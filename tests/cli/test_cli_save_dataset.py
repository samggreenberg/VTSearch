"""The CLI saves what it imports to the dashboard unless told not to (#4226).

``--autodetect`` used to import a dataset, score it, and throw it away. It now
imports the source through the GUI's own load pipeline, registers it (so the
dashboard lists it the next time the UI is opened), and scores the saved copy.
``--tempimport`` restores the old import-and-discard run, and is the only way
to stream, since a streamed source is never held whole.

These tests drive the library entry points with ``save_dataset=True`` (what the
flags resolve to) and pin the flag wiring separately.
"""

from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

from tests.helpers import make_dataset_file as _make_dataset_file
from vtscore.datasets.registry import list_datasets
from vtscore.media.audio.audio_generator import generate_wav
from vtsearch.settings import get_detectors_dir
from vtsearch.state import medias


@pytest.fixture(autouse=True)
def _clean_detectors_dir():
    d = get_detectors_dir()
    if d.is_dir():
        shutil.rmtree(d)
    yield
    d = get_detectors_dir()
    if d.is_dir():
        shutil.rmtree(d)


def _write_detector(name: str, media_type: str = "audio") -> Path:
    from vtscore.detectors.store import _detector_path, _write_detector

    labelset = {
        "labels": [
            {
                "md5": "a" * 32,
                "label": "good",
                "origin": {"importer": "ds_a", "params": {}},
                "origin_name": "alpha.wav",
            },
            {"md5": "b" * 32, "label": "good", "origin": {"importer": "ds_a", "params": {}}, "origin_name": "beta.wav"},
            {"md5": "c" * 32, "label": "bad", "origin": {"importer": "ds_a", "params": {}}, "origin_name": "gamma.wav"},
        ]
    }
    path = _detector_path(name)
    _write_detector(
        path,
        {"name": name, "text_query": "", "media_type": media_type, "examples": [], "labelset": labelset},
    )
    return path


def _stub_resolve(monkeypatch, tmp_path: Path) -> None:
    """Resolve the detector's label origins to freshly written wav files."""
    import vtscore.detectors.resolver as resolver_mod

    files: dict[str, Path] = {}
    for i, name in enumerate(["alpha.wav", "beta.wav", "gamma.wav"]):
        path = tmp_path / name
        path.write_bytes(generate_wav(220 + 110 * i, 0.1))
        files[name] = path

    @contextmanager
    def _fake_ctx(origin, origin_name="", filename=""):
        yield files.get(origin_name) or files.get(filename)

    monkeypatch.setattr(resolver_mod, "resolve_file_context", _fake_ctx)


def _settings_file(tmp_path: Path, autofind: list[str]) -> Path:
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"autofind_detectors": list(autofind), "detectors_dir": str(get_detectors_dir())}))
    return p


def _wav_folder(tmp_path: Path, n: int = 4) -> Path:
    folder = tmp_path / "field_recordings"
    folder.mkdir()
    for i in range(n):
        (folder / f"clip_{i}.wav").write_bytes(generate_wav(300 + 70 * i, 0.1))
    return folder


# ---------------------------------------------------------------------------
# Library entry points
# ---------------------------------------------------------------------------


class TestSavingRun:
    def test_pickle_run_saves_a_copy_and_scores_it(self, client, tmp_path, monkeypatch, capsys):
        _stub_resolve(monkeypatch, tmp_path)
        _write_detector("saver")
        source = _make_dataset_file(tmp_path, medias)
        out_path = tmp_path / "hits.json"

        from vtscore.cli import autodetect_main

        autodetect_main(
            str(source),
            settings_path=str(_settings_file(tmp_path, ["saver"])),
            exporter_name="server_json_file",
            exporter_field_values={"filepath": str(out_path)},
            save_dataset=True,
        )

        entries = list_datasets()
        assert len(entries) == 1
        entry = entries[0]
        assert entry["num_items"] == len(medias)
        assert entry["name"] == source.stem
        # A copy under saved_datasets, never the user's own file: deleting the
        # dataset on the dashboard deletes its pickle.
        saved = Path(entry["pkl_path"])
        assert saved.is_file()
        assert saved.resolve() != source.resolve()
        assert source.is_file()

        results = json.loads(out_path.read_text())["results"]
        assert "saver" in results
        assert "Saved dataset" in capsys.readouterr().out

    def test_importer_run_saves_under_the_folder_name(self, client, tmp_path):
        folder = _wav_folder(tmp_path)

        from vtscore.cli import autodetect_importer_main

        autodetect_importer_main(
            "server_folder",
            {"path": str(folder), "media_type": "audio", "reference_files": False},
            settings_path=str(_settings_file(tmp_path, [])),
            save_dataset=True,
        )

        entries = list_datasets()
        assert len(entries) == 1
        assert entries[0]["name"] == "field_recordings"
        assert entries[0]["num_items"] == 4
        assert entries[0]["media_type"] == "audio"

    def test_saved_dataset_belongs_to_the_running_user(self, client, tmp_path):
        """``--user alice`` imports onto alice's dashboard, not the default one."""
        from vtscore.cli import autodetect_main
        from vtscore.state.current_user import thread_user

        source = _make_dataset_file(tmp_path, medias)
        with thread_user("alice"):
            autodetect_main(str(source), settings_path=str(_settings_file(tmp_path, [])), save_dataset=True)

        assert [e["created_by"] for e in list_datasets()] == ["alice"]

    def test_no_autofind_detectors_saves_and_skips_detection(self, client, tmp_path, capsys):
        from vtscore.cli import autodetect_main

        source = _make_dataset_file(tmp_path, medias)
        # No SystemExit: the import is the product, detection the extra.
        autodetect_main(str(source), settings_path=str(_settings_file(tmp_path, [])), save_dataset=True)

        assert len(list_datasets()) == 1
        assert "Detection skipped: no Auto-Find detectors are configured" in capsys.readouterr().out

    def test_detector_for_another_media_type_saves_and_skips(self, client, tmp_path, capsys):
        _write_detector("videos-only", media_type="video")
        from vtscore.cli import autodetect_main

        source = _make_dataset_file(tmp_path, medias)
        autodetect_main(str(source), settings_path=str(_settings_file(tmp_path, ["videos-only"])), save_dataset=True)

        assert len(list_datasets()) == 1
        assert "Detection skipped: No Auto-Find detectors found for media type: audio" in capsys.readouterr().out

    def test_already_saved_pickle_is_not_imported_twice(self, client, tmp_path, capsys):
        from vtscore.cli import autodetect_main

        settings = str(_settings_file(tmp_path, []))
        autodetect_main(str(_make_dataset_file(tmp_path, medias)), settings_path=settings, save_dataset=True)
        [entry] = list_datasets()

        autodetect_main(entry["pkl_path"], settings_path=settings, save_dataset=True)

        assert [e["id"] for e in list_datasets()] == [entry["id"]]
        assert "already on the dashboard" in capsys.readouterr().out

    def test_import_failure_exits_non_zero_and_saves_nothing(self, client, tmp_path, capsys):
        from vtscore.cli import autodetect_main

        with pytest.raises(SystemExit) as exc:
            autodetect_main(str(tmp_path / "missing.pkl"), save_dataset=True)
        assert exc.value.code == 1
        assert "Dataset file not found" in capsys.readouterr().err
        assert list_datasets() == []

    def test_the_import_does_not_stay_resident(self, client, tmp_path):
        """The CLI scores the saved pickle, so the import's in-memory copy is
        released rather than doubling the run's footprint."""
        from vtscore.cli import autodetect_main
        from vtscore.datasets.registry import get_loaded_ids
        from vtscore.state.core import get_context

        autodetect_main(
            str(_make_dataset_file(tmp_path, medias)),
            settings_path=str(_settings_file(tmp_path, [])),
            save_dataset=True,
        )
        [entry] = list_datasets()
        assert entry["id"] not in get_loaded_ids()
        assert get_context(entry["id"]) is None

    def test_json_mode_reports_the_saved_dataset(self, client, tmp_path, capsys):
        from vtscore import cli_progress
        from vtscore.cli import autodetect_main

        cli_progress.set_format("json")
        try:
            autodetect_main(
                str(_make_dataset_file(tmp_path, medias)),
                settings_path=str(_settings_file(tmp_path, [])),
                save_dataset=True,
            )
        finally:
            cli_progress.set_format("text")

        events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
        saved = [e for e in events if e["event"] == "dataset_saved"]
        assert len(saved) == 1
        assert saved[0]["dataset_id"] == list_datasets()[0]["id"]
        assert saved[0]["already_saved"] is False
        assert any(e["event"] == "detection_skipped" for e in events)


class TestTemporaryRun:
    def test_default_library_call_keeps_nothing(self, client, tmp_path, monkeypatch):
        """Library callers that never asked to save still leave no trace."""
        _stub_resolve(monkeypatch, tmp_path)
        _write_detector("temp")
        out_path = tmp_path / "hits.json"

        from vtscore.cli import autodetect_main

        autodetect_main(
            str(_make_dataset_file(tmp_path, medias)),
            settings_path=str(_settings_file(tmp_path, ["temp"])),
            exporter_name="server_json_file",
            exporter_field_values={"filepath": str(out_path)},
        )

        assert "temp" in json.loads(out_path.read_text())["results"]
        assert list_datasets() == []

    def test_temporary_run_still_errors_without_detectors(self, client, tmp_path):
        from vtscore.cli import autodetect_main

        with pytest.raises(SystemExit) as exc:
            autodetect_main(str(_make_dataset_file(tmp_path, medias)), settings_path=str(_settings_file(tmp_path, [])))
        assert exc.value.code == 1

    def test_streaming_cannot_save(self, client, tmp_path, capsys):
        from vtscore.cli import autodetect_main_chunked

        with pytest.raises(SystemExit):
            autodetect_main_chunked(
                str(_make_dataset_file(tmp_path, medias)), 2, stream_results=True, save_dataset=True
            )
        assert "cannot save it to the dashboard" in capsys.readouterr().err
        assert list_datasets() == []


class TestDryRun:
    @pytest.mark.parametrize(("save", "expected"), [(True, "Save to dashboard: yes"), (False, "Save to dashboard: no")])
    def test_plan_says_whether_the_dataset_is_kept(self, client, tmp_path, capsys, save, expected):
        from vtscore.cli import autodetect_main

        autodetect_main(str(_make_dataset_file(tmp_path, medias)), dry_run=True, save_dataset=save)

        assert expected in capsys.readouterr().out
        assert list_datasets() == []


# ---------------------------------------------------------------------------
# YAML pipeline
# ---------------------------------------------------------------------------


class TestPipelineFile:
    @pytest.mark.parametrize(("tempimport", "saved"), [(False, 1), (True, 0)])
    def test_pipeline_saves_unless_tempimport(self, client, tmp_path, monkeypatch, tempimport, saved):
        _stub_resolve(monkeypatch, tmp_path)
        _write_detector("yaml-saver")
        out_path = tmp_path / "hits.json"
        pipeline = {
            "dataset": str(_make_dataset_file(tmp_path, medias)),
            "settings": str(_settings_file(tmp_path, ["yaml-saver"])),
            "exporter": {"name": "server_json_file", "fields": {"filepath": str(out_path)}},
        }
        if tempimport:
            pipeline["tempimport"] = True
        path = tmp_path / "pipeline.yaml"
        path.write_text(yaml.safe_dump(pipeline))

        from vtscore.cli_pipeline import run_pipeline_file

        run_pipeline_file(path)

        assert len(list_datasets()) == saved
        assert "yaml-saver" in json.loads(out_path.read_text())["results"]


# ---------------------------------------------------------------------------
# Flag wiring (entry points mocked)
# ---------------------------------------------------------------------------


def _run_main(monkeypatch, argv):
    from vtsearch import cli_main

    monkeypatch.setattr("sys.argv", ["app.py", *argv])
    monkeypatch.setattr(cli_main, "_run_server", lambda *a, **k: pytest.fail("server started"))
    cli_main.main(app=None, initialize_server=lambda **_: None)


def _record_entry_points(monkeypatch) -> dict:
    import vtscore.cli as vtcli

    calls: dict = {}
    for fn in (
        "autodetect_main",
        "autodetect_main_chunked",
        "autodetect_importer_main",
        "autodetect_importer_main_chunked",
    ):
        monkeypatch.setattr(vtcli, fn, lambda *a, _fn=fn, **k: calls.setdefault(_fn, (a, k)))
    return calls


class TestFlagWiring:
    def test_autodetect_saves_by_default(self, monkeypatch):
        calls = _record_entry_points(monkeypatch)
        _run_main(monkeypatch, ["--autodetect", "--dataset", "x.pkl"])
        assert calls["autodetect_main"][1]["save_dataset"] is True

    def test_tempimport_discards(self, monkeypatch):
        calls = _record_entry_points(monkeypatch)
        _run_main(monkeypatch, ["--autodetect", "--tempimport", "--dataset", "x.pkl"])
        assert calls["autodetect_main"][1]["save_dataset"] is False

    def test_tempimport_alone_implies_autodetect(self, monkeypatch):
        calls = _record_entry_points(monkeypatch)
        _run_main(monkeypatch, ["--tempimport", "--dataset", "x.pkl"])
        assert calls["autodetect_main"][1]["save_dataset"] is False

    def test_tempimport_alone_still_registers_importer_flags(self, monkeypatch):
        """The implied --autodetect must hold across the plugin re-parse."""
        calls = _record_entry_points(monkeypatch)
        _run_main(monkeypatch, ["--tempimport", "--importer", "server_folder", "--path", "/d", "--media-type", "audio"])
        args, kwargs = calls["autodetect_importer_main"]
        assert args[1]["path"] == "/d"
        assert kwargs["save_dataset"] is False

    def test_stream_results_requires_tempimport(self, monkeypatch, capsys):
        _record_entry_points(monkeypatch)
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--autodetect", "--dataset", "x.pkl", "--chunk-size", "10", "--stream-results"])
        assert exc.value.code == 2
        assert "--stream-results requires --tempimport" in capsys.readouterr().err

    def test_stream_results_with_tempimport_is_accepted(self, monkeypatch):
        calls = _record_entry_points(monkeypatch)
        _run_main(
            monkeypatch,
            ["--autodetect", "--tempimport", "--dataset", "x.pkl", "--chunk-size", "10", "--stream-results"],
        )
        assert calls["autodetect_main_chunked"][1]["stream_results"] is True

    def test_pipeline_rejects_tempimport_flag(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--pipeline", "flow.yaml", "--tempimport"])
        assert exc.value.code == 2
        assert "--pipeline cannot be combined with --tempimport" in capsys.readouterr().err
