"""Several datasets from one importer run on the command line (#4707).

``--autodetect --importer <name> --outputs '<json>'`` (and the library entry
points' ``outputs=``) run the importer once and make one dataset per entry:
each is saved (unless ``--tempimport``), scored by the AutoFind detectors that
apply to its media type, and exported on its own, with the exporter's file
named after the dataset.  A dataset nothing applies to is skipped with a note,
a dataset whose import failed is reported and the others still run, and the
dry-run plan lists them all.

These drive ``server_folder`` over a folder holding both wavs and pngs, so the
two outputs (audio, image) read real files through real loaders.
"""

from __future__ import annotations

import io
import json
import shutil
from contextlib import contextmanager
from pathlib import Path

import pytest

from vtscore.datasets.importers.base import ImporterBase, OutputSpec, PluginField, SourceSpec
from vtscore.datasets.registry import list_datasets
from vtscore.media.audio.audio_generator import generate_wav
from vtsearch.settings import get_detectors_dir


@pytest.fixture(autouse=True)
def _clean_detectors_dir():
    d = get_detectors_dir()
    if d.is_dir():
        shutil.rmtree(d)
    yield
    d = get_detectors_dir()
    if d.is_dir():
        shutil.rmtree(d)


@pytest.fixture(autouse=True)
def _bulk_embed_through_the_stub(monkeypatch):
    """Route every embedder's bulk path through the suite's per-item stub.

    The session stub covers ``embed_media``; an image embedder's own bulk
    implementation bypasses it and declines a real PNG without its model, so
    the image output would load empty.
    """
    from vtscore.media import all_embedders

    for emb in all_embedders():
        monkeypatch.setattr(emb, "embed_media_bulk", lambda items, _e=emb: [_e.embed_media(m) for m in items])


def _png_bytes(shade: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (shade, 255 - shade, 128)).save(buf, format="PNG")
    return buf.getvalue()


def _write_detector(name: str, media_type: str) -> Path:
    """A detector of *media_type* whose three labels resolve through :func:`_stub_resolve`."""
    from vtscore.detectors.store import _detector_path, _write_detector

    ext = "wav" if media_type == "audio" else "png"
    labelset = {
        "labels": [
            {"md5": c * 32, "label": label, "origin": {"importer": "ds_a", "params": {}}, "origin_name": f"{n}.{ext}"}
            for c, label, n in (("a", "good", "alpha"), ("b", "good", "beta"), ("c", "bad", "gamma"))
        ]
    }
    path = _detector_path(name)
    _write_detector(
        path,
        {"name": name, "text_query": "", "media_type": media_type, "examples": [], "labelset": labelset},
    )
    return path


def _stub_resolve(monkeypatch, tmp_path: Path) -> None:
    """Resolve every detector's label origins to freshly written files."""
    import vtscore.detectors.resolver as resolver_mod

    files: dict[str, Path] = {}
    labels = tmp_path / "labels"
    labels.mkdir(exist_ok=True)
    for i, stem in enumerate(["alpha", "beta", "gamma"]):
        wav = labels / f"{stem}.wav"
        wav.write_bytes(generate_wav(220 + 110 * i, 0.1))
        files[wav.name] = wav
        png = labels / f"{stem}.png"
        png.write_bytes(_png_bytes(40 * i))
        files[png.name] = png

    @contextmanager
    def _fake_ctx(origin, origin_name="", filename=""):
        yield files.get(origin_name) or files.get(filename)

    monkeypatch.setattr(resolver_mod, "resolve_file_context", _fake_ctx)


def _media_folder(tmp_path: Path, *, wavs: int = 4, pngs: int = 3) -> Path:
    folder = tmp_path / "field_kit"
    folder.mkdir()
    for i in range(wavs):
        (folder / f"clip_{i}.wav").write_bytes(generate_wav(300 + 70 * i, 0.1))
    for i in range(pngs):
        (folder / f"shot_{i}.png").write_bytes(_png_bytes(10 + 60 * i))
    return folder


def _settings_file(tmp_path: Path, autofind: list[str], *, delete_after: bool = False) -> Path:
    p = tmp_path / "settings.json"
    data = {"autofind_detectors": list(autofind), "detectors_dir": str(get_detectors_dir())}
    if delete_after:
        data["autofind_cli_delete_dataset"] = True
    p.write_text(json.dumps(data))
    return p


def _outputs() -> list[OutputSpec]:
    return [OutputSpec("audio"), OutputSpec("image")]


def _json_events(out: str) -> list[dict]:
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


@contextmanager
def _json_progress():
    from vtscore import cli_progress

    cli_progress.set_format("json")
    try:
        yield
    finally:
        cli_progress.set_format("text")


def _run(
    tmp_path: Path,
    *,
    save: bool,
    autofind: list[str],
    outputs=None,
    exporter_fields=None,
    chunk_size: int | None = None,
    **kwargs,
) -> Path:
    """One importer run over the media folder; returns the exporter's base path."""
    from vtscore.cli import autodetect_importer_main, autodetect_importer_main_chunked

    out_path = tmp_path / "hits.json"
    source: tuple = ("server_folder", {"path": str(_media_folder(tmp_path)), "reference_files": False})
    entry_point = autodetect_importer_main
    if chunk_size:
        source = (*source, chunk_size)
        entry_point = autodetect_importer_main_chunked
    entry_point(
        *source,
        settings_path=str(_settings_file(tmp_path, autofind)),
        exporter_name="server_json_file",
        exporter_field_values={"filepath": str(out_path), **(exporter_fields or {})},
        save_dataset=save,
        outputs=outputs if outputs is not None else _outputs(),
        **kwargs,
    )
    return out_path


def _per_dataset_files(out_path: Path) -> dict[str, dict]:
    """The exported file of each dataset, keyed by the dataset name it carries."""
    bodies = [json.loads(p.read_text()) for p in sorted(out_path.parent.glob("hits-*.json"))]
    return {b["dataset"]["name"]: b for b in bodies}


# ---------------------------------------------------------------------------
# Saving runs
# ---------------------------------------------------------------------------


class TestSavingRun:
    def test_one_dataset_per_output_each_scored_and_exported(self, client, tmp_path, monkeypatch):
        _stub_resolve(monkeypatch, tmp_path)
        _write_detector("ears", "audio")
        _write_detector("eyes", "image")

        out_path = _run(tmp_path, save=True, autofind=["ears", "eyes"])

        entries = {e["name"]: e for e in list_datasets()}
        assert set(entries) == {"field_kit – Audio", "field_kit – Image"}, "named like the dashboard's import"
        assert entries["field_kit – Audio"]["media_type"] == "audio"
        assert entries["field_kit – Audio"]["num_items"] == 4
        assert entries["field_kit – Image"]["media_type"] == "image"
        assert entries["field_kit – Image"]["num_items"] == 3

        assert not out_path.exists(), "two datasets never share one results file"
        files = _per_dataset_files(out_path)
        assert set(files) == set(entries)
        audio, image = files["field_kit – Audio"], files["field_kit – Image"]
        assert audio["media_type"] == "audio" and image["media_type"] == "image"
        # Each dataset is scored by the detectors that reach its type: the
        # image detector reaches audio through the audio2image route (the
        # CLI's ordinary converter routing), the audio one has no route to
        # images and is skipped there.
        assert set(audio["results"]) == {"ears", "eyes"}
        assert set(image["results"]) == {"eyes"}
        assert audio["dataset"] == {
            "name": "field_kit – Audio",
            "media_type": "audio",
            "category": "audio",
            "id": entries["field_kit – Audio"]["id"],
        }
        assert image["dataset"]["id"] == entries["field_kit – Image"]["id"]

    def test_the_saved_datasets_share_an_import_group(self, client, tmp_path, monkeypatch):
        """The CLI goes through the GUI's pipeline, so its datasets are linked the same way (#4747)."""
        _stub_resolve(monkeypatch, tmp_path)
        _write_detector("ears", "audio")

        _run(tmp_path, save=True, autofind=["ears"])

        entries = {e["name"]: e for e in list_datasets()}
        audio, image = entries["field_kit – Audio"], entries["field_kit – Image"]
        assert audio["import_group"] and audio["import_group"] == image["import_group"]
        assert (audio["output_category"], image["output_category"]) == ("audio", "image")

    def test_file_name_comes_from_the_dataset_name_template(self, client, tmp_path, monkeypatch):
        """``{dataset_name}`` anywhere in the path is the user's own naming; nothing is inserted."""
        _stub_resolve(monkeypatch, tmp_path)
        _write_detector("ears", "audio")

        _run(
            tmp_path, save=True, autofind=["ears"], exporter_fields={"filepath": str(tmp_path / "{dataset_name}.json")}
        )

        assert (tmp_path / "field_kit – Audio.json").exists()
        assert not list(tmp_path.glob("hits*")), "the template replaced the default file, not decorated it"

    def test_a_run_of_one_dataset_keeps_the_path_as_given(self, client, tmp_path, monkeypatch):
        _stub_resolve(monkeypatch, tmp_path)
        _write_detector("ears", "audio")

        out_path = _run(tmp_path, save=True, autofind=["ears"], outputs=[OutputSpec("audio")])

        assert out_path.exists()
        body = json.loads(out_path.read_text())
        assert body["dataset"]["name"] == "field_kit – Audio", "the dataset block still says which dataset it is"

    def test_dataset_nothing_applies_to_is_saved_and_skipped(self, client, tmp_path, monkeypatch, capsys):
        """An image dataset with only an audio detector configured is kept, noted, and the audio one scored."""
        _stub_resolve(monkeypatch, tmp_path)
        _write_detector("ears", "audio")

        with _json_progress():
            out_path = _run(tmp_path, save=True, autofind=["ears"])

        assert {e["name"] for e in list_datasets()} == {"field_kit – Audio", "field_kit – Image"}
        assert set(_per_dataset_files(out_path)) == {"field_kit – Audio"}
        events = _json_events(capsys.readouterr().out)
        [skipped] = [e for e in events if e["event"] == "detection_skipped"]
        assert skipped["dataset"] == "field_kit – Image"
        assert "image" in skipped["reason"]
        starts = [e for e in events if e["event"] == "dataset_start"]
        assert [(s["index"], s["count"], s["name"]) for s in starts] == [
            (0, 2, "field_kit – Audio"),
            (1, 2, "field_kit – Image"),
        ]
        saved = [e for e in events if e["event"] == "dataset_saved"]
        assert sorted(e["name"] for e in saved) == ["field_kit – Audio", "field_kit – Image"]

    def test_no_detectors_at_all_saves_both_and_exits_zero(self, client, tmp_path, capsys):
        _run(tmp_path, save=True, autofind=[])
        assert len(list_datasets()) == 2
        assert capsys.readouterr().out.count("Detection skipped for dataset") == 2

    def test_delete_after_detection_removes_only_the_scored_datasets(self, client, tmp_path, monkeypatch):
        """The ``autofind_cli_delete_dataset`` setting applies per dataset: a skipped one stays."""
        from vtscore.cli import autodetect_importer_main

        _stub_resolve(monkeypatch, tmp_path)
        _write_detector("ears", "audio")
        autodetect_importer_main(
            "server_folder",
            {"path": str(_media_folder(tmp_path)), "reference_files": False},
            settings_path=str(_settings_file(tmp_path, ["ears"], delete_after=True)),
            exporter_name="server_json_file",
            exporter_field_values={"filepath": str(tmp_path / "hits.json")},
            save_dataset=True,
            outputs=_outputs(),
        )
        assert [e["name"] for e in list_datasets()] == ["field_kit – Image"]

    def test_the_imports_do_not_stay_resident(self, client, tmp_path):
        from vtscore.datasets.registry import get_loaded_ids
        from vtscore.state.core import get_context

        _run(tmp_path, save=True, autofind=[])
        for entry in list_datasets():
            assert entry["id"] not in get_loaded_ids()
            assert get_context(entry["id"]) is None

    def test_import_progress_names_the_dataset(self, client, tmp_path, capsys):
        _run(tmp_path, save=True, autofind=[])
        out = capsys.readouterr().out
        assert "Importing 'field_kit – Audio':" in out
        assert "Importing 'field_kit – Image':" in out


# ---------------------------------------------------------------------------
# Temporary runs
# ---------------------------------------------------------------------------


class TestTemporaryRun:
    def test_scores_each_output_straight_off_the_importer(self, client, tmp_path, monkeypatch):
        _stub_resolve(monkeypatch, tmp_path)
        _write_detector("ears", "audio")
        _write_detector("eyes", "image")

        out_path = _run(tmp_path, save=False, autofind=["ears", "eyes"])

        assert list_datasets() == []
        files = _per_dataset_files(out_path)
        assert set(files) == {"field_kit – Audio", "field_kit – Image"}
        assert files["field_kit – Audio"]["dataset"]["id"] is None
        assert "ears" in files["field_kit – Audio"]["results"]
        assert set(files["field_kit – Image"]["results"]) == {"eyes"}, "no route from images to an audio detector"
        assert files["field_kit – Audio"]["results"]["ears"]["total_hits"] >= 1

    def test_chunked_run_scores_each_dataset_chunk_by_chunk(self, client, tmp_path, monkeypatch, capsys):
        _stub_resolve(monkeypatch, tmp_path)
        _write_detector("ears", "audio")
        _write_detector("eyes", "image")

        with _json_progress():
            out_path = _run(tmp_path, save=False, autofind=["ears", "eyes"], chunk_size=2)

        files = _per_dataset_files(out_path)
        assert set(files) == {"field_kit – Audio", "field_kit – Image"}
        done = [e for e in _json_events(capsys.readouterr().out) if e["event"] == "chunks_done"]
        assert [e["total_medias"] for e in done] == [4, 3], "each dataset is chunked on its own"

    def test_streaming_writes_one_ndjson_per_dataset(self, client, tmp_path, monkeypatch):
        _stub_resolve(monkeypatch, tmp_path)
        _write_detector("ears", "audio")
        _write_detector("eyes", "image")

        out_path = _run(tmp_path, save=False, autofind=["ears", "eyes"], chunk_size=2, stream_results=True)

        streams = {}
        for p in sorted(out_path.parent.glob("hits-*.json")):
            lines = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
            streams[lines[0]["_meta"]["dataset"]["name"]] = lines
        assert set(streams) == {"field_kit – Audio", "field_kit – Image"}
        meta = streams["field_kit – Image"][0]["_meta"]
        assert meta["media_type"] == "image"
        assert meta["dataset"] == {"name": "field_kit – Image", "media_type": "image", "category": "image", "id": None}
        assert [d["detector_name"] for d in meta["detectors"]] == ["eyes"]
        assert all(line["detector"] == "eyes" for line in streams["field_kit – Image"][1:])
        assert streams["field_kit – Audio"][0]["_meta"]["media_type"] == "audio"

    def test_one_dataset_without_a_detector_still_scores_the_other(self, client, tmp_path, monkeypatch, capsys):
        _stub_resolve(monkeypatch, tmp_path)
        _write_detector("ears", "audio")

        out_path = _run(tmp_path, save=False, autofind=["ears"])

        assert set(_per_dataset_files(out_path)) == {"field_kit – Audio"}
        out = capsys.readouterr().out
        assert "Detection skipped for dataset 'field_kit – Image'" in out
        assert "still saved to the dashboard" not in out, "a temporary run saved nothing"

    def test_nothing_to_score_anywhere_is_an_error(self, client, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc:
            _run(tmp_path, save=False, autofind=[])
        assert exc.value.code == 1
        assert "No AutoFind detector applies to any of the 2 dataset(s)" in capsys.readouterr().err
        assert list_datasets() == []

    def test_streaming_cannot_save(self, client, tmp_path, capsys):
        with pytest.raises(SystemExit):
            _run(tmp_path, save=True, autofind=[], chunk_size=2, stream_results=True)
        assert "cannot save it to the dashboard" in capsys.readouterr().err
        assert list_datasets() == []


# ---------------------------------------------------------------------------
# One output failing is that output's failure
# ---------------------------------------------------------------------------


class _Patchy(ImporterBase):
    """Yields media for the audio output and nothing for the image one."""

    name = "test_patchy"
    display_name = "Patchy"
    description = "test double"
    fields = [
        PluginField("path", "Folder", "folder"),
        PluginField("media_type", "Media type", "select", options=["audio", "image"], default="audio", required=False),
    ]

    def run(self, field_values, medias, thin=False):
        if field_values["media_type"] != "audio":
            return
        from vtscore.datasets.loader import load_dataset_from_folder

        load_dataset_from_folder(Path(field_values["path"]), "audio", medias, thin=thin)


def _plant_importer(monkeypatch, importer) -> None:
    import vtscore.datasets.importers as importers_mod

    real = importers_mod.get_importer
    monkeypatch.setattr(importers_mod, "get_importer", lambda name: importer if name == importer.name else real(name))


class TestOneOutputFailing:
    @pytest.mark.parametrize("save", [False, True])
    def test_the_other_dataset_is_still_scored_and_the_run_exits_one(self, client, tmp_path, monkeypatch, capsys, save):
        from vtscore.cli import autodetect_importer_main

        _stub_resolve(monkeypatch, tmp_path)
        _write_detector("ears", "audio")
        _plant_importer(monkeypatch, _Patchy())
        out_path = tmp_path / "hits.json"

        with _json_progress(), pytest.raises(SystemExit) as exc:
            autodetect_importer_main(
                "test_patchy",
                {"path": str(_media_folder(tmp_path))},
                settings_path=str(_settings_file(tmp_path, ["ears"])),
                exporter_name="server_json_file",
                exporter_field_values={"filepath": str(out_path)},
                save_dataset=save,
                outputs=_outputs(),
            )

        assert exc.value.code == 1
        files = _per_dataset_files(out_path)
        assert set(files) == {"field_kit – Audio"}, "the good output is exported before the run fails"
        events = _json_events(capsys.readouterr().out)
        [failed] = [e for e in events if e["event"] == "dataset_failed"]
        assert failed["name"] == "field_kit – Image"
        assert "no Image media" in failed["error"]
        [error] = [e for e in events if e["event"] == "error"]
        assert "1 of 2 dataset(s) failed to import" in error["message"]
        assert "field_kit – Image" in error["message"]
        order = [e["event"] for e in events]
        assert order.index("export_complete") < order.index("error")
        if save:
            assert [e["name"] for e in list_datasets()] == ["field_kit – Audio"], "the failed output registers nothing"


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_text_plan_lists_the_datasets(self, client, tmp_path, capsys):
        _run(
            tmp_path,
            save=True,
            autofind=[],
            dry_run=True,
            outputs=[
                OutputSpec("audio", embedder="clap"),
                OutputSpec(
                    "image",
                    category="document",
                    source_specs=[SourceSpec(source_type="document", converter="document2image")],
                    dataset_name="scans",
                ),
            ],
        )
        out = capsys.readouterr().out
        assert "Datasets (2; one importer run, one dataset per entry):" in out
        assert "- Audio  [media_type=audio, sources=audio, embedder=clap]" in out
        assert (
            "- Document  [media_type=image, category=document, sources=document→document2image, dataset_name=scans]"
            in out
        )
        assert list_datasets() == [], "a dry run imports nothing"
        assert not list(tmp_path.glob("hits*"))

    def test_json_plan_carries_the_outputs(self, client, tmp_path, capsys):
        with _json_progress():
            _run(tmp_path, save=True, autofind=[], dry_run=True)
        [plan] = [e for e in _json_events(capsys.readouterr().out) if e["event"] == "dry_run_plan"]
        assert [o["media_type"] for o in plan["source"]["outputs"]] == ["audio", "image"]
        assert plan["source"]["importer"] == "server_folder"

    def test_refuses_an_importer_that_makes_one_dataset(self, client, tmp_path, capsys):
        from vtscore.cli import autodetect_importer_main
        from vtscore.datasets.config import DEMO_DATASETS

        with pytest.raises(SystemExit) as exc:
            autodetect_importer_main(
                "demo",
                {"name": next(iter(DEMO_DATASETS))},
                settings_path=str(_settings_file(tmp_path, [])),
                dry_run=True,
                outputs=[OutputSpec("audio")],
            )
        assert exc.value.code == 1
        assert "produces one dataset per run" in capsys.readouterr().err


class TestSourceMediaType:
    """``--create-detector`` without ``--detector-media-type`` takes the type the outputs share."""

    def test_shared_type_is_the_sources(self):
        from vtscore.cli import _source_media_type, _SourceSpec

        spec = _SourceSpec(kind="importer", importer_name="server_folder", outputs=(OutputSpec("image"),))
        assert _source_media_type(spec) == "image"
        two = _SourceSpec(
            kind="importer",
            importer_name="server_folder",
            outputs=(OutputSpec("image"), OutputSpec("image", category="document")),
        )
        assert _source_media_type(two) == "image"

    def test_mixed_types_declare_none(self):
        from vtscore.cli import _source_media_type, _SourceSpec

        spec = _SourceSpec(kind="importer", importer_name="server_folder", outputs=tuple(_outputs()))
        assert _source_media_type(spec) == ""


# ---------------------------------------------------------------------------
# Flag wiring
# ---------------------------------------------------------------------------


def _run_main(monkeypatch, argv):
    from vtsearch import cli_main

    monkeypatch.setattr("sys.argv", ["app.py", *argv])
    monkeypatch.setattr(cli_main, "_run_server", lambda *a, **k: pytest.fail("server started"))
    cli_main.main(app=None, initialize_server=lambda **_: None)


def _record_entry_points(monkeypatch) -> dict:
    import vtscore.cli as vtcli

    calls: dict = {}
    for fn in ("autodetect_main", "autodetect_importer_main", "autodetect_importer_main_chunked"):
        monkeypatch.setattr(vtcli, fn, lambda *a, _fn=fn, **k: calls.setdefault(_fn, (a, k)))
    return calls


_OUTPUTS_JSON = json.dumps([{"media_type": "audio"}, {"media_type": "image", "embedder": "siglip"}])


class TestFlagWiring:
    def test_outputs_reach_the_importer_entry_point(self, monkeypatch):
        calls = _record_entry_points(monkeypatch)
        _run_main(
            monkeypatch, ["--autodetect", "--importer", "server_folder", "--path", "/d", "--outputs", _OUTPUTS_JSON]
        )
        args, kwargs = calls["autodetect_importer_main"]
        assert args[1]["path"] == "/d"
        assert [o.media_type for o in kwargs["outputs"]] == ["audio", "image"]
        assert kwargs["outputs"][1].embedder == "siglip"
        assert kwargs["save_dataset"] is True

    def test_chunked_entry_point_gets_them_too(self, monkeypatch):
        calls = _record_entry_points(monkeypatch)
        _run_main(
            monkeypatch,
            [
                "--tempimport",
                "--importer",
                "server_folder",
                "--path",
                "/d",
                "--chunk-size",
                "5",
                "--outputs",
                _OUTPUTS_JSON,
            ],
        )
        args, kwargs = calls["autodetect_importer_main_chunked"]
        assert args[2] == 5
        assert [o.media_type for o in kwargs["outputs"]] == ["audio", "image"]

    def test_without_the_flag_nothing_is_passed(self, monkeypatch):
        calls = _record_entry_points(monkeypatch)
        _run_main(monkeypatch, ["--autodetect", "--importer", "server_folder", "--path", "/d"])
        assert "outputs" not in calls["autodetect_importer_main"][1]

    def test_requires_an_importer(self, monkeypatch, capsys):
        _record_entry_points(monkeypatch)
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--autodetect", "--dataset", "x.pkl", "--outputs", _OUTPUTS_JSON])
        assert exc.value.code == 2
        assert "--outputs requires --importer" in capsys.readouterr().err

    def test_conflicts_with_media_type(self, monkeypatch, capsys):
        _record_entry_points(monkeypatch)
        with pytest.raises(SystemExit) as exc:
            _run_main(
                monkeypatch,
                [
                    "--autodetect",
                    "--importer",
                    "server_folder",
                    "--path",
                    "/d",
                    "--media-type",
                    "image",
                    "--outputs",
                    _OUTPUTS_JSON,
                ],
            )
        assert exc.value.code == 2
        assert "--outputs and --media-type both set" in capsys.readouterr().err

    def test_bad_json_is_a_usage_error(self, monkeypatch, capsys):
        _record_entry_points(monkeypatch)
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--autodetect", "--importer", "server_folder", "--path", "/d", "--outputs", "[{"])
        assert exc.value.code == 2
        assert "--outputs: Invalid outputs JSON" in capsys.readouterr().err

    def test_unknown_media_type_is_a_usage_error(self, monkeypatch, capsys):
        _record_entry_points(monkeypatch)
        with pytest.raises(SystemExit) as exc:
            _run_main(
                monkeypatch,
                [
                    "--autodetect",
                    "--importer",
                    "server_folder",
                    "--path",
                    "/d",
                    "--outputs",
                    '[{"media_type": "hologram"}]',
                ],
            )
        assert exc.value.code == 2
        assert "Unknown media type: 'hologram'" in capsys.readouterr().err

    def test_single_dataset_importer_is_refused(self, monkeypatch, capsys):
        from vtscore.datasets.config import DEMO_DATASETS

        _record_entry_points(monkeypatch)
        demo = next(iter(DEMO_DATASETS))
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--autodetect", "--importer", "demo", "--name", demo, "--outputs", _OUTPUTS_JSON])
        assert exc.value.code == 2
        assert "produces one dataset per run" in capsys.readouterr().err

    def test_pipeline_rejects_the_flag(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_main(monkeypatch, ["--pipeline", "flow.yaml", "--outputs", _OUTPUTS_JSON])
        assert exc.value.code == 2
        assert "--pipeline cannot be combined with --outputs" in capsys.readouterr().err
