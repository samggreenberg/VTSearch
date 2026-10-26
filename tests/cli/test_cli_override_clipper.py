"""``--override-clipper``: score every detector at a chosen granularity (#5282).

The app half: the flags as ``python app.py`` parses them, the override
reaching ``_load_and_train_detectors`` and the scoring pass of a real run,
and the dry-run plan.  The re-clip decision itself and the pipeline-file key
are pinned in ``tests_lib/cli/test_override_clipper.py``.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from tests.helpers import make_dataset_file
from vtscore.detectors.input_spec import resolve_clipper_spec
from vtscore.media.audio.audio_generator import generate_wav
from vtsearch import cli_main
from vtsearch.settings import get_detectors_dir
from vtsearch.state import medias

HALF_S = resolve_clipper_spec("sound_tiling", {"duration": "0.5", "min_overlap": "0"})
TWO_S = resolve_clipper_spec("sound_tiling", {"duration": "2"})

_LABELSET = {
    "labels": [
        {"md5": "a" * 32, "label": "good", "origin": {"importer": "ds_a", "params": {}}, "origin_name": "alpha.wav"},
        {"md5": "c" * 32, "label": "bad", "origin": {"importer": "ds_a", "params": {}}, "origin_name": "gamma.wav"},
    ]
}


def _write_detector(name: str, *, media_type: str = "audio", input_spec: dict | None = None) -> None:
    from vtscore.detectors.store import _detector_path, _write_detector

    data = {"name": name, "media_type": media_type, "labelset": _LABELSET}
    if input_spec is not None:
        data["input_spec"] = input_spec
    _write_detector(_detector_path(name), data)


@pytest.fixture
def label_files(tmp_path, monkeypatch) -> None:
    """Make the labels' origin files resolvable, as a detector's would be in a real install."""
    import vtscore.detectors.resolver as resolver_mod

    files = {}
    for i, name in enumerate(["alpha.wav", "gamma.wav"]):
        files[name] = tmp_path / name
        files[name].write_bytes(generate_wav(220 + 110 * i, 0.1))

    @contextmanager
    def _fake_ctx(origin, origin_name="", filename=""):
        yield files.get(origin_name) or files.get(filename)

    monkeypatch.setattr(resolver_mod, "resolve_file_context", _fake_ctx)


def _settings_file(tmp_path: Path, detectors: list[str]) -> Path:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"autofind_detectors": detectors, "detectors_dir": str(get_detectors_dir())}))
    return path


def _events(out: str) -> list[dict]:
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


class TestTraining:
    """What ``_load_and_train_detectors`` records for scoring under an override."""

    def test_overrides_the_detectors_own_input_spec(self, client, label_files):
        from vtscore.cli import _load_and_train_detectors

        _write_detector("dogs", input_spec=TWO_S)
        trained = _load_and_train_detectors(["dogs"], "audio", dict(medias), override_clipper=HALF_S)

        assert trained["dogs"]["clipper"] == "sound_tiling"
        assert trained["dogs"]["clipper_params"] == HALF_S["clipper_params"]

    def test_a_detector_with_no_input_spec_is_reclipped_too(self, client, label_files):
        from vtscore.cli import _load_and_train_detectors

        _write_detector("dogs")
        trained = _load_and_train_detectors(["dogs"], "audio", dict(medias), override_clipper=HALF_S)

        assert trained["dogs"]["clipper"] == "sound_tiling"

    def test_without_an_override_the_input_spec_decides(self, client, label_files):
        from vtscore.cli import _load_and_train_detectors

        _write_detector("dogs", input_spec=TWO_S)
        trained = _load_and_train_detectors(["dogs"], "audio", dict(medias))

        assert trained["dogs"]["clipper_params"]["duration"] == "2.0"

    def test_a_clipper_for_another_type_stops_the_run_before_training(self, client, label_files, monkeypatch):
        import vtscore.detectors.labelset_training as training
        from vtscore.cli import _load_and_train_detectors

        monkeypatch.setattr(training, "train_from_labelset", lambda *a, **k: pytest.fail("trained anyway"))
        _write_detector("dogs")
        with pytest.raises(ValueError, match="'image_tiling' clips image media, but detector 'dogs' scores audio"):
            _load_and_train_detectors(
                ["dogs"], "audio", dict(medias), override_clipper=resolve_clipper_spec("image_tiling")
            )

    def test_a_detector_skipped_for_its_type_is_not_held_to_it(self, client, label_files):
        """A video detector has no route from an audio dataset: it is skipped, not refused."""
        from vtscore.cli import _load_and_train_detectors

        _write_detector("clips", media_type="video")
        _write_detector("dogs")
        trained = _load_and_train_detectors(["clips", "dogs"], "audio", dict(medias), override_clipper=HALF_S)

        assert list(trained) == ["dogs"]


class TestScoredRun:
    def test_every_scoring_pass_reads_the_override(self, client, tmp_path, label_files, monkeypatch, capsys):
        """The override travels from the entry point to the routing that scores the run."""
        import vtscore.detectors.converter_routing as cr
        from vtscore import cli_progress
        from vtscore.cli import autodetect_main

        clippers: list[tuple[str, dict]] = []
        real_route = cr.route_and_embed

        def _spy(medias_, target_type, embedder_name, clipper="", clipper_params=None):
            clippers.append((clipper, dict(clipper_params or {})))
            return real_route(medias_, target_type, embedder_name, clipper=clipper, clipper_params=clipper_params)

        monkeypatch.setattr(cr, "route_and_embed", _spy)
        _write_detector("dogs", input_spec=TWO_S)
        out_path = tmp_path / "hits.json"

        cli_progress.set_format("json")
        autodetect_main(
            str(make_dataset_file(tmp_path, medias)),
            settings_path=str(_settings_file(tmp_path, ["dogs"])),
            exporter_name="server_json_file",
            exporter_field_values={"filepath": str(out_path)},
            override_clipper={"clipper": "sound_tiling", "clipper_params": {"duration": "0.5", "min_overlap": "0"}},
        )

        assert clippers and all(c == ("sound_tiling", HALF_S["clipper_params"]) for c in clippers)
        [reclip] = [e for e in _events(capsys.readouterr().out) if e["event"] == "detector_reclip"]
        assert reclip["override_input_spec"] == HALF_S
        assert "dogs" in json.loads(out_path.read_text())["results"]

    def test_an_unknown_clipper_fails_the_run(self, client, tmp_path, capsys):
        from vtscore.cli import autodetect_main

        _write_detector("dogs")
        with pytest.raises(SystemExit) as exc:
            autodetect_main(
                str(make_dataset_file(tmp_path, medias)),
                settings_path=str(_settings_file(tmp_path, ["dogs"])),
                override_clipper={"clipper": "sound_tiles"},
            )
        assert exc.value.code == 1
        assert "Unknown clipper 'sound_tiles'" in capsys.readouterr().err

    def test_the_dry_run_plan_names_it(self, client, tmp_path, capsys):
        from vtscore.cli import autodetect_main

        _write_detector("dogs")
        autodetect_main(
            str(make_dataset_file(tmp_path, medias)),
            settings_path=str(_settings_file(tmp_path, ["dogs"])),
            dry_run=True,
            override_clipper={"clipper": "sound_tiling", "clipper_params": {"duration": "5"}},
        )
        assert "Clipper override: 'sound_tiling' (duration=5.0, min_overlap=1.0)" in capsys.readouterr().out


class TestFlags:
    """``--override-clipper`` / ``--override-clipper-param`` as ``python app.py`` parses them."""

    def _run(self, monkeypatch, *flags: str) -> dict:
        import vtscore.cli as vtcli

        seen: dict = {}
        monkeypatch.setattr(vtcli, "autodetect_main", lambda *a, **kw: seen.update(kw))
        monkeypatch.setattr(sys, "argv", ["app.py", "--autodetect", "--dataset", "x.pkl", *flags])
        cli_main.main(None, None)
        return seen

    def test_the_flags_reach_the_run_as_a_resolved_spec(self, monkeypatch):
        seen = self._run(
            monkeypatch,
            "--override-clipper",
            "sound_tiling",
            "--override-clipper-param",
            "duration=0.5",
            "--override-clipper-param",
            "min_overlap=0",
        )
        assert seen["override_clipper"] == HALF_S

    def test_no_flag_no_override(self, monkeypatch):
        assert self._run(monkeypatch)["override_clipper"] is None

    @pytest.mark.parametrize(
        ("flags", "message"),
        [
            (["--override-clipper", "sound_tiles"], "--override-clipper: Unknown clipper 'sound_tiles'"),
            (
                ["--override-clipper", "sound_tiling", "--override-clipper-param", "duraton=5"],
                "--override-clipper: Clipper 'sound_tiling' has no parameter 'duraton'",
            ),
            (
                ["--override-clipper", "sound_tiling", "--override-clipper-param", "duration=0"],
                "duration must be positive",
            ),
            (
                ["--override-clipper", "sound_tiling", "--override-clipper-param", "duration"],
                "--override-clipper-param expects KEY=VALUE, got 'duration'",
            ),
            (
                ["--override-clipper-param", "duration=5"],
                "--override-clipper-param only applies with --override-clipper",
            ),
        ],
    )
    def test_a_bad_override_is_a_usage_error(self, monkeypatch, capsys, flags, message):
        with pytest.raises(SystemExit) as exc:
            self._run(monkeypatch, *flags)
        assert exc.value.code == 2
        assert message in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("flags", "named"),
        [
            (["--override-clipper", "sound_tiling"], "--override-clipper"),
            (["--override-clipper-param", "duration=5"], "--override-clipper-param"),
            (["--label-importer-field", "filepath=x"], "--label-importer-field"),
        ],
    )
    def test_pipeline_refuses_them_by_their_own_spelling(self, monkeypatch, capsys, flags, named):
        """The YAML file's ``override_clipper:`` declares it; the flags beside ``--pipeline`` are refused."""
        monkeypatch.setattr(sys, "argv", ["app.py", "--pipeline", "flow.yaml", *flags])
        with pytest.raises(SystemExit) as exc:
            cli_main.main(None, None)
        assert exc.value.code == 2
        assert f"--pipeline cannot be combined with {named};" in capsys.readouterr().err
