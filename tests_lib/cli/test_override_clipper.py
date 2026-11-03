"""The clipper override: score every detector at a chosen granularity (#5282).

A detector re-clips the dataset to its own ``input_spec`` at scoring time;
``--override-clipper`` (and a pipeline file's ``override_clipper:``) put a
chosen clipper in that spec's place.  These pin the library half: the
override's validation, the per-detector re-clip decision, the pipeline-file
key, and the dry-run plan.  The flags and a whole scored run are in
``tests/cli/test_cli_override_clipper.py``.
"""

from __future__ import annotations

import json

import pytest

from vtscore.cli import progress as cli_progress
from vtscore.cli.detectors import _override_reclip, _resolve_override_clipper
from vtscore.cli.dry_run import _emit_dry_run_plan
from vtscore.detectors.input_spec import resolve_clipper_spec

FIVE_S = resolve_clipper_spec("sound_tiling", {"duration": "5"})
TWO_S = resolve_clipper_spec("sound_tiling", {"duration": "2"})


def _events(out: str) -> list[dict]:
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


class TestResolveOverrideClipper:
    def test_no_override_stays_none(self):
        assert _resolve_override_clipper(None) is None

    def test_resolves_to_the_effective_spec(self):
        override = {"clipper": "sound_tiling", "clipper_params": {"duration": 5}}
        assert _resolve_override_clipper(override) == FIVE_S

    def test_needs_a_clipper_name(self):
        with pytest.raises(ValueError, match="'clipper' name"):
            _resolve_override_clipper({"clipper_params": {"duration": "5"}})

    def test_params_must_be_a_mapping(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            _resolve_override_clipper({"clipper": "sound_tiling", "clipper_params": ["duration=5"]})

    def test_an_unknown_clipper_is_refused(self):
        with pytest.raises(ValueError, match="Unknown clipper 'sound_tiles'"):
            _resolve_override_clipper({"clipper": "sound_tiles"})


class TestOverrideReclip:
    def test_reclips_a_dataset_with_no_clipper(self, capsys):
        cli_progress.set_format("json")
        assert _override_reclip("dogs", "audio", FIVE_S, None, None) == ("sound_tiling", FIVE_S["clipper_params"])

        [event] = [e for e in _events(capsys.readouterr().out) if e["event"] == "detector_reclip"]
        assert event["detector"] == "dogs"
        assert event["override_input_spec"] == FIVE_S
        assert event["detector_input_spec"] == {}

    def test_text_mode_says_the_override_did_it(self, capsys):
        _override_reclip("dogs", "audio", FIVE_S, TWO_S, None)
        assert (
            "Re-clipping dataset for detector 'dogs' with clipper 'sound_tiling' (duration=5.0, min_overlap=1.0) "
            "from the clipper override (its input_spec: 'sound_tiling' (duration=2.0, min_overlap=1.0); "
            "dataset clipper: (none))."
        ) in capsys.readouterr().out

    def test_replaces_the_detectors_own_spec(self, capsys):
        cli_progress.set_format("json")
        clipper, params = _override_reclip("dogs", "audio", FIVE_S, TWO_S, TWO_S)

        assert (clipper, params["duration"]) == ("sound_tiling", "5.0")
        [event] = [e for e in _events(capsys.readouterr().out) if e["event"] == "detector_reclip"]
        assert event["detector_input_spec"] == TWO_S
        assert event["dataset_input_spec"] == TWO_S

    def test_a_dataset_already_clipped_to_it_is_scored_as_is(self, capsys):
        cli_progress.set_format("json")
        assert _override_reclip("dogs", "audio", FIVE_S, TWO_S, FIVE_S) == ("", {})
        assert not [e for e in _events(capsys.readouterr().out) if e["event"] == "detector_reclip"]

    def test_a_default_clipper_scores_each_media_as_loaded(self):
        """Even when the detector's own spec would have re-clipped."""
        override = resolve_clipper_spec("sound_default")
        assert _override_reclip("dogs", "audio", override, TWO_S, None) == ("", {})

    def test_a_clipper_for_another_type_is_refused(self):
        override = resolve_clipper_spec("image_tiling")
        with pytest.raises(ValueError) as exc:
            _override_reclip("dogs", "audio", override, None, None)
        message = str(exc.value)
        assert "'image_tiling' clips image media, but detector 'dogs' scores audio media" in message
        assert "Clippers for audio: sound_default, sound_silence, sound_speech_activity, sound_tiling" in message


class TestPipelineFileKey:
    def _load(self, tmp_path, block: str):
        from vtscore.cli.pipeline import load_pipeline_file

        path = tmp_path / "flow.yaml"
        path.write_text(f"dataset: foo.pkl\n{block}")
        return load_pipeline_file(path)

    def test_absent_is_none(self, tmp_path):
        assert self._load(tmp_path, "")["override_clipper"] is None

    def test_parses_into_the_effective_spec(self, tmp_path):
        config = self._load(tmp_path, "override_clipper:\n  name: sound_tiling\n  params:\n    duration: 5\n")
        assert config["override_clipper"] == FIVE_S

    def test_params_are_optional(self, tmp_path):
        config = self._load(tmp_path, "override_clipper:\n  name: sound_default\n")
        assert config["override_clipper"] == {"clipper": "sound_default"}

    @pytest.mark.parametrize(
        ("block", "match"),
        [
            ("override_clipper: sound_tiling\n", "must be a mapping"),
            ("override_clipper:\n  params: {duration: 5}\n", "'override_clipper.name' is required"),
            ("override_clipper:\n  name: sound_tiling\n  fields: {}\n", "unknown key"),
            ("override_clipper:\n  name: sound_tiling\n  params: [5]\n", "'override_clipper.params' must be"),
            ("override_clipper:\n  name: sound_tiles\n", "'override_clipper': Unknown clipper 'sound_tiles'"),
            ("override_clipper:\n  name: sound_tiling\n  params: {duraton: 5}\n", "no parameter 'duraton'"),
        ],
    )
    def test_a_bad_block_fails_at_parse_time(self, tmp_path, block, match):
        with pytest.raises(ValueError, match=match):
            self._load(tmp_path, block)

    def test_dispatch_hands_it_to_the_run(self, monkeypatch):
        from vtscore.cli.pipeline import _dispatch

        seen: dict = {}
        monkeypatch.setattr("vtscore.cli.run._run_source", lambda spec, **kw: seen.update(kw))
        _dispatch(
            {
                "dataset": "foo.pkl",
                "importer": None,
                "importer_fields": {},
                "settings": None,
                "detectors": ["Dogs"],
                "chunk_size": None,
                "import_labels": None,
                "exporter": None,
                "exporter_fields": {},
                "tempimport": True,
                "override_clipper": FIVE_S,
            }
        )
        assert seen["override_clipper"] == FIVE_S


class TestDryRunPlan:
    def _plan(self, **kwargs) -> None:
        _emit_dry_run_plan({"kind": "pickle", "dataset": "foo.pkl"}, None, ["Dogs"], None, None, **kwargs)

    def test_text_plan_names_the_override(self, capsys):
        self._plan(override_clipper=FIVE_S)
        out = capsys.readouterr().out
        assert "Clipper override: 'sound_tiling' (duration=5.0, min_overlap=1.0)" in out

    def test_text_plan_without_one_says_nothing(self, capsys):
        self._plan()
        assert "Clipper override" not in capsys.readouterr().out

    def test_json_plan_carries_it(self, capsys):
        cli_progress.set_format("json")
        self._plan(override_clipper=FIVE_S)
        [plan] = [e for e in _events(capsys.readouterr().out) if e["event"] == "dry_run_plan"]
        assert plan["override_clipper"] == FIVE_S
