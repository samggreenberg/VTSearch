"""A CLI autodetect run exports the floor's unchecked starting candidate, and says so (#4247, #4272).

Nobody can vote in a headless run, so the precision floor's spot check never
runs: the detector is scored at the floor's starting candidate - the top 128
unvoted items at 10%, the top 64 at 25%, the top 32 at 50% and above - and
that set is what gets exported.  The run says so twice: a
``detector_unchecked`` progress event (a line on the terminal in text mode),
and a ``floor`` entry beside ``threshold`` in every result the detector
produces, which the JSON exporters write out verbatim.  ``threshold`` itself
stays one float, the one the hits were cut at.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
import torch

import vtscore.cli as cli
from tests.cli.test_cli_score_geometry import _fires_on_dim1_mlp, _patch_medias
from vtscore import cli_progress
from vtscore.detectors.training import scoring_rows_for_snap
from vtscore.state import set_min_precision
from vtscore.training.thresholds import LineRanking

#: ``(min_precision, the schedule's starting candidate, rounds, picks)``.
PRESETS = [
    pytest.param(0.1, 128, 3, 5, id="10%"),
    pytest.param(0.5, 32, 1, 5, id="50%"),
    pytest.param(0.9, 32, 1, 29, id="90%"),
]


def _train(monkeypatch, n_ranked: int | None) -> dict[str, dict[str, Any]]:
    """``_load_and_train_detectors`` for one detector, with a ranking of *n_ranked* items parked when given."""
    import vtscore.detectors.labelset_training as lt
    import vtscore.detectors.store as store_mod

    det = {
        "name": "det",
        "media_type": "image",
        "labelset": {"labels": [{"md5": "a", "label": "good"}, {"md5": "b", "label": "bad"}]},
    }
    monkeypatch.setattr(store_mod, "_read_detector", lambda _p: det)
    monkeypatch.setattr("vtscore.detectors.input_spec.extract_input_spec_from_medias", lambda _m: None)

    def _fake_train(det_ctx, labelset, *, media_type, snap, haystack_for=None, on_progress=None):
        det_ctx.embedder = "fake"
        det_ctx.model = torch.nn.Sequential(torch.nn.Linear(4, 1))
        det_ctx.threshold = 0.5
        if n_ranked is not None:
            det_ctx.line_ranking = LineRanking.from_scores(
                list(range(1, n_ranked + 1)), np.linspace(0.99, 0.01, n_ranked), set()
            )
        return True

    monkeypatch.setattr(lt, "train_from_labelset", _fake_train)
    return cli._load_and_train_detectors(["det"], "image", {1: {"id": 1, "media_type": "image"}})


def _events(capsys) -> list[dict]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]


class TestTrainingRecordsTheState:
    @pytest.mark.parametrize(("min_precision", "candidate", "rounds", "picks"), PRESETS)
    def test_each_preset_exports_its_starting_candidate(
        self, client, monkeypatch, capsys, min_precision, candidate, rounds, picks
    ):
        set_min_precision(min_precision)
        cli_progress.set_format("json")

        out = _train(monkeypatch, 200)

        assert out["det"]["floor"] == {
            "min_precision": min_precision,
            "status": "unchecked",
            "count": candidate,
            "range": None,
            "schedule": {"candidate": candidate, "rounds": rounds, "picks": picks},
        }
        assert out["det"]["threshold"] == 0.5, "the cut is still the one training drew"
        unchecked = [e for e in _events(capsys) if e["event"] == "detector_unchecked"]
        assert len(unchecked) == 1
        event = unchecked[0]
        assert event["detector"] == "det"
        assert (event["min_precision"], event["status"], event["count"]) == (min_precision, "unchecked", candidate)

    def test_a_small_corpus_is_its_own_candidate(self, client, monkeypatch, capsys):
        set_min_precision(0.1)
        cli_progress.set_format("json")
        out = _train(monkeypatch, 40)
        assert out["det"]["floor"]["count"] == 40
        assert [e for e in _events(capsys) if e["event"] == "detector_unchecked"][0]["count"] == 40

    def test_no_ranking_still_records_the_schedules_count(self, client, monkeypatch, capsys):
        set_min_precision(0.5)
        cli_progress.set_format("json")
        out = _train(monkeypatch, None)
        assert out["det"]["floor"]["status"] == "unchecked" and out["det"]["floor"]["count"] == 32

    def test_text_mode_says_what_the_export_is(self, client, monkeypatch, capsys):
        set_min_precision(0.25)
        _train(monkeypatch, 200)
        out = capsys.readouterr().out
        assert "Detector 'det' exports its top 64 unchecked (aiming at 25% right)" in out
        assert "nobody is here to check it" in out


FLOOR = {
    "min_precision": 0.5,
    "status": "unchecked",
    "count": 32,
    "range": None,
    "schedule": {"candidate": 32, "rounds": 1, "picks": 5},
}


class TestResultsCarryTheState:
    def test_direct_scoring(self):
        info = {"mlp": _fires_on_dim1_mlp(), "threshold": 0.5, "floor": FLOOR}
        out = cli._score_direct_all(["d"], {"d": info}, _patch_medias(), "dinov3_patch")
        assert out["d"]["floor"] == FLOOR
        assert {h["id"] for h in out["d"]["hits"]} == {1}, "the unchecked cut still exports its hits"

    def test_routed_scoring(self):
        medias = _patch_medias()
        info = {"mlp": _fires_on_dim1_mlp(), "threshold": 0.5, "floor": FLOOR}
        out = cli._score_one_detector(
            "d", info, medias, scoring_rows_for_snap(medias, "dinov3_patch"), {cid: cid for cid in medias}
        )
        assert out["floor"] == FLOOR
        assert {h["id"] for h in out["hits"]} == {1}

    def test_a_detector_with_no_state_reports_none(self):
        info = {"mlp": _fires_on_dim1_mlp(), "threshold": 0.5}
        out = cli._score_direct_all(["d"], {"d": info}, _patch_medias(), "dinov3_patch")
        assert out["d"]["floor"] is None
