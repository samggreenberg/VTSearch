"""A CLI autodetect run exports the balance's unchecked line, and says so (#4247, #4272, #4389, #4413).

Nobody can vote in a headless run, so the balance's spot check never runs:
the detector is scored at the balance's unchecked line - the smaller of the
balance's cap (the top 32 unvoted items at beta 1 and below, the top 128
above) and the mixture's F-beta argmax - and that set is what gets exported.
The run says so twice: a ``detector_unchecked`` progress event (a line on the
terminal in text mode), and a ``balance`` entry beside ``threshold`` in every
result the detector produces, which the JSON exporters write out verbatim.
``threshold`` itself stays one float, the one the hits were cut at.
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
from vtscore.state import set_beta
from vtscore.training.thresholds import LineRanking

#: ``(beta, the balance's cap, the bands the walk audits first, picks a band, the check's shape)`` (#4388, #4427).
# The app's presets (#4448); every check is advisory since #4452.
PRESETS = [
    pytest.param(0.25, 32, 3, 5, "advisory", id="F0.25"),
    pytest.param(1.0, 32, 3, 5, "advisory", id="F1"),
    pytest.param(4.0, 128, 5, 5, "advisory", id="F4"),
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
    @pytest.mark.parametrize(("beta", "candidate", "rounds", "picks", "shape"), PRESETS)
    def test_each_preset_exports_its_cap(
        self, client, monkeypatch, capsys, schedule_only, beta, candidate, rounds, picks, shape
    ):
        set_beta(beta)
        cli_progress.set_format("json")

        out = _train(monkeypatch, 200)

        assert out["det"]["balance"] == {
            "beta": beta,
            "status": "unchecked",
            "count": candidate,
            "precision": None,
            "recall": None,
            "fbeta": None,
            "schedule": {"candidate": candidate, "rounds": rounds, "picks": picks},
            "shape": shape,
            "audited": None,
            "checkable": True,
            "separation": None,
            "check_due": False,
        }
        assert "floor" not in out["det"]
        assert out["det"]["threshold"] == 0.5, "the cut is still the one training drew"
        unchecked = [e for e in _events(capsys) if e["event"] == "detector_unchecked"]
        assert len(unchecked) == 1
        event = unchecked[0]
        assert event["detector"] == "det" and "min_precision" not in event
        assert (event["beta"], event["status"], event["count"]) == (beta, "unchecked", candidate)

    def test_a_small_corpus_is_its_own_count(self, client, monkeypatch, capsys, schedule_only):
        set_beta(2.0)
        cli_progress.set_format("json")
        out = _train(monkeypatch, 40)
        assert out["det"]["balance"]["count"] == 40
        assert [e for e in _events(capsys) if e["event"] == "detector_unchecked"][0]["count"] == 40

    def test_no_ranking_still_records_the_caps_count(self, client, monkeypatch, capsys):
        set_beta(1.0)
        cli_progress.set_format("json")
        out = _train(monkeypatch, None)
        assert out["det"]["balance"]["status"] == "unchecked" and out["det"]["balance"]["count"] == 32

    def test_text_mode_says_what_the_export_is(self, client, monkeypatch, capsys, schedule_only):
        set_beta(2.0)
        _train(monkeypatch, 200)
        out = capsys.readouterr().out
        assert "Detector 'det' exports its top 128 unchecked (at F2)" in out
        assert "nobody is here to check it" in out

    def test_the_mixture_lowers_the_exported_count(self, client, monkeypatch, capsys):
        """The headless line is the smaller of the cap and the mixture's F-beta argmax (#4389), recorded as unchecked."""
        import vtscore.training.thresholds as thresholds

        seen: list[tuple] = []

        def _five(ranking, beta, labels, also_voted=()):
            if ranking is None:  # the balance setting re-cuts every context, ranking or not
                return None
            seen.append((ranking.size, beta, dict(labels), set(also_voted)))
            return 5

        monkeypatch.setattr(thresholds, "fbeta_count", _five)
        set_beta(1.0)
        cli_progress.set_format("json")
        out = _train(monkeypatch, 200)
        assert seen and seen[0] == (200, 1.0, {}, set()), "anchored on the (here empty) human votes"
        assert out["det"]["balance"]["status"] == "unchecked" and out["det"]["balance"]["count"] == 5
        assert out["det"]["balance"]["schedule"]["candidate"] == 32, "the walk would still start at the cap"
        event = [e for e in _events(capsys) if e["event"] == "detector_unchecked"][0]
        assert (event["status"], event["count"]) == ("unchecked", 5)


BALANCE = {
    "beta": 1.0,
    "status": "unchecked",
    "count": 32,
    "precision": None,
    "recall": None,
    "fbeta": None,
    "schedule": {"candidate": 32, "rounds": 3, "picks": 5},
    "shape": "advisory",
    "audited": None,
}


class TestResultsCarryTheState:
    def test_direct_scoring(self):
        info = {"mlp": _fires_on_dim1_mlp(), "threshold": 0.5, "balance": BALANCE}
        out = cli._score_direct_all(["d"], {"d": info}, _patch_medias(), "dinov3_patch")
        assert out["d"]["balance"] == BALANCE and "floor" not in out["d"]
        assert {h["id"] for h in out["d"]["hits"]} == {1}, "the unchecked cut still exports its hits"

    def test_routed_scoring(self):
        medias = _patch_medias()
        info = {"mlp": _fires_on_dim1_mlp(), "threshold": 0.5, "balance": BALANCE}
        out = cli._score_one_detector(
            "d", info, medias, scoring_rows_for_snap(medias, "dinov3_patch"), {cid: cid for cid in medias}
        )
        assert out["balance"] == BALANCE and "floor" not in out
        assert {h["id"] for h in out["hits"]} == {1}

    def test_a_detector_with_no_state_reports_none(self):
        info = {"mlp": _fires_on_dim1_mlp(), "threshold": 0.5}
        out = cli._score_direct_all(["d"], {"d": info}, _patch_medias(), "dinov3_patch")
        assert out["d"]["balance"] is None
