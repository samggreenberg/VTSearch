"""A CLI autodetect run records when a detector's cut carries no precision promise (#4247).

When the precision floor can promise nothing, the detector is still scored - at
its Inclusion 0 cut - and that set is what gets exported.  The run says so
twice: a ``detector_unpromised`` progress event (a line on the terminal in text
mode), and a ``floor`` entry beside ``threshold`` in every result the detector
produces, which the JSON exporters write out verbatim.  ``threshold`` itself
stays one float, the one the hits were cut at.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch

import vtscore.cli as cli
from tests.cli.test_cli_score_geometry import _fires_on_dim1_mlp, _patch_medias
from tests.helpers import planted_precision_floor_estimate
from vtscore import cli_progress
from vtscore.detectors.training import scoring_rows_for_snap
from vtscore.state import set_min_precision

#: ``(min_precision, positives per planted fold or None for no estimate, status)``.
STATES = [
    pytest.param(0.5, 8, "promised", id="promised"),
    pytest.param(1.0, 8, "unreachable", id="unreachable"),
    pytest.param(0.5, 3, "insufficient_evidence", id="insufficient_evidence"),
    pytest.param(0.5, None, "insufficient_evidence", id="no-estimate"),
    pytest.param(None, 8, None, id="no-floor"),
]

UNPROMISED = ("unreachable", "insufficient_evidence")


def _train(monkeypatch, n_pos: int | None) -> dict[str, dict[str, Any]]:
    """``_load_and_train_detectors`` for one detector, its estimate planted when *n_pos* is given."""
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
        if n_pos is not None:
            det_ctx.precision_floor_cache = planted_precision_floor_estimate(n_pos_per_fold=n_pos)
        return True

    monkeypatch.setattr(lt, "train_from_labelset", _fake_train)
    return cli._load_and_train_detectors(["det"], "image", {1: {"id": 1, "media_type": "image"}})


def _events(capsys) -> list[dict]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]


class TestTrainingRecordsTheVerdict:
    @pytest.mark.parametrize(("min_precision", "n_pos", "status"), STATES)
    def test_each_state(self, client, monkeypatch, capsys, min_precision, n_pos, status):
        set_min_precision(min_precision)
        cli_progress.set_format("json")

        out = _train(monkeypatch, n_pos)

        positives = 2 * n_pos if n_pos is not None else 0
        assert out["det"]["floor"] == {
            "min_precision": min_precision,
            "status": status,
            "calibration_positives": positives,
        }
        assert out["det"]["threshold"] == 0.5, "the cut is still the one training drew"
        unpromised = [e for e in _events(capsys) if e["event"] == "detector_unpromised"]
        if status in UNPROMISED:
            assert len(unpromised) == 1
            event = unpromised[0]
            assert event["detector"] == "det"
            assert (event["min_precision"], event["status"], event["calibration_positives"]) == (
                min_precision,
                status,
                positives,
            )
        else:
            assert unpromised == []

    def test_text_mode_says_what_the_export_is(self, client, monkeypatch, capsys):
        set_min_precision(1.0)
        _train(monkeypatch, 8)
        out = capsys.readouterr().out
        assert "Detector 'det' makes no 100% precision promise (unreachable, 16 calibration positives)" in out
        assert "exporting its Inclusion 0 cut" in out


FLOOR = {"min_precision": 0.5, "status": "insufficient_evidence", "calibration_positives": 4}


class TestResultsCarryTheVerdict:
    def test_direct_scoring(self):
        info = {"mlp": _fires_on_dim1_mlp(), "threshold": 0.5, "floor": FLOOR}
        out = cli._score_direct_all(["d"], {"d": info}, _patch_medias(), "dinov3_patch")
        assert out["d"]["floor"] == FLOOR
        assert {h["id"] for h in out["d"]["hits"]} == {1}, "the unpromised cut still exports its hits"

    def test_routed_scoring(self):
        medias = _patch_medias()
        info = {"mlp": _fires_on_dim1_mlp(), "threshold": 0.5, "floor": FLOOR}
        out = cli._score_one_detector(
            "d", info, medias, scoring_rows_for_snap(medias, "dinov3_patch"), {cid: cid for cid in medias}
        )
        assert out["floor"] == FLOOR
        assert {h["id"] for h in out["hits"]} == {1}

    def test_a_detector_with_no_verdict_reports_none(self):
        info = {"mlp": _fires_on_dim1_mlp(), "threshold": 0.5}
        out = cli._score_direct_all(["d"], {"d": info}, _patch_medias(), "dinov3_patch")
        assert out["d"]["floor"] is None
