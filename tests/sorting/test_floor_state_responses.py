"""Whether a line is a promise rides with the line, wherever it leaves the process (#4247).

When the precision floor can promise nothing - too few calibration positives,
or a floor this corpus cannot reach - the line falls back to the Inclusion 0
cut and every consumer keeps working on it.  What changes is that the consumer
is told: every response that carries a detector's ``threshold`` carries the
floor's verdict on it beside it, built by
:func:`vtscore.state.core.detector_floor_state`, and a headless run records
the unpromised set in its log and its payload.

The verdict's four values are pinned in
``tests_lib/sorting/test_precision_floor_wiring.py``; here each carrier is
pinned to report it.  States a real retrain here cannot reach (the fixture's
labels carry no learned-sort provenance, so a floor never has evidence) are
planted on the context with :func:`~tests.helpers.planted_precision_floor_estimate`.
"""

from __future__ import annotations

import logging

import pytest

from tests import load_detector_and_wait
from tests.helpers import planted_precision_floor_estimate, setup_trainable_model_in_registry
from vtscore.state.core import get_active_detector_context
from vtsearch.state import bad_votes, good_votes, set_min_precision, snapshot_medias

#: ``(min_precision, positives per planted fold, the status it reports)``.
STATES = [
    pytest.param(0.5, 8, "promised", id="promised"),
    pytest.param(1.0, 8, "unreachable", id="unreachable"),
    pytest.param(0.5, 3, "insufficient_evidence", id="insufficient_evidence"),
    pytest.param(None, 8, None, id="no-floor"),
]


def _load_detector(client, name: str = "floor-carrier") -> str:
    detector_id = setup_trainable_model_in_registry(
        name, good_ids=[1, 2, 3, 4, 5, 6], bad_ids=[7, 8, 9, 10, 11, 12], snap=snapshot_medias()
    )
    load_detector_and_wait(client, detector_id)
    return detector_id


class TestTheInclusionKnob:
    @pytest.mark.parametrize(("min_precision", "n_pos", "status"), STATES)
    def test_get_and_post_report_the_verdict_on_the_line_they_return(self, client, min_precision, n_pos, status):
        ctx = get_active_detector_context()
        ctx.precision_floor_cache = planted_precision_floor_estimate(n_pos_per_fold=n_pos)
        set_min_precision(min_precision)

        for data in (
            client.get("/api/inclusion").get_json(),
            client.post("/api/inclusion", json={"inclusion": 2}).get_json(),
        ):
            assert data["floor"] == {
                "min_precision": min_precision,
                "status": status,
                "calibration_positives": 2 * n_pos,
            }

    def test_no_trained_detector_is_no_evidence(self, client):
        set_min_precision(0.5)
        floor = client.get("/api/inclusion").get_json()["floor"]
        assert floor == {"min_precision": 0.5, "status": "insufficient_evidence", "calibration_positives": 0}


class TestFindLabel:
    @pytest.mark.parametrize(("min_precision", "status"), [(0.5, "insufficient_evidence"), (None, None)])
    def test_carries_the_verdict_on_its_threshold(self, client, min_precision, status):
        detector_id = _load_detector(client)
        set_min_precision(min_precision)

        data = client.post("/api/find-label", json={"detector_id": detector_id}).get_json()

        assert data["floor"]["status"] == status
        assert data["floor"]["min_precision"] == min_precision
        # The unpromised line is the Inclusion 0 cut, not an empty result.
        ctx = get_active_detector_context()
        if status == "insufficient_evidence":
            assert data["threshold"] == round(ctx.anchored_cut_cache.threshold_at(0), 4)
            assert any(r["score"] >= data["threshold"] for r in data["results"])


class TestLearnedSort:
    @pytest.mark.parametrize(("min_precision", "status"), [(0.5, "insufficient_evidence"), (None, None)])
    def test_the_done_payload_carries_the_verdict(self, client, min_precision, status):
        set_min_precision(min_precision)
        good_votes.update({k: None for k in [1, 2, 3]})
        bad_votes.update({k: None for k in [18, 19, 20]})

        data = client.post("/api/learned-sort", json={"wait": True}).get_json()

        assert data["floor"]["status"] == status
        assert data["floor"]["min_precision"] == min_precision
        assert data["threshold"] is not None

    def test_a_text_sort_has_no_line_and_no_verdict(self, client):
        data = client.post("/api/sort", json={"text": "a dog barking"}).get_json()
        assert data.get("floor") is None


class TestAutoRun:
    def _autorun(self, client, monkeypatch, n_pos: int | None) -> dict:
        """One Auto-Find detector, its trained context's estimate planted when *n_pos* is given."""
        import vtsearch.routes.detectors.scoring as scoring_mod
        from vtsearch.settings import add_autofind_detector

        setup_trainable_model_in_registry(
            "floor-autorun", good_ids=[1, 2, 3], bad_ids=[18, 19, 20], snap=snapshot_medias()
        )
        add_autofind_detector("floor-autorun")
        if n_pos is not None:
            real = scoring_mod.resolve_or_train_detector

            def _planting(*args, ctx_sink=None, **kwargs):
                trained: list = []
                out = real(*args, ctx_sink=trained, **kwargs)
                trained[0].precision_floor_cache = planted_precision_floor_estimate(n_pos_per_fold=n_pos)
                ctx_sink.extend(trained)
                return out

            monkeypatch.setattr(scoring_mod, "resolve_or_train_detector", _planting)

        resp = client.post("/api/auto-detect", json={})
        assert resp.status_code == 200, resp.get_json()
        return resp.get_json()["results"]["floor-autorun"]

    @pytest.mark.parametrize(("min_precision", "n_pos", "status"), STATES)
    def test_each_result_carries_the_verdict(self, client, monkeypatch, caplog, min_precision, n_pos, status):
        set_min_precision(min_precision)
        with caplog.at_level(logging.INFO, logger="vtsearch.routes.detectors.scoring"):
            result = self._autorun(client, monkeypatch, n_pos)

        assert result["floor"] == {
            "min_precision": min_precision,
            "status": status,
            "calibration_positives": 2 * n_pos,
        }
        # The exporters still take one float threshold, and the hits are cut at it.
        assert isinstance(result["threshold"], float)
        assert all(h["score"] >= result["threshold"] for h in result["hits"])
        unpromised = [r for r in caplog.records if "makes no" in r.getMessage()]
        if status in ("unreachable", "insufficient_evidence"):
            assert len(unpromised) == 1
            assert "floor-autorun" in unpromised[0].getMessage()
            assert status in unpromised[0].getMessage()
        else:
            assert unpromised == []

    def test_a_real_train_reports_through_the_context_sink(self, client, monkeypatch, caplog):
        """No planting: the context ``resolve_or_train_detector`` trained is the one asked."""
        set_min_precision(0.5)
        with caplog.at_level(logging.INFO, logger="vtsearch.routes.detectors.scoring"):
            result = self._autorun(client, monkeypatch, None)
        assert result["floor"]["status"] == "insufficient_evidence"
        assert result["hits"], "an unpromised run still exports the Inclusion 0 set"
        assert any("makes no 50% promise" in r.getMessage() for r in caplog.records)
