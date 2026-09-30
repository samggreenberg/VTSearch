"""The floor's state rides with the line, wherever it leaves the process (#4247, #4272).

Every response that carries a detector's ``threshold`` carries the floor's
state beside it, built by :func:`vtscore.state.core.detector_floor_state`:
``unchecked`` before a spot check has run at this floor, with the size of the
starting candidate the line keeps; ``confirmed`` or ``short`` after one, with
the set the check ended on and its likely range, flagged ``stale`` once the
ranking under it moved.  A headless run records the unchecked set in its log
and its payload.  The state's values are pinned in
``tests_lib/sorting/test_precision_floor_wiring.py``; here each carrier is
pinned to report it.

The fixture's labelset resolves 12 of the 20 media as voted, so the starting
candidate at any preset is the 8 unvoted items - and the line keeps all of
them.  A finished check is planted with :func:`~tests.helpers.planted_spot_check`.
"""

from __future__ import annotations

import logging

import pytest

from tests import load_detector_and_wait
from tests.helpers import planted_spot_check, setup_trainable_model_in_registry
from vtscore.state.core import get_active_detector_context, human_voted_ids
from vtsearch.state import bad_votes, good_votes, set_min_precision, snapshot_medias

UNCHECKED = {"status": "unchecked", "range": None}
SCHEDULES = {
    0.1: {"candidate": 128, "rounds": 5, "picks": 5},
    0.5: {"candidate": 32, "rounds": 3, "picks": 5},
    0.9: {"candidate": 32, "rounds": 3, "picks": 5},
}


def _load_detector(client, name: str = "floor-carrier") -> str:
    detector_id = setup_trainable_model_in_registry(
        name, good_ids=[1, 2, 3, 4, 5, 6], bad_ids=[7, 8, 9, 10, 11, 12], snap=snapshot_medias()
    )
    load_detector_and_wait(client, detector_id)
    return detector_id


def _run_find(client, name: str = "floor-carrier") -> dict:
    detector_id = _load_detector(client, name)
    return client.post("/api/find-label", json={"detector_id": detector_id}).get_json()


class TestTheFloorRoute:
    @pytest.mark.parametrize("min_precision", sorted(SCHEDULES))
    def test_get_and_post_report_the_unchecked_starting_candidate(self, client, min_precision):
        _run_find(client)
        set_min_precision(min_precision)
        ctx = get_active_detector_context()
        for data in (
            client.get("/api/min-precision").get_json(),
            client.post("/api/min-precision", json={"min_precision": min_precision}).get_json(),
        ):
            assert {k: data[k] for k in ("min_precision", "status", "range", "count")} == {
                "min_precision": min_precision,
                **UNCHECKED,
                # 8 unvoted items: the whole corpus is the candidate at every preset.
                "count": 8,
            }
            assert data["schedule"] == SCHEDULES[min_precision]
            assert data["threshold"] == ctx.threshold == ctx.line_ranking.threshold_for(8, human_voted_ids(ctx))
            assert data["n_returned"] == ctx.line_ranking.above(ctx.threshold)

    def test_a_finished_check_reports_its_set_and_range_and_goes_stale(self, client):
        _run_find(client)
        set_min_precision(0.5)
        ctx = get_active_detector_context()
        check = planted_spot_check(ctx, 0.5)
        assert check.status == "confirmed" and check.k == 8

        data = client.get("/api/min-precision").get_json()
        assert data["status"] == "confirmed" and data["count"] == 8
        assert data["range"] == {
            "lo": pytest.approx(0.5493, abs=1e-3),
            "hi": 1.0,
            "labelled": 5,
            "right": 5,
            "stale": False,
        }
        assert data["threshold"] == ctx.threshold == ctx.line_ranking.threshold_for(8, human_voted_ids(ctx))

        # A later vote inside the set moves the set under the result: only the flag changes.
        later = next(cid for cid in ctx.line_ranking.candidate(8, human_voted_ids(ctx)))
        client.post(f"/api/medias/{later}/vote", json={"target": "bad"})
        data = client.get("/api/min-precision").get_json()
        assert data["status"] == "confirmed" and data["range"]["stale"] is True
        assert data["range"]["lo"] == pytest.approx(0.5493, abs=1e-3)
        # A result belongs to its floor: another floor is unchecked until checked.
        assert client.post("/api/min-precision", json={"min_precision": 0.9}).get_json()["status"] == "unchecked"
        assert client.post("/api/min-precision", json={"min_precision": 0.5}).get_json()["status"] == "confirmed"


class TestFindLabel:
    @pytest.mark.parametrize("min_precision", [0.5, 1.0])
    def test_carries_the_unchecked_starting_candidate_and_keeps_it(self, client, min_precision):
        detector_id = _load_detector(client)
        set_min_precision(min_precision)

        data = client.post("/api/find-label", json={"detector_id": detector_id}).get_json()

        assert {k: data["floor"][k] for k in ("status", "range", "count", "min_precision")} == {
            **UNCHECKED,
            "count": 8,
            "min_precision": min_precision,
        }
        ctx = get_active_detector_context()
        unvoted = ctx.line_ranking.candidate(32, human_voted_ids(ctx))
        assert len(unvoted) == 8
        # The line sits at the last item of the set: every unvoted item clears it.
        assert data["threshold"] == round(ctx.line_ranking.threshold_for(8), 4)
        assert all(r["score"] >= data["threshold"] for r in data["results"] if r["id"] in unvoted)


class TestLearnedSort:
    @pytest.mark.parametrize("min_precision", [0.5, 1.0])
    def test_the_done_payload_carries_the_state(self, client, min_precision):
        set_min_precision(min_precision)
        good_votes.update({k: None for k in [1, 2, 3]})
        bad_votes.update({k: None for k in [18, 19, 20]})

        data = client.post("/api/learned-sort", json={"wait": True}).get_json()

        assert data["floor"]["status"] == "unchecked" and data["floor"]["range"] is None
        assert data["floor"]["min_precision"] == min_precision
        # 20 media, 6 voted: the 14 unvoted are the candidate, and the line keeps them all.
        assert data["floor"]["count"] == 14
        assert data["threshold"] is not None
        unvoted = [r for r in data["results"] if r["id"] not in (1, 2, 3, 18, 19, 20)]
        assert all(r["score"] >= data["threshold"] for r in unvoted)

    def test_a_finished_check_moves_the_line_the_next_sort_draws(self, client):
        set_min_precision(0.5)
        good_votes.update({k: None for k in [1, 2, 3]})
        bad_votes.update({k: None for k in [18, 19, 20]})
        client.post("/api/learned-sort", json={"wait": True})
        ctx = get_active_detector_context()
        check = planted_spot_check(ctx, 0.5, right=False)
        # 14 unvoted items make two bands (8 and 6); every pick wrong ends short on the first.
        assert check.status == "short" and check.k == 8

        data = client.post("/api/learned-sort", json={"wait": True}).get_json()
        assert data["floor"]["status"] == "short" and data["floor"]["count"] == 8
        assert data["floor"]["range"]["labelled"] == 5 and data["floor"]["range"]["right"] == 0
        assert data["threshold"] == round(ctx.line_ranking.threshold_for(8, human_voted_ids(ctx)), 4)

    def test_a_text_sort_has_no_line_and_no_verdict(self, client):
        data = client.post("/api/sort", json={"text": "a dog barking"}).get_json()
        assert data.get("floor") is None


class TestAutoRun:
    def _autorun(self, client) -> dict:
        from vtsearch.settings import add_autofind_detector

        setup_trainable_model_in_registry(
            "floor-autorun", good_ids=[1, 2, 3], bad_ids=[18, 19, 20], snap=snapshot_medias()
        )
        add_autofind_detector("floor-autorun")
        resp = client.post("/api/auto-detect", json={})
        assert resp.status_code == 200, resp.get_json()
        return resp.get_json()["results"]["floor-autorun"]

    @pytest.mark.parametrize("min_precision", sorted(SCHEDULES))
    def test_each_result_exports_the_unchecked_starting_candidate(self, client, caplog, min_precision):
        """Nobody can vote in a headless run, so it exports the starting candidate and says so."""
        set_min_precision(min_precision)
        with caplog.at_level(logging.INFO, logger="vtsearch.autorun_detectors"):
            result = self._autorun(client)

        assert result["floor"] == {
            "min_precision": min_precision,
            **UNCHECKED,
            "count": 14,
            "schedule": SCHEDULES[min_precision],
        }
        # The exporters still take one float threshold, and the hits are cut at it.
        assert isinstance(result["threshold"], float)
        assert all(h["score"] >= result["threshold"] for h in result["hits"])
        assert result["total_hits"] >= 14, "the unchecked set is exported whole"
        unchecked = [r for r in caplog.records if "unchecked" in r.getMessage()]
        assert len(unchecked) == 1
        assert "floor-autorun exports its top 14 unchecked" in unchecked[0].getMessage()
        assert f"{100 * min_precision:.0f}% right" in unchecked[0].getMessage()


class TestFindStats:
    def test_the_stats_carry_the_state(self, client):
        _run_find(client)
        set_min_precision(0.5)
        data = client.get("/api/find/stats").get_json()
        assert data["floor"]["status"] == "unchecked" and data["floor"]["count"] == 8
        assert data["floor"]["schedule"] == SCHEDULES[0.5]
        ctx = get_active_detector_context()
        planted_spot_check(ctx, 0.5)
        data = client.get("/api/find/stats").get_json()
        assert data["floor"]["status"] == "confirmed" and data["floor"]["range"]["right"] == 5
        assert data["threshold"] == round(ctx.threshold, 4)
