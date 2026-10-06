"""``/api/precision-check`` (#4272, #4413): each state of the balance's spot check through the endpoints.

Start fixes the unvoted ranking off the detector's current one and deals the
first band's picks; votes land as ordinary labels (provenance ``check``,
verified in Find mode) and decide the band; the walk ends ``checked``, and
the balance's state reports what it audited.  A later vote inside the audited
set makes the result ``stale``; cancelling leaves the balance's state as it
was; and there is no redraw on the same list.  How the two check shapes move
the line is pinned in ``test_balance_routes.py``.

The fixture's Find pass leaves 8 unvoted items, so the whole corpus is one
band and a check is one round.  The multi-band cases plant a 200-item ranking
on the context instead; votes on ids the dataset does not hold are in-memory
labels like any other, and the labelset persist skips them.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests import load_detector_and_wait
from tests.helpers import setup_trainable_model_in_registry
from vtscore.state.core import detector_walk_positives, get_active_detector_context, human_voted_ids
from vtscore.training.thresholds import BALANCE_CHECKED, CHECK_TRIM, LineRanking, SpotCheck
from vtsearch.state import set_beta, snapshot_medias


def _run_find(client) -> None:
    detector_id = setup_trainable_model_in_registry(
        "check-route", good_ids=[1, 2, 3, 4, 5, 6], bad_ids=[7, 8, 9, 10, 11, 12], snap=snapshot_medias()
    )
    load_detector_and_wait(client, detector_id)
    client.post("/api/find-label", json={"detector_id": detector_id})


def _plant_big_ranking(n: int = 200) -> LineRanking:
    """A 200-item ranking on the active context, ids 1..n on a descending ladder, nothing voted."""
    ctx = get_active_detector_context()
    ctx.line_ranking = LineRanking.from_scores(list(range(1, n + 1)), np.linspace(0.99, 0.01, n), set())
    ctx.verified_ids.clear()
    return ctx.line_ranking


def _votes(picks, right) -> dict:
    return {"votes": [{"id": cid, "label": "good" if right else "bad"} for cid in picks]}


def _start(client) -> dict:
    resp = client.post("/api/precision-check/start", json={})
    assert resp.status_code == 200, resp.get_json()
    return resp.get_json()


class TestBeforeAnyCheck:
    def test_get_reports_the_balance_and_no_check(self, client):
        _run_find(client)
        data = client.get("/api/precision-check").get_json()
        assert data["check"] is None
        assert data["balance"]["status"] == "unchecked" and data["balance"]["beta"] == 1.0
        assert "floor" not in data

    def test_start_needs_a_ranking(self, client):
        resp = client.post("/api/precision-check/start", json={})
        assert resp.status_code == 409
        assert "learned sort" in resp.get_json()["message"]

    def test_votes_need_a_running_check(self, client):
        _run_find(client)
        assert client.post("/api/precision-check/votes", json=_votes([13], True)).status_code == 409


class TestOneRound:
    def test_start_deals_round_one_from_the_unvoted_ranking(self, client):
        _run_find(client)
        ctx = get_active_detector_context()
        data = _start(client)
        check = data["check"]
        assert check["status"] == "running" and check["beta"] == 1.0
        assert (check["round"], check["rounds"], check["picks_per_round"]) == (1, 1, 5)
        assert (check["candidate"], check["start_candidate"]) == (8, 8)
        assert len(check["picks"]) == 5
        assert set(check["picks"]) <= set(ctx.line_ranking.candidate(32, human_voted_ids(ctx)))
        assert check["range"] is None and check["recall"] is None and check["labelled"] == 0
        # The balance's state is as it was until the check ends.
        assert data["balance"]["status"] == "unchecked"
        assert client.get("/api/precision-check").get_json()["check"]["picks"] == check["picks"]

    def test_a_stray_vote_is_refused(self, client):
        _run_find(client)
        picks = _start(client)["check"]["picks"]
        stray = next(cid for cid in range(1, 21) if cid not in picks)
        assert client.post("/api/precision-check/votes", json=_votes([stray], True)).status_code == 400

    def test_a_partial_round_waits_and_a_full_round_ends_the_walk(self, client):
        _run_find(client)
        ctx = get_active_detector_context()
        picks = _start(client)["check"]["picks"]

        data = client.post("/api/precision-check/votes", json=_votes(picks[:2], True)).get_json()
        assert data["check"]["status"] == "running" and data["check"]["picks"] == picks[2:]
        assert data["check"]["labelled"] == 2 and data["balance"]["status"] == "unchecked"

        data = client.post("/api/precision-check/votes", json=_votes(picks[2:], True)).get_json()
        assert data["check"]["status"] == BALANCE_CHECKED and data["check"]["picks"] == []
        assert data["balance"]["status"] == BALANCE_CHECKED and data["balance"]["audited"] == 8
        assert data["balance"]["precision"]["labelled"] == 5 and data["balance"]["precision"]["right"] == 5
        assert data["balance"]["precision"]["stale"] is False

        # The votes are ordinary labels: cast, verified in Find mode, tagged as the check's.
        assert all(cid in ctx.good_votes and cid in ctx.verified_ids for cid in picks)
        assert all(ctx.vote_provenance[cid]["flow"] == "check" for cid in picks)
        # The line keeps the set the balance keeps, and the check is the last finished one.
        assert ctx.threshold == ctx.line_ranking.threshold_for(data["balance"]["count"], human_voted_ids(ctx))
        assert ctx.precision_check is not None and ctx.precision_check_run is None
        assert client.get("/api/balance").get_json()["status"] == BALANCE_CHECKED
        # The weak-separation prompt counts its cooldown from the vote total the check ended at (#4496).
        assert ctx.check_ended_votes == len(human_voted_ids(ctx))

    def test_a_later_vote_inside_the_set_makes_the_result_stale(self, client):
        _run_find(client)
        ctx = get_active_detector_context()
        picks = _start(client)["check"]["picks"]
        client.post("/api/precision-check/votes", json=_votes(picks, True))
        kept = ctx.line_ranking.candidate(8, human_voted_ids(ctx))
        assert len(kept) == 3, "8 unvoted less the 5 picks"

        client.post(f"/api/medias/{kept[0]}/vote", json={"target": "good"})
        data = client.get("/api/precision-check").get_json()
        assert data["balance"]["status"] == BALANCE_CHECKED and data["balance"]["precision"]["stale"] is True
        assert data["check"]["status"] == BALANCE_CHECKED, "the last finished check is still reported"

    def test_cancel_leaves_the_balance_as_it_was_and_the_votes_cast(self, client):
        _run_find(client)
        ctx = get_active_detector_context()
        picks = _start(client)["check"]["picks"]
        client.post("/api/precision-check/votes", json=_votes(picks[:2], True))
        data = client.post("/api/precision-check/cancel", json={}).get_json()
        assert data["check"] is None and data["balance"]["status"] == "unchecked"
        assert ctx.precision_check_run is None and ctx.precision_check is None
        assert all(cid in ctx.good_votes for cid in picks[:2])
        # A closed check ended too: the weak-separation prompt counts its cooldown from here (#4496).
        assert ctx.check_ended_votes == len(human_voted_ids(ctx))
        # Nothing running: cancel again is a no-op.
        assert client.post("/api/precision-check/cancel", json={}).status_code == 200

    def test_no_redraw_on_the_same_list(self, client):
        _run_find(client)
        ctx = get_active_detector_context()
        unvoted = tuple(int(i) for i in ctx.line_ranking.unvoted_ids(human_voted_ids(ctx)))
        done = SpotCheck.start_balance(unvoted, 1.0, detector_walk_positives(ctx, 1.0), seed=1)
        done.record({cid: False for cid in done.pending})
        assert done.finished
        ctx.precision_check = done
        resp = client.post("/api/precision-check/start", json={})
        assert resp.status_code == 409 and "already checked" in resp.get_json()["message"]
        # A vote changes the list, and a new check may start.
        client.post(f"/api/medias/{unvoted[0]}/vote", json={"target": "bad"})
        assert _start(client)["check"]["status"] == "running"

    def test_a_running_check_is_replaced_by_a_new_start(self, client):
        _run_find(client)
        _start(client)
        second = _start(client)["check"]["picks"]
        assert len(second) == 5
        assert client.post("/api/precision-check/votes", json=_votes(second, True)).status_code == 200


class TestSeveralBands:
    """The walk (#4388): one round per band, starting at the bands that hold the balance's cap."""

    @pytest.mark.parametrize(("beta", "start", "bands"), [(0.5, 32, 3), (1.0, 32, 3), (2.0, 128, 5)])
    def test_the_schedule_says_where_the_walk_starts(self, client, beta, start, bands):
        _run_find(client)
        _plant_big_ranking()
        set_beta(beta)
        data = _start(client)
        check = data["check"]
        assert check["beta"] == beta
        assert (check["candidate"], check["start_candidate"], check["bands"], check["picks_per_round"]) == (
            start,
            start,
            bands,
            5,
        )
        assert check["rounds"] == 6 and len(check["picks"]) == 5, "the ranking of 200 has six bands"
        assert data["balance"]["schedule"] == {"candidate": start, "rounds": bands, "picks": 5}

    def test_a_trim_walk_whose_every_pick_is_wrong_ends_on_the_first_band(self, client):
        _run_find(client)
        ranking = _plant_big_ranking()
        set_beta(2.0)
        ctx = get_active_detector_context()

        data = _start(client)
        assert data["balance"]["shape"] == CHECK_TRIM
        assert data["check"]["band"] == {"index": 0, "lo": 1, "hi": 8} and data["check"]["direction"] == "start"
        seen: list[int] = []
        for band, (lo, hi) in enumerate(((1, 8), (9, 16), (17, 32), (33, 64), (65, 128))):
            assert data["check"]["band"] == {"index": band, "lo": lo, "hi": hi}
            picks = data["check"]["picks"]
            assert len(picks) == 5 and set(picks) <= set(range(lo, hi + 1)) and not set(picks) & set(seen)
            seen += picks
            assert data["balance"]["status"] == "unchecked", "not decided yet"
            data = client.post("/api/precision-check/votes", json=_votes(picks, False)).get_json()
        # The fifth band decided the starting set: the walk stepped shallower to the first band, with no new picks.
        assert data["check"]["status"] == BALANCE_CHECKED and data["check"]["direction"] == "shallower"
        assert data["check"]["round"] == 5 and data["check"]["picks"] == []
        assert data["balance"]["status"] == BALANCE_CHECKED and data["balance"]["count"] == 8
        assert data["balance"]["audited"] == 8
        assert ctx.threshold == ranking.threshold_for(8, human_voted_ids(ctx))


class TestSeed:
    """``VTSEARCH_SPOT_CHECK_SEED`` reaches the draw, so the screenshot harness frames one pick (#4330)."""

    def test_the_draw_is_unseeded_by_default(self, client, monkeypatch):
        seeds = []
        start = SpotCheck.start_balance.__func__

        def spy(cls, *args, **kwargs):
            seeds.append(kwargs.get("seed"))
            return start(cls, *args, **kwargs)

        monkeypatch.setattr(SpotCheck, "start_balance", classmethod(spy))
        _run_find(client)
        _start(client)
        assert seeds == [None]

    def test_a_seed_deals_the_same_picks_on_every_start(self, client, monkeypatch):
        # ``SPOT_CHECK_SEED`` is read from the environment at import time; the
        # route reads the name off ``vtscore.config`` on each start.
        monkeypatch.setattr("vtscore.config.SPOT_CHECK_SEED", 7)
        _run_find(client)
        ranking = _plant_big_ranking()
        first = _start(client)["check"]["picks"]
        client.post("/api/precision-check/cancel", json={})
        assert _start(client)["check"]["picks"] == first
        n_pos = detector_walk_positives(get_active_detector_context(), 1.0)
        unvoted = tuple(int(i) for i in ranking.unvoted_ids())
        assert first == list(SpotCheck.start_balance(unvoted, 1.0, n_pos, seed=7).pending)
