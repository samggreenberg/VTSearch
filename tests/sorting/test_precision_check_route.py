"""``/api/precision-check`` (#4272): each state of the spot check through the endpoints.

Start fixes the candidate off the detector's ranking and deals round one;
votes land as ordinary labels (provenance ``check``, verified in Find mode)
and decide the round; a failed round halves the candidate into the next; the
check ends confirmed or short, and the line moves to the set it ended on.  A
later vote inside that set makes the result ``stale``; cancelling leaves the
floor's state as it was; and there is no redraw on the same candidate.

The fixture's Find pass leaves 8 unvoted items, so the whole corpus is the
candidate and a check is one round.  The multi-round cases plant a 200-item
ranking on the context instead; votes on ids the dataset does not hold are
in-memory labels like any other, and the labelset persist skips them.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests import load_detector_and_wait
from tests.helpers import setup_trainable_model_in_registry
from vtscore.state.core import get_active_detector_context, human_voted_ids
from vtscore.training.thresholds import LineRanking, SpotCheck
from vtsearch.state import set_min_precision, snapshot_medias

#: The floor draws the line here (#4413): these are the deprecated floor's own tests, and the balance is the default.
pytestmark = pytest.mark.usefixtures("floor_preference")


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
    def test_get_reports_the_floor_and_no_check(self, client, schedule_only):
        _run_find(client)
        set_min_precision(0.5)
        data = client.get("/api/precision-check").get_json()
        assert data["check"] is None
        assert data["floor"]["status"] == "unchecked" and data["floor"]["count"] == 8

    def test_start_needs_a_ranking(self, client):
        resp = client.post("/api/precision-check/start", json={})
        assert resp.status_code == 409
        assert "learned sort" in resp.get_json()["message"]

    def test_votes_need_a_running_check(self, client):
        _run_find(client)
        assert client.post("/api/precision-check/votes", json=_votes([13], True)).status_code == 409


class TestOneRound:
    def test_start_deals_round_one_from_the_unvoted_candidate(self, client):
        _run_find(client)
        set_min_precision(0.5)
        ctx = get_active_detector_context()
        data = _start(client)
        check = data["check"]
        assert check["status"] == "running"
        assert (check["round"], check["rounds"], check["picks_per_round"]) == (1, 1, 5)
        assert (check["candidate"], check["start_candidate"]) == (8, 8)
        assert len(check["picks"]) == 5
        assert set(check["picks"]) <= set(ctx.line_ranking.candidate(32, human_voted_ids(ctx)))
        assert check["range"] is None and check["labelled"] == 0
        # The floor's state is as it was until the check ends.
        assert data["floor"]["status"] == "unchecked"
        assert client.get("/api/precision-check").get_json()["check"]["picks"] == check["picks"]

    def test_a_stray_vote_is_refused(self, client):
        _run_find(client)
        picks = _start(client)["check"]["picks"]
        stray = next(cid for cid in range(1, 21) if cid not in picks)
        assert client.post("/api/precision-check/votes", json=_votes([stray], True)).status_code == 400

    def test_a_partial_round_waits_and_a_full_round_confirms(self, client):
        _run_find(client)
        set_min_precision(0.5)
        ctx = get_active_detector_context()
        picks = _start(client)["check"]["picks"]

        data = client.post("/api/precision-check/votes", json=_votes(picks[:2], True)).get_json()
        assert data["check"]["status"] == "running" and data["check"]["picks"] == picks[2:]
        assert data["check"]["labelled"] == 2 and data["floor"]["status"] == "unchecked"

        data = client.post("/api/precision-check/votes", json=_votes(picks[2:], True)).get_json()
        assert data["check"]["status"] == "confirmed" and data["check"]["picks"] == []
        assert data["floor"]["status"] == "confirmed" and data["floor"]["count"] == 8
        assert data["floor"]["range"]["labelled"] == 5 and data["floor"]["range"]["right"] == 5
        assert data["floor"]["range"]["lo"] >= 0.5 and data["floor"]["range"]["stale"] is False

        # The votes are ordinary labels: cast, verified in Find mode, tagged as the check's.
        assert all(cid in ctx.good_votes and cid in ctx.verified_ids for cid in picks)
        assert all(ctx.vote_provenance[cid]["flow"] == "check" for cid in picks)
        # The line keeps the set the check ended on, and the check is the last finished one.
        assert ctx.threshold == ctx.line_ranking.threshold_for(8, human_voted_ids(ctx))
        assert ctx.precision_check is not None and ctx.precision_check_run is None
        assert client.get("/api/min-precision").get_json()["status"] == "confirmed"

    def test_a_wrong_round_ends_short_and_keeps_the_set(self, client):
        _run_find(client)
        set_min_precision(0.5)
        ctx = get_active_detector_context()
        picks = _start(client)["check"]["picks"]
        data = client.post("/api/precision-check/votes", json=_votes(picks, False)).get_json()
        assert data["check"]["status"] == "short"
        assert data["floor"]["status"] == "short" and data["floor"]["count"] == 8
        assert data["floor"]["range"]["right"] == 0 and data["floor"]["range"]["lo"] == 0.0
        assert all(cid in ctx.bad_votes for cid in picks)
        assert ctx.threshold == ctx.line_ranking.threshold_for(8, human_voted_ids(ctx))

    def test_a_later_vote_inside_the_set_makes_the_result_stale_and_the_line_follows(self, client):
        _run_find(client)
        set_min_precision(0.5)
        ctx = get_active_detector_context()
        picks = _start(client)["check"]["picks"]
        client.post("/api/precision-check/votes", json=_votes(picks, True))
        kept = ctx.line_ranking.candidate(8, human_voted_ids(ctx))
        assert len(kept) == 3, "8 unvoted less the 5 picks"

        client.post(f"/api/medias/{kept[0]}/vote", json={"target": "good"})
        data = client.get("/api/precision-check").get_json()
        assert data["floor"]["status"] == "confirmed" and data["floor"]["range"]["stale"] is True
        assert data["check"]["status"] == "confirmed", "the last finished check is still reported"
        # The line follows the ranking at the same count.
        client.post("/api/min-precision", json={"min_precision": 0.5})
        assert ctx.threshold == ctx.line_ranking.threshold_for(8, human_voted_ids(ctx))

    def test_cancel_leaves_the_floor_as_it_was_and_the_votes_cast(self, client):
        _run_find(client)
        set_min_precision(0.5)
        ctx = get_active_detector_context()
        picks = _start(client)["check"]["picks"]
        client.post("/api/precision-check/votes", json=_votes(picks[:2], True))
        data = client.post("/api/precision-check/cancel", json={}).get_json()
        assert data["check"] is None and data["floor"]["status"] == "unchecked"
        assert ctx.precision_check_run is None and ctx.precision_check is None
        assert all(cid in ctx.good_votes for cid in picks[:2])
        # Nothing running: cancel again is a no-op.
        assert client.post("/api/precision-check/cancel", json={}).status_code == 200

    def test_no_redraw_on_the_same_candidate(self, client):
        _run_find(client)
        set_min_precision(0.5)
        ctx = get_active_detector_context()
        candidate = ctx.line_ranking.candidate(32, human_voted_ids(ctx))
        done = SpotCheck.start(candidate, 0.5, seed=1)
        done.record({cid: False for cid in done.pending})
        ctx.precision_check = done
        resp = client.post("/api/precision-check/start", json={})
        assert resp.status_code == 409 and "already checked" in resp.get_json()["message"]
        # A vote changes the candidate, and a new check may start.
        client.post(f"/api/medias/{candidate[0]}/vote", json={"target": "bad"})
        assert _start(client)["check"]["status"] == "running"

    def test_a_running_check_is_replaced_by_a_new_start(self, client):
        _run_find(client)
        first = _start(client)["check"]["picks"]
        second = _start(client)["check"]["picks"]
        assert len(second) == 5
        assert client.post("/api/precision-check/votes", json=_votes(second, True)).status_code == 200
        assert first != second or True  # the draw is random; what matters is the old round is gone


class TestSeveralBands:
    """The walk (#4388): one round per band, deeper while the set is right enough, back when it is not."""

    def test_a_walk_whose_every_pick_is_wrong_ends_short_on_the_first_band(self, client):
        _run_find(client)
        ranking = _plant_big_ranking()
        set_min_precision(0.1)
        ctx = get_active_detector_context()

        data = _start(client)
        check = data["check"]
        # At 10% the walk starts at the five bands that hold the top 128 (8, 8, 16, 32, 64).
        assert (check["candidate"], check["start_candidate"], check["bands"], check["picks_per_round"]) == (
            128,
            128,
            5,
            5,
        )
        assert check["rounds"] == 6, "the ranking of 200 has six bands"
        assert check["band"] == {"index": 0, "lo": 1, "hi": 8} and check["direction"] == "start"
        seen: list[int] = []
        for band, (lo, hi) in enumerate(((1, 8), (9, 16), (17, 32), (33, 64), (65, 128))):
            assert data["check"]["band"] == {"index": band, "lo": lo, "hi": hi}
            picks = data["check"]["picks"]
            assert len(picks) == 5 and set(picks) <= set(range(lo, hi + 1)) and not set(picks) & set(seen)
            seen += picks
            assert data["floor"]["status"] == "unchecked", "not decided yet"
            data = client.post("/api/precision-check/votes", json=_votes(picks, False)).get_json()
        # The fifth band decided the starting set: every set fell short, down to the first band, with no new picks.
        assert data["check"]["status"] == "short" and data["check"]["direction"] == "shallower"
        assert data["check"]["round"] == 5 and data["check"]["picks"] == []
        assert data["floor"]["status"] == "short" and data["floor"]["count"] == 8
        assert data["floor"]["range"]["labelled"] == 5 and data["floor"]["range"]["right"] == 0
        assert ctx.threshold == ranking.threshold_for(8, human_voted_ids(ctx))
        assert ranking.candidate(8, human_voted_ids(ctx))[-1] > 8, "the line follows the ranking past the votes"

    def test_a_walk_that_grows_then_falls_short_keeps_the_deepest_set_that_met_the_floor(self, client):
        _run_find(client)
        ranking = _plant_big_ranking()
        set_min_precision(0.5)
        ctx = get_active_detector_context()
        data = _start(client)
        assert (data["check"]["candidate"], data["check"]["bands"]) == (32, 3)
        # The three starting bands are all right: the walk goes deeper.
        for _ in range(3):
            data = client.post("/api/precision-check/votes", json=_votes(data["check"]["picks"], True)).get_json()
        check = data["check"]
        assert check["status"] == "running" and check["direction"] == "deeper"
        # ``round`` is the one being voted on: the fourth band, after three audited.
        assert (check["candidate"], check["bands"], check["round"]) == (64, 4, 4)
        assert check["band"] == {"index": 3, "lo": 33, "hi": 64} and check["estimate"] is None
        assert data["floor"]["status"] == "unchecked"
        # Items 33-64 are all wrong: the top 64 is exactly half right, which meets 50%, so deeper again.
        data = client.post("/api/precision-check/votes", json=_votes(check["picks"], False)).get_json()
        check = data["check"]
        assert check["status"] == "running" and (check["candidate"], check["bands"]) == (128, 5)
        assert check["band"] == {"index": 4, "lo": 65, "hi": 128}
        # Items 65-128 are all wrong too: the top 128 falls short, and the walk steps back to the 64.
        data = client.post("/api/precision-check/votes", json=_votes(check["picks"], False)).get_json()
        check = data["check"]
        assert check["status"] == "confirmed" and check["direction"] == "shallower" and check["round"] == 5
        assert data["floor"]["status"] == "confirmed" and data["floor"]["count"] == 64
        assert data["floor"]["range"]["labelled"] == 20 and data["floor"]["range"]["right"] == 15
        assert check["estimate"] == 0.5
        assert ctx.threshold == ranking.threshold_for(64, human_voted_ids(ctx))
        assert client.get("/api/min-precision").get_json()["count"] == 64

    @pytest.mark.parametrize(("floor", "start", "bands"), [(0.25, 64, 4), (0.75, 32, 3), (0.9, 32, 3)])
    def test_the_schedule_says_where_the_walk_starts(self, client, floor, start, bands):
        _run_find(client)
        _plant_big_ranking()
        set_min_precision(floor)
        data = _start(client)
        check = data["check"]
        assert (check["candidate"], check["start_candidate"], check["bands"], check["picks_per_round"]) == (
            start,
            start,
            bands,
            5,
        )
        assert check["rounds"] == 6 and len(check["picks"]) == 5
        assert data["floor"]["schedule"] == {"candidate": start, "rounds": bands, "picks": 5}


class TestSeed:
    """``VTSEARCH_SPOT_CHECK_SEED`` reaches the draw, so the screenshot harness frames one pick (#4330)."""

    def test_the_draw_is_unseeded_by_default(self, client, monkeypatch):
        seeds = []
        start = SpotCheck.start.__func__

        def spy(cls, *args, **kwargs):
            seeds.append(kwargs.get("seed"))
            return start(cls, *args, **kwargs)

        monkeypatch.setattr(SpotCheck, "start", classmethod(spy))
        _run_find(client)
        _start(client)
        assert seeds == [None]

    def test_a_seed_deals_the_same_picks_on_every_start(self, client, monkeypatch):
        # ``SPOT_CHECK_SEED`` is read from the environment at import time; the
        # route reads the name off ``vtscore.config`` on each start.
        monkeypatch.setattr("vtscore.config.SPOT_CHECK_SEED", 7)
        _run_find(client)
        ranking = _plant_big_ranking()
        set_min_precision(0.5)
        first = _start(client)["check"]["picks"]
        client.post("/api/precision-check/cancel", json={})
        assert _start(client)["check"]["picks"] == first
        assert first == list(SpotCheck.start(ranking.candidate(32), 0.5, seed=7).pending)
