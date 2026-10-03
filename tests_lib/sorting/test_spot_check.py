"""Planted-answer tests for the spot check's machinery: the band walk (#4272, #4388, #4413).

The bands the owner ruled on #4383, priced in
``docs/experiments/2026-09-30-line-estimate-4383`` (``grow-fine`` in
``analyze_line_estimate_4383.py`` is the reference; the eval/app sync gate
pins the two against each other): the bands of a ranking (8, 8, 16, 32, ...),
the schedule that says where a walk starts and what a band costs, the
band-weighted estimate and the likely range built from each band's interval,
the rounds a walk deals, and a ranking that stays fixed while the model
retrains.  Where the walk stops - the F-beta peak - and the line it leaves are
pinned in ``test_balance.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vtscore.training.thresholds import (
    BALANCE_CHECKED,
    CHECK_ALPHA,
    CHECK_BASE_CANDIDATE,
    CHECK_CANCELLED,
    CHECK_MIN_PICKS,
    CHECK_PROVENANCE,
    CHECK_RECALL_CANDIDATE,
    WALK_DEEPER,
    WALK_START,
    LineRanking,
    SpotCheck,
    applicable_balance,
    balance_schedule,
    band_edges,
    bands_for,
    clopper_pearson_lower,
    clopper_pearson_upper,
    likely_range,
    line_under,
    range_tail,
    rounds_for,
)
from vtscore.utils.scores import NON_FINITE_SCORE_SENTINEL

#: (starting candidate, bands the walk audits before its first verdict, picks a band) at each preset balance.
PRESETS = {0.5: (32, 3, 5), 1.0: (32, 3, 5), 2.0: (128, 5, 5)}


def _ranking(n: int = 200, voted: set[int] | None = None, seed: int = 42) -> LineRanking:
    """*n* items scored on a strict descending ladder, so rank == id - 1."""
    scores = np.linspace(0.99, 0.01, n)
    return LineRanking.from_scores(list(range(1, n + 1)), scores, voted or set())


def _unvoted(r: LineRanking, also_voted: set[int] | None = None) -> tuple[int, ...]:
    return tuple(int(i) for i in r.unvoted_ids(also_voted or set()))


def _start(ids: tuple[int, ...], beta: float = 1.0, n_pos: float = 52.0, **kwargs) -> SpotCheck:
    """A balance walk over *ids*; *n_pos* is the walk's count of the ranking's positives."""
    return SpotCheck.start_balance(ids, beta, n_pos, **kwargs)


def _finish(check: SpotCheck, right_ids: set[int] | None = None) -> SpotCheck:
    """Vote every band until the walk ends: an item is right iff it is in *right_ids* (default: all)."""
    while check.running:
        check.record({cid: (right_ids is None or cid in right_ids) for cid in check.pending})
    return check


def _top(n: int) -> set[int]:
    """The ids of the top *n* ranks of :func:`_ranking`: rank == id - 1."""
    return set(range(1, n + 1))


class TestTheBands:
    def test_the_bands_double_from_eight_and_the_last_runs_to_the_end(self):
        assert band_edges(200) == (0, 8, 16, 32, 64, 128, 200)
        assert band_edges(128) == (0, 8, 16, 32, 64, 128)
        assert band_edges(5) == (0, 5)
        assert band_edges(0) == (0,)

    def test_bands_for_holds_the_count_in_the_fewest_bands(self):
        edges = band_edges(200)
        assert [bands_for(k, edges) for k in (1, 8, 9, 16, 32, 64, 128, 129, 200, 500)] == [
            1,
            1,
            2,
            2,
            3,
            4,
            5,
            6,
            6,
            6,
        ]

    def test_rounds_for_is_the_bands_that_hold_the_candidate(self):
        assert rounds_for(8) == 1
        assert rounds_for(16) == 2
        assert rounds_for(32) == 3
        assert rounds_for(50) == 4
        assert rounds_for(100) == 5
        assert rounds_for(128) == 5


class TestTheSchedule:
    def test_the_presets_resolve_as_the_owner_ruled_them(self):
        got = {b: (s.candidate, s.rounds, s.picks) for b, s in ((b, balance_schedule(b)) for b in PRESETS)}
        assert got == PRESETS

    def test_the_caps_are_the_precision_floors_counts_at_50_and_10_percent(self):
        """#4267's schedule, ``32 * 2**max(0, floor(log2(0.5 / P)))``, at the floor each preset leans toward."""
        for beta, (k, rounds, _m) in PRESETS.items():
            floor = 0.5 if beta <= 1 else 0.1
            assert k == CHECK_BASE_CANDIDATE * 2 ** max(0, math.floor(math.log2(0.5 / floor) + 1e-9))
            assert rounds == rounds_for(k) == bands_for(k, band_edges(k))
        assert (CHECK_BASE_CANDIDATE, CHECK_RECALL_CANDIDATE) == (32, 128)

    def test_every_band_costs_the_same_five_picks(self):
        assert all(balance_schedule(b).picks == CHECK_MIN_PICKS for b in (0.25, 0.5, 1.0, 2.0, 4.0))


class TestTheLikelyRange:
    def test_each_tail_splits_alpha_over_the_sets_bands(self):
        assert range_tail(1) == CHECK_ALPHA
        assert range_tail(3) == pytest.approx(CHECK_ALPHA / 3)

    def test_a_census_is_exact(self):
        rng = likely_range(right=20, labelled=32, candidate=32, tail=0.05)
        assert (rng.lo, rng.hi) == (20 / 32, 20 / 32)

    def test_no_labels_bound_nothing(self):
        rng = likely_range(0, 0, 32, 0.05)
        assert (rng.lo, rng.hi) == (0.0, 1.0)

    def test_the_bounds_are_clopper_pearson_and_nest_with_the_tail(self):
        wide = likely_range(3, 5, 32, range_tail(3))
        plain = likely_range(3, 5, 32, range_tail(1))
        assert wide.lo == clopper_pearson_lower(3, 5, range_tail(3))
        assert wide.hi == clopper_pearson_upper(3, 5, range_tail(3))
        assert wide.lo < plain.lo < 0.6 < plain.hi < wide.hi
        assert clopper_pearson_lower(0, 5, 0.05) == 0.0 and clopper_pearson_upper(5, 5, 0.05) == 1.0

    def test_a_sets_range_is_its_bands_intervals_weighted_by_size(self):
        """Two bands of 8: one censused at 100%, one with 5 picks 3 right at the tail alpha / 2."""
        r = _ranking(16)
        # 13 positives: the walk's F1 at both bands (12.8 right of 16) beats the first band's (8 of 8).
        check = _start(_unvoted(r), n_pos=13.0, seed=1, start_count=16)
        assert check.bands == 2 and check.band == 0
        check.record({cid: True for cid in check.pending})
        assert check.band == 1
        second = check.pending
        check.record({cid: i < 3 for i, cid in enumerate(second)})
        assert check.finished and check.k == 16
        tail = range_tail(2)
        band0 = likely_range(5, 5, 8, tail)
        band1 = likely_range(3, 5, 8, tail)
        rng = check.range()
        assert rng.lo == pytest.approx((8 * band0.lo + 8 * band1.lo) / 16)
        assert rng.hi == pytest.approx((8 * band0.hi + 8 * band1.hi) / 16)
        assert (rng.labelled, rng.right) == (10, 8)
        assert check.estimate() == pytest.approx((8 * 1.0 + 8 * 0.6) / 16)


class TestTheRanking:
    def test_sorted_by_score_then_id_with_unscorable_items_dropped(self):
        r = LineRanking.from_scores([3, 1, 2, 4], [0.5, 0.9, 0.5, NON_FINITE_SCORE_SENTINEL], set())
        assert r.ids.tolist() == [1, 2, 3]
        assert r.scores.tolist() == [0.9, 0.5, 0.5]

    def test_the_candidate_is_the_top_unvoted_in_rank_order(self):
        r = _ranking(voted={2})
        assert r.candidate(3) == (1, 3, 4)
        assert r.candidate(3, also_voted={1}) == (3, 4, 5)
        assert r.candidate(500) == tuple(i for i in range(1, 201) if i != 2)

    def test_the_line_sits_just_under_the_last_item_of_the_set(self):
        r = _ranking()
        assert r.threshold_for(32) == line_under(r.score_of(32))
        assert r.threshold_for(500) == line_under(r.score_of(200))
        assert LineRanking.from_scores([1], [0.5], {1}).threshold_for(32) is None

    @pytest.mark.parametrize("score", [0.4657, 0.12345, 0.9999, 0.0001])
    def test_the_kept_item_clears_its_own_line_raw_or_rounded(self, score):
        assert line_under(score) <= score
        assert line_under(score) <= round(score, 4)
        assert round(line_under(score), 4) == line_under(score)

    def test_above_counts_voted_items_too(self):
        r = _ranking(voted={1, 2})
        assert r.above(r.score_of(10)) == 10

    def test_the_fingerprint_is_the_sets_identity(self):
        r = _ranking()
        assert r.fingerprint(3) == (1, 2, 3)
        assert r.fingerprint(3, {1}) == (2, 3, 4)

    def test_ids_and_scores_must_align(self):
        with pytest.raises(ValueError):
            LineRanking.from_scores([1, 2], [0.5], set())


class TestTheWalk:
    def test_it_starts_at_the_bands_that_hold_the_balances_cap_and_deals_the_top_band(self):
        check = _start(_unvoted(_ranking()), seed=3)
        assert (check.start_k, check.bands, check.k, check.picks) == (32, 3, 32, 5)
        assert check.edges == (0, 8, 16, 32, 64, 128, 200)
        assert check.band == 0 and set(check.pending) <= _top(8) and len(check.pending) == 5
        assert check.round == 0 and check.running and check.direction == WALK_START
        deep = _start(_unvoted(_ranking()), beta=2.0, seed=3)
        assert (deep.start_k, deep.bands) == (128, 5)

    def test_it_audits_each_starting_band_in_turn_before_its_first_verdict(self):
        check = _start(_unvoted(_ranking()), seed=5)
        seen: list[int] = []
        for expected_band in (0, 1, 2):
            assert check.band == expected_band
            seen += check.pending
            check.record({cid: True for cid in check.pending})
        assert check.round == 3
        # The starting set's estimate is the first to compare: the walk went one band deeper, and only now.
        assert check.direction == WALK_DEEPER and check.bands == 4 and check.k == 64 and check.band == 3
        assert set(check.pending) <= set(range(33, 65)) and not set(check.pending) & set(seen)

    def test_a_walk_can_start_where_a_caller_proposes(self):
        check = _start(_unvoted(_ranking()), seed=1, start_count=100)
        assert (check.start_k, check.bands, check.k) == (128, 5, 128)
        tiny = _start(_unvoted(_ranking()), seed=1, start_count=3)
        assert (tiny.start_k, tiny.bands, tiny.k) == (8, 1, 8)

    def test_the_ranking_never_changes_after_the_start(self):
        """The model may retrain behind the check; every band samples the one list fixed at the start."""
        ranking = _ranking()
        check = _start(_unvoted(ranking), beta=2.0, seed=9)
        fixed = check.ranking_ids
        moved = LineRanking.from_scores(ranking.ids.tolist(), ranking.scores[::-1], set())
        check.record({cid: False for cid in check.pending})
        assert check.ranking_ids == fixed
        assert set(check.pending) <= set(fixed[8:16])
        assert _unvoted(moved) != fixed, "the retrained ranking would have said otherwise"

    def test_a_partial_round_waits_and_a_stray_vote_is_refused(self):
        check = _start(_unvoted(_ranking()), seed=2)
        first, rest = check.pending[0], check.pending[1:]
        assert check.record({first: True}) is False
        assert check.pending == rest and check.running
        with pytest.raises(ValueError, match="not this round"):
            check.record({999: True})
        assert check.record({cid: True for cid in rest}) is True
        assert check.band == 1, "the next band the starting set owes"

    def test_a_finished_check_takes_no_more_votes_and_a_cancelled_one_is_neither_state(self):
        done = _finish(_start(_unvoted(_ranking()), seed=2))
        with pytest.raises(ValueError):
            done.record({})
        with pytest.raises(ValueError):
            done.draw()
        cancelled = _start(_unvoted(_ranking()), seed=2)
        cancelled.cancel()
        assert cancelled.status == CHECK_CANCELLED and not cancelled.running and not cancelled.finished
        assert applicable_balance(1.0, cancelled) is None

    def test_a_small_corpus_is_one_band_and_a_census(self):
        tiny = _start(_unvoted(_ranking(3)), n_pos=3.0, seed=4)
        assert tiny.edges == (0, 3) and tiny.bands == 1 and len(tiny.pending) == 3
        tiny.record({cid: True for cid in tiny.pending})
        assert tiny.status == BALANCE_CHECKED and tiny.k == 3
        assert tiny.range().lo == tiny.range().hi == 1.0
        five = _start(_unvoted(_ranking(5)), n_pos=4.0, seed=1)
        five.record({cid: cid != 5 for cid in five.pending})
        assert (five.range().lo, five.range().hi) == (0.8, 0.8)
        assert five.status == BALANCE_CHECKED and five.k == 5

    def test_a_ranking_must_have_distinct_ids(self):
        with pytest.raises(ValueError):
            _start((1, 1, 2))
        with pytest.raises(ValueError):
            _start(())

    def test_the_picks_are_seeded_and_the_provenance_is_the_apps(self):
        a = _start(_unvoted(_ranking()), beta=2.0, seed=11).pending
        b = _start(_unvoted(_ranking()), beta=2.0, seed=11).pending
        assert a == b
        assert CHECK_PROVENANCE == {"flow": "check"}

    def test_the_state_a_client_sees(self):
        check = _start(_unvoted(_ranking()), seed=6)
        state = check.as_dict()
        assert state["status"] == "running" and state["picks"] == list(check.pending)
        assert (state["round"], state["rounds"], state["picks_per_round"]) == (1, 6, 5)
        assert (state["candidate"], state["start_candidate"], state["bands"]) == (32, 32, 3)
        assert state["band"] == {"index": 0, "lo": 1, "hi": 8} and state["direction"] == WALK_START
        assert state["range"] is None and state["labelled"] == 0 and state["estimate"] is None
        assert state["beta"] == 1.0 and state["fbeta"] is None and state["recall"] is None
        assert "min_precision" not in state
        _finish(check, right_ids=_top(40))
        state = check.as_dict()
        assert state["status"] == BALANCE_CHECKED and state["picks"] == [] and state["band"] is None
        assert state["fbeta"] is not None and state["range"]["labelled"] >= 15 and 0.0 <= state["estimate"] <= 1.0
