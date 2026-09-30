"""Planted-answer tests for the precision floor's spot check: the band walk (#4272, #4388).

The rule the owner ruled on #4383, priced in
``docs/experiments/2026-09-30-line-estimate-4383`` (``grow-fine`` in
``analyze_line_estimate_4383.py`` is the reference; the eval/app sync gate
pins the two against each other): the bands of a ranking (8, 8, 16, 32, ...),
the schedule that says where a walk starts and what a band costs, the
band-weighted estimate and the likely range built from each band's interval,
the walk (deeper while the set meets the floor, shallower while it does not,
stop on the first reversal), a ranking that stays fixed while the model
retrains, and the line that keeps the set the walk ended on - or the unchecked
starting candidate before any check, which is what a headless run exports.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vtscore.training.thresholds import (
    BAND_BASE,
    CHECK_ALPHA,
    CHECK_BASE_CANDIDATE,
    CHECK_CANCELLED,
    CHECK_MIN_PICKS,
    CHECK_PROVENANCE,
    FLOOR_CONFIRMED,
    FLOOR_SHORT,
    FLOOR_UNCHECKED,
    WALK_DEEPER,
    WALK_SHALLOWER,
    WALK_START,
    LineRanking,
    SpotCheck,
    applicable_result,
    band_edges,
    bands_for,
    check_schedule,
    clopper_pearson_lower,
    clopper_pearson_upper,
    floor_count,
    floor_line,
    floor_state,
    likely_range,
    line_under,
    range_tail,
    rounds_for,
)
from vtscore.utils.scores import NON_FINITE_SCORE_SENTINEL

#: (starting candidate, bands the walk audits before its first verdict, picks a band).
PRESETS = {0.10: (128, 5, 5), 0.25: (64, 4, 5), 0.50: (32, 3, 5), 0.75: (32, 3, 5), 0.90: (32, 3, 5)}


def _ranking(n: int = 200, voted: set[int] | None = None, seed: int = 42) -> LineRanking:
    """*n* items scored on a strict descending ladder, so rank == id - 1."""
    scores = np.linspace(0.99, 0.01, n)
    return LineRanking.from_scores(list(range(1, n + 1)), scores, voted or set())


def _unvoted(r: LineRanking, also_voted: set[int] | None = None) -> tuple[int, ...]:
    return tuple(int(i) for i in r.unvoted_ids(also_voted or set()))


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
        got = {x: (s.candidate, s.rounds, s.picks) for x, s in ((x, check_schedule(x)) for x in PRESETS)}
        assert got == PRESETS

    def test_the_candidate_doubles_as_the_floor_halves_and_the_bands_follow(self):
        for x, (k, rounds, _m) in PRESETS.items():
            assert k == CHECK_BASE_CANDIDATE * 2 ** max(0, math.floor(math.log2(0.5 / x) + 1e-9))
            assert rounds == rounds_for(k) == bands_for(k, band_edges(k))

    def test_every_band_costs_the_same_five_picks(self):
        assert all(check_schedule(x).picks == CHECK_MIN_PICKS for x in (0.1, 0.25, 0.5, 0.75, 0.9))

    def test_a_floor_of_one_censuses_each_band(self):
        assert check_schedule(1.0).picks == BAND_BASE
        check = SpotCheck.start(_unvoted(_ranking()), 1.0, seed=1)
        assert len(check.pending) == BAND_BASE and set(check.pending) == _top(8)

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_a_floor_outside_the_unit_interval_is_refused(self, bad):
        with pytest.raises(ValueError):
            check_schedule(bad)


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
        check = SpotCheck.start(_unvoted(r), 0.5, seed=1, start_count=16)
        assert check.bands == 2 and check.band == 0
        check.record({cid: True for cid in check.pending})
        assert check.band == 1
        second = check.pending
        check.record({cid: i < 3 for i, cid in enumerate(second)})
        assert check.finished
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
        assert floor_line(r, 0.5) == line_under(r.score_of(32))
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
    def test_it_starts_at_the_bands_that_hold_the_floors_count_and_deals_the_top_band(self):
        check = SpotCheck.start(_unvoted(_ranking()), 0.5, seed=3)
        assert (check.start_k, check.bands, check.k, check.picks) == (32, 3, 32, 5)
        assert check.edges == (0, 8, 16, 32, 64, 128, 200)
        assert check.band == 0 and set(check.pending) <= _top(8) and len(check.pending) == 5
        assert check.round == 0 and check.running and check.direction == WALK_START
        deep = SpotCheck.start(_unvoted(_ranking()), 0.1, seed=3)
        assert (deep.start_k, deep.bands) == (128, 5)

    def test_it_audits_each_starting_band_in_turn_before_its_first_verdict(self):
        check = SpotCheck.start(_unvoted(_ranking()), 0.5, seed=5)
        seen: list[int] = []
        for expected_band in (0, 1, 2):
            assert check.band == expected_band
            seen += check.pending
            check.record({cid: True for cid in check.pending})
        assert check.round == 3
        # The starting set met the floor: the walk went one band deeper, and only now.
        assert check.direction == WALK_DEEPER and check.bands == 4 and check.k == 64 and check.band == 3
        assert set(check.pending) <= set(range(33, 65)) and not set(check.pending) & set(seen)

    def test_it_grows_while_the_set_meets_the_floor_and_stops_on_the_first_reversal(self):
        """Positives are the top 40: the set of 64 is at least half right, the set of 128 is not."""
        check = _finish(SpotCheck.start(_unvoted(_ranking()), 0.5, seed=7), right_ids=_top(40))
        assert check.status == FLOOR_CONFIRMED and check.k == 64 and check.best == 4
        assert check.direction == WALK_SHALLOWER, "it stepped back from the set that fell short"
        # Five bands audited (the three it started with, then 64 and 128), 5 picks each.
        assert check.round == 5 and len(check.labels) == 25
        assert check.estimate() >= 0.5 and check.range().labelled == 20, "the range describes the kept set only"

    def test_it_walks_to_the_end_of_a_ranking_that_is_all_right(self):
        check = _finish(SpotCheck.start(_unvoted(_ranking()), 0.5, seed=7))
        assert check.status == FLOOR_CONFIRMED and check.k == 200 and check.bands == 6
        assert check.round == 6 and check.direction == WALK_DEEPER

    def test_it_shrinks_while_the_set_falls_short_and_ends_on_the_first_band(self):
        check = _finish(SpotCheck.start(_unvoted(_ranking()), 0.5, seed=5), right_ids=set())
        assert check.status == FLOOR_SHORT and check.k == BAND_BASE and check.bands == 1
        assert check.round == 3, "shrinking needs no new picks: the bands were audited already"
        assert check.direction == WALK_SHALLOWER and check.best is None

    def test_it_settles_one_band_shallower_when_the_start_falls_short_and_a_smaller_set_meets_it(self):
        """Positives are the top 12: the top 32 is under half right, the top 16 is not."""
        check = _finish(SpotCheck.start(_unvoted(_ranking()), 0.5, seed=9), right_ids=_top(12))
        assert check.status == FLOOR_CONFIRMED and check.k == 16 and check.round == 3

    def test_a_walk_can_start_where_a_caller_proposes(self):
        check = SpotCheck.start(_unvoted(_ranking()), 0.5, seed=1, start_count=100)
        assert (check.start_k, check.bands, check.k) == (128, 5, 128)
        tiny = SpotCheck.start(_unvoted(_ranking()), 0.5, seed=1, start_count=3)
        assert (tiny.start_k, tiny.bands, tiny.k) == (8, 1, 8)

    def test_the_ranking_never_changes_after_the_start(self):
        """The model may retrain behind the check; every band samples the one list fixed at the start."""
        ranking = _ranking()
        check = SpotCheck.start(_unvoted(ranking), 0.25, seed=9)
        fixed = check.ranking_ids
        moved = LineRanking.from_scores(ranking.ids.tolist(), ranking.scores[::-1], set())
        check.record({cid: False for cid in check.pending})
        assert check.ranking_ids == fixed
        assert set(check.pending) <= set(fixed[8:16])
        assert _unvoted(moved) != fixed, "the retrained ranking would have said otherwise"

    def test_a_partial_round_waits_and_a_stray_vote_is_refused(self):
        check = SpotCheck.start(_unvoted(_ranking()), 0.5, seed=2)
        first, rest = check.pending[0], check.pending[1:]
        assert check.record({first: True}) is False
        assert check.pending == rest and check.running
        with pytest.raises(ValueError, match="not this round"):
            check.record({999: True})
        assert check.record({cid: True for cid in rest}) is True
        assert check.band == 1, "the next band the starting set owes"

    def test_a_finished_check_takes_no_more_votes_and_a_cancelled_one_is_neither_state(self):
        done = _finish(SpotCheck.start(_unvoted(_ranking()), 0.5, seed=2))
        with pytest.raises(ValueError):
            done.record({})
        with pytest.raises(ValueError):
            done.draw()
        cancelled = SpotCheck.start(_unvoted(_ranking()), 0.5, seed=2)
        cancelled.cancel()
        assert cancelled.status == CHECK_CANCELLED and not cancelled.running and not cancelled.finished
        assert applicable_result(0.5, cancelled) is None

    def test_a_small_corpus_is_one_band_and_a_census(self):
        tiny = SpotCheck.start(_unvoted(_ranking(3)), 0.5, seed=4)
        assert tiny.edges == (0, 3) and tiny.bands == 1 and len(tiny.pending) == 3
        tiny.record({cid: True for cid in tiny.pending})
        assert tiny.status == FLOOR_CONFIRMED and tiny.k == 3
        assert tiny.range().lo == tiny.range().hi == 1.0
        five = SpotCheck.start(_unvoted(_ranking(5)), 0.9, seed=1)
        five.record({cid: cid != 5 for cid in five.pending})
        assert (five.range().lo, five.range().hi) == (0.8, 0.8)
        assert five.status == FLOOR_SHORT and five.k == 5

    def test_a_ranking_must_have_distinct_ids(self):
        with pytest.raises(ValueError):
            SpotCheck.start((1, 1, 2), 0.5)
        with pytest.raises(ValueError):
            SpotCheck.start((), 0.5)

    def test_the_picks_are_seeded_and_the_provenance_is_the_apps(self):
        a = SpotCheck.start(_unvoted(_ranking()), 0.1, seed=11).pending
        b = SpotCheck.start(_unvoted(_ranking()), 0.1, seed=11).pending
        assert a == b
        assert CHECK_PROVENANCE == {"flow": "check"}

    def test_the_state_a_client_sees(self):
        check = SpotCheck.start(_unvoted(_ranking()), 0.25, seed=6)
        state = check.as_dict()
        assert state["status"] == "running" and state["picks"] == list(check.pending)
        assert (state["round"], state["rounds"], state["picks_per_round"]) == (1, 6, 5)
        assert (state["candidate"], state["start_candidate"], state["bands"]) == (64, 64, 4)
        assert state["band"] == {"index": 0, "lo": 1, "hi": 8} and state["direction"] == WALK_START
        assert state["range"] is None and state["labelled"] == 0 and state["estimate"] is None
        _finish(check, right_ids=_top(40))
        state = check.as_dict()
        assert state["status"] == FLOOR_CONFIRMED and state["picks"] == [] and state["band"] is None
        # At 25% the top 128 (40 right of 128) meets the floor and the top 200 does not.
        assert state["candidate"] == 128 and state["direction"] == WALK_SHALLOWER
        assert state["range"]["labelled"] == 25 and 0.25 <= state["estimate"] <= 1.0


class TestTheLine:
    def test_before_any_check_the_line_keeps_the_unchecked_starting_candidate(self):
        """What a headless run exports: the top 128 at 10%, 64 at 25%, 32 at 50% and above."""
        r = _ranking(voted={1})
        for x, (k, _r, _m) in PRESETS.items():
            assert floor_count(x, None) == k
            assert floor_line(r, x) == r.threshold_for(k)
            state = floor_state(x, None, r)
            assert (state.status, state.count, state.range, state.stale) == (FLOOR_UNCHECKED, k, None, False)
            assert state.as_dict()["schedule"] == {"candidate": k, "rounds": _r, "picks": _m}

    def test_a_corpus_smaller_than_the_candidate_reports_what_it_has(self):
        r = _ranking(10, voted={1, 2})
        state = floor_state(0.5, None, r)
        assert state.count == 8 and floor_line(r, 0.5) == r.threshold_for(8) == r.score_of(10)
        assert floor_state(0.5, None, None).count == 32, "with no ranking the schedule's count is all there is"
        assert floor_line(None, 0.5) is None

    def test_a_confirmed_check_keeps_the_set_the_walk_ended_on(self):
        r = _ranking()
        check = _finish(SpotCheck.start(_unvoted(r), 0.25, seed=8), right_ids=_top(40))
        assert check.status == FLOOR_CONFIRMED and check.k == 128, "at 25% the set of 128 is right enough"
        check.fingerprint = r.fingerprint(128, set(check.labels))
        assert floor_count(0.25, check) == 128
        assert floor_line(r, 0.25, check, set(check.labels)) == r.threshold_for(128, set(check.labels))
        state = floor_state(0.25, check, r, set(check.labels))
        assert state.status == FLOOR_CONFIRMED and state.count == 128 and not state.stale
        assert state.range is not None and state.range.labelled == 25

    def test_a_short_check_keeps_the_first_band_it_ended_on(self):
        r = _ranking()
        check = _finish(SpotCheck.start(_unvoted(r), 0.1, seed=8), right_ids=set())
        assert check.status == FLOOR_SHORT and check.k == BAND_BASE
        check.fingerprint = r.fingerprint(BAND_BASE, set(check.labels))
        state = floor_state(0.1, check, r, set(check.labels))
        assert state.status == FLOOR_SHORT and state.count == BAND_BASE
        assert state.range is not None and state.range.hi < 0.6
        assert floor_line(r, 0.1, check, set(check.labels)) == r.threshold_for(BAND_BASE, set(check.labels))

    def test_a_result_belongs_to_its_floor(self):
        r = _ranking()
        check = _finish(SpotCheck.start(_unvoted(r), 0.5, seed=8), right_ids=_top(40))
        assert applicable_result(0.5, check) is check
        assert applicable_result(0.25, check) is None
        assert floor_state(0.25, check, r).status == FLOOR_UNCHECKED
        assert floor_count(0.25, check) == 64

    def test_a_finished_result_goes_stale_when_the_set_under_it_moves(self):
        r = _ranking()
        check = _finish(SpotCheck.start(_unvoted(r), 0.5, seed=8), right_ids=_top(40))
        assert check.k == 64
        voted = set(check.labels)
        check.fingerprint = r.fingerprint(64, voted)
        assert not floor_state(0.5, check, r, voted).stale
        # A later vote inside the set: the top 64 unvoted shifts by one.
        later = voted | {next(cid for cid in r.candidate(64, voted))}
        assert floor_state(0.5, check, r, later).stale
        # A retrain that reorders the ranking.
        retrained = LineRanking.from_scores(r.ids.tolist(), r.scores[::-1], set())
        assert floor_state(0.5, check, retrained, voted).stale
        # A vote far below the line leaves the set, and so the range, alone.
        below = voted | {200}
        assert not floor_state(0.5, check, r, below).stale
        assert floor_line(r, 0.5, check, later) == r.threshold_for(64, later), "the line follows at the same count"
