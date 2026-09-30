"""Planted-answer tests for the precision floor's spot check (#4272).

The rule the owner priced in ``docs/experiments/2026-09-29-floor-candidate-4267``
(``analyze_floor_candidate_4267.py`` is the reference; the eval/app sync gate
pins the two against each other): the schedule ``(K, R, m)`` at the presets,
the likely range and its tails, the rounds (halve on a failed round, keep the
labels inside the new candidate, no redraw on the same one), a candidate that
stays fixed while the model retrains, and the line that keeps the set the
check ended on - or the unchecked starting candidate before any check, which
is what a headless run exports.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vtscore.training.thresholds import (
    CHECK_ALPHA,
    CHECK_BASE_CANDIDATE,
    CHECK_CANCELLED,
    CHECK_MIN_PICKS,
    CHECK_PROVENANCE,
    FLOOR_CONFIRMED,
    FLOOR_SHORT,
    FLOOR_UNCHECKED,
    LineRanking,
    SpotCheck,
    applicable_result,
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

PRESETS = {0.10: (128, 3, 5), 0.25: (64, 2, 5), 0.50: (32, 1, 5), 0.75: (32, 1, 11), 0.90: (32, 1, 29)}


def _ranking(n: int = 200, voted: set[int] | None = None, seed: int = 42) -> LineRanking:
    """*n* items scored on a strict descending ladder, so rank == id - 1."""
    scores = np.linspace(0.99, 0.01, n)
    return LineRanking.from_scores(list(range(1, n + 1)), scores, voted or set())


def _finish(check: SpotCheck, right_ids: set[int] | None = None) -> SpotCheck:
    """Vote every round until the check ends: an item is right iff it is in *right_ids* (default: all)."""
    while check.running:
        check.record({cid: (right_ids is None or cid in right_ids) for cid in check.pending})
    return check


class TestTheSchedule:
    def test_the_presets_resolve_as_the_owner_priced_them(self):
        got = {x: (s.candidate, s.rounds, s.picks) for x, s in ((x, check_schedule(x)) for x in PRESETS)}
        assert got == PRESETS

    def test_the_candidate_doubles_as_the_floor_halves_and_the_rounds_follow(self):
        for x, (k, rounds, _m) in PRESETS.items():
            assert k == CHECK_BASE_CANDIDATE * 2 ** max(0, math.floor(math.log2(0.5 / x) + 1e-9))
            assert rounds == int(math.log2(k / CHECK_BASE_CANDIDATE)) + 1
            assert rounds_for(k) == rounds

    @pytest.mark.parametrize("x", [0.5, 0.6, 0.75, 0.9, 0.95])
    def test_at_one_round_the_picks_are_the_fewest_that_can_reach_the_floor(self, x):
        """m all-positive picks reach X at level alpha, and when m > 5, m - 1 do not."""
        m = check_schedule(x).picks
        assert m == max(CHECK_MIN_PICKS, math.ceil(math.log(CHECK_ALPHA) / math.log(x) - 1e-9))
        assert clopper_pearson_lower(m, m, CHECK_ALPHA) >= x - 1e-9
        if m > CHECK_MIN_PICKS:
            assert clopper_pearson_lower(m - 1, m - 1, CHECK_ALPHA) < x

    def test_below_fifty_percent_the_picks_are_priced_at_the_rounds_level(self):
        for x in (0.1, 0.25):
            s = check_schedule(x)
            assert s.picks == max(CHECK_MIN_PICKS, math.ceil(math.log(CHECK_ALPHA / s.rounds) / math.log(x) - 1e-9))

    def test_a_floor_of_one_is_a_census_of_the_candidate(self):
        assert check_schedule(1.0).picks == CHECK_BASE_CANDIDATE

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_a_floor_outside_the_unit_interval_is_refused(self, bad):
        with pytest.raises(ValueError):
            check_schedule(bad)

    def test_a_small_corpus_gets_the_halvings_it_really_has(self):
        assert rounds_for(8) == 1
        assert rounds_for(32) == 1
        assert rounds_for(50) == 2
        assert rounds_for(100) == 3
        assert rounds_for(128) == 3


class TestTheLikelyRange:
    def test_each_tail_is_the_level_a_round_is_tested_at(self):
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

    def test_all_five_right_clears_fifty_percent_and_four_does_not(self):
        """The 50% preset's whole design: 5 of 5 confirms, 4 of 5 falls short."""
        assert likely_range(5, 5, 32, range_tail(1)).lo >= 0.5
        assert likely_range(4, 5, 32, range_tail(1)).lo < 0.5

    def test_a_check_confirms_iff_its_ranges_lower_end_clears_the_floor(self):
        for x in PRESETS:
            for right in range(0, 6):
                check = SpotCheck.start(_ranking().candidate(check_schedule(x).candidate), x, seed=1)
                # One round only: mark the picks and stop the halving by reading
                # the range the round is decided on.
                picks = check.pending
                votes = {cid: i < right for i, cid in enumerate(picks)}
                check.record(votes)
                decided_confirmed = check.status == FLOOR_CONFIRMED
                first_round_range = likely_range(right, len(picks), check.start_k, range_tail(check.rounds))
                assert decided_confirmed == (first_round_range.lo >= x - 1e-9), (x, right)


class TestTheRanking:
    def test_sorted_by_score_then_id_with_unscorable_items_dropped(self):
        r = LineRanking.from_scores([3, 1, 2, 4], [0.5, 0.9, 0.5, NON_FINITE_SCORE_SENTINEL], voted={1})
        assert r.ids.tolist() == [1, 2, 3] and r.scores.tolist() == [0.9, 0.5, 0.5]
        assert r.size == 3 and r.voted == frozenset({1})

    def test_the_candidate_is_the_top_unvoted_in_rank_order(self):
        r = _ranking(10, voted={2})
        assert r.candidate(3) == (1, 3, 4)
        assert r.candidate(3, also_voted={1, 3}) == (4, 5, 6)
        assert r.candidate(100) == tuple(i for i in range(1, 11) if i != 2)

    def test_the_line_sits_just_under_the_last_item_of_the_set(self):
        r = _ranking(10, voted={2})
        assert r.threshold_for(3) == line_under(r.score_of(4)) <= r.score_of(4)
        assert r.threshold_for(100) == line_under(r.score_of(10)), "a count past the end keeps the whole remainder"
        assert r.threshold_for(3, also_voted=set(range(1, 11))) is None, "nothing unvoted: no set to keep"

    @pytest.mark.parametrize("score", [0.4657, 0.46572142839, 0.46568, 0.99995, 0.0, 1.0, 0.12345])
    def test_the_kept_item_clears_its_own_line_raw_or_rounded(self, score):
        """Responses round scores to 4 decimals; the line is floored to that grid, not rounded."""
        line = line_under(score)
        assert line <= score and round(score, 4) >= line
        assert round(line, 4) == line
        assert score - line < 1e-4

    def test_above_counts_voted_items_too(self):
        r = _ranking(10, voted={1, 2})
        line = r.threshold_for(3)
        assert line is not None and r.above(line) == 5

    def test_the_fingerprint_is_the_sets_identity(self):
        r = _ranking(10)
        assert r.fingerprint(3) == (1, 2, 3)
        assert r.fingerprint(3, also_voted={2}) == (1, 3, 4)

    def test_ids_and_scores_must_align(self):
        with pytest.raises(ValueError):
            LineRanking.from_scores([1, 2], [0.5])


class TestTheRounds:
    def test_round_one_deals_fresh_uniform_picks_from_the_fixed_candidate(self):
        check = SpotCheck.start(_ranking().candidate(128), 0.1, seed=3)
        assert (check.start_k, check.rounds, check.picks) == (128, 3, 5)
        assert len(check.pending) == 5 and set(check.pending) <= set(check.candidate_ids)
        assert check.round == 1 and check.k == 128 and check.running

    def test_a_failed_round_halves_and_keeps_the_labels_inside_the_new_candidate(self):
        check = SpotCheck.start(_ranking().candidate(128), 0.1, seed=5)
        first = check.pending
        # Force two picks into the top 64 by labelling everything Bad: whatever
        # was drawn, the kept labels are exactly those inside the halved list.
        check.record({cid: False for cid in first})
        assert check.running and check.round == 2 and check.k == 64
        kept = [cid for cid in first if cid in check.candidate_ids[:64]]
        labelled, right = check.counts()
        assert labelled == len(kept) and right == 0
        assert not set(check.pending) & set(first), "the next round draws fresh items"
        assert set(check.pending) <= set(check.candidate_ids[:64])
        check.record({cid: False for cid in check.pending})
        assert check.round == 3 and check.k == 32
        check.record({cid: False for cid in check.pending})
        assert check.status == FLOOR_SHORT and check.k == CHECK_BASE_CANDIDATE
        assert check.finished and not check.running

    def test_an_all_right_round_confirms_the_whole_candidate(self):
        check = _finish(SpotCheck.start(_ranking().candidate(128), 0.1, seed=7))
        assert check.status == FLOOR_CONFIRMED and check.k == 128 and check.round == 1
        assert check.range().lo >= 0.1

    def test_the_candidate_never_changes_after_the_start(self):
        """The ranking may retrain behind the check; every round samples the one list fixed at the start."""
        ranking = _ranking()
        check = SpotCheck.start(ranking.candidate(64), 0.25, seed=9)
        fixed = check.candidate_ids
        moved = LineRanking.from_scores(ranking.ids.tolist(), ranking.scores[::-1], set())
        check.record({cid: False for cid in check.pending})
        assert check.candidate_ids == fixed
        assert set(check.pending) <= set(fixed[:32])
        assert moved.candidate(64) != fixed, "the retrained ranking would have said otherwise"

    def test_a_partial_round_waits_and_a_stray_vote_is_refused(self):
        check = SpotCheck.start(_ranking().candidate(32), 0.5, seed=2)
        first, rest = check.pending[0], check.pending[1:]
        assert check.record({first: True}) is False
        assert check.pending == rest and check.running
        with pytest.raises(ValueError, match="not this round"):
            check.record({999: True})
        assert check.record({cid: True for cid in rest}) is True
        assert check.status == FLOOR_CONFIRMED

    def test_a_finished_check_takes_no_more_votes_and_a_cancelled_one_is_neither_state(self):
        done = _finish(SpotCheck.start(_ranking().candidate(32), 0.5, seed=2))
        with pytest.raises(ValueError):
            done.record({})
        with pytest.raises(ValueError):
            done.draw()
        cancelled = SpotCheck.start(_ranking().candidate(32), 0.5, seed=2)
        cancelled.cancel()
        assert cancelled.status == CHECK_CANCELLED and not cancelled.running and not cancelled.finished
        assert applicable_result(0.5, cancelled) is None

    def test_a_small_corpus_is_its_own_candidate_with_the_rounds_it_has(self):
        check = SpotCheck.start(_ranking(50).candidate(128), 0.1, seed=4)
        assert check.start_k == 50 and check.rounds == 2
        check.record({cid: False for cid in check.pending})
        assert check.k == 32 and check.round == 2
        tiny = SpotCheck.start(_ranking(3).candidate(32), 0.5, seed=4)
        assert tiny.start_k == 3 and tiny.rounds == 1 and len(tiny.pending) == 3, "a census of the whole corpus"
        tiny.record({cid: True for cid in tiny.pending})
        assert tiny.status == FLOOR_CONFIRMED and tiny.range().lo == tiny.range().hi == 1.0

    def test_the_range_is_exact_once_the_labels_cover_the_candidate(self):
        check = SpotCheck.start(_ranking(5).candidate(32), 0.9, seed=1)
        check.record({cid: cid != 5 for cid in check.pending})
        assert (check.range().lo, check.range().hi) == (0.8, 0.8)
        assert check.status == FLOOR_SHORT

    def test_the_picks_are_seeded_and_the_provenance_is_the_apps(self):
        a = SpotCheck.start(_ranking().candidate(128), 0.1, seed=11).pending
        b = SpotCheck.start(_ranking().candidate(128), 0.1, seed=11).pending
        assert a == b
        assert CHECK_PROVENANCE == {"flow": "check"}

    def test_the_state_a_client_sees(self):
        check = SpotCheck.start(_ranking().candidate(64), 0.25, seed=6)
        state = check.as_dict()
        assert state["status"] == "running" and state["picks"] == list(check.pending)
        assert (state["round"], state["rounds"], state["picks_per_round"]) == (1, 2, 5)
        assert (state["candidate"], state["start_candidate"]) == (64, 64)
        assert state["range"] is None and state["labelled"] == 0
        _finish(check)
        state = check.as_dict()
        assert state["status"] == FLOOR_CONFIRMED and state["picks"] == []
        assert state["range"]["labelled"] == 5 and state["range"]["right"] == 5


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

    def test_a_confirmed_check_keeps_the_set_it_confirmed(self):
        r = _ranking()
        check = _finish(SpotCheck.start(r.candidate(64), 0.25, seed=8))
        assert check.status == FLOOR_CONFIRMED and check.k == 64
        check.fingerprint = r.fingerprint(64, set(check.labels))
        assert floor_count(0.25, check) == 64
        assert floor_line(r, 0.25, check, set(check.labels)) == r.threshold_for(64, set(check.labels))
        state = floor_state(0.25, check, r, set(check.labels))
        assert state.status == FLOOR_CONFIRMED and state.count == 64 and not state.stale
        assert state.range is not None and state.range.lo >= 0.25

    def test_a_short_check_keeps_the_top_thirty_two_it_ended_on(self):
        r = _ranking()
        check = _finish(SpotCheck.start(r.candidate(128), 0.1, seed=8), right_ids=set())
        assert check.status == FLOOR_SHORT and check.k == 32
        check.fingerprint = r.fingerprint(32, set(check.labels))
        state = floor_state(0.1, check, r, set(check.labels))
        assert state.status == FLOOR_SHORT and state.count == 32
        assert state.range is not None and state.range.lo < 0.1
        assert floor_line(r, 0.1, check, set(check.labels)) == r.threshold_for(32, set(check.labels))

    def test_a_result_belongs_to_its_floor(self):
        r = _ranking()
        check = _finish(SpotCheck.start(r.candidate(32), 0.5, seed=8))
        assert applicable_result(0.5, check) is check
        assert applicable_result(0.25, check) is None
        assert floor_state(0.25, check, r).status == FLOOR_UNCHECKED
        assert floor_count(0.25, check) == 64

    def test_a_finished_result_goes_stale_when_the_set_under_it_moves(self):
        r = _ranking()
        check = _finish(SpotCheck.start(r.candidate(32), 0.5, seed=8))
        voted = set(check.labels)
        check.fingerprint = r.fingerprint(32, voted)
        assert not floor_state(0.5, check, r, voted).stale
        # A later vote inside the set: the top 32 unvoted shifts by one.
        later = voted | {next(cid for cid in r.candidate(32, voted))}
        assert floor_state(0.5, check, r, later).stale
        # A retrain that reorders the ranking.
        retrained = LineRanking.from_scores(r.ids.tolist(), r.scores[::-1], set())
        assert floor_state(0.5, check, retrained, voted).stale
        # A vote far below the line leaves the set, and so the range, alone.
        below = voted | {200}
        assert not floor_state(0.5, check, r, below).stale
        assert floor_line(r, 0.5, check, later) == r.threshold_for(32, later), "the line follows at the same count"
