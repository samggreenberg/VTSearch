"""The balance: the line at an F-beta optimum instead of a precision floor (#4413).

The owner's ruling of 2026-10-01 after #4411: the preference is a balance,
beta, and the line is the band walk stopped at the F-beta peak.  Pinned here,
on planted rankings: the F-beta arithmetic and its oracle, the mixture's
positives and F-beta argmax, the walk (deeper while the estimate does not fall,
shallower from a start whose first step falls, the peak's edge kept, no
``short``), the unchecked line (the mixture's argmax under the cap), and the
state a response carries (precision and recall ranges, the F-beta estimate,
stale once the ranking moves).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests_lib.sorting.test_mixture_count import _two_populations
from vtscore.training.thresholds import (
    BALANCE_CHECKED,
    BALANCE_PRESETS,
    DEFAULT_BETA,
    FLOOR_CONFIRMED,
    FLOOR_UNCHECKED,
    WALK_DEEPER,
    WALK_SHALLOWER,
    LineRanking,
    SpotCheck,
    applicable_balance,
    applicable_result,
    balance_count,
    balance_line,
    balance_schedule,
    balance_state,
    check_schedule,
    fbeta_count,
    fbeta_score,
    floor_count,
    floor_state,
    mixture_count,
    mixture_positives,
    walk_positives,
)


def _planted(n_pos: int = 52, n: int = 1000) -> tuple[LineRanking, set[int]]:
    """A ranking whose top *n_pos* unvoted items are the positives: ids 1..n on a ladder, nothing voted."""
    ranking = LineRanking.from_scores(list(range(1, n + 1)), np.linspace(0.99, 0.01, n), set())
    return ranking, set(range(1, n_pos + 1))


def _finish(check: SpotCheck, positives: set[int]) -> SpotCheck:
    while check.running:
        check.record({cid: cid in positives for cid in check.pending})
    return check


def _best_edge(check: SpotCheck, positives: set[int], beta: float) -> int:
    """The band edge with the best true F-beta on the walk's ranking."""
    n_pos = len(positives)
    best = max(
        check.edges[1:],
        key=lambda e: fbeta_score(sum(1 for cid in check.ranking_ids[:e] if cid in positives), e, n_pos, beta),
    )
    return int(best)


class TestTheArithmetic:
    def test_fbeta_is_the_textbook_one(self):
        assert fbeta_score(10, 20, 40, 1.0) == pytest.approx(2 * 10 / (40 + 20))
        assert fbeta_score(10, 20, 40, 2.0) == pytest.approx(5 * 10 / (4 * 40 + 20))
        assert fbeta_score(0, 0, 0, 1.0) == 0.0

    def test_the_presets_and_their_caps(self):
        assert BALANCE_PRESETS == (0.5, 1.0, 2.0) and DEFAULT_BETA == 1.0
        assert balance_schedule(0.5).candidate == balance_schedule(1.0).candidate == check_schedule(0.5).candidate == 32
        assert balance_schedule(2.0).candidate == check_schedule(0.1).candidate == 128

    @pytest.mark.parametrize("bad", [0.0, 0.1, 5.0, -1.0])
    def test_a_balance_outside_the_range_is_refused(self, bad):
        with pytest.raises(ValueError, match="beta must be in"):
            balance_schedule(bad)


class TestTheMixturesCount:
    def test_the_positives_and_the_argmax_track_the_truth_on_a_two_population_ranking(self):
        ranking, labels = _two_populations()  # 52 unvoted positives at ~0.85, 932 negatives at ~0.2
        n_pos = mixture_positives(ranking, labels)
        assert n_pos is not None and 40 <= n_pos <= 80, n_pos
        k = fbeta_count(ranking, 1.0, labels)
        assert k is not None and 40 <= k <= 70, k
        # Recall-leaning returns more, precision-leaning fewer.
        k2, k05 = fbeta_count(ranking, 2.0, labels), fbeta_count(ranking, 0.5, labels)
        assert k2 is not None and k05 is not None and k05 <= k <= k2
        # The floor's own count at 50% on the same fit runs deeper: it keeps going while the set is half right.
        assert (mixture_count(ranking, 0.5, labels) or 0) >= k

    def test_nothing_fits_is_none(self):
        assert fbeta_count(None, 1.0, {}) is None and mixture_positives(None, {}) is None
        empty = LineRanking.from_scores([], [])
        assert fbeta_count(empty, 1.0, {}) is None

    def test_the_walk_reads_the_mixtures_count_when_it_has_one(self):
        ranking, _ = _planted(52)
        labels = {c: c <= 52 + 8 for c in ranking.voted}
        assert walk_positives(ranking, 1.0, labels) == mixture_positives(ranking, labels)

    def test_and_the_cap_over_the_unvoted_when_it_has_none(self):
        """#4419: a check still starts; recall is read against the set the unchecked line would keep."""
        from vtscore.training.thresholds import balance_schedule

        assert walk_positives(None, 1.0, {}) == balance_schedule(1.0).candidate == 32
        assert walk_positives(None, 2.0, {}) == 128
        empty = LineRanking.from_scores([], [])
        assert walk_positives(empty, 0.5, {}) == 32
        # A collapsing fit (the app's 20-item corpus, see test_mixture_count): 16 unvoted under a cap of 32.
        scores = [0.66, 0.659, 0.515, 0.512, 0.511, 0.51, 0.509, 0.506, 0.501, 0.495]
        scores += [0.492, 0.491, 0.49, 0.489, 0.488, 0.486, 0.486, 0.476, 0.343, 0.34]
        labels = {1: True, 2: True, 19: False, 20: False}
        ranking = LineRanking.from_scores(list(range(1, 21)), scores, set(labels))
        assert mixture_positives(ranking, labels) is None
        assert walk_positives(ranking, 1.0, labels) == 16.0
        assert walk_positives(ranking, 2.0, labels) == 16.0
        check = SpotCheck.start_balance(ranking.unvoted_ids().tolist(), 1.0, walk_positives(ranking, 1.0, labels))
        assert check.n_pos == 16.0 and check.pending


class TestTheWalk:
    def test_it_walks_deeper_to_the_f1_peak_and_keeps_its_edge(self):
        ranking, positives = _planted(52)
        check = SpotCheck.start_balance(ranking.unvoted_ids().tolist(), 1.0, 52.0, seed=0)
        assert check.beta == 1.0 and check.start_k == 32 and check.status == "running"
        _finish(check, positives)
        assert check.status == BALANCE_CHECKED and check.finished
        assert check.k == _best_edge(check, positives, 1.0) == 64
        assert check.direction == WALK_DEEPER
        # Bands 8, 8-16, 16-32 (the start), 32-64, then 64-128 to see the fall: 25 picks.
        assert check.round == 5 and len(check.labels) == 25
        # The estimate is read off 5 picks a band, so it sits near the truth, not on it.
        assert check.best_estimate == pytest.approx(fbeta_score(52, 64, 52, 1.0), abs=0.2)

    def test_from_a_start_whose_first_step_falls_it_walks_shallower_to_the_peak(self):
        ranking, positives = _planted(52)
        check = SpotCheck.start_balance(ranking.unvoted_ids().tolist(), 1.0, 52.0, seed=0, start_count=128)
        _finish(check, positives)
        assert check.status == BALANCE_CHECKED and check.k == 64
        assert check.direction == WALK_SHALLOWER
        # Stepping back costs no picks: 128 (start, 5 bands) and 256 were audited, 64 and 32 read off them.
        assert check.round == 6

    def test_a_precision_leaning_balance_keeps_less_and_a_recall_leaning_one_more(self):
        ranking, positives = _planted(52)
        kept = {}
        for beta in BALANCE_PRESETS:
            check = SpotCheck.start_balance(ranking.unvoted_ids().tolist(), beta, 52.0, seed=0)
            kept[beta] = _finish(check, positives).k
            assert kept[beta] == _best_edge(check, positives, beta)
        assert kept[0.5] <= kept[1.0] <= kept[2.0]

    def test_an_all_wrong_ranking_ends_at_the_first_band_never_short(self):
        """Nothing right anywhere: the estimate ties at 0, and ties keep the smaller set, down to the first band."""
        ranking, _ = _planted(52)
        check = _finish(SpotCheck.start_balance(ranking.unvoted_ids().tolist(), 1.0, 52.0, seed=0), set())
        assert check.status == BALANCE_CHECKED and check.k == 8
        assert check.fbeta_estimate() == 0.0
        # The start's 3 bands and one deeper band were audited; stepping back cost nothing.
        assert check.round == 4

    def test_it_reports_precision_and_recall_ranges_and_the_estimate(self):
        ranking, positives = _planted(52)
        check = _finish(SpotCheck.start_balance(ranking.unvoted_ids().tolist(), 1.0, 52.0, seed=0), positives)
        d = check.as_dict()
        assert d["status"] == BALANCE_CHECKED and d["beta"] == 1.0 and d["min_precision"] is None
        assert d["fbeta"] == pytest.approx(check.best_estimate, abs=1e-4)
        assert d["fbeta"] == pytest.approx(fbeta_score(52, 64, 52, 1.0), abs=0.2)
        # The ranges describe the kept set (its 4 bands, 20 picks); the 5th band was audited only to see the fall.
        assert 0.0 <= d["recall"]["lo"] <= d["recall"]["hi"] <= 1.0 and d["range"]["labelled"] == 20
        assert len(check.labels) == 25
        recall = check.recall_range()
        assert recall is not None and recall.hi == pytest.approx(1.0)

    def test_a_balance_walk_needs_a_count_of_positives(self):
        ranking, _ = _planted(52)
        with pytest.raises(ValueError, match="positive count"):
            SpotCheck.start_balance(ranking.unvoted_ids().tolist(), 1.0, 0.0)
        with pytest.raises(ValueError, match="beta must be in"):
            SpotCheck.start_balance(ranking.unvoted_ids().tolist(), 9.0, 52.0)

    def test_a_floor_walk_is_unchanged(self):
        ranking, positives = _planted(52)
        check = _finish(SpotCheck.start(ranking.unvoted_ids().tolist(), 0.5, seed=0), positives)
        assert check.beta is None and check.status == FLOOR_CONFIRMED and check.fbeta_estimate() is None
        assert check.as_dict()["min_precision"] == 0.5 and "beta" not in check.as_dict()


class TestTheLineAndTheState:
    def test_the_unchecked_count_is_the_mixtures_argmax_under_the_cap(self):
        assert balance_count(1.0, None) == 32 and balance_count(2.0, None) == 128
        assert balance_count(1.0, None, proposal=10) == 10
        assert balance_count(1.0, None, proposal=500) == 32, "the cap holds the mixture's over-return"
        assert balance_count(2.0, None, proposal=500) == 128
        assert balance_count(1.0, None, proposal=0) == 1

    def test_the_line_sits_under_the_kept_set(self):
        ranking, _ = _planted(52)
        assert balance_line(ranking, 1.0, proposal=20) == ranking.threshold_for(20)
        assert balance_line(None, 1.0) is None

    def test_a_finished_walk_outranks_the_proposal_and_belongs_to_its_beta(self):
        ranking, positives = _planted(52)
        check = _finish(SpotCheck.start_balance(ranking.unvoted_ids().tolist(), 1.0, 52.0, seed=0), positives)
        assert applicable_balance(1.0, check) is check and applicable_balance(2.0, check) is None
        assert balance_count(1.0, check, proposal=3) == 64
        assert balance_count(2.0, check, proposal=3) == 3, "another balance is unchecked until it is walked"
        floor_walk = _finish(SpotCheck.start(ranking.unvoted_ids().tolist(), 0.5, seed=0), positives)
        assert applicable_balance(1.0, floor_walk) is None, "a floor's result never serves a balance"
        # And the other way: a balance walk's NaN floor must not compare equal to every floor.
        assert applicable_result(0.5, check) is None and floor_count(0.5, check) == 32
        assert floor_state(0.5, check, ranking).status == FLOOR_UNCHECKED

    def test_the_state_a_response_carries(self):
        ranking, positives = _planted(52)
        before = balance_state(1.0, None, ranking, proposal=20)
        assert before.status == FLOOR_UNCHECKED and before.count == 20 and before.precision is None
        assert before.as_dict()["schedule"]["candidate"] == 32 and before.as_dict()["fbeta"] is None
        check = _finish(SpotCheck.start_balance(ranking.unvoted_ids().tolist(), 1.0, 52.0, seed=0), positives)
        check.fingerprint = ranking.fingerprint(check.k)
        after = balance_state(1.0, check, ranking)
        d = after.as_dict()
        assert d["status"] == BALANCE_CHECKED and d["count"] == 64 and d["beta"] == 1.0
        assert d["precision"]["stale"] is False and d["recall"]["lo"] <= d["recall"]["hi"]
        assert d["fbeta"] == pytest.approx(check.best_estimate, abs=1e-4)
        # A later vote inside the set moves the ranking under the result.
        stale = balance_state(1.0, check, ranking, also_voted={1})
        assert stale.stale is True and stale.as_dict()["precision"]["stale"] is True
        assert balance_state(2.0, check, ranking).status == FLOOR_UNCHECKED, "a result belongs to its balance"
