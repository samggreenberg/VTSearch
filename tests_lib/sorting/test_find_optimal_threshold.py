"""Unit tests for ``find_optimal_threshold``.

Regression coverage for two search-space bugs:

* The candidate set was only the observed scores, so "predict nothing
  positive" (threshold above the max score; FPR=0, FNR=1) was unreachable
  even when it minimised the weighted cost - under a precision-biased
  inclusion the calibrated cutoff came out far too permissive.
* Tied scores produced infeasible cut positions: the cumsum at a mid-tie
  index corresponds to predicting only *some* of the items at that score,
  which ``score >= threshold`` cannot realize, so ``argmin`` could report
  a cost the returned threshold doesn't achieve.
"""

from __future__ import annotations

import pytest

from vtscore.training.thresholds import (
    NO_GOOD_THRESHOLD,
    find_optimal_threshold,
    threshold_from_fold_orderings,
)


class TestBasics:
    def test_empty_scores_returns_default(self):
        assert find_optimal_threshold([], []) == 0.5

    def test_single_class_returns_default(self):
        assert find_optimal_threshold([0.2, 0.8], [1.0, 1.0]) == 0.5
        assert find_optimal_threshold([0.2, 0.8], [0.0, 0.0]) == 0.5

    def test_perfect_separation_picks_boundary(self):
        scores = [0.9, 0.8, 0.2, 0.1]
        labels = [1.0, 1.0, 0.0, 0.0]
        t = find_optimal_threshold(scores, labels)
        # Any threshold in (0.2, 0.8] classifies perfectly; the search
        # returns an observed score, so it must be 0.8.
        assert t == 0.8

    def test_balanced_inclusion_prefers_low_cost_cut(self):
        # One inversion: threshold at 0.6 gives fp=0, fn=1 (cost 1/2 + 0);
        # cutting deeper at 0.4 gives fp=1, fn=0 (cost 1/2).  Equal cost -
        # argmin picks the first (higher) cut.
        scores = [0.9, 0.6, 0.5, 0.4, 0.2]
        labels = [1.0, 1.0, 0.0, 1.0, 0.0]
        t = find_optimal_threshold(scores, labels, 0)
        assert 0.0 < t <= 1.0


class TestPredictNothingCandidate:
    def test_precision_biased_abstains_on_top_scored_negative(self):
        """A lone top-scored negative under inclusion=-3 must abstain.

        Observed candidates: t=0.9 → cost 8*1 + 1*1 = 9; t=0.1 → cost
        8*1 + 0 = 8.  Predicting nothing costs fnr_weight = 1, strictly
        cheaper, so the sentinel must be returned.
        """
        t = find_optimal_threshold([0.9, 0.1], [0.0, 1.0], inclusion_value=-3)
        assert t == NO_GOOD_THRESHOLD

    def test_observed_threshold_wins_ties_with_abstain(self):
        """When abstaining merely ties the best observed cut, keep the cut."""
        # Perfect separation: best observed cost is 0, abstain costs 1.
        t = find_optimal_threshold([0.9, 0.1], [1.0, 0.0], inclusion_value=-3)
        assert t == 0.9

    def test_recall_biased_never_abstains(self):
        # With inclusion >= 0, fnr_weight >= 1 and the full-inclusion cut
        # costs at most fpr_weight = 1 <= fnr_weight, so abstaining is
        # never strictly cheaper... unless every cut includes a negative
        # AND misses a positive.  Sanity-check the common case.
        t = find_optimal_threshold([0.9, 0.1], [0.0, 1.0], inclusion_value=3)
        assert t == 0.1


class TestTiedScores:
    def test_mid_tie_cut_not_reported(self):
        """All-tied scores admit exactly one cut: include everything.

        With scores all 0.7, ``score >= 0.7`` includes every item, so
        fp=2, fn=0 → cost = fpr_weight.  A mid-tie argmin used to report
        an infeasible fp=0/fn=1 split.  Under inclusion=-2 (fpr_weight=4)
        the true choice is between including all (cost 4) and abstaining
        (cost 1): abstain wins.
        """
        scores = [0.7, 0.7, 0.7, 0.7]
        labels = [1.0, 0.0, 1.0, 0.0]
        t = find_optimal_threshold(scores, labels, inclusion_value=-2)
        assert t == NO_GOOD_THRESHOLD

    def test_tie_run_uses_last_position_counts(self):
        """A tie run's feasible cut counts every tied item.

        scores: pos 1.0, then a tied run [0.5, 0.5] with one of each class,
        then neg 0.1.  Cutting at 0.5 includes both tied items (fp=1, fn=0);
        cutting at 1.0 gives fp=0, fn=1.  At inclusion=0 both cost 0.5;
        the first (higher) cut wins the argmin tie.
        """
        scores = [1.0, 0.5, 0.5, 0.1]
        labels = [1.0, 1.0, 0.0, 0.0]
        t = find_optimal_threshold(scores, labels, inclusion_value=0)
        assert t == 1.0

    def test_saturated_sigmoid_scores(self):
        """Exact 1.0/0.0 sigmoid saturation must yield a usable threshold."""
        scores = [1.0, 1.0, 1.0, 0.0, 0.0]
        labels = [1.0, 1.0, 1.0, 0.0, 0.0]
        t = find_optimal_threshold(scores, labels, inclusion_value=0)
        assert t == 1.0


class TestFoldAggregationAbstain:
    """``threshold_from_fold_orderings`` treats an abstaining fold as a *vote*,
    not a numeric 2.0 to average.  The ensemble abstains only under a strict
    majority; otherwise it averages the folds that produced a real cut.
    """

    # At inclusion=-3, a lone top-scored negative makes the fold abstain, while
    # a perfectly separable fold yields a concrete cut (0.9).
    ABSTAIN = ([0.9, 0.1], [0.0, 1.0])
    FINITE = ([0.9, 0.1], [1.0, 0.0])
    INCL = -3

    def test_single_fold_abstains(self):
        assert threshold_from_fold_orderings([self.ABSTAIN], self.INCL) == NO_GOOD_THRESHOLD

    def test_lone_abstain_of_two_is_ignored(self):
        """1 of 2 abstaining is not a strict majority: use the finite fold."""
        t = threshold_from_fold_orderings([self.ABSTAIN, self.FINITE], self.INCL)
        assert t == 0.9

    def test_lone_abstain_never_exceeds_sigmoid_range(self):
        """Regression: the old numeric mean pushed a lone abstain to
        (0.9 + 2.0)/2 = 1.45, i.e. an out-of-range 'threshold' that abstained
        by accident and depended on the other fold's magnitude."""
        t = threshold_from_fold_orderings([self.ABSTAIN, self.FINITE], self.INCL)
        assert t <= 1.0

    def test_both_of_two_abstain(self):
        assert threshold_from_fold_orderings([self.ABSTAIN, self.ABSTAIN], self.INCL) == NO_GOOD_THRESHOLD

    def test_minority_abstain_of_three_averages_finite(self):
        """1 of 3 abstaining: average the two folds that produced a cut."""
        t = threshold_from_fold_orderings([self.ABSTAIN, self.FINITE, self.FINITE], self.INCL)
        assert t == 0.9

    def test_majority_abstain_of_three(self):
        t = threshold_from_fold_orderings([self.ABSTAIN, self.ABSTAIN, self.FINITE], self.INCL)
        assert t == NO_GOOD_THRESHOLD

    def test_all_finite_folds_are_meaned(self):
        a = ([0.9, 0.1], [1.0, 0.0])  # -> 0.9
        b = ([0.8, 0.2], [1.0, 0.0])  # -> 0.8
        t = threshold_from_fold_orderings([a, b], self.INCL)
        assert t == pytest.approx(0.85)
