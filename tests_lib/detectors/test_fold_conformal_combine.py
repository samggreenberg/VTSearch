"""Unit tests for the cross-calibration combine rules (issue #3115).

``threshold_from_fold_orderings`` **pools** every fold's held-out scores and
takes one conformal quantile; ``FoldAnchoredCut._combined_fold_quantile`` takes
one cut per fold and averages them in quantile space, on the stated premise that
fold scores are *not* directly comparable.  Both docstrings cannot be right.
:func:`~vtscore.training.thresholds.combined_fold_conformal_threshold` is the
challenger family that lets a run adjudicate it, and these pin the properties
the adjudication depends on:

* the score-space combine reproduces the pooled cut **bit for bit** at K=1, so
  the study has an exact control;
* the quantile-space combine deliberately does *not*, and the reason is a
  property of quantiles rather than a bug;
* mean and median coincide below three folds, which is why the question has
  never been askable at production's ``calibrate_count=2``;
* a single-class fold is **dropped and counted**, not averaged in as
  :func:`conformal_threshold`'s 0.5 sentinel.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.training.thresholds import (
    FOLD_CONFORMAL_COMBINES,
    combined_fold_conformal_threshold,
    conformal_threshold,
    per_fold_conformal_cuts,
    threshold_from_fold_orderings,
)


def _fold(rng, n=40, shift=0.0):
    """A separable held-out fold, offset on the sigmoid scale by *shift*."""
    half = n // 2
    pos = np.clip(rng.normal(0.75 + shift, 0.10, half), 0.0, 1.0)
    neg = np.clip(rng.normal(0.30 + shift, 0.10, half), 0.0, 1.0)
    return (np.concatenate([pos, neg]).tolist(), [1.0] * half + [0.0] * half)


@pytest.fixture
def folds():
    """Four folds whose scales deliberately *drift*, plus their haystacks.

    The drift is the whole point: if every fold sat on the same scale, score-
    space and quantile-space combining would agree by construction and the
    contrast the study runs would be empty on this fixture.
    """
    rng = np.random.default_rng(0)
    orderings = [_fold(rng, shift=0.05 * i) for i in range(4)]
    haystacks = [np.sort(np.clip(rng.normal(0.30 + 0.05 * i, 0.20, 5000), 0.0, 1.0)) for i in range(4)]
    final = np.clip(rng.normal(0.32, 0.20, 5000), 0.0, 1.0).tolist()
    return orderings, haystacks, final


def _combined(orderings, haystacks, final, combine, k):
    quantile_space = combine.startswith("q")
    return combined_fold_conformal_threshold(
        orderings[:k],
        0,
        combine=combine,
        fold_haystacks=haystacks[:k] if quantile_space else None,
        final_scores=final if quantile_space else None,
    )


class TestScoreSpaceCombine:
    def test_single_fold_reproduces_the_pooled_cut_exactly(self, folds):
        """The study's exact control, and it has to be *exact*.

        Averaging one number is the identity, so any drift here is the rule
        losing information the pooled cut carries - in particular the conformal
        gap midpoint, which is a specific point inside an empty band rather than
        an order statistic.  ``==`` rather than ``approx`` on purpose.
        """
        orderings, haystacks, final = folds
        pooled = threshold_from_fold_orderings(orderings[:1], 0)
        for combine in ("tmean", "tmedian"):
            value, prov = _combined(orderings, haystacks, final, combine, 1)
            assert value == pooled, combine
            assert prov == f"fold_conformal_{combine}[1/1]"

    def test_mean_and_median_coincide_below_three_folds(self, folds):
        """Why nobody could have asked this at production's two folds."""
        orderings, haystacks, final = folds
        for k in (1, 2):
            assert (
                _combined(orderings, haystacks, final, "tmean", k)[0]
                == (_combined(orderings, haystacks, final, "tmedian", k)[0])
            )
            assert (
                _combined(orderings, haystacks, final, "qmean", k)[0]
                == (_combined(orderings, haystacks, final, "qmedian", k)[0])
            )

    def test_mean_and_median_separate_at_three_folds(self, folds):
        """...and that they *do* separate once there are three, or the run is empty."""
        orderings, haystacks, final = folds
        assert (
            _combined(orderings, haystacks, final, "tmean", 3)[0]
            != (_combined(orderings, haystacks, final, "tmedian", 3)[0])
        )

    def test_combining_differs_from_pooling(self, folds):
        """The contrast the study is for: mean-of-quantiles is not quantile-of-mixture."""
        orderings, haystacks, final = folds
        pooled = threshold_from_fold_orderings(orderings, 0)
        assert _combined(orderings, haystacks, final, "tmean", 4)[0] != pooled

    def test_is_the_mean_of_the_folds_own_cuts(self, folds):
        """Stated as arithmetic, so a wrong axis fails rather than merely looking plausible."""
        orderings, haystacks, final = folds
        cuts = [conformal_threshold(*ordering, 0) for ordering in orderings]
        assert _combined(orderings, haystacks, final, "tmean", 4)[0] == pytest.approx(float(np.mean(cuts)))
        assert _combined(orderings, haystacks, final, "tmedian", 4)[0] == pytest.approx(float(np.median(cuts)))


class TestQuantileSpaceCombine:
    def test_lands_inside_the_final_haystacks_support(self, folds):
        orderings, haystacks, final = folds
        arr = np.asarray(final)
        for combine in ("qmean", "qmedian"):
            value, _prov = _combined(orderings, haystacks, final, combine, 4)
            assert arr.min() <= value <= arr.max(), combine

    def test_does_not_reproduce_the_pooled_cut_even_at_one_fold(self, folds):
        """Documented, not a defect: a quantile records which observed scores a
        cut sits *between*, not where inside that gap it sat, so the conformal
        midpoint cannot survive the round trip.  The ``tmean`` leg exists to
        separate this from the combine rule itself, and this test is what stops
        someone "fixing" the ``q*`` arms into a false control.
        """
        orderings, haystacks, final = folds
        pooled = threshold_from_fold_orderings(orderings[:1], 0)
        assert _combined(orderings, haystacks, final, "qmean", 1)[0] != pooled

    def test_requires_the_transfer_inputs(self, folds):
        orderings, _haystacks, _final = folds
        with pytest.raises(ValueError, match="needs fold_haystacks"):
            combined_fold_conformal_threshold(orderings, 0, combine="qmean")


class TestDegenerateFolds:
    #: A holdout the fold model saw only positives on - no negatives, so
    #: `conformal_threshold` has no quantile to take and answers its 0.5 "no
    #: evidence" sentinel.  This is the fold #3115's contamination hypothesis is
    #: about: it pours its scores straight into a pooled quantile.
    SINGLE_CLASS = ([0.90, 0.80, 0.85], [1.0, 1.0, 1.0])

    def test_single_class_fold_yields_no_cut(self):
        assert conformal_threshold(*self.SINGLE_CLASS, 0) == 0.5
        assert per_fold_conformal_cuts([self.SINGLE_CLASS], 0) == []

    def test_dropped_from_the_combine_and_counted(self, folds):
        """Dropped, not averaged in at 0.5 - and the provenance says how many.

        Averaging the sentinel would drag the combined cut toward the middle of
        the sigmoid for a reason that has nothing to do with where the classes
        sit, which would show up in a report as the combine rule "being worse"
        on exactly the steps the pooled rule handles quietly.
        """
        orderings, haystacks, final = folds
        clean, _ = _combined(orderings, haystacks, final, "tmean", 2)
        contaminated, prov = combined_fold_conformal_threshold([*orderings[:2], self.SINGLE_CLASS], 0, combine="tmean")
        assert contaminated == clean
        assert prov == "fold_conformal_tmean[2/3]"

    def test_falls_back_to_pooling_when_no_fold_contributes(self):
        """So the arm stays defined on exactly the steps its control is defined on."""
        value, prov = combined_fold_conformal_threshold([self.SINGLE_CLASS], 0, combine="tmean")
        assert prov == "fold_conformal_fallback_pooled"
        assert value == threshold_from_fold_orderings([self.SINGLE_CLASS], 0)


def test_unknown_combine_is_rejected(folds):
    orderings, _haystacks, _final = folds
    with pytest.raises(ValueError, match="unknown fold conformal combine"):
        combined_fold_conformal_threshold(orderings, 0, combine="qmode")


def test_the_family_is_the_documented_two_by_two():
    assert set(FOLD_CONFORMAL_COMBINES) == {"tmean", "tmedian", "qmean", "qmedian"}
