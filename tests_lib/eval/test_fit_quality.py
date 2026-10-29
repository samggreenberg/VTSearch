"""Planted-answer tests for the #3329 goodness-of-fit diagnostics.

Every test here draws from a distribution whose answer is known in advance, so a
statistic that silently stops working fails rather than merely changing.  The
two shapes that matter are "the model is right" (a sample drawn from the very
mixture it is scored against) and "the model is wrong in the predicted
direction" (a right-skewed Bad mode, which is what max-pooling over region nodes
is argued to produce).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vtscore.eval.fit_quality import (
    FIT_QUALITY_COLUMNS,
    anchor_mass_fraction,
    class_shape,
    ecdf_distances,
    fit_quality_row,
    identification,
    mixture_cdf,
    tail_calibration,
)
from vtscore.training.thresholds import GmmFit1D


def _draw_mixture(rng: np.random.Generator, fit: GmmFit1D, n: int) -> np.ndarray:
    """Sample *n* points from *fit* exactly."""
    take_hi = rng.random(n) < fit.w_hi
    lo = rng.normal(fit.mu_lo, math.sqrt(fit.var_lo), n)
    hi = rng.normal(fit.mu_hi, math.sqrt(fit.var_hi), n)
    return np.where(take_hi, hi, lo)


WELL_SEPARATED = GmmFit1D(w_lo=0.8, mu_lo=0.2, var_lo=0.01, w_hi=0.2, mu_hi=0.8, var_hi=0.01)


class TestMixtureCdf:
    def test_is_monotone_and_spans_unit_interval(self):
        x = np.linspace(-1.0, 2.0, 400)
        c = mixture_cdf(WELL_SEPARATED, x)
        assert np.all(np.diff(c) >= -1e-12)
        assert c[0] == pytest.approx(0.0, abs=1e-6)
        assert c[-1] == pytest.approx(1.0, abs=1e-6)

    def test_matches_the_weighted_component_cdfs_at_the_means(self):
        # At mu_lo the low component contributes exactly half its weight.
        at_lo = float(mixture_cdf(WELL_SEPARATED, np.array([WELL_SEPARATED.mu_lo]))[0])
        assert at_lo == pytest.approx(0.5 * WELL_SEPARATED.w_lo, abs=1e-6)


class TestEcdfDistances:
    def test_a_correct_model_scores_near_zero(self):
        rng = np.random.default_rng(0)
        sample = _draw_mixture(rng, WELL_SEPARATED, 4000)
        d = ecdf_distances(sample, WELL_SEPARATED)
        # KS for a correct model at n=4000 is ~1.36/sqrt(n) = 0.021 at the 5%
        # level; a generous bound still separates it from any real misfit.
        assert d["ks"] < 0.05
        assert d["ad"] < 10.0

    def test_a_wrong_model_is_caught(self):
        rng = np.random.default_rng(1)
        # Data is one broad blob; the model claims two tight, separated ones.
        sample = rng.normal(0.5, 0.3, 4000)
        d = ecdf_distances(sample, WELL_SEPARATED)
        assert d["ks"] > 0.15
        assert d["ad"] > 100.0

    def test_too_few_points_is_nan_not_an_exception(self):
        d = ecdf_distances(np.array([0.1, 0.2, 0.3]), WELL_SEPARATED)
        assert all(math.isnan(v) for v in d.values())

    def test_ks_is_the_sup_of_the_two_one_sided_gaps(self):
        # A hand-checkable case: two points against a model that puts them at
        # the 0.5 and 0.5 quantiles.  The ECDF steps 0 -> 0.5 -> 1.0 there, so
        # the sup gap is 0.5.
        flat = GmmFit1D(w_lo=0.5, mu_lo=0.0, var_lo=1e-6, w_hi=0.5, mu_hi=1.0, var_hi=1e-6)
        sample = np.full(40, 0.5)
        d = ecdf_distances(sample, flat)
        assert d["ks"] == pytest.approx(0.5, abs=1e-6)


class TestTailCalibration:
    def test_a_correct_model_has_ratio_near_one(self):
        rng = np.random.default_rng(2)
        sample = _draw_mixture(rng, WELL_SEPARATED, 20000)
        t = tail_calibration(sample, WELL_SEPARATED, cut=0.5)
        assert t["ratio"] == pytest.approx(1.0, abs=0.1)

    def test_a_heavy_tail_the_model_misses_reads_above_one(self):
        rng = np.random.default_rng(3)
        # The failure this statistic exists to catch: a fit that puts almost no
        # mass above the cut, against data whose Bad mode has a genuine heavy
        # right tail.  Both fitted components sit well below 0.5, so the model
        # predicts ~0.2% up there while ~1.2% of the sample actually clears it.
        #
        # NOT `WELL_SEPARATED`: that model carries a 20%-weight component
        # centred at 0.8, so it *expects* a fifth of the sample above the cut
        # and reads an all-Bad tail as gross OVER-prediction. Planting it that
        # way is how this test first tested the opposite of its own name.
        low_only = GmmFit1D(w_lo=0.97, mu_lo=0.20, var_lo=0.01, w_hi=0.03, mu_hi=0.30, var_hi=0.01)
        heavy = np.concatenate(
            [
                rng.normal(0.2, 0.01, 16000),
                rng.gumbel(0.25, 0.09, 4000),  # spills well past 0.5
            ]
        )
        t = tail_calibration(heavy, low_only, cut=0.5)
        assert t["empirical"] > t["predicted"]
        assert t["ratio"] > 1.5

    def test_zero_predicted_mass_is_nan_not_inf(self):
        far = GmmFit1D(w_lo=0.5, mu_lo=0.0, var_lo=1e-8, w_hi=0.5, mu_hi=0.01, var_hi=1e-8)
        sample = np.full(100, 5.0)
        t = tail_calibration(sample, far, cut=4.0)
        assert math.isnan(t["ratio"])

    def test_non_finite_cut_is_handled(self):
        t = tail_calibration(np.linspace(0, 1, 100), WELL_SEPARATED, cut=float("nan"))
        assert math.isnan(t["ratio"])


class TestClassShape:
    def test_symmetric_gaussian_classes_read_as_unskewed(self):
        rng = np.random.default_rng(4)
        # Built on the LOGIT axis, then squashed, because class_shape measures
        # in logit space - so a planted symmetric shape must be planted there.
        neg = 1.0 / (1.0 + np.exp(-rng.normal(-2.0, 0.5, 3000)))
        pos = 1.0 / (1.0 + np.exp(-rng.normal(2.0, 0.5, 3000)))
        scores = np.concatenate([neg, pos])
        labels = np.concatenate([np.zeros(3000), np.ones(3000)])
        sh = class_shape(scores, labels)
        assert abs(sh["skew_neg"]) < 0.2
        assert abs(sh["skew_pos"]) < 0.2
        assert abs(sh["kurt_neg"]) < 0.4

    def test_a_max_pooled_negative_mode_reads_as_right_skewed(self):
        rng = np.random.default_rng(5)
        # The geometry argument, simulated: each negative's score is the MAX
        # over 24 region nodes, which is an extreme-value statistic and is
        # right-skewed on the logit axis.
        nodes = rng.normal(-3.0, 1.0, size=(3000, 24))
        neg_logit = nodes.max(axis=1)
        neg = 1.0 / (1.0 + np.exp(-neg_logit))
        pos = 1.0 / (1.0 + np.exp(-rng.normal(2.0, 0.5, 500)))
        scores = np.concatenate([neg, pos])
        labels = np.concatenate([np.zeros(3000), np.ones(500)])
        sh = class_shape(scores, labels)
        assert sh["skew_neg"] > 0.3, f"expected a right-skewed max-pooled mode, got {sh['skew_neg']}"
        assert sh["ad_normal_neg"] > 1.0

    def test_counts_are_reported_per_class(self):
        scores = np.concatenate([np.full(40, 0.2), np.full(50, 0.8)])
        labels = np.concatenate([np.zeros(40), np.ones(50)])
        sh = class_shape(scores, labels)
        assert sh["n_neg"] == 40.0
        assert sh["n_pos"] == 50.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            class_shape(np.zeros(5), np.zeros(4))


class TestIdentification:
    def test_a_split_that_is_the_class_split_scores_one(self):
        rng = np.random.default_rng(6)
        neg = rng.normal(0.2, 0.01, 2000)
        pos = rng.normal(0.8, 0.01, 500)
        scores = np.concatenate([neg, pos])
        labels = np.concatenate([np.zeros(2000), np.ones(500)])
        ident = identification(scores, labels, WELL_SEPARATED)
        assert ident["bal_acc"] > 0.99
        assert ident["ari"] > 0.95
        assert abs(ident["mu_lo_err"]) < 0.02
        assert abs(ident["mu_hi_err"]) < 0.02

    def test_a_split_orthogonal_to_the_classes_scores_at_chance(self):
        rng = np.random.default_rng(7)
        # Both classes drawn from the same distribution: no split can separate
        # them, so balanced accuracy must sit at 0.5 and ARI at 0.
        scores = rng.normal(0.5, 0.2, 4000)
        labels = (rng.random(4000) < 0.3).astype(float)
        ident = identification(scores, labels, WELL_SEPARATED)
        assert ident["bal_acc"] == pytest.approx(0.5, abs=0.05)
        assert abs(ident["ari"]) < 0.05

    def test_balanced_accuracy_is_not_fooled_by_rare_positives(self):
        rng = np.random.default_rng(8)
        # 1% positives, and a fit whose components are BOTH below every
        # positive: the argmax calls everything low.  Raw accuracy would be
        # 0.99; balanced accuracy must expose it as chance.
        scores = np.concatenate([rng.normal(0.2, 0.01, 9900), rng.normal(0.21, 0.01, 100)])
        labels = np.concatenate([np.zeros(9900), np.ones(100)])
        ident = identification(scores, labels, WELL_SEPARATED)
        assert ident["bal_acc"] < 0.6

    def test_single_class_returns_nan_rather_than_a_verdict(self):
        scores = np.linspace(0.0, 1.0, 100)
        labels = np.zeros(100)
        ident = identification(scores, labels, WELL_SEPARATED)
        assert math.isnan(ident["bal_acc"])
        # The component-mean error against the class that IS present still reports.
        assert math.isfinite(ident["mu_lo_err"])


class TestAnchorMassFraction:
    def test_the_shipped_setting_is_a_ten_thousandth(self):
        # kappa = 0.3, 20 votes, 50k haystack -> 6 / 50006.
        frac = anchor_mass_fraction(50_000, 20, 0.3)
        assert frac == pytest.approx(6.0 / 50_006.0, rel=1e-9)
        assert frac < 2e-4

    def test_the_share_is_an_order_of_magnitude_larger_on_a_small_haystack(self):
        # The 1.2e-4 headline is mostly the 50k denominator, and the denominator
        # is not a constant: `vg_scale_any` fits ~2k. The same votes then carry
        # >1e-3, which is the far side of the bar H3 was pre-registered against,
        # so the share has to be read per step rather than argued once (#3329).
        assert anchor_mass_fraction(2_000, 20, 0.3) > 10 * anchor_mass_fraction(50_000, 20, 0.3)
        assert anchor_mass_fraction(2_000, 20, 0.3) > 1e-3

    def test_the_count_is_votes_not_folds(self):
        # Passing the FOLD count where the vote count belongs is exactly the
        # error that flattened this statistic to 2.9e-4 at every click of the
        # first real run: two folds against ~2k reads a hundredfold smaller
        # than the ~50 held-out votes those folds actually anchored on.
        folds, votes = 2, 50
        assert anchor_mass_fraction(2_000, votes, 0.3) > 20 * anchor_mass_fraction(2_000, folds, 0.3)

    def test_labels_dominate_at_a_large_kappa(self):
        assert anchor_mass_fraction(100, 50, 100.0) > 0.9

    def test_no_anchors_is_zero(self):
        assert anchor_mass_fraction(1000, 0, 0.3) == 0.0

    def test_degenerate_inputs_are_nan(self):
        assert math.isnan(anchor_mass_fraction(-1, 5, 0.3))
        assert math.isnan(anchor_mass_fraction(0, 0, 0.0))


class TestFitQualityRow:
    def test_every_column_is_present_even_with_no_fit(self):
        row = fit_quality_row(np.zeros(10), None)
        assert set(row) == set(FIT_QUALITY_COLUMNS)

    def test_every_column_is_present_with_a_full_call(self):
        rng = np.random.default_rng(9)
        sample = _draw_mixture(rng, WELL_SEPARATED, 2000)
        labels = (rng.random(2000) < 0.2).astype(float)
        row = fit_quality_row(
            sample,
            WELL_SEPARATED,
            cut=0.5,
            labels=labels,
            label_scores=sample,
            unanchored_fit=WELL_SEPARATED,
            n_anchors=12,
            anchor_weight=0.3,
        )
        assert set(row) == set(FIT_QUALITY_COLUMNS)
        # An anchored fit identical to the unanchored one must read as zero
        # drift - the H3 statistic's null.
        assert row["anchored_dmu_lo"] == 0.0
        assert row["anchored_dmu_hi"] == 0.0

    def test_the_drift_is_anchored_minus_unanchored_and_is_signed(self):
        # The H3 statistic's ALTERNATIVE, which the null above cannot see: with
        # the counterfactual displaced by a known amount, the row must report
        # that amount, with the sign of (shipped - counterfactual). The old
        # `anchored_fit` parameter was never passed by any call site, so these
        # columns were NaN on every row of the first real run and H3 read as a
        # refutation it had never measured (#3329).
        moved = GmmFit1D(
            w_lo=WELL_SEPARATED.w_lo - 0.05,
            mu_lo=WELL_SEPARATED.mu_lo - 0.20,
            var_lo=WELL_SEPARATED.var_lo,
            w_hi=WELL_SEPARATED.w_hi + 0.05,
            mu_hi=WELL_SEPARATED.mu_hi + 0.10,
            var_hi=WELL_SEPARATED.var_hi,
        )
        row = fit_quality_row(np.zeros(10), WELL_SEPARATED, unanchored_fit=moved)
        assert row["anchored_dmu_lo"] == pytest.approx(0.20)
        assert row["anchored_dmu_hi"] == pytest.approx(-0.10)
        assert row["anchored_dw_lo"] == pytest.approx(0.05)

    def test_no_counterfactual_leaves_the_drift_unmeasured(self):
        # NaN means "not measured", and must never be read as "did not move".
        row = fit_quality_row(np.zeros(10), WELL_SEPARATED)
        assert math.isnan(row["anchored_dmu_lo"])
        assert math.isnan(row["anchored_dmu_hi"])
        assert math.isnan(row["anchored_dw_lo"])

    def test_anchor_columns_are_filled_without_a_fit(self):
        row = fit_quality_row(np.zeros(500), None, n_anchors=10, anchor_weight=0.3)
        assert row["anchor_n"] == 10.0
        assert row["anchor_kappa"] == 0.3
        assert math.isfinite(row["anchor_mass_frac"])
