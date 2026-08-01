"""Tests for the equal-density cut rule in ``calculate_gmm_threshold`` (issue #2798).

The threshold used to be the midpoint between the two fitted component means.
That rule is only the Bayes boundary when the components are equal-weight and
equal-variance; under region voting a media's score is the max over ~24 region
nodes, so the Bad mode is an extreme-value statistic - wider, right-skewed, and
far heavier than the Good mode - and the midpoint lands inside Bad mass.  These
tests pin the crossing solver's algebra, its degenerate-case fallbacks, and the
end-to-end behaviour change on a max-pooled distribution.
"""

import math
from unittest import mock

import numpy as np

from vtscore.training.thresholds import (
    _weighted_gaussian_crossing,
    calculate_gmm_threshold,
)


def _log_density(w, mu, var, x):
    """Log of ``w * N(x; mu, var)``, up to the shared ``-0.5*log(2*pi)`` constant."""
    return math.log(w) - 0.5 * math.log(var) - (x - mu) ** 2 / (2.0 * var)


class TestWeightedGaussianCrossing:
    def test_equal_weight_equal_variance_is_the_midpoint(self):
        """The historical rule is the special case the new one generalises."""
        got = _weighted_gaussian_crossing(0.5, 0.2, 0.0025, 0.5, 0.8, 0.0025)
        assert got is not None
        assert abs(got - 0.5) < 1e-12

    def test_equal_variance_matches_closed_form(self):
        """With equal variances the crossing is ``mid + var*ln(w_lo/w_hi)/(mu_hi-mu_lo)``."""
        w_lo, mu_lo, w_hi, mu_hi, var = 0.8, 0.2, 0.2, 0.8, 0.01
        got = _weighted_gaussian_crossing(w_lo, mu_lo, var, w_hi, mu_hi, var)
        expected = (mu_lo + mu_hi) / 2.0 + var * math.log(w_lo / w_hi) / (mu_hi - mu_lo)
        assert got is not None
        assert abs(got - expected) < 1e-12

    def test_densities_are_equal_at_the_returned_score(self):
        """The defining property, checked directly on unequal weights *and* variances."""
        params = (0.9, 0.3, 0.02, 0.1, 0.8, 0.002)
        got = _weighted_gaussian_crossing(*params)
        assert got is not None
        w_lo, mu_lo, var_lo, w_hi, mu_hi, var_hi = params
        gap = _log_density(w_lo, mu_lo, var_lo, got) - _log_density(w_hi, mu_hi, var_hi, got)
        assert abs(gap) < 1e-9

    def test_wider_heavier_low_component_cuts_above_the_midpoint(self):
        """The max-pool shape: a fat, heavy Bad mode pushes the cut up, not down."""
        got = _weighted_gaussian_crossing(0.9, 0.3, 0.02, 0.1, 0.8, 0.002)
        assert got is not None
        assert got > (0.3 + 0.8) / 2.0

    def test_narrow_heavy_low_component_cuts_below_the_midpoint(self):
        """The rule is not a one-way ratchet - a tight Bad mode lets the cut fall."""
        got = _weighted_gaussian_crossing(0.7, 0.1, 0.001, 0.3, 0.6, 0.05)
        assert got is not None
        assert got < (0.1 + 0.6) / 2.0

    def test_crossing_lies_strictly_between_the_means(self):
        rng = np.random.default_rng(11)
        for _ in range(200):
            w_lo = float(rng.uniform(0.05, 0.95))
            mu_lo = float(rng.uniform(-1.0, 0.4))
            mu_hi = mu_lo + float(rng.uniform(0.05, 1.5))
            var_lo = float(rng.uniform(1e-4, 0.2))
            var_hi = float(rng.uniform(1e-4, 0.2))
            got = _weighted_gaussian_crossing(w_lo, mu_lo, var_lo, 1.0 - w_lo, mu_hi, var_hi)
            if got is not None:
                assert mu_lo < got < mu_hi

    def test_extreme_weight_ratio_has_no_crossing_between_the_means(self):
        """Equal variances plus a 999:1 weight ratio push the root past the Good mean."""
        assert _weighted_gaussian_crossing(0.999, 0.0, 1.0, 0.001, 1.0, 1.0) is None

    def test_degenerate_inputs_return_none(self):
        """Non-ordered means, and non-positive weights/variances, all fall back."""
        assert _weighted_gaussian_crossing(0.5, 0.5, 0.01, 0.5, 0.5, 0.01) is None  # equal means
        assert _weighted_gaussian_crossing(0.5, 0.8, 0.01, 0.5, 0.2, 0.01) is None  # mis-ordered
        assert _weighted_gaussian_crossing(0.0, 0.2, 0.01, 1.0, 0.8, 0.01) is None  # zero weight
        assert _weighted_gaussian_crossing(0.5, 0.2, 0.0, 0.5, 0.8, 0.01) is None  # zero variance
        assert _weighted_gaussian_crossing(0.5, float("nan"), 0.01, 0.5, 0.8, 0.01) is None

    def test_near_equal_variances_converge_to_the_linear_root(self):
        """``a -> 0`` must not blow up: the quadratic root tracks the linear one."""
        w_lo, mu_lo, w_hi, mu_hi, var = 0.75, 0.2, 0.25, 0.8, 0.01
        linear = (mu_lo + mu_hi) / 2.0 + var * math.log(w_lo / w_hi) / (mu_hi - mu_lo)
        for eps in (1e-6, 1e-9, 1e-12, 1e-15):
            got = _weighted_gaussian_crossing(w_lo, mu_lo, var, w_hi, mu_hi, var * (1.0 + eps))
            assert got is not None
            assert abs(got - linear) < 1e-6


def _pooled_scores(n, mu, sd, rng, k=24):
    """Region max-pooling: each media's score is the max over *k* region nodes."""
    logits = rng.normal(mu, sd, size=(n, k))
    return np.max(1.0 / (1.0 + np.exp(-logits)), axis=1)


class TestGmmThresholdCutRule:
    def test_symmetric_bimodal_still_cuts_at_the_midpoint(self):
        """Equal-weight, equal-variance modes: the rule change is a no-op."""
        rng = np.random.default_rng(1)
        lo = rng.normal(0.2, 0.05, size=5000)
        hi = rng.normal(0.8, 0.05, size=5000)
        scores = np.clip(np.concatenate([lo, hi]), 0.0, 1.0).tolist()
        assert abs(calculate_gmm_threshold(scores) - 0.5) < 0.02

    def test_max_pooled_distribution_cuts_above_the_midpoint(self):
        """On a region-voted score distribution the cut moves up off the midpoint."""
        from sklearn.mixture import GaussianMixture

        rng = np.random.default_rng(7)
        bad = _pooled_scores(950, -2.0, 1.0, rng)
        good = _pooled_scores(50, 1.5, 1.0, rng)
        scores = np.concatenate([bad, good])

        threshold = calculate_gmm_threshold(scores.tolist())
        fitted = GaussianMixture(n_components=2, random_state=42).fit(scores.reshape(-1, 1))
        assert fitted.means_ is not None
        midpoint = float(np.ravel(fitted.means_).mean())

        assert threshold > midpoint
        # And it buys real precision: same recall, far fewer Bad medias admitted.
        assert (good >= threshold).sum() == (good >= midpoint).sum()
        assert (bad >= threshold).sum() < (bad >= midpoint).sum()

    def test_falls_back_to_the_midpoint_when_no_crossing_exists(self):
        """A ``None`` from the solver must land exactly on the old midpoint rule."""
        from sklearn.mixture import GaussianMixture

        rng = np.random.default_rng(2)
        scores = np.clip(
            np.concatenate([rng.normal(0.25, 0.04, size=800), rng.normal(0.75, 0.09, size=200)]),
            0.0,
            1.0,
        )
        fitted = GaussianMixture(n_components=2, random_state=42).fit(scores.reshape(-1, 1))
        assert fitted.means_ is not None
        midpoint = float(np.ravel(fitted.means_).mean())

        with mock.patch("vtscore.training.thresholds._weighted_gaussian_crossing", return_value=None):
            assert calculate_gmm_threshold(scores.tolist()) == midpoint

    def test_threshold_stays_inside_the_score_range(self):
        rng = np.random.default_rng(5)
        scores = _pooled_scores(2000, -1.5, 1.2, rng)
        threshold = calculate_gmm_threshold(scores.tolist())
        assert float(scores.min()) <= threshold <= float(scores.max())

    def test_deterministic_across_calls(self):
        rng = np.random.default_rng(9)
        scores = np.concatenate([_pooled_scores(900, -2.0, 1.0, rng), _pooled_scores(100, 1.0, 0.8, rng)]).tolist()
        assert calculate_gmm_threshold(scores) == calculate_gmm_threshold(scores)
