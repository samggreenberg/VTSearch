"""The Gumbel(low) + Normal(high) score mixture (issue #2836).

The point of this component is not flexibility, it is *shape*: a max over region
nodes converges to a Gumbel, so the tests that matter are the ones that check
the fit recovers a planted Gumbel, that it beats the Gaussian mixture on data
that really is a maximum, and that it does **not** claim an advantage on data
that is not.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vtscore.training.evt_mixture import (
    _gumbel_logpdf,
    _weighted_gumbel_mle,
    fit_gumbel_normal_mixture,
    gaussian_mixture_mean_loglik,
)
from vtscore.training.thresholds import fit_score_gmm, gmm_fit_array


class TestGumbelMle:
    def test_recovers_planted_parameters(self):
        rng = np.random.default_rng(3)
        x = rng.gumbel(loc=0.4, scale=0.08, size=60_000)
        loc, scale = _weighted_gumbel_mle(x, np.ones_like(x))
        assert loc == pytest.approx(0.4, abs=5e-3)
        assert scale == pytest.approx(0.08, abs=5e-3)

    def test_survives_a_tiny_scale_on_unit_interval_scores(self):
        """The max-shifted sums must not underflow when ``e^{-x/scale}`` would.

        Sigmoid scores live in [0, 1] and a confident Bad mode is narrow, so the
        naive form evaluates ``exp(-1/0.002)`` and returns 0 for every point.
        """
        rng = np.random.default_rng(4)
        x = rng.gumbel(loc=0.9, scale=0.002, size=20_000)
        loc, scale = _weighted_gumbel_mle(x, np.ones_like(x))
        assert math.isfinite(loc) and math.isfinite(scale)
        assert scale == pytest.approx(0.002, rel=0.1)

    def test_weights_select_a_subpopulation(self):
        rng = np.random.default_rng(6)
        a = rng.gumbel(0.2, 0.05, 20_000)
        b = rng.gumbel(0.8, 0.05, 20_000)
        x = np.concatenate([a, b])
        w = np.concatenate([np.ones(20_000), np.zeros(20_000)])
        loc, _scale = _weighted_gumbel_mle(x, w)
        assert loc == pytest.approx(0.2, abs=1e-2)

    def test_degenerate_input_returns_none(self):
        x = np.full(100, 0.5)
        assert _weighted_gumbel_mle(x, np.ones_like(x)) is None
        assert _weighted_gumbel_mle(np.array([0.1, 0.2]), np.zeros(2)) is None


class TestGumbelNormalMixture:
    def test_recovers_a_planted_mixture(self):
        rng = np.random.default_rng(12)
        lo = rng.gumbel(0.25, 0.06, 18_000)
        hi = rng.normal(0.80, 0.05, 2_000)
        fit = fit_gumbel_normal_mixture(np.concatenate([lo, hi]))
        assert fit is not None
        assert fit.w_lo == pytest.approx(0.9, abs=0.05)
        assert fit.loc_lo == pytest.approx(0.25, abs=0.02)
        assert fit.scale_lo == pytest.approx(0.06, abs=0.02)
        assert fit.mu_hi == pytest.approx(0.80, abs=0.03)

    def test_beats_the_gaussian_mixture_on_max_pooled_scores(self):
        """The misspecification claim, on data generated the way region voting is.

        Each media's score is the max over 24 region logits, which is exactly the
        regime where a Gaussian low component is the wrong shape.
        """
        rng = np.random.default_rng(13)
        n, m = 8_000, 24
        bad = rng.normal(0.0, 1.0, size=(n, m)).max(axis=1)
        scores = 1.0 / (1.0 + np.exp(-bad))
        arr = gmm_fit_array(scores)
        gauss = fit_score_gmm(arr)
        evt = fit_gumbel_normal_mixture(arr, init_split=None if gauss is None else gauss.midpoint())
        assert gauss is not None and evt is not None
        assert evt.mean_loglik > gaussian_mixture_mean_loglik(arr, gauss)

    def test_claims_no_advantage_on_single_draw_scores(self):
        """m = 1 is not a maximum of anything, so the Gumbel must not win big.

        This is the control that keeps the previous test from merely showing
        "more flexible shape fits better".
        """
        rng = np.random.default_rng(14)
        lo = rng.normal(0.3, 0.08, 9_000)
        hi = rng.normal(0.75, 0.06, 1_000)
        arr = gmm_fit_array(np.concatenate([lo, hi]))
        gauss = fit_score_gmm(arr)
        evt = fit_gumbel_normal_mixture(arr, init_split=gauss.midpoint())
        assert gauss is not None and evt is not None
        gain = evt.mean_loglik - gaussian_mixture_mean_loglik(arr, gauss)
        assert gain < 0.05

    def test_too_few_points_returns_none(self):
        assert fit_gumbel_normal_mixture(np.array([0.1, 0.2])) is None


class TestEvtCrossing:
    def _fit(self):
        rng = np.random.default_rng(15)
        lo = rng.gumbel(0.25, 0.05, 18_000)
        hi = rng.normal(0.80, 0.05, 2_000)
        fit = fit_gumbel_normal_mixture(np.concatenate([lo, hi]))
        assert fit is not None
        return fit

    @pytest.mark.parametrize("lam", [0.5, 1.0, 5.0])
    def test_root_solves_the_density_equation(self, lam):
        fit = self._fit()
        x = fit.crossing(lam=lam)
        assert x is not None
        arr = np.array([x])
        lo_term = math.log(fit.w_lo) + float(_gumbel_logpdf(arr, fit.loc_lo, fit.scale_lo)[0])
        hi_term = math.log(lam * fit.w_hi) - 0.5 * (
            math.log(2 * math.pi * fit.var_hi) + (x - fit.mu_hi) ** 2 / fit.var_hi
        )
        assert lo_term - hi_term == pytest.approx(0.0, abs=1e-6)

    def test_root_lies_between_the_modes(self):
        fit = self._fit()
        x = fit.crossing()
        assert fit.mode_lo < x < fit.mu_hi

    def test_rate_crossing_is_weight_free(self):
        """Same defining property as the Gaussian family's prior-free cut."""
        base = self._fit()
        shifted = type(base)(
            w_lo=0.999,
            loc_lo=base.loc_lo,
            scale_lo=base.scale_lo,
            w_hi=0.001,
            mu_hi=base.mu_hi,
            var_hi=base.var_hi,
            mean_loglik=base.mean_loglik,
        )
        assert base.rate_crossing(1.0, 1.0) == pytest.approx(shifted.rate_crossing(1.0, 1.0), abs=1e-9)
        assert shifted.crossing() > base.crossing()

    def test_lo_survival_matches_the_gumbel_cdf(self):
        fit = self._fit()
        x = 0.5
        expected = 1.0 - math.exp(-math.exp(-(x - fit.loc_lo) / fit.scale_lo))
        assert fit.lo_survival(x) == pytest.approx(expected, rel=1e-9)
