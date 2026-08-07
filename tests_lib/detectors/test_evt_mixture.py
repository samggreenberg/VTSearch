"""The Gumbel + Normal score mixture (issues #2836, #2846).

The point of this component is not flexibility, it is *shape*: a max over region
nodes converges to a Gumbel, so the tests that matter are the ones that check
the fit recovers a planted Gumbel, that it beats the Gaussian mixture on data
that really is a maximum, and that it does **not** claim an advantage on data
that is not.

The #2846 half is about which *mode* the Gumbel lands on.  #2836 required the low
one and discarded every fit that said otherwise, which on production-like samples
was 14 % of them and the single largest reason the Gumbel arm silently degraded
to the midpoint.  So the tests here pin both the orientation-agnostic solve and
the fact that the incumbent rule still declines a swapped fit — the two have to
stay distinguishable or the re-measurement cannot tell them apart.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vtscore.training.evt_mixture import (
    CROSSING_REASONS,
    GumbelNormalFit1D,
    FIT_FAILURES,
    _gumbel_logpdf,
    _weighted_gumbel_mle,
    fit_gumbel_normal_mixture,
    fit_gumbel_normal_mixture_state,
    gaussian_mixture_mean_loglik,
)
from vtscore.eval.cut_rules import _to_logit
from vtscore.training.thresholds import fit_score_gmm, gmm_fit_array


def _mle(result: tuple[float, float] | None) -> tuple[float, float]:
    """An MLE narrowed to non-``None`` (its degenerate-input return)."""
    assert result is not None, "expected the MLE to converge on this sample"
    return result


def _root(x: float | None) -> float:
    """A crossing narrowed to non-``None`` (its no-root return)."""
    assert x is not None, "expected this fit to have a crossing"
    return x


class TestGumbelMle:
    def test_recovers_planted_parameters(self):
        rng = np.random.default_rng(3)
        x = rng.gumbel(loc=0.4, scale=0.08, size=60_000)
        loc, scale = _mle(_weighted_gumbel_mle(x, np.ones_like(x)))
        assert loc == pytest.approx(0.4, abs=5e-3)
        assert scale == pytest.approx(0.08, abs=5e-3)

    def test_survives_a_tiny_scale_on_unit_interval_scores(self):
        """The max-shifted sums must not underflow when ``e^{-x/scale}`` would.

        Sigmoid scores live in [0, 1] and a confident Bad mode is narrow, so the
        naive form evaluates ``exp(-1/0.002)`` and returns 0 for every point.
        """
        rng = np.random.default_rng(4)
        x = rng.gumbel(loc=0.9, scale=0.002, size=20_000)
        loc, scale = _mle(_weighted_gumbel_mle(x, np.ones_like(x)))
        assert math.isfinite(loc) and math.isfinite(scale)
        assert scale == pytest.approx(0.002, rel=0.1)

    def test_weights_select_a_subpopulation(self):
        rng = np.random.default_rng(6)
        a = rng.gumbel(0.2, 0.05, 20_000)
        b = rng.gumbel(0.8, 0.05, 20_000)
        x = np.concatenate([a, b])
        w = np.concatenate([np.ones(20_000), np.zeros(20_000)])
        loc, _scale = _mle(_weighted_gumbel_mle(x, w))
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
        assert fit.loc == pytest.approx(0.25, abs=0.02)
        assert fit.scale == pytest.approx(0.06, abs=0.02)
        assert fit.mu == pytest.approx(0.80, abs=0.03)

    def _max_pooled_logits(self, seed=13, n=8_000, m=24, prevalence=0.08):
        """Two modes, each media scored by the max over *m* region logits."""
        rng = np.random.default_rng(seed)
        labels = (rng.random(n) < prevalence).astype(float)
        bad = rng.normal(0.0, 1.0, size=(n, m))
        logits = bad.max(axis=1)
        idx = np.flatnonzero(labels == 1.0)
        others = bad[idx, 1:].max(axis=1) if m > 1 else np.full(idx.size, -np.inf)
        logits[idx] = np.maximum(rng.normal(3.0, 1.0, size=idx.size), others)
        return logits

    def test_beats_the_gaussian_mixture_on_max_pooled_logits(self):
        """The misspecification claim, on data generated the way region voting is.

        Negatives are all Bad regions, positives carry one object region, and each
        media scores by the max — exactly the regime where a Gaussian low
        component is the wrong shape.  But only on the **logit** axis, where the
        maximum is taken; see the sigmoid-axis control below, which is why the
        production fit and this one live on different axes.
        """
        logits = self._max_pooled_logits()
        arr = gmm_fit_array(logits)
        gauss = fit_score_gmm(arr)
        evt = fit_gumbel_normal_mixture(arr, init_split=None if gauss is None else gauss.midpoint())
        assert gauss is not None and evt is not None
        assert evt.mean_loglik > gaussian_mixture_mean_loglik(arr, gauss)

    def test_claims_no_advantage_on_the_sigmoid_axis(self):
        """The control for the *axis*: squashed scores are not Gumbel-shaped.

        A sigmoid is a strong nonlinear squash, so the max-of-m structure that the
        limit theorem describes is not visible on the score axis and a 2-Gaussian
        mixture fits it better.  This test exists so that a future change fitting
        the EVT component on scores instead of logits fails loudly rather than
        quietly reporting a worse mixture.
        """
        logits = self._max_pooled_logits()
        arr = gmm_fit_array(1.0 / (1.0 + np.exp(-logits)))
        gauss = fit_score_gmm(arr)
        evt = fit_gumbel_normal_mixture(arr, init_split=None if gauss is None else gauss.midpoint())
        assert gauss is not None and evt is not None
        assert evt.mean_loglik < gaussian_mixture_mean_loglik(arr, gauss)

    def test_claims_no_advantage_on_single_draw_logits(self):
        """m = 1 is not a maximum of anything, so the Gumbel must not win big.

        The control for the *family*: same axis and machinery, no max-pool. Keeps
        the max-pooled test from merely showing "more flexible shape fits better".
        """
        rng = np.random.default_rng(14)
        lo = rng.normal(-2.0, 1.0, 9_000)
        hi = rng.normal(2.0, 0.8, 1_000)
        arr = gmm_fit_array(np.concatenate([lo, hi]))
        gauss = fit_score_gmm(arr)
        evt = fit_gumbel_normal_mixture(arr, init_split=None if gauss is None else gauss.midpoint())
        assert gauss is not None and evt is not None
        gain = evt.mean_loglik - gaussian_mixture_mean_loglik(arr, gauss)
        assert gain < 0.02

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
        lo_term = math.log(fit.w_lo) + float(_gumbel_logpdf(arr, fit.loc, fit.scale)[0])
        hi_term = math.log(lam * fit.w_hi) - 0.5 * (math.log(2 * math.pi * fit.var) + (x - fit.mu) ** 2 / fit.var)
        assert lo_term - hi_term == pytest.approx(0.0, abs=1e-6)

    def test_root_lies_between_the_modes(self):
        fit = self._fit()
        x = _root(fit.crossing())
        assert fit.mode_lo < x < fit.mu

    def test_rate_crossing_is_weight_free(self):
        """Same defining property as the Gaussian family's prior-free cut."""
        base = self._fit()
        shifted = type(base)(
            w_gumbel=0.999,
            loc=base.loc,
            scale=base.scale,
            w_normal=0.001,
            mu=base.mu,
            var=base.var,
            mean_loglik=base.mean_loglik,
        )
        assert base.rate_crossing(1.0, 1.0) == pytest.approx(shifted.rate_crossing(1.0, 1.0), abs=1e-9)
        assert _root(shifted.crossing()) > _root(base.crossing())

    def test_lo_survival_matches_the_gumbel_cdf(self):
        fit = self._fit()
        x = 0.5
        expected = 1.0 - math.exp(-math.exp(-(x - fit.loc) / fit.scale))
        assert fit.lo_survival(x) == pytest.approx(expected, rel=1e-9)


class TestOrientation:
    """The #2846 repair: which mode the Gumbel lands on is an outcome, not a given."""

    def _swapped_fit(self):
        """Hand-built: the Gumbel on the **high** mode, a Normal below it.

        Constructed rather than fitted so the orientation under test is exact
        instead of a property of one EM trajectory; that EM really does reach
        this configuration on realistic samples is
        :meth:`test_em_reaches_a_swapped_fit_on_max_pooled_scores`.
        """
        return GumbelNormalFit1D(
            w_gumbel=0.25,
            loc=0.70,
            scale=0.09,
            w_normal=0.75,
            mu=0.20,
            var=0.03**2,
            mean_loglik=0.0,
        )

    def test_em_reaches_a_swapped_fit_on_max_pooled_scores(self):
        """The premise of the whole repair: EM does put the Gumbel on top.

        A sparse, weakly separated max-pooled sample — the early-ramp regime the
        safe threshold actually runs in — is a single right-skewed blob whose
        bulk *is* the negatives.  #2836 discarded this fit; #2846 measured that
        at 14 % of production-like fits.
        """
        rng = np.random.default_rng(525)
        n, m, prevalence = 500, 24, 0.005
        labels = (rng.random(n) < prevalence).astype(float)
        bad = rng.normal(0.0, 1.0, size=(n, m))
        logits = bad.max(axis=1)
        idx = np.flatnonzero(labels == 1.0)
        others = bad[idx, 1:].max(axis=1)
        logits[idx] = np.maximum(rng.normal(1.0, 1.0, size=idx.size), others)

        # The same pipeline `fit_both_mixtures` runs: fit the Gaussian on scores,
        # seed the EVT fit from its midpoint, fit on the logit axis.
        arr = gmm_fit_array(1.0 / (1.0 + np.exp(-logits)))
        gauss = fit_score_gmm(arr)
        assert gauss is not None
        u = _to_logit(arr)
        init = float(_to_logit(np.array([gauss.midpoint()]))[0])
        fit = fit_gumbel_normal_mixture(u, init_split=init)
        assert fit is not None, "#2836 returned None here — the bulk of the Gumbel arm's fallbacks"
        assert not fit.gumbel_is_low

    def test_a_swapped_fit_is_kept_not_discarded(self):
        """#2836 returned ``None`` here, which is the bulk of the Gumbel arm's fallbacks."""
        fit = self._swapped_fit()
        assert fit.mode_lo == pytest.approx(fit.mu)
        assert fit.mode_hi == pytest.approx(fit.loc)
        assert fit.w_lo == pytest.approx(fit.w_normal)
        assert fit.w_hi == pytest.approx(fit.w_gumbel)

    def test_the_incumbent_rule_still_declines_a_swapped_fit(self):
        """``gumbel_*`` must keep #2836's behaviour, or the two are not comparable."""
        fit = self._swapped_fit()
        reason, cut = fit.crossing_state()
        assert (reason, cut) == ("modes_swapped", None)
        assert fit.rate_crossing(1.0, 1.0) is None

    def test_allow_swapped_finds_the_root(self):
        fit = self._swapped_fit()
        reason, cut = fit.crossing_state(allow_swapped=True)
        assert reason == "ok"
        assert cut is not None
        assert fit.mode_lo < cut < fit.mode_hi

    def test_swapped_root_solves_the_density_equation(self):
        """With the roles reversed the low side is the Normal and the high the Gumbel."""
        fit = self._swapped_fit()
        x = _root(fit.crossing(allow_swapped=True))
        arr = np.array([x])
        lo_term = math.log(fit.w_lo) - 0.5 * (math.log(2 * math.pi * fit.var) + (x - fit.mu) ** 2 / fit.var)
        hi_term = math.log(fit.w_hi) + float(_gumbel_logpdf(arr, fit.loc, fit.scale)[0])
        assert lo_term - hi_term == pytest.approx(0.0, abs=1e-6)

    def test_lo_survival_reads_the_low_component_whichever_family_it_is(self):
        fit = self._swapped_fit()
        x = 0.5
        expected = 0.5 * math.erfc((x - fit.mu) / math.sqrt(2.0 * fit.var))
        assert fit.lo_survival(x) == pytest.approx(expected, rel=1e-9)

    @pytest.mark.parametrize("swapped", [False, True])
    def test_the_log_density_difference_is_monotone_between_the_modes(self, swapped):
        """Why endpoint sign-checking is sufficient and the bisection is exact.

        ``d/dx [log g - log n] = (e^{-z} - 1)/scale + (x - mu)/var`` has both
        terms non-positive on ``[loc, mu]`` (and both non-negative on ``[mu, loc]``
        when the fit is swapped), so the difference crosses zero at most once.  A
        bracket that shows no sign change really has no root in it.
        """
        fit = self._swapped_fit() if swapped else self._fit_low()
        xs = np.linspace(fit.mode_lo, fit.mode_hi, 500)
        d = np.array(
            [
                math.log(fit.w_lo)
                + float(fit._log_lo(np.array([x]))[0])  # noqa: SLF001 - the property under test
                - math.log(fit.w_hi)
                - float(fit._log_hi(np.array([x]))[0])  # noqa: SLF001
                for x in xs
            ]
        )
        assert float(np.max(np.diff(d))) <= 1e-9

    def _fit_low(self):
        rng = np.random.default_rng(15)
        lo = rng.gumbel(0.25, 0.05, 18_000)
        hi = rng.normal(0.80, 0.05, 2_000)
        fit = fit_gumbel_normal_mixture(np.concatenate([lo, hi]))
        assert fit is not None and fit.gumbel_is_low
        return fit


class TestFitFailureReasons:
    """A fallback has to stay attributable, not merely countable (issue #2846)."""

    def test_a_good_fit_reports_ok(self):
        rng = np.random.default_rng(22)
        arr = np.concatenate([rng.gumbel(0.25, 0.06, 9_000), rng.normal(0.80, 0.05, 1_000)])
        reason, fit = fit_gumbel_normal_mixture_state(arr)
        assert reason == "ok"
        assert fit is not None

    def test_too_few_points_is_named(self):
        reason, fit = fit_gumbel_normal_mixture_state(np.array([0.1, 0.2]))
        assert (reason, fit) == ("too_few", None)

    def test_every_reason_is_declared(self):
        """The vocabularies are the analyzer's join keys; typos must fail here."""
        rng = np.random.default_rng(23)
        arr = np.concatenate([rng.gumbel(0.25, 0.06, 9_000), rng.normal(0.80, 0.05, 1_000)])
        reason, fit = fit_gumbel_normal_mixture_state(arr)
        assert reason in FIT_FAILURES
        assert fit is not None
        for lam in (0.01, 1.0, 100.0):
            assert fit.crossing_state(lam)[0] in CROSSING_REASONS
            assert fit.crossing_state(lam, allow_swapped=True)[0] in CROSSING_REASONS

    def test_an_extreme_tilt_names_the_endpoint_that_failed(self):
        """The two ``owns`` reasons point at different repairs, so they must differ."""
        rng = np.random.default_rng(24)
        arr = np.concatenate([rng.gumbel(0.25, 0.06, 9_000), rng.normal(0.80, 0.05, 1_000)])
        fit = fit_gumbel_normal_mixture(arr)
        assert fit is not None
        # A huge lam prices the high component up until it owns the low mode too.
        assert fit.crossing_state(lam=1e100)[0] == "hi_owns_lo_mode"
        # A tiny one does the reverse.
        assert fit.crossing_state(lam=1e-100)[0] == "lo_owns_hi_mode"
