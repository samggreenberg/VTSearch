"""The unanchored mixture fit is ours, not sklearn's (issue #3585).

``fit_score_gmm`` is 91-95% of a cosine/text sort and the larger half of every
calibration fold, and until #3585 it was ``GaussianMixture(n_components=2,
random_state=42)`` - full-covariance machinery and a k-means init, on a problem
whose covariance is a scalar and whose EM loop already exists next door in
``_anchored_em``.  It is now a deterministic 2-means init plus that same loop
with no anchors.

That swap **moves the fit**, so these tests do not assert equality with the old
one.  They pin the three things that license the move instead:

* it is the *same estimator* - on a cleanly separated sample the two land on the
  same answer to within float noise;
* where they differ it is **not a coin flip**: the native fit has the higher
  mean log-likelihood, which is the objective both are maximising;
* the shapes production actually sees - saturated, max-pooled, constant - come
  back with a sane fit rather than the phantom second component sklearn's
  initialiser invented on a constant sample.

``docs/experiments/2026-09-13-gmm-init-3585/REPORT.md`` is the measurement on
real fold haystacks that decided it.
"""

import math
from unittest import mock

import numpy as np
import pytest

from vtscore.training.thresholds import (
    calculate_gmm_threshold,
    fit_score_gmm,
    fit_score_gmm_sklearn,
    gmm_fit_array,
)
from vtscore.training.thresholds import gmm as gmm_mod
from vtscore.training.thresholds.gmm import (
    _anchored_em,
    _EM_LOGLIK_TOL,
    _NO_ANCHORS,
    _plain_em,
    _two_means_init,
)


def _mean_loglik(fit, x: np.ndarray) -> float:
    """Mean log-likelihood of *fit* on *x* - what EM maximises, in both arms."""
    a = math.log(fit.w_lo) - 0.5 * np.log(2.0 * math.pi * fit.var_lo) - (x - fit.mu_lo) ** 2 / (2.0 * fit.var_lo)
    b = math.log(fit.w_hi) - 0.5 * np.log(2.0 * math.pi * fit.var_hi) - (x - fit.mu_hi) ** 2 / (2.0 * fit.var_hi)
    m = np.maximum(a, b)
    return float(np.mean(m + np.log(np.exp(a - m) + np.exp(b - m))))


def _bimodal(n=8000, prevalence=0.1, separation=0.6, sd=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n_hi = int(n * prevalence)
    return np.concatenate([rng.normal(0.2, sd, n - n_hi), rng.normal(0.2 + separation, sd, n_hi)])


def _saturated(n=9000, frac_pos=0.09, seed=0):
    """The #3166 shape: positives pinned at 1, negatives at 0, nothing between."""
    rng = np.random.default_rng(seed)
    n_pos = int(n * frac_pos)
    return np.clip(
        np.concatenate([np.abs(rng.normal(0.0, 1e-5, n - n_pos)), 1.0 - np.abs(rng.normal(0.0, 1e-5, n_pos))]),
        0.0,
        1.0,
    )


def _max_pooled(n=1000, mu=-2.0, sd=1.0, k=24, seed=0):
    """Region voting's Bad mode: a max over ~24 region nodes, so right-skewed."""
    rng = np.random.default_rng(seed)
    return np.max(1.0 / (1.0 + np.exp(-rng.normal(mu, sd, size=(n, k)))), axis=1)


class TestSameEstimator:
    """A separated sample has one answer, and both implementations find it."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_well_separated_agrees_with_sklearn(self, seed):
        x = _bimodal(seed=seed)
        native, reference = fit_score_gmm(x), fit_score_gmm_sklearn(x)
        assert native is not None and reference is not None
        assert native.midpoint() == pytest.approx(reference.midpoint(), abs=1e-9)
        assert native.mu_lo == pytest.approx(reference.mu_lo, abs=1e-9)
        assert native.mu_hi == pytest.approx(reference.mu_hi, abs=1e-9)
        assert native.w_lo == pytest.approx(reference.w_lo, abs=1e-9)

    def test_components_are_ordered_and_normalised(self):
        fit = fit_score_gmm(_bimodal())
        assert fit is not None
        assert fit.mu_lo < fit.mu_hi
        assert fit.w_lo + fit.w_hi == pytest.approx(1.0, abs=1e-12)
        assert fit.var_lo > 0.0 and fit.var_hi > 0.0

    def test_deterministic(self):
        x = _bimodal(seed=5)
        first, second = fit_score_gmm(x), fit_score_gmm(x)
        assert (first.w_lo, first.mu_lo, first.var_lo) == (second.w_lo, second.mu_lo, second.var_lo)
        assert (first.w_hi, first.mu_hi, first.var_hi) == (second.w_hi, second.mu_hi, second.var_hi)


class TestTheSameConvergence:
    """The two stop in the same place *in the same sense*, which is the point.

    Both loops stop when an EM iteration improves the mean log-likelihood by
    less than :data:`_EM_LOGLIK_TOL`, so neither is "more converged" than the
    other and the remaining difference is only where their inits started.  That
    equivalence is load-bearing: running our loop to a *tighter* stop finds a
    better fit (below) but a substantially different one, and on a barely
    bimodal sort that difference is thousands of medias (#3585's report).
    """

    @pytest.mark.parametrize("prevalence", [0.02, 0.1, 0.3])
    @pytest.mark.parametrize("separation", [0.15, 0.3, 0.6])
    def test_loglik_lands_within_the_stopping_tolerance_of_sklearns(self, prevalence, separation):
        x = _bimodal(n=6000, prevalence=prevalence, separation=separation, seed=11)
        native, reference = fit_score_gmm(x), fit_score_gmm_sklearn(x)
        assert native is not None and reference is not None
        # Two runs that stop at the same improvement threshold cannot end more
        # than a few of those thresholds apart on the objective.
        assert abs(_mean_loglik(native, x) - _mean_loglik(reference, x)) < 10 * _EM_LOGLIK_TOL

    @pytest.mark.parametrize("prevalence", [0.02, 0.1, 0.3])
    @pytest.mark.parametrize("separation", [0.15, 0.3, 0.6])
    def test_run_to_convergence_it_beats_sklearn(self, prevalence, separation, monkeypatch):
        """The loop is not the weaker optimiser; it stops early on purpose.

        With the stopping rule taken off, the native EM reaches a likelihood at
        least as high as sklearn's on every shape here - so what the shipped
        tolerance buys is speed and *agreement with the incumbent*, not a fit
        that could not be found.
        """
        monkeypatch.setattr(gmm_mod, "_EM_LOGLIK_TOL", 1e-12)
        monkeypatch.setattr(gmm_mod, "_EM_MAX_ITER", 2000)
        x = _bimodal(n=6000, prevalence=prevalence, separation=separation, seed=11)
        native, reference = fit_score_gmm(x), fit_score_gmm_sklearn(x)
        assert native is not None and reference is not None
        assert _mean_loglik(native, x) >= _mean_loglik(reference, x) - 1e-9

    def test_run_to_convergence_recovers_a_rare_component_sklearn_stops_short_of(self, monkeypatch):
        # 2% prevalence at 0.3 separation: both default tolerances stop with the
        # high component still spread across the gap; the mixture is really
        # there, and a converged fit finds it.
        monkeypatch.setattr(gmm_mod, "_EM_LOGLIK_TOL", 1e-12)
        monkeypatch.setattr(gmm_mod, "_EM_MAX_ITER", 2000)
        x = _bimodal(n=50_000, prevalence=0.02, separation=0.3, seed=13)
        native = fit_score_gmm(x)
        assert native is not None
        assert native.mu_lo == pytest.approx(0.2, abs=0.01)
        assert native.mu_hi == pytest.approx(0.5, abs=0.01)


class TestProductionShapes:
    def test_saturated_splits_at_the_two_pins(self):
        fit = fit_score_gmm(_saturated())
        assert fit is not None
        assert fit.mu_lo == pytest.approx(0.0, abs=1e-3)
        assert fit.mu_hi == pytest.approx(1.0, abs=1e-3)
        assert fit.midpoint() == pytest.approx(0.5, abs=1e-3)

    def test_max_pooled_mixture_is_fitted(self):
        scores = np.concatenate([_max_pooled(950, -2.0), _max_pooled(50, 1.5)])
        fit = fit_score_gmm(gmm_fit_array(scores))
        assert fit is not None
        assert fit.mu_lo < fit.mu_hi
        assert 0.0 < fit.midpoint() < 1.0

    def test_a_constant_haystack_cuts_at_the_constant(self):
        """sklearn answered this with a phantom component at zero; we do not.

        ``GaussianMixture`` on 50 copies of 0.3 returned means (0.0, 0.3) - so
        ``calculate_gmm_threshold`` said 0.15, a threshold below every score it
        was fitted on.  Two identical components on the value is the honest
        answer, and the anchored fit that initialises from here is what pulls
        them apart when there are votes to do it with.
        """
        fit = fit_score_gmm(np.full(50, 0.3))
        assert fit is not None
        assert fit.mu_lo == fit.mu_hi
        assert fit.mu_lo == pytest.approx(0.3, abs=1e-12)
        assert calculate_gmm_threshold([0.3] * 50) == pytest.approx(0.3, abs=1e-12)

    @pytest.mark.parametrize(
        "arr",
        [
            np.empty(0),
            np.array([0.5]),
            np.array([0.1, np.nan, 0.9]),
            np.array([0.1, np.inf, 0.9]),
        ],
    )
    def test_unfittable_samples_return_none(self, arr):
        assert fit_score_gmm(arr) is None


class TestTheInit:
    def test_split_is_a_2_means_fixed_point(self):
        """Every point is closer to its own component's mean than to the other's."""
        xs = np.sort(_bimodal(n=4000, seed=17))
        init = _two_means_init(xs)
        assert init is not None
        boundary = 0.5 * (init.mu_lo + init.mu_hi)
        split = int(np.searchsorted(xs, boundary, side="left"))
        assert split == int(round(init.w_lo * xs.size))

    def test_survives_a_saturated_sample_where_the_quartiles_coincide(self):
        # 9% positives: the 25th and 75th percentiles are both 0, so a
        # quartile-started Lloyd's would never move.  min/max does.
        init = _two_means_init(np.sort(_saturated()))
        assert init is not None
        assert init.mu_lo < 0.5 < init.mu_hi

    def test_constant_sample_initialises_two_identical_components(self):
        init = _two_means_init(np.full(10, 0.7))
        assert init is not None
        assert init.mu_lo == init.mu_hi == 0.7
        assert init.w_lo == init.w_hi == 0.5


class TestOneLoop:
    def test_the_plain_fit_is_the_anchored_loop_with_no_anchors(self):
        """Bit-for-bit: the two are the same code, not two copies of it."""
        x = _bimodal(n=3000, seed=23)
        init = _two_means_init(np.sort(x))
        assert init is not None
        direct = _anchored_em(x, _NO_ANCHORS, _NO_ANCHORS, init, 1.0, gmm_mod._EM_MAX_ITER, 1e-8, _EM_LOGLIK_TOL)
        via = _plain_em(x, init)
        assert direct is not None and via is not None
        assert (direct.w_lo, direct.mu_lo, direct.var_lo) == (via.w_lo, via.mu_lo, via.var_lo)
        assert (direct.w_hi, direct.mu_hi, direct.var_hi) == (via.w_hi, via.mu_hi, via.var_hi)

    def test_sklearn_is_not_consulted(self):
        """Production must not reach ``GaussianMixture`` at all any more.

        The point of the swap is that the sort path stops paying for it; an
        import left behind on some fallback branch would put it back on the
        path invisibly, since the answer would still look right.
        """
        x = _bimodal(n=2000, seed=29).tolist()
        expected = calculate_gmm_threshold(x)
        with mock.patch(
            "sklearn.mixture.GaussianMixture",
            side_effect=AssertionError("the production fit must not construct a GaussianMixture"),
        ):
            assert calculate_gmm_threshold(x) == expected
