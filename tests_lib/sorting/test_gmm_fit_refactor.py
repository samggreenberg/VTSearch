"""Tests for the GMM fit/cut/blend decomposition in ``thresholds.py`` (issue #2799).

The #2799 measurement harness re-cuts one fitted GMM under several rules and
re-blends pre-computed cuts on the production label ramp, so the monolithic
``calculate_gmm_threshold`` / ``calculate_safe_threshold`` pair was split into
``gmm_fit_array`` + ``fit_score_gmm`` + ``GmmFit1D`` cuts and
``safe_blend_weight`` + ``blend_gmm_threshold``.  These tests pin the split's
contract: recomposing the pieces must reproduce the production functions
exactly, so the study's ``pooled_cross`` variant is guaranteed to measure the
threshold production actually ships.
"""

import math

import numpy as np
import pytest

from vtscore.training.thresholds import (
    GmmFit1D,
    _GMM_MAX_SAMPLES,
    blend_gmm_threshold,
    calculate_gmm_threshold,
    calculate_safe_threshold,
    fit_score_gmm,
    gmm_fit_array,
    safe_blend_weight,
)


def _bimodal(n=400, seed=7):
    rng = np.random.default_rng(seed)
    lo = rng.normal(0.2, 0.05, size=int(n * 0.8))
    hi = rng.normal(0.8, 0.03, size=n - int(n * 0.8))
    return np.clip(np.concatenate([lo, hi]), 0.0, 1.0).tolist()


class TestFitScoreGmm:
    def test_components_ordered_by_mean(self):
        fit = fit_score_gmm(np.asarray(_bimodal()))
        assert fit is not None
        assert fit.mu_lo < fit.mu_hi
        assert fit.w_lo > 0 and fit.w_hi > 0
        assert fit.var_lo > 0 and fit.var_hi > 0
        assert fit.w_lo + fit.w_hi == pytest.approx(1.0, abs=1e-9)

    def test_fewer_than_two_scores_is_none(self):
        assert fit_score_gmm(np.asarray([0.5])) is None
        assert fit_score_gmm(np.empty(0)) is None

    def test_midpoint_is_mean_of_means(self):
        fit = GmmFit1D(w_lo=0.7, mu_lo=0.2, var_lo=0.01, w_hi=0.3, mu_hi=0.8, var_hi=0.01)
        assert fit.midpoint() == pytest.approx(0.5)

    def test_crossing_falls_back_to_midpoint(self):
        # Equal variances + a 999:1 weight ratio push the crossing outside the
        # means -> the cut method must fall back to the midpoint.
        fit = GmmFit1D(w_lo=0.999, mu_lo=0.0, var_lo=1.0, w_hi=0.001, mu_hi=1.0, var_hi=1.0)
        assert fit.crossing_or_midpoint() == pytest.approx(fit.midpoint())


class TestGmmFitArray:
    def test_small_input_unchanged(self):
        scores = _bimodal(n=100)
        assert np.array_equal(gmm_fit_array(scores), np.asarray(scores, dtype=np.float64))

    def test_large_input_subsampled_deterministically(self):
        rng = np.random.default_rng(3)
        scores = rng.random(_GMM_MAX_SAMPLES + 1000)
        a = gmm_fit_array(scores)
        b = gmm_fit_array(scores)
        assert a.shape[0] == _GMM_MAX_SAMPLES
        assert np.array_equal(a, b)


class TestRecomposition:
    """The split pieces must recompose to the production functions exactly."""

    def test_calculate_gmm_threshold_equals_fit_plus_cut(self):
        scores = _bimodal()
        fit = fit_score_gmm(gmm_fit_array(scores))
        assert fit is not None
        assert calculate_gmm_threshold(scores) == fit.crossing_or_midpoint()

    @pytest.mark.parametrize("n_labels", [2, 5, 6, 7, 13, 19, 20, 40])
    def test_calculate_safe_threshold_equals_blend_of_parts(self, n_labels):
        scores = _bimodal()
        expected = blend_gmm_threshold(0.9, calculate_gmm_threshold(scores), n_labels)
        assert calculate_safe_threshold(0.9, scores, n_labels) == expected


class TestBlend:
    def test_ramp_endpoints(self):
        assert safe_blend_weight(2) == 0.0
        assert safe_blend_weight(6) == 0.0
        assert safe_blend_weight(13) == pytest.approx(0.5)
        assert safe_blend_weight(20) == 1.0
        assert safe_blend_weight(100) == 1.0

    def test_pure_gmm_below_ramp(self):
        assert blend_gmm_threshold(0.9, 0.3, 4) == pytest.approx(0.3)

    def test_pure_xcal_above_ramp(self):
        assert blend_gmm_threshold(0.9, 0.3, 25) == pytest.approx(0.9)

    def test_non_finite_guards(self):
        nan = float("nan")
        assert blend_gmm_threshold(nan, 0.3, 10) == 0.3
        assert blend_gmm_threshold(0.9, nan, 10) == 0.9
        assert blend_gmm_threshold(nan, nan, 10) == 0.5
        assert math.isfinite(blend_gmm_threshold(float("inf"), 0.3, 10))
