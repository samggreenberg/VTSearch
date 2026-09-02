"""Tests for the subsampling fast-path in ``calculate_gmm_threshold``.

The GMM threshold runs on the *full* score distribution on every cosine/text
sort and in the safe-threshold blend, where N reaches ~250k (GUI Find) to 2M+
(CLI Find). Above ``_GMM_MAX_SAMPLES`` the fit uses a deterministic random
subsample; these tests pin that the subsample (a) produces a threshold
statistically indistinguishable from the full-sample fit and (b) is determinist
ic and bounded.
"""

from unittest import mock

import numpy as np

from vtscore.training.thresholds import _GMM_MAX_SAMPLES, calculate_gmm_threshold


def _bimodal_scores(n, seed=0):
    """A clean two-cluster score distribution centred at 0.2 (Bad) and 0.8 (Good)."""
    rng = np.random.default_rng(seed)
    lo = rng.normal(0.2, 0.05, size=n // 2)
    hi = rng.normal(0.8, 0.05, size=n - n // 2)
    return np.clip(np.concatenate([lo, hi]), 0.0, 1.0).tolist()


class TestGmmSubsample:
    def test_large_input_matches_subsample_fit(self):
        """Threshold on a >50k bimodal set sits at the midpoint (~0.5) of the two modes.

        The two component means are 0.2 and 0.8, so the midpoint threshold must
        land near 0.5 regardless of whether all N or only the subsample is fit.
        """
        scores = _bimodal_scores(200_000, seed=1)
        threshold = calculate_gmm_threshold(scores)
        assert 0.45 < threshold < 0.55

    def test_subsample_within_one_percent_of_full_fit(self):
        """The subsampled threshold is within 1% of fitting the entire population.

        Compare the production call (which subsamples internally) against a
        forced full-population fit on the same data: temporarily lift the cap so
        nothing is dropped.
        """
        from vtscore.training.thresholds import gmm as thr

        scores = _bimodal_scores(150_000, seed=2)
        subsampled = calculate_gmm_threshold(scores)

        original_cap = thr._GMM_MAX_SAMPLES
        try:
            thr._GMM_MAX_SAMPLES = len(scores) + 1  # disable subsampling
            full = calculate_gmm_threshold(scores)
        finally:
            thr._GMM_MAX_SAMPLES = original_cap

        assert abs(subsampled - full) <= 0.01 * full + 1e-3

    def test_deterministic_across_calls(self):
        """Seeded subsampling makes repeated calls on the same input identical."""
        scores = _bimodal_scores(120_000, seed=3)
        assert calculate_gmm_threshold(scores) == calculate_gmm_threshold(scores)

    def test_small_input_unchanged(self):
        """At or below the cap the full set is used (no subsample path)."""
        scores = _bimodal_scores(_GMM_MAX_SAMPLES, seed=4)
        threshold = calculate_gmm_threshold(scores)
        assert 0.45 < threshold < 0.55

    def test_large_input_is_bounded(self):
        """A 1M-point fit never hands the GMM more than ``_GMM_MAX_SAMPLES`` rows.

        Spy on ``GaussianMixture.fit`` and assert the row count it receives is
        capped by the subsample. This asserts the cap directly instead of timing
        the call, so it stays deterministic under xdist load (a wall-clock bound
        is flaky on a busy machine).
        """
        from sklearn.mixture import GaussianMixture  # noqa: PLC0415

        scores = _bimodal_scores(1_000_000, seed=5)
        seen_rows: list[int] = []
        original_fit = GaussianMixture.fit

        def spy_fit(self, X, *args, **kwargs):
            seen_rows.append(np.asarray(X).shape[0])
            return original_fit(self, X, *args, **kwargs)

        with mock.patch.object(GaussianMixture, "fit", spy_fit):
            calculate_gmm_threshold(scores)

        assert seen_rows, "GaussianMixture.fit was never called"
        assert all(n <= _GMM_MAX_SAMPLES for n in seen_rows)
