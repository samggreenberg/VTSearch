"""The GMM cut must be reproducible on a **saturated** score distribution (#3166).

`caltech101_m` / `siglip` / `airplanes` is saturated: positives score near 1.0,
negatives near 0, and nothing sits between them.  Re-running that benchmark cell
on a different node reproduced every other column bit-for-bit (`n_good`,
`n_bad`, `average_precision`, `auroc` all differed by exactly 0) while the
`threshold` moved by 0.0264.

The cause is amplification, not a wandering fit.  `FoldAnchoredCut.threshold_at`
realises its combined fold quantile with ``np.quantile``, which interpolates
*linearly* between adjacent order statistics.  Across the empty interval of a
saturated distribution that interpolation has gain ``(n - 1) * gap``: at
``n = 9000`` and a unit-wide gap, ``dt = 8999 * dq``, so a quantile difference of
2.9e-6 - well inside what differing BLAS kernels or thread counts produce
between two machines - lands exactly on the observed 0.026.

The fix (:func:`snap_cut_to_sample`) canonicalises a cut that falls strictly
inside an empty interval to that interval's midpoint, which removes the gain
entirely while leaving the admitted set untouched.  These tests pin both halves:
the cut is decision-exact, and it does not move under perturbations - of the
fit, of the quantile, or of the BLAS thread count - that leave the admitted set
alone.
"""

import numpy as np
import pytest

from vtscore.training.thresholds import (
    FoldAnchoredCut,
    GmmFit1D,
    calculate_gmm_threshold,
    fit_score_gmm,
    fold_anchored_gmm_threshold,
    gmm_fit_array,
    snap_cut_to_sample,
)


def saturated_scores(n=9000, frac_pos=0.09, seed=0, sigma=1e-5):
    """Two clusters pinned at 0 and 1 with no interior mass - the #3166 shape."""
    rng = np.random.default_rng(seed)
    n_pos = int(n * frac_pos)
    neg = np.abs(rng.normal(0.0, sigma, n - n_pos))
    pos = 1.0 - np.abs(rng.normal(0.0, sigma, n_pos))
    return np.clip(np.concatenate([neg, pos]), 0.0, 1.0)


def admitted(scores, threshold):
    """The Good set a threshold selects - what must not change when a cut moves."""
    return frozenset(np.flatnonzero(np.asarray(scores) >= threshold).tolist())


class TestSnapCutToSample:
    """Invariants of the canonicalisation itself."""

    def test_interior_cut_snaps_to_gap_midpoint(self):
        src = np.array([0.0, 0.1, 0.9, 1.0])
        assert snap_cut_to_sample(0.2, src) == pytest.approx(0.5)
        assert snap_cut_to_sample(0.8, src) == pytest.approx(0.5)

    def test_snap_is_decision_exact(self):
        """No element of the sample changes side, for any cut, anywhere."""
        src = np.sort(saturated_scores(n=400, seed=3))
        for cut in np.linspace(-0.2, 1.2, 401):
            assert admitted(src, snap_cut_to_sample(float(cut), src)) == admitted(src, float(cut))

    def test_cut_on_an_observed_score_is_left_alone(self):
        """That value *is* identifiable - it decides its own score's verdict."""
        src = np.array([0.0, 0.25, 0.75, 1.0])
        for v in src:
            assert snap_cut_to_sample(float(v), src) == float(v)

    def test_out_of_support_and_empty_pass_through(self):
        src = np.array([0.2, 0.8])
        assert snap_cut_to_sample(-1.0, src) == -1.0
        assert snap_cut_to_sample(2.0, src) == 2.0
        assert snap_cut_to_sample(0.5, np.array([])) == 0.5
        assert np.isnan(snap_cut_to_sample(float("nan"), src))

    def test_snap_is_idempotent_and_monotone(self):
        src = np.sort(saturated_scores(n=200, seed=4))
        cuts = np.linspace(0.0, 1.0, 257)
        snapped = [snap_cut_to_sample(float(c), src) for c in cuts]
        assert snapped == [snap_cut_to_sample(v, src) for v in snapped]
        assert all(a <= b for a, b in zip(snapped, snapped[1:], strict=False))


class TestSaturatedThresholdIsStable:
    """The amplifier is gone: sub-order-statistic wobble cannot move the cut."""

    def _cut(self, mu_lo, haystack):
        fit = GmmFit1D(w_lo=0.91, mu_lo=mu_lo, var_lo=1e-6, w_hi=0.09, mu_hi=1.0, var_hi=1e-6)
        return FoldAnchoredCut(
            fits=(fit,), fold_haystacks=(haystack,), final_haystack=haystack, n_anchored=1
        )

    def test_float_level_fit_wobble_leaves_the_threshold_bit_identical(self):
        """The regression: pre-#3166 this slid smoothly across the empty gap."""
        haystack = np.sort(saturated_scores())
        base = self._cut(0.0, haystack).threshold_at(0)
        for delta in (1e-15, 1e-12, 1e-9, 1e-6, 1e-3):
            assert self._cut(delta, haystack).threshold_at(0) == base

    def test_threshold_lands_in_the_empty_interval(self):
        """Sanity: the cut separates the two modes, it is not merely constant."""
        haystack = np.sort(saturated_scores())
        threshold = self._cut(0.0, haystack).threshold_at(0)
        assert float(haystack[haystack < 0.5].max()) < threshold < float(haystack[haystack > 0.5].min())

    def test_quantile_wobble_that_crosses_no_order_statistic_is_invisible(self):
        """``dt = dq * (n - 1) * gap`` was the whole bug; the gain is now zero."""
        haystack = np.sort(saturated_scores())
        snapped = {
            snap_cut_to_sample(float(np.quantile(haystack, 0.91 + dq)), haystack)
            for dq in (0.0, 1e-7, 1e-6, 3e-6, 1e-5)
        }
        assert len(snapped) == 1
        raw = {float(np.quantile(haystack, 0.91 + dq)) for dq in (0.0, 1e-7, 1e-6, 3e-6, 1e-5)}
        # Guard the guard: those same wobbles really did move the raw quantile
        # by ~0.09, three orders of magnitude more than the wobble itself.
        assert max(raw) - min(raw) > 0.05


class TestSaturatedFitUnderThreadCounts:
    """The issue's own reproduction: refit under different BLAS thread counts.

    Cross-*machine* BLAS differences cannot be staged in-process, but the thread
    count is the same knob and is the one the issue names.  The assertion is
    one-directional - the shipped threshold must not depend on it.
    """

    @pytest.mark.parametrize("frac_pos", [0.02, 0.09, 0.5])
    def test_threshold_is_independent_of_thread_count(self, frac_pos):
        threadpool_limits = pytest.importorskip("threadpoolctl").threadpool_limits
        scores = saturated_scores(frac_pos=frac_pos, seed=11)
        folds = [saturated_scores(frac_pos=frac_pos, seed=20 + k) for k in range(3)]
        rng = np.random.default_rng(5)
        anchors = []
        for fold in folds:
            idx = rng.choice(fold.size, 24, replace=False)
            picked = fold[idx]
            anchors.append((picked.tolist(), (picked > 0.5).astype(float).tolist()))

        anchored, plain = set(), set()
        for limit in (1, 2, 4):
            with threadpool_limits(limits=limit):
                anchored.add(fold_anchored_gmm_threshold(folds, anchors, scores, 0)[0])
                plain.add(calculate_gmm_threshold(scores.tolist()))
        assert len(anchored) == 1, anchored
        assert len(plain) == 1, plain


class TestSnappedCutPreservesTheAdmittedSet:
    """Canonicalising must not change what a detector calls Good."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_gmm_threshold_admits_exactly_the_unsnapped_set(self, seed):
        scores = saturated_scores(seed=seed)
        arr = gmm_fit_array(scores.tolist())
        fit = fit_score_gmm(arr)
        assert fit is not None
        assert admitted(arr, calculate_gmm_threshold(scores.tolist())) == admitted(arr, fit.midpoint())

    def test_non_saturated_distribution_is_barely_moved(self):
        """On a distribution with interior mass the gaps are ~1/n, so is the snap."""
        rng = np.random.default_rng(7)
        scores = np.clip(
            np.concatenate([rng.normal(0.3, 0.1, 5000), rng.normal(0.7, 0.1, 5000)]), 0.0, 1.0
        )
        arr = gmm_fit_array(scores.tolist())
        fit = fit_score_gmm(arr)
        assert fit is not None
        assert calculate_gmm_threshold(scores.tolist()) == pytest.approx(fit.midpoint(), abs=1e-3)
