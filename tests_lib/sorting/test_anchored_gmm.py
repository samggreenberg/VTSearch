"""Tests for the label-anchored mixture estimators (issue #2852).

Covers the anchored EM itself (seeded synthetic bimodal / unimodal /
anchors-contradict-modes cases, determinism, degeneracy fallbacks), the
rank-transfer scale carrier, and the fold-anchored ("cross-LabeledGMM")
combiner.  Library tier: pure ``vtscore.training.thresholds``, no app imports.

``TestAnchoredEmEquivalence`` additionally pins the #3558 optimisation of the
EM loop to bit-for-bit equality with the two-column form it replaced, which is
kept here verbatim as ``_reference_anchored_em``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vtscore.training.thresholds import (
    FOLD_ANCHOR_COMBINE,
    FOLD_ANCHOR_CUT_RULE,
    FOLD_ANCHOR_WEIGHT,
    CUT_KIND_CONTINUED,
    CUT_KIND_DEGENERATE_MIDPOINT,
    CUT_KIND_INTERIOR,
    FoldAnchoredCut,
    GmmFit1D,
    anchored_gmm_fit,
    fit_anchored_score_gmm,
    fit_fold_anchored_cut,
    fit_score_gmm,
    fold_anchored_gmm_threshold,
    gmm_cut_from_fit,
    gmm_fit_array,
    inclusion_cost_weights,
    rank_transfer,
)
from vtscore.training.thresholds.gmm import _anchored_em


def _bimodal(
    rng: np.random.Generator, n: int = 2000, mu_lo: float = 0.2, mu_hi: float = 0.8, sd: float = 0.05, w_hi: float = 0.3
) -> np.ndarray:
    n_hi = int(n * w_hi)
    return np.concatenate([rng.normal(mu_lo, sd, n - n_hi), rng.normal(mu_hi, sd, n_hi)])


def _anchors_from_modes(rng: np.random.Generator, n_each: int = 10) -> tuple[np.ndarray, np.ndarray]:
    scores = np.concatenate([rng.normal(0.2, 0.05, n_each), rng.normal(0.8, 0.05, n_each)])
    labels = np.concatenate([np.zeros(n_each), np.ones(n_each)])
    return scores, labels


class TestAnchoredEm:
    def test_bimodal_with_agreeing_anchors_recovers_modes(self):
        rng = np.random.default_rng(42)
        arr = _bimodal(rng)
        a_scores, a_labels = _anchors_from_modes(rng)
        fit, provenance = fit_anchored_score_gmm(arr, a_scores, a_labels)
        assert provenance == "anchored"
        assert fit is not None
        assert abs(fit.mu_lo - 0.2) < 0.03
        assert abs(fit.mu_hi - 0.8) < 0.03
        assert 0.3 < fit.midpoint() < 0.7

    def test_deterministic(self):
        rng = np.random.default_rng(7)
        arr = _bimodal(rng)
        a_scores, a_labels = _anchors_from_modes(rng)
        fit1, p1 = fit_anchored_score_gmm(arr, a_scores, a_labels)
        fit2, p2 = fit_anchored_score_gmm(arr, a_scores, a_labels)
        assert p1 == p2 == "anchored"
        assert fit1 == fit2  # frozen dataclass: bit-for-bit parameter equality

    def test_unimodal_anchors_identify_the_split(self):
        # A single broad mode gives the unanchored EM no class information;
        # strong anchors must place the components on the labelled sides.
        rng = np.random.default_rng(3)
        arr = rng.normal(0.5, 0.15, 3000)
        a_scores = np.concatenate([rng.normal(0.25, 0.03, 15), rng.normal(0.75, 0.03, 15)])
        a_labels = np.concatenate([np.zeros(15), np.ones(15)])
        fit, provenance = fit_anchored_score_gmm(arr, a_scores, a_labels, anchor_weight=100.0)
        assert provenance == "anchored"
        assert fit is not None
        assert fit.mu_lo < 0.5 < fit.mu_hi
        assert 0.3 < fit.midpoint() < 0.7

    def test_anchors_contradicting_modes_fall_back_to_unanchored(self):
        # Good anchors planted in the LOW population mode: the anchored means
        # invert, the anchored path must report the degeneracy, and the
        # production-shaped wrapper must fall back to the unanchored fit -
        # never to 0.5.
        rng = np.random.default_rng(11)
        arr = _bimodal(rng)
        a_scores = np.concatenate([rng.normal(0.2, 0.05, 20), rng.normal(0.8, 0.05, 20)])
        a_labels = np.concatenate([np.ones(20), np.zeros(20)])  # deliberately inverted
        fit, provenance = fit_anchored_score_gmm(arr, a_scores, a_labels, anchor_weight=1000.0)
        assert fit is None
        assert provenance == "inverted_means"

        wrapped, wrapped_prov = anchored_gmm_fit(arr, a_scores, a_labels, anchor_weight=1000.0)
        assert wrapped_prov == "unanchored:inverted_means"
        assert wrapped == fit_score_gmm(gmm_fit_array(arr))

    def test_interleaved_anchors_are_not_degenerate(self):
        # Ranking errors among the anchors are the NORMAL regime, not a
        # degeneracy: the estimator is moment-based, so a Bad scoring above a
        # Good (or a 24%-pairwise-inverted heavy overlap) just widens the
        # components.  The fallback must trigger only on *mean* inversion of
        # the anchor classes - the model ranking the labelset worse than
        # chance - never on individual crossing pairs.  (An order-statistic
        # sensitivity here would rebuild exactly the conformal rule's
        # small-sample noise this estimator exists to escape.)
        rng = np.random.default_rng(17)
        arr = _bimodal(rng)

        # One fully inverted pair on top of clean mode anchors.
        a_scores = np.concatenate([rng.normal(0.2, 0.05, 10), rng.normal(0.8, 0.05, 10), [0.95, 0.05]])
        a_labels = np.concatenate([np.zeros(10), np.ones(10), [0.0, 1.0]])
        for weight in (1.0, 100.0, 1000.0):
            fit, provenance = fit_anchored_score_gmm(arr, a_scores, a_labels, anchor_weight=weight)
            assert provenance == "anchored", (weight, provenance)
            assert fit is not None and 0.3 < fit.midpoint() < 0.7

        # Heavily overlapping anchor classes: many crossing pairs, ordered means.
        a_bad = rng.normal(0.45, 0.12, 20)
        a_good = rng.normal(0.55, 0.12, 20)
        assert any(b > g for b in a_bad for g in a_good)  # the overlap is real
        a_scores2 = np.concatenate([a_bad, a_good])
        a_labels2 = np.concatenate([np.zeros(20), np.ones(20)])
        for weight in (1.0, 100.0, 1000.0):
            fit, provenance = fit_anchored_score_gmm(arr, a_scores2, a_labels2, anchor_weight=weight)
            assert provenance == "anchored", (weight, provenance)
            assert fit is not None and fit.mu_hi > fit.mu_lo

    def test_huge_weight_converges_to_anchor_class_means(self):
        rng = np.random.default_rng(5)
        arr = _bimodal(rng)
        a_scores = np.concatenate([rng.normal(0.3, 0.02, 25), rng.normal(0.7, 0.02, 25)])
        a_labels = np.concatenate([np.zeros(25), np.ones(25)])
        fit, provenance = fit_anchored_score_gmm(arr, a_scores, a_labels, anchor_weight=1e6)
        assert provenance == "anchored"
        assert fit is not None
        assert abs(fit.mu_lo - float(a_scores[:25].mean())) < 0.02
        assert abs(fit.mu_hi - float(a_scores[25:].mean())) < 0.02

    def test_tiny_weight_stays_close_to_unanchored(self):
        rng = np.random.default_rng(9)
        arr = _bimodal(rng)
        a_scores, a_labels = _anchors_from_modes(rng, n_each=5)
        anchored, provenance = fit_anchored_score_gmm(arr, a_scores, a_labels, anchor_weight=1e-3)
        plain = fit_score_gmm(arr)
        assert provenance == "anchored"
        assert anchored is not None and plain is not None
        assert abs(anchored.mu_lo - plain.mu_lo) < 0.02
        assert abs(anchored.mu_hi - plain.mu_hi) < 0.02

    def test_no_anchor_edge_cases(self):
        rng = np.random.default_rng(1)
        arr = _bimodal(rng)
        assert fit_anchored_score_gmm(arr, [], []) == (None, "no_anchors")
        assert fit_anchored_score_gmm(arr, [0.5], [1.0], anchor_weight=0.0) == (None, "no_anchors")
        assert fit_anchored_score_gmm(np.array([0.5]), [0.5], [1.0]) == (None, "too_few_scores")

    def test_one_sided_anchors_are_allowed(self):
        rng = np.random.default_rng(13)
        arr = _bimodal(rng)
        fit, provenance = fit_anchored_score_gmm(arr, rng.normal(0.8, 0.05, 10), np.ones(10))
        assert provenance == "anchored"
        assert fit is not None
        assert fit.mu_hi > fit.mu_lo


def _reference_anchored_em(
    x: np.ndarray,
    a_lo: np.ndarray,
    a_hi: np.ndarray,
    init: GmmFit1D,
    anchor_weight: float,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> GmmFit1D | None:
    """The pre-#3558 anchored EM, verbatim, as the equivalence reference.

    Kept as the readable two-column form the shipped
    :func:`~vtscore.training.thresholds.gmm._anchored_em` was optimised out of:
    one ``(n, 2)`` responsibility array per iteration, ``axis=1`` reductions,
    fresh temporaries throughout.  It is a *specification*, not a copy to keep
    in sync - if a future change to the shipped loop makes this disagree, the
    shipped loop has moved the threshold, which is the thing #3558 was not
    allowed to do.
    """
    lam = float(anchor_weight)
    n = float(x.size)
    n_lo, n_hi = float(a_lo.size), float(a_hi.size)
    total_mass = n + lam * (n_lo + n_hi)

    w = np.array([init.w_lo, init.w_hi], dtype=np.float64)
    mu = np.array([init.mu_lo, init.mu_hi], dtype=np.float64)
    var = np.array([init.var_lo, init.var_hi], dtype=np.float64)

    pooled = np.concatenate([x, a_lo, a_hi])
    var_floor = max(1e-12, 1e-6 * float(np.var(pooled)))
    var = np.maximum(var, var_floor)

    sum_a_lo, sum_a_hi = float(a_lo.sum()), float(a_hi.sum())

    for _ in range(max_iter):
        log_p = (
            np.log(np.maximum(w, 1e-300))[None, :]
            - 0.5 * np.log(2.0 * math.pi * var)[None, :]
            - (x[:, None] - mu[None, :]) ** 2 / (2.0 * var[None, :])
        )
        log_p -= log_p.max(axis=1, keepdims=True)
        r = np.exp(log_p)
        r /= r.sum(axis=1, keepdims=True)

        m_lo = float(r[:, 0].sum()) + lam * n_lo
        m_hi = float(r[:, 1].sum()) + lam * n_hi
        if not (m_lo > 0.0 and m_hi > 0.0):
            return None
        mu_new = np.array(
            [
                (float(np.sum(r[:, 0] * x)) + lam * sum_a_lo) / m_lo,
                (float(np.sum(r[:, 1] * x)) + lam * sum_a_hi) / m_hi,
            ]
        )
        var_new = np.array(
            [
                (float(np.sum(r[:, 0] * (x - mu_new[0]) ** 2)) + lam * float(((a_lo - mu_new[0]) ** 2).sum())) / m_lo,
                (float(np.sum(r[:, 1] * (x - mu_new[1]) ** 2)) + lam * float(((a_hi - mu_new[1]) ** 2).sum())) / m_hi,
            ]
        )
        var_new = np.maximum(var_new, var_floor)
        w_new = np.array([m_lo, m_hi]) / total_mass

        if not (np.all(np.isfinite(mu_new)) and np.all(np.isfinite(var_new)) and np.all(np.isfinite(w_new))):
            return None
        delta = max(
            float(np.max(np.abs(mu_new - mu))),
            float(np.max(np.abs(var_new - var))),
            float(np.max(np.abs(w_new - w))),
        )
        mu, var, w = mu_new, var_new, w_new
        if delta < tol:
            break

    return GmmFit1D(
        w_lo=float(w[0]),
        mu_lo=float(mu[0]),
        var_lo=float(var[0]),
        w_hi=float(w[1]),
        mu_hi=float(mu[1]),
        var_hi=float(var[1]),
    )


def _params(fit: GmmFit1D | None) -> tuple[float, ...] | None:
    return None if fit is None else (fit.w_lo, fit.mu_lo, fit.var_lo, fit.w_hi, fit.mu_hi, fit.var_hi)


class TestAnchoredEmEquivalence:
    """#3558: the optimised anchored EM must be **bit-for-bit** the old one.

    The shipped loop was rewritten to run in preallocated 1-D buffers because
    it is the dominant term in a calibration fold (86% of one, per the
    2026-08-28 fold-count study), and the whole value of that rewrite depends
    on it being an *optimisation*: a fit that moved by even one ulp would move
    the threshold through the fold-quantile transfer, and that is a
    calibration change requiring a study rather than an engineering task.

    So these compare all six fitted parameters with ``==``, never
    ``pytest.approx``.  A failure here is not a tolerance to widen.
    """

    @staticmethod
    def _fit_pair(x, a_scores, a_labels, anchor_weight, max_iter=200, tol=1e-8):
        arr = gmm_fit_array(x)
        init = fit_score_gmm(arr)
        assert init is not None
        a = np.asarray(a_scores, dtype=np.float64)
        z = np.asarray(a_labels, dtype=np.float64)
        a_hi, a_lo = a[z == 1.0], a[z != 1.0]
        return (
            _reference_anchored_em(arr, a_lo, a_hi, init, anchor_weight, max_iter, tol),
            _anchored_em(arr, a_lo, a_hi, init, anchor_weight, max_iter, tol),
        )

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 7])
    @pytest.mark.parametrize("n_votes", [2, 5, 20, 120])
    @pytest.mark.parametrize("anchor_weight", [0.01, FOLD_ANCHOR_WEIGHT, 1.0, 100.0])
    def test_bit_identical_on_bimodal_haystacks(self, seed, n_votes, anchor_weight):
        rng = np.random.default_rng(seed)
        arr = _bimodal(rng, n=3000)
        scores = np.clip(rng.normal(0.6, 0.15, n_votes), 0.0, 1.0)
        labels = (np.arange(n_votes) % 2 == 0).astype(float)
        reference, optimised = self._fit_pair(arr, scores, labels, anchor_weight)
        assert _params(optimised) == _params(reference)

    def test_bit_identical_with_one_sided_anchors(self):
        rng = np.random.default_rng(19)
        arr = _bimodal(rng)
        reference, optimised = self._fit_pair(arr, rng.normal(0.8, 0.05, 12), np.ones(12), FOLD_ANCHOR_WEIGHT)
        assert _params(optimised) == _params(reference)

    def test_bit_identical_when_anchors_contradict_the_modes(self):
        rng = np.random.default_rng(23)
        arr = _bimodal(rng)
        scores = np.concatenate([rng.normal(0.8, 0.02, 8), rng.normal(0.2, 0.02, 8)])
        labels = np.concatenate([np.zeros(8), np.ones(8)])
        reference, optimised = self._fit_pair(arr, scores, labels, FOLD_ANCHOR_WEIGHT)
        assert _params(optimised) == _params(reference)

    def test_bit_identical_on_a_degenerate_haystack(self):
        # Every free score identical: the variance floor is what keeps this
        # finite, and it has to bind at the same iteration in both forms.
        arr = np.full(500, 0.5)
        reference, optimised = self._fit_pair(arr, [0.4, 0.41, 0.6], [0.0, 0.0, 1.0], FOLD_ANCHOR_WEIGHT)
        assert _params(optimised) == _params(reference)

    def test_bit_identical_at_an_extreme_score_scale(self):
        # Scores far from the unit interval: the shifted/squared terms are
        # where a reassociated expression would first show up.
        rng = np.random.default_rng(29)
        arr = rng.standard_normal(4000) * 1e6
        reference, optimised = self._fit_pair(arr, [-1.0e6, 1.0e6], [0.0, 1.0], FOLD_ANCHOR_WEIGHT)
        assert _params(optimised) == _params(reference)

    @pytest.mark.parametrize("max_iter", [1, 2, 7])
    def test_bit_identical_mid_run_so_the_iteration_path_matches(self, max_iter):
        # Stopping short compares the *trajectory*, not just the fixed point:
        # two loops can converge to the same answer by different routes, and
        # only the truncated runs can tell them apart.
        rng = np.random.default_rng(31)
        arr = _bimodal(rng, n=1500)
        reference, optimised = self._fit_pair(
            arr, rng.normal(0.75, 0.05, 10), np.ones(10), FOLD_ANCHOR_WEIGHT, max_iter=max_iter
        )
        assert _params(optimised) == _params(reference)

    def test_bit_identical_through_the_shipped_fold_threshold(self):
        # The end-to-end check: the equality that matters is the threshold the
        # app ships, not the mixture parameters on their own.
        rng = np.random.default_rng(37)
        fold_haystacks = [_bimodal(rng, n=2500) for _ in range(3)]
        orderings = [
            (list(np.concatenate([rng.normal(0.2, 0.05, 6), rng.normal(0.8, 0.05, 6)])), [0.0] * 6 + [1.0] * 6)
            for _ in range(3)
        ]
        final = _bimodal(rng, n=2500)
        thresholds = [fold_anchored_gmm_threshold(fold_haystacks, orderings, final, k) for k in (0, 2, -2)]
        # Deterministic across repeat calls, and the provenance says the
        # anchored path (not a fallback) is what produced them.
        again = [fold_anchored_gmm_threshold(fold_haystacks, orderings, final, k) for k in (0, 2, -2)]
        assert thresholds == again
        assert all(prov.startswith("fold_anchored[3/3]") for _thr, prov in thresholds)


class TestCutFromFit:
    _FIT = GmmFit1D(w_lo=0.7, mu_lo=0.2, var_lo=0.01, w_hi=0.3, mu_hi=0.8, var_hi=0.01)

    def test_mid(self):
        cut, kind = gmm_cut_from_fit(self._FIT, "mid")
        assert cut == self._FIT.midpoint()
        assert kind == CUT_KIND_INTERIOR

    def test_rate_root_between_means(self):
        cut, kind = gmm_cut_from_fit(self._FIT, "rate", 1.0, 1.0)
        assert kind == CUT_KIND_INTERIOR
        assert self._FIT.mu_lo < cut < self._FIT.mu_hi
        # Equal variances at equal cost weights: the prior-free crossing IS the
        # midpoint (see GmmFit1D.rate_crossing).
        assert abs(cut - self._FIT.midpoint()) < 1e-9

    def test_rate_without_root_continues_past_the_costlier_edge(self):
        # The rate rule divides the mixture weights back out, so only extreme
        # COST weights can push the root outside (mu_lo, mu_hi): with a huge
        # FPR weight and wide components the Bad component still out-densities
        # the Good one all the way to ``mu_hi``.  Returning the bare edge there
        # made the cut constant in the cost ratio, which flattened the
        # ``mid_tilt`` quantile over whole bands of the Inclusion slider and
        # silently collapsed the acquisition offset to a no-op (issue #2896).
        # The cut now continues past the edge at the rule's first-order slope
        # - for an equal-variance fit that is *exactly* the interior crossing
        # line ``mid + var*ln(fpr/fnr)/d`` extended, checked in closed form.
        fit = GmmFit1D(w_lo=0.7, mu_lo=0.2, var_lo=0.04, w_hi=0.3, mu_hi=0.8, var_hi=0.04)

        def line(wf: float, wn: float) -> float:
            return fit.midpoint() + 0.04 * math.log(wf / wn) / 0.6

        cut, kind = gmm_cut_from_fit(fit, "rate", 1e6, 1.0)
        assert kind == CUT_KIND_CONTINUED
        assert cut > fit.mu_hi
        assert cut == pytest.approx(line(1e6, 1.0), rel=1e-12)

        cut, kind = gmm_cut_from_fit(fit, "rate", 1.0, 1e6)
        assert kind == CUT_KIND_CONTINUED
        assert cut < fit.mu_lo
        assert cut == pytest.approx(line(1.0, 1e6), rel=1e-12)

        # Strictly monotone *within* the saturated regime - the property the
        # bare-edge return lacked and the whole point of the continuation.
        deeper, _ = gmm_cut_from_fit(fit, "rate", 1e8, 1.0)
        assert deeper > gmm_cut_from_fit(fit, "rate", 1e6, 1.0)[0]
        assert gmm_cut_from_fit(fit, "rate", 1.0, 1e8)[0] < gmm_cut_from_fit(fit, "rate", 1.0, 1e6)[0]

    def test_rate_is_monotone_non_increasing_across_the_inclusion_knob(self):
        """The knob's nesting contract, at the level of a single fit.

        Swept over a wide spread of fit geometries - including the wide-Good
        shapes where the interior root enters and leaves the interval
        non-monotonically, which a midpoint fallback got wrong - so the clamp,
        the two-root case, and the ordinary crossing are all exercised.
        """
        rng = np.random.default_rng(2860)
        for _ in range(2000):
            w_lo = float(rng.uniform(0.05, 0.95))
            mu_lo = float(rng.uniform(0.0, 0.5))
            fit = GmmFit1D(
                w_lo=w_lo,
                mu_lo=mu_lo,
                var_lo=float(rng.uniform(1e-5, 0.05)),
                w_hi=1.0 - w_lo,
                mu_hi=mu_lo + float(rng.uniform(0.01, 0.5)),
                var_hi=float(rng.uniform(1e-5, 0.05)),
            )
            cuts = [gmm_cut_from_fit(fit, "rate", *inclusion_cost_weights(k))[0] for k in range(-10, 11)]
            assert all(b <= a + 1e-9 for a, b in zip(cuts, cuts[1:], strict=False)), (fit, cuts)

    def test_rate_agrees_with_the_stationary_point_where_one_exists(self):
        """The clamp only decides the degenerate cases; elsewhere the shipped
        cut is exactly ``rate_crossing`` - the rule #2836/#2852 measured."""
        rng = np.random.default_rng(2861)
        seen = 0
        for _ in range(500):
            w_lo = float(rng.uniform(0.2, 0.8))
            mu_lo = float(rng.uniform(0.0, 0.4))
            fit = GmmFit1D(
                w_lo=w_lo,
                mu_lo=mu_lo,
                var_lo=float(rng.uniform(1e-3, 0.02)),
                w_hi=1.0 - w_lo,
                mu_hi=mu_lo + float(rng.uniform(0.2, 0.5)),
                var_hi=float(rng.uniform(1e-3, 0.02)),
            )
            for k in (-3, 0, 3):
                wf, wn = inclusion_cost_weights(k)
                root = fit.rate_crossing(wf, wn)
                cut, kind = gmm_cut_from_fit(fit, "rate", wf, wn)
                if root is None:
                    continue
                seen += 1
                assert kind == CUT_KIND_INTERIOR
                assert cut == pytest.approx(root, abs=1e-12)
        assert seen > 100, "the sweep never produced an interior crossing"

    def test_a_degenerate_fit_reports_a_midpoint_not_a_continuation(self):
        """Both branches set the fallback flag; only one still answers the knob.

        A continued cut keeps moving with the cost tilt, so it is still the rate
        rule; a degenerate fit's midpoint is *constant* in the tilt and is not.
        Reporting both as one flag is what made a fallback unattributable, so
        the kind has to tell them apart (issue #2900).
        """
        no_gap = GmmFit1D(w_lo=0.7, mu_lo=0.5, var_lo=0.01, w_hi=0.3, mu_hi=0.5, var_hi=0.01)
        cut, kind = gmm_cut_from_fit(no_gap, "rate", 1.0, 1.0)
        assert kind == CUT_KIND_DEGENERATE_MIDPOINT
        assert cut == no_gap.midpoint()
        assert gmm_cut_from_fit(no_gap, "rate", 1e6, 1.0)[0] == cut, "a degenerate cut must not move with the tilt"

        empty_hi = GmmFit1D(w_lo=1.0, mu_lo=0.2, var_lo=0.01, w_hi=0.0, mu_hi=0.8, var_hi=0.01)
        assert gmm_cut_from_fit(empty_hi, "rate", 1.0, 1.0)[1] == CUT_KIND_DEGENERATE_MIDPOINT

    def test_the_kind_is_exactly_the_no_interior_stationary_point_flag(self):
        """``bool(kind)`` has to keep meaning what ``cut_fallback`` meant, or the
        fallback *rates* in every existing analysis silently change meaning."""
        rng = np.random.default_rng(2900)
        seen = {CUT_KIND_INTERIOR: 0, CUT_KIND_CONTINUED: 0}
        for _ in range(500):
            w_lo = float(rng.uniform(0.05, 0.95))
            mu_lo = float(rng.uniform(0.0, 0.4))
            fit = GmmFit1D(
                w_lo=w_lo,
                mu_lo=mu_lo,
                var_lo=float(rng.uniform(1e-4, 0.05)),
                w_hi=1.0 - w_lo,
                mu_hi=mu_lo + float(rng.uniform(0.05, 0.5)),
                var_hi=float(rng.uniform(1e-4, 0.05)),
            )
            for k in (-6, 0, 6):
                cut, kind = gmm_cut_from_fit(fit, "rate", *inclusion_cost_weights(k))
                assert kind in (CUT_KIND_INTERIOR, CUT_KIND_CONTINUED, CUT_KIND_DEGENERATE_MIDPOINT)
                # An interior kind is exactly a genuine root; anything else is not.
                assert (kind == CUT_KIND_INTERIOR) == (fit.rate_crossing(*inclusion_cost_weights(k)) is not None)
                if kind == CUT_KIND_CONTINUED:
                    assert not (fit.mu_lo < cut < fit.mu_hi)
                seen[kind] = seen.get(kind, 0) + 1
        assert seen[CUT_KIND_INTERIOR] > 50 and seen[CUT_KIND_CONTINUED] > 50, seen

    def test_unknown_rule_raises(self):
        with pytest.raises(ValueError, match="unknown cut rule"):
            gmm_cut_from_fit(self._FIT, "bogus")


class TestRankTransfer:
    def test_identity_when_scales_match(self):
        rng = np.random.default_rng(21)
        scores = rng.normal(0.5, 0.1, 5000)
        assert abs(rank_transfer(0.5, scores, scores) - 0.5) < 0.005

    def test_recovers_affine_scale_change(self):
        rng = np.random.default_rng(22)
        src = rng.normal(0.5, 0.1, 5000)
        tgt = 2.0 * src + 1.0
        assert abs(rank_transfer(0.55, src, tgt) - (2.0 * 0.55 + 1.0)) < 0.01

    def test_empty_inputs_pass_through(self):
        assert rank_transfer(0.42, [], [1.0, 2.0]) == 0.42
        assert rank_transfer(0.42, [1.0, 2.0], []) == 0.42


class TestFoldAnchored:
    def test_transfers_fold_cuts_across_scales(self):
        # Fold 2 sees the same haystack squashed to half scale; the combined
        # quantile must land the cut between the FINAL distribution's modes.
        rng = np.random.default_rng(31)
        final = _bimodal(rng)
        fold1 = _bimodal(np.random.default_rng(32))
        fold2 = 0.5 * _bimodal(np.random.default_rng(33))
        a1, l1 = _anchors_from_modes(np.random.default_rng(34))
        a2, l2 = _anchors_from_modes(np.random.default_rng(35))
        threshold, provenance = fold_anchored_gmm_threshold(
            [fold1, fold2],
            [(list(a1), list(l1)), (list(0.5 * a2), list(l2))],
            final,
        )
        assert provenance == "fold_anchored[2/2]"
        assert 0.3 < threshold < 0.7

    def test_deterministic(self):
        rng = np.random.default_rng(41)
        final = _bimodal(rng)
        fold = _bimodal(np.random.default_rng(42))
        a, lbl = _anchors_from_modes(np.random.default_rng(43))
        args = ([fold], [(list(a), list(lbl))], final)
        assert fold_anchored_gmm_threshold(*args) == fold_anchored_gmm_threshold(*args)

    def test_qmedian_combine(self):
        rng = np.random.default_rng(51)
        final = _bimodal(rng)
        folds = [_bimodal(np.random.default_rng(52 + i)) for i in range(3)]
        anchors = [_anchors_from_modes(np.random.default_rng(60 + i)) for i in range(3)]
        threshold, provenance = fold_anchored_gmm_threshold(
            folds, [(list(a), list(lbl)) for a, lbl in anchors], final, combine="qmedian"
        )
        assert provenance == "fold_anchored[3/3]"
        assert 0.3 < threshold < 0.7

    def test_degenerate_fold_falls_back_to_unanchored_fold_fit(self):
        # One fold's anchors are inverted -> that fold falls back to its own
        # unanchored fit but still contributes a quantile.
        rng = np.random.default_rng(71)
        final = _bimodal(rng)
        fold = _bimodal(np.random.default_rng(72))
        a, lbl = _anchors_from_modes(np.random.default_rng(73))
        threshold, provenance = fold_anchored_gmm_threshold(
            [fold], [(list(a), list(1.0 - lbl))], final, anchor_weight=1000.0
        )
        assert provenance == "fold_anchored[0/1]"
        assert np.isfinite(threshold)

    def test_all_folds_unusable_falls_back_to_final_unanchored(self):
        rng = np.random.default_rng(81)
        final = _bimodal(rng)
        threshold, provenance = fold_anchored_gmm_threshold([np.array([0.5])], [([0.5], [1.0])], final)
        assert provenance == "fold_fallback_final_unanchored"
        plain = fit_score_gmm(gmm_fit_array(final))
        assert plain is not None
        assert threshold == plain.midpoint()

    def test_unknown_combine_raises(self):
        rng = np.random.default_rng(91)
        final = _bimodal(rng)
        fold = _bimodal(np.random.default_rng(92))
        a, lbl = _anchors_from_modes(np.random.default_rng(93))
        with pytest.raises(ValueError, match="unknown fold combine"):
            fold_anchored_gmm_threshold([fold], [(list(a), list(lbl))], final, combine="bogus")


class TestShippedFoldAnchoredDefaults:
    """The production settings the 2026-08-06 anchor-mass sweep picked.

    ``docs/experiments/2026-08-05-population-anchored-calibration/REPORT.md`` recommends
    ``fold_anchored κ=0.3, mid cut``: the first run's grid bottomed out at κ=1
    so its winner sat on the edge, and extending the grid two decades down
    across six environments moved both the mass *and* the rule.  The shipped
    rule is ``mid_tilt`` - the measured midpoint anchored at inclusion 0, with
    the rate rule's displacement as the Inclusion tilt (issue #2865) - so at
    inclusion 0 it *is* the recommended arm, bit for bit.  These pin the
    constants and the call defaults together so a change to either is
    deliberate rather than a drift between the report and the code.
    """

    def test_constants(self):
        assert FOLD_ANCHOR_WEIGHT == 0.3
        assert FOLD_ANCHOR_CUT_RULE == "mid_tilt"
        assert FOLD_ANCHOR_COMBINE == "qmean"

    def test_the_bare_call_uses_them(self):
        rng = np.random.default_rng(101)
        final = _bimodal(rng)
        fold = _bimodal(np.random.default_rng(102))
        a, lbl = _anchors_from_modes(np.random.default_rng(103))
        orderings = [(list(a), list(lbl))]
        default, _p = fold_anchored_gmm_threshold([fold], orderings, final)
        explicit, _p2 = fold_anchored_gmm_threshold(
            [fold],
            orderings,
            final,
            0,
            anchor_weight=FOLD_ANCHOR_WEIGHT,
            cut_rule=FOLD_ANCHOR_CUT_RULE,
            combine=FOLD_ANCHOR_COMBINE,
        )
        assert default == explicit


class TestFoldAnchoredInclusion:
    """How the estimator answers the Inclusion knob: monotonically under the
    shipped ``mid_tilt`` rule (anchored bit-for-bit to the measured midpoint
    at inclusion 0, issue #2865) and under ``rate``; not at all under plain
    ``mid``."""

    @staticmethod
    def _cut(sd=0.12, mu_lo=0.4, mu_hi=0.6, cut_rule=FOLD_ANCHOR_CUT_RULE):
        """A fit over *overlapping* modes - the regime a real haystack is in."""

        def hay(seed):
            r = np.random.default_rng(seed)
            return np.concatenate([r.normal(mu_lo, sd, 1400), r.normal(mu_hi, sd, 600)])

        def anchors(seed):
            r = np.random.default_rng(seed)
            return (
                list(np.concatenate([r.normal(mu_lo, sd, 10), r.normal(mu_hi, sd, 10)])),
                list(np.concatenate([np.zeros(10), np.ones(10)])),
            )

        cut = fit_fold_anchored_cut([hay(202), hay(203)], [anchors(210), anchors(211)], hay(201), cut_rule=cut_rule)
        assert cut is not None
        return cut

    def test_the_shipped_rule_reproduces_mid_exactly_at_inclusion_zero(self):
        """#2865's anchoring contract: at inclusion 0 the tilt term is
        identically zero (both rate quantiles are the same computation on the
        same fits), so the shipped ``mid_tilt`` rule is bit-for-bit the plain
        ``mid`` arm - the one operating point either calibration run actually
        scored."""
        assert self._cut().threshold_at(0) == self._cut(cut_rule="mid").threshold_at(0)

    def test_the_shipped_rule_answers_the_knob_and_stays_nested(self):
        """Raising inclusion never raises the cut, and the knob actually moves
        the line across its range - the assertion the inclusion-blind ``mid``
        rule could not pass (issue #2865)."""
        cuts = [self._cut().threshold_at(k) for k in range(-10, 11)]
        assert all(b <= a + 1e-12 for a, b in zip(cuts, cuts[1:], strict=False)), cuts
        assert cuts[0] > cuts[-1], "the knob must actually move the line"

    def test_the_shipped_rule_has_no_interior_plateau(self):
        """Issue #2896: the tilt must not saturate short of the haystack's support.

        Under the old edge-clamped rate cut, once the density crossing ran off
        the inter-mean interval the per-fold cut stopped moving, so the
        composed ``mid_tilt`` quantile was *constant* over whole bands of the
        knob - and the acquisition offset, mapping ``k`` and ``k - 3`` into the
        same band, silently collapsed to a no-op.  The estimator here is built
        by hand rather than fitted so the geometry is exact and independent of
        process ordering: an equal-variance fit whose crossing exits the
        interval at ``|k| ~ 6``, over a haystack wide enough that no cut in
        range reaches its edges.  Every step must strictly move the threshold.
        """
        fit = GmmFit1D(w_lo=0.7, mu_lo=0.3, var_lo=0.02, w_hi=0.3, mu_hi=0.7, var_hi=0.02)
        hay = np.linspace(-0.2, 1.6, 1801)
        cut = FoldAnchoredCut(fits=(fit,), fold_haystacks=(hay,), final_haystack=hay, n_anchored=1)
        thr = [cut.threshold_at(k) for k in range(-13, 11)]
        assert all(b < a for a, b in zip(thr, thr[1:], strict=False)), thr

    def test_plain_mid_is_inclusion_blind(self):
        """The property that motivated #2865, pinned on the rule that has it:
        the bare midpoint ignores the cost weights, so a ``mid``-rule estimator
        is constant across the whole knob.  ``mid`` stays in the tree as a
        harness arm and as ``mid_tilt``'s inclusion-0 anchor."""
        cuts = [self._cut(cut_rule="mid").threshold_at(k) for k in range(-10, 11)]
        assert len(set(cuts)) == 1, cuts

    def test_rate_thresholds_are_nested_across_the_whole_knob(self):
        """The ``rate`` rule still answers the knob, and answers it monotonically.

        Not the shipped rule on its own any more, but it is the donor of
        ``mid_tilt``'s tilt (and a harness arm), so its contract is what the
        shipped rule's monotonicity is inherited from: raising inclusion can
        only lower the cut.
        """
        cuts = [self._cut(cut_rule="rate").threshold_at(k) for k in range(-10, 11)]
        assert all(b <= a + 1e-12 for a, b in zip(cuts, cuts[1:], strict=False)), cuts
        assert cuts[0] > cuts[-1], "the knob must actually move the line"

    def test_a_cleanly_separated_haystack_leaves_the_knob_little_room(self):
        """Known property, pinned so it is a decision rather than a surprise.

        The cut is carried to the final model as a *quantile*, and inside an
        empty band between two well-separated modes every cut has the same
        empirical quantile - so inclusion moves the cut but not the set of
        media it admits.  That is the same "band the calibration data cannot
        resolve" the conformal rule names; it shrinks as soon as the modes
        overlap (the test above), which is the realistic case.

        Scored against ``rate``, whose tilt is exactly the shift the shipped
        ``mid_tilt`` rule applies - so a knob range this small under ``rate``
        bounds the shipped rule's realized range on the same haystack too.
        """
        rng = np.random.default_rng(401)
        final = _bimodal(rng)
        folds = [_bimodal(np.random.default_rng(402 + i)) for i in range(2)]
        anchors = [_anchors_from_modes(np.random.default_rng(410 + i)) for i in range(2)]
        cut = fit_fold_anchored_cut(folds, [(list(a), list(lbl)) for a, lbl in anchors], final, cut_rule="rate")
        assert cut is not None
        cuts = [cut.threshold_at(k) for k in range(-10, 11)]
        assert all(b <= a + 1e-12 for a, b in zip(cuts, cuts[1:], strict=False))
        assert max(cuts) - min(cuts) < 0.05

    def test_recut_matches_a_fresh_one_shot_call(self):
        """Re-cutting a cached fit is the same answer as fitting from scratch.

        This is what lets an Inclusion slide skip the EM and the scoring passes
        without diverging from a retrain at that inclusion.
        """
        rng = np.random.default_rng(301)
        final = _bimodal(rng)
        folds = [_bimodal(np.random.default_rng(302 + i)) for i in range(2)]
        anchors = [_anchors_from_modes(np.random.default_rng(310 + i)) for i in range(2)]
        orderings = [(list(a), list(lbl)) for a, lbl in anchors]
        cut = fit_fold_anchored_cut(folds, orderings, final)
        assert cut is not None
        for k in (-7, -1, 0, 3, 9):
            one_shot, _p = fold_anchored_gmm_threshold(folds, orderings, final, k)
            assert cut.threshold_at(k) == one_shot

    def test_inclusion_zero_weights_the_two_error_rates_equally(self):
        """Inclusion 0 is the neutral point of the knob - which is also the
        only inclusion either calibration run scored, hence #2865."""
        assert inclusion_cost_weights(0) == (1.0, 1.0)
