"""Tests for the label-anchored mixture estimators (issue #2852).

Covers the anchored EM itself (seeded synthetic bimodal / unimodal /
anchors-contradict-modes cases, determinism, degeneracy fallbacks), the
rank-transfer scale carrier, and the fold-anchored ("cross-LabeledGMM")
combiner.  Library tier: pure ``vtscore.training.thresholds``, no app imports.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.training.thresholds import (
    GmmFit1D,
    anchored_gmm_fit,
    fit_anchored_score_gmm,
    fit_score_gmm,
    fold_anchored_gmm_threshold,
    gmm_cut_from_fit,
    gmm_fit_array,
    rank_transfer,
)


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


class TestCutFromFit:
    _FIT = GmmFit1D(w_lo=0.7, mu_lo=0.2, var_lo=0.01, w_hi=0.3, mu_hi=0.8, var_hi=0.01)

    def test_mid(self):
        cut, fell_back = gmm_cut_from_fit(self._FIT, "mid")
        assert cut == self._FIT.midpoint()
        assert fell_back == 0

    def test_rate_root_between_means(self):
        cut, fell_back = gmm_cut_from_fit(self._FIT, "rate", 1.0, 1.0)
        assert fell_back == 0
        assert self._FIT.mu_lo < cut < self._FIT.mu_hi
        # Equal variances at equal cost weights: the prior-free crossing IS the
        # midpoint (see GmmFit1D.rate_crossing).
        assert abs(cut - self._FIT.midpoint()) < 1e-9

    def test_rate_without_root_falls_back_flagged(self):
        # The rate rule divides the mixture weights back out, so only extreme
        # COST weights can push the root outside (mu_lo, mu_hi): with a huge
        # FPR weight and wide components no cut between the means is
        # rate-optimal, and the rule must fall back to the midpoint and say so.
        fit = GmmFit1D(w_lo=0.7, mu_lo=0.2, var_lo=0.04, w_hi=0.3, mu_hi=0.8, var_hi=0.04)
        cut, fell_back = gmm_cut_from_fit(fit, "rate", 1e6, 1.0)
        assert fell_back == 1
        assert cut == fit.midpoint()

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
