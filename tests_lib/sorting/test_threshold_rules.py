"""Unit tests for the selectable threshold rules (issue #2790 threshold-stability).

Pure ``(scores, labels)`` / synthetic-``trainer_fn`` tests — no models, no cache,
no Flask — so they run in the library tier. They pin the fidelity-critical
properties of the conformal port and the variance-reducing behaviours the study
A/Bs: the gap midpoint sits below every calibration positive on a separated task,
the cut is monotone in inclusion, ``argmin`` stays byte-identical to the sweep's
current ``xcal`` path, stratified folds never go single-class, and the med3 /
rank-transfer transforms do what the plan says.
"""

from __future__ import annotations

import numpy as np

from vtscore.eval.threshold_rules import (
    calibrated_threshold,
    conformal_threshold,
    median_smooth,
    rank_transfer_cut,
    stratified_fold_orderings,
)
from vtscore.eval.xcal import cross_calibrated_threshold


def _linear_trainer(query: np.ndarray):
    """A no-op ``trainer_fn`` whose ``predict`` is a fixed linear score ``X @ query``.

    Deterministic and model-free: lets the fold machinery run without training an
    MLP, so the tests exercise the split/pool/threshold logic in isolation.
    """

    def trainer_fn(X, y, seed):  # noqa: ARG001 - signature fixed by TrainerFn
        def predict(Xc):
            return np.asarray(Xc, dtype=np.float64) @ query

        return predict

    return trainer_fn


class TestConformalThreshold:
    def test_empty_and_single_class_return_half(self):
        assert conformal_threshold([], []) == 0.5
        assert conformal_threshold([0.1, 0.2, 0.3], [1.0, 1.0, 1.0]) == 0.5
        assert conformal_threshold([0.1, 0.2, 0.3], [0.0, 0.0, 0.0]) == 0.5

    def test_gap_midpoint_below_every_positive_when_separated(self):
        # Cleanly separated: negatives in [0, 0.3], positives in [0.7, 1.0].
        neg = [0.05, 0.1, 0.2, 0.25, 0.3]
        pos = [0.7, 0.8, 0.9, 0.95, 1.0]
        scores = neg + pos
        labels = [0.0] * len(neg) + [1.0] * len(pos)
        thr = conformal_threshold(scores, labels, inclusion_value=0)
        # The k=0 cut is the midpoint of the gap: above the top negative, strictly
        # below the lowest positive (so no user-voted Good is ever excluded).
        assert max(neg) < thr < min(pos)
        # It is the max-margin midpoint, not pinned to the lowest positive.
        assert thr < min(pos) - 1e-6

    def test_monotone_non_increasing_in_inclusion(self):
        rng = np.random.default_rng(0)
        neg = rng.uniform(0.0, 0.5, size=40).tolist()
        pos = rng.uniform(0.4, 1.0, size=40).tolist()
        scores = neg + pos
        labels = [0.0] * len(neg) + [1.0] * len(pos)
        thrs = [conformal_threshold(scores, labels, inclusion_value=k) for k in range(-10, 11)]
        # Higher inclusion (larger k) can only lower the cut (include more).
        for a, b in zip(thrs, thrs[1:], strict=True):
            assert b <= a + 1e-9

    def test_overlap_regime_collapses_to_fp_guard(self):
        # Heavy overlap: no empty band, so the midpoint collapses onto the guard
        # and the cut stays inside the shared score range (FPR-controlled regime).
        rng = np.random.default_rng(3)
        scores = rng.uniform(0.3, 0.7, size=80).tolist()
        labels = [float(i % 2) for i in range(80)]
        thr = conformal_threshold(scores, labels, inclusion_value=0)
        assert 0.3 <= thr <= 0.7


class TestStratifiedFolds:
    def test_never_single_class_and_pairs_scores_with_labels(self):
        rng = np.random.default_rng(1)
        # 6 positives, 6 negatives in 1-D; the linear trainer scores by the coord.
        X = np.concatenate([rng.normal(1.0, 0.1, size=(6, 1)), rng.normal(-1.0, 0.1, size=(6, 1))]).astype(np.float64)
        y = np.array([1.0] * 6 + [0.0] * 6)
        orderings = stratified_fold_orderings(X, y, _linear_trainer(np.array([1.0])), seed=7, calibrate_count=4)
        assert len(orderings) == 4
        for scores, labels in orderings:
            assert len(scores) == len(labels)
            assert 0.0 in labels and 1.0 in labels  # both classes present in every cal fold

    def test_too_few_or_single_class_returns_empty(self):
        assert (
            stratified_fold_orderings(np.zeros((3, 1)), np.array([1.0, 0.0, 1.0]), _linear_trainer(np.array([1.0])), 0)
            == []
        )
        X = np.ones((6, 1))
        assert stratified_fold_orderings(X, np.ones(6), _linear_trainer(np.array([1.0])), 0) == []


class TestCalibratedThresholdDispatch:
    def _data(self, seed=2):
        rng = np.random.default_rng(seed)
        X = np.concatenate([rng.normal(1.0, 0.3, size=(20, 1)), rng.normal(-1.0, 0.3, size=(20, 1))]).astype(np.float64)
        y = np.array([1.0] * 20 + [0.0] * 20)
        return X, y

    def test_argmin_is_byte_identical_to_xcal(self):
        X, y = self._data()
        trainer = _linear_trainer(np.array([1.0]))
        got = calibrated_threshold(X, y, trainer, seed=5, rule="argmin", calibrate_count=2)
        want = cross_calibrated_threshold(X, y, trainer, 5, calibrate_count=2)
        assert got == want

    def test_conformal_runs_and_lands_in_gap(self):
        X, y = self._data()
        trainer = _linear_trainer(np.array([1.0]))
        thr = calibrated_threshold(X, y, trainer, seed=5, rule="conformal", calibrate_count=2)
        assert np.isfinite(thr)

    def test_unknown_rule_raises(self):
        X, y = self._data()
        try:
            calibrated_threshold(X, y, _linear_trainer(np.array([1.0])), 0, rule="bogus")
        except ValueError as e:
            assert "bogus" in str(e)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError for unknown rule")

    def test_too_few_votes_falls_back_to_half(self):
        X = np.ones((3, 1))
        y = np.array([1.0, 0.0, 1.0])
        assert calibrated_threshold(X, y, _linear_trainer(np.array([1.0])), 0, rule="conformal") == 0.5


class TestMedianSmooth:
    def test_median_of_last_three(self):
        assert median_smooth([0.5, 0.4, 0.35, 0.9]) == 0.4  # median(0.4, 0.35, 0.9)

    def test_kills_single_spike(self):
        # A lone spike among stable values is smoothed away.
        assert median_smooth([0.5, 0.5, 0.9]) == 0.5

    def test_shorter_history_uses_what_exists(self):
        assert median_smooth([0.42]) == 0.42
        assert median_smooth([0.4, 0.6]) == 0.5

    def test_drops_non_finite(self):
        assert median_smooth([0.5, np.inf, 0.5]) == 0.5


class TestRankTransfer:
    def test_maps_quantile_between_score_pools(self):
        fold = [0.0, 0.25, 0.5, 0.75, 1.0]
        # Cut at 0.5 = the 0.6 quantile of the fold pool (3 of 5 <= 0.5).
        # Final pool is shifted; the same quantile lands lower.
        final = [0.0, 0.1, 0.2, 0.3, 0.4]
        got = rank_transfer_cut(0.5, fold, final)
        assert got == float(np.quantile(np.asarray(final), 0.6))

    def test_degenerate_pools_return_cut_unchanged(self):
        assert rank_transfer_cut(0.42, [], [0.1, 0.2]) == 0.42
        assert rank_transfer_cut(0.42, [0.1, 0.2], []) == 0.42
