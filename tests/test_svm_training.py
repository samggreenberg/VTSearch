"""Tests for the SVM trainer prototype (vtsearch.models.svm_training)."""

from __future__ import annotations

import numpy as np
import pytest

from vtsearch.models.svm_training import (
    SVMClassifier,
    _effective_calibration,
    train_svm,
)


def _separable(n_per_class: int = 30, dim: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Build two linearly separable Gaussian blobs.

    Positives sit near ``+1`` along the first axis, negatives near ``-1``.
    Returns ``(X, y)`` already shuffled so callers can pass straight to
    ``train_svm``.
    """
    rng = np.random.default_rng(seed)
    pos = rng.standard_normal((n_per_class, dim)).astype(np.float32) * 0.2
    pos[:, 0] += 1.0
    neg = rng.standard_normal((n_per_class, dim)).astype(np.float32) * 0.2
    neg[:, 0] -= 1.0
    X = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(n_per_class, dtype=np.int32), np.zeros(n_per_class, dtype=np.int32)])
    order = rng.permutation(X.shape[0])
    return X[order], y[order]


class TestEffectiveCalibration:
    def test_auto_picks_decision_sigmoid_when_one_class_lt_2(self):
        assert _effective_calibration("auto", n_pos=1, n_neg=10) == "decision_sigmoid"
        assert _effective_calibration("auto", n_pos=10, n_neg=1) == "decision_sigmoid"

    def test_auto_picks_sigmoid_for_small_balanced(self):
        # 2 <= min < 10 → Platt
        assert _effective_calibration("auto", n_pos=3, n_neg=3) == "sigmoid"
        assert _effective_calibration("auto", n_pos=9, n_neg=9) == "sigmoid"

    def test_auto_picks_isotonic_for_larger_balanced(self):
        assert _effective_calibration("auto", n_pos=20, n_neg=20) == "isotonic"

    def test_explicit_mode_downgrades_when_unfittable(self):
        # User asked for isotonic with only 1 positive — must downgrade.
        assert _effective_calibration("isotonic", n_pos=1, n_neg=10) == "decision_sigmoid"

    def test_explicit_mode_honoured_when_fittable(self):
        assert _effective_calibration("sigmoid", n_pos=3, n_neg=3) == "sigmoid"
        assert _effective_calibration("isotonic", n_pos=3, n_neg=3) == "isotonic"


class TestTrainSVMShapeContracts:
    def test_returns_svm_classifier(self):
        X, y = _separable(n_per_class=20)
        clf = train_svm(X, y)
        assert isinstance(clf, SVMClassifier)

    def test_predict_proba_returns_1d_floats_in_range(self):
        X, y = _separable(n_per_class=20)
        clf = train_svm(X, y)
        probs = clf.predict_proba(X)
        assert probs.shape == (X.shape[0],)
        assert probs.dtype.kind == "f"
        assert float(probs.min()) >= 0.0
        assert float(probs.max()) <= 1.0

    def test_rejects_single_class(self):
        X = np.random.default_rng(0).standard_normal((10, 4)).astype(np.float32)
        with pytest.raises(ValueError, match="both classes"):
            train_svm(X, np.ones(10, dtype=np.int32))

    def test_rejects_size_one(self):
        with pytest.raises(ValueError):
            train_svm(np.zeros((1, 4), dtype=np.float32), np.array([1], dtype=np.int32))

    def test_rejects_mismatched_xy(self):
        X = np.zeros((5, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="rows"):
            train_svm(X, np.array([0, 1, 0], dtype=np.int32))

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            train_svm(np.zeros(10, dtype=np.float32), np.zeros(10, dtype=np.int32))


class TestSeparableData:
    def test_linear_kernel_separates_blobs(self):
        X, y = _separable(n_per_class=40, seed=1)
        clf = train_svm(X, y, kernel="linear")
        probs = clf.predict_proba(X)
        # Trivially separable → near-perfect AUROC
        pos = probs[y == 1]
        neg = probs[y == 0]
        assert pos.mean() > neg.mean() + 0.5

    def test_rbf_kernel_separates_blobs(self):
        X, y = _separable(n_per_class=40, seed=2)
        clf = train_svm(X, y, kernel="rbf")
        probs = clf.predict_proba(X)
        pos = probs[y == 1]
        neg = probs[y == 0]
        assert pos.mean() > neg.mean() + 0.5

    def test_unknown_kernel_raises(self):
        X, y = _separable()
        with pytest.raises(ValueError, match="kernel"):
            train_svm(X, y, kernel="polynomial")  # type: ignore[arg-type]


class TestDeterminism:
    def test_same_seed_same_predictions(self):
        X, y = _separable(n_per_class=30, seed=7)
        a = train_svm(X, y, seed=42).predict_proba(X)
        b = train_svm(X, y, seed=42).predict_proba(X)
        np.testing.assert_allclose(a, b, atol=1e-6)


class TestDefaultCalibration:
    """The default is ``decision_sigmoid`` — no CV calibration."""

    def test_default_is_decision_sigmoid(self):
        X, y = _separable(n_per_class=20)
        clf = train_svm(X, y)
        assert clf.calibration == "decision_sigmoid"
        assert clf.calibrator is None

    def test_default_emits_monotone_scores_in_unit_interval(self):
        # The wrapper is still sigmoid-over-decision_function, so scores
        # stay in [0, 1] and downstream threshold-finders see a familiar
        # range.  But ordering is what actually matters.
        X, y = _separable(n_per_class=20)
        clf = train_svm(X, y)
        probs = clf.predict_proba(X)
        assert ((probs >= 0.0) & (probs <= 1.0)).all()

    def test_explicit_isotonic_uses_cv_calibration(self):
        X, y = _separable(n_per_class=30)
        clf = train_svm(X, y, calibration="isotonic")
        assert clf.calibration == "isotonic"
        assert clf.calibrator is not None

    def test_explicit_sigmoid_uses_cv_calibration(self):
        X, y = _separable(n_per_class=20)
        clf = train_svm(X, y, calibration="sigmoid")
        assert clf.calibration == "sigmoid"
        assert clf.calibrator is not None

    def test_auto_picks_isotonic_when_data_supports_it(self):
        X, y = _separable(n_per_class=20)
        clf = train_svm(X, y, calibration="auto")
        # 20 per class → smallest >= 10 → isotonic
        assert clf.calibration == "isotonic"
        assert clf.calibrator is not None

    def test_unfittable_cv_calibration_falls_back(self):
        # 1 pos + 5 neg can't fit even cv=2 calibrator, must fall back.
        rng = np.random.default_rng(0)
        X = rng.standard_normal((6, 4)).astype(np.float32)
        y = np.array([1, 0, 0, 0, 0, 0], dtype=np.int32)
        clf = train_svm(X, y, calibration="isotonic")
        assert clf.calibration == "decision_sigmoid"
        assert clf.calibrator is None
        probs = clf.predict_proba(X)
        assert probs.shape == (6,)
        assert ((probs >= 0.0) & (probs <= 1.0)).all()


def _overlapping(n_per_class: int = 40, dim: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Build two Gaussian blobs with substantial overlap.

    Class weights have to actually move the decision boundary for the
    inclusion tests to be observable, which means the trainer needs to
    face genuine ambiguity.  ``_separable`` is too easy — both inclusion
    values learn essentially the same boundary.
    """
    rng = np.random.default_rng(seed)
    pos = rng.standard_normal((n_per_class, dim)).astype(np.float32)
    pos[:, 0] += 0.6
    neg = rng.standard_normal((n_per_class, dim)).astype(np.float32)
    neg[:, 0] -= 0.6
    X = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(n_per_class, dtype=np.int32), np.zeros(n_per_class, dtype=np.int32)])
    order = rng.permutation(X.shape[0])
    return X[order], y[order]


class TestInclusionWeights:
    """``inclusion_value`` becomes a class-weight multiplier on the SVM.

    Note that with CV-based probability calibration (Platt / isotonic) the
    effect of class weights is largely cancelled out at the probability
    layer — the calibrator maps decision scores back to the true base
    rate of the training data.  Inclusion bias only shows up if the
    caller picks a non-default threshold (analogous to the MLP path
    where ``find_optimal_threshold`` consumes the same inclusion value).
    So here we verify the trainer accepts and applies the bias, without
    asserting a post-calibration probability shift that's intentionally
    flattened by the calibrator.
    """

    @staticmethod
    def _class_weight_for(inclusion_value: int) -> dict[int, float]:
        X, y = _overlapping(n_per_class=20, seed=11)
        clf = train_svm(X, y, inclusion_value=inclusion_value, calibration="decision_sigmoid")
        cw = clf.base.get_params().get("class_weight")
        if isinstance(cw, dict):
            return {int(k): float(v) for k, v in cw.items()}
        # ``"balanced"`` for inclusion_value == 0 — represent as ratio 1.0.
        return {0: 1.0, 1: 1.0}

    def test_positive_inclusion_upweights_positives(self):
        balanced = self._class_weight_for(0)
        biased = self._class_weight_for(5)
        # Positive class weight grows; negative stays at 1.
        ratio_balanced = balanced[1] / balanced[0]
        ratio_biased = biased[1] / biased[0]
        assert ratio_biased > ratio_balanced

    def test_negative_inclusion_upweights_negatives(self):
        balanced = self._class_weight_for(0)
        biased = self._class_weight_for(-5)
        ratio_balanced = balanced[1] / balanced[0]
        ratio_biased = biased[1] / biased[0]
        assert ratio_biased < ratio_balanced

    def test_decision_sigmoid_bias_visible_in_probabilities(self):
        """Without CV calibration, inclusion bias propagates through to scores.

        ``decision_sigmoid`` is the small-data fallback path; here the
        sigmoid is applied to the raw decision function so a shifted
        boundary directly shifts the output probabilities.
        """
        X, y = _overlapping(n_per_class=40, seed=12)
        balanced = train_svm(X, y, inclusion_value=0, calibration="decision_sigmoid")
        positive = train_svm(X, y, inclusion_value=5, calibration="decision_sigmoid")
        negative = train_svm(X, y, inclusion_value=-5, calibration="decision_sigmoid")
        # Positive bias raises mean P; negative bias lowers it.
        assert positive.predict_proba(X).mean() > balanced.predict_proba(X).mean()
        assert negative.predict_proba(X).mean() < balanced.predict_proba(X).mean()


class TestStandardize:
    def test_standardize_path_runs(self):
        X, y = _separable(n_per_class=20)
        # Scale wildly to force StandardScaler to do something.
        X = X * 100.0
        clf = train_svm(X, y, standardize=True)
        assert clf.scaler is not None
        probs = clf.predict_proba(X)
        assert probs.shape == (X.shape[0],)
        # Probabilities still well-behaved.
        assert ((probs >= 0.0) & (probs <= 1.0)).all()
