"""SVM trainer prototype for the learned-sort comparison study.

This module is intentionally standalone - it does NOT wire into
``DetectorContext`` or the ``train_and_score`` production path.  Its job
is to expose a trainer with the same input/output contract as the MLP
(features in, calibrated [0, 1] probabilities out) so the label-curve
sweep in :mod:`vtscore.eval.label_curve` can compare them head-to-head.

If the sweep shows the SVM is competitive, the next step is to add a
trainer-selection field on detectors and route through this module from
``train_and_score``.  Until then, treat everything here as experimental.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


SVMKernel = Literal["linear", "rbf"]
CalibrationMode = Literal["sigmoid", "isotonic", "decision_sigmoid", "auto"]


@dataclass
class SVMClassifier:
    """Trained SVM wrapped with a probability source.

    ``calibrator`` is either a sklearn ``CalibratedClassifierCV`` (when CV
    calibration was feasible) or ``None`` (small-data fallback - we sigmoid
    the raw decision function instead).
    """

    base: Any
    """Underlying sklearn estimator (``LinearSVC`` or ``SVC``)."""

    calibrator: Any | None
    """``CalibratedClassifierCV`` fit on the same data, or ``None``."""

    scaler: Any | None
    """Optional ``StandardScaler`` applied before any predict call."""

    kernel: SVMKernel = "linear"
    calibration: CalibrationMode = "auto"

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(positive) for each row of *X* as a 1-D float array in [0, 1]."""
        X = np.asarray(X, dtype=np.float32)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        if self.calibrator is not None:
            return self.calibrator.predict_proba(X)[:, 1].astype(np.float32)
        # Fallback: sigmoid over decision function.  Not a true probability
        # but monotone in the SVM score, which is what the ranker cares
        # about; thresholding on 0.5 corresponds to decision_function >= 0.
        d = self.base.decision_function(X)
        d = np.clip(d, -30.0, 30.0)  # avoid overflow in exp
        return (1.0 / (1.0 + np.exp(-d))).astype(np.float32)


def _make_base_estimator(
    kernel: SVMKernel,
    C: float,
    seed: int,
    class_weight: str | dict[int, float] | None,
) -> Any:
    """Construct the underlying SVM (no calibration yet)."""
    if kernel == "linear":
        from sklearn.svm import LinearSVC  # noqa: PLC0415

        # ``dual='auto'`` lets sklearn pick the better solver based on
        # n_samples vs. n_features.  This is the sklearn-recommended default
        # on >=1.3 and avoids the FutureWarning about the previous default.
        return LinearSVC(
            C=C,
            class_weight=class_weight,
            dual="auto",
            max_iter=5000,
            random_state=seed,
        )
    if kernel == "rbf":
        from sklearn.svm import SVC  # noqa: PLC0415

        # ``probability=False`` here - we attach our own calibrator outside
        # so the calibration mode is uniform across kernels.
        return SVC(
            C=C,
            kernel="rbf",
            gamma="scale",
            class_weight=class_weight,
            probability=False,
            random_state=seed,
        )
    raise ValueError(f"Unsupported SVM kernel: {kernel!r}")


def _effective_calibration(
    requested: CalibrationMode,
    n_pos: int,
    n_neg: int,
) -> CalibrationMode:
    """Pick a calibration mode that's actually fittable for the label counts.

    ``CalibratedClassifierCV`` needs at least ``cv`` samples per class
    (default 5).  We drop to ``cv=2`` when possible and otherwise fall
    back to ``decision_sigmoid``, which works for any N >= 1 per class.

    ``isotonic`` additionally needs enough distinct decision-function
    values to be useful; below ~10 calibration points it overfits, so we
    downgrade to ``sigmoid`` (Platt) in that regime.
    """
    if requested != "auto":
        # Caller asked for a specific mode - try to honour it; fall back
        # only if the data literally can't support CV calibration.
        if requested in ("sigmoid", "isotonic") and min(n_pos, n_neg) < 2:
            return "decision_sigmoid"
        return requested

    smallest = min(n_pos, n_neg)
    if smallest < 2:
        return "decision_sigmoid"
    if smallest < 5:
        return "sigmoid"  # Platt is the right tool when calibration set is tiny
    if smallest < 10:
        return "sigmoid"
    return "isotonic"


def train_svm(
    X: np.ndarray,
    y: np.ndarray,
    *,
    kernel: SVMKernel = "linear",
    C: float = 1.0,
    calibration: CalibrationMode = "decision_sigmoid",
    inclusion_value: int = 0,
    seed: int = 42,
    standardize: bool = False,
) -> SVMClassifier:
    """Fit an SVM classifier and return a score-emitting wrapper.

    Mirrors the call shape of :func:`vtscore.training.mlp.train_model`
    (features, labels, inclusion bias, seed) and returns an object whose
    ``predict_proba`` produces a monotone score directly comparable to
    ``torch.sigmoid(mlp(X))`` for ranking and threshold-finding purposes.

    The default is ``"decision_sigmoid"``: a sigmoid wrapper over the raw
    ``decision_function``.  This is deliberate.  Production VTSearch
    treats the MLP's sigmoid output as an *uncalibrated score* and picks
    a threshold via cross-calibration (see
    :func:`vtscore.training.thresholds.calculate_cross_calibration_threshold`)
    rather than trusting it as a probability.  Wrapping the SVM in
    ``CalibratedClassifierCV`` would burn training data on k-fold CV
    (strictly less data per fold than fitting once on all of it) and
    invert the effect of ``inclusion_value`` by mapping decision scores
    back to the empirical base rate - both of which are undesirable
    when only the ranking and a learnable threshold matter.  Pass
    ``calibration="isotonic"`` or ``"sigmoid"`` (Platt) if you do
    explicitly want calibrated probabilities for some downstream task,
    or ``"auto"`` to let the data size pick (legacy behaviour).

    Args:
        X: ``(N, D)`` float array of training embeddings.
        y: ``(N,)`` array of 0/1 labels (1 = good, 0 = bad).
        kernel: ``"linear"`` (LinearSVC, fast) or ``"rbf"`` (SVC).
        C: Inverse regularisation strength.  Defaults to 1.0; bump up for
            noisy embeddings, down for very small N.
        calibration: How to map the SVM decision function to scores.
            Default ``"decision_sigmoid"`` (no CV calibration, just a
            sigmoid over ``decision_function``).  ``"auto"`` picks based
            on label counts; ``"sigmoid"`` forces Platt scaling;
            ``"isotonic"`` forces isotonic regression.  See
            :func:`_effective_calibration` for the auto rules and the
            fall-back behaviour when the data can't support CV
            calibration.
        inclusion_value: ``[-10, 10]`` bias toward including (positive) or
            excluding (negative) - translated to ``class_weight``.
        seed: Random seed for the SVM solver and the calibrator's CV splits.
        standardize: When ``True``, fit a ``StandardScaler`` first.  Every
            embedding is L2-normalised once at ingest (see
            :mod:`vtscore.embedding.normalize`), so features are already
            unit-norm and this is off by default; turn it on only if you're
            feeding raw or unnormalised features from an external source.

    Returns:
        A fitted :class:`SVMClassifier`.

    Raises:
        ValueError: when the training data has fewer than 2 samples or is
            single-class.  The MLP path silently returns ``None`` in that
            case; here we surface the error so the caller (the eval
            harness) can record it as a skip.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(np.int32).ravel()
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}")

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"train_svm needs both classes (got n_pos={n_pos}, n_neg={n_neg})")
    if n_pos + n_neg < 2:
        raise ValueError("train_svm needs at least 2 training samples")

    scaler = None
    if standardize:
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

        scaler = StandardScaler().fit(X)
        X_fit = scaler.transform(X)
    else:
        X_fit = X

    # Translate inclusion into a class weight.  Mirrors the MLP path:
    # the "balanced" baseline divides by class frequency, then we apply
    # an exponential multiplier in the requested direction.
    if inclusion_value == 0:
        class_weight: str | dict[int, float] = "balanced"
    elif inclusion_value > 0:
        base_pos = n_neg / max(n_pos, 1)
        class_weight = {0: 1.0, 1: float(base_pos * (2.0**inclusion_value))}
    else:
        base_neg = n_pos / max(n_neg, 1)
        class_weight = {0: float(base_neg * (2.0 ** (-inclusion_value))), 1: 1.0}

    base = _make_base_estimator(kernel, C, seed, class_weight)

    mode = _effective_calibration(calibration, n_pos, n_neg)

    calibrator: Any | None
    if mode == "decision_sigmoid":
        base.fit(X_fit, y)
        calibrator = None
    else:
        from sklearn.calibration import CalibratedClassifierCV  # noqa: PLC0415
        from sklearn.model_selection import StratifiedKFold  # noqa: PLC0415

        # cv must be <= min(n_pos, n_neg).  We already guaranteed >= 2 above.
        cv_n = min(5, min(n_pos, n_neg))
        cv_splitter = StratifiedKFold(n_splits=cv_n, shuffle=True, random_state=seed)
        calibrator = CalibratedClassifierCV(
            estimator=base,
            method=mode,  # "sigmoid" (Platt) or "isotonic"
            cv=cv_splitter,
        )
        calibrator.fit(X_fit, y)
        # ``calibrator`` holds its own fitted copy of the base estimator
        # via cross-validation; ``base`` itself is left unfit.  That's
        # fine because predict_proba only ever consults ``calibrator``
        # when one is present.

    return SVMClassifier(
        base=base,
        calibrator=calibrator,
        scaler=scaler,
        kernel=kernel,
        calibration=mode,
    )
