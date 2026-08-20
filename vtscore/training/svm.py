"""SVM trainers: the **production detector head** plus the kernel sweep arms.

:func:`train_svm` is the general trainer used by the eval sweeps
(:mod:`vtscore.eval.label_curve`, :mod:`vtscore.eval.voting_iterations`) to
compare kernels and hyperparameters head-to-head against the MLP.

:func:`fit_linear_svm_head` is the one production cares about.  It fits the
**linear** SVM through the very same :func:`train_svm` call the eval harness
scores as ``svm_linear``, then lifts the resulting hyperplane into a torch
``Linear(D, 1)`` so the rest of the detector pipeline - calibration folds,
max-pooled region scoring, weight serialisation, threshold fusion - keeps
working on an ``nn.Sequential`` exactly as it did under the logistic head.
The two heads have the *same architecture*; they differ only in the objective
that fits it (hinge + L2 versus balanced BCE).  Reached through
:func:`vtscore.training.mlp.train_model` via the
:data:`~vtscore.training.mlp.LINEAR_SVM_HEAD` sentinel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


SVMKernel = Literal["linear", "rbf", "poly", "sigmoid"]
CalibrationMode = Literal["sigmoid", "isotonic", "decision_sigmoid", "auto"]
SVMBackend = Literal["auto", "sklearn", "cuml"]


@dataclass
class SVMClassifier:
    """Trained SVM wrapped with a probability source.

    ``calibrator`` is either a sklearn ``CalibratedClassifierCV`` (when CV
    calibration was feasible) or ``None`` (small-data fallback - we sigmoid
    the raw decision function instead).
    """

    base: Any
    """Underlying estimator (sklearn ``LinearSVC``/``SVC`` or the cuML equivalent)."""

    calibrator: Any | None
    """``CalibratedClassifierCV`` fit on the same data, or ``None``."""

    scaler: Any | None
    """Optional ``StandardScaler`` applied before any predict call."""

    kernel: SVMKernel = "linear"
    calibration: CalibrationMode = "auto"
    backend: str = "sklearn-cpu"
    """Which backend actually fit ``base`` — ``"sklearn-cpu"`` or ``"cuml"``.

    Recorded so the eval harness can label result rows with the backend that
    produced the timing/score, and shout loudly (rather than silently comparing
    CPU to GPU) when a requested cuML fit fell back to sklearn.
    """

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(positive) for each row of *X* as a 1-D float array in [0, 1]."""
        X = np.asarray(X, dtype=np.float32)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        if self.calibrator is not None:
            return np.asarray(self.calibrator.predict_proba(X))[:, 1].astype(np.float32)
        # Fallback: sigmoid over decision function.  Not a true probability
        # but monotone in the SVM score, which is what the ranker cares
        # about; thresholding on 0.5 corresponds to decision_function >= 0.
        # ``np.asarray`` coerces cuML's cupy/host output back to a numpy array.
        d = np.asarray(self.base.decision_function(X), dtype=np.float64).ravel()
        d = np.clip(d, -30.0, 30.0)  # avoid overflow in exp
        return (1.0 / (1.0 + np.exp(-d))).astype(np.float32)


def _resolve_gamma(gamma: str | float, gamma_mult: float, X_fit: np.ndarray) -> str | float:
    """Return the effective ``gamma`` passed to the kernel estimator.

    ``gamma_mult`` rescales sklearn's data-driven ``"scale"`` heuristic
    (``1 / (n_features * X.var())``) — e.g. ``gamma_mult=4`` gives 4×scale — so
    the kernel bandwidth can be swept relative to the default without the caller
    knowing the feature statistics.  With ``gamma_mult == 1`` the string
    ``"scale"`` / ``"auto"`` is passed through untouched, keeping the default
    ``svm_rbf`` byte-identical.
    """
    if gamma_mult == 1.0:
        return gamma
    if gamma in ("scale", "auto"):
        var = float(np.asarray(X_fit, dtype=np.float64).var())
        scale = 1.0 / (X_fit.shape[1] * var) if var > 0 else 1.0
        return float(scale * gamma_mult)
    return float(gamma) * gamma_mult


def _make_base_estimator(
    kernel: SVMKernel,
    C: float,
    seed: int,
    class_weight: str | dict[int, float] | None,
    *,
    gamma: str | float = "scale",
    degree: int = 3,
) -> Any:
    """Construct the underlying sklearn SVM (no calibration yet)."""
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
    if kernel in ("rbf", "poly", "sigmoid"):
        from sklearn.svm import SVC  # noqa: PLC0415

        # No built-in Platt scaling: we attach our own calibrator outside so
        # the calibration mode is uniform across kernels. sklearn>=1.9 removed
        # the ``probability`` argument (deprecated), and disabled is the
        # default, so there is nothing to pass.  ``degree`` is only consulted
        # for the polynomial kernel; sklearn ignores it otherwise.
        return SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,  # type: ignore[arg-type]  # sklearn accepts str|float; stub is str-only
            degree=degree,
            class_weight=class_weight,
            random_state=seed,
        )
    raise ValueError(f"Unsupported SVM kernel: {kernel!r}")


def _make_cuml_base(
    kernel: SVMKernel,
    C: float,
    class_weight: str | dict[int, float] | None,
    *,
    gamma: str | float = "scale",
    degree: int = 3,
) -> Any:
    """Construct a cuML (GPU) SVM estimator API-compatible with the sklearn one.

    Raises ``ImportError`` if cuML is missing (the caller catches it and falls
    back to sklearn).  cuML's ``LinearSVC``/``SVC`` expose ``decision_function``,
    which is all ``decision_sigmoid`` scoring needs; ``output_type="numpy"``
    keeps downstream code on plain numpy arrays.
    """
    if kernel == "linear":
        from cuml.svm import LinearSVC as CuLinearSVC  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

        return CuLinearSVC(C=C, class_weight=class_weight, output_type="numpy")
    from cuml.svm import SVC as CuSVC  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

    return CuSVC(
        C=C,
        kernel=kernel,
        gamma=gamma,  # type: ignore[arg-type]
        degree=degree,
        class_weight=class_weight,
        output_type="numpy",
    )


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


def _class_weight_for(
    inclusion_value: int,
    n_pos: int,
    n_neg: int,
    *,
    weighted: bool,
) -> str | dict[int, float] | None:
    """Translate *inclusion_value* into an sklearn ``class_weight``.

    Mirrors the MLP path: the ``"balanced"`` baseline divides by class
    frequency, then an exponential multiplier tilts it in the requested
    direction.  When the caller supplies explicit per-row weights (*weighted*)
    the result is ``None`` - again mirroring the MLP path, where per-row
    weights replace the frequency balance entirely.  Stacking a class balance
    on top of per-bag weights would count the balance twice.
    """
    if weighted:
        return None
    if inclusion_value == 0:
        return "balanced"
    if inclusion_value > 0:
        base_pos = n_neg / max(n_pos, 1)
        return {0: 1.0, 1: float(base_pos * (2.0**inclusion_value))}
    base_neg = n_pos / max(n_neg, 1)
    return {0: float(base_neg * (2.0 ** (-inclusion_value))), 1: 1.0}


def train_svm(
    X: np.ndarray,
    y: np.ndarray,
    *,
    kernel: SVMKernel = "linear",
    C: float = 1.0,
    gamma: str | float = "scale",
    gamma_mult: float = 1.0,
    degree: int = 3,
    calibration: CalibrationMode = "decision_sigmoid",
    inclusion_value: int = 0,
    seed: int = 42,
    standardize: bool = False,
    backend: SVMBackend = "auto",
    sample_weight: np.ndarray | None = None,
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
        kernel: ``"linear"`` (LinearSVC, fast), or ``"rbf"`` / ``"poly"`` /
            ``"sigmoid"`` (SVC).
        C: Inverse regularisation strength.  Defaults to 1.0; bump up for
            noisy embeddings, down for very small N.
        gamma: Kernel coefficient for ``rbf``/``poly``/``sigmoid`` — sklearn's
            ``"scale"`` (default), ``"auto"``, or a float.  Ignored for
            ``linear``.
        gamma_mult: Multiplier applied to the ``"scale"`` heuristic (e.g. ``4``
            → 4×scale, ``0.25`` → ¼×scale), letting the kernel bandwidth be
            swept relative to the default.  ``1.0`` (default) passes ``gamma``
            through unchanged.
        degree: Polynomial degree for the ``poly`` kernel (default 3); ignored
            otherwise.
        backend: ``"auto"`` (default; cuML on a usable GPU, else sklearn),
            ``"sklearn"`` (force CPU), or ``"cuml"`` (force GPU, raising if the
            cuML fit can't be honoured).  Only the ``decision_sigmoid`` path can
            use cuML; CV calibration is always sklearn.
        calibration: How to map the SVM decision function to scores.
            Default ``"decision_sigmoid"`` (no CV calibration, just a
            sigmoid over ``decision_function``).  ``"auto"`` picks based
            on label counts; ``"sigmoid"`` forces Platt scaling;
            ``"isotonic"`` forces isotonic regression.  See
            :func:`_effective_calibration` for the auto rules and the
            fall-back behaviour when the data can't support CV
            calibration.
        inclusion_value: ``[-10, 10]`` bias toward including (positive) or
            excluding (negative) - translated to ``class_weight``.  Ignored
            when *sample_weight* is supplied (the caller owns class balance).
        sample_weight: Optional per-row fit weights of shape ``(N,)``.  When
            given they replace the ``class_weight`` balance entirely, exactly
            as they do in :func:`vtscore.training.mlp.train_model` - this is
            how the region-flooding path expresses **per-bag** balancing, so a
            Bad image's many correlated region negatives count as one image.
            Only the ``decision_sigmoid`` path accepts them; CV calibration
            would have to re-weight each fold, so it rejects them instead of
            silently dropping the balance.
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
    X_fit: np.ndarray = X
    if standardize:
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

        scaler = StandardScaler().fit(X)
        X_fit = np.asarray(scaler.transform(X), dtype=np.float32)

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64).ravel()
        if sample_weight.shape[0] != y.shape[0]:
            raise ValueError(f"sample_weight has {sample_weight.shape[0]} rows but y has {y.shape[0]}")

    class_weight = _class_weight_for(inclusion_value, n_pos, n_neg, weighted=sample_weight is not None)

    gamma_eff = _resolve_gamma(gamma, gamma_mult, X_fit)
    mode = _effective_calibration(calibration, n_pos, n_neg)

    calibrator: Any | None
    used_backend = "sklearn-cpu"
    if mode == "decision_sigmoid":
        # Only the raw ``decision_function`` is needed here, which cuML's SVMs
        # provide — so this is the branch that can run on the GPU.  cuML is
        # tried when a usable GPU + cuML resolve; any hiccup (missing install,
        # unsupported param, fit-time kernel-compile error) degrades to sklearn
        # and flips the process-global kill switch, mirroring
        # :mod:`vtscore.gpu_backends`.
        base, used_backend = _fit_decision_sigmoid(
            kernel,
            C,
            seed,
            class_weight,
            X_fit,
            y,
            gamma=gamma_eff,
            degree=degree,
            backend=backend,
            sample_weight=sample_weight,
        )
        calibrator = None
    else:
        # CV calibration (Platt/isotonic) is sklearn-only — cuML has no
        # CalibratedClassifierCV — so this path always runs on the CPU.
        if sample_weight is not None:
            raise ValueError(
                f"sample_weight is only supported with calibration='decision_sigmoid' (got {mode!r}); "
                "CV calibration would re-fit on unweighted folds and silently drop the balance"
            )
        from sklearn.calibration import CalibratedClassifierCV  # noqa: PLC0415
        from sklearn.model_selection import StratifiedKFold  # noqa: PLC0415

        base = _make_base_estimator(kernel, C, seed, class_weight, gamma=gamma_eff, degree=degree)
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
        backend=used_backend,
    )


# Process-global kill switch: once a cuML SVM fit fails at runtime (typically a
# lazy nvrtc kernel-compile error from a mismatched CUDA toolchain), stop paying
# the failure on every subsequent fit and go straight to sklearn — mirroring the
# pattern in :mod:`vtscore.gpu_backends`.
_cuml_svm_failed = False


def _fit_decision_sigmoid(
    kernel: SVMKernel,
    C: float,
    seed: int,
    class_weight: str | dict[int, float] | None,
    X_fit: np.ndarray,
    y: np.ndarray,
    *,
    gamma: str | float,
    degree: int,
    backend: SVMBackend,
    sample_weight: np.ndarray | None = None,
) -> tuple[Any, str]:
    """Fit the base SVM for ``decision_sigmoid`` scoring, preferring cuML on a GPU.

    Returns ``(fitted_base, backend_label)`` where ``backend_label`` is
    ``"cuml"`` or ``"sklearn-cpu"``.  A cuML failure (import, construction, or
    the lazy fit-time kernel compile) degrades to sklearn and disables cuML for
    the rest of the process.

    *sample_weight*, when given, is forwarded to ``fit``.  cuML's SVMs do not
    take per-row weights, so a weighted fit stays on sklearn rather than
    silently dropping the weights on the GPU path.
    """
    global _cuml_svm_failed

    want_cuml = backend in ("auto", "cuml") and not _cuml_svm_failed and sample_weight is None
    if sample_weight is not None and backend == "cuml":
        raise ValueError("backend='cuml' cannot honour sample_weight; cuML's SVMs take no per-row weights")
    if want_cuml and backend == "auto":
        from vtscore.gpu_backends import cuml_enabled  # noqa: PLC0415

        want_cuml = cuml_enabled()

    if want_cuml:
        try:
            import logging  # noqa: PLC0415

            base = _make_cuml_base(kernel, C, class_weight, gamma=gamma, degree=degree)
            base.fit(X_fit, y)
            return base, "cuml"
        except Exception as exc:  # noqa: BLE001 — any cuML hiccup degrades to CPU
            _cuml_svm_failed = True
            logging.getLogger(__name__).warning(
                "cuML SVM fit failed (%s: %s); falling back to sklearn and disabling "
                "cuML SVMs for the rest of this process.",
                type(exc).__name__,
                exc,
            )
            if backend == "cuml":
                # An explicit cuML request that can't be honoured is worth
                # surfacing to the caller rather than silently swapping backends.
                raise

    base = _make_base_estimator(kernel, C, seed, class_weight, gamma=gamma, degree=degree)
    base.fit(X_fit, y, sample_weight=sample_weight)
    return base, "sklearn-cpu"


def fit_linear_svm_head(
    X: np.ndarray,
    y: np.ndarray,
    input_dim: int,
    *,
    seed: int = 42,
    sample_weight: np.ndarray | None = None,
) -> Any:
    """Fit the **production detector head**: a linear SVM, returned as ``Linear(D, 1)``.

    This is the head every production fit trains (see
    :data:`vtscore.training.mlp.LINEAR_SVM_HEAD`).  It is the *same*
    architecture as the older logistic head - one ``Linear(input_dim, 1)``, no
    hidden layer - so weight serialisation, ``build_model_from_weights``,
    max-pooled region scoring, and the calibration folds are all untouched.
    What changes is the objective that fits it: hinge loss with L2
    regularisation (a maximum-margin boundary decided by the examples nearest
    it) instead of balanced binary cross-entropy.

    The fit is delegated to :func:`train_svm` with ``kernel="linear"`` rather
    than re-implemented, so the shipped head *is* the ``svm_linear`` arm the
    eval harness scores - there is no second definition to drift.

    Scores stay comparable to the logistic head's because both end in
    ``sigmoid(w·x + b)``: production applies the sigmoid downstream
    (:func:`vtscore.utils.scores.sigmoid_to_finite_scores`), which reproduces
    :meth:`SVMClassifier.predict_proba`'s ``decision_sigmoid`` mode value for
    value.  Neither head's output is a calibrated probability - the decision
    point is the separately calibrated threshold, not 0.5.

    Args:
        X: ``(N, input_dim)`` float array of training embeddings.
        y: ``(N,)`` array of 0/1 labels (1 = good, 0 = bad).
        input_dim: Embedding dimensionality; must match ``X.shape[1]``.
        seed: Seed for the solver (liblinear is deterministic given it).
        sample_weight: Optional per-row fit weights.  ``None`` balances the
            classes by inverse frequency (``class_weight="balanced"``, the
            analogue of the MLP path's default); supplying weights hands class
            balance to the caller, which is how region flooding weights a Bad
            image's many region rows down to one image's worth.

    Returns:
        An ``nn.Sequential(Linear(input_dim, 1))`` in eval mode on the active
        torch device, whose forward pass **is** the SVM's decision function.

    Raises:
        ValueError: propagated from :func:`train_svm` when the labels are
            single-class or there are fewer than 2 samples.
    """
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    from vtscore.embedding.loader import ensure_torch_configured, get_torch_device  # noqa: PLC0415
    from vtscore.training.mlp import LINEAR_SVM_HEAD, build_model  # noqa: PLC0415

    from vtscore import config  # noqa: PLC0415

    clf = train_svm(
        X,
        y,
        kernel="linear",
        C=config.SVM_HEAD_C,
        calibration="decision_sigmoid",
        inclusion_value=0,
        seed=seed,
        standardize=False,
        # sklearn only: the fit is milliseconds at any vote count a user
        # reaches, liblinear is deterministic given the seed, and cuML's
        # LinearSVC takes neither a seed nor per-row weights - so a GPU fit
        # would buy nothing and cost reproducibility.
        backend="sklearn",
        sample_weight=sample_weight,
    )

    # Lift the hyperplane into the torch head.  ``coef_`` is ``(1, D)`` and
    # ``intercept_`` is ``(1,)`` for a binary LinearSVC, which is exactly the
    # shape of ``Linear(D, 1)``'s weight and bias, so ``model(x)`` returns the
    # decision function unchanged.
    coef = np.asarray(clf.base.coef_, dtype=np.float32).reshape(1, -1)
    intercept = np.asarray(clf.base.intercept_, dtype=np.float32).reshape(1)
    if coef.shape[1] != input_dim:
        raise ValueError(f"SVM fit produced {coef.shape[1]} coefficients but input_dim is {input_dim}")

    ensure_torch_configured()
    model = build_model(input_dim, hidden_dim=LINEAR_SVM_HEAD)
    layer: nn.Linear = model[0]  # type: ignore[assignment]  # LINEAR_SVM_HEAD builds exactly this
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(coef))
        layer.bias.copy_(torch.from_numpy(intercept))
    model.eval()
    return model.to(get_torch_device())
