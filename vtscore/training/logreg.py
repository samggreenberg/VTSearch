"""The converged logistic-regression head, an eval arm (issue #4114).

:func:`fit_linear_logreg_head` fits balanced, L2-regularised logistic
regression **to convergence** with scikit-learn, then lifts it into the same
torch ``Linear(D, 1)`` the production SVM head uses, exactly as
:func:`vtscore.training.svm.fit_linear_svm_head` lifts its hyperplane.  It is
reached through :func:`vtscore.training.mlp.train_model` via the
:data:`~vtscore.training.mlp.LINEAR_LOGREG_HEAD` sentinel, so it flows through
the calibration folds, region max-pooling and weight serialisation untouched.

Why a second logistic head exists: #3197 found that the older logistic head
(:data:`~vtscore.training.mlp.LINEAR_HEAD`) loses to the SVM not because of its
loss but because of its **fit**.  Adam at lr 1e-3, early-stopped at <= 200
epochs, leaves its weight vector at cosine ~0.75 to any point on the logistic
regularisation path.  Fitted to convergence at C = 1, the same loss ranks as
well as the SVM on the same votes.  This head is that converged fit, so the
Autopilot loop can measure it.  It is not reachable from the app.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

#: Inverse L2 strength of the converged logistic head.  C = 1 is the value
#: #3197's Stage A replay measured (``fit_lr_sklearn`` in
#: ``scripts/experiments/svm_vs_logistic/heads.py``), and the SVM head's own
#: shipped C, so the two heads regularise on the same nominal scale.
LOGREG_HEAD_C: float = 1.0

#: Solver budget.  lbfgs converges in well under this at any vote count a
#: session reaches; the budget and tolerance match #3197's ``fit_lr_sklearn``,
#: so the in-loop arm is the fit Stage A measured.
LOGREG_HEAD_MAX_ITER: int = 20000
LOGREG_HEAD_TOL: float = 1e-6


def fit_linear_logreg_head(
    X: np.ndarray,
    y: np.ndarray,
    input_dim: int,
    *,
    seed: int = 42,
    sample_weight: np.ndarray | None = None,
) -> Any:
    """Fit converged, balanced, L2 logistic regression, returned as ``Linear(D, 1)``.

    Args:
        X: ``(N, input_dim)`` float array of training embeddings.
        y: ``(N,)`` array of 0/1 labels (1 = good, 0 = bad).
        input_dim: Embedding dimensionality; must match ``X.shape[1]``.
        seed: Passed to the estimator's ``random_state``.  lbfgs is
            deterministic and ignores it; it is accepted so the call shape
            matches the other heads.
        sample_weight: Optional per-row fit weights.  ``None`` balances the
            classes by inverse frequency (``class_weight="balanced"``).
            Supplying weights hands class balance to the caller, as on the SVM
            head, which is how region flooding weights a Bad image's many
            region rows down to one image's worth.

    Returns:
        An ``nn.Sequential(Linear(input_dim, 1))`` in eval mode on the active
        torch device, whose forward pass is the fitted logit ``w·x + b``.
    """
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415
    from sklearn.exceptions import ConvergenceWarning  # noqa: PLC0415
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    from vtscore.embedding.loader import ensure_torch_configured, get_torch_device  # noqa: PLC0415
    from vtscore.training.mlp import LINEAR_LOGREG_HEAD, build_model  # noqa: PLC0415

    X64 = np.asarray(X, dtype=np.float64)
    y_int = np.asarray(y).reshape(-1).astype(int)
    if X64.shape[1] != input_dim:
        raise ValueError(f"X has {X64.shape[1]} columns but input_dim is {input_dim}")

    clf = LogisticRegression(
        C=LOGREG_HEAD_C,
        class_weight=None if sample_weight is not None else "balanced",
        max_iter=LOGREG_HEAD_MAX_ITER,
        tol=LOGREG_HEAD_TOL,
        random_state=seed,
    )
    with warnings.catch_warnings():
        # At a tolerance this tight lbfgs can stop a hair short on a separable
        # set; the weight vector is already on the path, which is the point.
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(X64, y_int, sample_weight=sample_weight)

    coef = np.asarray(clf.coef_, dtype=np.float32).reshape(1, -1)
    intercept = np.asarray(clf.intercept_, dtype=np.float32).reshape(1)

    ensure_torch_configured()
    model = build_model(input_dim, hidden_dim=LINEAR_LOGREG_HEAD)
    layer: nn.Linear = model[0]  # type: ignore[assignment]  # a linear sentinel builds exactly this
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(coef))
        layer.bias.copy_(torch.from_numpy(intercept))
    model.eval()
    return model.to(get_torch_device())
