"""The heads #3197 compares, as plain ``(w, b)`` linear scorers.

Every arm here is a ``Linear(D, 1)`` (except ``mlp``, kept for reference), so
every arm reduces to a weight vector and a bias, and every mechanism question
reduces to "which ``w`` did the objective pick".  Two of the arms are the
production fits called through production code (``svm`` and ``lr`` go through
:func:`vtscore.training.mlp.train_model` with the shipped sentinels); every
other arm changes exactly one thing about one of them, and :func:`fidelity`
proves the replicas are the production fits before anything is measured.

The ladder from the shipped logistic head to a converged logistic regression
is the core of the "is it the loss, or the fit?" question, so it is spelled
out here rather than hidden behind flags:

========================  =====================================================
arm                       what it is
========================  =====================================================
``svm``                   SHIPPED.  liblinear LinearSVC, squared hinge, L2,
                          class-balanced, C = 1, penalised intercept.
``svm_C{c}``              the same fit at another C.
``svm_hinge``             plain (not squared) hinge at C = 1.
``svm_unbal``             the shipped SVM without the class balance.
``lr``                    the head the SVM replaced: balanced BCE, Adam lr 1e-3,
                          weight decay 1e-4, label smoothing 0.05, <= 200
                          epochs with patience 10, random (Kaiming) init.
``lr_nosmooth``           ``lr`` without label smoothing.
``lr_ep2000``             ``lr`` run 2000 epochs with no early stop (the
                          in-loop ``linconv`` arm).
``lr_zeroinit``           ``lr`` started from w = 0 instead of a random draw.
``lr_unbal``              ``lr`` without the class balance.
``lrconv_C{c}``           sklearn LogisticRegression, balanced, L2, lbfgs to
                          convergence: logistic loss ON the regularisation path.
``ridge_a{a}``            RidgeClassifier, balanced: squared loss + L2.
``centroid``              w = mean(Good) - mean(Bad): the limit BOTH losses
                          reach as regularisation goes to infinity.
``mlp``                   the retired auto-sized MLP (reference only).
========================  =====================================================
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

SVM_CS = (0.01, 0.1, 10.0, 100.0)
LR_CS = (0.01, 0.1, 1.0, 10.0, 100.0)
RIDGE_ALPHAS = (0.1, 1.0, 10.0)


@dataclass
class Fit:
    """A fitted arm: a scorer plus, for linear arms, its hyperplane."""

    score: Callable[[np.ndarray], np.ndarray]
    w: np.ndarray | None
    b: float | None
    seconds: float


def _linear(w: np.ndarray, b: float, seconds: float) -> Fit:
    w = np.asarray(w, dtype=np.float64).ravel()
    return Fit(score=lambda X: np.asarray(X, dtype=np.float64) @ w + b, w=w, b=float(b), seconds=seconds)


def _torch_xy(X: np.ndarray, y: np.ndarray):
    import torch

    return torch.from_numpy(np.asarray(X, dtype=np.float32)), torch.from_numpy(
        np.asarray(y, dtype=np.float32).reshape(-1, 1)
    )


def _from_module(model: Any, seconds: float) -> Fit:
    layer = model[0]
    w = layer.weight.detach().cpu().numpy().astype(np.float64).ravel()
    b = float(layer.bias.detach().cpu().numpy().ravel()[0])
    return _linear(w, b, seconds)


# --- production fits, through production code --------------------------------


def fit_svm_shipped(X: np.ndarray, y: np.ndarray) -> Fit:
    from vtscore.training.mlp import LINEAR_SVM_HEAD, train_model

    t0 = time.perf_counter()
    Xt, yt = _torch_xy(X, y)
    model = train_model(Xt, yt, X.shape[1], hidden_dim=LINEAR_SVM_HEAD)
    return _from_module(model, time.perf_counter() - t0)


def fit_lr_shipped(X: np.ndarray, y: np.ndarray) -> Fit:
    from vtscore.training.mlp import LINEAR_HEAD, train_model

    t0 = time.perf_counter()
    Xt, yt = _torch_xy(X, y)
    model = train_model(Xt, yt, X.shape[1], hidden_dim=LINEAR_HEAD)
    return _from_module(model, time.perf_counter() - t0)


def fit_mlp(X: np.ndarray, y: np.ndarray) -> Fit:
    import torch

    from vtscore.training.mlp import train_model

    t0 = time.perf_counter()
    Xt, yt = _torch_xy(X, y)
    model = train_model(Xt, yt, X.shape[1], hidden_dim=None)
    secs = time.perf_counter() - t0

    def score(Z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.from_numpy(np.asarray(Z, dtype=np.float32))).cpu().numpy().ravel().astype(np.float64)

    return Fit(score=score, w=None, b=None, seconds=secs)


# --- replicas that change one thing -------------------------------------------


def fit_svm(X: np.ndarray, y: np.ndarray, *, C: float = 1.0, loss: str = "squared_hinge", balanced: bool = True) -> Fit:
    """LinearSVC exactly as ``vtscore.training.svm._make_base_estimator`` builds it."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.svm import LinearSVC

    t0 = time.perf_counter()
    clf = LinearSVC(
        C=C,
        loss=loss,
        class_weight="balanced" if balanced else None,
        dual="auto" if loss == "squared_hinge" else True,
        max_iter=5000,
        random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(np.asarray(X, dtype=np.float32), np.asarray(y).astype(int))
    return _linear(clf.coef_.ravel(), float(clf.intercept_[0]), time.perf_counter() - t0)


def fit_lr_torch(
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 200,
    patience: int = 10,
    weight_decay: float = 1e-4,
    smoothing: float = 0.05,
    lr: float = 1e-3,
    zero_init: bool = False,
    balanced: bool = True,
    seed: int = 42,
) -> Fit:
    """A line-for-line replica of ``train_model``'s ``LINEAR_HEAD`` path on CPU.

    With every default it must reproduce :func:`fit_lr_shipped` exactly -
    :func:`fidelity` asserts that - so each non-default is a one-knob change to
    the shipped logistic head and nothing else.
    """
    import torch
    import torch.nn as nn

    from vtscore.config import MLP_DROPOUT
    from vtscore.training.mlp import LINEAR_HEAD, build_model

    t0 = time.perf_counter()
    Xt, yt = _torch_xy(X, y)
    g = torch.Generator()
    g.manual_seed(seed)
    model = build_model(X.shape[1], hidden_dim=LINEAR_HEAD, dropout=MLP_DROPOUT, generator=g)
    if zero_init:
        with torch.no_grad():
            model[0].weight.zero_()
            model[0].bias.zero_()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    num_true = int(yt.sum().item())
    num_false = len(yt) - num_true
    if balanced:
        weights = torch.where(yt == 1, num_false / num_true, 1.0).squeeze()
    else:
        weights = torch.ones(len(yt))
    ys = yt * (1.0 - 2.0 * smoothing) + smoothing
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    model.train()
    best = float("inf")
    since = 0
    n_epochs = 0
    with torch.random.fork_rng(), torch.enable_grad():
        torch.manual_seed(seed)
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = (loss_fn(model(Xt), ys).squeeze() * weights).mean()
            loss.backward()
            optimizer.step()
            n_epochs += 1
            cur = loss.item()
            if cur < best - 1e-4:
                best = cur
                since = 0
            else:
                since += 1
                if patience > 0 and since >= patience:
                    break
    model.eval()
    fit = _from_module(model, time.perf_counter() - t0)
    fit.epochs = n_epochs  # type: ignore[attr-defined]
    return fit


def fit_lr_sklearn(X: np.ndarray, y: np.ndarray, *, C: float) -> Fit:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    t0 = time.perf_counter()
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=20000, tol=1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(np.asarray(X, dtype=np.float64), np.asarray(y).astype(int))
    return _linear(clf.coef_.ravel(), float(clf.intercept_[0]), time.perf_counter() - t0)


def fit_ridge(X: np.ndarray, y: np.ndarray, *, alpha: float) -> Fit:
    from sklearn.linear_model import RidgeClassifier

    t0 = time.perf_counter()
    clf = RidgeClassifier(alpha=alpha, class_weight="balanced")
    clf.fit(np.asarray(X, dtype=np.float64), np.asarray(y).astype(int))
    return _linear(clf.coef_.ravel(), float(clf.intercept_[0]), time.perf_counter() - t0)


def fit_centroid(X: np.ndarray, y: np.ndarray) -> Fit:
    t0 = time.perf_counter()
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    mp, mn = X[y == 1].mean(0), X[y == 0].mean(0)
    w = mp - mn
    b = -float(w @ (mp + mn) / 2.0)
    return _linear(w, b, time.perf_counter() - t0)


# --- the arm table --------------------------------------------------------------


def arm_table(include_slow: bool) -> dict[str, Callable[[np.ndarray, np.ndarray], Fit]]:
    """Every arm by name.  ``include_slow`` adds the MLP and the 2000-epoch run."""
    arms: dict[str, Callable[[np.ndarray, np.ndarray], Fit]] = {
        "svm": fit_svm_shipped,
        "lr": fit_lr_shipped,
        "svm_hinge": lambda X, y: fit_svm(X, y, loss="hinge"),
        "svm_unbal": lambda X, y: fit_svm(X, y, balanced=False),
        "lr_nosmooth": lambda X, y: fit_lr_torch(X, y, smoothing=0.0),
        "lr_zeroinit": lambda X, y: fit_lr_torch(X, y, zero_init=True),
        "lr_unbal": lambda X, y: fit_lr_torch(X, y, balanced=False),
        "centroid": fit_centroid,
    }
    for c in SVM_CS:
        arms[f"svm_C{c:g}"] = lambda X, y, c=c: fit_svm(X, y, C=c)
    for c in LR_CS:
        arms[f"lrconv_C{c:g}"] = lambda X, y, c=c: fit_lr_sklearn(X, y, C=c)
    for a in RIDGE_ALPHAS:
        arms[f"ridge_a{a:g}"] = lambda X, y, a=a: fit_ridge(X, y, alpha=a)
    if include_slow:
        arms["lr_ep2000"] = lambda X, y: fit_lr_torch(X, y, epochs=2000, patience=0)
        arms["mlp"] = fit_mlp
    return arms


def fidelity(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Max |Δ| between each production fit and the replica that claims to be it.

    Raises when a replica is not the production fit: every one-knob arm is only
    a one-knob change if its all-defaults form IS the shipped head.
    """
    out = {}
    a, b = fit_svm_shipped(X, y), fit_svm(X, y)
    out["svm_vs_replica"] = float(max(np.abs(a.w - b.w).max(), abs(a.b - b.b)))
    a, b = fit_lr_shipped(X, y), fit_lr_torch(X, y)
    out["lr_vs_replica"] = float(max(np.abs(a.w - b.w).max(), abs(a.b - b.b)))
    # The SVM head is lifted into a float32 torch Linear, so it can only match
    # the float32 liblinear fit to float32 precision; the torch replica runs the
    # same float32 ops in the same order and must match bit for bit.
    if out["svm_vs_replica"] > 1e-5 or out["lr_vs_replica"] > 1e-6:
        raise AssertionError(f"a replica is not the production fit: {out}")
    return out
