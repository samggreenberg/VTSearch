"""The candidate unanchored-GMM fits #3585 gates, and the swap that installs one.

Every arm here fits the *same* model - two Gaussians over one dimension - to the
same score sample.  They differ only in how the EM is started and where it is
stopped, which is exactly what makes the comparison a gate rather than a
benchmark: a different starting point lands EM in a different place within its
tolerance, and sometimes in a different basin, so the threshold can move even
though nothing about the estimator's *definition* changed.

``baseline`` is what shipped until #3585 (sklearn's ``GaussianMixture``);
``native`` is what this branch ships.  The rest are the alternatives the issue
and its comment thread name, kept as arms so the choice between them is a table
rather than an argument.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from vtscore.training.thresholds import gmm as G
from vtscore.training.thresholds.gmm import GmmFit1D

#: An arm is ``name -> (fit callable, one-line description)``.  The callable
#: takes the score array a production caller would hand :func:`fit_score_gmm`
#: and returns a :class:`GmmFit1D` or ``None``, i.e. it is drop-in for it.
FitFn = Callable[[np.ndarray], "GmmFit1D | None"]


def _sklearn_variant(**kwargs: Any) -> FitFn:
    """A ``GaussianMixture`` arm differing from the pre-#3585 default by *kwargs*."""

    def fit(arr: np.ndarray) -> "GmmFit1D | None":
        if arr.shape[0] < 2:
            return None
        from sklearn.mixture import GaussianMixture

        try:
            gmm = GaussianMixture(n_components=2, random_state=42, **kwargs)
            gmm.fit(np.asarray(arr, dtype=np.float64).reshape(-1, 1))
            means = np.ravel(gmm.means_)
            variances = np.ravel(gmm.covariances_)
            weights = np.ravel(gmm.weights_)
            lo = 0 if means[0] < means[1] else 1
            hi = 1 - lo
            return GmmFit1D(
                w_lo=float(weights[lo]),
                mu_lo=float(means[lo]),
                var_lo=float(variances[lo]),
                w_hi=float(weights[hi]),
                mu_hi=float(means[hi]),
                var_hi=float(variances[hi]),
            )
        except Exception:
            return None

    return fit


def _native_variant(tol: float, max_iter: int) -> FitFn:
    """The branch's own fit, run at a named ``(tol, max_iter)`` instead of the shipped pair.

    Set on the module rather than passed, because :func:`_plain_em` reads both
    off it - which is also what lets one process run every arm without
    re-importing anything.
    """

    def fit(arr: np.ndarray) -> "GmmFit1D | None":
        prev = (G._EM_TOL, G._EM_MAX_ITER)
        G._EM_TOL, G._EM_MAX_ITER = tol, max_iter
        try:
            return G.fit_score_gmm(arr)
        finally:
            G._EM_TOL, G._EM_MAX_ITER = prev

    return fit


def _subsampled(inner: FitFn, cap: int) -> FitFn:
    """*inner*, but fitted on at most *cap* scores - the ``_GMM_MAX_SAMPLES`` arm.

    Takes the deterministic seed-42 draw :func:`gmm_fit_array` takes, so the
    only difference from *inner* is the sample size.  Not a candidate to ship
    from this issue (it changes what the fit *sees*, not how it is computed);
    it is here because the issue names it as the third lever and a number is
    cheaper than a follow-up study.
    """

    def fit(arr: np.ndarray) -> "GmmFit1D | None":
        arr = np.asarray(arr, dtype=np.float64).ravel()
        if arr.shape[0] > cap:
            arr = np.random.default_rng(42).choice(arr, size=cap, replace=False)
        return inner(arr)

    return fit


#: The gate's arms.  ``baseline`` first: every other arm is scored against it.
ARMS: "dict[str, tuple[FitFn, str]]" = {
    "baseline": (G.fit_score_gmm_sklearn, "sklearn GaussianMixture(2, random_state=42) - what shipped before #3585"),
    "native": (_native_variant(1e-8, 200), "2-means init + the anchored EM loop with no anchors, tol 1e-8"),
    "native_tol1e-6": (_native_variant(1e-6, 200), "native, stopped at parameter delta 1e-6"),
    "native_tol1e-4": (_native_variant(1e-4, 200), "native, stopped at parameter delta 1e-4"),
    "native_tol1e-3": (_native_variant(1e-3, 200), "native, stopped at parameter delta 1e-3"),
    "native_iter50": (_native_variant(1e-8, 50), "native at tol 1e-8, capped at 50 EM iterations"),
    "sklearn_kmeanspp": (
        _sklearn_variant(init_params="k-means++"),
        "sklearn with the k-means++ init (5-6x, per #3585)",
    ),
    "sklearn_spherical": (
        _sklearn_variant(covariance_type="spherical"),
        "sklearn with the scalar covariance (measured: 0.97x)",
    ),
    "native_10k": (_subsampled(_native_variant(1e-8, 200), 10_000), "native, fitted on at most 10k scores"),
}

#: Arms that are candidates to *ship*.  The rest are references or diagnostics:
#: ``baseline`` is the incumbent, ``native_10k`` changes the sample rather than
#: the fit, and the two sklearn variants are the issue's cheap options.
SHIPPABLE = ("native", "native_tol1e-6", "native_tol1e-4", "native_tol1e-3", "native_iter50")


def _bindings() -> "list[tuple[Any, str]]":
    """Every module attribute that names the unanchored fit, right now.

    ``fit_score_gmm`` is imported by name in several modules and fetched off the
    package by the ones that import it inside a function, so swapping the
    definition in :mod:`vtscore.training.thresholds.gmm` alone leaves half the
    call sites on the old fit - silently, and in the direction that makes an arm
    look like the baseline.  Collected by scanning the imported modules instead
    of by listing them, so a new call site cannot be forgotten here.
    """
    out: "list[tuple[Any, str]]" = []
    for mod in list(sys.modules.values()):
        name = getattr(mod, "__name__", "")
        if not (name.startswith("vtscore") or name.startswith("__main__")):
            continue
        if getattr(mod, "fit_score_gmm", None) is not None:
            out.append((mod, "fit_score_gmm"))
    return out


@contextmanager
def swap_fit(fit: FitFn) -> Iterator[None]:
    """Install *fit* as the unanchored GMM fit everywhere, for the duration.

    Restores every binding it touched, including the definition itself, so arms
    can be run one after another in one process against one corpus.
    """
    bindings = _bindings()
    saved = [(mod, attr, getattr(mod, attr)) for mod, attr in bindings]
    try:
        for mod, attr in bindings:
            setattr(mod, attr, fit)
        yield
    finally:
        for mod, attr, old in saved:
            setattr(mod, attr, old)


def mean_loglik(fit: "GmmFit1D | None", x: np.ndarray) -> float:
    """Mean log-likelihood of the fitted mixture on *x* - the objective EM maximises.

    The one comparison that says which of two fits of the same model is
    *better*, rather than only that they differ: both arms are maximising this,
    so a candidate that lands higher on it has not merely moved the threshold,
    it has moved it toward a better fit of the sample.
    """
    if fit is None:
        return float("nan")
    x = np.asarray(x, dtype=np.float64).ravel()
    a = (
        math.log(max(fit.w_lo, 1e-300))
        - 0.5 * np.log(2.0 * math.pi * fit.var_lo)
        - (x - fit.mu_lo) ** 2 / (2.0 * fit.var_lo)
    )
    b = (
        math.log(max(fit.w_hi, 1e-300))
        - 0.5 * np.log(2.0 * math.pi * fit.var_hi)
        - (x - fit.mu_hi) ** 2 / (2.0 * fit.var_hi)
    )
    m = np.maximum(a, b)
    return float(np.mean(m + np.log(np.exp(a - m) + np.exp(b - m))))
