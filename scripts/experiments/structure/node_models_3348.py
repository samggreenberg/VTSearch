"""Candidate per-node density models for the Coverage Atlas (#3348).

The shipped atlas models each node as one von Mises-Fisher direction ``mu`` and
calibrates a 21-point quantile grid of leave-one-out ``t = mu . x``.  #3329 part 2
measured that on held-out in-domain data and found it miscalibrated everywhere
(KS 0.103) and useless on ``dinov3_patch`` (guard separation 0.13).  This module
prices the replacements #3348 names, on the SAME tree, so the only thing that
varies between candidates is what a node knows about its points:

statistic (what a node scores a query with; higher = more typical)
    ``vmf``   the shipped mean direction, ``mu . x``
    ``knn``   mean cosine of the k nearest build points in the node (non-parametric)
    ``mix``   a small vMF mixture per node: ``max_j mu_j . x`` over 1-3 directions
    ``ppca``  a Gaussian on the node's top principal directions plus isotropic
              residual (probabilistic PCA), scored as minus its Mahalanobis term

calibration (how a node turns a score into a p-value)
    ``loo``   the shipped scheme: the tree is built on every build point and each
              node ranks a query among its OWN points' leave-one-out scores (exact
              for ``vmf``/``knn``, 5-fold cross-fit for ``mix``/``ppca``)
    ``split`` split-conformal: the tree is built on half the build points and
              each node ranks a query among the OTHER half's points routed to it
              exactly as a query is.  The partition never saw those points, which
              is what makes the deepest node's rank exchangeable with a fresh
              in-domain query's -- the property ``loo`` cannot have, because the
              k-means cell boundaries were drawn around the points it ranks against

combiner (how a root-to-leaf path of p-values becomes one)
    ``deepest`` / ``mean`` / ``median``, as priced in #3329

Two references ride alongside: ``grid_*`` is the shipped code path itself
(``typicality_pvalues`` and its path), so the baseline is the production
arithmetic and not a re-implementation of it; ``knn_global`` is a tree-free
split-conformal kNN over the whole build set -- uniform by construction, the
positive control #3348 asks for, and a candidate in its own right.

Every p-value here is a conformal rank ``(1 + #{cal <= s}) / (n_cal + 1)``, not a
grid interpolation, so "is the statistic the problem or the grid?" is separable:
``vmf_loo_*`` is the shipped statistic and calibration set read off exact ranks.
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections.abc import Callable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

#: A node calibrates only on this many calibration points and this
#: concentration -- the shipped floors (``_CALIBRATION_MIN_NODE`` /
#: ``_CALIBRATION_MIN_RBAR``), applied to every candidate so eligibility never
#: differs between them.
MIN_NODE = 20
MIN_RBAR = 0.1

#: Neighbour count for the kNN statistics (the app's conformal default).
KNN_K = 5

#: Principal directions kept by ``ppca`` (capped at n // 4 on small nodes).
PPCA_RANK = 10

#: Folds for the cross-fit leave-out on ``mix`` and ``ppca``.
CROSS_FOLDS = 5

STATISTICS = ("vmf", "knn", "mix", "ppca")
CALIBRATIONS = ("loo", "split")
COMBINERS = ("deepest", "mean", "median")

_EPS = 1e-12
_CHUNK = 2048


def normalize_rows(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return np.where(n > _EPS, m / np.maximum(n, _EPS), 0.0).astype(np.float32)


def to_frame(atlas, x: np.ndarray) -> np.ndarray:
    """Center into *atlas*'s frame and renormalize, exactly as the atlas does."""
    return normalize_rows(np.asarray(x, dtype=np.float32) - atlas.center)


# ---------------------------------------------------------------------------
# Statistics.  Each is (fit, score, leave_out): fit(V) -> model,
# score(model, X) -> (len(X),) higher = more typical, leave_out(V) -> each row
# of V scored by a model that did not see it.
# ---------------------------------------------------------------------------


def topk_mean_sims(q: np.ndarray, r: np.ndarray, k: int, exclude_self: bool = False) -> np.ndarray:
    """Mean of the *k* largest cosines from each row of *q* to the rows of *r*.

    ``exclude_self`` requires ``q is r`` row-for-row and drops each row's own
    entry, which is the exact leave-one-out score.
    """
    n_r = r.shape[0]
    k = max(1, min(k, n_r - (1 if exclude_self else 0)))
    out = np.empty(q.shape[0], dtype=np.float64)
    for s in range(0, q.shape[0], _CHUNK):
        sims = q[s : s + _CHUNK] @ r.T
        if exclude_self:
            rows = np.arange(sims.shape[0])
            sims[rows, rows + s] = -np.inf
        part = -np.partition(-sims, k - 1, axis=1)[:, :k]
        out[s : s + _CHUNK] = part.mean(axis=1)
    return out


def _vmf_fit(v: np.ndarray):
    r = v.sum(axis=0)
    n = float(np.linalg.norm(r))
    return (r / n).astype(np.float32) if n > _EPS else np.zeros(v.shape[1], np.float32)


def _vmf_score(mu, x: np.ndarray) -> np.ndarray:
    return (x @ mu).astype(np.float64)


def _vmf_leave_out(v: np.ndarray) -> np.ndarray:
    # The atlas's own closed form: mu_loo(x) . x = (R.x - 1) / ||R - x||.
    r = v.sum(axis=0).astype(np.float64)
    rl = float(np.linalg.norm(r))
    rd = v.astype(np.float64) @ r
    return (rd - 1.0) / np.sqrt(np.maximum(rl * rl - 2.0 * rd + 1.0, _EPS))


def _knn_fit(v: np.ndarray):
    return v


def _knn_score(ref, x: np.ndarray) -> np.ndarray:
    return topk_mean_sims(x, ref, KNN_K)


def _knn_leave_out(v: np.ndarray) -> np.ndarray:
    return topk_mean_sims(v, v, KNN_K, exclude_self=True)


def _mix_components(n: int) -> int:
    return 3 if n >= 90 else 2 if n >= 40 else 1


def _mix_fit(v: np.ndarray):
    m = _mix_components(v.shape[0])
    if m == 1:
        return _vmf_fit(v)[None, :]
    from sklearn.cluster import KMeans  # noqa: PLC0415

    km = KMeans(n_clusters=m, n_init=3, random_state=0).fit(v)
    return normalize_rows(km.cluster_centers_)


def _mix_score(mus, x: np.ndarray) -> np.ndarray:
    return (x @ mus.T).max(axis=1).astype(np.float64)


def _ppca_fit(v: np.ndarray):
    n, d = v.shape
    mean = v.mean(axis=0)
    c = (v - mean).astype(np.float64)
    r = max(1, min(PPCA_RANK, n // 4, d - 1))
    _, s, vt = np.linalg.svd(c, full_matrices=False)
    lam = (s[:r] ** 2) / n
    total = float((c * c).sum()) / n
    sigma2 = max((total - float(lam.sum())) / max(d - r, 1), 1e-8)
    return mean, vt[:r], np.maximum(lam, sigma2), sigma2


def _ppca_score(model, x: np.ndarray) -> np.ndarray:
    mean, comps, lam, sigma2 = model
    c = x.astype(np.float64) - mean
    z = c @ comps.T
    resid = np.maximum((c * c).sum(axis=1) - (z * z).sum(axis=1), 0.0)
    return -((z * z / lam).sum(axis=1) + resid / sigma2)


def _cross_fit(fit, score) -> Callable[[np.ndarray], np.ndarray]:
    def leave_out(v: np.ndarray) -> np.ndarray:
        n = v.shape[0]
        folds = np.arange(n) % CROSS_FOLDS
        np.random.default_rng(n).shuffle(folds)
        out = np.empty(n, dtype=np.float64)
        for f in range(CROSS_FOLDS):
            hold = folds == f
            if hold.any():
                out[hold] = score(fit(v[~hold]), v[hold])
        return out

    return leave_out


STAT_FUNCS = {
    "vmf": (_vmf_fit, _vmf_score, _vmf_leave_out),
    "knn": (_knn_fit, _knn_score, _knn_leave_out),
    "mix": (_mix_fit, _mix_score, _cross_fit(_mix_fit, _mix_score)),
    "ppca": (_ppca_fit, _ppca_score, _cross_fit(_ppca_fit, _ppca_score)),
}


# ---------------------------------------------------------------------------
# Routing, calibration, and scoring on a built atlas
# ---------------------------------------------------------------------------


def route(atlas, q: np.ndarray) -> dict[str, np.ndarray]:
    """Node name -> indices of the rows of *q* (already in frame) whose path visits it.

    The same cosine-nearest-child descent :meth:`CoverageAtlas.typicality_pvalues`
    uses, keeping every node on the path including the root.
    """
    out: dict[str, np.ndarray] = {}
    if not atlas.nodes or q.shape[0] == 0:
        return out

    def walk(name: str, idx: np.ndarray) -> None:
        out[name] = idx
        children = atlas.nodes[name]["children"]
        if not children:
            return
        mus = np.stack([atlas.nodes[c]["mu"] for c in children])
        nearest = np.argmax(q[idx] @ mus.T, axis=1)
        for ci, c in enumerate(children):
            sub = idx[nearest == ci]
            if sub.size:
                walk(c, sub)

    walk("0", np.arange(q.shape[0]))
    return out


def conformal_p(cal_sorted: np.ndarray, s: np.ndarray) -> np.ndarray:
    """``(1 + #{cal <= s}) / (n + 1)`` against an ascending calibration array."""
    return (1.0 + np.searchsorted(cal_sorted, s, side="right")) / (cal_sorted.size + 1.0)


class NodeModel:
    """One (statistic, calibration) candidate fitted on one atlas.

    ``loo``: *atlas* was built on *build* (rows in frame, keyed by the atlas's
    ids via *id_row*); each eligible node fits on its own points and calibrates
    on their leave-out scores.  ``split``: *atlas* was built on the tree half
    and *cal* holds the other half (in frame); each eligible node fits on its
    tree points and calibrates on the calibration points routed to it.
    """

    def __init__(
        self,
        atlas,
        statistic: str,
        calibration: str,
        build: np.ndarray,
        id_row: dict[int, int],
        cal: np.ndarray | None = None,
    ):
        fit, score, leave_out = STAT_FUNCS[statistic]
        self.atlas, self.statistic, self.calibration = atlas, statistic, calibration
        self._score = score
        self.models: dict[str, object] = {}
        self.cal: dict[str, np.ndarray] = {}
        t0 = time.time()
        cal_routes = route(atlas, cal) if calibration == "split" and cal is not None else {}
        for name, node in atlas.nodes.items():
            if node["rbar"] < MIN_RBAR:
                continue
            rows = np.fromiter((id_row[i] for i in node["ids"]), dtype=np.int64, count=len(node["ids"]))
            v = build[rows]
            if calibration == "loo":
                if v.shape[0] < MIN_NODE:
                    continue
                self.models[name] = fit(v)
                self.cal[name] = np.sort(leave_out(v))
            else:
                ci = cal_routes.get(name)
                if ci is None or ci.size < MIN_NODE or v.shape[0] < 2:
                    continue
                model = fit(v)
                self.models[name] = model
                self.cal[name] = np.sort(score(model, cal[ci]))
        self.fit_seconds = time.time() - t0

    def paths(self, q: np.ndarray) -> list[list[float]]:
        """Per query row (in frame), the eligible nodes' p-values, root first."""
        routes = route(self.atlas, q)
        out: list[list[tuple[int, float]]] = [[] for _ in range(q.shape[0])]
        for name, idx in routes.items():
            if name not in self.models:
                continue
            p = conformal_p(self.cal[name], self._score(self.models[name], q[idx]))
            depth = self.atlas.nodes[name]["depth"]
            for j, row in enumerate(idx):
                out[int(row)].append((depth, float(p[j])))
        return [[p for _, p in sorted(pp)] for pp in out]


def combine(paths: list[list[float]]) -> dict[str, np.ndarray]:
    """``deepest`` / ``mean`` / ``median`` of each path; an empty path reads 1.0."""
    n = len(paths)
    out = {c: np.ones(n, dtype=np.float64) for c in COMBINERS}
    for i, pv in enumerate(paths):
        if pv:
            a = np.asarray(pv, dtype=np.float64)
            out["deepest"][i] = a[-1]
            out["mean"][i] = a.mean()
            out["median"][i] = float(np.median(a))
    return out


class GlobalKnn:
    """Tree-free split-conformal kNN: fit on one half, calibrate on the other.

    Uniform by construction under exchangeability, so it is the positive
    control.  Works on the raw unit vectors (no atlas frame): the cosine the
    app's own evidence-coverage rule uses.
    """

    def __init__(self, fit_rows: np.ndarray, cal_rows: np.ndarray):
        t0 = time.time()
        self.ref = normalize_rows(fit_rows)
        self.cal = np.sort(topk_mean_sims(normalize_rows(cal_rows), self.ref, KNN_K))
        self.fit_seconds = time.time() - t0

    def pvalues(self, x: np.ndarray) -> np.ndarray:
        return conformal_p(self.cal, topk_mean_sims(normalize_rows(x), self.ref, KNN_K))


def shift_verdict(p: np.ndarray, alpha: float = 0.05) -> dict:
    """The shipped ``domain_shift_report`` rule applied to any p-value vector."""
    p = np.asarray(p, dtype=np.float64)
    n = int(p.size)
    if n == 0:
        return {"n_items": 0, "frac_atypical": 0.0, "z_score": 0.0, "shifted": False, "median_pvalue": 1.0}
    frac = float(np.mean(p < alpha))
    se = math.sqrt(alpha * (1.0 - alpha) / n)
    z = (frac - alpha) / se
    return {
        "n_items": n,
        "frac_atypical": frac,
        "z_score": z,
        "shifted": bool(z > 3.0 and frac >= 2.0 * alpha),
        "median_pvalue": float(np.median(p)),
    }
