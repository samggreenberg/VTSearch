#!/usr/bin/env python
"""One-shot goodness-of-fit measurement for the #3329 inventory's parts B and C.

    python structure_fits_3329.py --dataset vg_scale_any --embedder siglip --out DIR

Part A of #3329 (the score mixture) rode the click loop, because every quantity
it needed was already computed per step.  **Parts B and C do not**: the Coverage
Atlas, the kNN conformal support rule, the UMAP layout and the HDBSCAN
compaction radius are all fitted ONCE per dataset, so they need a one-shot stage
rather than 192 cells.  This is that stage.

Each family is measured against **its own stated assumption**, not against a
rival model - the distinction that makes #3329 a different question from every
diagnostic already in the tree:

``atlas``
    ``domain_shift_report``'s docstring states the null outright: "under the
    null ... typicality p-values are roughly uniform, so about *alpha* of them
    fall below *alpha*".  That is a PIT claim and it has never been checked.
    Measured on a **held-out in-domain split**, which is the only honest test:
    the build points are in their own calibration quantiles.

``atlas_deepest``
    The same p-values with the path averaging removed.  ``typicality_pvalues``
    scores at EVERY calibrated node along the root-to-leaf path and returns the
    mean, which cannot be uniform even if each node's own p-value is: the mean
    of m correlated uniforms concentrates on 0.5.  Emitting both is what
    separates "the atlas is mis-fitted" from "the aggregation is the problem".

``conformal``
    Split-conformal support p-values are uniform under exchangeability BY
    CONSTRUCTION, so this scope is partly a positive control on the harness -
    if it does not read uniform, the reading is the instrument's fault.

``umap``
    Trustworthiness and continuity against neighbourhood size, a Shepard rank
    correlation, and k-NN class purity before vs after projection.  The layout
    has no goodness measure anywhere in the tree.

``compaction``
    ``_build_units`` fits each cluster's extent as the 90th-percentile core
    radius.  A fitted 90th percentile should contain 90% of the points it
    describes; this measures what it actually contains.

Long-format CSV out (one row per statistic), plus an ``.npz`` of the p-value
vectors so the figures can draw distributions rather than summaries.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration"))

import common  # noqa: E402

common.setup_env()

#: Held-out share used as the in-domain test population everywhere.
HOLDOUT_FRACTION = 0.2

#: Neighbourhood sizes the projection is scored at.
UMAP_KS = (5, 10, 15, 30, 50)

#: Trustworthiness is O(n^2) in memory; above this the point set is subsampled.
#: 3000 keeps the full pairwise matrix at ~70 MB and the estimate stable.
PROJECTION_MAX_N = 3000

#: k for the conformal support rule, mirroring the app's default.
CONFORMAL_K = 5


def _rows(**base: Any):
    out: list[dict] = []

    def add(family: str, scope: str, statistic: str, value: float, n: int = 0) -> None:
        out.append(
            {**base, "family": family, "scope": scope, "statistic": statistic, "value": float(value), "n": int(n)}
        )

    return out, add


def ks_uniform(p: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance between *p* and U(0,1).

    Written out rather than imported so the number in the report is one this
    file defines: ``max|ECDF(p) - p|`` over both one-sided gaps.
    """
    p = np.sort(np.asarray(p, dtype=np.float64))
    n = p.size
    if n == 0:
        return float("nan")
    i = np.arange(1, n + 1, dtype=np.float64)
    return float(max(np.max(i / n - p), np.max(p - (i - 1) / n)))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="stable")
        r = np.empty(x.size, dtype=np.float64)
        r[order] = np.arange(x.size, dtype=np.float64)
        return r

    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def path_pvalues(atlas, matrix: np.ndarray) -> list[list[float]]:
    """Every calibrated node's p-value along each item's root-to-leaf path.

    ``typicality_pvalues`` collapses this to a mean before anyone sees it, so
    the shipped statistic cannot be re-aggregated after the fact.  Returning the
    path lets one pass price every candidate combiner against the same data,
    which is the difference between "the null is false" and "here is what to do
    about it".
    """
    from vtscore.coverage.atlas import (  # noqa: PLC0415
        _CALIBRATION_GRID,
        _CALIBRATION_MIN_NODE,
        _CALIBRATION_MIN_RBAR,
        _normalize_rows,
    )

    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    n_items = matrix.shape[0]
    paths: list[list[float]] = [[] for _ in range(n_items)]
    if not atlas.nodes or n_items == 0:
        return paths
    queries = _normalize_rows(matrix - atlas.center)

    def route(name: str, idx: np.ndarray) -> None:
        node = atlas.nodes[name]
        if node["n"] >= _CALIBRATION_MIN_NODE and node["rbar"] >= _CALIBRATION_MIN_RBAR and node["t_quantiles"]:
            t = queries[idx] @ node["mu"]
            grid = np.asarray(node["t_quantiles"], dtype=np.float64)
            pv = np.maximum(np.interp(t, grid, _CALIBRATION_GRID, left=0.0, right=1.0), 0.5 / max(node["n"], 1))
            for j, row in enumerate(idx):
                paths[int(row)].append(float(pv[j]))
        children = node["children"]
        if not children:
            return
        mus = np.stack([atlas.nodes[c]["mu"] for c in children])
        nearest = np.argmax(queries[idx] @ mus.T, axis=1)
        for ci in range(len(children)):
            sub = idx[nearest == ci]
            if sub.size:
                route(children[ci], sub)

    route("0", np.arange(n_items))
    return paths


def aggregate_paths(paths: list[list[float]]) -> dict[str, np.ndarray]:
    """Candidate combiners for a root-to-leaf path of p-values.

    ``mean`` is what ships.  The others are the obvious alternatives, priced on
    the same data so the report can name a repair rather than only a defect:

    ``deepest``  the leaf-most calibrated node alone - no combination at all;
    ``min``      the most conservative reading, and the one a "is this point
                 atypical ANYWHERE on its path?" question implies;
    ``median``   a robust mean, which keeps the smoothing but drops the tails;
    ``fisher``   Fisher's method, the textbook combiner for independent
                 p-values, mapped back through its own chi-square null so the
                 result is on the p-value scale the guard expects. The path's
                 p-values are NOT independent (a leaf is inside its parent), so
                 this is expected to over-reject; it is priced to show by how
                 much.
    """
    import math  # noqa: PLC0415

    n = len(paths)
    out = {k: np.full(n, np.nan) for k in ("mean", "deepest", "min", "median", "fisher")}
    for i, pv in enumerate(paths):
        if not pv:
            for k in out:
                out[k][i] = 1.0
            continue
        a = np.asarray(pv, dtype=np.float64)
        out["mean"][i] = float(a.mean())
        out["deepest"][i] = float(a[-1])
        out["min"][i] = float(a.min())
        out["median"][i] = float(np.median(a))
        stat = -2.0 * float(np.log(np.maximum(a, 1e-12)).sum())
        k2 = a.size
        # Survival function of chi-square with 2k dof, in closed form for even
        # dof: exp(-x/2) * sum_{j<k} (x/2)^j / j!
        x = stat / 2.0
        term, acc = 1.0, 1.0
        for j in range(1, k2):
            term *= x / j
            acc += term
        out["fisher"][i] = float(min(1.0, math.exp(-x) * acc))
    return out


def deepest_pvalues(atlas, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``typicality_pvalues`` WITHOUT the path averaging, plus the path length.

    Mirrors :meth:`CoverageAtlas.typicality_pvalues` exactly - same centering,
    same routing, same interpolation onto ``_CALIBRATION_GRID`` - and keeps only
    the DEEPEST calibrated node's p-value instead of the mean over the path.
    Returns ``(p_deepest, n_scored)``; ``n_scored`` is how many calibrated nodes
    the shipped version averaged over, which is the explanatory variable for any
    under-dispersion it shows.
    """
    from vtscore.coverage.atlas import (  # noqa: PLC0415
        _CALIBRATION_GRID,
        _CALIBRATION_MIN_NODE,
        _CALIBRATION_MIN_RBAR,
        _normalize_rows,
    )

    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    n_items = matrix.shape[0]
    if not atlas.nodes or n_items == 0:
        return np.ones(n_items, dtype=np.float64), np.zeros(n_items, dtype=np.int64)

    queries = _normalize_rows(matrix - atlas.center)
    p_last = np.ones(n_items, dtype=np.float64)
    n_scored = np.zeros(n_items, dtype=np.int64)

    def route(name: str, idx: np.ndarray) -> None:
        node = atlas.nodes[name]
        if node["n"] >= _CALIBRATION_MIN_NODE and node["rbar"] >= _CALIBRATION_MIN_RBAR and node["t_quantiles"]:
            t = queries[idx] @ node["mu"]
            grid = np.asarray(node["t_quantiles"], dtype=np.float64)
            p = np.interp(t, grid, _CALIBRATION_GRID, left=0.0, right=1.0)
            p_last[idx] = np.maximum(p, 0.5 / max(node["n"], 1))
            n_scored[idx] += 1
        children = node["children"]
        if not children:
            return
        mus = np.stack([atlas.nodes[c]["mu"] for c in children])
        nearest = np.argmax(queries[idx] @ mus.T, axis=1)
        for ci in range(len(children)):
            sub = idx[nearest == ci]
            if sub.size:
                route(children[ci], sub)

    route("0", np.arange(n_items))
    return p_last, n_scored


def _pit_stats(add, family: str, scope: str, p: np.ndarray) -> None:
    """Every PIT reading for one p-value vector, under one (family, scope)."""
    p = np.asarray(p, dtype=np.float64)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return
    add(family, scope, "ks_uniform", ks_uniform(p), p.size)
    add(family, scope, "mean", float(p.mean()), p.size)
    # sd is the tell for path averaging: U(0,1) has sd 0.289, and the mean of m
    # correlated uniforms is tighter than that by construction.
    add(family, scope, "sd", float(p.std()), p.size)
    add(family, scope, "median", float(np.median(p)), p.size)
    for alpha in (0.01, 0.05, 0.10):
        add(family, scope, f"frac_below_{alpha:g}", float((p < alpha).mean()), p.size)
    add(family, scope, "frac_above_0.95", float((p > 0.95).mean()), p.size)


def load_matrix(dataset: str, embedder: str) -> tuple[np.ndarray, list[int], list[list[str]]]:
    """``(N, D) float32`` matrix, ids, and per-item category lists from the pile."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration"))
    from _cells_io import load_medias  # noqa: PLC0415

    from vtscore.datasets import loader as _loader  # noqa: PLC0415

    pkl = _loader.EMBEDDINGS_DIR / f"{dataset}__{embedder}.pkl"
    medias = load_medias(pkl)
    ids = sorted(medias)
    vecs, keep, cats = [], [], []
    for i in ids:
        v = medias[i].get("embeddings", {}).get(embedder)
        if v is None:
            continue
        arr = np.asarray(v, dtype=np.float32).ravel()
        if arr.size == 0:
            continue
        vecs.append(arr)
        keep.append(i)
        # `categories` is the multi-label list, but not every pile dataset
        # carries one - caltech101_m has the single-label `category` instead,
        # and reading only the plural silently dropped its class purity and
        # conformal stages on the smoke run.
        c = medias[i].get("categories") or []
        if not c and medias[i].get("category"):
            c = [medias[i]["category"]]
        cats.append(list(c))
    if not vecs:
        raise RuntimeError(f"no {embedder} vectors in {pkl}")
    return np.stack(vecs), keep, cats


def main(argv: list[str] | None = None) -> int:  # noqa: C901, PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--embedder", required=True)
    ap.add_argument("--out", required=True, help="directory for the CSV and npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-projection", action="store_true", help="atlas + conformal only (fast)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    import pandas as pd

    from vtscore.coverage.atlas import CoverageAtlas, auto_max_depth, domain_shift_report

    t0 = time.time()
    matrix, ids, cats = load_matrix(args.dataset, args.embedder)
    common.log(f"{args.dataset}/{args.embedder}: {matrix.shape[0]} items, dim={matrix.shape[1]}")

    rows, add = _rows(dataset=args.dataset, embedder=args.embedder, seed=args.seed)
    store: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(matrix.shape[0])
    n_hold = max(1, int(round(HOLDOUT_FRACTION * matrix.shape[0])))
    hold_idx, build_idx = perm[:n_hold], perm[n_hold:]

    # --- Coverage Atlas -----------------------------------------------------
    # Built exactly as production builds it (labeling_progress / state.coverage):
    # k=3 with the size-derived depth cap, so the null under test is the shipped
    # estimator's null and not a variant of it.
    vectors = {int(ids[i]): matrix[i] for i in build_idx}
    atlas = CoverageAtlas(vectors, k=3, max_depth=auto_max_depth(len(vectors), k=3))
    add("atlas", "structure", "n_nodes", atlas.total_nodes, len(vectors))
    add("atlas", "structure", "depth", atlas.depth(), len(vectors))
    rbars = np.array([n["rbar"] for n in atlas.nodes.values()], dtype=np.float64)
    add("atlas", "structure", "rbar_median", float(np.median(rbars)), rbars.size)
    add("atlas", "structure", "rbar_p10", float(np.quantile(rbars, 0.10)), rbars.size)
    store["rbar"] = rbars

    hold_m, build_m = matrix[hold_idx], matrix[build_idx]
    p_hold = np.asarray(atlas.typicality_pvalues(hold_m), dtype=np.float64)
    p_build = np.asarray(atlas.typicality_pvalues(build_m), dtype=np.float64)
    p_deep, n_scored = deepest_pvalues(atlas, hold_m)
    _pit_stats(add, "atlas", "holdout", p_hold)
    _pit_stats(add, "atlas", "build", p_build)
    _pit_stats(add, "atlas_deepest", "holdout", p_deep)
    add("atlas", "holdout", "path_len_mean", float(n_scored.mean()), n_scored.size)
    add("atlas", "holdout", "path_len_median", float(np.median(n_scored)), n_scored.size)
    store["p_atlas_holdout"] = p_hold
    store["p_atlas_build"] = p_build
    store["p_atlas_deepest"] = p_deep
    store["atlas_path_len"] = n_scored.astype(np.float64)

    # Price every candidate combiner on the same paths, so the run can name a
    # repair instead of only a defect.  `mean` here must reproduce the shipped
    # `typicality_pvalues` -- it is the same arithmetic -- which makes it a free
    # consistency check on this whole path-capture route.
    aggs = aggregate_paths(path_pvalues(atlas, hold_m))
    for name, vals in aggs.items():
        _pit_stats(add, "atlas_agg", name, vals)
        store[f"p_agg_{name}"] = vals
    add(
        "atlas_agg",
        "mean",
        "max_abs_diff_vs_shipped",
        float(np.max(np.abs(aggs["mean"] - p_hold))) if p_hold.size else float("nan"),
        p_hold.size,
    )

    # The shipped guard, pointed at data it should call in-domain.  Anything but
    # z ~ 0 here is the null being wrong, since the holdout IS the build
    # distribution.
    rep = domain_shift_report(atlas, hold_m)
    for key in ("frac_atypical", "z_score", "median_pvalue"):
        add("atlas", "domain_shift_self", key, rep[key], rep["n_items"])
    add("atlas", "domain_shift_self", "shifted", float(bool(rep["shifted"])), rep["n_items"])
    common.log(f"  atlas: nodes={atlas.total_nodes} KS(holdout)={ks_uniform(p_hold):.3f} z_self={rep['z_score']:.1f}")

    # --- kNN conformal support ---------------------------------------------
    # Positive control AND a real reading: the rule is exchangeability-uniform by
    # construction, so a non-uniform result here indicts the harness first.
    from vtscore.detectors.evidence_coverage import support_pvalues  # noqa: PLC0415

    counts: dict[str, int] = {}
    for c in cats:
        for name in c:
            counts[name] = counts.get(name, 0) + 1
    if counts:
        target = max(counts, key=lambda c: counts[c])
        pos = np.array([i for i, c in enumerate(cats) if target in c], dtype=int)
        neg = np.array([i for i, c in enumerate(cats) if target not in c], dtype=int)
        if pos.size >= 40:
            pr = np.random.default_rng(args.seed + 1).permutation(pos.size)
            pos = pos[pr]
            cut = int(0.7 * pos.size)
            refs = matrix[pos[:cut]]
            refs = refs / np.maximum(np.linalg.norm(refs, axis=1, keepdims=True), 1e-12)
            q_in = matrix[pos[cut:]]
            q_in = q_in / np.maximum(np.linalg.norm(q_in, axis=1, keepdims=True), 1e-12)
            p_in = np.asarray(support_pvalues(q_in, refs, k=CONFORMAL_K), dtype=np.float64)
            _pit_stats(add, "conformal", "in_class_holdout", p_in)
            store["p_conformal_in"] = p_in
            add("conformal", "in_class_holdout", "target_prevalence", pos.size / len(cats), len(cats))
            if neg.size:
                nsub = np.random.default_rng(args.seed + 2).choice(neg, size=min(2000, neg.size), replace=False)
                q_out = matrix[nsub]
                q_out = q_out / np.maximum(np.linalg.norm(q_out, axis=1, keepdims=True), 1e-12)
                p_out = np.asarray(support_pvalues(q_out, refs, k=CONFORMAL_K), dtype=np.float64)
                _pit_stats(add, "conformal", "out_of_class", p_out)
                store["p_conformal_out"] = p_out
            common.log(f"  conformal: target={target!r} n_pos={pos.size} KS(in)={ks_uniform(p_in):.3f}")

    # --- Projection ---------------------------------------------------------
    if not args.skip_projection:
        from sklearn.manifold import trustworthiness  # noqa: PLC0415
        from sklearn.neighbors import NearestNeighbors  # noqa: PLC0415

        from vtscore.projection.compaction import _build_units, _cluster  # noqa: PLC0415
        from vtscore.projection.umap_projection import fit_projection  # noqa: PLC0415

        n = matrix.shape[0]
        sub = (
            np.random.default_rng(args.seed + 3).choice(n, size=PROJECTION_MAX_N, replace=False)
            if n > PROJECTION_MAX_N
            else np.arange(n)
        )
        X = matrix[sub]
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        proj = fit_projection(X, [int(ids[i]) for i in sub], random_state=args.seed)
        Y = np.asarray(proj.coords, dtype=np.float64)
        add("umap", "layout", "n_points", Y.shape[0], Y.shape[0])
        for k in UMAP_KS:
            if k >= Y.shape[0] - 1:
                continue
            add("umap", "layout", f"trustworthiness_k{k}", float(trustworthiness(Xn, Y, n_neighbors=k)), Y.shape[0])
            add("umap", "layout", f"continuity_k{k}", float(trustworthiness(Y, Xn, n_neighbors=k)), Y.shape[0])

        # Shepard: original cosine distance vs laid-out euclidean, over a random
        # pair sample (all pairs is 4.5M at n=3000 and adds nothing).
        pr = np.random.default_rng(args.seed + 4)
        ia, ib = pr.integers(0, Y.shape[0], 40000), pr.integers(0, Y.shape[0], 40000)
        ok = ia != ib
        ia, ib = ia[ok], ib[ok]
        d_hi = 1.0 - np.einsum("ij,ij->i", Xn[ia], Xn[ib])
        d_lo = np.linalg.norm(Y[ia] - Y[ib], axis=1)
        add("umap", "layout", "shepard_spearman", _spearman(d_hi, d_lo), ia.size)
        store["shepard_hi"] = d_hi[:5000]
        store["shepard_lo"] = d_lo[:5000]

        # Class purity before vs after: does a neighbourhood in the picture mean
        # what a neighbourhood in the embedding meant?
        sub_cats = [cats[i] for i in sub]
        has_cat = np.array([bool(c) for c in sub_cats])
        if has_cat.sum() >= 100:
            for space, M in (("embedding", Xn), ("layout", Y)):
                nn = NearestNeighbors(n_neighbors=11).fit(M)
                _, nbr = nn.kneighbors(M)
                shared = []
                for r in range(M.shape[0]):
                    if not sub_cats[r]:
                        continue
                    own = set(sub_cats[r])
                    hits = sum(1 for j in nbr[r][1:] if own & set(sub_cats[j]))
                    shared.append(hits / 10.0)
                add("umap", space, "knn_class_purity_k10", float(np.mean(shared)), len(shared))

        # Compaction: a fitted 90th percentile should contain 90%.  NOTE this
        # packer is OFF in production (PROJECTION_COMPACT_DEFAULT is False since
        # the 2026-07-22 UMAP sweep, and `compact` has no override path), and the
        # layout below is fit at that default, i.e. uncompacted.  What is measured
        # is the radius statistic the packer WOULD use, on the layout users see.
        labels = _cluster(Y, min_cluster_size=max(5, Y.shape[0] // 100))
        units, centres, radii = _build_units(Y, labels)
        # The radius is fitted to the CORE members only (`_build_units` excludes
        # folded-in noise on purpose), so the honest test of the fitted 90th
        # percentile is against those same points. The unit's full point list -
        # core plus the noise that rides along - is reported beside it, because
        # that is the set the packing actually treats as inside the circle, and
        # the gap between the two is how much room the pack under-claims.
        cluster_ids = sorted(int(k) for k in np.unique(labels) if k >= 0)
        core_contained, unit_contained, sizes = [], [], []
        for k, c, r in zip(cluster_ids, centres, radii, strict=False):
            core = Y[labels == k]
            if core.size == 0 or not np.isfinite(r) or r <= 0:
                continue
            core_contained.append(float(np.mean(np.linalg.norm(core - c, axis=1) <= r)))
            sizes.append(int(core.shape[0]))
        for u, c, r in zip(units, centres, radii, strict=False):
            idx = np.asarray(u)
            pts = Y[idx]
            if pts.size == 0 or not np.isfinite(r) or r <= 0:
                continue
            unit_contained.append(float(np.mean(np.linalg.norm(pts - c, axis=1) <= r)))
        if core_contained:
            add("compaction", "core", "n_clusters", len(core_contained), int(np.sum(sizes)))
            add("compaction", "core", "containment_mean", float(np.mean(core_contained)), len(core_contained))
            add("compaction", "core", "containment_median", float(np.median(core_contained)), len(core_contained))
            add("compaction", "core", "nominal", 0.90, len(core_contained))
            store["compaction_containment_core"] = np.asarray(core_contained, dtype=np.float64)
        if unit_contained:
            add("compaction", "unit", "containment_mean", float(np.mean(unit_contained)), len(unit_contained))
            store["compaction_containment_unit"] = np.asarray(unit_contained, dtype=np.float64)
        add("compaction", "clusters", "noise_fraction", float(np.mean(labels < 0)), labels.size)
        common.log(f"  projection: {len(core_contained)} clusters, noise={np.mean(labels < 0):.2f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # The seed is IN the stem: it moves the build/holdout split, so two seeds
    # are two measurements, and a stem without it would have silently kept only
    # the last of every three cells in the grid.
    stem = f"{args.dataset}__{args.embedder}__s{args.seed}"
    pd.DataFrame(rows).to_csv(out / f"struct_{stem}.csv", index=False)
    np.savez_compressed(out / f"struct_{stem}.npz", **store)
    (out / f"struct_{stem}.json").write_text(
        json.dumps({"dataset": args.dataset, "embedder": args.embedder, "seconds": time.time() - t0}, indent=2)
    )
    common.log(f"wrote {len(rows)} rows in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
