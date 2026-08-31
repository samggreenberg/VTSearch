#!/usr/bin/env python
"""Are the browse canvas's named regions coherent? (#3329, inventory item 10)

    python region_coherence_3329.py --dataset D --embedder E --out DIR

The inventory asks for "within-vs-between cosine coherence per named cluster".
The *naming* half of Toponymy needs the captioner stack, but the **regions** do
not: Toponymy clusters in its own dedicated ~5-D cosine UMAP
(`signpost_build._clusterable_vectors`), and that clustering is what decides
which items share a sign on the canvas.  This measures those regions directly,
with no texts, no namer and no LLM involved - so nothing here depends on a
caption whose quality would confound the answer.

Two questions per cluster, both of which the canvas implicitly claims:

``coherence``
    mean pairwise cosine WITHIN the cluster, against the mean cosine BETWEEN it
    and everything else, both computed in the ORIGINAL embedding rather than in
    the 5-D reduction the clustering happened in.  A region whose within- and
    between-similarities are equal is a region the user is being shown a sign
    for that means nothing.

``purity``
    the dominant ground-truth category's share of the cluster.  Available here
    and nowhere in production, which is the point: it says whether a region
    corresponds to anything a person would name.

Layers are Toponymy's own, finest first, because the canvas shows different
layers at different zooms and a coarse region is allowed to be less coherent
than a fine one.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration"))

import common  # noqa: E402

common.setup_env()

from structure_fits_3329 import load_matrix  # noqa: E402

#: Cap on the pairwise work per cluster; above this a random subsample is used.
MAX_PAIRS_N = 400


def _unit(m: np.ndarray) -> np.ndarray:
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)


def _mean_within(x: np.ndarray, rng: np.random.Generator) -> float:
    if x.shape[0] < 2:
        return float("nan")
    if x.shape[0] > MAX_PAIRS_N:
        x = x[rng.choice(x.shape[0], MAX_PAIRS_N, replace=False)]
    g = x @ x.T
    n = g.shape[0]
    return float((g.sum() - np.trace(g)) / (n * (n - 1)))


def _mean_between(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> float:
    if x.shape[0] == 0 or y.shape[0] == 0:
        return float("nan")
    if x.shape[0] > MAX_PAIRS_N:
        x = x[rng.choice(x.shape[0], MAX_PAIRS_N, replace=False)]
    if y.shape[0] > MAX_PAIRS_N:
        y = y[rng.choice(y.shape[0], MAX_PAIRS_N, replace=False)]
    return float((x @ y.T).mean())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--embedder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(list(argv) if argv is not None else None)

    import pandas as pd

    import vtscore.projection.signpost_build as sb
    from toponymy.clustering import ToponymyClusterer

    matrix, ids, cats = load_matrix(args.dataset, args.embedder)
    n = matrix.shape[0]
    if n < sb._MIN_POINTS:
        common.log(f"{n} items is below signpost_build._MIN_POINTS; nothing to measure")
        return 0
    common.log(f"{args.dataset}/{args.embedder}: {n} items")

    clusterable = sb._clusterable_vectors(matrix)
    clusterer = ToponymyClusterer(min_clusters=4, base_min_cluster_size=sb._base_min_cluster_size(n), verbose=False)
    layers, _tree = clusterer.fit_predict(clusterable, matrix)
    common.log(f"  {len(layers)} layers")

    unit = _unit(matrix.astype(np.float64))
    rng = np.random.default_rng(args.seed)
    rows = []
    for li, layer in enumerate(layers):
        labels = np.asarray(layer.cluster_labels)
        ncl = int(labels.max()) + 1 if labels.size else 0
        for k in range(ncl):
            member = np.flatnonzero(labels == k)
            if member.size < 2:
                continue
            other = np.flatnonzero((labels != k) & (labels >= 0))
            within = _mean_within(unit[member], rng)
            between = _mean_between(unit[member], unit[other], rng)
            member_cats = [cats[i] for i in member]
            counts: dict[str, int] = {}
            for c in member_cats:
                for name in c:
                    counts[name] = counts.get(name, 0) + 1
            labelled = sum(1 for c in member_cats if c)
            purity = (max(counts.values()) / labelled) if counts and labelled else float("nan")
            rows.append(
                {
                    "dataset": args.dataset,
                    "embedder": args.embedder,
                    "seed": args.seed,
                    "layer": li,
                    "n_layers": len(layers),
                    "cluster": k,
                    "size": int(member.size),
                    "within_cosine": within,
                    "between_cosine": between,
                    "coherence_gap": within - between,
                    "gt_purity": purity,
                    "n_labelled": int(labelled),
                }
            )
        common.log(f"  layer {li}: {ncl} clusters, noise {float(np.mean(labels < 0)):.2f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"regions_{args.dataset}__{args.embedder}__s{args.seed}.csv", index=False)
    common.log(f"wrote {len(rows)} clusters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
