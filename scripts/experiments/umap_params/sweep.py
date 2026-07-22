"""The UMAP parameter sweep: fit → score → emit rows, over cached matrices.

For one prepared ``(dataset, embedder)`` matrix, sweep the primary grid
(``n_neighbors`` × ``min_dist``) with 3 seeds, and score every fit **twice** —
raw UMAP coordinates and post-``compact_layout`` coordinates — so the
``compact`` boolean is evaluated on the same fit (no extra UMAP runs). Emits one
CSV row per (params, seed, compact) cell with the ceiling-normalized taxonomy
separability, the label-free structure guards, inter-seed stability, and fit
seconds.

Run on a GPU node:  python sweep.py <dataset>__<embedder>  [more...]
                    python sweep.py --all

The high-D kNN "ceiling" graph is invariant across the whole grid, so it is
computed once per matrix on the GPU (a normalized matmul + top-k — the O(N²·d)
step that would otherwise dominate) and reused for every cell.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

import common as C
import metric as M


def gpu_highd_knn(X: np.ndarray, k: int) -> np.ndarray:
    """k nearest neighbors (cosine, self-excluded) of each row of X, on GPU.

    Embeddings are L2-normalized, so cosine similarity is a plain matmul; we
    top-k the similarity in row chunks to bound memory. Falls back to the CPU
    sklearn path if torch/CUDA is unavailable.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("no cuda")
        dev = "cuda"
        Xt = torch.tensor(X, dtype=torch.float32, device=dev)
        Xt = Xt / Xt.norm(dim=1, keepdim=True).clamp_min(1e-12)
        n = Xt.shape[0]
        out = np.empty((n, min(k, n - 1)), dtype=np.int64)
        chunk = 2048
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            sims = Xt[s:e] @ Xt.T  # (chunk, n)
            # exclude self by setting the diagonal to -inf
            for r in range(s, e):
                sims[r - s, r] = -1e30
            top = torch.topk(sims, k=min(k, n - 1), dim=1).indices
            out[s:e] = top.cpu().numpy()
        del Xt
        torch.cuda.empty_cache()
        return out
    except Exception as exc:  # pragma: no cover - fallback
        print(f"  [gpu_knn fallback → sklearn: {exc}]")
        return M.knn_indices(X, k, metric="cosine")


FIELDS = [
    "dataset", "embedder", "N", "dim", "n_neighbors", "min_dist", "seed", "compact",
    "fit_seconds", "score_2d", "score_highd", "ratio", "trustworthiness", "continuity",
    "knn_recall", "seed_agreement", "n_nodes",
]


def sweep_matrix(tag: str):
    """tag = '<dataset>__<embedder>'."""
    npz = C.MATRIX_DIR / f"{tag}.npz"
    d = np.load(npz, allow_pickle=True)
    ids = d["ids"]
    X = d["X"].astype(np.float32)
    leaf = d["leaf"].astype(str)
    lineage = {}
    if "lineage" in d:
        # per-item lineage rows → map leaf label → lineage (species dir is the leaf)
        lin_rows = d["lineage"]
        lineage = {leaf[i]: list(map(str, lin_rows[i])) for i in range(len(leaf))}
    dataset, embedder = tag.split("__", 1)
    n, dim = X.shape
    print(f"[sweep] {tag}: N={n} dim={dim}")

    spec = C.ROSTER_BY_NAME[dataset]
    build = C.TAXONOMY_BUILDERS[spec.taxonomy]
    kw = dict(lineage=lineage)
    if "ml_labels" in d:  # FSD50K multi-label
        kw.update(ml_labels=d["ml_labels"], ml_names=d["ml_names"], ml_isroot=d["ml_isroot"])
    taxonomy = build(leaf, **kw)
    n_nodes = sum(len(v) for v in taxonomy.values())
    print(f"  taxonomy levels: { {k: len(v) for k, v in taxonomy.items()} } ({n_nodes} nodes)")
    if n_nodes == 0:
        print("  no scorable taxonomy nodes — skipping")
        return

    from vtscore.projection.compaction import compact_layout
    from vtscore.projection.umap_projection import fit_projection

    k = C.METRIC_K
    highd_knn = gpu_highd_knn(X, k)
    id_list = list(map(int, ids))

    out_csv = C.CSV_DIR / f"{tag}.csv"
    C.CSV_DIR.mkdir(parents=True, exist_ok=True)
    f = open(out_csv, "w", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()

    for nn in C.N_NEIGHBORS_GRID:
        if nn >= n:  # n_neighbors clamped to N-1 anyway; skip redundant top
            continue
        for md in C.MIN_DIST_GRID:
            raw_layouts = []
            rows_buffer = []
            for seed in C.SEEDS:
                t0 = time.time()
                proj = fit_projection(
                    X, id_list, n_neighbors=nn, min_dist=md, random_state=seed, compact=False
                )
                fit_s = time.time() - t0
                raw = np.ascontiguousarray(proj.coords, dtype=np.float32)
                raw_layouts.append(raw)
                compacted = compact_layout(raw)
                for compact_flag, coords in [(False, raw), (True, compacted)]:
                    sep = M.taxonomy_separability(coords, X, taxonomy, k=k, highd_knn=highd_knn)
                    guards = M.structure_guards_subsampled(coords, X, k=k, cap=2000, seed=0)
                    rows_buffer.append(dict(
                        dataset=dataset, embedder=embedder, N=n, dim=dim,
                        n_neighbors=nn, min_dist=md, seed=seed, compact=compact_flag,
                        fit_seconds=round(fit_s, 3), score_2d=round(sep.score_2d, 5),
                        score_highd=round(sep.score_highd, 5), ratio=round(sep.ratio, 5),
                        trustworthiness=round(guards["trustworthiness"], 5),
                        continuity=round(guards["continuity"], 5),
                        knn_recall=round(guards["knn_recall"], 5),
                        seed_agreement=None, n_nodes=sep.n_nodes_scored,
                    ))
            agree = M.layout_seed_agreement(raw_layouts, k=k)
            for r in rows_buffer:
                r["seed_agreement"] = round(agree, 5)
                w.writerow(r)
            f.flush()
            print(f"  nn={nn:3d} md={md:.2f}  ratio≈{rows_buffer[-1]['ratio']:.3f}  agree={agree:.3f}")
    f.close()
    print(f"  wrote {out_csv}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--all" in args:
        tags = sorted(p.stem for p in C.MATRIX_DIR.glob("*.npz"))
    else:
        tags = args
    for tag in tags:
        sweep_matrix(tag)
