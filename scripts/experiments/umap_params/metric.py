"""Layout-quality metrics for the VTSBrowse UMAP parameter sweep.

The primary metric is **ceiling-normalized taxonomy separability** (plan
§Metric): for a dataset with a class taxonomy, how cleanly do the labeled
subsets occupy locally-coherent regions of the 2-D layout, *relative to* how
cleanly they occupy the original high-D embedding space. Normalizing by the
high-D ceiling isolates what the UMAP parameters actually control from the
embedder's own (in)ability to separate a class.

All metrics are label-position/scale invariant and multi-island tolerant: a
class may form several clean blobs without penalty (the AUROC of a local
neighbor-fraction, not a single-centroid measure).

Pure NumPy / scikit-learn; no Flask, no VTSearch imports — so it runs anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


def knn_indices(X: np.ndarray, k: int, metric: str = "euclidean") -> np.ndarray:
    """Row i = indices of the k nearest neighbors of point i (self excluded).

    ``metric="cosine"`` is used for the high-D embedding space; the 2-D layout
    uses euclidean. We over-fetch one neighbor and drop self so co-located
    duplicates never make a point its own neighbor.
    """
    n = X.shape[0]
    k_eff = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric)
    nn.fit(X)
    idx = nn.kneighbors(X, return_distance=False)  # (n, k_eff+1), self usually col 0
    # Vectorized self-removal: drop each row's own index, keep the first k_eff of
    # the rest. A boolean mask over the (k_eff+1) fetched neighbors is exact and
    # O(n·k) — no Python per-row loop, which matters at N≈20k across a whole grid.
    rows = np.arange(n)[:, None]
    keep = idx != rows  # False exactly where a neighbor == self
    # For rows where self wasn't among the fetched neighbors (co-located dupes),
    # keep loses nothing; force exactly k_eff by taking the first k_eff True cols.
    out = np.empty((n, k_eff), dtype=np.int64)
    for_flat = idx[keep].reshape(-1)  # rows with self dropped are length k_eff...
    counts = keep.sum(axis=1)
    if np.all(counts == k_eff):
        out = for_flat.reshape(n, k_eff)
    else:  # rare: some rows had no self (dupes) → keep first k_eff columns
        for i in range(n):
            r = idx[i][keep[i]]
            out[i] = r[:k_eff] if r.shape[0] >= k_eff else idx[i][:k_eff]
    return out


def _node_auroc(knn_idx: np.ndarray, mask: np.ndarray) -> float | None:
    """AUROC of "fraction of my neighbors that are in-class" vs actual membership.

    High AUROC ⇔ members sit among members and non-members among non-members at
    the k scale: a clean boundary exists. Returns None for a degenerate node
    (all-in or all-out — AUROC undefined).
    """
    pos = int(mask.sum())
    if pos == 0 or pos == mask.shape[0]:
        return None
    frac = mask[knn_idx].mean(axis=1)  # per-point in-class neighbor fraction
    return float(roc_auc_score(mask.astype(np.int8), frac))


@dataclass
class SeparabilityResult:
    """Taxonomy-separability outcome for one layout, against one high-D ceiling."""

    score_2d: float  # level-averaged mean node AUROC on the 2-D layout
    score_highd: float  # same, on the original embedding space (the ceiling)
    ratio: float  # score_2d / score_highd — the quantity UMAP params control
    per_level_2d: dict[str, float] = field(default_factory=dict)
    per_level_highd: dict[str, float] = field(default_factory=dict)
    n_nodes_scored: int = 0


def _aggregate_levels(per_node: dict[str, list[float]]) -> tuple[float, dict[str, float]]:
    """Mean over nodes within a level, then average the levels (plan §Metric.3).

    Averaging per level first keeps a coarse 2-node level (animal/not) from
    being swamped by a 300-leaf level.
    """
    level_means = {lvl: float(np.mean(v)) for lvl, v in per_node.items() if v}
    overall = float(np.mean(list(level_means.values()))) if level_means else float("nan")
    return overall, level_means


def taxonomy_separability(
    coords2d: np.ndarray,
    highd: np.ndarray,
    taxonomy: dict[str, list[np.ndarray]],
    *,
    k: int = 20,
    highd_knn: np.ndarray | None = None,
) -> SeparabilityResult:
    """Ceiling-normalized taxonomy separability of *coords2d*.

    ``taxonomy`` maps a level name (e.g. ``"class"``) to a list of boolean
    member-masks, one per node at that level (one-vs-rest). ``highd`` is the
    original ``(N, d)`` embedding matrix; its kNN graph (cosine) is the ceiling
    and can be precomputed once per dataset via ``highd_knn`` (it never changes
    across the param sweep).
    """
    knn2d = knn_indices(coords2d, k, metric="euclidean")
    knnhd = highd_knn if highd_knn is not None else knn_indices(highd, k, metric="cosine")

    per_node_2d: dict[str, list[float]] = {}
    per_node_hd: dict[str, list[float]] = {}
    n_scored = 0
    for level, masks in taxonomy.items():
        for mask in masks:
            a2 = _node_auroc(knn2d, mask)
            ah = _node_auroc(knnhd, mask)
            if a2 is None or ah is None:
                continue
            per_node_2d.setdefault(level, []).append(a2)
            per_node_hd.setdefault(level, []).append(ah)
            n_scored += 1

    s2, pl2 = _aggregate_levels(per_node_2d)
    sh, plh = _aggregate_levels(per_node_hd)
    ratio = s2 / sh if sh and not np.isnan(sh) else float("nan")
    return SeparabilityResult(s2, sh, ratio, pl2, plh, n_scored)


def knn_recall(coords2d: np.ndarray, highd: np.ndarray, k: int = 20, highd_knn: np.ndarray | None = None) -> float:
    """Mean overlap between each point's 2-D and high-D k-neighbor sets (label-free).

    A layout that shatters the space to fake class purity scores badly here:
    real neighbors get scattered, so 2-D and high-D neighbor sets diverge.
    1.0 = perfectly preserved local neighborhoods; ~k/N = random.
    """
    knn2d = knn_indices(coords2d, k, metric="euclidean")
    knnhd = highd_knn if highd_knn is not None else knn_indices(highd, k, metric="cosine")
    overlaps = [np.intersect1d(knn2d[i], knnhd[i], assume_unique=False).size for i in range(coords2d.shape[0])]
    return float(np.mean(overlaps) / knn2d.shape[1])


def structure_guards(
    coords2d: np.ndarray, highd: np.ndarray, *, k: int = 20, highd_knn: np.ndarray | None = None
) -> dict[str, float]:
    """Label-free structure-preservation guards: trustworthiness, continuity, recall.

    - **Trustworthiness** ∈ [0,1]: penalizes 2-D neighbors that were *not* close
      in high-D (false neighbors / intrusions). Up is good.
    - **Continuity** ∈ [0,1]: the dual — penalizes high-D neighbors torn apart in
      2-D (missing neighbors). Up is good. Computed as trustworthiness with the
      two spaces swapped.
    - **kNN-recall@k**: see :func:`knn_recall`.
    """
    from sklearn.manifold import trustworthiness

    n = coords2d.shape[0]
    k_eff = min(k, (n - 1) // 2)  # trustworthiness requires k < N/2
    trust = float(trustworthiness(highd, coords2d, n_neighbors=k_eff, metric="cosine"))
    cont = float(trustworthiness(coords2d, highd, n_neighbors=k_eff, metric="euclidean"))
    return {
        "trustworthiness": trust,
        "continuity": cont,
        "knn_recall": knn_recall(coords2d, highd, k=k, highd_knn=highd_knn),
    }


def structure_guards_subsampled(
    coords2d: np.ndarray, highd: np.ndarray, *, k: int = 20, cap: int = 2000, seed: int = 0
) -> dict[str, float]:
    """:func:`structure_guards` on a deterministic subsample (trustworthiness is O(N²)).

    Trustworthiness/continuity compute full pairwise ranks, so they are
    infeasible at N=20k across a whole grid. A fixed random subsample of ``cap``
    points gives a stable estimate cheaply; below ``cap`` the full set is used.
    """
    n = coords2d.shape[0]
    if n <= cap:
        return structure_guards(coords2d, highd, k=k)
    rng = np.random.default_rng(seed)
    sub = rng.choice(n, size=cap, replace=False)
    return structure_guards(coords2d[sub], highd[sub], k=k)


def layout_seed_agreement(layouts: list[np.ndarray], k: int = 20) -> float:
    """Mean pairwise neighbor-set overlap across seeded layouts (stability).

    Each ``layouts[s]`` is an ``(N,2)`` layout from a different seed (same
    params). Returns the mean, over all seed pairs and points, of the fraction
    of shared k-neighbors — how much the unseeded production fit will wobble.
    1.0 = seed-invariant; low = knife-edge.
    """
    if len(layouts) < 2:
        return float("nan")
    knns = [knn_indices(L, k, metric="euclidean") for L in layouts]
    n = layouts[0].shape[0]
    kk = knns[0].shape[1]
    agrees: list[float] = []
    for a in range(len(knns)):
        for b in range(a + 1, len(knns)):
            ov = [np.intersect1d(knns[a][i], knns[b][i]).size for i in range(n)]
            agrees.append(np.mean(ov) / kk)
    return float(np.mean(agrees))
