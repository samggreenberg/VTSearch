"""Stage 1.5 of the VTSBrowse browse canvas: close the empty oceans.

UMAP lays clusters out faithfully but scatters them across a wide canvas with
large empty regions ("oceans") between the islands of related media.  Those
oceans waste screen real estate: after zoom-to-fit the actual points occupy a
small fraction of the frame.  This module compacts a finished UMAP layout by
sliding the clusters together until they nearly touch, **without distorting any
cluster internally** — each cluster translates as a rigid body, so the local
neighbourhood structure UMAP discovered is preserved exactly.

The algorithm is a collision-aware force-directed pack:

1. Cluster the 2-D layout with HDBSCAN.  Each cluster becomes a rigid *unit*
   with a centre and a radius; stray noise points fold into their nearest
   cluster so they ride along with it (and the unit count stays bounded by the
   cluster count, not the point count — this is what lets it scale).
2. Relax the unit centres: a gentle pull toward the global centroid closes the
   gaps, while hard circle-repulsion stops units the instant their discs touch,
   so islands pack tightly but never overlap.
3. Translate every point by its unit's net displacement.

Because step 3 is a pure per-cluster translation, the disparity (Procrustes
distance) between a cluster before and after compaction is exactly zero: the
islands keep their shape and only the dead water between them disappears.

This module is Flask-free and imports scikit-learn lazily so importing the
package never pulls in the clustering backend until a compaction actually runs.
"""

from __future__ import annotations

import numpy as np

# Below this many points there is nothing meaningful to pack (the UMAP path only
# engages above ``min_n_for_umap`` anyway); compaction is a no-op under it.
_MIN_POINTS = 12

# If HDBSCAN ever discovers more clusters than this, the O(U^2) relaxation would
# get expensive; we skip compaction rather than stall the projection build.  In
# practice density clustering yields far fewer units than this on real data.
_MAX_UNITS = 3000


def _cluster(coords: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """HDBSCAN labels for *coords* (``-1`` marks noise).  Deterministic."""
    from sklearn.cluster import HDBSCAN

    size = max(2, min(min_cluster_size, coords.shape[0]))
    return np.asarray(HDBSCAN(min_cluster_size=size).fit_predict(coords))


def _build_units(
    coords: np.ndarray, labels: np.ndarray
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Group points into rigid units (one per cluster, noise folded into nearest).

    Returns ``(point_index_lists, centres, radii)`` where ``centres[u]`` and
    ``radii[u]`` are the bounding circle of unit ``u``.  The circle is fitted to
    the cluster's **core** (HDBSCAN-assigned) members only — the 90th percentile
    of their distance to the centroid — so a few scattered noise points folded in
    for translation don't balloon the radius and loosen the pack.  Noise *does*
    join ``point_index_lists`` so it rides along with its island's translation;
    it just doesn't get a vote on how much space the island claims.  The radius is
    floored at a small fraction of the median so a tight single-cell cluster still
    claims room.
    """
    cluster_ids = sorted(int(k) for k in np.unique(labels) if k >= 0)
    centroids = np.stack([coords[labels == k].mean(axis=0) for k in cluster_ids])

    # Bounding circle from core members only (noise excluded).
    centres, radii = [], []
    for k, c in zip(cluster_ids, centroids):
        core = coords[labels == k]
        radii.append(float(np.percentile(np.linalg.norm(core - c, axis=1), 90)))
        centres.append(c)

    # Fold each noise point into the nearest cluster so it translates with that
    # island instead of becoming its own unit (keeps the unit count bounded by
    # cluster count, which is what lets the O(U^2) relaxation scale).
    members = {k: list(np.where(labels == k)[0]) for k in cluster_ids}
    noise_idx = np.where(labels < 0)[0]
    if noise_idx.size:
        d = np.linalg.norm(coords[noise_idx, None, :] - centroids[None, :, :], axis=2)
        nearest = np.argmin(d, axis=1)
        for pt, slot in zip(noise_idx, nearest):
            members[cluster_ids[int(slot)]].append(int(pt))

    point_lists = [np.array(sorted(members[k])) for k in cluster_ids]
    radii_arr = np.array(radii, dtype=np.float64)
    floor = 0.15 * float(np.median(radii_arr)) if radii_arr.size else 0.0
    radii_arr = np.maximum(radii_arr, floor)
    return point_lists, np.array(centres, dtype=np.float64), radii_arr


def _relax(
    centres: np.ndarray,
    radii: np.ndarray,
    *,
    target: np.ndarray,
    margin: float,
    attract: float,
    iters: int,
) -> np.ndarray:
    """Force-directed pack: pull centres toward *target*, stop them at contact.

    Returns the settled centres.  The main loop interleaves a gentle attraction
    with hard non-overlap repulsion; a final attraction-free pass cleans up any
    residual overlap so units end just-touching rather than intersecting.
    """
    from scipy.spatial.distance import cdist

    centres = centres.copy()
    rsum = radii[:, None] + radii[None, :] + margin
    np.fill_diagonal(rsum, 0.0)

    def _resolve(disp: np.ndarray) -> np.ndarray:
        d = cdist(centres, centres)
        np.fill_diagonal(d, np.inf)
        overlap = rsum - d
        ii, jj = np.where(overlap > 0)
        if ii.size:
            delta = centres[ii] - centres[jj]
            dist = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-9
            push = (overlap[ii, jj][:, None] * 0.5) * (delta / dist)
            np.add.at(disp, ii, push)
        return disp

    for _ in range(iters):
        centres += _resolve(attract * (target - centres))
    for _ in range(max(1, iters // 8)):  # final overlap-only cleanup
        centres += _resolve(np.zeros_like(centres))
    return centres


def compact_layout(
    coords: np.ndarray,
    *,
    min_cluster_size: int = 15,
    margin_frac: float = 0.15,
    attract: float = 0.02,
    iters: int = 400,
) -> np.ndarray:
    """Return a compacted copy of *coords* with the inter-cluster oceans closed.

    *coords* is an ``(N, 2)`` UMAP layout.  Clusters are slid together as rigid
    bodies until their bounding discs nearly touch (``margin_frac`` of the
    median radius of breathing room between them), so the big empty gaps vanish
    while every cluster keeps its exact internal shape.  When there is nothing to
    pack — too few points, or fewer than two clusters — the input is returned
    unchanged (as a copy).
    """
    coords = np.ascontiguousarray(coords, dtype=np.float32)
    if coords.shape[0] < _MIN_POINTS:
        return coords.copy()

    labels = _cluster(coords, min_cluster_size)
    n_clusters = int((np.unique(labels) >= 0).sum())
    if n_clusters < 2 or n_clusters > _MAX_UNITS:
        # One island (no oceans to close), all noise, or pathologically many
        # units: leave the layout as UMAP produced it.
        return coords.copy()

    point_lists, centres, radii = _build_units(coords, labels)
    target = coords.mean(axis=0).astype(np.float64)
    margin = margin_frac * float(np.median(radii))
    settled = _relax(
        centres, radii, target=target, margin=margin, attract=attract, iters=iters
    )

    out = coords.copy()
    shifts = (settled - centres).astype(np.float32)
    for idx, shift in zip(point_lists, shifts):
        out[idx] += shift
    return out
