"""Tests for bin-popup member ordering (the 1-D Hilbert traversal).

A bin's member ids are served in a locality-preserving 1-D order derived from
the frozen 2-D layout (``_hilbert_order``), not by media id, so a dense bin's
contents group by region — and hiding items leaves the survivors' order intact.
"""

from __future__ import annotations

import numpy as np

from vtscore.projection import build_pyramid, rebin_like, remove_ids, tile_member_ids
from vtscore.projection.pyramid import _hilbert_order
from vtscore.projection.umap_projection import Projection


def _single_bin_members(pyr, proj) -> list[int]:
    """The member list of the one cell in a single-bin pyramid (all points)."""
    members: list[int] = []
    for lvl, tx, ty in pyr.tiles:
        for mem in tile_member_ids(pyr, proj, lvl, tx, ty).values():
            members.extend(mem)
    return members


def _interleaved_clusters(seed: int = 7):
    """Three far-apart, tight blobs with ids assigned independently of cluster.

    Row order (== id order, ids are ``range(N)``) interleaves the clusters, so a
    spatial regrouping is observable: id order scatters the clusters, the Hilbert
    order must put each back together.
    """
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [100.0, 0.0], [50.0, 80.0]])
    blocks = [c + rng.standard_normal((100, 2)) * 0.3 for c in centers]
    coords = np.concatenate(blocks).astype(np.float32)
    cluster_of_row = np.repeat([0, 1, 2], 100)
    perm = rng.permutation(coords.shape[0])  # interleave the rows
    coords = np.ascontiguousarray(coords[perm])
    cluster_of_row = cluster_of_row[perm]
    proj = Projection("hilbert-test", list(range(coords.shape[0])), coords, "test")
    return proj, cluster_of_row


def _transitions(labels: list[int]) -> int:
    return sum(1 for a, b in zip(labels, labels[1:]) if a != b)


def test_dense_bin_groups_clusters():
    """A bin holding three interleaved clusters lists each cluster contiguously."""
    proj, cluster_of_row = _interleaved_clusters()
    # base_radius huge -> every point lands in one cell.
    pyr = build_pyramid(proj, n_levels=1, base_radius=1000.0)
    members = _single_bin_members(pyr, proj)
    assert len(members) == len(proj.ids)

    # ids are row indices, so cluster_of_row indexes directly by member id.
    hilbert_labels = [int(cluster_of_row[m]) for m in members]
    # Ideal grouping of 3 clusters is 2 transitions; allow a couple for curve
    # seams. The point is it's *nothing like* the interleaved id order below.
    assert _transitions(hilbert_labels) <= 4

    id_labels = [int(cluster_of_row[m]) for m in sorted(members)]
    assert _transitions(id_labels) > 50  # id order scatters the clusters


def test_members_follow_global_hilbert_order():
    """Within every cell at every level, members are in global Hilbert order."""
    rng = np.random.default_rng(3)
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 8.0]])
    coords = np.concatenate([c + rng.standard_normal((100, 2)) * 0.5 for c in centers]).astype(np.float32)
    proj = Projection("rank-test", list(range(coords.shape[0])), coords, "test")
    pyr = build_pyramid(proj, n_levels=4)

    # Same coords the membership pass sees (float64), so the curve matches.
    perm = _hilbert_order(np.ascontiguousarray(coords, dtype=np.float64))
    rank = np.empty(perm.shape[0], dtype=np.int64)
    rank[perm] = np.arange(perm.shape[0])  # rank[id] = position in the order

    for lvl, tx, ty in pyr.tiles:
        for mem in tile_member_ids(pyr, proj, lvl, tx, ty).values():
            ranks = [int(rank[m]) for m in mem]
            assert ranks == sorted(ranks)


def test_hiding_items_preserves_relative_order():
    """Removing ids leaves the survivors in the same relative order (subsequence)."""
    proj, _ = _interleaved_clusters(seed=5)
    pyr = build_pyramid(proj, n_levels=1, base_radius=1000.0)
    full_order = _single_bin_members(pyr, proj)

    removed = set(proj.ids[::3])  # cull every third item
    reduced = remove_ids(proj, removed)
    reduced_pyr = rebin_like(reduced, pyr)
    reduced_order = _single_bin_members(reduced_pyr, reduced)

    survivors = [i for i in full_order if i not in removed]
    assert reduced_order == survivors


def test_hilbert_order_handles_degenerate_inputs():
    """Empty and all-coincident layouts order deterministically without error."""
    assert _hilbert_order(np.empty((0, 2), dtype=np.float64)).tolist() == []
    coincident = np.zeros((5, 2), dtype=np.float64)
    # All on one point: ties break by row index (stable), so order is identity.
    assert _hilbert_order(coincident).tolist() == [0, 1, 2, 3, 4]
