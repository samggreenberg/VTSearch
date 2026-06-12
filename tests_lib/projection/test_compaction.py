"""Tests for Stage-1.5 layout compaction (``vtscore.projection.compaction``).

These exercise the collision-aware packer that slides UMAP clusters together to
close the empty oceans between them.  The defining guarantees are: it preserves
point count/identity order, it never distorts a cluster internally (rigid
translation only), and it actually shrinks the layout's empty space.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.projection.compaction import _MIN_POINTS, compact_layout


def _islands(n_per: int = 30, spread: float = 0.4, gap: float = 40.0, seed: int = 0):
    """Well-separated Gaussian blobs with wide oceans between them.

    Returns ``(coords, labels)`` where ``labels`` is the ground-truth blob of
    each row (so tests can check per-blob invariants independent of how the
    packer's own HDBSCAN happens to split things).
    """
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [gap, 0.0], [0.0, gap], [gap, gap]], dtype=np.float32)
    blocks, labels = [], []
    for k, c in enumerate(centres):
        blocks.append(c + spread * rng.standard_normal((n_per, 2)).astype(np.float32))
        labels.append(np.full(n_per, k))
    return np.concatenate(blocks).astype(np.float32), np.concatenate(labels)


def _bbox_area(c: np.ndarray) -> float:
    span = c.max(axis=0) - c.min(axis=0)
    return float(span[0] * span[1])


def test_preserves_count_dtype_and_finiteness():
    coords, _ = _islands(seed=1)
    out = compact_layout(coords, min_cluster_size=10)
    assert out.shape == coords.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_returns_independent_copy():
    coords, _ = _islands(seed=2)
    out = compact_layout(coords, min_cluster_size=10)
    out[:] = 0.0
    assert not np.allclose(coords, 0.0)  # input untouched


def test_closes_the_oceans():
    # Four far-apart blobs: packing must pull them together, shrinking the
    # bounding box substantially while keeping every point finite.
    coords, _ = _islands(gap=50.0, seed=3)
    out = compact_layout(coords, min_cluster_size=10)
    assert _bbox_area(out) < 0.5 * _bbox_area(coords)


def test_clusters_translate_rigidly_no_internal_distortion():
    # The headline guarantee: each well-separated blob moves as a rigid body, so
    # its points' offsets from the blob centroid are unchanged to float precision.
    coords, labels = _islands(seed=4)
    out = compact_layout(coords, min_cluster_size=10)
    for k in np.unique(labels):
        idx = labels == k
        before = coords[idx] - coords[idx].mean(axis=0)
        after = out[idx] - out[idx].mean(axis=0)
        np.testing.assert_allclose(before, after, atol=1e-4)


def test_clusters_do_not_overlap_after_packing():
    # Collision repulsion must keep blob centroids at least roughly a blob-extent
    # apart — packing tightly without piling islands on top of each other.
    coords, labels = _islands(spread=0.4, seed=5)
    out = compact_layout(coords, min_cluster_size=10)
    cents = np.stack([out[labels == k].mean(axis=0) for k in np.unique(labels)])
    dmin = min(float(np.linalg.norm(cents[i] - cents[j])) for i in range(len(cents)) for j in range(i + 1, len(cents)))
    # Each blob's own extent is ~spread; centroids must stay well separated.
    assert dmin > 1.0


def test_deterministic():
    coords, _ = _islands(seed=6)
    a = compact_layout(coords, min_cluster_size=10)
    b = compact_layout(coords, min_cluster_size=10)
    np.testing.assert_array_equal(a, b)


def test_noop_below_min_points():
    rng = np.random.default_rng(7)
    coords = rng.standard_normal((_MIN_POINTS - 1, 2)).astype(np.float32)
    out = compact_layout(coords)
    np.testing.assert_array_equal(out, coords)


def test_noop_single_cluster():
    # One tight blob: fewer than two clusters means no oceans to close, so the
    # layout is returned unchanged.
    rng = np.random.default_rng(8)
    coords = (0.3 * rng.standard_normal((60, 2))).astype(np.float32)
    out = compact_layout(coords, min_cluster_size=10)
    np.testing.assert_array_equal(out, coords)


def test_noop_all_identical_points():
    coords = np.zeros((20, 2), dtype=np.float32)
    out = compact_layout(coords)
    np.testing.assert_array_equal(out, coords)


@pytest.mark.parametrize("margin", [0.02, 0.15, 0.4])
def test_margin_frac_stays_valid(margin):
    coords, _ = _islands(seed=9)
    out = compact_layout(coords, min_cluster_size=10, margin_frac=margin)
    assert out.shape == coords.shape
    assert np.isfinite(out).all()
