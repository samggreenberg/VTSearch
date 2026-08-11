"""Tests for the vectorized d3-hexbin binning (``vtscore.projection.hexbin``)."""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.projection.hexbin import hex_center, hexbin_assign


def test_assign_shapes_and_dtype():
    rng = np.random.default_rng(42)
    pts = rng.standard_normal((200, 2))
    q, r = hexbin_assign(pts, radius=0.3)
    assert q.shape == (200,)
    assert r.shape == (200,)
    assert q.dtype == np.int64
    assert r.dtype == np.int64


def test_center_round_trips_to_same_cell():
    # A point placed at a cell's center must bin back to that cell.
    rng = np.random.default_rng(1)
    pts = rng.standard_normal((500, 2)) * 5.0
    radius = 0.7
    q, r = hexbin_assign(pts, radius)
    cx, cy = hex_center(q, r, radius)
    centers = np.stack([cx, cy], axis=1)
    q2, r2 = hexbin_assign(centers, radius)
    assert np.array_equal(q, q2)
    assert np.array_equal(r, r2)


def test_origin_maps_to_a_cell_at_origin_center():
    q, r = hexbin_assign(np.array([[0.0, 0.0]]), radius=1.0)
    cx, cy = hex_center(q, r, radius=1.0)
    assert float(cx[0]) == pytest.approx(0.0, abs=1e-9)
    assert float(cy[0]) == pytest.approx(0.0, abs=1e-9)


def test_nearby_points_share_a_cell():
    radius = 2.0
    # Two points well within one hex of each other collapse to one cell.
    pts = np.array([[0.05, 0.05], [-0.05, 0.1]])
    q, r = hexbin_assign(pts, radius)
    assert q[0] == q[1] and r[0] == r[1]


def test_smaller_radius_yields_more_distinct_cells():
    rng = np.random.default_rng(7)
    pts = rng.standard_normal((1000, 2)) * 3.0
    coarse_q, coarse_r = hexbin_assign(pts, radius=1.0)
    fine_q, fine_r = hexbin_assign(pts, radius=0.25)
    n_coarse = len(np.unique(np.stack([coarse_q, coarse_r], axis=1), axis=0))
    n_fine = len(np.unique(np.stack([fine_q, fine_r], axis=1), axis=0))
    assert n_fine > n_coarse


@pytest.mark.parametrize("radius", [0.37, 1.0, 2.5])
def test_assignment_is_the_exact_nearest_center_partition(radius):
    """Every point must land in the hexagon that actually contains it.

    Pins the deliberate divergence from d3-hexbin: d3 compares the
    near-boundary tie-break in per-axis-normalized units (no 0.75 y-weight),
    which misassigns ~2% of boundary points to a neighbor whose center is
    farther away.  Assignment must equal nearest-center over the lattice.
    """
    rng = np.random.default_rng(0)
    pts = rng.uniform(-10.0, 10.0, size=(20_000, 2))
    q, r = hexbin_assign(pts, radius)

    cx, cy = hex_center(q, r, radius)
    assigned_dist = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)

    # Brute-force nearest center over the neighborhood of the assigned cell.
    # Only even ``q`` are real lattice columns (``hex_center`` folds the odd-row
    # half-column offset in itself), so step the column index by two.
    nearest_dist = np.full(len(pts), np.inf)
    for d_row in (-2, -1, 0, 1, 2):
        for d_col in (-4, -2, 0, 2, 4):
            ncx, ncy = hex_center(q + d_col, r + d_row, radius)
            nearest_dist = np.minimum(nearest_dist, np.hypot(pts[:, 0] - ncx, pts[:, 1] - ncy))

    assert np.all(assigned_dist <= nearest_dist + 1e-9)


def test_invalid_radius_raises():
    with pytest.raises(ValueError):
        hexbin_assign(np.zeros((3, 2)), radius=0.0)


def test_invalid_shape_raises():
    with pytest.raises(ValueError):
        hexbin_assign(np.zeros((3, 3)), radius=1.0)
