"""Tests for the vectorized square-grid binning (``vtscore.projection.squarebin``)."""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.projection.squarebin import SQUARE_SIDE_PER_RADIUS, square_center, squarebin_assign


def test_assign_shapes_and_dtype():
    rng = np.random.default_rng(42)
    pts = rng.standard_normal((200, 2))
    q, r = squarebin_assign(pts, radius=0.3)
    assert q.shape == (200,)
    assert r.shape == (200,)
    assert q.dtype == np.int64
    assert r.dtype == np.int64


def test_center_round_trips_to_same_cell():
    # A point placed at a cell's center must bin back to that cell.
    rng = np.random.default_rng(1)
    pts = rng.standard_normal((500, 2)) * 5.0
    radius = 0.7
    q, r = squarebin_assign(pts, radius)
    cx, cy = square_center(q, r, radius)
    centers = np.stack([cx, cy], axis=1)
    q2, r2 = squarebin_assign(centers, radius)
    assert np.array_equal(q, q2)
    assert np.array_equal(r, r2)


def test_origin_maps_to_a_cell_at_origin_center():
    q, r = squarebin_assign(np.array([[0.0, 0.0]]), radius=1.0)
    cx, cy = square_center(q, r, radius=1.0)
    assert float(cx[0]) == pytest.approx(0.0, abs=1e-9)
    assert float(cy[0]) == pytest.approx(0.0, abs=1e-9)


def test_keys_are_plain_column_row_indices():
    # Unlike the hex key, square q/r are the bare (column, row) lattice indices:
    # a point one full cell to the right lands one column over.
    radius = 1.0
    side = radius * SQUARE_SIDE_PER_RADIUS
    q0, r0 = squarebin_assign(np.array([[0.0, 0.0]]), radius)
    q1, r1 = squarebin_assign(np.array([[side, 0.0]]), radius)
    assert int(q1[0]) - int(q0[0]) == 1
    assert int(r1[0]) == int(r0[0])


def test_cell_partitions_into_a_square():
    # Points within half a side of a center in both axes share its cell; a point
    # just past the half-side boundary falls into the neighbor.
    radius = 2.0
    side = radius * SQUARE_SIDE_PER_RADIUS
    inside = np.array([[0.49 * side, -0.49 * side]])
    outside = np.array([[0.51 * side, 0.0]])
    qi, ri = squarebin_assign(inside, radius)
    qo, ro = squarebin_assign(outside, radius)
    assert (int(qi[0]), int(ri[0])) == (0, 0)
    assert int(qo[0]) == 1 and int(ro[0]) == 0


def test_smaller_radius_yields_more_distinct_cells():
    rng = np.random.default_rng(7)
    pts = rng.standard_normal((1000, 2)) * 3.0
    coarse_q, coarse_r = squarebin_assign(pts, radius=1.0)
    fine_q, fine_r = squarebin_assign(pts, radius=0.25)
    n_coarse = len(np.unique(np.stack([coarse_q, coarse_r], axis=1), axis=0))
    n_fine = len(np.unique(np.stack([fine_q, fine_r], axis=1), axis=0))
    assert n_fine > n_coarse


def test_invalid_radius_raises():
    with pytest.raises(ValueError):
        squarebin_assign(np.zeros((3, 2)), radius=0.0)


def test_invalid_shape_raises():
    with pytest.raises(ValueError):
        squarebin_assign(np.zeros((3, 3)), radius=1.0)
