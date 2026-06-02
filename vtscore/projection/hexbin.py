"""Vectorized d3-hexbin binning over a 2-D point cloud (no d3 dependency).

VTSBrowse renders the UMAP projection as a field of hexagons whose color
encodes density.  To do that the server has to assign every projected point to
a hex cell.  This module reimplements the `d3-hexbin
<https://github.com/d3/d3-hexbin>`_ assignment algorithm in NumPy so a whole
``(N, 2)`` array is binned in one vectorized pass.

The lattice is anchored in *projection (data) space*, parameterized by a single
hexagon *radius* ``r`` (center → vertex).  Pointy-top hexagons tile with column
spacing ``dx = r·√3`` and row spacing ``dy = 1.5·r``; odd rows are offset by
half a column.  Anchoring in data space is what makes pure panning free in the
renderer (membership only changes when the radius changes, i.e. across a zoom
level), per *§Browse-canvas architecture* in
``docs/design/vtsbrowse.md``.

A bin is identified by an integer pair ``(q, r)`` where ``q = round(2·pi)``
(d3's column index ``pi`` can be a half-integer after the edge correction, so we
double it to stay integral) and ``r = pj`` is the row index.  :func:`hex_center`
inverts that key back to the cell's center in projection space.
"""

from __future__ import annotations

import math

import numpy as np

# 2·sin(π/3): the column spacing of a pointy-top hex lattice is ``radius · SQRT3``.
SQRT3 = math.sqrt(3.0)


def hexbin_assign(points: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Assign each point in *points* to a hex cell of the given *radius*.

    *points* is an ``(M, 2)`` float array of projection-space coordinates.
    Returns ``(q, r)``: two int64 arrays of length ``M`` holding the integer
    cell key for each point, where ``q = round(2·pi)`` (doubled column index)
    and ``r = pj`` (row index).  Feed the same ``(q, r)`` to :func:`hex_center`
    to recover the cell center.

    This is the d3-hexbin rounding rule, vectorized: round to the nearest row,
    then the nearest (row-offset) column, and where the point lands in the outer
    third of a row, compare against the staggered neighbor and snap to whichever
    center is closer.
    """
    if radius <= 0:
        raise ValueError(f"hex radius must be positive, got {radius!r}")
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be (M, 2), got shape {pts.shape}")

    dx = radius * SQRT3
    dy = radius * 1.5

    px0 = pts[:, 0] / dx
    py0 = pts[:, 1] / dy

    pj = np.round(py0)
    pj_odd = (pj.astype(np.int64) & 1) == 1
    px = px0 - np.where(pj_odd, 0.5, 0.0)
    pi = np.round(px)
    py1 = py0 - pj

    # Edge correction: only points more than a third of a row away from the
    # row center can be closer to a staggered neighbor's center.
    needs_check = np.abs(py1) * 3.0 > 1.0
    px1 = px - pi
    pi2 = pi + np.where(px < pi, -0.5, 0.5)
    pj2 = pj + np.where(py0 < pj, -1.0, 1.0)
    px2 = px - pi2
    py2 = py0 - pj2
    closer_to_neighbor = (px1 * px1 + py1 * py1) > (px2 * px2 + py2 * py2)
    swap = needs_check & closer_to_neighbor

    # When we snap to the neighbor the column shifts by ±0.5 depending on the
    # parity of the *original* row (d3's ``(pj & 1 ? 1 : -1) / 2``).
    pi_corrected = pi2 + np.where(pj_odd, 0.5, -0.5)
    pi_final = np.where(swap, pi_corrected, pi)
    pj_final = np.where(swap, pj2, pj)

    q = np.round(pi_final * 2.0).astype(np.int64)
    r = pj_final.astype(np.int64)
    return q, r


def hex_center(q: np.ndarray | int, r: np.ndarray | int, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`hexbin_assign`: the projection-space center of cell ``(q, r)``.

    Accepts scalars or arrays.  Returns ``(cx, cy)`` as float64 (scalars in,
    scalars out via 0-d arrays' ``.item()``-friendly values).
    """
    dx = radius * SQRT3
    dy = radius * 1.5
    q_arr = np.asarray(q, dtype=np.float64)
    r_arr = np.asarray(r, dtype=np.float64)
    pi = q_arr / 2.0
    r_odd = (r_arr.astype(np.int64) & 1) == 1
    cx = (pi + np.where(r_odd, 0.5, 0.0)) * dx
    cy = r_arr * dy
    return cx, cy
