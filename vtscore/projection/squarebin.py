"""Vectorized square-grid binning over a 2-D point cloud.

VTSBrowse can render the UMAP projection either as a field of hexagons (see
:mod:`vtscore.projection.hexbin`) or as a plain square grid, for users who
prefer rectangular tiles.  This module is the square counterpart of
``hexbin``: it assigns every projected point to a square cell and inverts the
cell key back to a center, so :mod:`vtscore.projection.pyramid` can build a
pyramid over either lattice through one uniform interface.

The lattice is anchored in *projection (data) space*, parameterized by the
same per-level ``radius`` scale the hex lattice uses.  A square cell has side
``s = radius·√3`` — chosen to equal the hex lattice's column spacing so a
square and a hexagon at the same pyramid level occupy a comparable on-screen
footprint and the level-of-detail picker stays shared.

**The square lattice is a quadtree.**  Unlike the hex lattice, square cells are
*corner-anchored* on a fixed origin ``(0, 0)`` in data space: cell ``(q, r)``
covers ``[q·s, (q+1)·s) × [r·s, (r+1)·s)``, so a point ``(x, y)`` lands in
``q = floor(x/s)``, ``r = floor(y/s)``.  Because the origin is the *same* at
every level and the pyramid halves ``s`` from one level to the next, a coarse
cell of side ``2s`` is exactly the union of the four fine cells of side ``s``
beneath it (``[2q, 2q+2) × [2r, 2r+2)`` in fine-cell keys).  Every fine cell
nests wholly inside one coarse cell — none straddle a coarse boundary — which
is what makes a child bin's representative provably a member of its parent and
gives :mod:`vtscore.projection.pyramid` *geometric* (not best-effort) zoom
persistence.  This is the reason the square path uses ``floor`` (corner
anchor) where the hex path uses round-to-nearest-center: a centered lattice
does not nest under doubling (fine cells would straddle coarse boundaries),
but hexagons cannot tile 4-into-1 self-similarly at all, so only the square
shape can be a true quadtree.

Anchoring in data space (a *fixed* origin, independent of ``radius``) is also
what keeps pure panning free in the renderer: membership only changes across a
zoom level, where the side halves, exactly as for the hex lattice.

A cell is identified by an integer pair ``(q, r)`` (column, row).  Unlike the
hex key, ``q`` is the plain column index (no doubling is needed because a
square lattice has no half-integer columns).  :func:`square_center` inverts the
key back to the cell's center in projection space (the cell midpoint,
``((q+0.5)·s, (r+0.5)·s)``).
"""

from __future__ import annotations

import numpy as np

from vtscore.projection.hexbin import SQRT3

# A square cell's side as a multiple of the per-level ``radius`` scale.  Equal
# to the hex lattice's column spacing (``radius·√3``) so the two lattices have
# matching cell footprints and share the renderer's level-of-detail picker.
SQUARE_SIDE_PER_RADIUS = SQRT3


def _square_side(radius: float) -> float:
    return radius * SQUARE_SIDE_PER_RADIUS


def squarebin_assign(points: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Assign each point in *points* to a square cell of the given *radius*.

    *points* is an ``(M, 2)`` float array of projection-space coordinates.
    Returns ``(q, r)``: two int64 arrays of length ``M`` holding the integer
    ``(column, row)`` cell key for each point.  Feed the same ``(q, r)`` to
    :func:`square_center` to recover the cell center.

    The cell side is ``radius·√3`` (see module docstring); cells are
    corner-anchored on the fixed origin ``(0, 0)``, so a point lands in the
    cell whose lower-left corner is at or below it, i.e. ``floor(coord / side)``
    per axis.  This corner anchor (rather than round-to-nearest-center) is what
    makes the levels nest 4-into-1 as a quadtree.
    """
    if radius <= 0:
        raise ValueError(f"square radius must be positive, got {radius!r}")
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be (M, 2), got shape {pts.shape}")

    side = _square_side(radius)
    q = np.floor(pts[:, 0] / side).astype(np.int64)
    r = np.floor(pts[:, 1] / side).astype(np.int64)
    return q, r


def square_center(q: np.ndarray | int, r: np.ndarray | int, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`squarebin_assign`: the projection-space center of cell ``(q, r)``.

    Cells are corner-anchored at ``(q·side, r·side)``, so the center is the
    cell midpoint ``((q+0.5)·side, (r+0.5)·side)``.  Accepts scalars or arrays.
    Returns ``(cx, cy)`` as float64.
    """
    side = _square_side(radius)
    q_arr = np.asarray(q, dtype=np.float64)
    r_arr = np.asarray(r, dtype=np.float64)
    return (q_arr + 0.5) * side, (r_arr + 0.5) * side
