"""Stage 2 of the VTSBrowse browse canvas: the hex-tile pyramid.

Given a frozen :class:`~vtscore.projection.umap_projection.Projection`, build a
multi-resolution pyramid of hexagon aggregates that the canvas streams as it
pans and zooms.  Per *§Browse-canvas architecture* in
``docs/plans/vtsbrowse.md``:

- **Zoom levels** ``z = 0 … n_levels-1``.  Level 0 is coarsest (the whole
  projection in a handful of big hexes); each deeper level *halves the hex
  radius*, revealing finer structure while keeping the on-screen hex count
  roughly constant.
- **Per hex** the server precomputes its axial key ``(q, r)`` and center, the
  **count** of points in it (the density channel), and a **representative**
  media id — the clip whose projected point is nearest the cell's centroid,
  which is what hover-to-hear auditions.  Representatives are per level, so the
  audition clip generalizes as you zoom out.
- **Tiles** group hexes into a spatial grid per level so the client fetches
  only the tiles covering its viewport.  Because the projection is frozen, a
  tile is immutable for the life of the dataset and trivially cacheable.

This module is pure NumPy (Flask-free) and does no I/O; persistence and the
HTTP tile endpoint live in the VTSearch Browse routes.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vtscore.projection.hexbin import SQRT3, hex_center, hexbin_assign
from vtscore.projection.squarebin import square_center, squarebin_assign
from vtscore.projection.umap_projection import Projection

#: The bin shapes VTSBrowse can tile a projection with.  ``"hex"`` is the
#: default (d3-hexbin lattice); ``"square"`` is the rectangular-grid
#: alternative.  Both share the per-level ``radius`` scale and pyramid
#: structure — only the assignment / center / tile-index geometry differs.
BIN_SHAPES: tuple[str, ...] = ("hex", "square")
DEFAULT_BIN_SHAPE = "hex"


@dataclass(frozen=True)
class HexCell:
    """One aggregated hexagon at one zoom level."""

    q: int  # doubled column key (see hexbin)
    r: int  # row key
    cx: float  # center in projection space
    cy: float
    count: int  # number of clips in this cell (density)
    rep_id: int  # representative media id (nearest clip to the cell centroid)

    def to_payload(self) -> dict[str, Any]:
        """JSON-serializable form for the tile endpoint."""
        return {
            "q": self.q,
            "r": self.r,
            "cx": self.cx,
            "cy": self.cy,
            "count": self.count,
            "rep_id": self.rep_id,
        }


@dataclass(frozen=True)
class Tile:
    """The non-empty hexes inside one ``(level, tx, ty)`` spatial cell."""

    level: int
    tx: int
    ty: int
    cells: list[HexCell]

    def to_payload(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "tx": self.tx,
            "ty": self.ty,
            "cells": [c.to_payload() for c in self.cells],
        }


@dataclass(frozen=True)
class LevelMeta:
    """Sizing/summary for one zoom level, surfaced via the meta endpoint."""

    level: int
    radius: float
    n_cells: int


@dataclass(frozen=True)
class Pyramid:
    """The full hex-tile pyramid for one frozen projection."""

    projection_id: str
    bounds: tuple[float, float, float, float]
    base_radius: float
    tile_span: float  # columns/rows of cells per tile (projection-index units)
    point_count: int
    levels: list[LevelMeta]
    # (level, tx, ty) -> Tile
    tiles: dict[tuple[int, int, int], Tile]
    bin_shape: str = DEFAULT_BIN_SHAPE  # "hex" | "square" — which lattice was binned
    # In-memory, process-scoped membership cache: level -> {(tx, ty): {(q, r): [ids]}}.
    # Lazily filled by ``tile_member_ids`` (one O(N) re-bin per level, shared across
    # that level's tiles) and never persisted — the frozen coords re-imply it on the
    # next load. Excluded from equality/repr so it stays a pure cache.
    _member_index: dict[int, dict[tuple[int, int], dict[tuple[int, int], list[int]]]] = field(
        default_factory=dict, compare=False, repr=False
    )

    def level_radius(self, level: int) -> float:
        """Cell radius at *level* (``base_radius / 2**level``)."""
        return self.base_radius / (2.0**level)

    def get_tile(self, level: int, tx: int, ty: int) -> Tile | None:
        """The tile at ``(level, tx, ty)``, or ``None`` if it holds no hexes."""
        return self.tiles.get((level, tx, ty))

    def meta(self) -> dict[str, Any]:
        """JSON-serializable projection/pyramid summary for ``/api/projection/meta``."""
        return {
            "projection_id": self.projection_id,
            "bin_shape": self.bin_shape,
            "bounds": list(self.bounds),
            "base_radius": self.base_radius,
            "tile_span": self.tile_span,
            "point_count": self.point_count,
            "levels": [{"level": lm.level, "radius": lm.radius, "n_cells": lm.n_cells} for lm in self.levels],
        }


def _base_radius_for(bounds: tuple[float, float, float, float], base_cols: float) -> float:
    """Level-0 hex radius so the larger extent spans ~``base_cols`` columns."""
    xmin, ymin, xmax, ymax = bounds
    extent = max(xmax - xmin, ymax - ymin)
    if extent <= 0 or base_cols <= 0:
        return 1.0  # all points coincide (or degenerate) — any positive radius works
    return (extent / base_cols) / SQRT3


def _hex_tile_index(q: np.ndarray, r: np.ndarray, tile_span: float) -> tuple[np.ndarray, np.ndarray]:
    """Map hex keys to integer tile coords ``(tx, ty)``.

    ``q`` is the doubled column key, so the column position is ``q / 2``; tiles
    bin ``tile_span`` hex columns/rows each.
    """
    tx = np.floor((q / 2.0) / tile_span).astype(np.int64)
    ty = np.floor(r / tile_span).astype(np.int64)
    return tx, ty


def _square_tile_index(q: np.ndarray, r: np.ndarray, tile_span: float) -> tuple[np.ndarray, np.ndarray]:
    """Map square keys to integer tile coords ``(tx, ty)``.

    ``q`` is the plain column key (no doubling), so the column position is
    ``q``; tiles bin ``tile_span`` square columns/rows each.
    """
    tx = np.floor(q / tile_span).astype(np.int64)
    ty = np.floor(r / tile_span).astype(np.int64)
    return tx, ty


@dataclass(frozen=True)
class _BinGeometry:
    """The lattice-specific functions a pyramid build needs for one bin shape."""

    assign: Callable[[np.ndarray, float], tuple[np.ndarray, np.ndarray]]
    center: Callable[[Any, Any, float], tuple[np.ndarray, np.ndarray]]
    tile_index: Callable[[np.ndarray, np.ndarray, float], tuple[np.ndarray, np.ndarray]]


_GEOMETRIES: dict[str, _BinGeometry] = {
    "hex": _BinGeometry(assign=hexbin_assign, center=hex_center, tile_index=_hex_tile_index),
    "square": _BinGeometry(assign=squarebin_assign, center=square_center, tile_index=_square_tile_index),
}


def _geometry_for(bin_shape: str) -> _BinGeometry:
    try:
        return _GEOMETRIES[bin_shape]
    except KeyError:
        raise ValueError(f"unknown bin_shape {bin_shape!r}; expected one of {tuple(_GEOMETRIES)}") from None


def _build_level(
    level: int,
    coords: np.ndarray,
    ids: np.ndarray,
    radius: float,
    tile_span: float,
    geom: _BinGeometry,
) -> list[Tile]:
    """Aggregate *coords* into cells at one *level* and group them into tiles."""
    q, r = geom.assign(coords, radius)
    keys = np.stack([q, r], axis=1)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    n_hexes = uniq.shape[0]

    counts = np.bincount(inverse, minlength=n_hexes)
    # Contiguous segments of point indices grouped by hex, points within a
    # segment ordered by ascending row index (== ascending id, since ids are
    # sorted) so representative tie-breaks are deterministic.
    order = np.argsort(inverse, kind="stable")
    seg_starts = np.cumsum(counts) - counts

    centers_x, centers_y = geom.center(uniq[:, 0], uniq[:, 1], radius)
    tx_all, ty_all = geom.tile_index(uniq[:, 0], uniq[:, 1], tile_span)

    tiles: dict[tuple[int, int], list[HexCell]] = {}
    for h in range(n_hexes):
        members = order[seg_starts[h] : seg_starts[h] + counts[h]]
        pts = coords[members]
        centroid = pts.mean(axis=0)
        rep_local = int(np.argmin(np.sum((pts - centroid) ** 2, axis=1)))
        rep_id = int(ids[members[rep_local]])
        cell = HexCell(
            q=int(uniq[h, 0]),
            r=int(uniq[h, 1]),
            cx=float(centers_x[h]),
            cy=float(centers_y[h]),
            count=int(counts[h]),
            rep_id=rep_id,
        )
        tiles.setdefault((int(tx_all[h]), int(ty_all[h])), []).append(cell)

    return [Tile(level=level, tx=tx, ty=ty, cells=cells) for (tx, ty), cells in tiles.items()]


def build_pyramid(
    projection: Projection,
    *,
    bin_shape: str = DEFAULT_BIN_SHAPE,
    n_levels: int | None = None,
    base_cols: float = 6.0,
    base_radius: float | None = None,
    tile_span: float = 16.0,
) -> Pyramid:
    """Build the tile :class:`Pyramid` for a frozen *projection*.

    *bin_shape* selects the lattice: ``"hex"`` (default, d3-hexbin) or
    ``"square"`` (rectangular grid).  Both share the per-level ``radius``
    scale, the pyramid structure, and every tunable knob below; only the
    point→cell assignment and cell geometry differ.  Re-binning the same
    projection under the other shape is the cheap operation that backs the
    Browse hex/square toggle (no re-fit of UMAP).

    Zoom levels are produced ``z = 0 … L-1``, each halving the cell radius.

    - **Auto depth (default, ``n_levels=None``):** descend until every occupied
      hex holds a single clip, so the deepest level resolves the dataset to
      one-clip-per-hex and hover-to-hear can audition individual sounds.
      Descent also stops if a radius halving no longer separates any points
      (co-located clips) and is capped at :func:`max_useful_levels` as a
      runaway guard.  This is the production path: a closed-form level count
      can't be used because occupied-hex growth depends on the (unknown a
      priori) shape of the projected cloud — clustered, manifold-like
      embeddings grow far slower per level than a uniform 2-D fill would, so a
      fixed estimate systematically bottoms out before reaching single clips.
    - **Fixed depth (``n_levels`` an int):** produce exactly that many levels.

    ``base_radius`` (level 0) defaults to a value sized so the projection's
    larger extent spans ~``base_cols`` hex columns; pass it explicitly to
    override.  ``tile_span`` is the number of hex columns/rows grouped into one
    tile.  All of these are tunable knobs, not baked constants (see
    *§Open problems* in the design doc).

    Empty projections yield a pyramid with no tiles (one level when auto).
    """
    if n_levels is not None and n_levels < 1:
        raise ValueError(f"n_levels must be >= 1, got {n_levels}")

    geom = _geometry_for(bin_shape)
    coords = np.ascontiguousarray(projection.coords, dtype=np.float64)
    ids = np.asarray(projection.ids, dtype=np.int64)
    bounds = projection.bounds
    r0 = base_radius if base_radius is not None else _base_radius_for(bounds, base_cols)

    adaptive = n_levels is None
    max_levels = max_useful_levels(int(coords.shape[0])) if adaptive else n_levels

    levels: list[LevelMeta] = []
    tiles: dict[tuple[int, int, int], Tile] = {}

    prev_n_cells: int | None = None
    prev_max_count: int | None = None
    for level in range(max_levels):
        radius = r0 / (2.0**level)
        if coords.shape[0] == 0:
            levels.append(LevelMeta(level=level, radius=radius, n_cells=0))
            continue
        level_tiles = _build_level(level, coords, ids, radius, tile_span, geom)
        n_cells = sum(len(t.cells) for t in level_tiles)
        levels.append(LevelMeta(level=level, radius=radius, n_cells=n_cells))
        for t in level_tiles:
            tiles[(t.level, t.tx, t.ty)] = t

        if adaptive:
            max_count = max((c.count for t in level_tiles for c in t.cells), default=0)
            # Fully resolved: every hex holds a single clip, so deeper levels
            # would only reproduce this one at finer (wasted) radii.
            if max_count <= 1:
                break
            # No progress across a radius halving — neither the hex count nor
            # the densest cell improved — means the remaining co-located clips
            # can never be separated; stop rather than grind to the cap.
            if n_cells == prev_n_cells and max_count == prev_max_count:
                break
            prev_n_cells = n_cells
            prev_max_count = max_count

    return Pyramid(
        projection_id=projection.projection_id,
        bounds=bounds,
        base_radius=r0,
        tile_span=tile_span,
        point_count=int(coords.shape[0]),
        levels=levels,
        tiles=tiles,
        bin_shape=bin_shape,
    )


def _level_membership(
    pyr: Pyramid,
    projection: Projection,
    level: int,
) -> dict[tuple[int, int], dict[tuple[int, int], list[int]]]:
    """All of *level*'s membership, ``{(tx, ty): {(q, r): [ids]}}``, cached in-memory.

    A :class:`HexCell` stores only its ``count`` (density) and a single
    ``rep_id`` — the per-cell member lists are computed during the build and
    discarded, since persisting them would bloat the container with an index
    that the frozen 2-D coordinates already imply.  So we re-derive them here by
    re-binning *projection*'s coordinates at *level*'s radius.

    A single :func:`_geometry_for` assignment pass already partitions *every*
    point into its ``(tx, ty)`` tile and ``(q, r)`` cell, so we build the whole
    level at once and memoize it on ``pyr._member_index``.  The first tile
    fetched at a level pays the O(N) re-bin; every other tile at that level is
    then a dict lookup — turning a per-tile O(N) scan into one O(N) pass per
    level.  The cache is process-scoped and never persisted; the frozen layout
    re-derives it on the next load.
    """
    cached = pyr._member_index.get(level)
    if cached is not None:
        return cached

    coords = np.ascontiguousarray(projection.coords, dtype=np.float64)
    index: dict[tuple[int, int], dict[tuple[int, int], list[int]]] = {}
    if coords.shape[0] == 0:
        pyr._member_index[level] = index
        return index

    ids = np.asarray(projection.ids, dtype=np.int64)
    geom = _geometry_for(pyr.bin_shape)
    radius = pyr.level_radius(level)
    q, r = geom.assign(coords, radius)
    tx_all, ty_all = geom.tile_index(q, r, pyr.tile_span)

    for txx, tyy, qq, rr, mid in zip(tx_all.tolist(), ty_all.tolist(), q.tolist(), r.tolist(), ids.tolist()):
        tile_cells = index.setdefault((int(txx), int(tyy)), {})
        tile_cells.setdefault((int(qq), int(rr)), []).append(int(mid))

    pyr._member_index[level] = index
    return index


def tile_member_ids(
    pyr: Pyramid,
    projection: Projection,
    level: int,
    tx: int,
    ty: int,
) -> dict[tuple[int, int], list[int]]:
    """Media ids per cell for one tile, re-derived from the frozen layout.

    The browse canvas needs each cell's full membership to render per-cell
    selection state (none / partial / full) and to toggle a whole bin's
    contents.  Returned keyed by ``(q, r)`` so the tile endpoint can hand each
    cell its members.  Backed by the per-level cache in :func:`_level_membership`
    (computed once per level, shared across that level's tiles); the result is
    also immutable for the dataset's life, so the tile endpoint HTTP-caches it.
    """
    return _level_membership(pyr, projection, level).get((tx, ty), {})


def max_useful_levels(point_count: int) -> int:
    """A generous ``n_levels`` ceiling for *point_count* clips.

    Used by :func:`build_pyramid`'s auto-depth path purely as a runaway guard:
    the build descends until each hex holds a single clip, and this only bounds
    how deep it may ever go (e.g. when many clips project to the same point and
    can never be separated).  Sized off ``log2`` of the point count with
    headroom and clamped to ``[1, 14]`` — deep enough that any non-degenerate
    cloud separates before the cap, while over-estimating stays harmless because
    the adaptive descent stops as soon as the data resolves.

    The old heuristic assumed each level quadruples the occupied-hex count (a
    uniform 2-D fill); real, clustered embeddings grow far slower (~2x), so that
    estimate stopped one or more levels short of single-clip resolution.
    """
    if point_count <= 1:
        return 1
    return max(1, min(14, 1 + int(math.ceil(math.log(point_count, 2.0)))))
