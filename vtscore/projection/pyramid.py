"""Stage 2 of the VTSBrowse browse canvas: the hex-tile pyramid.

Given a frozen :class:`~vtscore.projection.umap_projection.Projection`, build a
multi-resolution pyramid of hexagon aggregates that the canvas streams as it
pans and zooms.  Per *§Browse-canvas architecture* in
``docs/design/vtsbrowse.md``:

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
from dataclasses import dataclass
from typing import Any

import numpy as np

from vtscore.projection.hexbin import SQRT3, hex_center, hexbin_assign
from vtscore.projection.umap_projection import Projection


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
    tile_span: float  # columns/rows of hexes per tile (projection-index units)
    point_count: int
    levels: list[LevelMeta]
    # (level, tx, ty) -> Tile
    tiles: dict[tuple[int, int, int], Tile]

    def level_radius(self, level: int) -> float:
        """Hex radius at *level* (``base_radius / 2**level``)."""
        return self.base_radius / (2.0**level)

    def get_tile(self, level: int, tx: int, ty: int) -> Tile | None:
        """The tile at ``(level, tx, ty)``, or ``None`` if it holds no hexes."""
        return self.tiles.get((level, tx, ty))

    def meta(self) -> dict[str, Any]:
        """JSON-serializable projection/pyramid summary for ``/api/projection/meta``."""
        return {
            "projection_id": self.projection_id,
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


def _tile_index(q: np.ndarray, r: np.ndarray, tile_span: float) -> tuple[np.ndarray, np.ndarray]:
    """Map hex keys to integer tile coords ``(tx, ty)``.

    ``q`` is the doubled column key, so the column position is ``q / 2``; tiles
    bin ``tile_span`` hex columns/rows each.
    """
    tx = np.floor((q / 2.0) / tile_span).astype(np.int64)
    ty = np.floor(r / tile_span).astype(np.int64)
    return tx, ty


def _build_level(level: int, coords: np.ndarray, ids: np.ndarray, radius: float, tile_span: float) -> list[Tile]:
    """Aggregate *coords* into hexes at one *level* and group them into tiles."""
    q, r = hexbin_assign(coords, radius)
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

    centers_x, centers_y = hex_center(uniq[:, 0], uniq[:, 1], radius)
    tx_all, ty_all = _tile_index(uniq[:, 0], uniq[:, 1], tile_span)

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
    n_levels: int = 6,
    base_cols: float = 6.0,
    base_radius: float | None = None,
    tile_span: float = 16.0,
) -> Pyramid:
    """Build the hex-tile :class:`Pyramid` for a frozen *projection*.

    ``n_levels`` zoom levels are produced (``z = 0 … n_levels-1``), each
    halving the hex radius.  ``base_radius`` (level 0) defaults to a value
    sized so the projection's larger extent spans ~``base_cols`` hex columns;
    pass it explicitly to override.  ``tile_span`` is the number of hex
    columns/rows grouped into one tile.  All of these are tunable knobs, not
    baked constants (see *§Open problems* in the design doc).

    Empty projections yield a pyramid with no tiles.
    """
    if n_levels < 1:
        raise ValueError(f"n_levels must be >= 1, got {n_levels}")

    coords = np.ascontiguousarray(projection.coords, dtype=np.float64)
    ids = np.asarray(projection.ids, dtype=np.int64)
    bounds = projection.bounds
    r0 = base_radius if base_radius is not None else _base_radius_for(bounds, base_cols)

    levels: list[LevelMeta] = []
    tiles: dict[tuple[int, int, int], Tile] = {}

    for level in range(n_levels):
        radius = r0 / (2.0**level)
        if coords.shape[0] == 0:
            levels.append(LevelMeta(level=level, radius=radius, n_cells=0))
            continue
        level_tiles = _build_level(level, coords, ids, radius, tile_span)
        n_cells = sum(len(t.cells) for t in level_tiles)
        levels.append(LevelMeta(level=level, radius=radius, n_cells=n_cells))
        for t in level_tiles:
            tiles[(t.level, t.tx, t.ty)] = t

    return Pyramid(
        projection_id=projection.projection_id,
        bounds=bounds,
        base_radius=r0,
        tile_span=tile_span,
        point_count=int(coords.shape[0]),
        levels=levels,
        tiles=tiles,
    )


def max_useful_levels(point_count: int, base_cols: float = 6.0) -> int:
    """A reasonable ``n_levels`` ceiling for *point_count* clips.

    Heuristic: keep descending until a level could resolve roughly one clip per
    hex.  Each level multiplies the hex count by ~4 (radius halves in 2-D), and
    level 0 spans ~``base_cols**2`` hexes, so ``log4(point_count / base_cols**2)``
    extra levels suffice.  Clamped to ``[1, 12]``.  Advisory only — callers
    pass the result as ``n_levels`` to :func:`build_pyramid`.
    """
    if point_count <= 1:
        return 1
    level0_hexes = max(base_cols * base_cols, 1.0)
    extra = math.log(max(point_count / level0_hexes, 1.0), 4.0)
    return max(1, min(12, 1 + int(math.ceil(extra))))
