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
  media id — the clip the canvas shows as the bin's thumbnail and hover-to-hear
  auditions.  Representatives are chosen **bottom-up for zoom persistence** (see
  :func:`_assign_reps`): at the deepest level a cell's rep is the clip nearest
  its centroid, and each coarser cell *inherits* the rep of one of the finer
  cells beneath it — the one nearest the coarse cell's centroid.  So
  ``reps(level z) ⊆ reps(level z+1)``: every thumbnail you see persists as you
  zoom in (the eye can track it) while 3–4 genuinely new ones appear, instead of
  the whole grid reshuffling each level.
- **Tiles** group hexes into a spatial grid per level so the client fetches
  only the tiles covering its viewport.  Because the projection is frozen, a
  tile is immutable for the life of the dataset and trivially cacheable.

This module is pure NumPy (Flask-free) and does no I/O; persistence and the
HTTP tile endpoint live in the VTSearch Browse routes.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from vtscore.projection.hexbin import SQRT3, hex_center, hexbin_assign
from vtscore.projection.squarebin import square_center, squarebin_assign
from vtscore.projection.umap_projection import Projection

#: The bin shapes VTSBrowse can tile a projection with.  ``"square"`` is the
#: rectangular-grid (quadtree) lattice; ``"hex"`` is the d3-hexbin lattice.
#: Both share the per-level ``radius`` scale and pyramid structure — only the
#: assignment / center / tile-index geometry differs.
BIN_SHAPES: tuple[str, ...] = ("hex", "square")
DEFAULT_BIN_SHAPE = "hex"


def bin_shape_for_media_type(media_type: str | None) -> str:
    """Return the bin shape VTSBrowse tiles a *media_type* dataset with.

    The shape is a fixed property of the media type, not a user choice: media
    whose items have a *browsable thumbnail* (``MediaType.has_thumbnail`` —
    image / video / document, and audio via its waveform PNG) tile as
    **squares** so the thumbnails pack edge-to-edge with no wasted gaps (and
    the quadtree lattice keeps representatives perfectly zoom-persistent).
    Everything else (text) falls back to the **hex** density map.

    ``has_thumbnail`` on the media type is the single source of truth for this
    distinction (it is also surfaced to the frontend via
    ``GET /api/media-types``).  We resolve it through a lazy registry lookup so
    this projection module keeps its "pure NumPy, Flask-free" property and
    takes no module-load dependency on the (heavy) media package.  Empty,
    unknown, or unresolvable types fall back to hex.
    """
    if not media_type:
        return "hex"
    try:
        from vtscore.media import get as _get_media_type

        return "square" if _get_media_type(media_type).has_thumbnail else "hex"
    except (KeyError, ImportError):
        return "hex"


@dataclass(frozen=True)
class HexCell:
    """One aggregated hexagon at one zoom level."""

    q: int  # doubled column key (see hexbin)
    r: int  # row key
    cx: float  # center in projection space
    cy: float
    count: int  # number of clips in this cell (density)
    rep_id: int  # representative media id (see _assign_reps: nearest clip to the
    # cell centroid at the deepest level, an inherited finer rep above it)

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


@dataclass
class _LevelCells:
    """One level's binning, *without* representatives — the raw material both the
    tile assembly and the bottom-up :func:`_assign_reps` pass consume.

    All arrays are indexed by cell (``0 … n_cells-1``) except ``inverse``, which
    is indexed by point and maps each of the projection's points to its cell at
    this level.  ``members[h]`` are the point indices in cell ``h`` (ascending
    id, since ids are sorted) so rep tie-breaks are deterministic.
    """

    level: int
    radius: float
    keys: np.ndarray  # (n_cells, 2) int — (q, r) per cell
    centers_x: np.ndarray
    centers_y: np.ndarray
    counts: np.ndarray  # (n_cells,)
    tx: np.ndarray  # (n_cells,)
    ty: np.ndarray
    members: list[np.ndarray]  # per cell: point indices, ascending id
    centroids: np.ndarray  # (n_cells, 2) — mean of member coords
    inverse: np.ndarray  # (n_points,) — point -> cell index


def _level_cells(
    level: int,
    coords: np.ndarray,
    radius: float,
    tile_span: float,
    geom: _BinGeometry,
) -> _LevelCells:
    """Bin *coords* into cells at one *level* (geometry + membership, no reps)."""
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
    members = [order[seg_starts[h] : seg_starts[h] + counts[h]] for h in range(n_hexes)]
    # Per-cell centroid in one vectorized pass: sum member coords by segment.
    sums = np.add.reduceat(coords[order], seg_starts, axis=0)
    centroids = sums / counts[:, None]

    centers_x, centers_y = geom.center(uniq[:, 0], uniq[:, 1], radius)
    tx_all, ty_all = geom.tile_index(uniq[:, 0], uniq[:, 1], tile_span)

    return _LevelCells(
        level=level,
        radius=radius,
        keys=uniq,
        centers_x=centers_x,
        centers_y=centers_y,
        counts=counts,
        tx=tx_all,
        ty=ty_all,
        members=members,
        centroids=centroids,
        inverse=inverse,
    )


def _assign_reps(
    lcs: list[_LevelCells],
    coords: np.ndarray,
    ids: np.ndarray,
    prior_reps: dict[int, dict[tuple[int, int], int]] | None = None,
) -> dict[int, np.ndarray]:
    """Pick each cell's representative point, **bottom-up**, for zoom persistence.

    Returns ``{level: rep_point_index_per_cell}``.  *lcs* is in ascending level
    order (0 coarsest … deepest finest); we walk it finest→coarsest:

    - **Deepest level:** a cell's rep is the member nearest its centroid.
    - **Each coarser level:** a cell's candidates are its members that are *their
      own* finer rep (``point_rep_fine[m] == m``) — i.e. the finer reps that
      actually fall inside this cell.  The cell adopts the candidate nearest its
      centroid, so ``reps(coarse) ⊆ reps(fine)`` (the zoom-persistence invariant)
      **and** the rep is always a member of the bin — which the bin popup relies
      on to open/scroll to it within the member list.  A coarse cell that
      contains no finer rep at all (every finer cell beneath it straddles its
      boundary, the rep landing in a neighbour) falls back to its own
      centroid-nearest member; that one rep is not inherited, the single
      concession to keeping the rep in-bin.  This fallback only fires for the
      **hex** lattice, whose round-to-nearest-center cells do not nest under a
      radius halving.  The **square** lattice is a corner-anchored quadtree
      (see :mod:`vtscore.projection.squarebin`): every fine cell nests wholly
      inside one coarse cell, so the inherited pool is never empty and square
      reps persist by construction.

    *prior_reps* (``{level: {(q, r): rep_id}}``) lets a re-bin **keep a surviving
    representative in place**: if a cell's prior rep id is still present, it is
    retained verbatim regardless of where the centroid now sits.  This is what
    makes removing items a fast, stable repaint — thumbnails only move when the
    rep itself is among the removed (then the cell re-picks from its surviving
    inherited candidates, and that change propagates up only where a coarse cell
    had inherited the now-dead rep).  A slightly off-centre surviving rep is
    accepted on purpose; re-centring it is explicitly lower priority than not
    churning the grid.
    """
    id_to_pos = {int(m): i for i, m in enumerate(ids.tolist())} if prior_reps else None
    reps_by_level: dict[int, np.ndarray] = {}
    # For each point, the rep point-index of the cell it fell in one level finer
    # (None at the deepest level, where reps come straight from the centroid).
    point_rep_fine: np.ndarray | None = None
    for lc in reversed(lcs):
        n = lc.keys.shape[0]
        reps = np.empty(n, dtype=np.int64)
        prior = prior_reps.get(lc.level) if prior_reps else None
        for h in range(n):
            if prior is not None:
                pid = prior.get((int(lc.keys[h, 0]), int(lc.keys[h, 1])))
                if pid is not None and id_to_pos is not None and pid in id_to_pos:
                    reps[h] = id_to_pos[pid]  # keep-put: surviving rep stays
                    continue
            mem = lc.members[h]
            if point_rep_fine is None:
                pool = mem  # deepest level: any member may win
            else:
                # Inherit: the members that are their own finer rep — these are
                # the finer reps that fall inside this cell, so the choice both
                # persists and stays in-bin.  Empty only for a cell with no
                # interior finer rep; then fall back to all members.
                inherited = mem[point_rep_fine[mem] == mem]
                pool = inherited if inherited.size else mem
            d = np.sum((coords[pool] - lc.centroids[h]) ** 2, axis=1)
            reps[h] = int(pool[int(np.argmin(d))])
        reps_by_level[lc.level] = reps
        point_rep_fine = reps[lc.inverse]  # carry this level's reps down to points
    return reps_by_level


def _descent_resolved(
    lc: _LevelCells,
    coords: np.ndarray,
    n_cells: int,
    max_count: int,
    prev_n_cells: int | None,
    prev_max_count: int | None,
) -> bool:
    """Whether adaptive depth should stop descending after binning level *lc*.

    Two terminal conditions:

    - **Fully resolved:** every cell holds a single clip (``max_count <= 1``), so
      deeper levels would only reproduce this one at finer (wasted) radii.
    - **Co-located:** no progress across a radius halving — neither the cell count
      nor the densest cell improved — *and* the densest cell's members are
      genuinely coincident (zero spatial extent), so a finer radius can never
      separate them.  The coincidence check matters because "no cell split" alone
      does **not** imply co-location: a corner-anchored square cell wider than the
      whole cloud holds every point for several levels before its boundaries fall
      between them.  Only the spread test distinguishes "can't separate" from
      "haven't separated yet"; without it the descent would stop short on tight
      clouds under the square (quadtree) lattice.
    """
    if max_count <= 1:
        return True
    if n_cells == prev_n_cells and max_count == prev_max_count:
        pile = coords[lc.members[int(np.argmax(lc.counts))]]
        return float(np.max(pile.max(axis=0) - pile.min(axis=0))) == 0.0
    return False


def build_pyramid(
    projection: Projection,
    *,
    bin_shape: str = DEFAULT_BIN_SHAPE,
    n_levels: int | None = None,
    base_cols: float = 6.0,
    base_radius: float | None = None,
    tile_span: float = 16.0,
    prior_reps: dict[int, dict[tuple[int, int], int]] | None = None,
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

    *prior_reps* (``{level: {(q, r): rep_id}}``) is threaded to
    :func:`_assign_reps` so a re-bin can keep surviving representatives in place
    (see :func:`rebin_like`); leave it ``None`` for a fresh build, where reps are
    chosen purely bottom-up.

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

    # Phase 1: bin every level (geometry + membership), honouring adaptive depth.
    levels: list[LevelMeta] = []
    lcs: list[_LevelCells] = []
    prev_n_cells: int | None = None
    prev_max_count: int | None = None
    for level in range(max_levels):
        radius = r0 / (2.0**level)
        if coords.shape[0] == 0:
            levels.append(LevelMeta(level=level, radius=radius, n_cells=0))
            continue
        lc = _level_cells(level, coords, radius, tile_span, geom)
        n_cells = int(lc.keys.shape[0])
        levels.append(LevelMeta(level=level, radius=radius, n_cells=n_cells))
        lcs.append(lc)

        if adaptive:
            max_count = int(lc.counts.max()) if lc.counts.size else 0
            if _descent_resolved(lc, coords, n_cells, max_count, prev_n_cells, prev_max_count):
                break
            prev_n_cells = n_cells
            prev_max_count = max_count

    # Phase 2: pick representatives bottom-up (coarse reps inherit from finer
    # ones), then assemble the per-level tiles.
    reps_by_level = _assign_reps(lcs, coords, ids, prior_reps) if lcs else {}
    tiles: dict[tuple[int, int, int], Tile] = {}
    for lc in lcs:
        reps = reps_by_level[lc.level]
        level_tiles: dict[tuple[int, int], list[HexCell]] = {}
        for h in range(lc.keys.shape[0]):
            cell = HexCell(
                q=int(lc.keys[h, 0]),
                r=int(lc.keys[h, 1]),
                cx=float(lc.centers_x[h]),
                cy=float(lc.centers_y[h]),
                count=int(lc.counts[h]),
                rep_id=int(ids[reps[h]]),
            )
            level_tiles.setdefault((int(lc.tx[h]), int(lc.ty[h])), []).append(cell)
        for (tx, ty), cells in level_tiles.items():
            tiles[(lc.level, tx, ty)] = Tile(level=lc.level, tx=tx, ty=ty, cells=cells)

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


def _reps_by_level(pyr: Pyramid) -> dict[int, dict[tuple[int, int], int]]:
    """``{level: {(q, r): rep_id}}`` — a pyramid's current representatives, the
    ``prior_reps`` :func:`build_pyramid` honours on a re-bin."""
    out: dict[int, dict[tuple[int, int], int]] = {}
    for (level, _tx, _ty), tile in pyr.tiles.items():
        cells = out.setdefault(level, {})
        for c in tile.cells:
            cells[(c.q, c.r)] = c.rep_id
    return out


def rebin_like(projection: Projection, template: Pyramid, *, preserve_reps: bool = True) -> Pyramid:
    """Re-bin *projection* onto *template*'s exact grid, without re-fitting UMAP.

    Reuses the template pyramid's ``base_radius``, ``tile_span``, ``bin_shape``,
    level count **and** ``bounds`` so every surviving point lands in the same
    ``(q, r)`` cell — and the canvas's coord→screen transform (driven by
    ``bounds``) is unchanged, so nothing moves on screen.  Only counts,
    representatives, and now-empty cells differ.  This is the cheap operation
    behind removing items from a subset browse: the 2-D layout is frozen; we
    just recompute which items fall in which (unchanged) bins.

    With *preserve_reps* (the default), the template's representatives are fed
    back in as ``prior_reps`` so every cell whose rep **survived the removal
    keeps it** — the grid repaints without thumbnails jumping around.  Only
    cells whose rep was among the removed re-pick (from their surviving
    inherited candidates), and that change propagates upward just to the coarse
    cells that had inherited the now-dead rep.  Pass ``preserve_reps=False`` to
    re-derive reps from scratch (a fresh bottom-up pass).
    """
    pyr = build_pyramid(
        projection,
        bin_shape=template.bin_shape,
        n_levels=len(template.levels),
        base_radius=template.base_radius,
        tile_span=template.tile_span,
        prior_reps=_reps_by_level(template) if preserve_reps else None,
    )
    # Keep the original extent so the client never re-frames; bins are assigned
    # from absolute coords + radius (origin-independent), so a stable ``bounds``
    # is purely a metadata/transform concern, not a binning one.
    return replace(pyr, bounds=template.bounds)


def _hilbert_order(coords: np.ndarray, bits: int = 16) -> np.ndarray:
    """Indices that walk *coords* along a 2-D Hilbert space-filling curve.

    Returns a permutation ``perm`` such that ``coords[perm]`` traverses the frozen
    layout in a locality-preserving 1-D order: points adjacent in ``perm`` are
    adjacent in the plane, so a contiguous slice of the order is a contiguous
    region of the projection.  This is the cheap "derive a 1-D ordering from the
    2-D coords" (one quantize + bit-twiddle + stable argsort, no second UMAP fit)
    that orders bin-popup contents — because UMAP already places semantically
    similar items near each other, the curve keeps them grouped in the list
    (cats with cats, dogs with dogs) instead of scattering by media id, and the
    order of any subset is just the order with the hidden points dropped.

    The integer Hilbert distance is computed by the standard quadrant-rotation
    walk (Wikipedia's ``xy2d``/``rot``), vectorized over all points.  Ties
    (co-located points map to the same curve cell) break by ascending row index
    via a stable sort — ascending id, since ids are sorted — so the order is
    deterministic.
    """
    n = coords.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)
    side = 1 << bits  # curve resolution: side x side cells
    mins = coords.min(axis=0)
    span = np.maximum(coords.max(axis=0) - mins, 1e-12)  # guard all-coincident axes
    grid = np.clip(((coords - mins) / span * (side - 1)).astype(np.int64), 0, side - 1)
    x = grid[:, 0].copy()
    y = grid[:, 1].copy()
    d = np.zeros(n, dtype=np.int64)
    s = side >> 1
    while s > 0:
        rx = ((x & s) > 0).astype(np.int64)
        ry = ((y & s) > 0).astype(np.int64)
        d += s * s * ((3 * rx) ^ ry)
        # rot(): in the ry == 0 quadrant, flip both axes when rx == 1, then swap.
        flip = (ry == 0) & (rx == 1)
        x[flip] = (side - 1) - x[flip]
        y[flip] = (side - 1) - y[flip]
        swap = ry == 0
        x_swap = x[swap]
        x[swap] = y[swap]
        y[swap] = x_swap
        s >>= 1
    return np.argsort(d, kind="stable")


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

    Within each cell, members come out in :func:`_hilbert_order` (a 1-D Hilbert
    traversal of the frozen layout) rather than by id, so a dense bin's contents
    group by region — the popup shows similar items together and a subset keeps
    that order when items are hidden.
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

    # Visit points along the Hilbert curve so each cell's member list lands in a
    # locality-preserving 1-D order (shared across this level's tiles; the whole
    # index is memoized, so the curve is computed at most once per level).
    perm = _hilbert_order(coords)
    for txx, tyy, qq, rr, mid in zip(
        tx_all[perm].tolist(), ty_all[perm].tolist(), q[perm].tolist(), r[perm].tolist(), ids[perm].tolist()
    ):
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
    cell its members, each list ordered by a 1-D Hilbert traversal of the layout
    (see :func:`_level_membership`) so the popup groups similar items.  Backed by
    the per-level cache in :func:`_level_membership` (computed once per level,
    shared across that level's tiles); the result is also immutable for the
    dataset's life, so the tile endpoint HTTP-caches it.
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
