"""Tests for Stage-2 tile pyramid (``vtscore.projection.pyramid``)."""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.projection.pyramid import (
    BIN_SHAPES,
    Pyramid,
    build_pyramid,
    max_useful_levels,
    tile_member_ids,
)
from vtscore.projection.umap_projection import Projection


def _projection(coords: np.ndarray, ids: list[int] | None = None, pid: str = "pid-test") -> Projection:
    if ids is None:
        ids = list(range(coords.shape[0]))
    return Projection(pid, ids, np.ascontiguousarray(coords, dtype=np.float32), "test")


def _cluster_cloud(seed: int = 0) -> np.ndarray:
    """Three well-separated Gaussian blobs in 2-D."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 8.0]])
    pts = np.concatenate([c + rng.standard_normal((100, 2)) * 0.5 for c in centers])
    return pts.astype(np.float32)


def test_every_point_counted_at_each_level():
    coords = _cluster_cloud()
    proj = _projection(coords)
    pyr = build_pyramid(proj, n_levels=5)
    assert isinstance(pyr, Pyramid)
    assert pyr.point_count == coords.shape[0]
    for lvl in range(5):
        total = sum(cell.count for (level, _tx, _ty), tile in pyr.tiles.items() if level == lvl for cell in tile.cells)
        assert total == coords.shape[0], f"level {lvl} lost points"


def test_finer_levels_have_more_cells():
    pyr = build_pyramid(_projection(_cluster_cloud()), n_levels=5)
    n_cells = [lm.n_cells for lm in pyr.levels]
    # Coarsest level has the fewest cells; each level resolves at least as many.
    assert n_cells[0] <= n_cells[-1]
    assert n_cells[0] >= 1
    assert max(n_cells) > min(n_cells)


def test_level_radius_halves():
    pyr = build_pyramid(_projection(_cluster_cloud()), n_levels=4, base_radius=8.0)
    assert pyr.level_radius(0) == 8.0
    assert pyr.level_radius(1) == 4.0
    assert pyr.level_radius(3) == 1.0
    assert pyr.levels[2].radius == 2.0


def test_representative_is_member_nearest_centroid():
    # Two points; one is the centroid-nearest, both share a (big) hex.
    coords = np.array([[0.0, 0.0], [0.1, 0.0]], dtype=np.float32)
    proj = _projection(coords, ids=[100, 200])
    pyr = build_pyramid(proj, n_levels=1, base_radius=100.0)
    tiles = list(pyr.tiles.values())
    assert len(tiles) == 1
    cells = tiles[0].cells
    assert len(cells) == 1
    cell = cells[0]
    assert cell.count == 2
    # centroid is at (0.05, 0); the point at (0.0,0) and (0.1,0) are equidistant,
    # tie-break to the smaller id.
    assert cell.rep_id == 100


def test_rep_ids_are_valid_ids():
    coords = _cluster_cloud()
    ids = list(range(1000, 1000 + coords.shape[0]))
    pyr = build_pyramid(_projection(coords, ids=ids), n_levels=4)
    valid = set(ids)
    for tile in pyr.tiles.values():
        for cell in tile.cells:
            assert cell.rep_id in valid


def test_tile_indexing_matches_get_tile():
    pyr = build_pyramid(_projection(_cluster_cloud()), n_levels=4)
    for (level, tx, ty), tile in pyr.tiles.items():
        assert pyr.get_tile(level, tx, ty) is tile
        assert tile.level == level and tile.tx == tx and tile.ty == ty
    assert pyr.get_tile(999, 0, 0) is None


@pytest.mark.parametrize("bin_shape", BIN_SHAPES)
def test_tile_member_ids_partition_each_tile(bin_shape):
    """Every tile's members reproduce each cell's count and union to its ids."""
    coords = _cluster_cloud()
    ids = list(range(1000, 1000 + coords.shape[0]))
    proj = _projection(coords, ids=ids)
    pyr = build_pyramid(proj, bin_shape=bin_shape, n_levels=4)

    all_recovered: set[int] = set()
    for (level, tx, ty), tile in pyr.tiles.items():
        members = tile_member_ids(pyr, proj, level, tx, ty)
        # Re-deriving membership must agree with the cell aggregates: one entry
        # per cell, member count equals the stored count, and the representative
        # is among the members.
        assert set(members) == {(c.q, c.r) for c in tile.cells}
        for cell in tile.cells:
            cell_members = members[(cell.q, cell.r)]
            assert len(cell_members) == cell.count
            assert cell.rep_id in cell_members
            all_recovered.update(cell_members)

    # Across one level, every point is recovered exactly once.
    level0_members: list[int] = []
    for (level, tx, ty) in pyr.tiles:
        if level != 0:
            continue
        for member_list in tile_member_ids(pyr, proj, level, tx, ty).values():
            level0_members.extend(member_list)
    assert sorted(level0_members) == sorted(ids)
    assert all_recovered == set(ids)


def test_tile_member_ids_empty_for_missing_tile():
    proj = _projection(_cluster_cloud())
    pyr = build_pyramid(proj, n_levels=3)
    assert tile_member_ids(pyr, proj, 0, 999, 999) == {}


def test_tile_member_ids_empty_projection():
    proj = _projection(np.empty((0, 2), dtype=np.float32), ids=[])
    pyr = build_pyramid(proj, n_levels=2)
    assert tile_member_ids(pyr, proj, 0, 0, 0) == {}


def test_empty_projection_yields_no_tiles_but_keeps_levels():
    proj = _projection(np.empty((0, 2), dtype=np.float32), ids=[])
    pyr = build_pyramid(proj, n_levels=3)
    assert pyr.tiles == {}
    assert len(pyr.levels) == 3
    assert pyr.point_count == 0


def test_meta_is_json_friendly():
    pyr = build_pyramid(_projection(_cluster_cloud(), pid="abc123"), n_levels=3)
    meta = pyr.meta()
    assert meta["projection_id"] == "abc123"
    assert len(meta["levels"]) == 3
    assert len(meta["bounds"]) == 4
    assert meta["point_count"] == 300
    # payloads are plain Python scalars (no numpy types leaking into JSON)
    tile = next(iter(pyr.tiles.values()))
    payload = tile.to_payload()
    cell0 = payload["cells"][0]
    assert isinstance(cell0["q"], int)
    assert isinstance(cell0["count"], int)
    assert isinstance(cell0["cx"], float)
    assert isinstance(cell0["rep_id"], int)


def test_coincident_points_use_fallback_radius():
    # All points identical: zero extent must not crash (radius falls back to 1).
    coords = np.zeros((10, 2), dtype=np.float32)
    pyr = build_pyramid(_projection(coords), n_levels=2)
    assert pyr.base_radius > 0
    total = sum(cell.count for t in pyr.tiles.values() for cell in t.cells if t.level == 0)
    assert total == 10


def test_max_useful_levels_bounds():
    assert max_useful_levels(0) == 1
    assert max_useful_levels(1) == 1
    assert 1 <= max_useful_levels(100) <= 14
    assert max_useful_levels(10_000) >= max_useful_levels(100)
    assert max_useful_levels(10**9) <= 14
    # Generous enough that a dense small cloud separates before the cap: the old
    # log4 ceiling bottomed out at 3 levels for ~245 clips (the ESC-50 demo),
    # leaving ~5 clips merged per hex at max zoom.  log2 sizing gives real
    # headroom.
    assert max_useful_levels(245) >= 8


def _grid_cloud(side: int = 4, spacing: float = 1.0) -> np.ndarray:
    """``side x side`` distinct points on a regular grid (deterministic)."""
    xs, ys = np.meshgrid(np.arange(side) * spacing, np.arange(side) * spacing)
    return np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)


def test_auto_depth_resolves_to_single_clip_hexes():
    # The fix: with no explicit n_levels, the build descends until every hex
    # holds exactly one clip, so hover-to-hear can audition individual sounds.
    coords = _grid_cloud(side=4, spacing=1.0)  # 16 well-separated points
    pyr = build_pyramid(_projection(coords), base_radius=4.0)
    deepest = max(lm.level for lm in pyr.levels)
    deepest_cells = [c for (lvl, _, _), t in pyr.tiles.items() if lvl == deepest for c in t.cells]
    assert deepest_cells, "deepest level has no cells"
    assert max(c.count for c in deepest_cells) == 1, "deepest level still merges clips"
    assert sum(c.count for c in deepest_cells) == coords.shape[0]
    # Coarser levels genuinely aggregate — this isn't a one-level pyramid.
    assert pyr.levels[0].n_cells < coords.shape[0]


def test_fixed_shallow_depth_still_merges_clips():
    # Contrast: a too-shallow fixed depth leaves clips merged — exactly the
    # bottoming-out the auto path now avoids.
    coords = _grid_cloud(side=4, spacing=1.0)
    pyr = build_pyramid(_projection(coords), n_levels=2, base_radius=4.0)
    assert len(pyr.levels) == 2  # explicit depth is honored exactly
    cells = [c for t in pyr.tiles.values() for c in t.cells]
    assert max(c.count for c in cells) > 1


def test_auto_depth_stops_early_for_separated_data():
    # Three points 100 apart resolve at the very first level; the build must not
    # grind all the way to the cap.
    coords = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]], dtype=np.float32)
    pyr = build_pyramid(_projection(coords))
    assert len(pyr.levels) <= 2
    cells = [c for t in pyr.tiles.values() for c in t.cells]
    assert max(c.count for c in cells) == 1


def test_auto_depth_terminates_on_coincident_points():
    # Identical points can never be separated by any radius; the build must stop
    # (saturation guard) rather than descend to the cap.
    coords = np.zeros((10, 2), dtype=np.float32)
    pyr = build_pyramid(_projection(coords))
    assert len(pyr.levels) < max_useful_levels(10)
    # Every level holds the same single, unsplittable hex of all 10 clips.
    for lvl in {level for level, _, _ in pyr.tiles}:
        cells = [c for (level, _, _), t in pyr.tiles.items() if level == lvl for c in t.cells]
        assert len(cells) == 1
        assert cells[0].count == 10


# --- Bin shape (hex vs square) -------------------------------------------------


def test_default_bin_shape_is_hex():
    pyr = build_pyramid(_projection(_cluster_cloud()), n_levels=2)
    assert pyr.bin_shape == "hex"
    assert pyr.meta()["bin_shape"] == "hex"


def test_unknown_bin_shape_raises():
    with pytest.raises(ValueError):
        build_pyramid(_projection(_cluster_cloud()), bin_shape="triangle", n_levels=1)


@pytest.mark.parametrize("bin_shape", BIN_SHAPES)
def test_points_conserved_for_every_shape(bin_shape):
    coords = _cluster_cloud()
    pyr = build_pyramid(_projection(coords), bin_shape=bin_shape, n_levels=4)
    assert pyr.bin_shape == bin_shape
    for lvl in range(4):
        total = sum(c.count for (level, _, _), t in pyr.tiles.items() if level == lvl for c in t.cells)
        assert total == coords.shape[0], f"{bin_shape} level {lvl} lost points"


@pytest.mark.parametrize("bin_shape", BIN_SHAPES)
def test_auto_depth_resolves_to_single_cell_for_every_shape(bin_shape):
    coords = _grid_cloud(side=4, spacing=1.0)
    pyr = build_pyramid(_projection(coords), bin_shape=bin_shape, base_radius=4.0)
    deepest = max(lm.level for lm in pyr.levels)
    deepest_cells = [c for (lvl, _, _), t in pyr.tiles.items() if lvl == deepest for c in t.cells]
    assert deepest_cells
    assert max(c.count for c in deepest_cells) == 1


def test_square_uses_plain_column_tile_index():
    # The square lattice keys q as the bare column index, so cell centers sit on
    # a regular grid: well-separated points land in distinct cells whose centers
    # are integer multiples of the side apart.
    coords = _grid_cloud(side=3, spacing=4.0)
    pyr = build_pyramid(_projection(coords), bin_shape="square", base_radius=1.0)
    deepest = max(lm.level for lm in pyr.levels)
    cells = [c for (lvl, _, _), t in pyr.tiles.items() if lvl == deepest for c in t.cells]
    assert sum(c.count for c in cells) == coords.shape[0]


def test_same_projection_bins_both_shapes_independently():
    # The Browse toggle re-bins one frozen projection two ways; both share the
    # projection id but produce their own cells.
    proj = _projection(_cluster_cloud(), pid="shared-pid")
    hex_pyr = build_pyramid(proj, bin_shape="hex", n_levels=3)
    sq_pyr = build_pyramid(proj, bin_shape="square", n_levels=3)
    assert hex_pyr.projection_id == sq_pyr.projection_id == "shared-pid"
    assert hex_pyr.bin_shape == "hex"
    assert sq_pyr.bin_shape == "square"
