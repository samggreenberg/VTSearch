"""Tests for in-place subset edits: ``remove_ids`` + ``rebin_like``.

These back the Browser's "Remove from Good" cull, which drops hand-selected
false-positives from a subset browse *without* re-fitting UMAP: the surviving
points keep their exact 2-D coordinates and bins; only counts/membership change.
"""

from __future__ import annotations

import numpy as np

from vtscore.projection import build_pyramid, rebin_like, remove_ids
from vtscore.projection.umap_projection import Projection


def _projection(seed: int = 0, n: int = 60, pid: str = "pid-test") -> Projection:
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 8.0]])
    pts = np.concatenate([c + rng.standard_normal((n // 3, 2)) * 0.5 for c in centers]).astype(np.float32)
    return Projection(pid, list(range(pts.shape[0])), np.ascontiguousarray(pts), "test")


def test_remove_ids_preserves_id_and_coords():
    proj = _projection()
    removed = {0, 5, 17}

    out = remove_ids(proj, removed)

    # Layout identity is intentionally preserved (the canvas keys its
    # re-frame decision on it).
    assert out.projection_id == proj.projection_id
    assert set(out.ids) == set(proj.ids) - removed
    assert out.coords.shape[0] == len(proj.ids) - len(removed)
    # Every surviving point keeps its exact coordinate — nothing is re-fit.
    for mid in out.ids:
        np.testing.assert_array_equal(out.coords[out.ids.index(mid)], proj.coords[proj.ids.index(mid)])


def test_remove_ids_keeps_order():
    proj = _projection()
    out = remove_ids(proj, {proj.ids[3]})
    assert out.ids == [i for i in proj.ids if i != proj.ids[3]]


def test_remove_ids_ignores_unknown():
    proj = _projection()
    out = remove_ids(proj, {999998, 999999})
    assert out.ids == proj.ids
    np.testing.assert_array_equal(out.coords, proj.coords)


def test_rebin_like_keeps_bins_for_survivors():
    proj = _projection()
    template = build_pyramid(proj, n_levels=5)

    removed = {proj.ids[0], proj.ids[1], proj.ids[2]}
    reduced = remove_ids(proj, removed)
    rebinned = rebin_like(reduced, template)

    # Same grid geometry — never a re-fit-driven reshape.
    assert rebinned.bin_shape == template.bin_shape
    assert rebinned.base_radius == template.base_radius
    assert rebinned.tile_span == template.tile_span
    assert rebinned.bounds == template.bounds
    assert len(rebinned.levels) == len(template.levels)
    assert rebinned.point_count == proj.point_count - len(removed)

    # Each surviving id lands in the SAME (q, r) cell it occupied before.
    for level in range(len(template.levels)):
        before = {
            mid: (q, r)
            for (q, r), members in _members(template, proj, level).items()
            for mid in members
        }
        after = {
            mid: (q, r)
            for (q, r), members in _members(rebinned, reduced, level).items()
            for mid in members
        }
        for mid in reduced.ids:
            assert after[mid] == before[mid], f"id {mid} moved bins at level {level}"


def test_rebin_like_drops_emptied_cells():
    # One isolated point far from the rest gets its own cell; removing it must
    # drop that cell entirely rather than leave an empty bin.
    coords = np.array([[0.0, 0.0], [0.1, 0.0], [100.0, 100.0]], dtype=np.float32)
    proj = Projection("pid", [1, 2, 3], coords, "test")
    template = build_pyramid(proj, n_levels=3)
    total_cells_before = sum(len(t.cells) for t in template.tiles.values())

    rebinned = rebin_like(remove_ids(proj, {3}), template)
    total_cells_after = sum(len(t.cells) for t in rebinned.tiles.values())
    assert rebinned.point_count == 2
    assert total_cells_after < total_cells_before


def _members(pyr, proj, level: int) -> dict[tuple[int, int], list[int]]:
    """All ``(q, r) -> [ids]`` for a level, flattening the per-tile index."""
    from vtscore.projection import tile_member_ids

    out: dict[tuple[int, int], list[int]] = {}
    for tlevel, tx, ty in pyr.tiles:
        if tlevel != level:
            continue
        for qr, ids in tile_member_ids(pyr, proj, level, tx, ty).items():
            out[qr] = ids
    return out
