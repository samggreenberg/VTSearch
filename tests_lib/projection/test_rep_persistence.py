"""Tests for zoom-persistent bin representatives (``_assign_reps``).

The browse canvas shows one thumbnail per bin.  Reps are chosen bottom-up so a
coarse bin's rep is always one of the reps of the finer bins beneath it: when
you zoom in, every thumbnail you were looking at persists (the eye can track
it) while 3-4 genuinely new ones appear.  On a removal the surviving reps stay
put for a fast, stable repaint; only a removed rep forces a (minimal) re-pick.
"""

from __future__ import annotations

import numpy as np

from vtscore.projection import build_pyramid, rebin_like, remove_ids, tile_member_ids
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


def _reps_at(pyr, level: int) -> set[int]:
    return {c.rep_id for (lvl, _tx, _ty), t in pyr.tiles.items() if lvl == level for c in t.cells}


def _rep_of_cell(pyr, level: int, q: int, r: int) -> int | None:
    for (lvl, _tx, _ty), t in pyr.tiles.items():
        if lvl == level:
            for c in t.cells:
                if c.q == q and c.r == r:
                    return c.rep_id
    return None


def _members_of(pyr, proj, level: int, q: int, r: int) -> list[int]:
    for tlevel, tx, ty in pyr.tiles:
        if tlevel != level:
            continue
        cell_members = tile_member_ids(pyr, proj, level, tx, ty)
        if (q, r) in cell_members:
            return cell_members[(q, r)]
    return []


def _a_dense_level0_cell(pyr):
    """A level-0 cell holding more than one clip (so it has a real rep choice)."""
    return next(c for (lvl, _, _), t in pyr.tiles.items() if lvl == 0 for c in t.cells if c.count > 1)


def _nearest_member(proj, members: list[int]) -> int:
    """The member id nearest the members' centroid (the deepest-level / fallback rule)."""
    id_to_idx = {int(m): i for i, m in enumerate(proj.ids)}
    pts = np.asarray(proj.coords, dtype=np.float64)[[id_to_idx[m] for m in members]]
    centroid = pts.mean(axis=0)
    return members[int(np.argmin(np.sum((pts - centroid) ** 2, axis=1)))]


def _assert_reps_well_formed(pyr, proj):
    """The exact rep contract, true by construction at every level:

    1. a cell's rep is one of its own members (the bin popup opens/scrolls to it);
    2. a coarse rep is *inherited* (also a rep one level finer) — the persistence
       invariant — unless that cell holds no finer rep at all, in which case it
       falls back to its centroid-nearest member.
    """
    levels = sorted({lvl for lvl, _, _ in pyr.tiles})
    for i, lvl in enumerate(levels):
        finer = _reps_at(pyr, levels[i + 1]) if i + 1 < len(levels) else None
        for _key, tile in ((k, t) for k, t in pyr.tiles.items() if k[0] == lvl):
            for cell in tile.cells:
                members = _members_of(pyr, proj, lvl, cell.q, cell.r)
                assert cell.rep_id in members, f"rep {cell.rep_id} not a member of its bin"
                if finer is not None and cell.rep_id not in finer:
                    # Not inherited -> must be the centroid-nearest fallback.
                    assert cell.rep_id == _nearest_member(proj, members)


def _persistence_fractions(pyr) -> list[float]:
    """Per zoom-in hop, the fraction of coarse reps that persist into the finer level."""
    levels = sorted({lvl for lvl, _, _ in pyr.tiles})
    out = []
    for i in range(len(levels) - 1):
        coarse, finer = _reps_at(pyr, levels[i]), _reps_at(pyr, levels[i + 1])
        out.append(len(coarse & finer) / len(coarse))
    return out


def test_reps_are_well_formed_and_mostly_persist():
    # Most thumbnails persist on each zoom-in (the eye can track them); the few
    # that don't are the documented centroid-nearest fallback.
    proj = _projection(_cluster_cloud())
    pyr = build_pyramid(proj, n_levels=5)
    _assert_reps_well_formed(pyr, proj)
    assert min(_persistence_fractions(pyr)) >= 0.7


def test_deepest_rep_is_member_nearest_centroid():
    # At the deepest level there is nothing finer to inherit from, so the rep is
    # the member nearest the centroid (tie -> smallest id), unchanged behaviour.
    coords = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)  # coincident
    proj = _projection(coords, ids=[100, 200])
    pyr = build_pyramid(proj, n_levels=3)
    deepest = max(lm.level for lm in pyr.levels)
    assert _reps_at(pyr, deepest) == {100}


def test_rebin_keeps_surviving_rep_put():
    # Removing a non-rep member must NOT move the bin's rep (fast, stable repaint).
    proj = _projection(_cluster_cloud())
    template = build_pyramid(proj, n_levels=5)
    cell = _a_dense_level0_cell(template)
    members = _members_of(template, proj, 0, cell.q, cell.r)
    victim = next(m for m in members if m != cell.rep_id)

    rebinned = rebin_like(remove_ids(proj, {victim}), template)
    assert _rep_of_cell(rebinned, 0, cell.q, cell.r) == cell.rep_id


def test_rebin_repicks_removed_rep_to_a_survivor():
    # Removing the rep itself forces a re-pick to a surviving clip, and the
    # pyramid stays well-formed (in-bin + persistent-or-fallback) afterwards.
    proj = _projection(_cluster_cloud())
    template = build_pyramid(proj, n_levels=5)
    cell = _a_dense_level0_cell(template)
    removed_rep = cell.rep_id

    reduced = remove_ids(proj, {removed_rep})
    rebinned = rebin_like(reduced, template)

    new_rep = _rep_of_cell(rebinned, 0, cell.q, cell.r)
    assert new_rep is not None
    assert new_rep != removed_rep
    assert new_rep in set(reduced.ids)
    _assert_reps_well_formed(rebinned, reduced)


def test_rebin_preserve_reps_false_rederives():
    # Opt-out path: a from-scratch re-bin re-derives reps but stays well-formed.
    proj = _projection(_cluster_cloud())
    template = build_pyramid(proj, n_levels=4)
    rebinned = rebin_like(proj, template, preserve_reps=False)
    _assert_reps_well_formed(rebinned, proj)
