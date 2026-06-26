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


def test_coarse_reps_persist_into_finer_levels():
    # The invariant: reps(level z) ⊆ reps(level z+1) for every adjacent pair, so
    # a thumbnail never vanishes as you zoom in.
    proj = _projection(_cluster_cloud())
    pyr = build_pyramid(proj, n_levels=5)
    for lvl in range(len(pyr.levels) - 1):
        coarse = _reps_at(pyr, lvl)
        finer = _reps_at(pyr, lvl + 1)
        assert coarse <= finer, f"level {lvl} reps not all carried into level {lvl + 1}"


def test_deepest_rep_is_member_nearest_centroid():
    # At the deepest level there is nothing finer to inherit from, so the rep is
    # the member nearest the centroid (tie -> smallest id), unchanged behaviour.
    coords = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)  # coincident
    proj = _projection(coords, ids=[100, 200])
    pyr = build_pyramid(proj, n_levels=3)
    deepest = max(lm.level for lm in pyr.levels)
    assert _reps_at(pyr, deepest) == {100}


def test_coarse_rep_is_inherited_from_a_finer_cell():
    # Every coarse rep must literally be some finer cell's rep, not an arbitrary
    # nearest-centroid pick computed afresh at the coarse level.
    proj = _projection(_cluster_cloud())
    pyr = build_pyramid(proj, n_levels=4)
    for lvl in range(len(pyr.levels) - 1):
        finer = _reps_at(pyr, lvl + 1)
        for rep in _reps_at(pyr, lvl):
            assert rep in finer


def test_rebin_keeps_surviving_rep_put():
    # Removing a non-rep member must NOT move the bin's rep (fast, stable repaint).
    proj = _projection(_cluster_cloud())
    template = build_pyramid(proj, n_levels=5)
    cell = _a_dense_level0_cell(template)
    members = _members_of(template, proj, 0, cell.q, cell.r)
    victim = next(m for m in members if m != cell.rep_id)

    rebinned = rebin_like(remove_ids(proj, {victim}), template)
    assert _rep_of_cell(rebinned, 0, cell.q, cell.r) == cell.rep_id


def test_rebin_repicks_removed_rep_and_stays_persistent():
    # Removing the rep itself forces a re-pick — to a surviving clip — and the
    # zoom-persistence invariant must still hold across the whole pyramid.
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
    for lvl in range(len(rebinned.levels) - 1):
        assert _reps_at(rebinned, lvl) <= _reps_at(rebinned, lvl + 1)


def test_rebin_preserve_reps_false_rederives():
    # Opt-out path: a from-scratch re-bin still satisfies the invariant but does
    # not promise to keep any particular surviving rep.
    proj = _projection(_cluster_cloud())
    template = build_pyramid(proj, n_levels=4)
    rebinned = rebin_like(proj, template, preserve_reps=False)
    for lvl in range(len(rebinned.levels) - 1):
        assert _reps_at(rebinned, lvl) <= _reps_at(rebinned, lvl + 1)
