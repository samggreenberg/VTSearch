"""Tests for projection persistence (``vtscore.projection.persistence``)."""

from __future__ import annotations

import numpy as np

from vtscore.projection.persistence import (
    load_projection,
    projection_sidecar_path,
    save_projection,
)
from vtscore.projection.pyramid import build_pyramid
from vtscore.projection.umap_projection import Projection


def _make_projection(n: int = 50, seed: int = 42, pid: str = "test-pid-abc") -> Projection:
    rng = np.random.default_rng(seed)
    coords = rng.standard_normal((n, 2)).astype(np.float32)
    ids = list(range(n))
    return Projection(pid, ids, coords, "test")


def test_round_trip(tmp_path):
    """Save then load recovers identical projection + pyramid."""
    pkl = tmp_path / "dataset.pkl"
    pkl.write_bytes(b"fake")

    proj = _make_projection()
    pyr = build_pyramid(proj, n_levels=4)

    save_projection(pkl, proj, pyr)

    sidecar = projection_sidecar_path(pkl)
    assert sidecar.exists()
    assert sidecar.suffix == ".projection"

    loaded = load_projection(pkl)
    assert loaded is not None
    proj2, pyr2 = loaded

    assert proj2.projection_id == proj.projection_id
    assert proj2.method == proj.method
    assert proj2.ids == proj.ids
    np.testing.assert_array_equal(proj2.coords, proj.coords)

    assert pyr2.projection_id == pyr.projection_id
    assert pyr2.base_radius == pyr.base_radius
    assert pyr2.tile_span == pyr.tile_span
    assert pyr2.point_count == pyr.point_count
    assert len(pyr2.levels) == len(pyr.levels)
    for lm1, lm2 in zip(pyr.levels, pyr2.levels):
        assert lm1.level == lm2.level
        assert lm1.radius == lm2.radius
        assert lm1.n_cells == lm2.n_cells

    assert set(pyr2.tiles.keys()) == set(pyr.tiles.keys())
    for key in pyr.tiles:
        t1 = pyr.tiles[key]
        t2 = pyr2.tiles[key]
        assert t1.level == t2.level
        assert t1.tx == t2.tx
        assert t1.ty == t2.ty
        assert len(t1.cells) == len(t2.cells)
        for c1, c2 in zip(t1.cells, t2.cells):
            assert c1.q == c2.q
            assert c1.r == c2.r
            assert c1.count == c2.count
            assert c1.rep_id == c2.rep_id
            assert abs(c1.cx - c2.cx) < 1e-6
            assert abs(c1.cy - c2.cy) < 1e-6


def test_missing_sidecar_returns_none(tmp_path):
    pkl = tmp_path / "missing.pkl"
    assert load_projection(pkl) is None


def test_corrupt_sidecar_returns_none(tmp_path):
    pkl = tmp_path / "corrupt.pkl"
    sidecar = projection_sidecar_path(pkl)
    sidecar.write_bytes(b"not a valid npz file")
    assert load_projection(pkl) is None


def test_empty_projection(tmp_path):
    """Edge case: zero-point projection round-trips."""
    pkl = tmp_path / "empty.pkl"
    pkl.write_bytes(b"fake")

    proj = Projection("empty-pid", [], np.empty((0, 2), dtype=np.float32), "trivial")
    pyr = build_pyramid(proj, n_levels=2)
    save_projection(pkl, proj, pyr)

    loaded = load_projection(pkl)
    assert loaded is not None
    proj2, pyr2 = loaded
    assert proj2.ids == []
    assert proj2.coords.shape == (0, 2)
    assert pyr2.point_count == 0


def test_negative_tile_indices(tmp_path):
    """Projections with negative tile indices round-trip correctly."""
    pkl = tmp_path / "neg.pkl"
    pkl.write_bytes(b"fake")

    rng = np.random.default_rng(99)
    coords = (rng.standard_normal((200, 2)) * 20).astype(np.float32)
    proj = Projection("neg-pid", list(range(200)), coords, "test")
    pyr = build_pyramid(proj, n_levels=3)

    has_negative = any(k[1] < 0 or k[2] < 0 for k in pyr.tiles)
    assert has_negative, "test data should produce negative tile indices"

    save_projection(pkl, proj, pyr)
    loaded = load_projection(pkl)
    assert loaded is not None
    _, pyr2 = loaded
    assert set(pyr2.tiles.keys()) == set(pyr.tiles.keys())


def test_overwrite_existing(tmp_path):
    """A second save overwrites the previous sidecar."""
    pkl = tmp_path / "dataset.pkl"
    pkl.write_bytes(b"fake")

    proj1 = _make_projection(pid="first")
    pyr1 = build_pyramid(proj1, n_levels=2)
    save_projection(pkl, proj1, pyr1)

    proj2 = _make_projection(seed=99, pid="second")
    pyr2 = build_pyramid(proj2, n_levels=2)
    save_projection(pkl, proj2, pyr2)

    loaded = load_projection(pkl)
    assert loaded is not None
    assert loaded[0].projection_id == "second"
