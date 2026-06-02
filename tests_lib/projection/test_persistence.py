"""Tests for projection persistence via the ZIP container."""

from __future__ import annotations

import numpy as np

from vtscore.datasets.container import (
    append_projection,
    read_projection,
    write_container,
)
from vtscore.projection.pyramid import build_pyramid
from vtscore.projection.umap_projection import Projection

import pickle


def _make_container(tmp_path, n: int = 50):
    path = tmp_path / "dataset.pkl"
    medias = {i: {"id": i, "embedding": [0.1] * 8} for i in range(n)}
    pkl_bytes = pickle.dumps({"medias": medias})
    meta = {"format_version": 1, "embedder": "test", "clipper": "", "media_type": "audio"}
    write_container(path, pkl_bytes, meta)
    return path


def _make_projection(n: int = 50, seed: int = 42, pid: str = "test-pid-abc") -> Projection:
    rng = np.random.default_rng(seed)
    coords = rng.standard_normal((n, 2)).astype(np.float32)
    ids = list(range(n))
    return Projection(pid, ids, coords, "test")


def test_round_trip(tmp_path):
    """Append then read recovers identical projection + pyramid."""
    path = _make_container(tmp_path)

    proj = _make_projection()
    pyr = build_pyramid(proj, n_levels=4)

    append_projection(path, proj, pyr)

    loaded = read_projection(path)
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
        assert len(t1.cells) == len(t2.cells)


def test_missing_projection_returns_none(tmp_path):
    path = _make_container(tmp_path)
    assert read_projection(path) is None


def test_empty_projection(tmp_path):
    """Edge case: zero-point projection round-trips."""
    path = _make_container(tmp_path, n=0)

    proj = Projection("empty-pid", [], np.empty((0, 2), dtype=np.float32), "trivial")
    pyr = build_pyramid(proj, n_levels=2)
    append_projection(path, proj, pyr)

    loaded = read_projection(path)
    assert loaded is not None
    proj2, pyr2 = loaded
    assert proj2.ids == []
    assert proj2.coords.shape == (0, 2)
    assert pyr2.point_count == 0


def test_negative_tile_indices(tmp_path):
    """Projections with negative tile indices round-trip correctly."""
    path = _make_container(tmp_path, n=200)

    rng = np.random.default_rng(99)
    coords = (rng.standard_normal((200, 2)) * 20).astype(np.float32)
    proj = Projection("neg-pid", list(range(200)), coords, "test")
    pyr = build_pyramid(proj, n_levels=3)

    has_negative = any(k[1] < 0 or k[2] < 0 for k in pyr.tiles)
    assert has_negative, "test data should produce negative tile indices"

    append_projection(path, proj, pyr)
    loaded = read_projection(path)
    assert loaded is not None
    _, pyr2 = loaded
    assert set(pyr2.tiles.keys()) == set(pyr.tiles.keys())


def test_overwrite_existing(tmp_path):
    """A second append overwrites the previous projection."""
    path = _make_container(tmp_path)

    proj1 = _make_projection(pid="first")
    pyr1 = build_pyramid(proj1, n_levels=2)
    append_projection(path, proj1, pyr1)

    proj2 = _make_projection(seed=99, pid="second")
    pyr2 = build_pyramid(proj2, n_levels=2)
    append_projection(path, proj2, pyr2)

    loaded = read_projection(path)
    assert loaded is not None
    assert loaded[0].projection_id == "second"
