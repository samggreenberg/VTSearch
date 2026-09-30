"""Tiled VLAD (#3928): the library the app's tiled Stage 1 runs on.

No SIFT runs here. What is pinned: the geometry the published numbers were
measured under, the projection's slicing identity, the multi-query segment max,
the Good-box query, and that a missing cached projection fails loudly instead of
letting Stage 1 fall back in silence.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from vtscore.media.structural import SIFT_DESCRIPTOR_DIM, StructuralFeatures
from vtscore.media.structural_tiles import (
    PROJECTION_NAME,
    TILE_DIM,
    Projection,
    TileProjectionMissing,
    box_vlad,
    fit_projection,
    fit_sample_ids,
    load_tile_projection,
    page_scores,
    page_scores_per_query,
    projection_path,
    sample_digest,
    starts_from_counts,
    tile_windows,
    write_projection,
)


def _sample(rows=200, dims=64, seed=0):
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(8, dims))
    return (rng.normal(size=(rows, 8)) @ basis + 0.01 * rng.normal(size=(rows, dims))).astype(np.float32)


class TestGeometry:
    def test_the_measured_layout(self):
        wins = tile_windows()
        assert len(wins) == 7 * 10  # 7 columns (stride 0.125) x 10 rows (stride 0.09)
        assert max(w[3] for w in wins) == pytest.approx(0.99, abs=1e-9)


class TestScoring:
    TILES = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [1.0, 0.0]], dtype=np.float32)

    def test_several_queries_score_a_page_by_its_best_pair(self):
        starts = starts_from_counts([2, 2])
        one = page_scores(self.TILES, starts, np.array([0.0, 1.0], dtype=np.float32))
        assert list(np.round(one, 3)) == [1.0, 0.8]
        # Adding a second query can only raise a page's score, never lower it.
        two = page_scores(self.TILES, starts, np.array([[0.0, 1.0], [0.6, 0.8]], dtype=np.float32))
        assert list(np.round(two, 3)) == [1.0, 1.0]

    def test_per_query_keeps_one_column_per_query(self):
        starts = starts_from_counts([2, 2])
        cols = page_scores_per_query(self.TILES, starts, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32).T)
        assert cols.shape == (2, 2)
        assert list(np.round(cols[:, 1], 3)) == [1.0, 0.8]


class TestBoxQuery:
    def test_only_keypoints_inside_the_box_describe_the_mark(self):
        rng = np.random.default_rng(0)
        kp = np.zeros((40, 4), dtype=np.float32)
        kp[:20, :2] = 0.1  # the boxed mark
        kp[20:, :2] = 0.9  # the rest of the page
        desc = rng.random((40, SIFT_DESCRIPTOR_DIM)).astype(np.float32)
        feats = StructuralFeatures(keypoints=kp, descriptors=desc)
        inside = box_vlad(feats, (0.0, 0.0, 0.2, 0.2))
        whole = box_vlad(feats, (0.0, 0.0, 1.0, 1.0))
        assert inside is not None and whole is not None
        assert not np.allclose(inside, whole)
        assert box_vlad(feats, (0.4, 0.4, 0.6, 0.6)) is None  # an empty box is no query


class TestCachedProjection:
    def test_missing_asset_fails_loudly_and_names_the_fix(self, tmp_path):
        with pytest.raises(TileProjectionMissing, match="fit_tile_projection.py"):
            load_tile_projection(tmp_path)

    def test_round_trip_with_provenance(self, tmp_path):
        proj = fit_projection(_sample(), 16)
        path = projection_path(tmp_path)
        write_projection(proj, path, {"fit_on": "unit test", "sample_digest": "abc"})
        meta = json.loads(path.with_suffix(".json").read_text())
        assert meta["name"] == PROJECTION_NAME and meta["dim"] == 16 and meta["fit_on"] == "unit test"
        back = load_tile_projection(tmp_path, dim=8)
        assert back.dim == 8
        assert np.allclose(back.components, proj.slice(8).components)

    def test_too_narrow_a_cached_fit_is_refused(self, tmp_path):
        write_projection(fit_projection(_sample(), 8), projection_path(tmp_path / "n"), {})
        with pytest.raises(TileProjectionMissing, match="8-wide"):
            load_tile_projection(tmp_path / "n", dim=TILE_DIM)

    def test_the_fit_sample_is_deterministic(self):
        ids = [f"p{i:04d}" for i in range(1000)]
        a = fit_sample_ids(list(reversed(ids)), 10)
        assert a == fit_sample_ids(ids, 10) and len(a) == 10
        assert sample_digest(a) == sample_digest(list(a))

    def test_projection_slices_like_a_narrow_fit(self):
        sample = _sample()
        assert np.allclose(
            np.abs(fit_projection(sample, 16).slice(4).components),
            np.abs(fit_projection(sample, 4).components),
            atol=1e-4,
        )
        assert isinstance(fit_projection(sample, 4), Projection)
