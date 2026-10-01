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


class TestVectorisedTiles:
    """``raw_tiles`` is the load-time path; ``tile_rows`` + ``aggregate_vlad`` is the measured one."""

    def _features(self, n: int, seed: int) -> StructuralFeatures:
        rng = np.random.default_rng(seed)
        kp = np.zeros((n, 4), dtype=np.float32)
        kp[:, :2] = rng.random((n, 2))
        desc = (rng.random((n, SIFT_DESCRIPTOR_DIM)) * 255).astype(np.float32)
        return StructuralFeatures(keypoints=kp, descriptors=desc).compact()

    @pytest.mark.parametrize("n", [3000, 400, 5])
    def test_equals_the_reference_aggregation(self, n):
        from vtscore.media.structural import aggregate_vlad, load_vlad_codebook
        from vtscore.media.structural_tiles import raw_tiles, tile_rows

        feats = self._features(n, seed=n)
        fast, fast_boxes = raw_tiles(feats)
        ref, ref_boxes = tile_rows(feats.keypoints_f32(), feats.descriptors_f32(), load_vlad_codebook(), aggregate_vlad)
        assert fast.shape == ref.shape
        assert np.allclose(fast_boxes, ref_boxes)
        assert np.allclose(fast, ref, atol=1e-5)


def _doc_features(seed: int = 0, n: int = 800) -> StructuralFeatures:
    rng = np.random.default_rng(seed)
    kp = np.zeros((n, 4), dtype=np.float32)
    kp[:, :2] = rng.random((n, 2))
    return StructuralFeatures(
        keypoints=kp, descriptors=(rng.random((n, SIFT_DESCRIPTOR_DIM)) * 255).astype(np.float32)
    ).compact()


def _cache_projection(models_dir, dim: int = 16) -> None:
    """A small real fit, written where ``load_tile_projection`` looks."""
    from vtscore.media.structural_tiles import raw_tiles

    rows = np.concatenate([raw_tiles(_doc_features(s))[0] for s in range(3)])
    write_projection(fit_projection(rows, dim), projection_path(models_dir), {"fit_on": "unit test"})


class TestLoadBackfill:
    """Build step 2: ``sift_vlad_doc`` datasets get ``tile_vectors`` at load, derived from local features."""

    def _use_models_dir(self, monkeypatch, models_dir) -> None:
        import vtscore.media.structural_tiles as st

        real_load = st.load_tile_projection
        monkeypatch.setattr(st, "load_tile_projection", lambda *_a, **_k: real_load(models_dir, dim=16))
        st._load_cached.cache_clear()

    def test_the_document_embedder_opts_in_and_the_photo_one_does_not(self):
        from vtscore.media.image.embedder_sift_vlad import ImageSiftVladEmbedder
        from vtscore.media.image.embedder_sift_vlad_doc import ImageSiftVladDocEmbedder

        assert ImageSiftVladDocEmbedder().supports_tiled_stage1 is True
        assert ImageSiftVladEmbedder().supports_tiled_stage1 is False

    def test_forward_tiles_each_page_from_its_stored_features(self, monkeypatch, tmp_path):
        from vtscore.media.image.embedder_sift_vlad_doc import ImageSiftVladDocEmbedder
        from vtscore.media.structural_tiles import TileVectors

        _cache_projection(tmp_path)
        self._use_models_dir(monkeypatch, tmp_path)
        out = ImageSiftVladDocEmbedder().tile_vectors_forward_bulk(
            [{"local_features": _doc_features(1)}, {"local_features": None}]
        )
        assert isinstance(out[0], TileVectors) and out[0].vectors.shape[1] == 16
        assert out[0].vectors.dtype == np.float16 and out[0].count == out[0].boxes.shape[0]
        assert out[1] is None

    def test_a_missing_projection_warns_and_tiles_nothing(self, monkeypatch, tmp_path, caplog):
        from vtscore.media.image.embedder_sift_vlad_doc import ImageSiftVladDocEmbedder

        self._use_models_dir(monkeypatch, tmp_path / "empty")
        with caplog.at_level("WARNING"):
            out = ImageSiftVladDocEmbedder().tile_vectors_forward_bulk([{"local_features": _doc_features(1)}])
        assert out == [None]
        assert "fit_tile_projection.py" in caplog.text and "no tiled Stage 1" in caplog.text

    def test_a_reload_backfills_tiles_without_re_embedding(self, monkeypatch, tmp_path):
        """A pickle reload carries the VLAD vector but no side channels (they are never persisted)."""
        from vtscore.datasets.stages.embedding import embed_missing

        _cache_projection(tmp_path)
        self._use_models_dir(monkeypatch, tmp_path)
        media = {
            "media_type": "image",
            "embedder": "sift_vlad_doc",
            "embeddings": {"sift_vlad_doc": np.ones(8192, dtype=np.float32)},
            "local_features": _doc_features(2),
        }
        medias = {1: media}
        embed_missing(medias, "sift_vlad_doc")
        assert medias[1]["tile_vectors"].vectors.shape[1] == 16
        np.testing.assert_array_equal(medias[1]["embeddings"]["sift_vlad_doc"], np.ones(8192, dtype=np.float32))


class TestTileLayers:
    """#4415: a finer layer adds windows; the shipped default is one layer."""

    def test_the_shipped_default_is_the_measured_layer(self):
        from vtscore.media import structural_tiles as st

        assert st.TILE_LAYERS == ((st.TILE_W, st.TILE_H),)

    def test_a_fine_layer_adds_tiles_on_a_dense_page(self, monkeypatch):
        from vtscore.media import structural_tiles as st

        rng = np.random.default_rng(3)
        kp = np.zeros((6000, 4), dtype=np.float32)
        kp[:, :2] = rng.random((6000, 2))
        feats = StructuralFeatures(
            keypoints=kp, descriptors=(rng.random((6000, SIFT_DESCRIPTOR_DIM)) * 255).astype(np.float32)
        ).compact()
        coarse, _ = st.raw_tiles(feats)
        monkeypatch.setattr(st, "TILE_LAYERS", ((st.TILE_W, st.TILE_H), (st.TILE_W / 2, st.TILE_H / 2)))
        both, boxes = st.raw_tiles(feats)
        assert both.shape[0] > coarse.shape[0]
        assert np.allclose(both[: coarse.shape[0]], coarse, atol=1e-6)  # the coarse layer comes first, unchanged
        assert boxes[-1][2] - boxes[-1][0] == pytest.approx(st.TILE_W / 2, abs=1e-6)
