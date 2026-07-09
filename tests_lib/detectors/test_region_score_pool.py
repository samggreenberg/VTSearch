"""Region (patch) scoring path of ``_score_all_media`` + its matrix cache.

DINOv3-style patch datasets expose ``patch_regions`` (a list of
:class:`RegionVector`s) per media.  Scoring flattens every (media, region)
pair into one matrix, runs a single MLP forward pass, then max-pools back
down to one score + winning-region index per media.

The matrix used to be rebuilt from scratch on every vote - a
hundreds-of-thousands-row Python loop plus a multi-GB ``np.stack`` - which,
running in the background training thread, stalled the next vote's request.
These tests pin (a) the max-pool / best-region correctness and (b) that the
flattened matrix is cached on the dataset context and reused until the
media-id set changes.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
import torch.nn as nn

from vtscore.detectors.training import _score_all_media
from vtscore.embedding.matrix import get_region_matrix_for_snap, invalidate_embedding_matrix
from vtscore.embedding.media_vectors import media_embedding
from vtscore.media.patch_embed import RegionVector
from vtscore.state.core import get_active_context

DIM = 8


def _region(rng: np.random.Generator, ri: int) -> RegionVector:
    """A RegionVector with a deterministic L2-normalised vector."""
    vec = rng.standard_normal(DIM).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-8
    return RegionVector(box=(0.0, 0.0, ri / 10.0 + 0.1, 1.0), vec=vec)


def _region_media(media_id: int, n_regions: int) -> dict:
    rng = np.random.default_rng(media_id)
    regions = [_region(rng, ri) for ri in range(n_regions)]
    return {
        "id": media_id,
        "media_type": "image",
        "embedder": "dinov3_patch",
        # Image-level embedding is the fallback row for region-less media.
        "embeddings": {"dinov3_patch": rng.standard_normal(DIM).astype(np.float32)},
        "patch_regions": regions,
    }


def _linear_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(DIM, 1)).eval()


class TestRegionMaxPool:
    def test_scores_and_best_region_match_manual_pool(self):
        # Media 1 & 2 carry regions; media 3 has none -> image-level fallback.
        clips = {
            1: _region_media(1, 3),
            2: _region_media(2, 4),
            3: {**_region_media(3, 0), "patch_regions": []},
        }
        model = _linear_model()

        all_ids, scores, best_region = _score_all_media(cast(nn.Sequential, model), clips)

        assert all_ids == [1, 2, 3]

        # Recompute the expected per-media max-pool independently.
        for idx, cid in enumerate(all_ids):
            regions = clips[cid]["patch_regions"]
            if regions:
                vecs = np.stack([r.vec for r in regions]).astype(np.float32)
            else:
                vecs = media_embedding(clips[cid])[None, :].astype(np.float32)
            with torch.no_grad():
                logits = model(torch.from_numpy(vecs)).squeeze(-1)
                row_scores = torch.sigmoid(logits).numpy()
            expected_best = int(np.argmax(row_scores))
            assert abs(scores[idx] - float(row_scores.max())) < 1e-6
            assert best_region[idx] == expected_best

        # The region-less media's winning region is always 0 (its single row).
        assert best_region[2] == 0


class TestRegionMatrixFallbackSpace:
    """A region-less media's fallback row must be the *patch*-space vector.

    On a dataset that mixes patch-capable and patch-less media (e.g. two
    datasets combined, or a media type the patch embedder can't process),
    the primary embedder can differ from the one that produced the region
    vectors.  Stacking the primary vector alongside patch-space region rows
    in the same matrix would silently score a region-less media in the wrong
    space; the fallback must read the patch embedder's own vector instead.
    """

    def test_fallback_row_reads_patch_embedder_not_primary(self):
        rng = np.random.default_rng(0)
        region_vec = rng.standard_normal(DIM).astype(np.float32)
        region_vec /= np.linalg.norm(region_vec) + 1e-8
        primary_vec = rng.standard_normal(DIM).astype(np.float32)
        primary_vec /= np.linalg.norm(primary_vec) + 1e-8
        patch_vec = rng.standard_normal(DIM).astype(np.float32)
        patch_vec /= np.linalg.norm(patch_vec) + 1e-8

        clips = {
            1: {
                "id": 1,
                "media_type": "image",
                "embedder": "dinov3_patch",
                "embeddings": {"dinov3_patch": rng.standard_normal(DIM).astype(np.float32)},
                "patch_regions": [RegionVector(box=(0.0, 0.0, 1.0, 1.0), vec=region_vec)],
            },
            # Region-less media bound under a *different* primary embedder
            # (e.g. text-capable), but also carries a vector for the patch
            # embedder - the mixed-media-type case.
            2: {
                "id": 2,
                "media_type": "image",
                "embedder": "siglip",
                "embeddings": {"siglip": primary_vec, "dinov3_patch": patch_vec},
                "patch_regions": None,
            },
        }

        _all_ids, region_matrix, media_index, _region_index = get_region_matrix_for_snap(clips)

        # Media 2's single fallback row is its "dinov3_patch" vector, not its
        # "siglip" primary vector.
        fallback_row = region_matrix[media_index.tolist().index(1)]
        np.testing.assert_array_equal(fallback_row, patch_vec)
        assert not np.array_equal(fallback_row, primary_vec)

    def test_fallback_row_missing_patch_vector_raises(self):
        rng = np.random.default_rng(1)
        region_vec = rng.standard_normal(DIM).astype(np.float32)
        region_vec /= np.linalg.norm(region_vec) + 1e-8

        clips = {
            1: {
                "id": 1,
                "media_type": "image",
                "embedder": "dinov3_patch",
                "embeddings": {"dinov3_patch": rng.standard_normal(DIM).astype(np.float32)},
                "patch_regions": [RegionVector(box=(0.0, 0.0, 1.0, 1.0), vec=region_vec)],
            },
            # Region-less media with no vector at all under the patch
            # embedder - must raise loudly rather than mix in its primary.
            2: {
                "id": 2,
                "media_type": "video",
                "embedder": "siglip",
                "embeddings": {"siglip": rng.standard_normal(DIM).astype(np.float32)},
                "patch_regions": None,
            },
        }

        try:
            get_region_matrix_for_snap(clips)
        except ValueError as exc:
            assert "dinov3_patch" in str(exc)
            assert "2" in str(exc)
        else:
            raise AssertionError("expected ValueError for missing patch-embedder vector")


class TestRegionMatrixCache:
    def test_matrix_cached_and_invalidated(self):
        ctx = get_active_context()
        # Replace the active dataset's medias with region media so the snap
        # key set matches the context (the condition for caching).
        ctx.medias.clear()
        ctx.medias[1] = _region_media(1, 3)
        ctx.medias[2] = _region_media(2, 2)
        invalidate_embedding_matrix(ctx)

        snap = dict(ctx.medias)
        ids1, matrix1, media_idx1, region_idx1 = get_region_matrix_for_snap(snap)

        # 3 + 2 = 5 flattened rows, mapped back to media indices 0/1.
        assert ids1 == [1, 2]
        assert matrix1.shape == (5, DIM)
        assert media_idx1.tolist() == [0, 0, 0, 1, 1]
        assert region_idx1.tolist() == [0, 1, 2, 0, 1]

        # Second call reuses the very same cached arrays (no rebuild).
        _ids2, matrix2, media_idx2, _region_idx2 = get_region_matrix_for_snap(dict(ctx.medias))
        assert matrix2 is matrix1
        assert media_idx2 is media_idx1
        assert ctx._region_matrix is matrix1

        # Invalidation drops the cache; the next call rebuilds a fresh array.
        invalidate_embedding_matrix(ctx)
        assert ctx._region_matrix is None
        _ids3, matrix3, _media_idx3, _region_idx3 = get_region_matrix_for_snap(dict(ctx.medias))
        assert matrix3 is not matrix1
        np.testing.assert_array_equal(matrix3, matrix1)
