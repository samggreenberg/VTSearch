"""Patch scoring path of ``_score_all_media`` + its matrix cache.

DINOv3-style patch datasets expose a raw ``patch_grid`` per media.  Scoring
flattens every media's score-row stack - the image-level (CLS) vector plus every
raw patch, :func:`~vtscore.embedding.matrix.media_score_rows` - into one matrix,
runs one chunked MLP forward pass, then max-pools back down to one score +
winning-row index per media.

The matrix used to be rebuilt from scratch on every vote - a
hundreds-of-thousands-row Python loop plus a multi-GB ``np.stack`` - which,
running in the background training thread, stalled the next vote's request.
These tests pin (a) the max-pool / best-row correctness, (b) that the flattened
matrix is cached on the dataset context and reused until the media-id set
changes, and (c) that it stays float16 - MaxPatch stacks ~8x the rows the old
HAC tree did, so a float32 matrix would multiply the resident bytes with it.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
import torch.nn as nn

from vtscore.detectors.training import _score_all_media
from vtscore.embedding.matrix import (
    get_region_matrix_for_snap,
    invalidate_embedding_matrix,
    media_row_box,
    media_score_rows,
)
from vtscore.state.core import get_active_context

DIM = 8
GRID = 2  # 2x2 patch grid -> 1 + 4 = 5 score rows per media


def _unit(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-8)


def _grid_media(media_id: int, *, with_grid: bool = True) -> dict:
    """A synthetic patch media: CLS vector + (optionally) a raw ``patch_grid``."""
    rng = np.random.default_rng(media_id)
    media = {
        "id": media_id,
        "media_type": "image",
        "embedder": "dinov3_patch",
        # The image-level vector is row 0 of the score stack (and the only row
        # for a grid-less media).
        "embeddings": {"dinov3_patch": _unit(rng.standard_normal(DIM).astype(np.float32))},
    }
    if with_grid:
        media["patch_grid"] = _unit(rng.standard_normal((GRID, GRID, DIM)).astype(np.float32)).astype(np.float16)
    return media


def _linear_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(DIM, 1)).eval()


class TestScoreRowLayout:
    def test_row_zero_is_the_image_vector_then_patches_row_major(self):
        media = _grid_media(1)
        rows = media_score_rows(media, "dinov3_patch")
        assert rows.shape == (1 + GRID * GRID, DIM)
        np.testing.assert_allclose(rows[0], media["embeddings"]["dinov3_patch"], rtol=1e-3)
        flat = np.asarray(media["patch_grid"], dtype=np.float32).reshape(-1, DIM)
        np.testing.assert_allclose(rows[1:], flat, rtol=1e-3)

    def test_row_boxes_invert_the_layout(self):
        media = _grid_media(1)
        # Row 0 covers the whole image; rows 1.. are the grid cells row-major.
        assert media_row_box(media, 0) == [0.0, 0.0, 1.0, 1.0]
        assert media_row_box(media, 1) == [0.0, 0.0, 0.5, 0.5]
        assert media_row_box(media, 2) == [0.5, 0.0, 1.0, 0.5]
        assert media_row_box(media, 3) == [0.0, 0.5, 0.5, 1.0]
        assert media_row_box(media, 4) == [0.5, 0.5, 1.0, 1.0]
        # Out of range clamps rather than raising; a grid-less media has no box.
        assert media_row_box(media, 99) is None
        assert media_row_box(_grid_media(2, with_grid=False), 0) is None


class TestRegionMaxPool:
    def test_scores_and_best_row_match_manual_pool(self):
        # Media 1 & 2 carry grids; media 3 has none -> image-level fallback.
        clips = {
            1: _grid_media(1),
            2: _grid_media(2),
            3: _grid_media(3, with_grid=False),
        }
        model = _linear_model()

        all_ids, scores, best_region = _score_all_media(cast(nn.Sequential, model), clips)

        assert all_ids == [1, 2, 3]

        # Recompute the expected per-media max-pool independently.
        for idx, cid in enumerate(all_ids):
            vecs = media_score_rows(clips[cid], "dinov3_patch")
            with torch.no_grad():
                logits = model(torch.from_numpy(vecs)).squeeze(-1)
                row_scores = torch.sigmoid(logits).numpy()
            expected_best = int(np.argmax(row_scores))
            assert abs(scores[idx] - float(row_scores.max())) < 1e-3
            assert best_region[idx] == expected_best

        # The grid-less media's winning row is always 0 (its single row).
        assert best_region[2] == 0


class TestRegionMatrixFallbackSpace:
    """A grid-less media's fallback row must be the *patch*-space vector.

    On a dataset that mixes patch-capable and patch-less media (e.g. two
    datasets combined, or a media type the patch embedder can't process),
    the primary embedder can differ from the one that produced the patch
    vectors.  Stacking the primary vector alongside patch-space rows
    in the same matrix would silently score a grid-less media in the wrong
    space; the fallback must read the patch embedder's own vector instead.
    """

    def test_fallback_row_reads_patch_embedder_not_primary(self):
        rng = np.random.default_rng(0)
        primary_vec = _unit(rng.standard_normal(DIM).astype(np.float32))
        patch_vec = _unit(rng.standard_normal(DIM).astype(np.float32))

        clips = {
            1: _grid_media(1),
            # Grid-less media bound under a *different* primary embedder
            # (e.g. text-capable), but also carries a vector for the patch
            # embedder - the mixed-media-type case.
            2: {
                "id": 2,
                "media_type": "image",
                "embedder": "siglip",
                "embeddings": {"siglip": primary_vec, "dinov3_patch": patch_vec},
            },
        }

        _all_ids, region_matrix, media_index, _region_index = get_region_matrix_for_snap(clips)

        # Media 2's single fallback row is its "dinov3_patch" vector, not its
        # "siglip" primary vector.
        fallback_row = np.asarray(region_matrix[media_index.tolist().index(1)], dtype=np.float32)
        np.testing.assert_allclose(fallback_row, patch_vec, atol=1e-3)
        assert not np.allclose(fallback_row, primary_vec, atol=1e-3)

    def test_fallback_row_missing_patch_vector_raises(self):
        rng = np.random.default_rng(1)

        clips = {
            1: _grid_media(1),
            # Grid-less media with no vector at all under the patch
            # embedder - must raise loudly rather than mix in its primary.
            2: {
                "id": 2,
                "media_type": "video",
                "embedder": "siglip",
                "embeddings": {"siglip": _unit(rng.standard_normal(DIM).astype(np.float32))},
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
        # Replace the active dataset's medias with patch media so the snap
        # key set matches the context (the condition for caching).
        ctx.medias.clear()
        ctx.medias[1] = _grid_media(1)
        ctx.medias[2] = _grid_media(2)
        invalidate_embedding_matrix(ctx)

        snap = dict(ctx.medias)
        ids1, matrix1, media_idx1, region_idx1 = get_region_matrix_for_snap(snap)

        # (1 CLS + 4 patches) x 2 medias = 10 flattened rows, mapped back to
        # media indices 0/1, each numbered 0..4 within its own stack.
        rows_per_media = 1 + GRID * GRID
        assert ids1 == [1, 2]
        assert matrix1.shape == (2 * rows_per_media, DIM)
        # Kept float16 (the grid's own dtype): MaxPatch stacks ~8x the rows the
        # HAC tree did, so upcasting here would multiply the resident bytes too.
        assert matrix1.dtype == np.float16
        assert media_idx1.tolist() == [0] * rows_per_media + [1] * rows_per_media
        assert region_idx1.tolist() == list(range(rows_per_media)) * 2

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
