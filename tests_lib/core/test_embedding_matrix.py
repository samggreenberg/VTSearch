"""Regression tests for the embedding-matrix builders.

Covers audit M11: a media item with ``embedding=None`` must NOT silently
poison the matrix with a NaN row (numpy 2.x behaviour of
``matrix[i] = None``).  The builder raises ``ValueError`` naming the
offending cid instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.embedding.matrix import (
    get_embedding_matrix,
    get_embedding_matrix_for_snap,
    invalidate_embedding_matrix,
)
from vtscore.state.core import DatasetContext, set_thread_dataset_context


class TestGetEmbeddingMatrixRaisesOnNoneEmbedding:
    def test_first_media_none_embedding(self):
        ctx = DatasetContext("test_first_none")
        ctx.medias[1] = {"id": 1, "embedding": None}
        ctx.medias[2] = {"id": 2, "embedding": np.ones(4, dtype=np.float32)}

        with pytest.raises(ValueError, match=r"media 1.*has no embedding"):
            get_embedding_matrix(ctx)

    def test_later_media_none_embedding(self):
        """Without the guard, numpy 2.x silently fills row i with NaN."""
        ctx = DatasetContext("test_later_none")
        ctx.medias[1] = {"id": 1, "embedding": np.ones(4, dtype=np.float32)}
        ctx.medias[2] = {"id": 2, "embedding": np.full(4, 2.0, dtype=np.float32)}
        ctx.medias[3] = {"id": 3, "embedding": None}

        with pytest.raises(ValueError, match=r"media 3.*has no embedding"):
            get_embedding_matrix(ctx)

    def test_empty_medias_ok(self):
        ctx = DatasetContext("test_empty")
        ids, mat = get_embedding_matrix(ctx)
        assert ids == []
        assert mat.shape == (0, 0)

    def test_clean_medias_ok(self):
        ctx = DatasetContext("test_clean")
        for cid in (1, 2, 3):
            ctx.medias[cid] = {"id": cid, "embedding": np.full(4, float(cid), dtype=np.float32)}
        ids, mat = get_embedding_matrix(ctx)
        assert ids == [1, 2, 3]
        assert mat.shape == (3, 4)
        # Row order matches sorted IDs.
        assert mat[0, 0] == 1.0
        assert mat[1, 0] == 2.0
        assert mat[2, 0] == 3.0


class TestGetEmbeddingMatrixForSnapRaisesOnNoneEmbedding:
    """Same guarantee for the snap helper, including the cross-dataset
    'temp dict' path that does NOT hit the cached active-ctx branch."""

    def test_snap_with_none_in_temp_dict(self):
        # Active ctx is empty so the snap can't match its key set; this
        # forces the fresh-build (uncached) path.
        ctx = DatasetContext("test_snap_temp")
        set_thread_dataset_context(ctx)

        snap = {
            10: {"embedding": np.ones(4, dtype=np.float32)},
            11: {"embedding": None},
        }
        with pytest.raises(ValueError, match=r"media 11.*has no embedding"):
            get_embedding_matrix_for_snap(snap)

    def test_snap_matching_active_ctx_with_none(self):
        """The matching-keys fast path delegates to ``get_embedding_matrix``
        on the active ctx; the same guard fires there too."""
        ctx = DatasetContext("test_snap_match")
        ctx.medias[1] = {"id": 1, "embedding": np.ones(4, dtype=np.float32)}
        ctx.medias[2] = {"id": 2, "embedding": None}
        set_thread_dataset_context(ctx)
        invalidate_embedding_matrix(ctx)

        snap = {cid: ctx.medias[cid] for cid in ctx.medias}
        with pytest.raises(ValueError, match=r"media 2.*has no embedding"):
            get_embedding_matrix_for_snap(snap)
