"""Regression tests for the embedding-matrix builders.

Covers audit M11: a media item with ``embedding=None`` must NOT silently
poison the matrix with a NaN row (numpy 2.x behaviour of
``matrix[i] = None``).  The builder raises ``ValueError`` naming the
offending cid instead.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from vtscore.embedding import matrix as matrix_mod
from vtscore.embedding.matrix import (
    get_embedding_matrix,
    get_embedding_matrix_for_snap,
    get_embedding_submatrix,
    invalidate_embedding_matrix,
)
from vtscore.state.core import DatasetContext, _state_lock, set_thread_dataset_context


class TestGetEmbeddingMatrixRaisesOnNoneEmbedding:
    def test_first_media_none_embedding(self):
        ctx = DatasetContext("test_first_none")
        ctx.medias[1] = {"id": 1, "embeddings": {}}
        ctx.medias[2] = {"id": 2, "embedder": "e5", "embeddings": {"e5": np.ones(4, dtype=np.float32)}}

        with pytest.raises(ValueError, match=r"media 1.*has no embedding"):
            get_embedding_matrix(ctx)

    def test_later_media_none_embedding(self):
        """Without the guard, numpy 2.x silently fills row i with NaN."""
        ctx = DatasetContext("test_later_none")
        ctx.medias[1] = {"id": 1, "embedder": "e5", "embeddings": {"e5": np.ones(4, dtype=np.float32)}}
        ctx.medias[2] = {"id": 2, "embedder": "e5", "embeddings": {"e5": np.full(4, 2.0, dtype=np.float32)}}
        ctx.medias[3] = {"id": 3, "embeddings": {}}

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
            ctx.medias[cid] = {
                "id": cid,
                "embedder": "e5",
                "embeddings": {"e5": np.full(4, float(cid), dtype=np.float32)},
            }
        ids, mat = get_embedding_matrix(ctx)
        assert ids == [1, 2, 3]
        assert mat.shape == (3, 4)
        # Row order matches sorted IDs.
        assert mat[0, 0] == 1.0
        assert mat[1, 0] == 2.0
        assert mat[2, 0] == 3.0


class TestGetEmbeddingSubmatrix:
    """Subset matrix builder for VTSBrowse subset projections."""

    def _ctx(self) -> DatasetContext:
        ctx = DatasetContext("test_submatrix")
        for cid in (1, 2, 3, 4):
            ctx.medias[cid] = {
                "id": cid,
                "embedder": "e5",
                "embeddings": {"e5": np.full(4, float(cid), dtype=np.float32)},
            }
        return ctx

    def test_returns_only_requested_ids_sorted(self):
        ctx = self._ctx()
        ids, mat = get_embedding_submatrix(ctx, [3, 1])
        assert ids == [1, 3]
        assert mat.shape == (2, 4)
        assert mat[0, 0] == 1.0
        assert mat[1, 0] == 3.0

    def test_dedups_and_drops_unknown_ids(self):
        ctx = self._ctx()
        ids, mat = get_embedding_submatrix(ctx, [2, 2, 99])
        assert ids == [2]
        assert mat.shape == (1, 4)

    def test_empty_when_no_match(self):
        ctx = self._ctx()
        ids, mat = get_embedding_submatrix(ctx, [99, 100])
        assert ids == []
        assert mat.shape == (0, 0)

    def test_raises_on_none_embedding(self):
        ctx = self._ctx()
        ctx.medias[2]["embeddings"] = {}
        with pytest.raises(ValueError, match=r"media 2.*has no embedding"):
            get_embedding_submatrix(ctx, [1, 2])

    def test_does_not_populate_cache(self):
        """Subset builds must not poison the context-wide matrix cache."""
        ctx = self._ctx()
        get_embedding_submatrix(ctx, [1, 2])
        assert ctx._emb_matrix is None
        assert ctx._emb_matrix_ids is None


class TestEmbedderAwareMatrix:
    """The matrix builders can source rows from a specific bound embedder.

    A multi-embedder dataset carries ``media["embeddings"]`` (dict keyed by
    embedder name); requesting an explicit name builds that embedder's matrix,
    while the default (no name) follows the primary mirror.
    """

    def _ctx(self) -> DatasetContext:
        ctx = DatasetContext("test_embedder_aware")
        for cid in (1, 2, 3):
            ctx.medias[cid] = {
                "id": cid,
                "embedder": "siglip",
                "embeddings": {
                    "siglip": np.full(4, float(cid), dtype=np.float32),
                    "dinov3_patch": np.full(4, float(cid) + 100.0, dtype=np.float32),
                },
            }
        return ctx

    def test_named_embedder_selects_that_matrix(self):
        ctx = self._ctx()
        ids, mat = get_embedding_matrix(ctx, "dinov3_patch")
        assert ids == [1, 2, 3]
        assert mat[0, 0] == 101.0
        assert mat[2, 0] == 103.0

    def test_default_follows_primary(self):
        ctx = self._ctx()
        ids, mat = get_embedding_matrix(ctx)
        assert mat[0, 0] == 1.0
        assert mat[2, 0] == 3.0

    def test_named_path_does_not_populate_cache(self):
        """The named path is uncached; the primary cache stays untouched."""
        ctx = self._ctx()
        get_embedding_matrix(ctx, "dinov3_patch")
        assert ctx._emb_matrix is None
        assert ctx._emb_matrix_ids is None

    def test_named_missing_vector_raises_with_embedder_name(self):
        ctx = self._ctx()
        del ctx.medias[2]["embeddings"]["dinov3_patch"]
        ctx.medias[2]["embedder"] = "siglip"  # primary is siglip's, not the requested embedder
        with pytest.raises(ValueError, match=r"media 2.*has no embedding for embedder 'dinov3_patch'"):
            get_embedding_matrix(ctx, "dinov3_patch")

    def test_primary_name_collapses_to_cache(self):
        """Routing hands a name even for single-embedder datasets; a name equal
        to the primary must collapse to the cached primary path, not the
        uncached named path - keeping the hot path byte-for-byte unchanged."""
        ctx = self._ctx()
        # "siglip" is the primary (the recorded ``embedder``).
        ids, mat = get_embedding_matrix(ctx, "siglip")
        assert ids == [1, 2, 3]
        # Primary's vectors (1,2,3), not dinov3's (101,...).
        assert mat[0, 0] == 1.0
        # ...and the cache was populated (the named path would not cache).
        assert ctx._emb_matrix is not None
        assert ctx._emb_matrix_ids == [1, 2, 3]

    def test_primary_name_snap_collapses_to_cache(self):
        ctx = self._ctx()
        set_thread_dataset_context(ctx)
        invalidate_embedding_matrix(ctx)
        snap = {cid: ctx.medias[cid] for cid in ctx.medias}
        ids, mat = get_embedding_matrix_for_snap(snap, "siglip")
        assert ids == [1, 2, 3]
        assert mat[0, 0] == 1.0
        assert ctx._emb_matrix is not None

    def test_submatrix_named_embedder(self):
        ctx = self._ctx()
        ids, mat = get_embedding_submatrix(ctx, [3, 1], "dinov3_patch")
        assert ids == [1, 3]
        assert mat[0, 0] == 101.0
        assert mat[1, 0] == 103.0

    def test_snap_named_embedder_matching_active_ctx(self):
        ctx = self._ctx()
        set_thread_dataset_context(ctx)
        invalidate_embedding_matrix(ctx)
        snap = {cid: ctx.medias[cid] for cid in ctx.medias}
        ids, mat = get_embedding_matrix_for_snap(snap, "dinov3_patch")
        assert ids == [1, 2, 3]
        assert mat[0, 0] == 101.0
        # Delegated to the context builder, but the named path must not cache.
        assert ctx._emb_matrix is None

    def test_snap_named_embedder_temp_dict(self):
        # Active ctx empty → snap can't match → fresh-build path.
        ctx = DatasetContext("test_snap_named_temp")
        set_thread_dataset_context(ctx)
        snap = {
            10: {"embedder": "siglip", "embeddings": {"dinov3_patch": np.full(4, 5.0, dtype=np.float32)}},
            11: {"embedder": "siglip", "embeddings": {"dinov3_patch": np.full(4, 6.0, dtype=np.float32)}},
        }
        ids, mat = get_embedding_matrix_for_snap(snap, "dinov3_patch")
        assert ids == [10, 11]
        assert mat[0, 0] == 5.0
        assert mat[1, 0] == 6.0


class TestGetEmbeddingMatrixForSnapRaisesOnNoneEmbedding:
    """Same guarantee for the snap helper, including the cross-dataset
    'temp dict' path that does NOT hit the cached active-ctx branch."""

    def test_snap_with_none_in_temp_dict(self):
        # Active ctx is empty so the snap can't match its key set; this
        # forces the fresh-build (uncached) path.
        ctx = DatasetContext("test_snap_temp")
        set_thread_dataset_context(ctx)

        snap = {
            10: {"embedder": "e5", "embeddings": {"e5": np.ones(4, dtype=np.float32)}},
            11: {"embeddings": {}},
        }
        with pytest.raises(ValueError, match=r"media 11.*has no embedding"):
            get_embedding_matrix_for_snap(snap)

    def test_snap_matching_active_ctx_with_none(self):
        """The matching-keys fast path delegates to ``get_embedding_matrix``
        on the active ctx; the same guard fires there too."""
        ctx = DatasetContext("test_snap_match")
        ctx.medias[1] = {"id": 1, "embedder": "e5", "embeddings": {"e5": np.ones(4, dtype=np.float32)}}
        ctx.medias[2] = {"id": 2, "embeddings": {}}
        set_thread_dataset_context(ctx)
        invalidate_embedding_matrix(ctx)

        snap = {cid: ctx.medias[cid] for cid in ctx.medias}
        with pytest.raises(ValueError, match=r"media 2.*has no embedding"):
            get_embedding_matrix_for_snap(snap)


class TestEmbeddingMatrixLockScoping:
    """The contiguous (N, D) build must run OUTSIDE ``_state_lock``.

    Holding the global state lock across ``_stack_embeddings`` lets a large
    (or multi-embedder named-path) build stall every other request's
    ``before_request`` state-sync, which also takes ``_state_lock``, on the
    single worker. These guard against re-introducing the in-lock build.
    """

    def _ctx(self) -> DatasetContext:
        ctx = DatasetContext("test_lock_scoping")
        for cid in (1, 2, 3, 4):
            ctx.medias[cid] = {
                "id": cid,
                "embedder": "e5",
                "embeddings": {"e5": np.full(4, float(cid), dtype=np.float32)},
            }
        return ctx

    def _assert_lock_free_during_build(self, monkeypatch, call) -> None:
        real_stack = matrix_mod._stack_embeddings
        lock_was_free = threading.Event()

        def probing_stack(sorted_ids, source, embedder_name):
            def grab():
                if _state_lock.acquire(timeout=2.0):
                    lock_was_free.set()
                    _state_lock.release()

            t = threading.Thread(target=grab)
            t.start()
            t.join(3.0)
            return real_stack(sorted_ids, source, embedder_name)

        monkeypatch.setattr(matrix_mod, "_stack_embeddings", probing_stack)
        call()
        assert lock_was_free.is_set(), "_state_lock was held across the matrix build"

    def test_get_embedding_matrix_builds_outside_state_lock(self, monkeypatch):
        ctx = self._ctx()  # fresh context -> cache miss -> reaches the build
        self._assert_lock_free_during_build(monkeypatch, lambda: get_embedding_matrix(ctx))

    def test_get_embedding_submatrix_builds_outside_state_lock(self, monkeypatch):
        ctx = self._ctx()
        self._assert_lock_free_during_build(monkeypatch, lambda: get_embedding_submatrix(ctx, [1, 2, 3, 4]))

    def test_stale_matrix_not_cached_when_medias_change_during_build(self, monkeypatch):
        """Phase-3 double-check: a media-set change during the unlocked build
        must NOT populate the primary cache with the now-stale matrix.
        """
        ctx = self._ctx()
        real_stack = matrix_mod._stack_embeddings

        def mutating_stack(sorted_ids, source, embedder_name):
            built = real_stack(sorted_ids, source, embedder_name)
            # A concurrent media insert lands while we build outside the lock.
            ctx.medias[99] = {"id": 99, "embedder": "e5", "embeddings": {"e5": np.full(4, 9.0, dtype=np.float32)}}
            return built

        monkeypatch.setattr(matrix_mod, "_stack_embeddings", mutating_stack)
        ids, mat = get_embedding_matrix(ctx)

        assert ids == [1, 2, 3, 4]
        assert mat.shape == (4, 4)
        # Cache must not hold the stale [1,2,3,4] build now that medias changed.
        assert ctx._emb_matrix_ids != [1, 2, 3, 4]
