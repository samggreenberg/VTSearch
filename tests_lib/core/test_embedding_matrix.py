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

from vtscore.datasets import registry
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


class TestMediaRevisionCounter:
    """Root-cause Pattern #4: the matrix cache keys on ``media_revision``.

    Structural mutations bump the counter transparently through
    :class:`MediasDict`; an in-place vector rewrite (same id set, different
    vectors) is invisible to a dict subclass and must be signalled by
    ``invalidate_embedding_matrix`` / ``bump_media_revision`` — this is the
    exact C4 miscompute the counter neutralises.
    """

    def _ctx(self) -> DatasetContext:
        ctx = DatasetContext("test_media_revision")
        for cid in (1, 2, 3):
            ctx.medias[cid] = {
                "id": cid,
                "embedder": "e5",
                "embeddings": {"e5": np.full(4, float(cid), dtype=np.float32)},
            }
        return ctx

    def test_structural_mutations_bump_revision(self):
        ctx = DatasetContext("test_bump_structural")
        assert ctx.media_revision == 0
        ctx.medias[1] = {"id": 1, "embedder": "e5", "embeddings": {"e5": np.ones(4, dtype=np.float32)}}
        after_add = ctx.media_revision
        assert after_add > 0
        del ctx.medias[1]
        assert ctx.media_revision > after_add

    def test_wholesale_reassignment_bumps_revision(self):
        ctx = DatasetContext("test_bump_reassign")
        before = ctx.media_revision
        ctx.medias = {1: {"id": 1, "embedder": "e5", "embeddings": {"e5": np.ones(4, dtype=np.float32)}}}
        assert ctx.media_revision > before
        # The assigned mapping is wrapped so it keeps bumping on later edits.
        after_assign = ctx.media_revision
        ctx.medias[2] = {"id": 2, "embedder": "e5", "embeddings": {"e5": np.full(4, 2.0, dtype=np.float32)}}
        assert ctx.media_revision > after_assign

    def test_no_bump_without_mutation(self):
        ctx = self._ctx()
        rev = ctx.media_revision
        _ = ctx.medias[1]  # a read must not bump
        _ = list(ctx.medias.keys())
        assert ctx.media_revision == rev

    def test_cache_reused_across_calls_at_same_revision(self):
        ctx = self._ctx()
        ids1, mat1 = get_embedding_matrix(ctx)
        ids2, mat2 = get_embedding_matrix(ctx)
        assert ids1 == ids2 == [1, 2, 3]
        # Same underlying array object → served from cache, not rebuilt.
        assert mat1 is mat2
        assert ctx._emb_matrix_revision == ctx.media_revision

    def test_inplace_vector_rewrite_needs_explicit_bump(self):
        """A dict-subclass can't see ``medias[cid][...] = vec``; until the
        counter is bumped the cache keeps serving the pre-rewrite matrix."""
        ctx = self._ctx()
        _, mat_before = get_embedding_matrix(ctx)
        assert mat_before[0, 0] == 1.0

        # Rewrite media 1's vector in place — no structural change, no bump.
        ctx.medias[1]["embeddings"]["e5"] = np.full(4, 42.0, dtype=np.float32)
        _, mat_stale = get_embedding_matrix(ctx)
        assert mat_stale[0, 0] == 1.0  # still the cached pre-rewrite row

        # The embed/clip stages signal the in-place change via this call.
        invalidate_embedding_matrix(ctx)
        _, mat_fresh = get_embedding_matrix(ctx)
        assert mat_fresh[0, 0] == 42.0

    def test_same_id_set_new_content_not_served_stale(self):
        """Reassigning ``medias`` to a fresh dict with the *same ids* but new
        vectors must invalidate the cache (the id-list key could not tell)."""
        ctx = self._ctx()
        _, mat_before = get_embedding_matrix(ctx)
        assert mat_before[0, 0] == 1.0

        ctx.medias = {
            cid: {"id": cid, "embedder": "e5", "embeddings": {"e5": np.full(4, float(cid) + 100.0, dtype=np.float32)}}
            for cid in (1, 2, 3)
        }
        ids, mat_after = get_embedding_matrix(ctx)
        assert ids == [1, 2, 3]
        assert mat_after[0, 0] == 101.0


class TestEmbeddingMatrixSidecar:
    """S1 (docs/plans/scalability.md): the mmap embedding-matrix sidecar.

    A dataset backed by a registry entry gets its primary matrix persisted
    as a ``<pkl_stem>.embids.npy`` / ``<pkl_stem>.embmat.npy`` pair after the
    first build, so a fresh ``DatasetContext`` for the same dataset can mmap
    it instead of rebuilding from per-item embeddings.
    """

    def _register(self, tmp_path, num_items: int = 3) -> str:
        pkl_dir = tmp_path / "saved"
        pkl_dir.mkdir(parents=True, exist_ok=True)
        entry = registry.register_dataset(
            name="sidecar-test",
            media_type="audio",
            num_items=num_items,
            pkl_path=str(pkl_dir / "ds_sidecar.pkl"),
        )
        return entry["id"]

    def _medias(self, n: int = 3) -> dict:
        return {
            cid: {"id": cid, "embedder": "e5", "embeddings": {"e5": np.full(4, float(cid), dtype=np.float32)}}
            for cid in range(1, n + 1)
        }

    def test_sidecar_written_after_first_build(self, tmp_path):
        dataset_id = self._register(tmp_path)
        ctx = DatasetContext(dataset_id)
        ctx.medias = self._medias()

        get_embedding_matrix(ctx)

        entry = registry.get_dataset(dataset_id)
        ids_path, mat_path = matrix_mod._emb_sidecar_paths(entry["pkl_path"])
        assert ids_path.is_file()
        assert mat_path.is_file()
        assert np.array_equal(np.load(ids_path), np.array([1, 2, 3], dtype=np.int64))

    def test_fresh_context_mmaps_the_sidecar(self, tmp_path):
        dataset_id = self._register(tmp_path)
        ctx1 = DatasetContext(dataset_id)
        ctx1.medias = self._medias()
        get_embedding_matrix(ctx1)  # writes the sidecar

        # A second context for the same registered dataset stands in for a
        # fresh process re-loading the same pkl from disk.
        ctx2 = DatasetContext(dataset_id)
        ctx2.medias = self._medias()
        ids, mat = get_embedding_matrix(ctx2)

        assert ids == [1, 2, 3]
        assert mat[0, 0] == 1.0
        assert mat[2, 0] == 3.0
        assert isinstance(ctx2._emb_matrix, np.memmap), "expected the mmap sidecar path, not a fresh rebuild"

    def test_unregistered_dataset_never_writes_a_sidecar(self, tmp_path):
        """No registry entry -> no pkl_path -> the mmap cache stays opt-in only."""
        ctx = DatasetContext("not_registered")
        ctx.medias = self._medias()

        get_embedding_matrix(ctx)

        assert not any(tmp_path.iterdir())

    def test_id_mismatch_sidecar_falls_back_to_live_rebuild(self, tmp_path):
        """A sidecar written for a different id set must never be trusted."""
        dataset_id = self._register(tmp_path)
        entry = registry.get_dataset(dataset_id)
        ids_path, mat_path = matrix_mod._emb_sidecar_paths(entry["pkl_path"])
        matrix_mod._atomic_save_npy(ids_path, np.array([1, 2, 99], dtype=np.int64))
        matrix_mod._atomic_save_npy(mat_path, np.zeros((3, 4), dtype=np.float32))

        ctx = DatasetContext(dataset_id)
        ctx.medias = self._medias()
        ids, mat = get_embedding_matrix(ctx)

        assert ids == [1, 2, 3]
        assert mat[0, 0] == 1.0  # live value, not the bogus zero-filled sidecar

    def test_dim_mismatch_sidecar_falls_back_to_live_rebuild(self, tmp_path):
        """A same-id-count sidecar with the wrong dimension must never be trusted."""
        dataset_id = self._register(tmp_path)
        entry = registry.get_dataset(dataset_id)
        ids_path, mat_path = matrix_mod._emb_sidecar_paths(entry["pkl_path"])
        matrix_mod._atomic_save_npy(ids_path, np.array([1, 2, 3], dtype=np.int64))
        matrix_mod._atomic_save_npy(mat_path, np.zeros((3, 99), dtype=np.float32))

        ctx = DatasetContext(dataset_id)
        ctx.medias = self._medias()
        ids, mat = get_embedding_matrix(ctx)

        assert ids == [1, 2, 3]
        assert mat.shape == (3, 4)
        assert mat[0, 0] == 1.0

    def test_invalidate_disables_sidecar_for_rest_of_context_lifetime(self, tmp_path):
        """Root-cause Pattern #4 for the sidecar: a same-id in-place vector
        rewrite must never be served stale from the on-disk mmap cache."""
        dataset_id = self._register(tmp_path)
        ctx1 = DatasetContext(dataset_id)
        ctx1.medias = self._medias()
        get_embedding_matrix(ctx1)  # writes the sidecar with row 0 == 1.0

        ctx2 = DatasetContext(dataset_id)
        ctx2.medias = self._medias()
        ids, mat = get_embedding_matrix(ctx2)
        assert mat[0, 0] == 1.0
        assert isinstance(ctx2._emb_matrix, np.memmap)

        # An in-place rewrite (re-embed/clip) with the same id set - the
        # sidecar's id-list check alone cannot see this.
        ctx2.medias[1]["embeddings"]["e5"] = np.full(4, 42.0, dtype=np.float32)
        invalidate_embedding_matrix(ctx2)
        assert ctx2._emb_sidecar_disabled is True

        ids, mat = get_embedding_matrix(ctx2)
        assert mat[0, 0] == 42.0, "must reflect the in-place rewrite, not the stale mmap'd sidecar"
        assert not isinstance(ctx2._emb_matrix, np.memmap)

        # The on-disk sidecar itself must also be refreshed (not just this
        # context's in-memory view), or a third, later-loading context would
        # still mmap the stale pre-rewrite values.
        entry = registry.get_dataset(dataset_id)
        _, mat_path = matrix_mod._emb_sidecar_paths(entry["pkl_path"])
        assert np.load(mat_path)[0, 0] == 42.0
