"""Regression tests for C4: embedding-matrix cache must be invalidated
after the clip and dedup stages of the dataset-load pipeline.

Both ``_apply_clipper_stage`` and ``_collapse_duplicates_stage`` mutate
``ctx.medias`` in ways the matrix cache's id-keyed check cannot detect:

- ``_apply_clipper_stage`` (via ``clipper_chain.apply_chain_to_clips``)
  renumbers media IDs starting at 1 while replacing embeddings in place.
  A 1-to-1 chain produces the same id list as before clipping, so the
  cached matrix's ``cached_ids == sorted_ids`` check fails to fire and
  the next reader sees stale rows.

- ``_collapse_duplicates_stage`` removes duplicate media items. The id
  set changes, but only by removal — a cached matrix built before dedup
  still holds rows for the deleted ids that no longer correspond to any
  live media.

Both stages must drop the cache so the next ``get_embedding_matrix``
rebuilds against the live medias dict.
"""

from __future__ import annotations

import numpy as np

from vtscore.concurrency.progress import LoadingTasksTracker
from vtscore.datasets.load_pipeline import (
    _apply_clipper_stage,
    _collapse_duplicates_stage,
    _drop_none_embeddings_stage,
)
from vtscore.embedding.matrix import get_embedding_matrix
from vtscore.state.core import DatasetContext


def _populate_matrix(ctx: DatasetContext) -> np.ndarray:
    """Force the matrix cache to be built; return the cached matrix."""
    _, matrix = get_embedding_matrix(ctx)
    assert ctx._emb_matrix is matrix
    assert ctx._emb_matrix_ids == sorted(ctx.medias.keys())
    return matrix


def _make_tracker():
    return LoadingTasksTracker().create_task("test_task", "test")


class TestCollapseDuplicatesStageInvalidatesMatrix:
    def test_cache_dropped_after_dedup(self):
        ctx = DatasetContext("test_dedup_cache")
        ctx.medias[1] = {"id": 1, "md5": "aaa", "embedding": np.ones(4, dtype=np.float32)}
        ctx.medias[2] = {"id": 2, "md5": "aaa", "embedding": np.full(4, 2.0, dtype=np.float32)}
        ctx.medias[3] = {"id": 3, "md5": "bbb", "embedding": np.full(4, 3.0, dtype=np.float32)}

        _populate_matrix(ctx)

        _collapse_duplicates_stage(ctx, _make_tracker())

        assert ctx._emb_matrix_ids is None
        assert ctx._emb_matrix is None

        ids, matrix = get_embedding_matrix(ctx)
        assert ids == sorted(ctx.medias.keys())
        assert matrix.shape[0] == len(ctx.medias)


class TestApplyClipperStageInvalidatesMatrix:
    """A no-op clipper still re-stamps origins; once a real clipper renumbers
    ids, the cache would silently survive. Use a default clipper (no
    structural change) to exercise the stage's invalidation contract
    without depending on heavy media bytes."""

    def test_cache_dropped_after_clipper_stage(self):
        ctx = DatasetContext("test_clipper_cache")
        rng = np.random.default_rng(0)
        for cid in (1, 2, 3):
            ctx.medias[cid] = {
                "id": cid,
                "type": "audio",
                "filename": f"a_{cid}.wav",
                "md5": f"md5_{cid}",
                "embedding": rng.standard_normal(4).astype(np.float32),
                "origin": {"importer": "server_folder", "params": {"path": "/x"}},
                "origin_name": f"a_{cid}.wav",
            }

        _populate_matrix(ctx)

        # ``sound_default`` is a stamp-only clipper: no embedding rewrite,
        # no id renumbering. The stage must still invalidate so that a
        # real clip chain (which does rewrite embeddings in place) can't
        # leave stale rows in the cache.
        _apply_clipper_stage(ctx, _make_tracker(), "sound_default", None, None)

        assert ctx._emb_matrix_ids is None
        assert ctx._emb_matrix is None

    def test_noop_when_no_clipper(self):
        """No clipper, no chain: stage is a no-op and must not touch the cache."""
        ctx = DatasetContext("test_noop_cache")
        ctx.medias[1] = {"id": 1, "md5": "x", "embedding": np.ones(4, dtype=np.float32)}

        _populate_matrix(ctx)
        cached_matrix = ctx._emb_matrix

        _apply_clipper_stage(ctx, _make_tracker(), "", None, None)

        # No mutation happened; cache survives.
        assert ctx._emb_matrix is cached_matrix


class TestDropNoneEmbeddingsStage:
    """Regression for M11 finalize step.

    Audit M11 was framed as "stale media when some embeddings are None"
    via silent zip truncation; the real symptom under numpy 2.x is that
    a None embedding becomes a NaN row, propagates through the MLP, and
    forces an always-False threshold compare so the media silently lands
    in negative_hits with a NaN score (and ``NaN`` in the JSON response).
    The drop stage in ``_run_origin_load_in_background`` ensures None
    embeddings never reach the matrix builder, dedup, diversity tree,
    or registry — and surfaces the dropped count to the user via the
    progress tracker so the load row reflects the real N.
    """

    def test_drops_medias_with_none_embedding(self):
        ctx = DatasetContext("test_drop_none")
        ctx.medias[1] = {"id": 1, "embedding": np.ones(4, dtype=np.float32)}
        ctx.medias[2] = {"id": 2, "embedding": None}  # embedder failed silently
        ctx.medias[3] = {"id": 3, "embedding": np.full(4, 3.0, dtype=np.float32)}
        ctx.medias[4] = {"id": 4, "embedding": None}

        _populate_matrix(ctx)

        _drop_none_embeddings_stage(ctx, _make_tracker())

        assert set(ctx.medias.keys()) == {1, 3}
        # Cache must be invalidated since the id set changed.
        assert ctx._emb_matrix is None
        assert ctx._emb_matrix_ids is None

    def test_no_op_when_all_embeddings_present(self):
        ctx = DatasetContext("test_drop_none_noop")
        ctx.medias[1] = {"id": 1, "embedding": np.ones(4, dtype=np.float32)}
        ctx.medias[2] = {"id": 2, "embedding": np.full(4, 2.0, dtype=np.float32)}

        cached_matrix = _populate_matrix(ctx)

        _drop_none_embeddings_stage(ctx, _make_tracker())

        # No mutation: cache must NOT be invalidated.
        assert set(ctx.medias.keys()) == {1, 2}
        assert ctx._emb_matrix is cached_matrix

    def test_empty_medias_ok(self):
        ctx = DatasetContext("test_drop_none_empty")
        # Should not raise.
        _drop_none_embeddings_stage(ctx, _make_tracker())
        assert ctx.medias == {}
