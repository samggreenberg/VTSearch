"""A dataset that arrives partly embedded must leave the embed stage single-space.

Issue #3798.  A third-party importer ships vectors for *some* of its items
(``content_vectors`` / ``custom_metadata_map``, which the folder loader stores
under the blank :data:`UNKNOWN_EMBEDDER_KEY` sentinel because the importer did
not name the embedder) and leaves the rest for the framework to embed.  When the
load names no embedder - the CLI never does, and the GUI need not - the stage
resolved the media-type default, embedded the missing items under that name,
and left the pre-computed vectors under the sentinel.  The dataset was then two
"spaces" by name while being one space in fact, and the first thing to ask the
matrix layer for the routed embedder (Browse, Train, text sort) raised::

    media 7 has no embedding for embedder 'custom-embedder' (embedding=None)

These tests mock the importer's output shape directly - the medias dict as the
folder loader / a ``fetch_record`` importer hands it to the embed stage - with
some items carrying a vector and some not, and a fake embedder standing in for
the plugin's.
"""

from __future__ import annotations

import logging
import unittest.mock as mock

import numpy as np
import pytest

from vtscore.concurrency.progress import LoadingTasksTracker
from vtscore.datasets.stages.embedding import _embed_missing_stage, embed_missing
from vtscore.datasets.stages.finalize import _drop_none_embeddings_stage
from vtscore.embedding.matrix import _stack_embeddings, _uniform_primary_embedder, get_embedding_matrix
from vtscore.embedding.media_vectors import UNKNOWN_EMBEDDER_KEY, init_embeddings, media_embedding
from vtscore.embedding.precomputed import MismatchedVectorError
from vtscore.state.core import DatasetContext

DIM = 4
EMBEDDER = "custom_embedder"


def _fake_embedder(name: str = EMBEDDER, dim: int = DIM, *, fail_ids: frozenset[int] = frozenset()):
    """A MagicMock embedder whose bulk pass returns ``None`` for *fail_ids*."""
    emb = mock.MagicMock()
    emb.name = name
    emb.media_type_id = "audio"
    emb._model = True
    emb.supports_text = True
    emb.supports_patch_regions = False
    emb.supports_geometric_verification = False
    emb.embedding_dim = None

    def _bulk(medias):
        return [None if m["id"] in fail_ids else np.full(dim, float(m["id"]), dtype=np.float32) for m in medias]

    emb.embed_media_bulk.side_effect = _bulk
    return emb


def _media(mid: int, vec: np.ndarray | None) -> dict:
    """One media dict in the shape the folder loader emits for an importer.

    ``init_embeddings("", vec)`` is exactly what ``_build_folder_media_data``
    does with a ``content_vectors`` hit: the vector lands under the blank
    sentinel key because the importer named no embedder.
    """
    return {
        "id": mid,
        "media_type": "audio",
        "filename": f"{mid}.wav",
        "md5": f"md5-{mid}",
        "embedder": "",
        "embeddings": init_embeddings("", vec),
        "media_path": f"/tmp/{mid}.wav",
    }


def _partial_medias(rng: np.random.Generator, *, pre: tuple[int, ...], missing: tuple[int, ...]) -> tuple[dict, dict]:
    """Medias where *pre* carry a pre-computed vector and *missing* carry none."""
    vecs = {i: rng.standard_normal(DIM).astype(np.float32) for i in pre}
    medias = {i: _media(i, vecs[i]) for i in pre}
    medias.update({i: _media(i, None) for i in missing})
    return dict(sorted(medias.items())), vecs


class _registered:
    """Register *emb* in the process embedder registry for the block."""

    def __init__(self, emb) -> None:
        self._emb = emb

    def __enter__(self):
        from vtscore.media import _embedder_registry, register_embedder

        self._registry = _embedder_registry
        self._prev = _embedder_registry.get(self._emb.name)
        register_embedder(self._emb)
        return self._emb

    def __exit__(self, *exc):
        if self._prev is None:
            self._registry.pop(self._emb.name, None)
        else:
            self._registry[self._emb.name] = self._prev
        return False


def _tracker():
    return LoadingTasksTracker().create_task("test_partial_import", "test")


class TestPartialImportNoPick:
    """No embedder named for the load (CLI, or a GUI import with no pick)."""

    def test_precomputed_vectors_are_keyed_under_the_resolved_embedder(self):
        emb = _fake_embedder()
        medias, vecs = _partial_medias(np.random.default_rng(3798), pre=(1, 3), missing=(2,))

        with mock.patch("vtscore.media.embedders_for_type", return_value=[emb]):
            embed_missing(medias, "")

        # Only the item with no vector was embedded; the shipped vectors were kept.
        assert emb.embed_media_bulk.call_count == 1
        assert [m["id"] for m in emb.embed_media_bulk.call_args.args[0]] == [2]
        for i in (1, 3):
            np.testing.assert_array_equal(medias[i]["embeddings"][EMBEDDER], vecs[i])

        # ...and every media now lives under the one embedder that produced the
        # load, so nothing downstream can find a media "missing" that space.
        for m in medias.values():
            assert m["embedder"] == EMBEDDER
            assert set(m["embeddings"]) == {EMBEDDER}
            assert media_embedding(m, EMBEDDER) is not None
        assert _uniform_primary_embedder(medias) == EMBEDDER

    def test_matrix_for_the_resolved_embedder_builds(self):
        """The exact call Browse / Train / text sort make after the import."""
        emb = _fake_embedder()
        medias, _ = _partial_medias(np.random.default_rng(1), pre=(1, 3), missing=(2,))

        with mock.patch("vtscore.media.embedders_for_type", return_value=[emb]):
            embed_missing(medias, "")

        matrix = _stack_embeddings(sorted(medias), medias, EMBEDDER)
        assert matrix.shape == (3, DIM)

    def test_pipeline_stage_then_routed_matrix(self):
        """End to end through the load-pipeline stages and the routing table.

        ``_embed_missing_stage`` is what the GUI import runs; ``routed_embedder``
        plus ``get_embedding_matrix`` is what Browse asks for first.  Before the
        fix the second call raised ``media 2 has no embedding for embedder``.
        """
        emb = _fake_embedder()
        ctx = DatasetContext("test_partial_import_ctx")
        medias, _ = _partial_medias(np.random.default_rng(2), pre=(2, 3), missing=(1,))
        ctx.medias.update(medias)

        # Registered so the routing table's capability lookup resolves the
        # fake; patched so the media-type default resolves to it as well.
        with _registered(emb), mock.patch("vtscore.media.embedders_for_type", return_value=[emb]):
            _embed_missing_stage(ctx, _tracker(), [""])
            _drop_none_embeddings_stage(ctx, _tracker())
            routed = ctx.routed_embedder("score")
            assert routed == EMBEDDER
            ids, matrix = get_embedding_matrix(ctx, routed)

        assert ids == [1, 2, 3]
        assert matrix.shape == (3, DIM)

    def test_already_mixed_dataset_is_healed_on_reload(self):
        """A dataset saved in the broken shape loads clean once the fix exists.

        Nothing is missing, but some media already carry the resolved embedder's
        name, so the nameless ones must be its vectors too.
        """
        emb = _fake_embedder()
        rng = np.random.default_rng(4)
        nameless = rng.standard_normal(DIM).astype(np.float32)
        named = rng.standard_normal(DIM).astype(np.float32)
        medias = {
            1: _media(1, nameless),
            2: {**_media(2, None), "embedder": EMBEDDER, "embeddings": {EMBEDDER: named}},
        }

        with mock.patch("vtscore.media.get_embedder", return_value=emb):
            embed_missing(medias, "")

        assert emb.embed_media_bulk.call_count == 0
        assert set(medias[1]["embeddings"]) == {EMBEDDER}
        assert medias[1]["embedder"] == EMBEDDER
        np.testing.assert_array_equal(medias[1]["embeddings"][EMBEDDER], nameless)
        assert _uniform_primary_embedder(medias) == EMBEDDER

    def test_fully_precomputed_nameless_dataset_is_left_alone(self):
        """No pick, nothing to embed, no named vectors: the sentinel stays.

        A manifest of vectors from a model VTSearch has never heard of is a
        supported import; stamping the media-type default onto it would assert
        a space nothing asked for.
        """
        emb = _fake_embedder()
        medias, vecs = _partial_medias(np.random.default_rng(5), pre=(1, 2), missing=())

        with mock.patch("vtscore.media.embedders_for_type", return_value=[emb]):
            embed_missing(medias, "")

        assert emb.embed_media_bulk.call_count == 0
        for i, m in medias.items():
            assert set(m["embeddings"]) == {UNKNOWN_EMBEDDER_KEY}
            assert not m.get("embedder")
            np.testing.assert_array_equal(media_embedding(m), vecs[i])

    def test_wrong_width_precomputed_vector_is_rejected_at_import(self):
        """The stamp asserts the shipped vectors are the resolved embedder's.

        When the embedder declares a width and a shipped vector disagrees, the
        load fails here naming both widths - not at sort time, as a bare
        "has no embedding" for a media that visibly has one.
        """
        emb = _fake_embedder()
        emb.embedding_dim = DIM
        medias, _ = _partial_medias(np.random.default_rng(6), pre=(1,), missing=(2,))
        medias[1]["embeddings"][UNKNOWN_EMBEDDER_KEY] = np.ones(DIM + 3, dtype=np.float32)

        with (
            mock.patch("vtscore.media.embedders_for_type", return_value=[emb]),
            mock.patch("vtscore.media.get_embedder", return_value=emb),
            pytest.raises(MismatchedVectorError) as exc,
        ):
            embed_missing(medias, "")

        msg = str(exc.value)
        assert "1.wav" in msg
        assert str(DIM + 3) in msg and str(DIM) in msg
        assert EMBEDDER in msg


class TestPartialImportExplicitPick:
    def test_pick_still_stamps_and_embeds_the_rest(self):
        """The named-pick path (GUI import with a pick) is unchanged."""
        emb = _fake_embedder()
        medias, vecs = _partial_medias(np.random.default_rng(7), pre=(1,), missing=(2, 3))

        with (
            mock.patch("vtscore.media.get_embedder", return_value=emb),
            mock.patch("vtscore.media.embedders_for_type", return_value=[emb]),
        ):
            embed_missing(medias, EMBEDDER)

        assert [m["id"] for m in emb.embed_media_bulk.call_args.args[0]] == [2, 3]
        np.testing.assert_array_equal(medias[1]["embeddings"][EMBEDDER], vecs[1])
        for m in medias.values():
            assert set(m["embeddings"]) == {EMBEDDER}
        assert _stack_embeddings(sorted(medias), medias, EMBEDDER).shape == (3, DIM)


class TestEmbedFailuresAreLoud:
    """An embedder that produces nothing must say so in the log, not just drop rows."""

    def test_items_the_embedder_returns_none_for_are_named(self, caplog):
        emb = _fake_embedder(fail_ids=frozenset({2, 3}))
        medias, _ = _partial_medias(np.random.default_rng(8), pre=(), missing=(1, 2, 3))

        with (
            mock.patch("vtscore.media.embedders_for_type", return_value=[emb]),
            caplog.at_level(logging.WARNING, logger="vtscore.datasets.stages.embedding"),
        ):
            embed_missing(medias, "")

        assert media_embedding(medias[1]) is not None
        assert media_embedding(medias[2]) is None
        assert media_embedding(medias[3]) is None
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("2 of 3" in w and EMBEDDER in w for w in warnings), warnings

    def test_short_result_list_is_named_and_nothing_is_misattributed(self, caplog):
        emb = _fake_embedder()
        emb.embed_media_bulk.side_effect = lambda medias: [np.ones(DIM, dtype=np.float32)]
        medias, _ = _partial_medias(np.random.default_rng(9), pre=(), missing=(1, 2))

        with (
            mock.patch("vtscore.media.embedders_for_type", return_value=[emb]),
            caplog.at_level(logging.WARNING, logger="vtscore.datasets.stages.embedding"),
        ):
            embed_missing(medias, "")

        # A wrong-length answer cannot be paired with the inputs, so no vector
        # is attached to the wrong media; the mismatch is reported instead.
        assert all(media_embedding(m) is None for m in medias.values())
        assert any("1 vector(s) for 2 item(s)" in r.getMessage() for r in caplog.records)

    def test_no_embedder_for_the_media_type_is_named(self, caplog):
        medias, _ = _partial_medias(np.random.default_rng(10), pre=(), missing=(1, 2))

        with (
            mock.patch("vtscore.media.embedders_for_type", return_value=[]),
            caplog.at_level(logging.WARNING, logger="vtscore.datasets.stages.embedding"),
        ):
            embed_missing(medias, "")

        assert all(media_embedding(m) is None for m in medias.values())
        assert any("no embedder" in r.getMessage().lower() and "audio" in r.getMessage() for r in caplog.records)
