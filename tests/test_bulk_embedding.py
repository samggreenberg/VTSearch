"""Bulk (batch) embedding support tests.

Verifies that :class:`~vtsearch.media.embedder.MediaEmbedder` exposes an
opt-in batch API and that :func:`~vtsearch.datasets.loader.load_dataset_from_folder`
(and its chunked sibling) route through it when an embedder advertises
``supports_batch=True``.

The goal is to support embedders backed by a remote bulk API (e.g. Datawave)
without forcing every existing embedder to implement batching.
"""

from __future__ import annotations

import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest

from helpers import make_raw_wav_bytes as _make_wav_bytes


# ---------------------------------------------------------------------------
# Fake batch-capable embedder used across the loader tests.
# ---------------------------------------------------------------------------


def _make_batch_embedder(batch_size: int = 32, embed_return_dim: int = 3):
    """Return a MagicMock that mimics a batch-capable MediaEmbedder."""
    emb = mock.MagicMock()
    emb.name = "fake_batch"
    emb.media_type_id = "audio"
    emb._model = True  # skip eager model-load path
    emb.supports_batch = True
    emb.batch_size = batch_size

    def _batch(medias):
        return [np.full(embed_return_dim, float(i), dtype=np.float32) for i, _ in enumerate(medias)]

    emb.embed_media_batch.side_effect = _batch
    return emb


def _make_media_type_for_audio():
    mt = mock.MagicMock()
    mt.type_id = "audio"
    mt.file_extensions = ["*.wav"]
    mt.load_media_data.return_value = {"duration": 1.0}
    return mt


def _write_wav(path: Path) -> None:
    path.write_bytes(_make_wav_bytes())


# ---------------------------------------------------------------------------
# MediaEmbedder.embed_media_batch defaults
# ---------------------------------------------------------------------------


class TestEmbedderDefaults:
    """The ABC's defaults should preserve existing per-file behaviour."""

    def test_supports_batch_defaults_false(self):
        """Concrete embedders that don't override stay one-by-one."""
        from vtsearch.media import all_embedders

        for emb in all_embedders():
            assert emb.supports_batch is False, (
                f"{type(emb).__name__} reports supports_batch=True but has no overridden "
                "_embed_media_batch_impl"
            )

    def test_default_batch_impl_loops_over_single(self, tmp_path):
        """The default _embed_media_batch_impl dispatches to _embed_media_impl per item."""
        from vtsearch.media.embedder import MediaEmbedder

        class _Stub(MediaEmbedder):
            @property
            def name(self):
                return "stub"

            @property
            def media_type_id(self):
                return "audio"

            def _load_models_impl(self):
                self._model = True

            def _embed_media_impl(self, media):
                return np.array([float(Path(media["media_path"]).stat().st_size)], dtype=np.float32)

        emb = _Stub()
        emb._model = True
        medias = []
        for i, name in enumerate(["a.bin", "b.bin", "c.bin"]):
            p = tmp_path / name
            p.write_bytes(b"x" * (i + 1))
            medias.append({"media_path": str(p)})

        vecs = emb.embed_media_batch(medias)
        assert len(vecs) == 3
        assert vecs[0][0] == 1.0
        assert vecs[1][0] == 2.0
        assert vecs[2][0] == 3.0

    def test_empty_batch_returns_empty(self):
        """Passing an empty list must not acquire the lock or call the hook."""
        from vtsearch.media.embedder import MediaEmbedder

        class _Stub(MediaEmbedder):
            @property
            def name(self):
                return "stub"

            @property
            def media_type_id(self):
                return "audio"

            def _load_models_impl(self):
                self._model = True

            def _embed_media_impl(self, media):
                raise AssertionError("should not be called for empty batch")

        emb = _Stub()
        assert emb.embed_media_batch([]) == []


# ---------------------------------------------------------------------------
# load_dataset_from_folder routes through embed_media_batch when opted in
# ---------------------------------------------------------------------------


class TestLoaderRoutesToBatch:
    """load_dataset_from_folder must call embed_media_batch when supports_batch=True."""

    def test_single_batch_for_small_folder(self, tmp_path):
        from vtsearch.datasets.loader import load_dataset_from_folder

        for name in ("a.wav", "b.wav", "c.wav"):
            _write_wav(tmp_path / name)

        mt = _make_media_type_for_audio()
        emb = _make_batch_embedder(batch_size=32)

        medias: dict = {}
        with mock.patch("vtsearch.media.get_by_folder_name", return_value=mt), mock.patch(
            "vtsearch.media.embedders_for_type", return_value=[emb]
        ):
            load_dataset_from_folder(tmp_path, "audio", medias, on_progress=lambda *a: None)

        assert len(medias) == 3
        # Exactly one batch call — all three files in one request.
        assert emb.embed_media_batch.call_count == 1
        sent_medias = emb.embed_media_batch.call_args.args[0]
        assert sorted(Path(m["media_path"]).name for m in sent_medias) == ["a.wav", "b.wav", "c.wav"]
        # Per-file embed_media must NOT have been called.
        emb.embed_media.assert_not_called()

    def test_splits_into_multiple_batches_by_batch_size(self, tmp_path):
        from vtsearch.datasets.loader import load_dataset_from_folder

        # 5 files, batch_size=2 → 3 batches of sizes 2, 2, 1
        for i in range(5):
            _write_wav(tmp_path / f"f{i}.wav")

        mt = _make_media_type_for_audio()
        emb = _make_batch_embedder(batch_size=2)

        medias: dict = {}
        with mock.patch("vtsearch.media.get_by_folder_name", return_value=mt), mock.patch(
            "vtsearch.media.embedders_for_type", return_value=[emb]
        ):
            load_dataset_from_folder(tmp_path, "audio", medias, on_progress=lambda *a: None)

        assert len(medias) == 5
        assert emb.embed_media_batch.call_count == 3
        sizes = [len(call.args[0]) for call in emb.embed_media_batch.call_args_list]
        assert sizes == [2, 2, 1]

    def test_overrides_skip_batch_call(self, tmp_path):
        """Files with content_vectors or custom_metadata embedding aren't sent to batch."""
        from vtsearch.datasets.loader import load_dataset_from_folder

        for name in ("keep.wav", "pre1.wav", "pre2.wav"):
            _write_wav(tmp_path / name)

        mt = _make_media_type_for_audio()
        emb = _make_batch_embedder(batch_size=32)

        pre_vec = np.array([99.0, 99.0, 99.0], dtype=np.float32)
        cm_vec = np.array([42.0, 42.0, 42.0], dtype=np.float32)

        medias: dict = {}
        with mock.patch("vtsearch.media.get_by_folder_name", return_value=mt), mock.patch(
            "vtsearch.media.embedders_for_type", return_value=[emb]
        ):
            load_dataset_from_folder(
                tmp_path,
                "audio",
                medias,
                content_vectors={"pre1.wav": pre_vec},
                custom_metadata_map={"pre2.wav": {"embedding": cm_vec}},
                on_progress=lambda *a: None,
            )

        assert len(medias) == 3
        # Only the non-overridden file should have been batched.
        assert emb.embed_media_batch.call_count == 1
        sent = emb.embed_media_batch.call_args.args[0]
        assert [Path(m["media_path"]).name for m in sent] == ["keep.wav"]

        # Verify overrides won
        by_name = {m["filename"]: m["embedding"] for m in medias.values()}
        np.testing.assert_array_equal(by_name["pre1.wav"], pre_vec)
        np.testing.assert_array_equal(by_name["pre2.wav"], cm_vec)

    def test_skip_embedding_does_not_trigger_batch(self, tmp_path):
        """skip_embedding=True must bypass the batch API entirely."""
        from vtsearch.datasets.loader import load_dataset_from_folder

        _write_wav(tmp_path / "a.wav")

        mt = _make_media_type_for_audio()
        emb = _make_batch_embedder(batch_size=32)

        medias: dict = {}
        with mock.patch("vtsearch.media.get_by_folder_name", return_value=mt), mock.patch(
            "vtsearch.media.embedders_for_type", return_value=[emb]
        ):
            load_dataset_from_folder(
                tmp_path, "audio", medias, on_progress=lambda *a: None, skip_embedding=True
            )

        emb.embed_media_batch.assert_not_called()
        emb.embed_media.assert_not_called()
        assert medias[1]["embedding"] is None

    def test_batch_returning_none_skips_file(self, tmp_path):
        """A None entry in the batch response should skip that file, matching per-file semantics."""
        from vtsearch.datasets.loader import load_dataset_from_folder

        _write_wav(tmp_path / "good.wav")
        _write_wav(tmp_path / "bad.wav")

        mt = _make_media_type_for_audio()
        emb = mock.MagicMock()
        emb.name = "fake_batch"
        emb.media_type_id = "audio"
        emb._model = True
        emb.supports_batch = True
        emb.batch_size = 32

        def _batch(medias_in):
            return [
                None if Path(m["media_path"]).name == "bad.wav" else np.array([1.0, 2.0])
                for m in medias_in
            ]

        emb.embed_media_batch.side_effect = _batch

        medias: dict = {}
        with mock.patch("vtsearch.media.get_by_folder_name", return_value=mt), mock.patch(
            "vtsearch.media.embedders_for_type", return_value=[emb]
        ):
            load_dataset_from_folder(tmp_path, "audio", medias, on_progress=lambda *a: None)

        assert len(medias) == 1
        assert list(medias.values())[0]["filename"] == "good.wav"


# ---------------------------------------------------------------------------
# Non-batch embedders keep the old per-file path
# ---------------------------------------------------------------------------


class TestNonBatchEmbedderUnchanged:
    """Embedders with supports_batch=False must still use embed_media per file."""

    def test_per_file_path_still_used(self, tmp_path):
        from vtsearch.datasets.loader import load_dataset_from_folder

        for name in ("a.wav", "b.wav"):
            _write_wav(tmp_path / name)

        mt = _make_media_type_for_audio()
        emb = mock.MagicMock()
        emb.name = "single"
        emb.media_type_id = "audio"
        emb._model = True
        emb.supports_batch = False
        emb.embed_media.return_value = np.array([1.0, 2.0])

        medias: dict = {}
        with mock.patch("vtsearch.media.get_by_folder_name", return_value=mt), mock.patch(
            "vtsearch.media.embedders_for_type", return_value=[emb]
        ):
            load_dataset_from_folder(tmp_path, "audio", medias, on_progress=lambda *a: None)

        assert emb.embed_media.call_count == 2
        emb.embed_media_batch.assert_not_called()


# ---------------------------------------------------------------------------
# Chunked loader batches within each chunk
# ---------------------------------------------------------------------------


class TestChunkedLoaderBatches:
    """load_dataset_from_folder_chunked must batch within each chunk."""

    def test_batches_scoped_to_chunk(self, tmp_path):
        """Four files with chunk_size=2 and batch_size=10 → 2 batch calls, one per chunk.

        Ensures the batch is flushed per chunk (to preserve chunked loading's
        memory story) rather than across the whole folder.
        """
        from vtsearch.datasets.loader import load_dataset_from_folder_chunked

        for i in range(4):
            _write_wav(tmp_path / f"f{i}.wav")

        mt = _make_media_type_for_audio()
        emb = _make_batch_embedder(batch_size=10)

        with mock.patch("vtsearch.media.get_by_folder_name", return_value=mt), mock.patch(
            "vtsearch.media.embedders_for_type", return_value=[emb]
        ):
            chunks = list(
                load_dataset_from_folder_chunked(
                    tmp_path, "audio", chunk_size=2, on_progress=lambda *a: None
                )
            )

        assert len(chunks) == 2
        assert all(len(c) == 2 for c in chunks)
        # One batch call per chunk, each receiving exactly 2 files.
        assert emb.embed_media_batch.call_count == 2
        for call in emb.embed_media_batch.call_args_list:
            assert len(call.args[0]) == 2
