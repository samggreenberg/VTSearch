"""Bulk embedding support tests.

The :class:`~vtsearch.media.embedder.MediaEmbedder` ABC exposes a single
bulk entrypoint (:meth:`embed_media_bulk`) that the dataset loader
always routes through.  The default implementation loops per item and
emits progress via ``_on_progress`` so long embeds stay responsive;
subclasses backed by a service that natively accepts many items can
override :meth:`_embed_media_bulk_impl` and do their own internal
batching (emitting their own progress).
"""

from __future__ import annotations

import unittest.mock as mock
from pathlib import Path

import numpy as np

from helpers import make_raw_wav_bytes as _make_wav_bytes


# ---------------------------------------------------------------------------
# Fake embedders used across the loader tests.
# ---------------------------------------------------------------------------


def _make_bulk_embedder(embed_return_dim: int = 3):
    """Return a MagicMock that mimics an embedder with a native bulk path."""
    emb = mock.MagicMock()
    emb.name = "fake_bulk"
    emb.media_type_id = "audio"
    emb._model = True  # skip eager model-load path
    emb._on_progress = lambda *a, **kw: None

    def _bulk(medias):
        return [np.full(embed_return_dim, float(i), dtype=np.float32) for i, _ in enumerate(medias)]

    emb.embed_media_bulk.side_effect = _bulk
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
# MediaEmbedder.embed_media_bulk defaults
# ---------------------------------------------------------------------------


class TestEmbedderDefaults:
    """The ABC's default bulk impl loops per item and emits progress."""

    def test_default_bulk_impl_loops_over_single(self, tmp_path):
        """The default _embed_media_bulk_impl dispatches to embed_media per item."""
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

        vecs = emb.embed_media_bulk(medias)
        assert len(vecs) == 3
        assert vecs[0][0] == 1.0
        assert vecs[1][0] == 2.0
        assert vecs[2][0] == 3.0

    def test_default_bulk_impl_emits_per_item_progress(self, tmp_path):
        """The default bulk loop must call _on_progress on each iteration
        so the progress bar keeps moving for slow local embedders."""
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
                return np.array([1.0], dtype=np.float32)

        emb = _Stub()
        emb._model = True

        events: list[tuple[str, str, int, int]] = []
        emb._on_progress = lambda status, msg, cur, tot: events.append((status, msg, cur, tot))

        medias = []
        for name in ["a.bin", "b.bin", "c.bin"]:
            p = tmp_path / name
            p.write_bytes(b"x")
            medias.append({"media_path": str(p)})

        emb.embed_media_bulk(medias)

        # One progress event per item, each reporting (i+1)/3 with status "embedding".
        assert len(events) == 3
        for i, (status, _msg, cur, tot) in enumerate(events):
            assert status == "embedding"
            assert cur == i + 1
            assert tot == 3

    def test_empty_bulk_returns_empty(self):
        """Passing an empty list must not invoke the hook."""
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
                raise AssertionError("should not be called for empty bulk")

        emb = _Stub()
        assert emb.embed_media_bulk([]) == []


# ---------------------------------------------------------------------------
# load_dataset_from_folder always routes through embed_media_bulk
# ---------------------------------------------------------------------------


class TestLoaderRoutesToBulk:
    """The loader hands pending files to embed_media_bulk in a single call."""

    def test_single_bulk_call_for_folder(self, tmp_path):
        from vtsearch.datasets.loader import load_dataset_from_folder

        for name in ("a.wav", "b.wav", "c.wav"):
            _write_wav(tmp_path / name)

        mt = _make_media_type_for_audio()
        emb = _make_bulk_embedder()

        medias: dict = {}
        with (
            mock.patch("vtsearch.media.get_by_folder_name", return_value=mt),
            mock.patch("vtsearch.media.embedders_for_type", return_value=[emb]),
        ):
            load_dataset_from_folder(tmp_path, "audio", medias, on_progress=lambda *a: None)

        assert len(medias) == 3
        # Exactly one bulk call — all three files in one request.
        assert emb.embed_media_bulk.call_count == 1
        sent_medias = emb.embed_media_bulk.call_args.args[0]
        assert sorted(Path(m["media_path"]).name for m in sent_medias) == ["a.wav", "b.wav", "c.wav"]

    def test_loader_routes_embedder_progress_to_caller(self, tmp_path):
        """The loader sets emb._on_progress to its own on_progress for the
        duration of the bulk call, so progress emitted by the embedder (or
        its default per-item loop) reaches the UI."""
        from vtsearch.datasets.loader import load_dataset_from_folder

        for name in ("a.wav", "b.wav"):
            _write_wav(tmp_path / name)

        mt = _make_media_type_for_audio()

        captured_cb: list = []

        def _bulk_that_pings_progress(medias_in):
            # Whatever _on_progress is set to *at the moment of the bulk call*
            # is what the loader routed through.
            captured_cb.append(emb._on_progress)
            emb._on_progress("embedding", "halfway", 1, 2)
            return [np.array([1.0], dtype=np.float32) for _ in medias_in]

        emb = _make_bulk_embedder()
        emb.embed_media_bulk.side_effect = _bulk_that_pings_progress

        events: list[tuple] = []

        def loader_progress(status, msg, cur, tot):
            events.append((status, msg, cur, tot))

        medias: dict = {}
        with (
            mock.patch("vtsearch.media.get_by_folder_name", return_value=mt),
            mock.patch("vtsearch.media.embedders_for_type", return_value=[emb]),
        ):
            load_dataset_from_folder(tmp_path, "audio", medias, on_progress=loader_progress)

        # The embedder's _on_progress during the call is the loader's callback.
        assert captured_cb and captured_cb[0] is loader_progress
        # The "halfway" event from inside the bulk call reached the loader's callback.
        assert ("embedding", "halfway", 1, 2) in events

    def test_overrides_skip_bulk_call(self, tmp_path):
        """Files with content_vectors or custom_metadata embedding aren't sent to bulk."""
        from vtsearch.datasets.loader import load_dataset_from_folder

        for name in ("keep.wav", "pre1.wav", "pre2.wav"):
            _write_wav(tmp_path / name)

        mt = _make_media_type_for_audio()
        emb = _make_bulk_embedder()

        pre_vec = np.array([99.0, 99.0, 99.0], dtype=np.float32)
        cm_vec = np.array([42.0, 42.0, 42.0], dtype=np.float32)

        medias: dict = {}
        with (
            mock.patch("vtsearch.media.get_by_folder_name", return_value=mt),
            mock.patch("vtsearch.media.embedders_for_type", return_value=[emb]),
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
        assert emb.embed_media_bulk.call_count == 1
        sent = emb.embed_media_bulk.call_args.args[0]
        assert [Path(m["media_path"]).name for m in sent] == ["keep.wav"]

        by_name = {m["filename"]: m["embedding"] for m in medias.values()}
        np.testing.assert_array_equal(by_name["pre1.wav"], pre_vec)
        np.testing.assert_array_equal(by_name["pre2.wav"], cm_vec)

    def test_skip_embedding_does_not_trigger_bulk(self, tmp_path):
        """skip_embedding=True must bypass the bulk API entirely."""
        from vtsearch.datasets.loader import load_dataset_from_folder

        _write_wav(tmp_path / "a.wav")

        mt = _make_media_type_for_audio()
        emb = _make_bulk_embedder()

        medias: dict = {}
        with (
            mock.patch("vtsearch.media.get_by_folder_name", return_value=mt),
            mock.patch("vtsearch.media.embedders_for_type", return_value=[emb]),
        ):
            load_dataset_from_folder(tmp_path, "audio", medias, on_progress=lambda *a: None, skip_embedding=True)

        emb.embed_media_bulk.assert_not_called()
        assert medias[1]["embedding"] is None

    def test_bulk_returning_none_skips_file(self, tmp_path):
        """A None entry in the bulk response should skip that file."""
        from vtsearch.datasets.loader import load_dataset_from_folder

        _write_wav(tmp_path / "good.wav")
        _write_wav(tmp_path / "bad.wav")

        mt = _make_media_type_for_audio()
        emb = mock.MagicMock()
        emb.name = "fake_bulk"
        emb.media_type_id = "audio"
        emb._model = True
        emb._on_progress = lambda *a, **kw: None

        def _bulk(medias_in):
            return [None if Path(m["media_path"]).name == "bad.wav" else np.array([1.0, 2.0]) for m in medias_in]

        emb.embed_media_bulk.side_effect = _bulk

        medias: dict = {}
        with (
            mock.patch("vtsearch.media.get_by_folder_name", return_value=mt),
            mock.patch("vtsearch.media.embedders_for_type", return_value=[emb]),
        ):
            load_dataset_from_folder(tmp_path, "audio", medias, on_progress=lambda *a: None)

        assert len(medias) == 1
        assert list(medias.values())[0]["filename"] == "good.wav"


# ---------------------------------------------------------------------------
# Chunked loader bulk-embeds per chunk
# ---------------------------------------------------------------------------


class TestChunkedLoaderBulkPerChunk:
    """The chunked loader issues one bulk call per chunk — preserving the
    memory story where only one chunk's worth of embeddings is in RAM at a time."""

    def test_bulk_call_scoped_to_chunk(self, tmp_path):
        from vtsearch.datasets.loader import load_dataset_from_folder_chunked

        for i in range(4):
            _write_wav(tmp_path / f"f{i}.wav")

        mt = _make_media_type_for_audio()
        emb = _make_bulk_embedder()

        with (
            mock.patch("vtsearch.media.get_by_folder_name", return_value=mt),
            mock.patch("vtsearch.media.embedders_for_type", return_value=[emb]),
        ):
            chunks = list(
                load_dataset_from_folder_chunked(tmp_path, "audio", chunk_size=2, on_progress=lambda *a: None)
            )

        assert len(chunks) == 2
        assert all(len(c) == 2 for c in chunks)
        # One bulk call per chunk, each receiving exactly 2 files.
        assert emb.embed_media_bulk.call_count == 2
        for call in emb.embed_media_bulk.call_args_list:
            assert len(call.args[0]) == 2
