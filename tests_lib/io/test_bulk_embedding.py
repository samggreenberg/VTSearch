"""Bulk embedding support tests.

The :class:`~vtscore.media.embedder.MediaEmbedder` ABC exposes a single
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
        from vtscore.media.embedder import MediaEmbedder

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
        assert all(v is not None for v in vecs)
        assert vecs[0][0] == 1.0  # pyright: ignore[reportOptionalSubscript]
        assert vecs[1][0] == 2.0  # pyright: ignore[reportOptionalSubscript]
        assert vecs[2][0] == 3.0  # pyright: ignore[reportOptionalSubscript]

    def test_default_bulk_impl_emits_per_item_progress(self, tmp_path):
        """The default bulk loop must call _on_progress on each iteration
        so the progress bar keeps moving for slow local embedders."""
        from vtscore.media.embedder import MediaEmbedder

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
        from vtscore.media.embedder import MediaEmbedder

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
# embed_missing: framework stage that bulk-embeds items with embedding=None
# ---------------------------------------------------------------------------


class TestEmbedMissingRoutesToBulk:
    """The framework ``embed_missing`` stage hands every media with
    ``embedding=None`` to ``embed_media_bulk`` in a single call."""

    def test_single_bulk_call_for_unembedded_medias(self):
        from vtscore.datasets.load_pipeline import embed_missing

        mt = _make_media_type_for_audio()
        emb = _make_bulk_embedder()

        medias = {i: {"media_type": "audio", "embedding": None, "media_path": f"/tmp/{i}.wav"} for i in range(1, 4)}

        with (
            mock.patch("vtscore.media.get_by_folder_name", return_value=mt),
            mock.patch("vtscore.media.embedders_for_type", return_value=[emb]),
        ):
            embed_missing(medias)

        assert emb.embed_media_bulk.call_count == 1
        sent = emb.embed_media_bulk.call_args.args[0]
        assert len(sent) == 3
        assert all(m["embedding"] is not None for m in medias.values())

    def test_already_embedded_medias_are_left_alone(self):
        from vtscore.datasets.load_pipeline import embed_missing

        emb = _make_bulk_embedder()

        pre_vec = np.array([99.0, 99.0, 99.0], dtype=np.float32)
        medias = {
            1: {"media_type": "audio", "embedding": pre_vec, "embedder": "external"},
            2: {"media_type": "audio", "embedding": None, "media_path": "/tmp/2.wav"},
        }

        with mock.patch("vtscore.media.embedders_for_type", return_value=[emb]):
            embed_missing(medias)

        sent = emb.embed_media_bulk.call_args.args[0]
        assert len(sent) == 1
        np.testing.assert_array_equal(medias[1]["embedding"], pre_vec)
        assert medias[1]["embedder"] == "external"
        assert medias[2]["embedding"] is not None

    def test_no_op_when_nothing_missing(self):
        from vtscore.datasets.load_pipeline import embed_missing

        emb = _make_bulk_embedder()
        pre_vec = np.array([1.0], dtype=np.float32)
        medias = {1: {"media_type": "audio", "embedding": pre_vec, "embedder": "external"}}

        with mock.patch("vtscore.media.embedders_for_type", return_value=[emb]):
            embed_missing(medias)

        emb.embed_media_bulk.assert_not_called()

    def test_no_op_when_media_type_absent(self):
        """embed_missing silently skips all items when media_type is missing.

        This is the regression contract for the converter-ingestion bug:
        if _ingest_spec_stream forgets to stamp media_type, embed_missing
        returns early and leaves all embeddings as None.
        """
        from vtscore.datasets.load_pipeline import embed_missing

        emb = _make_bulk_embedder()
        # Mimic what a converter output looked like before the fix:
        # media_type absent, embedding still None.
        medias = {i: {"embedding": None, "media_path": f"/tmp/{i}.wav"} for i in range(1, 4)}

        with mock.patch("vtscore.media.embedders_for_type", return_value=[emb]):
            embed_missing(medias)

        emb.embed_media_bulk.assert_not_called()
        assert all(m["embedding"] is None for m in medias.values())

    def test_routes_progress_to_caller(self):
        from vtscore.datasets.load_pipeline import embed_missing

        captured_cb: list = []

        def _bulk_that_pings_progress(medias_in):
            captured_cb.append(emb._on_progress)
            emb._on_progress("embedding", "halfway", 1, 2)
            return [np.array([1.0], dtype=np.float32) for _ in medias_in]

        emb = _make_bulk_embedder()
        emb.embed_media_bulk.side_effect = _bulk_that_pings_progress

        events: list[tuple] = []

        def caller_progress(status, msg="", cur=0, tot=0):
            events.append((status, msg, cur, tot))

        medias = {1: {"media_type": "audio", "embedding": None, "media_path": "/tmp/a.wav"}}

        with mock.patch("vtscore.media.embedders_for_type", return_value=[emb]):
            embed_missing(medias, on_progress=caller_progress)

        assert captured_cb and captured_cb[0] is caller_progress
        assert ("embedding", "halfway", 1, 2) in events

    def test_bulk_returning_none_leaves_embedding_none(self):
        from vtscore.datasets.load_pipeline import embed_missing

        emb = _make_bulk_embedder()

        def _bulk(medias_in):
            return [None if m["media_path"].endswith("bad.wav") else np.array([1.0, 2.0]) for m in medias_in]

        emb.embed_media_bulk.side_effect = _bulk

        medias = {
            1: {"media_type": "audio", "embedding": None, "media_path": "/tmp/good.wav"},
            2: {"media_type": "audio", "embedding": None, "media_path": "/tmp/bad.wav"},
        }

        with mock.patch("vtscore.media.embedders_for_type", return_value=[emb]):
            embed_missing(medias)

        assert medias[1]["embedding"] is not None
        assert medias[2]["embedding"] is None


# ---------------------------------------------------------------------------
# embed_medias: dict-keyed convenience wrapper around embed_media_bulk
# ---------------------------------------------------------------------------


class TestEmbedMediasDictWrapper:
    """``embed_medias`` is a sugar wrapper for callers that already have
    medias keyed by ID (e.g. importers building the medias dict before
    embedding).  It delegates to ``embed_media_bulk`` and pairs vectors
    back to the original IDs."""

    def test_returns_dict_keyed_by_input_ids(self):
        from vtscore.media.embedder import MediaEmbedder

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
                return np.array([float(media["tag"])], dtype=np.float32)

        emb = _Stub()
        emb._model = True

        medias = {1: {"tag": 10}, 2: {"tag": 20}, 7: {"tag": 70}}
        out = emb.embed_medias(medias)

        assert set(out.keys()) == {1, 2, 7}
        assert all(v is not None for v in out.values())
        assert out[1][0] == 10.0  # pyright: ignore[reportOptionalSubscript]
        assert out[2][0] == 20.0  # pyright: ignore[reportOptionalSubscript]
        assert out[7][0] == 70.0  # pyright: ignore[reportOptionalSubscript]

    def test_handles_sparse_keys(self):
        """Non-contiguous keys (e.g. post-collapse_duplicates) round-trip."""
        from vtscore.media.embedder import MediaEmbedder

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
                return np.array([float(media["tag"])], dtype=np.float32)

        emb = _Stub()
        emb._model = True

        medias = {1: {"tag": 1}, 3: {"tag": 3}, 5: {"tag": 5}}
        out = emb.embed_medias(medias)
        assert list(out.keys()) == [1, 3, 5]
        assert out[3] is not None
        assert out[3][0] == 3.0

    def test_propagates_none_for_failed_embeddings(self):
        """A None vector from the underlying bulk call surfaces as None
        at the matching key — no silent dropping like the loader does."""
        from vtscore.media.embedder import MediaEmbedder

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
                if media["tag"] == "skip":
                    return None
                return np.array([1.0], dtype=np.float32)

        emb = _Stub()
        emb._model = True

        out = emb.embed_medias({1: {"tag": "ok"}, 2: {"tag": "skip"}, 3: {"tag": "ok"}})
        assert out[1] is not None
        assert out[2] is None
        assert out[3] is not None

    def test_empty_dict_returns_empty_dict(self):
        from vtscore.media.embedder import MediaEmbedder

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
                raise AssertionError("should not be called for empty dict")

        emb = _Stub()
        assert emb.embed_medias({}) == {}

    def test_delegates_to_embed_media_bulk(self):
        """The wrapper calls embed_media_bulk once with values in key order."""
        from vtscore.media.embedder import MediaEmbedder

        captured: list[list[dict]] = []

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
                return np.array([float(media["x"])], dtype=np.float32)

            def _embed_media_bulk_impl(self, medias):
                captured.append(list(medias))
                return [np.array([float(m["x"])], dtype=np.float32) for m in medias]

        emb = _Stub()
        emb._model = True

        medias = {10: {"x": 1}, 20: {"x": 2}, 30: {"x": 3}}
        out = emb.embed_medias(medias)

        assert len(captured) == 1
        assert captured[0] == [{"x": 1}, {"x": 2}, {"x": 3}]
        assert set(out.keys()) == {10, 20, 30}
        assert all(v is not None for v in out.values())
        assert out[10][0] == 1.0  # pyright: ignore[reportOptionalSubscript]
        assert out[20][0] == 2.0  # pyright: ignore[reportOptionalSubscript]
        assert out[30][0] == 3.0  # pyright: ignore[reportOptionalSubscript]
