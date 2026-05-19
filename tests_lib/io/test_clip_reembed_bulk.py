"""Phase C: clip re-embed goes through ``embed_media_bulk`` once per
``(clip-list, media_type)`` invocation, with no tempfile detour.

The old ``_reembed_clip`` helper wrote each clip to a ``tempfile.mkstemp``
and then called ``embed_file`` one clip at a time — defeating GPU
batching and adding two syscalls per clip.  Phase C replaces that with a
single ``embedder.embed_media_bulk`` call wired through the bulk surface
established in Phase A, handing the embedder ``media_bytes`` /
``media_string`` directly so the content never touches disk.
"""

from __future__ import annotations

import hashlib
import io
import unittest.mock as mock

import numpy as np
import pytest

from vtscore.media.audio.audio_generator import generate_wav


def _make_audio_media(media_id: int, duration: float = 5.1) -> dict:
    wav = generate_wav(441, duration)
    rng = np.random.default_rng(media_id)
    return {
        "id": media_id,
        "type": "audio",
        "filename": f"clip_{media_id}.wav",
        "media_bytes": wav,
        "duration": duration,
        "md5": hashlib.md5(wav).hexdigest(),
        "embedding": rng.standard_normal(512).astype(np.float32),
        "origin": {"importer": "server_folder", "params": {"path": "/data/audio", "media_type": "audio"}},
        "origin_name": f"clip_{media_id}.wav",
    }


def _make_image_media(media_id: int, width: int = 300, height: int = 100) -> dict:
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(media_id * 30 % 256, 100, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    rng = np.random.default_rng(media_id + 1000)
    return {
        "id": media_id,
        "type": "image",
        "filename": f"img_{media_id}.png",
        "media_bytes": img_bytes,
        "width": width,
        "height": height,
        "md5": hashlib.md5(img_bytes).hexdigest(),
        "embedding": rng.standard_normal(512).astype(np.float32),
        "origin": {"importer": "server_folder", "params": {"path": "/data/images", "media_type": "image"}},
        "origin_name": f"img_{media_id}.png",
    }


def _make_text_media(media_id: int, text: str = "First sentence. Second sentence. Third sentence.") -> dict:
    text_bytes = text.encode("utf-8")
    rng = np.random.default_rng(media_id + 2000)
    return {
        "id": media_id,
        "type": "text",
        "filename": f"text_{media_id}.txt",
        "media_string": text,
        "media_bytes": text_bytes,
        "md5": hashlib.md5(text_bytes).hexdigest(),
        "embedding": rng.standard_normal(512).astype(np.float32),
        "origin": {"importer": "server_folder", "params": {"path": "/data/texts", "media_type": "text"}},
        "origin_name": f"text_{media_id}.txt",
    }


def _fake_bulk_embedder(dim: int = 512):
    """Return a MagicMock embedder with a deterministic bulk hook."""
    emb = mock.MagicMock()
    emb._on_progress = lambda *a, **kw: None

    def _bulk(medias):
        # Return one distinct vector per media so callers can verify
        # the scatter step assigned to the correct slot.
        return [np.full(dim, float(i + 1), dtype=np.float32) for i, _ in enumerate(medias)]

    emb.embed_media_bulk.side_effect = _bulk
    return emb


@pytest.mark.parametrize(
    ("media_type", "make_media", "clipper_name", "clipper_params", "content_field"),
    [
        ("audio", _make_audio_media, "sound_tiling", {"duration": 2.0}, "media_bytes"),
        ("image", _make_image_media, "image_tiling", None, "media_bytes"),
        ("text", _make_text_media, "text_sentence", None, "media_string"),
    ],
)
class TestBulkClipReembed:
    """``embed_media_bulk`` is called exactly once per clipper invocation,
    and the dicts it receives carry the clip's in-memory content (no path).
    """

    def test_single_bulk_call_per_clipper_invocation(
        self, media_type, make_media, clipper_name, clipper_params, content_field
    ):
        from vtscore.datasets.load_pipeline import _apply_clipper

        emb = _fake_bulk_embedder()

        clips_dict = {1: make_media(1)}
        with mock.patch("vtscore.media.embedders_for_type", return_value=[emb]):
            _apply_clipper(clips_dict, clipper_name, clipper_params)

        assert emb.embed_media_bulk.call_count == 1, (
            f"{media_type}: expected one bulk call per invocation, got {emb.embed_media_bulk.call_count}"
        )
        sent = emb.embed_media_bulk.call_args.args[0]
        assert len(sent) == len(clips_dict)
        # Every dict carries the in-memory content the embedder needs —
        # and **never** a ``media_path``, which would mean the loader was
        # still routing through a tempfile.
        for media in sent:
            assert content_field in media
            assert "media_path" not in media

    def test_scatters_returned_vectors_back_into_clips(
        self, media_type, make_media, clipper_name, clipper_params, content_field
    ):
        from vtscore.datasets.load_pipeline import _apply_clipper

        emb = _fake_bulk_embedder()

        clips_dict = {1: make_media(1)}
        with mock.patch("vtscore.media.embedders_for_type", return_value=[emb]):
            _apply_clipper(clips_dict, clipper_name, clipper_params)

        # Each clip should hold the vector the bulk hook returned for its
        # slot — i.e. ``np.full(512, slot_idx + 1)`` from the fake.
        ordered = list(clips_dict.values())
        for slot_idx, clip in enumerate(ordered):
            expected = np.full(512, float(slot_idx + 1), dtype=np.float32)
            np.testing.assert_array_equal(clip["embedding"], expected)


class TestBulkClipReembedFailureFallback:
    """When ``embed_media_bulk`` returns ``None`` for a clip — or the
    whole call raises — the affected clip keeps the parent embedding it
    inherited from the clipper, matching the legacy ``except: pass``
    contract in the pre-refactor ``_reembed_clip`` helper."""

    def test_none_entries_leave_parent_embedding_intact(self):
        from vtscore.datasets.load_pipeline import _apply_clipper

        parent = _make_audio_media(1, duration=5.1)
        parent_vec = parent["embedding"].copy()

        emb = mock.MagicMock()
        emb._on_progress = lambda *a, **kw: None
        emb.embed_media_bulk.side_effect = lambda medias: [None] * len(medias)

        clips_dict = {1: parent}
        with mock.patch("vtscore.media.embedders_for_type", return_value=[emb]):
            _apply_clipper(clips_dict, "sound_tiling", {"duration": 2.0})

        assert emb.embed_media_bulk.call_count == 1
        for clip in clips_dict.values():
            np.testing.assert_array_equal(clip["embedding"], parent_vec)

    def test_bulk_exception_leaves_parent_embedding_intact(self):
        from vtscore.datasets.load_pipeline import _apply_clipper

        parent = _make_audio_media(1, duration=5.1)
        parent_vec = parent["embedding"].copy()

        emb = mock.MagicMock()
        emb._on_progress = lambda *a, **kw: None
        emb.embed_media_bulk.side_effect = RuntimeError("boom")

        clips_dict = {1: parent}
        with mock.patch("vtscore.media.embedders_for_type", return_value=[emb]):
            _apply_clipper(clips_dict, "sound_tiling", {"duration": 2.0})

        for clip in clips_dict.values():
            np.testing.assert_array_equal(clip["embedding"], parent_vec)

    def test_no_embedders_registered_skips_bulk_and_keeps_parent(self):
        from vtscore.datasets.load_pipeline import _apply_clipper

        parent = _make_audio_media(1, duration=5.1)
        parent_vec = parent["embedding"].copy()

        clips_dict = {1: parent}
        with mock.patch("vtscore.media.embedders_for_type", return_value=[]):
            _apply_clipper(clips_dict, "sound_tiling", {"duration": 2.0})

        for clip in clips_dict.values():
            np.testing.assert_array_equal(clip["embedding"], parent_vec)


class TestBulkClipReembedMD5UnchangedByRefactor:
    """The MD5 fixup behaviour from ``test_clipper_workflow.py`` keeps
    working with the bulk path: per-clip hashes still come from the
    actual clip bytes, not the parent."""

    def test_audio_md5s_match_clip_bytes(self):
        from vtscore.datasets.load_pipeline import _apply_clipper

        emb = _fake_bulk_embedder()
        clips_dict = {1: _make_audio_media(1, duration=5.1)}
        parent_md5 = clips_dict[1]["md5"]

        with mock.patch("vtscore.media.embedders_for_type", return_value=[emb]):
            _apply_clipper(clips_dict, "sound_tiling", {"duration": 2.0})

        for clip in clips_dict.values():
            assert clip["md5"] != parent_md5
            assert clip["md5"] == hashlib.md5(clip["media_bytes"]).hexdigest()


class TestBulkClipReembedNoTempfile:
    """The pre-refactor path wrote each clip to ``tempfile.mkstemp`` and
    called ``embed_file`` from ``vtscore.detectors.resolver`` one clip
    at a time.  The refactor deletes that detour entirely — verify
    neither code path is invoked from clip re-embed."""

    def test_no_tempfile_mkstemp_calls(self):
        from vtscore.datasets.load_pipeline import _apply_clipper
        import tempfile

        emb = _fake_bulk_embedder()
        clips_dict = {1: _make_audio_media(1, duration=5.1)}

        with (
            mock.patch("vtscore.media.embedders_for_type", return_value=[emb]),
            mock.patch.object(tempfile, "mkstemp", side_effect=AssertionError("clip re-embed must not hit tempfile")),
        ):
            _apply_clipper(clips_dict, "sound_tiling", {"duration": 2.0})

    def test_does_not_import_embed_file(self):
        """``embed_file`` was the single-item embedder dispatch used by
        the old per-clip loop.  The refactor resolves embedders inline
        and never reaches into ``vtscore.detectors.resolver``."""
        from vtscore.datasets.load_pipeline import _apply_clipper
        from vtscore.detectors import resolver

        emb = _fake_bulk_embedder()
        clips_dict = {1: _make_audio_media(1, duration=5.1)}

        with (
            mock.patch("vtscore.media.embedders_for_type", return_value=[emb]),
            mock.patch.object(resolver, "embed_file", side_effect=AssertionError("must not be called")) as guard,
        ):
            _apply_clipper(clips_dict, "sound_tiling", {"duration": 2.0})

        guard.assert_not_called()
