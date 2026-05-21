"""Tests for the video-clip embedding boundary fix.

Before the fix, ``VideoTilingClipper`` tiles all shared the parent video's
embedding because:

  1. ``_build_clip_embed_input`` dropped ``clip_start`` / ``clip_end``;
  2. ``_fixup_clip_md5_and_embeddings`` skipped re-embedding for video
     (since video clippers don't slice bytes);
  3. The video embedders sampled ``np.linspace(0, frame_count - 1, n)``
     unconditionally — ignoring the clip range.

These tests cover all three layers.
"""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# _frame_index_range — boundary math
# ---------------------------------------------------------------------------


class TestFrameIndexRange:
    def test_full_range_when_no_boundaries(self):
        from vtscore.media.video._frame_sampling import _frame_index_range

        media: dict[str, Any] = {}
        start, end = _frame_index_range(media, frame_count=100, fps=10.0)
        assert (start, end) == (0, 99)

    def test_clip_boundaries_map_to_indices(self):
        from vtscore.media.video._frame_sampling import _frame_index_range

        media = {"clip_start": 2.0, "clip_end": 5.0}
        start, end = _frame_index_range(media, frame_count=100, fps=10.0)
        # 2.0s @ 10fps = frame 20; 5.0s @ 10fps -> last frame = 50 - 1 = 49.
        assert start == 20
        assert end == 49

    def test_clip_boundaries_clamped_to_video(self):
        from vtscore.media.video._frame_sampling import _frame_index_range

        media = {"clip_start": 99.0, "clip_end": 200.0}
        start, end = _frame_index_range(media, frame_count=100, fps=10.0)
        # clip_start far past end: clamp to last frame; end must not precede start.
        assert start == 99
        assert end == 99

    def test_falls_back_to_full_when_fps_zero(self):
        from vtscore.media.video._frame_sampling import _frame_index_range

        media = {"clip_start": 1.0, "clip_end": 2.0}
        start, end = _frame_index_range(media, frame_count=50, fps=0.0)
        assert (start, end) == (0, 49)

    def test_falls_back_to_full_when_end_le_start(self):
        from vtscore.media.video._frame_sampling import _frame_index_range

        media = {"clip_start": 3.0, "clip_end": 3.0}
        start, end = _frame_index_range(media, frame_count=50, fps=10.0)
        assert (start, end) == (0, 49)

    def test_distinct_tiles_get_distinct_ranges(self):
        from vtscore.media.video._frame_sampling import _frame_index_range

        # Parent: 10s @ 25fps -> 250 frames.  Tile into 2s segments.
        tile_a = {"clip_start": 0.0, "clip_end": 2.0}
        tile_b = {"clip_start": 2.0, "clip_end": 4.0}
        tile_c = {"clip_start": 8.0, "clip_end": 10.0}

        a = _frame_index_range(tile_a, frame_count=250, fps=25.0)
        b = _frame_index_range(tile_b, frame_count=250, fps=25.0)
        c = _frame_index_range(tile_c, frame_count=250, fps=25.0)

        assert a == (0, 49)
        assert b == (50, 99)
        assert c == (200, 249)


# ---------------------------------------------------------------------------
# sample_video_frames — uses clip_start/clip_end to pick frame indices
# ---------------------------------------------------------------------------


class _FakeCapture:
    """Stand-in for ``cv2.VideoCapture`` that records seeks.

    Returns deterministic 1x1 BGR frames so the helper produces a list of
    PIL Images we can introspect.  ``positions_seen`` exposes every frame
    index the caller seeked to via ``CAP_PROP_POS_FRAMES``.
    """

    def __init__(self, *, frame_count: int, fps: float):
        self._frame_count = frame_count
        self._fps = fps
        self.positions_seen: list[int] = []
        self._next_pos = 0

    def isOpened(self) -> bool:  # noqa: N802 (cv2 API)
        return True

    def get(self, prop: int) -> float:
        import cv2  # noqa: PLC0415

        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._frame_count)
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        import cv2  # noqa: PLC0415

        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._next_pos = int(value)
            self.positions_seen.append(self._next_pos)
        return True

    def read(self):
        # Encode the position into the frame so different seeks produce
        # different PIL Images — useful as a sanity check.
        b = max(0, min(255, self._next_pos))
        frame = np.full((1, 1, 3), b, dtype=np.uint8)
        return True, frame

    def release(self) -> None:
        pass


@pytest.fixture
def fake_video(monkeypatch):
    """Patch cv2.VideoCapture with a captured _FakeCapture instance."""
    import cv2

    captures: list[_FakeCapture] = []

    def _factory(*, frame_count: int, fps: float):
        def _ctor(_path):
            cap = _FakeCapture(frame_count=frame_count, fps=fps)
            captures.append(cap)
            return cap

        monkeypatch.setattr(cv2, "VideoCapture", _ctor)
        return captures

    return _factory


class TestSampleVideoFrames:
    def test_full_range_sampled_without_boundaries(self, fake_video, tmp_path):
        from vtscore.media.video._frame_sampling import sample_video_frames

        captures = fake_video(frame_count=100, fps=10.0)
        video = tmp_path / "v.mp4"
        video.write_bytes(b"fake")
        frames = sample_video_frames({"media_path": str(video)}, num_frames=8)

        assert len(frames) == 8
        assert len(captures) == 1
        # 8 frames linspace'd over [0, 99].
        seen = captures[0].positions_seen
        assert seen[0] == 0
        assert seen[-1] == 99

    def test_clip_boundaries_restrict_sampled_frames(self, fake_video, tmp_path):
        from vtscore.media.video._frame_sampling import sample_video_frames

        captures = fake_video(frame_count=250, fps=25.0)
        video = tmp_path / "v.mp4"
        video.write_bytes(b"fake")

        frames = sample_video_frames(
            {"media_path": str(video), "clip_start": 2.0, "clip_end": 4.0},
            num_frames=8,
        )
        assert len(frames) == 8
        seen = captures[0].positions_seen
        # Range [50, 99] from clip_start=2.0s and clip_end=4.0s @ 25fps.
        assert min(seen) >= 50
        assert max(seen) <= 99
        assert seen[0] == 50
        assert seen[-1] == 99

    def test_distinct_tiles_produce_distinct_frame_indices(self, fake_video, tmp_path):
        from vtscore.media.video._frame_sampling import sample_video_frames

        # Tile A and Tile B share the same media_path/bytes but cover
        # disjoint time ranges; the frame indices the embedder samples
        # must therefore be disjoint.
        captures = fake_video(frame_count=250, fps=25.0)
        video = tmp_path / "v.mp4"
        video.write_bytes(b"fake")

        sample_video_frames(
            {"media_path": str(video), "clip_start": 0.0, "clip_end": 2.0},
            num_frames=8,
        )
        sample_video_frames(
            {"media_path": str(video), "clip_start": 8.0, "clip_end": 10.0},
            num_frames=8,
        )
        a_positions = set(captures[0].positions_seen)
        b_positions = set(captures[1].positions_seen)
        assert a_positions.isdisjoint(b_positions)

    def test_falls_back_to_tempfile_when_no_path(self, fake_video):
        from vtscore.media.video._frame_sampling import sample_video_frames

        captures = fake_video(frame_count=100, fps=10.0)
        frames = sample_video_frames({"media_bytes": b"\x00\x01\x02"}, num_frames=8)
        assert len(frames) == 8
        assert len(captures) == 1

    def test_returns_empty_without_path_or_bytes(self):
        from vtscore.media.video._frame_sampling import sample_video_frames

        assert sample_video_frames({}, num_frames=8) == []


# ---------------------------------------------------------------------------
# _build_clip_embed_input — propagates clip metadata for video
# ---------------------------------------------------------------------------


class TestBuildClipEmbedInput:
    def test_video_includes_clip_boundaries(self):
        from vtscore.datasets.load_pipeline import _build_clip_embed_input

        clip = {
            "origin_name": "x.mp4",
            "filename": "x.mp4",
            "media_bytes": b"parent",
            "media_path": "/data/x.mp4",
            "clip_start": 1.0,
            "clip_end": 3.0,
            "clip_index": 1,
        }
        out = _build_clip_embed_input(clip, "video")
        assert out["media_bytes"] == b"parent"
        assert out["media_path"] == "/data/x.mp4"
        assert out["clip_start"] == 1.0
        assert out["clip_end"] == 3.0

    def test_video_without_path_still_carries_boundaries(self):
        from vtscore.datasets.load_pipeline import _build_clip_embed_input

        clip = {
            "origin_name": "x.mp4",
            "filename": "x.mp4",
            "media_bytes": b"parent",
            "clip_start": 0.0,
            "clip_end": 2.0,
        }
        out = _build_clip_embed_input(clip, "video")
        assert "media_path" not in out
        assert out["clip_start"] == 0.0
        assert out["clip_end"] == 2.0

    def test_audio_unaffected(self):
        from vtscore.datasets.load_pipeline import _build_clip_embed_input

        clip = {
            "origin_name": "a.wav",
            "filename": "a.wav",
            "media_bytes": b"sliced",
            "clip_start": 1.0,
            "clip_end": 2.0,
        }
        out = _build_clip_embed_input(clip, "audio")
        assert out["media_bytes"] == b"sliced"
        # Audio clipper slices bytes, so it has no need for boundary metadata.
        assert "clip_start" not in out
        assert "clip_end" not in out


# ---------------------------------------------------------------------------
# _fixup_clip_md5_and_embeddings — video tiles each get their own embedding
# ---------------------------------------------------------------------------


def _make_tiled_video_clips() -> list[dict]:
    """Build three tiles of one parent video — same bytes, distinct boundaries."""
    parent_bytes = b"parent-video-bytes"
    parent_md5 = hashlib.md5(parent_bytes).hexdigest()
    tiles = []
    for idx, (t0, t1) in enumerate([(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]):
        tiles.append({
            "id": 100 + idx,
            "type": "video",
            "media_bytes": parent_bytes,
            "media_path": "/data/parent.mp4",
            "md5": parent_md5,  # stale; the fixup should rewrite it
            "duration": 2.0,
            "clip_index": idx,
            "clip_start": t0,
            "clip_end": t1,
            "filename": "parent.mp4",
            "origin_name": "parent.mp4",
            "embedding": None,
        })
    return tiles


class TestFixupClipMd5AndEmbeddingsVideo:
    def test_tiles_get_distinct_md5(self):
        from vtscore.datasets.load_pipeline import _fixup_clip_md5_and_embeddings

        clips = _make_tiled_video_clips()
        # Stub the embedder so we don't try to decode a fake mp4.
        fake = MagicMock()
        fake._on_progress = lambda *a, **k: None
        fake.embed_media_bulk.return_value = [np.array([0.0], dtype=np.float32)] * len(clips)

        with patch(
            "vtscore.datasets.load_pipeline._resolve_clip_embedder",
            return_value=fake,
        ):
            _fixup_clip_md5_and_embeddings(clips, [True] * len(clips), "video")

        md5s = [c["md5"] for c in clips]
        assert len(set(md5s)) == len(clips), f"Tiles share MD5: {md5s}"

    def test_tiles_get_distinct_embeddings(self):
        """The embedder must see each tile's boundaries and produce distinct
        vectors — modelled here by returning ``clip_start`` as the embedding."""
        from vtscore.datasets.load_pipeline import _fixup_clip_md5_and_embeddings

        clips = _make_tiled_video_clips()

        def fake_bulk(inputs: list[dict]) -> list[np.ndarray]:
            # Sanity: every input must carry the tile's clip boundaries.
            assert all("clip_start" in m and "clip_end" in m for m in inputs), inputs
            # Emit one-hot-ish vectors keyed by the tile's clip_start so the
            # caller can verify each tile gets its own embedding.
            return [np.array([m["clip_start"], m["clip_end"]], dtype=np.float32) for m in inputs]

        fake = MagicMock()
        fake._on_progress = lambda *a, **k: None
        fake.embed_media_bulk.side_effect = fake_bulk

        with patch(
            "vtscore.datasets.load_pipeline._resolve_clip_embedder",
            return_value=fake,
        ):
            _fixup_clip_md5_and_embeddings(clips, [True] * len(clips), "video")

        embs = [c["embedding"] for c in clips]
        assert all(e is not None for e in embs)
        # Tile boundaries: (0,2), (2,4), (4,6) -> vectors must all differ.
        assert {tuple(e.tolist()) for e in embs} == {(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)}
