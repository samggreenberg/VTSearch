"""Tests for the video-clip embedding boundary fix.

Before the fix, ``VideoTilingClipper`` tiles all shared the parent video's
embedding because:

  1. ``_build_clip_embed_input`` dropped ``clip_start`` / ``clip_end``;
  2. ``_fixup_clip_md5_and_embeddings`` skipped re-embedding for video
     (since video clippers don't slice bytes);
  3. The video embedders sampled the whole video unconditionally;
     ignoring the clip range.

These tests cover all three layers.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vtscore.embedding.media_vectors import media_embedding
from vtscore.media.video import decode
from vtscore.utils.hashing import content_md5


# ---------------------------------------------------------------------------
# _frame_time_range: boundary math
# ---------------------------------------------------------------------------


def _info(duration: float, fps: float) -> decode.VideoInfo:
    return decode.VideoInfo(duration=duration, fps=fps, width=1, height=1)


class TestFrameTimeRange:
    def test_full_range_when_no_boundaries(self):
        from vtscore.media.video._frame_sampling import _frame_time_range

        media: dict[str, Any] = {}
        start, end = _frame_time_range(media, _info(10.0, 10.0))
        # Stops one frame short of the duration, where nothing decodes.
        assert start == pytest.approx(0.0)
        assert end == pytest.approx(9.9)

    def test_clip_boundaries_map_to_times(self):
        from vtscore.media.video._frame_sampling import _frame_time_range

        media = {"clip_start": 2.0, "clip_end": 5.0}
        start, end = _frame_time_range(media, _info(10.0, 10.0))
        assert start == pytest.approx(2.0)
        assert end == pytest.approx(4.9)  # 5.0 minus one 10fps frame

    def test_clip_boundaries_clamped_to_video(self):
        from vtscore.media.video._frame_sampling import _frame_time_range

        media = {"clip_start": 99.0, "clip_end": 200.0}
        start, end = _frame_time_range(media, _info(10.0, 10.0))
        # clip_start far past the end: clamp to the last frame; end must not
        # precede start.
        assert start == pytest.approx(9.9)
        assert end == pytest.approx(9.9)

    def test_boundaries_honoured_when_fps_unknown(self):
        from vtscore.media.video._frame_sampling import _frame_time_range

        # Seconds don't need a frame rate to be meaningful, so an unknown fps
        # no longer forces the full range.
        media = {"clip_start": 1.0, "clip_end": 2.0}
        start, end = _frame_time_range(media, _info(5.0, 0.0))
        assert start == pytest.approx(1.0)
        assert end == pytest.approx(1.96)

    def test_falls_back_to_full_when_end_le_start(self):
        from vtscore.media.video._frame_sampling import _frame_time_range

        media = {"clip_start": 3.0, "clip_end": 3.0}
        start, end = _frame_time_range(media, _info(5.0, 10.0))
        assert start == pytest.approx(0.0)
        assert end == pytest.approx(4.9)

    def test_distinct_tiles_get_distinct_ranges(self):
        from vtscore.media.video._frame_sampling import _frame_time_range

        # Parent: 10s @ 25fps.  Tile into 2s segments.
        info = _info(10.0, 25.0)
        a = _frame_time_range({"clip_start": 0.0, "clip_end": 2.0}, info)
        b = _frame_time_range({"clip_start": 2.0, "clip_end": 4.0}, info)
        c = _frame_time_range({"clip_start": 8.0, "clip_end": 10.0}, info)

        assert a == pytest.approx((0.0, 1.96))
        assert b == pytest.approx((2.0, 3.96))
        assert c == pytest.approx((8.0, 9.96))


# ---------------------------------------------------------------------------
# sample_video_frames: uses clip_start/clip_end to pick timestamps
# ---------------------------------------------------------------------------


class _FakeDecoder:
    """Decode-layer stand-in that records every timestamp it was asked for.

    Returns deterministic 1x1 RGB frames so the helper produces a list of PIL
    Images we can introspect.
    """

    def __init__(self, *, frame_count: int, fps: float):
        self._fps = fps
        self._duration = frame_count / fps if fps > 0 else 0.0
        self.times_seen: list[float] = []

    def probe(self, _path):
        return decode.VideoInfo(duration=self._duration, fps=self._fps, width=1, height=1)

    def frame_at(self, _path, time_seconds: float):
        self.times_seen.append(float(time_seconds))
        value = max(0, min(255, int(round(time_seconds * self._fps))))
        return np.full((1, 1, 3), value, dtype=np.uint8)


@pytest.fixture
def fake_video(monkeypatch):
    """Return an installer that swaps the decode layer for a recording fake."""

    def _factory(*, frame_count: int, fps: float) -> _FakeDecoder:
        fake = _FakeDecoder(frame_count=frame_count, fps=fps)
        monkeypatch.setattr(decode, "probe", fake.probe)
        monkeypatch.setattr(decode, "frame_at", fake.frame_at)
        return fake

    return _factory


class TestSampleVideoFrames:
    def test_full_range_sampled_without_boundaries(self, fake_video, tmp_path):
        from vtscore.media.video._frame_sampling import sample_video_frames

        fake = fake_video(frame_count=100, fps=10.0)
        video = tmp_path / "v.mp4"
        video.write_bytes(b"fake")
        frames = sample_video_frames({"media_path": str(video)}, num_frames=8)

        assert len(frames) == 8
        # 8 timestamps linspace'd over the whole 10s video.
        assert fake.times_seen[0] == pytest.approx(0.0)
        assert fake.times_seen[-1] == pytest.approx(9.9)

    def test_clip_boundaries_restrict_sampled_frames(self, fake_video, tmp_path):
        from vtscore.media.video._frame_sampling import sample_video_frames

        fake = fake_video(frame_count=250, fps=25.0)
        video = tmp_path / "v.mp4"
        video.write_bytes(b"fake")

        frames = sample_video_frames(
            {"media_path": str(video), "clip_start": 2.0, "clip_end": 4.0},
            num_frames=8,
        )
        assert len(frames) == 8
        seen = fake.times_seen
        assert min(seen) >= 2.0
        assert max(seen) <= 4.0
        assert seen[0] == pytest.approx(2.0)
        assert seen[-1] == pytest.approx(3.96)

    def test_distinct_tiles_produce_distinct_timestamps(self, fake_video, tmp_path):
        from vtscore.media.video._frame_sampling import sample_video_frames

        # Tile A and Tile B share the same media_path/bytes but cover
        # disjoint time ranges; the timestamps the embedder samples must
        # therefore be disjoint.
        fake = fake_video(frame_count=250, fps=25.0)
        video = tmp_path / "v.mp4"
        video.write_bytes(b"fake")

        sample_video_frames(
            {"media_path": str(video), "clip_start": 0.0, "clip_end": 2.0},
            num_frames=8,
        )
        a_times = set(fake.times_seen)
        fake.times_seen.clear()
        sample_video_frames(
            {"media_path": str(video), "clip_start": 8.0, "clip_end": 10.0},
            num_frames=8,
        )
        assert a_times.isdisjoint(set(fake.times_seen))

    def test_falls_back_to_tempfile_when_no_path(self, fake_video):
        from vtscore.media.video._frame_sampling import sample_video_frames

        fake = fake_video(frame_count=100, fps=10.0)
        frames = sample_video_frames({"media_bytes": b"\x00\x01\x02"}, num_frames=8)
        assert len(frames) == 8
        assert len(fake.times_seen) == 8

    def test_returns_empty_without_path_or_bytes(self):
        from vtscore.media.video._frame_sampling import sample_video_frames

        assert sample_video_frames({}, num_frames=8) == []


# ---------------------------------------------------------------------------
# _build_clip_embed_input: propagates clip metadata for video
# ---------------------------------------------------------------------------


class TestBuildClipEmbedInput:
    def test_video_includes_clip_boundaries(self):
        from vtscore.datasets.stages.clipper import _build_clip_embed_input

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
        from vtscore.datasets.stages.clipper import _build_clip_embed_input

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
        from vtscore.datasets.stages.clipper import _build_clip_embed_input

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
# _fixup_clip_md5_and_embeddings: video tiles each get their own embedding
# ---------------------------------------------------------------------------


def _make_tiled_video_clips() -> list[dict]:
    """Build three tiles of one parent video; same bytes, distinct boundaries."""
    parent_bytes = b"parent-video-bytes"
    parent_md5 = content_md5(parent_bytes)
    tiles = []
    for idx, (t0, t1) in enumerate([(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]):
        tiles.append(
            {
                "id": 100 + idx,
                "media_type": "video",
                "media_bytes": parent_bytes,
                "media_path": "/data/parent.mp4",
                "md5": parent_md5,  # stale; the fixup should rewrite it
                "duration": 2.0,
                "clip_index": idx,
                "clip_start": t0,
                "clip_end": t1,
                "filename": "parent.mp4",
                "origin_name": "parent.mp4",
                "embeddings": {},
            }
        )
    return tiles


class TestFixupClipMd5AndEmbeddingsVideo:
    def test_tiles_get_distinct_md5(self):
        from vtscore.datasets.stages.clipper import _fixup_clip_md5_and_embeddings

        clips = _make_tiled_video_clips()
        # Stub the embedder so we don't try to decode a fake mp4.
        fake = MagicMock()
        fake._on_progress = lambda *a, **k: None
        fake.embed_media_bulk.return_value = [np.array([0.0], dtype=np.float32)] * len(clips)

        with patch(
            "vtscore.datasets.stages.clipper._resolve_clip_embedder",
            return_value=fake,
        ):
            _fixup_clip_md5_and_embeddings(clips, [True] * len(clips), "video")

        md5s = [c["md5"] for c in clips]
        assert len(set(md5s)) == len(clips), f"Tiles share MD5: {md5s}"

    def test_tiles_get_distinct_embeddings(self):
        """The embedder must see each tile's boundaries and produce distinct
        vectors; modelled here by returning ``clip_start`` as the embedding."""
        from vtscore.datasets.stages.clipper import _fixup_clip_md5_and_embeddings

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
            "vtscore.datasets.stages.clipper._resolve_clip_embedder",
            return_value=fake,
        ):
            _fixup_clip_md5_and_embeddings(clips, [True] * len(clips), "video")

        embs = [media_embedding(c) for c in clips]
        assert all(e is not None for e in embs)
        # Tile boundaries: (0,2), (2,4), (4,6) -> vectors must all differ.
        assert {tuple(e.tolist()) for e in embs} == {(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)}
