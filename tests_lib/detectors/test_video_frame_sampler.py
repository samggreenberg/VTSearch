"""M20-related tests for ``vtscore.media.video._frame_sampling.sample_video_frames``.

The clip-boundaries / full-range coverage lives in
``tests_lib/detectors/test_video_clip_embedding.py``. This file covers the
two concerns surfaced by the M20 investigation that the clip-boundaries
fix did not address:

* The **single-frame video contract** (logical-bug-audit M20): a 1-frame
  source video must produce ``num_frames`` identical frames with no
  warning; padding to fixed length is the only correct behaviour given
  X-CLIP / LanguageBind / VideoMAE all require a fixed frame stack, and a
  single-frame video genuinely has no temporal variation to encode. This
  test locks in the contract so a future contributor can't accidentally
  "fix" M20 by rejecting short videos.

* The **partial-read warning**: when some of the requested timestamps fail
  to decode (corrupted middle frames, VFR videos whose seeks land oddly,
  codec quirks), the helper silently dropped them and the pad step biased
  the embedding toward the last readable frame. The fix logs a warning so
  the failure is visible. We must also verify the M20 single-frame case
  does NOT trigger this warning (it's the legitimate-short-video path,
  not a partial-read failure).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from vtscore.media.video import decode


class _FakeDecoder:
    """Decode-layer stand-in with configurable per-timestamp success.

    ``read_results`` is consumed one entry per ``frame_at()`` call: ``True``
    yields a frame, ``False`` yields ``None``. Lets us simulate codec quirks
    where some seeks succeed and others don't.
    """

    def __init__(self, *, frame_count: int, fps: float, read_results: list[bool] | None) -> None:
        self._fps = fps
        self._duration = frame_count / fps if fps > 0 else 0.0
        self._read_results = iter(read_results if read_results is not None else [True] * frame_count)

    def probe(self, _path):
        return decode.VideoInfo(duration=self._duration, fps=self._fps, width=1, height=1)

    def frame_at(self, _path, time_seconds: float):
        try:
            ok = next(self._read_results)
        except StopIteration:
            ok = False
        if not ok:
            return None
        # Encode the seek position into the pixel value so different seeks
        # yield distinguishable frames; useful for all-identical assertions.
        value = max(0, min(255, int(round(time_seconds * self._fps))))
        return np.full((1, 1, 3), value, dtype=np.uint8)


@pytest.fixture
def fake_video(monkeypatch):
    """Return an installer that swaps the decode layer for a configured fake."""

    def _factory(*, frame_count: int, fps: float, read_results: list[bool] | None = None) -> _FakeDecoder:
        fake = _FakeDecoder(frame_count=frame_count, fps=fps, read_results=read_results)
        monkeypatch.setattr(decode, "probe", fake.probe)
        monkeypatch.setattr(decode, "frame_at", fake.frame_at)
        return fake

    return _factory


def _media(tmp_path: Path) -> dict:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    return {"media_path": str(video)}


class TestSingleFrameVideoContract:
    """M20 contract: 1-frame videos must produce ``num_frames`` identical frames."""

    def test_one_frame_video_pads_to_num_frames(self, fake_video, tmp_path):
        from vtscore.media.video._frame_sampling import sample_video_frames

        fake_video(frame_count=1, fps=10.0, read_results=[True])
        frames = sample_video_frames(_media(tmp_path), num_frames=8)
        assert len(frames) == 8

    def test_one_frame_video_all_padded_frames_identical(self, fake_video, tmp_path):
        from vtscore.media.video._frame_sampling import sample_video_frames

        fake_video(frame_count=1, fps=10.0, read_results=[True])
        frames = sample_video_frames(_media(tmp_path), num_frames=8)
        first = np.asarray(frames[0])
        for f in frames[1:]:
            assert np.array_equal(first, np.asarray(f))

    def test_one_frame_video_videomae_target(self, fake_video, tmp_path):
        """VideoMAE asks for 16 frames; same contract."""
        from vtscore.media.video._frame_sampling import sample_video_frames

        fake_video(frame_count=1, fps=10.0, read_results=[True])
        frames = sample_video_frames(_media(tmp_path), num_frames=16)
        assert len(frames) == 16

    def test_one_frame_video_does_not_log_warning(self, fake_video, tmp_path, caplog):
        """The legitimate short-video pad path must not trigger a partial-read warning."""
        from vtscore.media.video._frame_sampling import sample_video_frames

        fake_video(frame_count=1, fps=10.0, read_results=[True])
        with caplog.at_level(logging.WARNING, logger="vtscore.media.video._frame_sampling"):
            sample_video_frames(_media(tmp_path), num_frames=8)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert not warnings, "M20 single-frame case must not log a partial-read warning"


class TestPartialReadWarning:
    """A timestamp that fails to decode must log a warning."""

    def test_partial_read_logs_warning(self, fake_video, tmp_path, caplog):
        from vtscore.media.video._frame_sampling import sample_video_frames

        # 8 timestamps requested, only the first 5 succeed → 3 silent drops.
        fake_video(frame_count=100, fps=10.0, read_results=[True] * 5 + [False] * 3)

        with caplog.at_level(logging.WARNING, logger="vtscore.media.video._frame_sampling"):
            frames = sample_video_frames(_media(tmp_path), num_frames=8)

        assert len(frames) == 8  # still padded out
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "expected a warning when some frame reads fail"
        assert "Partial frame-read failure" in warnings[0].getMessage()

    def test_complete_read_does_not_warn(self, fake_video, tmp_path, caplog):
        from vtscore.media.video._frame_sampling import sample_video_frames

        fake_video(frame_count=100, fps=10.0, read_results=[True] * 8)

        with caplog.at_level(logging.WARNING, logger="vtscore.media.video._frame_sampling"):
            frames = sample_video_frames(_media(tmp_path), num_frames=8)

        assert len(frames) == 8
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert not warnings

    def test_all_reads_fail_returns_empty(self, fake_video, tmp_path):
        from vtscore.media.video._frame_sampling import sample_video_frames

        fake_video(frame_count=100, fps=10.0, read_results=[False] * 8)
        frames = sample_video_frames(_media(tmp_path), num_frames=8)
        assert frames == []

    def test_partial_read_with_short_video_warns(self, fake_video, tmp_path, caplog):
        """A 4-frame video where only 2 reads succeed: 4 timestamps requested,
        2 returned → still a partial-read failure (distinct from the
        legitimately-short single-frame case)."""
        from vtscore.media.video._frame_sampling import sample_video_frames

        fake_video(frame_count=4, fps=10.0, read_results=[True, True, False, False])

        with caplog.at_level(logging.WARNING, logger="vtscore.media.video._frame_sampling"):
            frames = sample_video_frames(_media(tmp_path), num_frames=8)

        assert len(frames) == 8
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "partial read on a short video should still warn"


class TestClipBoxCrop:
    """A unit's ``clip_box`` frames what the embedder sees.

    Video units are metadata-only, so a letterbox crop is *honoured* here
    rather than baked into a re-encoded copy (see
    :class:`~vtscore.media.video.cleaner.VideoLetterboxCropCleaner`).
    """

    @pytest.fixture
    def wide_video(self, monkeypatch):
        """Install a decoder serving 40x20 frames with a 4 px bar top/bottom."""
        frame = np.full((20, 40, 3), 200, dtype=np.uint8)
        frame[:4] = 0
        frame[16:] = 0

        def probe(_path):
            return decode.VideoInfo(duration=2.0, fps=10.0, width=40, height=20)

        monkeypatch.setattr(decode, "probe", probe)
        monkeypatch.setattr(decode, "frame_at", lambda _p, _t: frame)

    def test_frames_are_cropped_to_the_box(self, wide_video, tmp_path):
        from vtscore.media.video._frame_sampling import sample_video_frames

        media = _media(tmp_path)
        media["clip_box"] = [0, 4, 40, 16]
        frames = sample_video_frames(media, num_frames=4)
        assert len(frames) == 4
        assert all(f.size == (40, 12) for f in frames)
        # Only the bars were black, so nothing black survives the crop.
        assert all(np.asarray(f).min() == 200 for f in frames)

    def test_no_box_leaves_the_full_frame(self, wide_video, tmp_path):
        from vtscore.media.video._frame_sampling import sample_video_frames

        frames = sample_video_frames(_media(tmp_path), num_frames=2)
        assert all(f.size == (40, 20) for f in frames)

    def test_a_malformed_box_is_ignored_rather_than_fatal(self, wide_video, tmp_path):
        from vtscore.media.video._frame_sampling import sample_video_frames

        media = _media(tmp_path)
        for bad in ("nonsense", [1, 2], [0, 0, 0, 0], None):
            media["clip_box"] = bad
            frames = sample_video_frames(media, num_frames=2)
            assert all(f.size == (40, 20) for f in frames)


class TestUndecodableSource:
    """Guards on the real decode layer, no fake installed."""

    def test_missing_path_returns_empty(self, tmp_path):
        from vtscore.media.video._frame_sampling import sample_video_frames

        frames = sample_video_frames({"media_path": str(tmp_path / "nope.mp4")}, num_frames=8)
        assert frames == []

    def test_undecodable_bytes_return_empty(self):
        from vtscore.media.video._frame_sampling import sample_video_frames

        frames = sample_video_frames({"media_bytes": b"not a video at all"}, num_frames=8)
        assert frames == []


class TestRealSingleFrameVideo:
    """End-to-end check with a real (ffmpeg-encoded) 1-frame MP4."""

    def _make_video(self, frames: int, tmp_path: Path) -> Path:
        from vtscore.utils.synthetic.video import _encode_frames

        path = tmp_path / f"v_{frames}f.mp4"
        _encode_frames(
            path,
            [np.full((64, 64, 3), fill_value=(i * 23) % 256, dtype=np.uint8) for i in range(frames)],
            fps=10,
        )
        return path

    def test_real_one_frame_video_pads_without_warning(self, tmp_path, caplog):
        from vtscore.media.video._frame_sampling import sample_video_frames

        video_path = self._make_video(1, tmp_path)
        with caplog.at_level(logging.WARNING, logger="vtscore.media.video._frame_sampling"):
            frames = sample_video_frames({"media_path": str(video_path)}, num_frames=8)
        assert len(frames) == 8
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert not warnings

    def test_real_multi_frame_video_samples_across_the_clip(self, tmp_path):
        from vtscore.media.video._frame_sampling import sample_video_frames

        video_path = self._make_video(30, tmp_path)
        frames = sample_video_frames({"media_path": str(video_path)}, num_frames=4)
        assert len(frames) == 4
        # Distinct fills per source frame, so distinct samples across the clip.
        means = [float(np.asarray(f).mean()) for f in frames]
        assert len(set(round(m) for m in means)) > 1
