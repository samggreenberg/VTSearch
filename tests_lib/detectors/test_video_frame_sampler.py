"""M20-related tests for ``vtscore.media.video._frame_sampling.sample_video_frames``.

The clip-boundaries / full-range coverage lives in
``tests_lib/detectors/test_video_clip_embedding.py``. This file covers the
two concerns surfaced by the M20 investigation that the clip-boundaries
fix did not address:

* The **single-frame video contract** (logical-bug-audit M20): a 1-frame
  source video must produce ``num_frames`` identical frames with no
  warning - padding to fixed length is the only correct behaviour given
  X-CLIP / LanguageBind / VideoMAE all require a fixed frame stack, and a
  single-frame video genuinely has no temporal variation to encode. This
  test locks in the contract so a future contributor can't accidentally
  "fix" M20 by rejecting short videos.

* The **partial-read warning**: when ``cap.read()`` returns ``ret=False``
  for some of the requested indices (corrupted middle frames, VFR videos
  where ``CAP_PROP_POS_FRAMES`` seek doesn't actually move, codec
  quirks), the helper silently dropped them and the pad step biased the
  embedding toward the last readable frame. The fix logs a warning so
  the failure is visible. We must also verify the M20 single-frame case
  does NOT trigger this warning (it's the legitimate-short-video path,
  not a partial-read failure).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest


class _FakeCapture:
    """``cv2.VideoCapture`` stand-in with configurable per-read success.

    ``read_results`` is consumed one entry per ``read()`` call. ``True`` →
    return a (True, frame) pair; ``False`` → return (False, None). Lets us
    simulate codec quirks where some seeks succeed and others don't.
    """

    def __init__(self, *, frame_count: int, fps: float, read_results: list[bool]) -> None:
        self._frame_count = frame_count
        self._fps = fps
        self._read_results = iter(read_results)
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
        return True

    def read(self) -> tuple[bool, Any]:
        try:
            ok = next(self._read_results)
        except StopIteration:
            ok = False
        if not ok:
            return False, None
        # Encode position into the pixel value so different seeks yield
        # distinguishable frames; useful for the all-identical assertions.
        b = max(0, min(255, self._next_pos))
        frame = np.full((1, 1, 3), b, dtype=np.uint8)
        return True, frame

    def release(self) -> None:
        pass


@pytest.fixture
def fake_video(monkeypatch):
    """Build and install a ``_FakeCapture`` factory; return the captures list."""

    def _factory(*, frame_count: int, fps: float, read_results: list[bool] | None = None):
        import cv2

        captures: list[_FakeCapture] = []

        def _ctor(_path):
            # Default: every read succeeds - fast path for tests that don't
            # care about partial-read behaviour.
            results = read_results if read_results is not None else [True] * frame_count
            cap = _FakeCapture(frame_count=frame_count, fps=fps, read_results=list(results))
            captures.append(cap)
            return cap

        monkeypatch.setattr(cv2, "VideoCapture", _ctor)
        return captures

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
    """``cap.read()`` returning False for some indices must log a warning."""

    def test_partial_read_logs_warning(self, fake_video, tmp_path, caplog):
        from vtscore.media.video._frame_sampling import sample_video_frames

        # 8 indices requested, only the first 5 succeed → 3 silent drops.
        fake_video(frame_count=100, fps=10.0, read_results=[True] * 5 + [False] * 3)

        with caplog.at_level(logging.WARNING, logger="vtscore.media.video._frame_sampling"):
            frames = sample_video_frames(_media(tmp_path), num_frames=8)

        assert len(frames) == 8  # still padded out
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "expected a warning when some cap.read() calls fail"
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
        """A 4-frame video where only 2 reads succeed: 4 indices requested,
        2 returned → still a partial-read failure (distinct from the
        legitimately-short single-frame case)."""
        from vtscore.media.video._frame_sampling import sample_video_frames

        fake_video(frame_count=4, fps=10.0, read_results=[True, True, False, False])

        with caplog.at_level(logging.WARNING, logger="vtscore.media.video._frame_sampling"):
            frames = sample_video_frames(_media(tmp_path), num_frames=8)

        assert len(frames) == 8
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "partial read on a short video should still warn"


class TestRealSingleFrameVideo:
    """End-to-end check with a real (cv2-encoded) 1-frame MP4."""

    def _make_video(self, frames: int, tmp_path: Path) -> Path:
        try:
            import cv2
        except ImportError:
            pytest.skip("OpenCV not installed")
        path = tmp_path / f"v_{frames}f.mp4"
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp = Path(f.name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # pyright: ignore[reportAttributeAccessIssue]
        writer = cv2.VideoWriter(str(tmp), fourcc, 10.0, (64, 64))
        for i in range(frames):
            frame = np.full((64, 64, 3), fill_value=(i * 23) % 256, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        path.write_bytes(tmp.read_bytes())
        tmp.unlink(missing_ok=True)
        return path

    def test_real_one_frame_video_pads_without_warning(self, tmp_path, caplog):
        from vtscore.media.video._frame_sampling import sample_video_frames

        video_path = self._make_video(1, tmp_path)
        with caplog.at_level(logging.WARNING, logger="vtscore.media.video._frame_sampling"):
            frames = sample_video_frames({"media_path": str(video_path)}, num_frames=8)
        assert len(frames) == 8
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert not warnings
