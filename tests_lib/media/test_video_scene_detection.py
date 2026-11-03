"""Library-tier tests for the video scene detector's histogram machinery.

``_detect_scene_boundaries`` used to lean on OpenCV for both decoding and
the colour-histogram comparison.  Decoding now goes through ffmpeg (see
:mod:`vtscore.media.video.decode`) and the histogram work is numpy/PIL, so
these tests pin down the replacements: the histogram's shape and
normalisation, the correlation's agreement with ``HISTCMP_CORREL``
semantics, and an end-to-end detection over a real clip with hard cuts.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.media.video.clipper import (
    _detect_scene_boundaries,
    _histogram_correlation,
    _hue_saturation_histogram,
)
from vtscore.utils.synthetic.video import _encode_frames


def _solid(color: tuple[int, int, int], size: int = 32) -> np.ndarray:
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:] = color
    return frame


def _textured(color: tuple[int, int, int], seed: int, size: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = _solid(color, size)
    noise = rng.integers(0, 40, size=(size, size, 3), dtype=np.uint8)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


class TestHueSaturationHistogram:
    def test_shape_matches_the_declared_bins(self):
        hist = _hue_saturation_histogram(_textured((200, 30, 30), seed=1))
        assert hist.shape == (50, 60)

    def test_is_l2_normalised(self):
        hist = _hue_saturation_histogram(_textured((30, 200, 30), seed=2))
        assert float(np.linalg.norm(hist)) == pytest.approx(1.0)

    def test_black_frame_normalises_without_dividing_by_zero(self):
        # A pure black frame has zero saturation everywhere: one populated bin,
        # so the norm is finite and the result stays L2-normalised.
        hist = _hue_saturation_histogram(_solid((0, 0, 0)))
        assert np.isfinite(hist).all()
        assert float(np.linalg.norm(hist)) == pytest.approx(1.0)

    def test_different_hues_populate_different_bins(self):
        red = _hue_saturation_histogram(_textured((220, 20, 20), seed=3))
        blue = _hue_saturation_histogram(_textured((20, 20, 220), seed=3))
        assert not np.array_equal(red, blue)


class TestHistogramCorrelation:
    def test_identical_histograms_correlate_perfectly(self):
        hist = _hue_saturation_histogram(_textured((200, 30, 30), seed=4))
        assert _histogram_correlation(hist, hist) == pytest.approx(1.0)

    def test_similar_frames_correlate_highly(self):
        a = _hue_saturation_histogram(_textured((200, 30, 30), seed=5))
        b = _hue_saturation_histogram(_textured((200, 30, 30), seed=6))
        assert _histogram_correlation(a, b) > 0.5

    def test_different_scenes_correlate_weakly(self):
        a = _hue_saturation_histogram(_textured((220, 20, 20), seed=7))
        b = _hue_saturation_histogram(_textured((20, 20, 220), seed=8))
        assert _histogram_correlation(a, b) < 0.3

    def test_flat_histogram_reads_as_no_change(self):
        # Undefined correlation must not manufacture a scene boundary.
        flat = np.full((50, 60), 0.5)
        other = _hue_saturation_histogram(_textured((200, 30, 30), seed=9))
        assert _histogram_correlation(flat, other) == pytest.approx(1.0)
        assert _histogram_correlation(flat, flat) == pytest.approx(1.0)


class TestDetectSceneBoundaries:
    def _three_scene_clip(self, tmp_path, fps: int = 10, seconds: int = 3):
        """A clip of three hard-cut scenes, each *seconds* long."""
        frames: list[np.ndarray] = []
        for color in ((200, 30, 30), (30, 30, 200), (30, 200, 30)):
            for i in range(fps * seconds):
                frame = _solid(color, size=64)
                frame[10:30, 10 + i : 30 + i] = (255, 255, 255)  # motion within the scene
                frames.append(frame)
        path = tmp_path / "scenes.mp4"
        _encode_frames(path, frames, fps=fps)
        return path

    def test_finds_the_hard_cuts(self, tmp_path):
        path = self._three_scene_clip(tmp_path)
        boundaries = _detect_scene_boundaries(str(path), threshold=0.3, min_scene_duration=1.0)
        assert boundaries == pytest.approx([3.0, 6.0])

    def test_min_scene_duration_suppresses_close_boundaries(self, tmp_path):
        path = self._three_scene_clip(tmp_path)
        boundaries = _detect_scene_boundaries(str(path), threshold=0.3, min_scene_duration=5.0)
        # The 6.0s cut is 3s after the 3.0s one, so it can't open a new scene;
        # the trailing-scene guard then drops what's left.
        assert boundaries == []

    def test_single_scene_has_no_boundaries(self, tmp_path):
        frames = [_textured((200, 30, 30), seed=i, size=64) for i in range(30)]
        path = tmp_path / "one.mp4"
        _encode_frames(path, frames, fps=10)
        assert _detect_scene_boundaries(str(path), threshold=0.3, min_scene_duration=1.0) == []

    def test_undecodable_video_returns_no_boundaries(self, tmp_path):
        junk = tmp_path / "junk.mp4"
        junk.write_bytes(b"not a video at all")
        assert _detect_scene_boundaries(str(junk), threshold=0.3, min_scene_duration=1.0) == []

    def test_missing_video_returns_no_boundaries(self, tmp_path):
        assert _detect_scene_boundaries(str(tmp_path / "nope.mp4"), 0.3, 1.0) == []
