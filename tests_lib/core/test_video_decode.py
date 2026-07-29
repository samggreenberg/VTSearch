"""Library-tier tests for :mod:`vtscore.media.video.decode`.

The decode layer exists to keep OpenCV - and the OpenSSL its wheel vendors,
which aborts the interpreter on FIPS-enabled hosts - out of the process, so
these tests pin down both halves of it:

* the **banner parser**, exercised against canned ``ffmpeg -i`` output so the
  metadata edge cases (no video stream, unknown duration, odd resolutions)
  are covered without spawning anything;
* the **real decode round-trip**, against short mp4s encoded on the fly by the
  same bundled ffmpeg, so probing and frame extraction are checked end to end.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.media.video import decode
from vtscore.utils.synthetic.video import _encode_frames

# A representative ``ffmpeg -i`` banner for a plain H.264 mp4.
_BANNER = """
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'clip.mp4':
  Metadata:
    major_brand     : isom
  Duration: 00:01:03.48, start: 0.000000, bitrate: 1234 kb/s
  Stream #0:0(und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709), \
1920x1080 [SAR 1:1 DAR 16:9], 1200 kb/s, 29.97 fps, 29.97 tbr, 30k tbn (default)
  Stream #0:1(und): Audio: aac (LC) (mp4a / 0x6134706D), 48000 Hz, stereo, fltp, 128 kb/s
At least one output file must be specified
"""


# ---------------------------------------------------------------------------
# Banner parsing
# ---------------------------------------------------------------------------


class TestParseProbe:
    def test_parses_duration_fps_and_size(self):
        info = decode._parse_probe(_BANNER)
        assert info is not None
        assert info.duration == pytest.approx(63.48)
        assert info.fps == pytest.approx(29.97)
        assert (info.width, info.height) == (1920, 1080)

    def test_fourcc_tag_is_not_mistaken_for_a_resolution(self):
        info = decode._parse_probe(_BANNER)
        assert info is not None
        assert info.width == 1920  # not parsed out of "0x31637661"

    def test_no_video_stream_returns_none(self):
        audio_only = "\n".join(line for line in _BANNER.splitlines() if "Video:" not in line)
        assert decode._parse_probe(audio_only) is None

    def test_missing_file_output_returns_none(self):
        assert decode._parse_probe("clip.mp4: No such file or directory\n") is None

    def test_unknown_duration_is_zero(self):
        banner = _BANNER.replace("Duration: 00:01:03.48", "Duration: N/A")
        info = decode._parse_probe(banner)
        assert info is not None
        assert info.duration == 0.0
        assert info.fps == pytest.approx(29.97)

    def test_hours_are_carried(self):
        banner = _BANNER.replace("00:01:03.48", "02:03:04.50")
        info = decode._parse_probe(banner)
        assert info is not None
        assert info.duration == pytest.approx(2 * 3600 + 3 * 60 + 4.5)


class TestVideoInfo:
    def test_frame_count_from_duration_and_fps(self):
        assert decode.VideoInfo(duration=10.0, fps=30.0).frame_count == 300

    def test_frame_count_zero_when_unknown(self):
        assert decode.VideoInfo(duration=0.0, fps=30.0).frame_count == 0
        assert decode.VideoInfo(duration=10.0, fps=0.0).frame_count == 0

    def test_frame_count_never_rounds_a_real_video_to_zero(self):
        assert decode.VideoInfo(duration=0.01, fps=1.0).frame_count == 1

    def test_frame_seconds_from_fps(self):
        assert decode.VideoInfo(duration=1.0, fps=25.0).frame_seconds == pytest.approx(0.04)

    def test_frame_seconds_falls_back_when_fps_unknown(self):
        assert decode.VideoInfo(duration=1.0, fps=0.0).frame_seconds == pytest.approx(0.04)


class TestScaledSize:
    def test_leaves_narrow_frames_alone(self):
        assert decode._scaled_size(320, 240, 640) == (320, 240)

    def test_no_max_width_is_a_passthrough(self):
        assert decode._scaled_size(1920, 1080, None) == (1920, 1080)

    def test_clamps_and_preserves_aspect(self):
        assert decode._scaled_size(1920, 1080, 320) == (320, 180)

    def test_never_scales_height_to_zero(self):
        assert decode._scaled_size(1000, 2, 10) == (10, 1)


# ---------------------------------------------------------------------------
# Real decode round-trip
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> tuple[object, list[np.ndarray]]:
    """A 3 s, 10 fps, 64x48 mp4 whose frames step through distinct fills."""
    frames = [np.full((48, 64, 3), (i * 8) % 256, dtype=np.uint8) for i in range(30)]
    path = tmp_path_factory.mktemp("decode") / "clip.mp4"
    _encode_frames(path, frames, fps=10)
    return path, frames


class TestProbeRealFile:
    def test_reports_duration_fps_and_size(self, clip):
        path, _ = clip
        info = decode.probe(path)
        assert info is not None
        assert info.duration == pytest.approx(3.0, abs=0.2)
        assert info.fps == pytest.approx(10.0)
        assert (info.width, info.height) == (64, 48)
        assert info.frame_count == pytest.approx(30, abs=2)

    def test_missing_file_returns_none(self, tmp_path):
        assert decode.probe(tmp_path / "nope.mp4") is None

    def test_non_video_file_returns_none(self, tmp_path):
        junk = tmp_path / "junk.mp4"
        junk.write_bytes(b"not a video at all")
        assert decode.probe(junk) is None


class TestFrameAtRealFile:
    def test_returns_rgb_frame(self, clip):
        path, _ = clip
        frame = decode.frame_at(path, 0.0)
        assert frame is not None
        assert frame.shape == (48, 64, 3)
        assert frame.dtype == np.uint8
        assert frame.flags.writeable

    def test_seeks_to_the_requested_time(self, clip):
        path, frames = clip
        # Frame fills step by 8 per frame, so 0.0s and 2.0s differ markedly.
        early = decode.frame_at(path, 0.0)
        late = decode.frame_at(path, 2.0)
        assert early is not None and late is not None
        assert float(early.mean()) == pytest.approx(float(frames[0].mean()), abs=12)
        assert float(late.mean()) == pytest.approx(float(frames[20].mean()), abs=12)

    def test_negative_time_clamps_to_the_start(self, clip):
        path, _ = clip
        frame = decode.frame_at(path, -5.0)
        first = decode.frame_at(path, 0.0)
        assert frame is not None and first is not None
        assert np.array_equal(frame, first)

    def test_past_the_end_returns_none(self, clip):
        path, _ = clip
        assert decode.frame_at(path, 60.0) is None

    def test_missing_file_returns_none(self, tmp_path):
        assert decode.frame_at(tmp_path / "nope.mp4", 1.0) is None


class TestFramesAtRealFile:
    def test_returns_one_frame_per_timestamp(self, clip):
        path, _ = clip
        frames = decode.frames_at(path, [0.0, 1.0, 2.0])
        assert len(frames) == 3
        assert all(f.shape == (48, 64, 3) for f in frames)

    def test_drops_undecodable_timestamps(self, clip):
        path, _ = clip
        frames = decode.frames_at(path, [0.0, 60.0, 1.0])
        assert len(frames) == 2  # the out-of-range seek is dropped

    def test_empty_request_returns_empty(self, clip):
        path, _ = clip
        assert decode.frames_at(path, []) == []


class TestIterFramesRealFile:
    def test_walks_the_video_at_the_requested_cadence(self, clip):
        path, _ = clip
        got = list(decode.iter_frames(path, interval=1.0))
        assert len(got) == 3  # a 3 s clip at one frame per second
        times = [t for t, _ in got]
        assert times == pytest.approx([0.0, 1.0, 2.0])
        assert all(frame.shape == (48, 64, 3) for _, frame in got)

    def test_downscales_to_max_width(self, clip):
        path, _ = clip
        got = list(decode.iter_frames(path, interval=1.0, max_width=32))
        assert got
        assert all(frame.shape == (24, 32, 3) for _, frame in got)

    def test_early_exit_does_not_leak_the_process(self, clip):
        path, _ = clip
        frames = decode.iter_frames(path, interval=0.1)
        next(frames)
        frames.close()  # kills ffmpeg mid-stream; must not raise

    def test_missing_file_yields_nothing(self, tmp_path):
        assert list(decode.iter_frames(tmp_path / "nope.mp4", interval=1.0)) == []

    def test_rejects_non_positive_interval(self, clip):
        path, _ = clip
        with pytest.raises(ValueError, match="interval must be positive"):
            list(decode.iter_frames(path, interval=0.0))


class TestBackendSelection:
    def test_reports_ffmpeg_when_available(self):
        assert decode.backend() == "ffmpeg"

    def test_falls_back_to_opencv_without_ffmpeg(self, monkeypatch):
        monkeypatch.setattr(decode, "get_ffmpeg_exe", _missing_ffmpeg)
        assert decode.backend() == "opencv"

    def test_probe_routes_to_opencv_without_ffmpeg(self, monkeypatch):
        monkeypatch.setattr(decode, "get_ffmpeg_exe", _missing_ffmpeg)
        calls: list[str] = []
        monkeypatch.setattr(decode, "_cv2_probe", lambda path: calls.append(str(path)))
        decode.probe("clip.mp4")
        assert calls == ["clip.mp4"]

    def test_frame_at_routes_to_opencv_without_ffmpeg(self, monkeypatch):
        monkeypatch.setattr(decode, "get_ffmpeg_exe", _missing_ffmpeg)
        calls: list[tuple[str, float]] = []
        monkeypatch.setattr(decode, "_cv2_frame_at", lambda path, t: calls.append((str(path), t)))
        decode.frame_at("clip.mp4", 2.0)
        assert calls == [("clip.mp4", 2.0)]

    def test_iter_frames_routes_to_opencv_without_ffmpeg(self, monkeypatch):
        monkeypatch.setattr(decode, "get_ffmpeg_exe", _missing_ffmpeg)
        monkeypatch.setattr(
            decode,
            "_cv2_iter_frames",
            lambda path, *, interval, max_width=None: iter([(0.0, np.zeros((1, 1, 3), dtype=np.uint8))]),
        )
        assert len(list(decode.iter_frames("clip.mp4", interval=1.0))) == 1


def _missing_ffmpeg() -> str:
    raise FileNotFoundError("ffmpeg not found")
