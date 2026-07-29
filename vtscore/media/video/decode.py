"""Decode video frames through ffmpeg instead of loading OpenCV in-process.

Every video path used to call ``cv2.VideoCapture``.  That is a problem beyond
OpenCV itself: ``cv2.abi3.so`` lists ``libavformat`` in its ``DT_NEEDED``
entries, and the wheel's ``libavformat`` in turn needs the ``libssl`` /
``libcrypto`` 1.1.1 pair vendored inside ``opencv_python_headless.libs``.  So
merely importing ``cv2`` maps a second, non-FIPS OpenSSL into a process that
(on a FIPS-enabled host) already holds the system's FIPS-validated one.  The
duplicate trips the FIPS self-test and OpenSSL answers with ``abort()``:

    crypto/fips/fips.c:154 OpenSSL internal error: FATAL FIPS SELFTEST FAILURE

That kills the interpreter outright - no exception, no traceback, just a core
dump - so it cannot be caught and handled at the call site.  Downgrading does
not help either: every ``opencv-python-headless`` wheel from 4.9 through 5.0
vendors the same OpenSSL 1.1.1w.

The fix is to keep OpenSSL out of *our* address space.  ffmpeg runs as a
separate process, so whatever it links is its own business, and it is already
a hard dependency (``imageio-ffmpeg`` ships a static binary; a system ffmpeg on
``$PATH`` wins when present) used for audio decoding and video transcoding.

``cv2`` remains as a fallback for installs with no ffmpeg at all.  It is
deliberately *not* used when ffmpeg is present but fails on a particular file:
a decode failure is not worth risking the process on.
"""

from __future__ import annotations

import io
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from vtscore.media.audio.ffmpeg import get_ffmpeg_exe

#: Seconds to wait for a single-frame extract or a container probe.  Generous
#: for a slow seek into a large file, bounded so a wedged subprocess can't hang
#: a worker forever.
FFMPEG_TIMEOUT = 120.0

#: Header line ffmpeg prints for the container: ``Duration: 00:01:03.48, ...``.
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2}(?:\.\d+)?)")

#: The first video stream's summary line, e.g.
#: ``Stream #0:0(und): Video: h264 (High) (avc1 / 0x31637661), yuv420p, 640x480 ...``
_VIDEO_STREAM_RE = re.compile(r"^\s*Stream #\d+:\d+.*: Video: .*$", re.MULTILINE)

#: ``640x480`` inside a stream line.  Two digits minimum so the ``0x...`` fourcc
#: tag that precedes it can't match.
_SIZE_RE = re.compile(r"(?<![\dx])(\d{2,5})x(\d{2,5})(?![\dx])")

#: ``30 fps`` / ``29.97 fps`` (``tbr``/``tbn`` deliberately not matched).
_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s+fps")


@dataclass(frozen=True)
class VideoInfo:
    """Container metadata for a video file.  Zeroed fields mean "unknown"."""

    duration: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0

    @property
    def frame_count(self) -> int:
        """Frames in the video, derived from duration x fps (0 when unknown)."""
        if self.duration <= 0 or self.fps <= 0:
            return 0
        return max(1, int(round(self.duration * self.fps)))

    @property
    def frame_seconds(self) -> float:
        """How long one frame lasts, falling back to 25fps when unknown.

        Callers use it as the headroom that keeps a seek to the end of a video
        from landing past its last frame, where nothing decodes.
        """
        return 1.0 / self.fps if self.fps > 0 else 0.04


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def _ffmpeg_exe() -> str | None:
    """Return the ffmpeg binary, or ``None`` when this install has none."""
    try:
        return get_ffmpeg_exe()
    except FileNotFoundError:
        return None


def backend() -> str:
    """Return the decoder in use: ``"ffmpeg"`` or ``"opencv"``.

    Exposed so deployments can confirm which one a host resolved to - on a
    FIPS-enabled host, ``"opencv"`` means video decoding will crash the
    process, and ffmpeg needs installing.
    """
    return "ffmpeg" if _ffmpeg_exe() is not None else "opencv"


def _run(cmd: list[str], *, timeout: float = FFMPEG_TIMEOUT) -> subprocess.CompletedProcess[bytes] | None:
    """Run *cmd* to completion, returning ``None`` if it could not be run."""
    try:
        # Always hand ffmpeg an empty stdin so the child never inherits - and
        # blocks on - the parent's terminal.
        return subprocess.run(cmd, input=b"", capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------


def _parse_probe(stderr: str) -> VideoInfo | None:
    """Parse ``ffmpeg -i`` banner output into a :class:`VideoInfo`.

    Returns ``None`` when the input has no video stream (unreadable file,
    audio-only container, ...).
    """
    stream = _VIDEO_STREAM_RE.search(stderr)
    if stream is None:
        return None
    line = stream.group(0)

    width = height = 0
    size = _SIZE_RE.search(line)
    if size is not None:
        width, height = int(size.group(1)), int(size.group(2))

    fps = 0.0
    fps_match = _FPS_RE.search(line)
    if fps_match is not None:
        fps = float(fps_match.group(1))

    duration = 0.0
    dur = _DURATION_RE.search(stderr)
    if dur is not None:
        duration = int(dur.group(1)) * 3600 + int(dur.group(2)) * 60 + float(dur.group(3))

    return VideoInfo(duration=duration, fps=fps, width=width, height=height)


def probe(path: str | Path) -> VideoInfo | None:
    """Return metadata for the video at *path*, or ``None`` if it can't be read."""
    exe = _ffmpeg_exe()
    if exe is None:
        return _cv2_probe(path)

    # ffmpeg exits non-zero on "no output file specified" after printing the
    # stream summary we're after, so the return code is deliberately ignored.
    result = _run([exe, "-hide_banner", "-nostdin", "-i", str(path)])
    if result is None:
        return None
    return _parse_probe(result.stderr.decode(errors="replace"))


# ---------------------------------------------------------------------------
# Single-frame extraction
# ---------------------------------------------------------------------------


def _decode_png(blob: bytes) -> np.ndarray | None:
    """Decode a PNG blob to an ``(h, w, 3)`` uint8 RGB array."""
    from PIL import Image  # noqa: PLC0415

    try:
        with Image.open(io.BytesIO(blob)) as img:
            return np.array(img.convert("RGB"), dtype=np.uint8)
    except Exception:
        return None


def frame_at(path: str | Path, time_seconds: float) -> np.ndarray | None:
    """Return the frame at *time_seconds* as an ``(h, w, 3)`` uint8 RGB array.

    Returns ``None`` when the video can't be decoded or the timestamp lies
    past its last frame.
    """
    exe = _ffmpeg_exe()
    if exe is None:
        return _cv2_frame_at(path, time_seconds)

    cmd = [
        exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        # Input-side seek: jumps to the nearest keyframe before decoding
        # forward to the exact timestamp, so cost stays flat on long videos.
        "-ss",
        f"{max(0.0, float(time_seconds)):.6f}",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
    ]
    result = _run(cmd)
    if result is None or result.returncode != 0 or not result.stdout:
        return None
    return _decode_png(result.stdout)


def frames_at(path: str | Path, times: Sequence[float]) -> list[np.ndarray]:
    """Return the frames at *times*, skipping any that fail to decode.

    The result can therefore be shorter than *times*; callers that care about
    partial failure compare the two lengths.
    """
    frames: list[np.ndarray] = []
    for t in times:
        frame = frame_at(path, t)
        if frame is not None:
            frames.append(frame)
    return frames


# ---------------------------------------------------------------------------
# Streaming extraction (one process for the whole video)
# ---------------------------------------------------------------------------


def _scaled_size(width: int, height: int, max_width: int | None) -> tuple[int, int]:
    """Return the output size after clamping *width* to *max_width*."""
    if not max_width or width <= max_width:
        return width, height
    return max_width, max(1, int(round(height * max_width / width)))


def iter_frames(
    path: str | Path,
    *,
    interval: float,
    max_width: int | None = None,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield ``(timestamp, frame)`` every *interval* seconds through the video.

    One ffmpeg process decodes the file linearly and pipes raw RGB, which is
    far cheaper than seeking per sample.  *max_width* downscales wide frames
    (aspect preserved) for callers that only need coarse image statistics.
    """
    if interval <= 0:
        raise ValueError("interval must be positive")

    exe = _ffmpeg_exe()
    if exe is None:
        yield from _cv2_iter_frames(path, interval=interval, max_width=max_width)
        return

    info = probe(path)
    if info is None or info.width <= 0 or info.height <= 0:
        return
    out_w, out_h = _scaled_size(info.width, info.height, max_width)

    # ``-noautorotate`` keeps the raw stream dimensions we just probed: with
    # autorotation on, a 90-degree display matrix would swap them and every
    # reshape below would be off.
    filters = f"fps={1.0 / interval:.6f}"
    if (out_w, out_h) != (info.width, info.height):
        filters += f",scale={out_w}:{out_h}"
    cmd = [
        exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-noautorotate",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-vf",
        filters,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]

    frame_bytes = out_w * out_h * 3
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return

    try:
        assert proc.stdout is not None
        index = 0
        while True:
            buf = proc.stdout.read(frame_bytes)
            if buf is None or len(buf) < frame_bytes:
                break
            yield index * interval, np.frombuffer(bytearray(buf), dtype=np.uint8).reshape(out_h, out_w, 3)
            index += 1
    finally:
        # Covers both the normal end-of-stream and an early ``break`` in the
        # consumer, which would otherwise leave ffmpeg blocked on a full pipe.
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.poll() is None:
            proc.kill()
        proc.wait()


# ---------------------------------------------------------------------------
# OpenCV fallback - only reached when the host has no ffmpeg at all
# ---------------------------------------------------------------------------


def _cv2_probe(path: str | Path) -> VideoInfo | None:
    try:
        import cv2  # noqa: PLC0415

        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                return None
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()
    except Exception:
        return None
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    return VideoInfo(duration=duration, fps=fps, width=max(0, width), height=max(0, height))


def _cv2_frame_at(path: str | Path, time_seconds: float) -> np.ndarray | None:
    try:
        import cv2  # noqa: PLC0415

        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                return None
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                return None
            target = int(round(time_seconds * fps)) if fps > 0 else frame_count // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(target, frame_count - 1)))
            ok, frame = cap.read()
            if not ok:
                return None
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()
    except Exception:
        return None


def _cv2_iter_frames(
    path: str | Path,
    *,
    interval: float,
    max_width: int | None = None,
) -> Iterator[tuple[float, np.ndarray]]:
    try:
        import cv2  # noqa: PLC0415

        cap = cv2.VideoCapture(str(path))
    except Exception:
        return
    try:
        if not cap.isOpened():
            return
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            return
        step = max(1, int(round(fps * interval)))
        for frame_idx in range(0, frame_count, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                return
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out_w, out_h = _scaled_size(rgb.shape[1], rgb.shape[0], max_width)
            if (out_w, out_h) != (rgb.shape[1], rgb.shape[0]):
                rgb = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
            yield frame_idx / fps, rgb
    finally:
        cap.release()
