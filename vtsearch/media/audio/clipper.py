"""Audio clippers — tile or pass-through audio media."""

from __future__ import annotations

import io
import math
import wave
from typing import Any

from vtsearch.media.base import MediaClipper


def _wav_duration(wav_bytes: bytes) -> float:
    """Return the duration in seconds of a WAV byte string."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def _wav_slice(wav_bytes: bytes, start: float, end: float) -> bytes:
    """Extract a [start, end) slice from a WAV byte string.

    Returns a new WAV byte string containing only the requested segment.
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        start_frame = int(start * sr)
        end_frame = int(end * sr)
        total_frames = wf.getnframes()
        start_frame = max(0, min(start_frame, total_frames))
        end_frame = max(start_frame, min(end_frame, total_frames))
        wf.setpos(start_frame)
        frames = wf.readframes(end_frame - start_frame)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(n_channels)
        out.setsampwidth(sampwidth)
        out.setframerate(sr)
        out.writeframes(frames)
    return buf.getvalue()


class SoundDefaultClipper(MediaClipper):
    """Returns the audio media unchanged."""

    @property
    def name(self) -> str:
        return "sound_default"

    @property
    def media_type(self) -> str:
        return "audio"

    def clip(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        return [media]


class SoundTilingClipper(MediaClipper):
    """Tile audio into equally-spaced segments of a given duration.

    ``SoundTilingClipper(2)`` tiles a 9.5 s clip into five 2 s segments
    whose start times are equally spaced so the first starts at 0 and the
    last ends at 9.5 s (with a little overlap between neighbours when the
    total duration is not an exact multiple of the segment size).

    If *min_overlap* is set, the clipper ensures that consecutive segments
    overlap by at least that many seconds (producing more tiles when needed).

    If the audio is shorter than or equal to *duration*, a single segment
    covering the full audio is returned.
    """

    def __init__(self, duration: float, min_overlap: float = 0.0) -> None:
        if duration <= 0:
            raise ValueError("duration must be positive")
        if min_overlap < 0:
            raise ValueError("min_overlap must be non-negative")
        if min_overlap >= duration:
            raise ValueError("min_overlap must be less than duration")
        self._duration = duration
        self._min_overlap = min_overlap

    @property
    def name(self) -> str:
        return "sound_tiling"

    @property
    def media_type(self) -> str:
        return "audio"

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def min_overlap(self) -> float:
        return self._min_overlap

    def clip(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        wav_bytes = media.get("media_bytes")
        if wav_bytes is None:
            return [media]

        total = _wav_duration(wav_bytes)
        seg = self._duration

        if total <= seg:
            return [media]

        max_stride = seg - self._min_overlap
        n_tiles = max(1, math.ceil((total - seg) / max_stride) + 1)
        # Space n_tiles segments so that the first starts at 0 and the last
        # ends at *total*.  When n_tiles == 1 this degenerates to [0, seg).
        if n_tiles == 1:
            starts = [0.0]
        else:
            starts = [i * (total - seg) / (n_tiles - 1) for i in range(n_tiles)]

        results: list[dict[str, Any]] = []
        for idx, t0 in enumerate(starts):
            t1 = t0 + seg
            sliced = _wav_slice(wav_bytes, t0, t1)
            tile = dict(media)
            tile["media_bytes"] = sliced
            tile["duration"] = round(t1 - t0, 6)
            tile["file_size"] = len(sliced)
            tile["clip_index"] = idx
            tile["clip_start"] = round(t0, 6)
            tile["clip_end"] = round(t1, 6)
            results.append(tile)
        return results

    @property
    def parameters(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "duration",
                "label": "Clip length (seconds)",
                "type": "number",
                "default": self._duration,
                "min": 0.1,
                "max": 300,
                "step": 0.1,
            },
            {
                "key": "min_overlap",
                "label": "Minimum overlap (seconds)",
                "type": "number",
                "default": self._min_overlap,
                "min": 0,
                "max": 299.9,
                "step": 0.1,
            },
        ]

    def with_params(self, params: dict[str, Any]) -> "SoundTilingClipper":
        duration = float(params.get("duration", self._duration))
        min_overlap = float(params.get("min_overlap", self._min_overlap))
        return SoundTilingClipper(duration, min_overlap)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["duration"] = self._duration
        d["min_overlap"] = self._min_overlap
        return d
