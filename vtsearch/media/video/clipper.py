"""Video clippers — tile or pass-through video media."""

from __future__ import annotations

import math
from typing import Any

from vtsearch.media.base import MediaClipper


class VideoDefaultClipper(MediaClipper):
    """Returns the video media unchanged."""

    @property
    def name(self) -> str:
        return "video_default"

    @property
    def media_type(self) -> str:
        return "video"

    def clip(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        return [media]


class VideoTilingClipper(MediaClipper):
    """Tile a video into equally-spaced segments of a given duration.

    ``VideoTilingClipper(2)`` tiles a 9.5 s video into five 2 s segments
    whose start times are equally spaced so the first starts at 0 and the
    last ends at 9.5 s (with a little overlap between neighbours when the
    total duration is not an exact multiple of the segment size).

    If the video is shorter than or equal to *duration*, a single segment
    covering the full video is returned.

    Because video transcoding requires heavy dependencies (ffmpeg / moviepy),
    this clipper records the clip boundaries as metadata on the media dict
    (``clip_start``, ``clip_end``, ``clip_index``) without slicing the
    underlying bytes.  Downstream consumers can use these fields to seek or
    trim on playback.
    """

    def __init__(self, duration: float) -> None:
        if duration <= 0:
            raise ValueError("duration must be positive")
        self._duration = duration

    @property
    def name(self) -> str:
        return f"video_tiling_{self._duration}s"

    @property
    def media_type(self) -> str:
        return "video"

    @property
    def duration(self) -> float:
        return self._duration

    def clip(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        total = media.get("duration", 0)
        seg = self._duration

        if total <= seg:
            return [media]

        n_tiles = max(1, math.ceil(total / seg))
        if n_tiles == 1:
            starts = [0.0]
        else:
            starts = [i * (total - seg) / (n_tiles - 1) for i in range(n_tiles)]

        results: list[dict[str, Any]] = []
        for idx, t0 in enumerate(starts):
            t1 = t0 + seg
            tile = dict(media)
            tile["duration"] = round(t1 - t0, 6)
            tile["clip_index"] = idx
            tile["clip_start"] = round(t0, 6)
            tile["clip_end"] = round(t1, 6)
            results.append(tile)
        return results

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["duration"] = self._duration
        return d
