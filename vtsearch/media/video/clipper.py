"""Video clippers — tile, scene-detect, or pass-through video media."""

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

    @property
    def description(self) -> str:
        return "Import each video as-is, without splitting."

    def clip(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        return [media]


class VideoTilingClipper(MediaClipper):
    """Tile a video into equally-spaced segments of a given duration.

    ``VideoTilingClipper(2)`` tiles a 9.5 s video into five 2 s segments
    whose start times are equally spaced so the first starts at 0 and the
    last ends at 9.5 s (with a little overlap between neighbours when the
    total duration is not an exact multiple of the segment size).

    If *min_overlap* is set, the clipper ensures that consecutive segments
    overlap by at least that many seconds (producing more tiles when needed).

    If the video is shorter than or equal to *duration*, a single segment
    covering the full video is returned.

    Because video transcoding requires heavy dependencies (ffmpeg / moviepy),
    this clipper records the clip boundaries as metadata on the media dict
    (``clip_start``, ``clip_end``, ``clip_index``) without slicing the
    underlying bytes.  Downstream consumers can use these fields to seek or
    trim on playback.
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
        return "video_tiling"

    @property
    def media_type(self) -> str:
        return "video"

    @property
    def description(self) -> str:
        return "Split each video into fixed-length overlapping segments."

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def min_overlap(self) -> float:
        return self._min_overlap

    def clip(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        total = media.get("duration", 0)
        seg = self._duration

        if total <= seg:
            return [media]

        max_stride = seg - self._min_overlap
        n_tiles = max(1, math.ceil((total - seg) / max_stride) + 1)
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

    @property
    def parameters(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "duration",
                "label": "Clip length (seconds)",
                "description": "Duration of each video segment in seconds.",
                "type": "number",
                "default": self._duration,
                "min": 0.1,
                "max": 300,
                "step": 0.1,
            },
            {
                "key": "min_overlap",
                "label": "Minimum overlap (seconds)",
                "description": "Minimum overlap between consecutive segments. Higher values produce more tiles.",
                "type": "number",
                "default": self._min_overlap,
                "min": 0,
                "max": 299.9,
                "step": 0.1,
            },
        ]

    def with_params(self, params: dict[str, Any]) -> "VideoTilingClipper":
        duration = float(params.get("duration", self._duration))
        min_overlap = float(params.get("min_overlap", self._min_overlap))
        return VideoTilingClipper(duration, min_overlap)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["duration"] = self._duration
        d["min_overlap"] = self._min_overlap
        return d


def _detect_scene_boundaries(
    video_path: str,
    threshold: float,
    min_scene_duration: float,
) -> list[float]:
    """Return a list of scene-boundary timestamps (in seconds) for *video_path*.

    Scene changes are detected by comparing consecutive frames using
    normalised histogram correlation.  When the correlation drops below
    *threshold* (i.e. the frames look very different), a scene boundary is
    recorded — provided at least *min_scene_duration* seconds have elapsed
    since the previous boundary.

    The function samples one frame per second to keep the cost manageable
    for long videos.

    Returns a sorted list of boundary timestamps **not** including 0 or
    the video end.  An empty list means no scene changes were detected.
    """
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            return []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            return []
        total_duration = frame_count / fps

        # Sample roughly one frame per second.
        sample_interval = max(1, int(round(fps)))

        prev_hist = None
        boundaries: list[float] = []
        last_boundary_time = 0.0

        frame_idx = 0
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to HSV and compute a normalised hue-saturation histogram.
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist)

            if prev_hist is not None:
                corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                current_time = frame_idx / fps
                if corr < threshold and (current_time - last_boundary_time) >= min_scene_duration:
                    boundaries.append(round(current_time, 6))
                    last_boundary_time = current_time

            prev_hist = hist
            frame_idx += sample_interval
            if frame_idx >= frame_count:
                break

        # Drop any boundary that would create a trailing scene shorter than
        # min_scene_duration.
        if boundaries and (total_duration - boundaries[-1]) < min_scene_duration:
            boundaries.pop()

        return boundaries
    finally:
        cap.release()


class VideoSceneClipper(MediaClipper):
    """Split a video into clips at detected scene boundaries.

    Scene changes are found by comparing colour histograms of sampled
    frames (one per second).  When consecutive frames differ significantly
    (histogram correlation below *threshold*), a scene boundary is placed.

    Like :class:`VideoTilingClipper`, this clipper records boundaries as
    metadata (``clip_start``, ``clip_end``, ``clip_index``, ``scene_index``)
    without transcoding the underlying bytes — downstream consumers use
    these fields to seek or trim on playback.

    Parameters
    ----------
    threshold : float
        Histogram correlation threshold.  Lower values require a more
        dramatic visual change to trigger a scene break.  Defaults to 0.3.
    min_scene_duration : float
        Minimum duration (seconds) for a scene.  Boundaries that would
        produce shorter scenes are suppressed.  Defaults to 1.0.

    If OpenCV is not installed, or if no scene boundaries are found, the
    media is returned unchanged (single-element list).
    """

    def __init__(self, threshold: float = 0.3, min_scene_duration: float = 1.0) -> None:
        if threshold < 0 or threshold > 1:
            raise ValueError("threshold must be between 0 and 1")
        if min_scene_duration <= 0:
            raise ValueError("min_scene_duration must be positive")
        self._threshold = threshold
        self._min_scene_duration = min_scene_duration

    @property
    def name(self) -> str:
        return "video_scene"

    @property
    def media_type(self) -> str:
        return "video"

    @property
    def description(self) -> str:
        return "Automatically split each video at detected scene changes."

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def min_scene_duration(self) -> float:
        return self._min_scene_duration

    def clip(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        import os  # noqa: PLC0415
        import tempfile  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        total = media.get("duration", 0)
        if total <= 0:
            return [media]

        # Resolve a file path that OpenCV can read.
        media_path = media.get("media_path")
        media_bytes = media.get("media_bytes")
        tmp_file = None

        if media_path and Path(media_path).exists():
            video_path = str(media_path)
        elif media_bytes:
            filename = media.get("filename", "video.mp4")
            ext = Path(filename).suffix or ".mp4"
            tmp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            tmp_file.write(media_bytes)
            tmp_file.close()
            video_path = tmp_file.name
        else:
            return [media]

        try:
            try:
                boundaries = _detect_scene_boundaries(
                    video_path,
                    self._threshold,
                    self._min_scene_duration,
                )
            except ImportError:
                return [media]

            if not boundaries:
                return [media]

            # Build scene intervals: [0, b1), [b1, b2), ..., [bN, total)
            starts = [0.0] + boundaries
            ends = boundaries + [total]

            results: list[dict[str, Any]] = []
            for idx, (t0, t1) in enumerate(zip(starts, ends)):
                scene = dict(media)
                scene["duration"] = round(t1 - t0, 6)
                scene["clip_index"] = idx
                scene["scene_index"] = idx
                scene["clip_start"] = round(t0, 6)
                scene["clip_end"] = round(t1, 6)
                results.append(scene)
            return results
        finally:
            if tmp_file is not None:
                os.unlink(tmp_file.name)

    @property
    def parameters(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "threshold",
                "label": "Sensitivity (0\u20131)",
                "description": "Histogram correlation threshold. Lower values require more dramatic visual change to trigger a scene break.",
                "type": "number",
                "default": self._threshold,
                "min": 0,
                "max": 1,
                "step": 0.05,
            },
            {
                "key": "min_scene_duration",
                "label": "Min scene length (seconds)",
                "description": "Minimum duration for a scene. Boundaries that would create shorter scenes are suppressed.",
                "type": "number",
                "default": self._min_scene_duration,
                "min": 0.1,
                "max": 60,
                "step": 0.1,
            },
        ]

    def with_params(self, params: dict[str, Any]) -> "VideoSceneClipper":
        threshold = float(params.get("threshold", self._threshold))
        min_scene_duration = float(params.get("min_scene_duration", self._min_scene_duration))
        return VideoSceneClipper(threshold=threshold, min_scene_duration=min_scene_duration)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["threshold"] = self._threshold
        d["min_scene_duration"] = self._min_scene_duration
        return d
