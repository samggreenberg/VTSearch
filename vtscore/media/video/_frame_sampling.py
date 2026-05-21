"""Shared video frame sampling for the X-CLIP / LanguageBind / VideoMAE embedders.

Reads ``clip_start`` / ``clip_end`` off the media dict so tiled clips of the
same parent video sample different frame ranges.  Without this, every tile
would call ``np.linspace(0, frame_count - 1, num_frames)`` against the full
parent video and produce identical embeddings.

Also accepts ``media_bytes`` (round-tripped through a tempfile) when the
caller cannot supply a local ``media_path`` — matches the audio embedder's
two-source convention so re-embedding works from in-memory dataset bytes.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional


@contextmanager
def _resolve_video_path(media: dict) -> Iterator[Optional[Path]]:
    """Yield a usable filesystem path for *media*'s video, tempfile if needed."""
    path_str = media.get("media_path")
    if path_str:
        yield Path(path_str)
        return

    video_bytes = media.get("media_bytes")
    if not isinstance(video_bytes, (bytes, bytearray)) or not video_bytes:
        yield None
        return

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        tmp.write(bytes(video_bytes))
        tmp.close()
        yield Path(tmp.name)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _frame_index_range(media: dict, frame_count: int, fps: float) -> tuple[int, int]:
    """Map ``clip_start``/``clip_end`` to ``[start_frame, end_frame]`` indices.

    Falls back to the full video range when boundaries are absent or ``fps``
    is non-positive.  ``end_frame`` is inclusive.
    """
    clip_start = media.get("clip_start")
    clip_end = media.get("clip_end")
    if clip_start is None or clip_end is None or not fps or fps <= 0 or float(clip_end) <= float(clip_start):
        return 0, frame_count - 1

    start_idx = int(round(float(clip_start) * fps))
    end_idx = int(round(float(clip_end) * fps)) - 1
    start_idx = max(0, min(start_idx, frame_count - 1))
    end_idx = max(start_idx, min(end_idx, frame_count - 1))
    return start_idx, end_idx


def sample_video_frames(media: dict, num_frames: int) -> list[Any]:
    """Return up to *num_frames* PIL Images sampled from *media*'s video.

    When ``media`` carries ``clip_start`` and ``clip_end``, sampling is
    restricted to that interval — distinct tiles of the same parent video
    produce distinct frame sets (and therefore distinct embeddings).
    Otherwise the whole video is sampled.

    The returned list is padded by repeating the last frame to exactly
    *num_frames* entries, matching the per-embedder padding loops it
    replaces.  Returns ``[]`` on any decoding failure.
    """
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    with _resolve_video_path(media) as path:
        if path is None:
            return []
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return []
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                return []
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            start_idx, end_idx = _frame_index_range(media, frame_count, fps)
            n_avail = end_idx - start_idx + 1
            n_to_sample = min(num_frames, max(1, n_avail))
            if n_to_sample == 1:
                indices = np.array([start_idx], dtype=int)
            else:
                indices = np.linspace(start_idx, end_idx, n_to_sample, dtype=int)

            frames: list[Any] = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))
        finally:
            cap.release()

    if not frames:
        return []
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames
