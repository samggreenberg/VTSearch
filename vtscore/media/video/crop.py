"""The per-unit ``clip_box``: which pixel region of a video frame is the content.

Video units are **metadata-only**: every clip of a parent video shares the
parent's bytes and carries a time window (``clip_start`` / ``clip_end``) saying
which slice of it this unit is.  A crop is the spatial half of the same idea -
``clip_box`` says which *pixel region* of each frame this unit is - and it is
honoured, not baked in, for exactly the same reason: re-encoding a cropped copy
of the file per unit would duplicate the payload and desync the time window
that still indexes the parent's timeline.

Three readers therefore have to agree about the box, so the parsing and the
crop live here once:

* :func:`~vtscore.media.video._frame_sampling.sample_video_frames` - what the
  embedder actually sees.
* :mod:`vtscore.media.video.media_type`'s thumbnailers - what the user sees in
  the grid, so preview and embedding frame the same picture.
* :class:`~vtscore.media.video.cleaner.VideoLetterboxCropCleaner` - the gate
  that writes the box in the first place, and
  :class:`~vtscore.media.video.cleaner.VideoBlankTrimCleaner`, which measures
  blankness inside it.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def parse_clip_box(raw: Any) -> tuple[int, int, int, int] | None:
    """Parse a ``clip_box`` into a 4-int ``(x0, y0, x1, y1)`` tuple, or ``None``.

    Accepts the native list/tuple carried on a media dict as well as the
    ``"x0,y0,x1,y1"`` string form that rides in ``origin.params``.  Returns
    ``None`` for anything malformed - a bad box must degrade to "no crop"
    rather than raise in the middle of an embed.
    """
    if isinstance(raw, (list, tuple)):
        parts: list[Any] = list(raw)
    elif isinstance(raw, str):
        parts = [p for p in raw.split(",") if p != ""]
    else:
        return None
    if len(parts) != 4:
        return None
    try:
        return (int(float(parts[0])), int(float(parts[1])), int(float(parts[2])), int(float(parts[3])))
    except (TypeError, ValueError):
        return None


def clamp_clip_box(raw: Any, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Return *raw* as a usable box within a *width* x *height* frame, or ``None``.

    ``None`` means "no crop applies": the box is missing or malformed, it is
    degenerate once clamped to the frame, or it already covers the whole frame.
    """
    box = parse_clip_box(raw)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    x0, y0 = max(0, min(x0, width)), max(0, min(y0, height))
    x1, y1 = max(0, min(x1, width)), max(0, min(y1, height))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    if (x0, y0, x1, y1) == (0, 0, width, height):
        return None
    return (x0, y0, x1, y1)


def crop_frame(frame_rgb: np.ndarray, raw: Any) -> np.ndarray:
    """Return *frame_rgb* cropped to *raw*, or the frame itself when no crop applies.

    Safe to call unconditionally: a missing, malformed, degenerate, or
    full-frame box leaves the frame untouched.
    """
    height, width = frame_rgb.shape[:2]
    box = clamp_clip_box(raw, width, height)
    if box is None:
        return frame_rgb
    x0, y0, x1, y1 = box
    return frame_rgb[y0:y1, x0:x1]
