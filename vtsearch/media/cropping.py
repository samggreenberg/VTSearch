"""Apply a user-specified bounded clipper to an arbitrary media file.

Used by the example-sort and server-media-upload routes so the user can
supply a sub-region of a media item (e.g. a time range of an audio clip
or a bounding box of an image) without having to crop the file client
side.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any


def crop_file_bytes(
    file_path: Path,
    media_type: str,
    params: dict[str, Any],
) -> bytes:
    """Apply a single-clip bounded clipper to *file_path* and return the cropped bytes.

    *params* must be a dict matching the clipper's bounds:

    * Audio: ``{"start": float, "end": float}`` — seconds.
    * Image: ``{"box": [x1, y1, x2, y2]}`` — pixel coords in the original image.

    Raises :class:`ValueError` for unsupported media types or invalid params.
    Raises :class:`FileNotFoundError` if *file_path* does not exist.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    media_bytes = file_path.read_bytes()

    if media_type == "audio":
        from vtsearch.media.audio.clipper import SoundClipClipper

        start = float(params.get("start", 0.0))
        end = float(params.get("end", 0.0))
        clipper = SoundClipClipper(start, end)
        media = {"id": 0, "type": "audio", "media_bytes": media_bytes}
        clipped = clipper.clip(media)
        return clipped[0].get("media_bytes", media_bytes)

    if media_type == "image":
        from PIL import Image  # noqa: PLC0415

        from vtsearch.media.image.clipper import ImageBboxClipper

        box = params.get("box")
        if box is None or len(box) != 4:
            raise ValueError("box must be a 4-tuple [x1, y1, x2, y2]")
        clipper = ImageBboxClipper(box)
        with Image.open(io.BytesIO(media_bytes)) as img:
            width, height = img.size
        media = {
            "id": 0,
            "type": "image",
            "media_bytes": media_bytes,
            "width": width,
            "height": height,
        }
        clipped = clipper.clip(media)
        return clipped[0].get("media_bytes", media_bytes)

    raise ValueError(f"Cropping is not supported for media type: {media_type!r}")
