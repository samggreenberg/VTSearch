"""Image cleaners - 1→1 cleanup gates run on each image before embedding."""

from __future__ import annotations

import io
import logging
from typing import Any

from vtscore.media.cleaner import MediaCleaner

log = logging.getLogger(__name__)

#: Camera EXIF tag holding the display orientation (1 == already upright).
_ORIENTATION_TAG = 274


class ImageExifOrientCleaner(MediaCleaner):
    """Bake a photo's EXIF display orientation into its pixels.

    Phone and camera JPEGs are stored in sensor order with an EXIF
    ``Orientation`` tag telling the viewer how to rotate them.  Browsers and
    :func:`~vtscore.media.image.thumbnail.make_image_thumbnail` honour that tag,
    but the **embed path does not**: it hands the stored bytes straight to the
    embedder, so a portrait phone photo is embedded sideways while its thumbnail
    shows upright.  Every rotated photo therefore lands in the wrong part of the
    vector space, which no amount of training fixes.

    This gate rewrites the payload as the upright image with the tag cleared, so
    the bytes the embedder sees match the bytes the user sees.  Because it
    corrects an outright representation bug rather than making a judgment call
    about what counts as wasted content, it is the one cleaner that defaults
    **on**.
    """

    @property
    def name(self) -> str:
        return "image_exif_orient"

    @property
    def media_type(self) -> str:
        return "image"

    @property
    def display_name(self) -> str:
        return "EXIF Orientation"

    @property
    def description(self) -> str:
        return (
            "Rotate photos to their EXIF display orientation so the embedder sees them "
            "upright, the way thumbnails already show them."
        )

    @property
    def default_enabled(self) -> bool:
        return True

    def clean(self, media: dict[str, Any]) -> dict[str, Any]:
        """Return *media* with the rotation baked in, or unchanged.

        Unchanged when there are no bytes to work with, the payload can't be
        decoded (SVG, a corrupt file), the image carries no orientation tag, or
        the tag already says upright - the common case, so most images cost one
        header parse and nothing else.
        """
        media_bytes = media.get("media_bytes")
        if not isinstance(media_bytes, (bytes, bytearray)) or not media_bytes:
            return media

        from PIL import Image, ImageOps  # noqa: PLC0415

        try:
            with Image.open(io.BytesIO(media_bytes)) as src:
                exif = src.getexif()
                orientation = exif.get(_ORIENTATION_TAG) if exif else None
                if orientation in (None, 1):
                    return media
                upright = ImageOps.exif_transpose(src)
                if upright is None:
                    return media
                fmt = src.format or "PNG"
                # ``exif_transpose`` already strips the orientation tag from the
                # transposed copy, so re-encoding cannot re-apply the rotation.
                out = io.BytesIO()
                upright.save(out, format=fmt)
                width, height = upright.size
        except Exception:
            log.debug("image_exif_orient: undecodable payload, leaving unchanged", exc_info=True)
            return media

        rotated_bytes = out.getvalue()
        cleaned = dict(media)
        cleaned["media_bytes"] = rotated_bytes
        cleaned["file_size"] = len(rotated_bytes)
        cleaned["width"] = width
        cleaned["height"] = height
        return cleaned
