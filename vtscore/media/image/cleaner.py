"""Image cleaners - 1→1 cleanup gates run on each image before embedding."""

from __future__ import annotations

import io
import logging
from typing import Any

from vtscore.media.cleaner import MediaCleaner
from vtscore.media.image.edge_trim import (
    DEFAULT_EDGE_TOL,
    DEFAULT_MAX_EDGE_TRIM,
    DEFAULT_MIN_EDGE_TRIM,
    solid_edge_box,
)

log = logging.getLogger(__name__)

#: Camera EXIF tag holding the display orientation (1 == already upright).
_ORIENTATION_TAG = 274


class ImageExifOrientCleaner(MediaCleaner):
    """Bake a photo's EXIF display orientation into its pixels.

    Phone and camera JPEGs are stored in sensor order with an EXIF
    ``Orientation`` tag telling the viewer how to rotate them.  This gate
    rewrites the payload as the upright image with the tag cleared, so the
    *stored bytes* are upright too.

    It is **off by default**, and does not affect what VTSearch sees.  Every
    decode in :mod:`vtscore.media.image.decode` applies the orientation tag
    already, so the embedder, thumbnailer, OCR, face detection and the crop
    paths all work on upright pixels whether or not this cleaner ran.  What it
    buys is bytes that are upright *outside* VTSearch — for a consumer of an
    exported dataset that reads the payload without honouring EXIF.  What it
    costs is a re-encode of every rotated photo (lossy, for JPEG), which is why
    the default is off: paying it to fix a bug that no longer exists would be a
    poor trade.
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
            "Rewrite photos upright in their stored bytes. VTSearch already reads them "
            "upright either way; this only matters for tools that ignore EXIF."
        )

    @property
    def default_enabled(self) -> bool:
        return False

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


#: Formats whose Pillow encoder accepts an ``exif=`` keyword.  A cropped copy
#: keeps the source's EXIF block for these so the camera/GPS metadata survives
#: the trim.  The block is taken from the *upright* image, where Pillow has
#: already dropped the orientation tag, so a rotated photo is not re-rotated on
#: top of pixels that are already upright.
_EXIF_WRITABLE_FORMATS = frozenset({"JPEG", "PNG", "TIFF", "WEBP", "MPO"})


class ImageEdgeTrimCleaner(MediaCleaner):
    """Crop away near-solid white or black border margins.

    Letterbox bars, a pillarbox on one side, and the whitespace around a centred
    logo are all content-free: the embedder spends real capacity encoding "this
    is a black rectangle" and the padding dilutes whatever signal the subject
    carries.  This gate finds the tight bounding box of non-padding content and
    crops to it, each of the four edges independently, so a frame padded only on
    top loses just that margin.

    The same detector backs the grid thumbnail (see
    :mod:`vtscore.media.image.edge_trim`), so a trimmed item's preview and its
    embedding finally agree about where the picture starts.

    Conservative by construction: no side is ever pulled in by more than
    *max_edge_trim*, a frame whose margins are all thinner than *min_edge_trim*
    is left alone, and an image that is nothing *but* solid tone is left alone
    too (there is no content to pull the box toward).
    """

    def __init__(
        self,
        edge_tol: float = DEFAULT_EDGE_TOL,
        max_edge_trim: float = DEFAULT_MAX_EDGE_TRIM,
        min_edge_trim: float = DEFAULT_MIN_EDGE_TRIM,
    ) -> None:
        self._edge_tol = edge_tol
        self._max_edge_trim = max_edge_trim
        self._min_edge_trim = min_edge_trim

    @property
    def name(self) -> str:
        return "image_edge_trim"

    @property
    def media_type(self) -> str:
        return "image"

    @property
    def description(self) -> str:
        return (
            "Crop near-solid white or black margins - letterbox bars, pillarboxes, whitespace "
            "around a logo - so the embedder sees the picture, not its padding."
        )

    @property
    def summary_template(self) -> str:
        return "Crop near-solid white/black margins, never more than {max_edge_trim} of any one side."

    @property
    def edge_tol(self) -> float:
        return self._edge_tol

    @property
    def max_edge_trim(self) -> float:
        return self._max_edge_trim

    @property
    def min_edge_trim(self) -> float:
        return self._min_edge_trim

    @property
    def parameters(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "edge_tol",
                "label": "Solid tolerance (0-255)",
                "description": (
                    "How far from pure white or pure black a pixel may sit and still count as padding. "
                    "Higher values treat more of the border as blank."
                ),
                "type": "number",
                "default": self._edge_tol,
                "min": 0,
                "max": 64,
                "step": 1,
            },
            {
                "key": "max_edge_trim",
                "label": "Maximum trim per side",
                "description": (
                    "Never remove more than this fraction of the width or height from any single side, "
                    "so a small subject in a large blank field can't blow up to fill the frame."
                ),
                "type": "number",
                "default": self._max_edge_trim,
                "min": 0,
                "max": 0.5,
                "step": 0.05,
            },
            {
                "key": "min_edge_trim",
                "label": "Minimum trim to bother",
                "description": (
                    "Leave the image untouched when every margin is thinner than this fraction - not worth a re-encode."
                ),
                "type": "number",
                "default": self._min_edge_trim,
                "min": 0,
                "max": 0.5,
                "step": 0.01,
            },
        ]

    def with_params(self, params: dict[str, Any]) -> "ImageEdgeTrimCleaner":
        return ImageEdgeTrimCleaner(
            edge_tol=float(params.get("edge_tol", self._edge_tol)),
            max_edge_trim=float(params.get("max_edge_trim", self._max_edge_trim)),
            min_edge_trim=float(params.get("min_edge_trim", self._min_edge_trim)),
        )

    def clean(self, media: dict[str, Any]) -> dict[str, Any]:
        """Return *media* cropped to its content box, or unchanged.

        Unchanged when there are no bytes to work with, the payload can't be
        decoded or re-encoded in its own format (SVG, a corrupt file), or the
        detector finds nothing worth trimming - the common case, which costs one
        decode plus a small resample and re-encodes nothing.
        """
        media_bytes = media.get("media_bytes")
        if not isinstance(media_bytes, (bytes, bytearray)) or not media_bytes:
            return media

        from vtscore.media.image.decode import open_upright  # noqa: PLC0415

        try:
            # Upright decode: the trimmed copy's dimensions become the media's
            # stored ``width``/``height``, which are the displayed ones, and the
            # grid thumbnail runs the same detector over the same upright pixels.
            with open_upright(io.BytesIO(media_bytes)) as src:
                box = solid_edge_box(
                    src,
                    edge_tol=self._edge_tol,
                    max_edge_trim=self._max_edge_trim,
                    min_edge_trim=self._min_edge_trim,
                )
                if box is None:
                    return media
                fmt = src.format or "PNG"
                save_kwargs: dict[str, Any] = {}
                exif = src.info.get("exif")
                if exif and fmt in _EXIF_WRITABLE_FORMATS:
                    save_kwargs["exif"] = exif
                cropped = src.crop(box)
                out = io.BytesIO()
                cropped.save(out, format=fmt, **save_kwargs)
                width, height = cropped.size
        except Exception:
            log.debug("image_edge_trim: undecodable payload, leaving unchanged", exc_info=True)
            return media

        trimmed_bytes = out.getvalue()
        cleaned = dict(media)
        cleaned["media_bytes"] = trimmed_bytes
        cleaned["file_size"] = len(trimmed_bytes)
        cleaned["width"] = width
        cleaned["height"] = height
        return cleaned
