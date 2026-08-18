"""Bounded image decoding — open arbitrarily large images without choking.

Pillow ships a "decompression bomb" guard: :func:`PIL.Image.open` emits a
``DecompressionBombWarning`` once an image exceeds ``Image.MAX_IMAGE_PIXELS``
(~89 MP by default) and raises ``DecompressionBombError`` past *twice* that,
with a message about a possible DOS attack.  The check fires on the header
alone, so a merely *large* picture — a gigapixel panorama, a whole-slide
microscopy scan, a stitched satellite mosaic, a high-DPI poster scan — is
refused outright before a single pixel is decoded.

That trade is wrong for VTSearch.  Users point the importer at their own
media; a 330-megapixel photo is not hostile, it is just big, and silently
dropping it from a gallery is worse than the memory it costs.  So this
module lifts the ceiling (:func:`configure_pil_limits`) and replaces it
with the thing the guard was actually protecting against: an *unbounded
decode*.  :func:`decode_bounded` opens an image at any resolution but
downsamples it to a fixed pixel budget (:data:`~vtscore.config.MAX_DECODE_PIXELS`),
so peak memory stays capped no matter how large the source is, and the
downstream libraries that genuinely do choke on huge bitmaps (model
processors, OCR, face detection) never see one.

Two decode postures, and which one a call site wants depends on whether it
reasons in pixel coordinates:

* **Bounded** — thumbnails, embedders, extractors, converters.  These
  resize to a few hundred pixels anyway, so decoding a full gigapixel
  bitmap first is pure waste.  Call sites that report pixel coordinates
  multiply them back by ``1 / scale`` to stay in original-image space.
* **Full resolution** — cropping/clipping (:mod:`vtscore.media.cropping`,
  :mod:`vtscore.media.lazy_clip`, the image clippers).  A crop's box comes
  from the media's stored ``width``/``height`` and the user expects the
  region back at full fidelity, so those paths keep decoding at native
  size.  They still benefit from the lifted ceiling: before it, they
  raised instead of cropping.

:func:`configure_pil_limits` is invoked from ``vtscore/media/image/__init__.py``,
which the media registry imports eagerly while auto-discovering media types,
so the ceiling is lifted process-wide before any image work begins —
including in code that calls ``PIL.Image.open`` directly.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import IO, TYPE_CHECKING, Union

from vtscore.config import MAX_DECODE_PIXELS

if TYPE_CHECKING:
    from PIL import Image

#: Anything :func:`PIL.Image.open` accepts, plus a raw in-memory blob.
ImageSource = Union[str, Path, bytes, bytearray, memoryview, IO[bytes]]

_limits_configured = False


def configure_pil_limits() -> None:
    """Remove Pillow's decompression-bomb ceiling, process-wide.

    Idempotent and cheap to re-call.  After this, ``Image.open`` never
    refuses an image for being large; :func:`decode_bounded` is what keeps
    the decode itself from running away.
    """
    global _limits_configured
    if _limits_configured:
        return
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        # Pillow is a hard dependency, but this runs at media-registry import:
        # a missing install should surface where an image is actually decoded,
        # not by taking down media-type discovery for every other type.
        return

    # ``None`` disables the check entirely (both the warning and the error).
    Image.MAX_IMAGE_PIXELS = None
    _limits_configured = True


def _as_openable(source: ImageSource) -> Union[str, Path, IO[bytes]]:
    """Wrap a raw ``bytes`` blob in a stream; pass paths/streams through."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        return io.BytesIO(bytes(source))
    return source


def open_image(source: ImageSource) -> "Image.Image":
    """``Image.open`` with the decompression-bomb ceiling lifted.

    Use this when only the *header* is needed (dimensions, format, mode):
    the returned image is lazy, so nothing is decoded and an enormous source
    costs nothing.  When the pixels are actually wanted, prefer
    :func:`decode_bounded`.
    """
    from PIL import Image  # noqa: PLC0415

    configure_pil_limits()
    return Image.open(_as_openable(source))


def decode_bounded(
    source: ImageSource,
    max_pixels: int | None = None,
) -> tuple["Image.Image", float]:
    """Open *source*, downsampled to at most *max_pixels* total pixels.

    Returns ``(img, scale)`` where ``scale`` is the ratio of the returned
    image's width to the original's — ``1.0`` when the source already fit
    inside the budget, and ``< 1.0`` when it was reduced.  Call sites that
    report pixel coordinates (bounding boxes) divide by ``scale`` to map
    them back into the original image's coordinate space.

    Aspect ratio is preserved.  Small images are never upscaled.  For JPEGs
    the reduction goes through Pillow's DCT-scaled ``draft`` path, so a huge
    JPEG is never materialised at full resolution even transiently.

    Pass ``max_pixels=0`` (or a negative value) to decode at native size.
    """
    from PIL import Image  # noqa: PLC0415

    configure_pil_limits()
    budget = MAX_DECODE_PIXELS if max_pixels is None else max_pixels
    img = Image.open(_as_openable(source))
    orig_w, orig_h = img.size
    if budget > 0 and orig_w > 0 and orig_h > 0 and orig_w * orig_h > budget:
        ratio = math.sqrt(budget / float(orig_w * orig_h))
        target = (max(1, int(orig_w * ratio)), max(1, int(orig_h * ratio)))
        # ``thumbnail`` drafts first (a DCT-scaled JPEG decode at 1/2, 1/4 or
        # 1/8 size) and only then resamples, so the full-resolution bitmap
        # never has to fit in memory for the formats that support it.
        img.thumbnail(target, Image.Resampling.LANCZOS, reducing_gap=2.0)
    scale = (img.width / orig_w) if orig_w > 0 else 1.0
    return img, scale


def decode_bounded_rgb(
    source: ImageSource,
    max_pixels: int | None = None,
) -> tuple["Image.Image", float]:
    """:func:`decode_bounded`, converted to ``RGB``.

    The conversion happens *after* the downsample, so the RGB copy costs at
    most ``max_pixels * 3`` bytes rather than the source's full footprint.
    """
    img, scale = decode_bounded(source, max_pixels)
    with img:
        return img.convert("RGB"), scale
