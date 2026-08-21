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

**EXIF orientation is applied here, once, for everybody.**  Phone and camera
JPEGs are stored in sensor order with an EXIF ``Orientation`` tag telling the
viewer how to rotate them; browsers honour it, so the picture the user sees is
the *upright* one.  Every decode in this module therefore returns upright
pixels (:func:`decode_bounded`, :func:`decode_bounded_rgb`,
:func:`open_upright`), and :func:`upright_size` reports the matching
dimensions from the header alone.  That makes "the original image's space" mean
one thing everywhere: what the user sees.  Embeddings, thumbnails, OCR boxes,
face boxes, crop boxes and the media's stored ``width``/``height`` all agree,
and a call site that divides a bounded-decode coordinate by ``scale`` lands in
the same space the stored dimensions describe.

:func:`open_image` is the deliberate exception — it stays lazy and untransposed
because it exists for header-only reads, where forcing a decode to rotate
pixels nobody asked for would defeat the point.  Ask it for metadata, not for
pixels.
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

#: EXIF tag holding the display orientation.  ``1`` (and a missing tag) mean
#: "already upright"; ``5``-``8`` additionally swap the two axes.
EXIF_ORIENTATION_TAG = 274

#: Orientations that transpose the image, i.e. whose upright form has the
#: source's width and height exchanged.
_AXIS_SWAPPING_ORIENTATIONS = frozenset({5, 6, 7, 8})

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


def exif_orientation(img: "Image.Image") -> int:
    """Return *img*'s EXIF display orientation, or ``1`` when it has none.

    Reads the header only — safe to call on a lazy :func:`open_image` result
    without forcing a decode.  A payload whose EXIF block is missing or
    unparseable reports ``1`` rather than raising: an unreadable tag and an
    absent one both mean "no rotation is known to be needed".
    """
    try:
        exif = img.getexif()
    except Exception:
        return 1
    if not exif:
        return 1
    value = exif.get(EXIF_ORIENTATION_TAG)
    return value if isinstance(value, int) and 1 <= value <= 8 else 1


def apply_exif_orientation(img: "Image.Image") -> "Image.Image":
    """Return *img* rotated to its EXIF display orientation.

    Returns *img* **itself** when there is nothing to do (no tag, or a tag that
    already says upright) — the overwhelmingly common case, and the reason this
    exists instead of a bare ``ImageOps.exif_transpose`` call, which copies the
    full bitmap unconditionally.  When a rotation *is* applied the result is a
    new image with the orientation tag stripped, so re-applying is a no-op; the
    source's ``format`` is carried across so a caller re-encoding a crop keeps
    the original container instead of silently falling back to PNG.

    Callers that own *img* should close it when a different object comes back.
    """
    if exif_orientation(img) == 1:
        return img

    from PIL import ImageOps  # noqa: PLC0415

    upright = ImageOps.exif_transpose(img)
    if upright is None or upright is img:
        return img
    upright.format = img.format
    return upright


def _upright(img: "Image.Image") -> "Image.Image":
    """:func:`apply_exif_orientation`, closing *img* when it is superseded."""
    upright = apply_exif_orientation(img)
    if upright is not img:
        img.close()
    return upright


def exif_upright_size(img: "Image.Image") -> tuple[int, int]:
    """Return *img*'s ``(width, height)`` **as displayed**, from the header.

    Costs one EXIF parse and no decode: a 90°/270° orientation just exchanges
    the two numbers.  This is what a media item's stored ``width``/``height``
    must be, so that boxes expressed against them address the same pixels the
    user is looking at.
    """
    width, height = img.size
    if exif_orientation(img) in _AXIS_SWAPPING_ORIENTATIONS:
        return height, width
    return width, height


def upright_size(source: ImageSource) -> tuple[int, int]:
    """:func:`exif_upright_size` for a source that is not open yet."""
    with open_image(source) as img:
        return exif_upright_size(img)


def open_upright(source: ImageSource) -> "Image.Image":
    """Open *source* at native resolution, rotated to its display orientation.

    The full-resolution counterpart to :func:`decode_bounded`, for the paths
    that reason in pixel coordinates against a media's stored
    ``width``/``height`` — cropping, clipping, origin resolution.  Unlike
    :func:`open_image` the result is *not* lazy when a rotation is needed
    (rotating requires the pixels), which is exactly the trade those call sites
    want: they are about to decode anyway, and a crop box that disagreed with
    the stored dimensions by 90° would cut out the wrong region.
    """
    return _upright(open_image(source))


def decode_bounded(
    source: ImageSource,
    max_pixels: int | None = None,
) -> tuple["Image.Image", float]:
    """Open *source* upright, downsampled to at most *max_pixels* total pixels.

    Returns ``(img, scale)`` where ``scale`` is the ratio of the returned
    image's longest-side length to the original's — ``1.0`` when the source
    already fit inside the budget, and ``< 1.0`` when it was reduced.  Call
    sites that report pixel coordinates (bounding boxes) divide by ``scale``
    to map them back into the original image's coordinate space.

    EXIF display orientation is applied (see the module docstring), so "the
    original image's space" means the *upright* image — the same space the
    media's stored ``width``/``height`` describe and the same one the user
    sees in the browser.  The downsample happens first and the rotation
    second, so a rotated gigapixel photo is still never materialised at full
    resolution, and ``scale`` is unaffected either way: a transpose exchanges
    the axes without changing the linear reduction ratio.

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
    # Measured before the transpose, while both sides are still in source
    # axis order; the rotation below preserves the ratio but may swap w/h.
    scale = (img.width / orig_w) if orig_w > 0 else 1.0
    return _upright(img), scale


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
