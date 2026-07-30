"""Downscale image bytes to a bounded thumbnail for grid/list display.

Grid and list tiles render a media item at a few dozen to a couple hundred
pixels, but the ``/api/medias/<id>/image`` route streams the *original*
source bytes.  A gallery of a few hundred high-resolution photos therefore
forces the browser to download and decode every full-size bitmap at once
(tens of megabytes of decoded pixels each), which exhausts memory and can
hang the machine.  Serving a small thumbnail instead caps the per-tile
decode cost to a fixed budget regardless of the source resolution.

The full-resolution route is unchanged and still backs the detail viewer,
where a crisp original is wanted.
"""

from __future__ import annotations

import io

from vtscore.media.image.edge_trim import trim_solid_edges

#: Longest-side length (px) of a generated thumbnail.  Comfortably covers the
#: largest grid tile (XL ≈ 200 px CSS, ≈ 400 px on a 2× display) while keeping
#: the decoded bitmap small (≈ ``MAX_DIM`` × ``MAX_DIM`` × 4 bytes).  The same
#: thumbnail is reused at every zoom level, so the browser never re-fetches
#: when the user zooms in or out.
DEFAULT_MAX_DIM = 384

#: A region box whose width *and* height both reach this fraction of the image
#: covers (effectively) the whole frame, so cropping to it is a no-op worth
#: skipping.  Mirrors the frontend's near-full-image guard in
#: ``media-item.component.ts`` (``bestRegionStyle``).
_FULL_BOX_THRESHOLD = 0.99


def normalize_region_crop(
    box: object,
) -> tuple[float, float, float, float] | None:
    """Validate and canonicalise a normalised crop box.

    Accepts a 4-element sequence ``(x0, y0, x1, y1)`` of fractions of the image
    width/height.  Returns a clean tuple with ``x0 < x1`` and ``y0 < y1``, each
    coordinate clamped to ``[0, 1]``.  Returns ``None`` (meaning "don't crop")
    when the box is missing, malformed, degenerate (zero area), or covers
    effectively the whole image, so callers can pass a raw box through and let
    this decide whether a crop is warranted.
    """
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    if not all(v == v for v in (x0, y0, x1, y1)):  # reject NaN
        return None
    x0, x1 = sorted((min(max(x0, 0.0), 1.0), min(max(x1, 0.0), 1.0)))
    y0, y1 = sorted((min(max(y0, 0.0), 1.0), min(max(y1, 0.0), 1.0)))
    w, h = x1 - x0, y1 - y0
    if w <= 0.0 or h <= 0.0:
        return None
    if w >= _FULL_BOX_THRESHOLD and h >= _FULL_BOX_THRESHOLD:
        return None
    return (x0, y0, x1, y1)


def make_image_thumbnail(
    media_bytes: bytes,
    max_dim: int = DEFAULT_MAX_DIM,
    crop: tuple[float, float, float, float] | None = None,
) -> tuple[bytes, str] | None:
    """Return ``(thumbnail_bytes, mimetype)`` downscaled to ``max_dim`` px.

    The longest side of the result is at most ``max_dim``; images already
    within that bound are re-encoded (never upscaled) so callers still get a
    compact, decode-cheap artifact.  Opaque images become JPEG; images that
    carry an alpha channel become PNG so transparency survives.  Camera EXIF
    orientation is applied so portrait photos aren't shown sideways.

    When ``crop`` is given (a normalised ``(x0, y0, x1, y1)`` box, already
    canonicalised via :func:`normalize_region_crop`), the image is cropped to
    that sub-region *before* downscaling, so the thumbnail shows only the
    user's voted region rather than the whole frame.

    Returns ``None`` when the bytes can't be decoded as an image (e.g. SVG or
    a corrupt file), letting the caller fall back to the original bytes.
    """
    from PIL import Image, ImageOps  # noqa: PLC0415

    try:
        with Image.open(io.BytesIO(media_bytes)) as src:
            img = ImageOps.exif_transpose(src) or src
            if crop is not None:
                img = _crop_to_region(img, crop)
            else:
                # No explicit region: shave any baked-in solid white/black
                # letterbox/pillarbox margins so a padded source reports the
                # aspect ratio of its content rather than its padding.  Only the
                # thumbnail is affected — the full-resolution route still streams
                # the untouched original for the detail viewer / main canvas.
                img = trim_solid_edges(img)
            has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            out = io.BytesIO()
            if has_alpha:
                img.convert("RGBA").save(out, format="PNG", optimize=True)
                mimetype = "image/png"
            else:
                img.convert("RGB").save(out, format="JPEG", quality=82, optimize=True)
                mimetype = "image/jpeg"
            return out.getvalue(), mimetype
    except Exception:
        return None


def _crop_to_region(img, crop: tuple[float, float, float, float]):
    """Crop a PIL image to a normalised ``(x0, y0, x1, y1)`` box.

    Coordinates are fractions of the (EXIF-corrected) image dimensions.  The
    pixel box is rounded to integers and widened to at least one pixel on each
    axis so a sliver-thin box still yields a decodable crop.
    """
    x0, y0, x1, y1 = crop
    w, h = img.size
    px0, px1 = int(round(x0 * w)), int(round(x1 * w))
    py0, py1 = int(round(y0 * h)), int(round(y1 * h))
    if px1 <= px0:
        px1 = min(px0 + 1, w)
        px0 = max(px1 - 1, 0)
    if py1 <= py0:
        py1 = min(py0 + 1, h)
        py0 = max(py1 - 1, 0)
    return img.crop((px0, py0, px1, py1))
