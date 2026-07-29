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

#: Per-channel distance from pure white (255) or pure black (0) that a pixel may
#: still fall within and count as part of a "solid" edge margin.  Wide enough to
#: absorb JPEG ringing and video-level black (~16) without swallowing genuine
#: near-white/near-black content.
_EDGE_TOL = 16

#: Never trim more than this fraction off any single side.  A tiny subject
#: floating in a vast blank margin would otherwise blow up into a full-frame
#: crop; this caps how tight the content box may pull each edge.
_MAX_EDGE_TRIM = 0.45

#: Ignore a candidate trim smaller than this fraction on every side — not worth
#: the re-crop, and it keeps a frame that's merely *near* solid-free untouched.
_MIN_EDGE_TRIM = 0.02

#: Detect edges on a copy no larger than this (longest side, px).  The trim box
#: is fractional and applied to the full-resolution image, so the scan cost
#: stays fixed and tiny no matter how large the source is.
_EDGE_ANALYSIS_DIM = 256


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

    from vtscore.media.image.decode import decode_bounded  # noqa: PLC0415

    try:
        # Bounded decode: the result is at most ``max_dim`` px on its longest
        # side, so a gigapixel source has nothing to gain from being decoded at
        # full resolution first.  Both the crop and the edge-trim below work in
        # *fractional* coordinates, so they are unaffected by the downsample.
        decoded, _scale = decode_bounded(media_bytes)
        with decoded as src:
            img = ImageOps.exif_transpose(src) or src
            if crop is not None:
                img = _crop_to_region(img, crop)
            else:
                # No explicit region: shave any baked-in solid white/black
                # letterbox/pillarbox margins so a padded source reports the
                # aspect ratio of its content rather than its padding.  Only the
                # thumbnail is affected — the full-resolution route still streams
                # the untouched original for the detail viewer / main canvas.
                img = _trim_solid_edges(img)
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


def _trim_solid_edges(img):
    """Crop away near-solid white/black border margins from a PIL image.

    Many sources ship already padded — letterbox bars top/bottom, a pillarbox
    on one side, whitespace around a centred logo — which makes the image's
    stored aspect ratio a poor guide for a preview tile (the browse canvas'
    "half-crop, half-pad" fit then pads the padding).  This finds the tight
    bounding box of *content* (any pixel that is neither near-white nor
    near-black within :data:`_EDGE_TOL`) and crops to it, so each of the four
    edges is trimmed independently: a frame padded only on top, or only on the
    left, loses just that margin.

    Detection runs on a copy downscaled to at most :data:`_EDGE_ANALYSIS_DIM`
    px, so the added cost is a single small resample regardless of the source
    resolution; the resulting box is fractional and applied to the full image.
    Returns ``img`` unchanged when there is nothing worth trimming: a frame
    that's entirely one solid tone (no content to pull toward), or whose
    margins are all thinner than :data:`_MIN_EDGE_TRIM`.  No single side is ever
    trimmed by more than :data:`_MAX_EDGE_TRIM`, so a small subject in a large
    blank field can't explode into a full-frame crop.
    """
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    w, h = img.size
    if w < 4 or h < 4:
        return img

    scale = min(1.0, _EDGE_ANALYSIS_DIM / max(w, h))
    if scale < 1.0:
        aw, ah = max(1, round(w * scale)), max(1, round(h * scale))
        analysis = img.resize((aw, ah), Image.Resampling.BILINEAR)
    else:
        analysis = img

    arr = np.asarray(analysis.convert("RGB"), dtype=np.int16)
    near_white = np.all(arr >= 255 - _EDGE_TOL, axis=2)
    near_black = np.all(arr <= _EDGE_TOL, axis=2)
    content = ~(near_white | near_black)
    if not content.any():
        return img  # nothing but solid tone — no content to pull the box toward

    rows = np.flatnonzero(content.any(axis=1))
    cols = np.flatnonzero(content.any(axis=0))
    ch, cw = content.shape
    fx0, fx1 = cols[0] / cw, (cols[-1] + 1) / cw
    fy0, fy1 = rows[0] / ch, (rows[-1] + 1) / ch

    # Cap how far each edge may pull in so a tiny subject can't blow up.
    fx0, fy0 = min(fx0, _MAX_EDGE_TRIM), min(fy0, _MAX_EDGE_TRIM)
    fx1, fy1 = max(fx1, 1.0 - _MAX_EDGE_TRIM), max(fy1, 1.0 - _MAX_EDGE_TRIM)

    if fx0 < _MIN_EDGE_TRIM and fy0 < _MIN_EDGE_TRIM and (1.0 - fx1) < _MIN_EDGE_TRIM and (1.0 - fy1) < _MIN_EDGE_TRIM:
        return img  # margins negligible on every side; leave the frame as-is

    px0, px1 = int(round(fx0 * w)), int(round(fx1 * w))
    py0, py1 = int(round(fy0 * h)), int(round(fy1 * h))
    px1, py1 = max(px1, px0 + 1), max(py1, py0 + 1)
    return img.crop((px0, py0, px1, py1))


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
