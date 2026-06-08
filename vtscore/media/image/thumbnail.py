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


def make_image_thumbnail(media_bytes: bytes, max_dim: int = DEFAULT_MAX_DIM) -> tuple[bytes, str] | None:
    """Return ``(thumbnail_bytes, mimetype)`` downscaled to ``max_dim`` px.

    The longest side of the result is at most ``max_dim``; images already
    within that bound are re-encoded (never upscaled) so callers still get a
    compact, decode-cheap artifact.  Opaque images become JPEG; images that
    carry an alpha channel become PNG so transparency survives.  Camera EXIF
    orientation is applied so portrait photos aren't shown sideways.

    Returns ``None`` when the bytes can't be decoded as an image (e.g. SVG or
    a corrupt file), letting the caller fall back to the original bytes.
    """
    from PIL import Image, ImageOps  # noqa: PLC0415

    try:
        with Image.open(io.BytesIO(media_bytes)) as src:
            img = ImageOps.exif_transpose(src) or src
            has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
            img.thumbnail((max_dim, max_dim), Image.LANCZOS)
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
