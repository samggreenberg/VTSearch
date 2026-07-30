"""Near-solid border detection: the shared half of "shave the padding off".

Many sources ship images already padded - letterbox bars top/bottom, a
pillarbox on one side, whitespace around a centred logo - and that padding is
content-free in two different places:

* **Thumbnails** (:func:`~vtscore.media.image.thumbnail.make_image_thumbnail`)
  want the *content's* aspect ratio, or the browse canvas' "half-crop,
  half-pad" fit ends up padding the padding.
* **Embeddings** (:class:`~vtscore.media.image.cleaner.ImageEdgeTrimCleaner`)
  want the embedder to spend its capacity on the subject rather than on a
  frame of blank pixels.

Both callers ask the same question - *where does the content actually start?* -
so the detector lives here once and each caller supplies its own thresholds.
"""

from __future__ import annotations

from typing import Any

#: Per-channel distance from pure white (255) or pure black (0) that a pixel may
#: still fall within and count as part of a "solid" edge margin.  Wide enough to
#: absorb JPEG ringing and video-level black (~16) without swallowing genuine
#: near-white/near-black content.
DEFAULT_EDGE_TOL = 16

#: Never trim more than this fraction off any single side.  A tiny subject
#: floating in a vast blank margin would otherwise blow up into a full-frame
#: crop; this caps how tight the content box may pull each edge.
DEFAULT_MAX_EDGE_TRIM = 0.45

#: Ignore a candidate trim smaller than this fraction on every side - not worth
#: the re-crop, and it keeps a frame that's merely *near* solid-free untouched.
DEFAULT_MIN_EDGE_TRIM = 0.02

#: Detect edges on a copy no larger than this (longest side, px).  The trim box
#: is fractional and applied to the full-resolution image, so the scan cost
#: stays fixed and tiny no matter how large the source is.
EDGE_ANALYSIS_DIM = 256


def solid_edge_box(
    img: Any,
    *,
    edge_tol: float = DEFAULT_EDGE_TOL,
    max_edge_trim: float = DEFAULT_MAX_EDGE_TRIM,
    min_edge_trim: float = DEFAULT_MIN_EDGE_TRIM,
) -> tuple[int, int, int, int] | None:
    """Return the pixel crop box that drops *img*'s near-solid border margins.

    Finds the tight bounding box of *content* (any pixel that is neither
    near-white nor near-black within *edge_tol*), so each of the four edges is
    trimmed independently: a frame padded only on top, or only on the left,
    loses just that margin.

    Detection runs on a copy downscaled to at most :data:`EDGE_ANALYSIS_DIM` px,
    so the added cost is a single small resample regardless of the source
    resolution; the resulting box is fractional and applied to the full image.
    No single side is ever pulled in by more than *max_edge_trim*, so a small
    subject in a large blank field can't explode into a full-frame crop.

    Returns ``None`` when there is nothing worth trimming: an image too small to
    analyse, a frame that's entirely one solid tone (no content to pull the box
    toward), or margins all thinner than *min_edge_trim*.
    """
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    w, h = img.size
    if w < 4 or h < 4:
        return None

    scale = min(1.0, EDGE_ANALYSIS_DIM / max(w, h))
    if scale < 1.0:
        aw, ah = max(1, round(w * scale)), max(1, round(h * scale))
        analysis = img.resize((aw, ah), Image.Resampling.BILINEAR)
    else:
        analysis = img

    arr = np.asarray(analysis.convert("RGB"), dtype=np.int16)
    near_white = np.all(arr >= 255 - edge_tol, axis=2)
    near_black = np.all(arr <= edge_tol, axis=2)
    content = ~(near_white | near_black)
    if not content.any():
        return None  # nothing but solid tone - no content to pull the box toward

    rows = np.flatnonzero(content.any(axis=1))
    cols = np.flatnonzero(content.any(axis=0))
    ch, cw = content.shape
    fx0, fx1 = cols[0] / cw, (cols[-1] + 1) / cw
    fy0, fy1 = rows[0] / ch, (rows[-1] + 1) / ch

    # Cap how far each edge may pull in so a tiny subject can't blow up.
    fx0, fy0 = min(fx0, max_edge_trim), min(fy0, max_edge_trim)
    fx1, fy1 = max(fx1, 1.0 - max_edge_trim), max(fy1, 1.0 - max_edge_trim)

    if fx0 < min_edge_trim and fy0 < min_edge_trim and (1.0 - fx1) < min_edge_trim and (1.0 - fy1) < min_edge_trim:
        return None  # margins negligible on every side; leave the frame as-is

    px0, px1 = int(round(fx0 * w)), int(round(fx1 * w))
    py0, py1 = int(round(fy0 * h)), int(round(fy1 * h))
    px1, py1 = max(px1, px0 + 1), max(py1, py0 + 1)
    if (px0, py0, px1, py1) == (0, 0, w, h):
        return None
    return (px0, py0, px1, py1)


def trim_solid_edges(
    img: Any,
    *,
    edge_tol: float = DEFAULT_EDGE_TOL,
    max_edge_trim: float = DEFAULT_MAX_EDGE_TRIM,
    min_edge_trim: float = DEFAULT_MIN_EDGE_TRIM,
) -> Any:
    """Crop away *img*'s near-solid white/black border margins.

    Thin wrapper over :func:`solid_edge_box`: returns ``img`` unchanged (the
    same object) when there is nothing worth trimming, so callers can use it
    unconditionally.
    """
    box = solid_edge_box(
        img,
        edge_tol=edge_tol,
        max_edge_trim=max_edge_trim,
        min_edge_trim=min_edge_trim,
    )
    return img if box is None else img.crop(box)
