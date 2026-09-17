"""Render a slate image with its box and a magnified view of the box's contents.

Two framings share one crop:

* :func:`inset_crop` -- the magnified, context-padded crop of the box, used by both.
* ``make_positive_slate.draw_with_inset`` pastes it *over* a bottom corner of the photo.
  Fine for "is this box one?", where the reviewer looks at the box.
* :func:`draw_with_side_inset` extends the canvas with black padding and places it
  *beside* the photo, so no part of the scene is covered. Needed for any question
  about *other* instances in the frame -- a prominence triage asks whether a more
  prominent instance exists, and the corner inset can hide exactly that instance.

**The side framing changes the canvas, and a drawn box is normalised to the canvas.**
The app stores ``region_box`` in [0, 1] relative to the image as displayed, so a box
drawn on a side-inset render is relative to the padded canvas, not the photo. Band is
computed from box area, so using it unconverted would mis-band every redraw -- the very
error such a triage exists to fix. :func:`side_inset_to_original` converts it back,
using the geometry :func:`draw_with_side_inset` returns; record that geometry per image.

No import side effects (unlike ``make_positive_slate``, which calls
``pile_config.setup_env()`` at import), so this module is directly testable.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

#: How big the inset may get, as a fraction of the image's shorter side.
INSET_FRAC = 0.42
#: The inset magnifies at least this much, so a sub-patch box is actually visible.
MIN_ZOOM = 3.0


def inset_crop(im, box: tuple[float, float, float, float]):
    """``(crop, (x0, y0, x1, y1), line_width)`` -- the magnified crop and the box in pixels.

    *im* is an RGB PIL image; *box* is normalised. The crop is what both framings paste.
    """
    from PIL import Image  # noqa: PLC0415

    W, H = im.size
    # VG boxes are not guaranteed to lie inside their image -- some run past
    # the edge, and a few are inverted -- so clamp before any arithmetic
    # that assumes a well-formed rectangle.
    x0, x1 = sorted((box[0] * W, box[2] * W))
    y0, y1 = sorted((box[1] * H, box[3] * H))
    x0, x1 = max(0.0, min(x0, W - 1.0)), max(1.0, min(x1, float(W)))
    y0, y1 = max(0.0, min(y0, H - 1.0)), max(1.0, min(y1, float(H)))
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)

    # The crop is padded around the box so the object keeps its context --
    # a box cropped exactly to its edges is often unrecognisable, and for a
    # sub-patch object it is a smudge at any magnification. What makes a
    # 20-pixel backpack identifiable is seeing the person wearing it, so the
    # padding has a floor in absolute image terms rather than being a
    # multiple of a tiny box.
    pad = max(max(bw, bh) * 0.6, min(W, H) * 0.10)
    cx0, cy0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
    cx1, cy1 = min(W, int(x1 + pad)), min(H, int(y1 + pad))
    cx1, cy1 = max(cx1, cx0 + 2), max(cy1, cy0 + 2)
    crop = im.crop((cx0, cy0, min(cx1, W), min(cy1, H)))

    target = int(min(W, H) * INSET_FRAC)
    zoom = max(MIN_ZOOM, target / max(crop.width, crop.height))
    iw, ih = int(crop.width * zoom), int(crop.height * zoom)
    iw, ih = min(iw, target), min(ih, target)
    crop = crop.resize((max(1, iw), max(1, ih)), Image.LANCZOS)
    lw = max(2, int(min(W, H) * 0.006))
    return crop, (x0, y0, x1, y1), lw


def _box_px(im, box: Sequence[float]) -> tuple[float, float, float, float]:
    """*box* (normalised) in pixels, clamped exactly as :func:`inset_crop` clamps it."""
    W, H = im.size
    x0, x1 = sorted((box[0] * W, box[2] * W))
    y0, y1 = sorted((box[1] * H, box[3] * H))
    return (
        max(0.0, min(x0, W - 1.0)),
        max(0.0, min(y0, H - 1.0)),
        max(1.0, min(x1, float(W))),
        max(1.0, min(y1, float(H))),
    )


def draw_with_side_inset(
    src: Path,
    box: tuple[float, float, float, float],
    dest: Path,
    also: Sequence[Sequence[float]] = (),
) -> dict[str, int | str]:
    """Write *src* with *box* outlined, and the magnified crop on padding beside it.

    The canvas always grows to the RIGHT. The reviewer's screen is far wider than it is tall,
    so height is what limits how large the photo is drawn; padding sideways leaves it at full
    height, where padding below would shrink it. (An earlier draft padded along the photo's
    shorter side, which is the wrong constraint for a wide screen.) The photo sits at the
    canvas origin, so :func:`side_inset_to_original` is a pure rescale.

    *also* outlines further boxes of the same class on the photo, for an image whose
    class has several instances (a VG positive is banded by the union of them all, so
    a reviewer asked "is the most prominent one boxed?" has to see every one). The
    magnified panel then shows the union of *box* and *also*. With no *also* the
    render is byte-identical to the single-box one.

    Returns the geometry needed to convert a box drawn on this render back to the photo.
    """
    from PIL import Image, ImageDraw  # noqa: PLC0415

    with Image.open(src) as im:
        im = im.convert("RGB")
        W, H = im.size
        boxes = [tuple(box), *(tuple(b) for b in also)]
        union = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
        crop, (x0, y0, x1, y1), lw = inset_crop(im, union if also else box)
        gap = max(4, 2 * lw)
        side = "right"
        cw, ch = W + crop.width + 2 * gap, max(H, crop.height + 2 * gap)
        ix, iy = W + gap, (ch - crop.height) // 2
        out = Image.new("RGB", (cw, ch), (0, 0, 0))
        out.paste(im, (0, 0))
        d = ImageDraw.Draw(out)
        if also:
            for b in boxes:
                bx0, by0, bx1, by1 = _box_px(im, b)
                d.rectangle([bx0, by0, bx1, by1], outline=(255, 32, 32), width=lw)
        else:
            d.rectangle([x0, y0, x1, y1], outline=(255, 32, 32), width=lw)
        out.paste(crop, (ix, iy))
        d.rectangle([ix, iy, ix + crop.width - 1, iy + crop.height - 1], outline=(255, 32, 32), width=lw)
        out.save(dest, quality=92)
        return {"orig_w": W, "orig_h": H, "canvas_w": cw, "canvas_h": ch, "side": side}


def side_inset_to_original(box: list[float] | tuple[float, ...], geom: dict) -> list[float]:
    """Convert a box normalised to a side-inset CANVAS into one normalised to the PHOTO.

    The photo sits at the canvas origin, so this is a rescale by canvas/photo size.
    Anything drawn into the padding is clamped to the photo's edge.
    """
    sx = geom["canvas_w"] / geom["orig_w"]
    sy = geom["canvas_h"] / geom["orig_h"]
    x0, y0, x1, y1 = box
    return [min(1.0, max(0.0, v)) for v in (x0 * sx, y0 * sy, x1 * sx, y1 * sy)]
