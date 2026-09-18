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


#: The side panel magnifies at most this much, so a tiny box is not blown up to mush.
SIDE_MAX_ZOOM = 8.0
#: The whole canvas stays at most this wide for its height, so on a wide screen the photo's
#: HEIGHT is still what limits its size -- a panel as wide as a landscape photo would make it
#: ~2.9:1 and shrink the photo on anything narrower than an ultrawide.
MAX_CANVAS_ASPECT = 2.2
#: Each individual annotation of the class, when there are several. Thin and amber: VG often
#: holds many overlapping boxes for one object (21 on one bench image), and outlining them all
#: in red buried the one box the band actually comes from.
INSTANCE_EDGE = (255, 190, 0)
#: Two annotations overlapping by at least this much are the same object annotated twice, for
#: DISPLAY only: VG holds 21 boxes on one bench image and 23 on one bird image, mostly repeats.
DUPLICATE_IOU = 0.5
#: Outline of the side panel itself -- deliberately NOT red, so it cannot be read as the box.
PANEL_EDGE = (110, 110, 110)


def distinct_boxes(boxes: Sequence[Sequence[float]], iou: float = DUPLICATE_IOU) -> list[Sequence[float]]:
    """*boxes* with near-duplicates dropped, largest first -- for drawing, never for banding.

    VG is annotated by many people, so one object collects many boxes: 21 on one bench image,
    23 on one bird image. Outlining every one buries the object. A box is dropped when it
    overlaps a bigger kept box by *iou*, or sits almost entirely inside one.
    """
    kept: list[Sequence[float]] = []
    for b in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        ba = max(1e-12, (b[2] - b[0]) * (b[3] - b[1]))
        dup = False
        for k in kept:
            ix = max(0.0, min(b[2], k[2]) - max(b[0], k[0]))
            iy = max(0.0, min(b[3], k[3]) - max(b[1], k[1]))
            inter = ix * iy
            ka = max(1e-12, (k[2] - k[0]) * (k[3] - k[1]))
            if inter / (ba + ka - inter) >= iou or inter / ba >= 0.8:
                dup = True
                break
        if not dup:
            kept.append(b)
    return kept


def draw_with_side_inset(
    src: Path,
    box: tuple[float, float, float, float],
    dest: Path,
    also: Sequence[Sequence[float]] = (),
) -> dict[str, int | str]:
    """Write *src* with *box* outlined, and a magnified view of it on padding beside the photo.

    The canvas always grows to the RIGHT. The reviewer's screen is far wider than it is tall,
    so height is what limits how large the photo is drawn; padding sideways leaves it at full
    height, where padding below would shrink it. The photo sits at the canvas origin, so
    :func:`side_inset_to_original` is a pure rescale.

    **The panel is a faithful view, not the corner inset moved sideways.** The corner inset
    (:func:`inset_crop`) caps width and height to the same target independently, which squashes
    any crop that is not square, and its red frame surrounds the padded context rather than the
    box -- so the reviewer could not tell whether something near the box's edge was inside it
    (#3926, reported on a seat-check render). Here the crop keeps its aspect ratio, the box
    itself is drawn in red INSIDE the panel at the same place it has on the photo, and the
    panel's own edge is grey. The context padding is the corner inset's, so a small object still
    comes with its surroundings.

    *also* carries the class's further annotations in this image. The band comes from the
    UNION of them all (``vg_scale.band_for``), so the union is what is drawn in red -- it is
    the box the reviewer is being asked about -- and each annotation is drawn thin, in amber,
    underneath it. Drawing every annotation in red instead made a 21-box VG bench unreadable
    and hid the band's own box.

    Returns the geometry needed to convert a box drawn on this render back to the photo.
    """
    from PIL import Image, ImageDraw  # noqa: PLC0415

    with Image.open(src) as im:
        im = im.convert("RGB")
        W, H = im.size
        boxes = [_box_px(im, b) for b in (tuple(box), *(tuple(b) for b in also))]
        ux0, uy0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        ux1, uy1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        bw, bh = max(1.0, ux1 - ux0), max(1.0, uy1 - uy0)
        pad = max(max(bw, bh) * 0.6, min(W, H) * 0.10)
        cx0, cy0 = max(0, int(ux0 - pad)), max(0, int(uy0 - pad))
        cx1, cy1 = min(W, int(ux1 + pad) + 1), min(H, int(uy1 + pad) + 1)
        cx1, cy1 = max(cx1, cx0 + 2), max(cy1, cy0 + 2)
        crw, crh = cx1 - cx0, cy1 - cy0
        lw = max(2, int(min(W, H) * 0.006))
        gap = max(4, 2 * lw)
        # One scale for both axes. The panel may be as tall as the photo; its width is whatever
        # keeps the canvas within MAX_CANVAS_ASPECT, but never under 40% of a very wide photo.
        max_pw = max(MAX_CANVAS_ASPECT * H - W - 2 * gap, 0.4 * W)
        zoom = min((H - 2 * gap) / crh, max_pw / crw, SIDE_MAX_ZOOM)
        pw, ph = max(1, round(crw * zoom)), max(1, round(crh * zoom))
        panel = im.crop((cx0, cy0, cx1, cy1)).resize((pw, ph), Image.LANCZOS)
        cw, ch = W + pw + 2 * gap, H
        ix, iy = W + gap, (H - ph) // 2
        out = Image.new("RGB", (cw, ch), (0, 0, 0))
        out.paste(im, (0, 0))
        out.paste(panel, (ix, iy))
        d = ImageDraw.Draw(out)
        d.rectangle([ix - 1, iy - 1, ix + pw, iy + ph], outline=PANEL_EDGE, width=1)
        sx, sy = pw / crw, ph / crh

        def outline(b, colour, width):
            x0, y0, x1, y1 = b
            d.rectangle([x0, y0, x1, y1], outline=colour, width=width)
            d.rectangle(
                [ix + (x0 - cx0) * sx, iy + (y0 - cy0) * sy, ix + (x1 - cx0) * sx, iy + (y1 - cy0) * sy],
                outline=colour,
                width=width,
            )

        shown = distinct_boxes(boxes)
        if len(shown) > 1:
            for b in shown:
                outline(b, INSTANCE_EDGE, max(1, lw // 2))
        outline((ux0, uy0, ux1, uy1), (255, 32, 32), lw)
        # No chroma subsampling: at the default 4:2:0 a 2 px red outline blurs to a dull purple,
        # and the outline is exactly what the reviewer is reading.
        out.save(dest, quality=92, subsampling=0)
        return {"orig_w": W, "orig_h": H, "canvas_w": cw, "canvas_h": ch, "side": "right"}


def side_inset_to_original(box: list[float] | tuple[float, ...], geom: dict) -> list[float]:
    """Convert a box normalised to a side-inset CANVAS into one normalised to the PHOTO.

    The photo sits at the canvas origin, so this is a rescale by canvas/photo size.
    Anything drawn into the padding is clamped to the photo's edge.
    """
    sx = geom["canvas_w"] / geom["orig_w"]
    sy = geom["canvas_h"] / geom["orig_h"]
    x0, y0, x1, y1 = box
    return [min(1.0, max(0.0, v)) for v in (x0 * sx, y0 * sy, x1 * sx, y1 * sy)]
