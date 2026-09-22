"""Render a slate image with its box and a magnified view of the box's contents.

Two framings share one crop:

* :func:`inset_crop` -- the magnified, context-padded crop of the box.
  :func:`draw_with_inset` pastes it *over* a bottom corner of the photo.
  Fine for "is this box one?", where the reviewer looks at the box.
* :func:`draw_with_side_inset` derives its own crop (:func:`side_crop_rect`) and places it
  on black padding *beside* the photo, so no part of the scene is covered. Needed for any
  question about *other* instances in the frame -- a prominence triage asks whether a more
  prominent instance exists, and the corner inset can hide exactly that instance.

Both magnify with **one scale for both axes**, and neither draws anything over the
magnified view: the photo says WHERE the box is, the magnified view says WHAT IS UNDER
IT, and an outline across a magnified small object hides the pixels the question is about.
Their surrounds are grey (:data:`PANEL_EDGE`) for the same reason -- a red one reads as
the box. The corner inset acquired all three only in #3961; until then it capped width
and height to the same target independently, squashing every non-square crop.

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

#: How big the inset may get, as a fraction of the image's shorter side. The inset fits
#: inside a square this wide; its LONGER side is what touches the edge, so a 3:1 crop is
#: drawn 3:1 rather than squared off.
INSET_FRAC = 0.42
#: Outline of a magnified view itself, in either framing -- deliberately NOT red, so it cannot
#: be read as the box. The corner inset used red until #3961, around the padded *context*.
PANEL_EDGE = (110, 110, 110)


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

    # ONE scale for both axes, so the crop keeps its shape. This used to be
    # ``zoom = max(MIN_ZOOM, fit)`` followed by capping width and height to *target*
    # independently, which resized any crop wider or taller than the target into a
    # square (#3961). That was not the rare case it read as: the padding floor above
    # keeps the crop at roughly 0.2 * min(W, H) or more per side, so the 3x floor
    # essentially always won and essentially every inset came out target-by-target.
    # Fitting the longer side reproduces what the old expression did on the crops
    # small enough to escape the cap, and is honest about the rest.
    target = int(min(W, H) * INSET_FRAC)
    zoom = target / max(crop.width, crop.height)
    iw, ih = max(1, round(crop.width * zoom)), max(1, round(crop.height * zoom))
    crop = crop.resize((iw, ih), Image.LANCZOS)
    lw = max(2, int(min(W, H) * 0.006))
    return crop, (x0, y0, x1, y1), lw


def draw_with_inset(src: Path, box: tuple[float, float, float, float], dest: Path) -> tuple[int, int]:
    """Write *src* with *box* outlined and a magnified inset of its contents over a bottom corner.

    The crop is :func:`inset_crop`. :func:`draw_with_side_inset` is the other framing, which
    pays canvas geometry to leave the whole photo visible. The canvas is untouched here, so a
    box drawn on this render is already in photo coordinates and needs no conversion.

    **The box is drawn on the photo only, and the inset's surround is grey.** The photo says
    WHERE the box is; the inset says WHAT IS UNDER IT, so nothing is drawn across the
    magnified object. A red surround here framed the padded *context* rather than the box,
    and read as the box itself -- which, with the squashed crop that :func:`inset_crop` used
    to produce, is how a reviewer could not tell whether a speck near the edge was inside
    it (#3961). Both were fixed on the side panel first (#3960).
    """
    from PIL import Image, ImageDraw  # noqa: PLC0415

    with Image.open(src) as im:
        im = im.convert("RGB")
        W, H = im.size
        crop, (x0, y0, x1, y1), lw = inset_crop(im, box)

        out = im.copy()
        d = ImageDraw.Draw(out)
        d.rectangle([x0, y0, x1, y1], outline=(255, 32, 32), width=lw)

        # Inset in whichever bottom corner is furthest from the box, so the
        # magnifier never covers the thing it is magnifying.
        ix = 0 if (x0 + x1) / 2 > W / 2 else W - crop.width
        iy = H - crop.height
        out.paste(crop, (ix, iy))
        d.rectangle([ix, iy, ix + crop.width - 1, iy + crop.height - 1], outline=PANEL_EDGE, width=lw)
        # No chroma subsampling: at the default 4:2:0 a 2 px red outline blurs to a dull
        # purple, and the outline is exactly what the reviewer is reading.
        out.save(dest, quality=92, subsampling=0)
        return out.size


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


def side_crop_rect(W: int, H: int, union: Sequence[float]) -> tuple[int, int, int, int]:
    """The photo rectangle the side panel magnifies: *union* in pixels, with context padding.

    Shared with :func:`canvas_box_to_original`, which has to know exactly what the panel shows to
    convert a box drawn ON the panel. Deriving it twice is how a converter drifts from its renderer.
    """
    ux0, uy0, ux1, uy1 = union
    bw, bh = max(1.0, ux1 - ux0), max(1.0, uy1 - uy0)
    pad = max(max(bw, bh) * 0.6, min(W, H) * 0.10)
    cx0, cy0 = max(0, int(ux0 - pad)), max(0, int(uy0 - pad))
    cx1, cy1 = min(W, int(ux1 + pad) + 1), min(H, int(uy1 + pad) + 1)
    return cx0, cy0, max(cx1, cx0 + 2), max(cy1, cy0 + 2)


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

    **The panel is a faithful view, not the corner inset moved sideways.** It derives its own
    crop (:func:`side_crop_rect`), covering every annotation rather than one box, and scales it to
    the photo's height rather than to a fraction of it. What it shares with the fixed corner inset
    is the shape of the thing: one scale for both axes, a grey edge, and the context padding, so a
    small object still comes with its surroundings. (Both of those were the panel's alone until
    #3961; the corner inset squashed non-square crops and framed them in red.)

    **Nothing is drawn over the panel.** The photo carries the outlines, so the reviewer reads
    *where* the box is there, and the panel shows *what is under it* -- an outline drawn across a
    magnified small object hides the very pixels the question is about. Both framings before this
    one drew it in the panel too; the reviewer asked for the earlier, unobstructed view back.

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
        cx0, cy0, cx1, cy1 = side_crop_rect(W, H, (ux0, uy0, ux1, uy1))
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

        def outline(b, colour, width):
            # On the photo only -- the panel stays clear so the object under the box is visible.
            d.rectangle([b[0], b[1], b[2], b[3]], outline=colour, width=width)

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


def canvas_box_to_original(
    box: Sequence[float], geom: dict, boxes_shown: Sequence[Sequence[float]]
) -> tuple[list[float], str]:
    """``(box in PHOTO coordinates, where it was drawn)`` for a box drawn on a side-inset render.

    A reviewer may draw on either half, and for a small object the magnified panel is the natural
    place -- it is bigger and clearer. Both halves convert exactly:

    * **photo** -- a rescale, :func:`side_inset_to_original`.
    * **panel** -- the panel shows :func:`side_crop_rect` of the photo, scaled to fit, so a box on it
      maps back through that rectangle. *boxes_shown* (normalised, as recorded in the render
      manifest) is what the panel framed.
    * **across both** -- a photo box dragged past the photo's right edge into the padding, which is
      what a reviewer boxing a frame-filling object does. Clipped at the edge, as before.
    """
    W, H = geom["orig_w"], geom["orig_h"]
    cw, ch = geom["canvas_w"], geom["canvas_h"]
    frac = W / cw
    x0, x1 = sorted((float(box[0]), float(box[2])))
    y0, y1 = sorted((float(box[1]), float(box[3])))
    if x0 < frac - 1e-9:
        return side_inset_to_original([x0, y0, x1, y1], geom), ("photo" if x1 <= frac + 1e-9 else "across both")

    px = [_norm_to_px(b, W, H) for b in boxes_shown]
    union = (min(b[0] for b in px), min(b[1] for b in px), max(b[2] for b in px), max(b[3] for b in px))
    cx0, cy0, cx1, cy1 = side_crop_rect(W, H, union)
    crw, crh = cx1 - cx0, cy1 - cy0
    lw = max(2, int(min(W, H) * 0.006))
    gap = max(4, 2 * lw)
    pw, ph = cw - W - 2 * gap, ch - 2 * gap if ch - 2 * gap < H else min(H - 2 * gap, ch)
    ph = max(1, round(crh * (pw / crw)))
    ix, iy = W + gap, (ch - ph) // 2
    out = []
    for cxn, cyn in ((x0, y0), (x1, y1)):
        sx = (cxn * cw - ix) / max(1, pw)
        sy = (cyn * ch - iy) / max(1, ph)
        out += [(cx0 + sx * crw) / W, (cy0 + sy * crh) / H]
    return [min(1.0, max(0.0, v)) for v in (out[0], out[1], out[2], out[3])], "panel"


def _norm_to_px(b: Sequence[float], W: int, H: int) -> tuple[float, float, float, float]:
    x0, x1 = sorted((b[0] * W, b[2] * W))
    y0, y1 = sorted((b[1] * H, b[3] * H))
    return (
        max(0.0, min(x0, W - 1.0)),
        max(0.0, min(y0, H - 1.0)),
        max(1.0, min(x1, float(W))),
        max(1.0, min(y1, float(H))),
    )
