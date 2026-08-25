"""Shared rules for figures that are going on a slide rather than in a report.

A report figure and a slide figure are not the same object, and the difference
is arithmetic, not taste. A figure `W` inches wide is drawn at `W x 72` points;
dropped into a slide slot `P` pixels wide it is displayed at `P / (W x 72)`
pixels per point. A 12.8in report figure in a 717px sidebar renders its 10pt
tick labels at **8 pixels** — beside 28px body copy on the same slide.

So every generator here works to two rules:

* **Size to the slot.** The slot is the whole 1280x720 slide: these decks are
  full-bleed (`_class: full` + `![bg fit]`), because a sidebar figure was
  already height-bound and the text column beside it cost the drawing nothing.
  A 1:1 figure fills 56% of a 16:9 box and wastes the rest, which is the same
  as choosing to draw everything at three-quarter size. Give a six-panel figure
  2 rows x 3 cols, not 3 x 2 — and note that widening a *height-bound* drawing
  buys no type at all; only making it shorter does.
* **Hold the type floor.** Nothing renders below `TYPE_FLOOR_PX`. `save()`
  checks it and raises, so a later edit that adds panels or shrinks a label
  fails the build instead of quietly producing another unreadable figure.

Both rules are about the *rendered* size, which is why neither can be checked
by looking at the PNG on its own.

A third rule applies to schematics rather than plots, and is about *spacing*
rather than size: `LABEL_GAP_PT` and `OBJECT_GAP_PT` below. See
`slides/STYLE.md` for what they mean and why the ratio between them matters.

A fourth applies only to full-bleed slides, and is about *where the title
goes*: `TITLE_NOTCH_PX`. A figure that owns the whole 1280x720 slide has to
leave its top-left corner empty for the slide's own kicker and headline. That
rectangle is a fixed standard rather than a per-slide choice, because a title
that moves is worse than no title at all — a figure that cannot spare its
top-left corner carries no title, it does not put one somewhere else.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#: Marp renders these decks at 1280x720.
SLIDE_PX = 1280
#: `![bg right:56% fit]`, the standard sidebar slot in slides/fragments/.
SIDEBAR = 0.56
#: `![bg right:70% fit]`, for a slide the figure is meant to carry.
SIDEBAR_WIDE = 0.70
#: `![bg fit]` on a `_class: full` slide.
FULL_BLEED = 1.0
#: Body copy on a slide is 28px and the page number — the smallest thing the
#: theme draws — is 20px. Nothing inside a figure has more right to be small
#: than the page number.
TYPE_FLOOR_PX = 20.0

#: Spacing standard for schematic figures, in printed points. A label must sit
#: *closer* to the thing it names than that thing sits to its neighbours, or
#: the eye binds it to whichever object happens to be nearest. `LABEL_GAP_PT`
#: is the gap from a label to what it labels; `OBJECT_GAP_PT` is the gap
#: between two distinct objects — a box and the arrow leaving it, an arrowhead
#: and what it points at. Only the ratio really matters, and it wants to be
#: large: at 1:1 (which is what "eyeball it" produces) the pairing is a
#: coin-flip. Convert to a figure's own drawing units by dividing by the
#: points-per-unit that figure is drawn at.
LABEL_GAP_PT = 6.0
OBJECT_GAP_PT = 16.0

#: The title notch, in slide pixels from the top-left of a 1280x720 slide:
#: `(x, y, width, height)`. A `_class: full` slide draws its kicker and
#: headline here and nowhere else, so a full-bleed figure must keep its ink
#: out of it. Narrow and tall on purpose, and nothing but the headline: a
#: full-bleed slide carries no kicker, and the headline is *meant* to wrap into
#: a tall block rather than run across the top.
#:
#: The width is what the geometry turns on. A schematic drawn symmetrically
#: about a spine puts its first row — the block, plus whatever labels hang off
#: its left edge — at a fixed fraction of its own width, near 0.29. A notch
#: wider than that is unclearable at any aspect a slide can show: widening the
#: drawing moves the block and the notch together. At 300px the reserve ends at
#: 0.28 of a 16:9 drawing's width and the symmetric flows clear it, which is
#: the difference between those slides having a headline and not.
#:
#: The height is measured, not chosen. It used to be 250px, sized against the
#: sentence-length headlines the deck carried before #3242 retitled the figure
#: slides to short recognizable phrases. Every `_class: full` headline in the
#: deck now renders at or under 191.2px in this column ("Simulated Voters,
#: Thousands of Runs", the longest, at four lines), so 250 was 59px of reserve
#: nothing ever used — and every one of those pixels was taken out of a figure.
#: Re-measure before changing it, and re-measure if a full-bleed headline grows
#: past two lines: `slides/STYLE.md` records the recipe (render the deck to
#: HTML and read every `section.full h2` box in a browser). This number is a
#: measurement of the deck, so it is only true of the deck it was measured on.
#:
#: What still cannot clear it is a figure whose *top row spans the drawing* —
#: a score axis or a scatter that starts in the top-left corner by
#: construction. Shrinking the height does not help those: the blocker is
#: horizontal, so the drawing has to start right of the notch instead. The
#: Part 2 panel figures do exactly that. What the height buys is the figures
#: whose ink merely grazed the reserve's lower half.
TITLE_NOTCH_PX = (60.0, 42.0, 300.0, 200.0)

#: A figure may pass its *own* rectangle to `save(notch=...)`, and exactly one
#: does. The x, y and width are not negotiable — that is the standard, and a
#: headline that moved would stop being a headline — but the *height* is only
#: ever a reserve for the deck's **longest** headline, and a figure whose slide
#: carries a one-line one is holding back 100px it will never use. On a
#: schematic that costs nothing (the drawing does not reach up there anyway);
#: on `vote-boundary`, whose whole first page is a field of items filling the
#: slide, it left a band under the title with no title in it and no items
#: either, which reads as a mistake rather than as a margin (#3254).
#:
#: The height is measured the same way `TITLE_NOTCH_PX`'s is, on the slide the
#: figure actually appears on, and a figure that trims it owns that
#: measurement: re-take it if the headline changes. See `slides/STYLE.md`.

INK = "#14181f"
SOFT = "#5b6472"


def rendered_px_per_pt(fig: plt.Figure, column: float) -> float:
    """Slide pixels per printed point, once `fig` is fitted into its slot."""
    box_width, box_height = SLIDE_PX * column, SLIDE_PX * 9 / 16
    width_pt, height_pt = fig.get_size_inches() * 72
    return min(box_width / width_pt, box_height / height_pt)


def _slot_placement(width: float, height: float, column: float) -> tuple[float, float, float, float]:
    """Where a `width` x `height` drawing lands inside its slide slot.

    Returns `(offset_x, offset_y, drawn_w, drawn_h)` in slide pixels. Marp's
    `fit` scales the image to touch the slot on its binding axis and centres it
    on the other, so a drawing whose aspect differs from the slot's is
    letterboxed — and the empty bands that leaves are why a tall figure often
    owes the title notch nothing at all.
    """
    box_w, box_h = SLIDE_PX * column, SLIDE_PX * 9 / 16
    scale = min(box_w / width, box_h / height)
    drawn_w, drawn_h = scale * width, scale * height
    return (box_w - drawn_w) / 2, (box_h - drawn_h) / 2, drawn_w, drawn_h


def notch_box(
    width: float,
    height: float,
    column: float = FULL_BLEED,
    notch: tuple[float, float, float, float] | None = None,
) -> tuple[float, float, float, float] | None:
    """The title notch as a fraction of a `width` x `height` drawing, or None.

    The result is `(x0, y0, x1, y1)` in matplotlib's figure-fraction
    convention (origin bottom-left), ready to hand to `transFigure`, so a
    generator can lay out around the notch rather than guess at it.

    `None` means the notch falls entirely in the letterbox margin beside the
    drawing — the corner the title wants is empty *slide*, not empty *figure*,
    and the generator has nothing to do. That is the common case for the
    tall flow figures, and it is why going full-bleed costs them no redraw.

    `notch` overrides `TITLE_NOTCH_PX` for a figure that has measured its own
    slide's headline; see the constant's note for when that is legitimate.

    Only meaningful for `FULL_BLEED`: the notch is a full-bleed standard, and
    a split-background slot is not anchored at the slide's left edge anyway.
    """
    offset_x, offset_y, drawn_w, drawn_h = _slot_placement(width, height, column)
    notch_x, notch_y, notch_w, notch_h = notch or TITLE_NOTCH_PX
    left = (notch_x - offset_x) / drawn_w
    right = (notch_x + notch_w - offset_x) / drawn_w
    # Notch y runs down from the slide's top; figure fractions run up.
    top = (notch_y - offset_y) / drawn_h
    bottom = (notch_y + notch_h - offset_y) / drawn_h

    x0, x1 = max(0.0, left), min(1.0, right)
    y0, y1 = max(0.0, 1.0 - bottom), min(1.0, 1.0 - top)
    if x0 >= x1 or y0 >= y1:
        return None
    return x0, y0, x1, y1


def enforce_title_notch(
    path: Path,
    column: float = FULL_BLEED,
    notch: tuple[float, float, float, float] | None = None,
) -> None:
    """Raise if the written figure puts ink where the slide's title goes.

    Checked against the *file* rather than the live figure because `save`
    crops, so the drawing that reaches the slide is not the one the canvas
    holds. Background is read from the image's own top-left pixel, which the
    tight crop's padding guarantees is blank.
    """
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGBA"))
    height, width = pixels.shape[:2]
    box = notch_box(width, height, column, notch)
    if box is None:
        return
    x0, y0, x1, y1 = box
    top, bottom = int((1.0 - y1) * height), math.ceil((1.0 - y0) * height)
    left, right = int(x0 * width), math.ceil(x1 * width)
    region = pixels[top:bottom, left:right]
    background = pixels[0, 0]
    # Transparent counts as blank whatever its colour channels happen to hold.
    ink = (region[..., 3] > 8) & (np.abs(region.astype(int) - background.astype(int))[..., :3] > 8).any(-1)
    if ink.any():
        covered = 100.0 * ink.mean()
        raise SystemExit(
            f"title notch: {path.name} draws in the top-left corner the slide reserves for its "
            f"headline ({covered:.1f}% of the notch has ink). Move that ink out of the notch — "
            f"`slide_figure.notch_box()` gives the rectangle in figure fractions — or drop the "
            f"slide's title, which is the other honest answer. Do not move the title."
        )


def enforce_type_floor(fig: plt.Figure, column: float = SIDEBAR) -> None:
    """Raise unless every label in `fig` clears the slide type floor."""
    px_per_pt = rendered_px_per_pt(fig, column)
    smallest = min(
        (t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip()),
        default=float("inf"),
    )
    if smallest == float("inf"):
        # A figure with no labels at all cannot miss the floor. This is a real
        # case, not a degenerate one: the opening stage of a build often draws
        # the bare situation and lets the first advance name it.
        return
    rendered = smallest * px_per_pt
    if rendered < TYPE_FLOOR_PX:
        raise SystemExit(
            f"type floor: smallest text is {smallest:g}pt, which renders at {rendered:.1f}px "
            f"in a {column:.0%} slide slot — below the {TYPE_FLOOR_PX:g}px floor. "
            f"Raise the font size, or shrink the figure so it is scaled up less."
        )


def tight_box(fig: plt.Figure) -> matplotlib.transforms.Bbox:
    """The crop box (in inches) that `save`'s default tight crop would use.

    A figure drawn in stages for a slide build must save every stage with the
    *final* stage's box, not its own: cropping each stage to its own ink would
    reframe the drawing between build slides, and the mechanism would jump
    around the slot instead of assembling in place.
    """
    fig.canvas.draw()
    return fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.14)


def save(
    fig: plt.Figure,
    out: Path,
    name: str,
    column: float = SIDEBAR,
    *,
    tight: bool = True,
    box: matplotlib.transforms.Bbox | None = None,
    notch: bool | tuple[float, float, float, float] = True,
) -> None:
    """Write `fig`, refusing it if any label would miss the type floor.

    `tight=False` keeps the figure's declared bounds instead of cropping to the
    ink. That matters for a full-bleed slide, where the slide's own headline is
    overlaid on the top of the image: the reserved whitespace it needs is empty
    by definition, and a tight bbox would helpfully remove it.

    `box` crops to an explicit box (inches) instead — pass `tight_box(final)`
    when saving the stages of a build figure, so every stage shares the final
    stage's framing.

    A `FULL_BLEED` figure is additionally checked against `TITLE_NOTCH_PX`,
    the top-left corner its slide reserves for kicker and headline. Pass
    `notch=False` for a full-bleed slide that carries no title — a legitimate
    choice, and the only one available to a figure that needs its own corner —
    or a rectangle of its own for a slide whose headline has been *measured*
    shorter than the deck's longest (see `TITLE_NOTCH_PX`). The check is a
    no-op at any other `column`, where the title sits beside the figure rather
    than over it.
    """
    enforce_type_floor(fig, column)
    if box is not None:
        fig.savefig(out / name, bbox_inches=box)
    elif tight:
        fig.savefig(out / name, bbox_inches="tight", pad_inches=0.14)
    else:
        fig.savefig(out / name)
    plt.close(fig)
    if notch and column == FULL_BLEED:
        enforce_title_notch(out / name, column, None if notch is True else notch)
    print(f"wrote figs/{name}")
