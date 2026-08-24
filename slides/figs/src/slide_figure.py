"""Shared rules for figures that are going on a slide rather than in a report.

A report figure and a slide figure are not the same object, and the difference
is arithmetic, not taste. A figure `W` inches wide is drawn at `W x 72` points;
dropped into a slide slot `P` pixels wide it is displayed at `P / (W x 72)`
pixels per point. A 12.8in report figure in a 717px sidebar renders its 10pt
tick labels at **8 pixels** — beside 28px body copy on the same slide.

So every generator here works to two rules:

* **Size to the slot.** A `bg right:56%` box on a 1280x720 slide is 717x720 —
  very nearly square. A 2:1 figure fills half of it and wastes the rest, which
  is the same as choosing to draw everything at half size. Give a six-panel
  figure 3 rows x 2 cols, not 2 x 3.
* **Hold the type floor.** Nothing renders below `TYPE_FLOOR_PX`. `save()`
  checks it and raises, so a later edit that adds panels or shrinks a label
  fails the build instead of quietly producing another unreadable figure.

Both rules are about the *rendered* size, which is why neither can be checked
by looking at the PNG on its own.

A third rule applies to schematics rather than plots, and is about *spacing*
rather than size: `LABEL_GAP_PT` and `OBJECT_GAP_PT` below. See
`slides/STYLE.md` for what they mean and why the ratio between them matters.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

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

INK = "#14181f"
SOFT = "#5b6472"


def rendered_px_per_pt(fig: plt.Figure, column: float) -> float:
    """Slide pixels per printed point, once `fig` is fitted into its slot."""
    box_width, box_height = SLIDE_PX * column, SLIDE_PX * 9 / 16
    width_pt, height_pt = fig.get_size_inches() * 72
    return min(box_width / width_pt, box_height / height_pt)


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
) -> None:
    """Write `fig`, refusing it if any label would miss the type floor.

    `tight=False` keeps the figure's declared bounds instead of cropping to the
    ink. That matters for a full-bleed slide, where the slide's own headline is
    overlaid on the top of the image: the reserved whitespace it needs is empty
    by definition, and a tight bbox would helpfully remove it.

    `box` crops to an explicit box (inches) instead — pass `tight_box(final)`
    when saving the stages of a build figure, so every stage shares the final
    stage's framing.
    """
    enforce_type_floor(fig, column)
    if box is not None:
        fig.savefig(out / name, bbox_inches=box)
    elif tight:
        fig.savefig(out / name, bbox_inches="tight", pad_inches=0.14)
    else:
        fig.savefig(out / name)
    plt.close(fig)
    print(f"wrote figs/{name}")
