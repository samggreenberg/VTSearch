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
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

#: Marp renders these decks at 1280x720.
SLIDE_PX = 1280
#: `![bg right:56% fit]`, the standard sidebar slot in slides/fragments/.
SIDEBAR = 0.56
#: `![bg fit]` on a `_class: full` slide.
FULL_BLEED = 1.0
#: Body copy on a slide is 28px and the page number — the smallest thing the
#: theme draws — is 20px. Nothing inside a figure has more right to be small
#: than the page number.
TYPE_FLOOR_PX = 20.0

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
        default=0.0,
    )
    rendered = smallest * px_per_pt
    if rendered < TYPE_FLOOR_PX:
        raise SystemExit(
            f"type floor: smallest text is {smallest:g}pt, which renders at {rendered:.1f}px "
            f"in a {column:.0%} slide slot — below the {TYPE_FLOOR_PX:g}px floor. "
            f"Raise the font size, or shrink the figure so it is scaled up less."
        )


def save(fig: plt.Figure, out: Path, name: str, column: float = SIDEBAR) -> None:
    enforce_type_floor(fig, column)
    fig.savefig(out / name, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    print(f"wrote figs/{name}")
