#!/usr/bin/env python
"""One slide: eight results for *Coke logo*, arriving one at a time.

    python slides/figs/src/make-logo-figs.py

Writes `figs/logo-grid.webp` and its six build stages — the slide is a build, so
the room is asked about each result as it lands rather than being handed all
eight at once. The whole argument lives in the presenter notes
(`fragments/logo-grid.md`): *are these the same? …and this one? …what if the
colours invert? …this can't count, right?* Every answer is defensible and no
two people give the same set of them, which is the point.

There is nothing to compute here. The figure is a **compositor**: it lays the
committed thumbnails in `logo-src/` onto a fixed 3x3 grid and saves one stage
per reveal, with later cells simply empty. That fixed grid is the whole reason
this is a generated figure rather than eight `<img>` tags in the fragment — a
build marker reveals by *truncating* the fragment, so an HTML grid would
reflow on every page and the images would shuffle around the slide instead of
arriving in place. `slides/STYLE.md` is explicit that a reveal adds ink and
does nothing else.

The top-left cell is left empty for the slide's headline, which is what buys
this figure a title: eight tiles in a 4x2 grid would fill the corner
`slide_figure.TITLE_NOTCH_PX` reserves, and a full-bleed figure that cannot
spare that corner has to carry no title at all.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slide_figure import (  # noqa: E402
    FULL_BLEED,
    save,
)

OUT = Path(__file__).resolve().parent.parent
SRC = Path(__file__).resolve().parent / "logo-src"

INK = "#14181f"
SOFT = "#5b6472"
RULE = "#d8dee6"

#: WebP, not PNG, and this is the one figure in the deck that has to be.
#: Every other generated figure here is line art on white, which is what PNG is
#: for; these tiles are eight JPEG *photographs* — gradients, drop shadows and
#: compression noise, upscaled — which is the case `slides/README.md` reserves
#: WebP for, and the seven cumulative stages each carry every earlier tile
#: again. As PNG the group weighs 5.0 MB against a stated budget of about
#: 150 KB per figure. Marp rasterises through Chromium, which reads WebP
#: natively, and the deck already ships its two UI screenshots this way.
FIGURE_FORMAT = "webp"

#: Points per drawing unit, shared with the calibration schematics so type set
#: here is the size it is there. See `make-calib-figs.FLOW_UNIT_PT`.
UNIT_PT = 38.0

#: Exactly 16:9, and every stage is written with `tight=False`, so the canvas
#: maps one-to-one onto the 1280x720 slide and the grid geometry below is in
#: slide pixels divided by 64.65.
CANVAS = (19.8, 11.0)

COLS, ROWS = 3, 3

#: How much of the canvas's bottom edge the grid keeps clear. The slide draws
#: its own page number in the bottom-right corner, and the last row of tiles
#: otherwise runs underneath it.
GRID_FOOT = 0.75

CELL_W, CELL_H = CANVAS[0] / COLS, (CANVAS[1] - GRID_FOOT) / ROWS

#: How much of each cell is margin rather than image. Wide enough that two
#: tiles never touch — several of these thumbnails are white-on-white at the
#: edges and would otherwise read as one wide picture.
CELL_PAD = 0.42

#: The eight results, in the order the slide reveals them, as
#: `(file, what it is)`. The file is looked up in `logo-src/`; a slot whose
#: file is not there yet draws a dashed placeholder carrying its description,
#: so the deck builds and the layout can be reviewed before every asset has
#: arrived. **A placeholder is not a figure** — re-run this script once the
#: file lands and commit the result.
#:
#: The order is the argument's, not the search engine's: it starts with two
#: nobody argues about, walks out through the ones that are still obviously
#: the mark, and ends on three that each break a *different* attribute — the
#: colour, the typeface, the product. See the fragment's notes.
RESULTS = (
    ("01-wordmark-on-red.jpg", "wordmark, white on a solid red field"),
    ("02-red-disc.jpg", "the round red badge"),
    ("03-disc-with-bottle.jpg", "that badge, with a contour bottle"),
    ("04-wordmark-ribbon.png", "wordmark over the dynamic ribbon"),
    ("05-script-red-on-white.jpg", "the script, red on white"),
    ("06-script-black.jpg", "the script, in flat black"),
    ("07-coke-sans.png", "“Coke”, in a heavy sans"),
    ("08-diet-coke.jpg", "Diet Coke"),
)

#: How wide a placeholder's description may run before it wraps, in
#: characters. Measured against the cell rather than guessed: a placeholder
#: that overflows its own box is the one thing it must not do, since its whole
#: job is to show what the finished layout will look like.
PLACEHOLDER_CHARS = 24

#: Which grid cell each result lands in, as `(col, row)` with row 0 at the top.
#: The top-left cell is skipped: that is where the headline goes.
CELLS = ((1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2))

#: How many pages the slide is. The first shows **two** results, because the
#: opening question is a comparison — "are these the same?" needs two things to
#: be the same as each other — and every later page adds one.
STAGES = len(RESULTS) - 1

plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans"],
        "font.size": 15,
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 200,
    }
)


def _cell_box(index: int) -> tuple[float, float, float, float]:
    """Result *index*'s drawable rectangle as `(x0, y0, x1, y1)`, padded."""
    col, row = CELLS[index]
    x0 = col * CELL_W + CELL_PAD
    x1 = (col + 1) * CELL_W - CELL_PAD
    y1 = CANVAS[1] - row * CELL_H - CELL_PAD
    y0 = CANVAS[1] - (row + 1) * CELL_H + CELL_PAD
    return x0, y0, x1, y1


#: How far a pixel may sit from the corner colour and still count as border.
#: These are JPEG thumbnails, so a "white" margin is white plus ringing, and an
#: exact-match trim finds nothing at all on half of them.
TRIM_TOLERANCE = 12


def _trimmed(image: Image.Image) -> Image.Image:
    """`image` with its uniform border cropped off.

    The sources are search-result thumbnails, which are padded to a squarish
    box: several are a wordmark occupying a third of their own height with
    white above and below. Fitted untrimmed, that padding is what touches the
    cell and the art is drawn at a third of the size the slide is paying for.
    Trimming is what makes the eight tiles comparable to *each other*, too —
    otherwise a tile's apparent size records how much whitespace its thumbnail
    happened to carry.

    Returns the image unchanged when the trim finds nothing (art that already
    bleeds to all four edges) or everything (a solid tile, which cannot
    happen here but would otherwise crop to nothing).
    """
    border = Image.new("RGB", image.size, image.getpixel((0, 0)))
    difference = ImageChops.difference(image, border).convert("L")
    box = difference.point(lambda v: 255 if v > TRIM_TOLERANCE else 0).getbbox()
    return image if box is None else image.crop(box)


def _draw(ax: plt.Axes, index: int) -> None:
    """Draw result *index* in its cell, fitted and centred, or a placeholder."""
    x0, y0, x1, y1 = _cell_box(index)
    name, described = RESULTS[index]
    path = SRC / name
    if not path.exists():
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor="none",
                edgecolor=SOFT,
                linewidth=1.4,
                linestyle=(0, (4, 4)),
            )
        )
        ax.text(
            (x0 + x1) / 2,
            (y0 + y1) / 2,
            "awaiting\n" + "\n".join(textwrap.wrap(described, PLACEHOLDER_CHARS)),
            ha="center",
            va="center",
            fontsize=15,
            color=SOFT,
            linespacing=1.3,
        )
        return

    with Image.open(path) as image:
        pixels = _trimmed(image.convert("RGB"))
        width, height = pixels.size
        # `fit` semantics, done here rather than left to imshow: scale to touch
        # the cell on its binding axis and centre on the other, so a wide
        # wordmark and a square badge are as large as their cell allows and
        # neither is stretched.
        scale = min((x1 - x0) / width, (y1 - y0) / height)
        half_w, half_h = width * scale / 2, height * scale / 2
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ax.imshow(
            pixels,
            extent=(cx - half_w, cx + half_w, cy - half_h, cy + half_h),
            aspect="auto",
            interpolation="lanczos",
            zorder=2,
        )
        # A hairline frame, because several of these are black or red art on a
        # white ground and the slide is white: without it the tiles have no
        # edges and eight results read as one wide smear.
        ax.add_patch(
            Rectangle(
                (cx - half_w, cy - half_h),
                2 * half_w,
                2 * half_h,
                facecolor="none",
                edgecolor=RULE,
                linewidth=1.2,
                zorder=3,
            )
        )


def _stage(stage: int) -> plt.Figure:
    """The first *stage* reveals (1-based, cumulative): two results, then one more each."""
    fig, ax = plt.subplots(figsize=tuple(c * UNIT_PT / 72 for c in CANVAS))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, CANVAS[0])
    ax.set_ylim(0, CANVAS[1])
    ax.set_axis_off()
    for index in range(stage + 1):
        _draw(ax, index)
    return fig


def main() -> None:
    for stage in range(1, STAGES):
        save(_stage(stage), OUT, f"logo-grid.build{stage}.{FIGURE_FORMAT}", column=FULL_BLEED, tight=False)
    save(_stage(STAGES), OUT, f"logo-grid.{FIGURE_FORMAT}", column=FULL_BLEED, tight=False)


if __name__ == "__main__":
    main()
