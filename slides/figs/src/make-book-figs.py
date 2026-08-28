#!/usr/bin/env python
"""The deck's two photograph figures: what "book" means, and one ranking of them.

    python slides/figs/src/make-book-figs.py

Writes `figs/book-boundary.png` (the sidebar figure on *A Concept in Your
Head*) and `figs/book-rank.png` with its build stages (the full-bleed figure
that opens Part 3).

Both exist because a talk about searching images that shows no images is asking
the room to take the hard part on trust (#3265). The hard part is not "find the
books" — it is that **you** know where the edge of the concept is and no
labelled set does: a magazine, a DVD case, a spiral notebook and a game manual
are all rectangular, printed, shelved objects, and whether each one is a "book"
is a decision, not a fact. That is not hypothetical here. Visual Genome's own
`book` annotation is the worst of the twelve classes VTSearch re-reviewed for
`vg_scale`: 3 of 20 sampled negatives actually held one, against zero for eight
of the other classes (`docs/experiments/vg-scale/DATASHEET.md`).

The photographs are **COCO val2017**, which carries `book` as an annotated
class. `coco_fixture.ensure_corpus` downloads it on demand into
`data/coco-val2017/`; nothing of it is committed, and what is committed is the
rendered figure, exactly as for the two UI screenshots. The roster below pins
the ten frames by COCO image id and crops them by a hand-measured square, so a
re-run reproduces the same figure rather than resampling the corpus.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coco_fixture import IMAGES, ensure_corpus  # noqa: E402
from slide_figure import (  # noqa: E402
    FULL_BLEED,
    SIDEBAR,
    save,
    tight_box,
)

OUT = Path(__file__).resolve().parent.parent

INK = "#14181f"
SOFT = "#5b6472"
RULE = "#d8dee6"
RED = "#b91c1c"
GREEN = "#0d8a5f"
BLUE = "#0b5fa5"

#: Every frame the two figures use, as `name: (coco file, crop)`. The
#: crop is `(left, top, side)` in fractions of the image's own *width*, square
#: by construction, measured by eye against the frame rather than derived from
#: the annotation box — a box round one spine on a shelf is not the picture the
#: slide wants, and a box round the union of fourteen of them is the whole
#: photograph.
TILES = {
    "stack": ("000000520077.jpg", (0.10, 0.05, 0.80)),
    "shelf": ("000000183049.jpg", (0.05, 0.10, 0.62)),
    "bookcase": ("000000509260.jpg", (0.42, 0.02, 0.45)),
    "open": ("000000262938.jpg", (0.45, 0.32, 0.50)),
    "bird": ("000000542776.jpg", (0.22, 0.05, 0.62)),
    "magazine": ("000000375278.jpg", (0.50, 0.42, 0.50)),
    "dvd": ("000000125062.jpg", (0.30, 0.00, 0.55)),
    "notebook": ("000000176446.jpg", (0.20, 0.02, 0.45)),
    "newspaper": ("000000016249.jpg", (0.09, 0.14, 0.42)),
    "gamecase": ("000000379842.jpg", (0.02, 0.00, 0.60)),
}

#: The sidebar figure: three the room will call books, three it will argue
#: about. Deliberately *not* three obvious rejects — a slide that answers its
#: own question teaches nothing, and the argument the deck needs is that the
#: second row is genuinely undecidable without being told whose corpus it is.
BOUNDARY_YES = ("stack", "shelf", "bookcase")
BOUNDARY_MAYBE = ("magazine", "dvd", "notebook")

#: The full-bleed figure: ten items in the order a detector put them, and what
#: they actually are. Mostly right, wrong in the middle — the arrangement every
#: cut rule in Part 3 is arguing about.
#:
#: **Lowest score first**, so the row reads left to right the way every other
#: score axis in the deck does and the way a chart does: positives and high
#: scores on the right (#3296). The admitted set is therefore the *tail* of
#: this tuple, which is what `_cut_x` converts a "include N" count into.
RANKING = (
    ("notebook", False),
    ("newspaper", False),
    ("dvd", False),
    ("bird", True),
    ("gamecase", False),
    ("open", True),
    ("magazine", False),
    ("bookcase", True),
    ("shelf", True),
    ("stack", True),
)

#: Where the three drawn cuts fall, as a count of items admitted. Each is a
#: defensible answer on this ranking and they disagree about six of the ten.
CUTS = (3, 5, 7)


def tile(name: str, size: int = 512) -> np.ndarray:
    """One roster frame, cropped square and resampled to `size`."""
    file, (left, top, side) = TILES[name]
    with Image.open(IMAGES / file) as image:
        frame = image.convert("RGB")
        width, height = frame.size
        span = min(side * width, height)
        x = max(0.0, min(width - span, left * width))
        y = max(0.0, min(height - span, top * width))
        box = (int(x), int(y), int(x + span), int(y + span))
        return np.asarray(frame.resize((size, size), Image.LANCZOS, box=box))


def _photo(ax: plt.Axes, name: str, x: float, y: float, side: float, *, edge: str = INK) -> None:
    """Draw one square frame with its bottom-left corner at `(x, y)`."""
    ax.imshow(tile(name), extent=(x, x + side, y, y + side), zorder=2, interpolation="antialiased")
    ax.add_patch(Rectangle((x, y), side, side, facecolor="none", edgecolor=edge, linewidth=1.6, zorder=3))


def _canvas(width: float, height: float, unit: float) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width * unit / 72, height * unit / 72))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return fig, ax


# ---------------------------------------------------------------------------
# The sidebar figure: the concept, and its edge.
# ---------------------------------------------------------------------------

#: Points per drawing unit for the boundary figure. Chosen so the two row
#: labels clear the type floor in a 56% sidebar: the drawing is 5.34 units
#: wide, so a 15pt label renders at 15 * 717 / (5.34 * PT) pixels.
BOUNDARY_UNIT = 72.0
BOUNDARY_TILE = 1.62
BOUNDARY_GAP = 0.09


def boundary_fig() -> None:
    """Three books over three things that are not books, or are, or it depends."""
    cols = len(BOUNDARY_YES)
    width = cols * BOUNDARY_TILE + (cols - 1) * BOUNDARY_GAP
    label_h = 0.52
    height = 2 * (BOUNDARY_TILE + label_h) + 0.28
    fig, ax = _canvas(width, height, BOUNDARY_UNIT)

    rows = (
        (BOUNDARY_YES, "Book", GREEN),
        (BOUNDARY_MAYBE, "Book?", RED),
    )
    y = height
    for names, label, colour in rows:
        y -= label_h
        ax.text(0.0, y + 0.10, label, ha="left", va="bottom", fontsize=20, color=colour, fontweight="bold")
        y -= BOUNDARY_TILE
        for i, name in enumerate(names):
            _photo(ax, name, i * (BOUNDARY_TILE + BOUNDARY_GAP), y, BOUNDARY_TILE)
        y -= 0.28

    save(fig, OUT, "book-boundary.png", column=SIDEBAR)


# ---------------------------------------------------------------------------
# The full-bleed figure: one ranking, and every cut anyone could defend on it.
# ---------------------------------------------------------------------------

#: A 16:9 canvas at 80 slide pixels per unit, so the title notch — 300x200
#: pixels at a 60x42 inset — is the rectangle x 0.75..4.50, y 5.98..8.48 in
#: these coordinates. `RANK_TOP` is that rectangle's floor: nothing that spans
#: the drawing may sit above it, which is what pins the score axis and, under
#: it, the tile row.
RANK_CANVAS = (16.0, 9.0)
RANK_UNIT = 80.0
RANK_TOP = 5.97

RANK_TILE = 1.44
RANK_GAP = 0.11
#: The score axis runs the *exact* width of the tile row and sits directly
#: above it. It used to be indented past the notch and stop short of the last tile,
#: which left a line whose length matched nothing on the slide and read as a
#: second, unrelated object (#3296). Ducking under the notch's floor instead
#: buys the full span for the price of 1.2 units of headroom nothing was using.
RANK_AXIS_Y = 5.80
RANK_AXIS_LABEL_Y = 6.06
RANK_TILE_Y = 4.05
RANK_MARK_Y = 3.60
RANK_CUT_LABEL_Y = 2.78

#: Big enough to read from the back of the room. The mark is the *only* thing
#: on the slide that says which tiles are books — the thumbnails read as
#: photographs and not as their contents at this size — so it is sized like a
#: headline rather than like an annotation (#3296).
RANK_MARK_PT = 56


assert RANK_TILE_Y + RANK_TILE < RANK_AXIS_Y < RANK_TOP, "the score axis no longer clears the tile row"


def _rank_x(index: int) -> float:
    """Left edge of the `index`-th tile, laid out centred on the canvas."""
    return _rank_left() + index * (RANK_TILE + RANK_GAP)


def _rank_left() -> float:
    """Left edge of the tile row, and so of the score axis above it."""
    return (RANK_CANVAS[0] - _rank_span()) / 2


def _rank_span() -> float:
    """Width of the whole tile row."""
    return len(RANKING) * RANK_TILE + (len(RANKING) - 1) * RANK_GAP


def _cut_x(admitted: int) -> float:
    """The gap that leaves the top `admitted` items to its right.

    The row runs low score first, so "include 7 of 10" is the boundary before
    the 4th tile, not after the 7th.
    """
    return _rank_x(len(RANKING) - admitted) - RANK_GAP / 2


def _errors(admitted: int) -> str:
    """The two error counts a cut makes, in the words the rest of the deck uses."""
    kept = RANKING[len(RANKING) - admitted :]
    dropped = RANKING[: len(RANKING) - admitted]
    false_pos = sum(1 for _, positive in kept if not positive)
    false_neg = sum(1 for _, positive in dropped if positive)
    return f"{false_pos} false positive{'' if false_pos == 1 else 's'}\n{false_neg} false negative{'' if false_neg == 1 else 's'}"


def rank_fig() -> None:
    """The ranking, what it got wrong, and three cuts that all look reasonable."""
    final = _rank_stage(3)
    box = tight_box(final)
    for stage in (1, 2):
        save(_rank_stage(stage), OUT, f"book-rank.build{stage}.png", column=FULL_BLEED, box=box)
    save(final, OUT, "book-rank.png", column=FULL_BLEED, box=box)


def _rank_stage(stage: int) -> plt.Figure:
    """Stage 1 the ranking, 2 adds the truth, 3 adds the cuts anyone could pick."""
    width, height = RANK_CANVAS
    fig, ax = _canvas(width, height, RANK_UNIT)

    # The score axis, the exact width of the row it describes and immediately
    # above it, so that it reads as that row's axis rather than as a second
    # object. It ducks under the title notch's floor rather than indenting past
    # its right edge (slides/STYLE.md's "move the one thing that reaches left",
    # applied vertically): the notch reserves a corner, not a band.
    left, right = _rank_left(), _rank_left() + _rank_span()
    ax.annotate(
        "",
        xy=(right, RANK_AXIS_Y),
        xytext=(left, RANK_AXIS_Y),
        arrowprops={"arrowstyle": "-|>,head_width=0.16,head_length=0.34", "color": SOFT, "linewidth": 1.6},
    )
    ax.text(
        (left + right) / 2,
        RANK_AXIS_LABEL_Y,
        "detector score, low to high",
        ha="center",
        va="bottom",
        fontsize=21,
        color=SOFT,
    )

    for index, (name, positive) in enumerate(RANKING):
        x = _rank_x(index)
        _photo(ax, name, x, RANK_TILE_Y, RANK_TILE)
        if stage >= 2:
            ax.text(
                x + RANK_TILE / 2,
                RANK_MARK_Y,
                "✓" if positive else "✗",
                ha="center",
                va="top",
                fontsize=RANK_MARK_PT,
                color=GREEN if positive else RED,
            )

    if stage >= 3:
        for admitted in CUTS:
            x = _cut_x(admitted)
            ax.plot(
                [x, x],
                [RANK_CUT_LABEL_Y + 0.12, RANK_TILE_Y + RANK_TILE + 0.22],
                color=BLUE,
                linewidth=2.2,
                linestyle=(0, (4, 3)),
                zorder=4,
            )
            ax.text(
                x,
                RANK_CUT_LABEL_Y,
                f"Include {admitted}",
                ha="center",
                va="top",
                fontsize=22,
                color=BLUE,
                fontweight="bold",
            )
            ax.text(
                x,
                RANK_CUT_LABEL_Y - 0.42,
                _errors(admitted),
                ha="center",
                va="top",
                fontsize=21,
                color=INK,
            )

    return fig


if __name__ == "__main__":
    ensure_corpus()
    boundary_fig()
    rank_fig()
    print("wrote figures to", OUT)
