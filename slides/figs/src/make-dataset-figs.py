#!/usr/bin/env python
"""Dataset figures for the VTSearch decks.

Run from the repo root:

    python slides/figs/src/make-dataset-figs.py            # everything it can draw offline
    python slides/figs/src/make-dataset-figs.py --no-media # skip the cards that need pixels

Three kinds of figure, and the differences matter more than they look.

**Story figures** say how a dataset is *made* — the sources it starts from,
what is done to them, and what a reader may conclude. `coco_quarry` and DocMarks
get one each, drawn as a five-step stack with build markers so the room watches
the construction assemble rather than reading a paragraph. The steps are the
construction the build performs, not the order the decisions were reached in.

**Cards** say what a dataset *is*: how much of it there is, where to download
it, and — the part a drawing cannot fake — what the media look like. Every
card's strip is real pixels, fetched by `dataset_samples.py`.

**Argument figures** say why a dataset is *shaped* the way it is. There is one,
`coco_quarry`'s complement build, and it draws set theory rather than data: it
carries no counts at all, because the thing it is arguing about — that an
exhaustively annotated corpus can name the images a class is *absent* from —
is true of three classes and eighty alike. A dataset gets one of these only
where the design decision is the interesting part and a number would not say
it.

**Where the numbers come from.** Nothing here is typed in twice.

* The `coco_quarry` figures import `scripts/experiments/pile/pile_config.py`,
  so a slide cannot drift from the constants the pile actually builds against:
  a roster ruling that merges or adds a class moves the class count, the cell
  count and the class list here with no edit.
* The DocMarks figures read `scripts/experiments/docmarks/docmarks_config.py`
  for the tiers and the corpus version, and parse the committed `DATASHEET.md`
  for the per-source and per-class counts. The corpus itself lives on the GRID;
  a container cannot open it, and a figure that could only be built on the
  cluster would be a figure that silently stopped being rebuilt. Parsing the
  datasheet is the weaker guarantee of the two and is chosen for exactly that
  reason — it is the strongest one available from a checkout, and it fails
  loudly rather than going stale in silence.
* The COCO card counts its own annotation file.

**What the DocMarks slides still owe.** The one thing neither source carries is
*photographs of the marks*. Drawing them needs the corpus, so it is GRID work;
until then the DocMarks slides show structure and counts and say so.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import math
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slide_figure import FULL_BLEED, INK, SOFT, TITLE_NOTCH_PX, save  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "slides" / "figs"

#: 16:9 at a size where the type floor lands where we want it. A full-bleed
#: figure `W` inches wide renders at 1280/(W*72) px per point, so at 12.8in one
#: point is 1.389px and the 20px floor bites at 14.4pt. Everything here is 15pt
#: or more, with headroom rather than exactly at the line.
FIG_W, FIG_H = 12.8, 7.2
FLOOR_PT = 15

#: Characters per line in a story figure's step body. Five steps fit the
#: slide at two lines each and do not fit at three, so this number and the
#: length of the prose below are one decision; `_steps` refuses the figure
#: rather than letting the last step slide off the bottom.
STEP_WRAP = 66

#: The notch, as axes fractions of the 1280x720 slot, so a layout can be
#: written against it instead of against pixel arithmetic repeated in every
#: figure. `NOTCH_R` is its right edge and `NOTCH_B` its bottom, measured from
#: the top — so ink is safe at `x > NOTCH_R` at any height, or below `NOTCH_B`
#: at any x.
_nx, _ny, _nw, _nh = TITLE_NOTCH_PX
NOTCH_R = (_nx + _nw) / 1280.0
NOTCH_B = (_ny + _nh) / 720.0

CUT = "#2b6cb0"  # the deck's .cut blue
NEG = "#c0392b"  # .neg red
POS = "#2e7d51"  # .pos green
RULE = "#d5d9df"

#: The two columns every figure in this file is laid out on. The left one
#: starts under the title notch and carries the *conclusion* — counts, the
#: download line; the right one starts clear of the notch and carries the
#: *subject* — the steps, the photographs. Keeping them the same two columns
#: across nine figures is what lets a room stop re-learning the layout on every
#: slide of the appendix.
LEFT_X = 0.047
LEFT_TOP = 1.0 - NOTCH_B - 0.045
RIGHT_X = NOTCH_R + 0.045
RIGHT_W = 0.975 - RIGHT_X


def _save_photo(fig: plt.Figure, name: str) -> None:
    """Write a photograph-carrying card as WebP, through `save`'s own checks.

    `slides/README.md`'s rule: PNG for plots, WebP for photographs. These three
    cards are twelve web photographs each and cost 500-780 KB as PNG against
    about a sixth of that as WebP, at a quality no projector resolves — and a
    figure that is re-rendered whenever its dataset moves pays that repeatedly.

    Routed through `save` rather than around it so the type floor and the title
    notch are still checked, on the same pixels; matplotlib cannot write WebP,
    so the PNG is written, converted and removed.
    """
    from PIL import Image

    png = f"{name}.png"
    # `save` announces the file it writes, and this one is deleted three lines
    # later; printing it would put a name in the log that is not on disk.
    with contextlib.redirect_stdout(io.StringIO()):
        save(fig, OUT, png, column=FULL_BLEED, tight=False)
    with Image.open(OUT / png) as image:
        image.convert("RGB").save(OUT / f"{name}.webp", quality=88, method=6)
    (OUT / png).unlink()
    print(f"wrote figs/{name}.webp")


def count(n: int) -> str:
    """A count as a room would hear it said, not as the file reports it.

    `2,516,939` on a slide is seven digits nobody can hold and six of them
    nobody needs; what the number is doing there is saying *millions, a couple
    of them*. So: a suffix above ten thousand, at most three significant
    figures, and nothing under a thousand touched — small counts are usually
    exact-by-design (`23` classes, `40` categories) and rounding one would only
    make it disagree with the datasheet.

    >>> count(2_516_939), count(108_077), count(2_778), count(721)
    ('2.5M', '108K', '2,800', '721')

    The exact figures are not lost, they move to the presenter notes, which is
    where a number somebody might check belongs anyway.
    """
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}".rstrip("0").rstrip(".") + "M"
    if n >= 10_000:
        return f"{round(n / 1_000):,.0f}K"
    if n >= 1_000:
        return f"{round(n, -2):,}"
    return f"{n:,}"


def _blank_fig() -> plt.Figure:
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")
    return fig


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    finally:
        sys.path.remove(str(path.parent))
    return module


def _pile_config() -> Any:
    return _load_module(REPO / "scripts" / "experiments" / "pile" / "pile_config.py", "_slides_pile_config")


def _docmarks_config() -> Any:
    return _load_module(REPO / "scripts" / "experiments" / "docmarks" / "docmarks_config.py", "_slides_docmarks_config")


# --------------------------------------------------------------------------
# Shared drawing
# --------------------------------------------------------------------------


def _stats(fig: plt.Figure, rows: list[tuple[str, str]], *, top: float = LEFT_TOP, gap: float = 0.082) -> None:
    """A left-column block of `(value, label)` rows: big number, quiet caption.

    Two lines per row rather than one, because the values are of wildly
    different widths — `200,000` beside `23` — and a single line would set the
    captions on a ragged left edge that reads as a mistake.
    """
    for i, (value, label) in enumerate(rows):
        y = top - i * gap
        fig.text(LEFT_X, y, value, fontsize=FLOOR_PT + 9, color=INK, fontweight="bold", va="top")
        fig.text(LEFT_X, y - 0.042, label, fontsize=FLOOR_PT, color=SOFT, va="top")


def _steps(fig: plt.Figure, steps: list[tuple[str, str]], *, upto: int | None = None) -> None:
    """The right column's numbered stack: one row per construction step.

    Laid out by flow rather than on a fixed pitch — a row is as tall as its own
    wrapped body — because the alternative is a constant that is right for the
    longest step and leaves a hole under every other one. Rows are drawn from
    the same top in the same order whatever `upto` is, so a build reveal adds a
    row and moves nothing (`slides/STYLE.md`, *Builds*).

    `upto` draws only the first *n*, which is how the build stages are made.
    """
    shown = len(steps) if upto is None else upto
    y, line, lead, gap = 0.925, 0.0425, 0.052, 0.034
    for i, (verb, body) in enumerate(steps[:shown]):
        fig.text(RIGHT_X, y, f"{i + 1}", fontsize=FLOOR_PT + 11, color=RULE, fontweight="bold", va="top")
        fig.text(RIGHT_X + 0.042, y, verb, fontsize=FLOOR_PT + 5, color=INK, fontweight="bold", va="top")
        wrapped = textwrap.wrap(body, STEP_WRAP)
        fig.text(
            RIGHT_X + 0.042,
            y - lead,
            "\n".join(wrapped),
            fontsize=FLOOR_PT,
            color=SOFT,
            va="top",
            linespacing=1.5,
        )
        y -= lead + line * len(wrapped) + gap
    # Checked rather than eyeballed: the stack is laid out by flow, so one step
    # gaining a line silently pushes the last one off the bottom of the slide
    # — which is exactly what an edit to a *number* can do, months after the
    # layout was last looked at.
    if y < 0.045:
        raise SystemExit(
            f"step stack: {shown} steps overflow the slide (bottom at {y:.3f}, floor 0.045). "
            f"Shorten a step to {STEP_WRAP * 2} characters or fewer — every one of them is "
            f"meant to wrap to two lines."
        )


STRIP_TOP = 0.815


def _strip(fig: plt.Figure, tiles: list[Any], cols: int, rows: int, *, top: float = STRIP_TOP, captions=None) -> None:
    """A grid of real photographs in the right column, uniform tiles.

    Uniform because the tiles are the *evidence*, not a composition: a ragged
    mosaic invites the eye to read the layout, and the only thing worth reading
    here is the photographs.
    """
    pad = 0.008
    cell_w = RIGHT_W / cols
    cell_h = cell_w * (FIG_W / FIG_H) * 0.75
    for i, tile in enumerate(tiles[: cols * rows]):
        r, c = divmod(i, cols)
        ax = fig.add_axes(
            [RIGHT_X + c * cell_w, top - (r + 1) * cell_h + pad, cell_w - pad, cell_h - pad - (0.03 if captions else 0)]
        )
        ax.imshow(tile)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(RULE)
        if captions:
            ax.set_xlabel(captions[i], fontsize=FLOOR_PT, color=SOFT, labelpad=4)


def _fit_tile(image: Any, aspect: float = 4 / 3) -> Any:
    """Centre-crop to a common aspect so a grid of frames is a grid, not a mess."""
    return _fit_tile_offset(image, aspect)[0]


def _fit_tile_offset(image: Any, aspect: float = 4 / 3) -> tuple[Any, tuple[int, int]]:
    """As `_fit_tile`, and the pixels the crop removed from the left and top.

    A card that draws annotation over its frames has to move the boxes by the
    same crop; returning the offset is what stops that being a second, silently
    different piece of arithmetic.
    """
    width, height = image.size
    if width / height > aspect:
        side = int(round(height * aspect))
        left = (width - side) // 2
        box = (left, 0, left + side, height)
    else:
        side = int(round(width / aspect))
        top = (height - side) // 2
        box = (0, top, width, top + side)
    return image.convert("RGB").crop(box), (box[0], box[1])


def _source_line(fig: plt.Figure, text: str, *, caption: str = "where to get it", y: float = 0.105) -> None:
    """The pointer on the foot of a card: a download URL, or how it is built.

    Two captions rather than one because two of these datasets are not
    downloads at all — they are constructions over somebody else's download,
    and labelling a build recipe "where to get it" would send a reader looking
    for a zip that does not exist.
    """
    fig.text(LEFT_X, y, caption, fontsize=FLOOR_PT, color=SOFT, va="center")
    fig.text(LEFT_X, y - 0.052, text, fontsize=FLOOR_PT + 1, color=CUT, va="center", family="monospace")


def _caption(fig: plt.Figure, text: str) -> None:
    """One descriptive line at the top of the right column, and no title.

    For a figure whose *name* is already the slide's headline, sitting in the
    notch six centimetres to the left. Repeating "Caltech-101" inside the
    drawing is the one thing a full-bleed slide can get wrong for free: the
    room reads it twice and learns nothing the second time.
    """
    fig.text(
        RIGHT_X,
        0.945,
        "\n".join(textwrap.wrap(text, 76)),
        fontsize=FLOOR_PT + 1,
        color=SOFT,
        va="top",
        linespacing=1.4,
    )


def _heading(fig: plt.Figure, text: str, sub: str = "") -> None:
    """The right column's own heading, above whatever it carries.

    Only for a panel the slide's headline does not already name — the source
    breakdown, the three band sets. Where it would restate the headline, use
    `_caption`.
    """
    fig.text(RIGHT_X, 0.945, text, fontsize=FLOOR_PT + 5, color=INK, fontweight="bold", va="top")
    if sub:
        # Wrapped, not trusted to fit: the right column is 863px wide and a
        # subtitle written one word longer runs off the slide, where nothing
        # checks it — the notch guard only watches the other corner.
        fig.text(
            RIGHT_X,
            0.895,
            "\n".join(textwrap.wrap(sub, 76)),
            fontsize=FLOOR_PT,
            color=SOFT,
            va="top",
            linespacing=1.4,
        )


# --------------------------------------------------------------------------
# coco_quarry: the construction
# --------------------------------------------------------------------------


def _quarry_classes(pc: Any) -> list[str]:
    """The roster the cells are built for, in `pile_config`'s own order."""
    return list(pc.SCALE_CLASSES)


def _quarry_cell_count(pc: Any) -> int:
    """Every `class@band`, less the cells no honest supply fills."""
    return len(pc.SCALE_CLASSES) * len(pc.BOX_BANDS) - len(pc.SCALE_DROPPED_CELLS)


def _quarry_steps(pc: Any) -> list[tuple[str, str]]:
    """What `coco_quarry` is made of, one decision per step.

    Written as the construction the build performs, in the order it performs
    it — not the order the decisions were reached in. Every number is read out
    of `pile_config`, so a ruling that moves the roster moves the slide.
    """
    bands = pc.BOX_BANDS
    merged = [name for name in pc.SCALE_CLASSES if name in pc.SCALE_CLASS_MERGES]
    return [
        (
            "Start with all of COCO 2017",
            f"{count(123_287)} photographs and 80 classes, every class answered on every image — "
            "so a missing box is an absence.",
        ),
        (
            "Merge the names COCO cannot keep apart",
            "Where one object gets either label — car or truck, cup or wine glass — the two "
            f"become one class: {len(merged)} merges in all.",
        ),
        (
            "Band by the largest instance",
            f"Against the model's own grid: small is under one patch (1/{1 / bands['small'][1]:.0f}), "
            f"large runs to {bands['large'][1]:.0%} of the frame.",
        ),
        (
            "Check that the box is one object",
            "LVIS re-boxed the same photographs one object at a time. A COCO box round a pile of "
            "fruit or books is no positive.",
        ),
        (
            "Keep every class that fills its bands",
            f"By count alone, never by how easy a class looks: {len(pc.SCALE_CLASSES)} classes. "
            f"No honest supply fills {len(pc.SCALE_DROPPED_CELLS)} small-fruit cells.",
        ),
    ]


def fig_quarry_build(pc: Any, upto: int | None = None) -> plt.Figure:
    """How `coco_quarry` is cut out of COCO, in five steps."""
    fig = _blank_fig()
    _steps(fig, _quarry_steps(pc), upto=upto)
    if upto is None:
        _stats(
            fig,
            [
                (f"{len(pc.SCALE_CLASSES)}", "classes"),
                (f"{len(pc.BOX_BANDS)}", "size bands each"),
                (f"{_quarry_cell_count(pc)}", "cells, all the same shape"),
            ],
        )
    return fig


def fig_quarry_bands(pc: Any) -> plt.Figure:
    """What the three bands *are*: box area against the model's own geometry.

    The bands are not thirds of some range somebody chose. ``small`` is
    "below one patch" and ``medium`` tops out at the smallest HAC leaf, so the
    boundaries are properties of the embedder rather than of the dataset —
    which is what makes a small-vs-large result a statement about the method.
    Drawing the three boxes to scale inside one frame is the only way to say
    that without the audience taking it on trust.
    """
    fig = _blank_fig()
    bands = pc.BOX_BANDS

    # Three frames along the bottom, clear of the title reserve. The leftmost
    # one shares the notch's x-range, so the whole row sits low enough that its
    # titles clear the reserve's bottom edge rather than the row being shoved
    # right — which would leave the left third of a full-bleed slide empty.
    for i, (name, (lo, hi)) in enumerate(bands.items()):
        ax = fig.add_axes([0.06 + i * 0.31, 0.135, 0.26, 0.35])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec=SOFT, lw=2.0))

        # A box of the band's geometric-mean area, centred: the representative
        # member rather than either endpoint.
        area = (max(lo, 1 / 4000) * hi) ** 0.5
        side = area**0.5
        ax.add_patch(
            Rectangle(
                ((1 - side) / 2, (1 - side) / 2),
                side,
                side,
                facecolor=CUT,
                alpha=0.85,
                ec="none",
            )
        )
        ax.set_title(name, fontsize=FLOOR_PT + 8, color=INK, pad=14, fontweight="bold")
        ax.text(
            0.5,
            -0.11,
            f"up to 1/{1 / hi:.0f} of the frame" if hi < 0.5 else f"up to {hi:.0%} of the frame",
            transform=ax.transAxes,
            ha="center",
            fontsize=FLOOR_PT + 2,
            color=INK,
        )
        upper = {"small": "one model patch", "medium": "one poolable region", "large": "most of the picture"}[name]
        ax.text(
            0.5,
            -0.20,
            upper,
            transform=ax.transAxes,
            ha="center",
            fontsize=FLOOR_PT,
            color=SOFT,
        )

    # The class list, right of the reserve so the corner stays clear. Set as
    # wrapped running text rather than columns, so a roster ruling that adds or
    # merges a class re-flows the list instead of breaking a hand-tuned grid.
    names = _quarry_classes(pc)
    _caption(fig, f"the same {len(names)} classes, whatever size the thing is")
    fig.text(
        RIGHT_X,
        0.885,
        "\n".join(textwrap.wrap(", ".join(names), 72)),
        fontsize=FLOOR_PT,
        color=SOFT,
        va="top",
        linespacing=1.45,
    )
    return fig


def fig_quarry_cells(pc: Any) -> plt.Figure:
    """Every cell the same size, drawn from the same pool.

    The uniformity *is* the design: identical positives and one shared pool
    make small-vs-large a paired comparison instead of two datasets of
    different difficulty. A grid of identical tiles is the honest picture of
    that — there is no variation to plot. The three cells that are not built
    are drawn empty rather than left out, because a grid with a hole in it
    says "not built" and a grid with three fewer columns in one row says
    nothing at all.

    The columns carry no class names. At this roster's width a column is about
    twenty slide pixels, which is one line of the floor type set sideways and
    no room between neighbours; the names are on the slide before.
    """
    fig = _blank_fig()
    classes = _quarry_classes(pc)
    bands = list(pc.BOX_BANDS)
    dropped = set(pc.SCALE_DROPPED_CELLS)

    grid_x, grid_w, grid_y, grid_h = 0.215, 0.765, 0.43, 0.21
    ax = fig.add_axes([grid_x, grid_y, grid_w, grid_h])
    ax.set_xlim(-0.5, len(classes) - 0.5)
    ax.set_ylim(-0.5, len(bands) - 0.5)
    ax.axis("off")

    for r, band in enumerate(bands):
        for c, name in enumerate(classes):
            if f"{name}@{band}" in dropped:
                ax.add_patch(Rectangle((c - 0.42, r - 0.40), 0.84, 0.80, fill=False, ec=SOFT, lw=1.2, ls=":"))
                continue
            ax.add_patch(Rectangle((c - 0.42, r - 0.40), 0.84, 0.80, facecolor=CUT, alpha=0.18, ec=CUT, lw=1.2))
        # Set as figure text, not an axis label: an axis label is placed
        # outside the axes and runs off the canvas when the roster grows.
        fig.text(
            grid_x - 0.014,
            grid_y + grid_h * (r + 0.5) / len(bands),
            band,
            ha="right",
            va="center",
            fontsize=FLOOR_PT + 3,
            color=INK,
        )
    fruit = sorted({cell.split("@")[0] for cell in dropped}, key=classes.index)
    fig.text(
        grid_x,
        grid_y - 0.035,
        f"dotted: {', '.join(fruit)} @small — no honest supply, so not built",
        fontsize=FLOOR_PT,
        color=SOFT,
        va="top",
    )

    _stats(
        fig,
        [
            (f"{pc.SCALE_N_POS}", "positives in each cell"),
            (count(pc.SCALE_N_NEG), "shared negatives, holding no class"),
        ],
        top=0.30,
    )
    fig.text(
        RIGHT_X + 0.10,
        0.30,
        "\n".join(
            textwrap.wrap(
                "Every other cell's positives count as negatives too, wherever they do not hold "
                "the class. And the whole of COCO is embedded, so any other cut — a prevalence, a "
                "pool, a size to train at — is a filter, not a rebuild.",
                52,
            )
        ),
        fontsize=FLOOR_PT + 1,
        color=SOFT,
        va="top",
        linespacing=1.5,
    )
    fig.text(
        RIGHT_X,
        0.945,
        f"{len(classes)} classes × {len(bands)} bands, less {len(dropped)} = {_quarry_cell_count(pc)} cells",
        fontsize=FLOOR_PT + 5,
        color=INK,
        fontweight="bold",
        va="top",
    )
    fig.text(
        RIGHT_X,
        0.885,
        "\n".join(
            textwrap.wrap(
                "Every cell is exactly the same shape: the same positive count, the same shared "
                "pool of negatives. Nothing varies across the grid but the size of the thing you "
                "are looking for.",
                62,
            )
        ),
        fontsize=FLOOR_PT + 1,
        color=SOFT,
        va="top",
        linespacing=1.55,
    )
    return fig


# --------------------------------------------------------------------------
# DocMarks
# --------------------------------------------------------------------------


DATASHEET = REPO / "scripts" / "experiments" / "docmarks" / "DATASHEET.md"

#: Chart labels for the sources whose datasheet name is a sentence. A y-axis
#: label is drawn outside its axes, so a long one eats leftwards into the
#: slide's title notch; the full names are on the datasheet and in the notes.
_SOURCE_SHORT = {"UCSF Industry Documents": "UCSF"}


def _datasheet_rows(heading: str, columns: int) -> list[list[str]]:
    """Body rows of the first markdown table under `heading`.

    Parsing the committed datasheet rather than retyping it: the corpus itself
    is on the GRID, so this is the freshest source a checkout has, and a
    heading or column that moves fails the generator instead of producing a
    figure that quietly disagrees with the page beside it.
    """
    text = DATASHEET.read_text()
    start = text.find(heading)
    if start < 0:
        raise SystemExit(f"docmarks datasheet: no '{heading}' section in {DATASHEET}")
    rows: list[list[str]] = []
    seen_table = False
    for line in text[start + len(heading) :].splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            if seen_table and rows:
                break
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        seen_table = True
        if len(cells) != columns or set("".join(cells)) <= set("-: "):
            continue
        rows.append(cells)
    if not rows:
        raise SystemExit(f"docmarks datasheet: no {columns}-column table under '{heading}'")
    return rows[1:]  # drop the header row


def _clean(cell: str) -> str:
    """Markdown cell to plain text: strip emphasis, backticks and footnote marks."""
    return re.sub(r"[*`]", "", cell).strip()


def _number(cell: str) -> int:
    match = re.search(r"\d[\d,]*", _clean(cell))
    if not match:
        raise SystemExit(f"docmarks datasheet: no number in {cell!r}")
    return int(match.group().replace(",", ""))


def _docmarks_facts() -> dict[str, Any]:
    """Counts the figures need, read out of the datasheet's own tables.

    The per-class spread is computed from the roster table rather than read
    off the summary row above it, so the chart and its own caption can never
    disagree: the bars *are* that table.
    """
    sources = [
        {
            "name": _clean(r[0]),
            "what": _clean(r[1]),
            "pages": _number(r[2]),
            "classes": _number(r[3]) if any(ch.isdigit() for ch in r[3]) else 0,
            "instances": _number(r[4]),
        }
        for r in _datasheet_rows("### Sources", 5)
    ]
    instances = sorted((_number(r[2]) for r in _datasheet_rows("### The roster", 4)), reverse=True)
    if sum(s["instances"] for s in sources) != sum(instances):
        raise SystemExit(
            f"docmarks datasheet: the Sources table counts {sum(s['instances'] for s in sources)} instances "
            f"and the roster table {sum(instances)} — fix the datasheet before drawing it"
        )
    # The distractor source is the one the tiers are made of: every page the
    # three anchor sources carry sits in the smallest tier, so it is simply the
    # largest source by far.
    haystack = max(sources, key=lambda s: s["pages"])
    return {
        "classes": len(instances),
        "instances": sum(instances),
        "per_class": (min(instances), max(instances), int(np.median(instances))),
        "sources": sources,
        "haystack": haystack,
        "anchors": [s for s in sources if s is not haystack],
        "class_instances": instances,
    }


def _docmarks_steps(dc: Any, facts: dict[str, Any]) -> list[tuple[str, str]]:
    """What DocMarks is made of, one decision per step.

    Written as the construction the build performs, not the order the passes
    were first run in. Every number is the corpus's own.
    """
    haystack = facts["haystack"]
    return [
        (
            "Start with pages whose marks are outlined",
            ", ".join(f"{s['name']} ({count(s['pages'])})" for s in facts["anchors"])
            + f" ship every mark's outline; UCSF's {haystack['classes']} letterhead marks are boxed by hand.",
        ),
        (
            "Settle which marks are the same mark",
            "Hashing proposes the groups, and a person rules: is it one mark, is every member it, "
            "are two classes really two?",
        ),
        (
            "Find every copy",
            "SIFT searches every page for copies the sources missed, and a person confirms each: "
            f"{facts['instances']:,} copies of {facts['classes']} marks.",
        ),
        (
            "Bury them in real documents",
            f"{count(haystack['pages'])} scanned industry pages, in nested tiers of "
            # The tiers sit in one breath, so they take one form: `count` would
            # give "5,000 ⊂ 50K ⊂ 200K", which reads as three different units.
            + " ⊂ ".join(f"{round(size, -3) // 1000}K" for size in dc.TIERS.values())
            + " pages.",
        ),
        (
            "Fix what counts as a wrong answer",
            "Negatives are pages checked not to hold the mark, no archive scores against "
            "itself, and a mark-blind control runs alongside.",
        ),
    ]


def fig_docmarks_build(dc: Any, facts: dict[str, Any], upto: int | None = None) -> plt.Figure:
    """How DocMarks is made, in five steps."""
    fig = _blank_fig()
    _steps(fig, _docmarks_steps(dc, facts), upto=upto)
    if upto is None:
        _stats(
            fig,
            [
                (count(max(dc.TIERS.values())), "pages"),
                (f"{facts['classes']}", "marks to find"),
                (f"{facts['instances']:,}", "copies of them, all checked"),
            ],
        )
    return fig


def fig_docmarks_shape(dc: Any, facts: dict[str, Any]) -> plt.Figure:
    """What DocMarks is: where the pages come from, and how the roster falls.

    The two panels answer the two questions that decide whether a result on
    this corpus means anything. *Where the pages come from* — one source
    seventy times the size of the other three together — is why growing the
    haystack is a test against unrelated real scans. *How the copies fall
    across marks* is why a per-class number needs its n printed beside it.
    """
    fig = _blank_fig()
    sources = facts["sources"]
    total = max(dc.TIERS.values())

    _stats(
        fig,
        [
            (count(total), "pages, in three nested tiers"),
            (f"{facts['instances']:,}", f"copies of {facts['classes']} marks"),
            (f"{dc.CORPUS_VERSION}", "corpus version, and it moves"),
        ],
    )
    _source_line(fig, "scripts/experiments/docmarks/DATASHEET.md", caption="what it can be asked")

    _heading(
        fig,
        "Four sources, and one of them is the haystack",
        "the three anchor sources sit whole in the smallest tier; a bigger tier adds UCSF pages — "
        "distractors, and more copies of the four UCSF marks",
    )

    # Indented from the column: a y axis draws its ticks and its label to the
    # LEFT of the axes, and the notch is right there.
    bar_x = RIGHT_X + 0.065
    ax = fig.add_axes([bar_x, 0.47, 0.962 - bar_x, 0.29])
    names = [_SOURCE_SHORT.get(s["name"], s["name"]) for s in sources]
    pages = [s["pages"] for s in sources]
    ax.bar(range(len(names)), pages, color=[SOFT if s is facts["haystack"] else CUT for s in sources], width=0.6)
    ax.set_yscale("log")
    ax.set_ylim(top=max(pages) * 40)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=FLOOR_PT + 1, color=INK)
    ax.set_ylabel("pages (log)", fontsize=FLOOR_PT, color=SOFT)
    ax.tick_params(axis="y", labelsize=FLOOR_PT, colors=SOFT)
    ax.tick_params(axis="x", length=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    # Bars rather than a table because the point is the ratio: the distractor
    # source is seventy times the size of the other three together, and a
    # column of numbers does not say that at a glance.
    for i, source in enumerate(sources):
        ax.text(
            i,
            source["pages"] * 1.5,
            f"{count(source['pages'])}\n{source['instances']:,} copies",
            ha="center",
            va="bottom",
            fontsize=FLOOR_PT,
            color=INK,
            linespacing=1.4,
        )

    lo, hi, median = facts["per_class"]
    ax = fig.add_axes([bar_x, 0.175, 0.975 - bar_x, 0.20])
    ax.bar(range(len(facts["class_instances"])), facts["class_instances"], color=CUT, width=0.82)
    ax.axhline(median, color=NEG, lw=1.5, ls="--")
    ax.set_xlabel("one bar per mark, most copies first", fontsize=FLOOR_PT, color=SOFT)
    ax.set_ylabel("copies", fontsize=FLOOR_PT, color=SOFT)
    # No x ticks: the bars are marks, and their position in a sorted order
    # is not a quantity anybody should read off an axis.
    ax.set_xticks([])
    ax.tick_params(labelsize=FLOOR_PT, colors=SOFT)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.text(
        0.0,
        1.04,
        f"{lo} to {hi} copies per mark, median {median}",
        transform=ax.transAxes,
        fontsize=FLOOR_PT,
        color=NEG,
    )
    return fig


# --------------------------------------------------------------------------
# Cards: one per dataset the studies actually ran on
# --------------------------------------------------------------------------


def fig_card_visual_genome() -> plt.Figure:
    """Visual Genome: messy web photography, densely annotated in free text."""
    from PIL import Image

    import dataset_samples

    fig = _blank_fig()
    _stats(
        fig,
        [
            (count(108_077), "photographs"),
            (count(2_516_939), "objects, each a box and a name"),
            ("0.61", "VG's recall against COCO"),
        ],
    )
    _source_line(fig, "homes.cs.washington.edu/~ranjay/visualgenome")
    _caption(fig, "ordinary scenes with a dozen nameable things in each — in the app as visual_genome_s/m/l/a")
    tiles = [_fit_tile(Image.open(p)) for p in dataset_samples.visual_genome_samples()]
    _strip(fig, tiles, cols=4, rows=3)
    return fig


def fig_card_coco_val() -> plt.Figure:
    """COCO val2017: fewer classes than VG, annotated exhaustively — the reference."""
    from PIL import Image

    import dataset_samples

    images, coco = dataset_samples.coco_val()
    annotated = {a["image_id"] for a in coco["annotations"]}
    by_id = {i["id"]: i for i in coco["images"]}
    boxes: dict[int, list[list[float]]] = {}
    for annotation in coco["annotations"]:
        boxes.setdefault(annotation["image_id"], []).append(annotation["bbox"])

    fig = _blank_fig()
    _stats(
        fig,
        [
            (count(len(by_id)), "images in val2017"),
            (f"{len(coco['categories'])}", "classes, all always annotated"),
            (count(len(by_id) - len(annotated)), "hold none of them"),
        ],
    )
    _source_line(fig, "images.cocodataset.org/zips/val2017.zip")
    _caption(
        fig,
        "exhaustive — if a class is not boxed here it is not in the picture, which is what "
        "lets COCO say where a class is absent",
    )

    # Cropped to the strip's own aspect like every other card, with the boxes
    # moved by the same crop: a grid of raw COCO frames is four aspect ratios
    # in two rows, and the eye reads the ragged edges before it reads the
    # annotation the slide is about.
    tiles, moved = [], []
    for image_id in dataset_samples.COCO_SAMPLE_IDS[:12]:
        meta = by_id[image_id]
        with Image.open(images / meta["file_name"]) as frame:
            tile, (dx, dy) = _fit_tile_offset(frame)
        tiles.append(tile)
        moved.append([(x - dx, y - dy, w, h) for x, y, w, h in boxes.get(image_id, [])])

    pad = 0.008
    cols = 4
    cell_w = RIGHT_W / cols
    cell_h = cell_w * (FIG_W / FIG_H) * 0.75
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        ax = fig.add_axes([RIGHT_X + c * cell_w, STRIP_TOP - (r + 1) * cell_h + pad, cell_w - pad, cell_h - pad])
        ax.imshow(tile)
        for x, y, w, h in moved[i]:
            ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=CUT, lw=1.3))
        ax.set_xlim(0, tile.size[0])
        ax.set_ylim(tile.size[1], 0)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(RULE)
    return fig


def fig_card_caltech101() -> plt.Figure:
    """Caltech-101: one centred object, no boxes — the boxless control."""
    from PIL import Image

    import dataset_samples

    picks = dataset_samples.caltech101_samples()
    # `BACKGROUND_Google` is a 102nd directory and not one of the 101: it is a
    # pile of assorted web images the original paper shipped as a negative
    # class. Counting it would put a wrong number on the slide.
    categories = [c for c in dataset_samples.caltech101_categories() if not c.startswith("BACKGROUND")]

    fig = _blank_fig()
    _stats(
        fig,
        [
            (f"{len(categories)}", "categories, one per picture"),
            ("0", "boxes — whole-image labels only"),
            ("838", "of them in caltech101_m"),
        ],
    )
    _source_line(fig, "data.caltech.edu/records/mzrjq-6wc02")
    _caption(fig, "one object, centred, filling the frame — the set where region voting has nothing to point at")
    tiles = [_fit_tile(Image.open(p)) for _, p in picks]
    _strip(fig, tiles, cols=4, rows=3, captions=[c.replace("_", " ") for c, _ in picks])
    return fig


# --------------------------------------------------------------------------
# coco_quarry
# --------------------------------------------------------------------------

#: The Venn, in the drawing's own units. `QUARRY_D` is how far each circle's
#: centre sits from the group's, and the ratio to `QUARRY_R` is what decides
#: how big the three-way cell in the middle comes out — the one that has to
#: hold `ABC` and a superscript. At 0.46 / 0.70 that cell is 0.24 units of
#: inradius, which is 108 slide pixels across against a 45px label; the
#: textbook 0.55 / 0.70 halves it and the label spills over the lens edges.
QUARRY_R = 0.70
QUARRY_D = 0.46
#: The universe rectangle: every image there is, drawn so that "outside the
#: circles" is a *region* with room to be shaded and labelled rather than the
#: margin of the page.
QUARRY_RECT = (-1.55, -1.35, 3.10, 2.70)

#: The deck's calibration palette, not a pair of tints of it. Every shaded
#: region here is a *training side* — the Good pile or the Bad pile — which is
#: the thing `make-calib-figs._data_block` already draws, so it is drawn the
#: same way: hollow, white-faced, hatched in the deck's own red and green, the
#: hatch leaning the way that file leans it. Two solid fills side by side read
#: as a chart of two quantities; two hatches read as two *kinds*, and they stay
#: apart for a viewer who cannot separate the hues at all.
QUARRY_GOOD = "#0d8a5f"
QUARRY_BAD = "#b91c1c"
GOOD_HATCH = "//////"
BAD_HATCH = "\\\\\\"

#: The empty set is written **Ø** (U+00D8, the Latin letter O with stroke) and
#: not ∅ (U+2205, the mathematical empty set), for the one reason a deck cares
#: about: the proper symbol is a math glyph and does not take the bold face's
#: weight. Measured over DejaVu Sans Bold as ink area over bounding box,
#: `emptyset` fills 0.43 of its box against 0.51 for `A` and 0.56 for `O`,
#: while `Oslash` is 0.58 — right in with the letters. Beside a bold `A⁺` the
#: correct character reads as something pasted in from another document, which
#: is the same objection `slides/STYLE.md` raises against setting a formula in
#: Computer Modern. This is a *typographic* substitution of a glyph the symbol
#: was derived from, so it is written once, here, and referred to by name.
#:
#: The superscripts were checked the same way and are fine: `⁺` and `⁼` carry
#: exactly the ink of a full-size `+` and `=` in the bold face. They look light
#: because they are small, which is what a superscript is.
EMPTY = "\u00d8"

#: Characters per line in the left column, which is `RIGHT_X - LEFT_X` wide —
#: 357 slide pixels, about 31 characters of the 15pt body face.
#: The left column's two registers, in points, and the rhythm they are set on
#: in figure fractions. A *definition* is one line — bold term, then its gloss
#: in the body face on the same baseline — because a term over its own gloss is
#: two lines that look exactly like a heading over its own caption, and the
#: column then reads as six headings with no hierarchy at all (which is what it
#: did). A *heading* is the other register: half again the size, on its own
#: line, with room above it, so the two experiments read as the titles of the
#: two halves of the argument rather than as two more entries in a list.
TERM_PT = FLOOR_PT + 3
HEAD_PT = FLOOR_PT + 11
#: Gap between a term and its gloss on the shared baseline. Wider than a word
#: space: the point is that the two are different registers, not one phrase.
TERM_GAP = 0.009
#: Three gaps, and their *order* is the hierarchy the column is asking the eye
#: to read: tightest between two definitions, wider after a heading where the
#: argument moves on, widest of all above a heading.
DEF_PITCH = 0.075
HEAD_LEAD = 0.050
HEAD_PITCH = 0.095
#: The column starts a shade above `LEFT_TOP` because these are baselines and
#: that one is a top: the tallest thing here rises about 0.025 over its own
#: baseline, so 0.706 puts the first cap-height at 0.731, still clear of the
#: title notch's 0.761. The floor is the last baseline, not the flow position
#: after it — trailing air is not overflow.
STACK_TOP = LEFT_TOP - 0.010
STACK_FLOOR = 0.035
#: How wide a definition line may be, in figure fractions: the left column's
#: own width, less the gutter before the drawing starts.
STACK_MAX_W = 0.275

#: Every label sits on a chip of its own background, because most of them land
#: on hatching. Tight padding: the chip is there to stop the strokes running
#: through the glyphs, not to box the word.
QUARRY_CHIP = {"boxstyle": "square,pad=0.18", "facecolor": "white", "edgecolor": "none"}

#: The column, in the order the build introduces it, as `(kind, term, gloss)`.
#: A heading carries no gloss: the verdict on each experiment is a thing the
#: presenter says, and a slide that also writes it down is asking the room to
#: read the sentence it is being told.
#:
#: The two kinds alternate by accident of the argument rather than by design,
#: and the grouping they produce is the reason the order is worth reading:
#: two definitions, the easy experiment they buy, two more definitions, and the
#: hard experiment *those* buy.
#:
#: **A gloss is an instruction to the images, not a description of the set.**
#: `A⁺` does not hold an A — the pictures in it do, every one of them — so the
#: gloss is written as the entry requirement each of them meets: *Hold an A,
#: maybe more.* Which is also why all four start with the same verb: the sets
#: differ in what they demand, not in what kind of demand it is.
#:
#: The `Easy:` heading is the one entry whose term moves — the experiment is
#: shown three times, once per class, and `_quarry_easy_term` supplies the
#: spelling for the frame being drawn.
QUARRY_BLOCKS = [
    ("def", "A⁺", "Hold an A, maybe more."),
    ("def", EMPTY, "Hold none of the three."),
    ("head", f"Easy: A⁺ vs {EMPTY}", None),
    ("def", "AB⁼", "Hold exactly A and B."),
    ("def", "¬A", "Hold no A."),
    ("head", "Hard: A⁺ vs ¬A", None),
]

#: One entry per frame: `(good, negatives, cells, blocks)`.
#:
#: `good` is the circle hatched as the positive pile, `negatives` is `None`,
#: `"outside"` (∅ alone) or `"not_a"` (everything but A), `cells` says whether
#: the seven exact-set labels are drawn, and `blocks` is how much of the
#: notation column has been introduced.
#:
#: **The middle three frames rotate rather than accumulate**, which is why the
#: fragment declares `frames: equal`. Frame *c* is not frame *b* plus ink — it
#: is the same experiment run for a different class, and that is the whole
#: argument of those three pages: whichever one you look at, the crescents are
#: never shown to anybody. A build-up would have to pick one of them to stand
#: for the other two, which is the sentence ("and you'd rotate for the other
#: two") the three frames exist to not have to say.
#:
#: The cells arrive at *e* and not before. Until ¬A needs naming there is
#: nothing for `AB⁼` to do but sit on the drawing being read, and the
#: shading is carrying the argument on its own up to that point.
QUARRY_FRAMES = [
    (None, None, False, 2),
    (0, "outside", False, 3),
    (1, "outside", False, 3),
    (2, "outside", False, 3),
    (None, None, True, 4),
    (None, "not_a", True, 5),
    (0, "not_a", True, 6),
]


def _quarry_easy_term(frame: int) -> str:
    """The `Easy:` block's spelling on `frame`.

    It follows the rotation through B and C and then comes back to A, so that
    the last three frames read `Easy: A⁺ vs ∅` directly above
    `Better: A⁺ vs ¬A` — the two experiments named for the same class,
    which is the comparison the slide closes on. Leaving it on C would compare
    two different detectors.
    """
    return f"Easy: {'ABC'[frame - 1] if 1 <= frame <= 3 else 'A'}⁺ vs {EMPTY}"


def _quarry_centres() -> list[tuple[float, float]]:
    """The three circle centres, as a group centred on the universe rectangle.

    The triangle of centres is not symmetric about its own midline — one
    circle is up and two are down — so placing them at `QUARRY_D` from the
    origin and stopping would hang the whole Venn above centre by half a
    radius. The drop is the group's own midline, computed rather than nudged.
    """
    raw = [
        (QUARRY_D * math.cos(math.radians(angle)), QUARRY_D * math.sin(math.radians(angle)))
        for angle in (90.0, 210.0, 330.0)
    ]
    drop = ((QUARRY_D + QUARRY_R) + (-QUARRY_D / 2 - QUARRY_R)) / 2
    return [(x, y - drop) for x, y in raw]


def _quarry_cells(centres: list[tuple[float, float]]) -> dict[int, tuple[float, float]]:
    """Where each of the seven exact-set labels goes: `{membership bits: (x, y)}`.

    Measured off a raster of the drawing rather than derived from the centres.
    The seven regions of a three-circle Venn are four different shapes, and the
    three single-class ones are crescents whose middle is nowhere near the
    circle's own centre — so arithmetic on the centres puts `A` on the rim of
    the lens below it. Each centroid is then checked to fall inside the cell it
    names, which is what makes a later edit to `QUARRY_R` or `QUARRY_D` fail
    here instead of printing `AB` into the wrong lens.
    """
    axis = np.linspace(-1.2, 1.2, 601)
    grid_x, grid_y = np.meshgrid(axis, axis)
    inside = [((grid_x - cx) ** 2 + (grid_y - cy) ** 2) <= QUARRY_R**2 for cx, cy in centres]
    cells: dict[int, tuple[float, float]] = {}
    for bits in range(1, 8):
        want = [bool(bits >> i & 1) for i in range(3)]
        selected = np.ones_like(grid_x, dtype=bool)
        for i in range(3):
            selected &= inside[i] if want[i] else ~inside[i]
        x, y = float(grid_x[selected].mean()), float(grid_y[selected].mean())
        held = [((x - cx) ** 2 + (y - cy) ** 2) <= QUARRY_R**2 for cx, cy in centres]
        if held != want:
            raise SystemExit(
                f"venn cells: the centroid of {quarry_cell_name(bits)} lands outside its own cell "
                f"at ({x:.3f}, {y:.3f}). QUARRY_R / QUARRY_D have moved far enough that a region "
                f"is no longer convex about its own middle — place that label by hand, or put the "
                f"circles back."
            )
        cells[bits] = (x, y)
    return cells


def quarry_cell_name(bits: int) -> str:
    """`AB⁼` for the cell holding exactly A and B, and so on."""
    return "".join(c for i, c in enumerate("ABC") if bits >> i & 1) + "⁼"


def _text_width(fig: plt.Figure, text: "matplotlib.text.Text") -> float:
    """`text`'s rendered width as a fraction of the figure's own width."""
    return text.get_window_extent(fig.canvas.get_renderer()).width / fig.bbox.width


def _quarry_stack(fig: plt.Figure, frame: int) -> None:
    """The left column: the notation, introduced one block per frame.

    Laid out by flow over *every* block whether this frame draws it or not, so
    a block arriving never moves the ones above it and the `Easy:` term can
    change spelling without anything below it shifting. Both guards below
    measure the whole column for the same reason — a reworded gloss fails on
    the first figure rather than on the last one.

    Definitions set their gloss on the term's own baseline, which means
    measuring the term: the gap between the two is a gap between *registers*
    and has to be the same however wide the term is, so it cannot be a column
    position. `A⁺` and `AB⁼` differ by half the gloss's own indent.
    """
    shown = QUARRY_FRAMES[frame][3]
    y = lowest = STACK_TOP
    for index, (kind, term, gloss) in enumerate(QUARRY_BLOCKS):
        draw = index < shown
        if kind == "head":
            y -= HEAD_LEAD
            if draw:
                fig.text(
                    LEFT_X,
                    y,
                    _quarry_easy_term(frame) if index == 2 else term,
                    fontsize=HEAD_PT,
                    color=INK,
                    fontweight="bold",
                    va="baseline",
                )
            lowest, y = y, y - HEAD_PITCH
            continue
        head = fig.text(LEFT_X, y, term, fontsize=TERM_PT, color=INK, fontweight="bold", va="baseline")
        width = _text_width(fig, head)
        body = fig.text(LEFT_X + width + TERM_GAP, y, gloss, fontsize=FLOOR_PT, color=SOFT, va="baseline")
        line = width + TERM_GAP + _text_width(fig, body)
        if line > STACK_MAX_W:
            raise SystemExit(
                f'notation column: "{term} {gloss}" sets {line:.3f} of the figure wide, over the '
                f"{STACK_MAX_W} the left column has. A definition is one line by design — shorten "
                f"the gloss rather than letting it wrap, which would make it look like a heading."
            )
        if not draw:
            head.remove()
            body.remove()
        lowest, y = y, y - DEF_PITCH
    if lowest < STACK_FLOOR:
        raise SystemExit(
            f"notation column: the {len(QUARRY_BLOCKS)} blocks overflow the slide (last baseline "
            f"at {lowest:.3f}, floor {STACK_FLOOR}). Drop a block, or tighten DEF_PITCH / "
            f"HEAD_PITCH."
        )


def _quarry_chip(ax: plt.Axes, x: float, y: float, text: str, size: float, colour: str = INK, **kwargs) -> None:
    """A label on a chip of background, so hatching does not run through it."""
    ax.text(x, y, text, fontsize=size, color=colour, bbox=dict(QUARRY_CHIP), zorder=6, **kwargs)


def _quarry_tone(bits: int, good: int | None, negatives: str | None) -> str:
    """The colour a region's own label takes: the colour that region is shaded.

    `bits` is a membership mask over (A, B, C), with `0` meaning the outside.
    A label names one pile or it names none, so this answers only for regions
    that are *entirely* one pile — every cell of the Venn, and the outside. It
    is not true of a whole circle once A is the positive side: B is then green
    where it crosses A and red where it does not, and its label stays ink.
    Circle labels are coloured by the caller for that reason.
    """
    if good is not None and bits and bits >> good & 1:
        return QUARRY_GOOD
    if negatives == "outside" and not bits:
        return QUARRY_BAD
    if negatives == "not_a" and not bits & 1:
        return QUARRY_BAD
    return INK


def fig_coco_quarry_complement(frame: int = len(QUARRY_FRAMES) - 1) -> plt.Figure:
    """Why `coco_quarry` needs COCO's exhaustive annotation: the complement.

    Seven frames over one Venn. The circles are three classes, the rectangle is
    every image there is, and the slide walks the two ways to pick negatives
    for a detector. The cheap way is the outside of all three circles — run
    once per class, which is what frames *b* to *d* do — and it never once
    asks a detector to tell an A from a B, because nothing in the crescents is
    ever shown to it. The better way is the outside of *A* alone, which is only
    nameable because COCO answers for all eighty of its classes on every image
    it touches.
    """
    fig = _blank_fig()
    x0, y0, w, h = QUARRY_RECT
    # The axes take the rectangle's own aspect, so the drawing fills them
    # exactly instead of letterboxing inside a slot picked by hand — and a
    # later change to QUARRY_RECT keeps doing so.
    ax_h = 0.91
    ax = fig.add_axes((0.342, 0.045, ax_h * FIG_H * (w / h) / FIG_W, ax_h))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(x0, x0 + w)
    ax.set_ylim(y0, y0 + h)

    good, negatives, cells, _ = QUARRY_FRAMES[frame]
    centres = _quarry_centres()
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="white", edgecolor=SOFT, linewidth=1.6, zorder=1))
    if negatives:
        ax.add_patch(
            Rectangle(
                (x0, y0),
                w,
                h,
                facecolor="none",
                edgecolor=QUARRY_BAD,
                hatch=BAD_HATCH,
                linewidth=0,
                zorder=2,
            )
        )
        # Knocked back out of the Bad pile: under "outside" every circle is,
        # because only ∅ is a negative there; under "not_a" only A is, and the
        # crescents stay hatched — which is the whole of the last reveal.
        kept = centres if negatives == "outside" else centres[:1]
        for centre in kept:
            ax.add_patch(Circle(centre, QUARRY_R, facecolor="white", edgecolor="none", zorder=3))
    if good is not None:
        ax.add_patch(
            Circle(
                centres[good],
                QUARRY_R,
                facecolor="none",
                edgecolor=QUARRY_GOOD,
                hatch=GOOD_HATCH,
                linewidth=0,
                zorder=4,
            )
        )
    for centre in centres:
        ax.add_patch(Circle(centre, QUARRY_R, facecolor="none", edgecolor=INK, linewidth=2.0, zorder=5))

    # Each circle is named from just outside its own rim, anchored by the
    # corner facing the circle so the word grows *away* from the drawing —
    # centring it on the radial instead straddles the outline, and a label
    # lying across the thing it names is the one placement that reads as a
    # mistake rather than as a gap (`slides/STYLE.md`, *A label is closer*).
    for index, ((cx, cy), (dx, dy), anchor, name) in enumerate(
        zip(
            centres,
            ((0.0, 1.0), (-0.866, -0.5), (0.866, -0.5)),
            (("center", "bottom"), ("right", "top"), ("left", "top")),
            "ABC",
        )
    ):
        _quarry_chip(
            ax,
            cx + dx * (QUARRY_R + 0.06),
            cy + dy * (QUARRY_R + 0.06),
            f"{name}⁺",
            FLOOR_PT + 5,
            QUARRY_GOOD if good == index else INK,
            fontweight="bold",
            ha=anchor[0],
            va=anchor[1],
        )
    # ∅ bottom-left, ¬A top-right: they name nested regions once both are on
    # screen, so they go in opposite corners rather than along one edge.
    empty = _quarry_tone(0, good, negatives)
    _quarry_chip(ax, x0 + 0.13, y0 + 0.13, EMPTY, FLOOR_PT + 7, empty, fontweight="bold", ha="left", va="bottom")
    if cells:
        for bits, (x, y) in _quarry_cells(centres).items():
            tone = _quarry_tone(bits, good, negatives)
            _quarry_chip(ax, x, y, quarry_cell_name(bits), FLOOR_PT, tone, ha="center", va="center")
    if negatives == "not_a":
        # ¬A is red throughout by construction — it *is* the Bad pile here.
        _quarry_chip(
            ax, x0 + w - 0.13, y0 + h - 0.13, "¬A", FLOOR_PT + 7, QUARRY_BAD, fontweight="bold", ha="right", va="top"
        )

    _quarry_stack(fig, frame)
    return fig


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--no-media",
        action="store_true",
        help="skip the three cards that download real media (Visual Genome, COCO, Caltech-101)",
    )
    args = ap.parse_args()

    pc = _pile_config()
    dc = _docmarks_config()
    facts = _docmarks_facts()

    steps = len(_quarry_steps(pc))
    for n in range(1, steps):
        save(
            fig_quarry_build(pc, upto=n), OUT, f"dataset-coco-quarry-build.build{n}.png", column=FULL_BLEED, tight=False
        )
    save(fig_quarry_build(pc), OUT, "dataset-coco-quarry-build.png", column=FULL_BLEED, tight=False)
    save(fig_quarry_bands(pc), OUT, "dataset-coco-quarry-bands.png", column=FULL_BLEED, tight=False)
    save(fig_quarry_cells(pc), OUT, "dataset-coco-quarry-cells.png", column=FULL_BLEED, tight=False)

    steps = len(_docmarks_steps(dc, facts))
    for n in range(1, steps):
        save(
            fig_docmarks_build(dc, facts, upto=n),
            OUT,
            f"dataset-docmarks-build.build{n}.png",
            column=FULL_BLEED,
            tight=False,
        )
    save(fig_docmarks_build(dc, facts), OUT, "dataset-docmarks-build.png", column=FULL_BLEED, tight=False)
    save(fig_docmarks_shape(dc, facts), OUT, "dataset-docmarks-shape.png", column=FULL_BLEED, tight=False)

    for n in range(len(QUARRY_FRAMES) - 1):
        save(
            fig_coco_quarry_complement(frame=n),
            OUT,
            f"dataset-coco-quarry-complement.build{n + 1}.png",
            column=FULL_BLEED,
            tight=False,
        )
    save(fig_coco_quarry_complement(), OUT, "dataset-coco-quarry-complement.png", column=FULL_BLEED, tight=False)

    if args.no_media:
        print("--no-media: skipped the Visual Genome, COCO and Caltech-101 cards")
        return 0
    _save_photo(fig_card_visual_genome(), "dataset-card-visual-genome")
    _save_photo(fig_card_coco_val(), "dataset-card-coco-val")
    _save_photo(fig_card_caltech101(), "dataset-card-caltech101")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
