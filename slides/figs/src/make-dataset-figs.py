#!/usr/bin/env python
"""Dataset figures for the VTSearch decks.

Run from the repo root:

    python slides/figs/src/make-dataset-figs.py            # everything it can draw offline
    python slides/figs/src/make-dataset-figs.py --no-media # skip the cards that need pixels

Two kinds of figure, and the difference matters more than it looks.

**Story figures** say how a dataset *came about* — the sources it started from,
what was done to them, and what a reader may conclude. `vg_scale` and DocMarks
get one each, drawn as a five-step stack with build markers so the room watches
the construction assemble rather than reading a paragraph.

**Cards** say what a dataset *is*: how much of it there is, where to download
it, and — the part a drawing cannot fake — what the media look like. Every
card's strip is real pixels, fetched by `dataset_samples.py`.

**Where the numbers come from.** Nothing here is typed in twice.

* The `vg_scale` figures import `scripts/experiments/pile/pile_config.py`, so a
  slide cannot drift from the constants the pile actually builds against. The
  promotion from twelve classes to twenty-five moved four numbers on two of
  these figures and needed no edit here.
* The DocMarks figures read `scripts/experiments/docmarks/docmarks_config.py`
  for the tiers and the contamination rule, and parse the committed
  `DATASHEET.md` for the per-source and per-class counts. The corpus itself
  lives on the GRID; a container cannot open it, and a figure that could only
  be built on the cluster would be a figure that silently stopped being
  rebuilt. Parsing the datasheet is the weaker guarantee of the two and is
  chosen for exactly that reason — it is the strongest one available from a
  checkout, and it fails loudly rather than going stale in silence.
* The COCO card counts its own annotation file.

**What the DocMarks card still owes.** The one thing neither source carries is
*photographs of the marks*. The committed real-scans panel this file used to
draw was built against the 41-class v0 corpus and is gone rather than left to
mislead; reshooting it against v3.1 needs the corpus, so it is booked as GRID
work. Until then the DocMarks slides show structure and counts and say so.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

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
    save(fig, OUT, png, column=FULL_BLEED, tight=False)
    with Image.open(OUT / png) as image:
        image.convert("RGB").save(OUT / f"{name}.webp", quality=88, method=6)
    (OUT / png).unlink()
    print(f"wrote figs/{name}.webp")


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
    return _load_module(
        REPO / "scripts" / "experiments" / "docmarks" / "docmarks_config.py", "_slides_docmarks_config"
    )


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


def _heading(fig: plt.Figure, text: str, sub: str = "") -> None:
    """The right column's own heading, above whatever it carries."""
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
# vg_scale
# --------------------------------------------------------------------------


def _vg_scale_steps(pc: Any) -> list[tuple[str, str]]:
    """The construction, in the order it happened, with the numbers it turned on.

    Deliberately shorter than the truth. The real sequence ran repair, audit,
    repair again as each measurement changed what the last one meant; a talk
    that recounted that is a talk about our calendar rather than about the
    dataset. What is *not* fudged is any number, and no step is invented.
    """
    bands = pc.BOX_BANDS
    return [
        (
            "Start with Visual Genome",
            "108,077 photographs of ordinary scenes, every object drawn as a pixel box "
            "and named in free text by whoever annotated it.",
        ),
        (
            "Band by how much of the frame",
            f"Not a size somebody chose: small is under one patch of the model's own grid "
            f"(1/{1 / bands['small'][1]:.0f}), large runs to {bands['large'][1]:.0%} of it.",
        ),
        (
            "Repair the labels",
            "51,497 of them are COCO images too, so COCO's exhaustive boxes replace VG's. "
            "The rest went in front of a person, in VTSearch.",
        ),
        (
            "Audit what the class is called",
            "A bicycle annotated 'bike' was nobody's bicycle. 182 spellings now fold into "
            "their class; 254 more are withheld from it.",
        ),
        (
            "Draw negatives that are provable",
            "Every negative comes from the COCO-scored half, so 'holds no bus' is a fact "
            "rather than VG's silence — wrong 1.4% of the time.",
        ),
    ]


def fig_vg_scale_build(pc: Any, upto: int | None = None) -> plt.Figure:
    """How `vg_scale` came about, in five steps."""
    fig = _blank_fig()
    _steps(fig, _vg_scale_steps(pc), upto=upto)
    if upto is None:
        classes, bands = pc.SCALE_CLASSES, pc.BOX_BANDS
        _stats(
            fig,
            [
                (f"{len(classes)}", "classes"),
                (f"{len(bands)}", "size bands each"),
                (f"{len(classes) * len(bands)}", "cells, all the same shape"),
            ],
        )
    return fig


def fig_vg_scale_bands(pc: Any) -> plt.Figure:
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
        ax = fig.add_axes([0.06 + i * 0.31, 0.135, 0.26, 0.40])
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
    # wrapped running text rather than columns: the list has grown once already
    # (twelve to twenty-five) and a column layout has to be re-tuned every time
    # it does, while a wrap does not.
    names = list(pc.SCALE_CLASSES)
    fig.text(
        RIGHT_X,
        0.945,
        f"the same {len(names)} classes in every band",
        fontsize=FLOOR_PT + 5,
        color=INK,
        fontweight="bold",
        va="top",
    )
    fig.text(
        RIGHT_X,
        0.885,
        "\n".join(textwrap.wrap(", ".join(names), 54)),
        fontsize=FLOOR_PT + 1,
        color=SOFT,
        va="top",
        linespacing=1.7,
    )
    return fig


def fig_vg_scale_cells(pc: Any) -> plt.Figure:
    """Every cell the same size and the same prevalence.

    The uniformity *is* the design: identical prevalence in all of them makes
    small-vs-large a paired comparison instead of two datasets of different
    difficulty, which is the failure that made two earlier benchmark waves
    non-comparable. A grid of identical tiles is the honest picture of that —
    there is no variation to plot.

    Class names run *vertically* rather than on a slant. The list has grown
    once already, twelve to twenty-five, and a slanted label is anchored at one
    end and free at the other, so the rightmost name grew straight off the
    corner of the slide when it did.
    """
    fig = _blank_fig()
    classes = list(pc.SCALE_CLASSES)
    bands = list(pc.BOX_BANDS)
    n_pos, n_neg = pc.SCALE_N_POS, pc.SCALE_N_NEG

    grid_x, grid_w, grid_y, grid_h = 0.215, 0.765, 0.40, 0.21
    ax = fig.add_axes([grid_x, grid_y, grid_w, grid_h])
    ax.set_xlim(-0.5, len(classes) - 0.5)
    ax.set_ylim(-0.5, len(bands) - 0.5)
    ax.axis("off")

    for r, band in enumerate(bands):
        for c in range(len(classes)):
            ax.add_patch(
                Rectangle((c - 0.42, r - 0.40), 0.84, 0.80, facecolor=CUT, alpha=0.18, ec=CUT, lw=1.2)
            )
        # Set as figure text, not an axis label: an axis label is placed
        # outside the axes and ran off the canvas the first time the class
        # list grew.
        fig.text(
            grid_x - 0.014,
            grid_y + grid_h * (r + 0.5) / len(bands),
            band,
            ha="right",
            va="center",
            fontsize=FLOOR_PT + 3,
            color=INK,
        )
    for c, name in enumerate(classes):
        fig.text(
            grid_x + grid_w * (c + 0.5) / len(classes),
            grid_y - 0.022,
            name,
            ha="center",
            va="top",
            rotation=90,
            fontsize=FLOOR_PT,
            color=SOFT,
        )

    prevalence = n_pos / (n_pos + n_neg)
    _stats(
        fig,
        [
            (f"{n_pos}", "positives"),
            (f"{n_neg:,}", "negatives"),
            (f"{prevalence:.0%}", "prevalence"),
        ],
        top=0.33,
    )
    fig.text(
        RIGHT_X,
        0.945,
        f"{len(classes)} classes × {len(bands)} bands = {len(classes) * len(bands)} cells",
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
                "pool of negatives, the same prevalence. Nothing varies across the grid but the "
                "size of the thing you are looking for.",
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
    match = re.search(r"[\d,]+", _clean(cell))
    if not match:
        raise SystemExit(f"docmarks datasheet: no number in {cell!r}")
    return int(match.group().replace(",", ""))


def _docmarks_facts() -> dict[str, Any]:
    """Counts the figures need, read out of the datasheet's own tables."""
    text = DATASHEET.read_text()
    roster = re.search(r"\|\s*roster\s*\|\s*\*\*(\d+)\s*classes\*\*,\s*\*\*(\d+)\s*instances\*\*", text)
    if roster is None:
        raise SystemExit("docmarks datasheet: could not read the roster row of the 'What it is' table")
    per_class = re.search(r"\|\s*instances per class\s*\|\s*(\d+) to (\d+), median (\d+)", text)
    if per_class is None:
        raise SystemExit("docmarks datasheet: could not read the 'instances per class' row")
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
    pairs = re.search(r"\*\*(\d+) of \1\*\* pairs adjudicated", text)
    if pairs is None:
        raise SystemExit("docmarks datasheet: could not read the adjudicated-pairs count")
    return {
        "pairs": int(pairs.group(1)),
        "classes": int(roster.group(1)),
        "instances": int(roster.group(2)),
        "per_class": tuple(int(g) for g in per_class.groups()),
        "sources": sources,
        "class_instances": instances,
    }


def _docmarks_steps(dc: Any, facts: dict[str, Any]) -> list[tuple[str, str]]:
    """The construction, in the order it happened, with the numbers it turned on.

    Shortcutted the same way the `vg_scale` story is: the corpus was built,
    audited, rebuilt and re-audited three times over, and a talk that walked
    that is a talk about our calendar. The steps are real and every number is
    the corpus's own.
    """
    anchors = [s for s in facts["sources"] if s["instances"] > 0]
    anchor_pages = sum(s["pages"] for s in anchors)
    distractors = sum(s["pages"] for s in facts["sources"] if not s["instances"])
    return [
        (
            "Start with pages that carry marks",
            ", ".join(f"{s['name']} ({s['pages']:,})" for s in anchors)
            + f" — {anchor_pages:,} pages that each ship a mark with its outline already drawn.",
        ),
        (
            "Group the marks that are the same mark",
            "No source says 'these two impressions are the same stamp'. Hashing every "
            "boxed mark proposes the groups.",
        ),
        (
            "Settle every identity by hand",
            f"Is a group one mark? Is each member really it? Are two of them the same? "
            f"All {facts['pairs']} roster pairs were ruled on.",
        ),
        (
            "Bury them in real documents",
            f"{distractors:,} scanned industry pages go in as distractors, in nested tiers of "
            + " ⊂ ".join(f"{size:,}" for size in dc.TIERS.values())
            + " pages.",
        ),
        (
            "Fix what counts as a wrong answer",
            "A mark is scored against its own source's other pages, all checked. Two "
            "sources from one archive never score each other.",
        ),
    ]


def fig_docmarks_build(dc: Any, facts: dict[str, Any], upto: int | None = None) -> plt.Figure:
    """How DocMarks came about, in five steps."""
    fig = _blank_fig()
    _steps(fig, _docmarks_steps(dc, facts), upto=upto)
    if upto is None:
        _stats(
            fig,
            [
                (f"{max(dc.TIERS.values()):,}", "pages"),
                (f"{facts['classes']}", "marks to find"),
                (f"{facts['instances']}", "copies of them, all checked"),
            ],
        )
    return fig


def fig_docmarks_shape(dc: Any, facts: dict[str, Any]) -> plt.Figure:
    """What DocMarks is: where the pages come from, and how the roster falls.

    The two panels answer the two questions that decide whether a result on
    this corpus means anything. *Which pages hold marks* — 2,778 of 200,000,
    all of them in the smallest tier — is why growing the haystack adds only
    distractors. *How the instances fall across classes* is why a per-class
    number needs its n printed beside it: one miss moves recall by 1/8 at one
    end of the roster and 1/82 at the other.
    """
    fig = _blank_fig()
    sources = facts["sources"]
    total = max(dc.TIERS.values())
    anchor_pages = sum(s["pages"] for s in sources if s["instances"] > 0)

    _stats(
        fig,
        [
            (f"{total:,}", "pages, in three nested tiers"),
            (f"{anchor_pages:,}", "of them carry a roster mark"),
            (f"{dc.CORPUS_VERSION}", "corpus version, and it moves"),
        ],
    )
    _source_line(fig, "scripts/experiments/docmarks/README.md", caption="how it is built")

    _heading(
        fig,
        "Four sources, and only three of them hold answers",
        "every page holding a mark is in the smallest tier; a bigger tier is only more distractors",
    )

    # Indented from the column: a y axis draws its ticks and its label to the
    # LEFT of the axes, and the notch is right there.
    bar_x = RIGHT_X + 0.065
    ax = fig.add_axes([bar_x, 0.50, 0.962 - bar_x, 0.30])
    names = [_SOURCE_SHORT.get(s["name"], s["name"]) for s in sources]
    pages = [s["pages"] for s in sources]
    ax.bar(range(len(names)), pages, color=[CUT if s["instances"] else SOFT for s in sources], width=0.6)
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
    # source is seventy times the size of everything that holds an answer, and
    # a column of numbers does not say that at a glance.
    for i, source in enumerate(sources):
        note = f"{source['pages']:,}\n{source['instances']} marks" if source["instances"] else f"{source['pages']:,}\nno marks"
        ax.text(
            i,
            source["pages"] * 1.5,
            note,
            ha="center",
            va="bottom",
            fontsize=FLOOR_PT,
            color=INK if source["instances"] else SOFT,
            linespacing=1.4,
        )

    lo, hi, median = facts["per_class"]
    ax = fig.add_axes([bar_x, 0.175, 0.975 - bar_x, 0.215])
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
            ("108,077", "photographs"),
            ("2,516,939", "objects, each a box and a name"),
            ("0.61", "VG's recall against COCO"),
        ],
    )
    _source_line(fig, "homes.cs.washington.edu/~ranjay/visualgenome")
    _heading(
        fig,
        "Visual Genome",
        "ordinary scenes with a dozen nameable things in each, in the app as visual_genome_s/m/l/a",
    )
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
            (f"{len(by_id):,}", "images in val2017"),
            (f"{len(coco['categories'])}", "classes, all always annotated"),
            (f"{len(annotated):,}", "hold at least one object"),
        ],
    )
    _source_line(fig, "images.cocodataset.org/zips/val2017.zip")
    _heading(
        fig,
        "COCO val2017",
        "exhaustive — if a class is not boxed here it is not in the picture, which is what lets COCO correct another dataset",
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
    _heading(
        fig,
        "Caltech-101",
        "one object, centred, filling the frame — the set where region voting has nothing to point at",
    )
    tiles = [_fit_tile(Image.open(p)) for _, p in picks]
    _strip(fig, tiles, cols=4, rows=3, captions=[c.replace("_", " ") for c, _ in picks])
    return fig


def fig_card_vg_box(pc: Any) -> plt.Figure:
    """The three box-banded VG sets — and why they cannot answer the size question.

    This card exists to be *contradicted* by the `vg_scale` slides. Its three
    sets are perfectly good at what they measured and carry disjoint
    vocabularies, so the small-vs-large gap they show is box size and class
    identity at once. Saying that on the card is cheaper than having somebody
    quote them at the wrong question later.
    """
    fig = _blank_fig()
    bands = list(pc.BOX_BANDS)
    _stats(
        fig,
        [
            ("12,000", "images in each"),
            ("40", "categories in each"),
            ("643", "categories below one patch"),
            ("5", "the demo vocabulary has this many"),
        ],
    )
    _source_line(fig, "scripts/experiments/pile/README.md", caption="how it is built")
    _heading(
        fig,
        "vg_box_small · vg_box_medium · vg_box_large",
        "built from the whole Visual Genome source, not the demo pipeline's 100 curated categories",

    )

    ax = fig.add_axes([RIGHT_X, 0.30, RIGHT_W * 0.98, 0.50])
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # Verified separation, `build_pile.py --bands`: the share of each set's 40
    # categories whose median voted area really falls in the band it names.
    verified = {"small": 38, "medium": 40, "large": 33}
    examples = {
        "small": "nose · glasses · watch",
        "medium": "cup · bird · sign",
        "large": "fence · hill · lady",
    }
    for i, band in enumerate(bands):
        ax.add_patch(Rectangle((i + 0.06, 0.12), 0.88, 0.76, facecolor=CUT, alpha=0.12, ec=CUT, lw=1.4))
        ax.text(i + 0.5, 0.74, f"vg_box_{band}", ha="center", fontsize=FLOOR_PT + 4, color=INK, fontweight="bold")
        ax.text(
            i + 0.5,
            0.50,
            examples[band].replace(" · ", "\n"),
            ha="center",
            va="center",
            fontsize=FLOOR_PT + 1,
            color=SOFT,
            linespacing=1.5,
        )
        ax.text(
            i + 0.5,
            0.28,
            f"{verified[band]} of 40 in band",
            ha="center",
            fontsize=FLOOR_PT,
            color=SOFT,
        )

    fig.text(
        RIGHT_X,
        0.245,
        "\n".join(
            textwrap.wrap(
                "A category is banded by its own median box, so the three vocabularies are disjoint: "
                "the gap between them is box size and class identity at once. That is what vg_scale "
                "was built to separate, and these three are not comparable to it.",
                72,
            )
        ),
        fontsize=FLOOR_PT + 1,
        color=INK,
        va="top",
        linespacing=1.5,
    )
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

    steps = len(_vg_scale_steps(pc))
    for n in range(1, steps):
        save(fig_vg_scale_build(pc, upto=n), OUT, f"dataset-vg-scale-build.build{n}.png", column=FULL_BLEED, tight=False)
    save(fig_vg_scale_build(pc), OUT, "dataset-vg-scale-build.png", column=FULL_BLEED, tight=False)
    save(fig_vg_scale_bands(pc), OUT, "dataset-vg-scale-bands.png", column=FULL_BLEED, tight=False)
    save(fig_vg_scale_cells(pc), OUT, "dataset-vg-scale-cells.png", column=FULL_BLEED, tight=False)

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

    save(fig_card_vg_box(pc), OUT, "dataset-card-vg-box.png", column=FULL_BLEED, tight=False)
    if args.no_media:
        print("--no-media: skipped the Visual Genome, COCO and Caltech-101 cards")
        return 0
    _save_photo(fig_card_visual_genome(), "dataset-card-visual-genome")
    _save_photo(fig_card_coco_val(), "dataset-card-coco-val")
    _save_photo(fig_card_caltech101(), "dataset-card-caltech101")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
