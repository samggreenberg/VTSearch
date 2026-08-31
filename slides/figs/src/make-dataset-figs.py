#!/usr/bin/env python
"""Dataset figures for the VTSearch decks — `vg_scale` and DocMarks.

Run from the repo root:

    python slides/figs/src/make-dataset-figs.py --docmarks-corpus /path/to/corpus

Four figures, two per dataset: what the thing *is*, and how much of it there
is. They sit in the Hold The Line appendix, so they are answering a question
from the room rather than carrying an argument — which sets the bar: every
number on them has to be one the asker can check, and none of them may be
rounded into vagueness.

**Where the numbers come from.** The `vg_scale` figures read
`scripts/experiments/pile/pile_config.py` directly rather than restating it, so
a slide cannot drift from the constants the pile actually builds against. The
DocMarks figures read a built corpus's own `corpus.jsonl` and `classes.json`.
Neither figure has a number typed into it twice.

**The DocMarks overview uses real scans, not a schematic.** The whole point of
that slide is that the audience has never seen a stamp-detection corpus and
does not know what one looks like; a diagram of a rectangle labelled "mark"
would tell them nothing they could not have guessed. It is the one figure here
that has to be photographs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics
import sys
from collections import Counter
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

#: The notch, as axes fractions of the 1280x720 slot, so a layout can be
#: written against it instead of against pixel arithmetic repeated four times.
_nx, _ny, _nw, _nh = TITLE_NOTCH_PX
NOTCH_R = (_nx + _nw) / 1280.0  # right edge of the reserve
NOTCH_B = (_ny + _nh) / 720.0  # bottom edge, measured from the top

CUT = "#2b6cb0"  # the deck's .cut blue
NEG = "#c0392b"  # .neg red
POS = "#2e7d51"  # .pos green


def _blank_fig() -> plt.Figure:
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")
    return fig


def _pile_config() -> Any:
    path = REPO / "scripts" / "experiments" / "pile" / "pile_config.py"
    spec = importlib.util.spec_from_file_location("_slides_pile_config", path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# --------------------------------------------------------------------------
# vg_scale
# --------------------------------------------------------------------------


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
        ax = fig.add_axes([0.06 + i * 0.31, 0.09, 0.26, 0.45])
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
        upper = "one patch" if name == "small" else ("HAC leaf" if name == "medium" else "80% of frame")
        ax.set_title(name, fontsize=FLOOR_PT + 8, color=INK, pad=12, fontweight="bold")
        ax.text(
            0.5,
            -0.10,
            f"up to 1/{1 / hi:.0f}" if hi < 0.5 else f"up to {hi:.0%}",
            transform=ax.transAxes,
            ha="center",
            fontsize=FLOOR_PT + 2,
            color=INK,
        )
        ax.text(
            0.5,
            -0.19,
            f"({upper})",
            transform=ax.transAxes,
            ha="center",
            fontsize=FLOOR_PT,
            color=SOFT,
        )

    # The class list, right of the reserve so the corner stays clear.
    names = list(pc.SCALE_CLASSES)
    fig.text(
        NOTCH_R + 0.04,
        0.92,
        f"the same {len(names)} classes in every band",
        fontsize=FLOOR_PT + 4,
        color=INK,
        fontweight="bold",
        va="top",
    )
    columns = 4
    per = -(-len(names) // columns)
    for i, name in enumerate(names):
        col, row = divmod(i, per)
        fig.text(
            NOTCH_R + 0.05 + col * 0.135,
            0.855 - row * 0.048,
            name,
            fontsize=FLOOR_PT,
            color=SOFT,
            va="top",
        )
    return fig


def fig_vg_scale_cells(pc: Any) -> plt.Figure:
    """36 cells, every one the same size and the same prevalence.

    The uniformity *is* the design: identical prevalence in all 36 makes
    small-vs-large a paired comparison instead of two datasets of different
    difficulty, which is the failure that made two earlier benchmark waves
    non-comparable. A grid of identical tiles is the honest picture of that —
    there is no variation to plot.
    """
    fig = _blank_fig()
    classes = list(pc.SCALE_CLASSES)
    bands = list(pc.BOX_BANDS)
    n_pos, n_neg = pc.SCALE_N_POS, pc.SCALE_N_NEG

    ax = fig.add_axes([0.075, 0.16, 0.90, 0.46])
    ax.set_xlim(-0.5, len(classes) - 0.5)
    ax.set_ylim(-0.5, len(bands) - 0.5)
    ax.axis("off")

    for r, band in enumerate(bands):
        for c in range(len(classes)):
            ax.add_patch(
                Rectangle(
                    (c - 0.42, r - 0.40),
                    0.84,
                    0.80,
                    facecolor=CUT,
                    alpha=0.18,
                    ec=CUT,
                    lw=1.2,
                )
            )
        ax.text(-0.85, r, band, ha="right", va="center", fontsize=FLOOR_PT + 3, color=INK)
    for c, name in enumerate(classes):
        ax.text(
            c,
            -0.62,
            name,
            ha="right",
            va="top",
            rotation=40,
            rotation_mode="anchor",
            fontsize=FLOOR_PT,
            color=SOFT,
        )

    fig.text(
        NOTCH_R + 0.04,
        0.93,
        f"{len(classes)} classes × {len(bands)} bands = {len(classes) * len(bands)} cells",
        fontsize=FLOOR_PT + 5,
        color=INK,
        fontweight="bold",
        va="top",
    )
    prevalence = n_pos / (n_pos + n_neg)
    for i, line in enumerate(
        [
            f"every cell: {n_pos} positives + {n_neg:,} negatives",
            f"prevalence {prevalence:.1%} in all {len(classes) * len(bands)}, by construction",
            f"+{pc.SCALE_N_NEG_SPARE} spare negatives, designated into no cell",
        ]
    ):
        fig.text(NOTCH_R + 0.04, 0.855 - i * 0.055, line, fontsize=FLOOR_PT + 1, color=SOFT, va="top")
    return fig


# --------------------------------------------------------------------------
# DocMarks
# --------------------------------------------------------------------------


def _load_docmarks(corpus: Path) -> tuple[list[dict], dict[str, Any]]:
    pages = [json.loads(line) for line in (corpus / "corpus.jsonl").read_text().splitlines() if line.strip()]
    classes = json.loads((corpus / "classes.json").read_text())
    return pages, classes


def fig_docmarks_overview(corpus: Path, seed: int = 5) -> plt.Figure:
    """A real page with its marks boxed, beside real crops of the marks.

    Two things the room cannot get from a number: how *small* a mark is against
    the page it sits on, and how varied the marks are. The page is shown whole
    for the first and the crops are shown at a readable size for the second,
    which is the only honest way round — a gallery of crops alone would make
    the task look far easier than it is.
    """
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    pages, classes = _load_docmarks(corpus)
    by_id = {p["page_id"]: p for p in pages}
    rng = random.Random(seed)

    # A page carrying as many different kinds as possible, so one picture shows
    # the whole annotation vocabulary at once.
    best = max(
        (p for p in pages if p["source"] == "spods"),
        key=lambda p: (len({m["kind"] for m in p["marks"]}), len(p["marks"])),
    )

    # The page shares the notch's x-range, so its caption and legend go
    # *below* it rather than above: a title over this image would sit squarely
    # in the reserve, and the standard is that the figure moves, not the
    # slide's headline.
    fig = _blank_fig()
    ax = fig.add_axes([0.045, 0.145, 0.24, 0.50])
    with Image.open(best["path"]) as im:
        ax.imshow(im.convert("RGB"))
    for mark in best["marks"]:
        x, y, w, h = mark["box"]
        colour = {"logo": CUT, "stamp": NEG, "signature": POS}.get(mark["kind"], SOFT)
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=colour, lw=2.4))
    ax.axis("off")

    fig.text(
        0.045,
        0.105,
        f"one page · {best['width']}×{best['height']}",
        fontsize=FLOOR_PT,
        color=SOFT,
        va="center",
    )
    for i, (kind, colour) in enumerate((("logo", CUT), ("stamp", NEG), ("signature", POS))):
        fig.text(0.045 + i * 0.083, 0.048, "■", fontsize=FLOOR_PT + 2, color=colour, va="center")
        fig.text(0.064 + i * 0.083, 0.050, kind, fontsize=FLOOR_PT, color=SOFT, va="center")

    # Crops: one per class, largest instance, so each tile is a different mark.
    ordered = sorted(classes.items(), key=lambda kv: -kv[1]["n_instances"])
    tiles: list[Any] = []
    for _cid, meta in ordered:
        page = by_id.get(meta["page_ids"][0])
        if page is None:
            continue
        mark = next((m for m in page["marks"] if m["class_id"] == _cid), None)
        if mark is None:
            continue
        x, y, w, h = mark["box"]
        pad = int(round(max(w, h) * 0.08))
        with Image.open(page["path"]) as im:
            tiles.append(im.convert("RGB").crop((max(0, x - pad), max(0, y - pad), x + w + pad, y + h + pad)))
        if len(tiles) >= 24:
            break
    rng.shuffle(tiles)

    cols, rows = 8, 3
    for i, tile in enumerate(tiles[: cols * rows]):
        r, c = divmod(i, cols)
        ax = fig.add_axes([0.315 + c * 0.0855, 0.60 - r * 0.195, 0.078, 0.175])
        ax.imshow(tile)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#d5d9df")

    fig.text(
        0.315,
        0.90,
        f"{len(tiles[: cols * rows])} of the {len(classes)} marks, one instance each",
        fontsize=FLOOR_PT + 3,
        color=INK,
        fontweight="bold",
        va="top",
    )
    fig.text(
        0.315,
        0.845,
        "every one of these appears on ~30 different pages",
        fontsize=FLOOR_PT,
        color=SOFT,
        va="top",
    )
    return fig


def fig_docmarks_counts(corpus: Path) -> plt.Figure:
    """How many, how big, and how evenly the instances fall across classes."""
    pages, classes = _load_docmarks(corpus)

    queryable = [m for p in pages for m in p["marks"] if m["kind"] in ("logo", "stamp")]
    sides = sorted(max(m["box"][2], m["box"][3]) for m in queryable)
    fracs = sorted(
        (m["box"][2] * m["box"][3]) / (p["width"] * p["height"])
        for p in pages
        for m in p["marks"]
        if m["kind"] in ("logo", "stamp")
    )
    kinds = Counter(m["kind"] for p in pages for m in p["marks"])
    by_source = Counter(p["source"] for p in pages)
    sizes = sorted((v["n_instances"] for v in classes.values()), reverse=True)

    fig = _blank_fig()

    # Counts, under the reserve so the corner stays clear.
    lines = [
        (f"{by_source.get('spods', 0):,}", "SPODS pages"),
        (f"{by_source.get('ucsf', 0):,}", "UCSF distractor pages"),
        (f"{kinds.get('logo', 0):,}", "logo marks"),
        (f"{kinds.get('stamp', 0):,}", "stamp marks"),
        (f"{len(classes)}", "classes, ≥10 each"),
        (f"{statistics.median(fracs):.2%}", "median mark / page"),
    ]
    # The stat block sits under the reserve, in the left column the two upper
    # panels leave free. Six rows fit between the notch's bottom edge and the
    # foot of the slide with room to breathe; a seventh would not, so if this
    # list grows something else has to give.
    for i, (value, label) in enumerate(lines):
        y = 0.625 - i * 0.098
        fig.text(0.05, y, value, fontsize=FLOOR_PT + 8, color=INK, fontweight="bold", va="center")
        fig.text(0.155, y, label, fontsize=FLOOR_PT, color=SOFT, va="center")

    # Size distribution, against the floor the structural pipeline actually has.
    bands = [(32, 64), (64, 128), (128, 256), (256, 512), (512, 1024)]
    counts = [sum(1 for s in sides if lo <= s < hi) for lo, hi in bands]
    ax = fig.add_axes([0.40, 0.63, 0.24, 0.25])
    ax.bar(range(len(bands)), counts, color=CUT, width=0.72)
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([f"{lo}–{hi}" for lo, hi in bands], fontsize=FLOOR_PT, color=SOFT, rotation=30)
    ax.set_ylabel("marks", fontsize=FLOOR_PT, color=SOFT)
    ax.tick_params(axis="y", labelsize=FLOOR_PT, colors=SOFT)
    ax.set_title("longest side, px", fontsize=FLOOR_PT + 2, color=INK, pad=26)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    below = sum(1 for s in sides if s < 32)
    ax.text(
        0.0,
        1.03,
        f"{below} below the 32px floor",
        transform=ax.transAxes,
        fontsize=FLOOR_PT,
        color=POS if below == 0 else NEG,
    )

    # Instances per class: the regularity is the finding.
    ax = fig.add_axes([0.735, 0.63, 0.24, 0.25])
    ax.bar(range(len(sizes)), sizes, color=CUT, width=0.85)
    ax.set_xlabel("class, largest first", fontsize=FLOOR_PT, color=SOFT)
    ax.set_ylabel("instances", fontsize=FLOOR_PT, color=SOFT)
    ax.tick_params(labelsize=FLOOR_PT, colors=SOFT)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    modal = Counter(sizes).most_common(1)[0]
    ax.axhline(modal[0], color=NEG, lw=1.4, ls="--")
    ax.set_title("instances per class", fontsize=FLOOR_PT + 2, color=INK, pad=26)
    ax.text(
        0.0,
        1.03,
        f"{modal[1]} classes hold exactly {modal[0]}",
        transform=ax.transAxes,
        fontsize=FLOOR_PT,
        color=NEG,
    )

    # Page-area distribution, the number that makes this a small-object task.
    ax = fig.add_axes([0.44, 0.135, 0.535, 0.26])
    ax.hist([f * 100 for f in fracs], bins=44, color=CUT)
    ax.set_xlabel("mark size, % of the page it sits on", fontsize=FLOOR_PT, color=SOFT)
    ax.set_ylabel("marks", fontsize=FLOOR_PT, color=SOFT)
    ax.tick_params(labelsize=FLOOR_PT, colors=SOFT)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    median = statistics.median(fracs) * 100
    ax.axvline(median, color=NEG, lw=1.6)
    ax.text(
        median + 0.06,
        ax.get_ylim()[1] * 0.86,
        f"median {median:.2f}%",
        fontsize=FLOOR_PT,
        color=NEG,
    )
    return fig


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--docmarks-corpus",
        type=Path,
        default=None,
        help="a built DocMarks corpus; omit to render only the vg_scale figures",
    )
    args = ap.parse_args()

    pc = _pile_config()
    save(fig_vg_scale_bands(pc), OUT, "dataset-vg-scale-bands.png", column=FULL_BLEED, tight=False)
    save(fig_vg_scale_cells(pc), OUT, "dataset-vg-scale-cells.png", column=FULL_BLEED, tight=False)

    if args.docmarks_corpus:
        save(
            fig_docmarks_overview(args.docmarks_corpus),
            OUT,
            "dataset-docmarks-overview.png",
            column=FULL_BLEED,
            tight=False,
        )
        save(
            fig_docmarks_counts(args.docmarks_corpus),
            OUT,
            "dataset-docmarks-counts.png",
            column=FULL_BLEED,
            tight=False,
        )
    else:
        print("no --docmarks-corpus given; skipped the DocMarks figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
