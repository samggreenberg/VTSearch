#!/usr/bin/env python
"""Slide-scale redraws of the overview-benchmark figures.

Run from the repo root:

    python slides/figs/src/make-bench-figs.py

The four figures these produce were previously used on slides as-is, straight
out of `docs/experiments/overview-bench/figures/`. That does not work, and the
arithmetic says why. A report figure is ~12.8in wide with 10pt tick labels;
dropped into a `bg right:56%` slot it is displayed 717px wide, so each printed
point lands on 0.78 slide pixels and a 10pt label reads at **8px** — against
28px body copy on the same slide. Six panels, six legends, all of it unreadable
past the second row of the room.

So the numbers are read back out of the committed report SVGs (see
`report_svg.py`) and redrawn under two rules:

* **A type floor.** Nothing renders below `TYPE_FLOOR_PX` slide pixels, checked
  by `enforce_type_floor` rather than eyeballed — it fails the build, so a
  later edit that shrinks a label cannot slip through.
* **Size to the slot, not to the page.** A `bg right:56%` box is 717x720, very
  nearly square; a 2:1 report figure fills half of it and wastes the rest. The
  6-panel figures are therefore 3 rows x 2 cols here, not 2 x 3.

Everything else is subtraction: one shared legend instead of one per panel, no
suptitle (the slide's own headline says it), no footnote (that is a presenter
note). The numbers are untouched.

`--check` re-reads every report figure without writing PNGs, so a move or a
re-render on the report side surfaces as a failure here rather than as a stale
slide nobody looked at.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_svg  # noqa: E402
from slide_figure import FULL_BLEED, INK, SIDEBAR, SOFT, save  # noqa: E402

SRC = Path(__file__).resolve().parent
OUT = SRC.parent
REPO = SRC.parents[2]
REPORT = REPO / "docs" / "experiments" / "overview-bench"
FIGURES = REPORT / "figures"

#: One colour per representation, matching the published report exactly so a
#: slide and the report it summarises cannot disagree about which line is which.
COLOR = {
    "siglip": "#3B6EA8",
    "siglip2_l": "#D08428",
    "dinov3_patch": "#3E8A6E",
    "binary": "#B3574A",
}
CEILING = "#8b93a0"

plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans"],
        "font.size": 15,
        "text.color": INK,
        "axes.edgecolor": SOFT,
        "axes.labelcolor": INK,
        "axes.labelsize": 15,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": SOFT,
        "ytick.color": SOFT,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 200,
    }
)


# --- reading the report ------------------------------------------------------


def report_panels(figure: str) -> list[report_svg.Panel]:
    """Panels read back out of one committed report SVG."""
    return report_svg.read_panels(str(FIGURES / f"{figure}.svg"))


def decomposition_rows() -> dict[str, dict[str, tuple[float, float]]]:
    """(calibration shift, rule inefficiency) per dataset per embedder.

    Read from the committed analysis tables rather than the figure, because
    here the published numbers exist as text and a figure read back would only
    be a lossier copy of them.
    """
    wanted = "WHERE THE REGRET COMES FROM"
    rows: dict[str, dict[str, tuple[float, float]]] = {}
    for name in ("ANALYSIS_TABLES.txt", "ANALYSIS_TABLES_vgbox2.txt"):
        section = (REPORT / name).read_text().split(wanted, 1)[1]
        started = False
        for line in section.splitlines():
            # Stop at the next banner: later tables in the same file are also
            # `<dataset> x <embedder>` rows and would otherwise overwrite these.
            if line.startswith("=" * 20) and started:
                break
            match = re.match(r"^(\S+) x (\S+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)", line.strip())
            if match:
                dataset, embedder, _regret, rule, shift = match.groups()
                rows.setdefault(dataset, {})[embedder] = (float(shift), float(rule))
                started = True
    return rows


# --- the figures -------------------------------------------------------------

DATASETS = ["caltech101_m", "coco_val", "visual_genome_m", "vg_box_large", "vg_box_medium", "vg_box_small"]
EMBEDDERS = ["siglip", "siglip2_l", "dinov3_patch"]


def positives_fig() -> None:
    """Positives accumulated against votes, one panel per dataset."""
    panels = {p.title: p for p in report_panels("fig_positives")}
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.0), sharex=True, sharey=True)

    for ax, dataset in zip(axes.ravel(), DATASETS):
        panel = panels[dataset]
        ceiling = panel.lines["every vote a positive"]
        ax.plot(*zip(*ceiling), color=CEILING, linestyle=(0, (1.5, 2)), linewidth=1.6, zorder=1)
        for embedder in EMBEDDERS:
            ax.plot(*zip(*panel.lines[embedder]), color=COLOR[embedder], linewidth=2.0, zorder=3)
        ax.set_title(dataset, pad=6)
        ax.set_yscale("log")
        ax.set_ylim(0.8, 220)
        ax.set_xlim(0, 152)
        ax.set_xticks([0, 50, 100, 150])
        ax.set_yticks([1, 10, 100])
        ax.set_yticklabels(["1", "10", "100"])
        ax.grid(axis="y", color="#e6eaee", linewidth=0.8)
        ax.set_axisbelow(True)

    fig.supxlabel("votes cast", fontsize=15)
    fig.supylabel("positives held (median)", fontsize=15, x=0.005)

    handles = [plt.Line2D([], [], color=COLOR[e], linewidth=2.4) for e in EMBEDDERS]
    handles.append(plt.Line2D([], [], color=CEILING, linestyle=(0, (1.5, 2)), linewidth=1.8))
    fig.legend(
        handles,
        [*EMBEDDERS, "every vote a positive"],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
        handlelength=1.8,
    )
    fig.tight_layout(rect=(0.02, 0, 1, 0.93))
    save(fig, OUT, "positive-starvation.png")


def _split_traces(panel: report_svg.Panel) -> tuple[list, list, list]:
    """(individual runs, median, mean) — told apart by how they were stroked."""
    runs, median, mean = [], [], []
    for label, points in panel.lines.items():
        style = panel.styles[label]
        if style.get("stroke") == "#000000":
            median = points
        elif style.get("stroke-dasharray"):
            mean = points
        else:
            runs.append(points)
    return runs, median, mean


def _cost_panel(ax: plt.Axes, panel: report_svg.Panel, embedder: str) -> None:
    runs, median, mean = _split_traces(panel)
    for points in runs:
        ax.plot(*zip(*points), color=COLOR[embedder], linewidth=0.7, alpha=0.4, zorder=1)
    if mean:
        ax.plot(*zip(*mean), color=COLOR[embedder], linewidth=2.4, linestyle=(0, (5, 2)), zorder=3)
    if median:
        ax.plot(*zip(*median), color=INK, linewidth=2.4, zorder=4)
    for points in panel.markers.values():
        ax.scatter(*zip(*points), color=COLOR["binary"], s=26, zorder=5)
    ax.set_ylim(0, 1.28)
    ax.set_xlim(0, 152)
    ax.set_xticks([0, 50, 100, 150])
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(axis="y", color="#e6eaee", linewidth=0.8)
    ax.set_axisbelow(True)


def cost_traces_fig() -> None:
    """Every individual run, laid out dataset (columns) by embedder (rows)."""
    panels = {p.title: p for p in report_panels("fig_cost_traces")}
    layout = [("visual_genome_m", e) for e in EMBEDDERS], [("vg_box_small", e) for e in EMBEDDERS]
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.0), sharex=True, sharey=True)

    for column, cells in enumerate(layout):
        for row, (dataset, embedder) in enumerate(cells):
            ax = axes[row][column]
            title = next(t for t in panels if t.startswith(f"{dataset} x {embedder} "))
            _cost_panel(ax, panels[title], embedder)
            if row == 0:
                ax.set_title(dataset, pad=8)
            if column == 0:
                ax.set_ylabel(embedder, labelpad=8)

    fig.supxlabel("votes cast", fontsize=15)

    handles = [
        plt.Line2D([], [], color=INK, linewidth=2.4),
        plt.Line2D([], [], color=SOFT, linewidth=2.4, linestyle=(0, (5, 2))),
        plt.Line2D([], [], color=SOFT, linewidth=1.0, alpha=0.6),
        plt.Line2D([], [], color=COLOR["binary"], marker="o", linestyle="none", markersize=6),
    ]
    fig.legend(
        handles,
        ["median run", "mean", "one run", "first scored step"],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
        handlelength=1.8,
    )
    fig.supylabel("cost = FPR + FNR", fontsize=15, x=0.005)
    fig.tight_layout(rect=(0.02, 0, 1, 0.93))
    save(fig, OUT, "cost-traces.png")


BOX_SERIES = [
    ("siglip (no box possible)", "siglip", "siglip — binary", "solid"),
    ("siglip2_l (no box possible)", "siglip2_l", "siglip2_l — binary", "solid"),
    ("dinov3_patch — box NOT drawn", "binary", "dinov3_patch — binary", "solid"),
    ("dinov3_patch — box drawn", "dinov3_patch", "dinov3_patch — region voting", "dashed"),
]


def region_voting_fig(name: str, *, rows: int, size: tuple[float, float], column: float) -> None:
    """The same cells with and without a drawn box, stacked or side by side."""
    panels = {p.title: p for p in report_panels("fig_binary_vs_boxes")}
    order = ["visual_genome_m", "coco_val"]
    cols = 1 if rows == 2 else 2
    fig, axes = plt.subplots(rows, cols, figsize=size, sharex=True, squeeze=False)

    for ax, dataset in zip(axes.ravel(), order):
        panel = panels[dataset]
        for source, key, _label, dash in BOX_SERIES:
            ax.plot(
                *zip(*panel.lines[source]),
                color=COLOR[key],
                linewidth=2.6 if dash == "dashed" else 1.8,
                linestyle=(0, (5, 2)) if dash == "dashed" else "solid",
                zorder=4 if dash == "dashed" else 2,
            )
        ax.set_title(dataset, pad=6)
        ax.set_xlim(0, 152)
        ax.margins(y=0.08)
        ax.grid(axis="y", color="#e6eaee", linewidth=0.8)
        ax.set_axisbelow(True)
    for ax in axes[-1] if rows == 2 else axes[0]:
        ax.set_xlabel("votes cast")
    fig.supylabel("cost = FPR + FNR", fontsize=15, x=0.005)

    handles = [
        plt.Line2D(
            [],
            [],
            color=COLOR[key],
            linewidth=2.6 if dash == "dashed" else 1.8,
            linestyle=(0, (5, 2)) if dash == "dashed" else "solid",
        )
        for _source, key, _label, dash in BOX_SERIES
    ]
    fig.legend(
        handles,
        [label for _source, _key, label, _dash in BOX_SERIES],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
        handlelength=2.0,
    )
    fig.tight_layout(rect=(0.03, 0, 1, 0.88 if rows == 2 else 0.86))
    save(fig, OUT, name, column)


def regret_fig() -> None:
    """Where the deep-regime regret goes: the sim->test move, and the cut rule."""
    rows = decomposition_rows()
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True, sharey=True)
    positions = np.arange(len(DATASETS))
    height = 0.26

    for index, (ax, slot, title) in enumerate(
        zip(
            axes,
            (0, 1),
            ("The sim → test move", "The cut rule's own cost"),
        )
    ):
        for offset, embedder in enumerate(EMBEDDERS):
            values = [rows[d][embedder][slot] for d in DATASETS]
            ax.barh(
                positions + (offset - 1) * height,
                values,
                height=height * 0.92,
                color=COLOR[embedder],
                zorder=3,
            )
        ax.set_title(title, pad=6)
        ax.set_yticks(positions)
        ax.set_yticklabels(DATASETS)
        ax.axvline(0, color=SOFT, linewidth=1.0, zorder=4)
        ax.grid(axis="x", color="#e6eaee", linewidth=0.8)
        ax.set_axisbelow(True)
        if index == 1:
            ax.set_xlabel("cost units")

    # Both panels share one axis: that the left column is several times the
    # right is the finding, and separate scales would hide it.
    axes[0].invert_yaxis()
    handles = [plt.Line2D([], [], color=COLOR[e], linewidth=7) for e in EMBEDDERS]
    fig.legend(handles, EMBEDDERS, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, OUT, "regret-decomposition.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--check", action="store_true", help="re-derive everything, write no PNGs")
    args = parser.parse_args()

    if args.check:
        for figure in ("fig_positives", "fig_cost_traces", "fig_binary_vs_boxes"):
            report_svg.read_panels(str(FIGURES / f"{figure}.svg"))
        decomposition_rows()
        print("report figures still readable")
        return 0

    positives_fig()
    cost_traces_fig()
    region_voting_fig("region-voting-vs-binary.png", rows=2, size=(7.2, 7.0), column=SIDEBAR)
    region_voting_fig("region-voting-vs-binary.wide.png", rows=1, size=(11.6, 6.4), column=FULL_BLEED)
    regret_fig()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
