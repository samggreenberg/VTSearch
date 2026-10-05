#!/usr/bin/env python
"""How the app does today, one slide per path: the State of the App reviews.

    python slides/figs/src/make-sota-figs.py

Writes `figs/sota-photos.png` and `figs/sota-documents.png`, each with one build
stage. Both slides have the same two panels, so the room reads the second the
way it read the first:

* **left** — the returned set's F-beta over a session, one line per radio,
  each read off sessions run *at* that radio, so each line is scored at its
  own beta (and the three are not ranked against one another: an F at 1/4 and
  an F at 4 are different scores);
* **right** — the same three returned sets as precision against recall, which
  is the trade the radio sets, drawn on one pair of axes so the three can be
  compared, square and on one scale (#4533). On documents, where the per-step
  CSVs carry both at every click, each radio is a path through the session
  (#4517): a circle at each of `DOC_PR_CLICKS` with the click count written in
  it, joined in order, circle and path both in the radio's own weight (#4533).
  The photo slide gets the same paths once its review commits precision and
  recall per click (#4519); until then each radio is one circle, after the
  spot check, with the check's ✓ in it.

The photo slide is the binary-photo review (SigLIP, binary votes, COCO Better,
`docs/experiments/2026-10-04-state-of-the-app-binary-photo/`). Its per-run
data lives on the GRID, so the slide reads the report's own committed tables:
the headline table in `REPORT.md` and the per-beta summary beside it.

The document slide is FullMarks v5.0, tier `m`, 36 classes, 50 clicks, two
replicates, from the committed per-step CSVs of three runs of the app's own
path, one per radio: the document-logo review at beta 1
(`2026-10-03-state-of-the-app-document-logo-4457`), and the runs that shipped
the recall end (#4458) and the precision end (#4479).

**Nothing here is a guess**, as in `make-logo-line-fig.py`: `EXPECT` pins the
numbers the presenter notes quote, and `main` fails if the reports or CSVs say
anything else. A rerun of a review is a re-run of this script, then a look at
the notes.
"""

from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slide_figure import FULL_BLEED, INK, SOFT, save  # noqa: E402

SRC = Path(__file__).resolve().parent
OUT = SRC.parent
REPO = SRC.parents[2]
EXPERIMENTS = REPO / "docs" / "experiments"
PHOTO_REPORT = EXPERIMENTS / "2026-10-04-state-of-the-app-binary-photo"

#: The three radios, left panel's line weight and label, in the order
#: `calib-fbeta` stacks them: the precision end, the middle, the recall end.
#: The weights are that figure's `BALANCE_WEIGHTS`, so a radio is drawn the
#: same weight on every slide that draws it.
RADIOS: tuple[tuple[float, str, float], ...] = (
    (0.25, "β = ¼", 1.4),
    (1.0, "β = 1", 2.8),
    (4.0, "β = 4", 5.0),
)

#: The document runs, one per radio, each two replicates of the app's own path.
DOC_RUNS: dict[float, tuple[Path, ...]] = {
    0.25: tuple(
        EXPERIMENTS / "2026-10-04-precision-end-4479" / "measurements" / f"betaq-rep{r}-steps.csv" for r in (1, 2)
    ),
    1.0: tuple(
        EXPERIMENTS / "2026-10-03-state-of-the-app-document-logo-4457" / "measurements" / f"rep{r}" / "steps.csv"
        for r in (1, 2)
    ),
    4.0: tuple(
        EXPERIMENTS / "2026-10-03-balance-line-4458" / "measurements" / "round2" / f"beta4-rep{r}-steps.csv"
        for r in (1, 2)
    ),
}
#: Classes x replicates every document click must cover, and the clicks.
DOC_ROWS, DOC_CLICKS = 72, 50
#: The clicks the document slide's precision-recall paths put a circle at: the
#: query crop alone, then through the session to its last click. Not 25: every
#: radio has stopped moving by then, so its circle would sit on the one at 50
#: and cover its number (#4533).
DOC_PR_CLICKS = (0, 10, 50)

#: What the presenter notes quote, rounded as the notes say them.
EXPECT = {
    ("photos", 0.25, "after"): 0.62,
    ("photos", 1.0, "after"): 0.52,
    ("photos", 4.0, "after"): 0.60,
    ("photos", 0.25, "kept"): 26,
    ("photos", 1.0, "kept"): 47,
    ("photos", 4.0, "kept"): 80,
    ("photos", 0.25, "precision"): 0.70,
    ("photos", 4.0, "recall"): 0.68,
    ("documents", 0.25, "precision"): 0.98,
    ("documents", 1.0, "precision"): 0.93,
    ("documents", 4.0, "recall"): 0.95,
    ("documents", 0.25, "kept"): 13,
    ("documents", 1.0, "kept"): 14,
    ("documents", 4.0, "kept"): 25,
    ("documents", 1.0, "f50"): 0.87,
}

plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans"],
        # F-beta's subscript is set as mathtext, in the figure's own face, as on
        # F-ing Metrics (`make-calib-figs._sub`).
        "mathtext.fontset": "dejavusans",
        "font.size": 17,
        "text.color": INK,
        "axes.edgecolor": SOFT,
        "axes.labelcolor": INK,
        "axes.labelsize": 17,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": SOFT,
        "ytick.color": SOFT,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 200,
    }
)


def _fbeta(precision: float, recall: float, beta: float) -> float:
    """F-beta from a returned set's precision and recall; 0 for a set that returned nothing."""
    if not precision == precision or precision + recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def _table_rows(path: Path, first: str) -> list[list[str]]:
    """The cells of every markdown table row in *path* whose first cell matches *first*."""
    rows = []
    for line in path.read_text().splitlines():
        cells = [c.strip().strip("*") for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and re.fullmatch(first, cells[0]):
            rows.append(cells)
    return rows


def photo_data() -> dict[float, dict]:
    """Per radio: F-beta at 25, 50, 100 and 150 clicks and after the check; and the checked set.

    The curve is `REPORT.md`'s headline table; precision and recall after the
    check are `perbeta_summary.md`'s, which carries them to three places.
    """
    names = {"1/4": 0.25, "1": 1.0, "4": 4.0}
    head = {}
    for cells in _table_rows(PHOTO_REPORT / "REPORT.md", r"1/4|1|4"):
        # The headline table is the one whose rows read: preset, five Fs, a count, a share.
        if len(cells) == 8 and cells[7].endswith("%"):
            head[names[cells[0]]] = cells
    summary = {float(c[0]): c for c in _table_rows(PHOTO_REPORT / "perbeta_summary.md", r"0\.25|1|4") if len(c) == 12}
    if set(head) != {0.25, 1.0, 4.0} or set(summary) != {0.25, 1.0, 4.0}:
        raise SystemExit("make-sota-figs: the photo report's tables are not where this script reads them")
    out = {}
    for beta, cells in head.items():
        out[beta] = {
            "clicks": [25, 50, 100, 150],
            "f": [float(v) for v in cells[1:5]],
            "after": float(cells[5]),
            "kept": int(cells[6]),
            "precision": float(summary[beta][7]),
            "recall": float(summary[beta][8]),
        }
    return out


def doc_data() -> dict[float, dict]:
    """Per radio: mean returned-set F-beta at its own beta by click, and its set at every `DOC_PR_CLICKS`.

    The top-level precision, recall and kept are the session's last click,
    where the path ends and its label sits.
    """
    out = {}
    for beta, paths in DOC_RUNS.items():
        by_v: dict[int, list[tuple[float, float, float, int]]] = {}
        for path in paths:
            with path.open() as f:
                for row in csv.DictReader(f):
                    precision = float(row["gate_precision"]) if row["gate_precision"] not in ("", "nan") else math.nan
                    recall = float(row["gate_recall"])
                    by_v.setdefault(int(row["v"]), []).append(
                        (_fbeta(precision, recall, beta), precision, recall, int(row["gate_k"]))
                    )
        clicks = sorted(by_v)
        if clicks != list(range(DOC_CLICKS + 1)) or any(len(by_v[v]) != DOC_ROWS for v in clicks):
            raise SystemExit(f"make-sota-figs: the beta {beta:g} document runs do not cover every class and click")
        path = []
        for click in DOC_PR_CLICKS:
            at = np.array(by_v[click])
            path.append(
                {
                    # Precision is over the classes that returned something, as the reviews read it.
                    "precision": float(np.nanmean(at[:, 1])),
                    "recall": float(at[:, 2].mean()),
                    "kept": int(np.median(at[:, 3])),
                }
            )
        out[beta] = {
            "clicks": clicks,
            "f": [float(np.mean([r[0] for r in by_v[v]])) for v in clicks],
            "path": path,
            **path[-1],
        }
    return out


def _check(photos: dict, docs: dict) -> None:
    got = {}
    for (path, beta, key), want in EXPECT.items():
        data = photos if path == "photos" else docs
        value = data[beta]["f"][DOC_CLICKS] if key == "f50" else data[beta][key]
        got[(path, beta, key)] = value if isinstance(want, int) else round(value, 2)
    wrong = {k: (got[k], EXPECT[k]) for k in EXPECT if got[k] != EXPECT[k]}
    if wrong:
        raise SystemExit(f"make-sota-figs: the reviews no longer say what the notes quote (got, pinned): {wrong}")


def _spread(ys: list[float], gap: float) -> list[float]:
    """Nudge label heights apart by at least *gap*, keeping their order and mean."""
    order = np.argsort(ys)
    placed = np.array(ys, dtype=float)[order]
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + gap)
    placed += np.array(ys)[order].mean() - placed.mean()
    out = np.empty_like(placed)
    out[order] = placed
    return out.tolist()


#: The figure, in inches: the slide's own 16:9.
FIG_SIZE = (12.8, 7.2)
#: Both slides' panels, in figure fractions. The left starts below the title
#: notch's foot; the right sits beside it at the same height and is square,
#: because precision and recall run over the same span and a unit of one has to
#: be as long as a unit of the other (#4533).
PANEL_BOTTOM, PANEL_HEIGHT = 0.14, 0.58
RIGHT_WIDTH = PANEL_HEIGHT * FIG_SIZE[1] / FIG_SIZE[0]
RIGHT_AXES = (0.935 - RIGHT_WIDTH, PANEL_BOTTOM, RIGHT_WIDTH, PANEL_HEIGHT)
LEFT_AXES = (0.085, PANEL_BOTTOM, 0.37, PANEL_HEIGHT)
PR_LIM = (0.3, 1.0)

#: A circle on the precision-recall panel, per radio: the edge is the radio's
#: line weight from the left panel, and the type inside steps from light to
#: dark with it, so a circle reads as the same radio as the line it sits on.
CIRCLE_TEXT: dict[float, tuple[str, str]] = {
    0.25: (SOFT, "normal"),
    1.0: (INK, "normal"),
    4.0: (INK, "bold"),
}
#: The circle's diameter in points: room for a two-digit count at the type
#: floor, plus the heaviest radio's edge.
CIRCLE_PT = 30
CIRCLE_TEXT_PT = 15
#: The precision-recall grid: a faint line every step, so a set near the
#: top-right corner can still be read against something (#4533). Half a tick
#: on the documents' half-unit span; a whole one on the photos' wider span,
#: where half a tick would be fourteen lines a side.
DOC_GRID_STEP, PHOTO_GRID_STEP = 0.05, 0.1
GRID_COLOUR = "#e3e7ec"


def _pr_axes(fig: Figure, lim: tuple[float, float], ticks: list[float], step: float) -> plt.Axes:
    """The square precision-recall axes, gridded every *step* on both."""
    ax = fig.add_axes(RIGHT_AXES)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    lines = np.arange(lim[0], lim[1] + step / 2, step)
    ax.set_xticks(lines, minor=True)
    ax.set_yticks(lines, minor=True)
    ax.tick_params(which="minor", length=0)
    ax.grid(True, which="both", color=GRID_COLOUR, lw=1.0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    return ax


def _circle(ax: plt.Axes, x: float, y: float, beta: float, text: str) -> None:
    """One returned set on the precision-recall panel: a circle with *text* in it, in *beta*'s weight."""
    weight = next(w for b, _, w in RADIOS if b == beta)
    colour, font = CIRCLE_TEXT[beta]
    ax.plot(
        [x],
        [y],
        ls="none",
        marker="o",
        markersize=CIRCLE_PT,
        markerfacecolor="white",
        markeredgecolor=INK,
        markeredgewidth=weight,
        zorder=5,
        clip_on=False,
    )
    ax.text(x, y, text, ha="center", va="center_baseline", fontsize=CIRCLE_TEXT_PT, color=colour, fontweight=font, zorder=6)


#: How far right of a circle's centre its radio's name starts, in points.
CIRCLE_LABEL_PT = CIRCLE_PT / 2 + 6


def _pr_panel(fig: Figure, sets: dict[float, dict]) -> None:
    """Precision against recall, one circle per radio after the check, each labelled with how many it kept.

    The circle carries the check's ✓, as the left panel's last tick does: it
    is the set after the spot check, which is the one point per radio the
    photo review commits (#4519 brings the rest of the path).
    """
    ax = _pr_axes(fig, PR_LIM, [0.4, 0.6, 0.8, 1.0], PHOTO_GRID_STEP)
    xs = [sets[b]["recall"] for b, _, _ in RADIOS]
    ax.plot(xs, [sets[b]["precision"] for b, _, _ in RADIOS], color=SOFT, lw=1.2, zorder=2)
    for beta, label, _w in RADIOS:
        s = sets[beta]
        _circle(ax, s["recall"], s["precision"], beta, "✓")
        ax.annotate(
            f"{label} · {s['kept']} kept",
            (s["recall"], s["precision"]),
            xytext=(CIRCLE_LABEL_PT, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=16,
            color=INK,
            annotation_clip=False,
        )


#: The paths' axes: zoomed to where the document sets live, the same span on
#: both so precision and recall still share a scale.
PR_PATH_LIM = (0.5, 1.0)


def _pr_paths_panel(fig: Figure, sets: dict[float, dict]) -> None:
    """Precision against recall, one path per radio through the session's clicks.

    Each radio is drawn at the weight the left panel gives it, so a path and its
    F line are one object on two axes, and nothing joins the radios to one
    another: what the panel shows is how each one's set moves as votes arrive.
    Each of `DOC_PR_CLICKS` is a circle with its click count in it, so the
    points say when they are without a note to decode them (#4533), and each
    path is labelled where it ends; how many each returns is in the notes, since
    a count beside every end would not fit the panel.

    The middle and recall-end runs leave from one set, the query crop alone, so
    that circle is drawn once, by the first radio to reach it.
    """
    ax = _pr_axes(fig, PR_PATH_LIM, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], DOC_GRID_STEP)
    drawn: set[tuple[float, float]] = set()
    for beta, label, weight in RADIOS:
        path = sets[beta]["path"]
        xs = [point["recall"] for point in path]
        ys = [point["precision"] for point in path]
        ax.plot(xs, ys, color=INK, lw=weight, solid_capstyle="round", solid_joinstyle="round", zorder=3)
        for click, x, y in zip(DOC_PR_CLICKS, xs, ys, strict=True):
            if (round(x, 3), round(y, 3)) in drawn:
                continue
            drawn.add((round(x, 3), round(y, 3)))
            _circle(ax, x, y, beta, str(click))
        ax.annotate(
            label,
            (xs[-1], ys[-1]),
            xytext=(CIRCLE_LABEL_PT, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=16,
            color=INK,
            annotation_clip=False,
        )


def _f_panel(
    fig: Figure, lines: dict[float, tuple[list, list]], xlim, xticks, xlabel: str, floor: float, extra=None
) -> plt.Axes:
    """The returned set's F-beta over a session, one line per radio, labelled at its end.

    The axis starts at *floor*, the last 0.2 step under the lowest point the
    slide draws, rather than at 0: the panel is read for its shape (#4533).
    """
    ax = fig.add_axes(LEFT_AXES)
    ax.set_xlim(*xlim)
    ax.set_ylim(floor, 1.0)
    ax.set_xticks(xticks)
    ax.set_yticks(np.arange(floor, 1.0 + 1e-9, 0.2))
    ax.yaxis.grid(True, color=GRID_COLOUR, lw=1.0)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\mathregular{F_\beta}$ at that β")
    ends = {}
    for beta, label, weight in RADIOS:
        xs, ys = lines[beta]
        ax.plot(xs, ys, color=INK, lw=weight, solid_capstyle="round", zorder=3)
        ends[beta] = (xs[-1], ys[-1])
    if extra is not None:
        extra(ax, ends)
    label_y = _spread([ends[b][1] for b, _, _ in RADIOS], gap=0.075)
    for (beta, label, _w), y in zip(RADIOS, label_y, strict=True):
        x, yy = ends[beta]
        ax.annotate(
            label,
            (x, yy),
            xytext=(xlim[1] + (xlim[1] - xlim[0]) * 0.03, y),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=16,
            color=INK,
            annotation_clip=False,
            arrowprops={"arrowstyle": "-", "color": SOFT, "lw": 1.0, "shrinkA": 2, "shrinkB": 4},
        )
    return ax


def _floor(lines: dict[float, tuple[list, list]]) -> float:
    """The F panel's floor: the last 0.2 step at or under every point it draws."""
    lowest = min(min(ys) for _xs, ys in lines.values())
    return math.floor(round(lowest / 0.2, 6)) * 0.2


#: Where the photo slide draws "after the check": past the last click, joined
#: to it by a dotted step, because the check is not more clicks of the same kind.
PHOTO_CHECK_X = 168


def photo_figure(data: dict[float, dict], stage: int) -> Figure:
    fig = plt.figure(figsize=FIG_SIZE)
    lines = {b: (data[b]["clicks"] + [PHOTO_CHECK_X], data[b]["f"] + [data[b]["after"]]) for b in data}

    def check_step(ax: plt.Axes, _ends: dict) -> None:
        for beta, _label, _w in RADIOS:
            ax.plot([PHOTO_CHECK_X], [data[beta]["after"]], marker="o", markersize=8, color=INK, zorder=4)

    ax = _f_panel(
        fig,
        lines,
        (0, PHOTO_CHECK_X + 4),
        [25, 50, 100, 150, PHOTO_CHECK_X],
        "Clicks, then the spot check",
        _floor(lines),
        check_step,
    )
    ax.set_xticklabels(["25", "50", "100", "150", "✓"])
    if stage >= 2:
        _pr_panel(fig, data)
    return fig


def doc_figure(data: dict[float, dict], stage: int) -> Figure:
    fig = plt.figure(figsize=FIG_SIZE)
    lines = {b: (data[b]["clicks"], data[b]["f"]) for b in data}
    _f_panel(fig, lines, (0, DOC_CLICKS), [0, 10, 20, 30, 40, 50], "Clicks", _floor(lines))
    if stage >= 2:
        _pr_paths_panel(fig, data)
    return fig


def main() -> int:
    photos, docs = photo_data(), doc_data()
    _check(photos, docs)
    for stem, build in (("sota-photos", photo_figure), ("sota-documents", doc_figure)):
        data = photos if stem == "sota-photos" else docs
        # Saved at the canvas's declared bounds, so both stages frame the
        # panels identically and the top margin keeps the title notch empty.
        save(build(data, 2), OUT, f"{stem}.png", column=FULL_BLEED, tight=False)
        save(build(data, 1), OUT, f"{stem}.build1.png", column=FULL_BLEED, tight=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
