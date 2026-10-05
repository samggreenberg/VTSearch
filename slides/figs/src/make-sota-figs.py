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
  compared.

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
#: The click the document slide's right panel is read at.
DOC_PR_CLICK = 25

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
    ("documents", 1.0, "precision"): 0.90,
    ("documents", 4.0, "recall"): 0.94,
    ("documents", 0.25, "kept"): 13,
    ("documents", 1.0, "kept"): 14,
    ("documents", 4.0, "kept"): 26,
    ("documents", 1.0, "f50"): 0.87,
}

plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans"],
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
    """Per radio: mean returned-set F-beta at its own beta by click, and the set at `DOC_PR_CLICK`."""
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
        at = np.array(by_v[DOC_PR_CLICK])
        out[beta] = {
            "clicks": clicks,
            "f": [float(np.mean([r[0] for r in by_v[v]])) for v in clicks],
            # Precision is over the classes that returned something, as the reviews read it.
            "precision": float(np.nanmean(at[:, 1])),
            "recall": float(at[:, 2].mean()),
            "kept": int(np.median(at[:, 3])),
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


#: Both slides' panels, in figure fractions: the left starts below the title
#: notch's foot and the right sits beside it, square-ish, so precision and
#: recall share a scale.
LEFT_AXES = (0.085, 0.14, 0.43, 0.58)
RIGHT_AXES = (0.68, 0.14, 0.27, 0.58)
PR_LIM = (0.3, 1.0)


def _pr_panel(fig: Figure, sets: dict[float, dict], note: str) -> None:
    """Precision against recall, one dot per radio, each labelled with how many it kept."""
    ax = fig.add_axes(RIGHT_AXES)
    ax.set_xlim(*PR_LIM)
    ax.set_ylim(*PR_LIM)
    ticks = [0.4, 0.6, 0.8, 1.0]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid(True, color="#e3e7ec", lw=1.0)
    ax.set_axisbelow(True)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    xs = [sets[b]["recall"] for b, _, _ in RADIOS]
    ax.plot(xs, [sets[b]["precision"] for b, _, _ in RADIOS], color=SOFT, lw=1.2, zorder=2)
    for beta, label, _w in RADIOS:
        s = sets[beta]
        ax.plot([s["recall"]], [s["precision"]], marker="o", markersize=11, color=INK, zorder=4)
        # Labelled on whichever side has room inside the panel: a dot in its
        # left half would push a left-hand label across the precision axis.
        right = s["recall"] < sum(PR_LIM) / 2 + 0.05
        ax.annotate(
            f"{label} · {s['kept']} kept",
            (s["recall"], s["precision"]),
            xytext=(12 if right else -12, 0),
            textcoords="offset points",
            ha="left" if right else "right",
            va="center",
            fontsize=16,
            color=INK,
        )
    ax.set_title(note, fontsize=16, color=SOFT, loc="left", pad=10)


def _f_panel(fig: Figure, lines: dict[float, tuple[list, list]], xlim, xticks, xlabel: str, extra=None) -> plt.Axes:
    """The returned set's F-beta over a session, one line per radio, labelled at its end."""
    ax = fig.add_axes(LEFT_AXES)
    ax.set_xlim(*xlim)
    ax.set_ylim(0.2, 1.0)
    ax.set_xticks(xticks)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.yaxis.grid(True, color="#e3e7ec", lw=1.0)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("returned set, F at its own β")
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


#: Where the photo slide draws "after the check": past the last click, joined
#: to it by a dotted step, because the check is not more clicks of the same kind.
PHOTO_CHECK_X = 168


def photo_figure(data: dict[float, dict], stage: int) -> Figure:
    fig = plt.figure(figsize=(12.8, 7.2))
    lines = {b: (data[b]["clicks"] + [PHOTO_CHECK_X], data[b]["f"] + [data[b]["after"]]) for b in data}

    def check_step(ax: plt.Axes, _ends: dict) -> None:
        for beta, _label, _w in RADIOS:
            ax.plot([PHOTO_CHECK_X], [data[beta]["after"]], marker="o", markersize=8, color=INK, zorder=4)

    ax = _f_panel(
        fig,
        lines,
        (0, PHOTO_CHECK_X + 4),
        [25, 50, 100, 150, PHOTO_CHECK_X],
        "clicks, then the spot check",
        check_step,
    )
    ax.set_xticklabels(["25", "50", "100", "150", "✓"])
    if stage >= 2:
        _pr_panel(fig, data, "after the check")
    return fig


def doc_figure(data: dict[float, dict], stage: int) -> Figure:
    fig = plt.figure(figsize=(12.8, 7.2))
    lines = {b: (data[b]["clicks"], data[b]["f"]) for b in data}
    _f_panel(fig, lines, (0, DOC_CLICKS), [0, 10, 20, 30, 40, 50], "clicks")
    if stage >= 2:
        _pr_panel(fig, data, f"at {DOC_PR_CLICK} clicks")
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
