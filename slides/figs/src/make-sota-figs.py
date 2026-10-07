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
  compared, square and on one scale (#4533). Each radio is a path through the
  session in the radio's own weight: on photos (#4519) the set at every click,
  as the left panel draws the objective, then the step to after the check; on
  documents (#4517) its three points joined straight.

Both panels mark the same clicks with the same small hollow circle (`_mark`;
owner, 2026-10-07): the typed query, 25, 50, 100 and 150 clicks and after the
check (∞) on photos, and `DOC_PR_CLICKS` on documents.

The photo slide is the binary-photo review (SigLIP, binary votes, COCO Better,
10 seeds, `docs/experiments/2026-10-05-state-of-the-app-binary-photo/`). Its
per-run data lives on the GRID, so the slide reads what the report commits:
`precision_recall_path.csv`, the returned set's F-beta, precision, recall and
size at 25, 50, 100 and 150 clicks and after the check, per radio.

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
from slide_figure import FULL_BLEED, INK, SOFT, save, spread_labels  # noqa: E402

SRC = Path(__file__).resolve().parent
OUT = SRC.parent
REPO = SRC.parents[2]
EXPERIMENTS = REPO / "docs" / "experiments"
PHOTO_REPORT = EXPERIMENTS / "2026-10-05-state-of-the-app-binary-photo"
#: The review's committed path through the session (#4519): per radio, the returned set at
#: `PHOTO_CLICKS` and after the check, from `perp.py --kind balance` over 10 seeds.
PHOTO_PATH = PHOTO_REPORT / "precision_recall_path.csv"
#: The clicks the photo review reads the returned set at, before the check.
PHOTO_CLICKS = (25, 50, 100, 150)
#: The session's end, after the spot check: "∞", the last point the session reaches (owner, 2026-10-07:
#: "instead of 'check' meaning the theoretical last point on the graphs, use the infinity symbol").
END_LABEL = "∞"
#: The path's first point in `precision_recall_path.csv` (`perp.py`'s ``TYPED_QUERY``): the text sort under
#: its own blind GMM cut, before any vote. The F panel starts every radio there, at click 0 (owner,
#: 2026-10-06: "a text-sort (and GMM-thresh) notch at the far left for 0"); the precision-recall panel
#: leaves it out, at precision ~0.01 and recall ~0.9, far off its scale.
TYPED_QUERY = "typed query"
#: The Region Photo review (#4534): the same harness and the same table, on the
#: region path (DINOv3 patches, `max_patch`, opened on SigLIP's text sort).
REGION_REPORT = EXPERIMENTS / "2026-10-06-state-of-the-app-region-photo"
REGION_PATH = REGION_REPORT / "precision_recall_path.csv"
#: The region review's precision-recall panel: its recall span.
REGION_PR_LIM = (0.4, 0.9)

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
#: The clicks the document slide marks on both panels: the query crop alone, then
#: through the session to its last click. Not 25: every radio has stopped moving
#: by then, so its mark would sit on the one at 50.
DOC_PR_CLICKS = (0, 10, 50)

#: What the presenter notes quote, rounded as the notes say them.
EXPECT = {
    ("photos", 0.25, "after"): 0.64,
    ("photos", 1.0, "after"): 0.53,
    ("photos", 4.0, "after"): 0.60,
    ("photos", 0.25, "kept"): 21,
    ("photos", 1.0, "kept"): 44,
    ("photos", 4.0, "kept"): 80,
    ("photos", 0.25, "precision"): 0.73,
    ("photos", 4.0, "recall"): 0.68,
    ("documents", 0.25, "precision"): 0.98,
    ("documents", 1.0, "precision"): 0.93,
    ("documents", 4.0, "recall"): 0.95,
    ("documents", 0.25, "kept"): 13,
    ("documents", 1.0, "kept"): 14,
    ("documents", 4.0, "kept"): 25,
    ("documents", 1.0, "f50"): 0.87,
    ("photos", 0.25, "text"): 0.17,
    ("photos", 1.0, "text"): 0.24,
    ("photos", 4.0, "text"): 0.48,
    ("photos", 0.25, "at50"): 0.48,
    ("photos", 1.0, "at50"): 0.42,
    ("photos", 4.0, "at50"): 0.50,
    ("regions", 0.25, "at50"): 0.51,
    ("regions", 1.0, "at50"): 0.46,
    ("regions", 4.0, "at50"): 0.57,
    ("regions", 0.25, "text"): 0.18,
    ("regions", 4.0, "text"): 0.48,
    ("regions", 0.25, "after"): 0.73,
    ("regions", 1.0, "after"): 0.63,
    ("regions", 4.0, "after"): 0.72,
    ("regions", 0.25, "kept"): 30,
    ("regions", 1.0, "kept"): 48,
    ("regions", 4.0, "kept"): 92,
    ("regions", 0.25, "precision"): 0.81,
    ("regions", 4.0, "recall"): 0.82,
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


def photo_data(source: Path = PHOTO_PATH) -> dict[float, dict]:
    """Per radio: F-beta at `PHOTO_CLICKS` and after the check, the set after the check, and the path.

    All of it is a review's committed `precision_recall_path.csv` (#4519): means over
    the trained runs with a line by that click, the returned size a median. The
    binary review's by default; the region review's (#4534) has the same shape.
    """
    by_click: dict[float, list[tuple[int, float]]] = {}
    pr_click: dict[float, list[tuple[float, float]]] = {}
    clicks_csv = source.with_name("objective_by_click.csv")
    if not clicks_csv.exists():
        raise SystemExit(f"make-sota-figs: {clicks_csv} is missing: run state_of_app/by_click.py (#4599)")
    with clicks_csv.open() as f:
        for row in csv.DictReader(f):
            by_click.setdefault(float(row["beta"]), []).append((int(row["t"]), float(row["fbeta"])))
            if row.get("precision"):  # the per-click path behind it (#4605)
                pr_click.setdefault(float(row["beta"]), []).append((float(row["recall"]), float(row["precision"])))
    by: dict[float, dict[str, dict[str, str]]] = {}
    with source.open() as f:
        for row in csv.DictReader(f):
            by.setdefault(float(row["beta"]), {})[row["point"]] = row
    # The path starts at the typed query, "0" (owner, 2026-10-07), and ends after the check, "∞".
    points = [TYPED_QUERY] + [str(c) for c in PHOTO_CLICKS] + ["after the check"]
    if set(by) != {0.25, 1.0, 4.0} or any(set(points) - set(by[b]) for b in by):
        raise SystemExit(f"make-sota-figs: {source} does not carry every radio at every point")
    out = {}
    for beta, rows in by.items():
        path = [
            {
                "precision": float(rows[p]["precision"]),
                "recall": float(rows[p]["recall"]),
                "kept": int(round(float(rows[p]["returned, median"]))),
                "label": {"after the check": END_LABEL, TYPED_QUERY: "0"}.get(p, p),
            }
            for p in points
        ]
        curve = by_click[beta]
        out[beta] = {
            # The typed query at click 0: the text sort at the line the app draws for it (#4599).
            "text": float(rows[TYPED_QUERY]["fbeta"]),
            # Every click, filled: the typed query until the app shows a run's detector, the end of
            # Autopilot's opening (#4605, `by_click.py`).
            "curve_t": [t for t, _ in curve],
            "curve_f": [f for _, f in curve],
            # The same sessions' precision and recall at every click, for the path drawn as a curve.
            "curve_pr": pr_click.get(beta, []),
            "clicks": list(PHOTO_CLICKS),
            "f": [float(rows[str(c)]["fbeta"]) for c in PHOTO_CLICKS],
            # 50 clicks, which the notes quote: the first circle past the typed query (#4605).
            "at50": float(rows["50"]["fbeta"]),
            "after": float(rows["after the check"]["fbeta"]),
            "path": path,
            **{k: path[-1][k] for k in ("precision", "recall", "kept")},
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
                    "label": str(click),
                }
            )
        out[beta] = {
            "clicks": clicks,
            "f": [float(np.mean([r[0] for r in by_v[v]])) for v in clicks],
            "path": path,
            **path[-1],
        }
    return out


def _check(photos: dict, docs: dict, regions: dict) -> None:
    got = {}
    for (path, beta, key), want in EXPECT.items():
        data = {"photos": photos, "documents": docs, "regions": regions}[path]
        value = data[beta]["f"][DOC_CLICKS] if key == "f50" else data[beta][key]
        got[(path, beta, key)] = value if isinstance(want, int) else round(value, 2)
    wrong = {k: (got[k], EXPECT[k]) for k in EXPECT if got[k] != EXPECT[k]}
    if wrong:
        raise SystemExit(f"make-sota-figs: the reviews no longer say what the notes quote (got, pinned): {wrong}")


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

#: A marked click, the same on both panels: a small hollow circle, white inside so it
#: reads against the line it sits on (owner, 2026-10-07: solid dots were "melting into
#: the black line"; then "make the dots smaller ... they have white AND black"). Drawn
#: over the line, so even the heaviest radio's shows the white centre. The edge is the
#: middle radio's line weight (owner: "match the middle line, not the thinnest").
MARK_PT = 7
MARK_EDGE = next(w for b, _, w in RADIOS if b == 1.0)
#: The precision-recall grid: a faint line every step, so a set near the
#: top-right corner can still be read against something (#4533). Half a tick
#: on the documents' half-unit span; a whole one on the photos' wider span,
#: where half a tick would be fourteen lines a side.
DOC_GRID_STEP, PHOTO_GRID_STEP = 0.05, 0.1
GRID_COLOUR = "#e3e7ec"
#: The precision-recall panel's grid is darker than the F panel's (#4563): it
#: is the thing a set's position is read against, and at the F panel's shade it
#: vanished on a projector. Still lighter than the axes, so it stays behind.
PR_GRID_COLOUR, PR_GRID_LW = "#bcc4ce", 1.3


def _pr_axes(
    fig: Figure, lim: tuple[float, float], ticks: list[float], step: float, ylim: tuple[float, float] | None = None
) -> plt.Axes:
    """The square precision-recall axes, gridded every *step* on both.

    *lim* is recall's span and, unless *ylim* is given, precision's too. The results
    slides all give *ylim*: the F panel's span beside it, so a precision and an
    F-beta at one height read level across the slide (owner, 2026-10-07).
    """
    ylim = ylim or lim
    ax = fig.add_axes(RIGHT_AXES)
    ax.set_xlim(*lim)
    ax.set_ylim(*ylim)
    ax.set_xticks([t for t in ticks if lim[0] - 1e-9 <= t <= lim[1] + 1e-9])
    ax.set_yticks([t for t in ticks if ylim[0] - 1e-9 <= t <= ylim[1] + 1e-9])

    # From the first multiple of the step inside each span, so a span that starts
    # between steps (0.35) keeps its lines on the ticks.
    def grid(span: tuple[float, float]) -> np.ndarray:
        return np.arange(math.ceil(span[0] / step - 1e-9) * step, span[1] + step / 2, step)

    ax.set_xticks(grid(lim), minor=True)
    ax.set_yticks(grid(ylim), minor=True)
    ax.tick_params(which="minor", length=0)
    ax.grid(True, which="both", color=PR_GRID_COLOUR, lw=PR_GRID_LW)
    ax.set_axisbelow(True)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    return ax


def _mark(ax: plt.Axes, xs: list[float], ys: list[float]) -> None:
    """Marked clicks on either panel: :data:`MARK_PT` hollow circles over the lines."""
    ax.plot(
        xs,
        ys,
        ls="none",
        marker="o",
        markersize=MARK_PT,
        markerfacecolor="white",
        markeredgecolor=INK,
        markeredgewidth=MARK_EDGE,
        zorder=5,
        clip_on=False,
    )


#: How far right of a path's last mark its radio's name starts, in points.
END_LABEL_PT = MARK_PT / 2 + 6


#: The paths' recall axis: zoomed to where the document sets live. Precision's
#: span is the slide's own, shared with the F panel beside it (`_floor`).
PR_PATH_LIM = (0.5, 1.0)
#: The photo paths' recall axis: every photo set's recall lies between 0.38 and 0.68.
PHOTO_PR_LIM = (0.3, 0.8)


def _pr_paths_panel(
    fig: Figure,
    sets: dict[float, dict],
    lim: tuple[float, float] | None = None,
    ticks: list[float] | None = None,
    step: float = DOC_GRID_STEP,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Precision against recall, one path per radio through the session's clicks.

    Each radio is drawn at the weight the left panel gives it, so a path and its
    F line are one object on two axes, and nothing joins the radios to one
    another: what the panel shows is how each one's set moves as votes arrive.
    A radio with ``curve_pr`` (photos) is drawn at every click, then the step to
    after the check, as the left panel draws the objective (owner, 2026-10-07:
    "the left has high fidelity; it's weird for the right to be coarse"); one
    without (documents) joins its points straight. Every point in ``path`` is a
    :func:`_mark`, at the same clicks the left panel marks. Each path is labelled
    where it ends; how many each returns is in the notes.

    A point two radios share - the typed query, the query crop alone - is drawn
    once, by the first radio to reach it.
    """
    ax = _pr_axes(fig, lim or PR_PATH_LIM, ticks or [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], step, ylim)
    drawn: set[tuple[float, float]] = set()
    for beta, label, weight in RADIOS:
        path = sets[beta]["path"]
        xs = [point["recall"] for point in path]
        ys = [point["precision"] for point in path]
        along = sets[beta].get("curve_pr") or []
        cx = [r for r, _ in along] + [xs[-1]] if along else xs
        cy = [p for _, p in along] + [ys[-1]] if along else ys
        ax.plot(cx, cy, color=INK, lw=weight, solid_capstyle="round", solid_joinstyle="round", zorder=3)
        fresh = [(x, y) for x, y in zip(xs, ys, strict=True) if (round(x, 3), round(y, 3)) not in drawn]
        drawn.update((round(x, 3), round(y, 3)) for x, y in fresh)
        _mark(ax, [x for x, _ in fresh], [y for _, y in fresh])
        ax.annotate(
            label,
            (xs[-1], ys[-1]),
            xytext=(END_LABEL_PT, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=16,
            color=INK,
            annotation_clip=False,
        )


def _f_panel(
    fig: Figure,
    lines: dict[float, tuple[list, list]],
    xlim,
    xticks,
    xlabel: str,
    floor: float,
    extra=None,
    grid_step: float = 0.1,
    major: float = 0.2,
) -> plt.Axes:
    """The returned set's F-beta over a session, one line per radio, labelled at its end.

    The axis starts at *floor*, the last *major* step under the lowest point the
    slide draws, rather than at 0: the panel is read for its shape (#4533). Its
    numbers come every *major*, as the precision axis's beside it do.
    """
    ax = fig.add_axes(LEFT_AXES)
    ax.set_xlim(*xlim)
    ax.set_ylim(floor, 1.0)
    ax.set_xticks(xticks)
    ax.set_yticks(np.arange(floor, 1.0 + 1e-9, major))
    # The same horizontal lines as the precision-recall panel beside it, on the same
    # y range (owner, 2026-10-07): an F-beta and a precision at one height read
    # level across the slide.
    ax.set_yticks(
        np.arange(math.ceil(floor / grid_step - 1e-9) * grid_step, 1.0 + grid_step / 2, grid_step), minor=True
    )
    ax.tick_params(axis="y", which="minor", length=0)
    ax.yaxis.grid(True, which="both", color=PR_GRID_COLOUR, lw=PR_GRID_LW)
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
    # Each label sits just past its line's end, nudged apart only as far as the
    # type needs and in the ends' own order: never on a leader, which reads as
    # one more line (owner, 2026-10-05; STYLE.md). The gap is a share of the
    # span, so a type's height apart on any floor.
    label_y = spread_labels([ends[b][1] for b, _, _ in RADIOS], gap=0.075 * (1.0 - floor))
    # A fixed distance in points past the panel's edge, so a mark on the edge (documents end
    # there) never sits under its label.
    for (beta, label, _w), y in zip(RADIOS, label_y, strict=True):
        ax.annotate(
            label,
            (xlim[1], y),
            xytext=(END_LABEL_PT, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=16,
            color=INK,
            annotation_clip=False,
        )
    return ax


def _floor(lines: dict[float, tuple[list, list]], also: float | None = None, step: float = 0.2) -> float:
    """The slide's shared y floor: the last *step* at or under every point the F panel draws and, with
    *also*, the lowest precision the precision-recall panel draws (owner, 2026-10-07: one y scale)."""
    lowest = min(min(ys) for _xs, ys in lines.values())
    if also is not None:
        lowest = min(lowest, also)
    return round(math.floor(round(lowest / step, 6)) * step, 6)


def _lowest_precision(data: dict[float, dict]) -> float:
    """The lowest precision on any radio's precision-recall path."""
    return min(point["precision"] for d in data.values() for point in d["path"])


#: Where the photo slide draws "after the check": past the last click, joined
#: to it by a dotted step, because the check is not more clicks of the same kind.
PHOTO_CHECK_X = 168


def photo_figure(
    data: dict[float, dict],
    stage: int,
    recall_lim: tuple[float, float] = PHOTO_PR_LIM,
) -> Figure:
    """Photo Finish's two panels; Patch Notes (#4534) draws the region review with them."""
    fig = plt.figure(figsize=FIG_SIZE)
    # Every click, not five checkpoints joined (owner, 2026-10-07: "why is the left curve so coarse?"):
    # flat on the typed query through Autopilot's opening (#4605), then the climb as sessions hand over;
    # then the step to after the check.
    lines = {b: (data[b]["curve_t"] + [PHOTO_CHECK_X], data[b]["curve_f"] + [data[b]["after"]]) for b in data}

    def marks(ax: plt.Axes, _ends: dict) -> None:
        # The clicks the precision-recall path marks, at the same places: the typed query, each of
        # `PHOTO_CLICKS`, and after the check.
        for beta, _label, _w in RADIOS:
            d = data[beta]
            _mark(ax, [0, *PHOTO_CLICKS, PHOTO_CHECK_X], [d["text"], *d["f"], d["after"]])

    floor = _floor(lines, _lowest_precision(data))
    ax = _f_panel(
        fig,
        lines,
        (0, PHOTO_CHECK_X + 4),
        [0, 25, 50, 100, 150, PHOTO_CHECK_X],
        "Clicks, then the spot check",
        floor,
        marks,
        PHOTO_GRID_STEP,
    )
    ax.set_xticklabels(["typed\nquery", "25", "50", "100", "150", END_LABEL])
    if stage >= 2:
        _pr_paths_panel(fig, data, recall_lim, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], PHOTO_GRID_STEP, (floor, 1.0))
    return fig


def doc_figure(data: dict[float, dict], stage: int) -> Figure:
    fig = plt.figure(figsize=FIG_SIZE)
    lines = {b: (data[b]["clicks"], data[b]["f"]) for b in data}
    # Numbered every 0.1 on both panels, as the recall axis is: every document set lies above 0.5.
    floor = _floor(lines, _lowest_precision(data), step=0.1)

    def marks(ax: plt.Axes, _ends: dict) -> None:
        # The clicks the precision-recall path marks (`DOC_PR_CLICKS`), at the same places.
        for beta, _label, _w in RADIOS:
            _mark(ax, list(DOC_PR_CLICKS), [data[beta]["f"][c] for c in DOC_PR_CLICKS])

    _f_panel(fig, lines, (0, DOC_CLICKS), [0, 10, 20, 30, 40, 50], "Clicks", floor, marks, DOC_GRID_STEP, major=0.1)
    if stage >= 2:
        _pr_paths_panel(fig, data, ylim=(floor, 1.0))
    return fig


def region_figure(data: dict[float, dict], stage: int) -> Figure:
    return photo_figure(data, stage, REGION_PR_LIM)


def main() -> int:
    photos, docs, regions = photo_data(), doc_data(), photo_data(REGION_PATH)
    _check(photos, docs, regions)
    for stem, build, data in (
        ("sota-photos", photo_figure, photos),
        ("sota-documents", doc_figure, docs),
        ("sota-regions", region_figure, regions),
    ):
        # Saved at the canvas's declared bounds, so both stages frame the
        # panels identically and the top margin keeps the title notch empty.
        save(build(data, 2), OUT, f"{stem}.png", column=FULL_BLEED, tight=False)
        save(build(data, 1), OUT, f"{stem}.build1.png", column=FULL_BLEED, tight=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
