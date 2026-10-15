#!/usr/bin/env python
"""Experimentation's "Up and Up": what the work since the first rule bought, in the score the user sets (#4668, #4835).

Run from the repo root, once both studies' curve CSVs are committed:

    python slides/figs/src/make-up-and-up-fig.py

The owner wanted a slide whose takeaway is "the score they care about improved over and over" (2026-10-08). #4668
ran the F-beta era as a cumulative build-up on one set of sessions (COCO Better, the user's pool at 1%, 144 cells x 5
seeds a preset), each change added to the one before, and the owner picked the slide that draws two of its steps at
beta 1/4:

- the labels line at the preset (#4452), which lifts the second half of the session;
- today's app, which since #4603 includes the text query's own line, lifting the opening, where no detector is on
  screen yet.

Cross-calibration (rung b1) and the labels line (b2) are #4668's, measured 2026-10-08. Today's app moves every week,
so its line is the same rung re-run on today's app (#4835, `analyze_upandup_4835.py`): on 2026-10-08 it handed over
from the text query near vote 23 and dipped; the query's hold (#4736) keeps the text query's set on screen to 30.

One panel per preset, the objective (F-beta of the withheld half above the line the app shows) over votes, filled
from the text query's set until a detector shows. The x axis runs from the text query through every 25 votes to
150, then to the session's end after the spot check, "∞", as Photo Finish's does (owner, #4835). b1 and b2 ran no
check, so their lines stop at 150. One slide, six pages (owner, #4884: the three radios are frames of one slide).
At beta 1/4, a build, each page adding a line:

- a: cross-calibration (rung b1), grey;
- b: plus the line at F-beta's peak (b2, the labels line), dark;
- c: today's app, the deck's blue;
- d: the text query's own sort cut where an oracle would cut it, dashed: the best F-beta any cut of that ranking
  reaches, which no session sees (owner, #4884: not a fair comparison, since the sessions' lines got no oracle,
  but it is unfair against us). It takes no votes, so it is flat, and it stops where the sessions' votes do.

Then the beta 1 and beta 4 panels whole, one page each. Every session line carries its own click-0 dot: b1 and b2
start at the text query's old line (the mixture midpoint), today's app at its own.

The oracle is the State of the App review of 2026-10-08's (`ORACLE_SUMMARY`): its text sort's `oracle_fbeta`, on
the same bench at every class and size, 24 seeds a preset. The sort is the typed query's embedding against the
withheld half, which no app change moves, so the review's 24 seeds stand in for these sessions' 5.
"""

from __future__ import annotations

import csv
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
#: #4668's build-up: rungs b1 (cross-calibration) and b2 (the labels line), per preset.
BUILDUP_CSV = EXPERIMENTS / "2026-10-08-fbeta-buildup-4668" / "buildup_curve.csv"
#: Today's app on the same sessions (#4835): per preset, every vote and the end after the check.
TODAY_CSV = EXPERIMENTS / "2026-10-13-up-and-up-4835" / "today_curve.csv"
#: The text sort's best cut, from the State of the App review's per-beta table (the `app line` rows at `text`).
ORACLE_SUMMARY = EXPERIMENTS / "2026-10-08-state-of-the-app-binary-photo" / "perbeta_summary.md"
#: What the notes say the oracle reaches, per preset: `main` fails if the review's table says otherwise.
ORACLE_EXPECT = {"b025": 0.557, "b1": 0.449, "b4": 0.564}
#: The presets, as both CSVs name them, and how the axis writes each one's beta.
PRESETS: dict[str, str] = {"b025": "¼", "b1": "1", "b4": "4"}
#: The review's betas, as its table writes them.
SUMMARY_BETA = {"b025": "0.25", "b1": "1.0", "b4": "4.0"}
#: (line, label, colour, line width, bold label), in build order. ``today`` is `TODAY_CSV`'s, ``oracle`` is
#: `ORACLE_SUMMARY`'s; the rest are `BUILDUP_CSV` rungs. b2 is named for what the room met in section III, the
#: line at expected F-beta's peak, not for what it is fitted from (owner, #4884: "what does 'the labels line'
#: mean?").
LINES: tuple[tuple[str, str, str, float, bool], ...] = (
    ("b1", "cross-calibration", "#8a929e", 2.6, False),
    ("b2", "+ line at Fβ's peak", "#3d4450", 2.8, False),
    ("today", "today's app", "#2F6DB5", 3.6, True),
    ("oracle", "text query, oracle's cut", INK, 2.2, False),
)
#: The reference, not a session: dashed, and drawn under the sessions' lines.
DASHED = {"oracle"}
#: The pages, in order: (preset, how many of `LINES` are drawn, file). The beta 1/4 build, then the other two
#: radios whole; the last page is the slide's own figure.
PAGES: tuple[tuple[str, int, str], ...] = (
    ("b025", 1, "up-and-up.build1"),
    ("b025", 2, "up-and-up.build2"),
    ("b025", 3, "up-and-up.build3"),
    ("b025", 4, "up-and-up"),
    ("b1", 4, "up-and-up-b1"),
    ("b4", 4, "up-and-up-b4"),
)
YLIM = (0.0, 0.7)
#: The last vote the sessions ran to, and where the end after the check is drawn: past it, as on Photo Finish
#: (`make-sota-figs.PHOTO_CHECK_X`), with the infinity sign that figure uses.
HORIZON = 150
END_X = 168
END_LABEL = "∞"
#: The canvas: the whole slide, at 100 slide pixels to the inch.
FIG_W, FIG_H = 12.8, 7.2

plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans"],
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

#: One line: its votes and means, and its end after the check (None when the rung ran no check).
Curve = tuple[np.ndarray, np.ndarray, float | None]


def _need(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"{path} does not exist yet: run its study's analyzer and commit the CSV")
    return path


def read_oracle(preset: str) -> float:
    """The text sort's best-cut F-beta at *preset*: the review's ``app line`` row at ``text``."""
    for line in _need(ORACLE_SUMMARY).read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 13 and cells[1:5] == ["SigLIP binary", "app line", SUMMARY_BETA[preset], "text"]:
            oracle = float(cells[9])
            if round(oracle, 3) != ORACLE_EXPECT[preset]:
                raise SystemExit(f"the review's oracle at {preset} is {oracle}, the notes say {ORACLE_EXPECT[preset]}")
            return oracle
    raise SystemExit(f"{ORACLE_SUMMARY} has no text-sort row at {preset}")


def read_curves(preset: str) -> dict[str, Curve]:
    """``line -> (votes, mean, end after the check)`` at *preset*."""
    pts: dict[str, list[tuple[int, float]]] = {}
    with _need(BUILDUP_CSV).open() as f:
        for row in csv.DictReader(f):
            if row["preset"] == preset and row["rung"] in ("b1", "b2"):
                pts.setdefault(row["rung"], []).append((int(row["t"]), float(row["mean"])))
    end = None
    with _need(TODAY_CSV).open() as f:
        for row in csv.DictReader(f):
            if row["preset"] != preset:
                continue
            if row["point"] == "vote":
                pts.setdefault("today", []).append((int(float(row["t"])), float(row["mean"])))
            elif row["point"] == "after the check":
                end = float(row["mean"])
    pts["oracle"] = [(t, read_oracle(preset)) for t in range(HORIZON + 1)]
    missing = [name for name, *_ in LINES if name not in pts]
    if missing or end is None:
        raise SystemExit(f"no {preset} curve for {', '.join(missing) or 'today after the check'}")
    out: dict[str, Curve] = {}
    for name, p in pts.items():
        p.sort()
        if [t for t, _ in p] != list(range(HORIZON + 1)):
            raise SystemExit(f"{name} at {preset}: votes are not 0..{HORIZON}")
        out[name] = (np.array([t for t, _ in p]), np.array([y for _, y in p]), end if name == "today" else None)
    return out


def _figure(curves: dict[str, Curve], beta_name: str, shown: int) -> Figure:
    """The slide with the first *shown* lines drawn."""
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    # Up to the title notch's foot (172 of 720 px), the full width but a column on the right for the line names.
    ax = fig.add_axes((0.105, 0.135, 0.665, 0.62))
    ax.set_xlim(0, END_X + 4)
    ax.set_ylim(*YLIM)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.yaxis.grid(True, color="#e3e7ec", lw=1.0)
    ax.set_axisbelow(True)
    ticks = [0, 25, 50, 75, 100, 125, HORIZON, END_X]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["text\nquery", *[str(t) for t in ticks[1:-1]], END_LABEL])
    ax.set_xlabel("Votes, then the spot check")
    ax.set_ylabel(f"Fβ at β = {beta_name}")
    # Each line's name level with its own end, just past it, fanned only where ends meet; computed from every
    # line so a name holds its place from the page it appears on. The lines that stop at the last vote are
    # named under today's, which runs on over their names to the end after the check: those names are held a
    # gap below its last stretch, and fan among themselves.
    gap = 0.075 * (YLIM[1] - YLIM[0])
    ends = []
    for name, *_ in LINES:
        t, y, end = curves[name]
        ends.append((END_X, end) if end is not None else (float(t[-1]), float(y[-1])))
    _t, today_y, today_end = curves["today"]
    assert today_end is not None
    ceiling = min(float(today_y[-1]), today_end) - gap
    stopped = [i for i, (x, _y) in enumerate(ends) if x < END_X]
    if any(ends[i][1] > ceiling + gap for i in stopped):
        raise SystemExit("a line that stops at the last vote ends over today's app: place its name by hand")
    lab = [y for _, y in ends]
    for i, y in zip(stopped, spread_labels([-ends[i][1] for i in stopped], gap, floor=-ceiling), strict=True):
        lab[i] = -y
    for i, (name, label, colour, lw, bold) in enumerate(LINES[:shown]):
        t, y, end = curves[name]
        xs, ys = list(t), list(y)
        if end is not None:
            # The session's end after the check, joined straight, as Photo Finish joins it.
            xs.append(END_X)
            ys.append(end)
            ax.plot([END_X], [end], marker="o", color=colour, markersize=9, zorder=6, clip_on=False)
        if name in DASHED:
            ax.plot(xs, ys, color=colour, lw=lw, linestyle=(0, (5, 4)), zorder=2)
        else:
            ax.plot(xs, ys, color=colour, lw=lw, solid_capstyle="round", zorder=3 + i)
            ax.plot([0], [float(y[0])], marker="o", color=colour, markersize=9, zorder=6, clip_on=False)
        ax.text(ends[i][0] + 4, lab[i], label, color=colour if bold else INK, fontweight="bold" if bold else "normal",
                ha="left", va="center", clip_on=False)  # fmt: skip
    return fig


def main() -> int:
    curves = {preset: read_curves(preset) for preset in PRESETS}
    for preset, shown, name in PAGES:
        save(_figure(curves[preset], PRESETS[preset], shown), OUT, f"{name}.webp", column=FULL_BLEED, tight=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
