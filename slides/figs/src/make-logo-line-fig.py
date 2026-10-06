#!/usr/bin/env python
"""Section 7's line slide: where the app draws the line on document pages.

    python slides/figs/src/make-logo-line-fig.py

Writes `figs/logo-line.png` and two build stages. Each draws the app's
*returned set* (every page at or above the line) as mean F1 over FullMarks'
36 classes, by vote, on tier `m` (~50,000 pages, v5.0), scored on each class's
held-out half. One line per rule, in the order they shipped:

* **the 8-inlier gate**, the line before #4367: the first State of the App:
  Structural Document (`docs/experiments/2026-10-01-state-of-the-app-structural-document/`);
* **beat every Bad** (#4367): a page must fit better than the best-fitting Bad
  vote did. Drawn from #4440's run of it, so that it and the next line are the
  same app on the same votes (#4367's own run is within 0.04 of it at every vote);
* **fit tight until a Bad** (#4440, shipped): before the first Bad, a fit must
  also pass the geometry cuts; after it, the Bad ceiling takes over;
* the **best cut of the ranking**, dashed: the best F1 any line on the shipped
  ranking could reach, which is the ceiling the rules are chasing.

**Nothing here is a guess.** Every point is a mean of the committed per-step
CSVs those reports wrote through the app's own path, so a rerun of a study
that changes them changes the slide. `EXPECT` pins the numbers the presenter
notes quote, and `main` fails if the CSVs say anything else.

**A reveal adds ink and restyles nothing** (`slides/STYLE.md`): every stage is
laid out from the final set of lines, and each line is drawn in its final style
from the page it first appears on. The superseded rules are grey and darken as
they climb; the shipped rule is the deck's blue.
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
from slide_figure import FULL_BLEED, INK, SOFT, save  # noqa: E402

SRC = Path(__file__).resolve().parent
OUT = SRC.parent
REPO = SRC.parents[2]
EXPERIMENTS = REPO / "docs" / "experiments"

#: The deck's `.cut` blue (`themes/vtsearch.css`): the shipped decision.
SHIPPED = "#0b5fa5"

#: The rules in the order they shipped, as `(key, label, steps CSV)`.
RULES: tuple[tuple[str, str, Path], ...] = (
    (
        "gate",
        "8 inliers",
        EXPERIMENTS / "2026-10-01-state-of-the-app-structural-document" / "measurements" / "steps.csv",
    ),
    (
        "bads",
        "beat every Bad",
        EXPERIMENTS / "2026-10-02-early-line-4440" / "measurements" / "shipped-steps.csv",
    ),
    (
        "shipped",
        "fit tight until a Bad",
        EXPERIMENTS / "2026-10-02-early-line-4440" / "measurements" / "validate-steps.csv",
    ),
)
#: The best cut is read off the shipped run: the ranking the app ships.
ORACLE_LABEL = "best cut of the ranking"

#: What the presenter notes quote, at `(votes)`. Rounded as the notes say them.
EXPECT = {
    ("gate", 0): 0.53,
    ("gate", 10): 0.44,
    ("gate", 25): 0.43,
    ("bads", 10): 0.56,
    ("bads", 25): 0.87,
    ("shipped", 10): 0.79,
    ("shipped", 25): 0.88,
    ("oracle", 25): 0.94,
}
#: Classes and votes every run must cover, or the means are of different things.
CLASSES, MAX_VOTES = 36, 50

plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans"],
        # F₁'s subscript is set as mathtext, in the figure's own face, as on
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


def read_steps(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(votes, mean returned-set F1, mean best-cut F1)` over the classes, from a steps CSV."""
    by_v: dict[int, list[tuple[float, float]]] = {}
    classes: set[str] = set()
    with path.open() as f:
        for row in csv.DictReader(f):
            classes.add(row["class_id"])
            by_v.setdefault(int(row["v"]), []).append((float(row["gate_f1"]), float(row["best_f1"])))
    votes = np.array(sorted(by_v))
    if len(classes) != CLASSES or votes.tolist() != list(range(MAX_VOTES + 1)):
        raise SystemExit(f"{path.relative_to(REPO)}: {len(classes)} classes, votes {votes.min()}-{votes.max()}")
    if any(len(by_v[v]) != CLASSES for v in votes):
        raise SystemExit(f"{path.relative_to(REPO)}: a vote count is missing a class")
    rows = [np.array(by_v[v]) for v in votes]
    return votes, np.array([r[:, 0].mean() for r in rows]), np.array([r[:, 1].mean() for r in rows])


def curves() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """`key -> (votes, mean F1)` for every rule and the oracle, checked against `EXPECT`."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, _label, path in RULES:
        votes, returned, best = read_steps(path)
        out[key] = (votes, returned)
        if key == "shipped":
            out["oracle"] = (votes, best)
    got = {(k, v): round(float(out[k][1][v]), 2) for k, v in EXPECT}
    if got != EXPECT:
        wrong = {k: (got[k], EXPECT[k]) for k in EXPECT if got[k] != EXPECT[k]}
        raise SystemExit(f"logo-line: the CSVs no longer say what the notes quote (got, pinned): {wrong}")
    return out


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


YLIM = (0.3, 1.0)


def _styles() -> dict[str, dict]:
    greys = [plt.cm.Greys(0.4), plt.cm.Greys(0.62)]
    return {
        "gate": {"color": greys[0], "lw": 2.6},
        "bads": {"color": greys[1], "lw": 2.6},
        "shipped": {"color": SHIPPED, "lw": 3.6},
        "oracle": {"color": SOFT, "lw": 2.0, "linestyle": (0, (4, 3))},
    }


def _figure(lines: dict[str, tuple[np.ndarray, np.ndarray]], label_y: dict[str, float], k: int) -> Figure:
    """The slide with the first *k* rules drawn; the oracle is on every page."""
    styles = _styles()
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    # The top margin keeps the title notch empty; the right one holds the labels.
    fig.subplots_adjust(left=0.1, right=0.7, top=0.74, bottom=0.14)
    ax.set_xlim(0, MAX_VOTES)
    ax.set_ylim(*YLIM)
    ax.set_xticks(range(0, MAX_VOTES + 1, 10))
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.yaxis.grid(True, color="#e3e7ec", lw=1.0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Votes")
    # Two characters read upright; turned on its side, F₁ is a thing to tilt
    # your head at.
    ax.set_ylabel(r"$\mathregular{F_1}$", rotation=0, ha="right", va="center", labelpad=12)

    shown = [("oracle", ORACLE_LABEL)] + [(key, label) for key, label, _ in RULES[:k]]
    for key, label in shown:
        votes, f1 = lines[key]
        style = styles[key]
        ax.plot(votes, f1, solid_capstyle="round", zorder=4 if key == "shipped" else 3, **style)
        ax.annotate(
            label,
            (votes[-1], f1[-1]),
            xytext=(MAX_VOTES * 1.03, label_y[key]),
            textcoords="data",
            color=SHIPPED if key == "shipped" else (SOFT if key == "oracle" else INK),
            fontweight="bold" if key == "shipped" else "normal",
            ha="left",
            va="center",
            annotation_clip=False,
        )
    return fig


def draw(lines: dict[str, tuple[np.ndarray, np.ndarray]], out: Path, stem: str = "logo-line") -> None:
    """The final page and one build stage per earlier rule, laid out from the final set.

    Saved at the canvas's declared bounds, not cropped, so every stage frames
    the axes identically and the top margin keeps the title notch empty.
    """
    keys = ["oracle", *(key for key, _, _ in RULES)]
    span = YLIM[1] - YLIM[0]
    label_y = dict(zip(keys, _spread([float(lines[key][1][-1]) for key in keys], gap=0.075 * span), strict=True))
    n = len(RULES)
    save(_figure(lines, label_y, n), out, f"{stem}.png", column=FULL_BLEED, tight=False)
    for k in range(1, n):
        save(_figure(lines, label_y, k), out, f"{stem}.build{k}.png", column=FULL_BLEED, tight=False)


def main() -> int:
    draw(curves(), OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
