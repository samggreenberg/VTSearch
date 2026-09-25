#!/usr/bin/env python
"""The calibration ladder on COCO Better: one cost curve per rung (#4184).

Run from the repo root, once the study's curve CSV is committed:

    python slides/figs/src/make-progression-fig.py

Reads `progression_curve.csv` from the study directory
(`docs/experiments/2026-09-25-progression-4184/`, written by
`scripts/experiments/calibration/analyze_progression_4184.py`) and writes
`figs/progression.png` plus one build stage per earlier rung, so the room
watches the curves arrive in the order the deck argued for them.

    python slides/figs/src/make-progression-fig.py --demo DIR

draws the same figure from synthetic curves into DIR, for checking the layout
before the run exists. It never writes into `figs/`: a slide made of invented
numbers must not be one `git add` away from the deck.

**Every curve is drawn in its final style from the page it appears on.**
`slides/STYLE.md`'s build rule is that a reveal adds ink and restyles nothing,
so a rung cannot be black on its own page and grey on the next. The styling
does the work a restyle would have done: the rungs darken as they climb, and
the last one - the app as it ships - is the deck's blue, the colour that
means "the threshold, and the shipped decision it makes".

**Each curve starts at the same notch.** Click 0 is the typed query's own
ranking, and every click before the app shows a detector reads as that same
ranking (the analyzer's *filled* mean), so all seven share their left end and
differ only in how fast and how far they fall.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slide_figure import FULL_BLEED, INK, SOFT, save  # noqa: E402

SRC = Path(__file__).resolve().parent
OUT = SRC.parent
REPO = SRC.parents[2]
STUDY = REPO / "docs" / "experiments" / "2026-09-25-progression-4184"
CSV = STUDY / "progression_curve.csv"

#: The rungs in the deck's order, and the words each is labelled with at its
#: right end - the slide's own name for the idea, not the harness's.
RUNGS: tuple[tuple[str, str], ...] = (
    ("r1_xcal", "cross-calibration"),
    ("r2_gmm", "mixture midpoint"),
    ("r3_blend", "blend"),
    ("r4_rawmean", "fused, raw average"),
    ("r5_anchored", "fused, rank transfer"),
    ("r6_split70", "70/30 split"),
    ("r7_acq4", "second cut"),
)

#: The deck's `.cut` blue (`themes/vtsearch.css`, `make-calib-figs.py`).
SHIPPED = "#2F6DB5"

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


def read_curves(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``rung -> (clicks, mean cost)`` from the analyzer's curve CSV."""
    import csv  # noqa: PLC0415

    by: dict[str, list[tuple[int, float]]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            by.setdefault(row["rung"], []).append((int(row["t"]), float(row["mean"])))
    out = {}
    for rung, pts in by.items():
        pts.sort()
        out[rung] = (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
    missing = [r for r, _ in RUNGS if r not in out]
    if missing:
        raise SystemExit(f"{path}: no curve for {', '.join(missing)} - is the study complete?")
    return out


def demo_curves(horizon: int = 150) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Plausible shapes only: a shared notch, a delay, a fall to a rung-dependent floor."""
    t = np.arange(horizon + 1)
    out = {}
    for i, (rung, _label) in enumerate(RUNGS):
        floor = 0.42 - 0.035 * i
        speed = 18.0 - 1.4 * i
        fall = 1.0 - np.exp(-np.clip(t - 4, 0, None) / speed)
        out[rung] = (t, 0.72 - (0.72 - floor) * fall)
    return out


def _spread(ys: list[float], gap: float) -> list[float]:
    """Nudge label heights apart by at least *gap*, keeping their order."""
    order = np.argsort(ys)
    placed = np.array(ys, dtype=float)[order]
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + gap)
    shift = (np.array(ys)[order].mean() - placed.mean()) if len(placed) else 0.0
    placed += shift
    out = np.empty_like(placed)
    out[order] = placed
    return out.tolist()


def _layout(curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """Everything every stage shares, computed once from the FINAL set of curves:
    the axes' limits, the notch, the line styles and the label slots."""
    n = len(RUNGS)
    greys = [plt.cm.Greys(0.35 + 0.5 * i / max(n - 2, 1)) for i in range(n - 1)]
    all_y = np.concatenate([curves[r][1] for r, _ in RUNGS])
    lo, hi = float(np.nanmin(all_y)), float(np.nanmax(all_y))
    pad = 0.06 * (hi - lo)
    finals = [float(curves[r][1][-1]) for r, _ in RUNGS]
    return {
        "styles": [*({"color": g, "lw": 2.4} for g in greys), {"color": SHIPPED, "lw": 3.6}],
        "t_max": max(int(curves[r][0].max()) for r, _ in RUNGS),
        "ylim": (lo - pad, hi + pad),
        "t0": float(np.mean([curves[r][1][0] for r, _ in RUNGS])),
        "label_y": _spread(finals, gap=0.075 * (hi - lo + 2 * pad)),
    }


def _figure(curves: dict[str, tuple[np.ndarray, np.ndarray]], layout: dict, k: int) -> Figure:
    """The slide with the first *k* rungs drawn - the final page is ``k = len(RUNGS)``."""
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    fig.subplots_adjust(left=0.1, right=0.73, top=0.74, bottom=0.14)
    t_max = layout["t_max"]
    ax.set_xlim(0, t_max)
    ax.set_ylim(*layout["ylim"])
    ax.set_xlabel("votes")
    ax.set_ylabel("cost = FPR + FNR")

    # The notch every curve leaves from: the typed query, before any vote.
    t0 = layout["t0"]
    ax.plot([0], [t0], marker="o", color=INK, markersize=9, zorder=5, clip_on=False)
    ax.annotate("typed query", (0, t0), xytext=(14, 8), textcoords="offset points", color=INK, va="bottom")

    last = RUNGS[-1][0]
    for (rung, name), style, y_lab in list(zip(RUNGS, layout["styles"], layout["label_y"], strict=True))[:k]:
        t, y = curves[rung]
        ax.plot(t, y, color=style["color"], lw=style["lw"], solid_capstyle="round")
        ax.annotate(
            name,
            (t[-1], y[-1]),
            xytext=(t_max * 1.02, y_lab),
            textcoords="data",
            color=SHIPPED if rung == last else INK,
            fontweight="bold" if rung == last else "normal",
            va="center",
            annotation_clip=False,
            arrowprops={"arrowstyle": "-", "color": style["color"], "lw": 1.0, "shrinkA": 0, "shrinkB": 4},
        )
    return fig


def draw(curves: dict[str, tuple[np.ndarray, np.ndarray]], out: Path, stem: str = "progression") -> None:
    """The final page and one build stage per earlier rung.

    Saved at the canvas's declared bounds, not cropped: the canvas is already the
    slide's 16:9, its top margin is what keeps the title notch empty, and a
    tight crop would both remove that margin and reframe each stage to its own
    ink - the two things a full-bleed build must not do.
    """
    layout = _layout(curves)
    n = len(RUNGS)
    save(_figure(curves, layout, n), out, f"{stem}.png", column=FULL_BLEED, tight=False)
    for k in range(1, n):
        save(_figure(curves, layout, k), out, f"{stem}.build{k}.png", column=FULL_BLEED, tight=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--demo", metavar="DIR", help="draw synthetic curves into DIR (never into figs/)")
    args = ap.parse_args(argv)
    if args.demo:
        out = Path(args.demo).resolve()
        if out == OUT.resolve():
            raise SystemExit("--demo refuses to write into slides/figs/")
        out.mkdir(parents=True, exist_ok=True)
        draw(demo_curves(), out)
        return 0
    if not CSV.exists():
        raise SystemExit(f"{CSV.relative_to(REPO)} does not exist yet: run the #4184 study and commit its curve CSV")
    draw(read_curves(CSV), OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
