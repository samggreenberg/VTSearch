"""Figures for the #4934 keypoint-cut study, from ``keypoint_cut_analyze.py``'s ``steps.csv``.

* ``quality_over_clicks.png``: mean ranking AP and F1 at the line over clicks 0-50, one line per arm;
* ``per_class_ap25.png``: each class's AP change at 25 clicks against the control, one row per cut.

A session that ended before click 50 holds its last state, as in the tables.

    python keypoint_cut_figures.py <steps.csv> <figure dir> [--control all]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

MAX_V = 50
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dc"
#: Categorical slots 1-5 of the dataviz reference palette, in fixed order; the control is ink.
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
LABELS = {
    "all": "all keypoints (control)",
    "scale2048": "largest scale 2,048",
    "strongfrac035": "strongest 35%",
    "strong2048": "strongest 2,048",
    "rows2048": "first 2,048 rows",
    "strong1024": "strongest 1,024",
}


def series(rows: list[dict[str, str]], key: str) -> np.ndarray:
    """*key* at clicks 0..MAX_V for one session, holding its last value past its end."""
    by_v = {int(r["v"]): float(r[key]) for r in rows}
    out, last = np.empty(MAX_V + 1), np.nan
    for v in range(MAX_V + 1):
        last = by_v.get(v, last)
        out[v] = last
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", type=Path)
    ap.add_argument("figdir", type=Path)
    ap.add_argument("--control", default="all")
    args = ap.parse_args(argv)
    sessions: dict[str, dict[tuple[str, str], list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(args.steps.open(encoding="utf-8")):
        sessions[r["arm"]][(r["class_id"], r["rep"])].append(r)
    # The control first, then the cuts from the smallest loss to the largest (fixed for both figures).
    ap25 = {a: np.nanmean([series(s, "ap")[25] for s in sessions[a].values()]) for a in sessions}
    cuts = sorted((a for a in sessions if a != args.control), key=lambda a: -ap25[a])
    color = {args.control: INK, **{a: SLOTS[i % len(SLOTS)] for i, a in enumerate(cuts)}}
    plt.rcParams.update(
        {"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED}
    )
    args.figdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    for ax, key, title in ((axes[0], "ap", "Ranking AP (test half)"), (axes[1], "gate_f1", "F1 at the line (beta 1)")):
        for arm in [args.control, *cuts]:
            mean = np.nanmean(np.vstack([series(s, key) for s in sessions[arm].values()]), axis=0)
            ax.plot(
                range(MAX_V + 1),
                mean,
                color=color[arm],
                lw=2.4 if arm == args.control else 2,
                label=LABELS.get(arm, arm),
            )
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("clicks")
        ax.set_ylim(0, 1)
        ax.grid(True, color=GRID, lw=0.8)
        ax.axvline(25, color=MUTED, lw=1, ls=":")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    n = len(sessions[args.control])
    n_cls = len({c for c, _ in sessions[args.control]})
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle(
        f"CPU path: mean over {n} sessions ({n_cls} classes); dotted line = the decision click",
        x=0.01,
        ha="left",
        color=MUTED,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(args.figdir / "quality_over_clicks.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 0.6 + 0.62 * len(cuts)))
    rng = np.random.default_rng(0)
    for row, arm in enumerate(cuts):
        per_class: dict[str, list[float]] = defaultdict(list)
        for (cid, rep), s in sessions[arm].items():
            ctrl = sessions[args.control].get((cid, rep))
            if ctrl:
                per_class[cid].append(series(s, "ap")[25] - series(ctrl, "ap")[25])
        d = np.array([np.mean(v) for v in per_class.values()])
        y = len(cuts) - 1 - row
        ax.scatter(
            d,
            y + rng.uniform(-0.18, 0.18, len(d)),
            s=22,
            color=color[arm],
            alpha=0.75,
            edgecolors="white",
            linewidths=0.6,
        )
        ax.plot([d.mean()] * 2, [y - 0.32, y + 0.32], color=INK, lw=2)
    ax.set_yticks(range(len(cuts)), [LABELS.get(a, a) for a in reversed(cuts)])
    ax.axvline(0, color=MUTED, lw=1)
    ax.axvspan(-0.01, 0, color=GRID, lw=0)
    ax.set_xlabel("AP at 25 clicks, cut minus control (one dot a class; bar = mean; shaded = the -0.01 margin)")
    ax.grid(True, axis="x", color=GRID, lw=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.figdir / "per_class_ap25.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.figdir}/quality_over_clicks.png and per_class_ap25.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
