#!/usr/bin/env python
"""#4786's figures, from ``line_4786.py``'s ``curves.csv`` / ``first_clicks.csv`` and ``prior_4776.py``'s curves.

* ``fig_find_by_vote.png`` - what a Find returns at every vote at beta 4 on FHIBE's faces (1024 px,
  1 and 4 starting photos) and on COCO Better's Binary opening: today's line, the head's count line
  priced on today's sessions, the arm (the rule in every fit's line, its own sessions, paired with
  today's) and the per-click oracle (the best count on the withheld half; a bound, not a rule).
* ``fig_first_click.png`` - what each line keeps at the first trained click on faces at beta 4, as
  the share of runs keeping at most k (log k), beside the person's withheld photos.
* ``fig_per_run.png`` - every run's paired difference, the arm minus today, in Find's session mean
  over votes 1-150, sorted: faces (four sets) and COCO.

Colours are the validated categorical slots 1-4 in fixed order; every line is labelled at its end
(slides/STYLE.md: no leader lines) except on the step plot, whose steps cross any direct label, so it
takes a legend in its empty corner; the report's tables carry every number.

Usage::

    python line_4786_figures.py --curves <curves.csv> ... --first <first_clicks.csv> ... \
        [--arm <prior_curves.csv> ...] [--per-run <prior_per_run.csv> ...] [--rule headcount@neg32pos6] --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
        }
    )


def _spread(ys: list[float], gap: float) -> list[float]:
    """Push labels apart by at least *gap*, keeping their mean where it was."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    shift = (sum(out) - sum(ys)) / max(len(ys), 1)
    return [y - shift for y in out]


Series = tuple[pd.DataFrame, str, str, str, str]  # (frame with vote + column, column, label, colour, style)


def _draw(ax, series: list[Series]) -> None:
    ends = []
    for frame, col, label, color, ls in series:
        if frame.empty:
            continue
        f = frame.sort_values("vote")
        ax.plot(f["vote"], f[col], color=color, lw=2, ls=ls)
        ends.append((float(f[col].iloc[-1]), label, color))
    ax.set_xlim(0, 150)
    ax.grid(axis="y", color=GRID, lw=0.8)
    lo, hi = ax.get_ylim()
    for (_y, label, color), yy in zip(ends, _spread([e[0] for e in ends], 0.075 * (hi - lo))):
        ax.text(153, yy, label, color=color, va="center", fontsize=8, clip_on=False)


def fig_find(c: pd.DataFrame, p: pd.DataFrame | None, rule: str, out: Path) -> None:
    panels = [
        ("k1-b4-1024", "faces, 1 starting photo"),
        ("k4-b4-1024", "faces, 4 starting photos"),
        ("coco-b4", "COCO Better, Binary opening"),
    ]
    panels = [x for x in panels if (c["set"] == x[0]).any()]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.7 * len(panels), 4.0), squeeze=False)
    for ax, (setname, title) in zip(axes[0], panels):
        cs = c[c["set"] == setname]
        arm = p[(p["set"] == setname) & (p["metric"] == "F")] if p is not None else pd.DataFrame()
        _draw(
            ax,
            [
                (cs[cs["rule"] == "today"], "F", "today", SLOTS[0], "-"),
                (cs[cs["rule"] == rule], "F", "head's count (priced)", SLOTS[1], "-"),
                (arm, "arm", "head's count (arm)", SLOTS[2], "-"),
                (cs[cs["rule"] == "oracle"], "F", "per-click oracle", SLOTS[3], "--"),
            ],
        )
        ax.set_title(f"{title}, β 4", fontsize=10)
        ax.set_xlabel("vote")
    axes[0][0].set_ylabel("Find's F-beta (withheld half)")
    fig.suptitle("β 4: what a Find returns at every vote, today's line and the head's count line")
    fig.tight_layout(rect=(0, 0, 0.92, 0.93))
    fig.savefig(out / "fig_find_by_vote.png", dpi=150)
    plt.close(fig)


def fig_first(fc: pd.DataFrame, out: Path) -> None:
    panels = [("k1-b4-1024", "1 starting photo"), ("k4-b4-1024", "4 starting photos")]
    panels = [x for x in panels if (fc["set"] == x[0]).any()]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.9 * len(panels), 3.8), squeeze=False, sharey=True)
    for ax, (setname, title) in zip(axes[0], panels):
        d = fc[fc["set"] == setname]
        for col, label, color in (
            ("n_test_pos", "the person's withheld photos", INK2),
            ("k_test", "today's line keeps", SLOTS[0]),
            ("k_head_count", "the head's count keeps", SLOTS[1]),
        ):
            v = np.sort(np.clip(d[col].to_numpy(float), 1, None))
            y = np.arange(1, v.size + 1) / v.size
            ax.step(v, y, where="post", color=color, lw=2, ls=":" if col == "n_test_pos" else "-", label=label)
        # The steps cross where a direct label would sit; the lower right is empty.
        ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor="linecolor")
        ax.set_xscale("log")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_title(f"faces, {title}, β 4, 1024 px", fontsize=10)
        ax.set_xlabel("images kept at the first trained click (log)")
    axes[0][0].set_ylabel("share of runs keeping at most this many")
    fig.suptitle("The first trained click: what each line keeps, beside the person's withheld photos")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "fig_first_click.png", dpi=150)
    plt.close(fig)


def fig_per_run(pr: pd.DataFrame, out: Path) -> None:
    face = [s for s in ("k1-b4-1024", "k1-b4-640", "k4-b4-1024", "k4-b4-640") if (pr["set"] == s).any()]
    coco = [s for s in ("coco-b4",) if (pr["set"] == s).any()]
    panels = [(face, "faces, β 4"), (coco, "COCO Better, Binary opening, β 4")]
    panels = [p for p in panels if p[0]]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 3.8), squeeze=False, sharey=True)
    for ax, (sets, title) in zip(axes[0], panels):
        for i, s in enumerate(sets):
            d = np.sort(pr.loc[pr["set"] == s, "dF_1_150"].to_numpy(float))
            ax.plot(np.linspace(0, 1, d.size), d, color=SLOTS[i], lw=2, label=s)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("runs, sorted by the difference (share)")
        ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor="linecolor")
    axes[0][0].set_ylabel("arm minus today, F-beta over votes 1-150")
    fig.suptitle("Every run: the head's count line against today's, paired run for run")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "fig_per_run.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--curves", type=Path, action="append", required=True, help="line_4786.py curves.csv")
    ap.add_argument("--first", type=Path, action="append", default=[], help="line_4786.py first_clicks.csv")
    ap.add_argument("--arm", type=Path, action="append", default=[], help="prior_4776.py prior_curves.csv")
    ap.add_argument("--per-run", type=Path, action="append", default=[], help="prior_4776.py prior_per_run.csv")
    ap.add_argument("--rule", default="headcount@neg32pos6")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    _style()
    c = pd.concat([pd.read_csv(x) for x in args.curves], ignore_index=True)
    p = pd.concat([pd.read_csv(x) for x in args.arm], ignore_index=True) if args.arm else None
    args.out.mkdir(parents=True, exist_ok=True)
    fig_find(c, p, args.rule, args.out)
    if args.first:
        fig_first(pd.concat([pd.read_csv(x) for x in args.first], ignore_index=True), args.out)
    if args.per_run:
        fig_per_run(pd.concat([pd.read_csv(x) for x in args.per_run], ignore_index=True), args.out)
    print(f"figures -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
