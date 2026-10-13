#!/usr/bin/env python
"""#4798's figures, from ``cut_4798.py``'s ``clicks.csv`` / ``curves.csv`` and ``prior_4776.py``'s curves.

* ``fig_per_click.png`` - past #4786's Bads gate, while the class model holds fewer than 6 held-out
  Goods: what keeping the head's count instead of the class model's cut does to Find's F-beta at a
  click, by how much more the cut keeps than the count (``k_class / k_head``), faces against COCO, at
  beta 1 and 4. The same clicks, read by every count the line holds, go opposite ways on the two
  benches: no fix that reads only the line can tell a face session's runaway cut from a hard COCO
  category's sound deep one.
* ``fig_find_by_vote.png`` - what a Find returns at every vote, faces from four photos (1024 px) and
  COCO: today's line, the Goods-only gate (the arm, its own sessions paired with today's; the offline
  re-cut where no arm is given) and the per-click oracle (a bound, not a rule), at each preset.

Colours are the validated categorical slots in fixed order (``line_4786_figures.py``); lines are
labelled at their ends (slides/STYLE.md: no leader lines); the report's tables carry every number.

Usage::

    python cut_4798_figures.py --clicks b4=<clicks.csv> --clicks b1=<clicks.csv> \
        --curves b4=<curves.csv> ... [--arm b4=<prior_curves.csv> ...] --out <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from line_4786_figures import GRID, INK2, SLOTS, _spread, _style  # noqa: E402

BINS = [0, 0.5, 0.9, 1.1, 2, 4, 10, np.inf]
BIN_LABELS = ["<½", "½–0.9", "≈1", "1.1–2", "2–4", "4–10", ">10"]
BETA_NAMES = {"b025": "β ¼", "b1": "β 1", "b4": "β 4"}


def _pairs(specs: list[str] | None) -> dict[str, Path]:
    return {k: Path(v) for k, _, v in (s.partition("=") for s in specs or [])}


def per_click(clicks: dict[str, Path], out: Path) -> None:
    betas = [b for b in ("b1", "b4") if b in clicks]
    fig, axes = plt.subplots(1, len(betas), figsize=(5.2 * len(betas), 3.6), sharey=True, squeeze=False)
    for ax, b in zip(axes[0], betas):
        c = pd.read_csv(clicks[b])
        c = c[c["model_n_pos"] < 6].copy()
        c["bench"] = np.where(c["set"].str.startswith("coco"), "COCO", "faces")
        c["bin"] = pd.cut(c["k_class"] / c["k_head_count"].clip(lower=1), BINS, labels=BIN_LABELS)
        c["d"] = c["f_head_count"] - c["f_class"]
        g = c.groupby(["bench", "bin"], observed=False)["d"].mean().unstack("bench")
        x = np.arange(len(BIN_LABELS))
        for j, (bench, color) in enumerate((("faces", SLOTS[0]), ("COCO", SLOTS[1]))):
            if bench in g:
                ax.bar(x + (j - 0.5) * 0.4, g[bench].to_numpy(), width=0.38, color=color, label=bench)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(x, BIN_LABELS, fontsize=8)
        ax.set_xlabel("today's cut keeps ÷ the head's count")
        ax.set_title(BETA_NAMES[b], fontsize=10, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("F-beta: head's count − today's cut")
    axes[0][0].legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_per_click.png", dpi=150)
    plt.close(fig)


def _line(ax: plt.Axes, series: list[tuple[pd.Series, pd.Series, str, str, str]]) -> None:
    ends = []
    for v, y, label, color, ls in series:
        ax.plot(v, y, color=color, lw=2, ls=ls)
        ends.append((float(y.iloc[-1]), label, color))
    lo, hi = ax.get_ylim()
    for (_y, label, color), yy in zip(ends, _spread([e[0] for e in ends], 0.075 * (hi - lo))):
        ax.text(153, yy, label, color=color, va="center", fontsize=8, clip_on=False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_xlim(0, 150)


def find_by_vote(curves: dict[str, Path], arms: dict[str, Path], out: Path) -> None:
    betas = [b for b in ("b025", "b1", "b4") if b in curves]
    benches = (("faces, 4 photos", lambda s: s.startswith("k4-")), ("COCO", lambda s: s.startswith("coco")))
    fig, axes = plt.subplots(len(betas), 2, figsize=(10, 2.9 * len(betas)), squeeze=False)
    for i, b in enumerate(betas):
        cv = pd.read_csv(curves[b])
        arm = pd.read_csv(arms[b]) if b in arms else None
        for j, (name, pick) in enumerate(benches):
            ax = axes[i][j]
            sets = [s for s in cv["set"].unique() if pick(s)]
            if not sets:
                ax.set_visible(False)
                continue
            s = sets[0]
            f = cv[cv["set"] == s].pivot(index="vote", columns="rule", values="F")
            series = [(f.index.to_series(), f["today"], "today", SLOTS[0], "-")]
            if arm is not None and s in set(arm["set"]):
                a = arm[(arm["set"] == s) & (arm["metric"] == "F")].set_index("vote")
                series.append((a.index.to_series(), a["arm"], "Goods-only gate", SLOTS[1], "-"))
            else:
                series.append((f.index.to_series(), f["headgoods"], "Goods-only gate (re-cut)", SLOTS[1], "--"))
            series.append((f.index.to_series(), f["oracle"], "oracle", SLOTS[2], ":"))
            _line(ax, series)
            ax.set_title(f"{name}, {BETA_NAMES[b]}", fontsize=10, loc="left")
            if j == 0:
                ax.set_ylabel("Find's F-beta")
            if i == len(betas) - 1:
                ax.set_xlabel("vote")
    fig.tight_layout(w_pad=6)
    fig.savefig(out / "fig_find_by_vote.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clicks", action="append", help="BETA=clicks.csv (b1, b4)")
    ap.add_argument("--curves", action="append", help="BETA=curves.csv (b025, b1, b4)")
    ap.add_argument("--arm", action="append", help="BETA=prior_curves.csv")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    _style()
    if args.clicks:
        per_click(_pairs(args.clicks), args.out)
    if args.curves:
        find_by_vote(_pairs(args.curves), _pairs(args.arm), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
