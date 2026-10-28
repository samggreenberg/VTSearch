#!/usr/bin/env python
"""The figures of #5321's report: the imagekit provider against FaceNet on the face review.

* ``fig_curves.png`` - the objective at every click, FaceNet and imagekit, from 1 and from 4 photos
  (rows) at beta 1/4, 1 and 4 (columns), 1024 px, from each arm's ``k<K>-<size>-objective_by_click.csv``;
* ``fig_gaps.png`` - she/her minus he/him and 18-29 minus 45+ after the check, +-1 SE, per arm, beta
  and K at 1024 px, from ``ik_compare_5321.py``'s ``gaps.csv``.

    python ik_figures_5321.py --roll <roll analysis> --ik <ik analysis> --compare <compare dir> --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from face_gap_4780_figures import INK2, GRID, _style  # noqa: E402

#: The arms take the reference palette's blue (FaceNet, as in #4780) and orange (imagekit).
ARMS = (("FaceNet", "#2a78d6"), ("imagekit (buffalo_l)", "#eb6834"))
BETAS = (0.25, 1.0, 4.0)


def fig_curves(roll: Path, ik: Path, size: str, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
    for i, k in enumerate((1, 4)):
        for j, beta in enumerate(BETAS):
            ax = axes[i, j]
            for (label, color), d in zip(ARMS, (roll, ik)):
                c = pd.read_csv(d / f"k{k}-{size}-objective_by_click.csv")
                c = c[np.isclose(c["beta"], beta)]
                ax.plot(c["t"], c["fbeta"], color=color, lw=2, label=label)
            ax.set_title(f"{k} photo{'s' if k > 1 else ''}, β {beta:g}", fontsize=9)
            ax.grid(color=GRID, lw=0.6)
            if i == 1:
                ax.set_xlabel("click")
            if j == 0:
                ax.set_ylabel("objective (F-beta after the line)")
    axes[0, 0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_curves.png", dpi=150)
    plt.close(fig)


def fig_gaps(gaps: pd.DataFrame, size: int, out: Path) -> None:
    g = gaps[(gaps["size"] == size) & (gaps["metric"] == "after the check")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (key, title) in zip(axes, (("she/her-he/him", "she/her minus he/him"), ("18-29-45+", "18-29 minus 45+"))):
        rows = [(k, b) for k in (1, 4) for b in BETAS]
        y = np.arange(len(rows))
        for n, ((label, color), col) in enumerate(zip(ARMS, ("facenet", "imagekit"))):
            d = g[g["gap"] == key].set_index(["K", "beta"]).loc[rows]
            ax.errorbar(d[col], y + (n - 0.5) * 0.25, xerr=d[f"{col}_se"], fmt="o", ms=6, color=color,
                        elinewidth=1.2, capsize=0, label=label)  # fmt: skip
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(y, [f"{k} photo{'s' if k > 1 else ''}, β {b:g}" for k, b in rows])
        ax.set_title(f"{title}, after the check", fontsize=9)
        ax.grid(axis="x", color=GRID, lw=0.6)
    axes[0].invert_yaxis()  # shared: once for both
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "fig_gaps.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--roll", required=True, type=Path)
    ap.add_argument("--ik", required=True, type=Path)
    ap.add_argument("--compare", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--size", default="1024")
    args = ap.parse_args()
    _style()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_curves(args.roll, args.ik, args.size, args.out)
    fig_gaps(pd.read_csv(args.compare / "gaps.csv"), int(args.size), args.out)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
