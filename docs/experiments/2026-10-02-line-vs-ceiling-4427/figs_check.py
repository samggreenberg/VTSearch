#!/usr/bin/env python
"""The check's shape per preset (#4427): the objective after the check, per arm and beta, with the unchecked line
as the reference. One panel per beta; a dot per arm with its paired SE against the app's walk.

    python figs_check.py --out fig.png @0.5 app-walk=<dir> advisory=<dir> shallow=<dir> @1 app-walk=<dir> ... @2 ...

(``@<beta>`` opens a panel; the first arm of each panel is the control the SEs are paired against.)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORS = {
    "app-walk": "#555555",
    "advisory": "#1f77b4",
    "shallow": "#d62728",
    "guard 1.0": "#9467bd",
    "guard 0.5": "#ff7f0e",
}


def _cells(d: Path) -> pd.DataFrame:
    c = pd.read_csv(d / "cells.csv")
    return c[~c["never_trained"].astype(bool)][["category", "seed", "thr_fbeta_unchecked", "thr_fbeta_final"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("rest", nargs="+")
    args = ap.parse_args()
    groups: list[tuple[float, list[tuple[str, Path]]]] = []
    it = iter(args.rest)
    for tok in it:
        if tok.startswith("@"):
            groups.append((float(tok[1:]), []))
        else:
            name, _, d = tok.partition("=")
            groups[-1][1].append((name, Path(d)))
    fig, axes = plt.subplots(1, len(groups), figsize=(3.6 * len(groups), 4.0), sharey=False)
    axes = list(axes) if len(groups) > 1 else [axes]
    for ax, (beta, arms) in zip(axes, groups):
        ctl = _cells(arms[0][1])
        ax.axhline(ctl["thr_fbeta_unchecked"].mean(), color="#999999", ls="--", lw=1.2)
        for i, (name, d) in enumerate(arms):
            c = _cells(d)
            m = ctl.merge(c, on=["category", "seed"], suffixes=("_c", "_a"))
            diff = m["thr_fbeta_final_a"] - m["thr_fbeta_final_c"]
            se = diff.std(ddof=1) / math.sqrt(len(diff)) if i else 0.0
            y = float(c["thr_fbeta_final"].mean())
            ax.errorbar([i], [y], yerr=[se], fmt="o", ms=9, color=COLORS.get(name, "#2ca02c"), capsize=4, lw=1.5)
            ax.annotate(f"{y:.3f}", (i, y), xytext=(10, 0), textcoords="offset points", va="center", fontsize=9)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([n for n, _d in arms], fontsize=9)
        ax.set_xlim(-0.6, len(arms) - 0.2)
        ax.set_title(f"beta {beta:g}", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("withheld F-beta above the threshold, after the check")
    fig.suptitle(
        "The check's shape per preset: the objective after the check (5 seeds, line - 4; ± paired SE vs the app's walk; dashed = the unchecked line)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
