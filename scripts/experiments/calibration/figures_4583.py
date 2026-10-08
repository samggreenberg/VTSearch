#!/usr/bin/env python3
"""#4583 figures, from the CSVs ``analyze_calsplit_4583.py`` wrote.

    python figures_4583.py --data ANALYSIS_DIR --out FIGURES_DIR

* ``contrasts.png`` - every contrast's paired Δ-objective, ±2 SE (clustered on
  category), per preset: votes 1-150 and after the check.
* ``run_deltas.png`` - each run's paired Δ-objective after the check, per
  contrast and preset: the spread behind those means.
* ``curves.png`` - the objective over votes, one panel per preset, the shipped
  arm against the split and fold-count arms.
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

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
BETAS = (0.25, 1.0, 4.0)
BETA_LABEL = {0.25: "beta 1/4", 1.0: "beta 1", 4.0: "beta 4"}
BTAG = {0.25: "b025", 1.0: "b1", 4.0: "b4"}
CONTRAST_LABEL = {
    "split": "0.5 held out (vs 0.3)",
    "count": "4 folds (vs 2)",
    "both": "0.5 and 4 folds",
    "one fold": "1 fold (vs 2)",
    "interaction": "interaction",
}
ARM_LABEL = {"f03k2": "shipped: 0.3, 2 folds", "f05k2": "0.5, 2 folds", "f03k4": "0.3, 4 folds", "f03k1": "0.3, 1 fold"}


def _style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def contrasts(data: Path, out: Path) -> None:
    p = pd.read_csv(data / "paired.csv")
    p = p[(p.stratum == "all") & ~p.contrast.str.startswith("region")]
    names = [c for c in CONTRAST_LABEL if c in set(p.contrast)]
    reads = ("objective, votes 1-150", "objective, after the check")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True, sharex=True)
    for ax, beta in zip(axes, BETAS, strict=True):
        for j, read in enumerate(reads):
            q = p[(p.beta == beta) & (p.read == read)].set_index("contrast")
            for i, c in enumerate(names):
                if c not in q.index:
                    continue
                y = i + (j - 0.5) * 0.28
                d, se = q.loc[c, "delta"], q.loc[c, "se"]
                ax.errorbar(
                    d,
                    y,
                    xerr=2 * se,
                    fmt="o",
                    color=SERIES[j],
                    ms=5,
                    capsize=3,
                    lw=1.5,
                    label=read if (i == 0 and beta == BETAS[0]) else None,
                )
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([CONTRAST_LABEL[c] for c in names], fontsize=8)
        ax.set_title(BETA_LABEL[beta], fontsize=9, color=INK, loc="left")
        ax.set_xlabel("paired Δ-objective (right of 0: the changed arm is better)", fontsize=8, color=MUTED)
        _style(ax)
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle(
        "Every contrast is within ±0.006 of the shipped arm, at every preset (bars: ±2 SE)",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out / "contrasts.png", dpi=130)
    plt.close(fig)


def run_deltas(data: Path, out: Path) -> None:
    r = pd.read_csv(data / "run_deltas.csv")
    names = [c for c in CONTRAST_LABEL if c in set(r.contrast) and c != "interaction"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True, sharex=True)
    for ax, beta in zip(axes, BETAS, strict=True):
        for i, c in enumerate(names):
            v = r[(r.beta == beta) & (r.contrast == c)]["delta"].dropna().to_numpy(float)
            if v.size == 0:
                continue
            jitter = (np.random.default_rng(i).random(v.size) - 0.5) * 0.5
            ax.scatter(v, i + jitter, s=5, color=SERIES[0], alpha=0.3, linewidths=0)
            ax.plot([v.mean()] * 2, [i - 0.35, i + 0.35], color=INK, lw=2)
            ax.text(
                0.98,
                i,
                f"better {np.mean(v > 0.01):.0%} / worse {np.mean(v < -0.01):.0%}",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=7,
                color=MUTED,
            )
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([CONTRAST_LABEL[c] for c in names], fontsize=8)
        ax.set_title(f"{BETA_LABEL[beta]}: each run after the check", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("paired Δ-objective (bar = mean)", fontsize=8, color=MUTED)
        _style(ax)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / "run_deltas.png", dpi=130)
    plt.close(fig)


def curves(data: Path, out: Path) -> None:
    c = pd.read_csv(data / "curves.csv").set_index("t")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
    for ax, beta in zip(axes, BETAS, strict=True):
        for i, stem in enumerate(ARM_LABEL):
            col = f"{stem}_{BTAG[beta]}"
            if col in c.columns:
                ax.plot(c.index, c[col], color=SERIES[i], lw=1.6, label=ARM_LABEL[stem])
        ax.set_title(f"{BETA_LABEL[beta]}: the objective over votes", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("votes", fontsize=8, color=MUTED)
        _style(ax)
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    contrasts(args.data, args.out)
    run_deltas(args.data, args.out)
    curves(args.data, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
