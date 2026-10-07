#!/usr/bin/env python3
"""#4582 figures, from the CSVs ``rescore_fbeta_4582.py`` wrote.

    python figures_4582.py --data docs/experiments/2026-10-06-cost-fbeta-rescore-4582/tables \\
        --out docs/experiments/2026-10-06-cost-fbeta-rescore-4582/figures

* ``ladder_044.png`` - #4184's seven rungs on the 0.44% pool, per click, in cost
  and at F-beta 1/4, 1 and 4: one panel per metric (one axis each).
* ``ladder_runs.png`` - each run's cost and F1 at click 150, r4 (the cost
  winner) against r1 (cross-calibration): the per-run view of the reversal.
* ``fraction_runs.png`` - #3287: each run's paired 0.3 - 0.5 at click 150, per
  space, at F-beta 1/4 and 4: where the fraction decision flips with beta.
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

#: The reference categorical order (dataviz palette), fixed per rung.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
RUNG_LABEL = {
    "r1_xcal": "r1 cross-calibration",
    "r2_gmm": "r2 mixture midpoint",
    "r3_blend": "r3 blend",
    "r4_rawmean": "r4 fused, raw mean",
    "r5_anchored": "r5 fused, rank transfer",
    "r6_split70": "r6 70/30 split",
    "r7_acq4": "r7 second cut",
}
METRIC_LABEL = {"cost": "cost (FPR + FNR; lower is better)", "f025": "F-beta 1/4", "f1": "F1", "f4": "F-beta 4"}


def _style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def ladder(data: Path, out: Path) -> None:
    c = pd.read_csv(data / "curves_4184.csv").set_index(["metric", "arm"])
    clicks = np.array([int(x) for x in c.columns])
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharex=True)
    for ax, metric in zip(axes, METRIC_LABEL, strict=True):
        for i, rung in enumerate(RUNG_LABEL):
            ax.plot(
                clicks, c.loc[(metric, rung)].to_numpy(float), color=SERIES[i], linewidth=1.6, label=RUNG_LABEL[rung]
            )
        ax.set_title(METRIC_LABEL[metric], fontsize=9, color=INK, loc="left")
        ax.set_xlabel("votes", fontsize=8, color=MUTED)
        _style(ax)
    axes[1].set_ylim(0, None)
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle(
        "#4184's ladder on the 0.44% pool, re-scored: r4 wins on cost, r1 wins at every beta",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out / "ladder_044.png", dpi=130)
    plt.close(fig)


def ladder_runs(data: Path, out: Path) -> None:
    r = pd.read_csv(data / "runs_4184.csv")
    key = ["dataset", "embedder", "category", "seed"]
    w = r.pivot_table(index=key, columns="arm", values=["cost", "f1"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, metric, title in zip(
        axes, ("cost", "f1"), ("cost at vote 150 (lower is better)", "F1 at vote 150"), strict=True
    ):
        x, y = w[(metric, "r1_xcal")], w[(metric, "r4_rawmean")]
        ax.scatter(x, y, s=9, color=SERIES[0], alpha=0.45, linewidths=0)
        lim = [0, max(float(np.nanmax(x)), float(np.nanmax(y))) * 1.02]
        ax.plot(lim, lim, color=MUTED, linewidth=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        better = float(np.mean(y < x)) if metric == "cost" else float(np.mean(y > x))
        ax.set_title(f"{title}: r4 better in {better:.0%} of runs", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("r1 cross-calibration", fontsize=8, color=MUTED)
        ax.set_ylabel("r4 fused, raw mean", fontsize=8, color=MUTED)
        _style(ax)
    fig.suptitle("Each dot is one run (category x seed), 720 runs", fontsize=9, color=MUTED, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "ladder_runs.png", dpi=130)
    plt.close(fig)


def fraction_runs(data: Path, out: Path) -> None:
    r = pd.read_csv(data / "runs_3287.csv")
    r["arm"] = r["arm"].astype(float)
    key = ["dataset", "embedder", "category", "seed"]
    spaces = sorted(r["embedder"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, metric in zip(axes, ("f025", "f4"), strict=True):
        w = r.pivot_table(index=key, columns="arm", values=metric)
        d = (w[0.3] - w[0.5]).dropna().reset_index()
        for i, sp in enumerate(spaces):
            v = d.loc[d["embedder"] == sp, 0].to_numpy(float)
            jitter = (np.random.default_rng(i).random(v.size) - 0.5) * 0.5
            ax.scatter(v, i + jitter, s=8, color=SERIES[0], alpha=0.4, linewidths=0)
            ax.plot([v.mean()] * 2, [i - 0.35, i + 0.35], color=INK, linewidth=2)
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.set_yticks(range(len(spaces)))
        ax.set_yticklabels(spaces, fontsize=8)
        ax.set_title(f"{METRIC_LABEL[metric]}: 0.3 minus 0.5, per run at vote 150", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("right of 0: the 0.3 split is better (bar = mean)", fontsize=8, color=MUTED)
        _style(ax)
    fig.tight_layout()
    fig.savefig(out / "fraction_runs.png", dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    ladder(args.data, args.out)
    ladder_runs(args.data, args.out)
    fraction_runs(args.data, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
