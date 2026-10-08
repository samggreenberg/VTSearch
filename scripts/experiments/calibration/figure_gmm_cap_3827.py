#!/usr/bin/env python3
"""Figures for the ``_GMM_MAX_SAMPLES`` pricing (#3827), from ``analyze_gmm_cap_3827.py``'s output.

* ``cost_by_cap.png`` - median seconds per fit against the cap, per corpus
  size: the no-vote line's mixture and the fold-anchored cut.
* ``cut_by_cap.png`` - where the fold-anchored cut sits (the share of the
  corpus it returns) against the cap, per world, with the bench-size corpus
  as a dashed reference: the votes' share of the fit falls as the cap rises.
* ``line_by_cap.png`` - the shipped no-vote line's change in F1 against the
  shipped cap, per world and size, at P = 50% and 90%.

    python figure_gmm_cap_3827.py --run OUT --out FIGDIR [--t 150]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

CAP_ORDER = (5_000, 10_000, 25_000, 50_000, 100_000, 0)
CAP_LABEL = {5_000: "5k", 10_000: "10k", 25_000: "25k", 50_000: "50k\n(shipped)", 100_000: "100k", 0: "none"}
SIZE_LABEL = {100_000: "100k items", 250_000: "250k", 2_000_000: "2M"}
WORLDS = ("0.1%", "0.44%", "5%")


def _x() -> tuple[list[int], list[str]]:
    return list(range(len(CAP_ORDER))), [CAP_LABEL[c] for c in CAP_ORDER]


def cost(timing: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    xs, labels = _x()
    for ax, col, title in (
        (axes[0], "mix_median_s", "no-vote line: the mixture fit"),
        (axes[1], "cut_median_s", "fold-anchored cut (both folds + transfer)"),
    ):
        for n, g in timing[timing["size"] > 0].groupby("size"):
            g = g.set_index("cap").reindex(CAP_ORDER)
            ax.plot(xs, g[col], marker="o", label=SIZE_LABEL.get(n, str(n)))
        ax.set_yscale("log")
        ax.set_xticks(xs, labels)
        ax.set_xlabel("cap (scores the fit sees)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("median seconds per fit")
    axes[1].legend(title="corpus", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "cost_by_cap.png", dpi=110)
    plt.close(fig)


def cut(cuts: pd.DataFrame, out: Path, t: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
    xs, labels = _x()
    c = cuts[cuts["t"] == t]
    for ax, world in zip(axes, WORLDS, strict=True):
        w = c[c["world"] == world]
        for n, g in w[w["size"] > 0].groupby("size"):
            g = g.set_index("cap").reindex(CAP_ORDER)
            ax.plot(xs, 100 * g["cut0_q"], marker="o", label=SIZE_LABEL.get(n, str(n)))
        bench = w[(w["size"] == 0) & (w["cap"] == 50_000)]
        if not bench.empty:
            ax.axhline(100 * float(bench["cut0_q"].iloc[0]), ls="--", color="k", lw=1, label="bench-size corpus")
        ax.set_xticks(xs, labels)
        ax.set_title(f"{world} prevalence", fontsize=10)
        ax.set_xlabel("cap")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"% of the corpus the cut returns (click {t})")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "cut_by_cap.png", dpi=110)
    plt.close(fig)


def line(lines: pd.DataFrame, out: Path, t: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    xs, labels = _x()
    x = lines[(lines["t"] == t) & (lines["rule"] == "min-fixed-gmm") & (lines["size"] > 0)]
    for ax, floor in zip(axes, (0.5, 0.9), strict=True):
        for (world, n), g in x[x["floor"] == floor].groupby(["world", "size"]):
            if world == "5%":
                continue  # identical in every arm: the mixture never says fewer than 32 there
            g = g.set_index("cap").reindex(CAP_ORDER)
            ax.errorbar(xs, g["d_f1"], yerr=g["se_d_f1"], marker="o", capsize=2, label=f"{world}, {SIZE_LABEL[n]}")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(xs, labels)
        ax.set_title(f"P = {round(floor * 100)}%", fontsize=10)
        ax.set_xlabel("cap")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"F1 change of the no-vote line vs shipped (click {t})")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "line_by_cap.png", dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--run", type=Path, required=True, help="analyze_gmm_cap_3827.py's --out")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--t", type=int, default=150)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    cost(pd.read_csv(args.run / "timing.csv"), args.out)
    cut(pd.read_csv(args.run / "summary_cuts.csv"), args.out, args.t)
    line(pd.read_csv(args.run / "summary_lines.csv"), args.out, args.t)
    print(f"figures -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
