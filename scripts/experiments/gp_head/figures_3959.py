"""#3959 figures: the quality-over-clicks pair per dataset, plus the paired contrasts.

The curves go through the one implementation (``calibration/curves.py``); the
paired panels read only ``analyze_grid_3959.py``'s ``paired.csv``.

    python figures_3959.py --root /expscratch/$USER/gp-grid-3959 --analysis DIR --out FIGDIR
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "calibration"))
sys.path.insert(0, str(HERE))

from analyze_grid_3959 import ARMS, ENVS, PAIRS  # noqa: E402

GREY = "#8a8a8a"
BLUE, ORANGE = "#2a6fdb", "#e07b00"
DATASET_OF = {"better": "coco_better", "natural": "coco_val"}


def curves_pair(root: Path, out: Path, metric: str) -> None:
    import curves

    for env in ENVS:
        base = root / env
        dirs = [f"{a}/results" for a in ARMS if (base / a / "results" / "cells").exists()]
        if not dirs:
            continue
        frame = curves._load(base, dirs)
        frame["arm"] = frame["arm"].str.replace("/results", "", regex=False)
        arms = [d.split("/")[0] for d in dirs]
        bl_path = base / "prepare" / "results" / "text_baseline.csv"
        baseline = curves.text_sort_baseline(bl_path) if bl_path.exists() else None
        # One panel per EMBEDDER, never an average over two: `curves` draws a
        # panel per `dataset`, so the embedder is folded into that label (and
        # into the baseline's, which is keyed the same way).
        frame["dataset"] = frame["dataset"] + "·" + frame["embedder"]
        if baseline is not None:
            baseline["dataset"] = baseline["dataset"] + "·" + baseline["embedder"]
        curves.quality_vs_clicks(
            frame,
            out / DATASET_OF[env],
            arms=arms,
            metric=metric,
            baseline=baseline,
            lower_is_better=metric not in ("average_precision", "auroc"),
        )


def fig_paired(an: Path, out: Path) -> None:
    """Area-under-the-curve contrasts, split into cost = ranking + cut, per dataset."""
    p = pd.read_csv(an / "paired.csv")
    p = p[p.window == "aulc"]
    labels = [f"{a} − {r}\n({w})" for a, r, w in PAIRS]
    for scope in ["ALL", *sorted(s for s in p.scope.unique() if "/" not in s and s != "ALL")]:
        sub = p[p.scope == scope]
        if sub.empty:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(15, 0.42 * len(PAIRS) + 1.6), sharey=True)
        for ax, m, title in zip(
            axes,
            ("cost", "oracle_cost", "regret"),
            ("cost (what the user gets)", "oracle cost (the ranking alone)", "regret (the cut's own loss)"),
            strict=True,
        ):
            y = np.arange(len(PAIRS))
            means, ses = [], []
            for a, r, _ in PAIRS:
                row = sub[(sub.arm == a) & (sub.ref == r) & (sub.metric == m)]
                means.append(row["mean"].iloc[0] if len(row) else np.nan)
                ses.append(row["se"].iloc[0] if len(row) else np.nan)
            means, ses = np.asarray(means), np.asarray(ses)
            colors = [BLUE if abs(mu) >= 2 * s else GREY for mu, s in zip(means, ses, strict=True)]
            for yi, mu, s, c in zip(y, means, ses, colors, strict=True):
                ax.errorbar(mu, yi, xerr=2 * s, fmt="o", color=c, capsize=3)
            ax.axvline(0, color=GREY, lw=1)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("arm − ref, mean over clicks 8-150 (±2 SE)\n< 0: the arm is better")
        axes[0].set_yticks(np.arange(len(PAIRS)), labels, fontsize=8)
        axes[0].invert_yaxis()
        n = int(sub[sub.metric == "cost"]["n"].max())
        fig.suptitle(
            f"#3959 paired contrasts, {scope}: up to {n} cells per pair (grey = within 2 SE of zero, not resolvable)",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out / f"paired_aulc__{scope}.png", dpi=130)
        plt.close(fig)


def fig_by_click(an: Path, out: Path) -> None:
    """The three rule-matched head contrasts and the shipped-rule contrast, by click."""
    p = pd.read_csv(an / "paired.csv")
    picks = [PAIRS[0], PAIRS[1], PAIRS[2], PAIRS[3]]
    for scope in sorted(s for s in p.scope.unique() if "/" not in s and s != "ALL"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        for ax, m in zip(axes, ("cost", "oracle_cost"), strict=True):
            for (a, r, what), c in zip(picks, (BLUE, ORANGE, "#2f9e44", "#c92a2a"), strict=True):
                pts = []
                for w in ("t10", "t25", "t50", "t100", "t150"):
                    row = p[(p.window == w) & (p.scope == scope) & (p.arm == a) & (p.ref == r) & (p.metric == m)]
                    if len(row):
                        pts.append((int(w[1:]), row["mean"].iloc[0], row["se"].iloc[0]))
                if pts:
                    t, mu, s = zip(*pts, strict=True)
                    ax.errorbar(t, mu, yerr=2 * np.asarray(s), color=c, marker="o", capsize=3, label=f"{a} − {r}")
            ax.axhline(0, color=GREY, lw=1)
            ax.set_xscale("log")
            ax.set_xticks([10, 25, 50, 100, 150], ["10", "25", "50", "100", "150"])
            ax.set_xlabel("clicks")
            ax.set_ylabel(f"{m}: arm − ref (±2 SE)")
            ax.set_title(f"{m} by click, {scope} (< 0: arm better)", fontsize=10)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"by_click__{scope}.png", dpi=130)
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--analysis", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for metric in ("cost", "average_precision"):
        curves_pair(args.root, args.out, metric)
    fig_paired(args.analysis, args.out)
    fig_by_click(args.analysis, args.out)
    print(f"figures -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
