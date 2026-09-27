"""#3945 figures, from the analysis CSVs only (``analyze_c_path.py`` / ``analyze_stage_d.py``).

python figures.py --rpath DIR --stageD DIR --out docs/experiments/2026-09-25-overtrain-3945/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, GREEN, GREY, RED = "#2a6fdb", "#e07b00", "#2f9e44", "#8a8a8a", "#c92a2a"
DS = {"vg": "visual_genome_m", "coco": "coco_val", "caltech": "caltech101_m"}


def fig_c_path(rp: Path, out: Path) -> None:
    """Held-out oracle cost against C, one line per click count, one panel per dataset."""
    p = pd.read_csv(rp / "R_path.csv")
    ts = sorted(p["t"].unique())
    show = [t for t in (10, 20, 40, 80, 150, 250, 400) if t in ts]
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, ds in zip(axes, ("vg", "coco", "caltech")):
        sub = p[p["ds"] == ds]
        for i, t in enumerate(show):
            s = sub[sub["t"] == t].sort_values("C")
            # Each curve relative to its own C = 1 point, so the lines share a zero.
            ref = s.loc[np.isclose(s["C"], 1.0), "test_oracle_cost"].iloc[0]
            ax.plot(
                s["C"],
                s["test_oracle_cost"] - ref,
                marker="o",
                ms=3,
                color=cmap(i / max(len(show) - 1, 1)),
                label=f"click {t}",
            )
        ax.axvline(1.0, color=GREY, ls="--", lw=1)
        ax.axhline(0, color=GREY, lw=0.8)
        ax.set_xscale("log")
        ax.set_title(DS[ds])
        ax.set_xlabel("C (shipped = 1, dashed)")
    axes[0].set_ylabel("held-out oracle cost minus at C = 1\n(below zero: better than shipped)")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "c_path.png", dpi=130)
    plt.close(fig)


def fig_penalty(rp: Path, out: Path) -> None:
    p = pd.read_csv(rp / "R_penalty.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    # analyze_c_path.py writes the dataset scopes by their short names.
    for scope, color in (("all", "black"), ("vg", BLUE), ("coco", ORANGE), ("caltech", GREEN)):
        s = p[p["scope"] == scope].sort_values("t")
        ax.errorbar(
            s["t"],
            s["penalty"],
            yerr=2 * s["se"],
            color=color,
            marker="o",
            ms=3,
            capsize=2,
            lw=1.4,
            label=DS.get(scope, "pooled"),
        )
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xlabel("clicks (votes) in the session")
    ax.set_ylabel("oracle cost at C = 1 minus at best fixed C\n(cross-fitted, ±2 SE)")
    ax.set_title("What C = 1 gives away on the same votes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "penalty_vs_clicks.png", dpi=130)
    plt.close(fig)


def fig_gauges(rp: Path, out: Path, t: int) -> None:
    """The truth (held-out AUROC) and the vote-only gauges along C, at one click count."""
    p = pd.read_csv(rp / "R_path.csv")
    ts = sorted(p["t"].unique())
    t = min(ts, key=lambda x: abs(x - t))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, ds in zip(axes, ("vg", "coco")):
        s = p[(p["ds"] == ds) & (p["t"] == t)].sort_values("C")
        for col, color, label in (
            ("test_auroc", "black", "held-out AUROC (truth)"),
            ("app_auroc", RED, "app's cross-calibration split AUROC"),
            ("cv_auroc", ORANGE, "5-fold CV AUROC on the votes"),
        ):
            v = s[col] - s[col].max()
            ax.plot(s["C"], v, marker="o", ms=3, color=color, label=label)
        ax.axvline(1.0, color=GREY, ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_title(f"{DS[ds]}, click {t}")
        ax.set_xlabel("C")
    axes[0].set_ylabel("AUROC minus its own maximum over C")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "gauges_vs_truth.png", dpi=130)
    plt.close(fig)


def fig_stage_d_curves(sd: Path, out: Path) -> None:
    c = pd.read_csv(sd / "D_curve.csv")
    for metric, title in (("cost", "cost at the shipped cut"), ("oracle_cost", "oracle cost (ranking only)")):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
        for ax, ds in zip(axes, ("vg", "coco", "caltech")):
            for arm, color, label in (("svm", BLUE, "C = 1 (shipped)"), ("svmc01", ORANGE, "C = 0.1")):
                s = c[(c["ds"] == ds) & (c["arm"] == arm)].sort_values("t")
                ax.plot(s["t"], s[metric], color=color, lw=1.4, label=label)
            ax.axvline(150, color=GREY, ls=":", lw=1)
            ax.set_title(DS[ds])
            ax.set_xlabel("clicks")
        axes[0].set_ylabel(f"{title}, mean over categories × seeds")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"stageD_{metric}_vs_clicks.png", dpi=130)
        plt.close(fig)


def fig_stage_d_runs(sd: Path, out: Path, dataset: str = "visual_genome_m") -> None:
    """Every C = 1 session on one dataset as its own line: cost and oracle cost."""
    r = pd.read_csv(sd / "D_runs.csv.gz")
    r = r[(r["arm"] == "svm") & (r["dataset"] == dataset)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, metric, title in zip(
        axes, ("cost", "oracle_cost"), ("cost at the shipped cut", "oracle cost (ranking only)")
    ):
        for _, g in r.groupby(["embedder", "category", "seed"]):
            g = g.sort_values("t")
            ax.plot(g["t"], g[metric].rolling(10, min_periods=1).mean(), color=BLUE, alpha=0.08, lw=0.8)
        m = r.groupby("t")[metric].mean()
        ax.plot(m.index, m.to_numpy(), color="black", lw=2, label="mean")
        ax.set_title(f"{dataset}, C = 1: {title}")
        ax.set_xlabel("clicks")
    axes[0].set_ylabel("per session (10-click rolling mean)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / f"stageD_runs_{dataset}.png", dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rpath", required=True, type=Path, help="analyze_c_path.py --out")
    ap.add_argument("--stageD", required=True, type=Path, help="analyze_stage_d.py --out")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--gauge-t", type=int, default=80)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_c_path(args.rpath, args.out)
    fig_penalty(args.rpath, args.out)
    fig_gauges(args.rpath, args.out, args.gauge_t)
    fig_stage_d_curves(args.stageD, args.out)
    for ds in ("visual_genome_m", "coco_val"):
        fig_stage_d_runs(args.stageD, args.out, ds)
    print(f"figures -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
