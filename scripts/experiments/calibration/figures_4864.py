#!/usr/bin/env python3
"""#4864's figures, from the CSVs ``split_cadence_4864.py`` and ``analyze_split_rounding_4864.py`` write.

    python figures_4864.py --step0 STEP0_DIR --data ANALYSIS_DIR --out FIGDIR

* ``step0_cadence.png``  - step 0: the held-out total's rounding residual, its
  shared amplitude per frequency under each rule, against the sign-flip null.
* ``contrasts.png``      - Δ-objective (nearest - dither) ±2 SE per path and reading,
  with the ship rule's -0.003.
* ``curves.png``         - the objective over votes per arm, and the paired Δ ±2 SE.
* ``ripple.png``         - the high-passed mean curves' periodograms against the null.
* ``switches.png``       - count line <-> labels line switches per run, per arm.
* ``coin.png``           - which line each rule draws by Goods, and the dither's coin
  as a randomized trial of the two lines at one count of Goods.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DPI = 130
RULE_COLORS = {"dither": "#4C72B0", "nearest": "#DD8452", "round": "#8C8C8C"}
ARM_COLORS = {"dither": "#4C72B0", "nearest": "#DD8452", "r_dither": "#4C72B0", "r_nearest": "#DD8452"}
PATHS = {"binary": ("dither", "nearest"), "region": ("r_dither", "r_nearest")}
PATH_TITLES = {"binary": "binary, SigLIP (split 0.3)", "region": "region, SigLIP+DINOv3 patches (split 0.5)"}
TEST_FREQS = {"binary": [0.3, 0.1, 0.4], "region": [0.5, 0.25]}


def step0(step0_dir: Path, out: Path) -> None:
    spec = pd.read_csv(step0_dir / "spectra.csv")
    spec = spec[spec["quantity"] == "held_total"]
    grids = list(dict.fromkeys(spec["grid"]))
    fig, axes = plt.subplots(1, len(grids), figsize=(6.2 * len(grids), 3.8), squeeze=False)
    for ax, grid in zip(axes[0], grids):
        g = spec[spec["grid"] == grid]
        for rule, c in RULE_COLORS.items():
            r = g[g["rule"] == rule]
            ax.plot(r["freq"], r["amplitude"], color=c, lw=1.4, label=rule)
            ax.fill_between(r["freq"], 0, r["null_p95"], color=c, alpha=0.12, lw=0)
        for f in TEST_FREQS["binary" if grid == "binary" else "region"]:
            ax.axvline(f, color="k", lw=0.6, ls=":")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1)
        ax.set_xlabel("frequency (cycles per vote)")
        ax.set_title(PATH_TITLES.get(grid, grid), fontsize=10)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("shared amplitude of the held-out total's\nrounding residual (votes, per fold)")
    axes[0][0].legend(frameon=False, fontsize=9)
    fig.suptitle("Step 0: do the split sizes step together across runs? (#4583's trajectories)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "step0_cadence.png", dpi=DPI)
    plt.close(fig)


def contrasts(data: Path, out: Path) -> None:
    p = pd.read_csv(data / "paired.csv")
    p = p[p["stratum"] == "all"]
    reads = ["objective, votes 1-150", "objective, after the check"]
    labels = ["mean over votes 1-150", "after the check"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    y = 0
    ticks, names = [], []
    for path in PATHS:
        for read, lab in zip(reads, labels):
            r = p[(p["path"] == path) & (p["read"] == read)]
            if r.empty:
                continue
            d, se = float(r["delta"].iloc[0]), float(r["se"].iloc[0])
            ax.errorbar(d, y, xerr=2 * se, fmt="o", color="#DD8452" if "after" in read else "#4C72B0", capsize=3)
            ticks.append(y)
            names.append(f"{path}: {lab}")
            y -= 1
        y -= 0.5
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(-0.003, color="#C44E52", lw=1, ls="--", label="ship rule's -0.003")
    ax.set_yticks(ticks, names)
    ax.set_xlabel("Δ-objective, nearest - dither (±2 SE, clustered on category)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "contrasts.png", dpi=DPI)
    plt.close(fig)


def curves(data: Path, out: Path) -> None:
    c = pd.read_csv(data / "curves.csv", index_col="t")
    d = pd.read_csv(data / "delta_curves.csv", index_col="t")
    paths = [p for p in PATHS if all(a in c.columns for a in PATHS[p])]
    fig, axes = plt.subplots(
        2,
        len(paths),
        figsize=(6.2 * len(paths), 5.6),
        squeeze=False,
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    for j, path in enumerate(paths):
        top, bot = axes[0][j], axes[1][j]
        for arm in PATHS[path]:
            top.plot(c.index, c[arm], color=ARM_COLORS[arm], lw=1.3, label=arm)
        top.set_title(PATH_TITLES[path], fontsize=10)
        top.set_ylabel("objective (F1, withheld half)")
        top.legend(frameon=False, fontsize=9)
        top.grid(alpha=0.3)
        m, se = d[f"{path}:mean"], d[f"{path}:se"]
        bot.plot(m.index, m, color="k", lw=1)
        bot.fill_between(m.index, m - 2 * se, m + 2 * se, color="k", alpha=0.15, lw=0)
        bot.axhline(0, color="k", lw=0.6)
        bot.set_ylabel("nearest - dither\n(±2 SE)")
        bot.set_xlabel("votes")
        bot.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=DPI)
    plt.close(fig)


def ripple(data: Path, out: Path) -> None:
    s = pd.read_csv(data / "spectra.csv")
    paths = list(dict.fromkeys(s["path"]))
    curves_ = ["dither/r_dither", "nearest/r_nearest", "nearest - dither"]
    fig, axes = plt.subplots(len(paths), 3, figsize=(13, 3.2 * len(paths)), squeeze=False, sharey=True)
    for i, path in enumerate(paths):
        for j, name in enumerate(curves_):
            ax = axes[i][j]
            arm = name if name == "nearest - dither" else name.split("/")[0 if path == "binary" else 1]
            g = s[(s["path"] == path) & (s["curve"] == arm)]
            if g.empty:
                ax.set_visible(False)
                continue
            ax.plot(g["freq"], g["amplitude"], color=ARM_COLORS.get(arm, "k"), lw=1.2)
            ax.fill_between(g["freq"], 0, g["null_p95"], color="#8C8C8C", alpha=0.25, lw=0, label="null, 95th pct")
            ax.plot(g["freq"], 2 * g["null_median"], color="#C44E52", lw=0.8, ls="--", label="2x null median")
            for f in TEST_FREQS[path]:
                ax.axvline(f, color="k", lw=0.6, ls=":")
            ax.set_title(f"{path}: {arm}", fontsize=10)
            ax.set_xlabel("frequency (cycles per vote)")
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel("amplitude of the high-passed\nmean objective curve")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "ripple.png", dpi=DPI)
    plt.close(fig)


def switches(data: Path, out: Path) -> None:
    sw = pd.read_csv(data / "switches.csv")
    paths = list(dict.fromkeys(sw["path"]))
    fig, axes = plt.subplots(1, len(paths), figsize=(5.4 * len(paths), 3.2), squeeze=False)
    for ax, path in zip(axes[0], paths):
        g = sw[sw["path"] == path]
        top = int(max(g["switches"].max(), 1))
        bins = np.arange(-0.5, top + 1.5)
        for k, arm in enumerate(PATHS[path]):
            v = g[g["arm"] == arm]["switches"]
            h, _ = np.histogram(v, bins=bins)
            ax.bar(
                np.arange(top + 1) + (k - 0.5) * 0.4,
                h / max(len(v), 1),
                width=0.4,
                color=ARM_COLORS[arm],
                label=f"{arm} (mean {v.mean():.2f})",
            )
        ax.set_xlabel("switches between the count line and the labels line, per run")
        ax.set_ylabel("share of runs")
        ax.set_title(PATH_TITLES[path], fontsize=10)
        ax.legend(frameon=False, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "switches.png", dpi=DPI)
    plt.close(fig)


def coin(data: Path, out: Path) -> None:
    """The band's two sides: which line each rule draws by Goods, and what the labels line is worth there."""
    c = pd.read_csv(data / "coin.csv")
    c = c[pd.to_numeric(c["n_good"], errors="coerce").notna()].astype({"n_good": int})
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.4))
    g = c["n_good"]
    a.step(g, c["labels_share"], where="mid", color=ARM_COLORS["dither"], lw=1.6, label="dither (the app)")
    a.step(g, (g >= 9).astype(float), where="mid", color=ARM_COLORS["nearest"], lw=1.6, ls="--", label="nearest")
    a.set_xlabel("Goods voted")
    a.set_ylabel("share of clicks on the labels line")
    a.set_title("Which line the gate draws (binary, split 0.3)", fontsize=10)
    a.legend(frameon=False, fontsize=9)
    a.grid(alpha=0.3)
    k = c.dropna(subset=["labels_minus_count"])
    b.errorbar(k["n_good"], k["labels_minus_count"], yerr=2 * k["se"], fmt="o", color="k", capsize=3)
    for _, r in k.iterrows():
        b.annotate(
            f"{int(r['runs_both'])} runs",
            (r["n_good"], r["labels_minus_count"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
        )
    b.axhline(0, color="k", lw=0.6)
    b.axvspan(6.5, 8.5, color=ARM_COLORS["nearest"], alpha=0.12, lw=0)
    b.set_xlabel("Goods voted")
    b.set_ylabel("objective: labels line - count line\n(same run, ±2 SE)")
    b.set_title("The dither's coin as a trial (shaded: nearest draws the count line)", fontsize=10)
    b.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "coin.png", dpi=DPI)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--step0", type=Path)
    ap.add_argument("--data", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.step0:
        step0(args.step0, args.out)
    if args.data:
        contrasts(args.data, args.out)
        curves(args.data, args.out)
        ripple(args.data, args.out)
        switches(args.data, args.out)
        if (args.data / "coin.csv").exists():
            coin(args.data, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
