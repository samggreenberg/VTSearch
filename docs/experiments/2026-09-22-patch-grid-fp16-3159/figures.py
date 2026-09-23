#!/usr/bin/env python3
"""Figures for #3159, from the files under ``measurements/``.

    python figures.py            # regenerate every figure into figures/

Inputs live beside this script, so the figures rebuild without the GRID, the
pile or a GPU.  ``measurements/drift/`` is written by
``scripts/experiments/precision/grid_drift_3159.py``; ``measurements/bench/`` by
``scripts/experiments/precision/analyze_grid_3159.py``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
MEAS = HERE / "measurements"
FIGS = HERE / "figures"
DPI = 130

# Validated categorical slots 1 and 2 (the #4056 figures' pair: worst adjacent
# CVD dE 9.2).  Every series is also labelled, so identity is never colour alone.
FP32 = "#2a78d6"
FP16 = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8d7d2"
MARGIN = 0.005
BANDS = ["sub_patch", "patch_to_leaf", "leaf_to_4x", "above_4x"]
BAND_LABEL = {
    "sub_patch": "sub-patch\n(<1/196)",
    "patch_to_leaf": "patch→leaf",
    "leaf_to_4x": "leaf→4×",
    "above_4x": "above 4×",
    "image_level": "image-level\n(Caltech)",
}
DATASETS = ["visual_genome_m", "coco_val", "caltech101_m"]


def _style(ax) -> None:
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def drift_figures() -> None:
    q = pd.read_csv(MEAS / "drift" / "queries.csv.gz")
    q = q[q["kind"] == "exemplar"]
    order = [*BANDS, "image_level"]

    # 1. pooled-score drift per query, by band: median and worst medias
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    for ax, col, title in (
        (axes[0], "abs_drift_median", "median |Δ pooled score| over the gallery"),
        (axes[1], "abs_drift_max", "worst |Δ pooled score| over the gallery"),
    ):
        bands = [b for b in order if b in set(q["band"])]
        # In units of 1e-5 so the axis carries no offset exponent.
        data = [q.loc[q["band"] == b, col].to_numpy() * 1e5 for b in bands]
        ax.boxplot(data, widths=0.5, showfliers=True, flierprops={"markersize": 3, "markeredgecolor": MUTED})
        ax.set_xticks(range(1, len(bands) + 1), [BAND_LABEL[b] for b in bands])
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_ylabel("|fp16 − fp32| score, in units of 0.00001", fontsize=8, color=MUTED)
        _style(ax)
    fig.suptitle("One query = one Good box vote (nearest patch); score = max over 197 rows", fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(FIGS / "drift_by_band.png", dpi=DPI)
    plt.close(fig)

    # 2. dAP per query, by band and dataset, against the decision margin
    fig, ax = plt.subplots(figsize=(10, 3.8))
    rng = np.random.default_rng(0)
    xt, xl = [], []
    x = 0
    for ds in DATASETS:
        for b in order:
            g = q[(q["dataset"] == ds) & (q["band"] == b)]
            if g.empty:
                continue
            ax.scatter(x + rng.uniform(-0.25, 0.25, len(g)), g["d_ap"], s=6, color=FP32, alpha=0.5, lw=0)
            xt.append(x)
            xl.append(
                f"{ds.replace('_m', '').replace('visual_genome', 'VG').replace('caltech101', 'Caltech')}\n{BAND_LABEL[b]}"
            )
            x += 1
        x += 0.6
    for s in (1, -1):
        ax.axhline(s * MARGIN, color=FP16, lw=1, ls="--")
    ax.text(x - 0.8, MARGIN, " 0.005 margin", color=INK, fontsize=7, va="bottom", ha="right")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(xt, xl, fontsize=6.5)
    ax.set_ylabel("ΔAP per query (fp16 − fp32)", fontsize=8, color=MUTED)
    ax.set_title("Ranking quality per exemplar query: every dot is one Good vote", fontsize=9, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "dap_per_query.png", dpi=DPI)
    plt.close(fig)


def bench_figures() -> None:
    bench = MEAS / "bench"
    if not (bench / "trajectory.csv").exists():
        print("no bench measurements yet; skipping bench figures")
        return
    _bench_trajectories(pd.read_csv(bench / "trajectory.csv"))
    _bench_per_cell(pd.read_csv(bench / "per_step.csv.gz"))
    _bench_by_band(pd.read_csv(bench / "paired_by_split.csv"))


def _bench_trajectories(traj: pd.DataFrame) -> None:
    # 3. mean cost / AP trajectory per dataset, both arms (weighted over bands)
    for metric, a16, a32, label in (
        ("cost", "cost_fp16", "cost_fp32", "cost (FPR + FNR)"),
        ("ap", "ap_fp16", "ap_fp32", "average precision"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
        for ax, ds in zip(axes, ["visual_genome_m", "coco_val"], strict=True):
            g = traj[traj["dataset"] == ds]
            w = g.groupby("t").apply(
                lambda d, a16=a16, a32=a32: pd.Series(
                    {"fp16": np.average(d[a16], weights=d["n"]), "fp32": np.average(d[a32], weights=d["n"])}
                ),
                include_groups=False,
            )
            ax.plot(w.index, w["fp32"], color=FP32, lw=2, label="fp32 grid")
            ax.plot(w.index, w["fp16"], color=FP16, lw=2, ls="--", label="fp16 grid (shipped)")
            ax.set_title(ds, fontsize=9, color=INK)
            ax.set_xlabel("votes spent", fontsize=8, color=MUTED)
            _style(ax)
        axes[0].set_ylabel(f"mean {label}", fontsize=8, color=MUTED)
        axes[0].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(FIGS / f"bench_trajectory_{metric}.png", dpi=DPI)
        plt.close(fig)


def _bench_per_cell(steps: pd.DataFrame) -> None:
    # 4. every cell's paired difference, per dataset (the per-run view)
    key = ["dataset", "category", "seed"]
    for metric, ref, arm in (
        ("cost", "cost_ref", "cost_arm"),
        ("ap", "average_precision_ref", "average_precision_arm"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
        for ax, ds in zip(axes, ["visual_genome_m", "coco_val"], strict=True):
            g = steps[steps["dataset"] == ds]
            for _, c in g.groupby(key):
                c = c.sort_values("t")
                ax.plot(c["t"], c[ref] - c[arm], color=FP32, lw=0.4, alpha=0.12)
            m = g.assign(d=g[ref] - g[arm]).groupby("t")["d"].mean()
            ax.plot(m.index, m.values, color=INK, lw=1.8, label="mean over cells")
            for s in (1, -1):
                ax.axhline(s * MARGIN, color=FP16, lw=1, ls="--", label="0.005 margin" if s == 1 else None)
            ax.set_title(ds, fontsize=9, color=INK)
            ax.set_xlabel("votes spent", fontsize=8, color=MUTED)
            _style(ax)
        axes[0].set_ylabel(f"Δ{metric}, fp16 − fp32 (one line per cell)", fontsize=8, color=MUTED)
        axes[0].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(FIGS / f"bench_percell_{metric}.png", dpi=DPI)
        plt.close(fig)


def _bench_by_band(tab: pd.DataFrame) -> None:
    # 5. paired difference by dataset x band, with 2 SE bars, whole trajectory
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.8))
    for ax, metric in zip(axes, ["cost", "average_precision"], strict=True):
        t = tab[(tab["metric"] == metric) & (tab["window"] == "all") & (tab["split"] == "dataset+band")]
        labels, xs = [], []
        for i, (_, r) in enumerate(t.iterrows()):
            # table stores fp32 - fp16; plot fp16 - fp32 to match the other figures
            ax.errorbar(i, -r["diff_fp32_minus_fp16"], yerr=2 * r["se"], fmt="o", color=FP32, ms=5, capsize=3)
            labels.append(
                f"{r['dataset'].replace('visual_genome_m', 'VG').replace('coco_val', 'COCO')}\n{BAND_LABEL.get(r['band'], r['band'])}\nn={int(r['n_cells'])}"
            )
            xs.append(i)
        for s in (1, -1):
            ax.axhline(s * MARGIN, color=FP16, lw=1, ls="--")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(xs, labels, fontsize=6)
        ax.set_title(f"Δ{metric} (fp16 − fp32), mean over the trajectory, ±2 SE", fontsize=9, color=INK)
        _style(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "bench_by_band.png", dpi=DPI)
    plt.close(fig)


def main() -> int:
    FIGS.mkdir(exist_ok=True)
    drift_figures()
    bench_figures()
    print(f"wrote {sorted(p.name for p in FIGS.glob('*.png'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
