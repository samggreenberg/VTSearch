#!/usr/bin/env python
"""Figures for the #3348 report.

    python figures_node_models_3348.py --results DIR --analysis DIR --out DIR

``pit_patch.png``      in-domain PIT on dinov3_patch: shipped vs the split fix vs the control
``separation.png``     guard separation per embedder for the shipped model and the passers
``calibration_axis.png`` the same statistic, loo vs split: rate below 0.05 per embedder
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

EMBEDDERS = ["dinov3_patch", "clip", "clip_l", "siglip", "siglip2_l"]
SHOWN = {
    "grid_mean_shipped": ("shipped (vMF, LOO grid, path mean)", "#b2182b"),
    "vmf_loo_deepest": ("vMF, LOO, deepest", "#ef8a62"),
    "vmf_split_deepest": ("vMF, split, deepest", "#67a9cf"),
    "knn_split_median": ("kNN node, split, median", "#2166ac"),
    "knn_global": ("kNN global, split (shipped now)", "#1a9850"),
}


def pit_patch(results: Path, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    grid = np.linspace(0, 1, 201)
    for var, (label, color) in SHOWN.items():
        first = True
        for f in sorted(results.glob("pvals_dinov3_patch__s*.npz")):
            z = np.load(f)
            for key in z.files:
                if key.startswith("pit|") and key.endswith(f"|{var}"):
                    p = np.sort(z[key].astype(np.float64))
                    ecdf = np.searchsorted(p, grid, side="right") / p.size
                    ax.plot(grid, ecdf, color=color, lw=0.8, alpha=0.6, label=label if first else None)
                    first = False
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.axvline(0.05, color="grey", lw=0.5)
    ax.set_xlabel("p-value")
    ax.set_ylabel("share of held-out in-domain items at or below")
    ax.set_title("dinov3_patch, in-domain PIT (15 cells per model)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "pit_patch.png", dpi=130)
    plt.close(fig)


def separation(table: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    vars_ = list(SHOWN)
    w = 0.8 / len(vars_)
    x = np.arange(len(EMBEDDERS))
    t = table.set_index(["embedder", "variant"])
    for i, v in enumerate(vars_):
        ys = [t.loc[(e, v), "separation"] if (e, v) in t.index else np.nan for e in EMBEDDERS]
        ax.bar(x + (i - (len(vars_) - 1) / 2) * w, ys, w, color=SHOWN[v][1], label=SHOWN[v][0])
    ax.axhline(0.5, color="grey", lw=0.6, ls=":")
    ax.set_xticks(x, EMBEDDERS)
    ax.set_ylabel("separation\n(detect different corpus - fire on own data)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "separation.png", dpi=130)
    plt.close(fig)


def calibration_axis(table: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    t = table.set_index(["embedder", "variant"])
    stats = ["vmf", "knn", "mix", "ppca"]
    x = np.arange(len(EMBEDDERS))
    for j, s in enumerate(stats):
        for calib, marker in (("loo", "o"), ("split", "s")):
            v = f"{s}_{calib}_deepest"
            ys = [t.loc[(e, v), "rate05"] for e in EMBEDDERS]
            ax.scatter(
                x + (j - 1.5) * 0.12,
                ys,
                marker=marker,
                s=28,
                facecolor="none" if calib == "loo" else f"C{j}",
                edgecolor=f"C{j}",
                label=f"{s}, {calib}",
            )
    ax.axhline(0.05, color="k", lw=0.8, ls="--")
    ax.axhspan(0.03, 0.075, color="grey", alpha=0.12)
    ax.set_xticks(x, EMBEDDERS)
    ax.set_ylabel("held-out share below 0.05\n(deepest node, nominal 0.05)")
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "calibration_axis.png", dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(list(argv) if argv is not None else None)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(Path(args.analysis) / "table.csv")
    pit_patch(Path(args.results), out)
    separation(table, out)
    calibration_axis(table, out)
    print(json.dumps(sorted(p.name for p in out.glob("*.png"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
