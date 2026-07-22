"""Report figures from the aggregated sweep (master.csv / per_cell.csv).

Chart forms follow the data's job:
  - separability over the (n_neighbors × min_dist) grid → SEQUENTIAL single-hue
    heatmap (magnitude; darker = better), one small-multiple per embedder.
  - separability vs n_neighbors → line, one CATEGORICAL color per dataset
    (Okabe-Ito, colorblind-safe, fixed order), single y-axis.
  - compaction delta (compacted − raw) → SIGN-keyed bars with a zero baseline
    (polarity: blue = compaction helps, red = it hurts).
  - seed stability vs n_neighbors → line per embedder.
  - separability vs neighbor-recall guard → scatter (shows purity isn't gamed).

Run after summarize.py:  python plots.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

# Okabe-Ito colorblind-safe categorical palette, fixed order (never cycled).
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7", "#F0E442", "#999999", "#000000"]
# Embedder colors: a validated 4-subset (all adjacent-pair CVD ΔE ≥ 11) so the
# four embedder lines never rely on a marginal color pair.
EMB_ORDER = ["clap", "clip", "siglip", "siglip_l"]
EMB_COLORS = {"clap": "#0072B2", "clip": "#E69F00", "siglip": "#009E73", "siglip_l": "#D55E00"}
HELP_C, HURT_C = "#0072B2", "#D55E00"  # compaction helps / hurts
INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
    }
)


def _cell():
    return pd.read_csv(C.RESULTS_ROOT / "per_cell.csv")


def _emb_present(df):
    return [e for e in EMB_ORDER if e in set(df["embedder"])]


def heatmaps(df):
    """Ratio over (n_neighbors × min_dist), averaged over that embedder's datasets."""
    raw = df[df["compact"] == False]  # noqa: E712
    embs = _emb_present(raw)
    fig, axes = plt.subplots(1, len(embs), figsize=(3.5 * len(embs), 3.6), squeeze=False)
    nns = sorted(raw["n_neighbors"].unique())
    mds = sorted(raw["min_dist"].unique())
    vmin, vmax = raw["ratio"].quantile(0.02), raw["ratio"].max()
    for ax, emb in zip(axes[0], embs):
        sub = raw[raw["embedder"] == emb]
        grid = sub.pivot_table(index="min_dist", columns="n_neighbors", values="ratio", aggfunc="mean")
        grid = grid.reindex(index=mds, columns=nns)
        im = ax.imshow(grid.values, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
        ax.set_xticks(range(len(nns)))
        ax.set_xticklabels(nns, fontsize=8)
        ax.set_yticks(range(len(mds)))
        ax.set_yticklabels(mds, fontsize=8)
        ax.set_title(emb)
        ax.set_xlabel("n_neighbors")
        ax.grid(False)
        # mark the argmax cell
        gv = np.nan_to_num(grid.values, nan=-1)
        r, c = np.unravel_index(np.argmax(gv), gv.shape)
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="#111111", lw=2))
    axes[0][0].set_ylabel("min_dist")
    cb = fig.colorbar(im, ax=axes[0], fraction=0.025, pad=0.02)
    cb.set_label("separability ratio (2-D / high-D)  ·  darker = better", fontsize=8)
    fig.suptitle(
        "Ceiling-normalized taxonomy separability over the UMAP grid (raw layouts; black box = best cell)", fontsize=11
    )
    p = C.FIG_DIR / "fig_heatmaps.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def nn_curves(df):
    """Separability vs n_neighbors, one line per dataset, faceted by embedder."""
    raw = df[df["compact"] == False]  # noqa: E712
    embs = _emb_present(raw)
    fig, axes = plt.subplots(1, len(embs), figsize=(3.6 * len(embs), 3.6), squeeze=False, sharey=True)
    for ax, emb in zip(axes[0], embs):
        sub = raw[raw["embedder"] == emb]
        for i, (ds, dsub) in enumerate(sorted(sub.groupby("dataset"), key=lambda x: x[1]["N"].iloc[0])):
            curve = dsub.groupby("n_neighbors")["ratio"].mean()
            n = int(dsub["N"].iloc[0])
            ax.plot(curve.index, curve.values, "-o", ms=4, lw=2, color=OKABE[i % len(OKABE)], label=f"{ds} (N={n})")
        ax.set_xscale("log")
        ax.set_title(emb)
        ax.set_xlabel("n_neighbors (log)")
        ax.legend(fontsize=6.5, framealpha=0.9)
    axes[0][0].set_ylabel("separability ratio (up = better)")
    fig.suptitle("Where separability peaks vs n_neighbors — per embedder, per dataset size N", fontsize=11)
    p = C.FIG_DIR / "fig_nn_curves.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def compaction_delta(df):
    """Per (dataset, embedder) mean separability delta compacted − raw."""
    piv = df.pivot_table(index=["dataset", "embedder"], columns="compact", values="ratio").dropna()
    delta = (piv[True] - piv[False]).sort_values()
    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.28 * len(delta))))
    colors = [HELP_C if v >= 0 else HURT_C for v in delta.values]
    ax.barh([f"{d} · {e}" for d, e in delta.index], delta.values, color=colors, height=0.7)
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel("Δ separability ratio (compacted − raw)   ←  compaction hurts | helps  →")
    ax.set_title("Compaction verdict: does closing the empty oceans cost separability?", fontsize=11)
    ax.grid(axis="y", visible=False)
    p = C.FIG_DIR / "fig_compaction_delta.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def stability(df):
    """Seed agreement vs n_neighbors, one line per embedder (averaged over datasets)."""
    raw = df[df["compact"] == False]  # noqa: E712
    fig, ax = plt.subplots(figsize=(6.2, 4))
    for i, emb in enumerate(_emb_present(raw)):
        sub = raw[raw["embedder"] == emb].groupby("n_neighbors")["seed_agreement"].mean()
        ax.plot(sub.index, sub.values, "-o", ms=4, lw=2, color=EMB_COLORS.get(emb, OKABE[i % len(OKABE)]), label=emb)
    ax.set_xscale("log")
    ax.set_xlabel("n_neighbors (log)")
    ax.set_ylabel("inter-seed neighbor agreement (up = more stable)")
    ax.set_title("Run-to-run stability falls as n_neighbors grows", fontsize=11)
    ax.legend(fontsize=8)
    p = C.FIG_DIR / "fig_stability.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def guard_scatter(df):
    """Separability vs neighbor-recall — a layout can't fake purity without losing recall."""
    raw = df[df["compact"] == False]  # noqa: E712
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for i, emb in enumerate(_emb_present(raw)):
        sub = raw[raw["embedder"] == emb]
        ax.scatter(
            sub["knn_recall"],
            sub["ratio"],
            s=14,
            color=EMB_COLORS.get(emb, OKABE[i % len(OKABE)]),
            label=emb,
            alpha=0.7,
            linewidths=0,
        )
    ax.set_xlabel("kNN-recall guard (high-D ↔ 2-D neighbor overlap)")
    ax.set_ylabel("separability ratio")
    ax.set_title("Separability tracks neighbor-recall — purity isn't gamed by shattering", fontsize=10)
    ax.legend(fontsize=8)
    p = C.FIG_DIR / "fig_guard_scatter.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def main():
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = _cell()
    heatmaps(df)
    nn_curves(df)
    compaction_delta(df)
    stability(df)
    guard_scatter(df)


if __name__ == "__main__":
    main()
