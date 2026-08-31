#!/usr/bin/env python
"""Figures for the #3329 inventory's parts B and C.

Four, each carrying one of the run's claims:

1. **The atlas PIT** - the ECDF of in-domain typicality p-values against the
   uniform diagonal that ``domain_shift_report``'s docstring asserts. A PIT plot
   is the right picture for a stated-null claim: uniform means the curve lies on
   the diagonal, and the SHAPE of the departure says which way it is wrong.
2. **Dispersion against path length** - the mechanism, if there is one.
3. **The domain-shift matrix** - build dataset x query dataset, per embedder.
   The null lives on the diagonal and the power lives off it; one heatmap shows
   both, which is the whole reason B4 and B5 were pre-registered as a pair.
4. **The projection panel** - trustworthiness and continuity against
   neighbourhood size, the Shepard diagram, k-NN class purity before vs after,
   and the compaction radius's realised containment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

DPI = 130
EMB_COLOURS = {
    "siglip": "#2b6cb0",
    "dinov3_patch": "#c0392b",
    "clip": "#38a169",
    "clip_l": "#8e44ad",
    "siglip2_l": "#d68910",
}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(x, dtype=np.float64))
    return x, np.arange(1, x.size + 1, dtype=np.float64) / max(x.size, 1)


def atlas_pit(results: Path, outdir: Path) -> list[str]:
    """The PIT plot, shipped p-values against the deepest-node-only variant."""
    plt = _plt()
    files = sorted(Path(results).glob("struct_*.npz"))
    if not files:
        return []
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.1), layout="constrained")
    titles = (
        "shipped `typicality_pvalues`\n(mean over the root-to-leaf path)",
        "deepest calibrated node only\n(no path averaging)",
        "build points, shipped\n(in-sample, LOO-corrected)",
    )
    keys = ("p_atlas_holdout", "p_atlas_deepest", "p_atlas_build")
    for ax, key, title in zip(axes, keys, titles, strict=True):
        for f in files:
            z = np.load(f)
            if key not in z.files:
                continue
            emb = f.stem.replace("struct_", "").split("__")[-1]
            xs, ys = _ecdf(z[key])
            ax.plot(xs, ys, lw=1.0, alpha=0.75, color=EMB_COLOURS.get(emb, "#666"))
        ax.plot([0, 1], [0, 1], color="#111", ls="--", lw=1.2, zorder=5)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("typicality p-value")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("empirical CDF")
    handles = [plt.Line2D([], [], color=c, lw=1.6, label=e) for e, c in EMB_COLOURS.items()]
    handles.append(plt.Line2D([], [], color="#111", ls="--", lw=1.2, label="uniform (the stated null)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=6, fontsize=8, frameon=False)
    outdir.mkdir(parents=True, exist_ok=True)
    name = "atlas_pit_uniformity.png"
    fig.savefig(outdir / name, dpi=DPI)
    plt.close(fig)
    return [name]


def dispersion_vs_pathlen(uni: pd.DataFrame, outdir: Path) -> list[str]:
    plt = _plt()
    if uni.empty:
        return []
    fig, ax = plt.subplots(figsize=(6.2, 4.4), layout="constrained")
    for emb, g in uni.groupby("embedder"):
        ax.scatter(g["path_len"], g["sd_shipped"], s=42, color=EMB_COLOURS.get(emb, "#666"), label=emb, zorder=3)
    ax.axhline(uni["sd_uniform"].iloc[0], color="#111", ls="--", lw=1.2)
    ax.annotate(
        "sd of U(0,1) = 0.289",
        xy=(0.99, uni["sd_uniform"].iloc[0]),
        xycoords=("axes fraction", "data"),
        ha="right",
        va="bottom",
        fontsize=8,
    )
    ax.set_xlabel("mean number of calibrated nodes averaged per item")
    ax.set_ylabel("sd of the in-domain p-values")
    ax.set_title("Under-dispersion tracks the averaging, not the data", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, frameon=False)
    outdir.mkdir(parents=True, exist_ok=True)
    name = "atlas_dispersion_vs_pathlen.png"
    fig.savefig(outdir / name, dpi=DPI)
    plt.close(fig)
    return [name]


def domain_shift_matrix(shift: pd.DataFrame, outdir: Path) -> list[str]:
    """build x query z-scores. The null is the diagonal; the power is off it."""
    plt = _plt()
    if shift.empty:
        return []
    embs = sorted(shift["embedder"].unique())
    fig, axes = plt.subplots(1, len(embs), figsize=(3.15 * len(embs), 3.6), layout="constrained")
    axes = np.atleast_1d(axes)
    order: Sequence[str] = sorted(shift["build_dataset"].unique())
    for ax, emb in zip(axes, embs, strict=True):
        g = shift[shift["embedder"] == emb]
        M = np.full((len(order), len(order)), np.nan)
        for _, r in g.iterrows():
            M[order.index(r["build_dataset"]), order.index(r["query_dataset"])] = r["z_score"]
        im = ax.imshow(M, cmap="RdBu_r", vmin=-30, vmax=30)
        for i in range(len(order)):
            for j in range(len(order)):
                if np.isfinite(M[i, j]):
                    fired = bool(
                        g[(g["build_dataset"] == order[i]) & (g["query_dataset"] == order[j])]["shifted"].any()
                    )
                    ax.text(
                        j,
                        i,
                        f"{M[i, j]:.0f}" + ("*" if fired else ""),
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color="#111" if abs(M[i, j]) < 18 else "#fff",
                    )
        ax.set_xticks(range(len(order)))
        ax.set_yticks(range(len(order)))
        ax.set_xticklabels([o.replace("_", "\n") for o in order], fontsize=6, rotation=90)
        ax.set_yticklabels([o.replace("_", "\n") for o in order] if ax is axes[0] else [], fontsize=6)
        ax.set_title(emb, fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("atlas built on")
    fig.colorbar(im, ax=axes, shrink=0.8, label="domain-shift z  (* = `shifted` fired)")
    fig.supxlabel("queried with", fontsize=9)
    outdir.mkdir(parents=True, exist_ok=True)
    name = "atlas_domain_shift_matrix.png"
    fig.savefig(outdir / name, dpi=DPI)
    plt.close(fig)
    return [name]


def projection_panel(df: pd.DataFrame, proj: pd.DataFrame, results: Path, outdir: Path) -> list[str]:
    plt = _plt()
    if proj.empty:
        return []
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0), layout="constrained")

    ax = axes[0][0]
    ks = sorted(
        int(s.split("_k")[-1])
        for s in df.loc[df["statistic"].str.startswith("trustworthiness_k"), "statistic"].unique()
    )
    for emb, g in df[df["family"] == "umap"].groupby("embedder"):
        t = [g[g["statistic"] == f"trustworthiness_k{k}"]["value"].median() for k in ks]
        c = [g[g["statistic"] == f"continuity_k{k}"]["value"].median() for k in ks]
        ax.plot(ks, t, "-o", ms=3, lw=1.5, color=EMB_COLOURS.get(emb, "#666"), label=emb)
        ax.plot(ks, c, "--", lw=1.1, color=EMB_COLOURS.get(emb, "#666"), alpha=0.7)
    ax.set_xlabel("neighbourhood size k")
    ax.set_ylabel("trustworthiness (solid) / continuity (dashed)")
    ax.set_title("Local structure is kept; it decays with k", fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[0][1]
    f = sorted(Path(results).glob("struct_*.npz"))
    drawn = False
    for p in f[:6]:
        z = np.load(p)
        if "shepard_hi" not in z.files:
            continue
        emb = p.stem.replace("struct_", "").split("__")[-1]
        ax.scatter(z["shepard_hi"], z["shepard_lo"], s=1.5, alpha=0.10, color=EMB_COLOURS.get(emb, "#666"))
        drawn = True
    if drawn:
        ax.set_xlabel("cosine distance in the embedding")
        ax.set_ylabel("euclidean distance in the layout")
        med = proj["shepard"].median()
        ax.set_title(f"Shepard diagram — global distance is NOT preserved\nmedian Spearman {med:.2f}", fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)

    ax = axes[1][0]
    # Plotted as the DROP, not as layout-vs-embedding: purity runs 0.28-1.0 and
    # the effect is ~0.02, so on a shared 0-1 scatter every point sits on the
    # diagonal and the panel says nothing. The quantity of interest is the gap.
    d = proj.dropna(subset=["purity_embedding", "purity_layout"])
    if not d.empty:
        for emb, g in d.groupby("embedder"):
            ax.scatter(g["purity_embedding"], g["purity_drop"], s=42, color=EMB_COLOURS.get(emb, "#666"), label=emb)
        ax.axhline(0.0, color="#111", ls="--", lw=1.1)
        ax.set_xlabel("k-NN class purity in the embedding")
        ax.set_ylabel("purity lost by projecting (embedding − layout)")
        ax.set_title(f"The projection costs ~{d['purity_drop'].median():.3f} of class purity", fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=7, frameon=False)

    ax = axes[1][1]
    vals = proj["containment_core"].dropna().to_numpy()
    if vals.size:
        ax.hist(vals, bins=14, color="#2b6cb0", alpha=0.85)
        ax.axvline(0.90, color="#c0392b", ls="--", lw=1.4)
        ax.annotate("nominal 0.90", xy=(0.90, ax.get_ylim()[1] * 0.92), fontsize=8, color="#c0392b", ha="right")
        ax.set_xlabel("realised containment of the fitted 90th-percentile radius")
        ax.set_ylabel("cells")
        ax.set_title("Does the compaction radius contain what it claims?", fontsize=9)
    outdir.mkdir(parents=True, exist_ok=True)
    name = "projection_quality.png"
    fig.savefig(outdir / name, dpi=DPI)
    plt.close(fig)
    return [name]


COMBINER_COLOURS = {
    "mean": "#c0392b",
    "median": "#2b6cb0",
    "deepest": "#38a169",
    "fisher": "#8e44ad",
    "min": "#d68910",
}


def combiner_panel(results: Path, comb: pd.DataFrame, outdir: Path) -> list[str]:
    """What would a calibrated aggregation look like? Priced on the same paths.

    Left: the pooled PIT of each candidate combiner. Right: the two numbers a
    reader has to see together - distance from uniform over the WHOLE
    distribution, and the flag rate at the single alpha the guard actually
    reads. The shipped `mean` wins the second and loses the first, which is
    exactly how a miscalibrated guard passes the one check anyone runs.
    """
    plt = _plt()
    files = sorted(Path(results).glob("struct_*.npz"))
    if not files or comb.empty:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), layout="constrained")

    ax = axes[0]
    pooled: dict[str, list[np.ndarray]] = {}
    for f in files:
        z = np.load(f)
        for name in COMBINER_COLOURS:
            key = f"p_agg_{name}"
            if key in z.files:
                pooled.setdefault(name, []).append(z[key])
    for name, chunks in pooled.items():
        xs, ys = _ecdf(np.concatenate(chunks))
        ax.plot(xs, ys, lw=1.6, color=COMBINER_COLOURS[name], label=name + (" (shipped)" if name == "mean" else ""))
    ax.plot([0, 1], [0, 1], color="#111", ls="--", lw=1.2)
    ax.set_xlabel("typicality p-value")
    ax.set_ylabel("empirical CDF")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Every candidate combiner, pooled over all 75 cells", fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    d = comb.sort_values("ks_uniform_median")
    y = np.arange(len(d))
    ax.barh(y - 0.2, d["ks_uniform_median"], height=0.38, color="#2b6cb0", label="KS distance from uniform")
    ax.barh(y + 0.2, d["frac_below_05_median"], height=0.38, color="#d68910", label="flag rate at alpha = 0.05")
    ax.axvline(0.05, color="#111", ls="--", lw=1.1)
    ax.annotate("nominal 0.05", xy=(0.055, -0.42), fontsize=7.5, ha="left", va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([c + ("\n(shipped)" if c == "mean" else "") for c in d["combiner"]], fontsize=8)
    ax.set_title(
        "Shape vs the operating point — the shipped\ncombiner is best at one and worst at the other", fontsize=9
    )
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.5, axis="x")

    outdir.mkdir(parents=True, exist_ok=True)
    name = "atlas_combiner_comparison.png"
    fig.savefig(outdir / name, dpi=DPI)
    plt.close(fig)
    return [name]


def all_figures(
    df: pd.DataFrame,
    shift: pd.DataFrame,
    uni: pd.DataFrame,
    proj: pd.DataFrame,
    outdir: Path,
    results: Path,
    comb: pd.DataFrame | None = None,
) -> list[str]:
    written: list[str] = []
    written += atlas_pit(Path(results), outdir)
    if comb is not None:
        written += combiner_panel(Path(results), comb, outdir)
    written += dispersion_vs_pathlen(uni, outdir)
    written += domain_shift_matrix(shift, outdir)
    written += projection_panel(df, proj, Path(results), outdir)
    return written
