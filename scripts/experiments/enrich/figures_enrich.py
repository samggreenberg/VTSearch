"""Figures for the description-enrichment study (#3127).

Every figure is drawn from the same CSVs the report's tables are, so a number in
a caption and a number in a table cannot drift apart.  Imported by
``analyze_enrich.py``; also runnable to redraw a finished study's figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MT_COLOURS = {"audio": "#1f77b4", "image": "#d95f02", "text": "#7570b3", "video": "#1b9e77"}
CONTROL_COLOUR = "#999999"
DEFAULTS = {"audio": "clap_general", "image": "siglip", "text": "e5", "video": "xclip"}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _row_colour(media_type: str, embedder: str) -> str:
    """Controls are grey: only the media-type *defaults* answer the question."""
    if DEFAULTS.get(media_type) != embedder:
        return CONTROL_COLOUR
    return MT_COLOURS.get(media_type, "#444444")


def forest(per_mt: pd.DataFrame, outdir: Path, dpi: int = 130) -> str:
    """Per media type: the paired mean Δ with its ±2 clustered-SE interval."""
    plt = _plt()
    d = per_mt.sort_values(["media_type", "embedder"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.4, 0.62 * len(d) + 1.9), layout="constrained")
    ys = np.arange(len(d))[::-1]
    for y, (_, r) in zip(ys, d.iterrows()):
        colour = _row_colour(r["media_type"], r["embedder"])
        ax.errorbar(
            r["delta"],
            y,
            xerr=2 * r["se_cluster"],
            fmt="o",
            color=colour,
            capsize=4,
            lw=1.8,
            markersize=6,
        )
    ax.axvline(0, color="#444", lw=1.0, ls="--")
    labels = [
        f"{r['media_type']} · {r['embedder']}"
        + ("" if DEFAULTS.get(r["media_type"]) == r["embedder"] else "  (control)")
        for _, r in d.iterrows()
    ]
    ax.set_yticks(ys)
    ax.set_yticklabels(labels[: len(ys)])
    ax.set_xlabel("Δ average precision  (enriched − plain, paired per category)")
    ax.set_title("Description enrichment, per media-type default\nbars are ±2 SE clustered by (corpus, category)")
    for y, (_, r) in zip(ys, d.iterrows()):
        ax.annotate(
            f"{r['delta']:+.3f}  ({r['n_pairs']} pairs)",
            (r["delta"], y),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8,
            color="#333",
        )
    ax.margins(x=0.22)
    name = "delta_by_media_type.png"
    fig.savefig(outdir / name, dpi=dpi)
    plt.close(fig)
    return name


def by_dataset(per_ds: pd.DataFrame, outdir: Path, dpi: int = 130) -> str:
    """The same difference, one row per dataset: does it hold across domains?"""
    plt = _plt()
    d = per_ds.sort_values(["media_type", "embedder", "dataset"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.8, 0.36 * len(d) + 2.0), layout="constrained")
    ys = np.arange(len(d))[::-1]
    for y, (_, r) in zip(ys, d.iterrows()):
        colour = _row_colour(r["media_type"], r["embedder"])
        ax.errorbar(r["delta"], y, xerr=2 * r["se_cluster"], fmt="o", color=colour, capsize=3, lw=1.4, markersize=5)
    ax.axvline(0, color="#444", lw=1.0, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels(
        [f"{r['dataset']}  ({r['embedder']}, n={int(r['n_media'])})" for _, r in d.iterrows()][: len(ys)],
        fontsize=8,
    )
    ax.set_xlabel("Δ average precision  (enriched − plain, paired per category)")
    ax.set_title(
        "Per dataset — one global setting, many haystacks\nbars are ±2 SE clustered by category; grey = control arm"
    )
    ax.margins(x=0.12)
    name = "delta_by_dataset.png"
    fig.savefig(outdir / name, dpi=dpi)
    plt.close(fig)
    return name


def wrappers(per_wrapper: pd.DataFrame, per_mt: pd.DataFrame, outdir: Path, dpi: int = 130) -> str:
    """Each wrapper on its own, against the ensemble that averages them.

    The question the docs note needs answered: is enrichment *ensembling*, or is
    it one good template carrying four passengers?
    """
    plt = _plt()
    embs = list(per_wrapper.sort_values(["media_type", "embedder"])["embedder"].unique())
    ncol = min(3, len(embs))
    nrow = int(np.ceil(len(embs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 3.1 * nrow), layout="constrained", squeeze=False)
    for ax, emb in zip(axes.flat, embs):
        sub = per_wrapper[per_wrapper["embedder"] == emb].sort_values("arm")
        mt = sub["media_type"].iloc[0]
        colour = _row_colour(mt, emb)
        xs = np.arange(len(sub))
        ax.bar(xs, sub["delta"], color=colour, alpha=0.85)
        ax.errorbar(xs, sub["delta"], yerr=2 * sub["se_cluster"], fmt="none", ecolor="#333", capsize=3, lw=1.0)
        ens = per_mt[(per_mt["embedder"] == emb)]["delta"]
        if len(ens):
            ax.axhline(float(ens.iloc[0]), color="#c0392b", ls="--", lw=1.4)
            ax.annotate(
                "enriched (mean of all 5)",
                (0.02, float(ens.iloc[0])),
                xycoords=("axes fraction", "data"),
                fontsize=7.5,
                color="#c0392b",
                va="bottom",
            )
        ax.axhline(0, color="#444", lw=0.9)
        ax.set_xticks(xs)
        ax.set_xticklabels([w.replace("{text}", "…") for w in sub["wrapper"]], rotation=30, ha="right", fontsize=7)
        ax.set_title(f"{emb}  ({mt})" + ("" if DEFAULTS.get(mt) == emb else "  — control"), fontsize=9)
        ax.set_ylabel("Δ AP vs plain")
    for ax in axes.flat[len(embs) :]:
        ax.axis("off")
    fig.suptitle("Every wrapper on its own, against the ensemble of all five", fontsize=11)
    name = "wrappers.png"
    fig.savefig(outdir / name, dpi=dpi)
    plt.close(fig)
    return name


def category_scatter(pair: pd.DataFrame, outdir: Path, dpi: int = 130) -> str:
    """Where the movement is: per-category AP, plain against enriched."""
    plt = _plt()
    d = pair[pair.apply(lambda r: DEFAULTS.get(r["media_type"]) == r["embedder"], axis=1)]
    mts = sorted(d["media_type"].unique())
    fig, axes = plt.subplots(1, len(mts), figsize=(3.2 * len(mts), 3.5), layout="constrained", squeeze=False)
    for ax, mt in zip(axes.flat, mts):
        sub = d[d["media_type"] == mt]
        ax.scatter(sub["base"], sub["arm"], s=13, alpha=0.6, color=MT_COLOURS.get(mt, "#444"))
        ax.plot([0, 1], [0, 1], color="#444", lw=1.0, ls="--")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_aspect("equal")
        ax.set_title(f"{mt} · {DEFAULTS.get(mt, '')}", fontsize=9)
        ax.set_xlabel("AP, plain query")
        above = int((sub["arm"] > sub["base"]).sum())
        ax.annotate(f"{above}/{len(sub)} above the line", (0.04, 0.94), xycoords="axes fraction", fontsize=8)
    axes.flat[0].set_ylabel("AP, enriched query")
    fig.suptitle("Per-category average precision: enrichment moves which categories?", fontsize=11)
    name = "category_scatter.png"
    fig.savefig(outdir / name, dpi=dpi)
    plt.close(fig)
    return name


def make_figures(pair, per_ds, per_mt, per_wrapper, outdir: Path, dpi: int = 130) -> list[str]:
    outdir.mkdir(parents=True, exist_ok=True)
    made = [forest(per_mt, outdir, dpi), by_dataset(per_ds, outdir, dpi), category_scatter(pair, outdir, dpi)]
    if per_wrapper is not None and len(per_wrapper):
        made.append(wrappers(per_wrapper, per_mt, outdir, dpi))
    return made


def main() -> int:
    ap = argparse.ArgumentParser(description="Redraw the study's figures from its committed tables.")
    ap.add_argument("--study", required=True, type=Path, help="study directory holding tables/")
    ap.add_argument("--dpi", type=int, default=130)
    args = ap.parse_args()
    t = args.study / "tables"
    made = make_figures(
        pd.read_csv(t / "paired_categories.csv"),
        pd.read_csv(t / "per_dataset.csv"),
        pd.read_csv(t / "per_media_type.csv"),
        pd.read_csv(t / "per_wrapper.csv") if (t / "per_wrapper.csv").exists() else None,
        args.study / "figures",
        args.dpi,
    )
    for name in made:
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
