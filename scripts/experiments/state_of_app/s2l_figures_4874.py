#!/usr/bin/env python
"""Figures and tables for #4874 (SigLIP 2 L against SigLIP), from ``eg2_compare_4837.py --arm siglip2_l``'s
outputs and ``s2l_timing_4874.py``'s ``timing.jsonl``.

* ``timing.csv`` / ``figures/runtime.png``: images a second per embedder (and per precision, where a half-precision
  run exists), GPU and 8 CPU threads, the median of the interleaved repeats, and each one's time as a multiple of
  SigLIP's fp32 time on the same node;
* ``figures/objective_by_seed.png`` (with ``--run``): the objective over clicks, one line per seed, per arm and preset;
* ``figures/by_band.png``: the paired change in the session mean and in the starved share (fewer than 3 positives
  by vote 150, #4216), per size band and preset, +-2 SE over classes.

    python s2l_figures_4874.py --report <report dir> --timing <timing.jsonl> [--run 1=<b1 dir> ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

import matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK2, SURFACE, GRIDC = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a")  # the reports' categorical order; the first three pass all-pairs
LABEL = {"siglip": "SigLIP", "siglip2_l": "SigLIP 2 L", "embeddinggemma2": "EmbeddingGemma 2 (280 tokens)"}
ORDER = {e: i for i, e in enumerate(LABEL)}
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    }
)


def runtime(timing: Path, out: Path) -> pd.DataFrame:
    # A library warning can land in the file between records (one did on 2026-10-14); keep the JSON lines only.
    rows = [json.loads(line) for line in timing.read_text().splitlines() if line.startswith("{")]
    t = pd.DataFrame([r for r in rows if r["what"] == "bulk"])
    t["where"] = np.where(t["device"].str.startswith("cuda"), t["gpu"].str.replace("Tesla ", ""), "CPU, 8 threads")
    # Half precision (VTSEARCH_EMBED_PRECISION) is its own arm: it changes the vectors, not only the speed.
    t["arm"] = t["embedder"] + np.where(t["precision"].eq("fp32"), "", "@" + t["precision"])
    agg = t.groupby(["where", "arm", "embedder", "precision"], as_index=False).agg(
        per_s=("per_s", "median"), reps=("rep", "nunique"), batch=("batch", "first"), n=("n", "first")
    )
    sig = agg[agg.arm == "siglip"].set_index("where").per_s
    agg["x_siglip_time"] = agg.apply(lambda r: sig.get(r["where"], np.nan) / r.per_s, axis=1)
    pile = pd.DataFrame([r for r in rows if r["what"] == "vs_pile"])
    if len(pile):
        pile["arm"] = pile["embedder"] + np.where(pile["precision"].eq("fp32"), "", "@" + pile["precision"])
        agg = agg.merge(pile.groupby("arm", as_index=False)[["median_1mcos", "max_1mcos"]].max(), on="arm", how="left")
    agg = agg.assign(o=agg.embedder.map(ORDER)).sort_values(["where", "o", "precision"], ascending=[True, True, False])
    agg = agg.drop(columns="o")
    agg.round({"per_s": 3, "x_siglip_time": 2}).to_csv(out / "timing.csv", index=False)
    wheres = list(agg["where"].unique())
    fig, axes = plt.subplots(1, len(wheres), figsize=(5.6 * len(wheres), 3.0), squeeze=False)
    for ax, w in zip(axes[0], wheres, strict=True):
        a = agg[agg["where"] == w]
        y = np.arange(len(a))[::-1]
        ax.barh(
            y,
            a.per_s,
            color=[SLOTS[ORDER[e]] for e in a.embedder],
            hatch=["//" if p != "fp32" else "" for p in a.precision],
            edgecolor=SURFACE,
            height=0.6,
        )
        for yi, r in zip(y, a.itertuples()):
            note = "" if r.arm == "siglip" else f"  ({r.x_siglip_time:.1f}\u00d7 SigLIP's time)"
            ax.text(r.per_s, yi, f"  {r.per_s:.3g}/s{note}", va="center", fontsize=8.5, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL[e] + ("" if p == "fp32" else f", {p}") for e, p in zip(a.embedder, a.precision)])
        ax.set_xlim(0, a.per_s.max() * 1.7)
        ax.set_xlabel("images a second, end to end (embed_media_bulk)")
        ax.set_title(w, loc="left", fontsize=10)
        ax.grid(axis="x", color=GRIDC, lw=0.8)
    fig.tight_layout()
    fig.savefig(out / "figures" / "runtime.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return agg


def by_band(report: Path) -> None:
    b = pd.read_csv(report / "by_band.csv")
    panels = [
        ("session_1_150", "change in the session mean, votes 1-150", 1.0),
        ("starved", "change in the starved share (points)", 100.0),
    ]
    bands = ["large", "medium", "small"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
    for ax, (col, xlabel, k) in zip(axes, panels, strict=True):
        if f"d_{col}" not in b:
            ax.set_visible(False)
            continue
        for yi, band in enumerate(bands):
            for bi, (beta, color) in enumerate(zip((0.25, 1.0, 4.0), SLOTS, strict=True)):
                r = b[(b.band == band) & (b.beta == beta)]
                if r.empty:
                    continue
                m, se = k * float(r[f"d_{col}"].iloc[0]), k * float(r[f"se_{col}"].iloc[0])
                yy = yi - (bi - 1) * 0.22
                ax.plot([m - 2 * se, m + 2 * se], [yy, yy], color=color, lw=1.4)
                lab = {0.25: "beta 1/4", 1.0: "beta 1", 4.0: "beta 4"}[beta]
                ax.plot([m], [yy], "o", color=color, ms=5.5, mec=SURFACE, mew=1.2, label=lab if yi == 0 else None)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.grid(axis="x", color=GRIDC, lw=0.8)
        ax.set_xlabel(xlabel)
    axes[0].set_yticks(range(len(bands)))
    axes[0].set_yticklabels(bands)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].set_title("SigLIP 2 L minus SigLIP, paired, ±2 SE", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(report / "figures" / "by_band.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def individuals(runs: list[str], report: Path) -> None:
    """Every seed as its own line: the objective over clicks, averaged over the 144 cells within a seed."""
    import eg2_compare_4837 as cmp  # the compare's own filled curves (the typed query's set until the app shows a detector)

    specs = sorted((float(b), Path(d)) for b, d in (r.split("=", 1) for r in runs))
    fig, axes = plt.subplots(len(specs), 2, figsize=(10, 2.9 * len(specs)), sharex=True, sharey="row", squeeze=False)
    for row, (beta, d) in enumerate(specs):
        for col, emb in enumerate(("siglip", "siglip2_l")):
            _, filled = cmp.arm_runs(d, emb, beta)
            clicks = [c for c in filled.columns if 0 <= int(c) <= cmp.HORIZON]
            seeds = (
                filled.index.get_level_values("seed") if "seed" in filled.index.names else filled.reset_index()["seed"]
            )
            ax = axes[row][col]
            for s in sorted(set(seeds)):
                m = filled[np.asarray(seeds) == s][clicks].mean()
                ax.plot([int(c) for c in clicks], m.to_numpy(), color=SLOTS[col], lw=1.2, alpha=0.85)
            ax.set_title(f"{LABEL[emb]}, beta {beta:g}: each line one seed (144 cells)", loc="left", fontsize=9)
            ax.grid(color=GRIDC, lw=0.8)
        axes[row][0].set_ylabel("objective")
    for ax in axes[-1]:
        ax.set_xlabel("click")
    fig.tight_layout()
    fig.savefig(report / "figures" / "objective_by_seed.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--timing", required=True, type=Path)
    ap.add_argument("--run", action="append", default=[], help="beta=run dir, for the per-seed figure")
    a = ap.parse_args()
    (a.report / "figures").mkdir(parents=True, exist_ok=True)
    print(runtime(a.timing, a.report).round(3).to_string(index=False))
    by_band(a.report)
    if a.run:
        individuals(a.run, a.report)
    print(f"-> {a.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
