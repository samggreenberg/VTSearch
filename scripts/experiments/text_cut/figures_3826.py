#!/usr/bin/env python
"""Figures for the #3826 report, from the same gate CSVs as the tables.

    python figures_3826.py --parts <analysis>/parts --corpus <corpus> --out docs/experiments/<study>/figures

``tradeoff.png``      stability against quality, one point per rule (two panels: data, optimiser)
``admitted.png``      per sort: admitted fraction against prevalence, one panel per rule
``boot_ecdf.png``     per sort: bootstrap flip, ECDF per headline rule
``by_dataset.png``    excess cost per dataset, headline rules
``worked_cases.png``  three literal sorts: score histograms (positives vs negatives) with every line drawn
``per_sort.png``      per sort: paired cost difference against the shipped line, sorted
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_3826 as A  # noqa: E402

# Categorical slots in fixed order (the dataviz reference palette, light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DATASETS = ["caltech101_m", "coco_val", "visual_genome_m", "vg_scale"]
DS_COLOR = dict(zip(DATASETS, SERIES))
#: Rules drawn in the per-rule figures, each with a fixed colour (entity, not rank).
SHOWN = {
    "gmm_shipped": SERIES[7],
    "gmm_converged": SERIES[6],
    "gmm_multistart": SERIES[4],
    "gmm_guarded_z3": SERIES[3],
    "quantile0.05": SERIES[1],
    "tail_z3": SERIES[5],
    "tail_fdr0.2": SERIES[0],
    "oracle_cost": INK2,
}
WORKED = [
    ("vg_scale", "siglip", "backpack@large"),
    ("coco_val", "siglip", "bear"),
    ("visual_genome_m", "siglip", "sky"),
]

plt.rcParams.update(
    {
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "figure.dpi": 130,
    }
)


def tradeoff(st: pd.DataFrame, ql: pd.DataFrame, out: Path) -> None:
    j = st.set_index("rule").join(ql.set_index("rule")[["excess_cost"]]).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, col, title in (
        (axes[0], "boot_mean_flip", "Data: bootstrap resample of the sort"),
        (axes[1], "opt_mean_flip", "Optimiser: random start or tighter tolerance"),
    ):
        for _, r in j.iterrows():
            x = r[col]
            if not np.isfinite(x):
                if col == "opt_mean_flip":
                    x = 1e-5  # closed form: no optimiser to perturb
                else:
                    continue
            x = max(x, 1e-5)
            shown = r["rule"] in SHOWN
            c = SHOWN.get(r["rule"], "#b5b4ae")
            ax.scatter(
                100 * x,
                r["excess_cost"],
                s=40 if shown else 12,
                color=c,
                edgecolor="white",
                linewidth=1,
                zorder=3 if shown else 2,
                label=r["rule"] if (shown and ax is axes[0]) else None,
            )
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _p: f"{v:g}"))
        ax.set_xlabel("mean % of the haystack whose verdict flips (log)")
        ax.set_title(title, fontsize=9, color=INK)
    axes[1].axvline(100 * 1e-5, color=GRID, lw=4, zorder=1)
    axes[1].text(
        100 * 1.2e-5, axes[1].get_ylim()[1] * 0.95, "no optimiser\n(closed form)", fontsize=7, color=INK2, va="top"
    )
    axes[0].set_ylabel("mean excess cost over the oracle line (FPR+FNR)")
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle(
        "Stability against quality, one point per rule (small grey: the other quantile / tail constants)", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(out / "tradeoff.png")
    plt.close(fig)


def admitted(q: pd.DataFrame, out: Path) -> None:
    rules = [r for r in SHOWN if r in set(q["rule"])]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.2), sharex=True, sharey=True)
    for ax, rule in zip(axes.flat, rules):
        g = q[q["rule"] == rule]
        for ds in DATASETS:
            h = g[g["dataset"] == ds]
            ax.scatter(
                h["prevalence"],
                h["adm_frac"].clip(lower=1e-4),
                s=6,
                color=DS_COLOR[ds],
                alpha=0.7,
                label=ds,
                linewidth=0,
            )
        ax.plot([1e-3, 1], [1e-3, 1], color=INK2, lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(rule, fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("prevalence (true positives / haystack)")
    for ax in axes[:, 0]:
        ax.set_ylabel("admitted fraction")
    axes[0, 0].legend(fontsize=7, markerscale=2, frameon=False)
    fig.suptitle(
        "Each dot is one typed-query sort. The dashed diagonal admits exactly as many as there are matches.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out / "admitted.png")
    plt.close(fig)


def boot_ecdf(d: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    boot = d[d["frame"].str.startswith("boot:")]
    for rule, c in SHOWN.items():
        g = boot[boot["rule"] == rule]
        if not len(g):
            continue
        v = np.sort(100 * g.groupby(A.SORT_KEYS)["flip"].mean().to_numpy().clip(min=1e-3))
        ax.step(v, np.arange(1, v.size + 1) / v.size, where="post", color=c, lw=2, label=rule)
    ax.set_xscale("log")
    ax.set_xlabel("% of the haystack whose verdict flips under a bootstrap resample (per sort, mean of 20)")
    ax.set_ylabel("fraction of sorts at or below")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "boot_ecdf.png")
    plt.close(fig)


def by_dataset(bd: pd.DataFrame, out: Path) -> None:
    rules = [r for r in SHOWN if r != "oracle_cost"]
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.8 / len(rules)
    for i, rule in enumerate(rules):
        g = bd[bd["rule"] == rule].set_index("dataset").reindex(DATASETS)
        ax.bar(np.arange(len(DATASETS)) + i * w, g["excess_cost"], width=w * 0.9, color=SHOWN[rule], label=rule)
    ax.set_xticks(np.arange(len(DATASETS)) + 0.4 - w / 2)
    ax.set_xticklabels(DATASETS)
    ax.set_ylabel("mean excess cost over the oracle line")
    ax.legend(fontsize=7, frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(out / "by_dataset.png")
    plt.close(fig)


def worked(q: pd.DataFrame, corpus: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, len(WORKED), figsize=(13, 6.2))
    for ax, (ds, emb, cat) in zip(axes, WORKED):
        z = np.load(corpus / f"{ds}__{emb}.npz")
        key = f"{ds}|{emb}|{cat}"
        x, y = z[f"{key}|scores"], z[f"{key}|labels"].astype(bool)
        bins = np.linspace(x.min(), x.max(), 70)
        ax.hist(x[~y], bins=bins, color="#b5b4ae", label=f"negatives ({(~y).sum()})")
        ax.hist(x[y], bins=bins, color=SERIES[2], alpha=0.85, label=f"positives ({y.sum()})")
        ax.set_yscale("log")
        g = q[(q["dataset"] == ds) & (q["embedder"] == emb) & (q["category"] == cat)].set_index("rule")
        for rule, c in SHOWN.items():
            if rule in g.index and rule not in ("tail_z3", "gmm_converged"):
                r = g.loc[rule]
                ax.axvline(
                    float(r["cut"]),
                    color=c,
                    lw=1.6,
                    ls="--" if rule == "oracle_cost" else "-",
                    label=f"{rule}: admits {int(r['n_adm'])}, {int(r['tp'])} true",
                )
        meta = json.loads(bytes(z["_meta"].tobytes()).decode("utf-8"))
        ax.set_title(f'{ds} / {emb} / "{meta["queries"][key]["query"]}"', fontsize=8)
        ax.set_xlabel("cosine score")
        ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
    axes[0].set_ylabel("medias (log)")
    fig.tight_layout()
    fig.savefig(out / "worked_cases.png")
    plt.close(fig)


def per_sort(q: pd.DataFrame, out: Path) -> None:
    base = q[q["rule"] == "gmm_shipped"].set_index(A.SORT_KEYS)["cost"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for rule in ("gmm_converged", "quantile0.05", "tail_fdr0.2"):
        g = q[q["rule"] == rule].set_index(A.SORT_KEYS)["cost"]
        dlt = np.sort((g - base.reindex(g.index)).dropna().to_numpy())
        ax.plot(np.linspace(0, 1, dlt.size), dlt, color=SHOWN[rule], lw=2, label=rule)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("sorts, ordered by the difference")
    ax.set_ylabel("cost(rule) - cost(gmm_shipped), per sort")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "per_sort.png")
    plt.close(fig)


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parts", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(list(argv) if argv is not None else None)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    new, _old = A.load(Path(args.parts))
    d = A.with_flips(new)
    st = A.stability(d[d["family"] != "oracle"])
    q = A.quality_frame(d)
    ql = A.quality(q)
    tradeoff(st, ql, out)
    admitted(q, out)
    boot_ecdf(d, out)
    by_dataset(A.by_dataset(q, d), out)
    worked(q, Path(args.corpus), out)
    per_sort(q, out)
    print(f"wrote figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
