"""#5213: SAM3's zero-shot text prompt against SigLIP's typed sort and #4711's Region voting, on #4711's grid.

Reads ``text_baseline.py`` rows (SigLIP from the grid's own ``text_baseline.csv``; SAM3 from ``--sam3`` runs of the
same script, one CSV per prompt set) and #4711's committed per-run / per-vote tables. Every row is scored on the
cell's own pool and withheld half, so a SAM3 row and a SigLIP row are the same images cell for cell.

Per cell and seed, for each zero-shot sort:

- ``AP`` and ``oracle`` (the best F1 of any cut on the ranking): how good the ranking is;
- ``F_line``: F1 of the set the app's text-sort line would show on these scores (``text_line_*_b1``, the number
  #4711's analyzer puts at vote 0);
- ``F_native`` (SAM3 only): F1 of SAM3's own operating point, a detection over 0.5.

Region's numbers are #4711's seed-0 arms (``mp`` shipped, ``hac`` the tree) at votes 50 and 150, from the committed
tables; the per-vote curve gives the vote at which Region's mean passes each zero-shot mean. SEs are over classes
(a class's bands are not independent), as in #4711.

    python sam3_analyze_5213.py --siglip <grid>/text_baseline.csv --sam3 <root>/sam3_typed.csv <root>/sam3_noun.csv \\
        <root>/sam3_both.csv --region docs/experiments/2026-10-25-region-styles-4711/tables --out <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BANDS = ("all", "small", "medium", "large")


def f1(p, r):
    p, r = np.asarray(p, float), np.asarray(r, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(p + r > 0, 2 * p * r / (p + r), 0.0)


def zero_shot(path: Path, label: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    if label == "siglip":
        d = d[d["embedder"] == "siglip"]
    out = pd.DataFrame(
        {
            "category": d["category"],
            "seed": d["seed"],
            "arm": label,
            "query": d["query"],
            "AP": d["text_AP"],
            "oracle": d["text_oracle_fbeta_b1"],
            "F_line": f1(d["text_line_precision_b1"], d["text_line_recall_b1"]),
            "P_line": d["text_line_precision_b1"],
            "R_line": d["text_line_recall_b1"],
        }
    )
    if "text_native_f1" in d:
        out["F_native"] = d["text_native_f1"].values
        out["P_native"] = d["text_native_precision"].values
        out["R_native"] = d["text_native_recall"].values
        out["k_native"] = d["text_native_k"].values
    return out


def band_of(cat: str) -> str:
    return cat.rpartition("@")[2]


def class_of(cat: str) -> str:
    return cat.rpartition("@")[0]


def summarize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Mean over cells and its SE over classes (class means), per band group and arm."""
    rows = []
    for band in BANDS:
        sub = df if band == "all" else df[df["band"] == band]
        for arm, g in sub.groupby("arm", sort=False):
            row = {"group": band, "arm": arm, "cells": g["category"].nunique()}
            for c in cols:
                if c not in g or g[c].isna().all():
                    continue
                cm = g.groupby("class")[c].mean()
                row[c] = g[c].mean()
                row[f"{c}_se"] = cm.std(ddof=1) / np.sqrt(len(cm))
            rows.append(row)
    return pd.DataFrame(rows)


def paired(df: pd.DataFrame, a: str, b: str, col: str) -> dict:
    """Mean of a - b over shared cells, SE over classes, and the share of classes where a is ahead."""
    x = df[df["arm"] == a].set_index(["category", "seed"])[col]
    y = df[df["arm"] == b].set_index(["category", "seed"])[col]
    d = (x - y).dropna().rename("d").reset_index()
    d["class"] = d["category"].map(class_of)
    cm = d.groupby("class")["d"].mean()
    return {
        "a": a,
        "b": b,
        "metric": col,
        "gain": d["d"].mean(),
        "se": cm.std(ddof=1) / np.sqrt(len(cm)),
        "classes_ahead": float((cm > 0).mean()),
        "n_cells": len(d),
    }


def crossover(curves: pd.DataFrame, group: str, col: str, level: float) -> float:
    """First vote at which Region's mean ``col`` reaches *level* (NaN if never within 150)."""
    c = curves[curves["group"] == (group if group == "all" else f"band:{group}")].sort_values("t")
    hit = c[c[col] >= level]
    return float(hit["t"].iloc[0]) if len(hit) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--siglip", required=True)
    ap.add_argument("--sam3", nargs="+", required=True, help="text_baseline.py --sam3 CSVs (embedder sam3_<set>)")
    ap.add_argument("--region", required=True, help="#4711's tables dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    zs = [zero_shot(Path(a.siglip), "siglip")]
    for p in a.sam3:
        label = pd.read_csv(p, usecols=["embedder"])["embedder"].iloc[0]
        zs.append(zero_shot(Path(p), label))
    z = pd.concat(zs, ignore_index=True)

    per_run = pd.read_csv(Path(a.region) / "rs4711_per_run.csv")
    curves = pd.read_csv(Path(a.region) / "rs4711_curves.csv")
    reg = []
    for arm in ("mp", "hac"):
        for t in (150,):
            reg.append(
                pd.DataFrame(
                    {
                        "category": per_run["category"],
                        "seed": per_run["seed"],
                        "arm": f"region_{arm}@{t}",
                        "AP": per_run[f"AP{t}_{arm}"],
                        "oracle": per_run[f"oracle{t}_{arm}"],
                        "F_mean_1_150": per_run[f"meanF_1_150_{arm}"],
                    }
                )
            )
    r = pd.concat(reg, ignore_index=True)

    allr = pd.concat([z, r], ignore_index=True)
    allr["band"] = allr["category"].map(band_of)
    allr["class"] = allr["category"].map(class_of)
    allr.to_csv(out / "sam3_5213_per_cell.csv", index=False)

    cols = ["AP", "oracle", "F_line", "F_native", "P_native", "R_native", "k_native", "F_mean_1_150"]
    s_all = summarize(allr, cols)  # zero-shot over all three seeds
    s_all.to_csv(out / "sam3_5213_summary.csv", index=False)
    s0 = summarize(allr[allr["seed"] == 0], cols)  # seed 0, the seed #4711's Region arms ran
    s0.to_csv(out / "sam3_5213_summary_seed0.csv", index=False)

    sam3_arms = [x for x in z["arm"].unique() if x != "siglip"]
    pairs = []
    for arm in sam3_arms:
        for col in ("AP", "oracle", "F_line"):
            pairs.append(paired(allr, arm, "siglip", col))
        s0r = allr[allr["seed"] == 0]
        for reg_arm in ("region_mp@150", "region_hac@150"):
            for col in ("AP", "oracle"):
                pairs.append(paired(s0r, arm, reg_arm, col))
    pd.DataFrame(pairs).to_csv(out / "sam3_5213_paired.csv", index=False)

    # The vote at which Region (seed 0, per-vote means) first reaches each zero-shot arm's seed-0 mean.
    cross = []
    for band in BANDS:
        for arm in z["arm"].unique():
            row = s0[(s0["group"] == band) & (s0["arm"] == arm)]
            if row.empty:
                continue
            for metric, col in (("AP", "AP"), ("oracle", "oracle")):
                lvl = float(row[metric].iloc[0])
                cross.append(
                    {
                        "group": band,
                        "arm": arm,
                        "metric": metric,
                        "level": lvl,
                        "vote_mp": crossover(curves, band, f"{col}_mp", lvl),
                        "vote_hac": crossover(curves, band, f"{col}_hac", lvl),
                    }
                )
    pd.DataFrame(cross).to_csv(out / "sam3_5213_crossover.csv", index=False)

    # Per class (bands pooled, 3 seeds): where SAM3 wins and loses against SigLIP, per prompt set.
    pc = allr[allr["arm"].isin(z["arm"].unique())].pivot_table(index="class", columns="arm", values="AP")
    pc.to_csv(out / "sam3_5213_per_class_AP.csv")

    _figures(allr, s0, curves, sam3_arms, out)
    (out / "sam3_5213_counts.json").write_text(
        json.dumps(
            {
                "cells": int(z["category"].nunique()),
                "seeds": sorted(map(int, z["seed"].unique())),
                "arms": list(allr["arm"].unique()),
            }
        )
    )
    print(s0.round(3).to_string())
    print(pd.DataFrame(pairs).round(3).to_string())
    print(pd.DataFrame(cross).round(3).to_string())
    return 0


def _figures(allr, s0, curves, sam3_arms, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = ["siglip", *sam3_arms, "region_mp@150", "region_hac@150"]
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#17becf", "#d62728", "#ff7f0e"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, metric in zip(axes, ("AP", "oracle")):
        w = 0.8 / len(arms)
        for i, (arm, col) in enumerate(zip(arms, colors)):
            ys, es = [], []
            for band in BANDS:
                row = s0[(s0["group"] == band) & (s0["arm"] == arm)]
                ys.append(float(row[metric].iloc[0]) if len(row) else np.nan)
                es.append(2 * float(row[f"{metric}_se"].iloc[0]) if len(row) else 0)
            ax.bar(np.arange(len(BANDS)) + i * w, ys, w, yerr=es, color=col, label=arm, capsize=2)
        ax.set_xticks(np.arange(len(BANDS)) + 0.4 - w / 2, BANDS)
        ax.set_title({"AP": "AP on the withheld half", "oracle": "Oracle F1 (best cut of the ranking)"}[metric])
        ax.set_ylim(0, 1)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(arms), fontsize=8, frameon=False)
    fig.suptitle(
        "Seed 0, 144 cells: zero-shot sorts (vote 0) vs Region voting at vote 150, ±2 SE over classes", fontsize=9
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "sam3_5213_bands.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    c = curves[curves["group"] == "all"].sort_values("t")
    ax = axes[2]
    ax.plot(c["t"], c["F_mp"], color="#d62728", label="Region max_patch")
    ax.plot(c["t"], c["F_hac"], color="#ff7f0e", label="Region max_patch_hac")
    ax.plot(c["t"], c["F_bin"], color="#9467bd", label="SigLIP Binary")
    sig = s0[(s0["group"] == "all") & (s0["arm"] == "siglip")]
    ax.axhline(float(sig["F_line"].iloc[0]), color=colors[0], ls="--", lw=1.2, label="siglip typed set (app's line)")
    for arm, col in zip(sam3_arms, colors[1:]):
        row = s0[(s0["group"] == "all") & (s0["arm"] == arm)]
        ax.axhline(float(row["F_native"].iloc[0]), color=col, ls="--", lw=1.2, label=f"{arm} (score > 0.5)")
    ax.set_xlabel("votes")
    ax.set_title("F1 of the set shown")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper right")
    for ax, metric in zip(axes, ("AP", "oracle")):
        ax.plot(c["t"], c[f"{metric}_mp"], color="#d62728", label="Region max_patch")
        ax.plot(c["t"], c[f"{metric}_hac"], color="#ff7f0e", label="Region max_patch_hac")
        for arm, col in zip(["siglip", *sam3_arms], colors):
            row = s0[(s0["group"] == "all") & (s0["arm"] == arm)]
            if len(row):
                ax.axhline(float(row[metric].iloc[0]), color=col, ls="--", lw=1.2, label=f"{arm} (zero-shot)")
        ax.set_xlabel("votes")
        ax.set_title({"AP": "AP", "oracle": "Oracle F1"}[metric])
        ax.set_ylim(0, 1)
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("All 144 cells, seed 0: Region's ranking over the votes against each zero-shot sort", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "sam3_5213_over_votes.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
