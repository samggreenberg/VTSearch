"""#5395: GroundingDINO's zero-shot text prompt against SAM3 (#5213), SigLIP's typed sort and #4711's Region voting.

Every input is a ``text_baseline.py`` CSV scored on the cell's own pool and withheld half, so all arms are the same
images cell for cell: SigLIP from the grid's ``text_baseline.csv``, SAM3 from #5213's ``sam3_{typed,noun}.csv``,
GroundingDINO from ``gdino_5395.sh baseline`` (embedder ``gdino_<model>_<set>``). The measures and the SEs (over
classes) are #5213's; ``F_native`` is each detector's own cut (SAM3 0.5, GroundingDINO 0.35, its box threshold).

With ``--overlap DIR`` it also grades the images Visual Genome holds apart from the rest (#5240): GoldG, in every
GroundingDINO checkpoint's training, carries GQA's VG images with grounded phrases, as SA-Co/EXT does for SAM3.
DIR holds ``{arm}_{vg,novg}.csv`` for every arm, SigLIP included (the control).

    python gdino_analyze_5395.py --siglip <grid>/text_baseline.csv --zs <sam3>/sam3_*.csv <root>/gdino_*_*.csv \\
        --region docs/experiments/2026-10-25-region-styles-4711/tables --out <dir> [--overlap DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sam3_analyze_5213 import BANDS, band_of, class_of, crossover, paired, summarize, zero_shot

COLS = ["AP", "oracle", "F_line", "F_native", "P_native", "R_native", "k_native", "F_mean_1_150"]


def prompt_set(arm: str) -> str:
    return arm.rpartition("_")[2]


def region_arms(tables: Path) -> pd.DataFrame:
    per_run = pd.read_csv(tables / "rs4711_per_run.csv")
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "category": per_run["category"],
                    "seed": per_run["seed"],
                    "arm": f"region_{arm}@150",
                    "AP": per_run[f"AP150_{arm}"],
                    "oracle": per_run[f"oracle150_{arm}"],
                    "F_mean_1_150": per_run[f"meanF_1_150_{arm}"],
                }
            )
            for arm in ("mp", "hac")
        ],
        ignore_index=True,
    )


def overlap(root: Path, arms: list[str]) -> pd.DataFrame:
    """Seen (VG) minus unseen AP per arm, and its difference from SigLIP's (the DiD), SE over classes."""
    rows = []
    halves = {}
    for arm in ["siglip", *arms]:
        for half in ("vg", "novg"):
            p = root / f"{arm}_{half}.csv"
            if p.exists():
                halves[arm, half] = zero_shot(p, "siglip" if arm == "siglip" else arm).set_index(["category", "seed"])
    if ("siglip", "vg") not in halves:
        return pd.DataFrame()
    base = halves["siglip", "vg"]["AP"] - halves["siglip", "novg"]["AP"]
    for arm in arms:
        if (arm, "vg") not in halves or (arm, "novg") not in halves:
            continue
        gap = halves[arm, "vg"]["AP"] - halves[arm, "novg"]["AP"]
        did = (gap - base).dropna().rename("d").reset_index()
        did["class"] = did["category"].map(class_of)
        cm = did.groupby("class")["d"].mean()
        rows.append(
            {
                "arm": arm,
                "AP_vg": halves[arm, "vg"]["AP"].mean(),
                "AP_novg": halves[arm, "novg"]["AP"].mean(),
                "lead_over_siglip_novg": (halves[arm, "novg"]["AP"] - halves["siglip", "novg"]["AP"]).mean(),
                "did": did["d"].mean(),
                "did_se": cm.std(ddof=1) / np.sqrt(len(cm)),
                "classes_did_pos": float((cm > 0).mean()),
                "n_cells": len(did),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--siglip", required=True)
    ap.add_argument("--zs", nargs="+", required=True, help="text_baseline.py --sam3 CSVs (SAM3 and GroundingDINO)")
    ap.add_argument("--region", required=True, help="#4711's tables dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--overlap", default=None, help="dir of {arm}_{vg,novg}.csv (#5240's split)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    zs = [zero_shot(Path(a.siglip), "siglip")]
    for p in a.zs:
        zs.append(zero_shot(Path(p), pd.read_csv(p, usecols=["embedder"])["embedder"].iloc[0]))
    z = pd.concat(zs, ignore_index=True)
    allr = pd.concat([z, region_arms(Path(a.region))], ignore_index=True)
    allr["band"] = allr["category"].map(band_of)
    allr["class"] = allr["category"].map(class_of)
    allr.to_csv(out / "gdino_5395_per_cell.csv", index=False)

    summarize(allr, COLS).to_csv(out / "gdino_5395_summary.csv", index=False)
    s0 = summarize(allr[allr["seed"] == 0], COLS)
    s0.to_csv(out / "gdino_5395_summary_seed0.csv", index=False)

    detectors = [x for x in z["arm"].unique() if x != "siglip"]
    gdino = [x for x in detectors if x.startswith("gdino_")]
    pairs = []
    s0r = allr[allr["seed"] == 0]
    for arm in gdino:
        for col in ("AP", "oracle", "F_line", "F_native"):
            pairs.append(paired(allr, arm, "siglip", col))
            sam3 = f"sam3_{prompt_set(arm)}"
            if sam3 in detectors:
                pairs.append(paired(allr, arm, sam3, col))
        for reg in ("region_mp@150", "region_hac@150"):
            for col in ("AP", "oracle"):
                pairs.append(paired(s0r, arm, reg, col))
    pd.DataFrame(pairs).to_csv(out / "gdino_5395_paired.csv", index=False)

    curves = pd.read_csv(Path(a.region) / "rs4711_curves.csv")
    cross = []
    for band in BANDS:
        for arm in z["arm"].unique():
            row = s0[(s0["group"] == band) & (s0["arm"] == arm)]
            if row.empty:
                continue
            lvl = float(row["AP"].iloc[0])
            cross.append(
                {
                    "group": band,
                    "arm": arm,
                    "level": lvl,
                    "vote_mp": crossover(curves, band, "AP_mp", lvl),
                    "vote_hac": crossover(curves, band, "AP_hac", lvl),
                }
            )
    pd.DataFrame(cross).to_csv(out / "gdino_5395_crossover.csv", index=False)

    pc = allr[allr["arm"].isin(z["arm"].unique())].pivot_table(index="class", columns="arm", values="AP")
    pc.to_csv(out / "gdino_5395_per_class_AP.csv")

    ov = overlap(Path(a.overlap), detectors) if a.overlap else pd.DataFrame()
    if len(ov):
        ov.to_csv(out / "gdino_5395_overlap.csv", index=False)

    _figure(s0, ["siglip", *detectors, "region_mp@150", "region_hac@150"], out)
    _scatter(pc, out)
    (out / "gdino_5395_counts.json").write_text(
        json.dumps(
            {
                "cells": int(z["category"].nunique()),
                "seeds": sorted(map(int, z["seed"].unique())),
                "arms": list(allr["arm"].unique()),
            }
        )
    )
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(
            s0[[c for c in ("group", "arm", "cells", "AP", "AP_se", "oracle", "F_line", "F_native") if c in s0]]
            .round(3)
            .to_string()
        )
        print(pd.DataFrame(pairs).round(3).to_string())
        print(pd.DataFrame(cross).round(1).to_string())
        if len(ov):
            print(ov.round(3).to_string())
    return 0


def _figure(s0, arms, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = _colors(arms)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, metric in zip(axes, ("AP", "F_native")):
        w = 0.85 / len(arms)
        for i, arm in enumerate(arms):
            if not s0.loc[s0["arm"] == arm, metric].notna().any():
                continue  # no native cut (SigLIP, Region): its slot stays empty, so the panels line up
            ys, es = [], []
            for band in BANDS:
                row = s0[(s0["group"] == band) & (s0["arm"] == arm)]
                ys.append(float(row[metric].iloc[0]) if len(row) else np.nan)
                es.append(2 * float(row[f"{metric}_se"].iloc[0]) if len(row) else 0)
            ax.bar(np.arange(len(BANDS)) + i * w, ys, w, yerr=es, color=colors[arm], label=arm, capsize=2)
        ax.set_xticks(np.arange(len(BANDS)) + 0.425 - w / 2, BANDS)
        ax.set_title({"AP": "AP on the withheld half", "F_native": "F1 at the detector's own cut"}[metric])
        ax.set_ylim(0, 1)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=min(len(lab), 6), fontsize=8, frameon=False)
    fig.suptitle(
        "Seed 0, 144 cells: zero-shot detectors (vote 0) vs Region voting at vote 150, ±2 SE over classes", fontsize=9
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out / "gdino_5395_bands.png", dpi=130)
    plt.close(fig)


def _colors(arms) -> dict:
    """One hue per model, dark for the noun prompt and light for the typed one (tab20's pairs)."""
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")
    hue = {"siglip": 14, "sam3": 0, "gdino_tiny": 2, "gdino_mmt": 4, "gdino_mml": 6, "region_mp": 10, "region_hac": 11}
    out = {}
    for arm in arms:
        base = arm.split("@")[0]
        if base.endswith(("_typed", "_noun")):
            stem, _, kind = base.rpartition("_")
            out[arm] = cmap(hue[stem] + (kind == "typed"))
        else:
            out[arm] = cmap(hue[base])
    return out


def _scatter(pc, out: Path) -> None:
    """Per class: each GroundingDINO checkpoint's noun-prompt AP against SAM3's."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [a for a in ("gdino_tiny_noun", "gdino_mmt_noun", "gdino_mml_noun") if a in pc]
    if "sam3_noun" not in pc or not arms:
        return
    fig, axes = plt.subplots(1, len(arms), figsize=(4.2 * len(arms), 4.2), sharey=True)
    for ax, arm in zip(np.atleast_1d(axes), arms):
        x, y = pc["sam3_noun"], pc[arm]
        ax.plot([0, 1], [0, 1], color="0.7", lw=1)
        ax.scatter(x, y, s=14, color=_colors([arm])[arm])
        d = (y - x).sort_values()
        for c in [*d.index[:3], *d.index[-2:]]:
            ax.annotate(c, (x[c], y[c]), fontsize=7, xytext=(3, -3), textcoords="offset points")
        ax.set_title(f"{arm}: {d.mean():+.2f} mean, ahead in {(d > 0).sum()} of {len(d)}", fontsize=9)
        ax.set_xlabel("SAM3 noun AP")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    np.atleast_1d(axes)[0].set_ylabel("GroundingDINO noun AP")
    fig.suptitle("Per class, mean over the class's cells and 3 seeds", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "gdino_5395_per_class.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
