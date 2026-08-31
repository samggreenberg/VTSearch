#!/usr/bin/env python
"""Analysis for the #3329 inventory's parts B and C.

Scores the eight pre-registered claims in
``docs/experiments/2026-08-30-fit-quality-3329/PREREG-part2.md`` and writes one
CSV per family under ``agg/`` plus ``summary_structure3329.json``.

Every bar is a module-level constant, so the selftest plants an answer against
the same number the report quotes.

    python analyze_structure_3329.py --results <DIR> --out <DIR>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# --- Pre-registered bars (PREREG-part2.md, do not retune after the run) -------

#: B1: in-domain typicality p-values are not uniform.
B1_KS_MIN = 0.05
#: B2: and they are under-dispersed. U(0,1) has sd = 1/sqrt(12) = 0.2887.
UNIFORM_SD = 1.0 / np.sqrt(12.0)
B2_SD_MAX = 0.27
#: B2's second half and B3: the averaging, not the data, is the cause.
B2_MIN_EMBEDDERS = 4
B3_RHO_MIN = 0.5
#: B4: the guard is conservative on its own data.
B4_MAX_SELF_FIRES = 0
#: B5: but still separates real domains.
B5_MIN_OFF_DIAGONAL_SHARE = 0.5
#: C1: local structure kept, global structure not.
C1_TRUST_MIN = 0.95
C1_SHEPARD_MAX = 0.6
#: C2: the projection costs class purity, on at least this many embedders.
C2_MIN_EMBEDDERS = 4
#: C3: a fitted 90th percentile should contain about 90%.
C3_LO, C3_HI = 0.85, 0.95

#: Below this many in-class items the conformal scope is unresolved, not null.
MIN_CONFORMAL_N = 40
#: Below this path length there is no averaging for B2/B3 to blame.
MIN_PATH_LEN = 3.0


def load_cells(results: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Long-format statistic rows from every cell, plus what was dropped."""
    files = sorted(Path(results).glob("struct_*.csv"))
    frames, dropped = [], {"missing": 0, "empty": 0, "unreadable": 0}
    expected = 0
    cmap = Path(results) / "cell_map.csv"
    if cmap.exists():
        expected = len(pd.read_csv(cmap))
    for p in files:
        try:
            if p.stat().st_size == 0:
                dropped["empty"] += 1
                continue
            f = pd.read_csv(p)
        except Exception:
            dropped["unreadable"] += 1
            continue
        if f.empty:
            dropped["empty"] += 1
            continue
        frames.append(f)
    dropped["missing"] = max(0, expected - len(files))
    if not frames:
        return pd.DataFrame(), dropped
    return pd.concat(frames, ignore_index=True), dropped


def load_shift(results: Path) -> pd.DataFrame:
    files = sorted(Path(results).glob("domainshift_*.csv"))
    frames = [pd.read_csv(p) for p in files if p.stat().st_size > 0]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


#: C4: a region whose within-similarity does not exceed its between-similarity
#: is a sign the canvas draws over nothing.
C4_MIN_GAP = 0.0
#: And one whose dominant ground-truth category holds less than half its members
#: is a region no single name describes.
C4_PURITY_FLOOR = 0.5


def load_regions(results: Path) -> pd.DataFrame:
    """Per-cluster coherence rows from ``region_coherence_3329.py``."""
    frames = [pd.read_csv(p) for p in sorted(Path(results).glob("regions_*.csv")) if p.stat().st_size > 0]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def c4_region_coherence(regions: pd.DataFrame) -> pd.DataFrame:
    """Are the browse canvas's regions coherent, and by how much, per zoom layer?

    Reported per LAYER because the canvas shows different layers at different
    zooms: a coarse region is allowed to be broader than a fine one, and a
    single pooled number would hide whether the degradation is orderly.
    """
    if regions.empty:
        return pd.DataFrame()
    rows = []
    for layer, g in regions.groupby("layer", sort=True):
        gap = g["coherence_gap"].dropna()
        pur = g["gt_purity"].dropna()
        rows.append(
            {
                "layer": int(layer),
                "n_clusters": int(len(g)),
                "size_median": float(g["size"].median()),
                "within_cosine_median": float(g["within_cosine"].median()),
                "between_cosine_median": float(g["between_cosine"].median()),
                "coherence_gap_median": float(gap.median()) if len(gap) else float("nan"),
                "share_gap_at_or_below_zero": float((gap <= C4_MIN_GAP).mean()) if len(gap) else float("nan"),
                "gt_purity_median": float(pur.median()) if len(pur) else float("nan"),
                "share_purity_below_floor": float((pur < C4_PURITY_FLOOR).mean()) if len(pur) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _pick(df: pd.DataFrame, family: str, scope: str, statistic: str) -> pd.DataFrame:
    m = (df["family"] == family) & (df["scope"] == scope) & (df["statistic"] == statistic)
    return df[m]


#: A cell is a (dataset, embedder, seed) triple. The seed moves the
#: build/holdout split, so it is a repeat measurement and NOT a key to pool over
#: before the statistic is formed -- merging without it fans every join out into
#: a cross product of the three splits.
CELL_KEYS = ["dataset", "embedder", "seed"]


def _wide(df: pd.DataFrame, family: str, scope: str, statistic: str, name: str) -> pd.DataFrame:
    """One row per (dataset, embedder, seed) carrying *statistic* as *name*."""
    sub = _pick(df, family, scope, statistic)[[*CELL_KEYS, "value"]]
    return sub.rename(columns={"value": name})


def _median_se(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Median and its bootstrap standard error, so a difference can be read."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    if x.size == 1:
        return float(x[0]), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.median(rng.choice(x, size=(n_boot, x.size), replace=True), axis=1)
    return float(np.median(x)), float(np.std(boots))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3:
        return float("nan")

    def rank(x):
        o = np.argsort(x, kind="stable")
        r = np.empty(x.size)
        r[o] = np.arange(x.size)
        return r

    ra, rb = rank(a) - rank(a).mean(), rank(b) - rank(b).mean()
    d = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def b1_b2_uniformity(df: pd.DataFrame) -> pd.DataFrame:
    """Is the atlas's stated null true, and is the path averaging why?"""
    ks = _wide(df, "atlas", "holdout", "ks_uniform", "ks_shipped")
    sd = _wide(df, "atlas", "holdout", "sd", "sd_shipped")
    ksd = _wide(df, "atlas_deepest", "holdout", "ks_uniform", "ks_deepest")
    sdd = _wide(df, "atlas_deepest", "holdout", "sd", "sd_deepest")
    pl = _wide(df, "atlas", "holdout", "path_len_mean", "path_len")
    f05 = _wide(df, "atlas", "holdout", "frac_below_0.05", "frac_below_05")
    build = _wide(df, "atlas", "build", "ks_uniform", "ks_build")
    out = ks
    for other in (sd, ksd, sdd, pl, f05, build):
        out = out.merge(other, on=CELL_KEYS, how="outer")
    out["sd_uniform"] = UNIFORM_SD
    out["deepest_widens"] = out["sd_deepest"] > out["sd_shipped"]
    out["path_len_resolvable"] = out["path_len"] >= MIN_PATH_LEN
    return out.sort_values(["embedder", "dataset", "seed"]).reset_index(drop=True)


def b3_dispersion_vs_path(uni: pd.DataFrame) -> pd.DataFrame:
    """Does the under-dispersion track how many nodes were averaged?"""
    d = uni[uni["path_len_resolvable"]] if uni["path_len_resolvable"].any() else uni
    rho = _spearman(d["path_len"].to_numpy(), -d["sd_shipped"].to_numpy())
    return pd.DataFrame(
        [
            {
                "n_cells": int(len(d)),
                "spearman_pathlen_vs_neg_sd": rho,
                "bar": B3_RHO_MIN,
                "meets_bar": bool(np.isfinite(rho) and rho > B3_RHO_MIN),
            }
        ]
    )


def b4_b5_power(shift: pd.DataFrame) -> pd.DataFrame:
    """The guard's null AND its power, which only make sense read together."""
    if shift.empty:
        return pd.DataFrame()
    rows = []
    for is_self, g in shift.groupby("is_self", sort=True):
        rows.append(
            {
                "pairs": "self (held-out split)" if is_self else "cross-dataset",
                "n": int(len(g)),
                "z_median": float(g["z_score"].median()),
                "z_min": float(g["z_score"].min()),
                "z_max": float(g["z_score"].max()),
                "frac_atypical_median": float(g["frac_atypical"].median()),
                "n_shifted": int(g["shifted"].sum()),
                "share_shifted": float(g["shifted"].mean()),
            }
        )
    return pd.DataFrame(rows)


#: Which source each dataset is a slice of. Two slices of Visual Genome are not
#: the "different domain" B5 is about, and pooling them with a Caltech-vs-COCO
#: pair would understate the guard on the shifts that matter and overstate it on
#: the ones that do not. PREREG-part2 named this before the run.
DATASET_SOURCE = {
    "vg_scale_any": "visual_genome",
    "vg_scale": "visual_genome",
    "vg_box_large": "visual_genome",
    "vg_box_medium": "visual_genome",
    "vg_box_small": "visual_genome",
    "visual_genome_m": "visual_genome",
    "coco_val": "coco",
    "caltech101_m": "caltech",
}


def b4_by_embedder(shift: pd.DataFrame) -> pd.DataFrame:
    """The guard's whole operating point, per embedder.

    Three rates side by side, because no one of them is a verdict:

    - **false positives**: it fired on a held-out split of its OWN build data;
    - **detection (different source)**: it fired on a genuinely different corpus,
      which is the job;
    - **detection (same source)**: it fired on another slice of the same corpus,
      where firing is arguably not even wrong.

    Pooled across embedders a 20% self-fire rate reads as noise. Split, it is one
    embedder, and that embedder's detection rate has to be read next to it -
    a guard that fires on everything "detects" everything.
    """
    if shift.empty:
        return pd.DataFrame()
    d = shift.copy()
    d["build_source"] = d["build_dataset"].map(DATASET_SOURCE).fillna("unknown")
    d["query_source"] = d["query_dataset"].map(DATASET_SOURCE).fillna("unknown")
    rows = []
    for emb, g in d.groupby("embedder"):
        self_rows = g[g["is_self"]]
        cross = g[~g["is_self"]]
        diff = cross[cross["build_source"] != cross["query_source"]]
        same = cross[cross["build_source"] == cross["query_source"]]
        rows.append(
            {
                "embedder": emb,
                "n_self": int(len(self_rows)),
                "false_positive_rate": float(self_rows["shifted"].mean()) if len(self_rows) else float("nan"),
                "self_z_median": float(self_rows["z_score"].median()) if len(self_rows) else float("nan"),
                "self_frac_atypical_median": float(self_rows["frac_atypical"].median())
                if len(self_rows)
                else float("nan"),
                "n_diff_source": int(len(diff)),
                "detect_different_source": float(diff["shifted"].mean()) if len(diff) else float("nan"),
                "n_same_source": int(len(same)),
                "detect_same_source": float(same["shifted"].mean()) if len(same) else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    # The one number that says whether the guard separates at all for this
    # embedder: how much more often it fires on a real corpus change than on its
    # own data. Below 1 it is anti-informative.
    out["separation"] = out["detect_different_source"] - out["false_positive_rate"]
    return out.sort_values("false_positive_rate", ascending=False).reset_index(drop=True)


def b5_by_source(shift: pd.DataFrame) -> pd.DataFrame:
    """Cross-dataset power split by whether the two datasets are the same SOURCE.

    A `vg_scale_any` -> `vg_box_large` pair is two slices of Visual Genome; a
    `caltech101_m` -> `coco_val` pair is a different corpus entirely. Reporting
    one number over both answers neither question.
    """
    if shift.empty:
        return pd.DataFrame()
    d = shift[~shift["is_self"]].copy()
    d["build_source"] = d["build_dataset"].map(DATASET_SOURCE).fillna("unknown")
    d["query_source"] = d["query_dataset"].map(DATASET_SOURCE).fillna("unknown")
    d["same_source"] = d["build_source"] == d["query_source"]
    rows = []
    for same, g in d.groupby("same_source", sort=True):
        rows.append(
            {
                "pairs": "same source (slices of one corpus)" if same else "different source",
                "n": int(len(g)),
                "n_shifted": int(g["shifted"].sum()),
                "share_shifted": float(g["shifted"].mean()),
                "z_median": float(g["z_score"].median()),
            }
        )
    return pd.DataFrame(rows)


#: The false-alarm rate the guard's alpha is *supposed* to deliver. `alpha` is
#: both the p-value cut and the claimed rate, which is exactly the identity this
#: run finds broken.
TARGET_FALSE_ALARM = 0.05


def per_embedder_alpha_repair(results: Path, shift: pd.DataFrame) -> pd.DataFrame:
    """Does the recommended repair - a per-embedder alpha - actually work?

    A report that ends at "the guard is miscalibrated, calibrate it per
    embedder" is a recommendation nobody has priced.  This prices it: for each
    embedder, read the alpha that WOULD have produced a 5% false-alarm rate on
    its own held-out data, then re-score every cross-dataset pair at that alpha
    and report the detection rate it buys.

    The repair is honest only if the alpha is fitted on the SELF pairs and
    scored on the CROSS pairs, which is what happens here - fitting it on the
    pairs it is then evaluated against would guarantee a flattering answer.

    ``shifted`` in the shipped guard is ``z > 3 and frac >= 2*alpha``; the same
    rule is applied at the repaired alpha so the comparison is like for like.
    """
    files = sorted(Path(results).glob("domainshift_*.npz"))
    if not files or shift.empty:
        return pd.DataFrame()
    rows = []
    for f in files:
        emb = f.stem.replace("domainshift_", "")
        z = np.load(f)
        g = shift[shift["embedder"] == emb]
        if g.empty:
            continue
        # The alpha that gives a 5% flag rate on this embedder's OWN data: the
        # 5th percentile of the in-domain p-values, pooled over its datasets.
        self_p = [z[k] for k in z.files if k.split("|")[0] == k.split("|")[1] and k in z.files]
        if not self_p:
            continue
        pooled_self = np.concatenate(self_p)
        alpha_star = float(np.quantile(pooled_self, TARGET_FALSE_ALARM))
        fired_self, fired_diff, fired_same = [], [], []
        for _, r in g.iterrows():
            key = f"{r['build_dataset']}|{r['query_dataset']}"
            if key not in z.files:
                continue
            pv = z[key]
            frac = float(np.mean(pv < alpha_star))
            n = pv.size
            se = math.sqrt(alpha_star * (1.0 - alpha_star) / n) if n else 0.0
            zz = (frac - alpha_star) / se if se > 0 else 0.0
            fired = bool(zz > 3.0 and frac >= 2.0 * alpha_star)
            if r["is_self"]:
                fired_self.append(fired)
            elif DATASET_SOURCE.get(r["build_dataset"]) != DATASET_SOURCE.get(r["query_dataset"]):
                fired_diff.append(fired)
            else:
                fired_same.append(fired)
        rows.append(
            {
                "embedder": emb,
                "alpha_shipped": 0.05,
                "alpha_star": alpha_star,
                "fp_rate_shipped": float(g[g["is_self"]]["shifted"].mean()),
                "fp_rate_repaired": float(np.mean(fired_self)) if fired_self else float("nan"),
                "detect_different_source_shipped": float(
                    g[
                        (~g["is_self"])
                        & (g["build_dataset"].map(DATASET_SOURCE) != g["query_dataset"].map(DATASET_SOURCE))
                    ]["shifted"].mean()
                ),
                "detect_different_source_repaired": float(np.mean(fired_diff)) if fired_diff else float("nan"),
                "detect_same_source_repaired": float(np.mean(fired_same)) if fired_same else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["separation_shipped"] = out["detect_different_source_shipped"] - out["fp_rate_shipped"]
        out["separation_repaired"] = out["detect_different_source_repaired"] - out["fp_rate_repaired"]
    return out.sort_values("separation_repaired", ascending=False).reset_index(drop=True)


def c1_c3_projection(df: pd.DataFrame) -> pd.DataFrame:
    trust = _wide(df, "umap", "layout", "trustworthiness_k10", "trust_k10")
    trust50 = _wide(df, "umap", "layout", "trustworthiness_k50", "trust_k50")
    cont = _wide(df, "umap", "layout", "continuity_k10", "cont_k10")
    shep = _wide(df, "umap", "layout", "shepard_spearman", "shepard")
    pe = _wide(df, "umap", "embedding", "knn_class_purity_k10", "purity_embedding")
    pl = _wide(df, "umap", "layout", "knn_class_purity_k10", "purity_layout")
    con = _wide(df, "compaction", "core", "containment_mean", "containment_core")
    conu = _wide(df, "compaction", "unit", "containment_mean", "containment_unit")
    noise = _wide(df, "compaction", "clusters", "noise_fraction", "noise")
    out = trust
    for other in (trust50, cont, shep, pe, pl, con, conu, noise):
        out = out.merge(other, on=CELL_KEYS, how="outer")
    out["purity_drop"] = out["purity_embedding"] - out["purity_layout"]
    return out.sort_values(["embedder", "dataset", "seed"]).reset_index(drop=True)


def conformal_control(df: pd.DataFrame) -> pd.DataFrame:
    ks = _wide(df, "conformal", "in_class_holdout", "ks_uniform", "ks_in")
    n = _pick(df, "conformal", "in_class_holdout", "ks_uniform")[[*CELL_KEYS, "n"]]
    out = ks.merge(n, on=CELL_KEYS, how="left")
    ksout = _wide(df, "conformal", "out_of_class", "median", "median_p_out")
    out = out.merge(ksout, on=CELL_KEYS, how="outer")
    out["resolvable"] = out["n"].fillna(0) >= MIN_CONFORMAL_N
    return out.sort_values(["embedder", "dataset", "seed"]).reset_index(drop=True)


def _by_embedder_count(frame: pd.DataFrame, mask_col: str) -> int:
    """How many embedders have a majority of their datasets satisfying a flag."""
    if frame.empty or mask_col not in frame.columns:
        return 0
    g = frame.groupby("embedder")[mask_col].mean()
    return int((g > 0.5).sum())


def combiner_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Every candidate path combiner, priced on the same held-out data.

    `mean` is what ships. A run that only says "the null is false" leaves the
    reader with nothing to do; this says which of the obvious alternatives is
    closer to uniform, and whether any of them is close enough to use.
    """
    sub = df[df["family"] == "atlas_agg"]
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for scope, g in sub.groupby("scope"):
        ks = g[g["statistic"] == "ks_uniform"]["value"].to_numpy()
        sd = g[g["statistic"] == "sd"]["value"].to_numpy()
        f05 = g[g["statistic"] == "frac_below_0.05"]["value"].to_numpy()
        ks_m, ks_se = _median_se(ks)
        rows.append(
            {
                "combiner": scope,
                "n_cells": int(ks.size),
                "ks_uniform_median": ks_m,
                "ks_uniform_se": ks_se,
                "sd_median": float(np.median(sd)) if sd.size else float("nan"),
                "sd_uniform": UNIFORM_SD,
                "frac_below_05_median": float(np.median(f05)) if f05.size else float("nan"),
                "nominal_frac_below_05": 0.05,
                "is_shipped": scope == "mean",
            }
        )
    return pd.DataFrame(rows).sort_values("ks_uniform_median").reset_index(drop=True)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Analyse the #3329 part-B/C structure run.")
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    results, out = Path(args.results), Path(args.out)
    (out / "agg").mkdir(parents=True, exist_ok=True)

    df, dropped = load_cells(results)
    if df.empty:
        print("no structure rows found; nothing to analyse")
        return 1
    shift = load_shift(results)

    uni = b1_b2_uniformity(df)
    b3 = b3_dispersion_vs_path(uni)
    power = b4_b5_power(shift)
    proj = c1_c3_projection(df)
    conf = conformal_control(df)
    b4emb = b4_by_embedder(shift)
    comb = combiner_comparison(df)
    b5src = b5_by_source(shift)
    repair = per_embedder_alpha_repair(results, shift)
    regions = load_regions(results)
    c4 = c4_region_coherence(regions)

    for name, frame in (
        ("b1_b2_atlas_uniformity", uni),
        ("b3_dispersion_vs_pathlen", b3),
        ("b4_b5_domain_shift_power", power),
        ("c1_c3_projection", proj),
        ("conformal_control", conf),
        ("b4_guard_operating_point", b4emb),
        ("b5_power_by_source", b5src),
        ("b2_combiner_comparison", comb),
        ("b6_alpha_repair", repair),
        ("c4_region_coherence", c4),
    ):
        frame.to_csv(out / "agg" / f"{name}.csv", index=False)

    self_rows = shift[shift["is_self"]] if not shift.empty else pd.DataFrame()
    cross_rows = shift[~shift["is_self"]] if not shift.empty else pd.DataFrame()

    summary: dict[str, Any] = {
        "n_cells": int(df[CELL_KEYS].drop_duplicates().shape[0]),
        "n_environments": int(df[["dataset", "embedder"]].drop_duplicates().shape[0]),
        "n_seeds": int(df["seed"].nunique()),
        "n_cells_dropped": dropped,
        "n_shift_pairs": int(len(shift)),
        "b1_not_uniform": bool(uni["ks_shipped"].median() > B1_KS_MIN),
        "b1_ks_median": _median_se(uni["ks_shipped"].to_numpy())[0],
        "b1_ks_se": _median_se(uni["ks_shipped"].to_numpy())[1],
        "b2_under_dispersed": bool(uni["sd_shipped"].median() < B2_SD_MAX),
        "b2_sd_median": _median_se(uni["sd_shipped"].to_numpy())[0],
        "b2_sd_se": _median_se(uni["sd_shipped"].to_numpy())[1],
        "b2_sd_deepest_median": _median_se(uni["sd_deepest"].to_numpy())[0],
        "b2_ks_deepest_median": _median_se(uni["ks_deepest"].to_numpy())[0],
        "b2_deepest_widens_on_n_embedders": _by_embedder_count(uni, "deepest_widens"),
        "best_combiner": str(comb["combiner"].iloc[0]) if not comb.empty else "",
        "best_combiner_ks": float(comb["ks_uniform_median"].iloc[0]) if not comb.empty else float("nan"),
        "shipped_combiner_ks": float(comb.loc[comb["is_shipped"], "ks_uniform_median"].iloc[0])
        if not comb.empty and comb["is_shipped"].any()
        else float("nan"),
        "b2_meets_bar": bool(
            uni["sd_shipped"].median() < B2_SD_MAX and _by_embedder_count(uni, "deepest_widens") >= B2_MIN_EMBEDDERS
        ),
        "b3_rho": float(b3["spearman_pathlen_vs_neg_sd"].iloc[0]) if not b3.empty else float("nan"),
        "b3_meets_bar": bool(b3["meets_bar"].iloc[0]) if not b3.empty else False,
        "b4_self_fires": int(self_rows["shifted"].sum()) if not self_rows.empty else 0,
        "b4_self_z_median": float(self_rows["z_score"].median()) if not self_rows.empty else float("nan"),
        "b4_meets_bar": bool(
            not self_rows.empty
            and self_rows["shifted"].sum() <= B4_MAX_SELF_FIRES
            and self_rows["z_score"].median() <= 0
        ),
        "b4_worst_embedder_fp_rate": float(b4emb["false_positive_rate"].max()) if not b4emb.empty else float("nan"),
        "b4_min_separation": float(b4emb["separation"].min()) if not b4emb.empty else float("nan"),
        "b4_worst_embedder": str(b4emb["embedder"].iloc[0]) if not b4emb.empty else "",
        "b5_cross_share_shifted": float(cross_rows["shifted"].mean()) if not cross_rows.empty else float("nan"),
        "b5_share_shifted_different_source": float(
            b5src.loc[b5src["pairs"] == "different source", "share_shifted"].iloc[0]
        )
        if not b5src.empty and (b5src["pairs"] == "different source").any()
        else float("nan"),
        "repair_min_separation_shipped": float(repair["separation_shipped"].min())
        if not repair.empty
        else float("nan"),
        "repair_min_separation_repaired": float(repair["separation_repaired"].min())
        if not repair.empty
        else float("nan"),
        "repair_helps": bool(
            not repair.empty and repair["separation_repaired"].min() > repair["separation_shipped"].min()
        ),
        "b5_share_shifted_same_source": float(
            b5src.loc[b5src["pairs"] == "same source (slices of one corpus)", "share_shifted"].iloc[0]
        )
        if not b5src.empty and (b5src["pairs"] == "same source (slices of one corpus)").any()
        else float("nan"),
        "b5_meets_bar": bool(not cross_rows.empty and cross_rows["shifted"].mean() > B5_MIN_OFF_DIAGONAL_SHARE),
        "c1_trust_k10_median": float(proj["trust_k10"].median()),
        "c1_shepard_median": float(proj["shepard"].median()),
        "c1_meets_bar": bool(proj["trust_k10"].median() > C1_TRUST_MIN and proj["shepard"].median() < C1_SHEPARD_MAX),
        "c2_purity_drop_median": float(proj["purity_drop"].median()),
        "c2_meets_bar": _by_embedder_count(proj.assign(drops=proj["purity_drop"] > 0), "drops") >= C2_MIN_EMBEDDERS,
        "c3_containment_median": float(proj["containment_core"].median()),
        "c3_meets_bar": bool(C3_LO <= proj["containment_core"].median() <= C3_HI),
        "c4_n_clusters": int(len(regions)),
        "c4_share_gap_at_or_below_zero": float((regions["coherence_gap"].dropna() <= C4_MIN_GAP).mean())
        if not regions.empty
        else float("nan"),
        "c4_share_purity_below_floor": float((regions["gt_purity"].dropna() < C4_PURITY_FLOOR).mean())
        if not regions.empty and regions["gt_purity"].notna().any()
        else float("nan"),
        "c4_gap_finest_layer": float(c4.loc[c4["layer"] == 0, "coherence_gap_median"].iloc[0])
        if not c4.empty and (c4["layer"] == 0).any()
        else float("nan"),
        "c4_gap_coarsest_layer": float(c4["coherence_gap_median"].iloc[-1]) if not c4.empty else float("nan"),
        "conformal_control_ks_median": float(conf.loc[conf["resolvable"], "ks_in"].median())
        if conf["resolvable"].any()
        else float("nan"),
        "conformal_unresolvable_cells": int((~conf["resolvable"]).sum()),
        "path_len_unresolvable_cells": int((~uni["path_len_resolvable"]).sum()),
    }
    (out / "summary_structure3329.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if not args.no_figures:
        import figures_structure_3329 as F  # noqa: PLC0415

        written = F.all_figures(df, shift, uni, proj, out / "figures", results, comb, regions)
        if written:
            print("figures: " + ", ".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
