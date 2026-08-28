"""Folds addendum (#2861): does K=4 change the answer, and does the combine matter?

Two questions the main run structurally could not ask, both scoped to
VG x siglip (whole_image):

1. **qmean vs qmedian** — byte-identical at production's two folds, distinct at
   four.  This one is a *paired* contrast: both arms re-cut the same per-step
   fold fits, so it is tested cell-paired within the K=4 run.
2. **K=4 vs K=2** — NOT paired: changing the fold count changes the splits, the
   per-fold models and therefore the trajectory.  The contrast that survives is
   each run's own `Delta vs xcal_only`, because the control moves with the
   trajectory; it is compared across runs as an unpaired difference of those
   deltas, with a Mann-Whitney over cell means rather than a paired test.

Usage: python analyze_folds.py <k4_results_dir> <k2_results_dir>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _cells_io import assert_one_opening, main_frame_files
from scipy.stats import mannwhitneyu, wilcoxon

FOLD_RE = re.compile(r"^fold_anchored_w(?P<w>[\d.]+)_(?P<rule>mid|rate)_(?P<combine>\w+)$")
DEEP = 100
ENV = "visual_genome_m/siglip/whole_image"
CELL = ["category", "seed", "window"]


def load(results: Path) -> pd.DataFrame:
    files = main_frame_files(results)
    frames = [pd.read_csv(p) for p in files if p.stat().st_size > 0]
    df = pd.concat(frames, ignore_index=True)
    assert_one_opening(df, "analyze_folds.py")
    df["gmm_variant"] = df["gmm_variant"].fillna("")
    df["env"] = df["dataset"] + "/" + df["embedder"] + "/" + df["style"]
    df = df[df["env"] == ENV]
    df["n_votes"] = df["n_good"] + df["n_bad"]
    edges, labels = [1, 20, 50, 100, 200, 300], ["le_20", "le_50", "le_100", "le_200", "le_300"]
    df["window"] = pd.cut(df["n_votes"], bins=edges, labels=labels)
    df = df[df["window"].notna()]
    df["window_hi"] = df["window"].map({"le_20": 20, "le_50": 50, "le_100": 100, "le_200": 200, "le_300": 300})
    print(f"  {results}: {len(files)} cells, {len(df):,} rows, {df['category'].nunique()} categories")
    return df


def deltas_vs_xcal(df: pd.DataFrame) -> pd.DataFrame:
    """Cell-mean regret delta of every arm against xcal_only, deep windows."""
    keys = ["category", "seed", "t", "window"]
    c = df[df["gmm_variant"] == "xcal_only"].set_index(keys)["regret"]
    c = c[~c.index.duplicated()]
    out = []
    for name, a in df.groupby("gmm_variant", observed=True):
        if name in ("", "xcal_only"):
            continue
        a = a.set_index(keys)["regret"]
        a = a[~a.index.duplicated()]
        j = pd.concat([a.rename("a"), c.rename("c")], axis=1, join="inner").reset_index()
        if j.empty:
            continue
        j["d"] = j["a"] - j["c"]
        g = j.groupby(CELL, observed=True)["d"].mean().reset_index()
        g["gmm_variant"] = name
        out.append(g)
    d = pd.concat(out, ignore_index=True)
    d["window_hi"] = d["window"].map({"le_20": 20, "le_50": 50, "le_100": 100, "le_200": 200, "le_300": 300})
    return d[d["window_hi"] >= DEEP]


def annotate(d: pd.DataFrame) -> pd.DataFrame:
    m = d["gmm_variant"].str.extract(FOLD_RE)
    d = d.copy()
    d["kappa"] = pd.to_numeric(m["w"], errors="coerce")
    d["rule"] = m["rule"]
    d["combine"] = m["combine"]
    return d


def main() -> int:
    k4 = load(Path(sys.argv[1]) / "cells")
    k2 = load(Path(sys.argv[2]) / "cells")
    d4, d2 = annotate(deltas_vs_xcal(k4)), annotate(deltas_vs_xcal(k2))

    print("\n=== 1. qmean vs qmedian at K=4 (paired within run; negative = qmedian better) ===")
    rows = []
    f4 = d4[d4["combine"].notna()]
    for (kappa, rule), g in f4.groupby(["kappa", "rule"], observed=True):
        w = g.pivot_table(index=CELL, columns="combine", values="d", observed=True)
        if not {"qmean", "qmedian"} <= set(w.columns):
            continue
        w = w.dropna()
        diff = w["qmedian"] - w["qmean"]
        identical = bool(np.allclose(diff, 0))
        p = float("nan") if identical or len(diff) < 6 else float(wilcoxon(diff, zero_method="zsplit").pvalue)
        rows.append(
            {
                "kappa": kappa,
                "rule": rule,
                "qmean": w["qmean"].mean(),
                "qmedian": w["qmedian"].mean(),
                "diff": diff.mean(),
                "identical": identical,
                "n": len(w),
                "p": p,
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    print("\n=== 2. K=4 vs K=2, deep Delta-vs-xcal by (kappa, rule); qmean arm ===")
    rows = []
    for (kappa, rule), g4 in d4[d4["combine"] == "qmean"].groupby(["kappa", "rule"], observed=True):
        g2 = d2[(d2["combine"] == "qmean") & (d2["kappa"] == kappa) & (d2["rule"] == rule)]
        if g2.empty:
            continue
        p = float(mannwhitneyu(g4["d"], g2["d"]).pvalue) if min(len(g4), len(g2)) > 5 else float("nan")
        rows.append(
            {
                "kappa": kappa,
                "rule": rule,
                "K4": g4["d"].mean(),
                "K2": g2["d"].mean(),
                "K4_minus_K2": g4["d"].mean() - g2["d"].mean(),
                "n4": len(g4),
                "n2": len(g2),
                "p_unpaired": p,
            }
        )
    t = pd.DataFrame(rows).sort_values(["rule", "kappa"])
    print(t.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    print("\n=== 3. argmin kappa under each fold count (qmean) ===")
    for name, d in (("K=4", d4), ("K=2", d2)):
        f = d[d["combine"] == "qmean"]
        for rule, g in f.groupby("rule", observed=True):
            means = g.groupby("kappa", observed=True)["d"].mean()
            print(
                f"  {name} {rule}: argmin kappa={means.idxmin():g}  best={means.min():+.4f}  "
                f"at kappa=0.3 {means.get(0.3, float('nan')):+.4f}  at kappa=1 {means.get(1.0, float('nan')):+.4f}"
            )

    print("\n=== 4. controls under K=4 (deep, vs xcal) ===")
    ctl = d4[d4["combine"].isna() & d4["gmm_variant"].isin(["rank_transfer", "sched:slow_cap50", "pooled_mid"])]
    if not ctl.empty:
        print(ctl.groupby("gmm_variant")["d"].agg(["mean", "size"]).to_string(float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
