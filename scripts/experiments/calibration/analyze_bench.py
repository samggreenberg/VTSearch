"""Analysis for the production-defaults overview benchmark.

Reports cost = fnr + fpr and calibration regret against the oracle threshold,
banded on the axis the user actually spends (votes), plus per-cell traces.

Everything it drops is counted and printed: analysing N-of-M while reporting
neither number is how a disk incident becomes a wrong verdict.
"""

import sys
from pathlib import Path

import pandas as pd

RESULTS = Path(sys.argv[1] if len(sys.argv) > 1 else "/expscratch/sgreenberg/bench-overview/results")
CELLS = RESULTS / "cells"
EXPECTED = int(sys.argv[2]) if len(sys.argv) > 2 else 189

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def load() -> tuple[pd.DataFrame, dict]:
    files = sorted(f for f in CELLS.glob("task_*.csv") if not any(k in f.name for k in ("sweep", "cutdiag", "cutincl")))
    frames, bad = [], []
    for f in files:
        try:
            df = pd.read_csv(f)
            if df.empty:
                bad.append((f.name, "empty"))
                continue
            frames.append(df)
        except Exception as exc:  # noqa: BLE001
            bad.append((f.name, repr(exc)[:60]))
    prov = {"files_found": len(files), "expected": EXPECTED, "unreadable": bad}
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), prov


df, prov = load()

print("=" * 100)
print("COVERAGE")
print("=" * 100)
print(f"cell files found : {prov['files_found']} / {prov['expected']} expected")
print(f"unreadable/empty : {len(prov['unreadable'])} {prov['unreadable'][:5]}")
print(f"rows loaded      : {len(df)}")
if df.empty:
    sys.exit("no rows")
cells = df.groupby(["dataset", "embedder", "category", "seed"]).ngroups
print(f"distinct cells   : {cells}")
print(
    f"steps per cell   : min={df.groupby(['dataset', 'embedder', 'category', 'seed']).size().min()} "
    f"median={int(df.groupby(['dataset', 'embedder', 'category', 'seed']).size().median())} "
    f"max={df.groupby(['dataset', 'embedder', 'category', 'seed']).size().max()}"
)
print(f"styles           : {sorted(df['style'].unique())}")
print(f"heads            : {sorted(df['head'].unique())}")

# Region voting is a property of the cell, not a flag we trust: patch style on a
# boxed dataset is the only combination that region-votes.
df["arm"] = df["dataset"] + " x " + df["embedder"]
df["region"] = df["style"].ne("whole_image")

print()
print("=" * 100)
print("HEADLINE - deep regime (t >= 100), cost = fpr + fnr, regret vs oracle threshold")
print("=" * 100)
deep = df[df["t"] >= 100]
g = (
    deep.groupby("arm")
    .agg(
        n_steps=("cost", "size"),
        prevalence=("realized_prevalence", "mean"),
        cost=("cost", "mean"),
        fpr=("fpr", "mean"),
        fnr=("fnr", "mean"),
        regret=("regret", "mean"),
        oracle_cost=("oracle_cost", "mean"),
        AP=("average_precision", "mean"),
        auroc=("auroc", "mean"),
    )
    .sort_values("cost")
)
print(g.round(4).to_string())

print()
print("=" * 100)
print("BANDED ON VOTES SPENT - the axis the user actually pays")
print("=" * 100)
bands = [(1, 20), (21, 50), (51, 100), (101, 150)]
rows = []
for lo, hi in bands:
    sub = df[(df["t"] >= lo) & (df["t"] <= hi)]
    for arm, s in sub.groupby("arm"):
        rows.append(
            {
                "arm": arm,
                "votes": f"{lo}-{hi}",
                "n": len(s),
                "cost": s["cost"].mean(),
                "fpr": s["fpr"].mean(),
                "fnr": s["fnr"].mean(),
                "regret": s["regret"].mean(),
                "AP": s["average_precision"].mean(),
            }
        )
band = pd.DataFrame(rows).pivot(index="arm", columns="votes", values=["cost", "regret"])
print(band.round(4).to_string())

print()
print("=" * 100)
print("WHERE THE REGRET COMES FROM (rule inefficiency vs calibration shift)")
print("=" * 100)
dec = df.dropna(subset=["rule_inefficiency", "calibration_shift"])
if dec.empty:
    print("no decomposition rows emitted (needs the calibration-metrics path)")
else:
    print(f"rows with a decomposition: {len(dec)} / {len(df)}")
    d = (
        dec[dec["t"] >= 100]
        .groupby("arm")
        .agg(
            regret=("regret", "mean"),
            rule_ineff=("rule_inefficiency", "mean"),
            cal_shift=("calibration_shift", "mean"),
        )
    )
    d["cal_share"] = d["cal_shift"] / (d["rule_ineff"] + d["cal_shift"])
    print(d.round(4).to_string())

print()
print("=" * 100)
print("FAILURE MODES - how the threshold was chosen")
print("=" * 100)
prov_tab = df.groupby(["arm", "threshold_provenance"]).size().unstack(fill_value=0)
prov_pct = (prov_tab.T / prov_tab.sum(axis=1)).T * 100
print(prov_pct.round(1).to_string())
print()
print(f"degenerate steps : {int(df['degenerate'].sum())} / {len(df)} ({100 * df['degenerate'].mean():.2f}%)")
print(f"cut_fallback     : {int(df['cut_fallback'].sum())} / {len(df)} ({100 * df['cut_fallback'].mean():.2f}%)")
zero_fpr_deep = deep[deep["fpr"] == 0]
print(
    f"deep steps with fpr==0 : {len(zero_fpr_deep)} / {len(deep)} ({100 * len(zero_fpr_deep) / max(len(deep), 1):.2f}%)"
    f"  [mean fnr there: {zero_fpr_deep['fnr'].mean():.4f}]"
)

print()
print("=" * 100)
print("STARVATION - do runs actually find positives?")
print("=" * 100)
last = df.sort_values("t").groupby(["dataset", "embedder", "category", "seed"]).tail(1)
st = last.groupby("arm").agg(
    cells=("n_good", "size"),
    median_good=("n_good", "median"),
    zero_good=("n_good", lambda s: int((s == 0).sum())),
    median_t=("t", "median"),
)
st["pct_zero_good"] = (100 * st["zero_good"] / st["cells"]).round(1)
print(st.to_string())

print()
print("=" * 100)
print("COST OF A STEP (seconds)")
print("=" * 100)
print(df.groupby("arm")["elapsed_seconds"].agg(["mean", "median", "max"]).round(2).to_string())

# --- traces --------------------------------------------------------------
print()
print("=" * 100)
print("INDIVIDUAL TRACES")
print("=" * 100)
trace_cols = [
    "t",
    "n_good",
    "n_bad",
    "cost",
    "fpr",
    "fnr",
    "regret",
    "average_precision",
    "threshold_provenance",
    "degenerate",
]
picks = [
    ("visual_genome_m", "dinov3_patch"),
    ("visual_genome_m", "siglip"),
    ("caltech101_m", "dinov3_patch"),
    ("coco_val", "siglip2_l"),
]
for ds, emb in picks:
    sub = df[(df["dataset"] == ds) & (df["embedder"] == emb)]
    if sub.empty:
        continue
    key = sub.groupby(["category", "seed"]).size().idxmax()
    one = sub[(sub["category"] == key[0]) & (sub["seed"] == key[1])].sort_values("t")
    print(
        f"\n--- {ds} x {emb} | category={key[0]} seed={key[1]} "
        f"(prevalence={one['realized_prevalence'].iloc[0]:.4f}, style={one['style'].iloc[0]}) ---"
    )
    step = max(1, len(one) // 12)
    print(one[trace_cols].iloc[::step].round(4).to_string(index=False))

out = RESULTS / "ANALYSIS_TABLES.txt"
print(f"\n(tables also written to {out})")
