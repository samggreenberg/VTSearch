"""Analysis for the production-defaults overview benchmark.

    python analyze_bench.py <results> [expected_cells] [tables_out] [deep_from]


Reports cost = fnr + fpr and calibration regret against the oracle threshold,
banded on the axis the user actually spends (votes), plus per-cell traces.

Everything it drops is counted and printed: analysing N-of-M while reporting
neither number is how a disk incident becomes a wrong verdict.

Arm-vs-arm differences are reported **paired** (same category, same seed, same
split) with a standard error, because that is the only form in which the size of
a difference is readable. An unpaired mean of 0.0462 against 0.0508 invites a
"the margin grows" claim that a +-0.03 standard error cannot support; the paired
contrast says plainly which comparisons the sample can resolve and which it
cannot.
"""

import sys
from pathlib import Path

import pandas as pd
from bench_cells import load_cells, paired_contrasts

RESULTS = Path(sys.argv[1] if len(sys.argv) > 1 else "/expscratch/sgreenberg/bench-overview/results")
CELLS = RESULTS / "cells"
EXPECTED = int(sys.argv[2]) if len(sys.argv) > 2 else 189
TABLES = Path(sys.argv[3]) if len(sys.argv) > 3 else RESULTS / "ANALYSIS_TABLES.txt"
#: First vote of the deep-regime window. Derived from the run's own horizon (its
#: last third) unless given, so a 250-vote study is not summarised on the window
#: a 150-vote study happened to use.
DEEP_FROM = int(sys.argv[4]) if len(sys.argv) > 4 else 0

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


class _Tee:
    """Print to the terminal *and* to the tables file in one pass.

    The old version told you the tables were "also written to" a path it never
    wrote to; whoever ran it had to remember to redirect. A file the script
    claims to write should exist.
    """

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        # pandas pads table rows to a fixed width, and the repo's pre-commit hook
        # strips that trailing whitespace - which would leave the committed
        # tables differing from what this script emits. Strip it here instead.
        stripped = "\n".join(line.rstrip() for line in data.split("\n"))
        for stream in self._streams:
            stream.write(data if stream is sys.__stdout__ else stripped)
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


TABLES.parent.mkdir(parents=True, exist_ok=True)
_tables_fh = TABLES.open("w")
sys.stdout = _Tee(sys.__stdout__, _tables_fh)


df, prov = load_cells(RESULTS, quiet=True)

print("=" * 100)
print("COVERAGE")
print("=" * 100)
print(f"cell files found : {prov['n_files']} / {EXPECTED} expected")
print(f"header-only      : {len(prov['header_only'])} starved cells {prov['header_only'][:5]}")
print(f"zero-byte        : {len(prov['zero_byte'])} {prov['zero_byte'][:5]}")
print(f"unreadable       : {len(prov['unreadable'])} {prov['unreadable'][:5]}")
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
HORIZON = int(df["t"].max())
DEEP_FROM = DEEP_FROM or int(round(HORIZON * 2 / 3))
print(f"horizon          : {HORIZON} votes; deep regime = t >= {DEEP_FROM}")
print(f"heads            : {sorted(df['head'].unique())}")

# Region voting is a property of the cell, not a flag we trust: patch style on a
# boxed dataset is the only combination that region-votes.
df["arm"] = df["dataset"] + " x " + df["embedder"]
df["region"] = df["style"].ne("whole_image")

print()
print("=" * 100)
print(f"HEADLINE - deep regime (t >= {DEEP_FROM}), cost = fpr + fnr, regret vs oracle threshold")
print("=" * 100)
deep = df[df["t"] >= DEEP_FROM]
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
# Vote bands: fixed early edges (that is where the curve moves) plus a final
# band that stretches to whatever horizon this run used.
bands = [(1, 20), (21, 50), (51, 100), (101, min(150, HORIZON))]
if HORIZON > 150:
    bands.append((151, HORIZON))
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
        dec[dec["t"] >= DEEP_FROM]
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
print("PAIRED ARM CONTRASTS - deep regime, same (category, seed) on both sides")
print("=" * 100)
print("mean difference +- standard error over paired cells; |mean| < 2*SE is NOT resolvable here")
pairs = paired_contrasts(deep)
print(pairs.to_string(index=False) if not pairs.empty else "(only one embedder in this run; nothing to pair)")

print()
print("=" * 100)
print("COST OF A STEP (seconds)")
print("=" * 100)
# `elapsed_seconds` is CUMULATIVE from the cell's start, so aggregating it directly
# reports "how far into the cell was this step", not what a step costs. Difference it
# within each cell first; the per-cell total is reported separately as wall time.
cell_key = ["dataset", "embedder", "category", "seed"]
ordered = df.sort_values(cell_key + ["t"])
ordered["step_seconds"] = ordered.groupby(cell_key)["elapsed_seconds"].diff()
per_step = ordered.groupby("arm")["step_seconds"].agg(["mean", "median", "max"]).round(2)
per_cell = ordered.groupby(cell_key + ["arm"])["elapsed_seconds"].max().groupby("arm").agg(["median", "max"])
per_step[["cell_median", "cell_max"]] = per_cell.round(1)
print(per_step.to_string())
print("(mean/median/max are per STEP; cell_* are whole-cell wall time, both in seconds)")

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

print(f"\n(these tables were also written to {TABLES})")
sys.stdout = sys.__stdout__  # put the real stream back before the file goes away
_tables_fh.close()
