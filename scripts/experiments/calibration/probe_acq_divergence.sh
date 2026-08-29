#!/bin/bash
# Does the lever diverge from k=0 at all in THIS environment, before 14 arrays
# are committed to it?  #2905's re-run note is explicit that this is the first
# question, not the last: an arm whose acquisition cut is pinned above the whole
# pool is inert, and an inert arm looks exactly like a lever that does nothing.
#
# Reads the `size` cells directly (they are one cell each, so this is a picture
# and not a test) across an EASY category and the HARDEST one.
set -u
source /exp/sgreenberg/projects/vts-acq-2877/gridenv.sh >/dev/null 2>&1
cd /exp/sgreenberg/projects/vts-acq-2877/scripts/experiments/calibration || exit 1
export CALIB_EXP=/expscratch/sgreenberg/acq-2877/sizing
CUDA_VISIBLE_DEVICES= python - <<'PY'
import glob
import pathlib

import common

common.setup_env()
import pandas as pd

import analyze_spikes as sp  # noqa: F401  (for _base_rows)
from analyze_spikes import _base_rows

rows = []
for d in sorted(glob.glob("/expscratch/sgreenberg/acq-2877/sizing/*-*")):
    half, arm = pathlib.Path(d).name.split("-", 1)
    for f in sorted(glob.glob(f"{d}/task_*.csv")):
        if "__" in pathlib.Path(f).name:
            continue
        fr = _base_rows(pd.read_csv(f))
        if fr.empty:
            continue
        for (style, cat), g in fr.groupby(["style", "category"]):
            g = g.sort_values("t")
            rows.append(
                dict(
                    half=half,
                    arm=arm,
                    style=style,
                    category=cat,
                    seed=int(g["seed"].iloc[0]),
                    steps=len(g),
                    pos100=int(g["n_good"].iloc[-1]),
                    cost=float(g["cost"].iloc[-1]),
                    ap=float(g["average_precision"].iloc[-1]),
                    oracle=float(g["oracle_cost"].iloc[-1]),
                    acq_pct=float(g["acq_pool_percentile"].median()),
                    rep_pct=float(g["report_pool_percentile"].median()),
                    acq_pinned=float((g["acq_pool_percentile"] >= 0.9999).mean()),
                    acq_moved=float((g["acq_threshold"] != g["threshold"]).mean()),
                )
            )
df = pd.DataFrame(rows).sort_values(["style", "category", "seed", "arm"])
pd.set_option("display.width", 220)
print(df.to_string(index=False))
print()
print("--- paired arm-vs-prod, per (style, category, seed) ---")
# `half` is in the key: `siglip x whole_image` and the pair's `whole_image`
# style are BOTH binary, and joining without it pairs a cell against another
# environment's control.
key = ["half", "style", "category", "seed"]
p = df[df.arm == "prod"].set_index(key)
for arm in sorted(set(df.arm) - {"prod"}):
    a = df[df.arm == arm].set_index(key)
    j = a.join(p, rsuffix="_prod", how="inner")
    if j.empty:
        continue
    for k, r in j.iterrows():
        print(
            f"{arm:8s} {k[0]:4s} {k[1]:11s} {k[2]:10s} s{k[3]}  "
            f"pos {r.pos100_prod:3.0f}->{r.pos100:3.0f}   "
            f"cost {r.cost_prod:.3f}->{r.cost:.3f}   "
            f"AP {r.ap_prod:.3f}->{r.ap:.3f}   "
            f"acq_pct {r.acq_pct_prod:.3f}->{r.acq_pct:.3f}   "
            f"pinned@1.0 {100 * r.acq_pinned_prod:.0f}%->{100 * r.acq_pinned:.0f}%"
        )
PY
