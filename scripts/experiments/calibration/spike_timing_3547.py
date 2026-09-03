"""When does a cell acquire its FIRST deep spike?

`frontier_3547.py` reports identical deep-spike incidence at t=100 and t=400 for
all seven arms. That is either the finding H2 predicts (late spikes were
exhaustion, and this pile has none) or a masking bug. Distinguished by asking
the raw trajectories directly: if no cell's FIRST spike lands after t=100, the
two horizons agree because nothing happens between them.
"""

import sys
import numpy as np

sys.path.insert(0, "/exp/sgreenberg/projects/vts-acq-3547/scripts/experiments/calibration")
from _cells_io import load_arm  # noqa: E402
import pathlib  # noqa: E402

WARM_T, DEEP_COST, DEEP_EXCESS = 20, 0.25, 0.20
BASE = pathlib.Path("/expscratch/sgreenberg/acq-3547/bin")

print("%-8s %6s %8s %8s %8s   %s" % ("arm", "cells", "any", "first<=100", "first>100", "first-spike t (quartiles)"))
for arm in ["prod", "acq_m1", "acq_m3", "acq_m4", "acq_m5", "acq_m6", "acq_p2"]:
    raw, _ = load_arm(BASE / arm / "results")
    raw = raw[(raw["embedder"] == "siglip") & (raw["style"] == "whole_image")]
    firsts, n = [], 0
    for _, g in raw.groupby(["category", "seed"], dropna=False):
        n += 1
        g = g.sort_values("t")
        t = g["t"].to_numpy()
        c = g["cost"].to_numpy(dtype=float)
        o = g["oracle_cost"].to_numpy(dtype=float)
        d = (t >= WARM_T) & (c >= DEEP_COST) & ((c - o) >= DEEP_EXCESS)
        firsts.append(float(t[np.argmax(d)]) if d.any() else np.nan)
    f = np.array(firsts, dtype=float)
    has = np.isfinite(f)
    early = int(np.sum(f[has] <= 100))
    late = int(np.sum(f[has] > 100))
    q = np.percentile(f[has], [25, 50, 75]) if has.any() else [np.nan] * 3
    print(
        "%-8s %6d %8d %8d %8d   %s"
        % (arm, n, has.sum(), early, late, "%.0f / %.0f / %.0f" % tuple(q) if has.any() else "-")
    )
