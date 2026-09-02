import csv
import glob
import os
import statistics

BASE = "/expscratch/sgreenberg/acq-3319-deep/bin"
for arm in ["prod", "acq_m1", "acq_m3", "acq_m4"]:
    d = os.path.join(BASE, arm, "results", "cells")
    files = [f for f in sorted(glob.glob(d + "/task_*.csv")) if "__" not in os.path.basename(f)]
    best = {}  # (seed, cat) -> (t, n_good, sim_pos)
    for f in files:
        with open(f) as fh:
            for row in csv.DictReader(fh):
                key = (row["seed"], row["category"])
                t = int(row["t"])
                sim_pos = int(round(float(row["n_haystack"]) * float(row["realized_prevalence"])))
                cur = best.get(key)
                if cur is None or t > cur[0]:
                    best[key] = (t, int(row["n_good"]), sim_pos)
    harv = [g / p for (_, g, p) in best.values() if p]
    tt = [t for (t, _, _) in best.values()]
    ng = [g for (_, g, _) in best.values()]
    if not harv:
        print(arm, "NO DATA", len(files))
        continue
    harv.sort()
    over90 = sum(1 for h in harv if h > 0.90) / len(harv)
    over80 = sum(1 for h in harv if h > 0.80) / len(harv)
    print(
        "%-8s cells=%4d  final_t=%s  median n_good=%5.1f  sim_pos=%d  "
        "harvest med=%.1f%%  p90=%.1f%%  >80%%:%.1f%%  >90%%:%.1f%%"
        % (
            arm,
            len(best),
            sorted(set(tt)),
            statistics.median(ng),
            statistics.median([p for (_, _, p) in best.values()]),
            100 * statistics.median(harv),
            100 * harv[int(0.9 * len(harv))],
            100 * over80,
            100 * over90,
        )
    )
