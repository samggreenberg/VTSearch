"""Does the pool shape change how the four columns RANK, or only their spacing?"""

import json
import glob
import statistics
import itertools

M = "docs/experiments/2026-09-20-negative-pool-3986/measurements/"
data = {}
for p in sorted(glob.glob(M + "*.json")):
    d = json.load(open(p))
    if "d_auc_mean" not in d:
        continue
    data[d["embedder"]] = {c["cell"]: c for c in d["cells"]}

embs = sorted(data, key=lambda e: statistics.mean(c["ap_shared"] for c in data[e].values()))
cells = sorted(set.intersection(*(set(v) for v in data.values())))
print("cells:", len(cells), " embedders:", embs, "\n")

print("%-12s %9s %9s %8s" % ("embedder", "AP shared", "AP p/c", "delta"))
for e in embs:
    a = statistics.mean(data[e][c]["ap_shared"] for c in cells)
    b = statistics.mean(data[e][c]["ap_perclass"] for c in cells)
    print("%-12s %9.3f %9.3f %+8.3f" % (e, a, b, b - a))

print("\npairwise GAP between columns (mean AP difference):")
print("%-24s %9s %9s %9s" % ("pair", "shared", "per-class", "change"))
for x, y in itertools.combinations(embs, 2):
    gs = statistics.mean(data[y][c]["ap_shared"] - data[x][c]["ap_shared"] for c in cells)
    gp = statistics.mean(data[y][c]["ap_perclass"] - data[x][c]["ap_perclass"] for c in cells)
    print("%-24s %9.3f %9.3f %+8.1f%%" % (y + " - " + x, gs, gp, 100 * (gp - gs) / gs if gs else float("nan")))

# Per cell: does the winning column change?
flips = 0
for c in cells:
    ws = max(embs, key=lambda e: data[e][c]["ap_shared"])
    wp = max(embs, key=lambda e: data[e][c]["ap_perclass"])
    if ws != wp:
        flips += 1
        print("  FLIP %-20s shared->%s  per-class->%s" % (c, ws, wp))
print("\nper-cell winner changes: %d of %d cells" % (flips, len(cells)))
