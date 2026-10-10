"""#4671: Goods found by votes 25 / 50 / 150 per arm (the cells' n_good on headline rows), paired against ctl."""

import glob
import sys

import numpy as np
import pandas as pd

B = sys.argv[1] if len(sys.argv) > 1 else "/expscratch/sgreenberg/autopilot-4671"
rows = []
for p in ("b025", "b1", "b4"):
    per = {}
    for arm in ("ctl", "open", "nowalk"):
        recs = []
        for f in sorted(glob.glob(f"{B}/{arm}_{p}/results/cells/task_[0-9][0-9][0-9][0-9].csv.gz")):
            try:
                d = pd.read_csv(f, usecols=["category", "seed", "t", "n_good", "gmm_variant"], low_memory=False)
            except Exception:
                continue
            d = d[d["gmm_variant"].isna()]
            if d.empty:
                continue
            g = d.groupby("t")["n_good"].max()
            rec = {"category": d["category"].iloc[0], "seed": int(d["seed"].iloc[0])}
            for t in (25, 50, 150):
                rec[f"g{t}"] = float(g[g.index <= t].max()) if (g.index <= t).any() else 0.0
            recs.append(rec)
        per[arm] = pd.DataFrame(recs).set_index(["category", "seed"])
    for arm in ("ctl", "open", "nowalk"):
        row = {"preset": p, "arm": arm}
        for t in (25, 50, 150):
            row[f"goods@{t}"] = per[arm][f"g{t}"].mean()
            if arm != "ctl":
                j = per[arm].join(per["ctl"], rsuffix="_c", how="inner")
                dd = j[f"g{t}"] - j[f"g{t}_c"]
                row[f"d@{t}"] = f"{dd.mean():+.2f} ± {dd.std(ddof=1) / np.sqrt(len(dd)):.2f}"
        rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(f"{B}/analysis/goods.csv", index=False, float_format="%.4g")
print(out.to_string(index=False))
