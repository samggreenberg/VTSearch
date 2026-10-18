"""#4910 (fold_transfer_4910.sh's runs): where do arm q's picks first leave arm c's? Must be past the opening (the line is read from the first trained step)."""

import glob
import sys
import pandas as pd

S = "/expscratch/sgreenberg/state-of-the-app"
tag, beta = (sys.argv[1] if len(sys.argv) > 1 else "smoke"), (sys.argv[2] if len(sys.argv) > 2 else "1")
arm2 = sys.argv[3] if len(sys.argv) > 3 else "q"
OPEN = {"good", "bad", "more"}
bad = 0
for f in sorted(glob.glob(f"{S}/2026-10-15-xfer4910{tag}-c-binary-b{beta}/results/cells/task_*__picks.csv.gz")):
    g = f.replace("-c-binary", f"-{arm2}-binary")
    try:
        a, b = pd.read_csv(f), pd.read_csv(g)
    except FileNotFoundError:
        continue
    a = a[a["embedder"] == "siglip"].reset_index(drop=True)
    b = b[b["embedder"] == "siglip"].reset_index(drop=True)
    n = min(len(a), len(b))
    i = next((k for k in range(n) if a.loc[k, "picked_id"] != b.loc[k, "picked_id"]), None)
    last_open = max((k for k in range(len(a)) if a.loc[k, "phase"] in OPEN), default=-1)
    ok = i is None or i > last_open
    bad += not ok
    print(
        f"{f.split('/')[-1][:9]} {a.loc[0, 'category']}: picks {len(a)}/{len(b)} first diff idx {i} "
        f"(t={a.loc[i, 't'] if i is not None else '-'} phase c={a.loc[i, 'phase'] if i is not None else '-'} "
        f"{arm2}={b.loc[i, 'phase'] if i is not None else '-'}) opening ends idx {last_open} {'OK' if ok else 'IN OPENING'}"
    )
print("in-opening departures:", bad)
