"""#4910 (fold_transfer_4910.sh's runs), in F-beta only: from the rank frames (each withheld positive's rank under the step's final model,
and the line's set size), the F-beta the same ranking reaches at other set sizes. Per arm, preset, click: the line's
size over the F-beta-best size (median), and the mean F-beta change from keeping x0.5 / x0.71 / x1.41 / x2."""

import glob
import sys
from multiprocessing import Pool
import numpy as np
import pandas as pd

S = "/expscratch/sgreenberg/state-of-the-app"
TAG = {"025": 0.25, "1": 1.0, "4": 4.0}
SCALES = (0.5, 0.71, 1.41, 2.0)


def fb(tp, k, P, b):
    b2 = b * b
    return np.where(k > 0, (1 + b2) * tp / (b2 * P + k), 0.0)


def read(args):
    f, bt = args
    m = pd.read_csv(f, low_memory=False)
    m = m[(m.embedder == "siglip") & (m.kind == "step")]
    b = TAG[bt]
    out = []
    for _, r in m.iterrows():
        k = r[f"test_line_k_b{bt}"]
        if not np.isfinite(k) or k < 0 or not isinstance(r.test_pos_ranks, str):
            continue
        ranks = np.sort(np.array(r.test_pos_ranks.split(), dtype=np.int64))
        P = int(r.n_test_pos)
        n = int(r.n_test)
        ks = np.arange(1, n + 1)
        tp = np.searchsorted(ranks, ks, side="left")  # positives ranked above k
        F = fb(tp, ks, P, b)
        kbest = int(ks[np.argmax(F)])

        def Fk(kk, ranks=ranks, P=P, b=b):
            return float(fb(np.searchsorted(ranks, kk, side="left"), kk, P, b)) if kk >= 1 else 0.0

        k = int(k)
        f0 = Fk(k)
        out.append(
            {
                "t": int(r.t),
                "k": k,
                "kbest": kbest,
                "P": P,
                "F": f0,
                "Fbest": float(F.max()),
                **{f"dF_x{s}": Fk(max(1, int(round(k * s)))) - f0 for s in SCALES},
            }
        )
    return out


rows = []
for bt in ("025", "1", "4"):
    for arm in ("c", "q", "qt"):
        fs = sorted(glob.glob(f"{S}/2026-10-15-xfer4910-{arm}-binary-b{bt}/results/cells/task_*__rankframes.csv.gz"))
        with Pool(12) as p:
            d = pd.DataFrame([x for part in p.map(read, [(f, bt) for f in fs]) for x in part])
        for t, g in d.groupby("t"):
            if t < 25:
                continue
            rows.append(
                {
                    "beta": TAG[bt],
                    "arm": arm,
                    "t": t,
                    "n": len(g),
                    "k_med": g.k.median(),
                    "kbest_med": g.kbest.median(),
                    "P_med": g.P.median(),
                    "log2_k/kbest_med": float(np.median(np.log2((g.k + 1) / (g.kbest + 1)))),
                    "F": g.F.mean(),
                    "Fbest": g.Fbest.mean(),
                    **{c: g[c].mean() for c in g.columns if c.startswith("dF_x")},
                }
            )
df = pd.DataFrame(rows)
df.to_csv(sys.argv[1] if len(sys.argv) > 1 else "fold_transfer_4910_sizes.csv", index=False)
with pd.option_context("display.width", 250, "display.max_columns", 30):
    print(df.round(3).to_string(index=False))
