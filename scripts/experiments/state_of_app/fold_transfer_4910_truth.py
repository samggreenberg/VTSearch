"""#4910 (fold_transfer_4910.sh's runs): each arm's Find line against the truth on the withheld half, by preset and
click bucket, in F-beta only: the gap to the best cut's F-beta, the line's own expected F-beta (find_efbeta) against
the F-beta it got, and its prevalence estimate against the true prevalence."""

import glob
import sys
from multiprocessing import Pool
import pandas as pd

S = "/expscratch/sgreenberg/state-of-the-app"
COLS = [
    "category",
    "seed",
    "embedder",
    "t",
    "phase",
    "detector_tier",
    "fbeta",
    "oracle_fbeta",
    "n_flagged",
    "n_test_pos",
    "n_test_neg",
    "find_prevalence",
    "find_efbeta",
    "precision",
    "recall",
    "gmm_variant",
    "pool_variant",
    "line_head_count",
]


def read(f):
    h = pd.read_csv(f, nrows=0).columns
    m = pd.read_csv(f, usecols=[c for c in COLS if c in h], low_memory=False)
    m = m[
        (m.embedder == "siglip")
        & (m.pool_variant.fillna("max") == "max")
        & (m.gmm_variant.isna() | (m.gmm_variant.astype(str) == ""))
    ]
    m = m[(m.phase.astype(str) != "check") & (m.t <= 150) & (m.detector_tier.astype(str) == "trained")]
    return m.sort_values("t").drop_duplicates(["category", "seed", "t"], keep="last")


def load(arm, b):
    fs = sorted(
        x for x in glob.glob(f"{S}/2026-10-15-xfer4910-{arm}-binary-b{b}/results/cells/task_*.csv.gz") if "__" not in x
    )
    with Pool(12) as p:
        return pd.concat(p.map(read, fs))


out = []
for b in ("025", "1", "4"):
    for arm in ("c", "q", "qt"):
        m = load(arm, b)
        P, N = m.n_test_pos.astype(float), m.n_test_neg.astype(float)
        prev = P / (P + N)
        for lo, hi in ((31, 50), (51, 100), (101, 150)):
            k = (m.t >= lo) & (m.t <= hi)
            out.append(
                {
                    "beta": b,
                    "arm": arm,
                    "clicks": f"{lo}-{hi}",
                    "rows": int(k.sum()),
                    "kept_med": float(m.n_flagged[k].median()),
                    "F": float(m.fbeta[k].mean()),
                    "oracle_F": float(m.oracle_fbeta[k].mean()),
                    "F_gap": float((m.oracle_fbeta - m.fbeta)[k].mean()),
                    "prev_est/true_med": float((m.find_prevalence / prev)[k].median()),
                    "efbeta-F_mean": float((m.find_efbeta - m.fbeta)[k].mean()),
                    "P": float(m.precision[k].mean()),
                    "R": float(m.recall[k].mean()),
                }
            )
df = pd.DataFrame(out)
df.to_csv(sys.argv[1] if len(sys.argv) > 1 else "fold_transfer_4910_truth.csv", index=False)
with pd.option_context("display.width", 250, "display.max_columns", 30):
    print(df.round(3).to_string(index=False))
