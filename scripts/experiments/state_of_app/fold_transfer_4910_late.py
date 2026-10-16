"""#4910 (fold_transfer_4910.sh's runs): qt's late dip at beta 1. Same-head (arm's own heads: own line minus the raw shadow) by click bucket, and the
set at the last ordinary step <= 150 (P, R, kept, F) for c and qt, paired by (category, seed)."""

import glob
import sys
from multiprocessing import Pool
import numpy as np
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
    "precision",
    "recall",
    "n_flagged",
    "shadow_fbeta",
    "shadow_n_flagged",
    "xfer_clip_good",
    "line_separation",
    "shadow_separation",
    "gmm_variant",
    "schedule",
    "pool_variant",
    "n_test_pos",
]


def read(f):
    h = pd.read_csv(f, nrows=0).columns
    m = pd.read_csv(f, usecols=[c for c in COLS if c in h], low_memory=False)
    m = m[
        (m.embedder == "siglip")
        & (m.pool_variant.fillna("max") == "max")
        & (m.gmm_variant.isna() | (m.gmm_variant.astype(str) == ""))
    ]
    m = m[m.phase.astype(str) != "check"]
    return m[m.t <= 150].sort_values("t").drop_duplicates(["category", "seed", "t"], keep="last")


def load(arm, beta):
    fs = sorted(
        f
        for f in glob.glob(f"{S}/2026-10-15-xfer4910-{arm}-binary-b{beta}/results/cells/task_*.csv.gz")
        if "__" not in f
    )
    with Pool(8) as p:
        return pd.concat(p.map(read, fs))


beta = sys.argv[1] if len(sys.argv) > 1 else "1"
for arm in ("q", "qt"):
    m = load(arm, beta)
    tr = m[np.isfinite(m.shadow_fbeta)]
    print(f"== {arm} beta {beta}: same head on the arm's own heads (own line minus raw shadow), by clicks")
    for lo, hi in ((26, 50), (51, 75), (76, 100), (101, 125), (126, 150)):
        b = tr[(tr.t >= lo) & (tr.t <= hi)]
        print(
            f"  {lo:3d}-{hi:3d}: dF {np.mean(b.fbeta - b.shadow_fbeta):+.4f}  kept own-raw mean {np.mean(b.n_flagged - b.shadow_n_flagged):+.1f} "
            f"median {np.median(b.n_flagged - b.shadow_n_flagged):+.0f}  clip_good {np.nanmean(b.xfer_clip_good):.2f}  rows {len(b)}"
        )
c = load("c", beta)
qt = load("qt", beta)


def last(m):
    return m.groupby(["category", "seed"]).tail(1).set_index(["category", "seed"])


lc, lq = last(c), last(qt)
j = lc.join(lq, lsuffix="_c", rsuffix="_qt", how="inner")
print(f"== last step <= 150, beta {beta}, {len(j)} pairs (means; P over non-empty sets)")
for k in ("fbeta", "precision", "recall", "n_flagged"):
    print(
        f"  {k:10s} c {np.nanmean(j[k + '_c']):.4f}  qt {np.nanmean(j[k + '_qt']):.4f}  median c {np.nanmedian(j[k + '_c']):.1f} qt {np.nanmedian(j[k + '_qt']):.1f}"
    )
d = j.fbeta_qt - j.fbeta_c
print(
    f"  dF mean {d.mean():+.4f}; share qt better {np.mean(d > 0):.2f}, worse {np.mean(d < 0):.2f}, equal {np.mean(d == 0):.2f}"
)
big = (
    j.assign(d=d)
    .sort_values("d")
    .head(8)[
        [
            "fbeta_c",
            "fbeta_qt",
            "n_flagged_c",
            "n_flagged_qt",
            "precision_c",
            "precision_qt",
            "recall_c",
            "recall_qt",
            "n_test_pos_c",
        ]
    ]
)
print(big.round(3).to_string())
