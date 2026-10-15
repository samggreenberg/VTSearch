"""#4910 (fold_transfer_4910.sh's runs): is qt's beta-1/4 gain broad? Per-session mean F over votes 1-150 (ffill, rows only, opening identical in
both arms so the text baseline cancels), qt - c by seed, by band, and the share of sessions better/worse."""

import glob
import sys
from multiprocessing import Pool
import numpy as np
import pandas as pd

S = "/expscratch/sgreenberg/state-of-the-app"


def read(f):
    m = pd.read_csv(
        f,
        usecols=[
            "category",
            "seed",
            "embedder",
            "t",
            "phase",
            "fbeta",
            "app_trained",
            "gmm_variant",
            "pool_variant",
            "n_flagged",
        ],
        low_memory=False,
    )
    m = m[
        (m.embedder == "siglip")
        & (m.pool_variant.fillna("max") == "max")
        & (m.gmm_variant.isna() | (m.gmm_variant.astype(str) == ""))
    ]
    m = (
        m[(m.phase.astype(str) != "check") & (m.t <= 150)]
        .sort_values("t")
        .drop_duplicates(["category", "seed", "t"], keep="last")
    )
    out = []
    for (c, s), g in m.groupby(["category", "seed"]):
        f = np.full(151, np.nan)
        f[g.t.to_numpy(int)] = g.fbeta.to_numpy(float)
        f = pd.Series(f).ffill().fillna(0).to_numpy()
        on = g.loc[g.app_trained == 1, "t"]
        shown = on.min() if len(on) else 151
        f[: int(shown)] = 0.0  # identical opening in both arms: cancels in the difference
        out.append((c, int(s), f[1:151].mean(), f[51:151].mean()))
    return out


def load(arm, b):
    fs = sorted(
        x for x in glob.glob(f"{S}/2026-10-15-xfer4910-{arm}-binary-b{b}/results/cells/task_*.csv.gz") if "__" not in x
    )
    with Pool(12) as p:
        rows = [r for part in p.map(read, fs) for r in part]
    return pd.DataFrame(rows, columns=["category", "seed", "m150", "m51"]).set_index(["category", "seed"])


for b in sys.argv[1:]:
    c, qt = load("c", b), load("qt", b)
    d = (qt - c).dropna().reset_index()
    d["cls"] = d.category.str.split("@").str[0]
    d["band"] = d.category.str.split("@").str[1]

    def se(x, g):
        by = x.groupby(g).mean()
        return by.std(ddof=1) / np.sqrt(by.size)

    print(
        f"== beta {b}: qt - c, {len(d)} pairs: 1-150 {d.m150.mean():+.4f} (SE {se(d.m150, d.cls):.4f}); 51-150 {d.m51.mean():+.4f}"
    )
    print("   by seed :", "  ".join(f"{s}:{v:+.4f}" for s, v in d.groupby("seed").m150.mean().items()))
    print("   by band :", "  ".join(f"{s}:{v:+.4f}" for s, v in d.groupby("band").m150.mean().items()))
    print(
        f"   sessions better {np.mean(d.m150 > 1e-9):.2f} worse {np.mean(d.m150 < -1e-9):.2f} same {np.mean(np.abs(d.m150) <= 1e-9):.2f}"
    )
    byc = d.groupby("cls").m150.mean()
    print(
        f"   classes better {np.mean(byc > 0):.2f} of {len(byc)}; trimmed (drop top/bottom 5% sessions) {d.m150.sort_values().iloc[int(0.05 * len(d)) : int(0.95 * len(d))].mean():+.4f}"
    )
