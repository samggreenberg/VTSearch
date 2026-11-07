#!/usr/bin/env python
"""#5373 sizing tables (aggregates only) from face_rotation_5373_size.py's per-face rows.

For each subject face, the 0-degree record (which reproduces the cell) and the turned records that
overlap it. A retry rule is (cut, gain): faces under ``cut`` are retried and a turned face replaces
the face when its quality beats it by more than ``gain`` (the highest such). Writes:

* ``cuts.csv``: per size, cut and gain: faces retried, switched, switched faces matching their own
  photos worse (mean cosine) or with own-photo AP down by more than 0.01, and the AP gained in sum;
* ``gain_bins.csv``: every turned record, by its quality gain: how often its cosine to the person's
  own photos rose, and by how much on average (does a quality gain predict a match gain?).

    python face_rotation_5373_tables.py --size 1024=<glob> --size 640=<glob> --out <dir>
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

CUTS = (12, 14, 16, 18, 20, 22, 25)
GAINS = (0, 1, 2, 3, 5)
BINS = [(-99, 0), (0, 2), (2, 5), (5, 99)]


def read(pattern: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(pattern))], ignore_index=True)
    m = df[df.n_own > 0].dropna(subset=["q"])
    base = m[m.rot == 0].set_index("idx")
    assert (base.q - base.q_pile).abs().max() < 1e-3, "the 0-degree record must reproduce the cell"
    nz = m[m.rot != 0].copy()
    nz["q0"], nz["c0"], nz["a0"] = nz.idx.map(base.q), nz.idx.map(base.cos_own), nz.idx.map(base.ap)
    return base, nz


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--size", action="append", required=True, help="<size>=<glob of size rows>")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    cuts, bins = [], []
    for spec in args.size:
        size, pattern = spec.split("=", 1)
        base, nz = read(pattern)
        qmax = float(base.q.max())
        for cut in CUTS:
            if cut > qmax + 1:
                continue
            for gain in GAINS:
                t = nz[(nz.q0 < cut) & (nz.q > nz.q0 + gain)]
                b = t.loc[t.groupby("idx").q.idxmax()]
                dcos, dap = b.cos_own - b.c0, b.ap - b.a0
                cuts.append(
                    {"size": size, "cut": cut, "gain": gain, "faces": len(base),
                     "retried": int((base.q < cut).sum()), "switched": len(b),
                     "worse_cos": int((dcos < 0).sum()), "worse_ap": int((dap < -0.01).sum()),
                     "mean_dcos": round(float(dcos.mean()), 3), "sum_dap": round(float(dap.sum()), 1)}
                )  # fmt: skip
        dq, dc = nz.q - nz.q0, nz.cos_own - nz.c0
        for lo, hi in BINS:
            s = (dq > lo) & (dq <= hi)
            bins.append(
                {"size": size, "gain_from": lo, "gain_to": hi, "records": int(s.sum()),
                 "cos_up_pct": round(100 * float((dc[s] > 0).mean()), 0), "mean_dcos": round(float(dc[s].mean()), 3),
                 "corr_all": round(float(np.corrcoef(dq, dc)[0, 1]), 2)}
            )  # fmt: skip
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cuts).to_csv(args.out / "cuts.csv", index=False)
    pd.DataFrame(bins).to_csv(args.out / "gain_bins.csv", index=False)
    with pd.option_context("display.width", 200):
        print(pd.DataFrame(cuts).to_string(index=False))
        print(pd.DataFrame(bins).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
