#!/usr/bin/env python
"""What a bigger iteration budget costs where it binds - the slow folds (#3839).

    python bench_3839.py --corpus <dir> --fits <analysis dir> --out bench3839.csv

The gate's ``refit_seconds`` is one timing per fold at the fold's own size,
which prices the *mean* honestly and the *tail* poorly - and the tail is the
whole cost of raising the cap: a median fold converges in ~50 iterations and
never meets any cap, while the folds #3839 is about run to 2,000.  So this
takes the folds that were capped under the shipped rule, times the anchored
refit min-of-*reps* under each budget arm at the fold's own size and at a
bootstrap resample to :data:`_GMM_MAX_SAMPLES` (the app's own ceiling on a fit
sample), anchors held fixed.  Resampled rows carry ``resampled=1``: the shape
that drives the iteration count survives a resample, the granularity does not,
so they are projections, not measurements (the #3825 convention).
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402

common.setup_env()

import arms_3839 as A  # noqa: E402
import gate_3825 as G25  # noqa: E402

ARMS = ("shipped", "cap1000", "cap2000")


def main(argv: "list[str] | None" = None) -> int:
    import pandas as pd

    from vtscore.training.thresholds import FOLD_ANCHOR_WEIGHT, fit_anchored_score_gmm, gmm_fit_array, scored_ordering
    from vtscore.training.thresholds.gmm import _GMM_MAX_SAMPLES
    from vtscore.utils.scores import scored_only

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--fits", required=True, help="analysis dir holding gate3839_fits.*.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-folds", type=int, default=60)
    args = ap.parse_args(list(argv) if argv is not None else None)

    fits = pd.concat(pd.read_csv(f, dtype={"case": str}) for f in sorted(glob.glob(f"{args.fits}/gate3839_fits.*.csv")))
    capped = fits[(fits["arm"] == "shipped") & (fits["provenance"] == "anchored") & (fits["converged"] == 0)]
    capped = capped.sample(n=min(args.max_folds, len(capped)), random_state=3839)
    rng = np.random.default_rng(3839)
    rows = []
    for _, r in capped.iterrows():
        z = np.load(Path(args.corpus) / f"{r['cell']}.npz")
        arrays = G25._cases(z)[("fold", r["style"], r["case"])]
        hays, orderings, _final = G25._fold_inputs(arrays)
        a_s, a_l = scored_ordering(orderings[int(r["fold"])])
        base = gmm_fit_array(scored_only(hays[int(r["fold"])]))
        for size in (None, _GMM_MAX_SAMPLES):
            x = base if size is None else rng.choice(base, size=size, replace=True)
            for arm in ARMS:
                best, st = np.inf, {}
                with A.swap_anchored_stop(A.ARMS[arm][0]):
                    for _ in range(args.reps):
                        st = {}
                        t0 = time.perf_counter()
                        fit_anchored_score_gmm(x, a_s, a_l, anchor_weight=FOLD_ANCHOR_WEIGHT, stats=st)
                        best = min(best, time.perf_counter() - t0)
                rows.append(
                    {
                        "cell": r["cell"],
                        "style": r["style"],
                        "case": r["case"],
                        "fold": r["fold"],
                        "dataset": r["dataset"],
                        "embedder": r["embedder"],
                        "n": int(x.size),
                        "resampled": 0 if size is None else 1,
                        "arm": arm,
                        "n_iter": int(st.get("n_iter", 0)),
                        "converged": int(st.get("converged", 0)),
                        "seconds": f"{best:.6f}",
                    }
                )
        z.close()
        print(f"{r['cell']} {r['style']} {r['case']} fold {r['fold']}: done", flush=True)
    with Path(args.out).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
