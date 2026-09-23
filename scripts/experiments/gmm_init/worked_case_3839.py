#!/usr/bin/env python
"""Print one captured fold case end to end under several #3839 arms - the report's worked case.

    python worked_case_3839.py --corpus <dir> --cell cell_0080 --style whole_image --case 75

For each arm: every fold's refit (iterations, converged, weights, means,
midpoint, anchored objective) and the shipped chain's threshold and admitted
count.  Then the fold that moves most, traced through its iterations, so the
report can show *what* the loop was still doing when the cap stopped it.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402

common.setup_env()

import arms_3839 as A  # noqa: E402
import gate_3825 as G25  # noqa: E402

ARMS = ("shipped", "cap1000", "cap2000", "limit", "stall1e-4", "fallback")
CHECKPOINTS = (10, 50, 100, 200, 400, 1000, 2000, 5000)


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--style", required=True)
    ap.add_argument("--case", required=True)
    args = ap.parse_args(list(argv) if argv is not None else None)

    from vtscore.training.thresholds import (
        FOLD_ANCHOR_WEIGHT,
        fit_anchored_score_gmm,
        fit_fold_anchored_cut,
        fit_score_gmm,
        gmm_fit_array,
        scored_ordering,
    )
    from vtscore.utils.scores import scored_only

    z = np.load(Path(args.corpus) / f"{args.cell}.npz")
    print(G25._meta(z))
    arrays = G25._cases(z)[("fold", args.style, args.case)]
    hays, orderings, final = G25._fold_inputs(arrays)
    print(f"final haystack {final.size}; folds {len(hays)}; anchors per fold {[len(o[0]) for o in orderings]}")

    for arm in ARMS:
        rule, _ = A.ARMS[arm]
        with A.swap_anchored_stop(rule):
            cut = fit_fold_anchored_cut(hays, orderings, final.tolist())
            thr = cut.threshold_at(0)
            adm = int(np.count_nonzero(cut.final_haystack >= thr))
            print(f"\n[{arm}] threshold {thr:.6f} admits {adm}/{cut.final_haystack.size}  provenance {cut.provenance}")
            for i, hay in enumerate(hays):
                a_s, a_l = scored_ordering(orderings[i])
                arr = gmm_fit_array(scored_only(hay))
                st: dict[str, float] = {}
                fit, prov = fit_anchored_score_gmm(arr, a_s, a_l, anchor_weight=FOLD_ANCHOR_WEIGHT, stats=st)
                if fit is None:
                    print(f"  fold {i}: {prov} -> unanchored fallback")
                    continue
                obj = A.objective_value(fit, arr, a_s, a_l, True, FOLD_ANCHOR_WEIGHT)
                q = float(np.mean(np.sort(arr) < fit.midpoint()))
                print(
                    f"  fold {i}: iters {int(st.get('n_iter', 0))} converged {int(st.get('converged', 0))} "
                    f"w_hi {fit.w_hi:.4f} mu_lo {fit.mu_lo:.4f} mu_hi {fit.mu_hi:.4f} "
                    f"midpoint {fit.midpoint():.4f} (fold quantile {q:.4f}) objective {obj:.7f}"
                )

    # Trace the fold whose midpoint differs most between shipped and limit.
    best, gap = 0, -1.0
    for i, hay in enumerate(hays):
        a_s, a_l = scored_ordering(orderings[i])
        arr = gmm_fit_array(scored_only(hay))
        mids = []
        for arm in ("shipped", "limit"):
            with A.swap_anchored_stop(A.ARMS[arm][0]):
                f, _ = fit_anchored_score_gmm(arr, a_s, a_l, anchor_weight=FOLD_ANCHOR_WEIGHT)
            mids.append(np.nan if f is None else f.midpoint())
        if abs(mids[0] - mids[1]) > gap:
            best, gap = i, abs(mids[0] - mids[1])
    a_s, a_l = scored_ordering(orderings[best])
    arr = gmm_fit_array(scored_only(hays[best]))
    a = np.asarray(a_s, dtype=np.float64)
    lab = np.asarray(a_l, dtype=np.float64)
    cur = fit_score_gmm(arr)
    print(f"\ntrace of fold {best} (|Δ midpoint| {gap:.4f}), n={arr.size}, anchors {a.size} ({int(lab.sum())} Good)")
    prev = None
    for k in range(1, max(CHECKPOINTS) + 1):
        st: dict[str, float] = {}
        nxt = A._REAL(arr, a[lab != 1.0], a[lab == 1.0], cur, FOLD_ANCHOR_WEIGHT, 1, 1e-30, 1e30, st, True)
        d = None if prev is None else st["loglik"] - prev
        prev = st["loglik"]
        cur = nxt
        if k in CHECKPOINTS:
            print(
                f"  iter {k:5d}: objective {st['loglik']:.8f}  last step {d:.2e}  w_hi {cur.w_hi:.4f} "
                f"mu_lo {cur.mu_lo:.4f} mu_hi {cur.mu_hi:.4f} midpoint {cur.midpoint():.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
