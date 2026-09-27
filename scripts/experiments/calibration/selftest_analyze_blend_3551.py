#!/usr/bin/env python3
"""Planted-answer self-test for ``analyze_blend_3551.py``.

Builds a synthetic screen frame with known effects and checks the analyzer
recovers them:

* a Q2 (replacement) effect of -0.02 on every step for one schedule;
* a Q1 (fallback) effect of -0.10 on fallback steps only for another, with a
  fallback share of exactly 1/4, so its diluted contrast must be -0.025 and
  its Q2 contrast must equal the diluted one;
* a schedule that wins at 1:1 but loses at ``fpr x4`` (a lower cut), which the
  promotion rule must refuse;
* a planted threshold mismatch on one fallback step, which must fail the gate.

    python selftest_analyze_blend_3551.py
"""

from __future__ import annotations

import sys

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analyze_blend_3551 as A  # noqa: E402

ENVS = [("visual_genome_m", "siglip+dinov3_patch"), ("coco_val", "siglip")]
SCHEDS = ["cap50", "slow_cap50", "rare:lo=1:hi=16", "corridor:w=0.2", "pure_gmm"]


def build(mismatch: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for ds, emb in ENVS:
        mode = A.mode_of(emb)
        shipped = A.SHIPPED[mode]
        for cat in ("a", "b", "c", "d", "e", "f"):
            for seed in range(3):
                for draw in (42, 0):
                    for t in range(4, 44):  # 40 steps, the first 10 on the fallback
                        fb = t < 14
                        fpr0, fnr0 = rng.uniform(0.1, 0.3), rng.uniform(0.2, 0.4)
                        thr0 = rng.uniform(0.3, 0.7)
                        common_cols = {
                            "dataset": ds,
                            "embedder": emb,
                            "category": cat,
                            "seed": seed,
                            "calibration_seed": draw,
                            "t": t,
                            "n_good": 1 if fb else 5,
                            "n_bad": t - (1 if fb else 5),
                            "gmm_variant": "",
                            "pool_variant": "max",
                            "seed_mode": "text",
                        }
                        prov = "gmm_blend" if fb else "fold_anchored[2/2]"
                        rows.append(
                            {
                                **common_cols,
                                "schedule": "",
                                "threshold": thr0,
                                "threshold_provenance": prov,
                                "cost": fpr0 + fnr0,
                                "fpr": fpr0,
                                "fnr": fnr0,
                            }
                        )
                        for s in SCHEDS:
                            fpr, fnr, thr = fpr0 + rng.normal(0, 0.01), fnr0 + rng.normal(0, 0.01), thr0 + 0.01
                            if s == shipped and fb:
                                fpr, fnr, thr = fpr0, fnr0, thr0
                            if s == "rare:lo=1:hi=16":
                                fpr, fnr = fpr0 - 0.01, fnr0 - 0.01  # Q2 -0.02 everywhere
                            if s == "corridor:w=0.2":
                                fpr, fnr = (fpr0 - 0.05, fnr0 - 0.05) if fb else (fpr0, fnr0)
                            if s == "pure_gmm":
                                fpr, fnr = fpr0 + 0.02, fnr0 - 0.05  # wins 1:1 (-0.03), loses fpr x4 (+0.03)
                            rows.append(
                                {
                                    **common_cols,
                                    "schedule": s,
                                    "threshold": thr,
                                    "shipped_provenance": prov,
                                    "fold_fallback": "0.5" if fb else "",
                                    "cost": fpr + fnr,
                                    "fpr": fpr,
                                    "fnr": fnr,
                                }
                            )
    df = pd.DataFrame(rows)
    if mismatch:
        # On the binary environment, whose shipped fallback schedule is cap50.
        idx = df[(df["schedule"] == "cap50") & (df["t"] == 5) & (df["embedder"] == "siglip")].index[0]
        df.loc[idx, "threshold"] += 1e-3
    return df


def check(cond: bool, what: str, failures: list[str]) -> None:
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        failures.append(what)


def main() -> int:
    failures: list[str] = []
    m = A.paired(build())
    fid = A.fidelity(m)
    check(fid["passed"], f"fidelity passes on a faithful frame ({fid})", failures)
    check(not A.fidelity(A.paired(build(mismatch=True)))["passed"], "a planted mismatch fails the gate", failures)

    cells = A.cell_contrasts(m)
    q2 = A.summarise(cells, "q2_1:1")
    r = q2[q2["schedule"] == "rare:lo=1:hi=16"]["mean"]
    check(np.allclose(r, -0.02), f"Q2 planted -0.02 recovered ({r.round(4).tolist()})", failures)

    q1c = A.summarise(cells, "q1c_1:1")
    q1d = A.summarise(cells, "q1d_1:1")
    c = q1c[q1c["schedule"] == "corridor:w=0.2"]["mean"]
    d = q1d[q1d["schedule"] == "corridor:w=0.2"]["mean"]
    check(np.allclose(c, -0.10), f"Q1 conditional -0.10 recovered ({c.round(4).tolist()})", failures)
    check(np.allclose(d, -0.025), f"Q1 diluted = share x conditional = -0.025 ({d.round(4).tolist()})", failures)
    q2c = q2[q2["schedule"] == "corridor:w=0.2"]["mean"]
    check(np.allclose(q2c, d.to_numpy()), "a fallback-only effect has Q2 == Q1 diluted", failures)

    cen = A.census(m)
    check(np.allclose(cen["fallback_share"], 0.25), f"fallback share 1/4 ({cen['fallback_share'].tolist()})", failures)

    p2 = A.promote(cells, "q2")
    pg = p2[p2["schedule"] == "pure_gmm"]
    check(bool(len(pg)) and not pg["promoted"].iloc[0], "a win that reverses at fpr x4 is not promoted", failures)
    check(bool((pg["wins_every_env"]).all()), "... although it does win at 1:1", failures)
    rr = p2[p2["schedule"] == "rare:lo=1:hi=16"]
    check(bool(len(rr)) and bool(rr["promoted"].all()), "the planted Q2 winner is promoted in both modes", failures)
    p1 = A.promote(cells, "q1c")
    cc = p1[p1["schedule"] == "corridor:w=0.2"]
    check(
        bool(len(cc)) and bool(cc["promoted"].all()), "the planted Q1 winner is promoted on the conditional", failures
    )
    shipped_listed = any(
        ((p["mode"] == mode) & (p["schedule"] == s)).any() for p in (p1, p2) for mode, s in A.SHIPPED.items()
    )
    check(not shipped_listed, "the shipped schedule is never its own candidate", failures)

    band = A.cell_contrasts(m, ("x", 4, 13), "t")
    b = A.summarise(band, "q1d_1:1")
    bm = b[b["schedule"] == "corridor:w=0.2"]["mean"]
    check(
        np.allclose(bm, -0.10), f"inside the fallback band, diluted == conditional ({bm.round(4).tolist()})", failures
    )

    print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
