"""Planted-answer self-test for ``analyze_acq.py``.

Fabricates arms whose answer is known by construction and asserts the analyzer
recovers it — in particular the three things that would otherwise read as good
news:

* an arm whose acquisition cut never moved must be reported as having measured
  nothing, not as "the lever does nothing";
* a falsification arm that fails to falsify must withhold the verdict;
* the ship rule must reject an arm that buys positives at the cost of a
  regression, and the cost criterion must read the **CI**, not the p-value.

Run: ``python selftest_analyze_acq.py``
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import analyze_acq as A  # noqa: E402

N_CAT, N_SEED, N_STEP = 4, 6, 100

#: (positives at t=100, final cost, acq percentile, deep-spike?) per arm.
#: `acq_m2` is the planted winner: more positives, cost unchanged, no new spikes.
#: `acq_m3` buys positives but regresses cost -> must be REJECTED.
#: `acq_m4` never moved its cut -> must be flagged as having measured nothing.
PLANT = {
    "prod": (4, 0.140, 0.885, False),
    "acq_m1": (6, 0.139, 0.910, False),
    "acq_m2": (9, 0.138, 0.940, False),
    "acq_m3": (11, 0.190, 0.960, True),
    "acq_m4": (4, 0.140, 0.885, False),  # lever stuck
    "acq_p2": (2, 0.145, 0.840, False),  # falsifier: fewer positives
    "rank_pin": (9, 0.138, 0.959, False),
}


def _cell(arm, cat, seed, rng):
    pos, cost_end, acq_pct, deep = PLANT[arm]
    t = np.arange(1, N_STEP + 1)
    cost = 0.30 * np.exp(-t / 20.0) + cost_end + rng.normal(0, 0.004, N_STEP)
    oracle = 0.6 * cost
    if deep:  # a mid-run threshold blip on a healthy ranking
        cost[70] = 0.62
        oracle[70] = 0.05
    n_good = np.clip((t * pos / N_STEP).astype(int), 0, None)
    return pd.DataFrame(
        {
            "seed": seed,
            "dataset": "coco_val",
            "embedder": "siglip2",
            "category": cat,
            "strategy": "autopilot",
            "trainer": "mlp",
            "head": "linear",
            "style": "whole_image",
            "prevalence_arm": "",
            "realized_prevalence": 0.037,
            "t": t,
            "n_good": n_good,
            "n_bad": t - n_good,
            "phase": "hard",
            "app_trained": 1,
            "acq_threshold": 0.2 if arm == "prod" else 0.25,
            "acq_pool_percentile": acq_pct,
            "report_pool_percentile": 0.885,
            "pool_variant": "max",
            "gmm_variant": "",
            "schedule": "",
            "threshold": 0.2,
            "threshold_provenance": "fold_anchored[2/2]",
            "degenerate": 0,
            "xcal_threshold": 0.12,
            "gmm_cut": "",
            "cost": cost,
            "fpr": cost * 0.3,
            "fnr": cost * 0.7,
            "auroc": 0.9,
            "average_precision": 0.5,
            "oracle_threshold": 0.2,
            "oracle_cost": oracle,
            "oracle_fpr": oracle * 0.3,
            "oracle_fnr": oracle * 0.7,
            "regret": cost - oracle,
        }
    )


def build(root: Path):
    for arm in PLANT:
        cells = root / arm / "cells"
        cells.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(abs(hash(arm)) % 2**31)
        i = 0
        for ci in range(N_CAT):
            for seed in range(N_SEED):
                _cell(arm, f"cat{ci}", seed, rng).to_csv(cells / f"task_{i:04d}.csv", index=False)
                i += 1


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="acqselftest-"))
    try:
        build(root)
        df, prov = A.load_all(root)
        traj = A.trajectory_stats(df)
        s = A.build_summary(traj, prov)
        fails = []

        if len(df) != len(PLANT) * N_CAT * N_SEED * N_STEP:
            fails.append(f"row count wrong: {len(df)}")

        # 1. Lever verification.
        if s["lever_verification"]["acq_m4"]["moved"]:
            fails.append("acq_m4's cut never moved but was not flagged")
        if not s["lever_verification"]["acq_m2"]["moved"]:
            fails.append("acq_m2 moved its cut but was flagged as stuck")

        # 2. Falsifier.
        if not s["falsifier_behaved"]:
            fails.append("acq_p2 was planted with FEWER positives but did not register as falsifying")

        # 3. Ship rule.
        adopt = set(s["adopt"])
        if "acq_m2" not in adopt:
            fails.append(f"planted winner acq_m2 not adopted (ship={s['ship_rule'].get('acq_m2')})")
        if "acq_m3" in adopt:
            fails.append("acq_m3 regresses cost (+0.05) but was adopted")
        if "acq_m4" in adopt:
            fails.append("acq_m4's lever never moved but was adopted")

        # 4. The cost criterion must be an interval, not a point.
        c = s["contrasts_vs_control"]["acq_m3"]["final_cost"]
        if not (c["ci95_lo"] < c["ci95_hi"]) or c["ci95_hi"] <= 0:
            fails.append(f"cost CI degenerate or wrong side for a planted regression: {c}")

        # 5. Direction sanity: positives must rise monotonically m1 -> m2.
        pa = s["per_arm"]
        if not pa["acq_m1"]["median_positives_100"] < pa["acq_m2"]["median_positives_100"]:
            fails.append("planted positive ordering lost")

        # 6. Withheld verdict propagates to the report.
        s_bad = dict(s)
        s_bad["falsifier_behaved"] = False
        out = Path(tempfile.mkdtemp(prefix="acqrep-"))
        rep = A.write_report(s_bad, [], out).read_text()
        if "VERDICT WITHHELD" not in rep:
            fails.append("report does not withhold the verdict when the falsifier fails")
        if "acq_m4" not in rep:
            fails.append("report omits the stuck-lever arm")
        shutil.rmtree(out, ignore_errors=True)

        if fails:
            print("SELFTEST FAILED:")
            for f in fails:
                print("  -", f)
            return 1
        print(f"selftest OK: {len(df)} rows, {len(PLANT)} arms; planted winner={s['adopt']}")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
