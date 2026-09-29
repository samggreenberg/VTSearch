#!/usr/bin/env python3
"""Planted-answer self-test for ``analyze_drystop_4222.py``.

Plants two pools with four arms and known answers:

* ``g3`` (control): a 7-vote opening; its ``@small`` hunts find 1
  positive (starved), the rest 5; AP 0.50;
* ``g6``: meets G at vote 12; nothing starves; AP +0.03 in both pools;
* ``g20d8``: runs dry at vote 30 with 4 Goods; nothing starves; AP +0.06 in
  both pools - the arm the rule must pick;
* ``g20``: never hands over; AP +0.10 in one pool but -0.10 in the other - the
  quality guard must reject it.

Checks: how each opening ended and its length, the starved share per band and
its paired delta, the paired AP delta, the guard, and the rule's choice.

    python selftest_analyze_drystop_4222.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analyze_drystop_4222 as A  # noqa: E402

CELLS = [(f"cls{i}@{band}", seed) for i, band in enumerate(["large", "medium", "small"]) for seed in range(4)]
#: arm -> (opening votes, Goods at handover or None if it never hands over, AP delta per pool)
PLAN = {
    "g3": (7, 3, {"0.44%": 0.0, "0.1%": 0.0}),
    "g6": (12, 6, {"0.44%": 0.03, "0.1%": 0.03}),
    "g20d8": (30, 4, {"0.44%": 0.06, "0.1%": 0.06}),
    "g20": (150, None, {"0.44%": 0.10, "0.1%": -0.10}),
}


def build_arm(root: Path, pool: str, label: str) -> Path:
    opening, goods, dap = PLAN[label]
    results = root / pool.replace("%", "pct") / label / "results"
    cells = results / "cells"
    cells.mkdir(parents=True)
    for i, (cat, seed) in enumerate(CELLS):
        t = np.arange(1, 151)
        label_col = np.zeros(150, dtype=int)
        small = cat.endswith("@small")
        found = 1 if (small and label == "g3") else max(5, goods or 0)
        if goods is None:
            label_col[:found] = 1
        else:
            label_col[: min(goods, found)] = 1
            label_col[opening : opening + max(found - goods, 0)] = 1
        phase = np.where(t <= opening, "s0", "hard") if goods is not None else np.full(150, "s0")
        pd.DataFrame({"category": cat, "seed": seed, "t": t, "phase": phase, "picked_label": label_col}).to_csv(
            cells / f"task_{i:04d}__picks.csv", index=False
        )
        noise = 0.001 * ((i * 7) % 5)  # a paired SE above zero, identical across arms
        pd.DataFrame(
            {
                "dataset": "planted",
                "embedder": "e",
                "category": cat,
                "seed": seed,
                "t": list(A.CHECKPOINTS),
                "average_precision": 0.5 + dap[pool] + noise + (0.002 * (i % 2) if label != "g3" else 0.0),
                "oracle_cost": 0.2,
                "cost": 0.3,
                "app_trained": 1,
                "gmm_variant": "",
                "schedule": "",
                "pool_variant": "max",
            }
        ).to_csv(cells / f"task_{i:04d}.csv", index=False)
    return results


def check(cond: bool, what: str, failures: list[str]) -> None:
    print(("ok    " if cond else "FAIL  ") + what)
    if not cond:
        failures.append(what)


def main() -> int:
    failures: list[str] = []
    root = Path(tempfile.mkdtemp(prefix="selftest-drystop-4222-"))
    specs = [f"{pool}/{label}={build_arm(root, pool, label)}" for pool in ("0.44%", "0.1%") for label in PLAN]
    out = root / "out"
    rc = A.main([x for s in specs for x in ("--arm", s)] + ["--out", str(out)])
    check(rc == 0, "the analyzer runs on planted arms", failures)

    o = pd.read_csv(out / "opening.csv").set_index(["pool", "arm"])
    check(o.loc[("0.44%", "g20d8"), "ran_dry"] == 1.0, "a dry-stop opening that ended short of G ran dry", failures)
    check(o.loc[("0.44%", "g6"), "met_G"] == 1.0, "a fixed opening that reached G met it", failures)
    check(o.loc[("0.1%", "g20"), "never_handed_over"] == 1.0, "an opening that never ends is 'never'", failures)
    check(o.loc[("0.44%", "g20d8"), "median_opening_votes"] == 30, "opening length read back", failures)

    s = pd.read_csv(out / "starved.csv").set_index(["pool", "arm", "band"])
    check(s.loc[("0.44%", "g3", "small"), "starved"] == 1.0, "control's small hunts starve", failures)
    check(s.loc[("0.44%", "g3", "large"), "starved"] == 0.0, "control's large hunts do not", failures)
    d = s.loc[("0.44%", "g20d8", "small"), "delta_vs_g3"]
    check(abs(d - (-1.0)) < 1e-9, f"starved share paired against the control ({d:+.2f})", failures)

    q = pd.read_csv(out / "quality_paired.csv")
    row = q[(q.pool == "0.44%") & (q.arm == "g20d8") & (q.band == "all") & (q.t == 150)]
    dap = row[row.metric == "average_precision"]["delta_vs_g3"].iloc[0]
    check(abs(dap - 0.061) < 1e-9, f"paired AP delta read back ({dap:+.3f})", failures)

    v = json.loads((out / "verdict.json").read_text())
    check(v["arms"]["g20"]["safe"] is False, "the guard rejects an arm that loses AP in one pool", failures)
    check(v["winner"] == "g20d8", f"the rule picks g20d8 (got {v['winner']})", failures)

    print()
    print("SELFTEST " + ("PASSED" if not failures else f"FAILED ({len(failures)})"))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
