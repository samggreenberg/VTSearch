#!/usr/bin/env python3
"""Planted-answer self-test for ``analyze_calsplit_4583``.

Builds four arms per beta whose precision is planted (recall held at 1), so the
objective of every run is known in closed form, and checks that the analyzer:

* recovers each contrast's planted Δ (split, count, both, and the interaction);
* reads "after the check" off the last check row, not the vote-150 row;
* fills a starved run with the typed query's line only when the arm is complete,
  and leaves a missing run out otherwise (a fill would plant Δ = 0).

    python selftest_analyze_calsplit_4583.py
"""

from __future__ import annotations

import sys

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analyze_calsplit_4583 as A  # noqa: E402

CATS = [f"c{i}@{b}" for i, b in enumerate(["small", "medium", "large"] * 4)]
SEEDS = range(3)
#: Planted precision per arm stem; recall is 1, so F-beta = (1+b2) p / (b2 p + 1).
PLANT = {"f03k2": 0.40, "f05k2": 0.50, "f03k4": 0.45, "f05k4": 0.55, "f03k1": 0.35}
CHECK_P = 0.9


def fb(p: float, beta: float) -> float:
    b2 = beta * beta
    return (1 + b2) * p / (b2 * p + 1)


def frame(stem: str, beta: float, drop: tuple[str, int] | None = None) -> pd.DataFrame:
    rows = []
    for c in CATS:
        for s in SEEDS:
            if drop == (c, s):
                continue
            for t in range(10, A.HORIZON + 1):
                rows.append(
                    {
                        "category": c,
                        "seed": s,
                        "t": t,
                        "phase": "hard",
                        "app_trained": 1,
                        "precision": PLANT[stem],
                        "recall": 1.0,
                        "average_precision": 0.7,
                        "n_flagged": 40.0,
                    }
                )
            rows.append(
                {
                    "category": c,
                    "seed": s,
                    "t": A.HORIZON + 5,
                    "phase": "check",
                    "app_trained": 1,
                    "precision": CHECK_P,
                    "recall": 1.0,
                    "average_precision": 0.7,
                    "n_flagged": 20.0,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    ok = True
    anchor = pd.Series(0.01, index=pd.MultiIndex.from_product([CATS, SEEDS], names=A.RUN))

    def check(label: str, cond: bool) -> None:
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")
        ok &= bool(cond)

    for btag, beta in A.BETAS.items():
        tables = {f"{stem}_{btag}": A.arm_tables(frame(stem, beta), anchor, beta, True) for stem in PLANT}
        paired, sigma = A.contrasts(tables)
        at150 = paired[(paired.read == "objective, vote 150") & (paired.stratum == "all")].set_index("contrast")
        for name, (a, b) in A.CONTRASTS.items():
            want = fb(PLANT[a], beta) - fb(PLANT[b], beta)
            check(f"beta {beta:g}: {name} recovers {want:+.4f}", abs(at150.loc[name, "delta"] - want) < 1e-9)
        # Precision is planted additively, F-beta is not linear in it: the
        # interaction is the closed-form difference of differences, not zero.
        inter = (fb(PLANT["f05k4"], beta) - fb(PLANT["f05k2"], beta)) - (
            fb(PLANT["f03k4"], beta) - fb(PLANT["f03k2"], beta)
        )
        check(
            f"beta {beta:g}: interaction recovers {inter:+.4f}", abs(at150.loc["interaction", "delta"] - inter) < 1e-9
        )
        after = paired[(paired.read == "objective, after the check") & (paired.stratum == "all")]
        check(f"beta {beta:g}: after-the-check reads the check row (Δ 0)", np.allclose(after["delta"], 0.0))
        check(f"beta {beta:g}: a planted constant has zero per-run SD", np.allclose(sigma["sd_paired"], 0.0))

    # Starved vs missing: one run absent from the control arm.
    gone = (CATS[0], 0)
    t_inc = A.arm_tables(frame("f03k2", 1.0, drop=gone), anchor, 1.0, complete=False)
    check("an incomplete arm leaves a missing run out", gone not in t_inc["F"].index)
    t_full = A.arm_tables(frame("f03k2", 1.0, drop=gone), anchor, 1.0, complete=True)
    check(
        "a complete arm fills a starved run with the typed query's line",
        gone in t_full["F"].index and np.allclose(t_full["F"].loc[gone].to_numpy(), 0.01),
    )
    print("ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
