#!/usr/bin/env python3
"""Planted-answer self-test for ``analyze_textgood_4222.py``.

Plants three arms of one world with known answers and checks the analyzer
reads them back:

* ``G = 3`` (control): 3 Good votes by vote 25, a 7-vote opening, few
  calibration positives, AP 0.50;
* ``G = 10``: 10 Good votes by vote 25, a 20-vote opening, calibration
  positives past the gate, AP 0.50 - the arm the rule must pick;
* ``G = 20``: the same gate share as ``G = 10`` but AP 0.40 - it must be
  rejected by the quality guard, and cannot win the tie on size anyway.

Checks: Good votes and "reached G" per checkpoint, the opening length, the
share past the gate, the paired AP delta against the control, and the
decision rule's choice.

    python selftest_analyze_textgood_4222.py
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

import analyze_textgood_4222 as A  # noqa: E402
import selftest_analyze_pframes_4220 as F  # noqa: E402

CELLS = [("cls0@large", 0), ("cls1@medium", 0), ("cls2@small", 1)]
PLAN = {
    3: (3, 7, 8, 0.50),  # 8 votes cannot hold 10 calibration positives
    10: (10, 20, 300, 0.50),
    20: (10, 20, 300, 0.40),
}  # G: (goods by 25, opening, votes in frames, AP)


def build_arm(root: Path, g: int) -> None:
    goods_by_25, opening, n_votes, ap = PLAN[g]
    cells = root / f"natural-g{g}" / "results" / "cells"
    cells.mkdir(parents=True)
    for i, (cat, seed) in enumerate(CELLS):
        # Picks: `goods_by_25` Goods in the first 25 votes, none after; the
        # learned phases begin right after the opening.
        t = np.arange(1, 151)
        label = np.zeros(150, dtype=int)
        label[:goods_by_25] = 1
        phase = np.where(t <= opening, "s0", "hard")
        pd.DataFrame({"category": cat, "seed": seed, "t": t, "phase": phase, "picked_label": label}).to_csv(
            cells / f"task_{i:04d}__picks.csv", index=False
        )
        # Main frame: one base row per checkpoint, AP planted.
        pd.DataFrame(
            {
                "dataset": "planted",
                "embedder": "e",
                "category": cat,
                "seed": seed,
                "t": list(A.CHECKPOINTS),
                "average_precision": ap,
                "oracle_cost": 0.2,
                "cost": 0.3,
                "app_trained": 1,
                "gmm_variant": "",
                "schedule": "",
                "pool_variant": "max",
            }
        ).to_csv(cells / f"task_{i:04d}.csv", index=False)
        # Frames: the #4220 selftest's planted cell; many votes put the
        # calibration positives past the gate, few keep them under it.
        tmp = cells / f"_f{i}"
        tmp.mkdir()
        F.build_cell(tmp, i, seed=100 * g + i, pool_prev=0.05, test_prev=0.05, n_votes=n_votes)
        z = np.load(tmp / f"task_{i:04d}__pframes.npz")
        # Re-keyed to vote 50, and the frame's own `t` field with it - the
        # analyzer reads the field, not the key.
        packed = {k.replace("t150/", "t50/"): z[k] for k in z.files}
        packed["t50/t"] = np.int32(50)
        np.savez_compressed(cells / f"task_{i:04d}__pframes.npz", **packed)
        for p in tmp.iterdir():
            p.unlink()
        tmp.rmdir()


def check(cond: bool, what: str, failures: list[str]) -> None:
    print(("ok    " if cond else "FAIL  ") + what)
    if not cond:
        failures.append(what)


def main() -> int:
    failures: list[str] = []
    root = Path(tempfile.mkdtemp(prefix="selftest-textgood-4222-"))
    for g in PLAN:
        build_arm(root, g)
    out = root / "out"
    rc = A.main(["--base", str(root), "--out", str(out)])
    check(rc == 0, "the analyzer runs on planted arms", failures)

    h = pd.read_csv(out / "harvest.csv").set_index(["G", "t"])
    check(
        h.loc[(10, 25), "mean_goods"] == 10 and h.loc[(3, 25), "mean_goods"] == 3,
        "Good votes by vote 25 read back",
        failures,
    )
    check(
        h.loc[(10, 25), "reached_G"] == 1.0 and h.loc[(20, 150), "reached_G"] == 0.0,
        "reached-G is G-relative",
        failures,
    )
    check(
        h.loc[(10, 150), "median_opening_votes"] == 20 and h.loc[(3, 150), "median_opening_votes"] == 7,
        "opening length read back",
        failures,
    )

    tg = pd.read_csv(out / "trust_gate.csv").set_index(["G", "t"])
    check(
        tg.loc[(10, 50), "past_gate"] == 1.0 and tg.loc[(3, 50), "past_gate"] == 0.0,
        "gate share separates the arms",
        failures,
    )

    q = pd.read_csv(out / "quality_paired.csv")
    d20 = q[(q.G == 20) & (q.t == 150) & (q.metric == "average_precision")]["delta_vs_G3"].iloc[0]
    check(abs(d20 - (-0.10)) < 1e-9, f"paired AP delta vs control read back ({d20:+.2f})", failures)

    v = json.loads((out / "verdict.json").read_text())["natural"]
    check(
        v["ap_safe"]["20"] is False and v["ap_safe"]["10"] is True,
        "the quality guard rejects the AP-costly arm",
        failures,
    )
    check(v["smallest_G_at_max"] == 10, f"the rule picks G=10 (got {v['smallest_G_at_max']})", failures)

    print()
    print("SELFTEST " + ("PASSED" if not failures else f"FAILED ({len(failures)})"))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
