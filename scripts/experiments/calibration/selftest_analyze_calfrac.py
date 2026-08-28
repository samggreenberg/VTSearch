#!/usr/bin/env python
"""Planted-answer check on `analyze_calfrac.py`, run BEFORE the array.

A sign error in a paired contrast is invisible in the output and expensive in
the input: this study is five full simulation arms, and the first moment anyone
would notice "0.3 wins" actually meaning "0.3 loses" is the report.  So the
analyzer is run here on fabricated cells whose answer is known by construction.

Three things are planted, each of which is a different way the analyzer could be
wrong:

1. **A per-mode split.**  Binary geometries are built so that MORE Calibrate is
   better (0.7 wins); the region geometry so that MORE Train is better (0.3
   wins).  A report that names one winner for the whole study has pooled across
   the mode.
2. **A band reversal.**  Inside the binary geometries the ordering FLIPS between
   the early band and the deep band.  An analyzer that reads only a pooled
   number cannot see it, which is the failure the banded table exists to prevent.
3. **A pick log beside the cells.**  `run_cells.py` writes `task_*__picks.csv`
   unconditionally.  If it reaches the metric frame, the cell counts and every
   mean move.  The check is that they do not.

Run: `python selftest_analyze_calfrac.py`
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FRACTIONS = (0.3, 0.4, 0.5, 0.6, 0.7)
GEOMETRIES = (
    ("siglip", "whole_image"),
    ("siglip+dinov3_patch", "whole_image"),
    ("siglip+dinov3_patch", "max_patch"),
)
CATEGORIES = [f"cls{i}" for i in range(6)]
SEEDS = (0, 1, 2, 3)
STEPS = 150


def planted_cost(style: str, frac: float, t: int, rng: np.random.Generator) -> float:
    """The answer, written down.

    Region (`max_patch`): cost rises with the fraction at every band, so 0.3 is
    the winner everywhere.  Binary (`whole_image`): the sign of the slope flips
    at t=60, so 0.3 wins early and 0.7 wins deep, and no single pooled number is
    a faithful summary of it.
    """
    noise = rng.normal(0.0, 0.004)
    if style == "max_patch":
        return 0.30 + 0.20 * (frac - 0.5) + noise
    slope = +0.20 if t <= 60 else -0.20
    return 0.30 + slope * (frac - 0.5) + noise


def build(base: Path) -> None:
    rng = np.random.default_rng(7)
    for frac in FRACTIONS:
        cells = base / f"f{round(frac * 100):03d}" / "results" / "cells"
        cells.mkdir(parents=True, exist_ok=True)
        idx = 0
        for emb, style in GEOMETRIES:
            for cat in CATEGORIES:
                for seed in SEEDS:
                    rows = []
                    for t in range(1, STEPS + 1):
                        cost = planted_cost(style, frac, t, rng)
                        # `regret_honest` carries the same planted structure, so
                        # a verdict computed on either metric must agree.
                        rows.append(
                            {
                                "seed": seed,
                                "dataset": "vg_scale_any",
                                "category": cat,
                                "style": style,
                                "t": t,
                                "n_good": t // 2,
                                "n_bad": t - t // 2,
                                "gmm_variant": "",
                                "schedule": "",
                                "pool_variant": "",
                                "threshold": 0.5 + 0.1 * (frac - 0.5) + rng.normal(0, 0.01 * frac),
                                "cost": cost,
                                "regret_honest": cost - 0.25,
                                "regret": cost - 0.25,
                                # The trap: these two are built to sum to
                                # `regret` and slide against each other, exactly
                                # as the real ones do.
                                "rule_inefficiency": (cost - 0.25) * 0.5 + 0.3 * (frac - 0.5),
                                "calibration_shift": (cost - 0.25) * 0.5 - 0.3 * (frac - 0.5),
                                "n_cal_scores": max(1, round(t * frac)),
                                "average_precision": 1.0 - cost,
                                "embedder": emb,
                                "seed_mode": "text",
                                "seed_query": cat,
                                "seed_embedder": "siglip",
                                "calibration_fraction": frac,
                            }
                        )
                    pd.DataFrame(rows).to_csv(cells / f"task_{idx:04d}.csv", index=False)
                    # (3) the pick log, beside the metric frame, as a real run
                    # writes it.  If it leaks into the main frame the cell counts
                    # below double and every mean moves.
                    pd.DataFrame(
                        [
                            {"seed": seed, "dataset": "vg_scale_any", "category": cat, "t": t, "picked_id": t}
                            for t in range(1, 6)
                        ]
                    ).to_csv(cells / f"task_{idx:04d}__picks.csv", index=False)
                    idx += 1


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="calfrac-selftest-"))
    try:
        base = tmp / "arms"
        out = tmp / "out"
        build(base)
        # In-process, not a subprocess: the analyzer is a library here, and a
        # planted-answer test that shells out can only ever assert on an exit
        # code and a captured string.
        import analyze_calfrac

        if analyze_calfrac.main(["--base", str(base), "--out", str(out), "--no-figures", "--no-viewer"]) != 0:
            raise SystemExit("analyze_calfrac failed on the fabricated cells")

        failures: list[str] = []

        # (3) the pick log stayed out of the metric frame.
        counts = pd.read_csv(out / "agg" / "cell_counts.csv")
        expected = len(CATEGORIES) * len(SEEDS)
        got = sorted(counts["n_cells"].unique())
        if got != [expected]:
            failures.append(f"cell counts {got} != [{expected}] - a side frame leaked into the main frame")

        paired = pd.read_csv(out / "agg" / "paired_vs_incumbent.csv")
        cost = paired[paired["metric"] == "cost"]

        # (1) the region geometry prefers LESS calibration, at every band.
        reg = cost[cost["mode"] == "region"]
        for band in reg["band"].unique():
            b = reg[reg["band"] == band].sort_values("fraction")
            if not (b[b["fraction"] < 0.5]["delta"] < 0).all():
                failures.append(f"region/{band}: fractions below 0.5 should BEAT 0.5, got {b['delta'].tolist()}")
            if not (b[b["fraction"] > 0.5]["delta"] > 0).all():
                failures.append(f"region/{band}: fractions above 0.5 should LOSE to 0.5")

        # (2) the binary geometries reverse between the early and the deep band.
        bina = cost[cost["mode"] == "binary"]
        early = bina[(bina["band"] == "early 1-25") & (bina["fraction"] == 0.7)]["delta"].mean()
        deep = bina[(bina["band"] == "deep 101-150") & (bina["fraction"] == 0.7)]["delta"].mean()
        if not (early > 0 > deep):
            failures.append(f"binary: planted reversal not recovered (early Δ={early:.3g}, deep Δ={deep:.3g})")

        # ... and the verdict names a different winner per mode, which is the
        # per-mode default the issue is asking about.
        vd = pd.read_csv(out / "agg" / "verdict.csv")
        vdc = vd[vd["metric"] == "cost"]
        reg_c = vdc[(vdc["mode"] == "region") & vdc["candidate"]]["fraction"].tolist()
        if 0.3 not in reg_c:
            failures.append(f"region: 0.3 should be a candidate, candidates were {reg_c}")
        if 0.7 in reg_c:
            failures.append("region: 0.7 must not be a candidate - it is worse at every band")

        # The harm gate must bite: 0.7 harms the binary early band by 0.04,
        # four times HARM_TOLERANCE, so it cannot be a candidate however good
        # its pooled number is.
        bin_c = vdc[(vdc["mode"] == "binary") & (vdc["fraction"] == 0.7)]
        if bin_c.empty or bool(bin_c["candidate"].iloc[0]):
            failures.append("binary: 0.7 harms the early band and must fail the pointwise gate")

        # The trap check must SHOW the anti-correlation and the pinned sum.
        trap = pd.read_csv(out / "agg" / "trap_check.csv")
        if trap.empty or trap["corr_terms_across_arms"].max() > -0.9:
            failures.append("trap_check did not recover the planted anti-correlation")
        if not trap.empty and trap["max_abs_sum_minus_regret"].max() > 1e-9:
            failures.append("trap_check: the two terms should sum to regret exactly")

        # A mislabelled arm must be refused, not analysed.
        bad = tmp / "bad"
        shutil.copytree(base, bad)
        f = next((bad / "f030" / "results" / "cells").glob("task_0000.csv"))
        df = pd.read_csv(f)
        df["calibration_fraction"] = 0.6
        df.to_csv(f, index=False)
        try:
            analyze_calfrac.main(["--base", str(bad), "--out", str(tmp / "out2"), "--no-figures", "--no-viewer"])
        except SystemExit:
            pass  # what it is supposed to do
        else:
            failures.append("a cell stamped with the wrong fraction was analysed instead of refused")

        if failures:
            for x in failures:
                print(f"FAIL: {x}")
            return 1
        print("selftest_analyze_calfrac: OK (per-mode split, band reversal, harm gate, trap, picks, mislabel guard)")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
