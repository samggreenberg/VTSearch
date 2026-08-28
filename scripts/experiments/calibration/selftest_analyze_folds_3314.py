#!/usr/bin/env python
"""Planted-answer check on `analyze_folds_3314.py`, run BEFORE the array.

The screen is 96 cells of simulation and its verdict is a mechanical read of
three pre-registered rules.  A sign error in the paired contrast, a cost ratio
built on the wrong column, or a harm gate that never bites are all invisible in
the output and only discoverable in the report - by which point the grid has
been paid for.  So the analyzer is run here on fabricated cells whose answer is
known by construction.

Six things are planted, each a different way this analyzer could be wrong:

1. **A geometry split.**  The region geometry is built so more folds help; the
   two single-vector ones are built K-invariant (the laptop bench's prediction).
   A verdict that names one K for the whole study has pooled across geometry.
2. **A decay inside the horizon.**  The region benefit is real in the two early
   bands and exactly zero in the two deep ones - the shape the adaptive
   schedule exists to exploit.  A pooled read cannot see it.
3. **The cost ceiling bites.**  K=8's per-step wall clock is planted at 1.9x
   production's, above the 1.5x ceiling, while K<=6 sits under it.  K=8 has the
   LARGEST benefit, so an analyzer that forgets rule 3 will pick it.
4. **The harm gate bites.**  K=1 is planted worse than K=2 everywhere by more
   than `HARM_TOLERANCE`, so it must fail rule 2 whatever else it does.
5. **The cost model is `cal_seconds`, not `fold_seconds`.**  The two are
   planted to disagree: `fold_seconds` alone would put every K under the
   ceiling and wave K=8 through.  A run built WITHOUT `cal_seconds` must fall
   back loudly rather than silently price K at a third of its cost.
6. **The arm split.**  The pooled `xcal` rows carry the OPPOSITE sign - #2897's
   monotone worsening - so an analyzer reading the wrong arm reverses the
   verdict.  A pick log sits beside the cells, as a real run writes it, and
   must not reach the metric frame.

Run: `python selftest_analyze_folds_3314.py`
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

KS = (1, 2, 3, 4, 6, 8)
GEOMETRIES = (
    ("siglip", "whole_image"),
    ("siglip+dinov3_patch", "whole_image"),
    ("siglip+dinov3_patch", "max_patch"),
)
REGION = "dinov3_patch/max_patch"
CATEGORIES = [f"cls{i}" for i in range(6)]
SEEDS = (0, 1, 2, 3)
STEPS = 150

#: Planted per-step wall clock at K, as a multiple of K=2's.  K=8 is over the
#: 1.5x ceiling on purpose and has the biggest benefit, so rule 3 is the only
#: thing that can reject it.
STEP_RATIO = {1: 0.9, 2: 1.0, 3: 1.15, 4: 1.3, 6: 1.45, 8: 1.9}
#: A production step's wall clock, split so the ratios above come out EXACTLY.
#: The non-calibration part is fixed across K, as it is in a real screen: the
#: trajectory stays at `calibrate_count=2`, so the model fit, the pool scoring
#: and the test scoring are the same work at every K.  Only calibration moves,
#: which is the whole reason a cost ceiling can be written on the step at all.
STEP_SECONDS_K2 = 10.0
OTHER_SECONDS = 8.0


def planted_level(geometry: str, k: int, votes: int) -> float:
    """Cost offset at fold count *k*, before differencing against production.

    Note the level, not the contrast: the analyzer differences against K=2, so
    the delta a table reports is ``planted_level(k) - planted_level(2)``.
    Writing the fixture as a level rather than as a delta is what keeps the two
    from silently meaning the same thing - the first version of this file
    planted deltas and then asserted them as if K=2 sat at zero, which made a
    correct analyzer look wrong at K=3.
    """
    if geometry != REGION:
        return 0.0  # K-invariant on a single-vector geometry
    if k == 1:
        return 0.03  # strictly worse than 2, everywhere: the harm gate's target
    if votes > 60:
        return 0.0  # the benefit has decayed out of the deep bands
    # Saturating variance reduction.  Against K=2 this is -0.025 * (1 - 2/k):
    # -0.008 at K=3, so even the smallest challenger clears the 0.005 margin,
    # and 8 is only marginally better than 6.
    return -0.05 * (1.0 - 1.0 / k)


def build(cells: Path, *, with_cal_seconds: bool = True) -> None:
    rng = np.random.default_rng(3314)
    cells.mkdir(parents=True, exist_ok=True)
    idx = 0
    for emb, style in GEOMETRIES:
        learn = emb.partition("+")[2] or emb
        geometry = f"{learn}/{style}"
        for cat in CATEGORIES:
            for seed in SEEDS:
                rows = []
                for t in range(1, STEPS + 1):
                    n_good, n_bad = t // 2, t - t // 2
                    votes = n_good + n_bad
                    base_cost = 0.30 + rng.normal(0.0, 0.002)
                    for k in KS:
                        # Exactly `STEP_RATIO[k]` times production's step, with
                        # every extra second landing in calibration.
                        cal = STEP_SECONDS_K2 * STEP_RATIO[k] - OTHER_SECONDS
                        cost = base_cost + planted_level(geometry, k, votes)
                        common = {
                            "seed": seed,
                            "dataset": "vg_scale_any",
                            "embedder": emb,
                            "category": cat,
                            "style": style,
                            "t": t,
                            "n_good": n_good,
                            "n_bad": n_bad,
                            "pool_variant": "max",
                            "schedule": "",
                            "fold_count": k,
                            # Deliberately NOT proportional to the real cost:
                            # an analyzer that prices K off this column puts
                            # every K under the ceiling (max ratio 1.15) and
                            # ships the unaffordable one.
                            "fold_seconds": 1.0 + 0.02 * k,
                            "n_cal_scores": 10 * k,
                            "n_folds_used": k,
                            "train_seconds": OTHER_SECONDS * 0.5,
                            "pool_score_seconds": OTHER_SECONDS * 0.3,
                            "test_score_seconds": OTHER_SECONDS * 0.2,
                            # `fold_seconds` is only the fold FITS; see the
                            # planted disagreement with `cal_seconds` above.
                            "average_precision": 1.0 - cost,
                            "seed_mode": "text",
                            "seed_query": cat,
                            "seed_embedder": "siglip",
                        }
                        if with_cal_seconds:
                            common["cal_seconds"] = cal
                        rows.append(
                            {
                                **common,
                                "gmm_variant": f"folds_k{k}_anchored",
                                "threshold": 0.5 + rng.normal(0, 0.02 / np.sqrt(k)),
                                "cost": cost,
                                "regret_honest": cost - 0.25,
                                "regret": cost - 0.25,
                            }
                        )
                        # (6) the pooled arm, carrying the OPPOSITE sign.
                        rows.append(
                            {
                                **common,
                                "gmm_variant": f"folds_k{k}_xcal",
                                "threshold": 0.5,
                                "cost": base_cost + 0.004 * k,
                                "regret_honest": base_cost + 0.004 * k - 0.25,
                                "regret": base_cost + 0.004 * k - 0.25,
                            }
                        )
                pd.DataFrame(rows).to_csv(cells / f"task_{idx:04d}.csv", index=False)
                # The pick log a real run writes beside the metric frame.
                pd.DataFrame(
                    [
                        {"seed": seed, "dataset": "vg_scale_any", "category": cat, "t": t, "picked_id": t}
                        for t in range(1, 6)
                    ]
                ).to_csv(cells / f"task_{idx:04d}__picks.csv", index=False)
                idx += 1


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="folds3314-selftest-"))
    try:
        results = tmp / "results"
        out = tmp / "out"
        build(results / "cells")
        import analyze_folds_3314 as A

        if A.main(["--results", str(results), "--out", str(out), "--no-figures", "--no-viewer"]) != 0:
            raise SystemExit("analyze_folds_3314 failed on the fabricated cells")

        failures: list[str] = []
        agg = out / "agg"
        paired = pd.read_csv(agg / "paired_cost.csv")
        cost = pd.read_csv(agg / "cost_ratios.csv")
        ship = pd.read_csv(agg / "ship_rules.csv")
        sched = pd.read_csv(agg / "schedule_family.csv")
        summary = __import__("json").loads((out / "summary_folds3314.json").read_text())

        # (5) the cost model is the whole calibration clock, and it is named.
        if summary["cost_model"] != "cal_seconds":
            failures.append(f"cost model is {summary['cost_model']!r}, not cal_seconds")

        # The pick log stayed out of the metric frame: 6 categories x 4 seeds x
        # 3 geometries, but a cell is (dataset, category, seed, geometry).
        want_cells = len(CATEGORIES) * len(SEEDS) * len(GEOMETRIES)
        if summary["n_cells"] != want_cells:
            failures.append(f"n_cells {summary['n_cells']} != {want_cells} - a side frame leaked in")

        # (1) the two single-vector geometries are flat, so nothing ships there.
        flat = ship[ship["geometry"] != REGION]
        if flat.empty or bool(flat["ship_candidate"].any()):
            failures.append("a K shipped on a K-invariant geometry")

        # (2) the decay: the region benefit is in the early bands and gone deep.
        reg = paired[(paired["geometry"] == REGION) & (paired["k"] == 6)]
        early = float(reg[reg["band"] == "early 1-25"]["delta"].iloc[0])
        deep = float(reg[reg["band"] == "deep 101-150"]["delta"].iloc[0])
        if not (early < -A.MARGIN):
            failures.append(f"region K=6 early delta {early:.3g} did not recover the planted benefit")
        if abs(deep) > 1e-6:
            failures.append(f"region K=6 deep delta {deep:.3g} should be exactly zero")

        # (3) + (4) the two gates bite, in the right direction.
        r = ship[ship["geometry"] == REGION].set_index("k")
        if bool(r.loc[8, "ship_candidate"]) or bool(r.loc[8, "rule3_affordable"]):
            failures.append("K=8 is over the 1.5x ceiling and must not ship")
        if not bool(r.loc[8, "rule1_benefit"]):
            failures.append("K=8 has the largest planted benefit; rule 1 should still pass for it")
        if bool(r.loc[1, "ship_candidate"]) or bool(r.loc[1, "rule2_no_harm"]):
            failures.append("K=1 is worse everywhere and must fail the harm gate")
        for k in (3, 4, 6):
            if not bool(r.loc[k, "ship_candidate"]):
                # Name the rule that rejected it: "did not ship" is not enough
                # to tell a broken analyzer from a mis-planted fixture, which
                # cost this file one debugging round.
                why = [c for c in ("rule1_benefit", "rule2_no_harm", "rule3_affordable") if not bool(r.loc[k, c])]
                failures.append(
                    f"region K={k} clears all three planted rules but did not ship "
                    f"(failed {','.join(why) or 'nothing?'}; best delta {r.loc[k, 'best_delta']:.3g}, "
                    f"worst {r.loc[k, 'worst_delta']:.3g}, max ratio {r.loc[k, 'max_step_ratio']:.3g})"
                )

        # The measured ratio must be the PLANTED one, not the fold-fit one.
        got = float(cost[(cost["geometry"] == REGION) & (cost["k"] == 8)]["step_ratio"].iloc[0])
        if abs(got - STEP_RATIO[8]) > 0.01:
            failures.append(f"K=8 step ratio {got:.3g} != planted {STEP_RATIO[8]} - wrong cost column?")

        # The gate books the smallest fixed candidate on the region geometry.
        gate = summary["gate"]
        if not gate["gate_open"]:
            failures.append("the gate closed on a screen with a planted, affordable, decaying benefit")
        if gate["k_best"] != 3 or gate["k_best_geometry"] != REGION:
            failures.append(f"k_best {gate['k_best']}@{gate['k_best_geometry']} != 3@{REGION}")
        # The schedule with the most benefit inside the ceiling: K_early=6 (8 is
        # unaffordable) held to the wider cut, since the benefit survives to 60.
        if gate["schedule"] != "6@60":
            failures.append(f"schedule pick {gate['schedule']!r} != '6@60'")
        s = sched[(sched["geometry"] == REGION) & (sched["k_early"] == 8)]
        if s.empty or bool(s["eligible"].any()):
            failures.append("an 8-fold schedule is over the ceiling and must not be eligible")

        # (6) the arm split: the pooled rows carry the opposite sign, and the
        # shipped verdict must not be reading them.
        pooled = pd.read_csv(agg / "pooled_replication.csv")
        pr = pooled[(pooled["geometry"] == REGION) & (pooled["band"] == "early 1-25")]
        if pr.empty or not (pr.sort_values("k")["delta"].diff().dropna() > 0).all():
            failures.append("the pooled arm's planted monotone worsening was not recovered")

        # (5, second half) a pre-#3314 run must fall back and SAY so.
        old = tmp / "old"
        build(old / "results" / "cells", with_cal_seconds=False)
        if (
            A.main(["--results", str(old / "results"), "--out", str(tmp / "out_old"), "--no-figures", "--no-viewer"])
            != 0
        ):
            failures.append("the analyzer refused a pre-#3314 run outright instead of falling back")
        else:
            s2 = __import__("json").loads((tmp / "out_old" / "summary_folds3314.json").read_text())
            if not s2["cost_model"].startswith("fold_seconds"):
                failures.append("a run without cal_seconds must declare the fallback cost model")

        if failures:
            for x in failures:
                print(f"FAIL: {x}")
            return 1
        print(
            "selftest_analyze_folds_3314: OK (geometry split, decay, cost ceiling, harm gate, "
            "cost model + fallback, arm split, picks)"
        )
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
