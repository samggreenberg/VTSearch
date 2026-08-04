"""Self-test for analyze_mixin.py against planted answers (issue #2841).

The analyzer's job is to recover a difference between schedules from noisy
paired cells.  Run it on synthetic frames where the answer is known by
construction, so a bug shows up as a wrong verdict rather than as a plausible
table nobody can check.  Mirrors ``selftest_analyze_ab.py``'s approach for the
#2799 A/B analyzer, including the **sign-flip** case - the cheapest way for a
pairing bug to hide is to report a real effect backwards.

    python selftest_analyze_mixin.py
"""

from __future__ import annotations

import sys

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analyze_mixin as am  # noqa: E402

RNG = np.random.default_rng(2841)


def _frame(schedules: dict[str, float], n_cat: int = 12, n_seed: int = 8, style: str = "max_patch") -> pd.DataFrame:
    """Cells whose cost is a per-cell baseline plus a fixed per-schedule offset.

    The per-cell baseline is shared across schedules (that is what makes the
    comparison paired) and is drawn far wider than the planted offsets, so an
    analyzer that forgets to pair cannot recover the effect.
    """
    rows = []
    for cat in range(n_cat):
        for seed in range(n_seed):
            cell_level = RNG.normal(0.5, 0.25)  # the shared, dominant term
            for sched, offset in schedules.items():
                for t, votes in enumerate(range(7, 21), start=1):
                    rows.append(
                        {
                            "dataset": "ds",
                            "embedder": "emb",
                            "style": style,
                            "category": f"cat{cat}",
                            "seed": seed,
                            "t": t,
                            "n_good": votes // 2,
                            "n_bad": votes - votes // 2,
                            "schedule": sched,
                            "gmm_variant": "",
                            "cost": cell_level + offset + RNG.normal(0, 0.02),
                            "fnr": 0.2 + offset / 2,
                            "fpr": 0.2 + offset / 2,
                            "regret": 0.1 + offset,
                            "average_precision": 0.7,
                            "auroc": 0.8,
                            "degenerate": 0.0,
                        }
                    )
    df = pd.DataFrame(rows)
    df["n_votes"] = df["n_good"] + df["n_bad"]
    df["mode"] = df.apply(am._voting_mode, axis=1)
    return df


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    return ok


def main() -> int:
    ok = True
    print("planted effect: 'better' is 0.05 cheaper than prod, 'worse' 0.05 dearer")
    df = _frame({"prod": 0.0, "better": -0.05, "worse": +0.05})
    deltas = am.paired_vs_baseline(am.cell_means(df))
    by = {r["schedule"]: r for _, r in deltas.iterrows()}

    ok &= check("recovers the winner's sign", by["better"]["d_cost"] < 0, f"d_cost={by['better']['d_cost']:+.4f}")
    ok &= check("recovers the loser's sign", by["worse"]["d_cost"] > 0, f"d_cost={by['worse']['d_cost']:+.4f}")
    ok &= check(
        "recovers the winner's magnitude",
        abs(by["better"]["d_cost"] + 0.05) < 0.005,
        f"{by['better']['d_cost']:+.4f} vs planted -0.0500",
    )
    ok &= check("winner is significant", by["better"]["p_wilcoxon"] < 1e-6, f"p={by['better']['p_wilcoxon']:.2g}")
    ok &= check("winner improves nearly every cell", by["better"]["pct_improved"] > 95)
    ok &= check("baseline is not compared with itself", "prod" not in by)

    print("null effect: an identical schedule under a different name")
    dfn = _frame({"prod": 0.0, "twin": 0.0})
    dn = am.paired_vs_baseline(am.cell_means(dfn))
    twin = dn[dn.schedule == "twin"].iloc[0]
    ok &= check("null effect is small", abs(twin["d_cost"]) < 0.01, f"d_cost={twin['d_cost']:+.4f}")
    ok &= check("null effect is not significant", twin["p_wilcoxon"] > 0.01, f"p={twin['p_wilcoxon']:.2g}")

    print("mode split: region and binary must be reported separately")
    dfr = _frame({"prod": 0.0, "split": -0.05}, style="max_patch")
    dfb = _frame({"prod": 0.0, "split": +0.05}, style="whole_image")
    dm = am.paired_vs_baseline(am.cell_means(pd.concat([dfr, dfb], ignore_index=True)))
    region = dm[(dm["mode"] == "region") & (dm.schedule == "split")].iloc[0]
    binary = dm[(dm["mode"] == "binary") & (dm.schedule == "split")].iloc[0]
    ok &= check(
        "opposite per-mode effects are not averaged away",
        region["d_cost"] < -0.04 and binary["d_cost"] > 0.04,
        f"region={region['d_cost']:+.4f} binary={binary['d_cost']:+.4f}",
    )

    print("window: steps outside 7-20 votes must not enter the headline")
    dfw = _frame({"prod": 0.0, "late_only": 0.0})
    # Plant a huge penalty that lives only past the window.
    extra = dfw[dfw.schedule == "late_only"].copy()
    extra["n_good"], extra["n_bad"] = 30, 30
    extra["n_votes"] = 60
    extra["cost"] = extra["cost"] + 5.0
    dw = am.paired_vs_baseline(am.cell_means(pd.concat([dfw, extra], ignore_index=True)))
    late = dw[dw.schedule == "late_only"].iloc[0]
    ok &= check("out-of-window rows are excluded", abs(late["d_cost"]) < 0.01, f"d_cost={late['d_cost']:+.4f}")

    print("fidelity guard: a drifted prod row must abort the run")
    dff = _frame({"prod": 0.0})
    base = dff[dff.schedule == "prod"].copy()
    base["schedule"] = ""
    base["threshold"] = 0.5
    dff["threshold"] = 0.5
    good = pd.concat([dff, base], ignore_index=True)
    try:
        am.assert_screen_fidelity(good)
        fidelity_ok = True
    except SystemExit:
        fidelity_ok = False
    ok &= check("matching rows pass the fidelity check", fidelity_ok)

    drifted = good.copy()
    drifted.loc[drifted.schedule == "prod", "threshold"] = 0.9
    try:
        am.assert_screen_fidelity(drifted)
        caught = False
    except SystemExit:
        caught = True
    ok &= check("drifted rows abort", caught)

    print("\n" + ("ALL SELF-TESTS PASSED" if ok else "SELF-TESTS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
