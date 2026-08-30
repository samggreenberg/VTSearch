#!/usr/bin/env python
"""Planted-answer selftest for ``analyze_fitq_3329.py`` (issue #3329).

Run this **before** submitting the array.  The analyzer is run here on
fabricated cells whose answer is known by construction, so a decision rule that
reads the wrong column, pools the wrong scope, or inverts a bar fails here
rather than in a report.

The traps, each a different way this analyzer could be wrong:

1. **Scope leakage.**  The fold rows carry NaN in every labelled column, and the
   ``sim:image`` rows carry a *deliberately different* skew from ``sim:pooled``.
   An analyzer that pools scopes gets a diluted skew and misses H2's bar.
2. **The paired contrast is not the level difference.**  ``sim:pooled`` minus
   ``sim:image`` is planted at a value that is NOT the difference of the two
   arms' medians, so a paired reading computed as a difference of levels lands
   on the wrong number.
3. **H3 needs both halves.**  One arm is planted with inert mass but a LARGE
   mean movement.  An analyzer testing only the mass share calls it inert.
4. **H4's control matters.**  Regret is planted as a pure function of
   ``n_test_pos`` with *no* independent tail-ratio effect, while tail ratio is
   correlated with ``n_test_pos``.  An analyzer that omits the control finds a
   strong spurious slope; the partial R² must stay under the bar.
5. **The base-row filter.**  Variant rows (``gmm_variant`` set) carry wild
   regret values; an analyzer that fails to drop them corrupts H4.
6. **Side frames must not reach the main frame.**  A ``__fitq`` file is written
   beside each ``task_*.csv``; ``main_frame_files`` must not pick it up.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

ARMS = [
    ("siglip", "whole_image"),
    ("siglip+dinov3_patch", "whole_image"),
    ("siglip+dinov3_patch", "max_patch"),
]
CATEGORIES = ["bird", "boat", "clock"]
SEEDS = [0, 1]
STEPS = [5, 10, 15, 20]

#: Planted per-arm levels.  Region clears H1 and H2; the binary control clears
#: H1's lower bar and fails H2, which is the study's predicted shape.
TAIL_RATIO = {"siglip/whole_image": 1.35, "siglip+dinov3_patch/whole_image": 1.30,
              "siglip+dinov3_patch/max_patch": 2.10}
SKEW_POOLED = {"siglip/whole_image": 0.10, "siglip+dinov3_patch/whole_image": 0.15,
               "siglip+dinov3_patch/max_patch": 0.95}
#: Trap 2: the image-scope skew is planted so the WITHIN-run difference is 0.60
#: on the region arm, which is not any difference of the level medians above.
SKEW_IMAGE = {a: SKEW_POOLED[a] - 0.60 for a in SKEW_POOLED}
#: Trap 3: the middle arm has inert mass but a large mean movement, so "inert"
#: must be False for it even though its mass share is tiny.
DMU_LO = {"siglip/whole_image": 0.001, "siglip+dinov3_patch/whole_image": 0.25,
          "siglip+dinov3_patch/max_patch": 0.002}
MASS = 1e-4


def build(root: Path) -> None:
    cells = root / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    idx = 0
    for emb, style in ARMS:
        arm = f"{emb}/{style}"
        for cat in CATEGORIES:
            for seed in SEEDS:
                fq_rows, main_rows = [], []
                for t in STEPS:
                    # Trap 4: n_test_pos drives regret, and tail ratio is
                    # correlated with it, but tail ratio has no OWN effect.
                    n_pos = 30 + 10 * t
                    tail = TAIL_RATIO[arm] * (1.0 + 0.02 * t)
                    regret = 0.5 - 0.03 * np.log(n_pos) + float(rng.normal(0, 0.001))
                    base = {
                        "seed": seed, "dataset": "vg_scale_any", "category": cat,
                        "style": style, "t": t, "embedder": emb,
                        "seed_mode": "text", "seed_embedder": "siglip",
                    }
                    fq_rows.append({
                        **base, "scope": "sim:pooled", "fit_ok": True,
                        "tail_ratio": tail, "shape_skew_neg": SKEW_POOLED[arm],
                        "shape_n_neg": 500.0, "shape_n_pos": 100.0,
                        "anchored_dmu_lo": np.nan, "anchor_mass_frac": np.nan,
                        "anchor_kappa": np.nan,
                    })
                    fq_rows.append({
                        **base, "scope": "sim:image", "fit_ok": True,
                        "tail_ratio": tail * 0.8, "shape_skew_neg": SKEW_IMAGE[arm],
                        "shape_n_neg": 500.0, "shape_n_pos": 100.0,
                        "anchored_dmu_lo": np.nan, "anchor_mass_frac": np.nan,
                        "anchor_kappa": np.nan,
                    })
                    # Trap 1: fold rows carry NaN in every labelled column.
                    for f in (0, 1):
                        fq_rows.append({
                            **base, "scope": f"fold{f}", "fit_ok": True,
                            "tail_ratio": np.nan, "shape_skew_neg": np.nan,
                            "shape_n_neg": np.nan, "shape_n_pos": np.nan,
                            "anchored_dmu_lo": DMU_LO[arm], "anchor_mass_frac": MASS,
                            "anchor_kappa": 0.3,
                        })
                    main_rows.append({
                        **base, "gmm_variant": "", "pool_variant": "",
                        "regret_honest": regret, "n_test_pos": n_pos,
                    })
                    # Trap 5: a variant row with wild regret that must be dropped.
                    main_rows.append({
                        **base, "gmm_variant": "folds_k4_anchored", "pool_variant": "",
                        "regret_honest": regret + 5.0, "n_test_pos": n_pos,
                    })
                pd.DataFrame(main_rows).to_csv(cells / f"task_{idx:04d}.csv", index=False)
                pd.DataFrame(fq_rows).to_csv(cells / f"task_{idx:04d}__fitq.csv", index=False)
                idx += 1


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="fitq3329-selftest-"))
    failures: list[str] = []
    try:
        results = tmp / "results"
        build(results)
        import analyze_fitq_3329 as A

        rc = A.main(["--results", str(results), "--out", str(tmp / "analysis"), "--no-figures"])
        if rc != 0:
            return 1

        agg = tmp / "analysis" / "agg"
        h1 = pd.read_csv(agg / "h1_tail_calibration.csv").set_index("arm")
        h2l = pd.read_csv(agg / "h2_shape_levels.csv").set_index("arm")
        h2p = pd.read_csv(agg / "h2_shape_paired.csv").set_index("arm")
        h3 = pd.read_csv(agg / "h3_anchoring.csv").set_index("arm")
        h4 = pd.read_csv(agg / "h4_regret.csv").set_index("arm")

        region = "siglip+dinov3_patch/max_patch"
        binary = "siglip/whole_image"
        middle = "siglip+dinov3_patch/whole_image"

        # Trap 6 / trap 1: the fitq frame must not have leaked into the main one,
        # and the fold rows must not have diluted the labelled medians.
        got = h1.loc[region, "tail_ratio_median"]
        want = TAIL_RATIO[region] * (1.0 + 0.02 * float(np.median(STEPS)))
        if not np.isclose(got, want, rtol=0.05):
            failures.append(
                f"H1 region tail ratio {got:.3f} != planted {want:.3f} - fold rows (NaN) or "
                f"sim:image rows (0.8x) may have been pooled into the median"
            )
        if not bool(h1.loc[region, "meets_bar"]):
            failures.append(f"H1 region planted at {got:.2f} above bar {A.H1_TAIL_RATIO_REGION} but did not meet it")
        if not bool(h1.loc[binary, "meets_bar"]):
            failures.append(f"H1 binary planted above bar {A.H1_TAIL_RATIO_BINARY} but did not meet it")

        # H2 levels: only the region arm clears the skew bar.
        if not np.isclose(h2l.loc[region, "skew_neg_median"], SKEW_POOLED[region], rtol=1e-6):
            failures.append(f"H2 region skew {h2l.loc[region, 'skew_neg_median']} != planted {SKEW_POOLED[region]}")
        if not bool(h2l.loc[region, "meets_bar"]):
            failures.append("H2 region planted at 0.95 above the 0.5 bar but did not meet it")
        if bool(h2l.loc[binary, "meets_bar"]):
            failures.append("H2 binary planted at 0.10 below the bar but was reported as meeting it")

        # Trap 2: the paired contrast is the within-run difference, 0.60 - NOT
        # the difference of the arms' level medians (0.95 - 0.15 = 0.80).
        got_pair = h2p.loc[region, "d_skew_pooled_minus_image"]
        if not np.isclose(got_pair, 0.60, atol=1e-6):
            failures.append(
                f"H2 paired contrast {got_pair:.3f} != planted within-run 0.600; "
                f"0.800 would mean it was computed as a difference of arm levels"
            )

        # Trap 3: inert needs BOTH halves.
        if not bool(h3.loc[region, "inert"]):
            failures.append("H3 region planted inert (dmu 0.002, mass 1e-4) but was not reported inert")
        if bool(h3.loc[middle, "inert"]):
            failures.append(
                "H3 middle arm planted with tiny mass but a LARGE dmu_lo (0.25) and was still called "
                "inert - the mean-movement half of the test is not being applied"
            )

        # Trap 4: no independent tail-ratio effect survives the n_pos control.
        if region in h4.index:
            pr = h4.loc[region, "partial_r2"]
            if bool(h4.loc[region, "meets_bar"]):
                failures.append(
                    f"H4 planted with NO independent tail-ratio effect (regret is a pure function of "
                    f"n_test_pos) but partial R^2 came out {pr:.3f} >= {A.H4_PARTIAL_R2} - the "
                    f"n_test_pos control is probably missing from the base model"
                )
        else:
            failures.append("H4 produced no row for the region arm")

        # Trap 5: the variant rows must have been dropped; if they were not, the
        # +5.0 regret outlier moves the fit enormously.
        summary = pd.read_json(tmp / "analysis" / "summary_fitq3329.json", typ="series")
        if bool(summary["h4_misfit_predicts_regret"]):
            failures.append("H4 verdict is positive on a fixture with no planted effect")
        if "does not predict regret" not in str(summary["verdict"]):
            failures.append(f"verdict text does not match the planted null: {summary['verdict']!r}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print(
        "selftest_analyze_fitq_3329: OK - scope leakage, paired-vs-level contrast, "
        "H3's two halves, H4's n_pos control, the base-row filter and side-frame "
        "isolation all behave as planted"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
