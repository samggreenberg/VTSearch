"""Self-test for :mod:`analyze_cut` on fabricated cells (no cluster data needed).

The cut analyzer is read exactly once, on data that took an hour of cluster time
to produce, and it has to be right about a *sign*: whether a candidate rule beats
the incumbent, and which term in the derivation dominates. Both are easy to get
backwards. This plants known answers in a synthetic ``results`` tree and checks
they come back out:

* a candidate rule made cheaper inside the ramp window is found, with the right
  sign, size, and improved-cell fraction, and does not leak into the other window;
* the pairing unit is the **cell**, not the step (otherwise 29 autocorrelated
  steps per cell would inflate every p-value's confidence);
* the decomposition telescopes and names the term that was actually planted as
  dominant;
* a rule whose *blended* cost wins while its *raw cut* does not is reported as
  such, since that is the trap the plan pre-registers.

Usage::

    python selftest_analyze_cut.py     # exits non-zero on failure
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

#: How much cheaper the planted winner is inside the ramp window.
RAMP_EFFECT = -0.04
#: Planted decomposition, in threshold units: the prior/loss term dominates.
PLANTED_TERMS = {"prior_loss": 0.10, "identification": 0.02, "misspecification": 0.01, "transfer": 0.005}
#: The same chain in cost units, anchored to a noise-free oracle cost, so the
#: analyzer's "which term dominates" verdict has a known right answer.
ORACLE_COST = 0.20
PLANTED_COST_TERMS = {"prior_loss": 0.10, "identification": 0.02, "misspecification": 0.01, "transfer": 0.005}
#: Raw-cut cost per chain variant, built backwards from the oracle.
_CHAIN_RAW_COST = {
    "pooled_sim_oracle": ORACLE_COST + PLANTED_COST_TERMS["transfer"],
    "pooled_supervised": ORACLE_COST + PLANTED_COST_TERMS["transfer"] + PLANTED_COST_TERMS["misspecification"],
}
_CHAIN_RAW_COST["pooled_priorfree"] = _CHAIN_RAW_COST["pooled_supervised"] + PLANTED_COST_TERMS["identification"]
_CHAIN_RAW_COST["pooled_cross"] = _CHAIN_RAW_COST["pooled_priorfree"] + PLANTED_COST_TERMS["prior_loss"]
CATEGORIES = ["cat_a", "cat_b", "cat_c"]
SEEDS = [0, 1, 2, 3]
ARMS = [("dinov3_patch", "max_patch"), ("siglip", "whole_image")]

WINNER = "pooled_priorfree"
INCUMBENT = "pooled_mid"


def _ident(cat: str, seed: int, t: int, embedder: str, style: str) -> dict:
    n_votes = t
    return {
        "seed": seed,
        "dataset": "visual_genome_m",
        "category": cat,
        "strategy": "autopilot",
        "trainer": "mlp",
        "head": "linear",
        "style": style,
        "prevalence_arm": "natural",
        "realized_prevalence": 0.05,
        "t": t,
        "n_good": n_votes // 2,
        "n_bad": n_votes - n_votes // 2,
        "phase": "hard",
        "app_trained": 1,
        "embedder": embedder,
    }


def _fabricate(root: Path, rng: np.random.Generator) -> None:
    from vtscore.eval.voting_iterations import _CALIBRATION_COLUMNS, _CUT_DIAGNOSTIC_COLUMNS, _SAFE_GMM_VARIANTS

    cells = root / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    variants = [name for name, _f, _r in _SAFE_GMM_VARIANTS]

    idx = 0
    for embedder, style in ARMS:
        for cat in CATEGORIES:
            for seed in SEEDS:
                rows, diag = [], []
                for t in range(2, 31):
                    ident = _ident(cat, seed, t, embedder, style)
                    in_ramp = 6 <= t <= 20
                    base_cost = 0.30 + 0.002 * rng.standard_normal()
                    oracle_cost = ORACLE_COST
                    oracle_threshold = 0.50
                    # Shared jitter: the realised offset and its closed-form
                    # prediction move together, so the identity stays exact while
                    # the correlation the analyzer reports is still computable.
                    jitter = 0.01 * rng.standard_normal()

                    # The base (production) row, and the variant that must match it.
                    threshold = 0.55
                    for variant in ["", *variants]:
                        effect = RAMP_EFFECT if (variant == WINNER and in_ramp) else 0.0
                        # The blended and raw columns move together here except
                        # for the decoy, which wins only after blending.
                        decoy = variant == "pooled_gumbel_cross" and in_ramp
                        raw_cost = _CHAIN_RAW_COST.get(variant, base_cost) + (0.02 if decoy else 0.0)
                        row = dict(ident)
                        row.update(
                            pool_variant="max",
                            gmm_variant=variant,
                            threshold=threshold,
                            threshold_provenance="gmm_blend",
                            degenerate=0,
                            threshold_percentile=0.9,
                            xcal_threshold=0.52,
                            gmm_cut=oracle_threshold + (0.0 if variant == WINNER else 0.05),
                            blend_weight=0.5,
                            cut_fallback=0,
                            raw_cut_cost=raw_cost,
                            raw_cut_fpr=0.1,
                            raw_cut_fnr=0.2,
                            cost=base_cost + effect + (RAMP_EFFECT if decoy else 0.0),
                            fpr=0.1,
                            fnr=0.2,
                            auroc=0.9,
                            average_precision=0.5,
                            oracle_threshold=oracle_threshold,
                            oracle_cost=oracle_cost,
                            oracle_fpr=0.05,
                            oracle_fnr=0.1,
                            regret=base_cost - oracle_cost,
                            cal_oracle_threshold=0.5,
                            cal_oracle_cost=oracle_cost,
                            rule_inefficiency=0.0,
                            calibration_shift=0.0,
                            n_pool_rows=100.0,
                            train_seconds=1.0,
                            xcal_seconds=1.0,
                            pool_score_seconds=1.0,
                            test_score_seconds=1.0,
                            backend="torch",
                            device="cpu",
                            elapsed_seconds=1.0,
                            exemplar_id=-1,
                        )
                        rows.append(row)

                    # The decomposition frame: cuts placed so each successive
                    # gap is exactly the planted term.
                    for geometry in ("pooled", "image"):
                        tau_test_oracle = oracle_threshold
                        tau_sim_oracle = tau_test_oracle + PLANTED_TERMS["transfer"]
                        tau_supervised = tau_sim_oracle + PLANTED_TERMS["misspecification"]
                        tau_priorfree = tau_supervised + PLANTED_TERMS["identification"]
                        tau_cross = tau_priorfree + PLANTED_TERMS["prior_loss"] + jitter
                        d = dict(ident)
                        d.update(
                            geometry=geometry,
                            sim_n=2000.0,
                            sim_prevalence=0.05,
                            fallback_median=0.4,
                            gmm_ok=1,
                            w_lo=0.95,
                            mu_lo=0.30,
                            var_lo=0.02,
                            w_hi=0.05,
                            mu_hi=0.80,
                            var_hi=0.01,
                            gmm_loglik=1.0,
                            gmm_logit_loglik=1.0,
                            # Equal-variance closed form for this fit, so the
                            # identity check has something exact to recover.
                            pred_offset_equal_var=PLANTED_TERMS["prior_loss"] + jitter,
                            evt_ok=1,
                            evt_fit_fail="ok",
                            evt_gumbel_is_low=1,
                            evt_w_gumbel=0.95,
                            evt_loc=-1.0,
                            evt_scale=0.5,
                            evt_mu=1.5,
                            evt_var=0.5,
                            evt_loglik=1.1 if geometry == "pooled" else 1.0,
                            evt_loglik_gain=0.1 if geometry == "pooled" else 0.0,
                            s_mu_neg=0.30,
                            s_var_neg=0.02,
                            s_mu_pos=0.80,
                            s_var_pos=0.01,
                            s_prevalence=0.05,
                            tau_mid=tau_cross - PLANTED_TERMS["prior_loss"] - jitter,
                            tau_cross=tau_cross,
                            tau_priorfree=tau_priorfree,
                            tau_rate=tau_priorfree,
                            tau_gumbel_cross=tau_cross,
                            tau_gumbel_priorfree=tau_priorfree,
                            tau_gumbel_rate=tau_priorfree,
                            tau_gumbel_any_cross=tau_cross,
                            tau_gumbel_any_priorfree=tau_priorfree,
                            tau_gumbel_any_rate=tau_priorfree,
                            tau_supervised=tau_supervised,
                            tau_sim_oracle=tau_sim_oracle,
                            tau_test_oracle=tau_test_oracle,
                            oracle_lo_sf_gauss=0.02,
                            oracle_lo_sf_evt=0.02,
                        )
                        diag.append(d)

                pd.DataFrame(rows, columns=pd.Index([*_CALIBRATION_COLUMNS, "embedder", "exemplar_id"])).to_csv(
                    cells / f"task_{idx:04d}.csv", index=False
                )
                pd.DataFrame(diag, columns=pd.Index([*_CUT_DIAGNOSTIC_COLUMNS, "embedder"])).to_csv(
                    cells / f"task_{idx:04d}__cutdiag.csv", index=False
                )
                idx += 1


def _check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    return ok


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "results"
        root.mkdir(parents=True)
        # analyze_cut reads CALIB_RESULTS at import time, via common.setup_env.
        os.environ["CALIB_RESULTS"] = str(root)
        os.environ.setdefault("CALIB_EXP", str(Path(tmp) / "exp"))

        import analyze_cut  # noqa: PLC0415

        _fabricate(root, np.random.default_rng(2836))
        rc = analyze_cut.main()
        if rc != 0:
            print("analyze_cut.main() returned non-zero")
            return 1

        import json  # noqa: PLC0415

        summary = json.loads((root / "summary_cut.json").read_text())
        contrasts = pd.read_csv(root / "agg" / "cut_contrasts.csv")
        decomp = pd.read_csv(root / "agg" / "cut_decomposition.csv")

        ok = True
        prod = contrasts[contrasts["arm"].str.contains("dinov3_patch/max_patch")]
        ramp = prod[(prod["window"] == "ramp_6_20") & (prod["variant"] == WINNER)]
        ok &= _check("winner recovered in the ramp window", len(ramp) == 1)
        if len(ramp) == 1:
            r = ramp.iloc[0]
            ok &= _check(
                "planted effect size",
                abs(r["mean_d_cost"] - RAMP_EFFECT) < 1e-6,
                f"{r['mean_d_cost']:.5f} vs {RAMP_EFFECT}",
            )
            ok &= _check("all cells improved", abs(r["frac_cells_improved"] - 1.0) < 1e-9)
            ok &= _check(
                "pairing unit is the cell",
                int(r["n_cells"]) == len(CATEGORIES) * len(SEEDS),
                f"n_cells={r['n_cells']}",
            )

        other = prod[(prod["window"] == "pure_gmm_2_5") & (prod["variant"] == WINNER)]
        ok &= _check(
            "effect does not leak into the sub-ramp window",
            len(other) == 1 and abs(other.iloc[0]["mean_d_cost"]) < 1e-9,
        )

        dec = summary["decisions"]
        ok &= _check("winner chosen", dec["best_by_cost"]["variant"] == WINNER, str(dec["best_by_cost"]["variant"]))
        ok &= _check("beats the incumbent", bool(dec["beats_midpoint"]))
        ok &= _check("closest to the oracle cut", dec["closest_to_oracle"] == WINNER, str(dec["closest_to_oracle"]))

        # The decoy wins on the blended column but not on the raw cut; the
        # analyzer must expose both so it cannot be shipped on the wrong one.
        decoy = prod[(prod["window"] == "ramp_6_20") & (prod["variant"] == "pooled_gumbel_cross")]
        ok &= _check(
            "decoy's raw cut is worse than its blended cost",
            len(decoy) == 1 and decoy.iloc[0]["mean_d_raw_cut_cost"] > 0 > decoy.iloc[0]["mean_d_cost"],
        )

        pooled = decomp[(decomp["window"] == "ramp_6_20") & (decomp["geometry"] == "pooled")]
        ok &= _check("decomposition telescopes", bool((pooled["residual"].abs() < 1e-9).all()))
        for term, planted in PLANTED_TERMS.items():
            got = float(pooled[f"term_{term}"].iloc[0])
            # prior_loss carries the shared jitter, which averages out over cells.
            tol = 2e-3 if term == "prior_loss" else 1e-9
            ok &= _check(f"term {term}", abs(got - planted) < tol, f"{got:.5f} vs {planted}")
        ok &= _check(
            "dominant term named",
            dec["dominant_error_term"] == "prior_loss",
            str(dec["dominant_error_term"]),
        )

        offs = summary["offsets"]["identity"]
        pooled_off = [o for o in offs if o["geometry"] == "pooled" and "max_patch" in o["arm"]]
        ok &= _check(
            "offset identity recovered",
            bool(pooled_off) and abs(pooled_off[0]["mean_abs_residual"]) < 1e-9,
        )
        ok &= _check(
            "offset correlation computable",
            bool(pooled_off) and abs(pooled_off[0]["corr"] - 1.0) < 1e-6,
            "" if not pooled_off else f"corr={pooled_off[0]['corr']}",
        )

        evt = pd.read_csv(root / "agg" / "cut_evt_evidence.csv")
        ok &= _check(
            "EVT gain is geometry-specific",
            bool(
                (evt[evt["geometry"] == "pooled"]["evt_loglik_gain"] > 0).all()
                and (evt[evt["geometry"] == "image"]["evt_loglik_gain"].abs() < 1e-9).all()
            ),
        )
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
