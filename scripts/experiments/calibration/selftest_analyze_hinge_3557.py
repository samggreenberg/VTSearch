"""Planted-answer self-test for ``analyze_hinge_3557.py`` - no cluster data.

Writes two fake arms of cell CSVs whose answers are known by construction, runs
the analyzer on them, and asserts it recovers each planted answer:

* ``hinge`` is ``mid_tilt`` at k >= 0 and **0.020 cheaper** (rate scale) at every
  k < 0, on the re-cut and on the full ship, in all three environments;
* ``hinge_raw`` breaks nesting at every step (it cuts at 0.30 below the seam,
  under the 0.50 it cuts at k=0), and the guarded ``hinge`` never does;
* arm B reaches the incumbent's end-of-session cost in fewer clicks;
* a second fixture plants **+0.020 of harm at k=+1** in one environment, and the
  decision rule must then refuse to ship.

The failure each assertion guards: a units slip (rate scale vs cost scale is a
factor 2**|k|, so the -0.020 would read -0.040 at k=-1), a pairing slip (arm B's
rows joined to the wrong cell), and a contract audit that sorts by the wrong
column and passes a violating rule.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
KS = [-2, -1, -0.5, -0.25, 0, 1, 2]
RULES = ["mid_tilt", "mid", "rate", "cross_tilt", "hinge", "hinge_raw", "hinge_cont"]
ENVS = [
    ("visual_genome_m", "siglip+dinov3_patch", "max_patch"),
    ("coco_val", "siglip", "whole_image"),
    ("caltech101_m", "siglip", "whole_image"),
]
CATS = ["c0", "c1", "c2", "c3"]
SEEDS = [0, 1]
STEPS = np.arange(1, 161)
GAIN = -0.020
HARM = 0.020


def _thr(rule: str, k: float) -> float:
    base = 0.5 - 0.05 * k
    if rule == "mid":
        return 0.5
    if rule in ("hinge", "hinge_cont") and k < 0:
        return base + 0.01
    if rule == "hinge_raw" and k < 0:
        return 0.30  # below the 0.50 it cuts at k=0: a planted nesting violation
    return base


def _rcost(rule: str, k: float, arm: str, env_i: int, harm_env: int | None, noise: float) -> float:
    c = 0.20 + noise
    if rule in ("hinge", "hinge_raw", "hinge_cont") and k < 0:
        c += GAIN
    if arm == "hinge" and rule == "hinge" and k == 1 and harm_env == env_i:
        c += HARM
    return c


def write_arm(root: Path, arm: str, harm_env: int | None) -> None:
    cells = root / arm / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0 if arm == "incumbent" else 1)
    task = 0
    for env_i, (ds, emb, style) in enumerate(ENVS):
        for cat in CATS:
            for seed in SEEDS:
                cell_noise = rng.normal(0, 0.003)
                rows = []
                for t in STEPS:
                    for rule in RULES:
                        for k in KS:
                            thr = _thr(rule, k)
                            rc = _rcost(rule, k, arm, env_i, harm_env, cell_noise)
                            scale = 2.0 ** abs(k)
                            rows.append(
                                {
                                    "seed": seed,
                                    "dataset": ds,
                                    "category": cat,
                                    "style": style,
                                    "t": int(t),
                                    "n_good": int(t) // 5,
                                    "n_bad": int(t) - int(t) // 5,
                                    "embedder": emb,
                                    "cut_rule": rule,
                                    "inclusion_k": k,
                                    "fold_quantile": thr,
                                    "cut_threshold": thr,
                                    "cut_cost": rc * scale,
                                    "cut_fpr": 0.1,
                                    "cut_fnr": 0.1,
                                    "cut_regret": (rc - 0.1) * scale,
                                    "admitted_frac": 1 - thr,
                                    "n_admitted": int(round(1000 * (1 - thr))),
                                    "n_test": 1000,
                                    "seam_q_mid": 0.9,
                                    "seam_q_cross0": 0.8,
                                    "seam_q_rate0": 0.88,
                                    "fit_log2_prior_odds": 4.0,
                                    "fit_log2_var_ratio": 1.0,
                                }
                            )
                pd.DataFrame(rows).to_csv(cells / f"task_{task:04d}__cutincl.csv", index=False)
                tau = 50.0 if arm == "incumbent" else 30.0
                main = pd.DataFrame(
                    {
                        "seed": seed,
                        "dataset": ds,
                        "category": cat,
                        "style": style,
                        "t": STEPS,
                        "n_good": STEPS // 5,
                        "n_bad": STEPS - STEPS // 5,
                        "embedder": emb,
                        "gmm_variant": "",
                        "pool_variant": "max",
                        "threshold": 0.5,
                        "threshold_provenance": "fold_anchored[2/2]",
                        "cost": 0.5 * np.exp(-STEPS / tau) + 0.2 + cell_noise,
                        "fpr": 0.1,
                        "fnr": 0.1,
                        "average_precision": 0.6,
                        "live_cut_rule": "mid_tilt" if arm == "incumbent" else "hinge",
                    }
                )
                main.to_csv(cells / f"task_{task:04d}.csv", index=False)
                task += 1


def run(harm_env: int | None) -> dict:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_arm(root, "incumbent", harm_env)
        write_arm(root, "hinge", harm_env)
        out = root / "analysis"
        subprocess.run(  # noqa: S603 - argv is this interpreter and a sibling script
            [
                sys.executable,
                str(HERE / "analyze_hinge_3557.py"),
                "--incumbent",
                str(root / "incumbent"),
                "--hinge",
                str(root / "hinge"),
                "--out",
                str(out),
                "--workers",
                "2",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        summary = json.loads((out / "hinge3557_summary.json").read_text())
        summary["_paired"] = pd.read_csv(out / "hinge3557_paired.csv")
        summary["_contract"] = pd.read_csv(out / "hinge3557_contract.csv")
        summary["_speed"] = pd.read_csv(out / "hinge3557_speed.csv")
        return summary


def main() -> None:
    s = run(harm_env=None)
    paired, contract, speed = s["_paired"], s["_contract"], s["_speed"]

    # The planted -0.020 comes back on the RATE scale at every k < 0 (a units
    # slip would read -0.040 at k=-1 and -0.080 at k=-2), and 0 at k >= 0.
    ship = paired[paired["contrast"] == "ship_cost"]
    for k, want in [(-2, GAIN), (-1, GAIN), (-0.25, GAIN), (0, 0.0), (1, 0.0), (2, 0.0)]:
        got = ship[ship["inclusion_k"] == k]["d_mean"]
        assert len(got) == len(ENVS), (k, len(got))
        assert np.allclose(got, want, atol=0.004), (k, got.tolist())
    recut = paired[(paired["contrast"] == "recut") & (paired["arm_b"] == "hinge")]
    assert np.allclose(recut[recut["inclusion_k"] == -1]["d_mean"], GAIN, atol=1e-9)

    # Contract: the guarded hinge is clean, the literal one violates everywhere.
    h = contract[contract["cut_rule"] == "hinge"]
    raw = contract[contract["cut_rule"] == "hinge_raw"]
    assert (h["n_violating_steps"] == 0).all()
    assert (raw["violation_rate"] == 1.0).all(), raw["violation_rate"].tolist()
    assert s["rule1_contract"] is True

    # Speed: arm B (tau 30) reaches the incumbent's (tau 50) end cost sooner.
    assert (speed["clicks_to_target_d"] < 0).all(), speed["clicks_to_target_d"].tolist()

    assert s["rule2_no_harm"] is True
    assert s["rule3_gain"] is True and len(s["rule3_gain_envs"]) == 3

    # Second fixture: one environment harmed by +0.020 at k=+1 -> no ship.
    s2 = run(harm_env=1)
    assert s2["rule2_no_harm"] is False
    harmed = s2["rule2_harmed_ship"]
    assert len(harmed) == 1 and harmed[0]["inclusion_k"] == 1 and harmed[0]["dataset"] == "coco_val", harmed
    assert s2["ships"] is False
    print("selftest_analyze_hinge_3557: all planted answers recovered")


if __name__ == "__main__":
    main()
