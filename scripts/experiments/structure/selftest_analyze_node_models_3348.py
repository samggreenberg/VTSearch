#!/usr/bin/env python
"""Planted-answer self-test for ``analyze_node_models_3348`` (#3348).

    python selftest_analyze_node_models_3348.py

Writes run-shaped CSVs whose answer is chosen in advance -- one tree variant
that passes every gate, one that is well separated but miscalibrated, a shipped
baseline that reproduces the #3329 failure on the patch embedder, and the
control -- then checks that the analyzer computes the planted rates and picks
the planted winner.  The rows carry exactly the columns the run writes,
``scope`` values included, because a fixture that plants what the harness never
emits tests a filter nothing runs (the #3333 lesson).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyze_node_models_3348 as az  # noqa: E402

DATASETS = ["coco_val", "coco_better", "caltech101_m"]  # 3 self, 2 same-source, 4 different-source
EMBEDDERS = ["dinov3_patch", "siglip"]

# variant -> embedder -> (ks, rate05, fires on self, fires same-source, fires different-source)
PLAN = {
    "grid_mean_shipped": {
        "dinov3_patch": (0.15, 0.14, True, True, True),
        "siglip": (0.09, 0.04, False, False, True),
    },
    "knn_global": {
        "dinov3_patch": (0.02, 0.05, False, False, True),
        "siglip": (0.02, 0.05, False, False, True),
    },
    "knn_split_deepest": {
        "dinov3_patch": (0.03, 0.05, False, False, True),
        "siglip": (0.03, 0.05, False, False, True),
    },
    "vmf_loo_mean": {  # separated but miscalibrated: must fail on KS
        "dinov3_patch": (0.12, 0.05, False, False, True),
        "siglip": (0.12, 0.05, False, False, True),
    },
    "vmf_split_deepest": {  # calibrated, a quarter of the detection on the patch space
        "dinov3_patch": (0.03, 0.05, False, False, False),
        "siglip": (0.03, 0.05, False, False, True),
    },
}


def write(results: Path) -> None:
    pit, shift = [], []
    for e in EMBEDDERS:
        for d in DATASETS:
            for v, by in PLAN.items():
                ks, rate, *_ = by[e]
                for stat, val in (("ks_uniform", ks), ("frac_below_0.05", rate), ("sd", 0.29)):
                    pit.append(
                        {
                            "embedder": e,
                            "seed": 0,
                            "dataset": d,
                            "family": v,
                            "scope": "holdout",
                            "statistic": stat,
                            "value": val,
                            "n": 1000,
                        }
                    )
            # Rows the analyzer must ignore: fit/structure bookkeeping.
            pit.append(
                {
                    "embedder": e,
                    "seed": 0,
                    "dataset": d,
                    "family": "fit",
                    "scope": "knn_split",
                    "statistic": "seconds",
                    "value": 3.0,
                    "n": 0,
                }
            )
            pit.append(
                {
                    "embedder": e,
                    "seed": 0,
                    "dataset": d,
                    "family": "structure",
                    "scope": "full",
                    "statistic": "n_nodes",
                    "value": 99,
                    "n": 0,
                }
            )
        for b in DATASETS:
            for q in DATASETS:
                same = az.SOURCES[b] == az.SOURCES[q]
                for v, by in PLAN.items():
                    _, _, f_self, f_same, f_diff = by[e]
                    # vmf_split_deepest on the patch space detects 1 of the 4 different-source pairs.
                    if v == "vmf_split_deepest" and e == "dinov3_patch" and b != q and not same:
                        f_diff = (b, q) == ("coco_val", "caltech101_m")
                    fire = f_self if b == q else (f_same if same else f_diff)
                    shift.append(
                        {
                            "embedder": e,
                            "seed": 0,
                            "build_dataset": b,
                            "query_dataset": q,
                            "is_self": b == q,
                            "variant": v,
                            "n_items": 1000,
                            "frac_atypical": 0.5 if fire else 0.05,
                            "z_score": 50.0 if fire else 0.0,
                            "shifted": fire,
                            "median_pvalue": 0.5,
                        }
                    )
    pd.DataFrame(pit).to_csv(results / "pit_x__s0.csv", index=False)
    pd.DataFrame(shift).to_csv(results / "shift_x__s0.csv", index=False)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        results, out = Path(tmp) / "r", Path(tmp) / "a"
        results.mkdir()
        write(results)
        assert az.main(["--results", str(results), "--out", str(out)]) == 0
        import json  # noqa: PLC0415

        v = json.loads((out / "verdict.json").read_text())
        t = pd.read_csv(out / "table.csv").set_index(["embedder", "variant"])

    checks = {
        "shipped patch false alarm = 1": t.loc[("dinov3_patch", "grid_mean_shipped"), "false_alarm"] == 1.0,
        "shipped patch separation = 0": t.loc[("dinov3_patch", "grid_mean_shipped"), "separation"] == 0.0,
        "shipped siglip separation = 1": t.loc[("siglip", "grid_mean_shipped"), "separation"] == 1.0,
        "pair counts 3/2/4": tuple(t.loc[("siglip", "knn_global"), ["n_self", "n_same", "n_diff"]]) == (3, 2, 4),
        "vmf_split patch detect = 0.25": np.isclose(t.loc[("dinov3_patch", "vmf_split_deepest"), "detect"], 0.25),
        "passers": set(v["passers"]) == {"knn_global", "knn_split_deepest"},
        "winner is the tree passer": v["winner"] == "knn_split_deepest",
        "top is the control (higher min separation)": v["top"] == "knn_global",
        "vmf_loo_mean fails on KS": any("KS" in f for f in v["gates"]["vmf_loo_mean"]["fails"]),
        "P1 replicated": v["predictions"]["P1"] is True,
        "P2 control holds": v["predictions"]["P2"] is True,
        "P3 vmf_split calibrated": v["predictions"]["P3"] is True,
        "P4 knn beats vmf by >= 0.1": v["predictions"]["P4"] is True,
        "fit rows not read as a variant": not any(k[1] == "fit" for k in t.index),
    }
    bad = [k for k, ok in checks.items() if not ok]
    for k, ok in checks.items():
        print(f"{'ok  ' if ok else 'FAIL'} {k}")
    print("SELFTEST", "FAILED" if bad else "PASSED")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
