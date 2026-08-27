"""Planted-answer self-test for ``analyze_startup.py`` (issue #3267).

Fabricates arms whose answer is known by construction and asserts the analyzer
recovers it - in particular the four things that would otherwise read as good
news:

* an arm whose opening never left the control's sampling depth must be reported
  as having **measured nothing**, not as "depth does not matter";
* the falsification arm failing to falsify must **withhold** the verdict;
* an arm that mines more positives only because it spent more opening clicks
  must be visible as such against the length-matched control;
* cells whose opening never found both classes must be **counted**, not
  silently dropped - they are the starvation regime the study is about.

Run: ``python selftest_analyze_startup.py``
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import analyze_startup as A  # noqa: E402

N_CAT, N_SEED, N_STEP = 4, 6, 100

#: Per arm: (positives per 100 clicks, opening clicks, opening pick depth,
#: final cost).  `easy_med_hard` is the planted winner - more positives, cost
#: unchanged, and it beats the length-matched control.  `band_wide` mines more
#: only by spending more clicks, so it must NOT beat `flat_mid`.  `incl_k` never
#: moved its opening depth -> must be flagged as having measured nothing.
PLANT = {
    "prod": (18, 7, 0.05, 0.140),
    "top_long": (24, 12, 0.03, 0.139),
    "easy_med_hard": (30, 16, 0.02, 0.138),
    "band_wide": (26, 16, 0.12, 0.141),
    "incl_k": (18, 16, 0.05, 0.140),  # depth stuck at the control's
    "incl_k_wide": (22, 16, 0.03, 0.139),
    "flat_mid": (20, 16, 0.26, 0.140),
    "deep_first": (9, 16, 0.35, 0.145),  # falsifier: fewer positives
}


def _write_arm(root: Path, arm: str, *, starved_cells: int = 0) -> None:
    """One arm's cells, with the planted mining rate and outcome."""
    cells = root / arm / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    positives, open_clicks, depth, cost = PLANT[arm]
    idx = 0
    for cat in range(N_CAT):
        for seed in range(N_SEED):
            rng = np.random.default_rng(hash((arm, cat, seed)) % (2**31))
            starved = idx < starved_cells
            # --- pick log: one row per click, `positives` of them positive ---
            labels = np.zeros(N_STEP, dtype=int)
            hits = rng.choice(N_STEP, size=positives, replace=False)
            labels[hits] = 1
            picks = pd.DataFrame(
                {
                    "seed": seed,
                    "dataset": "ds",
                    "category": f"cat{cat}",
                    "startup_schedule": A.ARM_SCHEDULE[arm],
                    "style": "whole_image",
                    "t": np.arange(1, N_STEP + 1),
                    "phase": ["s0"] * open_clicks + ["hard"] * (N_STEP - open_clicks),
                    "startup_round": [0] * open_clicks + [-1] * (N_STEP - open_clicks),
                    "startup_cut": depth,
                    "startup_cut_percentile": [depth] * open_clicks + [np.nan] * (N_STEP - open_clicks),
                    "picked_id": np.arange(N_STEP),
                    "picked_label": labels,
                    "picked_seed_rank": np.arange(N_STEP),
                    "picked_seed_percentile": np.clip(rng.normal(depth, 0.01, N_STEP), 0, 1),
                    "picked_seed_score": 0.5,
                    "picked_detector_score": 0.5,
                    "acq_threshold": 0.5,
                    "n_good": labels.cumsum(),
                    "n_bad": (1 - labels).cumsum(),
                    "n_pool": N_STEP - np.arange(1, N_STEP + 1),
                    "embedder": "siglip",
                }
            )
            picks.to_csv(cells / f"task_{idx:04d}__picks.csv", index=False)
            # --- main frame: absent for a starved cell (no detector trained) ---
            main = pd.DataFrame(
                {
                    "seed": seed,
                    "dataset": "ds",
                    "category": f"cat{cat}",
                    "t": np.arange(open_clicks + 1, N_STEP + 1),
                    "n_good": 5,
                    "n_bad": 5,
                    "phase": "hard",
                    "gmm_variant": "",
                    "pool_variant": "max",
                    "schedule": "",
                    "cost": cost,
                    "oracle_cost": cost - 0.02,
                    "average_precision": 0.70 + (positives - 18) * 0.004,
                    "embedder": "siglip",
                }
            )
            if starved:
                main = main.iloc[0:0]
            main.to_csv(cells / f"task_{idx:04d}.csv", index=False)
            idx += 1


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{(' - ' + detail) if detail and not ok else ''}")
    return ok


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="gm-selftest-"))
    try:
        root = tmp / "results"
        for arm in A.ARMS:
            _write_arm(root, arm, starved_cells=3 if arm == "deep_first" else 0)
        summary = A.analyze(root, tmp / "analysis")

        ok = True
        print("planted-answer checks:")

        # 1. The stuck lever is called out rather than reported as a null.
        ok &= _check(
            "incl_k is flagged as having measured nothing",
            not summary["arms"]["incl_k"]["lever"]["moved"],
            str(summary["arms"]["incl_k"]["lever"]),
        )
        ok &= _check(
            "every arm that did move is recognised as having moved",
            all(summary["arms"][a]["lever"]["moved"] for a in ("top_long", "easy_med_hard", "band_wide", "flat_mid")),
        )

        # 2. The planted winner is found, with the right sign on every leg.
        emh = summary["arms"]["easy_med_hard"]
        ok &= _check("easy_med_hard mines more positives", emh["positives_100"]["median_delta"] > 0)
        ok &= _check("easy_med_hard holds cost", emh["final_cost"]["ci95_hi"] <= A.COST_REGRESSION_TOLERANCE)
        ok &= _check(
            "easy_med_hard beats the length-matched control",
            emh["vs_length_control"]["positives_100"]["median_delta"] > 0,
        )
        ok &= _check("the verdict names it", "easy_med_hard" in summary["verdict"], summary["verdict"])

        # 3. A click-budget win is not mistaken for a depth win.
        bw = summary["arms"]["band_wide"]
        ok &= _check("band_wide beats prod on positives", bw["positives_100"]["median_delta"] > 0)
        ok &= _check(
            "band_wide's edge over prod does NOT survive the length-matched control",
            bw["vs_length_control"]["positives_100"]["median_delta"]
            < emh["vs_length_control"]["positives_100"]["median_delta"],
        )

        # 4. Starved cells are counted, not dropped in silence.
        ok &= _check(
            "cells that trained no detector are counted",
            len(summary["provenance"]["deep_first"]["main"]["no_positive_found"]) == 3,
            str(summary["provenance"]["deep_first"]["main"]),
        )

        # 5. A falsifier that fails to falsify withholds the verdict.
        broken = dict(summary)
        broken["arms"] = dict(summary["arms"])
        broken["arms"][A.FALSIFIER] = dict(summary["arms"][A.FALSIFIER])
        broken["arms"][A.FALSIFIER]["positives_100"] = {"n_pairs": 24, "median_delta": +4.0}
        ok &= _check("a non-falsifying falsifier withholds the verdict", A.verdict(broken).startswith("WITHHELD"))

        # 6. The report is written and says what it found.
        report = (tmp / "analysis" / "REPORT_startup.md").read_text()
        ok &= _check("report names every arm", all(f"`{a}`" in report for a in A.ARMS))
        ok &= _check("report flags the stuck lever", "**no**" in report)

        print("\nSELFTEST", "PASSED" if ok else "FAILED")
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
