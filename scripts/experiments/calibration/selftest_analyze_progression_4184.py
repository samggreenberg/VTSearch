#!/usr/bin/env python3
"""Planted-answer self-test for ``analyze_progression_4184.py``.

Builds three rungs' cell files with known costs and checks the analyzer
recovers them:

* every cell's click-0 notch is its text-sort cost, and so is every click
  before the app shows a detector (``app_trained == 0`` at t=3);
* a cell that never found a positive in one rung (a header-only file) is
  filled with its text-sort cost at every click, not dropped - so that rung's
  mean is ``(17 * level + text) / 18``, and the paired step says so exactly;
* a rung that lost a file (zero bytes) drops only that cell from its mean;
* the rung premise is read off the rows: acquisition may leave the reporting
  cut only on ``r7_acq4``, and a mislabelled rule fails the run;
* the figures and the viewer build on the result without crashing.

    python selftest_analyze_progression_4184.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analyze_progression_4184 as A  # noqa: E402

CATS = ["a", "b", "c", "d", "e", "f"]
SEEDS = [0, 1, 2]
HORIZON = 20
TEXT = 0.6
LEVEL = {"r1_xcal": 0.5, "r2_gmm": 0.45, "r7_acq4": 0.4}
STARVED = ("coco_better", "siglip", "c", 1)
#: The curve CSV is written at six significant digits.
TOL = 1e-5


def _rows(rung: str, cat: str, seed: int, *, live: str | None = None) -> pd.DataFrame:
    want = A.RUNGS[rung]
    rows = []
    for t in range(3, HORIZON + 1):
        thr = 0.5
        acq = thr - 0.1 if (want["acq"] and t > 5) else thr
        rows.append(
            {
                "dataset": "coco_better",
                "embedder": "siglip",
                "category": cat,
                "seed": seed,
                "t": t,
                "cost": LEVEL[rung],
                "average_precision": 0.7,
                "app_trained": 0 if t == 3 else 1,
                "threshold": thr,
                "acq_threshold": acq,
                "live_threshold": live or want["live_threshold"],
                "calibration_fraction": want["calibration_fraction"],
                "head": "linear_svm",
                "gmm_variant": "",
                "schedule": "",
                "pool_variant": "max",
                "threshold_provenance": "fold_anchored[2/2]",
                "seed_mode": "text",
            }
        )
    return pd.DataFrame(rows)


def build(root: Path, *, mislabel: bool = False, lose: bool = False) -> str:
    specs = []
    for rung in LEVEL:
        cells = root / rung / "cells"
        cells.mkdir(parents=True, exist_ok=True)
        i = 0
        for cat in CATS:
            for seed in SEEDS:
                f = cells / f"task_{i:04d}.csv"
                i += 1
                df = _rows(rung, cat, seed, live="blend" if (mislabel and rung == "r2_gmm") else None)
                if rung == "r2_gmm" and ("coco_better", "siglip", cat, seed) == STARVED:
                    df.iloc[0:0].to_csv(f, index=False)  # header-only: never found a positive
                elif lose and rung == "r1_xcal" and (cat, seed) == ("f", 2):
                    f.write_bytes(b"")  # died mid-write: data loss
                else:
                    df.to_csv(f, index=False)
        specs.append(f"{root / rung}={rung}")
    base = pd.DataFrame(
        [
            {
                "dataset": "coco_better",
                "embedder": "siglip",
                "category": c,
                "seed": s,
                "supports_text": 1,
                "text_cost": TEXT,
                "text_AP": 0.3,
            }
            for c in CATS
            for s in SEEDS
        ]
    )
    base.to_csv(root / "text_baseline.csv", index=False)
    return ",".join(specs)


def run(**kw) -> tuple[int, Path]:
    tmp = Path(tempfile.mkdtemp(prefix="selftest-prog4184-"))
    arms = build(tmp, **kw)
    out = tmp / "analysis"
    rc = A.main(["--arms", arms, "--baseline", str(tmp / "text_baseline.csv"), "--out", str(out), "--horizon", "20"])
    return rc, out


def check(cond: bool, what: str, failures: list[str]) -> None:
    print(("ok    " if cond else "FAIL  ") + what)
    if not cond:
        failures.append(what)


def main() -> int:
    failures: list[str] = []

    rc, out = run()
    check(rc == 0, "a clean grid passes its premise checks", failures)
    curve = pd.read_csv(out / "progression_curve.csv").set_index(["rung", "t"])
    for rung in LEVEL:
        check(abs(curve.loc[(rung, 0), "mean"] - TEXT) < TOL, f"{rung}: t=0 is the text notch", failures)
        check(abs(curve.loc[(rung, 3), "mean"] - TEXT) < TOL, f"{rung}: an unshown detector reads as text", failures)
    check(abs(curve.loc[("r1_xcal", 10), "mean"] - 0.5) < TOL, "r1 level recovered", failures)
    want_r2 = (17 * 0.45 + TEXT) / 18
    check(
        abs(curve.loc[("r2_gmm", 10), "mean"] - want_r2) < TOL, "starved cell filled with text, not dropped", failures
    )
    check(int(curve.loc[("r2_gmm", 10), "n"]) == 18, "r2 keeps all 18 cells", failures)
    check(abs(curve.loc[("r2_gmm", 10), "coverage"] - 17 / 18) < TOL, "coverage counts the starved cell", failures)
    check(abs(curve.loc[("r1_xcal", 3), "coverage"]) < TOL, "coverage is 0 where no detector is shown", failures)

    pairs = pd.read_csv(out / "paired.csv")
    step = pairs[(pairs["kind"] == "step") & (pairs["from"] == "r1_xcal") & (pairs["at"] == "t=10")].iloc[0]
    want_d = (17 * (0.45 - 0.5) + (TEXT - 0.5)) / 18
    check(abs(step["mean"] - want_d) < TOL, f"paired r1->r2 at t=10 = {want_d:+.4f}", failures)
    check(int(step["n"]) == 18, "paired step is over every cell", failures)
    vs = pairs[(pairs["kind"] == "vs_first") & (pairs["to"] == "r7_acq4") & (pairs["at"] == "t=10")].iloc[0]
    check(abs(vs["mean"] - (0.4 - 0.5)) < TOL, "r7 vs r1 = -0.10", failures)
    figs = sorted(p.name for p in (out / "figures").glob("*.png"))
    check(any(n.startswith("cost_vs_clicks") for n in figs), "the quality-over-clicks figures were written", failures)
    check((out / "viewer.html").exists(), "the viewer was written", failures)

    rc, out = run(lose=True)
    curve = pd.read_csv(out / "progression_curve.csv").set_index(["rung", "t"])
    check(int(curve.loc[("r1_xcal", 10), "n"]) == 17, "a lost file drops only that cell", failures)
    check(int(curve.loc[("r2_gmm", 10), "n"]) == 18, "...and only from the rung that lost it", failures)

    rc, out = run(mislabel=True)
    report = (out / "REPORT_progression.md").read_text()
    check(rc == 1 and "r2_gmm: live_threshold" in report, "a mislabelled rung fails its premise", failures)

    print()
    print("SELFTEST " + ("PASSED" if not failures else f"FAILED ({len(failures)})"))
    return 0 if not failures else 1


if __name__ == "__main__":
    np.seterr(all="ignore")
    sys.exit(main())
