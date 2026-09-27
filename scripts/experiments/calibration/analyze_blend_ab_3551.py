#!/usr/bin/env python3
"""#3551 A/B analyzer: full trajectories under a promoted fallback schedule vs the shipped one.

Each arm dir (``launch_blend_3551.sh ab``) is one independent run of the whole
grid with ``CALIB_BLEND_SCHEDULE`` pinned.  Arms diverge from the first step
their fallback cuts differ, so this is the only measurement that sees the
acquisition feedback the screen cannot.

Pairing is on the cell - (environment, category, seed, calibration draw) - and
the unit is the cell's trajectory-mean cost, per weighting.  The pre-registered
ship rule (``docs/experiments/2026-09-22-blend-endpoints-3551/PLAN.md``):

1. within-mode pooled 95% upper bound on Δcost below 0;
2. every environment's 95% upper bound below +0.01;
3. neither reweighting resolvably (> 2 SE) worse in any environment.

    python analyze_blend_ab_3551.py --control DIR --arm DIR [--arm DIR ...] --out DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _cells_io  # noqa: E402
from analyze_blend_3551 import WEIGHTS, mode_of  # noqa: E402

CELL = ["dataset", "embedder", "category", "seed", "calibration_seed"]


def load_arm(results: Path) -> tuple[pd.DataFrame, dict]:
    """Per-cell trajectory means of each weighted cost, from the base rows only."""
    frame, prov = _cells_io.load_cells(results / "cells", per_file=_cells_io._base_rows)
    if frame.empty:
        return frame, prov
    frame["calibration_seed"] = frame["calibration_seed"].fillna(-1).astype(int)
    for w, (wf, wn) in WEIGHTS.items():
        frame[f"c_{w}"] = wf * frame["fpr"] + wn * frame["fnr"]
    cols = [f"c_{w}" for w in WEIGHTS] + ["average_precision"]
    g = frame.groupby(CELL)[cols].mean().reset_index()
    g["n_steps"] = frame.groupby(CELL).size().to_numpy()
    return g, prov


def compare(control: pd.DataFrame, arm: pd.DataFrame) -> pd.DataFrame:
    """Per-cell paired differences (arm - control), inner-joined on the cell."""
    m = arm.merge(control, on=CELL, suffixes=("", "_ctl"), how="inner")
    out = m[CELL].copy()
    out["mode"] = out["embedder"].map(mode_of)
    out["env"] = out["dataset"] + " / " + out["mode"]
    for w in WEIGHTS:
        out[f"d_{w}"] = m[f"c_{w}"] - m[f"c_{w}_ctl"]
    out["d_ap"] = m["average_precision"] - m["average_precision_ctl"]
    return out


def _stats(x: np.ndarray) -> dict:
    n = len(x)
    mean = float(x.mean()) if n else float("nan")
    se = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"n": n, "mean": mean, "se": se, "ci_hi": mean + 1.96 * se, "resolvable": bool(n > 1 and abs(mean) > 2 * se)}


def ship_rule(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per-env table and the three-clause verdict for one arm (one mode)."""
    rows = []
    for env, e in d.groupby("env"):
        for w in WEIGHTS:
            rows.append({"env": env, "weighting": w, **_stats(e[f"d_{w}"].to_numpy(float))})
        rows.append({"env": env, "weighting": "AP (higher=better)", **_stats(e["d_ap"].to_numpy(float))})
    t = pd.DataFrame(rows)
    pooled = _stats(d["d_1:1"].to_numpy(float))
    one = t[t["weighting"] == "1:1"]
    rew = t[t["weighting"].isin(["fpr x4", "fnr x4"])]
    verdict = {
        "pooled": pooled,
        "clause1_pooled_ci_below_0": bool(pooled["ci_hi"] < 0),
        "clause2_every_env_ci_below_+0.01": bool((one["ci_hi"] < 0.01).all()),
        "clause3_no_reweighting_resolvably_worse": bool(not ((rew["mean"] > 0) & rew["resolvable"]).any()),
    }
    verdict["ships"] = all(verdict[k] for k in verdict if k.startswith("clause"))
    return t, verdict


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--control", required=True)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    control, cprov = load_arm(Path(args.control))
    lines = ["# #3551 A/B — machine summary", "", f"Control `{args.control}`: {_cells_io.describe_load(cprov)}", ""]
    verdicts = {}
    for arm_dir in args.arm:
        arm, aprov = load_arm(Path(arm_dir))
        d = compare(control, arm)
        name = Path(arm_dir).parent.name if Path(arm_dir).name == "results" else Path(arm_dir).name
        d.to_csv(out / f"paired_{name}.csv", index=False)
        t, v = ship_rule(d)
        t.to_csv(out / f"table_{name}.csv", index=False)
        verdicts[name] = v
        lines += [f"## {name}", "", f"Arm: {_cells_io.describe_load(aprov)}; paired cells: {len(d)}", ""]
        lines += ["| env | weighting | n | Δ mean | SE | 95% hi |", "|---|---|---|---|---|---|"]
        for r in t.itertuples(index=False):
            star = "**" if r.resolvable else ""
            lines.append(
                f"| {r.env} | {r.weighting} | {r.n} | {star}{r.mean:+.2g}{star} | {r.se:.2g} | {r.ci_hi:+.2g} |"
            )
        lines += ["", f"Verdict: `{json.dumps(v)}`", ""]
    (out / "verdicts.json").write_text(json.dumps(verdicts, indent=2))
    (out / "REPORT_ab.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
