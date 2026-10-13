#!/usr/bin/env python3
"""#4835: today's app on #4668's build-up sessions, for the deck's "Up and Up" slide.

The slide charts the objective over votes for cross-calibration, the labels line, and today's app, each at the
balance's three presets. The first two are #4668's rungs b1 and b2, measured 2026-10-08. Today's app moves every
week, so this re-runs the build-up's "today's app" rung (``launch_buildup_4668.sh rungs b5_app_b025 b5_app_b1
b5_app_b4`` from a frozen tree on today's dev, ``BUILDUP_BASE`` its own dir) and reads it the way
``analyze_buildup_4668.py`` reads its b6 rung: the objective (F-beta at the preset's beta of the withheld half above
the line the app shows) at every vote, every run in every average, filled with the typed query's own set at today's
per-preset line until the app shows a detector (#4631, #4603).

It adds the point the slide ends on, as Photo Finish does: the objective **after the end-of-run spot check** (the
State of the App's "after the check", drawn at ∞), read as ``state_of_app/analyze.py`` reads it. A run whose last
check row the app shows scores that row's withheld set; any other run keeps its value at the last vote. b1 and b2
ran no check, so only today's app has the point.

    python analyze_upandup_4835.py --base /expscratch/$USER/upandup-4835 \\
        --out docs/experiments/2026-10-13-up-and-up-4835

Writes ``today_curve.csv`` (preset, beta, point, t, mean, se, n: ``point`` is ``vote`` for t = 0..horizon and
``after the check`` for the end) and ``REPORT_machine.md``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _cells_io  # noqa: E402
import curves  # noqa: E402
from _rank_metrics import beta_tag  # noqa: E402
from analyze_buildup_4668 import PRESETS, text_fbeta  # noqa: E402
from analyze_progression_4184 import CELL, baseline_mismatches, fbeta, filled_matrix, summarize  # noqa: E402

#: The pool the build-up's sessions ran on (``CALIB_HAYSTACK_PREVALENCE=0.01``), checked off every rung's rows.
POOL = "haystack_0.01"


def after_check(checks: pd.DataFrame, filled: pd.DataFrame, beta: float) -> pd.Series:
    """Per run, the objective after the end-of-run check: its last check row if the app shows it, else the last vote.

    ``state_of_app/analyze.py``'s rule (#4631, #4653): the check moves the returned set only when the app shows
    the detector it checked; a run still on the typed query's set at its end keeps that set.
    """
    out = filled.iloc[:, -1].copy()
    if checks.empty:
        return out
    last = checks.sort_values("t").groupby(CELL).tail(1).set_index(CELL)
    shows = pd.to_numeric(last["app_trained"], errors="coerce").fillna(0).astype(int) == 1
    scored = pd.Series(fbeta(last["precision"], last["recall"], beta), index=last.index)[shows]
    common_runs = out.index.intersection(scored.index)
    out.loc[common_runs] = scored.loc[common_runs]
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="the BUILDUP_BASE the three b5_app rungs ran under")
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=int, default=150)
    args = ap.parse_args(argv)
    base, out = Path(args.base), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw_bl = pd.read_csv(base / "text_baseline.csv")
    bl = curves.text_sort_baseline(base / "text_baseline.csv")
    grid = bl[CELL].drop_duplicates().reset_index(drop=True)

    rows: list[pd.DataFrame] = []
    lines = ["# #4835 Up and Up: today's app on #4668's sessions - machine summary", ""]
    failures: list[str] = []
    for preset, beta in PRESETS.items():
        arm = f"b5_app_{preset}"
        frame, prov = _cells_io.load_arm(base / arm / "results", keep_check=True)
        lines.append(f"- `{arm}`: {_cells_io.describe_load(prov)}")
        failures += [f"{arm}: {m}" for m in baseline_mismatches(raw_bl, {arm: frame})]
        b = pd.to_numeric(frame["beta"], errors="coerce").dropna().unique()
        if not np.allclose(b, beta):
            failures.append(f"{arm}: rows carry beta {sorted(b)}, expected {beta}")
        pools = sorted({str(v) for v in frame["prevalence_arm"].dropna().unique()})
        if pools != [POOL]:
            failures.append(f"{arm}: pool {pools}, expected {POOL}")
        # Every run in the grid wrote cells, or starved (a run that never found a positive writes a header only
        # and is read as its typed query's set, #4631); any other gap is a lost task, not a run to fill.
        starved = len(prov.get("no_positive_found") or [])
        missing = len(grid) - frame[CELL].drop_duplicates().shape[0] - starved
        if missing > 0:
            failures.append(f"{arm}: {missing} runs of {len(grid)} wrote nothing and did not starve")
        check = _cells_io.check_rows(frame)
        votes, checks = frame[~check].copy(), frame[check]
        votes["objective"] = fbeta(votes["precision"], votes["recall"], beta)
        tag = beta_tag(beta)
        anchors = text_fbeta(bl, f"text_line_precision_{tag}", f"text_line_recall_{tag}", beta)
        filled, _shown = filled_matrix(votes, grid, {}, args.horizon, metric="objective", baseline_metric=anchors)
        rows.append(summarize(filled).assign(preset=preset, beta=beta, point="vote"))
        end = after_check(checks, filled, beta)
        n = int(end.notna().sum())
        rows.append(
            pd.DataFrame(
                {
                    "t": [np.nan],
                    "mean": [float(end.mean())],
                    "se": [float(end.std(ddof=1) / np.sqrt(max(n, 1)))],
                    "n": [n],
                    "preset": [preset],
                    "beta": [beta],
                    "point": ["after the check"],
                }
            )
        )
        checked = int(checks[CELL].drop_duplicates().shape[0])
        lines.append(
            f"  - beta {beta:g}: {len(filled)} runs; mean 1-{args.horizon} {filled.iloc[:, 1:].mean(axis=1).mean():.3f}; "
            f"t=0 {filled[0].mean():.3f}, t=50 {filled[50].mean():.3f}, t={args.horizon} "
            f"{filled[args.horizon].mean():.3f}; after the check {end.mean():.3f} ({checked} runs ran one)"
        )
    curve = pd.concat(rows, ignore_index=True)[["preset", "beta", "point", "t", "mean", "se", "n"]]
    curve.to_csv(out / "today_curve.csv", index=False, float_format="%.6g")
    if failures:
        lines += ["", "## Premise failures", "", *[f"- {f}" for f in failures]]
    (out / "REPORT_machine.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
