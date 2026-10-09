"""Where the app said stop, per preset (#3560, #4611): one small table a report can commit.

``analyze.py`` writes each run's ``stops.csv`` (one row per session, 3,456 at 24 seeds, ~1 MB) and the
"Where the app said stop" block of its ``summary.md``, one review set per preset. Neither belongs in a
report directory: the per-session file is too big to commit three times, and the block is per set. This
reduces the three to ``stopping_by_preset.csv``, one row per preset and scope (``all`` and each size band).
That is what the report's stopping section quotes and what the deck's Past the Exit slide draws
(``slides/figs/src/make-autopilot-figs.py``).

The columns follow ``docs/plans/stopping-rules-in-eval.md``: the fire rate first, then the stop (the
Kaplan-Meier median, which carries the sessions that never fired as censored), then the objective at the
stop against the session's own best. Every column after ``fire_rate`` describes only the sessions that
fired. Two more come from the per-session rows, because a median of the paired gain hides its spread:

* ``mean_delta`` / ``se_delta``: the objective at the budget minus at the stop, paired within session.
  The SE is over the per-class means, because a class's sessions share their hard cases.
* ``share_worse`` / ``share_better``: the share of sessions that ended at least ``MARGIN`` below (above)
  where the app said they could stop.

    python stops_by_preset.py --run 0.25=<b025>/analysis-binary --run 1=<b1>/... --run 4=<b4>/... \\
        --out <report>/stopping_by_preset.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calibration"))
import objective  # noqa: E402
import stopping  # noqa: E402

#: How far the objective has to move after the stop for a session to count as ending worse (or better).
MARGIN = 0.02


def preset_rows(stops: pd.DataFrame, beta: float) -> pd.DataFrame:
    """One preset's rows: ``all``, then each size band."""
    metric = objective.OBJECTIVE
    scopes = [("all", stops), *((str(b), g) for b, g in stops.groupby("band", sort=True))]
    out = []
    for scope, g in scopes:
        row = stopping.summarise(g.assign(scope=scope), by=("scope",), metrics=(metric,)).iloc[0].to_dict()
        fired = g[g["stopped"]]
        delta = fired[f"{metric}_delta"].dropna()
        by_class = delta.groupby(fired["category"]).mean()
        row.update(
            beta=beta,
            mean_delta=float(delta.mean()),
            se_delta=float(by_class.std() / np.sqrt(len(by_class))),
            share_worse=float((delta <= -MARGIN).mean()),
            share_better=float((delta >= MARGIN).mean()),
        )
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="BETA=<analysis dir holding stops.csv>")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    frames = []
    for spec in args.run:
        beta, path = spec.split("=", 1)
        stops = pd.read_csv(Path(path) / "stops.csv")
        stops["stopped"] = stops["stopped"].astype(bool)
        frames.append(preset_rows(stops, float(beta)))
    table = pd.concat(frames, ignore_index=True)
    lead = ["beta", "scope", "n_runs", "n_fired", "fire_rate", "km_t_stop"]
    table = table[lead + [c for c in table.columns if c not in lead]]
    table.to_csv(args.out, index=False, float_format="%.4f")
    print(f"wrote {args.out} ({len(table)} rows)")


if __name__ == "__main__":
    main()
