#!/usr/bin/env python
"""Pair two Rico Suave analyses session for session (#4356): an arm against the app.

Reads the ``runs.csv`` that ``analyze.py`` writes for each, keeps the rico_suave rows, and pairs them
on (preset, class, seed) -- the same cell, so the same split and the same opening. Writes
``pair.md`` beside ``--out``: the mean difference (arm - control) of the session mean F-beta over
votes 1-150 and 1-50, with its standard error over sessions and the share of sessions the arm wins,
per preset and per class.

Usage::

    python pair.py --control <analysis>/runs.csv --arm <analysis>/runs.csv --label "no hold" --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["preset", "category", "seed"]
METRICS = {"fbeta_mean": "1-150", "fbeta_mean_short": "1-50"}


def _se(d: pd.Series) -> float:
    return float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--control", type=Path, required=True)
    ap.add_argument("--arm", type=Path, required=True)
    ap.add_argument("--label", default="arm")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    ctl = pd.read_csv(args.control)
    arm = pd.read_csv(args.arm)
    ctl, arm = ctl[ctl["dataset"] == "rico_suave"], arm[arm["dataset"] == "rico_suave"]
    both = ctl.merge(arm, on=KEY, suffixes=("_ctl", "_arm"))
    if len(both) != len(ctl) or len(both) != len(arm):
        raise SystemExit(f"unpaired sessions: control {len(ctl)}, arm {len(arm)}, paired {len(both)}")
    for m in METRICS:
        both[f"d_{m}"] = both[f"{m}_arm"] - both[f"{m}_ctl"]

    lines = [
        f"# {args.label} - the app, paired session for session (#4356)",
        "",
        f"{len(both)} sessions. Difference = {args.label} minus the app, in session mean F-beta.",
        "",
        "| preset | sessions | app (1-150) | difference (1-150) | arm wins | app (1-50) | difference (1-50) | arm wins |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p, g in both.groupby("preset", sort=False):
        cells = []
        for m in METRICS:
            d = g[f"d_{m}"]
            cells += [f"{g[f'{m}_ctl'].mean():.3f}", f"{d.mean():+.3f} ± {_se(d):.3f}", f"{(d > 1e-9).mean():.0%}"]
        lines.append(f"| {p} | {len(g)} | " + " | ".join(cells) + " |")
    presets = list(dict.fromkeys(both["preset"]))
    lines += [
        "",
        "Per class (difference over votes 1-50, mean over seeds):",
        "",
        "| class | " + " | ".join(presets) + " |",
        "|---|" + "---|" * len(presets),
    ]
    for c, g in both.groupby("category"):
        lines.append(
            f"| {c} | "
            + " | ".join(f"{g.loc[g['preset'] == p, 'd_fbeta_mean_short'].mean():+.3f}" for p in presets)
            + " |"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pair.md").write_text("\n".join(lines) + "\n")
    print((args.out / "pair.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
