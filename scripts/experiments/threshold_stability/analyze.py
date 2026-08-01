"""Aggregate threshold-stability cells into the pre-registered #2790 deliverables.

Concatenates every cell's Stage B ``results.jsonl`` (per-arm live-loop curves) and
Stage A ``replay.csv`` (frozen-trace variance decomposition), then computes the
headline metrics from the plan: per-arm ``sd(Δthreshold)`` and spike incidence over
``t >= 20``, mean regret over ``t in [40, 60]``, and the across-seed cost sd vs the
oracle-cost sd (the ranking-variance floor — threshold noise is the gap). Writes
``results/summary.json``, ``results/agg/*.csv``, and a ``results/REPORT.md`` draft.

Kept deliberately small: the figures + verdict prose are filled in once real Grid
data exists (see ``docs/plans/threshold-stability-experiment.md`` decision rules).
"""

from __future__ import annotations

import json
from pathlib import Path

import common

common.setup_env()

import experiment_config as cfg  # noqa: E402


def _load_stage_a(cells_dir: Path):
    import pandas as pd  # noqa: PLC0415

    files = sorted(cells_dir.glob("*/replay_seed*.csv"))
    frames = []
    for p in files:
        try:
            df = pd.read_csv(p)
        except Exception:  # noqa: BLE001 - empty/partial cell
            continue
        if df.empty:
            continue
        df["cell"] = p.parent.name  # <class_slug>
        df["seed"] = p.stem.removeprefix("replay_seed")  # replay_seed<N>.csv
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_stage_b(cells_dir: Path):
    import pandas as pd  # noqa: PLC0415

    rows = []
    for jl in sorted(cells_dir.glob("*/arm_*/results.jsonl")):
        arm = jl.parent.name.removeprefix("arm_")
        for line in jl.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                rec["arm"] = arm
                rows.append(rec)
    return pd.DataFrame(rows)


def main() -> int:
    cells_dir = common.RESULTS / "cells"
    agg_dir = common.RESULTS / "agg"
    agg_dir.mkdir(parents=True, exist_ok=True)

    stage_a = _load_stage_a(cells_dir)
    stage_b = _load_stage_b(cells_dir)
    summary: dict = {"n_cells_stage_a": int(stage_a["cell"].nunique()) if not stage_a.empty else 0}

    if not stage_a.empty:
        stage_a.to_csv(agg_dir / "stage_a_replay.csv", index=False)
        # Headline: mean sd_threshold + spike_rate per rule over t>=20 (replay already filters warmup).
        by_rule = stage_a.groupby("rule").agg(
            mean_sd_threshold=("sd_threshold", "mean"),
            mean_spike_rate=("spike_rate", "mean"),
        )
        by_rule.to_csv(agg_dir / "stage_a_by_rule.csv")
        summary["stage_a_by_rule"] = by_rule.reset_index().to_dict("records")

    if not stage_b.empty:
        stage_b.to_csv(agg_dir / "stage_b_curves.csv", index=False)
        if {"arm", "t", "cost"} <= set(stage_b.columns):
            late = stage_b[stage_b["t"].between(40, 60)]
            summary["stage_b_mean_cost_late"] = late.groupby("arm")["cost"].mean().reset_index().to_dict("records")

    (common.RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    _write_report(summary)
    common.log(f"summary -> {common.RESULTS / 'summary.json'}; report -> {common.RESULTS / 'REPORT.md'}")
    return 0


def _write_report(summary: dict) -> None:
    lines = [
        "# Threshold-stability study (#2790) — results",
        "",
        "Reframe: `argmin` = the sweep's current (infidelic) rule; `conformal` = production",
        "Autopilot's rule. Arms: " + ", ".join(a[0] for a in cfg.ARMS) + ".",
        "",
        f"Stage A cells replayed: {summary.get('n_cells_stage_a', 0)}",
        "",
        "## Stage A — per-rule threshold noise (mean over cells, t>=20)",
        "",
        "| rule | mean sd(threshold) | mean spike_rate |",
        "|---|---|---|",
    ]
    for r in summary.get("stage_a_by_rule", []):
        lines.append(f"| {r['rule']} | {r['mean_sd_threshold']:.4f} | {r['mean_spike_rate']:.4f} |")
    lines += ["", "_Verdict per the plan's decision rules is filled in once the full Grid run lands._", ""]
    (common.RESULTS / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
