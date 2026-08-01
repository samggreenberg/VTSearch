"""Aggregate the threshold-stability Stage B cells into the #2790 deliverables.

Reads every arm's ``results.jsonl`` (one row per (seed, t), carrying ``threshold``,
``cost``, ``oracle_cost``) and derives the pre-registered comparison — no trace
files, no replay needed:

* ``sd_threshold`` — across-seed spread of the trained threshold, averaged over the
  post-ramp window ``t >= WARMUP``. The threshold noise the study wants shrunk.
* ``sd_dthreshold`` — within-seed step-to-step |Δthreshold| spread (jumpiness).
* ``spike_rate`` — fraction of (seed, t) steps with ``|Δcost| > 0.1`` (the #2790
  single-step cost excursions).
* ``cost_sd`` vs ``oracle_sd`` — across-seed sd of cost vs of the oracle cost (the
  ranking-variance floor); the gap between them is threshold noise (plan H5).
* ``mean_regret`` — ``cost - oracle_cost`` averaged over ``t in [40, 60]``.

Writes ``results/summary.json``, ``results/agg/by_arm.csv``, and a ``REPORT.md``
ranking the arms. (rank-transfer runs identically to conformal in the *live* loop —
its rank remap needs the final pool scores, applied only in Stage-A replay — so it
is reported but expected to match conformal here.)
"""

from __future__ import annotations

import json

import common

common.setup_env()

import experiment_config as cfg  # noqa: E402

WARMUP = int(__import__("os").environ.get("THRSTAB_WARMUP_T", "20"))


def _load():
    import pandas as pd  # noqa: PLC0415

    frames = []
    for jl in sorted((common.RESULTS / "cells").glob("*/arm_*/results.jsonl")):
        arm = jl.parent.name.removeprefix("arm_")
        cls = jl.parent.parent.name
        rows = [json.loads(x) for x in jl.read_text().splitlines() if x.strip()]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["arm"] = arm
        df["cls"] = cls
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _per_arm(df):
    df = df.sort_values(["arm", "cls", "seed", "t"])
    # Within-seed step-to-step threshold jump.
    df["dthreshold"] = df.groupby(["arm", "cls", "seed"])["threshold"].diff().abs()
    df["dcost"] = df.groupby(["arm", "cls", "seed"])["cost"].diff().abs()
    df["spike"] = (df["dcost"] > 0.1).astype(float)
    df["regret"] = df["cost"] - df["oracle_cost"]

    warm = df[df["t"] >= WARMUP]
    # Across-seed spread of threshold / cost / oracle at each (arm, cls, t), then mean.
    grp = warm.groupby(["arm", "cls", "t"])
    spread = grp.agg(
        sd_threshold=("threshold", "std"),
        cost_sd=("cost", "std"),
        oracle_sd=("oracle_cost", "std"),
    ).reset_index()
    by_spread = spread.groupby("arm").agg(
        sd_threshold=("sd_threshold", "mean"),
        cost_sd=("cost_sd", "mean"),
        oracle_sd=("oracle_sd", "mean"),
    )

    late = df[(df["t"] >= 40) & (df["t"] <= 60)]
    out = by_spread.copy()
    out["sd_dthreshold"] = warm.groupby("arm")["dthreshold"].std()
    out["spike_rate"] = warm.groupby("arm")["spike"].mean()
    out["mean_regret"] = late.groupby("arm")["regret"].mean()
    # Threshold-noise share of cost variance: how much of cost_sd the ranking floor
    # (oracle_sd) does NOT explain.
    out["cost_noise_gap"] = (out["cost_sd"] - out["oracle_sd"]).clip(lower=0)
    return out.reindex([a[0] for a in cfg.ARMS]).reset_index().rename(columns={"index": "arm"})


def main() -> int:
    df = _load()
    agg_dir = common.RESULTS / "agg"
    agg_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        (common.RESULTS / "REPORT.md").write_text("# Threshold-stability (#2790)\n\nNo cell output found.\n")
        common.log("no results found")
        return 0

    by_arm = _per_arm(df)
    by_arm.to_csv(agg_dir / "by_arm.csv", index=False)
    n_cells = df.groupby(["arm", "cls"]).ngroups
    summary = {
        "warmup_t": WARMUP,
        "arm_cells": int(n_cells),
        "classes": sorted(df["cls"].unique().tolist()),
        "seeds": sorted(int(s) for s in df["seed"].unique().tolist()),
        "by_arm": by_arm.to_dict("records"),
    }
    (common.RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    _report(by_arm, summary)
    common.log(f"wrote REPORT.md + summary.json ({len(df)} rows, arms={by_arm['arm'].tolist()})")
    return 0


def _report(by_arm, summary: dict) -> None:
    base = by_arm[by_arm["arm"] == cfg.BASELINE_ARM]

    def _fmt(x):
        return "n/a" if x != x else f"{x:.4f}"  # noqa: PLR0124 - NaN check

    lines = [
        "# Threshold-stability study (#2790) — Stage B (live loop)",
        "",
        f"Classes: {', '.join(summary['classes'])}. Seeds: {len(summary['seeds'])}. "
        f"Window: t >= {summary['warmup_t']}.",
        "",
        "Reframe: `argmin` = the sweep's current (infidelic) rule; `conformal` = production",
        "Autopilot's rule. Lower `sd_threshold` / `spike_rate` / `cost_sd` is better;",
        "`oracle_sd` is the ranking-variance floor (threshold noise is `cost_sd - oracle_sd`).",
        "",
        "| arm | sd_threshold | sd_Δthreshold | spike_rate | cost_sd | oracle_sd | mean_regret |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in by_arm.to_dict("records"):
        lines.append(
            f"| {r['arm']} | {_fmt(r['sd_threshold'])} | {_fmt(r['sd_dthreshold'])} | "
            f"{_fmt(r['spike_rate'])} | {_fmt(r['cost_sd'])} | {_fmt(r['oracle_sd'])} | {_fmt(r['mean_regret'])} |"
        )
    lines.append("")
    if not base.empty:
        b = base.iloc[0]
        best = by_arm.loc[by_arm["sd_threshold"].idxmin()]
        lines += [
            f"**Baseline** `{cfg.BASELINE_ARM}`: sd_threshold={_fmt(b['sd_threshold'])}, "
            f"spike_rate={_fmt(b['spike_rate'])}.",
            f"**Lowest threshold noise:** `{best['arm']}` "
            f"(sd_threshold={_fmt(best['sd_threshold'])}, spike_rate={_fmt(best['spike_rate'])}).",
            "",
            "Decision rules (plan): adopt the cheapest arm cutting spike incidence >=80% and",
            "across-seed cost sd >=50% vs argmin-k2 without worsening mean regret by >0.01.",
        ]
    (common.RESULTS / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
