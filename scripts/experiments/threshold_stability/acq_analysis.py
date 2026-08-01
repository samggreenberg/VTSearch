"""Compare sparse-positive-fix arms: instability vs. the convergence cost (#2790).

Each arm is a sweep out-dir (e.g. ``acq/gts6`` for ``good_to_start=6``, or a
``defer`` dir) holding one ``results.jsonl`` (all class/seed/t rows) and a
``labeling_trace/`` tree. For each arm this reports, side by side:

* **instability** — ``spike_rate`` (|Δcost|>0.1, t≥20), across-seed ``cost_sd``
  (t≥20), and the **sparse-positive spike share** (from spike_analysis);
* **convergence cost** (the tradeoff) — ``cost_auc`` (mean test cost over the whole
  trajectory), ``final_cost`` (last t), and **votes_in_good** (mean steps spent in
  the ``good`` phase before the first bad/hard vote — how much of the budget the
  intervention spends collecting positives);
* **accuracy** — ``mean_regret`` (t∈[40,60]).

Usage: ``python acq_analysis.py --root <dir of arm subdirs> [--arms gts3,gts6,...]``
Prints a table and writes ``acq_summary.csv`` under --root.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from spike_analysis import collect_spikes  # sibling module


def _load_results(arm_dir: Path) -> list[dict]:
    jl = arm_dir / "results.jsonl"
    if not jl.exists():
        # sweep may nest results under the out-dir; find the first results.jsonl.
        found = sorted(arm_dir.rglob("results.jsonl"))
        if not found:
            return []
        jl = found[0]
    return [json.loads(x) for x in jl.read_text().splitlines() if x.strip()]


def _spike_rate_and_costs(rows: list[dict]) -> dict:
    # Group per (class, seed) trajectory.
    traj: dict[tuple, list[dict]] = {}
    for r in rows:
        traj.setdefault((r.get("dataset", ""), r.get("category", r.get("cls", "")), r["seed"]), []).append(r)
    spikes = tot = 0
    cost_auc: list[float] = []
    final_cost: list[float] = []
    regret: list[float] = []
    for series in traj.values():
        series.sort(key=lambda r: r["t"])
        costs = [float(r["cost"]) for r in series]
        cost_auc.append(statistics.fmean(costs))
        final_cost.append(costs[-1])
        for r in series:
            if 40 <= r["t"] <= 60:
                regret.append(float(r["cost"]) - float(r.get("oracle_cost", 0.0)))
        warm = [r for r in series if r["t"] >= 20]
        for a, b in zip(warm, warm[1:], strict=False):
            tot += 1
            if abs(float(b["cost"]) - float(a["cost"])) > 0.1:
                spikes += 1
    # Across-seed cost sd at each (class, t>=20), then mean.
    by_ct: dict[tuple, list[float]] = {}
    for r in rows:
        if r["t"] >= 20:
            by_ct.setdefault((r.get("category", r.get("cls", "")), r["t"]), []).append(float(r["cost"]))
    cost_sds = [statistics.pstdev(v) for v in by_ct.values() if len(v) > 1]
    return {
        "spike_rate": spikes / tot if tot else float("nan"),
        "cost_sd": statistics.fmean(cost_sds) if cost_sds else float("nan"),
        "cost_auc": statistics.fmean(cost_auc) if cost_auc else float("nan"),
        "final_cost": statistics.fmean(final_cost) if final_cost else float("nan"),
        "mean_regret": statistics.fmean(regret) if regret else float("nan"),
        "n_cells": len(traj),
    }


def _votes_in_good(arm_dir: Path) -> float:
    """Mean steps spent in the ``good`` phase per trace (before the first bad/hard)."""
    vals: list[int] = []
    for tj in sorted(arm_dir.rglob("trace.json")):
        trace = json.loads(tj.read_text())
        vals.append(sum(1 for e in trace if e.get("phase") == "good"))
    return statistics.fmean(vals) if vals else float("nan")


def _sparse_share(arm_dir: Path) -> dict:
    rows = collect_spikes(arm_dir, 0.1)
    if not rows:
        return {"n_spikes": 0, "sparse_share": float("nan"), "runaway_share": float("nan")}
    n = len(rows)
    sparse = sum(1 for r in rows if (r["n_good"] or 0) <= 6)
    runaway = sum(1 for r in rows if isinstance(r.get("d_fnr"), (int, float)) and r["d_fnr"] > 0.2)
    return {"n_spikes": n, "sparse_share": sparse / n, "runaway_share": runaway / n}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sparse-positive-fix arm comparison (#2790).")
    ap.add_argument("--root", required=True, help="Dir containing per-arm subdirs.")
    ap.add_argument("--arms", default=None, help="Comma list of arm subdir names; default = all subdirs.")
    args = ap.parse_args(argv)
    root = Path(args.root)
    arms = args.arms.split(",") if args.arms else sorted(p.name for p in root.iterdir() if p.is_dir())

    out: list[dict] = []
    for arm in arms:
        d = root / arm
        rows = _load_results(d)
        if not rows:
            continue
        rec = {"arm": arm, **_spike_rate_and_costs(rows), "votes_in_good": _votes_in_good(d), **_sparse_share(d)}
        out.append(rec)

    cols = [
        "arm", "spike_rate", "cost_sd", "sparse_share", "runaway_share",
        "cost_auc", "final_cost", "votes_in_good", "mean_regret", "n_spikes", "n_cells",
    ]  # fmt: skip
    with (root / "acq_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)

    def f(x):
        return "n/a" if not isinstance(x, (int, float)) or x != x else (f"{x:.4f}" if abs(x) < 100 else f"{x:.1f}")

    print("arm       spike_rate cost_sd sparse% runaway% cost_auc final_cost votes_good regret n_spk")
    for r in out:
        print(
            f"{r['arm']:<9} {f(r['spike_rate']):>9} {f(r['cost_sd']):>7} {f(r['sparse_share']):>7} "
            f"{f(r['runaway_share']):>8} {f(r['cost_auc']):>8} {f(r['final_cost']):>10} "
            f"{f(r['votes_in_good']):>10} {f(r['mean_regret']):>6} {r['n_spikes']:>5}"
        )
    print(f"\n[{len(out)} arms -> {root / 'acq_summary.csv'}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
