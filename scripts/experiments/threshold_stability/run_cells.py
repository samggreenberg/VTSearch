"""Threshold-stability study: one SLURM-array task per ``(class, seed)`` (issue #2790).

Each task runs **all arms** for its cell (they share the SigLIP 2 / whole embedding
cache, so arms are paired on identical data + startup exemplar), then replays the
baseline arm's frozen trace for the Stage A variance decomposition:

* **Stage B (live loop)** — one ``scripts/sod/sweep.py`` run per arm, with that arm's
  ``--threshold-rule`` / ``--threshold-smooth`` / ``--calibrate-count`` and
  ``--labeling-trace``, into a per-arm out-dir. Includes selection feedback (S5).
* **Stage A (frozen replay)** — ``scripts/sod/replay_thresholds.py`` over the
  ``argmin-k2`` baseline trace, recomputing every arm's rule on the frozen votes
  (exactly paired; no S5). CPU-only, reuses the cache.

Run directly with ``--index N`` for one cell, or via SLURM with
``$SLURM_ARRAY_TASK_ID``; ``--print-cells`` prints the count for the array sizing.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import common

common.setup_env()

import experiment_config as cfg  # noqa: E402

SOD = common.REPO / "scripts" / "sod"


def _sweep_cmd(cls: str, arm: tuple[str, str, str, int], out_dir: Path) -> list[str]:
    _name, rule, smooth, kcount = arm
    return [
        sys.executable,
        str(SOD / "sweep.py"),
        "--datasets",
        cfg.DATASET,
        "--classes",
        cls,
        "--embedders",
        cfg.EMBEDDER,
        "--proposals",
        cfg.PROPOSAL,
        "--iterations",
        str(len(cfg.SEEDS)),  # all seeds 0..N-1 in one sweep
        "--max-labels",
        str(cfg.MAX_LABELS),
        "--neg-multiple",
        str(cfg.NEG_MULTIPLE),
        "--min-box-frac",
        str(cfg.MIN_BOX_FRAC),
        "--inclusion",
        str(cfg.INCLUSION),
        "--calibrate-count",
        str(kcount),
        "--threshold-rule",
        rule,
        "--threshold-smooth",
        smooth,
        "--cache-dir",
        str(common.CACHE_DIR),
        "--out-dir",
        str(out_dir),
        # No --labeling-trace: it renders ~2 PNGs/step (tens of thousands of files,
        # multiple GB) and Stage B needs none of it — results.jsonl already carries
        # per-(seed, t) threshold/cost/oracle_cost, which is everything the analyzer
        # needs. (Stage A replay, which needs the frozen-vote trace.json, is a
        # separate PNG-free follow-up; see THRESHOLD_STABILITY_STATUS.)
    ]


def _run(cmd: list[str]) -> int:
    common.log("$ " + " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode  # noqa: S603 - fixed argv, no shell


def run_cell(cell: dict) -> None:
    cls = cell["cls"]
    cell_dir = common.RESULTS / "cells" / cfg.class_slug(cls)
    cell_dir.mkdir(parents=True, exist_ok=True)

    # Stage B: every arm, all seeds in one sweep each. results.jsonl per arm carries
    # per-(seed, t) threshold/cost/oracle_cost — analyze.py derives the full
    # threshold-stability comparison from it (no trace files, no PNGs, no replay).
    for arm in cfg.ARMS:
        out_dir = cell_dir / f"arm_{arm[0]}"
        _run(_sweep_cmd(cls, arm, out_dir))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Threshold-stability: one cell (class, seed).")
    ap.add_argument("--index", type=int, default=None, help="Cell index; defaults to $SLURM_ARRAY_TASK_ID.")
    ap.add_argument("--print-cells", action="store_true")
    args = ap.parse_args(argv)

    cells = cfg.array_cells()
    if args.print_cells:
        print(len(cells))
        return 0
    idx = args.index if args.index is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if idx >= len(cells):
        common.log(f"index {idx} >= {len(cells)} cells; nothing to do")
        return 0
    cell = cells[idx]
    common.log(f"cell {idx}/{len(cells)}: class={cell['cls']!r} arms={[a[0] for a in cfg.ARMS]}")
    run_cell(cell)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
