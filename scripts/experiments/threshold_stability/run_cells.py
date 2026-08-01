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


def _sweep_cmd(cls: str, seed: int, arm: tuple[str, str, str, int], out_dir: Path) -> list[str]:
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
        str(seed + 1),  # sweep seeds 0..N-1; run just this seed's row via array
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
        "--labeling-trace",
    ]


def _replay_cmd(cls: str, trace_dir: Path, out_csv: Path) -> list[str]:
    return [
        sys.executable,
        str(SOD / "replay_thresholds.py"),
        "--trace-dir",
        str(trace_dir),
        "--cache-dir",
        str(common.CACHE_DIR),
        "--dataset",
        cfg.DATASET,
        "--embedder",
        cfg.EMBEDDER,
        "--class-slug",
        cfg.class_slug(cls),
        "--proposal-slug",
        cfg.PROPOSAL,
        "--rules",
        "argmin,conformal,rank-transfer",
        "--fold-seeds",
        str(cfg.REPLAY_FOLD_SEEDS),
        "--trainer-seeds",
        str(cfg.REPLAY_TRAINER_SEEDS),
        "--inclusion",
        str(cfg.INCLUSION),
        "--out",
        str(out_csv),
    ]


def _run(cmd: list[str]) -> int:
    common.log("$ " + " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode  # noqa: S603 - fixed argv, no shell


def run_cell(cell: dict) -> None:
    cls, seed = cell["cls"], cell["seed"]
    cell_dir = common.RESULTS / "cells" / f"{cfg.class_slug(cls)}__seed{seed}"
    cell_dir.mkdir(parents=True, exist_ok=True)

    baseline_trace: Path | None = None
    for arm in cfg.ARMS:
        name = arm[0]
        out_dir = cell_dir / f"arm_{name}"
        _run(_sweep_cmd(cls, seed, arm, out_dir))
        # The sweep writes labeling_trace/<slug>/<config>/seed<seed>/; find this seed's dir.
        traces = sorted(out_dir.glob(f"labeling_trace/*/*/seed{seed}"))
        if name == cfg.BASELINE_ARM and traces:
            baseline_trace = traces[0]

    if baseline_trace is not None:
        _run(_replay_cmd(cls, baseline_trace, cell_dir / "replay.csv"))
    else:
        common.log(f"WARN: no baseline trace for {cls} seed{seed}; Stage A replay skipped")


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
    common.log(f"cell {idx}/{len(cells)}: class={cell['cls']!r} seed={cell['seed']} arms={[a[0] for a in cfg.ARMS]}")
    run_cell(cell)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
