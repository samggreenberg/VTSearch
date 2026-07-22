"""Stage B: the definitive Autopilot voting run (one SLURM-array task per cell).

Each array task handles exactly one ``(dataset, category, prevalence_arm, seed)``
cell and runs *all* trainers inside it (they share the one loaded dataset), so
the MLP and SVM trajectories are paired on identical data.  The cell for this
task is ``array_cells(...)[SLURM_ARRAY_TASK_ID]`` — a stable enumeration derived
from ``prepare_info.json`` (so every task agrees on the mapping without
coordinating).  Results land in ``results/stage_b/task_<idx>.csv``; the
summariser concatenates whatever tasks completed, so a few stragglers or a
trimmed grid still produce a valid (clearly-annotated) report.

Run directly with ``--index N`` for a single cell, or via SLURM with
``$SLURM_ARRAY_TASK_ID``.
"""

from __future__ import annotations

import argparse
import json
import os

import common

common.setup_env()


def _categories_by_dataset(prepare_info: dict, cfg) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for ds in cfg.DATASETS:
        counts = prepare_info["datasets"].get(ds, {}).get("category_counts", {})
        out[ds] = cfg.select_categories(counts, cfg.N_CATEGORIES)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage B: one Autopilot cell (dataset,category,arm,seed).")
    parser.add_argument("--index", type=int, default=None, help="Cell index; defaults to $SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--outdir", default=str(common.RESULTS / "stage_b"))
    args = parser.parse_args(argv)

    idx = args.index if args.index is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

    from vtscore.datasets.loader_demo import load_demo_dataset
    from vtscore.embedding import initialize_models
    from vtscore.eval.seed_scores import build_seed_scores
    from vtscore.eval.voting_iterations import run_voting_iterations_eval

    import experiment_config as cfg

    prepare_info = json.loads((common.RESULTS / "prepare_info.json").read_text())
    cats_by_ds = _categories_by_dataset(prepare_info, cfg)
    cells = cfg.array_cells(cats_by_ds)
    if idx >= len(cells):
        common.log(f"index {idx} >= {len(cells)} cells; nothing to do")
        return 0
    cell = cells[idx]
    ds, cat, arm, seed = cell["dataset"], cell["category"], cell["arm"], cell["seed"]
    common.log(f"cell {idx}/{len(cells)}: dataset={ds} category={cat} arm={arm} seed={seed}")

    initialize_models()

    medias: dict[int, dict] = {}
    load_demo_dataset(ds, medias, embedder_name=cfg.EMBEDDER)

    # Text-sort seed scores for this category (the Autopilot seed the user gets
    # after typing the query).
    seed_scores = build_seed_scores(
        {ds: medias}, media_type=cfg.MEDIA_TYPE, embedder_name=cfg.EMBEDDER, categories={ds: [cat]}
    )

    df = run_voting_iterations_eval(
        {ds: medias},
        seeds=[seed],
        categories={ds: [cat]},
        inclusion=cfg.INCLUSION,
        sim_fraction=cfg.SIM_FRACTION,
        safe_thresholds=cfg.SAFE_THRESHOLDS,
        calibrate_count=cfg.CALIBRATE_COUNT,
        calibration_fraction=cfg.CALIBRATION_FRACTION,
        max_steps=cfg.MAX_STEPS,
        seed_scores=seed_scores,
        trainers=cfg.STAGE_B_TRAINERS,
        prevalence_arms=[arm],
    )

    outdir = common.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"task_{idx:04d}.csv"
    df.to_csv(out, index=False)
    common.log(f"wrote {len(df)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
