"""Stage A: the label-curve sweep - each estimator given N labels, no voting.

Isolates the *head* question from the acquisition question: every trainer sees
the same N labels (balanced, drawn from the simulation half) and scores the
same held-out half.  Ranking (AUROC, AP, best F1), the production-path F1 at
the cross-calibrated cut, and the two probability-calibration diagnostics
(Brier, ECE) come out per cell.  Writes ``stage_a.csv`` beside the results.
"""

from __future__ import annotations

import argparse
import json

import common

common.setup_env()

import experiment_config as cfg  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage A: label-curve sweep for the GP-head pilot.")
    parser.add_argument("--out", default=str(common.RESULTS / "stage_a.csv"))
    args = parser.parse_args(argv)

    from vtscore.datasets.loader_demo import load_demo_dataset
    from vtscore.embedding import initialize_models
    from vtscore.eval.label_curve import run_label_curve_eval

    prepare = json.loads((common.RESULTS / "prepare_info.json").read_text())
    initialize_models()

    dataset_clips: dict[str, dict[int, dict]] = {}
    categories: dict[str, list[str]] = {}
    for ds in cfg.DATASETS:
        medias: dict[int, dict] = {}
        load_demo_dataset(ds, medias, embedder_name=cfg.EMBEDDER)
        dataset_clips[ds] = medias
        categories[ds] = prepare["datasets"][ds]["selected_categories"]

    df = run_label_curve_eval(
        dataset_clips=dataset_clips,
        trainers=cfg.STAGE_A_TRAINERS,
        label_counts=cfg.LABEL_COUNTS,
        seeds=cfg.SEEDS,
        categories=categories,
        sim_fraction=cfg.SIM_FRACTION,
        inclusion_value=cfg.INCLUSION,
        calibrate_count=cfg.CALIBRATE_COUNT,
        cal_fraction=0.3 if cfg.CALIBRATION_FRACTION is None else cfg.CALIBRATION_FRACTION,
        progress=True,
    )
    df.to_csv(args.out, index=False)
    common.log(f"wrote {len(df)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
