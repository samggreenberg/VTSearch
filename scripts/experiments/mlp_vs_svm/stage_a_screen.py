"""Stage A: SVM kernel/hyperparameter screen (cheap; decides which SVMs advance).

Runs the label-count sweep (random balanced labels — *not* autopilot; this stage
only prunes the SVM configuration space) over a widened trainer grid on two
datasets, and writes ``stage_a.csv``.  The report then keeps the best config per
kernel family by mean AUROC in the autopilot-relevant label regime.

This is deliberately a static train/test sweep: it is fast and only needs to rank
SVM configurations relative to each other, not decide the winner (that is Stage B).
"""

from __future__ import annotations

import argparse

import common

common.setup_env()

# The widened SVM configuration grid from the plan, plus the MLP baseline.
SCREEN_TRAINERS = [
    "mlp",
    *[f"svm_linear@C={c}" for c in (0.03, 0.3, 1, 3, 30)],
    *[f"svm_rbf@C={c},gamma={g}" for c in (0.3, 1, 3, 30) for g in ("scale", "4x", "0.25x")],
    *[f"svm_poly@degree={d},C={c}" for d in (2, 3) for c in (0.3, 3)],
    *[f"svm_sigmoid@C={c}" for c in (0.3, 3)],
]

SCREEN_DATASETS = ["caltech256_a", "visual_genome_m"]
LABEL_COUNTS = [5, 10, 20, 50, 100, 200]
SEEDS = list(range(10))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage A: SVM kernel/hyperparameter screen.")
    parser.add_argument("--datasets", nargs="+", default=SCREEN_DATASETS)
    parser.add_argument("--n-categories", type=int, default=6)
    parser.add_argument("--output", default=str(common.RESULTS / "stage_a.csv"))
    args = parser.parse_args(argv)

    from vtscore.datasets.loader_demo import load_demo_dataset
    from vtscore.embedding import initialize_models
    from vtscore.eval.label_curve import run_label_curve_eval

    import experiment_config as cfg

    initialize_models()

    dataset_clips: dict[str, dict] = {}
    categories: dict[str, list[str]] = {}
    for ds in args.datasets:
        medias: dict[int, dict] = {}
        common.log(f"loading {ds} ...")
        load_demo_dataset(ds, medias, embedder_name=cfg.EMBEDDER)
        dataset_clips[ds] = medias
        counts: dict[str, int] = {}
        for m in medias.values():
            for c in m.get("categories") or [m.get("category")]:
                if c:
                    counts[c] = counts.get(c, 0) + 1
        categories[ds] = cfg.select_categories(counts, args.n_categories)
        common.log(f"  {ds}: {len(medias)} medias, screen categories={categories[ds]}")

    common.log(f"\nScreening {len(SCREEN_TRAINERS)} trainer configs over {len(SEEDS)} seeds ...")
    df = run_label_curve_eval(
        dataset_clips=dataset_clips,
        trainers=SCREEN_TRAINERS,
        label_counts=LABEL_COUNTS,
        seeds=SEEDS,
        categories=categories,
        sim_fraction=cfg.SIM_FRACTION,
        inclusion_value=cfg.INCLUSION,
        calibrate_count=cfg.CALIBRATE_COUNT,
        cal_fraction=cfg.CALIBRATION_FRACTION,
        progress=True,
    )
    df.to_csv(args.output, index=False)
    common.log(f"\nWrote {len(df)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
