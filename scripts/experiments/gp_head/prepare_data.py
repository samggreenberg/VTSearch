"""Stage 0: load + embed each dataset once, then pick the study's categories.

``load_demo_dataset`` caches the SigLIP pickle under the experiment
``VTSEARCH_DATA_DIR``, so every later stage loads from the warm pickle.  Writes
``prepare_info.json`` (per-category counts, the selected categories) beside the
results, which the cell enumerator and the summariser both read.

Usage::

    python prepare_data.py            # cfg.DATASETS
    python prepare_data.py caltech101_m
"""

from __future__ import annotations

import argparse
import json

import common

common.setup_env()

import experiment_config as cfg  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load + embed the datasets for the GP-head pilot.")
    parser.add_argument("datasets", nargs="*", default=cfg.DATASETS)
    args = parser.parse_args(argv)

    from vtscore.datasets.loader_demo import load_demo_dataset
    from vtscore.embedding import initialize_models
    from vtscore.eval.config import EVAL_DATASETS

    initialize_models()

    info: dict[str, object] = {"embedder": cfg.EMBEDDER, "datasets": {}, "config": _config_snapshot()}
    for ds in args.datasets:
        timings: dict[str, float] = {}
        medias: dict[int, dict] = {}
        common.log(f"\n=== {ds} ===")
        with common.timed(f"load:{ds}", timings):
            load_demo_dataset(ds, medias, embedder_name=cfg.EMBEDDER)
        cats: dict[str, int] = {}
        for m in medias.values():
            for c in m.get("categories") or [m.get("category")]:
                if c:
                    cats[c] = cats.get(c, 0) + 1
        query_cats = {q.target_category for q in EVAL_DATASETS.get(ds, {}).get("queries", [])}
        query_counts = {c: k for c, k in cats.items() if c in query_cats}
        selected = cfg.select_categories(query_counts)
        common.log(f"{ds}: {len(medias)} medias, {len(cats)} categories, {len(query_counts)} with a query")
        common.log(f"  selected ({len(selected)}): {[(c, cats[c]) for c in selected]}")
        info["datasets"][ds] = {  # type: ignore[index]
            "n_medias": len(medias),
            "n_categories": len(cats),
            "category_counts": cats,
            "query_category_counts": query_counts,
            "selected_categories": selected,
            "load_seconds": timings.get(f"load:{ds}"),
        }
        del medias

    out = common.RESULTS / "prepare_info.json"
    out.write_text(json.dumps(info, indent=2))
    common.log(f"\nWrote {out}")
    return 0


def _config_snapshot() -> dict:
    return {
        "datasets": cfg.DATASETS,
        "embedder": cfg.EMBEDDER,
        "n_categories": cfg.N_CATEGORIES,
        "seeds": cfg.SEEDS,
        "max_steps": cfg.MAX_STEPS,
        "stage_a_trainers": cfg.STAGE_A_TRAINERS,
        "label_counts": cfg.LABEL_COUNTS,
        "arms": {k: list(v) for k, v in cfg.ARMS.items() if k in cfg.STAGE_B_ARMS},
        "inclusion": cfg.INCLUSION,
        "sim_fraction": cfg.SIM_FRACTION,
        "calibrate_count": cfg.CALIBRATE_COUNT,
        "calibration_fraction": cfg.CALIBRATION_FRACTION,
        "min_category_count": cfg.MIN_CATEGORY_COUNT,
    }


if __name__ == "__main__":
    raise SystemExit(main())
