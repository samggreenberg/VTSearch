"""Stage B: the Autopilot voting simulation, one process per cell.

A **cell** is ``(dataset, category, seed)``; every arm in
``experiment_config.ARMS`` runs inside it on the same split and the same seed
sort, so the arms are paired.  Each arm's rows land in
``<results>/<arm>/cells/task_<idx>.csv`` - the layout
``scripts/experiments/calibration/curves.py`` and ``viewer.py`` read, so the
study's figures come from the one shared implementation.

The dataset is loaded once in the parent and inherited by the worker pool
through ``fork``.  Run ``--index N`` for one cell, ``--text-baseline`` for the
click-0 anchor alone, or with neither for the whole grid across
``GPHEAD_WORKERS`` processes.  Cells whose CSV already exists for every arm are
skipped, so a run can be resumed.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# One BLAS thread per worker, pinned BEFORE numpy loads: the pool forks four
# workers and each would otherwise spin up a thread per core, which put the
# load average at 4x the core count on the pilot box and slowed every cell.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import common  # noqa: E402

common.setup_env()

import experiment_config as cfg  # noqa: E402

_DATASETS: dict[str, dict[int, dict]] = {}
_SEED_SCORES: dict[str, dict[str, dict[int, float]]] = {}


def _load_all() -> None:
    from vtscore.datasets.loader_demo import load_demo_dataset
    from vtscore.embedding import initialize_models
    from vtscore.eval.seed_scores import build_seed_scores

    initialize_models()
    prepare = json.loads((common.RESULTS / "prepare_info.json").read_text())
    for ds in cfg.DATASETS:
        medias: dict[int, dict] = {}
        load_demo_dataset(ds, medias, embedder_name=cfg.EMBEDDER)
        _DATASETS[ds] = medias
    cats = {ds: prepare["datasets"][ds]["selected_categories"] for ds in cfg.DATASETS}
    _SEED_SCORES.update(
        build_seed_scores(_DATASETS, media_type=cfg.MEDIA_TYPE, embedder_name=cfg.EMBEDDER, categories=cats)
    )


def _cell_path(arm: str, idx: int) -> Path:
    return common.RESULTS / arm / "cells" / f"task_{idx:04d}.csv"


def run_cell(idx: int, cell: dict, arms: list[str]) -> str:
    """Run every arm of one cell; returns a one-line log."""
    import torch

    from vtscore.eval.voting_iterations import run_voting_iterations_eval

    torch.set_num_threads(1)
    ds, cat, seed = cell["dataset"], cell["category"], cell["seed"]
    clips = _DATASETS[ds]
    seed_scores = {ds: {cat: _SEED_SCORES[ds][cat]}}
    lines = []
    for arm in arms:
        out = _cell_path(arm, idx)
        if out.exists():
            lines.append(f"{arm}=cached")
            continue
        trainer, strategy, safe, cut = cfg.ARMS[arm]
        t0 = time.time()
        df = run_voting_iterations_eval(
            {ds: clips},
            seeds=[seed],
            categories={ds: [cat]},
            inclusion=cfg.INCLUSION,
            sim_fraction=cfg.SIM_FRACTION,
            safe_thresholds=safe,
            calibrate_count=cfg.CALIBRATE_COUNT,
            calibration_fraction=cfg.CALIBRATION_FRACTION,
            strategies=[strategy],
            max_steps=cfg.MAX_STEPS,
            seed_scores=seed_scores,
            trainers=[trainer],
            prevalence_arms=cfg.PREVALENCE_ARMS,
            standalone_cut=cut,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        lines.append(f"{arm}={len(df)}rows/{time.time() - t0:.0f}s")
    return f"cell {idx} {ds}/{cat}/s{seed}: " + " ".join(lines)


def text_baseline(cells: list[dict], out: Path) -> None:
    """The zero-click anchor: the seed sort each cell opened on, cut at its GMM line, scored on the cell's test half.

    Mirrors ``scripts/experiments/calibration/text_baseline.py`` for this
    study's split rule: ``simulate_voting_iterations`` draws
    ``RandomState(seed)`` and, at natural prevalence, the media split is its
    first draw.
    """
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    from vtscore.eval.calibration_metrics import detection_metrics, inclusion_weights
    from vtscore.eval.labels import media_is_positive
    from vtscore.eval.voting_iterations import _split_media_ids
    from vtscore.training.thresholds import calculate_gmm_threshold

    wf, wn = inclusion_weights(cfg.INCLUSION)
    rows = []
    for cell in cells:
        ds, cat, seed = cell["dataset"], cell["category"], cell["seed"]
        clips = _DATASETS[ds]
        sims = _SEED_SCORES[ds][cat]
        gmm_cut = float(calculate_gmm_threshold([float(v) for v in sims.values()]))
        rng = np.random.RandomState(seed)
        _, test_ids = _split_media_ids(clips, cfg.SIM_FRACTION, rng)
        s = np.asarray([sims[i] for i in test_ids], dtype=np.float64)
        y = np.asarray([1 if media_is_positive(clips[i], cat) else 0 for i in test_ids])
        npos, nneg = int(y.sum()), int((1 - y).sum())
        if npos == 0 or nneg == 0:
            continue
        pred = s >= gmm_cut
        fpr = float(((pred == 1) & (y == 0)).sum() / nneg)
        fnr = float(((pred == 0) & (y == 1)).sum() / npos)
        det = detection_metrics(s, y, gmm_cut)
        order = np.argsort(-s)
        ys = y[order]
        tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
        ocost = float(np.min(wf * (fp / nneg) + wn * ((npos - tp) / npos)))
        rows.append(
            {
                "dataset": ds,
                "embedder": cfg.EMBEDDER,
                "category": cat,
                "seed": seed,
                "supports_text": 1,
                "n_test": int(len(test_ids)),
                "n_test_pos": npos,
                "prevalence": round(npos / (npos + nneg), 6),
                "text_gmm_cut": round(gmm_cut, 6),
                "text_cost": round(wf * fpr + wn * fnr, 6),
                "text_fpr": round(fpr, 6),
                "text_fnr": round(fnr, 6),
                "text_precision": round(det["precision"], 6),
                "text_recall": round(det["recall"], 6),
                "text_f1": round(det["f1"], 6),
                "text_oracle_cost": round(ocost, 6),
                "text_AP": round(float(average_precision_score(y, s)), 6),
                "text_auroc": round(float(roc_auc_score(y, s)), 6),
            }
        )
    import pandas as pd

    pd.DataFrame(rows).to_csv(out, index=False)
    common.log(f"wrote {len(rows)} text-baseline rows to {out}")


def _cells() -> list[dict]:
    prepare = json.loads((common.RESULTS / "prepare_info.json").read_text())
    return cfg.cells({ds: prepare["datasets"][ds]["selected_categories"] for ds in cfg.DATASETS})


def _run_one(args: tuple[int, dict, list[str]]) -> str:
    idx, cell, arms = args
    return run_cell(idx, cell, arms)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage B: Autopilot cells for the GP-head pilot.")
    ap.add_argument("--index", type=int, default=None, help="run one cell (default: the whole grid)")
    ap.add_argument("--arms", default=",".join(cfg.STAGE_B_ARMS), help="comma-separated arm names")
    ap.add_argument("--workers", type=int, default=cfg.WORKERS)
    ap.add_argument("--text-baseline", action="store_true", help="write text_baseline.csv and exit")
    ap.add_argument("--print-cells", action="store_true")
    args = ap.parse_args(argv)

    cells = _cells()
    if args.print_cells:
        for i, c in enumerate(cells):
            print(i, c)
        return 0
    arms = [a for a in args.arms.split(",") if a]
    unknown = [a for a in arms if a not in cfg.ARMS]
    if unknown:
        ap.error(f"unknown arms {unknown}; choices: {sorted(cfg.ARMS)}")

    _load_all()
    if args.text_baseline:
        text_baseline(cells, common.RESULTS / "text_baseline.csv")
        return 0
    if args.index is not None:
        common.log(run_cell(args.index, cells[args.index], arms))
        return 0

    # The click-0 anchor is cheap; refresh it with every full run.
    text_baseline(cells, common.RESULTS / "text_baseline.csv")
    jobs = [(i, c, arms) for i, c in enumerate(cells)]
    common.log(f"{len(jobs)} cells x {len(arms)} arms on {args.workers} workers")
    t0 = time.time()
    if args.workers <= 1:
        for job in jobs:
            common.log(_run_one(job))
    else:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(args.workers) as pool:
            for line in pool.imap_unordered(_run_one, jobs):
                common.log(line)
    common.log(f"done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
