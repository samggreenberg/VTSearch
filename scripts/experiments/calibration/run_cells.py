"""Calibration study: one SLURM-array task per (dataset, embedder, category, seed).

Each task runs *all* styles for that embedder inside it (they share the one
loaded pickle), so an embedder's arms are paired on identical data, sim/test
splits, and the identical startup exemplar.  Every style emits the #2781
calibration metrics (regret + its rule-inefficiency/calibration-shift
decomposition, threshold provenance, the degenerate flag) and the near-free
inclusion-budget sweep; the raw-patch tree style additionally re-pools its own
per-node scores under ``topk`` / ``pnorm`` (extra rows tagged ``pool_variant``).

Main rows -> ``results/cells/task_<idx>.csv``; the inclusion sweep ->
``results/cells/task_<idx>__sweep.csv``.

Run directly with ``--index N`` for one cell, or via SLURM with
``$SLURM_ARRAY_TASK_ID``.
"""

from __future__ import annotations

import argparse
import json
import os

import common

common.setup_env()

import numpy as np  # noqa: E402

import experiment_config as cfg  # noqa: E402


def _categories_by_dataset(prepare_info: dict) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for ds, per_emb in prepare_info.get("datasets", {}).items():
        out[ds] = {emb: entry.get("selected_categories", []) for emb, entry in per_emb.items()}
    return out


def _load_exemplar(ds: str, emb: str, cat: str, seed: int) -> tuple[int | None, np.ndarray | None]:
    base = common.RESULTS / "crops" / cfg.crops_basename(ds, emb)
    json_path = base.with_suffix(".json")
    npz_path = base.with_suffix(".npz")
    if not json_path.exists() or not npz_path.exists():
        return None, None
    candidates = json.loads(json_path.read_text()).get(cat) or []
    if not candidates:
        return None, None
    exemplar_id = int(candidates[seed % len(candidates)])
    with np.load(npz_path) as z:
        key = f"{cat}::{exemplar_id}"
        if key not in z:
            return None, None
        return exemplar_id, z[key].astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibration: one cell (dataset,embedder,category,seed).")
    parser.add_argument("--index", type=int, default=None, help="Cell index; defaults to $SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--outdir", default=str(common.RESULTS / "cells"))
    parser.add_argument("--print-cells", action="store_true", help="Print the total cell count and exit.")
    args = parser.parse_args(argv)

    prepare_info = json.loads((common.RESULTS / "prepare_info.json").read_text())
    cells = cfg.array_cells(_categories_by_dataset(prepare_info))

    if args.print_cells:
        print(len(cells))
        return 0

    idx = args.index if args.index is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if idx >= len(cells):
        common.log(f"index {idx} >= {len(cells)} cells; nothing to do")
        return 0
    cell = cells[idx]
    ds, emb, cat, seed = cell["dataset"], cell["embedder"], cell["category"], cell["seed"]
    styles = cfg.styles_for_embedder(emb)
    region_voting = cfg.REGION_VOTING_BY_DATASET.get(ds, False)
    common.log(
        f"cell {idx}/{len(cells)}: dataset={ds} embedder={emb} category={cat} seed={seed} "
        f"styles={styles} head={cfg.HEAD} safe_thresholds={cfg.SAFE_THRESHOLDS}"
    )

    import pandas as pd

    from vtscore.eval.patch_styles import resolve_style
    from vtscore.eval.voting_iterations import (
        _CALIBRATION_COLUMNS,
        _CUT_DIAGNOSTIC_COLUMNS,
        _INCLUSION_SWEEP_COLUMNS,
        simulate_voting_iterations,
    )

    from vtscore.datasets import loader as _loader  # isort: skip

    from _cells_io import load_medias  # noqa: PLC0415

    pkl = _loader.EMBEDDINGS_DIR / cfg.pickle_name(ds, emb)
    medias: dict[int, dict] = load_medias(pkl)
    common.log(f"loaded {len(medias)} medias from {pkl}")

    exemplar_id, crop_vec = _load_exemplar(ds, emb, cat, seed)
    if crop_vec is None:
        common.log(f"WARNING: no exemplar crop for ({ds}, {emb}, {cat}); seeding from random known-goods instead")

    all_rows: list[dict] = []
    all_sweep: list[dict] = []
    all_cutdiag: list[dict] = []
    for style in styles:
        seed_scores = None
        if crop_vec is not None:
            seed_scores = resolve_style(style).exemplar_sims(medias, crop_vec)
        variants = cfg.REPOOL_VARIANTS if style == cfg.REPOOL_STYLE else []
        sweep_local: list[dict] = []
        cutdiag_local: list[dict] = []
        rows = simulate_voting_iterations(
            medias,
            target_category=cat,
            seed=seed,
            dataset_name=ds,
            inclusion=cfg.INCLUSION,
            sim_fraction=cfg.SIM_FRACTION,
            safe_thresholds=cfg.SAFE_THRESHOLDS,
            calibrate_count=cfg.CALIBRATE_COUNT,
            calibration_fraction=cfg.CALIBRATION_FRACTION,
            region_voting=region_voting,
            max_steps=cfg.MAX_STEPS,
            seed_scores=seed_scores,
            trainer="mlp",
            head=cfg.HEAD,
            style=style,
            emit_calibration_metrics=True,
            repool_variants=variants,
            repool_topk=cfg.REPOOL_TOPK,
            inclusion_sweep_ks=cfg.INCLUSION_SWEEP_KS,
            sweep_sink=sweep_local,
            cut_diag_sink=cutdiag_local,
        )
        for r in rows:
            r["embedder"] = emb
            r["exemplar_id"] = exemplar_id if exemplar_id is not None else -1
        for sr in sweep_local:
            sr["embedder"] = emb
        for dr in cutdiag_local:
            dr["embedder"] = emb
        all_rows.extend(rows)
        all_sweep.extend(sweep_local)
        all_cutdiag.extend(cutdiag_local)
        common.log(
            f"  style={style}: {len(rows)} rows, {len(sweep_local)} sweep rows, "
            f"{len(cutdiag_local)} cut-diagnostic rows"
        )

    outdir = common.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    main_cols = [*_CALIBRATION_COLUMNS, "embedder", "exemplar_id"]
    out = outdir / f"task_{idx:04d}.csv"
    pd.DataFrame(all_rows, columns=pd.Index(main_cols)).to_csv(out, index=False)
    sweep_cols = [*_INCLUSION_SWEEP_COLUMNS, "embedder"]
    sweep_out = outdir / f"task_{idx:04d}__sweep.csv"
    pd.DataFrame(all_sweep, columns=pd.Index(sweep_cols)).to_csv(sweep_out, index=False)
    # The #2836 cut-decomposition frame (one row per step per fit geometry).
    cutdiag_cols = [*_CUT_DIAGNOSTIC_COLUMNS, "embedder"]
    cutdiag_out = outdir / f"task_{idx:04d}__cutdiag.csv"
    pd.DataFrame(all_cutdiag, columns=pd.Index(cutdiag_cols)).to_csv(cutdiag_out, index=False)
    common.log(
        f"wrote {len(all_rows)} rows to {out}, {len(all_sweep)} sweep rows to {sweep_out}, "
        f"and {len(all_cutdiag)} cut-diagnostic rows to {cutdiag_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
