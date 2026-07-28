"""The definitive Max-Patch voting run (one SLURM-array task per cell).

Each array task handles exactly one ``(dataset, embedder, category, seed)``
cell and runs *all* styles for that embedder inside it (they share the one
loaded pickle), so the MaxHAC / MaxPatch / whole-image trajectories are paired
on identical data, identical sim/test splits, and the identical startup
exemplar.  The cell for this task is ``array_cells(...)[SLURM_ARRAY_TASK_ID]``
- a stable enumeration derived from ``prepare_info.json``.

The startup sort mirrors the "train a new detector from an example" flow: the
cell's exemplar (a cropped positive, pre-embedded by ``prepare_data.py``) is
scored against the dataset **per style** - whole-image cosine, max cosine over
HAC region nodes, or max cosine over raw patches - and the Autopilot seed
phase votes down that ranking.

Results land in ``results/cells/task_<idx>.csv``; the summariser concatenates
whatever tasks completed.

Run directly with ``--index N`` for a single cell, or via SLURM with
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
    """Return ``(exemplar_media_id, crop_vector)`` for this cell, or ``(None, None)``."""
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
    parser = argparse.ArgumentParser(description="Max-Patch: one cell (dataset,embedder,category,seed).")
    parser.add_argument("--index", type=int, default=None, help="Cell index; defaults to $SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--outdir", default=str(common.RESULTS / "cells"))
    parser.add_argument(
        "--print-cells", action="store_true", help="Print the total cell count and exit (for array sizing)."
    )
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
    common.log(f"cell {idx}/{len(cells)}: dataset={ds} embedder={emb} category={cat} seed={seed} styles={styles}")

    import pandas as pd

    from vtscore.datasets.loader_pickle import load_dataset_from_pickle
    from vtscore.eval.patch_styles import resolve_style
    from vtscore.eval.voting_iterations import _VOTING_COLUMNS, simulate_voting_iterations

    from vtscore.datasets import loader as _loader  # isort: skip

    medias: dict[int, dict] = {}
    pkl = _loader.EMBEDDINGS_DIR / cfg.pickle_name(ds, emb)
    load_dataset_from_pickle(pkl, medias)
    common.log(f"loaded {len(medias)} medias from {pkl}")

    exemplar_id, crop_vec = _load_exemplar(ds, emb, cat, seed)
    if crop_vec is None:
        common.log(f"WARNING: no exemplar crop for ({ds}, {emb}, {cat}); seeding from random known-goods instead")

    all_rows: list[dict] = []
    for style in styles:
        # The exemplar startup sort, computed in this style's own geometry.
        seed_scores = None
        if crop_vec is not None:
            seed_scores = resolve_style(style).exemplar_sims(medias, crop_vec)
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
            region_voting=cfg.REGION_VOTING,
            max_steps=cfg.MAX_STEPS,
            seed_scores=seed_scores,
            trainer="mlp",
            style=style,
        )
        for r in rows:
            r["embedder"] = emb
            r["exemplar_id"] = exemplar_id if exemplar_id is not None else -1
        all_rows.extend(rows)
        common.log(f"  style={style}: {len(rows)} rows")

    outdir = common.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"task_{idx:04d}.csv"
    columns = [*_VOTING_COLUMNS, "embedder", "exemplar_id"]
    pd.DataFrame(all_rows, columns=pd.Index(columns)).to_csv(out, index=False)
    common.log(f"wrote {len(all_rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
