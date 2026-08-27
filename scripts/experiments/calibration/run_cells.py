"""Calibration study: one SLURM-array task per (dataset, embedder, category, seed).

Each task runs *all* styles for that embedder inside it (they share the one
loaded pickle), so an embedder's arms are paired on identical data, sim/test
splits, and the identical startup exemplar.  Every style emits the #2781
calibration metrics (regret + its rule-inefficiency/calibration-shift
decomposition, threshold provenance, the degenerate flag) and the near-free
inclusion-budget sweep; the raw-patch tree style additionally re-pools its own
per-node scores under ``topk`` / ``pnorm`` (extra rows tagged ``pool_variant``).

Main rows -> ``results/cells/task_<idx>.csv``; the inclusion-budget sweep ->
``results/cells/task_<idx>__sweep.csv``; the #2836 cut decomposition ->
``__cutdiag.csv``; the #2865 cut-rule x inclusion sweep -> ``__cutincl.csv``.

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


def _seed_query_text(ds: str, cat: str) -> str:
    """The text a user would type to find *cat* in *ds*, or "" if none is known.

    Two tables, because there are two kinds of dataset.  ``EXPERIMENT_QUERIES``
    covers fixtures that exist only inside this experiment (``vg_scale``);
    ``vtscore.eval.config.EVAL_DATASETS`` covers the real demo datasets the app
    ships (``visual_genome_m``, ``caltech101_m``).  The experiment table wins so
    a fixture can override, but neither is required -- an unknown dataset simply
    has no query, and the autopilot seeds from known-goods instead.
    """
    try:
        local = cfg.EXPERIMENT_QUERIES.get(ds) or {}
    except AttributeError:
        local = {}
    if cat in local:
        return local[cat]

    from vtscore.eval.config import EVAL_DATASETS  # noqa: PLC0415

    info = EVAL_DATASETS.get(ds)
    if not info:
        return ""
    for query in info["queries"]:
        if query.target_category == cat:
            return query.text
    return ""


def _text_seed_scores(ds: str, emb: str, cat: str, medias: dict) -> "dict[int, float] | None":
    """The app's text sort: cosine from the typed query to every media.

    This is what a real user starts from -- they type "boat" and vote down the
    ranking -- and it is what ``seed_scores`` has always been documented to hold
    (``al_strategies``, ``EVAL.md``, ``voting_iterations`` all say "similarity to
    the typed query").

    Returns ``None`` when no query is defined for the cell, or when the embedder
    has no text tower -- DINOv3 does not, so ``embed_text`` is the base class's
    ``return None`` and ``embed_text_query`` yields nothing.  ``None`` is the
    signal for the autopilot to seed from *three random known-good examples*
    instead, the app's other real start ("3 random examples pulled from the
    Good").  Both are things a user does; ranking by cosine to a cropped box is
    not, which is why this no longer seeds from crops.
    """
    from vtscore.embedding.helpers import embed_text_query  # noqa: PLC0415
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    text = _seed_query_text(ds, cat)
    if not text:
        return None
    qvec = embed_text_query(text, "image", embedder_name=emb)
    if qvec is None:
        return None

    def _unit(vec):
        v = np.asarray(vec, dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-12 else v

    ids = list(medias.keys())
    if not ids:
        return None
    matrix = np.stack([_unit(media_embedding(medias[c])) for c in ids])
    cos = matrix @ _unit(qvec)
    return {ids[k]: float(cos[k]) for k in range(len(ids))}


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
    styles = cfg.styles_for(ds, emb)
    region_voting = cfg.region_voting_for(ds, emb)
    common.log(
        f"cell {idx}/{len(cells)}: dataset={ds} embedder={emb} category={cat} seed={seed} "
        f"styles={styles} head={cfg.HEAD or 'default (production)'} safe_thresholds={cfg.SAFE_THRESHOLDS} "
        f"calibrate_count={cfg.CALIBRATE_COUNT} fold_counts={cfg.FOLD_COUNTS or 'off'} "
        f"cut_incl_ks={cfg.CUT_INCLUSION_KS or 'off'} "
        f"acq_inclusion_offset={cfg.ACQ_INCLUSION_OFFSET} acq_rank_percentile={cfg.ACQ_RANK_PERCENTILE}"
    )

    import pandas as pd

    from vtscore.eval.voting_iterations import (
        _CALIBRATION_COLUMNS,
        _CUT_DIAGNOSTIC_COLUMNS,
        _CUT_INCLUSION_COLUMNS,
        _INCLUSION_SWEEP_COLUMNS,
        simulate_voting_iterations,
    )

    from vtscore.datasets import loader as _loader  # isort: skip

    from _cells_io import load_medias  # noqa: PLC0415

    pkl = _loader.EMBEDDINGS_DIR / cfg.pickle_name(ds, emb)
    medias: dict[int, dict] = load_medias(pkl)
    common.log(f"loaded {len(medias)} medias from {pkl}")

    seed_scores = _text_seed_scores(ds, emb, cat, medias)
    seed_mode = "text" if seed_scores is not None else "known_good"
    seed_query = _seed_query_text(ds, cat) if seed_scores is not None else ""
    common.log(f"seed: mode={seed_mode} query={seed_query!r}")

    all_rows: list[dict] = []
    all_sweep: list[dict] = []
    all_cutdiag: list[dict] = []
    all_cutincl: list[dict] = []
    for style in styles:
        variants = cfg.REPOOL_VARIANTS if style == cfg.REPOOL_STYLE else []
        sweep_local: list[dict] = []
        cutdiag_local: list[dict] = []
        cutincl_local: list[dict] = []
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
            blend_schedule=cfg.BLEND_SCHEDULE,
            schedule_variants=cfg.SCHEDULE_VARIANTS,
            cut_diag_sink=cutdiag_local,
            anchored_thresholds=cfg.ANCHORED,
            anchored_weights=cfg.ANCHORED_WEIGHTS,
            anchored_rules=cfg.ANCHORED_RULES,
            anchored_fold_arms=cfg.ANCHORED_FOLD_ARMS,
            anchored_fold_combines=cfg.ANCHORED_FOLD_COMBINES,
            fold_count_variants=cfg.FOLD_COUNTS or None,
            cut_inclusion_ks=cfg.CUT_INCLUSION_KS or None,
            cut_inclusion_sink=cutincl_local,
            cut_inclusion_qtilt_steps=cfg.CUT_INCLUSION_QTILT_STEPS or None,
            acq_inclusion_offset=cfg.ACQ_INCLUSION_OFFSET,
            acq_rank_percentile=cfg.ACQ_RANK_PERCENTILE,
        )
        for r in rows:
            r["embedder"] = emb
            r["seed_mode"] = seed_mode
            r["seed_query"] = seed_query
        for sr in sweep_local:
            sr["embedder"] = emb
        for dr in cutdiag_local:
            dr["embedder"] = emb
        for cr in cutincl_local:
            cr["embedder"] = emb
        all_rows.extend(rows)
        all_sweep.extend(sweep_local)
        all_cutdiag.extend(cutdiag_local)
        all_cutincl.extend(cutincl_local)
        common.log(
            f"  style={style}: {len(rows)} rows, {len(sweep_local)} sweep rows, "
            f"{len(cutdiag_local)} cut-diagnostic rows, {len(cutincl_local)} cut-inclusion rows"
        )

    outdir = common.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    main_cols = [*_CALIBRATION_COLUMNS, "embedder", "seed_mode", "seed_query"]
    out = outdir / f"task_{idx:04d}.csv"
    pd.DataFrame(all_rows, columns=pd.Index(main_cols)).to_csv(out, index=False)
    sweep_cols = [*_INCLUSION_SWEEP_COLUMNS, "embedder"]
    sweep_out = outdir / f"task_{idx:04d}__sweep.csv"
    pd.DataFrame(all_sweep, columns=pd.Index(sweep_cols)).to_csv(sweep_out, index=False)
    # The #2836 cut-decomposition frame (one row per step per fit geometry).
    cutdiag_cols = [*_CUT_DIAGNOSTIC_COLUMNS, "embedder"]
    cutdiag_out = outdir / f"task_{idx:04d}__cutdiag.csv"
    pd.DataFrame(all_cutdiag, columns=pd.Index(cutdiag_cols)).to_csv(cutdiag_out, index=False)
    # The #2865 cut-rule x inclusion frame (one row per step per arm per k).
    # Written unconditionally, like the frames above: an empty CSV with the
    # right header is what tells the analyzer the run had the sweep switched
    # off, rather than that its cells silently failed.
    cutincl_cols = [*_CUT_INCLUSION_COLUMNS, "embedder"]
    cutincl_out = outdir / f"task_{idx:04d}__cutincl.csv"
    pd.DataFrame(all_cutincl, columns=pd.Index(cutincl_cols)).to_csv(cutincl_out, index=False)
    common.log(
        f"wrote {len(all_rows)} rows to {out}, {len(all_sweep)} sweep rows to {sweep_out}, "
        f"{len(all_cutdiag)} cut-diagnostic rows to {cutdiag_out}, "
        f"and {len(all_cutincl)} cut-inclusion rows to {cutincl_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
