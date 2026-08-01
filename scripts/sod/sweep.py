#!/usr/bin/env python3
"""Small-object-detection evaluation sweep (orchestrator).

Sweeps ``dataset × class × embedder × proposal`` and, for each config, runs the
realistic Autopilot active-learning labeling loop, reporting cost/fpr/fnr/f1/IoU vs
the total annotation count ``t`` (plus an oracle reference). See docs/plans/...
(small-object sweep) for the design.

Single-GPU loop for the pilot; ``--array-index/--array-total`` filter cells so the
same command drops into a SLURM array later. Per-image embeddings are cached to
npz (``--cache-dir``) so re-runs reuse forward passes.

Example (stop sign on COCO):
    srun --partition=gpu --gres=gpu:l40s:1 --cpus-per-task=8 --mem=46G --time=8:00:00 \\ # or --pty bash -l
      .venv/bin/python scripts/sod/sweep.py \\
        --datasets coco --classes "stop sign" --embedders dinov2,dinov3 \\
        --proposals hac --region-voting --iterations 3 --max-labels 60 \\
        --out-dir docs/experiments/sod-sweep --viz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

# scripts/sod is sys.path[0] when run as `python scripts/sod/sweep.py`.
from datasets import GUI_MIN_BOX_FRAC, SodDataset
from features import FeatureCache, build_curve_inputs, dump_split, partition_split, slugify

from vtscore.eval.region_curve import evaluate_realistic_curve
from vtscore.eval.region_sources import build_region_source

EMBEDDER_ALIASES = {"dinov2": "dinov2_patch", "dinov3": "dinov3_patch"}
TEXT_EMBEDDERS = {"siglip", "siglip2", "clip"}

# Short aliases for the gated DINOv3 ViT checkpoints (all patch-size 16, so the
# grid density is identical across sizes — a larger model buys richer per-patch
# features, NOT finer localization; use --resolution for that). ``vitb16`` is the
# app default (``config.DINOV3_MODEL_ID``); the others are downloaded on first use.
DINOV3_VARIANTS = {
    "vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "vits16plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "vith16plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "vit7b16": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
}


def _resolve_dinov3_model(token: str) -> str:
    """Map a ``--dinov3-model`` token to a full HF repo id.

    A short alias (``vitl16``) expands via :data:`DINOV3_VARIANTS`; a token
    containing ``/`` is treated as a full HF id and passed through verbatim.
    """
    t = str(token).strip()
    if "/" in t:
        return t
    try:
        return DINOV3_VARIANTS[t.lower()]
    except KeyError:
        raise SystemExit(
            f"unknown --dinov3-model {token!r}; choose from {sorted(DINOV3_VARIANTS)} or pass a full HF id"
        ) from None


def _floats(s: str) -> tuple[float, ...]:
    return tuple(float(x) for x in s.split(",") if x.strip())


def _strs(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _proposal_slug(
    proposal: str,
    args,
    alpha: float,
    region_voting: bool = False,
    seeding: str = "topk",
    leaf_assign: str = "spatial",
    pca_dims: int | None = None,
    hac_k: int = 12,
    leaf_beta: float | None = None,
) -> str:
    if proposal == "sliding":
        sc = "-".join(str(x) for x in args.scales)
        return f"sliding_s{sc}_o{args.overlap}_w{args.min_window}"
    if proposal == "dino":
        return f"dino_k{hac_k}_a{args.hac_alpha_default}"
    if proposal == "hac":
        # Region-voting rewrites the exemplar (snapped, one per image) and stores a
        # leaf_mask, so it must not share a cache with the box-pool hac variant.
        rv = "_rv" if region_voting else ""
        # Per-image PCA changes the tree topology (and thus cached region vecs/
        # boxes/children), so a PCA run must not reuse non-PCA cached regions.
        pca = f"_pca{pca_dims}" if pca_dims else ""
        # Leaf seeding/assignment also change the tree; tag only when non-default
        # so existing default-run caches keep their slug.
        seed_tag = "" if seeding == "topk" else f"_seed-{seeding}"
        asg_tag = "" if leaf_assign == "spatial" else f"_asg-{leaf_assign}"
        # Assignment blend beta (feature only): None = reuse alpha (no tag, keeps the
        # historical `_asg-feature` slug); an explicit value decouples it and tags.
        beta_tag = f"_b{leaf_beta}" if (leaf_assign == "feature" and leaf_beta is not None) else ""
        return f"hac{rv}_k{hac_k}_a{alpha}{pca}{seed_tag}{asg_tag}{beta_tag}"
    return "whole"


def _build_source(
    proposal: str,
    embedder,
    args,
    alpha: float,
    region_voting: bool = False,
    seeding: str = "topk",
    leaf_assign: str = "spatial",
    pca_dims: int | None = None,
    hac_k: int = 12,
    leaf_beta: float | None = None,
):
    from vtscore.config import DINOV2_MODEL_ID, resolve_device

    return build_region_source(
        proposal,
        embedder,
        scales=args.scales,
        overlap=args.overlap,
        min_window=args.min_window,
        hac_k=hac_k,
        hac_alpha=alpha,
        hac_pca_dims=pca_dims,
        hac_seeding=seeding,
        hac_leaf_assign=leaf_assign,
        hac_leaf_beta=leaf_beta,
        dino_model_id=DINOV2_MODEL_ID,
        dino_device=resolve_device(),
        dino_register_tokens=0,
        region_voting=region_voting,
    )


def _apply_resolution(embedder, resolution: int | None) -> None:
    """Set a DINO patch embedder's HF image processor to emit ``resolution×resolution``
    tensors (patch grid = ``resolution // patch_size`` per side), or restore the
    checkpoint default when ``resolution is None``.

    Loads the model first (idempotent) so ``_processor`` exists, then mirrors whichever
    size schema the processor uses (DINOv2's shortest-edge + crop vs DINOv3's
    height/width). **Since the embedder is a shared singleton across sweep cells**, the
    checkpoint-default size is snapshotted on first touch so a later cell with
    ``resolution=None`` (or a different value) restores/overwrites it cleanly — never
    inherits the previous cell's resolution. A no-op for embedders whose processor has no
    ``size`` dict (e.g. open_clip text embedders), so ``--resolution`` only affects DINO.
    Use a multiple of the patch size (14 dinov2, 16 dinov3) or the grid isn't square.
    """
    embedder.load_models()
    processor = getattr(embedder, "_processor", None)
    size = getattr(processor, "size", None)
    if size is None:
        return  # not a resizing processor (e.g. an open_clip text embedder) — resolution is a no-op

    def _sz_get(sz, k):
        # Works for both a modern fast-processor ``SizeDict`` (attribute access) and a
        # legacy plain-``dict`` size (``BitImageProcessor``). DINOv3's ``SizeDict`` is NOT
        # a ``dict`` subclass — the old ``isinstance(size, dict)`` gate silently made this
        # a no-op, which is exactly why --resolution did nothing on DINOv3.
        v = getattr(sz, k, None)
        if v is None and isinstance(sz, dict):
            v = sz.get(k)
        return v

    # Snapshot the pristine checkpoint default once so a later cell can restore it (the
    # embedder is a shared singleton across cells).
    if not hasattr(processor, "_sod_orig_size"):
        processor._sod_orig_size = size
        processor._sod_orig_crop = getattr(processor, "crop_size", None)
    if resolution is None:  # restore checkpoint default (undo a prior cell's override)
        processor.size = processor._sod_orig_size
        if processor._sod_orig_crop is not None:
            processor.crop_size = processor._sod_orig_crop
        return
    # Assign a plain dict — fast processors accept it and normalise to a SizeDict.
    if _sz_get(size, "shortest_edge") is not None:
        processor.size = {"shortest_edge": resolution}
    else:
        processor.size = {"height": resolution, "width": resolution}
    if getattr(processor, "do_center_crop", False) and getattr(processor, "crop_size", None) is not None:
        processor.crop_size = {"height": resolution, "width": resolution}


def _parse_pca_dims(tokens: list[str]) -> list[int | None]:
    """Parse ``--pca-dims`` tokens into (int | None) values, order-preserving + deduped.

    ``'none'`` / ``'0'`` → ``None`` (full-dim baseline); everything else → ``int``.
    """
    out: list[int | None] = []
    for tok in tokens:
        t = str(tok).strip().strip(",").lower()  # tolerate `none, 10` (stray commas/spaces)
        if not t:
            continue
        pv = None if t in ("none", "0") else int(t)
        if pv not in out:
            out.append(pv)
    return out


def _parse_resolutions(tokens: list[str]) -> list[int | None]:
    """Parse ``--resolution`` tokens into (int | None), order-preserving + deduped.

    ``'none'`` / ``'0'`` / ``'224'``-style; ``none`` → ``None`` (checkpoint default).
    """
    out: list[int | None] = []
    for tok in tokens:
        t = str(tok).strip().strip(",").lower()
        if not t:
            continue
        rv = None if t in ("none", "0") else int(t)
        if rv not in out:
            out.append(rv)
    return out


def _parse_betas(tokens: list[str]) -> list[float | None]:
    """Parse ``--leaf-beta`` tokens into (float | None), order-preserving + deduped.

    ``'none'`` → ``None`` (reuse the HAC-merge ``alpha`` for assignment). Unlike the
    other parsers, ``'0'`` is kept as ``0.0`` (pure-spatial assignment) — it is a
    meaningful value here, not a synonym for the default.
    """
    out: list[float | None] = []
    for tok in tokens:
        t = str(tok).strip().strip(",").lower()
        if not t:
            continue
        bv = None if t == "none" else float(t)
        if bv not in out:
            out.append(bv)
    return out


def _cells(args) -> list[dict]:
    """Enumerate (dataset, class, embedder, proposal, alpha, hac_k, leaf_seeding, leaf_assign,
    leaf_beta, pca_dims, resolution) cells.

    ``--hac-alpha``, ``--hac-k``, ``--leaf-seeding``, ``--leaf-assign``, ``--leaf-beta``,
    ``--pca-dims``, ``--resolution`` are all sweep axes: pass multiple values and every combination
    becomes its own cell (its own cache slug + result row). The leaf/pca/alpha/k axes only affect the
    hac tree, so other proposals collapse to a single value there. ``--leaf-beta`` (the feature
    assignment blend) only matters for ``leaf_assign="feature"``, so it collapses to a single value
    (None = reuse alpha) elsewhere — no redundant cells. ``--resolution`` affects the embedder forward,
    so it applies to any DINO embedder (whole/sliding/hac) but collapses to the checkpoint default for
    text embedders. ``--dinov3-model`` (checkpoint size) is likewise an embedder-forward axis, but only
    for the ``dinov3`` embedder — a sentinel ``None`` (app default) for every other embedder.
    """
    cells: list[dict] = []
    for dataset in args.datasets:
        for cls in args.classes:
            for embedder in args.embedders:
                is_dino = embedder in ("dinov2", "dinov3")
                resolutions = _parse_resolutions(args.resolution) if is_dino else [None]
                # DINOv3 checkpoint size is a sweep axis (separate curves) but only
                # for the dinov3 embedder; a sentinel None everywhere else.
                models = args.dinov3_model if embedder == "dinov3" else [None]
                for proposal in args.proposals:
                    is_hac = proposal == "hac"
                    alphas = args.hac_alpha if is_hac else [args.hac_alpha_default]
                    ks = args.hac_k if is_hac else [args.hac_k[0]]
                    seedings = args.leaf_seeding if is_hac else ["topk"]
                    assigns = args.leaf_assign if is_hac else ["spatial"]
                    pcas = _parse_pca_dims(args.pca_dims) if is_hac else [None]
                    betas = _parse_betas(args.leaf_beta) if is_hac else [None]
                    for alpha in alphas:
                        for hac_k in ks:
                            for seeding in seedings:
                                for leaf_assign in assigns:
                                    # beta only bites on feature assignment; collapse to
                                    # None (reuse alpha) for spatial to avoid dup cells.
                                    betas_here = betas if leaf_assign == "feature" else [None]
                                    for leaf_beta in betas_here:
                                        for pca_dims in pcas:
                                            for resolution in resolutions:
                                                for dinov3_model in models:
                                                    cells.append(
                                                        {
                                                            "dataset": dataset,
                                                            "class": cls,
                                                            "embedder": embedder,
                                                            "proposal": proposal,
                                                            "alpha": alpha,
                                                            "hac_k": hac_k,
                                                            "leaf_seeding": seeding,
                                                            "leaf_assign": leaf_assign,
                                                            "leaf_beta": leaf_beta,
                                                            "pca_dims": pca_dims,
                                                            "resolution": resolution,
                                                            "dinov3_model": dinov3_model,
                                                        }
                                                    )
    return cells


def _run_cell(cell: dict, args, cache_root: Path) -> list[dict]:
    from vtscore.media import get_embedder

    from vtscore.config import DINOV3_MODEL_ID

    reg_name = EMBEDDER_ALIASES.get(cell["embedder"], cell["embedder"])
    proposal, alpha = cell["proposal"], cell["alpha"]
    seeding, leaf_assign, pca_dims = cell["leaf_seeding"], cell["leaf_assign"], cell["pca_dims"]
    resolution, hac_k, leaf_beta = cell["resolution"], cell["hac_k"], cell["leaf_beta"]
    dinov3_model = cell.get("dinov3_model")
    # Resolve the DINOv3 checkpoint token and decide whether it's non-default (a
    # non-default checkpoint changes the embeddings, so it must namespace the cache
    # exactly like --resolution). Default/None → no tag, keeps existing caches.
    dinov3_full_id = _resolve_dinov3_model(dinov3_model) if dinov3_model else None
    model_tag = f"_m{dinov3_model}" if (dinov3_full_id and dinov3_full_id != DINOV3_MODEL_ID) else ""

    # Validity guards (skip invalid combos cleanly).
    if proposal == "hac" and cell["embedder"] not in {"dinov2", "dinov3"}:
        print(f"  skip {cell}: hac needs a patch embedder", flush=True)
        return []

    # Region-voting (faithful app-detector label construction) applies only to the
    # hac proposal on a patch embedder; a no-op elsewhere even when --region-voting
    # is set, so `--proposals whole,hac --region-voting` runs whole normally.
    region_voting = bool(args.region_voting) and proposal == "hac" and cell["embedder"] in {"dinov2", "dinov3"}
    if args.region_voting and not region_voting and proposal == "hac":
        print(f"  skip {cell}: --region-voting needs a patch embedder", flush=True)
        return []

    embedder = get_embedder(reg_name)
    # Re-point the DINOv3 singleton at this cell's checkpoint BEFORE loading (so
    # _apply_resolution's load_models sees the right weights). Always call (even
    # None) so a prior cell's larger model is reset rather than leaking forward.
    if cell["embedder"] == "dinov3":
        embedder.set_model_id(dinov3_full_id)
    # Always call (even for None) so a prior cell's resolution is reset on the shared
    # singleton embedder rather than leaking into this cell.
    if cell["embedder"] in ("dinov2", "dinov3"):
        _apply_resolution(embedder, resolution)
    try:
        source = _build_source(
            proposal,
            embedder,
            args,
            alpha,
            region_voting=region_voting,
            seeding=seeding,
            leaf_assign=leaf_assign,
            pca_dims=pca_dims,
            hac_k=hac_k,
            leaf_beta=leaf_beta,
        )
    except Exception as exc:
        print(f"  skip {cell}: {exc}", flush=True)
        return []

    with SodDataset(cell["dataset"]) as ds:
        split = ds.class_split(
            cell["class"], neg_multiple=args.neg_multiple, seed=args.split_seed, min_box_frac=args.min_box_frac
        )
        if not split.positive_ids:
            print(f"  skip {cell}: no positives for class {cell['class']!r}", flush=True)
            return []
        buckets = partition_split(split, args.test_fraction, args.split_seed)
        # Dump the split once per (dataset, class) — it's identical across cells.
        split_path = args.out_dir / f"{cell['dataset']}_{slugify(cell['class'])}_split.json"
        if not split_path.exists():
            dump_split(
                split_path,
                dataset=cell["dataset"],
                class_name=cell["class"],
                split_seed=args.split_seed,
                neg_multiple=args.neg_multiple,
                test_fraction=args.test_fraction,
                split=buckets,
            )
        slug = _proposal_slug(
            proposal,
            args,
            alpha,
            region_voting=region_voting,
            seeding=seeding,
            leaf_assign=leaf_assign,
            pca_dims=pca_dims,
            hac_k=hac_k,
            leaf_beta=leaf_beta,
        )
        if resolution:
            # Resolution changes the embedding itself (grid + vectors), so it must
            # namespace the cache for EVERY proposal — a resolution run must never
            # reuse default-res cached vectors.
            slug = f"{slug}_r{resolution}"
        if model_tag:
            # A non-default DINOv3 checkpoint changes the embeddings too (different
            # weights, possibly different embed dim), so it namespaces the cache like
            # resolution — all sizes share the 'dinov3_patch' reg_name otherwise.
            slug = f"{slug}{model_tag}"
        cache = FeatureCache(cache_root, cell["dataset"], reg_name, slug)
        meta = {
            "dataset": cell["dataset"],
            "class": cell["class"],
            "embedder": cell["embedder"],
            "reg_name": reg_name,  # registry name; joins rows to prep_timing_summary
            "proposal": proposal,
            "proposal_slug": slug,
            "alpha": alpha,
            "hac_k": hac_k,
            "leaf_seeding": seeding,
            "leaf_assign": leaf_assign,
            "leaf_beta": leaf_beta,
            "pca_dims": pca_dims,
            "resolution": resolution,
            "dinov3_model": dinov3_model,
            "region_voting": region_voting,
            "negatives_exhaustive": split.negatives_exhaustive,
            "n_pos_total": len(split.positive_ids),
            "n_neg_total": len(split.negative_ids),
        }
        inputs = build_curve_inputs(
            ds,
            source,
            buckets,
            cache,
            class_name=cell["class"],
            meta=meta,
            region_voting=region_voting,
            neg_regions=args.neg_regions,
            build_pool=True,
        )

    # One active-learning labeling loop per cell (x-axis = total annotations t). The
    # MLP head does the work; a text query only seeds the cold-start ranking for
    # text-capable embedders (DINO uses the seed exemplar).
    query_vec = None
    if cell["embedder"] in TEXT_EMBEDDERS:
        query_vec = source.embed_text(args.prompt_template.format(cell["class"]))
    result = evaluate_realistic_curve(
        inputs,
        "mlp",
        seeds=range(args.iterations),
        max_labels=args.max_labels,
        inclusion=args.inclusion,
        safe_thresholds=args.safe_thresholds,
        calibrate_count=args.calibrate_count,
        cal_fraction=args.cal_fraction,
        query_vec=query_vec,
        select_strategy=args.select_strategy,
        good_to_start=args.good_to_start,
        bad_to_start=args.bad_to_start,
        retrain_cadence=args.retrain_cadence,
        stop_at_done=args.stop_at_done,
        return_finals=args.viz or args.labeling_trace,
        threshold_rule=args.threshold_rule,
        threshold_smooth=args.threshold_smooth,
    )
    rows: list[dict] = []
    if args.viz or args.labeling_trace:
        cell_rows, finals = result
        rows.extend(cell_rows)
        # Prediction overlays (viz-seed, --viz) + per-iteration labeling trace
        # (all seeds, --labeling-trace); each gated independently inside.
        _render_realistic_viz(args, cell, cache_root, buckets, slug, finals)
    else:
        rows.extend(result)
    return rows


def _render_realistic_viz(args, cell, cache_root: Path, buckets, slug: str, finals: dict) -> None:
    """Realistic-mode overlays from the in-memory finals of
    ``evaluate_realistic_curve(return_finals=True)``:

    - with ``--viz``: final-detector TP/FP/FN/TN prediction overlays for the ``--viz-seed``
      (default 0);
    - with ``--labeling-trace``: the per-iteration labeling trace (numbered ordered images +
      trace.csv/json) for **every** seed under ``labeling_trace/``.

    Reopens + primes the dataset once (the loop ran after the cell's ``SodDataset`` block,
    and a fresh ``SodDataset`` has no ``load_image`` locator until ``class_split``).
    """
    from viz import render_labeling_trace, render_predictions_realistic

    viz_seed = args.viz_seed if args.viz_seed is not None else 0
    try:
        with SodDataset(cell["dataset"]) as ds:
            ds.class_split(
                cell["class"], neg_multiple=args.neg_multiple, seed=args.split_seed, min_box_frac=args.min_box_frac
            )
            if args.viz:
                fin = finals.get(viz_seed)
                if fin is not None:
                    render_predictions_realistic(
                        ds,
                        buckets,
                        cache_dir=cache_root,
                        # Namespace by slug so each leaf/pca/alpha variant writes to its
                        # own dir — viz.py's tag is proposal+alpha only, so without this
                        # the 8 hac variants would overwrite one another.
                        out_dir=args.out_dir / "predictions" / slug,
                        dataset=cell["dataset"],
                        cls=cell["class"],
                        embedder=cell["embedder"],
                        proposal=cell["proposal"],
                        alpha=cell["alpha"],
                        slug=slug,
                        predict=fin["predict"],
                        thr=fin["threshold"],
                        t=fin["t"],
                        gallery_n=args.viz_n,
                    )
                else:
                    print(f"  [predict] skip {cell}: no final head for seed {viz_seed}", flush=True)
            if args.labeling_trace:
                for s, f in sorted(finals.items()):
                    render_labeling_trace(
                        ds,
                        buckets,
                        f["trace"],
                        cache_dir=cache_root,
                        slug=slug,
                        out_dir=args.out_dir / "labeling_trace" / slug,
                        dataset=cell["dataset"],
                        cls=cell["class"],
                        embedder=cell["embedder"],
                        proposal=cell["proposal"],
                        alpha=cell["alpha"],
                        seed=s,
                        images=args.trace_images,
                    )
    except Exception:
        print(f"  viz realistic error {cell}:\n{traceback.format_exc()}", flush=True)


def _write_results(rows: list[dict], out_dir: Path) -> None:
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if rows:
        keys: list[str] = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with (out_dir / "results.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", type=_strs, default=["coco"])
    ap.add_argument("--classes", type=_strs, default=["stop sign"])
    ap.add_argument("--embedders", type=_strs, default=["clip", "siglip", "siglip2", "dinov2", "dinov3"])
    ap.add_argument("--proposals", type=_strs, default=["whole", "sliding", "dino", "hac"])
    ap.add_argument("--split-seed", type=int, default=0, help="seed for class split + pool partition")
    ap.add_argument(
        "--min-box-frac",
        type=float,
        default=GUI_MIN_BOX_FRAC,
        help="drop GT boxes smaller than this fraction of the image on either axis — the GUI's "
        f"drawable-box floor (default {GUI_MIN_BOX_FRAC}); an image stays positive if any box "
        "survives. 0 disables (keep all boxes).",
    )
    ap.add_argument(
        "--neg-multiple",
        type=int,
        default=100,
        help="evaluation negative-pool size as a multiple of the class's positive count "
        "(pool = neg_multiple × n_positives; prevalence ≈ 1/(1+neg_multiple), constant across classes)",
    )
    ap.add_argument(
        "--region-voting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="faithful app-detector label construction for the hac proposal on dinov2/dinov3: "
        "good votes snap to the nearest HAC node, bad votes flood CLS+leaves as negatives, "
        "bag-aware per-image weighting + grouped cross-calibration (mirrors `vtscore.eval --region-voting`). "
        "No-op for non-hac proposals. Uses a distinct 'hac_rv' cache slug.",
    )
    ap.add_argument(
        "--neg-regions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="realistic loop: a Bad vote contributes ALL of that image's region/window vectors as "
        "one per-image negative bag (via train_rv_head) instead of just the whole-image vector "
        "('No → all windows'). For sliding/dino/box-pool; no-op for whole; subsumed by --region-voting "
        "on hac. No re-embed (reuses the cached region vecs); results carry neg_regions in meta.",
    )
    ap.add_argument("--max-labels", type=int, default=60, help="realistic mode: max total annotations t per seed")
    ap.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="realistic mode: number of labeling-loop runs (seeds 0..N-1); the plots' seed "
        "band/lines span these iterations",
    )
    ap.add_argument("--good-to-start", type=int, default=3, help="realistic: goods before leaving the 'good' phase")
    ap.add_argument("--bad-to-start", type=int, default=4, help="realistic: bads before leaving the 'bad' phase")
    ap.add_argument(
        "--retrain-cadence", type=int, default=1, help="realistic: retrain every N labels (1 = faithful, per-vote)"
    )
    ap.add_argument(
        "--select-strategy",
        choices=("autopilot",),
        default="autopilot",
        help="realistic: item-selection strategy (only the faithful Autopilot phase machine is wired today)",
    )
    ap.add_argument(
        "--stop-at-done",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="realistic: stop the loop when Autopilot reaches the 'done' phase instead of running to --max-labels",
    )
    ap.add_argument(
        "--labeling-trace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="realistic: emit labeling_trace/<config>/seed{N}/ — the images labeled in order (per iteration) "
        "+ trace.csv/json (phase/head/calib/threshold/score per step). Off by default (heavy: one image per "
        "labeled item × every iteration). Independent of --viz.",
    )
    ap.add_argument(
        "--trace-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="with --labeling-trace, also render the per-step PNGs (default on). "
        "--no-trace-images writes only trace.json/trace.csv — KB-sized, for the vote/threshold "
        "record without the multi-GB image dump (spike analysis, Stage-A replay).",
    )
    ap.add_argument("--test-fraction", type=float, default=0.5)
    ap.add_argument("--inclusion", type=int, default=0, help="0=FPR+FNR; >0 favors recall; <0 favors precision")
    ap.add_argument("--safe-thresholds", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--calibrate-count", type=int, default=2)
    # Calibration rule for the whole/box-pool path + an optional temporal smoother.
    # Default is the shipped app's conformal inclusion rule (#2784); ``argmin`` (the
    # pre-#2784 min-cost cut) is selectable only to reproduce old runs.
    ap.add_argument("--threshold-rule", choices=["conformal", "rank-transfer", "argmin"], default="conformal")
    ap.add_argument("--threshold-smooth", choices=["none", "med3"], default="none")
    ap.add_argument("--cal-fraction", type=float, default=0.5)
    ap.add_argument("--scales", type=_floats, default=(1.0, 0.75, 0.5, 0.3))
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--min-window", type=int, default=48)
    ap.add_argument(
        "--hac-k",
        nargs="+",
        type=int,
        default=[12],
        help="hac leaf count K; SWEEP AXIS (pass multiple → one row each, e.g. --hac-k 8 12 16). "
        "Folds into the cache slug (k<K>). Non-hac proposals use the first value.",
    )
    ap.add_argument(
        "--hac-alpha",
        nargs="+",
        type=float,
        default=[0.5],
        help="hac cosine/spatial blend α; SWEEP AXIS (pass multiple → one row each, e.g. "
        "--hac-alpha 0.5 0.9). Space-separated (was comma-separated), matching the other axes.",
    )
    ap.add_argument("--hac-alpha-default", type=float, default=0.5, help="alpha for the dino proposer's tree")
    ap.add_argument(
        "--pca-dims",
        nargs="+",
        default=["none"],
        help="hac proposal only; SWEEP AXIS (pass multiple → one row each, e.g. --pca-dims none 10 32). "
        "Fit a per-image PCA of this many dims on the patch grid and decide HAC merge order in that space "
        "(tree topology only; stored vecs stay full-dim). 'none'/'0' = full-dim baseline (default). "
        "Clamped to min(dims, n_patches, embed_dim). Each value gets its own cache slug.",
    )
    ap.add_argument(
        "--leaf-seeding",
        nargs="+",
        choices=("topk", "spread"),
        default=["topk"],
        help="hac proposal only; SWEEP AXIS (pass multiple → one row each). How leaf seeds are placed. "
        "'topk' (default) = the K highest-saliency patches; 'spread' = greedy peaks with spatial non-max "
        "suppression so seeds spread across objects (small objects can win a seed). Non-default gets its "
        "own cache slug.",
    )
    ap.add_argument(
        "--leaf-assign",
        nargs="+",
        choices=("spatial", "feature"),
        default=["spatial"],
        help="hac proposal only; SWEEP AXIS (pass multiple → one row each). How patch cells bind to seeds. "
        "'spatial' (default) = nearest seed by grid distance (Voronoi); 'feature' = argmax of beta*cosine "
        "+ (1-beta)*spatial (see --leaf-beta), so leaves follow content. Non-default gets its own cache slug.",
    )
    ap.add_argument(
        "--leaf-beta",
        nargs="+",
        default=["none"],
        help="feature-assignment cosine/spatial blend β; SWEEP AXIS (pass multiple → one row each, e.g. "
        "--leaf-beta none 0 0.5 0.9). Only affects --leaf-assign feature. 'none' (default) = reuse the "
        "HAC-merge α (backward-compatible); β=0 == spatial assignment; β=1 = pure cosine. Independent of "
        "the merge α. An explicit value gets its own cache slug (_b<β>).",
    )
    ap.add_argument(
        "--resolution",
        nargs="+",
        default=["none"],
        help="square input edge in px for the DINO patch embedders; SWEEP AXIS (pass multiple → one "
        "row each, e.g. --resolution none 448; 'none'/'0' = checkpoint default 224). Higher = finer "
        "patch grid (grid = resolution // patch_size; 16 for dinov3, 14 for dinov2), so a small object "
        "spans more patches. Use a multiple of the patch size. A no-op for non-DINO embedders (collapses "
        "to the default there). Folds into the cache slug (_r<res>), so a resolution run never reuses "
        "default-res cached vectors.",
    )
    ap.add_argument(
        "--dinov3-model",
        nargs="+",
        default=["vitb16"],
        help="DINOv3 checkpoint(s) for the dinov3 embedder; SWEEP AXIS (pass multiple → one curve each, "
        "e.g. --dinov3-model vitb16 vitl16). Short aliases: "
        + ", ".join(sorted(DINOV3_VARIANTS))
        + " (or a full HF repo id). All are patch-size 16, so a bigger model gives richer features but the "
        "SAME grid density — use --resolution for finer localization. 'vitb16' is the app default (no cache "
        "tag); other sizes get their own cache slug (_m<alias>) and are downloaded on first use (gated: needs "
        "HF_TOKEN). A no-op for non-dinov3 embedders.",
    )
    ap.add_argument("--prompt-template", default="a photo of a {}")
    ap.add_argument("--out-dir", type=Path, default=Path("docs/experiments/sod-sweep"))
    ap.add_argument("--cache-dir", type=Path, default=None, help="npz cache dir (default: <out-dir>/cache)")
    ap.add_argument("--array-index", type=int, default=0)
    ap.add_argument("--array-total", type=int, default=1)
    # Built-in visualization (opt-in): plots + split galleries + prediction overlays.
    ap.add_argument(
        "--viz",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="emit plots/galleries/overlays after the sweep",
    )
    ap.add_argument(
        "--viz-seed", type=int, default=None, help="iteration to render prediction overlays for (default: 0)"
    )
    ap.add_argument("--viz-n", type=int, default=12, help="max images per montage")
    ap.add_argument(
        "--viz-band",
        choices=("minmax", "std", "none", "all"),
        default="std",
        help="seed spread on --viz plots: minmax/std band, none, or 'all' (one thin line per seed)",
    )
    ap.add_argument(
        "--summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="with --viz and ≥2 classes, also emit summary_<dataset>_<metric>.png (per-config curves "
        "macro-averaged across classes; band = across-class spread, or one line per class with "
        "--viz-band all)",
    )
    ap.add_argument(
        "--show-oracle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="overlay the oracle companion (faint dashed) on the cost and F1 plots: oracle_cost = "
        "min achievable cost, oracle_f1 = max achievable F1 (both true bounds the calibrated curve "
        "can't cross). The gap is threshold-placement noise vs detector quality — under extreme "
        "imbalance the calibrated value swings while the oracle stays flat. fpr/fnr get no oracle. "
        "Off by default.",
    )
    args = ap.parse_args()

    cache_root = args.cache_dir or (args.out_dir / "cache")
    cells = _cells(args)
    if args.array_total > 1:
        cells = [c for i, c in enumerate(cells) if i % args.array_total == args.array_index]
    print(f"{len(cells)} cell(s) to run (array {args.array_index}/{args.array_total})", flush=True)

    all_rows: list[dict] = []
    t0 = time.monotonic()
    for i, cell in enumerate(cells):
        print(f"[{i + 1}/{len(cells)}] {cell}", flush=True)
        try:
            all_rows.extend(_run_cell(cell, args, cache_root))
        except Exception:
            print(f"  ERROR in {cell}:\n{traceback.format_exc()}", flush=True)

    suffix = f".{args.array_index}" if args.array_total > 1 else ""
    out_dir = args.out_dir
    _write_results(all_rows, out_dir)
    if suffix:  # array shards write their own jsonl to merge later
        (out_dir / f"results{suffix}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in all_rows))
    print(f"\nwrote {len(all_rows)} rows to {out_dir} in {time.monotonic() - t0:.0f}s", flush=True)

    if args.viz and args.array_total == 1:
        _run_viz(args, cells, cache_root, all_rows, out_dir)
    return 0


def _build_total_timing(prep: list[dict], rows: list[dict]) -> list[dict]:
    """Combine embed+propose seconds (from ``prep_timing_summary``) with the MLP
    ``compute_ms`` summed from result rows into a total-time entry per config.

    Joined on ``(dataset, reg_name, proposal_slug)``: ``prep`` is keyed by registry
    name + slug, and rows now carry the same keys. Embedding is a cache-miss cost
    (0 on a fully-cached run); the MLP compute always runs, so the total always has
    data. Returns one dict per config with ``embed_s``/``compute_s``/``total_s``.
    """
    from collections import defaultdict

    # Reuse the plot's config label so time.png bars read identically to the curve
    # legends — one bar per distinct config (leaf/pca/alpha/k/resolution), not a single
    # "dinov3/hac" lumping every variant together.
    from plots import _config_key, _config_label

    embed_by = {(t["dataset"], t["embedder"], t["slug"]): t["embed_s"] for t in prep}
    compute_by: dict[tuple, float] = defaultdict(float)
    label_by: dict[tuple, str] = {}
    for r in rows:
        key = (r["dataset"], r.get("reg_name", r["embedder"]), r.get("proposal_slug", r["proposal"]))
        compute_by[key] += float(r.get("compute_ms", 0.0)) / 1000.0
        label_by[key] = _config_label(_config_key(r))
    out: list[dict] = []
    for key in sorted(set(embed_by) | set(compute_by)):
        e = embed_by.get(key, 0.0)
        c = compute_by.get(key, 0.0)
        out.append(
            {
                "label": label_by.get(key, f"{key[1]}/{key[2]}"),
                "embed_s": round(e, 3),
                "compute_s": round(c, 3),
                "total_s": round(e + c, 3),
            }
        )
    return out


def _run_viz(args, cells: list[dict], cache_root: Path, all_rows: list[dict], out_dir: Path) -> None:
    """Emit plots + split galleries after the sweep. (Per-cell prediction overlays and
    labeling traces are rendered inline in ``_run_cell`` from the loop's final heads.)"""
    from features import prep_timing_summary
    from plots import render_all, render_inference_time, render_summary
    from viz import render_split_gallery

    print("=== viz ===", flush=True)
    _metrics = ("cost", "fpr", "fnr", "f1", "mean_iou", "corloc")
    if all_rows:
        render_all(
            all_rows,
            out_dir / "plots",
            metrics=_metrics,
            band=args.viz_band,
            show_oracle=args.show_oracle,
            x_label="total annotations t",
            x_tag="t",
        )
        if args.summary:
            # Cross-class summary: per-config curves macro-averaged over each dataset's classes
            # (no-op unless a dataset has ≥2 classes). Same x-axis (total annotations t) as render_all.
            render_summary(
                all_rows,
                out_dir / "plots",
                metrics=_metrics,
                band=args.viz_band,
                show_oracle=args.show_oracle,
                x_label="total annotations t",
                x_tag="t",
            )
    # Total-time bar chart (per-config): embed+propose (cache misses this run) + MLP
    # train+score (always runs, so this has data even on a fully-cached run).
    total_timing = _build_total_timing(prep_timing_summary(), all_rows)
    if total_timing:
        render_inference_time(total_timing, out_dir / "plots" / "time.png")
        (out_dir / "time.json").write_text(json.dumps(total_timing, indent=2))
    else:
        print("  timing: no rows to summarize", flush=True)
    for dataset in args.datasets:
        for cls in args.classes:
            try:
                with SodDataset(dataset) as ds:
                    cs = ds.class_split(
                        cls, neg_multiple=args.neg_multiple, seed=args.split_seed, min_box_frac=args.min_box_frac
                    )
                    if not cs.positive_ids:
                        continue
                    sp = partition_split(cs, args.test_fraction, args.split_seed)
                    render_split_gallery(
                        ds,
                        sp,
                        out_dir=out_dir / "splits_gallery",
                        dataset=dataset,
                        cls=cls,
                        gallery_n=args.viz_n,
                        sample_seed=args.split_seed,
                    )
            except Exception:
                print(f"  viz error {dataset}/{cls}:\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
