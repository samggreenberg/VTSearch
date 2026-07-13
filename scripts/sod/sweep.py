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
from datasets import SodDataset
from features import FeatureCache, build_curve_inputs, dump_split, partition_split, slugify

from vtscore.eval.region_curve import evaluate_realistic_curve
from vtscore.eval.region_sources import build_region_source

EMBEDDER_ALIASES = {"dinov2": "dinov2_patch", "dinov3": "dinov3_patch"}
TEXT_EMBEDDERS = {"siglip", "siglip2", "clip"}


def _floats(s: str) -> tuple[float, ...]:
    return tuple(float(x) for x in s.split(",") if x.strip())


def _strs(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _proposal_slug(proposal: str, args, alpha: float, region_voting: bool = False) -> str:
    if proposal == "sliding":
        sc = "-".join(str(x) for x in args.scales)
        return f"sliding_s{sc}_o{args.overlap}_w{args.min_window}"
    if proposal == "dino":
        return f"dino_k{args.hac_k}_a{args.hac_alpha_default}"
    if proposal == "hac":
        # Region-voting rewrites the exemplar (snapped, one per image) and stores a
        # leaf_mask, so it must not share a cache with the box-pool hac variant.
        rv = "_rv" if region_voting else ""
        return f"hac{rv}_k{args.hac_k}_a{alpha}"
    return "whole"


def _build_source(proposal: str, embedder, args, alpha: float, region_voting: bool = False):
    from vtscore.config import DINOV2_MODEL_ID, resolve_device

    return build_region_source(
        proposal,
        embedder,
        scales=args.scales,
        overlap=args.overlap,
        min_window=args.min_window,
        hac_k=args.hac_k,
        hac_alpha=alpha,
        dino_model_id=DINOV2_MODEL_ID,
        dino_device=resolve_device(),
        dino_register_tokens=0,
        region_voting=region_voting,
    )


def _cells(args) -> list[dict]:
    """Enumerate (dataset, class, embedder, proposal, alpha) cells."""
    cells: list[dict] = []
    for dataset in args.datasets:
        for cls in args.classes:
            for embedder in args.embedders:
                for proposal in args.proposals:
                    alphas = args.hac_alpha if proposal == "hac" else [args.hac_alpha_default]
                    for alpha in alphas:
                        cells.append(
                            {
                                "dataset": dataset,
                                "class": cls,
                                "embedder": embedder,
                                "proposal": proposal,
                                "alpha": alpha,
                            }
                        )
    return cells


def _run_cell(cell: dict, args, cache_root: Path) -> list[dict]:
    from vtscore.media import get_embedder

    reg_name = EMBEDDER_ALIASES.get(cell["embedder"], cell["embedder"])
    proposal, alpha = cell["proposal"], cell["alpha"]

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
    try:
        source = _build_source(proposal, embedder, args, alpha, region_voting=region_voting)
    except Exception as exc:
        print(f"  skip {cell}: {exc}", flush=True)
        return []

    with SodDataset(cell["dataset"]) as ds:
        split = ds.class_split(cell["class"], neg_multiple=args.neg_multiple, seed=args.split_seed)
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
        slug = _proposal_slug(proposal, args, alpha, region_voting=region_voting)
        cache = FeatureCache(cache_root, cell["dataset"], reg_name, slug)
        meta = {
            "dataset": cell["dataset"],
            "class": cell["class"],
            "embedder": cell["embedder"],
            "reg_name": reg_name,  # registry name; joins rows to prep_timing_summary
            "proposal": proposal,
            "proposal_slug": slug,
            "alpha": alpha,
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
            ds.class_split(cell["class"], neg_multiple=args.neg_multiple, seed=args.split_seed)
            if args.viz:
                fin = finals.get(viz_seed)
                if fin is not None:
                    render_predictions_realistic(
                        ds,
                        buckets,
                        cache_dir=cache_root,
                        out_dir=args.out_dir / "predictions",
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
                        out_dir=args.out_dir / "labeling_trace",
                        dataset=cell["dataset"],
                        cls=cell["class"],
                        embedder=cell["embedder"],
                        proposal=cell["proposal"],
                        alpha=cell["alpha"],
                        seed=s,
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
        "--neg-multiple",
        type=int,
        default=100,
        help="evaluation negative-pool size as a multiple of the class's positive count "
        "(pool = neg_multiple × n_positives; prevalence ≈ 1/(1+neg_multiple), constant across classes)",
    )
    ap.add_argument(
        "--region-voting",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="faithful app-detector label construction for the hac proposal on dinov2/dinov3: "
        "good votes snap to the nearest HAC node, bad votes flood CLS+leaves as negatives, "
        "bag-aware per-image weighting + grouped cross-calibration (mirrors `vtscore.eval --region-voting`). "
        "No-op for non-hac proposals. Uses a distinct 'hac_rv' cache slug.",
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
    ap.add_argument("--test-fraction", type=float, default=0.5)
    ap.add_argument("--inclusion", type=int, default=0, help="0=FPR+FNR; >0 favors recall; <0 favors precision")
    ap.add_argument("--safe-thresholds", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--calibrate-count", type=int, default=2)
    ap.add_argument("--cal-fraction", type=float, default=0.5)
    ap.add_argument("--scales", type=_floats, default=(1.0, 0.75, 0.5, 0.3))
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--min-window", type=int, default=48)
    ap.add_argument("--hac-k", type=int, default=12)
    ap.add_argument("--hac-alpha", type=_floats, default=(0.5,), help="swept for the hac proposal")
    ap.add_argument("--hac-alpha-default", type=float, default=0.5, help="alpha for the dino proposer's tree")
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

    embed_by = {(t["dataset"], t["embedder"], t["slug"]): t["embed_s"] for t in prep}
    compute_by: dict[tuple, float] = defaultdict(float)
    label_by: dict[tuple, str] = {}
    for r in rows:
        key = (r["dataset"], r.get("reg_name", r["embedder"]), r.get("proposal_slug", r["proposal"]))
        compute_by[key] += float(r.get("compute_ms", 0.0)) / 1000.0
        tag = f" α{r['alpha']}" if r.get("proposal") == "hac" else ""
        label_by[key] = f"{r['embedder']}/{r['proposal']}{tag}"
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
    from plots import render_all, render_inference_time
    from viz import render_split_gallery

    print("=== viz ===", flush=True)
    if all_rows:
        render_all(
            all_rows,
            out_dir / "plots",
            metrics=("cost", "fpr", "fnr", "f1", "mean_iou", "corloc"),
            band=args.viz_band,
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
                    cs = ds.class_split(cls, neg_multiple=args.neg_multiple, seed=args.split_seed)
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
