#!/usr/bin/env python3
"""Small-object-detection evaluation sweep (orchestrator).

Sweeps ``dataset × class × embedder × proposal × K`` and, for each config, reports
the cross-calibrated (realistic) inclusion-weighted FPR+FNR vs the few-shot
annotation count K (plus an oracle reference). The MLP head runs for every
embedder (primary); the cosine head runs as a zero-shot baseline for text-capable
embedders. See docs/plans/... (small-object sweep) for the design.

Single-GPU loop for the pilot; ``--array-index/--array-total`` filter cells so the
same command drops into a SLURM array later. Per-image embeddings are cached to
npz (``--cache-dir``) so re-runs and multiple K reuse forward passes.

Example (stop sign on COCO):
    srun --partition=gpu --gres=gpu:l40s:1 --cpus-per-task=8 --mem=46G --time=8:00:00 \\ # or --pty bash -l
      .venv/bin/python scripts/sod/sweep.py \\
        --datasets coco --classes "stop sign" --embedders siglip,dinov2 \\
        --proposals whole,sliding,dino,hac --k-values 1,2,4,8,16 \\
        --heads mlp,cosine --seeds 0,1,2 --out-dir docs/experiments/sod-sweep
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

from vtscore.eval.region_curve import evaluate_region_curve
from vtscore.eval.region_sources import build_region_source

EMBEDDER_ALIASES = {"dinov2": "dinov2_patch", "dinov3": "dinov3_patch"}
TEXT_EMBEDDERS = {"siglip", "siglip2", "clip"}


def _floats(s: str) -> tuple[float, ...]:
    return tuple(float(x) for x in s.split(",") if x.strip())


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _strs(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _proposal_slug(proposal: str, args, alpha: float) -> str:
    if proposal == "sliding":
        sc = "-".join(str(x) for x in args.scales)
        return f"sliding_s{sc}_o{args.overlap}_w{args.min_window}"
    if proposal == "dino":
        return f"dino_k{args.hac_k}_a{args.hac_alpha_default}"
    if proposal == "hac":
        return f"hac_k{args.hac_k}_a{alpha}"
    return "whole"


def _build_source(proposal: str, embedder, args, alpha: float):
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

    embedder = get_embedder(reg_name)
    try:
        source = _build_source(proposal, embedder, args, alpha)
    except Exception as exc:
        print(f"  skip {cell}: {exc}", flush=True)
        return []

    with SodDataset(cell["dataset"]) as ds:
        split = ds.class_split(cell["class"], neg_count=args.neg_count, seed=args.split_seed)
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
                neg_count=args.neg_count,
                test_fraction=args.test_fraction,
                split=buckets,
            )
        slug = _proposal_slug(proposal, args, alpha)
        cache = FeatureCache(cache_root, cell["dataset"], reg_name, slug)
        meta = {
            "dataset": cell["dataset"],
            "class": cell["class"],
            "embedder": cell["embedder"],
            "reg_name": reg_name,  # registry name; joins rows to prep_timing_summary
            "proposal": proposal,
            "proposal_slug": slug,
            "alpha": alpha,
            "negatives_exhaustive": split.negatives_exhaustive,
            "neg_regions": args.neg_regions,
            "n_pos_total": len(split.positive_ids),
            "n_neg_total": len(split.negative_ids),
        }
        inputs = build_curve_inputs(
            ds, source, buckets, cache, class_name=cell["class"], meta=meta, neg_regions=args.neg_regions
        )

    rows: list[dict] = []
    heads = list(args.heads)
    for head in heads:
        if head == "cosine" and cell["embedder"] not in TEXT_EMBEDDERS:
            continue  # DINO has no text encoder
        query_vec = None
        k_list = [k for k in args.k_values if k > 0]
        if head == "cosine":
            query_vec = source.embed_text(args.prompt_template.format(cell["class"]))
            if query_vec is None:
                continue
            k_list = [0, *k_list]  # zero-shot baseline point
        rows.extend(
            evaluate_region_curve(
                inputs,
                head,
                k_values=k_list,
                seeds=args.seeds,
                neg_ratio=args.neg_ratio,
                inclusion=args.inclusion,
                safe_thresholds=args.safe_thresholds,
                calibrate_count=args.calibrate_count,
                cal_fraction=args.cal_fraction,
                query_vec=query_vec,
            )
        )
    return rows


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
    ap.add_argument("--heads", type=_strs, default=["mlp", "cosine"])
    ap.add_argument("--k-values", type=_ints, default=[1, 2, 4, 8, 16])
    ap.add_argument("--seeds", type=_ints, default=[0, 1, 2])
    ap.add_argument("--split-seed", type=int, default=0, help="seed for class split + pool partition")
    ap.add_argument("--neg-count", type=int, default=690, help="size of the sampled evaluation negative pool")
    ap.add_argument("--neg-ratio", type=int, default=1, help="training negatives per positive (per K)")
    ap.add_argument(
        "--neg-regions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="train MLP negatives on proposed-region crops of negative images (matches the test "
        "distribution for crop/HAC proposals) instead of whole-image vectors",
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
    # Built-in visualization (opt-in): plots + split galleries + MLP prediction overlays.
    ap.add_argument(
        "--viz",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="emit plots/galleries/overlays after the sweep",
    )
    ap.add_argument("--viz-k", type=int, default=None, help="K to render prediction overlays at (default: max K)")
    ap.add_argument("--viz-seed", type=int, default=None, help="seed for prediction overlays (default: first --seeds)")
    ap.add_argument("--viz-n", type=int, default=12, help="max images per montage")
    ap.add_argument(
        "--viz-band", choices=("minmax", "std", "none"), default="std", help="seed-variance band on --viz plots"
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
    """Emit plots + split galleries + MLP prediction overlays after the sweep."""
    from features import prep_timing_summary
    from plots import render_all, render_inference_time
    from viz import render_predictions, render_split_gallery

    print("=== viz ===", flush=True)
    if all_rows:
        render_all(all_rows, out_dir / "plots", band=args.viz_band)
    # Total-time bar chart (per-config): embed+propose (cache misses this run) + MLP
    # train+score (always runs, so this has data even on a fully-cached run).
    total_timing = _build_total_timing(prep_timing_summary(), all_rows)
    if total_timing:
        render_inference_time(total_timing, out_dir / "plots" / "time.png")
        (out_dir / "time.json").write_text(json.dumps(total_timing, indent=2))
    else:
        print("  timing: no rows to summarize", flush=True)
    viz_k = args.viz_k if args.viz_k is not None else max((k for k in args.k_values if k > 0), default=1)
    viz_seed = args.viz_seed if args.viz_seed is not None else (args.seeds[0] if args.seeds else 0)
    for dataset in args.datasets:
        for cls in args.classes:
            try:
                with SodDataset(dataset) as ds:
                    cs = ds.class_split(cls, neg_count=args.neg_count, seed=args.split_seed)
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
                    for cell in cells:
                        if cell["dataset"] != dataset or cell["class"] != cls:
                            continue
                        if cell["proposal"] == "hac" and cell["embedder"] not in {"dinov2", "dinov3"}:
                            continue  # no MLP cache for this combo
                        try:
                            render_predictions(
                                ds,
                                sp,
                                cache_dir=cache_root,
                                out_dir=out_dir / "predictions",
                                dataset=dataset,
                                cls=cls,
                                embedder=cell["embedder"],
                                proposal=cell["proposal"],
                                alpha=cell["alpha"],
                                slug=_proposal_slug(cell["proposal"], args, cell["alpha"]),
                                k=viz_k,
                                seed=viz_seed,
                                neg_ratio=args.neg_ratio,
                                inclusion=args.inclusion,
                                safe_thresholds=args.safe_thresholds,
                                gallery_n=args.viz_n,
                                neg_regions=args.neg_regions,
                            )
                        except Exception:
                            print(f"  viz predict error {cell}:\n{traceback.format_exc()}", flush=True)
            except Exception:
                print(f"  viz error {dataset}/{cls}:\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
