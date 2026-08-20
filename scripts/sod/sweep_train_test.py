#!/usr/bin/env python3
"""SOD realistic sweep over a pre-labeled, pre-split image directory (boxless / whole-image).

Unlike ``scripts/sod/sweep.py`` (which picks a dataset + class and auto-splits the
matching/negative images into a sim/test partition), this script runs the SAME
realistic Autopilot cost-vs-``t`` evaluation on images you have already organized on
disk as a fixed binary, pre-split tree:

    <image-dir>/train/pos/*   <image-dir>/train/neg/*
    <image-dir>/test/pos/*    <image-dir>/test/neg/*

``train/`` is the labeling pool Autopilot reveals in active-learning order; ``test/``
is the held-out set the cost/fpr/fnr/f1 curve is measured on. Positives are ``pos/``,
negatives are ``neg/`` - no class concept, no dataset registry.

Scope: the ``whole`` proposal only (image-level CLS/embedding vector). The folders
carry no bounding boxes, and ``hac``/region-voting build each positive's training
exemplar FROM a box, so they are out of scope here (see the plan). IoU/CorLoc are
therefore ``NaN`` (no boxes to score against). The evaluation core
(:func:`vtscore.eval.region_curve.evaluate_realistic_curve`) is reused verbatim.

Example:
    srun --partition=gpu --gres=gpu:l40s:1 --cpus-per-task=4 --mem=32G --time=1:00:00 \\
      .venv/bin/python scripts/sod/sweep_train_test.py \\
        --image-dir /exp/$USER/my_corpus_split --name my_corpus \\
        --embedders dinov3,siglip2 --iterations 3 --max-labels 60 \\
        --out-dir /exp/$USER/sweep-train-test/my_corpus
        
Matthew Usage:
# Cats SigLIP
python scripts/sod/sweep_train_test.py --image-dir data/cats --cache-dir docs/experiments/sod-sweep/cache  --name cats --out-dir docs/experiments/sod-sweep/cats --reference-csv data/cats/reference/reference.csv --embedders siglip --iterations 3 --max-labels 150 --viz --viz-band all --labeling-trace --show-oracle --query "cat"

# Cats SigLIP2
python scripts/sod/sweep_train_test.py --image-dir data/cats --cache-dir docs/experiments/sod-sweep/cache  --name cats --out-dir docs/experiments/sod-sweep/cats-siglip2large --reference-csv data/cats/reference/reference.csv --embedders siglip2_l --iterations 3 --max-labels 150 --viz --viz-band all --labeling-trace --show-oracle --query "cat"

# Cats HAC
python scripts/sod/sweep_train_test.py --image-dir data/cats --name cats --embedders dinov3 -- --iterations 3 --max-labels 150 --resolution 448 --out-dir docs/experiments/sod-sweep/cats --viz --viz-band all --labeling-trace
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

# This file lives in scripts/sod/ alongside the machinery it reuses, which imports
# its siblings bare (``from datasets import ...``). Ensure that dir is importable
# whether run as a script (sys.path[0] is already scripts/sod) or imported as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import FeatureCache, Split, build_curve_inputs, dump_split, slugify  # noqa: E402
from sweep import (  # noqa: E402
    EMBEDDER_ALIASES,
    TEXT_EMBEDDERS,
    _apply_resolution,
    _resolve_dinov3_model,
    _write_results,
)

from vtscore.eval.region_curve import evaluate_realistic_curve  # noqa: E402
from vtscore.eval.region_sources import build_region_source  # noqa: E402

# Mirrors scripts/vg/split_train_test.py's set; the loader walks these under the
# four bucket dirs.
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"}

# The full-frame box every TRAIN positive gets: the ``whole`` source ignores box
# coordinates (it hardcodes [0,0,1,1]) but emits one image-level exemplar PER box
# entry, so a positive needs exactly one entry or training gets zero positive rows.
_FULL_FRAME_BOX = (0.0, 0.0, 1.0, 1.0)

# Passed to the viz renderers as their ``alpha``; only used to build an ``_a{alpha}``
# cache tag when proposal == "hac", so its value is irrelevant for the whole path.
_WHOLE_ALPHA = 0.5

_BUCKETS = ("train_pos", "train_neg", "test_pos", "test_neg")


class DirDataset:
    """Folder-backed stand-in for ``SodDataset``: the ``load_image(id)`` + context-manager
    contract that ``build_curve_inputs`` depends on, over a ``train/{pos,neg}`` +
    ``test/{pos,neg}`` tree.

    Assigns stable integer ids by walking the four buckets in a fixed order and
    sorting filenames within each, so the ``FeatureCache`` npz keys (which are the
    image ids) are reproducible across runs on the same directory.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._paths: dict[int, Path] = {}
        self.buckets: dict[str, list[int]] = {b: [] for b in _BUCKETS}
        next_id = 1
        for split_name in ("train", "test"):
            for label in ("pos", "neg"):
                d = self.root / split_name / label
                files = (
                    sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
                    if d.is_dir()
                    else []
                )
                for p in files:
                    self._paths[next_id] = p
                    self.buckets[f"{split_name}_{label}"].append(next_id)
                    next_id += 1

    def load_image(self, image_id: int) -> Image.Image:
        return Image.open(self._paths[image_id]).convert("RGB")

    def close(self) -> None:  # no open handles; kept for the SodDataset-compatible contract
        pass

    def __enter__(self) -> DirDataset:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _build_split(ds: DirDataset, negatives_exhaustive: bool) -> Split:
    """A ``features.Split`` straight from the directory buckets (no auto-partition).

    Every TRAIN positive gets one dummy full-frame box (required: ``build_curve_inputs``
    indexes ``split.gt_boxes[iid]`` for train positives, and the ``whole`` source emits
    one exemplar per box). TEST positives get no entry, so ``test_gt_boxes`` stays empty
    and IoU/CorLoc resolve to ``NaN``.
    """
    return Split(
        train_pos=ds.buckets["train_pos"],
        test_pos=ds.buckets["test_pos"],
        train_neg=ds.buckets["train_neg"],
        test_neg=ds.buckets["test_neg"],
        gt_boxes={pid: [_FULL_FRAME_BOX] for pid in ds.buckets["train_pos"]},
        negatives_exhaustive=negatives_exhaustive,
    )


def _run_embedder(embedder_name: str, ds: DirDataset, split: Split, args, cache_root: Path) -> list[dict]:
    from vtscore.config import DINOV3_MODEL_ID
    from vtscore.media import get_embedder

    reg_name = EMBEDDER_ALIASES.get(embedder_name, embedder_name)
    embedder = get_embedder(reg_name)

    # DINOv3 checkpoint + resolution knobs (embedder-forward axes) — mirror sweep.py's
    # handling so a non-default checkpoint/resolution changes the embedding and gets its
    # own cache slug. Always reset the shared singleton so a prior call doesn't leak.
    dinov3_full_id = None
    if embedder_name == "dinov3":
        dinov3_full_id = _resolve_dinov3_model(args.dinov3_model) if args.dinov3_model else None
        embedder.set_model_id(dinov3_full_id)
    if embedder_name in ("dinov2", "dinov3"):
        _apply_resolution(embedder, args.resolution)

    source = build_region_source("whole", embedder)

    slug = "whole"
    if args.resolution:
        slug = f"{slug}_r{args.resolution}"
    if dinov3_full_id and dinov3_full_id != DINOV3_MODEL_ID:
        slug = f"{slug}_m{args.dinov3_model}"

    cache = FeatureCache(cache_root, args.name, reg_name, slug)
    meta = {
        "dataset": args.name,
        "class": args.label,
        "embedder": embedder_name,
        "reg_name": reg_name,
        "proposal": "whole",
        "proposal_slug": slug,
        "region_voting": False,
        "negatives_exhaustive": split.negatives_exhaustive,
        "resolution": args.resolution,
        "dinov3_model": args.dinov3_model,
        "n_pos_total": len(split.train_pos) + len(split.test_pos),
        "n_neg_total": len(split.train_neg) + len(split.test_neg),
    }
    inputs = build_curve_inputs(
        ds,
        source,
        split,
        cache,
        class_name=args.label,
        meta=meta,
        region_voting=False,
        neg_regions=False,
        build_pool=True,
    )

    # Cold-start ranking: a text query seeds it for text-capable embedders; DINO uses
    # the seed positive's exemplar (query_vec=None).
    query_vec = None
    if embedder_name in TEXT_EMBEDDERS and args.query:
        query_vec = source.embed_text(args.query)

    want_finals = args.viz or args.confidence_gallery or args.labeling_trace
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
        return_finals=want_finals,
        threshold_rule=args.threshold_rule,
        threshold_smooth=args.threshold_smooth,
        head_strategy=args.head_strategy,
    )
    if not want_finals:
        return result
    rows, finals = result
    _render_viz(ds, split, args, cache_root, embedder_name, slug, finals)
    return rows


def _render_viz(ds, split, args, cache_root: Path, embedder_name: str, slug: str, finals: dict) -> None:
    """Per-embedder overlays/galleries from the loop's final heads (``return_finals=True``).

    Mirrors sweep.py's ``_render_realistic_viz`` but passes the ``DirDataset`` directly
    (no ``SodDataset`` reopen). The renderers only need ``load_image`` + the ``Split``,
    and are boxless-safe (they key on ``proposal == "hac"``, which is never true here).
    ``--training-nodes`` is HAC-only, so it has no analog on the whole path.
    """
    import traceback

    from viz import render_confidence_gallery, render_labeling_trace, render_predictions_realistic

    pred_dir = args.out_dir / "predictions" / slug
    common = dict(
        cache_dir=cache_root,
        out_dir=pred_dir,
        dataset=args.name,
        cls=args.label,
        embedder=embedder_name,
        proposal="whole",
        alpha=_WHOLE_ALPHA,
        slug=slug,
    )
    try:
        if args.viz:
            viz_seed = args.viz_seed if args.viz_seed is not None else 0
            fin = finals.get(viz_seed)
            if fin is not None:
                render_predictions_realistic(
                    ds, split, **common, predict=fin["predict"], thr=fin["threshold"], t=fin["t"], gallery_n=args.viz_n
                )
            else:
                print(f"  [predict] skip {embedder_name}: no final head for seed {viz_seed}", flush=True)
        if args.confidence_gallery:
            for s, f in sorted(finals.items()):
                render_confidence_gallery(
                    ds, split, **common, predict=f["predict"], thr=f["threshold"], t=f["t"], seed=s
                )
        if args.labeling_trace:
            for s, f in sorted(finals.items()):
                render_labeling_trace(
                    ds,
                    split,
                    f["trace"],
                    cache_dir=cache_root,
                    slug=slug,
                    out_dir=args.out_dir / "labeling_trace" / slug,
                    dataset=args.name,
                    cls=args.label,
                    embedder=embedder_name,
                    proposal="whole",
                    alpha=_WHOLE_ALPHA,
                    seed=s,
                    images=args.trace_images,
                )
    except Exception:
        print(f"  viz error {embedder_name}:\n{traceback.format_exc()}", flush=True)


def _run_plots(args, all_rows: list[dict], ds, split, out_dir: Path) -> None:
    """Cost-curve plots (seed band) + split gallery + a total-time bar, after the sweep.

    IoU/CorLoc are omitted from the metric set because they are always ``NaN`` on the
    boxless whole path (plotting them would emit blank axes).
    """
    import traceback

    import plots_train_test
    from features import prep_timing_summary
    from plots import render_inference_time
    from sweep import _build_total_timing
    from viz import render_split_gallery

    if all_rows:
        # plots_train_test adds the confusion-matrix family (derived from the stored
        # rates, so it also works on older results.jsonl) plus threshold-free AUROC,
        # and supports the dual-output 'all+std' band.
        reference = plots_train_test.load_reference_csv(args.reference_csv) if args.reference_csv else None
        plots_train_test.render_all(
            all_rows,
            out_dir / "plots",
            metrics=plots_train_test.TRAIN_TEST_METRICS,
            band=args.viz_band,
            show_oracle=args.show_oracle,
            x_label="total annotations t",
            x_tag="t",
            reference=reference,
        )
    total_timing = _build_total_timing(prep_timing_summary(), all_rows)
    if total_timing:
        render_inference_time(total_timing, out_dir / "plots" / "time.png")
    try:
        render_split_gallery(
            ds,
            split,
            out_dir=out_dir / "splits_gallery",
            dataset=args.name,
            cls=args.label,
            gallery_n=args.viz_n,
            sample_seed=0,
        )
    except Exception:
        print(f"  split gallery error:\n{traceback.format_exc()}", flush=True)


def _strs(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-dir", type=Path, required=True, help="root holding train/{pos,neg} + test/{pos,neg}")
    ap.add_argument("--name", required=True, help="corpus slug (namespaces the feature cache + meta.dataset)")
    ap.add_argument("--label", default=None, help="positive-class name for meta/rows (default: --name)")
    ap.add_argument("--embedders", type=_strs, default=["dinov3"], help="comma list; proposal is always 'whole'")
    ap.add_argument("--out-dir", type=Path, required=True, help="where results.jsonl/.csv + split.json go")
    ap.add_argument("--cache-dir", type=Path, default=None, help="feature cache root (default: <out-dir>/cache)")
    # Realistic-loop knobs (defaults mirror scripts/sod/sweep.py).
    ap.add_argument("--iterations", type=int, default=3, help="labeling-loop runs (seeds 0..N-1)")
    ap.add_argument("--max-labels", type=int, default=60, help="max total annotations t per seed")
    ap.add_argument("--inclusion", type=int, default=0, help="0=FPR+FNR; >0 favors recall; <0 favors precision")
    ap.add_argument("--calibrate-count", type=int, default=2)
    ap.add_argument("--cal-fraction", type=float, default=0.5)
    ap.add_argument("--good-to-start", type=int, default=3)
    ap.add_argument("--bad-to-start", type=int, default=4)
    ap.add_argument("--retrain-cadence", type=int, default=1, help="retrain every N labels (1 = per-vote)")
    ap.add_argument("--select-strategy", choices=("autopilot",), default="autopilot")
    ap.add_argument("--stop-at-done", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--safe-thresholds", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--threshold-rule", choices=["conformal", "rank-transfer", "argmin"], default="conformal")
    ap.add_argument("--threshold-smooth", choices=["none", "med3"], default="none")
    ap.add_argument(
        "--head-strategy",
        default="mlp",
        choices=["mlp", "linear", "svm", "reg-mlp", "anneal-svm", "anneal-linear", "anneal-reg"],
    )
    ap.add_argument("--query", default=None, help="text query seeding cold-start (text embedders only)")
    ap.add_argument("--resolution", type=int, default=None, help="DINO forward resolution (multiple of patch size)")
    ap.add_argument("--dinov3-model", default=None, help="DINOv3 checkpoint alias/HF id (dinov3 embedder only)")
    ap.add_argument(
        "--negatives-exhaustive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="record whether neg/ enumerates every negative (feeds meta/row flags only)",
    )
    # Built-in visualization (opt-in): overlays + galleries + cost-curve plots. Reuses
    # scripts/sod/viz.py + plots.py with the DirDataset. --training-nodes (HAC-only) has
    # no analog on the boxless whole path.
    ap.add_argument(
        "--viz",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="after the sweep: TP/FP/FN/TN prediction overlays + cost/f1/fpr/fnr plots + split gallery",
    )
    ap.add_argument("--viz-seed", type=int, default=None, help="iteration to render overlays for (default: 0)")
    ap.add_argument("--viz-n", type=int, default=12, help="max images per montage/gallery")
    ap.add_argument(
        "--viz-band",
        choices=("minmax", "std", "none", "all", "all+std"),
        default="std",
        help="seed spread on --viz plots: minmax/std band, none, 'all' (one line per seed), or "
        "'all+std' (BOTH, as two files per metric: <metric>_vs_t.png with every seed drawn "
        "individually, and <metric>_vs_t_summary.png with mean +/- stdev). Prefer 'all+std' when "
        "reading a new run: a single outlier seed and a genuinely wide spread look identical in a "
        "summary alone",
    )
    ap.add_argument(
        "--reference-csv",
        type=Path,
        default=None,
        help="path to a 'metric,value' CSV of external reference numbers (e.g. "
        "data/cats/reference/reference.csv) drawn as a flat black dash-dot line on each "
        "metric it names. No header needed; every metric is optional and the order is "
        "arbitrary, so a file may carry any subset of cost/fpr/fnr/f1/accuracy/"
        "balanced_accuracy/precision/recall/auroc. Names match case/space/hyphen-"
        "insensitively; a non-float value is an error",
    )
    ap.add_argument(
        "--show-oracle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="overlay the oracle bound (faint dashed) on every metric with a non-degenerate "
        "best-over-τ value: cost (min achievable) and f1/accuracy/balanced_accuracy (max "
        "achievable). fpr/fnr/precision/recall get none (degenerate optimum), nor does auroc "
        "(threshold-free). Needs a run made after the oracle columns landed; older "
        "results.jsonl carry NaN there and simply draw no dashed line",
    )
    ap.add_argument(
        "--confidence-gallery",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="per embedder + seed: every test image sorted by descending detector confidence, captioned "
        "id/confidence/threshold (can be many images per seed)",
    )
    ap.add_argument(
        "--labeling-trace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="per seed: the images labeled in order + trace.csv/json (phase/head/threshold per step)",
    )
    ap.add_argument(
        "--trace-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="with --labeling-trace, also render the per-step PNGs (--no-trace-images = trace.json/csv only)",
    )
    args = ap.parse_args(argv)

    if args.label is None:
        args.label = args.name
    if not args.image_dir.is_dir():
        raise SystemExit(f"--image-dir not a directory: {args.image_dir}")

    ds = DirDataset(args.image_dir)
    counts = {b: len(ds.buckets[b]) for b in _BUCKETS}
    print(f"corpus '{args.name}': {counts}", flush=True)
    if not ds.buckets["train_pos"]:
        raise SystemExit(f"no train positives found under {args.image_dir}/train/pos (need >=1)")
    if not ds.buckets["train_neg"]:
        raise SystemExit(f"no train negatives found under {args.image_dir}/train/neg (need >=1)")
    if not ds.buckets["test_pos"] or not ds.buckets["test_neg"]:
        raise SystemExit(f"need >=1 test positive AND >=1 test negative to measure the curve; got {counts}")

    split = _build_split(ds, args.negatives_exhaustive)
    cache_root = args.cache_dir or (args.out_dir / "cache")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dump_split(
        args.out_dir / f"{slugify(args.name)}_split.json",
        dataset=args.name,
        class_name=args.label,
        split_seed=0,
        neg_multiple=0,
        test_fraction=0.0,
        split=split,
    )

    all_rows: list[dict] = []
    with ds:
        for embedder_name in args.embedders:
            print(f"  embedder={embedder_name} (whole)...", flush=True)
            all_rows.extend(_run_embedder(embedder_name, ds, split, args, cache_root))

    _write_results(all_rows, args.out_dir)
    print(f"wrote {len(all_rows)} rows to {args.out_dir}/results.jsonl", flush=True)

    if args.viz:
        _run_plots(args, all_rows, ds, split, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
