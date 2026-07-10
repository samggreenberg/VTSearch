# Small-object-detection (SOD) evaluation sweep

Sweeps `dataset × class × embedder × proposal × K` and plots the
**cross-calibrated (realistic) inclusion-weighted FPR+FNR** vs the few-shot
annotation count K. Design + rationale: `docs/plans/` (small-object sweep).

## Layers
- **Library core** (`vtscore/eval/`, reused by any caller):
  `error_metrics.py` (weighted FPR+FNR + oracle), `xcal.py` (cross-calibration),
  `region_sources.py` (whole / sliding / dino / hac → `(box, vector)` bags),
  `scoring_heads.py` (MLP + cosine), `region_curve.py` (the K-loop).
- **Driver** (this dir): `datasets.py` (COCO/LVIS/VG adapter over the staged
  derived extracts + zips), `features.py` (exemplars + npz cache + split),
  `sweep.py` (orchestrator), `plots.py`.

## Method matrix
- **MLP head** (primary): every embedder. Positives = K GT-box-crop exemplars,
  negatives = whole-image vectors; threshold via cross-calibration.
- **Cosine head** (baseline): text-capable embedders only (siglip/clip/siglip2);
  K=0 is the zero-shot text point.
- DINOv2/v3 (`dinov2`/`dinov3` → patch embedders): MLP head only, and the only
  embedders valid for the `hac` proposal.
- **`--neg-regions`** (optional, default off): train the MLP's negatives on proposed-region
  crops of negative images instead of whole-image vectors. For crop/HAC proposals this
  matches the train and test distributions (test scores max-pool over region crops), which
  sharply cuts max-pool false positives — e.g. siglip/sliding FPR 0.72→0.11 in the pilot
  (at the cost of higher FNR). No-op for `whole`. `viz.py --kind predict` takes the same flag,
  so overlays match a `--neg-regions` run.
- **`--region-voting`** (optional, default off; **hac + dinov2/dinov3 only**): the faithful
  app-detector label construction, so a sweep cell reproduces what a user swiping in VTSearch
  produces (`python -m vtscore.eval --region-voting`). It changes the whole `hac` train path:
  - **Good vote** → the covering GT box (union of all instances) is **snapped to its best-IoU
    HAC node** (`snap_box_to_region`) — one positive per image (K = good swipes), an actual
    candidate the detector max-pools over, not a uniform grid pool.
  - **Bad vote** → floods the image's **childless nodes (CLS + HAC leaves)**, dropping internals,
    as negatives (MIL: no region should score high).
  - **Bag-aware** training: per-image loss weights (a busy negative image counts once, not once
    per leaf), hidden width + safe-threshold ramp sized on **votes not rows**, cross-calibration
    folds split by image, via the production `cross_calibration_threshold_cached`.
  Overrides `--neg-regions` for `hac`; a no-op for other proposals (so `--proposals whole,hac
  --region-voting` runs `whole` normally). Uses a distinct `hac_rv_*` cache slug (don't reuse a
  plain-`hac` cache). `viz.py --kind predict --region-voting` renders matching overlays.
  Only alpha=0.5 / k=12 reproduce production's tree (its fixed build params).

## Run (pilot: stop sign on COCO, single GPU)
One command — add `--viz` and the sweep also emits all the figures at the end:
```bash
srun --partition=gpu --gres=gpu:l40s:1 --cpus-per-task=8 --mem=46G --time=2:00:00 \
  .venv/bin/python scripts/sod/sweep.py \
    --datasets coco --classes "stop sign" --embedders siglip,dinov2 \
    --proposals whole,sliding,dino,hac --k-values 1,2,4,8,16 \
    --heads mlp,cosine --seeds 0,1,2 --out-dir docs/experiments/sod-sweep --viz
```
Without `--viz` you get just the data; re-render anytime with
`python scripts/sod/plots.py --results docs/experiments/sod-sweep/results.jsonl`
(and `viz.py` for galleries/overlays). `--array-index/--array-total` filter cells
for a later SLURM array; per-image embeddings cache to `--cache-dir` (default
`<out-dir>/cache`) so re-runs reuse forward passes — point it at an existing cache
to avoid re-embedding.

Outputs (under `--out-dir`, gitignored):
- `results.jsonl` / `results.csv` — one row per config × K × seed (`cost`, `fpr`, `fnr`,
  `f1`, `mean_iou`, `corloc`, `oracle_cost`, `calib_mode`, `negatives_exhaustive`, …)
- `<ds>_<class>_split.json` — the train/test split image ids (always written)
- with `--viz`: `plots/` (cost/fpr/fnr/f1/mean_iou curves + variance bands, and an
  `time.png` total-time stacked bar), `time.json`, `splits_gallery/` (bucket montages
  with GT boxes), `predictions/` (per-config MLP TP/FP/FN/TN overlays)

## Metric / threshold
`cost = w_fpr·FPR + w_fnr·FNR` (rates), weights from `--inclusion` (0 → FPR+FNR).
Primary is the cross-calibrated threshold (production path); `oracle_cost` is the
best-case min-over-threshold reference. At low K the threshold falls back to a GMM
blend (`--safe-thresholds`, on by default) — the `calib_mode` column
(`fallback|gmm|blend|xcal`) records which regime each point used.

## Plots
`plots.py` writes one figure per metric per (dataset, class): `--metrics` defaults to
`cost,fpr,fnr,f1,mean_iou` (also available: `corloc`) → `<ds>_<class>_<metric>_vs_k.png`.
`f1` is at the cross-calibrated threshold; `mean_iou` is the top-scoring region's box vs the
best GT box, averaged over test positives (`whole` → ~flat/low by construction); `corloc` =
fraction with IoU≥0.5. So you see misses (FNR) vs false alarms (FPR), detection quality (F1),
and localization (IoU) — not just the combined cost. **Total time** is a separate per-config
stacked bar (`inference_time.png`, seconds): embed+propose (a cache-miss cost, 0 s on a
fully-cached run) plus the MLP calibrate+fit+score summed over all K×seeds. Because the MLP
work always runs, this chart has data even on a fully-cached run. Each mean curve
gets a **seed-variance band** (`--band-kind minmax|std|none`, default minmax) showing the
spread across `--seeds` — at small positive counts this band is large (calibration noise),
so treat close mean curves as not significant. Overlays are off by default: `--show-oracle`
(oracle companion lines) and `--show-text-baseline` (cosine zero-shot ref) — render-only;
the data is always in `results.jsonl` (one row per seed).

## Interpreting beyond the curves
- **Built-in (opt-in):** pass `--viz` to `sweep.py` and it emits everything below after the
  run — `plots/`, `splits_gallery/`, and `predictions/` (MLP overlays per config at
  `--viz-k` [default max K] / `--viz-seed` [default first seed], `--viz-n` per montage).
  `--viz-band {minmax,std,none}` (default minmax) controls the plots' seed-variance band.
  The per-config TP/FP/FN/TN counts are also printed. (Skipped for array shards.)
- **Train/test split** is dumped by `sweep.py` to `<out-dir>/<ds>_<class>_split.json`
  (the exact image ids in each bucket: `train_pos`, `test_pos`, `train_neg`, `test_neg`).
- **`viz.py`** renders montages, offline from the cache (no GPU):
  - `--kind split` — galleries of each bucket (positives drawn with GT boxes) so you can
    see what training vs testing looks like.
  - `--kind predict` — for one config (`--embedder --proposal --k --seed`, MLP head), the
    per-test-image prediction: predicted winning region box + score (red), GT boxes
    (green), grouped **TP/FP/FN/TN** at the cross-calibrated threshold.
  ```bash
  python scripts/sod/viz.py --kind split --dataset coco --cls "stop sign" \
      --out-dir docs/experiments/sod-sweep-.../splits_gallery
  python scripts/sod/viz.py --kind predict --dataset coco --cls "stop sign" \
      --cache-dir docs/experiments/sod-sweep-.../cache \
      --embedder siglip --proposal sliding --k 8 --seed 0 \
      --out-dir docs/experiments/sod-sweep-.../predictions
  ```
  (`viz.py` recomputes the split from `--split-seed/--neg-count/--test-fraction`, which
  must match the run being inspected; predict mode is MLP-head only since cosine needs the
  text encoder.)

## Notes
- COCO negatives are exhaustive (clean FPR); LVIS/VG are non-exhaustive, so their
  FPR is an upper bound (`negatives_exhaustive=false` per row).
- Images stay zipped; pixels are streamed from the zips and cached as vectors.

## Matthew Run:
```bash
srun --partition=gpu --gres=gpu:l40s:1 --cpus-per-task=8 --mem=46G --time=8:00:00 --pty bash -l

# coco stop sign (CLIP/SigLIP/SigLIP 2, whole/sliding)
# neg regions:
python scripts/sod/sweep.py --datasets coco --classes "stop sign" --cache-dir docs/experiments/cache --embedders clip,siglip,siglip2 --proposals whole,sliding --k-values 0,1,2,3,4,5,8,12,16,24,32 --out-dir docs/experiments/sod-sweep-coco-stopsign-test5 --viz --neg-regions

# coco stop sign HAC sweep (dinov2/dinov3, alpha=0.3,0.5,0.7):

# no region voting
python scripts/sod/sweep.py --datasets coco --classes "stop sign" --cache-dir docs/experiments/cache --embedders dinov2,dinov3 --proposals hac --hac-alpha 0.3,0.5,0.7 --k-values 0,1,2,3,4,5,8,12,16,24,32 --out-dir docs/experiments/sod-sweep-coco-stopsign-dinov2-dinov3-hac-alphas-3-5-7-no-region-voting --viz

# region voting
python scripts/sod/sweep.py --datasets coco --classes "stop sign" --cache-dir docs/experiments/cache --embedders dinov2,dinov3 --proposals hac --hac-alpha 0.3,0.5,0.7 --k-values 0,1,2,3,4,5,8,12,16,24,32 --out-dir docs/experiments/sod-sweep-coco-stopsign-dinov2-dinov3-hac-alphas-3-5-7-region-voting --region-voting --viz

# traffic light
python scripts/sod/sweep.py --datasets lvis --classes "traffic light" --cache-dir docs/experiments/cache --embedders dinov2,dinov3 --proposals hac --hac-alpha 0.1,0.3,0.5,0.7,0.9 --k-values 0,1,2,3,4,5,8,12,16,24,32 --out-dir docs/experiments/sod-sweep-lvis-traffic-light-dinov2-dinov3-hac-alphas-1-3-5-7-9-region-voting --region-voting --viz
```