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
- `results.jsonl` / `results.csv` — one row per config × K × seed (`cost`, `fpr`,
  `fnr`, `oracle_cost`, `calib_mode`, `negatives_exhaustive`, …)
- `<ds>_<class>_split.json` — the train/test split image ids (always written)
- with `--viz`: `plots/` (cost/fpr/fnr curves + variance bands), `splits_gallery/`
  (bucket montages with GT boxes), `predictions/` (per-config MLP TP/FP/FN/TN overlays)

## Metric / threshold
`cost = w_fpr·FPR + w_fnr·FNR` (rates), weights from `--inclusion` (0 → FPR+FNR).
Primary is the cross-calibrated threshold (production path); `oracle_cost` is the
best-case min-over-threshold reference. At low K the threshold falls back to a GMM
blend (`--safe-thresholds`, on by default) — the `calib_mode` column
(`fallback|gmm|blend|xcal`) records which regime each point used.

## Plots
`plots.py` writes one figure per metric per (dataset, class): `--metrics cost,fpr,fnr`
(default all three) → `<ds>_<class>_{cost,fpr,fnr}_vs_k.png`, so you can see whether
error is misses (FNR) vs false alarms (FPR), not just the combined cost. Each mean curve
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
srun --partition=gpu --gres=gpu:l40s:1 --cpus-per-task=8 --mem=46G --time=4:00:00 --pty bash -l

python scripts/sod/sweep.py --datasets coco --classes "stop sign" --cache-dir docs/experiments/sod-sweep-coco-stopsign-test4/cache --k-values 0,1,2,3,4,5,8,16,32 --out-dir docs/experiments/sod-sweep-coco-stopsign-testx


python scripts/sod/sweep.py --datasets coco --classes "stop sign" --cache-dir docs/experiments/cache --embedders clip,siglip,siglip2 --proposals whole,sliding --k-values 0,1,2,3,4,5,8,12,16,24,32 --out-dir docs/experiments/sod-sweep-coco-stopsign-test5 --viz 
```