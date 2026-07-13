# Small-object-detection (SOD) evaluation sweep

Sweeps `dataset × class × embedder × proposal`, running the **realistic Autopilot
active-learning labeling loop** for each config, and plots **cross-calibrated
inclusion-weighted FPR+FNR** (and F1/IoU) vs the total annotation count `t`. Design +
rationale: `docs/plans/` (small-object sweep).

## Layers
- **Library core** (`vtscore/eval/`, reused by any caller):
  `error_metrics.py` (weighted FPR+FNR + oracle), `xcal.py` (cross-calibration),
  `region_sources.py` (whole / sliding / dino / hac → `(box, vector)` bags),
  `scoring_heads.py` (MLP + cosine), `region_curve.py` (the K-loop).
- **Driver** (this dir): `datasets.py` (COCO/LVIS/VG adapter over the staged
  derived extracts + zips), `features.py` (exemplars + npz cache + split),
  `sweep.py` (orchestrator), `plots.py`.

## Method matrix
- **Detector**: an MLP trained on the labeled good/bad votes accumulated by the loop;
  threshold via cross-calibration (GMM-blended at low label counts, `--safe-thresholds`).
- **Cold-start**: before both a good and a bad vote exist, items are ranked by cosine to
  the seed exemplar (DINO/patch) or to the class text prompt (text-capable embedders).
- DINOv2/v3 (`dinov2`/`dinov3` → patch embedders) are the only embedders valid for the
  `hac` proposal.
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
  A no-op for other proposals (so `--proposals whole,hac --region-voting` runs `whole`
  normally). Uses a distinct `hac_rv_*` cache slug (don't reuse a plain-`hac` cache). Only
  alpha=0.5 / k=12 reproduce production's tree (its fixed build params).

## Labeling loop (realistic Autopilot)
Each cell simulates the app's **Autopilot active-learning loop**, so the x-axis is
**total annotations `t`** (good+bad) and the pos/neg mix **emerges** (it is not a knob).
Per step: seed one known positive, rank the training pool (cosine-to-seed exemplar for
text-less DINO/patch, else the text query), label the item the current Autopilot phase's
select mode surfaces — `top` (highest), `hard` (nearest the decision boundary), or `new`
(diversity via `DiversityTree.next_sample`) — reveal its ground-truth label, retrain from
scratch, re-score, and record `cost/fpr/fnr/f1/mean_iou/corloc` at `t`. Faithful port of
`autopilot-state.service.ts` (good→bad→hard→new phase machine) + the frontend `autoSelectNext`
select modes; the smart-indicator regresses held-out test cost (a documented approximation of
the app's cached vote-eval error cost). Region-voting good/bad construction (snap positive,
flood leaves) is reused when `--region-voting` is set on `hac`+dinov2/dinov3.
- Knobs: `--max-labels` (default 60), `--iterations` (default 3 — number of labeling-loop
  runs, seeds 0..N-1), `--good-to-start`/`--bad-to-start` (3/4), `--retrain-cadence` (1 =
  per-vote, faithful), `--stop-at-done` (default off — the curve runs to `--max-labels`,
  marking the recommended stop with `stop_recommended`), `--select-strategy` (only `autopilot`
  wired today).
- Prevalence is set by `--neg-multiple` (default 100 → negative pool = 100 × positives, so
  prevalence ≈ 1/(1+m), constant across classes but positive-rich vs a real deployment);
  a true-prevalence knob is future work.

## Run (pilot: stop sign on COCO, single GPU)
One command — add `--viz` and the sweep also emits all the figures at the end:
```bash
srun --partition=gpu --gres=gpu:l40s:1 --cpus-per-task=8 --mem=46G --time=2:00:00 \
  .venv/bin/python scripts/sod/sweep.py \
    --datasets coco --classes "stop sign" --embedders dinov2,dinov3 \
    --proposals hac --region-voting --iterations 3 --max-labels 60 \
    --out-dir docs/experiments/sod-sweep --viz
```
Without `--viz` you get just the data; re-render anytime with
`python scripts/sod/plots.py --results docs/experiments/sod-sweep/results.jsonl`
(and `viz.py` for galleries/overlays). `--array-index/--array-total` filter cells
for a later SLURM array; per-image embeddings cache to `--cache-dir` (default
`<out-dir>/cache`) so re-runs reuse forward passes — point it at an existing cache
to avoid re-embedding.

Outputs (under `--out-dir`, gitignored):
- `results.jsonl` / `results.csv` — one row per config × iteration × `t` (`cost`, `fpr`, `fnr`,
  `f1`, `mean_iou`, `corloc`, `oracle_cost`, `calib_mode`, `phase`, `select_mode`,
  `n_good`, `n_bad`, `negatives_exhaustive`, …)
- `<ds>_<class>_split.json` — the train/test split image ids (always written)
- with `--viz`: `plots/` (cost/fpr/fnr/f1/mean_iou curves + variance bands, and an
  `time.png` total-time stacked bar), `time.json`, `splits_gallery/` (bucket montages
  with GT boxes), `predictions/` (per-config MLP TP/FP/FN/TN overlays)
- with `--labeling-trace` in **realistic** mode (off by default; independent of `--viz`),
  `labeling_trace/<config>/seed{N}/` (one dir **per iteration**): the images the loop labeled **in order**
  (`{t:03d}_{iid}_{good|bad}.png`, GT green + the detector's surfacing box/score red,
  captioned with step/label/select-mode→phase/head/calib_mode/threshold), plus
  `trace.csv` + `trace.json` with the full per-step record (order, id, gt label, select
  mode, phase, head, `calib_mode` [cosine_coldstart vs gmm/blend/xcal], threshold,
  surface score/margin, pred box, n_good/n_bad/n_votes, smart/stable/span indicators,
  cost/fpr/fnr/f1)

## Metric / threshold
`cost = w_fpr·FPR + w_fnr·FNR` (rates), weights from `--inclusion` (0 → FPR+FNR).
Primary is the cross-calibrated threshold (production path); `oracle_cost` is the
best-case min-over-threshold reference. At low label counts the threshold falls back to a GMM
blend (`--safe-thresholds`, on by default) — the `calib_mode` column
(`cosine_coldstart|fallback|gmm|blend|xcal`) records which regime each point used.

## Plots
`plots.py` writes one figure per metric per (dataset, class): `--metrics` defaults to
`cost,fpr,fnr,f1,mean_iou` (also available: `corloc`) → `<ds>_<class>_<metric>_vs_t.png`.
`f1` is at the cross-calibrated threshold; `mean_iou` is the top-scoring region's box vs the
best GT box, averaged over test positives (`whole` → ~flat/low by construction); `corloc` =
fraction with IoU≥0.5. So you see misses (FNR) vs false alarms (FPR), detection quality (F1),
and localization (IoU) — not just the combined cost. **Total time** is a separate per-config
stacked bar (`inference_time.png`, seconds): embed+propose (a cache-miss cost, 0 s on a
fully-cached run) plus the MLP calibrate+fit+score summed over all iterations × `t`. Because the
MLP work always runs, this chart has data even on a fully-cached run. Each mean curve
gets a **seed-variance band** (`--band-kind minmax|std|none|all`, default minmax) showing the
spread across the `--iterations` runs; `all` instead plots every iteration's own curve as its
own distinctly-colored dotted line, one legend entry per `config × seed`
(`embedder/proposal α — seed N`), so each individual iteration is identifiable — at small `t`
this spread is large (calibration noise), so treat close curves as not significant.
`--show-oracle` (oracle companion lines, off by default) is render-only; the data is always in
`results.jsonl` (one row per iteration × `t`).

## Interpreting beyond the curves
- **Built-in (opt-in):** pass `--viz` to `sweep.py` and it emits everything below after the
  run — `plots/`, `splits_gallery/`, and `predictions/` (final-detector TP/FP/FN/TN overlays
  per config at the max `t`, for `--viz-seed` [default iteration 0], `--viz-n` per montage).
  `--viz-band {minmax,std,none,all}` (default minmax) controls the plots' iteration spread
  (`all` = one line per iteration). The per-config TP/FP/FN/TN counts are also printed.
  (Skipped for array shards.)
- **Train/test split** is dumped by `sweep.py` to `<out-dir>/<ds>_<class>_split.json`
  (the exact image ids in each bucket: `train_pos`, `test_pos`, `train_neg`, `test_neg`).
- **`viz.py`** renders `--kind split` montages, offline from the cache (no GPU): galleries of
  each bucket (positives drawn with GT boxes) so you can see what training vs testing looks like.
  ```bash
  python scripts/sod/viz.py --kind split --dataset coco --cls "stop sign" \
      --out-dir docs/experiments/sod-sweep-.../splits_gallery
  ```
  (`viz.py` recomputes the split from `--split-seed/--neg-multiple/--test-fraction`, which
  must match the run being inspected.)

## Notes
- COCO negatives are exhaustive (clean FPR); LVIS/VG are non-exhaustive, so their
  FPR is an upper bound (`negatives_exhaustive=false` per row).
- Images stay zipped; pixels are streamed from the zips and cached as vectors.

## Matthew Run:

```bash
srun --partition=gpu --gres=gpu:l40s:1 --cpus-per-task=8 --mem=46G --time=8:00:00 --pty bash -l

# coco stop sign (CLIP/SigLIP/SigLIP 2, whole/sliding)
python scripts/sod/sweep.py --datasets coco --classes "stop sign" --cache-dir docs/experiments/sod-sweep/cache --embedders clip,siglip,siglip2 --proposals whole,sliding --max-labels 50 --out-dir docs/experiments/sod-sweep-realistic/coco-stopsign-clip-siglip-siglip2-whole-sliding --viz

# coco stop sign HAC sweep (dinov2/dinov3, alpha=0.3,0.5,0.7):

# no region voting
python scripts/sod/sweep.py --datasets coco --classes "stop sign" --cache-dir docs/experiments/sod-sweep/cache --embedders dinov2,dinov3 --proposals hac --hac-alpha 0.3,0.5,0.7 --max-labels 50 --out-dir docs/experiments/sod-sweep-realistic/coco-stopsign-dinov2-dinov3-hac-alphas-3-5-7-no-region-voting --viz

# region voting
python scripts/sod/sweep.py --datasets coco --classes "stop sign" --cache-dir docs/experiments/sod-sweep/cache --embedders dinov2,dinov3 --proposals hac --hac-alpha 0.3,0.5,0.7 --max-labels 50 --out-dir docs/experiments/sod-sweep-realistic/coco-stopsign-dinov2-dinov3-hac-alphas-3-5-7-region-voting --region-voting --viz

# traffic light
python scripts/sod/sweep.py --datasets lvis --classes "traffic light" --cache-dir docs/experiments/sod-sweep/cache --embedders dinov2,dinov3 --proposals hac --hac-alpha 0.1,0.3,0.5,0.7,0.9 --max-labels 50 --out-dir docs/experiments/sod-sweep-lvis-traffic-light-dinov2-dinov3-hac-alphas-1-3-5-7-9-region-voting --region-voting --viz

# 5 iterations with band=all + labeling trace
python scripts/sod/sweep.py --datasets coco --classes "stop sign" --cache-dir docs/experiments/sod-sweep/cache --embedders dinov3 --proposals hac --hac-alpha 0.5 --max-labels 50 --out-dir docs/experiments/sod-sweep-realistic/coco-stopsign-dinov3-hac-alphas-5-region-voting-viz-band-all --region-voting --viz --viz-band all --iterations 5 --labeling-trace
```