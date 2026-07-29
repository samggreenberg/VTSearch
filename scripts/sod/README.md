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
- **`--region-voting` / `--no-region-voting`** (default **ON**; **hac + dinov2/dinov3 only**): the
  faithful app-detector label construction, so a sweep cell reproduces what a user swiping in VTSearch
  produces (`python -m vtscore.eval --region-voting`). On by default; pass `--no-region-voting` to
  disable. It changes the whole `hac` train path:
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
- **`--leaf-seeding {topk,spread}…`** / **`--leaf-assign {spatial,feature}…`** (optional, **sweep axes** —
  pass multiple values and every combination becomes its own cell/result row in a single run; e.g.
  `--leaf-seeding topk spread --leaf-assign spatial feature` runs all four in one command,
  default `topk`/`spatial` = production baseline; **hac only**): how the HAC *leaves* are
  proposed. `topk` seeds = the K highest-saliency patches (which pile onto the brightest
  object); `spread` = greedy peaks with spatial non-max suppression, so seeds spread across
  objects and a small object can win its own seed. `spatial` assignment = nearest seed by
  grid distance (Voronoi); `feature` = `β·cosine + (1-β)·spatial`, so leaf boundaries follow
  content. Non-default values get their own cache slug (`…_seed-spread`, `…_asg-feature`), so
  they never reuse baseline caches.
- **`--leaf-beta none 0 0.5 0.9 …`** (optional, default `none`; **sweep axis, hac + `feature` only**):
  the **assignment** blend `β` in `β·cosine + (1-β)·spatial`, *independent of the HAC-merge `α`* (they
  control different stages — patch→seed binding vs node clustering). `none` (default) reuses `α`
  (backward-compatible); **`β=0` is exactly `spatial`**; `β=1` is pure-cosine. Inert for `spatial`
  assignment. An explicit value gets its own slug tag (`…_b0.9`).
- **`--resolution none 448 …`** (optional, default `none` = checkpoint's 224; **sweep axis, DINO
  embedders only**): square input edge fed to the embedder. The patch grid is
  `resolution // patch_size` per side (16 for dinov3, 14 for dinov2), so higher resolution = finer
  patches and a small object spans more of them — the cleanest lever against the small-object floor.
  Pass multiple values (`--resolution none 448`) and each becomes its own cell/result row. Use a
  multiple of the patch size. A no-op for text embedders (collapses to the default there). Folds into
  the cache slug (`…_r448`), so a resolution run never reuses default-res cached vectors; cost scales
  ~resolution² (the HAC step is unaffected — K leaves is fixed).
- **`--pca-dims none 10 32 …`** (optional, default `none`; **sweep axis, hac only**): fit a per-image
  PCA of this many dims on the patch grid and decide the HAC **merge order** in that reduced space
  (tree topology only — stored region vectors stay full-dim). `none`/`0` = full-dim baseline. Pass
  multiple values (`--pca-dims none 10 32`, stray commas tolerated) and each becomes its own
  cell/result row; each value gets its own cache slug (`…_pca10`). Note this reshapes *internal*
  merges, not the leaves, so it's largely orthogonal to the small-object leaf question.
- **`--dinov3-model vitb16 vitl16 …`** (optional, default `vitb16`; **sweep axis, `dinov3` only**):
  which DINOv3 checkpoint the `dinov3` embedder loads. Short aliases `vits16`/`vits16plus`/`vitb16`/
  `vitl16`/`vith16plus`/`vit7b16` (or a full HF repo id). All are **patch-size 16**, so a larger model
  gives richer per-patch features but the **same grid density** — use `--resolution` for finer
  localization, not a bigger model. Pass multiple (`--dinov3-model vitb16 vitl16`) and each becomes its
  own curve. `vitb16` is the app default (no cache tag, reuses existing caches); other sizes get their
  own cache slug (`…_mvitl16`) and are downloaded on first use (gated — needs `HF_TOKEN`). A no-op for
  non-`dinov3` embedders. Bigger models are markedly slower/heavier (7B ≈ 27 GB weights), so validate a
  single cell before launching a full cartesian sweep.

## Sweep axes (one command, many rows)
`--hac-alpha`, `--hac-k`, `--leaf-seeding`, `--leaf-assign`, `--leaf-beta`, `--pca-dims`,
`--resolution`, and `--dinov3-model` are all **multi-value sweep axes** (all space-separated `nargs`,
e.g. `--hac-k 8 12 16`): pass several values to any of them and the run takes the cartesian product,
one result row (and cache slug) per combination — all in a single `results.jsonl`/`results.csv`,
distinguished by the
`alpha`/`hac_k`/`leaf_seeding`/`leaf_assign`/`leaf_beta`/`pca_dims`/`resolution`/`dinov3_model` columns.
The leaf/pca/alpha/k/β axes only affect the `hac` tree (collapse to one value for other proposals; `β`
further collapses unless `--leaf-assign feature`); `--resolution` and `--dinov3-model` affect the
embedder forward (collapse to the default for text embedders, and `--dinov3-model` only expands for the
`dinov3` embedder). Example — the full 2×2 leaf ablation ×
two PCA settings on five classes in one command:
```bash
.venv/bin/python scripts/sod/sweep.py --datasets coco \
  --classes "traffic light,stop sign,car,person,bus" --embedders dinov3 \
  --proposals hac --region-voting --iterations 3 --max-labels 60 \
  --leaf-seeding topk spread --leaf-assign spatial feature --pca-dims none 10 \
  --out-dir docs/experiments/sod-sweep-leaf --viz
```

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
- `--min-box-frac` (default `0.01`, the GUI's drawable-box floor) drops GT boxes below that
  fraction of the image on **either** axis — the same rule the annotation GUI enforces on a
  drawn box — so the sweep never trains/evaluates on an un-drawable annotation. Filtering is
  **per-box**: an image stays a positive as long as ≥1 of its class boxes survives (only the
  survivors feed the covering box / exemplars / metrics); an image whose class boxes are *all*
  sub-floor is dropped entirely and, since it still contains the class, is also kept out of the
  negative pool. Pass `--min-box-frac 0` to keep every box. (The exemplar cache is keyed on a
  hash of each image's GT boxes, so changing this value recomputes cleanly rather than reusing a
  stale file.)
- `--neg-regions` (default off): in the realistic loop a **Bad vote contributes all of that image's
  region/window vectors** as one per-image negative **bag** (bag-balanced via `train_rv_head`),
  instead of just the single whole-image vector — "No → all windows." For `sliding`/`dino`/box-pool
  proposals; a **no-op for `whole`** (one region) and **subsumed by `--region-voting` on hac** (which
  already bags leaf nodes and takes precedence). Only changes how the in-memory MLP is trained from
  the already-cached region vecs, so it needs **no re-embed and no separate cache** — use a distinct
  `--out-dir` to A/B it (`results.jsonl` rows carry `neg_regions`).

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
- with `--viz`: `plots/` — per-class `<ds>_<class>_<metric>_vs_t.png` (config curves + seed-variance
  bands) plus, when a dataset has **≥2 classes**, cross-class `summary_<ds>_<metric>_vs_t.png`
  (each config **macro-averaged over the dataset's classes**; band = across-class spread, or one line
  per class with `--viz-band all`; toggle with `--summary/--no-summary`) — and a `time.png`
  total-time stacked bar; `time.json`, `splits_gallery/` (bucket montages with GT boxes),
  `predictions/<slug>/` (per-config MLP TP/FP/FN/TN overlays — **one subdir per sweep-axis variant**,
  keyed by the cache slug, so leaf-seeding/leaf-assign/pca/alpha combos never overwrite each other)
- with `--labeling-trace` in **realistic** mode (off by default; independent of `--viz`),
  `labeling_trace/<slug>/<config>/seed{N}/` (one dir **per iteration**): the images the loop labeled **in order**,
  two PNGs per step (both prefixed `{t:03d}_{iid}_{good|bad}` so they sort in labeling order):
  - `…_pred.png` — GT green + the detector's surfacing box/score red (captioned with
    step/label/select-mode→phase/head/calib_mode/threshold),
  - `…_hac.png` — the HAC **composite** (the `run_hac_tree_sweep.py` look): the region tree
    drawn **twice** side by side — **left** = masked patch-cell **pixel** thumbnails laid out by
    merge depth with parent→child edges (leaves→merges, the tree "being built"), **right** = the
    same tree over the **inferno attention/saliency** overlay — each node labeled with its **MLP
    detector score**, the **surfacing match** ringed red (region that made the model pick this
    image) and the good-vote **snapped match** ringed blue (best-IoU HAC node vs the GT covering box);
  plus `trace.csv` + `trace.json` with the full per-step record (order, id, gt label, select
  mode, phase, head, `calib_mode` [cosine_coldstart vs gmm/blend/xcal], threshold,
  surface score/margin, pred box, matched/snapped region index, per-region scores [json only],
  n_good/n_bad/n_votes, smart/stable/span indicators, cost/fpr/fnr/f1).
  The composite reads HAC geometry (region boxes, `children`, per-node `cell_masks`, patch
  `saliency`) from the region npz; caches written before this feature degrade gracefully
  (box-crop thumbnails when `cell_masks` absent, pixel-only when `saliency` absent) until re-run.

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

# coco scissors (CLIP/SigLIP/SigLIP 2/DINOv2/DINOv3, whole/sliding)
python scripts/sod/sweep.py --datasets coco --classes "scissors" --cache-dir docs/experiments/sod-sweep/cache --embedders clip,siglip,siglip2,dinov2,dinov3 --proposals whole,sliding --max-labels 30 --out-dir docs/experiments/sod-sweep/coco-scissors-clip-siglip-siglip2-dinov2-dinov3-whole-sliding --viz --neg-regions

# coco stop sign HAC sweep (dinov2/dinov3, alpha=0.3,0.5,0.7):

# no region voting
python scripts/sod/sweep.py --datasets coco --classes "stop sign" --cache-dir docs/experiments/sod-sweep/cache --embedders dinov2,dinov3 --proposals hac --hac-alpha 0.3,0.5,0.7 --max-labels 50 --out-dir docs/experiments/sod-sweep/coco-stopsign-dinov2-dinov3-hac-alphas-3-5-7-no-region-voting --viz --no-region-voting

# region voting
python scripts/sod/sweep.py --datasets coco --classes "stop sign" --cache-dir docs/experiments/sod-sweep/cache --embedders dinov2,dinov3 --proposals hac --hac-alpha 0.3,0.5,0.7 --max-labels 50 --out-dir docs/experiments/sod-sweep/coco-stopsign-dinov2-dinov3-hac-alphas-3-5-7-region-voting --viz

# traffic light
python scripts/sod/sweep.py --datasets lvis --classes "traffic light" --cache-dir docs/experiments/sod-sweep/cache --embedders dinov2,dinov3 --proposals hac --hac-alpha 0.1,0.3,0.5,0.7,0.9 --max-labels 50 --out-dir docs/experiments/sod-sweep-lvis-traffic-light-dinov2-dinov3-hac-alphas-1-3-5-7-9-region-voting --viz

# 5 iterations with band=all + labeling trace + min-box-frac 0.05
python scripts/sod/sweep.py --datasets coco --classes "stop sign" --cache-dir docs/experiments/sod-sweep/cache --embedders dinov3 --proposals hac --hac-alpha 0.5 --max-labels 50 --out-dir docs/experiments/sod-sweep/coco-stopsign-dinov3-hac-alphas-5-region-voting-5-viz-bands --viz --viz-band all --iterations 5 --labeling-trace --min-box-frac 0.05

# multiple classes (summary), pca-dim=10
python scripts/sod/sweep.py --datasets coco --classes "stop sign, traffic light" --cache-dir docs/experiments/sod-sweep/cache --embedders dinov3 --proposals hac --hac-alpha 0.5 --max-labels 50 --out-dir docs/experiments/sod-sweep/coco-stopsign-trafficlight-dinov3-hac-alphas-5-summary-10-pca-dims --viz --iterations 5 --labeling-trace --min-box-frac 0.05 --summary --pca-dims 10

# different leaf node approaches ablation test, pca-dim = None, 10
python scripts/sod/sweep.py --datasets coco --classes "traffic light,stop sign,car,person,bus" --cache-dir docs/experiments/sod-sweep/cache --embedders dinov3 --proposals hac --iterations 3 --max-labels 50 --leaf-seeding topk spread --leaf-assign spatial feature --out-dir docs/experiments/sod-sweep/coco-5-classes-dinov3-hac--None-10-pca-leaf-ablation --viz --summary --pca-dims none 10 # should add --min-box-frac 0.05

# seeding: spread, assign: feature, sweep k and beta, resolution
python scripts/sod/sweep.py --datasets coco --classes "traffic light,stop sign" --cache-dir docs/experiments/sod-sweep/cache --embedders dinov3 --proposals hac --iterations 3 --max-labels 50 --leaf-seeding spread --leaf-assign feature --out-dir docs/experiments/sod-sweep/coco-stopsign-trafficlight-dinov3-hac-3-5-beta-8-16-32-k-value-224-448-resolution --hac-k 8 16 32 --leaf-beta 0.3 0.5 --resolution none 448 --viz --min-box-frac 0.05 --labeling-trace --summary

# DINOv3 ViT-B vs ViT-L, seeding: spread, assign: feature, sweep k and reslution
python scripts/sod/sweep.py --datasets coco --classes "traffic light,stop sign" --cache-dir docs/experiments/sod-sweep/cache --embedders dinov3 --proposals hac --leaf-seeding spread --leaf-assign feature --hac-k 8 16 32 --iterations 3 --resolution none 448 --max-labels 50 --dinov3-model vitb16 vitl16 --out-dir docs/experiments/sod-sweep/coco-stopsign-trafficlight-dinov3-vitb16-vs-vitl16 --viz --summary --min-box-frac 0.05 --labeling-trace

# DINOv3 ViT-B vs ViT-L, seeding: spread, assign: feature, k=32, reslution=448, vitl16, 5 iterations with band=all
python scripts/sod/sweep.py --datasets coco --classes "traffic light,stop sign" --cache-dir docs/experiments/sod-sweep/cache --embedders dinov3 --proposals hac --leaf-seeding spread --leaf-assign feature --hac-k 32 --iterations 5 --resolution 448 --max-labels 50 --dinov3-model vitl16 --out-dir docs/experiments/sod-sweep/coco-stopsign-trafficlight-dinov3-vitl16-5-iterations --viz --viz-band all --summary --min-box-frac 0.05 --labeling-trace --pca-dims none 10 

# SigLIP2 whole interpretability threshold test
python scripts/sod/sweep.py --datasets coco --classes "stop sign" --cache-dir docs/experiments/sod-sweep/cache --embedders siglip2 --proposals whole --leaf-seeding spread --leaf-assign feature --hac-k 32 --iterations 3 --resolution 448 --max-labels 50 --dinov3-model vitl16 --out-dir docs/experiments/sod-sweep/coco-stopsign-siglip2-whole-3-iterations-oracle --viz --viz-band all --min-box-frac 0.05 --labeling-trace --show-oracle

# DINOv3 HAC interpretability threshold test
python scripts/sod/sweep.py --datasets coco --classes "stop sign" --cache-dir docs/experiments/sod-sweep/cache --embedders dinov3 --proposals hac --leaf-seeding spread --leaf-assign feature --hac-k 32 --iterations 3 --resolution 448 --max-labels 50 --dinov3-model vitl16 --out-dir docs/experiments/sod-sweep/coco-stopsign-dinov3-hac-3-iterations-oracle --viz --viz-band all --min-box-frac 0.05 --labeling-trace --show-oracle
```