# Progress-weight calibration

**Background:** the checked-in coefficients this plan produced are now the
*shipped defaults* for `dataset_load`, not the last word. A deployment can
measure its own hardware with `scripts/profiling/tune_timing_profile.py` and
override any cell (plus every non-dataset task) through a
`VTSEARCH_TIMING_PROFILE` JSON — see `vtscore/timing/` and
`docs/DEPLOYMENT.md` § *Progress-bar timing profile*. The remaining work below
is about widening the checked-in defaults' coverage; it is not a prerequisite
for a tuned deployment.

**Status:** Load-progress weights are calibrated for every demo-backed media
type (image/audio/video/text/document) × every loadable registered embedder ×
{cpu, cuda, cuda+cuml} (issue #2623; HLTCOE Grid rack8n06 v100, 2026-07-18/19
sweep, raw JSONL under `/exp/sgreenberg/calib-2623`). The cuML-off `cuda`
variant is measured for the default embedders; other cuda cells fall back to
their `cuda+cuml` row (same device, different finalize cost — closer than the
static profile). Not calibratable because they cannot load in the measured
environment (and equally cannot in a served app on it): `video/videomae` and
`video/languagebind` (transformers version skew) and `audio/paraspeechclap`
(upstream HF weights file removed); the `face` media type has no demo
datasets. Those cells use the static fallback.

## Open follow-ups (remaining cells to calibrate)

<!-- item-sep -->

<!-- item-sep -->

- [ ] #2624 — Calibrate FinalizeProgress sub-slot shares from measured per-sub-stage durations

<!-- item-sep -->

<!-- item-sep -->

## Reproducing calibration for a cell

```
CUDA_VISIBLE_DEVICES=0 python scripts/profiling/calibrate_load_weights.py --out gpu.jsonl --embedders all
CUDA_VISIBLE_DEVICES=0 python scripts/profiling/calibrate_load_weights.py --out gpu-cumloff.jsonl --cuml off
CUDA_VISIBLE_DEVICES=  python scripts/profiling/calibrate_load_weights.py --out cpu.jsonl --sizes s,m,l
python scripts/profiling/fit_load_weights.py gpu.jsonl gpu-cumloff.jsonl cpu.jsonl   # prints the coefficient table
```
(Needs `VTSEARCH_MODELS_DIR` pointed at a warm model cache so the model phase is
a load, not a cold HuggingFace download.) Add the fitted rows to
`vtscore/datasets/stages/_load_cost_model.py`.

The driver covers all five demo-backed media types (image/audio/video/text/
document; face has no demo datasets and is logged as skipped). `--embedders
all` sweeps every registered embedder for the media type, one subprocess per
(media, embedder) cell so encoder models don't accumulate in RAM/VRAM — pass
`--data-dir` so the cells share one source-archive cache. `--cuml off` sets
`VTSEARCH_DISABLE_CUML=1` to measure the CPU-clustering finalize variant on a
GPU host; the profiler stamps every row with the live cuML state and the fit
keys cuML-on CUDA rows as device `"cuda+cuml"`. Runtime lookup tries the
variant matching the live cuML state first and falls back to the other.

The same run also records one `finalize:<slot>` row per finalize sub-stage
(`FinalizeProgress.begin` stamps them automatically while the recorder is
armed), and `fit_load_weights.py` emits a `FINALIZE_SLOT_SHARES` body from them —
the measured per-`(device, media)` finalize sub-slot shares that replace the
static `FinalizeProgress._SLOTS` ballpark (issue #2624). Paste that body into
`_load_cost_model.py` alongside `LOAD_COST_MODEL`; uncalibrated cells keep the
static fallback. The GPU cells are the ones worth measuring first: on a non-cuML
GPU host the coverage k-means can outweigh the registry save, which the static
shares don't capture.

---

## Reference: method & cost model

**What the weights do.** The unified load bar paces across four phases —
**download, model load, embed, finalize** — using a weight vector
(`vtscore/datasets/stages/_common.py`, applied by
`ProgressTracker._overall_raw_fraction`). The weights only shape *pacing*; the
overall ETA self-corrects from real elapsed-vs-fraction, so bad weights make the
bar race one phase and crawl another rather than making the final ETA wrong.
That bounds the value of this work: smooth honest pacing, not a time oracle.

**Why a single vector can't be right.** The phases scale with `n` differently:

| Phase      | Scales with                        | Cost model              |
|------------|------------------------------------|-------------------------|
| download   | archive **bytes**, not `n`         | `download_size_mb / bandwidth` |
| model load | nothing (encoder loaded once)      | fixed per (embedder, device) |
| embed      | `n` (one forward per item)         | `a_embed + b_embed · n` |
| finalize   | `n` (dedup + coverage + registry) | `a_fin + b_fin · n` |

Model load is a **constant**, so its *fraction* is large for small datasets and
negligible for large ones — no single vector is right at both ends. The target
is an affine per-phase cost fit per (device, media_type, embedder), normalized
to a weight vector at task creation, with a static fallback when `n` is unknown.

**Measurement matrix** (for the remaining cells): device `{cpu, cuda(±cuML)}` ×
media `{image, audio, video, text, document}` × each registered embedder × ≥2
size points per media (vary slice window / `items_per_category` for ≥3 distinct
`n`). Run each cell ≥3× (median); record cold vs warm model-load and download
separately; log skipped cells so coverage gaps are explicit.

**Fitting.** Per (device, media_type, embedder): `a_model` = median warm
model-load; `(a_embed, b_embed)` and `(a_fin, b_fin)` = least-squares vs `n`;
`bandwidth(device)` = fit of cold download vs `download_size_mb`. Sanity-check
R²/residuals; fall back to the generic profile where a cell is thin.

This is the calibration-coverage counterpart to `AdaptiveLoadPacer`
(`vtscore/datasets/stages/_common.py`), which paces the bar at runtime from
observed rates but doesn't replace the value of a better prior. Non-goals:
predicting absolute load time (ETA self-corrects); a persistent online model
(coefficients are checked-in constants); calibrating non-dataset bars
(detector load, Find, sort).

---

## Results (2026-07-18/19 sweep, rack8n06 v100 — issue #2623)

Fit diagnostics from `fit_load_weights.py` over the calib-2623 JSONL (embed
and finalize are least-squares vs `n`; slopes shown in ms/item). The audio
rows' checked-in `b_load` (0.11 s/item decode) is carried from the live
GTZAN measurement — see the note in `_load_cost_model.py`.

| device | media | embedder | load a+b·n s (cold) | R² | embed a+b·n | R² | finalize a+b·n | R² | pts |
|--------|-------|----------|---------------------|----|-------------|----|----------------|----|----|
| cpu | audio | ast | 0.50+0.00m·n (cold 25.6) | 0.00 | 5.67+1213.09m·n | 1.00 | 0.17+3.87m·n | 0.98 | 3 |
| cpu | audio | clap | 0.50+0.00m·n (cold 28.2) | 0.00 | 3.54+288.77m·n | 1.00 | -0.15+4.19m·n | 0.98 | 5 |
| cpu | audio | clap_general | 0.50+0.00m·n (cold 25.4) | 0.00 | 16.10+402.23m·n | 1.00 | 0.06+3.87m·n | 0.95 | 3 |
| cpu | audio | clap_music | 0.50+0.00m·n (cold 13.6) | 0.00 | 16.92+402.43m·n | 0.98 | 0.19+3.36m·n | 0.95 | 3 |
| cpu | audio | whisper_encoder | 0.50+0.00m·n (cold 10.7) | 0.00 | 22.12+409.27m·n | 1.00 | 0.03+3.76m·n | 0.98 | 3 |
| cpu | document |  | 0.50+0.00m·n (cold 70.1) | 0.00 | 44.73+0.00m·n | 0.00 | 15.28+0.00m·n | 0.00 | 1 |
| cpu | image | clip | 0.50+0.00m·n (cold 21.9) | 0.00 | 16.72+51.00m·n | 0.64 | 0.13+5.72m·n | 1.00 | 3 |
| cpu | image | dinov2_patch | 0.50+0.00m·n (cold 22.8) | 0.00 | 18.88+387.00m·n | 0.99 | -0.69+8.27m·n | 0.99 | 3 |
| cpu | image | dinov2_single | 0.50+0.00m·n (cold 21.7) | 0.00 | 14.25+200.84m·n | 0.96 | -0.88+8.61m·n | 0.99 | 3 |
| cpu | image | dinov3_patch | 0.50+0.00m·n (cold 22.5) | 0.00 | 22.09+312.76m·n | 0.98 | -1.05+8.91m·n | 1.00 | 3 |
| cpu | image | dinov3_single | 0.50+0.00m·n (cold 23.9) | 0.00 | 23.41+153.09m·n | 0.94 | -0.59+7.97m·n | 0.99 | 3 |
| cpu | image | eupe_patch | 0.50+0.00m·n (cold 3.6) | 0.00 | 32.41+284.59m·n | 0.96 | 1.40+5.83m·n | 0.54 | 3 |
| cpu | image | eupe_single | 0.50+0.00m·n (cold 4.7) | 0.00 | 25.20+150.85m·n | 0.89 | 1.01+6.43m·n | 0.61 | 3 |
| cpu | image | sift_vlad | 0.50+0.00m·n (cold 0.0) | 0.00 | -9.37+108.49m·n | 0.99 | 0.31+22.27m·n | 0.98 | 3 |
| cpu | image | siglip | 0.50+0.00m·n (cold 22.9) | 0.00 | 7.50+179.47m·n | 0.99 | -0.33+7.64m·n | 0.99 | 5 |
| cpu | image | siglip2 | 0.50+0.00m·n (cold 25.5) | 0.00 | 20.54+160.31m·n | 0.95 | -0.90+8.36m·n | 1.00 | 3 |
| cpu | image | siglip_l | 0.50+0.00m·n (cold 37.6) | 0.00 | -50.13+2620.10m·n | 1.00 | 0.28+7.37m·n | 0.99 | 3 |
| cpu | text | bge | 0.50+0.00m·n (cold 8.1) | 0.00 | 17.24+119.65m·n | 1.00 | -0.66+3.47m·n | 1.00 | 3 |
| cpu | text | e5 | 0.50+0.00m·n (cold 24.3) | 0.00 | 6.81+137.53m·n | 1.00 | -2.15+4.27m·n | 1.00 | 5 |
| cpu | video | xclip | 0.50+0.00m·n (cold 29.5) | 0.00 | 0.49+444.65m·n | 0.99 | 0.04+3.19m·n | 0.84 | 5 |
| cuda | audio | clap | 0.50+0.00m·n (cold 25.4) | 0.00 | 9.85+95.12m·n | 0.98 | -0.19+4.23m·n | 1.00 | 9 |
| cuda | document |  | 53.63+0.00m·n (cold 75.5) | 0.00 | 17.82+0.00m·n | 0.00 | 18.41+0.00m·n | 0.00 | 2 |
| cuda | image | siglip | 0.50+0.00m·n (cold 21.1) | 0.00 | 4.53+14.69m·n | 0.94 | -0.33+8.61m·n | 1.00 | 8 |
| cuda | text | e5 | 0.50+0.00m·n (cold 21.6) | 0.00 | 2.40+15.76m·n | 0.99 | -1.11+3.94m·n | 0.99 | 8 |
| cuda | video | xclip | 0.50+0.00m·n (cold 22.2) | 0.00 | 1.95+147.66m·n | 0.99 | -0.01+3.82m·n | 0.98 | 8 |
| cuda+cuml | audio | ast | 0.50+0.00m·n (cold 16.6) | 0.00 | 2.04+47.17m·n | 1.00 | 0.49+3.93m·n | 0.94 | 8 |
| cuda+cuml | audio | clap | 0.50+0.00m·n (cold 28.0) | 0.00 | 2.67+109.96m·n | 1.00 | 0.59+3.84m·n | 0.96 | 8 |
| cuda+cuml | audio | clap_general | 0.50+0.00m·n (cold 42.3) | 0.00 | 0.66+113.99m·n | 1.00 | 0.77+3.84m·n | 0.93 | 8 |
| cuda+cuml | audio | clap_music | 0.50+0.00m·n (cold 43.9) | 0.00 | 2.55+110.02m·n | 1.00 | 0.69+3.78m·n | 0.95 | 8 |
| cuda+cuml | audio | whisper_encoder | 0.50+0.00m·n (cold 27.5) | 0.00 | 2.45+33.55m·n | 0.99 | 0.39+3.79m·n | 0.97 | 8 |
| cuda+cuml | document |  | 45.47+0.00m·n (cold 72.4) | 0.00 | 13.67+0.00m·n | 0.00 | 15.90+0.00m·n | 0.00 | 4 |
| cuda+cuml | image | clip | 0.50+0.00m·n (cold 20.2) | 0.00 | 1.92+14.38m·n | 0.99 | 0.01+7.30m·n | 1.00 | 8 |
| cuda+cuml | image | dinov2_patch | 0.50+0.00m·n (cold 26.6) | 0.00 | 0.93+33.09m·n | 1.00 | -0.05+7.59m·n | 1.00 | 8 |
| cuda+cuml | image | dinov2_single | 0.50+0.00m·n (cold 20.6) | 0.00 | 1.85+15.57m·n | 0.98 | -0.13+7.17m·n | 1.00 | 8 |
| cuda+cuml | image | dinov3_patch | 0.50+0.00m·n (cold 26.7) | 0.00 | 2.63+34.17m·n | 1.00 | -0.17+7.62m·n | 1.00 | 8 |
| cuda+cuml | image | dinov3_single | 0.50+0.00m·n (cold 21.8) | 0.00 | 2.56+19.12m·n | 0.98 | -0.17+7.35m·n | 1.00 | 8 |
| cuda+cuml | image | eupe_patch | 0.50+0.00m·n (cold 20.4) | 0.00 | 2.64+40.34m·n | 1.00 | 0.63+7.43m·n | 0.98 | 8 |
| cuda+cuml | image | eupe_single | 0.50+0.00m·n (cold 4.5) | 0.00 | 3.02+26.01m·n | 0.98 | 0.33+7.19m·n | 1.00 | 8 |
| cuda+cuml | image | sift_vlad | 0.50+0.00m·n (cold 0.0) | 0.00 | 4.30+81.43m·n | 1.00 | -0.20+14.84m·n | 1.00 | 8 |
| cuda+cuml | image | siglip | 0.50+0.00m·n (cold 9.1) | 0.00 | 1.41+16.01m·n | 1.00 | -0.26+7.32m·n | 1.00 | 8 |
| cuda+cuml | image | siglip2 | 0.50+0.00m·n (cold 32.8) | 0.00 | 3.60+14.94m·n | 0.96 | 0.44+7.31m·n | 0.99 | 8 |
| cuda+cuml | image | siglip_l | 0.50+0.00m·n (cold 76.5) | 0.00 | 2.88+88.09m·n | 1.00 | 0.10+7.67m·n | 1.00 | 8 |
| cuda+cuml | text | bge | 0.50+0.00m·n (cold 27.0) | 0.00 | 6.09+12.07m·n | 1.00 | 0.41+2.22m·n | 1.00 | 8 |
| cuda+cuml | text | e5 | 0.50+0.00m·n (cold 30.5) | 0.00 | 3.84+12.50m·n | 1.00 | 0.15+2.22m·n | 0.99 | 9 |
| cuda+cuml | video | xclip | 0.50+0.00m·n (cold 28.3) | 0.00 | -0.34+146.07m·n | 0.99 | 0.25+3.05m·n | 0.83 | 9 |

### Finalize sub-slot shares (issue #2624)

Fitted from the same sweep's `finalize:<slot>` rows (the profiler was already
stamping them during the #2623 run, so no separate sweep was needed) and pasted
into `FINALIZE_SLOT_SHARES` in `_load_cost_model.py`: median seconds per slot
across loads, normalized per `(device, media)` cell.

- The static ballpark (`registry 0.45 > coverage 0.30`) has the ratio backwards
  for most cells: measured coverage outweighs registry for audio, text, and
  cuda video — for text it is ~99/1 (the coverage k-means over the text
  embeddings is essentially the whole phase). Document is the opposite extreme
  (registry ~0.95: few items, big page-image payload to serialize).
- Each `cuda` row pools cuML-on and cuML-off loads: splitting them moved no
  slot's share by more than ~0.11 (image coverage 0.41 with cuML vs 0.44
  without; video 0.65 vs 0.54), so the table keeps two device keys rather than
  growing a `cuda+cuml` variant like `LOAD_COST_MODEL`.
- `dedup` / `cleanup` did run and measure ~0 of the phase (default fast-hash
  dedup; floored at 0.0001 rather than rounded to an invalid 0 weight). The
  opt-in `projection` / `signpost_texts` never ran during calibration, emit no
  row, and keep their static ballpark share via the `_finalize_slots` merge.

| device | media | slot shares (fraction of finalize) | loads |
|--------|-------|-------------------------------------|-------|
| cpu | audio | cleanup 0.00, dedup 0.00, coverage 0.65, registry 0.35 | 17 |
| cpu | document | cleanup 0.00, dedup 0.00, coverage 0.05, registry 0.95 | 1 |
| cpu | image | cleanup 0.00, dedup 0.00, coverage 0.50, registry 0.50 | 35 |
| cpu | text | cleanup 0.00, dedup 0.00, coverage 0.99, registry 0.01 | 8 |
| cpu | video | cleanup 0.00, dedup 0.00, coverage 0.47, registry 0.53 | 5 |
| cuda | audio | cleanup 0.00, dedup 0.00, coverage 0.66, registry 0.34 | 48 |
| cuda | document | cleanup 0.00, dedup 0.00, coverage 0.04, registry 0.96 | 5 |
| cuda | image | cleanup 0.00, dedup 0.00, coverage 0.41, registry 0.59 | 96 |
| cuda | text | cleanup 0.00, dedup 0.00, coverage 0.99, registry 0.01 | 24 |
| cuda | video | cleanup 0.00, dedup 0.00, coverage 0.61, registry 0.39 | 16 |
