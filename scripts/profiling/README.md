# Progress-bar timing profiling

Tools for measuring how long each phase of a long-running task takes on a given
machine, and for fitting the coefficients that pace VTSearch's progress bars.

There are two consumers of these measurements:

- **The checked-in defaults.** `LOAD_COST_MODEL` and `FINALIZE_SLOT_SHARES` in
  `vtscore/datasets/stages/_load_cost_model.py` are fitted constants shipped with
  the app, covering every demo-backed media type (image / audio / video / text /
  document) × loadable registered embedder × `{cpu, cuda, cuda+cuml}`. Cells that
  cannot load in the measuring environment (`video/videomae`,
  `video/languagebind`, `audio/paraspeechclap`) and the `face` media type (no demo
  datasets) fall back to the static profile. Re-fit these when the load pipeline's
  cost shape changes or a new embedder lands.
- **A deployment's own profile.** An operator can measure their hardware with
  `tune_timing_profile.py` and override any cell — plus every non-dataset task —
  through a `VTSEARCH_TIMING_PROFILE` JSON. See `vtscore/timing/` and
  `docs/DEPLOYMENT.md` § *Progress-bar timing profile*. That path needs nothing
  from this directory beyond the recorder env vars.

## Re-fitting the checked-in defaults

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/profiling/calibrate_load_weights.py --out gpu.jsonl --embedders all
CUDA_VISIBLE_DEVICES=0 python scripts/profiling/calibrate_load_weights.py --out gpu-cumloff.jsonl --cuml off
CUDA_VISIBLE_DEVICES=  python scripts/profiling/calibrate_load_weights.py --out cpu.jsonl --sizes s,m,l
python scripts/profiling/fit_load_weights.py gpu.jsonl gpu-cumloff.jsonl cpu.jsonl   # prints the coefficient table
```

Point `VTSEARCH_MODELS_DIR` at a warm model cache so the model phase measures a
load rather than a cold HuggingFace download. Paste the fitted rows into
`vtscore/datasets/stages/_load_cost_model.py`.

The driver covers all five demo-backed media types (`face` has no demo datasets
and is logged as skipped). `--embedders all` sweeps every registered embedder for
the media type, one subprocess per (media, embedder) cell so encoder models don't
accumulate in RAM/VRAM — pass `--data-dir` so the cells share one source-archive
cache. `--cuml off` sets `VTSEARCH_DISABLE_CUML=1` to measure the CPU-clustering
finalize variant on a GPU host; the profiler stamps every row with the live cuML
state and the fit keys cuML-on CUDA rows as device `"cuda+cuml"`. Runtime lookup
tries the variant matching the live cuML state first and falls back to the other.

The same run also records one `finalize:<slot>` row per finalize sub-stage
(`FinalizeProgress.begin` stamps them automatically while the recorder is armed),
and `fit_load_weights.py` emits a `FINALIZE_SLOT_SHARES` body from them — the
measured per-`(device, media)` finalize sub-slot shares that replace the static
`FinalizeProgress._SLOTS` ballpark. Paste that body into `_load_cost_model.py`
alongside `LOAD_COST_MODEL`; uncalibrated cells keep the static fallback. GPU
cells are the ones worth measuring first: on a non-cuML GPU host the coverage
k-means can outweigh the registry save, which the static shares don't capture.

`dedup` / `cleanup` measure ~0 of the finalize phase under the default fast-hash
dedup and are floored at `0.0001` rather than rounded to an invalid `0` weight.
The opt-in `projection` / `signpost_texts` slots never run during calibration,
emit no row, and keep their static ballpark share via the `_finalize_slots` merge.

## Method & cost model

**What the weights do.** The unified load bar paces across four phases —
**download, model load, embed, finalize** — using a weight vector
(`vtscore/datasets/stages/_common.py`, applied by
`ProgressTracker._overall_raw_fraction`). The weights only shape *pacing*; the
overall ETA self-corrects from real elapsed-vs-fraction, so bad weights make the
bar race one phase and crawl another rather than making the final ETA wrong. That
bounds the value of this work: smooth honest pacing, not a time oracle.

**Why a single vector can't be right.** The phases scale with `n` differently:

| Phase      | Scales with                        | Cost model              |
|------------|------------------------------------|-------------------------|
| download   | archive **bytes**, not `n`         | `download_size_mb / bandwidth` |
| model load | nothing (encoder loaded once)      | fixed per (embedder, device) |
| embed      | `n` (one forward per item)         | `a_embed + b_embed · n` |
| finalize   | `n` (dedup + coverage + registry)  | `a_fin + b_fin · n` |

Model load is a **constant**, so its *fraction* is large for small datasets and
negligible for large ones — no single vector is right at both ends. The shipped
shape is therefore an affine per-phase cost fit per (device, media_type,
embedder), normalized to a weight vector at task creation, with a static fallback
when `n` is unknown.

**Measurement matrix.** device `{cpu, cuda(±cuML)}` × media `{image, audio, video,
text, document}` × each registered embedder × ≥2 size points per media (vary the
slice window / `items_per_category` for ≥3 distinct `n`). Run each cell ≥3×
(median); record cold vs warm model-load and download separately; log skipped
cells so coverage gaps are explicit.

**Fitting.** Per (device, media_type, embedder): `a_model` = median warm
model-load; `(a_embed, b_embed)` and `(a_fin, b_fin)` = least-squares vs `n`;
`bandwidth(device)` = fit of cold download vs `download_size_mb`. Sanity-check
R²/residuals; fall back to the generic profile where a cell is thin. The audio
rows' checked-in `b_load` (0.11 s/item decode) is carried from a live GTZAN
measurement rather than the fit — see the note in `_load_cost_model.py`.

This is the calibration-coverage counterpart to `AdaptiveLoadPacer`
(`vtscore/datasets/stages/_common.py`), which paces the bar at runtime from
observed rates but doesn't replace the value of a better prior. Non-goals:
predicting absolute load time (the ETA self-corrects); a persistent online model
(the coefficients are checked-in constants); calibrating non-dataset bars
(detector load, Find, sort).
