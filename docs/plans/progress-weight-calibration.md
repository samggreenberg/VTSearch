# Progress-weight calibration

**Status: executed for image + audio × CPU + GPU (2026-07).** This doc specifies
how to replace the hand-guessed dataset-load progress weights with *measured*
ones, and how to make the weights depend on dataset size `n` rather than being a
single fixed vector per device. It is the concrete follow-up to the "Weights are
static guesses" item in `docs/plans/progress-bar-consolidation.md`. The
instrumentation, driver, affine fit, and `n`-aware wiring shipped; fitted
coefficients for the image/audio × CPU/GPU cells are in `_load_cost_model.py`.
See **Results** for the runs; the sections above are the design/method.

It is written to be run **on a CUDA box** — the Claude Cloud container has no GPU.
The runs below were done on the HLTCOE Grid (a100).

## Background: what the weights do (and don't do)

The unified load bar paces itself across four phases — **download, model load,
embed, finalize** — using a weight vector that reflects each phase's share of
wall-clock (`vtscore/datasets/stages/_common.py`, consumed at task creation in
`load_pipeline.py` and applied by `ProgressTracker._overall_raw_fraction` in
`vtscore/concurrency/progress.py`). Today there are three hand-tuned vectors:

```
                  download  model  embed  finalize
CPU (generic)   [ 0.25,    0.10,  0.55,  0.10 ]
CPU (image)     [ 0.35,    0.10,  0.35,  0.20 ]   # added because images embed cheaply
GPU             [ 0.20,    0.10,  0.30,  0.40 ]   # embed fast, finalize not GPU-accelerated
```

Crucially, **the weights only shape pacing**. The overall ETA self-corrects from
the real elapsed-vs-fraction rate (`progress.py:_compute_overall`), so bad
weights don't make the *final* ETA wrong — they make the bar **race through one
phase and crawl another**, and they misplace the model-load floor. That's the
symptom we're fixing, and it bounds the value of this work: we want smooth,
honest pacing, not a perfect time oracle. Don't over-engineer past that.

## Why a single vector can't be right: `n` belongs in the model

The four phases scale with the item count `n` *differently*:

| Phase      | Scales with                              | Cost model              |
|------------|------------------------------------------|-------------------------|
| download   | archive **bytes**, not selected `n` (demo archives are a fixed size, then subset) | `download_size_mb / bandwidth` |
| model load | nothing — the encoder is loaded once     | fixed per (embedder, device) |
| embed      | `n` (one encoder forward per item)       | `a_embed + b_embed · n` |
| finalize   | `n` (dedup + diversity k-means + registry serialize) | `a_fin + b_fin · n` |

The model-load phase is what breaks a fixed vector: it is a **constant**, so its
*fraction* of wall-clock is large for a small dataset and negligible for a large
one. No single weight vector can be right at both ends — which is exactly the
"big vs small" axis in this plan.

So the target model is an **affine per-phase cost**, fit per
(device, media_type, embedder):

```
T_download ≈ download_size_mb / bandwidth(device)     # or a measured per-dataset constant
T_model    ≈ a_model                                   # fixed; b ≈ 0
T_embed    ≈ a_embed + b_embed · n
T_finalize ≈ a_fin   + b_fin   · n
weight_phase = T_phase / Σ T_phase                     # normalized at task creation
```

We usually **do** know `n` and `download_size_mb` up front for demo datasets:
`vtsearch/routes/datasets/ui.py:104-121` already computes expected item counts
from `items_per_category` (and the slice window), and `download_size_mb` lives in
the demo registry. When `n` is genuinely unknown (e.g. a folder importer that
streams), fall back to today's static vector (the large-`n` asymptote).

This makes calibration (#1) and the `n`-aware formula (#2) the same effort: the
run below measures `(a, b)` per phase; the formula consumes them.

## Goal & non-goals

**Goal:** produce measured affine coefficients `(a_phase, b_phase)` per
(device, media_type, embedder), and wire an `n`-aware `load_step_weights()` that
computes the weight vector from those coefficients + the known `n` /
`download_size_mb`, with a static fallback.

**Non-goals:**
- Not trying to predict absolute load time (the ETA already self-corrects).
- Not building a persistent learned/online model — coefficients are checked-in
  constants, refreshed by re-running this harness. (No vectors/weights persisted
  at runtime; this is consistent with the "No Persisted Vectors or MLPs" rule —
  the coefficients are source constants, not derived artifacts.)
- Not calibrating non-dataset bars (detector load, Find, sort) — same method
  would apply, but out of scope here.

## The measurement matrix

Run every reachable cell of:

- **Device:** `cpu`, `cuda`. (`mps` optional; skip if no Apple box. For `cuda`,
  run both **with and without cuML** installed if feasible — the diversity-tree
  k-means only moves to GPU under cuML, which materially changes finalize.)
- **Media type:** `image`, `audio`, `video`, `text`, `document`.
- **Embedder:** each registered embedder for that media type (not just the
  default), since embed cost is per-encoder. Enumerate via the embedder registry.
- **Size:** at least two points per (media_type) so the affine fit has a slope —
  use the existing small demo variants (e.g. `caltech101_s`) **and** a larger
  one, and/or vary the slice window / `items_per_category` to get ≥3 distinct
  `n` values. More `n` points = better `b` estimate.

Skip cells that don't exist (no such embedder for that media type). `log()` /
record every skipped cell so the coverage gap is explicit, not silent.

**Repetitions & caching:** run each cell ≥3× and take the median. Distinguish:
- **Cold model load** (first use downloads the encoder) vs **warm** — record
  separately; the bar should pace against the *warm* steady state, but note the
  cold delta.
- **Cold vs warm download** — the demo archive is cached on disk after the first
  fetch, so a second run has `T_download ≈ 0`. Measure the **cold** download (to
  fit `bandwidth`) and also record the warm case (so the formula can zero the
  download slice when the pickle/archive is already present, which the loader
  already short-circuits — see `loader_demo.py` cached-pickle path).

## Instrumentation

Add an **env-gated per-phase timing recorder** — no behavior change when off:

- Gate on `VTSEARCH_PROFILE_LOAD=<path>`. When set, the load pipeline records, at
  each phase boundary, a JSONL row:
  ```json
  {"device": "cuda", "media_type": "image", "embedder": "siglip",
   "dataset_id": "caltech101_s", "n": 860, "download_size_mb": 40.0,
   "phase": "embed", "seconds": 3.21, "cold_model": false, "cold_download": true}
  ```
- Capture boundaries where the status transitions are already emitted: the
  `_STATUS_TO_STEP` transitions (`downloading`→`loading`→`embedding`) plus the
  `FinalizeProgress` sub-slot `begin()` calls (cleanup/dedup/diversity/registry/
  projection) so finalize sub-shares get measured too (addresses the second
  open follow-up in the consolidation doc). A `ProgressTracker` subscriber that
  timestamps status changes is the least invasive hook; if sub-slot granularity
  needs more than status strings, add explicit `time.monotonic()` stamps in the
  finalize block keyed by slot.
- Keep it in `vtscore` (library tier) so it works from a plain CLI autodetect
  run, not just the app.

A small **driver script** (`scripts/profiling/calibrate_load_weights.py`,
new) iterates the matrix by invoking the existing demo-load path per cell
(reusing the CLI/`load_demo_dataset` entry point), sets the env var, clears the
relevant cache between cold runs, and concatenates the JSONL.

## Fitting

A second script (or a notebook) reads the JSONL and, per
(device, media_type, embedder):

- `a_model`  = median warm model-load seconds.
- `(a_embed, b_embed)` = least-squares fit of embed seconds vs `n`.
- `(a_fin, b_fin)`     = least-squares fit of finalize seconds vs `n`
  (optionally per sub-slot for the finalize sub-shares).
- `bandwidth(device)` = fit of cold download seconds vs `download_size_mb`
  (likely device-independent; collapse if so).

Emit the coefficients as a checked-in Python table (e.g.
`vtscore/datasets/stages/_load_cost_model.py`) plus a human-readable summary
appended below the line in this doc. Sanity-check fits (R², residuals); where a
cell has too few points or a poor fit, fall back to the generic profile and note
it.

## Wiring it back in

1. Extend `load_step_weights(media_type)` →
   `load_step_weights(media_type, *, n=None, download_size_mb=None, embedder=None)`.
2. When `n` is known and a coefficient row exists for
   (resolved device, media_type, embedder): compute `T_phase` from the affine
   model, normalize to a weight vector, return it.
3. When `n` is unknown or no row matches: return today's static vector for that
   (device, media_type) — unchanged behavior, so this is a strict superset.
4. Pass `n` / `download_size_mb` / `embedder` at the call site
   (`load_pipeline.py:419`); the demo importer already knows the expected count
   (`ui.py` computes it) — thread it through, don't recompute.

## Acceptance / tests

- `tests_lib/datasets/test_load_step_weights.py`: extend with cases proving the
  `n`-aware path (small `n` weights model-load heavier than large `n`; download
  slice collapses when `download_size_mb≈0` / cached; unknown `n` returns the
  static vector unchanged; GPU stays media-agnostic unless a coefficient row
  says otherwise).
- The coefficient table is validated for shape (all phases present, weights
  normalize to ~1.0) in a unit test, mirroring `test_profiles_are_well_formed`.
- No new runtime persistence; coefficients are source constants.

## Open follow-ups

- **Executed** for `{cpu, cuda} × {image, audio}`, default embedder — see
  Results below. Remaining media types (`video / text / document`), non-default
  embedders, and the cuML-on/off split are still uncalibrated and fall back to
  the static profiles; re-run `scripts/profiling/calibrate_load_weights.py` for
  those cells when needed.
- Once coefficients exist, revisit the **finalize sub-slot shares** (the
  `FinalizeProgress._SLOTS` ballparks) with the same measured data — they have
  the same "static guess" problem one level down.
- Consider whether the **multi-embedder (v3 trio)** embed loop needs its own
  `b_embed` summed across bound embedders (see the trio follow-up in the
  consolidation doc).

---

## Results

Ran on the HLTCOE Grid (a100 for the GPU cells) via
`scripts/profiling/calibrate_load_weights.py` with the env-gated recorder, then
`scripts/profiling/fit_load_weights.py` over the JSONL. 242 phase rows across the
four cells. Coefficients are checked into `_load_cost_model.py`.

**What ran.** `{cpu, cuda} × {image (caltech101 / siglip), audio (esc50 / clap)}`,
default embedder only. Sizes `caltech101_{s,m,l,a}` (n = 412/838/1704/2954) and
`esc50_{s,m,l,a}` (n = 245/588/1127/1960); ≥2 reps on the fast cells. Each cell's
first load pays the cold model + cold archive download; the driver clears only the
*embeddings* cache between loads, so the source stays warm and every load
re-embeds. **Skipped, logged not silent:** media `video/text/document`; all
non-default embedders (image: siglip2, eupe_\*, face, sift_vlad; audio:
whisper_encoder…); cuML on/off. Uncalibrated cells fall back to the static
per-(device, media) profile — unchanged behaviour.

**Model-load is warm-≈0 by construction.** A warm load *skips the model phase
entirely* (the encoder is resident, so no `loading model` status fires), so there
is no warm row to fit — `a_model` is floored to 0.5 s and the model slice stays
deliberately tiny. The cold first-load model cost (below) is recorded as a note,
not paced against (per the plan).

### Fitted coefficients (seconds; `n` = item count)

| device | media | embedder | a_model (cold) | embed `a + b·n` | R² | finalize `a + b·n` | R² | n pts |
|--------|-------|----------|---------------|-----------------|----|--------------------|----|------|
| cpu  | image | siglip | 0.5 (39.9) | 1.97 + 0.2927·n | 1.00 | 0.47 + 0.00212·n | 0.90 | 5 |
| cpu  | audio | clap   | 0.5 (4.5)  | 0.00 + 0.1846·n | 1.00 | 0.00 + 0.00253·n | 1.00 | 5 |
| cuda | image | siglip | 0.5 (119.7)| 2.12 + 0.00678·n | 0.89 | 8.01 (fixed, b≈0) | 0.00 | 12 |
| cuda | audio | clap   | 0.5 (40.1) | 0.69 + 0.03663·n | 1.00 | 0.00 + 0.00362·n | 1.00 | 12 |

Download bandwidth ≈ **10.05 MB/s** (device-pooled; cuda 9.5 / cpu 10.6 agreed
within ~10%), so `T_download ≈ download_size_mb / 10.05`.

Physically sensible: per-item embed is **5–43× slower on CPU** than GPU (image
0.293 vs 0.0068 s/item; audio 0.185 vs 0.0366) — SigLIP's ViT benefits from the
GPU far more than CLAP. Audio finalize scales with `n` (diversity-tree k-means);
GPU-image finalize is a ~8 s fixed overhead at these sizes (b≈0).

### Sample resulting weights `[download, model, embed, finalize]`

| device | media | n | archive | weights |
|--------|-------|---|---------|---------|
| cuda | image | 400   | 131 MB (cold) | [0.49, 0.02, 0.18, 0.31] |
| cuda | image | 2000  | 0 (warm)      | [0.00, 0.02, 0.64, 0.34] |
| cuda | image | 20000 | 0 (warm)      | [0.00, 0.00, 0.92, 0.08] |
| cuda | audio | 400   | 600 MB (cold) | [0.78, 0.01, 0.20, 0.02] |
| cuda | audio | 2000  | 0 (warm)      | [0.00, 0.01, 0.91, 0.09] |
| cpu  | image | 400   | 131 MB (cold) | [0.10, 0.00, 0.89, 0.01] |
| cpu  | audio | 2000  | 0 (warm)      | [0.00, 0.00, 0.99, 0.01] |

The n-awareness a single static vector can't give: small `n` weighs
model/finalize/download heavier, large `n` lets embed dominate; the download
slice collapses when there is no archive (local imports, cached re-adds — which
also take the static path since they don't know `n`). Cross-checks an independent
scout run: warm large-`n` GPU-image `≈ [0, 0.02, 0.64, 0.34]` vs the scout's
`[0, 0.01, 0.68, 0.31]` — the model and the raw data agree.

### Reproduce

```
# per device (device is fixed at process start):
CUDA_VISIBLE_DEVICES=0 python scripts/profiling/calibrate_load_weights.py --out gpu.jsonl
CUDA_VISIBLE_DEVICES=  python scripts/profiling/calibrate_load_weights.py --out cpu.jsonl --sizes s,m,l
python scripts/profiling/fit_load_weights.py gpu.jsonl cpu.jsonl   # prints the table above
```
(Needs `VTSEARCH_MODELS_DIR` pointed at a warm model cache so the model phase is
a load, not a cold HuggingFace download.)
