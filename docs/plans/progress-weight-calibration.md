# Progress-weight calibration

**Status: executed for image + audio × CPU + GPU (2026-07).** The
instrumentation, driver, affine fit, and `n`-aware wiring shipped; fitted
coefficients for those four cells are in
`vtscore/datasets/stages/_load_cost_model.py`. Remaining media types /
embedders / the cuML split are still on the static fallback. Runs were on the
HLTCOE Grid (a100) — the Claude Cloud container has no GPU.

## Open follow-ups (remaining cells to calibrate)

- **Uncalibrated cells fall back to the static per-(device, media) profile.**
  Calibrate by re-running `scripts/profiling/calibrate_load_weights.py` for:
  the remaining media types (`video / text / document`), the non-default
  embedders (image: siglip2, eupe_*, face, sift_vlad; audio: whisper_encoder…),
  and the **cuML on/off** split (cuML moves diversity-tree k-means to GPU,
  materially changing finalize).
- **Finalize sub-slot shares** (`FinalizeProgress._SLOTS` ballparks) have the
  same "static guess" problem one level down — revisit with the measured data
  once coefficients exist.
- **Multi-embedder (v3 trio)** embed loop may need its own `b_embed` summed
  across bound embedders (see the trio follow-up in the consolidation doc).

## What shipped

- Env-gated per-phase timing recorder (`VTSEARCH_PROFILE_LOAD=<path>`, JSONL,
  library-tier), `scripts/profiling/calibrate_load_weights.py` (matrix driver),
  `scripts/profiling/fit_load_weights.py` (affine fit).
- `n`-aware `load_step_weights(media_type, *, n, download_size_mb, embedder)`:
  computes the weight vector from the affine cost model + known `n` /
  `download_size_mb` when a coefficient row exists, else returns today's static
  vector (strict superset). Threaded through the demo call site
  (`load_pipeline.py`), reusing the count `ui.py` already computes.
- Coefficients checked into `_load_cost_model.py`; shape validated by a unit
  test (all phases present, weights normalize to ~1.0). No runtime persistence —
  coefficients are source constants (consistent with "No Persisted Vectors").
- Tests in `tests_lib/datasets/test_load_step_weights.py` (small-`n` weights
  model/finalize heavier; download slice collapses when cached; unknown `n`
  returns the static vector).

## Results (the four calibrated cells)

Ran `{cpu, cuda} × {image (caltech101 / siglip), audio (esc50 / clap)}`,
default embedder only, over 242 phase rows. Sizes `caltech101_{s,m,l,a}`
(n = 412/838/1704/2954) and `esc50_{s,m,l,a}` (n = 245/588/1127/1960). The
driver clears only the *embeddings* cache between loads, so the source stays
warm and every load re-embeds. Warm loads skip the model phase entirely, so
`a_model` is floored to 0.5 s (cold first-load cost recorded as a note, not
paced against).

**Fitted coefficients** (seconds; `n` = item count):

| device | media | embedder | a_model (cold) | embed `a + b·n` | R² | finalize `a + b·n` | R² | n pts |
|--------|-------|----------|---------------|-----------------|----|--------------------|----|------|
| cpu  | image | siglip | 0.5 (39.9) | 1.97 + 0.2927·n | 1.00 | 0.47 + 0.00212·n | 0.90 | 5 |
| cpu  | audio | clap   | 0.5 (4.5)  | 0.00 + 0.1846·n | 1.00 | 0.00 + 0.00253·n | 1.00 | 5 |
| cuda | image | siglip | 0.5 (119.7)| 2.12 + 0.00678·n | 0.89 | 8.01 (fixed, b≈0) | 0.00 | 12 |
| cuda | audio | clap   | 0.5 (40.1) | 0.69 + 0.03663·n | 1.00 | 0.00 + 0.00362·n | 1.00 | 12 |

Download bandwidth ≈ **10.05 MB/s** (device-pooled), so
`T_download ≈ download_size_mb / 10.05`. Per-item embed is 5–43× slower on CPU
than GPU (SigLIP's ViT benefits far more than CLAP); GPU-image finalize is a
~8 s fixed overhead at these sizes.

**Sample resulting weights** `[download, model, embed, finalize]`:

| device | media | n | archive | weights |
|--------|-------|---|---------|---------|
| cuda | image | 400   | 131 MB (cold) | [0.49, 0.02, 0.18, 0.31] |
| cuda | image | 2000  | 0 (warm)      | [0.00, 0.02, 0.64, 0.34] |
| cuda | image | 20000 | 0 (warm)      | [0.00, 0.00, 0.92, 0.08] |
| cuda | audio | 400   | 600 MB (cold) | [0.78, 0.01, 0.20, 0.02] |
| cpu  | image | 400   | 131 MB (cold) | [0.10, 0.00, 0.89, 0.01] |
| cpu  | audio | 2000  | 0 (warm)      | [0.00, 0.00, 0.99, 0.01] |

Small `n` weighs model/finalize/download heavier; large `n` lets embed
dominate; the download slice collapses when there is no archive.

**Reproduce:**
```
CUDA_VISIBLE_DEVICES=0 python scripts/profiling/calibrate_load_weights.py --out gpu.jsonl
CUDA_VISIBLE_DEVICES=  python scripts/profiling/calibrate_load_weights.py --out cpu.jsonl --sizes s,m,l
python scripts/profiling/fit_load_weights.py gpu.jsonl cpu.jsonl   # prints the table above
```
(Needs `VTSEARCH_MODELS_DIR` pointed at a warm model cache so the model phase is
a load, not a cold HuggingFace download.)

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
| finalize   | `n` (dedup + diversity + registry) | `a_fin + b_fin · n` |

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

This is the concrete follow-up to the "Weights are static guesses" item in
`docs/plans/progress-bar-consolidation.md`. Non-goals: predicting absolute load
time (ETA self-corrects); a persistent online model (coefficients are checked-in
constants); calibrating non-dataset bars (detector load, Find, sort).
