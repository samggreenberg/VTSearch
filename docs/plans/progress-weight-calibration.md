# Progress-weight calibration

**Status:** Load-progress weights are calibrated only for image + audio × CPU +
GPU (default embedders); the remaining media types (video/text/document),
non-default embedders, and the cuML on/off split still use the static fallback
and need calibration.

## Open follow-ups (remaining cells to calibrate)

- **Uncalibrated cells fall back to the static per-(device, media) profile.**
  Calibrate by re-running `scripts/profiling/calibrate_load_weights.py` for:
  the remaining media types (`video / text / document`), the non-default
  embedders (image: siglip2, eupe_*, face, sift_vlad; audio: whisper_encoder…),
  and the **cuML on/off** split (cuML moves coverage-atlas k-means to GPU,
  materially changing finalize).
- **Finalize sub-slot shares** (`FinalizeProgress._SLOTS` ballparks) have the
  same "static guess" problem one level down — revisit with the measured data
  once coefficients exist.
- **Multi-embedder (v3 trio)** embed loop may need its own `b_embed` summed
  across bound embedders (see the trio follow-up in the consolidation doc).

## Reproducing calibration for a remaining cell

```
CUDA_VISIBLE_DEVICES=0 python scripts/profiling/calibrate_load_weights.py --out gpu.jsonl
CUDA_VISIBLE_DEVICES=  python scripts/profiling/calibrate_load_weights.py --out cpu.jsonl --sizes s,m,l
python scripts/profiling/fit_load_weights.py gpu.jsonl cpu.jsonl   # prints the coefficient table
```
(Needs `VTSEARCH_MODELS_DIR` pointed at a warm model cache so the model phase is
a load, not a cold HuggingFace download.) Add the fitted rows to
`vtscore/datasets/stages/_load_cost_model.py`.

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

This is the concrete follow-up to the "Weights are static guesses" item in
`docs/plans/progress-bar-consolidation.md`. Non-goals: predicting absolute load
time (ETA self-corrects); a persistent online model (coefficients are checked-in
constants); calibrating non-dataset bars (detector load, Find, sort).
