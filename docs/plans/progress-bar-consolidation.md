# Progress-bar consolidation

**Status:** The core consolidation shipped (one whole-job bar with overall ETA and step count); the open follow-ups below remain.

## Open follow-ups

- **Finalize lumps ~6 sub-stages into one step.** The finalize step (drop-none,
  relazify, collapse-dup, diversity-tree, registry, projection) is a single
  weighted slice. The `FinalizeProgress` proxy (`stages/_common.py`) now spreads
  it across ordered sub-ranges (cleanup 0.05, dedup 0.15, diversity 0.30,
  registry 0.45, projection 0.05) so finishing one sub-stage no longer pins the
  bar at 100 % while the rest run — the bar advances once, monotonically, across
  the phase. The remaining roughness is that those sub-slot shares are static
  ballparks (e.g. on a non-cuML GPU host the diversity k-means can outweigh the
  registry save), not measured; a future pass could record real per-sub-stage
  durations and feed them back in, mirroring the device-aware top-level weights.
- **Multi-embedder (v3 trio) holds mid-embed.** `_embed_missing_stage` loops
  over each bound embedder; each `embed_missing` call restarts its `current` at
  0, so after the first embedder fills the embed slice the clamp holds the bar
  steady through the 2nd/3rd embedders. No backslide, but a long static stretch.
  Rare for demo imports (single embedder); fix by reporting cumulative progress
  across the embedder loop.
- **Text sort (`sort` channel) is not unified.** `vtsearch/routes/sorting.py`
  encodes its 3 phases as `current/total` (current = step index) rather than
  the `step`/`total_steps` fields, so it gets no `overall` and stays a coarse
  3-step bar. Refactor it to report `step`/`total_steps` (with the embedder
  load as a real sub-progress) to get the unified bar + overall ETA there too.
- **Indeterminate sub-steps park at the step floor.** A phase that reports no
  `total` (e.g. model load) sits the bar at its weighted floor until the next
  phase. That's honest but static; per-step weights (now shipped) mitigate it,
  but a phase with no `total` still can't animate within itself.
- **Weights are static guesses.** *(Largely resolved.)* The per-step weights are
  now fit to measured per-phase durations via an affine `n`-aware cost model —
  see `docs/plans/progress-weight-calibration.md` (executed 2026-07 for
  `{cpu,cuda} × {image,audio}`, default embedder; coefficients in
  `vtscore/datasets/stages/_load_cost_model.py`). The fixed model-load cost is why
  one static vector can't serve both small and large datasets, so
  `load_step_weights(media_type, *, n, download_size_mb, embedder)` computes the
  vector from the cost model when `n` is known and a coefficient row exists, and
  otherwise returns the static per-(device, media) profile (the large-`n`
  asymptote). Remaining media/embedders/cuML cells are uncalibrated and keep the
  static profiles (incl. the CPU **image** stopgap
  `_LOAD_STEP_WEIGHTS_CPU_IMAGE`).
- **Promote builds the diversity tree synchronously.** Promote now builds + caches
  the tree in-request (the app runs with `VTSEARCH_TIMEOUT=0`, so long creates
  are tolerated), which means no fine-grained progress during a large promote. If
  that becomes a pain, background the promote like the import pipeline. Other
  cache-miss reloads (old pickles predating tree caching, or a media set that
  shifts on load) still rebuild once; a "write the rebuilt tree back to the
  pickle on first reload" pass would cover those too.
- **Dead dashboard fields.** `progressValue` / `progressTotal` /
  `progressIndeterminate` on `DashboardComponent` are set by the Find polling
  but never rendered; safe to remove in a cleanup pass.
- **Eval (train-and-score)** is intentionally left single-phase (one smooth
  `current/total` bar); revisit only if it grows distinct phases.
