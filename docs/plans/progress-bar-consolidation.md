# Progress-bar consolidation

**Status: shipped** (consolidation + smoothness pass + GPU pacing +
registry-reload pacing). Multi-step jobs now render one whole-job progress bar
(0→100 %, never resetting) with a true overall ETA and a visible step count,
instead of a fresh per-phase bar that filled, snapped back to 0 %, and started
over. Open follow-ups below.

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

## What shipped

### Consolidation (the core)

- **Unified `overall` fraction (backend).** `ProgressTracker._compute_overall`
  (`vtscore/concurrency/progress.py`) computes one whole-job completion fraction
  whenever a caller reports `step`/`total_steps`; the within-step `current/total`
  maps into the job's span via `_overall_raw_fraction` (weighted:
  `(Σ weights[:step-1] + weights[step-1]·within) / Σ weights`; equal:
  `((step-1) + within) / total_steps`). Per-step weights
  (`set_step_weights` / the `step_weights` arg to
  `LoadingTasksTracker.create_task`) pace by real time — dataset load
  `[0.25,0.15,0.50,0.10]`, detector load `[0.15,0.15,0.70]`, Find
  `[0.10,0.30,0.60]`, Find-label `[0.10,0.45,0.40,0.05]`; no-weight flows fall
  back to equal weight. Fraction is clamped monotonic non-decreasing; a step
  going backwards / `total_steps` changing is read as a new job and resets the
  clock. Exposed as a new `overall` key in `_PROGRESS_COMMON_EXTRAS`.
- **True overall ETA (backend).** When `overall` is present, `eta_seconds` is
  derived from elapsed-vs-overall-fraction over a single whole-job clock with EMA
  smoothing; self-corrects and no longer resets between phases. Single-phase ops
  keep the per-phase ETA.
- **One bar + step count (frontend).** `progressBarState()`
  (`utils/format-progress.ts`) prefers `overall` → `current/total` →
  indeterminate; `formatProgressHeader` surfaces a title-cased `Step S of T` and
  the per-item `detail` drops the parens + redundant leading verb. Wired into the
  dashboard loading rows (dataset-card, detector-card, orphan-task rows), the
  Find/learned-sort overlay (find-view), and the left-panel sort indicator (via
  `SortStateService.sortOverall` / `sortEtaSeconds`).
- **Docs + tests.** `docs/api/events.md` + `docs/api/dashboard.md` document
  `overall` / `eta_seconds`; `tests_lib/core/test_progress_overall.py` +
  `frontend/src/app/utils/format-progress.spec.ts`.

### Smoothness pass

- **Clipper backslide fixed.** The clipper stage
  (`vtscore/datasets/stages/clipper.py`) reported the *finalize* step while
  running *before* embed, ramming the bar ~100 %→mid-job on the backwards-step
  reset. Clipping is really the embed phase for clipped datasets, so it now
  reports the **embedding** step; the whole-job fraction stays monotonic.
- **`converting` status added to `_STATUS_TO_STEP`.** It was missing, so
  `converters/runner.py`'s `"converting"` resolved to `step=None`, nulled
  `overall`, and bounced onto the raw `current/total` scale; now maps to the
  loading step (pre-embed work). Map docstring states the every-load-status
  invariant.
- **Model-load weight shift** `0.25/0.15/0.50/0.10 → 0.25/0.10/0.55/0.10` — trims
  the un-measurable model-load slice, hands the weight to embedding (which
  reports per-item progress), shrinking the between-stage jump.
- **Gentle fill ease (frontend).** `ProgressBarComponent` opt-in `smooth` input
  (a `0.6s ease-out` fill, class `progress-fill--smooth`) on the dataset-load
  bars; only ever eases *toward* the real reported value (no fake/timer motion).

### GPU pacing profile

- **Device-aware weights** (`load_step_weights()` in `stages/_common.py`,
  resolved once at task creation). CPU profile unchanged; GPU profile
  `[0.20,0.10,0.30,0.40]` shrinks embed and grows finalize, because on a GPU
  embed is fast but finalize (registry save + non-cuML diversity k-means) is CPU
  and dominates wall-clock. Detector-load weights unaffected. Tests:
  `tests_lib/datasets/test_load_step_weights.py`.

### Registry-reload pacing + redundant rebuild

- **Frozen bar fixed.** The registry/pickle reload path
  (`load_registered_dataset`, `vtsearch/routes/datasets/registry.py`) is a 2-step
  mini-pipeline (read pickle → dedup + diversity) that set `total_steps=2` with
  no weights, so the near-instant dedup pinned the overall bar at ~100 % while the
  ~45 s diversity rebuild sat frozen. Fixed by moving dedup into step 1 and
  `set_step_weights([0.15, 0.85])` so the diversity build owns the bulk of the
  bar; a terminal `Finalizing…` fills the slice to 100 % in every branch.
- **Redundant rebuild fixed.** Promoted datasets (`/api/dataset/promote`) omitted
  the cached diversity tree and renumber IDs (so the source tree can't be reused),
  rebuilding the full k-means on every reload. Now the tree is built + cached at
  promote time (`_diversity_tree_pickle_keys` +
  `build_diversity_tree_serializable`), gated by the same
  `should_auto_build_diversity_tree` threshold, so reopening a promoted dataset
  restores instead of rebuilding.
