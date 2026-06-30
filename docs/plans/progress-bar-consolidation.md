# Progress-bar consolidation

**Status: shipped.** Multi-step jobs now render one whole-job progress bar
(0→100 %, never resetting) with a true overall ETA and a visible step count,
instead of a fresh per-phase bar that filled, snapped back to 0 %, and started
over for the next phase.

## Problem

Adding a demo dataset (and other multi-phase loads) showed a sequence of
separate progress bars — download, model load, embed, finalize — each with its
own `current/total` and its own per-phase ETA. The bar visually reset at every
phase, and the ETA counted down toward an arrival we never reached: it just
restarted as the next phase began. The user never saw how many phases the whole
job had, and there was no honest "% of the total job done".

## What shipped

- **Unified `overall` fraction (backend).**
  `ProgressTracker._compute_overall` (`vtscore/concurrency/progress.py`)
  computes a single whole-job completion fraction whenever a caller reports a
  `step`/`total_steps` structure. The within-step `current/total` is mapped into
  the job's overall span by `_overall_raw_fraction`:

  ```
  weighted: (Σ weights[:step-1] + weights[step-1] * within) / Σ weights
  equal:    ((step - 1) + within_step_fraction) / total_steps   # when no weights
  ```

  **Per-step weights** (`ProgressTracker.set_step_weights`, or the `step_weights`
  arg to `LoadingTasksTracker.create_task`) let each flow reflect the phases it
  *knows* are longer, so the bar paces by real time instead of lurching:
  - dataset load `[0.25, 0.15, 0.50, 0.10]` — embed dominates (download, model
    load, embed, finalize); see `stages/_common.py:_LOAD_STEP_WEIGHTS`.
  - detector load `[0.15, 0.15, 0.70]` — MLP training dominates.
  - Find `[0.10, 0.30, 0.60]` — scoring dominates.
  - Find-label `[0.10, 0.45, 0.40, 0.05]` — train + score carry the cost.

  Weights need only be in the right ballpark: they shape *pacing* only, since
  the overall ETA self-corrects from the real elapsed-vs-fraction rate. Flows
  that supply no weights fall back to equal weight per step. The fraction is
  clamped monotonic non-decreasing within a job so the bar never visibly
  retreats; a step going backwards (or `total_steps` changing) is read as a new
  job and resets the clock. Exposed as a new `overall` key in
  `_PROGRESS_COMMON_EXTRAS`, so every tracker carries it.

- **True overall ETA (backend).** When `overall` is present, `eta_seconds` is
  derived from the elapsed-vs-overall-fraction rate over a single clock that
  runs for the whole job (not per-phase), with the same EMA smoothing the
  per-phase ETA used. It self-corrects as the real rate emerges and no longer
  resets between phases. Single-phase operations keep the per-phase ETA.

- **One bar + step count (frontend).**
  `progressBarState()` in `utils/format-progress.ts` is the shared resolver:
  prefer `overall` (one bar across the whole job) → else `current/total` → else
  indeterminate. `formatProgressHeader` now surfaces a capitalized `Step S of T`
  in the header (each `·`-separated segment is title-cased so the line reads as a
  row of labels, e.g. `Loading dataset · Step 3 of 4 · Embedding files`), and the
  per-item `detail` line drops the parentheses around the count and strips the
  redundant leading verb (`012/345 cats/img.png`, not `(012/345) Embedding
  cats/img.png`) so the narrow, ellipsized detail slot spends its characters on
  the filename rather than repeating the phase the header already names. Wired
  into the dashboard loading rows (dataset-card, detector-card,
  orphan-task rows), the Find/learned-sort overlay (find-view) and the
  left-panel sort indicator (via `SortStateService.sortOverall` /
  `sortEtaSeconds`).

- **Docs.** `docs/api/events.md` and `docs/api/dashboard.md` document the
  `overall` / `eta_seconds` fields.

- **Tests.** `tests_lib/core/test_progress_overall.py` (fraction math,
  monotonicity, new-job reset, overall ETA) and
  `frontend/src/app/utils/format-progress.spec.ts` (`progressBarState`,
  step-count header).

## Smoothness pass (shipped)

Follow-up to the consolidation: the unified bar was monotonic *in principle* but
still got "knocked around" between stages. Fixed the concrete jump sources:

- **Clipper backslide (the worst offender).** The clipper stage
  (`vtscore/datasets/stages/clipper.py`) reported the *finalize* step while
  running *before* the embed step. It ran the bar to ~100 %, then the following
  embed step (a lower number) tripped `_compute_overall`'s "step went backwards
  = new job" reset and slammed the bar from ~100 % back to mid-job. Clipping is
  really the embed phase for clipped datasets (it cuts + embeds clips; the later
  `_embed_missing_stage` is then a no-op), so it now reports the **embedding**
  step. The whole-job fraction stays monotonic across a clipped load.
- **`converting` status nulled the bar.** `converters/runner.py` emits a
  `"converting"` status (document→image, video→frames) that was missing from
  `_STATUS_TO_STEP`, so it resolved to `step=None`, nulled `overall`, and bounced
  the bar onto the raw `current/total` scale mid-job. It now maps to the loading
  step (it's pre-embed work). The map's docstring states the invariant: every
  load-path status must be present.
- **Smaller model-load jump.** `_LOAD_STEP_WEIGHTS` shifted `0.25/0.15/0.50/0.10`
  → `0.25/0.10/0.55/0.10`: the un-measurable model-load slice (the one that
  fills in a single step the moment embedding starts) is trimmed and the freed
  weight handed to embedding, the phase that reports per-item progress — so more
  of the bar advances smoothly and the between-stage jump is smaller.
- **Gentle fill ease (frontend).** `ProgressBarComponent` gained an opt-in
  `smooth` input (a longer `0.6s ease-out` fill transition, class
  `progress-fill--smooth`) wired into the dataset-load bars (dashboard orphan
  rows, dataset-card, detector-card). Unavoidable between-phase jumps glide
  instead of snapping. Honest by construction: the fill only ever eases *toward*
  the real reported value, never ahead of it (no fake/timer-driven motion — that
  approach was explicitly rejected).

## GPU pacing profile (shipped)

The static load weights `[0.25, 0.10, 0.55, 0.10]` assumed embedding dominates
wall-clock — true on a CPU host, false on a GPU one. On a GPU the embed phase is
several times faster, but the finalize phase is **not** GPU-accelerated: the
registry save (serialize → zip → disk write) is always CPU, and the
diversity-tree k-means only moves to the GPU when cuML is installed (a
best-effort RAPIDS dep that is frequently absent). So on a GPU host finalize
dominated wall-clock while still getting only 10 % of the bar — the bar raced to
~90 % during embed, then crawled through the last 10 % for 20+ seconds while the
rate-based ETA (computed mostly from the fast embed phase) reported "< 5 sec
left".

Fixed by making the weights device-aware (`load_step_weights()` in
`stages/_common.py`, resolved once at task creation): the CPU profile is
unchanged, and a GPU profile `[0.20, 0.10, 0.30, 0.40]` shrinks embed and grows
finalize so the bar paces by real time and the overall ETA has room to
self-correct through the diversity-tree build + registry save. Detector-load
weights are unaffected. Tests: `tests_lib/datasets/test_load_step_weights.py`.

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
- **Weights are static guesses.** The per-step weights are fixed ballparks, not
  learned. A future refinement could record real per-phase durations per
  media-type/embedder and feed measured weights back in. **Specified in
  `docs/plans/progress-weight-calibration.md`** — including making the weights
  depend on dataset size `n` via an affine per-phase cost model (the fixed
  model-load cost is why one static vector can't serve both small and large
  datasets). Not yet run (needs a GPU host). A CPU **image** profile
  (`_LOAD_STEP_WEIGHTS_CPU_IMAGE`) shipped as a stopgap: images embed cheaply, so
  the generic 55%-embed CPU profile mispaced image imports.
- **Dead dashboard fields.** `progressValue` / `progressTotal` /
  `progressIndeterminate` on `DashboardComponent` are set by the Find polling
  but never rendered; safe to remove in a cleanup pass.
- **Eval (train-and-score)** is intentionally left single-phase (one smooth
  `current/total` bar); revisit only if it grows distinct phases.
