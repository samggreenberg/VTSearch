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
  indeterminate. `formatProgressHeader` now surfaces `step S of T` in the
  header. Wired into the dashboard loading rows (dataset-card, detector-card,
  orphan-task rows), the Find/learned-sort overlay (find-view) and the
  left-panel sort indicator (via `SortStateService.sortOverall` /
  `sortEtaSeconds`).

- **Docs.** `docs/api/events.md` and `docs/api/dashboard.md` document the
  `overall` / `eta_seconds` fields.

- **Tests.** `tests_lib/core/test_progress_overall.py` (fraction math,
  monotonicity, new-job reset, overall ETA) and
  `frontend/src/app/utils/format-progress.spec.ts` (`progressBarState`,
  step-count header).

## Open follow-ups

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
  media-type/embedder and feed measured weights back in.
- **Dead dashboard fields.** `progressValue` / `progressTotal` /
  `progressIndeterminate` on `DashboardComponent` are set by the Find polling
  but never rendered; safe to remove in a cleanup pass.
- **Eval (train-and-score)** is intentionally left single-phase (one smooth
  `current/total` bar); revisit only if it grows distinct phases.
