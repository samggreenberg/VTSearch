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
  `step`/`total_steps` structure:

  ```
  overall = ((step - 1) + within_step_fraction) / total_steps
  ```

  Equal weight per step (plus the within-step `current/total`). Equal weighting
  is deliberate — the exact cost split between phases is unpredictable (a cached
  load skips the download; a tiny dataset embeds in a blink) and matters far
  less than knowing *how many* phases there are. The fraction is clamped
  monotonic non-decreasing within a job so the bar never visibly retreats; a
  step going backwards (or `total_steps` changing) is read as a new job and
  resets the clock. Exposed as a new `overall` key in `_PROGRESS_COMMON_EXTRAS`,
  so every tracker (dataset/detector loads, find, …) carries it.

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
- **Indeterminate sub-steps park at the step floor.** With equal weighting, a
  phase that reports no `total` (e.g. model load) sits the bar at `(step-1)/N`
  until the next phase. That's honest but static; if a phase reliably dominates
  wall-clock time, per-step weights (a `step_weights` arg on `ProgressTracker`)
  would make the bar track time better.
- **Dead dashboard fields.** `progressValue` / `progressTotal` /
  `progressIndeterminate` on `DashboardComponent` are set by the Find polling
  but never rendered; safe to remove in a cleanup pass.
- **Eval (train-and-score)** is intentionally left single-phase (one smooth
  `current/total` bar); revisit only if it grows distinct phases.
