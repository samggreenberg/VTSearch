# Progress UX

**What this is:** The surviving open work from a browser-driven long-running-op
observability review (2026-05-28, `O#` ids). Most of that report's findings —
the ETA/bar disagreement (O3), the duplicated count (O4), the "saving to
registry" flicker (O5), the concurrent-import gate contradiction (O8), the
greyed-out sibling rows (O10) — were resolved by the shipped progress-bar
consolidation and dashboard refactor and are **not** repeated here. What's left
is one solid cancel-feedback gap plus a few low-value follow-ups.

This plan is distinct from `progress-bar-consolidation.md` and
`progress-weight-calibration.md`, which track *bar pacing / per-phase weight*
work; the items below are about *what the UI communicates* during long ops and
don't overlap either.

Items are named (stable labels, never renumbered) and separated by
`<!-- item-sep -->` sentinels; when you ship a slice, delete only your item's
own lines and leave the sentinels intact (see the plan-file policy in
`CLAUDE.md`).

---

<!-- item-sep -->

- **Cancel acknowledgement state (High)** — after clicking Cancel on a
  dataset-load row, the UI freezes in its pre-cancel state (same bar fill, same
  live "Cancel" button, same sub-status) for the ~15–20 s the backend takes to
  unwind, inviting repeated clicks. There is no "cancelling…" state anywhere in
  the frontend (grep finds only comments): `onCancelTask`
  (`dataset-card.component.ts:233-237`) just emits, and `cancelLoadingTask`
  (`dashboard-loading-tasks.service.ts:272-274`) just hits the API. **Fix:** on
  cancel, set a per-row "cancelling" flag → swap the sub-status to "cancelling…"
  and replace the Cancel button with a disabled "Cancelling…" badge; clear it
  when the task leaves the active list. Pure frontend — no backend change
  needed (`progress.py`'s `is_cancelled`/`mark_finished` already expose enough
  state if a backend "cancelling" status is later wanted). **Files:**
  `job-progress.component.{ts,html}`, `dataset-card.component.ts`,
  `dashboard-loading-tasks.service.ts`. (O11.)

<!-- item-sep -->

- **Deferred train/rerank progress badge (Low)** — a per-vote re-rank shows no
  feedback; on a large dataset or a constrained host the gap between voting and
  the next item is silent. **Fix:** a lower-rail "training…" pill that only
  appears when the rerank exceeds a threshold (~500 ms), keeping fast cycles
  silent. **Files:** the label-view / autopilot area. (O6.)

<!-- item-sep -->

- **Warm/cold cache-hit hint (Low)** — a warm re-import skips the model-load
  phase and is much faster, but nothing tells the user why, so the two runs look
  inconsistent. **Fix (optional nicety):** show a brief "(embedder cache hit)"
  subtitle for the first second of a warm import so the faster path is expected.
  Low value — include only when bundling other progress-UX polish. (O2.)

<!-- item-sep -->

- **Large-dataset Find/autodetect progress verification (Low, needs browser)** —
  the `find_progress` tracker exists but the large-dataset Find affordance was
  never exercised at scale; confirm the Find view surfaces progress for a
  1000+ item dataset rather than showing a blank view. This is a
  verification/test task that needs a real browser (unavailable in the standard
  cloud container); it may turn up a small code gap. (O12.)
