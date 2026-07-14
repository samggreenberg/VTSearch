# Standard-Workflow Polish

**What this is:** The open work from a hands-on audit (2026-06-04) of VTSearch's
core "train a detector on dataset A, find matches on dataset B" loop, exercised
end-to-end on image media through Autopilot labeling and clipboard export. The
happy path was solid (zero console/server errors); the remaining items are
papercuts concentrated in the demo-dataset picker, export-format polish, and a
couple of small affordances. Findings already fixed on `dev` (the media-type
dropdown a11y, the shared clipboard-copy component, the Good-default export
filter) are **not** repeated here.

Each item is independently shippable with file pointers. Items are named
(stable labels, never renumbered) and separated by `<!-- item-sep -->`
sentinels; when you ship a slice, delete only your item's own lines and leave
the sentinels intact (see the plan-file policy in `CLAUDE.md`). `P2`/`P3` mark
the original priority.

---

<!-- item-sep -->

- [ ] #2355 — Fill in missing demo media counts

<!-- item-sep -->

- **Goods count over-reports with no confidence signal (P2)** —
  `right-panel/labelset-list/labelset-list.component.html:3` renders just
  `Goods ({{ elements().length }})`; because the auto threshold leans toward
  recall, this reads as N confident hits when many are low-confidence. **Fix:**
  surface the score/threshold or a confidence band in the hits header so users
  calibrate trust. **Files:** `labelset-list.component.html` (+ whatever feeds
  the count).

<!-- item-sep -->

- [ ] #2357 — "Email us" mailto has no recipient

<!-- item-sep -->

- [ ] #2358 — UCSF Documents listed under Image media type

<!-- item-sep -->

- **Bads-phase default focus on the Good button (P3)** — during Autopilot's
  "Find Initial Bads" phase nothing moves keyboard focus to the Bad action, so
  pressing Enter can mislabel. There's no focus-follows-phase logic in
  `label-view.component.{html,ts}`. **Fix:** move default focus to the phase's
  primary action (or to neither button) as the phase changes. Verify in a live
  browser before shipping — the repro is hard to confirm statically. **Files:**
  `label-view.component.ts`.

<!-- item-sep -->

- [ ] #2360 — Header "Data:" label lags the active dataset
