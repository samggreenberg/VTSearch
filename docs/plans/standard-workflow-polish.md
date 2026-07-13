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

- **Fill in missing demo media counts (P2)** — `vtscore/datasets/demo_counts.py`
  `DEMO_MEDIA_COUNTS` has exact counts for caltech101, eurosat, reuters21578,
  20newsgroups, arxiv_abstracts, oxford_flowers_102, ucf101, roxford5k, and
  bbc_news, but still falls back to an inaccurate estimate for caltech256,
  places365, urbansound8k, gtzan, speech_commands_v2, hmdb51, kth, ucf101_full,
  ucsf_documents, and dbpedia. **Fix:** download each and run
  `scripts/compute_demo_counts.py <id>`, then paste the emitted lines into
  `demo_counts.py`. **Files:** `vtscore/datasets/demo_counts.py`.

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

- **Goods count over-reports with no confidence signal (P2)** —
  `right-panel/labelset-list/labelset-list.component.html:3` renders just
  `Goods ({{ elements().length }})`; because the auto threshold leans toward
  recall, this reads as N confident hits when many are low-confidence. **Fix:**
  surface the score/threshold or a confidence band in the hits header so users
  calibrate trust. **Files:** `labelset-list.component.html` (+ whatever feeds
  the count).

<!-- item-sep -->

- **"Email us" mailto has no recipient (`mailto-recipient`, P3)** — the
  "Email us" affordance now lives in the Help modal
  (`keyboard-help-modal.component.html`, `.help-footer`) as
  `mailto:?subject=VTSearch%20Issue%3A` — the subject typo is fixed but the
  to-address is still empty, so "Email us" opens a blank-recipient compose
  window. **Fix:** add the recipient address. **Files:**
  `keyboard-help-modal.component.html`. (The "Simplify header and layout IA"
  slice that moved this out of the header logo has already shipped.)

<!-- item-sep -->

- **UCSF Documents listed under Image media type (P3)** —
  `vtscore/media/image/_demo_sources.py:556` registers `ucsf_documents_a` under
  the image `_MEDIA_TYPE_ID`, so scanned document pages appear atop the Image
  demo list. Arguably intentional (pages are images), so this is a labeling
  call. **Fix:** relabel to signal it's scanned documents, or recategorize.
  **Files:** `vtscore/media/image/_demo_sources.py`.

<!-- item-sep -->

- **Bads-phase default focus on the Good button (P3)** — during Autopilot's
  "Find Initial Bads" phase nothing moves keyboard focus to the Bad action, so
  pressing Enter can mislabel. There's no focus-follows-phase logic in
  `label-view.component.{html,ts}`. **Fix:** move default focus to the phase's
  primary action (or to neither button) as the phase changes. Verify in a live
  browser before shipping — the repro is hard to confirm statically. **Files:**
  `label-view.component.ts`.

<!-- item-sep -->

- **Header "Data:" label lags the active dataset (P3, verify first)** —
  reported drift between the dashboard row-checkbox "selected" notion and the
  header's "active dataset" label. Lower confidence this still reproduces and it
  overlaps the nav-picker-lag item in `ui-ux-papercuts.md`; confirm in a live
  browser before scoping, and fold into that item if it's the same root cause.
