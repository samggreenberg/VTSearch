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

- [ ] #2382 — Add a confidence signal to the Goods count header

<!-- item-sep -->

- [ ] #2357 — "Email us" mailto has no recipient

<!-- item-sep -->

- [ ] #2358 — UCSF Documents listed under Image media type

<!-- item-sep -->

- [ ] #2383 — Move keyboard focus off the Good button during Autopilot "Find Initial Bads"

<!-- item-sep -->

- [ ] #2360 — Header "Data:" label lags the active dataset
