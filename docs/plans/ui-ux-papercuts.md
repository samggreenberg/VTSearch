# UI/UX Papercuts

**What this is:** The open work distilled from two browser-driven UX sweeps —
an end-to-end flow walkthrough (2026-05-28, `U#`/`O#` ids) and an empty/
edge-state sweep (2026-05-28, `V#`/`U#`/`B#`/`O#` ids). Findings the two
reports shared (the stale-request storm, the nav-picker lag, the "fresh
detector reads as broken" nit) are deduped into single items here. Findings
already fixed on `dev` (the long-name table collapse, the toast overlap, the
non-optimistic rename, the picker media-type preselect) are **not** repeated.

Each item is independently shippable with file pointers and an approach. Items
are named (stable labels, never renumbered) and separated by `<!-- item-sep -->`
sentinels; when you ship a slice, delete only your item's own lines and leave
the sentinels intact (see the plan-file policy in `CLAUDE.md`).

Severity is carried from the reports (High = broken/blank-pane/data-integrity;
Med = confusing; Low = cosmetic) so you can pick the high-value ones first.

---

<!-- item-sep -->

- **Reset Find sub-view state on dataset change (High)** — opening Find (or
  Train) against a dataset *smaller* than the previous one fires image requests
  for media ids that only existed in the prior dataset, producing a storm of
  404s (edge-states amplified it to ~20 simultaneous). Root cause: the
  singleton sub-view state isn't reset on entry.
  `find-view.component.ts` `ngOnInit` (~lines 191-240) does **not** clear shared
  state — only the in-place `reloadForNewPair()` (~242-254) does
  (`sortState.setSortResults([])` / `voteState.clear()`). On a fresh
  Dashboard → Find navigation, `SortStateService.sortOrder` still holds the
  previous session's ranking, so the grid briefly renders old ids against the
  new dataset context. **Fix:** clear `sortState`/`voteState` at the top of
  `ngOnInit` (mirror `reloadForNewPair`'s reset) before `loadMedias()`/
  `runFindLabel()`, or drop/remap in-flight requests when
  `activeContext.datasetId` changes. **Files:** `find-view.component.ts`.
  (Reported as O1 in both sweeps.)

<!-- item-sep -->

- **Autopilot exhausted-state for tiny datasets (High)** — on a 1-item (or
  otherwise small) dataset, autopilot's "Find Initial Goods" needs 3 goods and
  the bad phase needs 4 bads; those targets are unreachable, so autopilot never
  advances, shows no "exhausted" state, and the center pane goes blank while the
  metadata strip keeps the stale item. `autopilot-state.service.ts`
  `INITIAL_STATE` hardcodes `goodToStart: 3, badToStart: 4`, and
  `checkPhaseTransition()` (~87-108) gates purely on `count < target` with no
  dataset-size awareness and no terminal branch (the `'done'` phase needs
  ≥5 good/5 bad, so small datasets can never reach it). **Fix:** cap each phase
  target at `min(target, remainingUnlabeledCount)` and add an explicit
  "dataset exhausted / nothing left to label" terminal state that renders a
  message instead of a blank pane. **Files:** `autopilot-state.service.ts`,
  the label-view template that renders the blank pane. (edge-states B1.)

<!-- item-sep -->

- **Length guard + safe filename + path-scrub for renames (High)** — no
  input-length cap exists anywhere, so a very long dataset/detector name
  overruns the filesystem `NAME_MAX` and raises an uncaught `OSError` whose
  toast still contains the absolute server path (a path-leak). The rename
  schemas validate only `Length(min=1)`
  (`vtsearch/schemas/detectors.py:200`, `vtsearch/schemas/datasets.py:303,592`);
  `vtscore/detectors/store.py:_slug()` (~line 38) slugifies but never
  truncates, and `_write_detector` (~line 62) builds a `<slug>.json` plus a
  longer `.tmp` sibling. **Fix:** (a) add a `Length(max=...)` to the rename/
  create schemas, (b) truncate/hash the derived filename in `_slug`, and
  (c) strip or relocate absolute paths out of the user-visible error message
  (see the error surface `frontend/src/app/utils/api-error.ts` /
  `toast-container`). **Files:** `schemas/detectors.py`, `schemas/datasets.py`,
  `store.py`, `api-error.ts`. (edge-states B2 + V5; the optimistic-UI half of
  B2 is already fixed — the dashboard rename now refreshes only on success.)

<!-- item-sep -->

- **Nav picker lags dashboard selection (Med)** — after importing/creating a
  dataset or detector, the dashboard treats the row as selected (the bottom CTA
  enables) but the top-bar nav pickers still read "Select a dataset" / "Select a
  detector" and show the gray inactive dot; the picker only catches up after
  navigating into a downstream view. The literal placeholder is at
  `context-pulldown.component.ts:220`. The dashboard mirrors table selection
  into the top bar but not into the shared active observable the picker reads.
  **Fix:** on implicit select, also update the observable the pulldown binds to
  (mirror `DashboardSelectionService` into the active-context intent the picker
  reads). **Files:** `context-pulldown.component.ts`, `dashboard.component.ts`.
  (e2e-flows U1 + edge-states U1/V2.)

<!-- item-sep -->

- **Offer a backup before settings reset (Med)** — `resetDefaults()`
  (`settings-modal.component.ts:561`) calls `dialog.confirmDestructive` with
  "…overwritten and cannot be recovered" but never surfaces Export as an escape
  hatch, so a user who wants to keep their config has no in-flow way to save it
  first. **Fix:** add a secondary "Export current settings before resetting"
  affordance in or beside the confirm dialog. **Files:**
  `settings-modal.component.ts`. (e2e-flows U13.)

<!-- item-sep -->

- **Multi-source importer (High, spec decision)** — the Add-Dataset picker only
  offers single-source importers (Services / Server / Local / Demo); there is no
  importer that accepts multiple source rows with per-row converters (e.g. a
  `video → video2image` row and a `document → document2image` row feeding one
  image dataset). Combine Datasets is adjacent but only merges same-media-type
  pickles. This is a scope/spec decision, not a bug fix: either design + build
  the multi-source importer, or decide it's out of scope and remove the
  multi-media flow from this plan and the `browser-vision-testing.md` Task 3
  template (which currently assumes it exists). **Files:** new importer plugin +
  picker/form UI, or a doc-only descope. (e2e-flows U10.)

<!-- item-sep -->

- **New-detector "example required" hint (Low)** — on the Blank tab, the
  "Example" label (`new-detector-modal.component.html:71`) has no required
  marker (unlike the `<span class="required">*</span>` on Trained-tab fields),
  there's no empty-state hint when both Text and Media examples are blank, and
  the Create button (~line 335) is `[disabled]="!canSubmitBlank"` with a title
  that describes success rather than the blocker. **Fix:** add a required marker
  on the Example group or an inline "Provide a text or image example" hint when
  empty. **Files:** `new-detector-modal.component.html`. (e2e-flows U6.)

<!-- item-sep -->

- **Combine dedup-summary toast (Low)** — `onCombineStarted()`
  (`dashboard.component.ts:543`) closes the modal and starts progress polling
  but emits no post-combine summary, so a user whose sources collapsed (e.g.
  80 → 50) is never told how many unique items were kept vs. duplicates dropped.
  **Fix:** emit a completion toast ("Combined 2 datasets into 1 — 50 unique
  kept, 30 duplicates dropped"). **Files:** `dashboard.component.ts`.
  (e2e-flows U11.)

<!-- item-sep -->

- **Fresh detector "awaiting labels" affordance (Low)** — a just-created
  detector renders `num_training` as raw `0` and `last_trained_at` as `-`
  (`detector-card.component.html:88-92`) with no framing, so it reads as
  "broken" rather than "new". **Fix:** show a contextual "Awaiting labels" /
  "Empty" hint on zero-training detectors. **Files:**
  `detector-card.component.html`. (e2e-flows U7 + edge-states U3.)

<!-- item-sep -->

- **Cold-boot Add CTA placement (Low)** — on first boot (no datasets), the "+"
  Add button sits in the section header, away from the centered empty-state
  text, so the call-to-action and the "nothing here yet" message are visually
  disconnected. **Fix:** co-locate the primary Add action with the empty-state
  copy. **Files:** `dashboard.component.{html,scss}`. (edge-states V1.)

<!-- item-sep -->

- **`solo_media_type` scope in the Add-Dataset picker (Low)** — with
  `solo_media_type` set, the server-folder picker now preselects that media type
  (`server-folder-picker.component.ts:179-181`), but incompatible importers/tabs
  aren't filtered out and the Settings wording doesn't state the scope, so it's
  unclear what "solo" actually constrains. **Fix:** filter the picker
  tabs/importers to the solo media type, and clarify the Settings help text.
  **Files:** the Add-Dataset picker components, settings copy. (edge-states U4.)
