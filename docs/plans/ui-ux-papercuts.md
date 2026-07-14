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

<!-- item-sep -->

<!-- item-sep -->

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
