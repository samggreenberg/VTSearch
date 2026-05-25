# Frontend bundle organization

Status: **#1 shipped (all five checkpoints). #2 shipped. #3 shipped.
#4 shipped. #5 investigation shipped (per-domain split deferred as a
follow-up).** Item #6 is scoped but not started.

## What shipped

- **#1 Checkpoint 1** (`fe2ef1e2`): extracted `<vt-import-advanced>`
  from `dataset-importer-modal.component.ts`. The Advanced block
  (Include media + Embedder + Clipper) was inlined four times in the
  modal with `sf*` / `lf*` / `form*` / `demo` parallel state and a
  fleet of context-dispatching helpers (`clipperDisplayName`,
  `isDefault*`, `show*Picker`, `showAdvancedToggle`,
  `contextHasIncludeMedia`, `toggleAdvanced`, `recommendedEmbedders`,
  `advancedEmbedders`, `licenseNoticeFor`, `embedderLabel`). All four
  call sites now render the same `<vt-import-advanced>` with per-flow
  inputs; the helpers and the shared `advancedOpen` flag are gone.
  Parent .ts dropped from 1814 → 1668 lines; template from 669 → 545
  lines. **Initial bundle went from 540 kB → 526.40 kB** (gzip
  136.35 kB), back under the previous (525 kB) warning threshold.
- **#1 Checkpoint 2** (`d1d3e2d7`): extracted `<vt-import-config>` —
  the output media-type select + auto-detection hint chip — from the
  `sf` and `lf` flows. The block was inlined twice with identical
  markup (same label, same select shape, same `.detection-hint`
  rendering) and only the bound state and field id differing. Both
  call sites now consume `<vt-import-config>`, and the
  `.detection-hint` rule moved out of the parent SCSS into the new
  component. The parent gains a `mediaTypeOptionLabels` cache
  (folder_import_name → label) so the child does not need
  `MediaTypeInfo`. Template dropped from 545 → 529 lines; .ts grew
  slightly (1668 → 1687) for the cache getter + import. **Lazy
  `dataset-importer-modal-component` chunk dropped 78.96 kB →
  77.64 kB raw** (15.99 → 15.76 kB gzip). Initial bundle unchanged at
  526.40 kB — the modal is `@defer`-loaded, so all gains land in the
  lazy chunk.
  Wrapping `<vt-import-advanced>` inside the new component (the
  plan's optional collapse) turned out to be infeasible without a UX
  re-order: the source widget (folder browser / dropzone) and the
  recursive checkbox sit between the media-type select and the
  Advanced block in both flows. Kept the scope to just the media-type
  + hint block; Advanced stays in its current position.
- **#1 Checkpoint 3**: migrated the generic `form` flow's
  `media_type` select to `<vt-import-config>`. The form flow's
  field iteration now special-cases `field_type === 'select' &&
  field.key === 'media_type'` and renders `<vt-import-config>`
  instead of an inline `<select>` wrapped in the outer
  `<div class="form-group">` + `<label>` + required-star scaffolding.
  All importers that declare a `media_type` select (`server_folder`,
  `server_files`, `http_archive`, `synthetic`, `recaller`) use the
  label "Dataset MediaType", which matches the label
  `<vt-import-config>` renders. The `detectionHint` input is left at
  its default (empty) because the form flow has no auto-detection.
  The now-dead `getMediaTypeOptionLabel` helper was removed (the
  child consumes `mediaTypeOptionLabels` directly).
  The previously-suppressed required-star (`@if (field.required && !(
  field.field_type === 'select' && field.key === 'media_type'))`)
  loses its special case — media-type lives outside the iteration
  branch that renders the star at all. Initial bundle unchanged at
  526.40 kB.
- **#1 Checkpoint 5**: migrated `new-detector-modal`'s media picker
  to consume `<vt-source-picker>`.  Replaces the inline duplicate of
  the picker chrome (category-tab + sub-tab bar), demo media-type
  tabs + sortable demo table, server-folder typed-path input, and
  local-folder/local-files dropzone (~213 lines of template) with a
  single `<vt-source-picker>` call.  Five dead helpers
  (`onDemoHeaderClick`, `statusBadgeClass`, `statusBadgeLabel`,
  `getDemoTabIcon`, `getDemoTabLabel`) and the SCSS rules they backed
  fell out of the parent.  `.server-folder-browser` stays in the
  parent SCSS because the demoFileBrowsing widget (typed-path input
  inside a clicked demo) still uses the same chrome.
  To support new-detector's UX needs, source-picker grew a handful of
  customization inputs: `alwaysShowSubtabBar` (always render the
  sub-tab row, even with a single importer — new-detector wants every
  category visibly labelled), `demoRowDisabledFn` /
  `demoRowTitleFn` (so non-browsable demos render with `.disabled`
  styling and an explanatory tooltip), `sfApplyOnBlur` (off in
  new-detector so the explicit Load button isn't double-triggered),
  `lfShowFieldLabel` / `lfShowFileCount` (off in new-detector), and
  the dropzone label / sublabel / accept overrides
  (`lfFolderDropLabel`, `lfFolderDropSublabel`, `lfFilesDropLabel`,
  `lfFilesDropSublabel`, `lfFilesAcceptAttr`).  A new `[lfInfo]`
  content-projection slot replaces the previously hard-coded "Pick a
  folder…" / "Upload a single file…" paragraphs (each parent now
  projects its own copy), keeping each modal's exact prose intact.
  **Bundle effect.** Initial bundle unchanged (526.40 kB / 136.37 kB
  gzip; both modals are `@defer`-loaded).  Lazy chunks rebalanced:
  the bundler now extracts source-picker into its own shared lazy
  chunk that both modals reference.  `dataset-importer-modal-component`
  dropped 83.09 → 68.11 kB raw (16.94 → 13.87 kB gzip);
  `new-detector-modal-component` dropped 49.78 → 43.30 kB raw (10.87
  → 9.79 kB gzip); the new shared chunk is ~21 kB raw / 5.6 kB gzip.
  The total deferred-chunk weight is slightly higher than before
  Checkpoint 4 (the per-component Angular plumbing adds overhead),
  but the structural goal — single source of truth for the picker
  chrome and demo table — is achieved, and any future edits to the
  picker UI now touch one component instead of two.
- **#2 Split `dashboard.component.ts`**: extracted two focused
  services from the page-shell god component.  Dashboard dropped
  1419 → 1175 lines.
  - **`DashboardModalsService`** consolidates the six row-action /
    selection-action modal states (combine-datasets, combine-detectors,
    label-exporter, label-importer, find-results, dataset-stats).  Each
    modal now has a single source of truth with `openX()` / `closeX()`
    methods; template `@if` blocks read directly off
    `modals.x.open` / `modals.x.payload`.  Replaces six parallel
    `xModalOpen` boolean + `xPayload` field pairs and their open/close
    glue scattered through the component.  Importer / new-detector
    flows continue to live on `NewThingFlowsService` because they're
    also opened from the top-bar context pulldowns — the new service
    only owns dashboard-local modals.
  - **`DashboardLoadingTasksService`** owns the per-task loading lists
    (`loadingTasks` for datasets, `detectorLoadingTasks` for
    detectors), the polling subscriptions, and the bookkeeping for
    `awaitedTaskIds` / `completedTaskIds` / `completedModelTaskIds`.
    Polling auto-resumes on construction whenever the SSE stream
    shows an active task, replacing the dashboard's bespoke
    `resumeActivePolling` and two `startProgressPolling` methods.
    `inlineTaskMap`, `orphanLoadingTasks`, `getInlineTask`,
    `getInlineDetectorTask`, and the cancel / dismiss helpers move
    with the state.
  - **DatasetStateService cleanup**: removed the now-dead
    `loadingTasksSubject` / `loadingTasks$` / `setLoadingTasks` /
    `loadingTasks` surface — nobody read it after the polling moved.
  - **What we did NOT do** (deviation from the original plan, noted
    here so the next contributor doesn't try and back it out):
    - **Row actions stay on the dashboard, not on cards.** The plan
      suggested pushing per-row handlers (rename / delete / load /
      unload / security / export / addLabels) into
      `DatasetCardComponent` / `DetectorCardComponent`.  Each handler
      is 1-15 lines of API + dialog glue, and pushing them in would
      require the card components to inject `DatasetsApiService`,
      `DetectorsApiService`, `VtDialogService`, and
      `DatasetStateService` — coupling presentation to backend +
      dialog system.  Cards stay presentational; the dashboard keeps
      the thin glue methods.  Extracting them into yet another service
      was also rejected: they have a single caller (the dashboard
      template) so the move would just shuffle names without making
      the code more reusable.
    - **Column resize / drag-reorder forwarding stays in the
      component.** The two `@HostListener('document:mousemove')` /
      `('document:mouseup')` methods can't move to a service (host
      listeners need to live on a `@Component` / `@Directive`).  The
      actual logic already lives in `ManagedColumns` /
      `DashboardColumnsService`; the dashboard's contribution is two
      lines of "forward to both managers" — already minimal.
  **Bundle effect.** Initial bundle essentially unchanged
  (527.48 kB → 527.25 kB raw, 136.49 kB → 136.47 kB gzip — both
  modals are `@defer`-loaded so service code is in the dashboard's
  lazy chunk).  Lazy `dashboard-component` chunk grew from
  129.71 → 131.61 kB raw (22.57 → 22.82 kB gzip) because the service
  plumbing adds Angular DI overhead without yet enabling cross-
  component dedup.  Bundle size was not the goal of this checkpoint —
  the structural goal (clear ownership of modal state and polling, a
  smaller dashboard shell) is what shipped.
- **#3 Split `label-view.component.ts`**: extracted three focused
  units out of the coordinator component to drop it from 1196 → 1071
  lines.
  - **`PanelResizeDirective`** (`vtPanelResize`) folds the two
    near-identical 50-line divider drag handlers (left + right
    mousedown / mousemove / mouseup) into a single standalone
    directive bound to each divider in the template.  The directive
    runs mousemove outside the Angular zone (matching the previous
    inline behaviour) and emits a `widthChange` per move plus one
    `resizeEnd` on release; the component receives those events and
    handles side-specific concerns (auto-uncollapse on left, grid
    snap on release, save-to-settings) in two small handlers per
    side.  The four bound handler properties
    (`boundMouseMove` / `boundMouseUp` / `boundRightMouseMove` /
    `boundRightMouseUp`), the two `dragging` flags, the `NgZone`
    injection, and the four `document.removeEventListener` calls in
    `ngOnDestroy` all leave the parent.
  - **`LabelViewPanelStateService`** owns the six per-media-type
    preference dicts (`viewModeLeftDict`, `gridIconSizeLeftDict`,
    `focusModeLeftDict`, `focusModeRightDict`, `panelPxLeftDict`,
    `panelPxRightDict`) and the active `currentMediaType` pointer.
    The component reads getters off the service (`viewModeLeft`,
    `gridGoalWidthLeft`, `focusModeLeft`, `focusModeRight`) in the
    template, hands fresh settings blobs to
    `panelState.loadFromSettings()`, and calls
    `panelState.savePanelPx(side, px)` after a drag releases.  The
    `loadSettings()` body shrunk from a six-key cascade with
    duplicate type-guards down to a single `loadFromSettings` call
    plus an `applyPanelPx()` re-clamp.  The service is component-
    scoped (provided in the component's `providers`) so it resets
    cleanly between test instances.
  - **`buildMediaContextMenuItems(mediaType)`** factory replaces the
    ~30-line inline switch on `cropAble` inside
    `onMediaContextRequest`.  The factory lives next to the
    component so audio/image-specific menu items stay near the
    flow that consumes them.
  - **What we did NOT do** (deviations from the plan, noted so the
    next contributor doesn't try and back them out):
    - **`applyPanelPx` stays in the component.**  The plan suggested
      a "panel-px ↔ panel-pct conversion utility module", but the
      existing code already stores raw px (the `panel_pct_*`
      settings-key names are a misnomer from an earlier draft).
      The remaining `applyPanelPx` helper clamps stored widths
      against current layout bounds (which needs `LEFT_MIN` /
      `RIGHT_MIN` / `DIVIDER_TOTAL` / `CENTER_MIN` plus the live
      `leftWidth` / `rightWidth` / `autopilotCollapsed` state),
      so pulling it out would force the component to pass all that
      state in on every call — a wash.  The service owns the
      per-media-type dicts; the clamping math stays where the
      constants live.
    - **`MediaContextMenuComponent` itself was not rebuilt.**  The
      plan mentioned moving the configuration data "into a small
      per-flow factory", which is exactly what
      `buildMediaContextMenuItems` does.  No structural change to
      the rendering component was needed.
  **Bundle effect.** Initial bundle unchanged at 527.25 kB raw /
  136.46 kB gzip — label-view is a lazy-loaded route, so all gains
  land in its chunk.  Lazy `label-view-component` chunk dropped from
  ~58 kB → 55.72 kB raw (12.44 kB gzip) as the four divider-drag
  handlers and the inline dict-loading cascade were replaced with
  smaller delegations.  Structural goal — three focused files
  (directive, service, pure factory) instead of one 1200-line
  coordinator — is what shipped.
- **#4 Lazy-load icon SVGs**: split `IconComponent`'s 42 inline SVGs
  out of the eager bundle.  Shell now ships only the 5 icons the
  `DialogHostComponent` needs (`warning`, `x-circle`, `check`,
  `info`, plus the `file` fallback) plus the dynamic letter glyph;
  the other 37 SVGs live in `icon-svgs-extended.ts` which the
  component lazy-loads via dynamic `import()` the first time a
  non-shell icon is requested.  Each loaded SVG is sanitised and
  cached on a static `Map` shared across every `IconComponent`
  instance — sanitisation per icon happens once per process.
  Rendering switched from a giant `@switch` template with one
  `<svg>` branch per icon name to `[innerHTML]="svgHtml"` on a
  wrapper span sized via CSS variables; the inner SVG (with `viewBox
  0 0 24 24`) scales to the wrapper's `width` / `height`, replacing
  the previous per-SVG `[attr.width]` / `[attr.height]` bindings.
  No call-site changes — `<vt-icon [icon]>` / `[type]` / `[size]`
  inputs are byte-identical.
  **Bundle effect.** Initial bundle dropped from 537.45 kB → 525.73
  kB raw (139.53 → 137.54 kB gzip); a new `icon-svgs-extended` lazy
  chunk weighs 12.09 kB raw / 1.81 kB gzip and is referenced by
  every lazy chunk that uses non-shell icons (dashboard,
  label-view, all of the deferred modals).  First render of an
  extended icon waits one microtask for the dynamic import to
  resolve — visually invisible in practice because the lazy chunk
  the icon ships inside already gates on its own download.
- **#5 API services audit** (investigation only): documented that
  the four big hand-written API services (`detectors-api`,
  `datasets-api`, `sorting-api`, `medias-api`) already delegate
  almost every call to the `ng-openapi-gen` generated `fn/`
  modules — the "fold-into-generated-client" half of the original
  plan is effectively done. The remaining direct `this.http.*`
  calls are all justified (FormData uploads for plugin endpoints
  whose body shape isn't in the OpenAPI spec, plus one blob
  download). The real eager-bundle cost identified by the audit is
  that `activeContextGuard` → `ContextSwitchService` statically
  imports the full `DetectorsApiService` and `DatasetsApiService`
  even though `ContextSwitchService` only calls 4 endpoints across
  the two. A per-domain split of both services is the next move,
  but is left as a follow-up (see Open follow-ups) — the
  investigation is what shipped here, not the refactor.
- **#1 Checkpoint 4**: extracted `<vt-source-picker>` — the importer
  category-tab + sub-tab chrome plus the source-side widgets (demo
  media-type tabs + sortable demo table, server-folder typed-path
  input, local-folder/local-files dropzone with file-count display).
  Migrated `dataset-importer-modal` to consume it.
  The component is presentational: the parent owns all state and
  passes in precomputed `visibleImporterTabs` /
  `importersForActiveTab` lists; user actions surface as `@Output`
  events the parent handles (`(activeTabChange)="selectImporterTab($event)"`
  for top-level tab clicks, `(importerSelected)="selectImporter($event)"`
  for sub-tab clicks, `(demoSelected)`, `(sfPathApplied)`,
  `(lfFilesDropped)`).  Per-flow output config (Dataset Name input,
  `<vt-import-config>`, Include-subfolders checkbox,
  `<vt-import-advanced>`) projects through named content slots
  (`[demoExtras]`, `[sfBefore]`/`[sfAfter]`,
  `[lfBefore]`/`[lfAfter]`) so the visual order on every flow stays
  byte-identical to before.
  Picker chrome SCSS (the `display: flex` rules, the indent for
  `.local-folder-uploader` / `.server-folder-browser`, the `gap`
  values) moved out of the parent and into
  `source-picker.component.scss`.  Dead helpers removed from the
  parent: `getTabIcon`, `getTabText`, `statusBadgeClass`,
  `statusBadgeLabel`, `onDemoHeaderClick`.  Dead component imports
  removed: `IconComponent`, `DropZoneComponent` (both still used —
  inside the new sub-component).  Parent template dropped from 529 →
  362 lines; .ts from 1687 → 1648 lines.
  **Bundle effect.** The Add Dataset modal is `@defer`-loaded, so the
  initial bundle is unaffected (526.40 kB → 526.40 kB).  The lazy
  `dataset-importer-modal-component` chunk grew from 77.64 kB → 81.99
  kB raw (15.76 → 16.73 kB gzip) because the new component adds its
  own Angular plumbing without yet eliminating cross-modal
  duplication — Checkpoint 5 (migrating `new-detector-modal` to
  consume `<vt-source-picker>`) is where the real dedup land.

## Why

The Angular initial-bundle warning budget has been bumped twice in a
week (500 → 525 → 540 kB). Each bump was triggered by routine feature
work, not by genuinely heavy new functionality:

- `b22c2b37` (May 16, 500 → 525 kB): "cumulative weight of recent
  dashboard, modal, and OpenAPI work."
- `4bd4cc40` (May 22, 525 → 540 kB): "design token system replaces
  literals like 8px with var(--space-md) which is longer raw text but
  compresses identically."

The pattern says the budget keeps creeping because the codebase has
structural smells that turn ordinary changes into bundle growth:
duplicated giant components, an over-stuffed app shell, eagerly-loaded
helpers that most callers don't need, and a budget metric (raw bytes)
that punishes readable code.

The biggest single offender is
`frontend/src/app/components/dashboard/dataset-importer-modal/dataset-importer-modal.component.ts`
— **1814 lines / 68 KB raw**, ~206 methods, four near-identical inner
flows in one class. Its own section comment in `new-detector-modal`
admits the duplication ("Media picker (shared structure with Add
Dataset)"). Both modals are `@defer`-loaded so they don't sit in the
initial bundle, but the duplication forces parallel edits on every
"add an importer style" change and bloats two deferred chunks
simultaneously.

The eager bundle's worst offender is `IconComponent` (663 lines /
20 KB of inline SVGs), pulled into `main.js` via `DialogHostComponent`,
even though most icons are referenced only by deferred modals.

## What's in the initial bundle today

Routes are lazy (`loadComponent` for dashboard / label / find). Five
top-level modals are `@defer`-loaded inside `app.component.html`:
settings, achievements, keyboard-help, dataset-importer,
new-detector. So those are NOT in the 540 kB initial.

The 540 kB is:

- App shell: `app.component` + `DialogHostComponent` +
  `ToastContainerComponent` + `AchievementUnlockHostComponent` +
  `LoginComponent` + `ContextPulldownComponent` (505 lines / 18 KB,
  added since the last bump) + `IncompatiblePairExplainerComponent` +
  `IconComponent` (20 KB of inline SVGs) + `ModalComponent`.
- Services transitively pulled in, dominated by `detectors-api`
  (21 KB), `datasets-api` (17 KB), `vote-state` (15 KB),
  `sorting-api` (14 KB). The `activeContextGuard` (used by the
  label/find routes) drags in `DetectorsApiService` +
  `DatasetsApiService`.
- Global SCSS: `_variables.scss` (with three theme blocks) +
  `_components.scss` (415 lines / 9.7 KB) + `_picker-shared.scss` +
  `_data-table.scss`.
- Angular framework / runtime / polyfills.

The May 16→22 bump was driven mostly by *new eager code*: the
`context-pulldown` component (505 lines, new), `context-switch.service`
(326 lines, new), `folder-browser` (+416 lines), and growth across all
four big API services. The design-token sweep was a red herring — that
commit explicitly noted gzipped size was flat.

## The plan

### #1 — Shared media-source picker / config widgets (in progress)

Goal: collapse the four-flow god class
`dataset-importer-modal.component.ts` into a thin shell + small
sub-components, and stop duplicating the same picker chrome in
`new-detector-modal.component.ts`.

Two extracted sub-components, both standalone Angular components in
the same shape as the existing
`SourceSpecsPickerComponent`:

1. **`<vt-source-picker>`** — the importer-tab/subtab chrome plus the
   demo table, server-folder typed path, and local-folder / local-files
   dropzone affordances. Emits "user picked source X with args Y" via
   `@Output`. Both modals subscribe; each handles the event
   differently (dataset-importer runs the importer; new-detector
   materialises a single example file).

   Eliminates: the four-tab category chrome duplicated between the
   two modals, the demo `ManagedColumns` table (currently spelled out
   twice with the same storage key), the typed-path server-folder
   widget, and the local-file dropzone wrapper.

2. **`<vt-import-config>`** — "output media type + embedder + clipper
   + clipper params + source-specs" form, used by the three
   import-configuring flows (`form`, `lf`, `sf`) inside
   `dataset-importer-modal`. Two-way bound state. Owns the
   media-type/embedder/clipper loading APIs it needs.

   Eliminates: the `sf*` / `lf*` / `form*` parallel state blocks
   inside the importer modal. Each flow shell becomes a thin wrapper
   around `<vt-import-config>` plus the import-trigger button.

Order of work (reordered after reading the code: start with the more
contained extraction so the pattern is proved on one flow before
touching the cross-modal picker chrome):

1. **Checkpoint 1 (shipped, `fe2ef1e2`)**: Extracted
   `<vt-import-advanced>` — the Advanced ▾ block (Include media +
   Embedder + Clipper). Migrated all four call sites (`sf`, `lf`,
   `form`, `demo`) in `dataset-importer-modal` to use it.
2. **Checkpoint 2 (shipped, `d1d3e2d7`)**: Extracted
   `<vt-import-config>` — the output media-type select + detection
   hint chip. Migrated the `sf` and `lf` flows in
   `dataset-importer-modal` to use it. Scope stayed at just the
   media-type widget (not the optional collapse-with-Advanced) because
   the source widget and recursive checkbox sit between media-type and
   Advanced in the template — a wrap would have forced a UX re-order.
3. **Checkpoint 3 (shipped)**: Migrated the `form` flow's
   `media_type` select to `<vt-import-config>`. Special-cased
   `field_type === 'select' && field.key === 'media_type'` inside the
   field iteration so the child renders its own form-group + label
   (rather than nesting inside the generic form-group / label
   scaffold), with `detectionHint` left empty because the form flow
   has no auto-detection. The previously dead-coded "suppress the
   required-star for media_type" condition in the generic-label
   template is gone with the case.
4. **Checkpoint 4 (shipped)**: Extracted `<vt-source-picker>` (the
   importer-tab / sub-tab chrome + demo table + server-folder typed
   path + local dropzone) into a presentational component with named
   content-projection slots ([demoExtras], [sfBefore]/[sfAfter],
   [lfBefore]/[lfAfter]) for the parent's per-flow output config.
   Migrated `dataset-importer-modal` to consume it.
5. **Checkpoint 5 (shipped)**: Migrated `new-detector-modal` to
   consume `<vt-source-picker>`.  Source-picker grew the
   customization knobs new-detector needed
   (`alwaysShowSubtabBar`, `demoRowDisabledFn`/`demoRowTitleFn`,
   `sfApplyOnBlur`, `lfShowFieldLabel`/`lfShowFileCount`, dropzone
   label overrides, and the `[lfInfo]` content slot).  The bundler
   now extracts source-picker into its own shared lazy chunk
   referenced by both modals — see the bundle effect line in the
   "What shipped" entry above.

The demo flow inside `dataset-importer-modal` stays a separate path
(its picker layout is genuinely different from the import flows).

### #2 — Split `dashboard.component.ts` (shipped — see "What shipped")

Goal: shrink the page-shell god component (1419 lines, ~200 methods).
Section dividers in the file already document the boundaries:
column resize/drag-reorder, dataset selection, model selection,
dataset actions, model actions, **9 separate modal openers** (export,
add-labels, importer, new-model, combine-datasets, combine-detectors,
label-exporter, label-importer, dataset-stats), cancel, progress
polling, sorting, button state, loading-task helpers.

Concrete moves:

- All 9 modal-open booleans (`exportModalOpen`, `addLabelsModalOpen`,
  `statsModalOpen`, …) → consolidate into `NewThingFlowsService`
  (which already owns the importer and new-detector flows).
- Row-action handlers (delete, rename, load, unload) → push into
  `DatasetCardComponent` / `DetectorCardComponent` (they exist;
  they're currently mostly view-only).
- Column resize + drag-reorder → already lives in
  `DashboardColumnsService`. Audit and remove any local copies still
  inlined in dashboard.
- Progress polling → already lives in `ProgressEventsService`.
  Audit and remove any local subscription bookkeeping.

After: `DashboardComponent` is roughly a layout + selection +
button-wiring component, with each domain concern owned by its
service or card component.

### #3 — Split `label-view.component.ts` (shipped — see "What shipped")

Goal: shrink the coordinator component (1196 lines, ~50 fields,
~160 methods). Existing section dividers: divider drag (left),
right-divider drag, data loading, sort handlers, select mode,
inclusion, media selection, right-click context menu, indicators,
autopilot, re-sort prompt, panel-percentage helpers.

Concrete moves:

- Left + right divider drag → a `vtPanelResize` directive (one
  instance per divider). Today this is two near-identical 50-line
  blocks of mousedown/mousemove/mouseup bookkeeping inlined into
  the component.
- Per-media-type panel-state dicts (`viewModeLeftDict`,
  `gridIconSizeLeftDict`, `focusModeLeftDict`,
  `focusModeRightDict`, `panelPxLeftDict`, `panelPxRightDict`) →
  a small per-media-type state service.
- Right-click context menu items → already has a
  `MediaContextMenuComponent`; the configuration data inlined in
  label-view should move into a small per-flow factory.
- Panel-px ↔ panel-pct conversion → utility module.

After: `LabelViewComponent` is route-level wiring (data loading +
inclusion + autopilot coordination), with layout / interaction
concerns owned by directives and services.

### #4 — Lazy-load icon SVGs (shipped — see "What shipped")

### #5 — Audit API services vs auto-generated client

`detectors-api.service.ts` (435 lines / 21 KB),
`datasets-api.service.ts` (344 lines / 17 KB),
`vote-state.service.ts` (401 lines / 15 KB),
`sorting-api.service.ts` (278 lines / 14 KB) are each flat method
bags directly mirroring REST endpoints. `ng-openapi-gen` runs on
`prebuild` and ships its own copy of the request/response models and
operation wrappers under `src/app/generated/api-client/`.

#### Investigation findings

**Overlap with the generated client: already high.** Each hand-written
API service is already delegating almost every call to the generated
`fn/` modules:

| Service                         | Generated imports | Direct `this.http.*` calls |
|---------------------------------|-------------------|----------------------------|
| `detectors-api.service.ts`      | 87                | 2 (plugin upload only)     |
| `datasets-api.service.ts`       | 61                | 8 (FormData / blob only)   |
| `sorting-api.service.ts`        | 49                | 6 (FormData uploads)       |
| `medias-api.service.ts`         | 11                | 1 (FormData upload)        |
| `label-importers-api.service.ts`| 5                 | 4 (FormData uploads)       |

The remaining direct `http.*` calls are all justified — they hit
plugin endpoints whose body shape is plugin-defined (not in the
OpenAPI spec) or stream a blob download. So the "fold-into-generated-
client" half of the original plan is **already done**; there is no
meaningful overlap left to remove. The wrapper services still add
value: cleaner names (`list()` vs `listDetectors()`), `r.body`
unwrapping so callers don't see `StrictHttpResponse`, and type
narrowing where the generated types are loose (e.g. the
`ImporterInfo` cast on the importer-listing return).

**`vote-state.service.ts` is not in scope.** Despite its size it is
not an API service — it owns vote state with optimistic updates and
undo/redo bookkeeping, using `MediasApiService` and
`SortingApiService` for HTTP. Folding it into the generated client
isn't applicable.

**The real eager-bundle cost: per-domain splitting is justified.**
The eager bundle pulls in the full `DetectorsApiService` (435 lines)
and `DatasetsApiService` (344 lines) via this static chain:

```
app.routes.ts  →  activeContextGuard  →  ContextSwitchService
                                          →  DatasetsApiService
                                          →  DetectorsApiService
```

But `ContextSwitchService` only calls **4 endpoints** across both
services:

- `datasetsApi.loadRegistered(id)`
- `datasetsApi.cancelTask(id)`
- `detectorsApi.loadDetector(id)`
- `detectorsApi.cancelDetectorLoadingTask(id)`

The other ~80 methods (find, scoring, CRUD, registry, extractors,
localizers, pregen processors, labels, …) are only called from lazy
chunks (dashboard, label-view, find-view, the deferred modals), but
they all ship in the eager bundle today because Angular cannot
tree-shake methods off an injectable class.

#### Follow-up: per-domain split (not started, scope-only)

Split `DetectorsApiService` and `DatasetsApiService` along the
section boundaries already documented in the source:

- `DetectorsApiService` → `DetectorsCrudApiService` (list/create/
  get/delete/rename/labels/combine/examples) +
  `DetectorsRegistryApiService` (registry list/load/unload/rename/
  cancel/autorun/labelset-move/from-labelset) +
  `DetectorScoringApiService` (auto-detect, extract/auto-extract,
  localize/auto-localize) + `DetectorFindApiService` (find,
  find-label, find-check-labels, cancelFind) +
  `ProcessorsApiService` (autorun-extractors, autorun-localizers,
  pregen-processors).
- `DatasetsApiService` → `DatasetsCrudApiService` (importers
  listings, import, stage, clear, export, detect-media-type) +
  `DatasetsRegistryApiService` (registry list/load/unload/rename/
  stats/readers + cancel) + `DatasetsListingsApiService` (clippers,
  embedders, converters, media-types, demo categories/list) +
  `DatasetsUiApiService` (dashboard disk/RAM usage, browse-media-
  files, select-browsed-file).

`ContextSwitchService` then imports only the two registry slices,
keeping CRUD/scoring/find/listings/UI out of the eager bundle.

The split is mostly mechanical (no logic change, no API change to
the generated client) but touches **all consumers** of the two
services — ~10 components and a handful of other services for
detectors-api, similar fan-out for datasets-api. Defer until after
#6 lands so the gzip-budget reading isn't disturbed by an unrelated
refactor.

Outcome: investigation report shipped (this section). No refactor
required for the "fold-into-generated-client" angle. The per-domain
split is a scoped follow-up.

### #6 — Switch budget metric to compressed size

The current Angular budget (`maximumWarning: 540kB`) is raw,
uncompressed bytes. This actively punishes readable code: the design
token sweep replaced literals like `8px` with `var(--space-md)` —
gzipped size unchanged, raw bytes up. Combined with #1–#4, the budget
metric should be flipped to a compressed size so cleanup work isn't
fighting the tooling.

Concrete moves:

- Confirm whether `@angular-devkit/build-angular` exposes a budget
  type for compressed size in the installed version (Angular 19.x).
  If not, document this as a "wait for upstream" item.
- If yes, switch the `maximumWarning` / `maximumError` budget type,
  re-baseline against the current gzipped initial bundle (~136 kB as
  noted in `4bd4cc40`), and add comfortable headroom.
- This is a one-line `angular.json` change once the upstream support
  is confirmed.

## What success looks like

After #1–#4 land, the initial bundle should drop comfortably back
under 500 kB on the same gzip metric we're using now, with the
biggest god-classes broken into focused sub-components and shared
widgets. After #6, future bumps will only happen when real new
functionality is added.

## Open follow-ups

- **#5 follow-up: per-domain split of `DetectorsApiService` /
  `DatasetsApiService`** to keep CRUD / scoring / find / listings /
  UI out of the eager bundle. Mechanical refactor — no logic change —
  but touches every consumer (~10 components per service). Defer
  until #6 lands so a gzip-budget reading isn't disturbed by an
  unrelated refactor. Scope and rationale documented inline under
  "#5 — Audit API services vs auto-generated client".

- **Roll the warning budget back to 525 kB** (or lower) now that
  #1–#4 are in. Current initial total is 525.73 kB raw / 137.54 kB
  gzip against the 540 kB warning threshold; the previous 525 kB
  threshold was bumped on `4bd4cc40` and the gains from #1
  (Checkpoints 1-5) plus #4 mean the bump is no longer needed.
  Requires user approval (CLAUDE.md says budget bumps — including
  reductions that could break future PRs — are user-decisions).
  Combine with #6 (flip to gzip metric) if that lands first.
