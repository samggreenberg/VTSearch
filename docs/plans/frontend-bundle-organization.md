# Frontend bundle organization

Status: **#1 in progress.** The other items are scoped but not started.

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

1. **Checkpoint 1**: Extract `<vt-import-config>` (media-type +
   detection hint + Advanced section with source-specs, embedder,
   clipper). Migrate the `sf` flow to use it. The dataset name and
   folder-path widget stay in the parent — only the configuration
   form is extracted.
2. **Checkpoint 2**: Migrate the `lf` flow to `<vt-import-config>`.
3. **Checkpoint 3**: Migrate the `form` flow to `<vt-import-config>`.
4. **Checkpoint 4**: Extract `<vt-source-picker>` (the importer-tab /
   sub-tab chrome + demo table + server-folder typed path + local
   dropzone). Migrate `dataset-importer-modal` to consume it.
5. **Checkpoint 5**: Migrate `new-detector-modal` to consume
   `<vt-source-picker>`.

The demo flow inside `dataset-importer-modal` stays a separate path
(its picker layout is genuinely different from the import flows).

### #2 — Split `dashboard.component.ts`

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

### #3 — Split `label-view.component.ts`

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

### #4 — Lazy-load icon SVGs

`IconComponent` (663 lines / 20 KB) ships every one of its 42 inline
SVGs into `main.js` because `DialogHostComponent` imports it. Most
icons are only used inside deferred modals.

Concrete moves:

- Split the SVG table out into a separate data module
  `icon-svgs.ts` that exports `{ name: string }`.
- Partition into "shell icons" (the 5-10 the shell actually needs)
  vs "modal icons" (the rest). The shell tier imports its set
  statically; modal icons are looked up via a dynamic import on
  first use, or partitioned into a separate `IconComponent` variant
  that the modals import.

After: ~15 KB of SVG markup moves out of the initial bundle into
the deferred chunks that actually use it.

### #5 — Audit API services vs auto-generated client

`detectors-api.service.ts` (435 lines / 21 KB),
`datasets-api.service.ts` (338 lines / 17 KB),
`vote-state.service.ts` (401 lines / 15 KB),
`sorting-api.service.ts` (278 lines / 14 KB) are each flat method
bags directly mirroring REST endpoints. `ng-openapi-gen` runs on
`prebuild` and ships its own copy of the request/response models and
operation wrappers under `src/app/generated/api-client/`.

Investigation:

- Survey overlap: how many endpoints in the four hand-written
  services already exist in the generated client?
- For the overlapping ones, can the hand-written wrappers delegate to
  the generated functions, removing the duplicate model imports?
- If `detectors-api` is irreducibly big, split per-concern
  (`detectors-crud`, `detectors-train`, `detectors-find`,
  `detectors-labels`) so consumers only drag in the slice they need.

Outcome: investigation report inline below before any refactor; then
follow-up either as a fold-into-generated-client task or a per-domain
split task.

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

(none yet — populated as items ship)
