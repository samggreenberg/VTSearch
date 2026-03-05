# Angular Migration Plan

Migration from vanilla JS (app.js + dialogs.js + charts.js + results.js + index.html) to Angular.

## Current Frontend Inventory

| File | Lines | Role |
|------|-------|------|
| `index.html` | 650 | SPA shell: 3-panel layout, ~12 modals, all markup |
| `app.js` | 7,075 | All application logic, API calls, event handlers, autopilot state machine |
| `dialogs.js` | 118 | Custom alert/confirm/prompt replacing `window.alert` |
| `charts.js` | 338 | Chart.js wrappers (error-cost, stability, diversity) |
| `results.js` | 564 | Auto-detect results table rendering and copy/export UI |
| `styles.css` | 785 | Theming (dark/light/highviz via `data-theme`), layout, components |

**API surface:** ~90 REST endpoints under `/api/` (sorting, voting, datasets, detectors, exporters, importers, settings, trainable-models, labeling-progress, diversity-tree).

**External deps:** None (no jQuery, no framework). Chart.js is the only library (loaded from CDN or bundled).

---

## Migration Strategy: Side-by-Side with Angular Elements

The safest approach is to build the Angular app alongside the existing vanilla JS, then swap views one at a time. Flask continues to serve both during the transition.

### How it works

1. Angular CLI project lives in `frontend/` with its own `package.json`, `angular.json`, build scripts.
2. `ng build` outputs to `static/ng/` (or a configurable output path).
3. Flask serves Angular's `index.html` at `/` when migration is complete; during migration, individual Angular components are loaded as web components (Angular Elements) into the existing `index.html`.
4. Alternatively (simpler): each phase replaces the vanilla `index.html` entirely once that phase's components are ready, and the vanilla JS for those sections is removed.

**Recommended: Phase-by-phase full replacement** (Option 4 above). This avoids Angular Elements complexity and each phase produces a standalone testable Angular app.

---

## Phases

### Phase 0: Project Scaffolding (estimated: 1 session)

**Goal:** Angular project skeleton that builds, serves, and passes the existing `test_frontend.py` tests.

**Tasks:**
1. `ng new frontend --routing --style=scss --skip-git` inside the repo root.
2. Configure `angular.json` to output build artifacts to `static/ng/`.
3. Add a `build:prod` npm script that builds to `static/ng/`.
4. Update Flask's `main.py` to serve `static/ng/index.html` at `/` when the build exists (fall back to old `static/index.html` otherwise).
5. Create a minimal `AppComponent` that renders the same outer HTML shell (header, 3-panel layout) so `test_frontend.py` assertions pass:
   - `<!DOCTYPE html>`, `app.js` reference (or equivalent), `charts.js` reference
   - Elements: `media-list`, `center`, `vote-section`, `sort-mode`, `settings-modal`, `dashboard-view`
6. Proxy `/api/*` requests to Flask dev server (via `proxy.conf.json` for `ng serve`).

**How to test:**
```bash
cd frontend && npm install && npm run build:prod
cd .. && ./run-tests.sh core   # test_frontend.py must pass
```

---

### Phase 1: API Service Layer (estimated: 1 session)

**Goal:** A typed Angular `HttpClient` service layer covering all API endpoints, with unit tests.

**Tasks:**
1. Create `frontend/src/app/services/api.service.ts` with methods grouped by domain:
   - `MediasApi` — `getMedias()`, `getAudio(id)`, `getVideo(id)`, `getImage(id)`, `getParagraph(id)`, `getMedia(id)`, `vote(id, label)`
   - `SortingApi` — `sort(params)`, `learnedSort()`, `getVotes()`, `clearVotes()`, `getInclusion()`, `setInclusion(value)`, `exportLabels()`, `importLabels(data)`, `fillFromSort(params)`, `exampleSort(params)`, `labelFileSort(params)`, `getLabelingProgress()`, `getLabelingStatus()`, `getDiversityTreeNext()`
   - `DatasetsApi` — `getStatus()`, `getProgress()`, `getImporters()`, `getDemoList()`, `loadDemo(name)`, `loadFile(file)`, `loadFolder(params)`, `clearDataset()`, `getRegistry()`, etc.
   - `DetectorsApi` — `getAutorunDetectors()`, `createDetector(params)`, `deleteDetector(name)`, `renameDetector(name, newName)`, `exportDetector(name)`, `autoDetect(params)`, etc.
   - `SettingsApi` — `getSettings()`, `updateSettings(data)`, `getDefaults()`
   - `ExportersApi` — `getExporters()`, `runExport(params)`
   - `TrainableModelsApi` — full CRUD
2. Define TypeScript interfaces for all API response shapes (reference `test_api_contracts.py` for the expected shapes).
3. Write Jasmine/Karma unit tests using `HttpClientTestingModule` for each service method.

**How to test:**
```bash
cd frontend && ng test --watch=false
```

---

### Phase 2: Shared Components & Theming (estimated: 1 session)

**Goal:** Reusable Angular components and the theme system.

**Tasks:**
1. **Theme service** (`ThemeService`): manages `data-theme` attribute on `<html>`, persists via `SettingsApi`. Supports `dark`, `light`, `highviz`.
2. **Port `styles.css`** to SCSS, split into:
   - `_variables.scss` — CSS custom properties per theme
   - `_layout.scss` — 3-panel grid
   - `_components.scss` — buttons, modals, forms
3. **Modal component** (`<vt-modal>`): generic modal with open/close, title slot, body slot. Replaces all 12 inline modal `<div>`s.
4. **Dialog service** (`VtDialogService`): replaces `dialogs.js` — `alert()`, `confirm()`, `prompt()` returning `Observable` or `Promise`.
5. **Progress bar component** (`<vt-progress-bar>`): reusable, takes `value`, `max`, `indeterminate` inputs.

**How to test:**
```bash
cd frontend && ng test --watch=false
# Manual: ng serve, open browser, toggle themes, open/close modals
```

---

### Phase 3: Dashboard View (estimated: 1 session)

**Goal:** The dashboard (dataset grid + model grid + Label/Find actions) works in Angular.

**Tasks:**
1. `DashboardComponent` — dataset grid, model grid, action buttons (Label, Find).
2. `DatasetCardComponent` — displays dataset info, load/unload/delete actions.
3. `ModelCardComponent` — displays model info, delete/rename.
4. `DatasetImporterModalComponent` — importer picker + dynamic form.
5. Wire up `DatasetsApi` and `TrainableModelsApi` services.
6. Progress polling for dataset loading (reuse `<vt-progress-bar>`).
7. Route: `/dashboard` (or show conditionally like the vanilla JS does).

**How to test:**
```bash
cd frontend && ng test --watch=false
# Integration: ng serve, load the dashboard, create/delete datasets and models
# Backend: ./run-tests.sh api datasets models
```

---

### Phase 4: Left Panel — Sorting & Media List (estimated: 1-2 sessions)

**Goal:** The left panel (Manual tab + Autopilot tab) with media list and sort controls.

**Tasks:**
1. `LeftPanelComponent` with tab switching (Manual / Autopilot).
2. **Manual tab:**
   - `SortBarComponent` — text/learned/load radio, text input, status, progress.
   - `SelectModeComponent` — top/hard/new radio.
   - `InclusionSliderComponent` — range slider with API sync.
   - `ProgressIndicatorsComponent` — Smart/Stable/Diverse buttons.
   - `MediaListComponent` — scrollable list of media items.
   - `StripeOverviewComponent` — labeled-media minimap.
3. **Autopilot tab:**
   - `AutopilotPanelComponent` — steps display, examples section.
   - Port the autopilot state machine from `app.js` (this is the most complex piece — ~1000 lines).
4. `MediaItemComponent` — single media row with thumbnail, name, click handler.

**How to test:**
```bash
cd frontend && ng test --watch=false
# Integration: ng serve, load a dataset, try text sort, learned sort, vote from list
# Backend: ./run-tests.sh sorting core
```

---

### Phase 5: Center Panel — Media Viewer & Voting (estimated: 1 session)

**Goal:** The main content area that displays the selected media and handles voting (good/bad with keyboard and swipe).

**Tasks:**
1. `CenterPanelComponent` — media display area.
2. `AudioPlayerComponent` — `<audio>` with waveform, volume control.
3. `ImageViewerComponent` — `<img>` with zoom/pan.
4. `VideoPlayerComponent` — `<video>` element.
5. `TextViewerComponent` — paragraph display.
6. `DocumentViewerComponent` — document page display.
7. `VotingOverlayComponent` — swipe animation, good/bad visual feedback.
8. `KeyboardService` — arrow-key / swipe voting, keyboard shortcuts.
9. Wire up `MediasApi.vote()` and favicon changes on vote.

**How to test:**
```bash
cd frontend && ng test --watch=false
# Integration: ng serve, load dataset, navigate medias, vote with keys and clicks
# Backend: ./run-tests.sh core
```

---

### Phase 6: Right Panel — Labels & Vote History (estimated: 1 session)

**Goal:** Good/Bad label lists with sort, counts, and click-to-navigate.

**Tasks:**
1. `RightPanelComponent` — good list, bad list, counts.
2. `LabelListComponent` — sorted label entries with thumbnails.
3. `LabelSortComponent` — dropdown (newest, oldest, name, confidence, ID).
4. `DetectorContextBarComponent` — detector name + rename.
5. Wire up `SortingApi.getVotes()` for reactive updates.

**How to test:**
```bash
cd frontend && ng test --watch=false
# Integration: vote on several items, verify labels appear, change sort order
```

---

### Phase 7: Modals — Import/Export/Settings (estimated: 1-2 sessions)

**Goal:** All modal dialogs ported to Angular components.

**Tasks:**
1. `SettingsModalComponent` — appearance, sorting, calibration, autopilot, autoload, storage paths, import/export settings JSON.
2. `ProgressModalComponent` — Smart/Stable/Diverse charts (wraps Chart.js).
3. `AutoDetectResultsModalComponent` — results table, copy, export section (port `results.js`).
4. `AutoDetectProgressModalComponent` — progress bar during detection.
5. `DatasetImporterModalComponent` (from Phase 3, extend if needed).
6. `LabelImporterModalComponent` — importer picker + form.
7. `LabelExporterModalComponent` — exporter list.
8. `DetectorExportModalComponent` — detector export picker.
9. `ProcessorImporterModalComponent` — processor import picker + form.
10. `LoadSortModalComponent` — detector sort + example sort file pickers.
11. `ExamplesEditorModalComponent` — add/edit/remove examples.
12. `ChartsService` — wraps Chart.js (port `charts.js`), or switch to `ng2-charts`.

**How to test:**
```bash
cd frontend && ng test --watch=false
# Integration: open each modal, verify forms submit, settings persist
# Backend: ./run-tests.sh io settings
```

---

### Phase 8: State Management (estimated: 1 session)

**Goal:** Centralized reactive state replacing the global variables scattered in `app.js`.

**Tasks:**
1. Choose approach: NgRx (if you want full Redux) or simple BehaviorSubject services (lighter).
2. Create state services:
   - `MediaStateService` — current media list, selected media, medias metadata.
   - `VoteStateService` — good_votes, bad_votes, vote history.
   - `SortStateService` — current sort mode, sort results, learned scores.
   - `AutopilotStateService` — phase, step, pending transitions.
   - `DatasetStateService` — loaded dataset info, loading progress.
   - `DetectorStateService` — autorun detectors/extractors/localizers.
3. Refactor all components from Phases 3-7 to use these state services instead of local state.
4. This phase can also be done incrementally during Phases 3-7 — up to you.

**How to test:**
```bash
cd frontend && ng test --watch=false
# State services should have comprehensive unit tests
```

---

### Phase 9: Cleanup & Cutover (estimated: 1 session)

**Goal:** Remove vanilla JS, update Flask to serve only Angular, update all tests.

**Tasks:**
1. Remove `static/app.js`, `static/dialogs.js`, `static/charts.js`, `static/results.js`, `static/index.html`.
2. Keep `static/styles.css` only if Angular references it (otherwise delete).
3. Keep static assets: `favicon*.ico`, `logo.svg`, `logo.png`.
4. Update Flask `main.py` to always serve the Angular build output.
5. Update `test_frontend.py`:
   - Assertions about `app.js`, `charts.js` references change to Angular bundle references.
   - DOM structure assertions may need updating for Angular-generated markup.
   - Content integrity tests should verify Angular app boots correctly.
6. Run full test suite: `./run-tests.sh`
7. Update `CLAUDE.md` with new frontend build commands.

**How to test:**
```bash
cd frontend && npm run build:prod
cd .. && ./run-tests.sh   # full suite must pass
```

---

## Phase Dependency Graph

```
Phase 0 (Scaffold)
  └─> Phase 1 (API Services)
        └─> Phase 2 (Shared Components)
              ├─> Phase 3 (Dashboard)
              ├─> Phase 4 (Left Panel)
              ├─> Phase 5 (Center Panel)
              ├─> Phase 6 (Right Panel)
              └─> Phase 7 (Modals)
                    └─> Phase 8 (State Management) ← can overlap with 3-7
                          └─> Phase 9 (Cleanup)
```

Phases 3-7 are largely independent of each other and can be done in any order (or in parallel by different people).

---

## Testing Strategy per Phase

Each phase should be testable in isolation:

| Phase | Angular Tests | Flask Tests | Manual Check |
|-------|---------------|-------------|--------------|
| 0 | `ng build` succeeds | `test_frontend.py` passes | Page loads |
| 1 | Karma unit tests for all API methods | N/A | N/A |
| 2 | Component tests for modal, dialog, progress | N/A | Theme toggle works |
| 3 | Dashboard component tests | `./run-tests.sh datasets` | Create/load dataset |
| 4 | Sort bar, media list tests | `./run-tests.sh sorting core` | Text sort, navigate |
| 5 | Media viewer, voting tests | `./run-tests.sh core` | Vote with keys |
| 6 | Label list tests | N/A | Labels appear |
| 7 | Modal component tests | `./run-tests.sh io settings` | All modals open |
| 8 | State service tests | N/A | N/A |
| 9 | N/A | `./run-tests.sh` (full) | Full workflow |

---

## Key Risks & Decisions

1. **Autopilot state machine** (~1000 lines in `app.js`) is the hardest piece. Consider extracting it to a pure TypeScript class first, with unit tests, before wiring it into Angular components.

2. **Chart.js integration:** Use `ng2-charts` (Angular wrapper) or raw Chart.js. `ng2-charts` is simpler.

3. **Build integration:** During development, `ng serve` with proxy to Flask is the best DX. For production, `ng build` outputs to `static/ng/` and Flask serves it.

4. **No SSR needed:** This is a single-user local tool, so Angular's default client-side rendering is fine.

5. **Routing:** The app is essentially a single page with view toggling (dashboard vs. labeling). Angular Router can handle this, but keep it simple — maybe just two routes: `/dashboard` and `/label`.

---

## How to Kick Off Each Phase with Claude

For each phase, tell Claude:

> "Implement Phase N of the Angular migration plan in `docs/ANGULAR_MIGRATION_PLAN.md`. Work on the branch `claude/plan-angular-migration-CMLkD`. Run the specified tests before pushing."

Claude will read this plan, implement the phase, run tests, and push.
