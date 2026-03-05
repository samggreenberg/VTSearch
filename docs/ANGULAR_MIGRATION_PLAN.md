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

## Migration Strategy: Side-by-Side via `/ng/` Route

The Angular app is built and served alongside the vanilla JS app throughout the migration. Both versions are accessible at all times so you can compare behavior and test incrementally.

### How it works

1. Angular CLI project lives in `frontend/` with its own `package.json`, `angular.json`, build scripts.
2. `ng build` outputs to `static/ng/`.
3. Flask serves the **vanilla app at `/`** (unchanged) and the **Angular app at `/ng/`** throughout the migration.
4. During development, `ng serve` with a proxy config forwards `/api/*` to the Flask backend.
5. You can open both side-by-side: `http://localhost:5000/` (vanilla) and `http://localhost:5000/ng/` (Angular).
6. After all phases are complete, Phase 9 flips `/` to serve Angular and removes the vanilla files.

### Why this approach

- **Vanilla app stays untouched** until you're fully satisfied with the Angular version.
- **Every phase is manually testable** by visiting `/ng/` in your browser.
- **No risk of breaking the working app** during migration.
- **Easy rollback** — just keep using `/` if something isn't ready.

### Flask routing during migration

Add to `main.py`:
```python
@main_bp.route("/ng/")
@main_bp.route("/ng/<path:path>")
def serve_angular(path=""):
    ng_dir = os.path.join(app.static_folder, "ng")
    if path and os.path.exists(os.path.join(ng_dir, path)):
        return send_from_directory(ng_dir, path)
    return send_from_directory(ng_dir, "index.html")
```

Angular's `<base href="/ng/">` ensures all asset paths resolve correctly under the `/ng/` prefix.

---

## Phases

### Phase 0: Project Scaffolding (estimated: 1 session)

**Goal:** Angular project skeleton that builds, is served at `/ng/`, and shows a basic shell.

**Tasks:**
1. `ng new frontend --routing --style=scss --skip-git` inside the repo root.
2. Configure `angular.json`:
   - Output build artifacts to `static/ng/`.
   - Set `baseHref` to `/ng/`.
3. Add a `build:prod` npm script that builds to `static/ng/`.
4. Add the `/ng/` catch-all route to Flask's `main.py` (see above).
5. Create a minimal `AppComponent` that renders the same outer HTML shell (header, 3-panel layout) with placeholder content in each panel.
6. Proxy config (`proxy.conf.json`) so `ng serve` forwards `/api/*` to Flask.
7. Verify: vanilla app at `/` is completely unchanged.

**How to test yourself:**
```bash
cd frontend && npm install && npm run build:prod
cd .. && python app.py --local
# Open http://localhost:5000/    → vanilla app (unchanged)
# Open http://localhost:5000/ng/ → Angular shell with placeholder panels
```

**Automated tests:**
```bash
./run-tests.sh core   # test_frontend.py still passes (vanilla app untouched)
cd frontend && ng test --watch=false
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

**How to test yourself:**
This phase is behind-the-scenes (no visible UI change at `/ng/`), but you can verify the service layer works:
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

**How to test yourself:**
```bash
cd frontend && ng test --watch=false
# Manual: open http://localhost:5000/ng/, verify theming matches vanilla app
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
7. Route: `/ng/dashboard`.

**How to test yourself:**
```bash
cd frontend && npm run build:prod && cd ..
python app.py --local
# Open http://localhost:5000/ng/dashboard
# Try: add a dataset, load it, create a model, delete it
# Compare with http://localhost:5000/ (vanilla dashboard)
```

**Automated tests:**
```bash
cd frontend && ng test --watch=false
./run-tests.sh api datasets
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

**How to test yourself:**
```bash
cd frontend && npm run build:prod && cd ..
python app.py --local
# Open http://localhost:5000/ng/ — load a dataset, then:
# - Try text sort (type a query, see results reorder)
# - Switch to learned sort
# - Click media items in the list
# - Compare list ordering with vanilla app at http://localhost:5000/
```

**Automated tests:**
```bash
cd frontend && ng test --watch=false
./run-tests.sh sorting core
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

**How to test yourself:**
```bash
cd frontend && npm run build:prod && cd ..
python app.py --local
# Open http://localhost:5000/ng/ — load a dataset, then:
# - Click a media item → it should display in the center
# - Press right arrow → vote good (green flash)
# - Press left arrow → vote bad (red flash)
# - Try swipe gestures on mobile/trackpad
# - Verify audio/video playback works
# - Compare with vanilla app
```

**Automated tests:**
```bash
cd frontend && ng test --watch=false
./run-tests.sh core
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

**How to test yourself:**
```bash
cd frontend && npm run build:prod && cd ..
python app.py --local
# Open http://localhost:5000/ng/ — load dataset, vote on a few items, then:
# - Right panel should show Good and Bad lists with counts
# - Change sort dropdown → labels reorder
# - Click a label → center panel navigates to that media
# - Compare with vanilla app
```

**Automated tests:**
```bash
cd frontend && ng test --watch=false
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

**How to test yourself:**
```bash
cd frontend && npm run build:prod && cd ..
python app.py --local
# Open http://localhost:5000/ng/ then test each modal:
# - Burger menu → Settings → change theme, toggle options
# - Burger menu → Import Labels → pick importer
# - Burger menu → Export Labels → pick exporter
# - Burger menu → Export Detector → pick detector
# - Click Smart/Stable/Diverse indicators → progress charts
# - Compare each modal with vanilla app
```

**Automated tests:**
```bash
cd frontend && ng test --watch=false
./run-tests.sh io settings
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

**How to test yourself:**
This is a refactor — no visible behavior change. Verify the app still works:
```bash
cd frontend && npm run build:prod && cd ..
python app.py --local
# Open http://localhost:5000/ng/ and run through a full workflow:
# Load dataset → sort → vote → export → verify nothing regressed
```

**Automated tests:**
```bash
cd frontend && ng test --watch=false
```

---

### Phase 9: Cleanup & Cutover (estimated: 1 session)

**Goal:** Promote Angular to `/`, remove vanilla JS, update all tests.

**Tasks:**
1. Update Flask `main.py`: serve Angular build at `/` instead of `/ng/`. Remove the vanilla `index.html` route.
2. Remove `static/app.js`, `static/dialogs.js`, `static/charts.js`, `static/results.js`, `static/index.html`.
3. Keep `static/styles.css` only if Angular references it (otherwise delete).
4. Keep static assets: `favicon*.ico`, `logo.svg`, `logo.png`.
5. Update Angular's `baseHref` from `/ng/` to `/`.
6. Update `test_frontend.py`:
   - Assertions about `app.js`, `charts.js` references change to Angular bundle references.
   - DOM structure assertions may need updating for Angular-generated markup.
   - Content integrity tests should verify Angular app boots correctly.
7. Run full test suite: `./run-tests.sh`
8. Update `CLAUDE.md` with new frontend build commands.

**How to test yourself:**
```bash
cd frontend && npm run build:prod
cd .. && python app.py --local
# Open http://localhost:5000/ → should be the Angular app now
# http://localhost:5000/ng/ → can be removed or redirect to /
# Full regression test of all features
```

**Automated tests:**
```bash
./run-tests.sh   # full suite must pass
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

Each phase is testable at `/ng/` without touching the vanilla app at `/`:

| Phase | Angular Tests | Flask Tests | Manual Test at `/ng/` |
|-------|---------------|-------------|----------------------|
| 0 | `ng build` succeeds | `test_frontend.py` passes (vanilla unchanged) | Shell loads at `/ng/` |
| 1 | Karma unit tests for all API methods | N/A | No visible change |
| 2 | Component tests for modal, dialog, progress | N/A | Theme toggle works at `/ng/` |
| 3 | Dashboard component tests | `./run-tests.sh datasets` | Create/load dataset at `/ng/dashboard` |
| 4 | Sort bar, media list tests | `./run-tests.sh sorting core` | Text sort, navigate at `/ng/` |
| 5 | Media viewer, voting tests | `./run-tests.sh core` | Vote with keys at `/ng/` |
| 6 | Label list tests | N/A | Labels appear at `/ng/` |
| 7 | Modal component tests | `./run-tests.sh io settings` | All modals open at `/ng/` |
| 8 | State service tests | N/A | Full workflow at `/ng/` (regression) |
| 9 | N/A | `./run-tests.sh` (full) | Everything works at `/` |

---

## Key Risks & Decisions

1. **Autopilot state machine** (~1000 lines in `app.js`) is the hardest piece. Consider extracting it to a pure TypeScript class first, with unit tests, before wiring it into Angular components.

2. **Chart.js integration:** Use `ng2-charts` (Angular wrapper) or raw Chart.js. `ng2-charts` is simpler.

3. **Build integration:** During development, `ng serve` with proxy to Flask is the best DX. For production, `ng build` outputs to `static/ng/` and Flask serves it.

4. **No SSR needed:** This is a single-user local tool, so Angular's default client-side rendering is fine.

5. **Routing:** The app is essentially a single page with view toggling (dashboard vs. labeling). Angular Router can handle this, but keep it simple — maybe just two routes: `/ng/dashboard` and `/ng/label` (becomes `/dashboard` and `/label` after cutover).

6. **Shared backend state:** Both `/` and `/ng/` hit the same Flask backend and share the same `medias`, `votes`, etc. This means you can load a dataset in the vanilla app and immediately see it at `/ng/`, which is great for comparison testing.

---

## How to Kick Off Each Phase with Claude

For each phase, tell Claude:

> "Implement Phase N of the Angular migration plan in `docs/ANGULAR_MIGRATION_PLAN.md`. Work on the branch `claude/plan-angular-migration-CMLkD`. Run the specified tests before pushing."

Claude will read this plan, implement the phase, run tests, and push.
