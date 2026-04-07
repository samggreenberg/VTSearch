# Refactoring Plan

Structural improvements identified March 2026. Each section is a self-contained
task that can be implemented and merged independently. Ordered by impact and risk.

**Progress:** 5 of 14 items complete (Phase 1 done, Phase 2–5 mostly pending).

---

## Phase 1: Split Oversized Files

### 1.1 ~~Split `datasets/loader.py`~~ ✅ DONE

Completed. Pickle security moved to `datasets/pickle_security.py`, metadata
extraction to `datasets/metadata.py`, converter integration to
`converters/runner.py` (as `run_converters_on_folder`). `loader.py` is now
~1,200 lines (down from 1,766).

---

### 1.2 ~~Split `routes/sorting.py`~~ ✅ DONE

Completed. Label endpoints moved to `routes/labels.py`, server media files
to `routes/media_server.py`, evaluation endpoints to `routes/eval.py`.
`sorting.py` is now ~565 lines (down from 1,138).

---

### 1.3 ~~Extract business logic from `routes/trainable_models.py`~~ ✅ DONE

Completed. `_seed_good_votes_from_examples()` moved to
`models/media_seeding.py`, `_restore_labels_from_trainable_model()` to
`models/label_restoration.py`, `_apply_and_retrain()` to
`models/training_workflow.py`. `trainable_models.py` is now ~868 lines
(down from 1,194).

---

## Phase 2: Eliminate Duplication (all pending)

### 2.1 Consolidate the training pipeline ⬜

**Problem:** The pattern of collecting embeddings, validating splits, training a model, and computing a threshold is reimplemented in `detectors_training.py` (4 handlers), `sorting.py` (`learned_sort`, `label_file_sort`), and `trainable_models.py` (`_apply_and_retrain`).

**Steps:**
1. Create `models/training_pipeline.py` with a unified `train_from_labels(embeddings, labels, **opts)` function
2. Refactor each call site to use the shared pipeline
3. Keep model-specific configuration (snap, threshold mode, etc.) as parameters

**Files affected:** `routes/detectors_training.py`, `routes/sorting.py`, `routes/trainable_models.py`, `routes/detectors_helpers.py`

---

### 2.2 Consolidate legacy detector weight fallback ⬜

**Problem:** Identical legacy weight format handling in 3 locations.

**Steps:**
1. Create `models/weights_compat.py` with a `normalize_detector_weights(raw)` function
2. Replace the 3 inline implementations with calls to this function

**Files affected:** `routes/detectors_crud.py:365-385`, `cli.py:55-76`, `processors/importers/server_detector_file/__init__.py:110-135`

---

### 2.3 Deduplicate embedder boilerplate ⬜

**Problem:** All 7 embedders repeat the same `_load_models_impl()` ceremony. `_extract_tensor()` is copy-pasted between image and video embedders.

**Steps:**
1. Move `_extract_tensor()` to `media/base.py` (shared by image and video)
2. Create a helper `_embedder_load_setup(on_progress, message)` in `media/base.py` that does: null check, `ensure_torch_configured()`, `gc.collect()`, cache dir, progress callback
3. Each embedder calls the shared setup, then does only its model-specific loading

**Files affected:** `media/image/embedder.py`, `media/video/embedder.py`, `media/audio/embedder.py`, `media/text/embedder.py`, `media/image/embedder_siglip.py`, `media/audio/embedder_clap_music.py`, `media/text/embedder_bge.py`, `media/base.py`

---

### 2.4 Create shared route helpers for plugin lookup and JSON extraction ⬜

**Problem:** "Unknown plugin" error handling duplicated in 5+ routes with minor inconsistencies. `request.get_json(force=True, silent=True) or {}` appears 30+ times with variations.

**Steps:**
1. Add `get_plugin_or_404(get_fn, list_fn, name, type_label)` to `routes/helpers.py`
2. Add `get_json_safe()` (returns `{}` on missing/invalid) to `routes/helpers.py`
3. Replace inline implementations across all route files
4. Ensure `datasets.py` uses `extract_plugin_fields()` instead of reimplementing field extraction

**Files affected:** `routes/helpers.py`, `routes/exporters.py`, `routes/label_importers.py`, `routes/processor_importers.py`, `routes/detectors_training.py`, `routes/datasets.py`, `routes/trainable_models.py`

---

## Phase 3: Architectural Improvements (1 of 4 done)

### 3.1 Break the `models/` ↔ `datasets/` circular dependency ⬜

**Problem:** `models/resolver.py` does late imports from `datasets/sources/` and `datasets/importers/`. `datasets/ingest.py` imports from `models/resolver.py`.

**Steps:**
1. Define a `FileResolver` protocol/ABC in `models/resolver.py` (or a shared location)
2. Have `datasets/sources/` implement this protocol
3. Inject the resolver into `models/resolver.py` functions instead of hardcoding late imports
4. Remove the bidirectional dependency

**Files affected:** `models/resolver.py`, `datasets/sources/__init__.py`, `datasets/ingest.py`

---

### 3.2 ~~Extend `PluginRegistry` to converters and sources~~ ✅ DONE

Completed. Converters now use `PluginRegistry` with `CONVERTER` sentinel.
Media sources use `PluginRegistry` with `SOURCE` sentinel. Both are
auto-discovered like all other plugin families.

---

### 3.3 Reduce `settings.py` boilerplate (~839 lines) ⬜

**Problem:** ~70% of the file is copy-pasted accessor patterns for left/right per-media-type settings, each with legacy scalar coercion.

**Steps:**
1. Create a `_per_side_setting(key, default, coercer=None)` factory that generates `get_X_left()`, `get_X_right()`, `set_X_left()`, `set_X_right()` and the internal `_get_X_dict()`/`_set_X_dict()`
2. Replace the 4 repetitive blocks (view_mode, grid_icon_size, focus_mode, panel_pct) with factory calls
3. Centralize the legacy scalar-to-dict coercion logic in one place

**Estimated reduction:** ~889 lines to ~400 lines

**Files affected:** `settings.py`

---

### 3.4 Clean up `utils/state_core.py` proxy complexity ⬜

**Problem:** 35+ manually overridden proxy methods and hidden data dependencies.

**Steps:**
1. Document the proxy pattern with clear warnings about implicit context dependency
2. Consider adding a `with_context(dataset_id)` context manager for explicit, scoped context switching
3. Long-term: evaluate migrating to explicit context passing (breaking change, needs careful planning)

**Files affected:** `utils/state_core.py`, potentially all files that import from `utils/`

---

## Phase 4: Cleanup (all pending)

### 4.1 Move `_build_extractor`/`_build_localizer` out of `detectors_crud.py` ⬜

**Problem:** Private functions in a CRUD module are imported by `detectors_scoring.py` and `detectors.py`.

**Steps:**
1. Move these functions to `detectors_helpers.py`
2. Update imports

**Files affected:** `routes/detectors_crud.py`, `routes/detectors_scoring.py`, `routes/detectors.py`, `routes/detectors_helpers.py`

---

### 4.2 Review autoload media embedder settings functions ⬜

**Problem:** `get_autoload_media_embedders`, `set_autoload_media_embedders`, `toggle_autoload_media_embedder` in `settings.py` may have limited usage.

**Steps:**
1. Grep for callers of these functions
2. If no callers remain, delete them
3. If callers exist, evaluate whether they should use a different API

**Files affected:** `settings.py`, any callers

---

### 4.3 Relocate `medias.py` test infrastructure ⬜

**Problem:** `vtsearch/medias.py` (91 lines) generates test media and embeddings. It's only used by `conftest.py` and `app.py` demo mode.

**Steps:**
1. Move test-only code to `tests/test_media_factory.py` or similar
2. If `app.py` demo mode needs it, keep a minimal `init_demo_medias()` in production code and move the rest to tests

**Files affected:** `vtsearch/medias.py`, `tests/conftest.py`, `app.py`

---

## Phase 5: Frontend (all pending)

### 5.1 Split `dashboard.component.ts` (1,301 lines) ⬜

**Problem:** Manages dataset selection, model selection, loading tasks, column resizing, sorting state, and modal orchestration in one component.

**Steps:**
1. Extract table column resize logic into a `TableResizeDirective` or service
2. Extract dataset management into a `DashboardDatasetsComponent`
3. Extract model management into a `DashboardModelsComponent`
4. Keep `DashboardComponent` as a thin orchestrator

---

### 5.2 Split `label-view.component.ts` (810 lines) ⬜

**Problem:** Manages panel layout, resize, polling, keyboard shortcuts, and autopilot in one component.

**Steps:**
1. Extract panel resize/layout logic into a `PanelLayoutService`
2. Extract per-media-type state dicts (viewMode, gridIconSize, focusMode, panelPx) into a shared `PanelStateService`
3. Keep `LabelViewComponent` focused on orchestration

---

### 5.3 Extract `BaseImporterComponent` for modal reuse ⬜

**Problem:** `dataset-importer-modal`, `label-importer-modal`, and `processor-importer-modal` independently implement the same picker→form flow, form state, file selection, and submission logic.

**Steps:**
1. Create `BaseImporterComponent` or shared mixin with: view state, `selectedImporter`, `formValues`, `selectedFile`, `submitting`, `error`, and common methods
2. Each concrete modal extends the base with its specific API service and plugin type

---

### 5.4 Add barrel files ⬜

**Steps:**
1. Create `frontend/src/app/services/index.ts` re-exporting all services
2. Create barrel files for feature component folders
3. Update imports across components

---

## Notes

- Each section can be done as a separate PR targeting `dev`
- Run `./run-tests.sh` after each change to verify nothing breaks
- Frontend changes should also run `cd frontend && npm run build:prod` to verify the build
- Phases are roughly ordered by impact, but items within a phase are independent
