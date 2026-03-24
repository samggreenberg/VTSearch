# VTSearch

Media explorer web app for browsing/voting on audio, images, text, video, or documents. Semantic sorting (LAION-CLAP, CLIP, X-CLIP, E5 embeddings) and learned sorting (neural net trained on votes). Flask + Angular + PyTorch.

## Branch Policy (CRITICAL)

- **All pull requests MUST target `dev`**, never `main` or `master`.
- **Claude must NEVER open a PR that merges into `main` or `master`.** The `main`/`master` branch is protected and only updated by human maintainers.
- When creating a PR, always use `--base dev` (e.g., `gh pr create --base dev ...` or the equivalent MCP tool parameter).
- If your feature branch was forked from `master`/`main` instead of `dev`, rebase or merge onto `dev` before opening a PR.

## Commands
- **Run tests (CPU, fast)**: `./run-tests.sh` (also checks frontend TypeScript build)
- **Run tests by group**: `./run-tests.sh core`, `./run-tests.sh sorting`, `./run-tests.sh api` (see Test Groups below; `core` includes frontend build check)
- **Run multiple groups**: `./run-tests.sh core sorting api`
- **Run tests with extra args**: `./run-tests.sh core -- -x --tb=long` (args after `--` go to pytest)
- **Run tests (CPU, full)**: `bash .claude/hooks/ensure-test-deps.sh && python -m pytest tests/ -q --tb=short -m 'not gpu'`
- **Run slow CLI subprocess tests only**: `python -m pytest tests/ -q --tb=short -m slow`
- **Run GPU tests**: `python -m pytest tests/test_gpu.py -q --tb=short -m gpu` (requires CUDA GPU; downloads models on first run)
- **Run all tests (CPU + GPU)**: `python -m pytest tests/ -q --tb=short -m ''`
- **Start app**: `bash .claude/hooks/ensure-test-deps.sh && python app.py` (or `python app.py --local` for dev)
- **CLI autodetect**: `bash .claude/hooks/ensure-test-deps.sh && python app.py --autodetect --dataset <file.pkl> --settings <settings.json>`
- **CLI autodetect + exporter**: `bash .claude/hooks/ensure-test-deps.sh && python app.py --autodetect --dataset <file.pkl> --settings <settings.json> --exporter server_json_file --filepath results.json`
- **CLI autodetect + importer**: `bash .claude/hooks/ensure-test-deps.sh && python app.py --autodetect --importer folder --path /data/sounds --media-type sounds --settings <settings.json>`
- **Install deps (CPU)**: `pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[cpu,dev]"`
- **Install deps (GPU)**: `bash install-gpu.sh` (or `bash install-gpu.sh cu121` for CUDA 12.1)
- **Build frontend**: `cd frontend && npm install && npm run build:prod` (builds Angular app to `static/`)
- **Frontend dev server**: `cd frontend && npm start` (proxies `/api/*` to Flask at localhost:5000)
- **Frontend tests**: `cd frontend && ng test --watch=false` (requires Chrome/Chromium; see Environment Notes below)
- **Lint**: `ruff check .`
- **Format**: `ruff format .`

## Architecture
- `app.py` — Flask entry point, registers blueprints, startup logic, CLI argument parsing, per-request user context via `before_request` middleware
- `vtsearch/auth/` — Authentication: `LoginProvider` ABC, `DefaultLoginProvider` (single-user, no-op), `get_current_user()`, `get_user_data_dir()`, `set_login_provider()`
- `vtsearch/config.py` — Constants (CLAP_SAMPLE_RATE, paths, model IDs)
- `vtsearch/medias.py` — Test media generation and embedding cache management
- `vtsearch/cli.py` — CLI utilities: autodetect (load dataset + detectors from settings, run inference, export results)
- `vtsearch/settings.py` — Persistent settings (volume, inclusion, theme, enrich_descriptions, safe_thresholds, calibrate_count, calibration_fraction, audio_playing, swipe_animation, show_metadata, view_mode_left, view_mode_right, focus_mode_left, focus_mode_right, grid_icon_size_left, grid_icon_size_right, panel_pct_left, panel_pct_right, autoload_media_types, autoload_media_embedders, autorun_processors, autorun_detector_names, autopilot_enabled, hide_autopilot, autopilot_top_greens, autopilot_hard_reds, autopilot_resort_interval, autopilot_goal_diversity, saved_datasets_dir, detectors_dir, trainable_models_dir); auto-saves to `data/settings.json`
- `vtsearch/routes/` — Flask blueprints: `auth.py`, `main.py`, `medias.py`, `sorting.py`, `detectors.py` (with sub-modules `detectors_crud.py`, `detectors_scoring.py`, `detectors_training.py`, `detectors_helpers.py`), `datasets.py` (with sub-modules `datasets_loading.py`, `datasets_ui.py`), `exporters.py`, `label_importers.py`, `processor_importers.py`, `settings.py`, `trainable_models.py`; shared utilities in `helpers.py`
- `vtsearch/models/` — Embeddings, training, model loading, progress tracking, diversity tree
- `vtsearch/datasets/` — Dataset loading, downloading, ingestion, origin tracking, labelsets, splitting, importers (folder/pickle/http_zip/combine_datasets/demo); auto-discovered via `IMPORTER` sentinel. Note: the `http_zip` directory registers as `http_archive` (its API/CLI name). `sources/` sub-package provides the `MediaSource` abstraction for resolving media files from origins
- `vtsearch/eval/` — Evaluation framework: runner, metrics, visualisation, voting iterations
- `vtsearch/exporters/` — Results exporters (server_json_file/server_csv_file/email_smtp/webhook/gui); auto-discovered via `EXPORTER` sentinel
- `vtsearch/labels/importers/` — Label importers (server_json_file/server_csv_file); auto-discovered via `LABEL_IMPORTER` sentinel
- `vtsearch/processors/importers/` — Processor importers (server_detector_file); auto-discovered via `PROCESSOR_IMPORTER` sentinel
- `vtsearch/media/` — Media type plugins: audio, image, text, video, document
- `vtsearch/converters/` — Media converters: document→image, document→text, video→audio, video→image
- `vtsearch/utils/` — Global state (`DatasetContext`, proxy dicts for `medias`/votes, multi-dataset context store), progress utilities
- `frontend/` — Angular SPA source (components, services, SCSS); builds to `static/` via `npm run build:prod`
- `static/` — Angular build output (index.html, main.js, polyfills.js, styles.css) and assets (favicons, logo.svg, logo.png)
- `docs/` — Extended docs (API.md, ARCHITECTURE.md, CLI.md, DEPLOYMENT.md, EVAL.md, EXTENDING.md, HANDOFF.md, ML.md, SETUP.md, demos.md, old_io.md, plan-media-sources.md, design/cli-detector-converter.md)
- `tests/` — Test suite split by module:
  - `conftest.py` — Shared fixtures: `reset_state` (autouse, clears all mutable global state), `isolated_settings` (autouse, redirects settings to tmp_path), `client` (Flask test client)
  - `test_api_contracts.py` — API response shape verification: status codes, content types, required keys, error format consistency
  - `test_ssrf_validation.py` — SSRF URL validation: blocking private/internal network addresses, public URL allowlisting, integration with HTTP archive importer and webhook exporter
  - `test_path_validation.py` — Server file-path validation: path traversal prevention
  - `test_audio.py` — WAV generation
  - `test_medias.py` — Media init, listing, audio endpoint, MD5
  - `test_votes.py` — Voting and vote retrieval
  - `test_sorting.py` — Text sort, learned sort, example sort, train_and_score
  - `test_labels.py` — Label export/import (via /api/labels/export and /api/labels/import)
  - `test_label_importers.py` — Label importer base class, registry, built-in server json_file/csv_file importers, GET /api/label-importers endpoint
  - `test_label_import_endpoint.py` — Label import POST endpoint, resolve_media_ids, find_missing_entries, next_media_id, missing element handling
  - `test_label_import_ingestion.py` — Label import ingestion: ingest-missing endpoint, _group_by_origin, _media_type_from_origin, _ingest_via_resolver
  - `test_inclusion.py` — Inclusion GET/POST
  - `test_detectors.py` — Detector export, detector sort, autorun detectors, auto-detect
  - `test_clippers.py` — MediaClipper ABC tests and concrete clipper implementations
  - `test_new_embedders.py` — Alternative embedder class properties and registration (SigLIP, CLAP Music, BGE) without downloading model weights
  - `test_resolver.py` — Media file resolution from origin trails: ResolvedLabels and resolve_file_from_origin
  - `test_cli_autodetect.py` — CLI autodetect: run_autodetect function, --autodetect flag, --exporter flag. Subprocess tests marked `slow` (~16s each, excluded from default run)
  - `test_datasets.py` — Dataset endpoints, startup state, importers
  - `test_dataset_split.py` — Train/test dataset splitting
  - `test_csv_webhook_exporters.py` — CSV and Webhook exporter metadata, CLI args, export logic
  - `test_dashboard.py` — Dashboard API endpoint tests
  - `test_exporters.py` — Results exporter base classes, registry, built-in exporters, API routes
  - `test_importers.py` — Importer base class, HTTP archive/folder importer metadata, archive extraction
  - `test_extractors.py` — Image class extractor
  - `test_processors.py` — Media processor tests
  - `test_processor_importers.py` — Processor importer base class, registry, server_detector_file importer, API routes
  - `test_origin_labelset.py` — Origin class, LabeledElement, LabelSet, build_origin(), label export/import with origins, integration
  - `test_combine_datasets.py` — Combine-datasets importer: metadata, dedup, media type validation, CLI, API routes
  - `test_corrections_export.py` — Corrections tracking: _find_initial_labels state, is_correction annotation on label export, label_filter=corrections filtering
  - `test_creation_info.py` — Legacy creation_info handling in pickle datasets
  - `test_parallel_loading.py` — Parallel dataset loading: LoadingTasksTracker, thread-local progress, per-task cancel, loading-tasks API endpoints, build_diversity_tree_for_context
  - `test_pickle_safety.py` — Restricted pickle unpickler: RCE prevention while allowing legitimate VTSearch dataset pickles
  - `test_duplicates.py` — Duplicate-content collapsing: collapse_duplicates function, dupe counting, label export/import integration
  - `test_enrich_descriptions.py` — Enriched text-sort description embedding
  - `test_error_recovery.py` — Error handling and edge cases: invalid requests, missing fields, type mismatches, empty state, nonexistent resources
  - `test_eval.py` — Evaluation framework runner and metrics
  - `test_eval_visualize.py` — Evaluation visualisation chart generation
  - `test_eval_voting_iterations.py` — Voting iterations evaluation
  - `test_safe_thresholds.py` — Safe threshold blending
  - `test_settings.py` — Settings persistence (volume, inclusion, theme, swipe_animation, show_metadata, view_mode_left, view_mode_right, focus_mode_left, focus_mode_right, hide_autopilot, autopilot_top_greens, autopilot_hard_reds, autopilot_resort_interval, autorun processors, autorun_detector_names)
  - `test_thin_loading.py` — Thin (lazy) dataset loading mode for CLI
  - `test_chunked_loading.py` — Chunked (piecewise) dataset loading: folder/pickle chunked loaders, importer run_chunked interface, merge helper
  - `test_integration.py` — End-to-end user workflow simulations: chained API calls, state interactions, response format consistency
  - `test_label_sorting.py` — Label sorting features: click-time tracking, learned-sort score storage, enriched /api/votes response
  - `test_diversity_tree.py` — DiversityTree: hierarchical k-means clustering, seen tracking, diversity level, next sample
  - `test_diversity_tree_integration.py` — DiversityTree integration with Flask app: build, rebuild, voting, diversity-level endpoint
  - `test_document_and_converters.py` — Document media type and media converters (document→image, document→text, video→audio, video→image)
  - `test_converter_selection.py` — ConverterChooser: converter registry, API, importer integration, run_converters_on_folder
  - `test_download_and_extract.py` — Generic _download_and_extract() helper: tar.gz, zip, nested archives
  - `test_tqdm_progress.py` — tqdm-based progress bar tracking
  - `test_ag_news_download.py` — AG News dataset download and load_demo_source integration
  - `test_bbc_news_download.py` — BBC News dataset download and load_demo_source integration
  - `test_export_options.py` — Export boolean options, negative_hits in CLI scoring, fill-from-sort
  - `test_frontend.py` — Frontend serving: Angular SPA entry point, static files (main.js, polyfills.js, styles.css), favicon variants, logo, legacy /ng/ redirect, content types
  - `test_gtzan_download.py` — GTZAN dataset download and load_demo_source integration
  - `test_image_sources_download.py` — Image dataset downloads: Oxford Flowers 102, Food-101, EuroSAT, Stanford Dogs, and load_demo_source integration
  - `test_imdb_download.py` — IMDB dataset download and load_demo_source integration
  - `test_load_sort_window.py` — Load Sort window endpoints: example-sort, server-media-files, detector server-files
  - `test_media_sources.py` — MediaSource abstraction: get_source_for_origin, LocalFolderSource, HTTP archive source
  - `test_multi_dataset.py` — Multi-dataset support: DatasetContext, proxy dicts/lists, context store, switching preserves votes/history/scores, activate/unload API endpoints, scalar state isolation, empty context fallback
  - `test_memory_errors.py` — Graceful MemoryError handling during dataset loading
  - `test_multi_user_dataset_access.py` — Multi-user dataset access control: readers list, access filtering, ownership checks, PUT readers endpoint
  - `test_multi_user_security.py` — LoginProvider ABC, DefaultLoginProvider, g.user middleware, created_by ownership, auth status endpoint, user data dir isolation
  - `test_pdf_import.py` — PDF-to-image import: render_pdf_pages conversion, folder importer PDF handling, origin tracking
  - `test_preload_progress.py` — Console progress output during embedding model preloading
  - `test_slow_integration.py` — Slow integration tests: chunked loading, detector scoring, export, label round-trips, settings persistence (marked slow)
  - `test_thread_safety.py` — Thread-safe global state operations: _state_lock verification, concurrent vote/click-time/label history access
  - `test_trainable_models.py` — Trainable models CRUD API: create, list, get, delete, rename, save labels, examples
  - `test_ucsf_documents_download.py` — UCSF Industry Documents demo dataset download and load_demo_source integration
  - `test_gpu.py` — GPU tests: training, cross-calibration, detectors, embedding models (CLAP/CLIP/X-CLIP/E5), CPU↔GPU equivalence, memory cleanup (skipped without CUDA)

## Test Groups

Tests are auto-grouped by area. Run a focused subset instead of the full suite:

| Group | Files | Description |
|-------|-------|-------------|
| `core` | audio, medias, votes, inclusion, settings, frontend | Basic app functionality |
| `api` | api_contracts, error_recovery, dashboard, path_validation, multi_user_security, multi_user_dataset_access, ssrf_validation | API contracts, error handling, security |
| `sorting` | sorting, label_sorting, safe_thresholds, enrich_descriptions, diversity_tree* | Sort algorithms and diversity |
| `datasets` | datasets, dataset_split, combine_datasets, creation_info, duplicates, origin_labelset, thin/chunked_loading, memory_errors, pickle_safety, media_sources, multi_dataset, parallel_loading | Dataset loading and management |
| `io` | exporters, csv_webhook_exporters, export_options, importers, label_importers, labels, processor_importers, pdf_import, corrections_export | Import/export |
| `models` | detectors, extractors, processors, trainable_models, clippers, eval*, resolver, new_embedders | ML models and evaluation |
| `downloads` | ag_news, bbc_news, gtzan, image_sources, imdb, ucsf, download_and_extract | Demo dataset downloads |
| `integration` | integration, slow_integration, thread_safety | End-to-end workflows |
| `cli` | cli_autodetect, load_sort_window, preload_progress, tqdm_progress | CLI and progress |
| `converters` | document_and_converters, converter_selection | Media converters |

**Recommended workflow**: Run `./run-tests.sh <group>` for the area you changed, then `./run-tests.sh` for the full suite.

## Test Markers
- **Default** (`./run-tests.sh` or `pytest tests/`): Runs fast CPU tests only (~35s). Excludes `gpu` and `slow` markers.
- **`slow`**: CLI subprocess tests that spawn `python app.py --autodetect` (each ~16s, total ~290s). Run with `-m slow` or include with `-m 'not gpu'`.
- **`gpu`**: CUDA-only tests. Run with `-m gpu`.
- **All tests**: Use `-m ''` to run everything.

## Reading Test Results (IMPORTANT)

The test suite prints a clear summary as its very last output:
- `ALL 1600 TESTS PASSED (3 skipped, total: 1603)` → all good
- `TESTS FAILED: 2 failed, 0 errors, 1598 passed, 3 skipped (total: 1603)` → 2 failures

**ONLY look at this final summary block** (bordered by `====` lines) to determine pass/fail. Many test names contain the word "error" (e.g., `test_memory_errors.py`, `TestErrorResponseFormat`). These test **error-handling behavior** — they are not failures.

**Do NOT scan test names or output for the word "error" to detect failures.** A line like:
```
tests/test_memory_errors.py::TestPickleMemoryError::test_importer_background_oom_reports_error PASSED
```
means the test **passed** — the word "error" is part of the test name, not an indication of failure.

## Test Workflow (IMPORTANT)

Testing can crash the session. To avoid losing work, follow this workflow:

1. **Commit and push before running tests.** Before running `pytest` or any test command, commit all current changes and push to your working branch. Use a message like `"WIP: pre-test checkpoint"` if the work isn't finalized yet.
2. **Run tests in the foreground (never in the background).** The test command has a slow startup phase: `ensure-test-deps.sh` installs dependencies (~1-2 min on first run), then `conftest.py` imports `app.py` and generates test media/embeddings before any tests execute. There may be no output for 1-3 minutes — this is normal. Do NOT run tests with `run_in_background` or assume output capture is broken because of the delay. Use a timeout of at least 300000ms (5 minutes).
3. **If tests fail and fixes are needed**, make the fixes, then commit and push again before re-running tests.
4. **Repeat** until tests pass. Every cycle of fixes should be committed and pushed before the next test run.

This ensures work is recoverable if the session crashes during a test run.

## Test Isolation (IMPORTANT)

All mutable global state is reset automatically before each test via two autouse fixtures in `conftest.py`:

1. **`reset_state`** — Clears all dataset contexts and creates a fresh `_test_default` context with the pre-generated test medias replayed into it. Also clears:
   - `autorun_detectors`, `autorun_extractors`, `autorun_localizers` (global state)
   - Progress cache and progress trackers
   - Login provider and dataset/model registries

2. **`isolated_settings`** — Redirects `SETTINGS_PATH` to a per-test temp file so settings writes never touch `data/settings.json`. Yields the temp path for tests that need to inspect the file.

**When writing new tests:**
- Do NOT add per-file or per-class autouse fixtures to clear autorun state, reset settings, or reset votes — `conftest.py` handles all of this automatically.
- Do NOT add inline `.pop()` or `.clear()` cleanup at the end of tests — the conftest fixtures run before each test regardless of whether the previous test passed or failed.
- If a test needs to temporarily empty `medias`, use the save/restore pattern with try/finally (since `medias` is intentionally NOT reset between tests to avoid expensive re-generation):
  ```python
  saved = dict(medias)
  medias.clear()
  try:
      # ... test logic ...
  finally:
      medias.update(saved)
  ```
- If a test needs to read the settings file path (e.g. to verify persistence), use `isolated_settings` as a parameter: `def test_foo(self, isolated_settings): ...`

## Avoiding Flaky Tests (IMPORTANT)

When writing new tests, avoid these two common sources of flakiness:

### 1. Always seed random number generators
Never call `np.random.randn()`, `np.random.rand()`, `torch.randn()`, or similar without a fixed seed. Random embeddings feed into neural net training and sorting, where different values cause non-deterministic convergence — making assertions pass or fail depending on the random draw.

**Do this:**
```python
rng = np.random.default_rng(42)
fake_embeddings = rng.standard_normal((n, dim)).astype(np.float32)
```

**Not this:**
```python
fake_embeddings = np.random.randn(n, dim).astype(np.float32)  # FLAKY — unseeded
```

### 2. Never use `time.sleep()` for thread synchronization
`time.sleep(0.2)` to "wait for a thread to start" is unreliable on loaded machines. Use `threading.Event` for deterministic synchronization, and set generous polling timeouts.

**Do this:**
```python
started = threading.Event()
def target():
    started.set()
    # ... work ...
thread = threading.Thread(target=target)
thread.start()
started.wait(timeout=5)
```

**Not this:**
```python
thread.start()
time.sleep(0.2)  # FLAKY — may not be enough on a loaded machine
```

### 3. Never use bounded loops to simulate "cancellable" or "interruptible" work
A `for i in range(100): sleep(0.05)` loop finishes in 5 seconds — but on a loaded machine the code that's supposed to interrupt it (e.g. setting a cancel flag) can take longer than 5 seconds to run. If the loop completes before the interrupt arrives, the test follows the wrong code path and fails.

**Do this:**
```python
def slow_load():
    started.set()
    while True:                            # exits ONLY via CancelledError
        dataset_progress.check_cancelled()
        time.sleep(0.05)
```

**Not this:**
```python
def slow_load():
    started.set()
    for i in range(100):                   # FLAKY — can finish before cancel arrives
        dataset_progress.check_cancelled()
        time.sleep(0.05)
```

## Environment Notes (Claude Code on the web)
- **No Chrome/Chromium available.** The cloud container (Ubuntu 24.04) does not have Chrome or Chromium installed, and they cannot be installed (`chromium` is snap-only on 24.04, snap is unavailable in containers, and Google's download servers are unreachable). Frontend Karma tests (`ng test`) will fail. Do NOT spend time trying to install Chrome/Chromium — it won't work. The Python backend tests (`./run-tests.sh`) work fine without a browser.

## Key Details
- **Multi-dataset support**: Multiple datasets can be loaded in memory simultaneously. Per-dataset state is bundled in `DatasetContext` objects (`vtsearch/utils/state_core.py`). The module-level names `medias`, `good_votes`, `bad_votes`, etc. are **proxy objects** (`_ProxyDict`/`_ProxyList`) that delegate to the active context. Switching the active dataset via `set_active_dataset_id()` is instant — no re-embedding. Key functions: `register_context()`, `unregister_context()`, `get_context()`, `list_loaded_dataset_ids()`. Global (non-per-dataset) state: `autorun_detectors`, `autorun_extractors`, `autorun_localizers`. API: `POST /api/datasets/registry/<id>/activate` (instant switch), `POST /api/datasets/registry/<id>/unload` (free RAM). Registry tracks `_loaded_ids` (set) and `_loaded_id` (active).
- Per-dataset state in `DatasetContext`: `medias`, `good_votes`, `bad_votes`, `label_history`, `vote_click_times`, `last_learned_scores`, `textsort_suggestions`, `click_counter`, `diversity_tree`, `find_initial_labels`, `inclusion`, `dataset_display_name`
- All mutable state protected by `_state_lock` (RLock) in `vtsearch/utils/state_core.py`
- Votes are `dict[int, None]` (not sets) — use `votes[id] = None` syntax
- Persistent settings live in `vtsearch/settings.py` (auto-saves to `data/settings.json`): volume, inclusion, theme, enrich_descriptions, safe_thresholds, calibrate_count, calibration_fraction, audio_playing, swipe_animation, show_metadata, view_mode_left, view_mode_right, focus_mode_left, focus_mode_right, grid_icon_size_left, grid_icon_size_right, panel_pct_left, panel_pct_right, autoload_media_types, autoload_media_embedders, autorun_processors, autorun_detector_names, autopilot_enabled, hide_autopilot, autopilot_top_greens, autopilot_hard_reds, autopilot_resort_interval, autopilot_goal_diversity, saved_datasets_dir, detectors_dir, trainable_models_dir
- Each media item has `origin` (dict or None) and `origin_name` (str) for per-element provenance tracking
- `Origin` class in `vtsearch/datasets/origin.py`; `LabelSet`/`LabeledElement` in `vtsearch/datasets/labelset.py`
- Label export (`/api/labels/export`) returns a `LabelSet` with per-element origin info (superset of legacy format)
- `data/` dir created at runtime for embeddings, model cache, media files
- OMP_NUM_THREADS and MKL_NUM_THREADS set to 1 for memory optimization
- Linter/formatter: ruff (E402 ignored, line-length 120, target-version py310, see pyproject.toml)
