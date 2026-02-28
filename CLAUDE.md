# VTSearch

Media explorer web app for browsing/voting on audio, images, text, or video. Semantic sorting (LAION-CLAP, CLIP, X-CLIP, E5 embeddings) and learned sorting (neural net trained on votes). Flask + vanilla JS + PyTorch.

## Commands
- **Run tests (CPU, fast)**: `bash .claude/hooks/ensure-test-deps.sh && python -m pytest tests/ -v`
- **Run tests (CPU, full)**: `bash .claude/hooks/ensure-test-deps.sh && python -m pytest tests/ -v -m 'not gpu'`
- **Run slow CLI subprocess tests only**: `python -m pytest tests/ -v -m slow`
- **Run GPU tests**: `python -m pytest tests/test_gpu.py -v -m gpu` (requires CUDA GPU; downloads models on first run)
- **Run all tests (CPU + GPU)**: `python -m pytest tests/ -v -m ''`
- **Start app**: `bash .claude/hooks/ensure-test-deps.sh && python app.py` (or `python app.py --local` for dev)
- **CLI autodetect**: `bash .claude/hooks/ensure-test-deps.sh && python app.py --autodetect --dataset <file.pkl> --settings <settings.json>`
- **CLI autodetect + exporter**: `bash .claude/hooks/ensure-test-deps.sh && python app.py --autodetect --dataset <file.pkl> --settings <settings.json> --exporter local_json_file --filepath results.json`
- **CLI autodetect + importer**: `bash .claude/hooks/ensure-test-deps.sh && python app.py --autodetect --importer folder --path /data/sounds --media-type sounds --settings <settings.json>`
- **Install deps**: `pip install -r requirements-cpu.txt` (or `requirements-gpu.txt`)
- **Lint**: `ruff check .`
- **Format**: `ruff format .`

## Architecture
- `app.py` — Flask entry point, registers blueprints, startup logic, CLI argument parsing
- `vtsearch/config.py` — Constants (SAMPLE_RATE, NUM_MEDIAS, paths, model IDs)
- `vtsearch/medias.py` — Test media generation and embedding cache management
- `vtsearch/cli.py` — CLI utilities: autodetect (load dataset + detectors from settings, run inference, export results)
- `vtsearch/settings.py` — Persistent settings (volume, inclusion, theme, enrich_descriptions, safe_thresholds, calibrate_count, calibration_fraction, swipe_animation, show_thumbnails_left, show_thumbnails_right, autoload_media_types, autorun_processors); auto-saves to `data/settings.json`
- `vtsearch/routes/` — Flask blueprints: `main.py`, `medias.py`, `sorting.py`, `detectors.py`, `datasets.py`, `exporters.py`, `label_importers.py`, `processor_importers.py`, `settings.py`
- `vtsearch/models/` — Embeddings, training, model loading, progress tracking, diversity tree
- `vtsearch/datasets/` — Dataset loading, downloading, ingestion, origin tracking, labelsets, splitting, importers (folder/pickle/http_zip/rss_feed/youtube_playlist/combine_datasets); auto-discovered via `IMPORTER` sentinel
- `vtsearch/eval/` — Evaluation framework: runner, metrics, visualisation, voting iterations
- `vtsearch/exporters/` — Results exporters (local_json_file/server_json_file/local_csv_file/server_csv_file/email_smtp/webhook/gui); auto-discovered via `EXPORTER` sentinel
- `vtsearch/labels/importers/` — Label importers (local_json_file/server_json_file/local_csv_file/server_csv_file); auto-discovered via `LABEL_IMPORTER` sentinel
- `vtsearch/processors/importers/` — Processor importers (detector_file/server_detector_file/label_file/csv_label_file); auto-discovered via `PROCESSOR_IMPORTER` sentinel
- `vtsearch/media/` — Media type plugins: audio, image, text, video
- `vtsearch/utils/` — Global state (`medias` dict, votes), progress utilities
- `static/` — Frontend (index.html, app.js, styles.css) and assets (favicons, logo.svg, logo.png)
- `docs/` — Extended docs (ARCHITECTURE.md, CLI.md, DEPLOYMENT.md, EVAL.md, EXTENDING.md, FEATURE_IDEAS.md, HANDOFF.md, ML.md, SETUP.md, demos.md)
- `tests/` — Test suite split by module:
  - `conftest.py` — Shared fixtures: `reset_state` (autouse, clears all mutable global state), `isolated_settings` (autouse, redirects settings to tmp_path), `client` (Flask test client)
  - `test_audio.py` — WAV generation
  - `test_medias.py` — Media init, listing, audio endpoint, MD5
  - `test_votes.py` — Voting and vote retrieval
  - `test_sorting.py` — Text sort, learned sort, example sort, train_and_score
  - `test_labels.py` — Label export/import (via /api/labels/export and /api/labels/import)
  - `test_label_importers.py` — Label importer base class, registry, local/server json_file/csv_file importers, API routes
  - `test_inclusion.py` — Inclusion GET/POST
  - `test_detectors.py` — Detector export, detector sort, favorites, auto-detect
  - `test_clippers.py` — MediaClipper ABC tests and concrete clipper implementations
  - `test_cli_autodetect.py` — CLI autodetect: run_autodetect function, --autodetect flag, --exporter flag. Subprocess tests marked `slow` (~16s each, excluded from default run)
  - `test_datasets.py` — Dataset endpoints, startup state, importers, archive extraction
  - `test_dataset_split.py` — Train/test dataset splitting
  - `test_rss_youtube_importers.py` — RSS feed and YouTube playlist importer metadata, CLI args, run logic
  - `test_csv_webhook_exporters.py` — CSV and Webhook exporter metadata, CLI args, export logic
  - `test_exporters.py` — Results exporter base classes, registry, built-in exporters, API routes
  - `test_importers.py` — Importer base class, HTTP archive/folder importer metadata, archive extraction
  - `test_extractors.py` — Image class extractor
  - `test_processors.py` — Media processor tests
  - `test_processor_importers.py` — Processor importer base class, registry, detector_file/label_file importers, API routes
  - `test_origin_labelset.py` — Origin class, LabeledElement, LabelSet, build_origin(), label export/import with origins, integration
  - `test_combine_datasets.py` — Combine-datasets importer: metadata, dedup, media type validation, CLI, API routes
  - `test_creation_info.py` — Legacy creation_info handling in pickle datasets
  - `test_duplicates.py` — Duplicate-content collapsing: collapse_duplicates function, dupe counting, label export/import integration
  - `test_enrich_descriptions.py` — Enriched text-sort description embedding
  - `test_eval.py` — Evaluation framework runner and metrics
  - `test_eval_visualize.py` — Evaluation visualisation chart generation
  - `test_eval_voting_iterations.py` — Voting iterations evaluation
  - `test_safe_thresholds.py` — Safe threshold blending
  - `test_settings.py` — Settings persistence (volume, inclusion, theme, swipe_animation, show_thumbnails_left, show_thumbnails_right, favorites)
  - `test_thin_loading.py` — Thin (lazy) dataset loading mode for CLI
  - `test_chunked_loading.py` — Chunked (piecewise) dataset loading: folder/pickle chunked loaders, importer run_chunked interface, merge helper
  - `test_integration.py` — End-to-end user workflow simulations: chained API calls, state interactions, response format consistency
  - `test_label_sorting.py` — Label sorting features: click-time tracking, learned-sort score storage, enriched /api/votes response
  - `test_diversity_tree.py` — DiversityTree: hierarchical k-means clustering, seen tracking, diversity level, next sample
  - `test_diversity_tree_integration.py` — DiversityTree integration with Flask app: build, rebuild, voting, diversity-level endpoint
  - `test_tqdm_progress.py` — tqdm-based progress bar tracking
  - `test_ag_news_download.py` — AG News dataset download and load_demo_source integration
  - `test_bbc_news_download.py` — BBC News dataset download and load_demo_source integration
  - `test_export_options.py` — Export boolean options, negative_hits in CLI scoring, fill-from-sort
  - `test_gtzan_download.py` — GTZAN dataset download and load_demo_source integration
  - `test_image_sources_download.py` — Image dataset downloads: Oxford Flowers 102, Food-101, EuroSAT, Stanford Dogs, and load_demo_source integration
  - `test_imdb_download.py` — IMDB dataset download and load_demo_source integration
  - `test_load_sort_window.py` — Load Sort window endpoints: example-sort, server-media-files, detector server-files
  - `test_memory_errors.py` — Graceful MemoryError handling during dataset loading
  - `test_pdf_import.py` — PDF-to-image import: render_pdf_pages conversion, folder importer PDF handling, origin tracking
  - `test_preload_progress.py` — Console progress output during embedding model preloading
  - `test_slow_integration.py` — Slow integration tests: chunked loading, detector scoring, export, label round-trips, settings persistence (marked slow)
  - `test_thread_safety.py` — Thread-safe global state operations: _state_lock verification, concurrent vote/click-time/label history access
  - `test_ucsf_documents_download.py` — UCSF Industry Documents demo dataset download and load_demo_source integration
  - `test_gpu.py` — GPU tests: training, cross-calibration, detectors, embedding models (CLAP/CLIP/X-CLIP/E5), CPU↔GPU equivalence, memory cleanup (skipped without CUDA)

## Test Markers
- **Default** (`pytest tests/ -v`): Runs fast CPU tests only (~35s). Excludes `gpu` and `slow` markers.
- **`slow`**: CLI subprocess tests that spawn `python app.py --autodetect` (each ~16s, total ~290s). Run with `-m slow` or include with `-m 'not gpu'`.
- **`gpu`**: CUDA-only tests. Run with `-m gpu`.
- **All tests**: Use `-m ''` to run everything.

## Test Workflow (IMPORTANT)

Testing can crash the session. To avoid losing work, follow this workflow:

1. **Commit and push before running tests.** Before running `pytest` or any test command, commit all current changes and push to your working branch. Use a message like `"WIP: pre-test checkpoint"` if the work isn't finalized yet.
2. **Run tests in the foreground (never in the background).** The test command has a slow startup phase: `ensure-test-deps.sh` installs dependencies (~1-2 min on first run), then `conftest.py` imports `app.py` and generates test media/embeddings before any tests execute. There may be no output for 1-3 minutes — this is normal. Do NOT run tests with `run_in_background` or assume output capture is broken because of the delay. Use a timeout of at least 300000ms (5 minutes).
3. **If tests fail and fixes are needed**, make the fixes, then commit and push again before re-running tests.
4. **Repeat** until tests pass. Every cycle of fixes should be committed and pushed before the next test run.

This ensures work is recoverable if the session crashes during a test run.

## Test Isolation (IMPORTANT)

All mutable global state is reset automatically before each test via two autouse fixtures in `conftest.py`:

1. **`reset_state`** — Clears all mutable state in `vtsearch/utils/state.py`:
   - `good_votes`, `bad_votes`, `label_history`, `textsort_suggestions`, `vote_click_times`, `last_learned_scores`
   - `autorun_detectors`, `favorite_extractors`, `favorite_localizers`
   - `_click_counter`, `inclusion`, `_diversity_tree`
   - Progress cache

2. **`isolated_settings`** — Redirects `SETTINGS_PATH` to a per-test temp file so settings writes never touch `data/settings.json`. Yields the temp path for tests that need to inspect the file.

**When writing new tests:**
- Do NOT add per-file or per-class autouse fixtures to clear favorites, reset settings, or reset votes — `conftest.py` handles all of this automatically.
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

## Key Details
- Global state lives in `vtsearch/utils/state.py`: `medias`, `good_votes`, `bad_votes`, `label_history`, `vote_click_times`, `last_learned_scores`, `inclusion`, `textsort_suggestions`, `autorun_detectors`, `favorite_extractors`, `favorite_localizers` are module-level dicts/lists
- Votes are `dict[int, None]` (not sets) — use `votes[id] = None` syntax
- Persistent settings live in `vtsearch/settings.py` (auto-saves to `data/settings.json`): volume, inclusion, theme, enrich_descriptions, safe_thresholds, calibrate_count, calibration_fraction, swipe_animation, show_thumbnails_left, show_thumbnails_right, autoload_media_types, autorun_processors
- Each media item has `origin` (dict or None) and `origin_name` (str) for per-element provenance tracking
- `Origin` class in `vtsearch/datasets/origin.py`; `LabelSet`/`LabeledElement` in `vtsearch/datasets/labelset.py`
- Label export (`/api/labels/export`) returns a `LabelSet` with per-element origin info (superset of legacy format)
- `data/` dir created at runtime for embeddings, model cache, media files
- OMP_NUM_THREADS and MKL_NUM_THREADS set to 1 for memory optimization
- Linter/formatter: ruff (E402 ignored, line-length 120, target-version py310, see pyproject.toml)
