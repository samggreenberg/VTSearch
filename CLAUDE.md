# VTSearch

Media explorer web app for browsing/voting on audio, images, text, video, or documents. Semantic sorting (LAION-CLAP, SigLIP, X-CLIP, E5 embeddings) and learned sorting (neural net trained on votes). Flask + Angular + PyTorch.

## Branch Policy (CRITICAL)

- **Always base work on `dev`.** At the start of every session, before making any changes, run `git fetch origin --prune && git rebase origin/dev`. The harness cuts the working branch off `main` (the GitHub default), so this rebase is required to pick up work already merged to `dev`. The GitHub default stays `main` so new users land on the stable branch — `dev` is Claude's starting point, not the public default.
- **All pull requests MUST target `dev`**, never `main`.
- **Claude must NEVER open a PR that merges into `main`.** The `main` branch is protected and only updated by human maintainers.
- When creating a PR, always use `--base dev` (e.g., `gh pr create --base dev ...` or the equivalent MCP tool parameter).
- If your feature branch was forked from `main` instead of `dev`, rebase or merge onto `dev` before opening a PR.

## Git Fetch Hygiene

Before comparing branches (`git log a..b`, `git diff a...b`, etc.), always run `git fetch origin --prune` first. Do **not** trust `origin/<branch>` refs after a partial fetch like `git fetch origin main` — that only updates the branch you named, leaving other remote-tracking refs stale and producing misleading diffs.

## Auto-PR

When you're done with your changes, open a PR targeting `dev`. Do not ask — just create it. Always pass `base=dev` explicitly (the GitHub PR-creation URL printed by `git push` defaults to `main`).

## PR Activity Subscription (do not ask)

Never ask the user whether to subscribe to PR activity, and never call `subscribe_pr_activity`. The user does not want Claude to watch PRs or respond to review comments / CI. This overrides the default GitHub Integration instruction to offer PR subscription after creating a PR.

## Versioning (do NOT bump by hand)

`vtsearch.__version__` is the UTC timestamp of `HEAD`'s commit (ISO 8601, Z-terminated), computed from git at import time in `vtsearch/__init__.py`. There is no tracked version constant to bump — every commit on `dev` automatically becomes the new version, and parallel branches cannot collide on a hand-edited version line. Do not add a `VERSION` file, do not write a hand-bumped string into `vtsearch/__init__.py`, and do not include version bumps in feature PRs. For Docker images (where `.git` is excluded from the build context), the host passes `--build-arg VTSEARCH_VERSION=$(TZ=UTC git log -1 --format=%cd --date=format:%Y-%m-%dT%H:%M:%SZ HEAD)` and the Dockerfile bakes it into `vtsearch/_version.txt` (gitignored). If git is unavailable and the baked file is missing, the version falls back to `0.0.0-unknown`.

## Backwards Compatibility

Breaking backwards compatibility is acceptable — do not add shims, feature flags, legacy re-exports, or other compatibility layers to preserve old behavior. Just make the clean change. When a change does break backwards compatibility, mention it to the user so they're aware.

## Ask Questions (use the Question tool)

When you have a question for the user — to disambiguate requirements, choose between approaches, confirm scope, or surface a non-obvious tradeoff — **ask it**. Do not guess silently and hope the choice was right. A 10-second clarification beats a 10-minute wrong-direction implementation.

**Always ask via the `AskUserQuestion` tool when the question fits its shape** (a discrete choice with a small number of options). Do not leave dangling questions like "Want me to go with approach A or approach B?" at the end of a prose response — those are easy to miss and force the user to type out an answer that could have been a single click. The tool also captures the choice cleanly in the transcript.

Use plain prose questions only when the answer is genuinely open-ended (e.g. "What should this field be named?") and a multiple-choice list would be artificial.

## No Persisted Vectors or MLPs (CRITICAL)

**Embeddings and trained MLP weights are in-memory artifacts only.** Never serialize them to disk, to `data/settings.json`, to detector / detector JSON files, or to any other persistent store. Origins are the canonical persisted form: the system rederives `origin → file → embedding → MLP` on demand.

This rule applies to all detector- and model-related code:

- Trainable-model JSON files store `LabeledElement`s with origin info, never embeddings.
- Detector JSON files (legacy) store origins; the `weights` field on disk is treated as deprecated and must not be written by new code.
- In-memory caches are fine and encouraged: `DetectorContext.label_embeddings`, `DetectorContext.model`, etc. — they live for the lifetime of the process and are repopulated from origins on the next start.
- New features that cache vectors must use a process-scoped data structure (e.g. a field on `DetectorContext`), not a file or settings key.
- Embedder version drift is impossible by construction because every load resolves+re-embeds against the active embedder.

The single exception is **dataset pickle files**, which are by design a snapshot of media + their embeddings — they ARE the dataset, not a cache.

If a feature seems to require persisting a vector or MLP, push back: either re-derive on demand, or change the design.

## Fix All Errors (CRITICAL)

When you run a build, typecheck, linter, or test suite, **fix every error and failure you see — not only the ones you introduced**. Do not dismiss errors as "pre-existing", "unrelated to my change", or "not my fault" and move on. Do not announce them and ask the user to triage. The user does not want to scan your output for problems you decided to ignore.

This applies to:
- TypeScript errors from `tsc` / `npm run build:prod` (including in `*.spec.ts` files, even though specs do not currently run — they must still typecheck).
- Angular build warnings of any kind, including `anyComponentStyle` budget warnings (e.g. `▲ [WARNING] ... exceeded maximum budget`). `run-tests.sh` treats every `▲ [WARNING]` line from `build:prod` as a hard test failure, so do not just bump budgets to silence them — fix the underlying bloat (split the component, extract shared styles, or remove dead rules). Bumping a budget is only acceptable when the size is genuinely justified, and requires the user's explicit approval.
- Python test failures from `./run-tests.sh` and `pytest` runs.
- Linter errors from `ruff`.
- Any other diagnostics surfaced by tooling you invoke.

If a failure is genuinely outside the scope of the current task (e.g. a flaky network test, a failure in unrelated infrastructure you cannot reproduce), explicitly call it out in your end-of-turn summary with one sentence explaining why you did not fix it. The default is **fix it**; skipping requires justification.

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
- **CLI autodetect + importer**: `bash .claude/hooks/ensure-test-deps.sh && python app.py --autodetect --importer server_folder --path /data/sounds --media-type audio --settings <settings.json>`
- **Install deps (CPU)**: `bash install-cpu.sh`
- **Install deps (GPU)**: `bash install-gpu.sh` (or `bash install-gpu.sh cu121` for CUDA 12.1)
- **Build frontend**: `cd frontend && npm install && npm run build:prod` (builds Angular app to `static/`)
- **Frontend dev server**: `cd frontend && npm start` (proxies `/api/*` to Flask at localhost:5000)
- **Frontend audit**: `cd frontend && npm audit` (checks for known vulnerabilities in dependencies)
- **Lint**: `ruff check .`
- **Format**: `ruff format .`

## Architecture
- `app.py` — Flask entry point, registers blueprints, startup logic, CLI argument parsing, per-request user context via `before_request` middleware, per-request dataset/model context resolution from `X-Dataset-Id`/`X-Detector-Id` headers
- `vtsearch/auth/` — Authentication: `LoginProvider` ABC, `DefaultLoginProvider` (single-user, no-op), `get_current_user()`, `get_user_data_dir()`, `set_login_provider()`
- `vtsearch/config.py` — Constants (CLAP_SAMPLE_RATE, paths, model IDs)
- `tests/fixtures/medias.py` — Test media generation and embedding cache management (loaded by `tests/conftest.py`; not part of the runtime app)
- `vtsearch/cli.py` — CLI utilities: autodetect (load dataset + detectors from settings, run inference, export results)
- `vtsearch/settings.py` — Persistent settings (volume, inclusion, theme, enrich_descriptions, safe_thresholds, calibrate_count, calibration_fraction, audio_playing, swipe_animation, show_metadata, view_mode_left, view_mode_right, focus_mode_left, focus_mode_right, grid_icon_size_left, grid_icon_size_right, panel_pct_left, panel_pct_right, autoload_media_embedders, autorun_detectors, autopilot_enabled, hide_autopilot, autopilot_top_greens, autopilot_hard_reds, autopilot_resort_interval, autopilot_goal_diversity, saved_datasets_dir, detectors_dir, max_concurrent_dataset_downloads, max_concurrent_dataset_embeddings, settings_source); auto-saves to `data/settings.json`. When a `settings_source` is configured, every save also syncs to the source (with a `_syncing` guard to prevent circular re-export during import). At startup, `sync_from_settings_source()` auto-imports from the active source. Also contains `_apply_settings()` for applying a settings dict via `set_*` functions
- `vtsearch/routes/` — Flask blueprints: `auth.py`, `eval.py`, `file_browser.py`, `labels.py`, `media_server.py`, `main.py`, `medias.py`, `sorting.py`, `processors.py` (extractor/localizer/pregen-processor CRUD + execution; sub-modules `processors_crud.py` and `processors_scoring.py`), `detector_scoring.py` (`/api/auto-detect`, `/api/find-label` against detector labelsets), `detector_find.py` (multi-dataset Find), `datasets.py` (with sub-module `datasets_ui.py`), `datasets_registry.py` (dataset registry CRUD at `/api/datasets/registry/*`), `exporters.py`, `label_importers.py`, `settings.py`, `settings_io.py`, `sync_sources.py`, `detectors.py` (on-disk labelset+query store at `/api/detectors/*`), `detectors_registry.py` (in-memory model registry at `/api/detectors/registry/*`, autorun toggle); shared utilities in `helpers.py`. Note: dataset-load orchestration lives at `vtsearch/datasets/load_pipeline.py` (background-task helpers, ConcurrencyGate, clip fix-up).
- `vtsearch/models/` — Embeddings, training, model loading, progress tracking, diversity tree
- `vtsearch/datasets/` — Dataset loading, downloading, ingestion, origin tracking, labelsets, splitting. `loader.py` is the public façade that re-exports the actual loaders from sibling modules: `loader_folder.py` (folder loaders), `loader_pickle.py` (pickle loaders + sidecars + image embed), `loader_demo.py` (demo dataset loader). Importers live in `importers/` — `local_folder`/`local_files` (browser-side upload placeholders), `server_folder`/`server_files` (server filesystem paths), `pickle`, `http_archive`, `combine_datasets`, `demo`, `synthetic`, `recaller`; auto-discovered via the `IMPORTER` sentinel. `sources/` sub-package provides the `MediaSource` abstraction for resolving media files from origins (local_folder, http_archive, pullwrest)
- `vtsearch/eval/` — Evaluation framework: runner, metrics, visualisation, voting iterations
- `vtsearch/exporters/` — Results exporters (server_json_file/server_csv_file/email_smtp/webhook/gui/holder); auto-discovered via `EXPORTER` sentinel
- `vtsearch/settings_io/` — Settings import/export plugins; `importers/` (local_json_file/server_json_file; auto-discovered via `SETTINGS_IMPORTER` sentinel), `exporters/` (local_json_file/server_json_file; auto-discovered via `SETTINGS_EXPORTER` sentinel), and `sources/` (server_json_file; auto-discovered via `SETTINGS_SOURCE` sentinel) for bidirectional sync
- `vtsearch/labels/` — Label importers and sync sources; `importers/` (server_json_file/server_csv_file/holder; auto-discovered via `LABEL_IMPORTER` sentinel) and `sources/` (server_json_file; auto-discovered via `LABELSET_SOURCE` sentinel) for bidirectional label sync
- `vtsearch/labels/sync.py` — Labelset source sync utilities: `sync_to_labelset_source()` (auto-export on vote change) and `sync_from_labelset_source()` (manual import), with `_syncing` guard to prevent circular re-export
- `vtsearch/media/` — Media type plugins: audio, image, text, video, document
- `vtsearch/converters/` — Media converters: document→image, document→text, video→audio, video→image. Each is a `MediaConverter` (a `PluginBase`) that can declare user-configurable params via `fields: list[PluginField]`. `convert(media, params)` reads those at conversion time. Used by the multi-media import flow (see `docs/plans/multi-media-import.md`) and the legacy per-importer `converters` field
- `vtsearch/utils/` — Global state (`DatasetContext`, proxy dicts for `medias`/votes, multi-dataset context store), progress utilities, plugin registry (`registry.py`), generic `SyncSource[LoadT, SaveT]` base class (`sync_source.py`) shared by `SettingsSource` and `LabelsetSource`, synthetic WAV generator (`audio_generator.py`), offline media synthesis (`synthetic/` — `images.py`, `audio.py`, `video.py` for the SyntheticDatasetImporter)
- `vtsearch/settings_factory.py` — Accessor factories (`make_accessors`, `make_per_side_setting`, `clamp`, `one_of`) used by `vtsearch/settings.py` to generate get/set pairs from the `_SETTING_SPECS` table
- `frontend/` — Angular SPA source (components, services, SCSS); builds to `static/` via `npm run build:prod`. `ActiveContextService` tracks which dataset/model the user selected; `activeContextInterceptor` attaches `X-Dataset-Id`/`X-Detector-Id` headers to every API request
- `static/` — Angular build output (index.html, main.js, polyfills.js, styles.css) and assets (favicons, logo.svg, logo.png)
- `docs/` — Extended docs (API.md, ARCHITECTURE.md, CLI.md, DEPLOYMENT.md, EVAL.md, EXTENDING.md + EXTENDING-plugins.md + EXTENDING-media.md + EXTENDING-processors.md, HANDOFF.md, ML.md, SETUP.md, USER_GUIDE.md, demos.md, design/cli-detector-converter.md, plans/README.md + open plan docs under plans/ — see that index)
- `tests/` — Test suite. Files are bucketed by group folder; the folder name **is** the pytest marker (`tests/core/test_*.py` → marker `core`). New test files inherit their group from where they live — no registry to update.
  - `conftest.py` — Shared fixtures: `reset_state` (autouse, clears all mutable global state), `isolated_settings` (autouse, redirects settings to tmp_path), `client` (Flask test client). Also stubs all embedders so tests never download real model weights.
  - `helpers.py` — Shared helpers (`make_wav_bytes`, `make_dataset_file`, media-builder fns) — imported as `from helpers import ...` via the `pythonpath = ["tests"]` in `pyproject.toml`.
  - `fixtures/medias.py` — Test media generation + per-worker embedding cache, loaded by conftest.
  - `tests/__init__.py` — Test-package helpers (`load_detector_and_wait`).
  - Test groups (folders): `core/`, `api/`, `sorting/`, `datasets/`, `io/`, `detectors/`, `downloads/`, `integration/`, `cli/`, `converters/`, `gpu/`. To find tests for an area, look in the corresponding folder; to add a new test, drop it in the folder that matches its concern.

## Test Groups

Tests are grouped by folder under `tests/`. Each folder is a pytest marker — `./run-tests.sh <group>` runs all tests in `tests/<group>/`. New tests inherit their group from the folder they're added to.

| Group | Description |
|-------|-------------|
| `core` | Basic app functionality (audio, medias, votes, inclusion, settings, frontend, torch config) |
| `api` | API contracts, error handling, security, dashboard, embed |
| `sorting` | Sort algorithms, diversity, safe thresholds, enriched text sort |
| `datasets` | Dataset loading, splitting, dedup, parallel/chunked/thin loading, multi-dataset context |
| `io` | Importers, exporters, label I/O, settings I/O, sync sources, PDF/NPZ import |
| `detectors` | Detectors, embedders, clippers, eval, processors, training |
| `downloads` | Demo dataset downloads (AG News, BBC, GTZAN, IMDB, image sources, UCSF, video, generic extract) |
| `integration` | End-to-end workflows, thread safety, async jobs |
| `cli` | CLI autodetect, load sort window, progress bars |
| `converters` | Media converters (document, video, image) |
| `gpu` | CUDA-only tests (excluded by default) |

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
   - `autorun_extractors`, `autorun_localizers` (global state)
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
- **No Chrome/Chromium available.** The cloud container (Ubuntu 24.04) does not have Chrome or Chromium installed. Karma has been removed from frontend devDependencies. The Python backend tests (`./run-tests.sh`) work fine without a browser.

## Key Details
- **Multi-dataset support**: Multiple datasets can be loaded in memory simultaneously. Per-dataset state is bundled in `DatasetContext` objects (`vtsearch/utils/state_core.py`). The module-level names `medias`, `good_votes`, `bad_votes`, etc. are **proxy objects** (`_ProxyDict`/`_ProxyList`) that delegate to the context resolved per-request. The frontend sends `X-Dataset-Id` and `X-Detector-Id` HTTP headers to specify which loaded dataset/model each request operates on (see `ActiveContextService` and `activeContextInterceptor` in the Angular frontend). The `before_request` handler in `app.py` stashes the resolved contexts on Flask's `g`, and the proxy objects check `g` first. Outside a request context (background threads, tests), proxies fall back to a **thread-local** context set via `set_thread_dataset_context()` / `set_thread_detector_context()`. There is no global "active" pointer. Key functions: `register_context()`, `unregister_context()`, `get_context()`, `list_loaded_dataset_ids()`. Global (non-per-dataset) state: `autorun_extractors`, `autorun_localizers`. API: `POST /api/datasets/registry/<id>/load` (load from pkl), `POST /api/datasets/registry/<id>/unload` (free RAM). Registry tracks `_loaded_ids` (set of in-memory dataset IDs).
- Per-dataset state in `DatasetContext`: `medias`, `diversity_tree`, `dataset_display_name`
- Per-detector state in `DetectorContext`: `good_votes`, `bad_votes`, `label_history`, `vote_click_times`, `click_counter`, `last_learned_scores`, `textsort_suggestions`, `find_initial_labels`, `inclusion`, `training_medias`, `model`, `threshold`, `labelset_source`
- All mutable state protected by `_state_lock` (RLock) in `vtsearch/utils/state_core.py`
- Votes are `dict[int, None]` (not sets) — use `votes[id] = None` syntax
- Persistent settings live in `vtsearch/settings.py` (auto-saves to `data/settings.json`): volume, inclusion, theme, enrich_descriptions, safe_thresholds, calibrate_count, calibration_fraction, audio_playing, swipe_animation, show_metadata, view_mode_left, view_mode_right, focus_mode_left, focus_mode_right, grid_icon_size_left, grid_icon_size_right, panel_pct_left, panel_pct_right, autoload_media_embedders, autorun_detectors, autopilot_enabled, hide_autopilot, autopilot_top_greens, autopilot_hard_reds, autopilot_resort_interval, autopilot_goal_diversity, saved_datasets_dir, detectors_dir, max_concurrent_dataset_downloads, max_concurrent_dataset_embeddings, settings_source
- **Concurrent dataset loading**: Dataset loads go through two independent `ConcurrencyGate`s in `vtsearch/datasets/load_pipeline.py` — `_download_gate` covers the importer's download/import phase (bandwidth/disk-bound) and `_embed_gate` covers all CPU/GPU-bound embedding work (importer-side per-file embedding plus post-load clipping, dedup, diversity tree, and embedder warm-up). A task acquires the download gate first, swaps to the embed gate when the importer emits its first `"embedding"` progress status (so another dataset can start downloading while this one embeds), and post-load steps also run under the embed gate. Limits are read fresh on every acquire, so changes to `max_concurrent_dataset_downloads` / `max_concurrent_dataset_embeddings` take effect for queued and future tasks (running tasks are never preempted). Defaults are 1/1, preserving serialised behaviour out of the box.
- **Sync sources** provide bidirectional sync for settings and detector labels. A `SettingsSource` auto-exports settings to an external target (e.g. server JSON file) on every change, and can auto-import on startup. A `LabelsetSource` does the same for a model's labels — auto-exporting on vote changes and auto-importing on model load. Both use the `PluginRegistry` discovery pattern (sentinels `SETTINGS_SOURCE` and `LABELSET_SOURCE`). Standalone importers/exporters remain fully functional regardless of whether a source is active. Config: `settings_source` key in `settings.json` (excluded from defaults and source export to avoid circularity); `labelset_source` field on `DetectorContext`. Template variables: `{username}` for settings sources, `{detector_id}`/`{detector_name}` for labelset sources
- Each media item has `origin` (dict or None), `origin_name` (str), and optionally `media_url` (str) for per-element provenance and URL-based lazy-fetch
- `Origin` class in `vtsearch/datasets/origin.py`; `LabelSet`/`LabeledElement` in `vtsearch/datasets/labelset.py`. `LabeledElement` has an optional `metadata` dict for arbitrary per-label data that round-trips through serialisation
- Label export (`/api/labels/export`) returns a `LabelSet` with per-element origin info (superset of legacy format). With `enrich=true`, `origin.params` are flattened into `custom_metadata` and `available_columns`
- **Extension scaffolds** (hidden_from_picker=True until API clients implemented): `vtsearch/datasets/importers/recaller/` (ReCaller dataset importer), `vtsearch/exporters/holder/` (Holder labelset exporter), `vtsearch/labels/importers/holder/` (Holder label importer), `vtsearch/datasets/sources/pullwrest.py` (PullWrest media source). See `docs/plans/RCDatasetImporter.md` for dev instructions
- **Multi-media imports**: importers can set the class attribute `multi_media = True` and iterate `self.effective_source_specs(field_values)` inside `run()` to pull in multiple source media types (e.g. images + videos-as-images + documents-as-images) with per-converter params (e.g. `n_clips`). Each `SourceSpec` is `(source_type, converter|None, params)`. Legacy importers (`multi_media = False`, the default) still work unchanged via the comma-separated `converters` field; `effective_source_specs()` also returns a useful list for them so they can migrate the body of `run()` before changing their form schema. See `docs/plans/multi-media-import.md`. Migrated: `server_folder`, `server_files`, `local_folder`, `local_files` (the lf-* importers are upload placeholders that re-enter `server_folder`). Remaining on the shim: `pickle`, `combine_datasets`, `synthetic`, `http_archive`, `recaller`, `demo`
- `data/` dir created at runtime for embeddings, model cache, media files
- OMP_NUM_THREADS and MKL_NUM_THREADS set to 1 for memory optimization
- Linter/formatter: ruff (E402 ignored, line-length 120, target-version py310, see pyproject.toml)
