# Extract `vtscore` Library Plan

Status: **Phases 0–4 functionally shipped** — public API surface captured in [`docs/vtscore-api.md`](../vtscore-api.md). Every library-candidate package is Flask-free, settings-free, proxy-free, *and* free of hardcoded `data/` path strings: every reference to the data directory goes through `vtsearch.config.DATA_DIR` (the canonical constant snapshotted into `CoreConfig.data_dir`). The module-level proxy view (`medias`, `good_votes`, etc.) lives in `vtsearch/shim/state_proxies.py` and is re-exported by `vtsearch.state.__init__` for app callers. Library code resolves the active `DatasetContext` / `DetectorContext` explicitly via `get_active_*_context()`. Phase 8's bridge relocation moved the body of `CoreConfig.from_settings()` into `vtsearch/shim/` (`build_core_config()`), wired via `register_core_config_builder()`. Phase 5's entry-point hook landed ahead of the split. Phases 5-9 are next.

Goal: split VTSearch into two distributions in one repo:

- **`vtscore`** — reusable Python library: dataset origins, MediaSources, clippers/croppers, embedders, MLP/detector training and scoring, evaluation. No Flask, no Angular, no auto-writing JSON configs.
- **`vtsearch`** — the Flask + Angular application that wraps `vtscore`. Owns user-facing concerns: HTTP routes, auth, persistent user preferences, the SPA.

The expensive work is **introducing seams in the current monolith**. Once `vtsearch` itself runs cleanly through those seams with no behaviour change, the actual `git mv` is mechanical.

**Path-name drift note.** This plan was written when the embedder/detector code lived under `vtsearch/models/`. That package has since been split into three: `vtsearch/embedding/` (embedder loaders, matrix cache), `vtsearch/detectors/` (detector lifecycle, workflow, store, training), and `vtsearch/training/` (media-agnostic MLP / threshold primitives). Path references below have been updated to the current layout; the underlying refactor work is unchanged.

## Naming

- Library import name: `vtscore` (recommended). Alternatives considered and rejected: `VTSearchLib` (non-idiomatic camelCase), `vtsearch.lib` (inverts the dependency direction — the app depends on the library, not vice versa), `vtsearch.core` namespace package (works but adds packaging complexity for marginal benefit).
- App import name: stays `vtsearch`.
- PyPI distribution names if/when published: `vtscore` and `vtsearch`.

## Phase 0 — Preparation (non-blocking, do anytime)

- [x] Inventory the actual public surface that external consumers would call. Captured in [`docs/vtscore-api.md`](../vtscore-api.md) as a docstring-only API sketch. This is the contract the refactor must preserve.
- [ ] Add a CI job that runs the full test suite with `flask` *uninstalled* against a candidate library subset, to prove import-cleanliness as the seams land. *(Caveat: VTSearch has no GitHub Actions workflows today — `./run-tests.sh` is the source of truth per CLAUDE.md. The equivalent gate here is a new `./run-tests.sh vtscore-clean` mode that runs library-candidate tests in a venv with Flask absent; revisit when Phase 5 is ready.)*

## Phase 1 — Cut the Flask seam

The library cannot import Flask. Today the leakage outside `routes/` is small:

- [x] `vtsearch/state/core.py` — `_request_*_context()` helpers replaced with module-level resolver callables (`_dataset_context_resolver`, `_detector_context_resolver`) installable via `register_dataset_context_resolver()` / `register_detector_context_resolver()`. Default = `lambda: None`. The Flask-aware versions live in the new `vtsearch/shim/` package and are installed once at `app.py` startup via `register_flask_context_resolvers()`.
- [x] `vtsearch/media/base.py` — formerly imported Flask; now Flask-free. The route layer converts the abstract `MediaResponse` into a `flask.Response` via `media_response_to_flask`. Module-level docstring still *mentions* Flask for orientation, but no `import flask`.
- [x] `vtsearch/detectors/workflow.py` (`apply_and_retrain`) — formerly wrote `g._detector_context = det_ctx` to swap the request-scoped context. Now uses `override_detector_context()` from `vtsearch.state.core`, a new context manager that sits at the top of `get_active_detector_context()`'s resolution chain (above the resolver and the thread-local). Works in both Flask requests and background threads.

Audit command (run before declaring this phase done):

```sh
grep -rn "^import flask\|^from flask\|^\s*import flask\|^\s*from flask" \
  vtsearch/{datasets,detectors,embedding,training,media,converters,exporters,labels,eval,plugins,concurrency,state,sync,utils,security}
```

**Exit criteria** ✅ that command returns zero hits as of this commit. The remaining Flask imports live in `vtsearch/auth/`, `vtsearch/logging_config.py`, `vtsearch/routes/`, and `app.py` — all unambiguously app-layer.

## Phase 2 — Cut the settings seam

Each library-candidate call site that pulls from `vtsearch.settings` needs to accept its config as an argument instead.

Files converted:

- [x] `vtsearch/cli.py` — was `set_settings_path` + `get_autorun_detectors`. Now builds a `CoreConfig` once at the top of `autodetect()`; the optional `settings_path` flows through `CoreConfig.from_settings(settings_path=…)`, and `autorun_detectors` becomes `config.autorun_detectors`.
- [x] `vtsearch/cli_pipeline.py` — same treatment.
- [x] `vtsearch/datasets/load_pipeline.py` — `ConcurrencyGate` caps now read through `CoreConfig.from_settings()`. The `set_last_embedder_for_media_type` write becomes an opt-in app-installed hook (`register_last_embedder_persistence_hook`) wired by `vtsearch/shim/`.
- [x] `vtsearch/datasets/registry.py` — routes through `CoreConfig.from_settings().saved_datasets_dir`.
- [x] `vtsearch/detectors/labeling_progress.py` — routes through `CoreConfig.from_settings().autopilot_goal_diversity`.
- [x] `vtsearch/detectors/store.py` — routes through `CoreConfig.from_settings().detectors_dir`.
- [x] `vtsearch/state/__init__.py` — read wrappers (`get_inclusion`, `get_calibrate_count`, …) read from `CoreConfig.from_settings()`. Write wrappers (`set_inclusion`, …) delegate persistence to `register_setting_persister(key, fn)`, which `vtsearch/shim/` wires to the matching `vtsearch.settings.set_*` at app startup.

Already clean (no `vtsearch.settings` imports today):

- `vtsearch/embedding/loader.py` (was `vtsearch/models/loader.py`).
- `vtsearch/sync/__init__.py` (mentions settings only in docstrings).

Approach:

1. [x] Define a `vtscore.config.CoreConfig` dataclass with the knobs library code actually consumes (`safe_thresholds`, `calibrate_count`, `calibration_fraction`, `enrich_descriptions`, `data_dir`, `saved_datasets_dir`, `detectors_dir`, `autopilot_goal_diversity`, `max_concurrent_dataset_downloads`, `max_concurrent_dataset_embeddings`, `inclusion`). **Shipped** as `vtsearch.config.CoreConfig` (will move to `vtscore.config.CoreConfig` in Phase 8). `CoreConfig.from_settings()` is the app-side bridge.
2. Replace direct `from vtsearch.settings import get_X` calls with reading from a `CoreConfig` argument or attribute of an enclosing context object (`DatasetContext`/`DetectorContext`). **In progress**: the two easy-win files (`detectors/store.py`, `datasets/registry.py`) now route through `CoreConfig.from_settings()`.
3. App-side: at request boundary, build a `CoreConfig` from `vtsearch.settings` and pass it down. Settings auto-save behaviour stays in the app.
4. The `vtsearch/state/__init__.py` setter wrappers (e.g. `set_inclusion`, `set_calibrate_count`) currently both update the in-memory cache **and** call `vtsearch.settings.set_X` to persist. Library callers should only do the in-memory half; the app-side persistence half moves into a thin shim that the app installs.

**Exit criteria** ✅ as of this commit: the same `grep` from Phase 1 (extended to look for `vtsearch.settings` instead of `flask`) returns **zero** hits in library-candidate modules. The Phase 8 bridge relocation (see Open follow-ups #6) moved the body of `CoreConfig.from_settings()` into `vtsearch/shim/__init__.py:build_core_config()`; the classmethod in `vtsearch/config.py` now delegates to a builder installed by the app at startup via `register_core_config_builder()`. Library-only consumers without an app construct `CoreConfig(...)` directly.

## Phase 3 — Cut the global-state seam

Currently `medias`, `good_votes`, `label_history`, etc. are module-level proxies. Library consumers should be able to pass `DatasetContext`/`DetectorContext` explicitly.

- [x] Audit every public library function and ensure it accepts a context object as a parameter. The state submodules (`vtsearch/state/votes.py`, `clicks.py`, `media_lookup.py`, `diversity.py`) and `vtsearch/labels/sync.py` now resolve the active context internally via `get_active_*_context()` and operate on `ctx.attr` instead of importing the module-level proxy names — the dependency on "the active context" is explicit in each function body.
- [x] Keep the proxy module in the *app* layer, not the library. The proxy classes (`_ProxyDict`, `_ProxyList`) and module-level instances (`medias`, `good_votes`, `bad_votes`, `label_history`, `vote_click_times`, `vote_region_boxes`, `last_learned_scores`, `textsort_suggestions`) now live in `vtsearch/shim/state_proxies.py`. `vtsearch/state/__init__.py` re-exports them so `from vtsearch.state import medias` still works for routes/tests; the library subpackage submodules never import them.
- [x] Move `autorun_extractors`, `autorun_localizers` off the library-candidate `vtsearch/state/` package onto an app-side singleton (`vtsearch/autorun_processors.py`). The dicts + CRUD now live in a single file outside `state/`; `state/processors.py` is gone, and `state/__init__.py` no longer re-exports any autorun names. Route + test imports were updated to point at the new module. `autorun_detectors` is already an app-side server-tier setting in `vtsearch/settings.py`, so the relocation is complete.

**Exit criteria** ✅ — `grep -rn "^medias\|^good_votes\|^_ProxyDict\|^_ProxyList" vtsearch/{datasets,detectors,embedding,training,media,converters,exporters,labels,eval,plugins,concurrency,state,sync,utils,security}` returns zero hits as of this commit. The only file that defines or instantiates the proxy names is `vtsearch/shim/state_proxies.py` (app-side).

## Phase 4 — Cut the filesystem seam

The library should never assume a `data/` directory exists at CWD.

- [x] All `data/` path resolution flows through `CoreConfig.data_dir` or a passed-in `Path`. Every library-candidate module derives data paths from `vtsearch.config.DATA_DIR` (a module-level constant resolved at import time from `VTSEARCH_DATA_DIR` or `<repo>/data`), which is snapshotted into `CoreConfig.data_dir` by `build_core_config()`. No library file hardcodes a `"data/"` string literal.
- [x] Embedding cache, model cache, ingestion staging — all parameterised. `EMBEDDINGS_DIR`, `MODELS_CACHE_DIR`, and `STAGING_DIR` are all derived from `DATA_DIR` (the first two in `vtsearch/config.py`, `STAGING_DIR` in `vtsearch/datasets/load_pipeline.py`).
- [x] App default: `data/` relative to the app's run directory (today's behaviour, preserved). `DATA_DIR` defaults to `_REPO_ROOT / "data"` so existing installations keep working; `VTSEARCH_DATA_DIR` overrides.

**Exit criteria** ✅ — `grep -rn '"data/\|"data\|\x27data/\|Path(\x27data\x27)\|Path("data")' vtsearch/{datasets,detectors,embedding,training,media,converters,exporters,labels,eval,plugins,concurrency,state,sync,utils,security,cli.py,cli_pipeline.py,cli_progress.py,config.py}` returns zero hardcoded path hits as of this commit. The five plugin form-field placeholders that previously hardcoded `"data/..."` (server_csv_file + server_json_file exporters, server_csv_file + server_json_file label importers, server_json_file labelset source) now interpolate `DATA_DIR` so they always honor `VTSEARCH_DATA_DIR`. Placeholders are absolute paths instead of relative.

## Phase 5 — Plugin discovery

The sentinel-based registry (`IMPORTER`, `EXPORTER`, `SETTINGS_SOURCE`, `LABELSET_SOURCE`, `PROCESSOR_IMPORTER`, `LABEL_IMPORTER`, `SETTINGS_IMPORTER`, `SETTINGS_EXPORTER`) walks packages by name. After the split, plugins live in two distributions.

- [ ] Library exposes `vtscore.utils.registry.PluginRegistry` (already generic).
- [ ] Library auto-discovers its own plugins (importers, exporters, label sources, etc.) at registry creation.
- [ ] App registers app-only plugins (`settings_io/`, settings sources) on top of the library's registries at startup.
- [x] Add an `importlib.metadata` entry-point hook so third-party packages can register plugins without monkey-patching. (Landed ahead of the library split as feature-brainstorm §12.11 — `PluginRegistry(entry_point_group=…)` scans `vtsearch.<family>` groups. Built-ins win on name clashes; broken entry points warn and are skipped.)

## Phase 6 — Pickle compatibility

Existing dataset pickles and detector weights reference classes by `vtsearch.X` import paths. Renaming breaks unpickling.

- [ ] Audit which classes appear in saved pickles (origin classes, labelset, MLP weights). Use `weights_compat.py` as the precedent.
- [ ] Add a custom `Unpickler.find_class` that maps `vtsearch.datasets.origin.Origin` → `vtscore.datasets.origin.Origin`, etc.
- [ ] Re-export old paths from the app for one release as a compat shim (since CLAUDE.md allows breaking compatibility, this can be skipped if we accept users re-saving — but the unpickler shim is cheap insurance).

## Phase 7 — Test split

- [ ] Identify tests that don't use the Flask `client` fixture and don't reach into `vtsearch.routes` or `vtsearch.settings`. These are library-test candidates.
- [ ] Create `tests_lib/` for library-only tests; keep `tests/` for app tests. Both run from `./run-tests.sh`.
- [ ] Library tests must pass with `pip install vtscore` only — no Flask installed in their virtualenv. CI enforces.
- [ ] Conftest fixtures split: `reset_state` and `client` stay app-side; library gets a smaller `reset_contexts` fixture.

## Phase 8 — Physical move

Once Phases 1–7 are green and behaviour-identical:

1. Create `vtscore/` directory at repo root.
2. `git mv` library subpackages into it: `datasets/`, `embedding/`, `detectors/`, `training/`, `media/`, `converters/`, `exporters/`, `labels/`, `eval/`, `plugins/`, `concurrency/`, `state/`, `sync/`, `security/`, plus `cli.py`, `cli_pipeline.py`, `cli_progress.py`, `config.py`, and the relevant `utils/` modules. (`processors/` does not exist as a top-level package today — processor *plugins* are reached via `vtsearch.media.processors` and per-route handlers; verify the move target before Phase 8.)
3. Search-and-replace `vtsearch.datasets` → `vtscore.datasets` etc., across the codebase. Note: the existing `vtsearch/state/__init__.py` is **app-tier** (it re-exports the proxy view from `vtsearch.shim.state_proxies` and hosts the settings-persistence hooks). When `state/` moves to `vtscore/state/`, leave the library submodules (`core.py`, `votes.py`, `clicks.py`, `media_lookup.py`, `diversity.py`, `diversity_tree.py`) under `vtscore/state/` and keep a thin `vtsearch/state/__init__.py` shim that re-exports the library names alongside the proxies — that file becomes pure app-tier glue.
4. Add `vtscore/pyproject.toml` and `pyproject.toml` workspace config so both distributions build.
5. App imports become `from vtscore.X import ...`.
6. Run full suite. Fix straggler imports.

## Phase 9 — Release plumbing

- [ ] Independent semver for `vtscore` (start at 0.1.0).
- [ ] CHANGELOG.md per distribution.
- [ ] `vtscore` README with quickstart: load a dataset, train a detector, score a folder.
- [ ] Decide on PyPI publication cadence (probably defer until a real external consumer exists).

## What goes where (final shape)

### `vtscore/` (library)

```
vtscore/
├── config.py              # CoreConfig dataclass + constants
├── cli.py                 # autodetect entrypoint (Flask-free)
├── cli_pipeline.py        # CLI orchestration (Flask-free)
├── cli_progress.py        # CLI progress bars
├── datasets/              # origins, labelsets, loaders, importers, sources, ingestion, split
├── embedding/             # embedder façades, loader, matrix cache, torch runtime helpers
├── detectors/             # detector lifecycle (registry/store), training pipeline, resolver,
│                          # workflow, label_sync, labeling_progress
├── training/              # media-agnostic MLP/threshold primitives + region similarity + SVM
├── media/                 # audio/image/text/video/document type plugins + clipper/cropping
├── converters/            # document→image, video→audio, etc.
├── exporters/             # results exporters (file/CSV/webhook/email/holder)
├── labels/                # label importers + labelset sync sources
├── eval/                  # evaluation runner, metrics, visualisation
├── plugins/               # PluginRegistry + sentinel discovery
├── concurrency/           # async jobs, memory budget, progress trackers
├── state/                 # DatasetContext, DetectorContext (no Flask)
├── sync/                  # SyncSource[L,S] ABC
├── security/              # path/URL validation, safe_pickle_load
└── utils/                 # hits, synthetic media generators, settings_factory helpers
```

### `vtsearch/` (application)

```
vtsearch/
├── app.py                 # Flask entry point
├── settings.py            # auto-saving JSON user prefs (server + per-user tiers)
├── settings_factory.py
├── settings_models.py
├── settings_io/           # settings import/export plugins + sync sources
├── auth/                  # LoginProvider ABC + default impl
├── achievements.py        # per-user achievement tracking
├── logging_config.py
├── routes/                # all Flask blueprints
├── schemas/               # marshmallow request/response schemas
├── shim/                  # NEW: glue between Flask g and vtscore contexts;
│                          #      builds CoreConfig from settings on each request;
│                          #      provides the Flask-aware context resolver vtscore consumes
tests/                     # app tests (Flask client, settings I/O, routes)
tests_lib/                 # NEW (Phase 7): library-only tests, run without Flask
frontend/                  # Angular SPA source
static/                    # built Angular output
```

**Cross-boundary notes**:
- ~~`state/processors.py`'s autorun-extractor/localizer CRUD operates on globals defined in `state/core.py` — Phase 3 needs to relocate those before `state/` can move into `vtscore/`.~~ **Done** — the autorun registry now lives in `vtsearch/autorun_processors.py` (app-side); `state/processors.py` is gone and `state/core.py` no longer defines `autorun_extractors`/`autorun_localizers`.
- `plugins/inventory.py` currently imports from `vtsearch.settings_io.*` to enumerate app-side plugin families. After the split, inventory either lives app-side or accepts the app-side registries via DI.

## Risks and open questions

- **Settings sources / labelset sources straddle the boundary.** *Resolved (Phase 0 review):* `SettingsSource` is a user-pref concern and stays app-side. `LabelsetSource` ships in `vtscore.labels` — pulling labels in from external systems is a core library affordance, and the registry (`get_labelset_source` / `list_labelset_sources`) ships with it. The shared `SyncSource[L,S]` ABC remains in `vtscore.sync`.
- **`medias.py` test-media generator** is used both at startup (app concern) and by `conftest.py`. The generator already moved to `tests/fixtures/medias.py` and is not part of the runtime app — no Phase action needed.
- **`eval/visualize.py`** *Resolved (Phase 0 review):* plotting (`plot_eval_results`, `plot_voting_iterations`) is presentation, not computation, and stays in `vtsearch/`. The library exports the data (`DatasetResult`, `QueryMetrics`, `LearnedSortMetrics`, `format_results_json`); the app renders it. No `vtscore[viz]` extra needed.
- **`autorun_*` global state.** *Resolved (Phase 0 review):* the registry of which processors to autorun (`autorun_detectors`, `autorun_extractors`, `autorun_localizers` and their CRUD) is *policy* and stays app-side. The library keeps the `Processor` / `Detector` / `Localizer` / `Extractor` ABCs plus the code that applies them; the app passes in the resolved list. Phase 3 wires this as an explicit argument.
- **Per-thread progress callback on `vtscore.media`.** *Resolved (Phase 0 review):* mirror `vtscore.concurrency.progress.set_thread_progress` — add `set_thread_progress_callback` that takes priority when set on the calling thread, keep `set_progress_callback` as the global default. Library consumers running multi-threaded ingestion won't have one thread clobber another's callback.
- **Heavy ML deps** (torch/transformers/CLAP/CLIP) — declare as required for now; revisit extras (`vtscore[embedders]`) if a consumer asks for a leaner install.

## Order of operations recap

Phases 1 → 4 are independent and can land in parallel PRs. Phase 5 depends on 1–4. Phase 6 can land any time after Phase 0. Phase 7 should land before Phase 8. Phase 8 is one big PR; Phase 9 is post-merge cleanup.

## Open follow-ups (what's left)

Phase 3 is functionally done. Picking up from here, ordered smallest-first:

1. ~~Phase 0 inventory doc, Phase 1 (Flask seam), Phase 2 (settings seam).~~ **All shipped.** See the per-phase status blocks above.
2. ~~**Phase 3, easy bit: relocate `autorun_extractors` / `autorun_localizers`.**~~ **Shipped.** Moved to the new app-side singleton `vtsearch/autorun_processors.py` (dicts + CRUD); `state/processors.py` deleted; `state/core.py` and `state/__init__.py` no longer reference the autorun names. Route + test imports updated.
3. ~~**Phase 3, audit: explicit context parameters.**~~ **Shipped.** The state submodules (`vtsearch/state/votes.py`, `clicks.py`, `media_lookup.py`, `diversity.py`) and `vtsearch/labels/sync.py` now resolve the active context internally via `get_active_*_context()` and operate on `ctx.attr` directly — no library-candidate code imports the module-level proxy names anymore.
4. ~~**Phase 3, finish: move the proxies to the app layer.**~~ **Shipped.** Proxy classes (`_ProxyDict`, `_ProxyList`) and instances (`medias`, `good_votes`, …) moved out of `vtsearch/state/core.py` into `vtsearch/shim/state_proxies.py`. `vtsearch/state/__init__.py` re-exports them so app-tier imports (`from vtsearch.state import medias`) keep working. Test imports that reached into `state.core` for proxies were updated to use `vtsearch.state` or `vtsearch.shim.state_proxies`.
5. ~~**Phase 4, filesystem seam.**~~ **Shipped.** All `data/` references now route through `vtsearch.config.DATA_DIR` (which is what `CoreConfig.data_dir` snapshots). The five plugin form-field placeholders that previously hardcoded `"data/..."` strings (server_csv_file + server_json_file exporters, server_csv_file + server_json_file label importers, server_json_file labelset source) now build their placeholder/default from `DATA_DIR` via f-strings. Tests in `tests/io/test_csv_webhook_exporters.py` were updated to assert the dynamic value. Side effect: placeholders are now absolute paths (e.g. `/home/user/VTSearch/data/...`) instead of relative — this is more correct in a library context because relative-from-CWD was always brittle.
6. ~~**Phase 8 bridge relocation.**~~ **Shipped.** The body of `CoreConfig.from_settings()` moved out of `vtsearch/config.py` into `vtsearch/shim/__init__.py:build_core_config()`. The classmethod is now a thin wrapper that delegates to whatever the app installed via `vtsearch.config.register_core_config_builder()`; `vtsearch/shim/register_app_config_builder()` is the app-side wiring (called from `app.py` startup alongside the other shim hooks). Library-only consumers without an app skip `from_settings()` entirely and construct `CoreConfig` directly — they get a clear `RuntimeError` if they try otherwise. `vtsearch/config.py` no longer imports `vtsearch.settings`.

When you take one of these on, check the box and add a one-line "shipped in #PR" note so the plan tracks reality.
