# Extract `vtscore` Library Plan

Status: **Phases 0–8 shipped, Phase 6 audited as a no-op** — public API surface captured in [`docs/vtscore-api.md`](../vtscore-api.md). The library now lives at its final home `vtscore/` (Phase 8's physical move). Every library-candidate package is Flask-free, settings-free, proxy-free, *and* free of hardcoded `data/` path strings: every reference to the data directory goes through `vtscore.config.DATA_DIR` (the canonical constant snapshotted into `CoreConfig.data_dir`). The module-level proxy view (`medias`, `good_votes`, etc.) lives in `vtsearch/shim/state_proxies.py` and is re-exported by `vtsearch/state/__init__.py` (a thin app-tier shim that re-exports `vtscore.state` and adds the proxy view on top). Library code resolves the active `DatasetContext` / `DetectorContext` explicitly via `get_active_*_context()`. Phase 8's bridge relocation moved the body of `CoreConfig.from_settings()` into `vtsearch/shim/` (`build_core_config()`), wired via `register_core_config_builder()`. Phase 5's entry-point hook, plugin-family DI seam, and eager registry discovery are all in (plugins are populated by the time their package `__init__.py` returns); library plugin entry-point group names are now `vtscore.<family>`, settings plugin groups remain `vtsearch.<family>`. Phase 6's audit found that VTSearch pickles store only plain Python types + numpy (enforced by `safe_pickle_load`'s allowlist) — `Origin` round-trips as a dict, detectors are JSON, MLP weights live only in memory — so the unpickler `find_class` shim Phase 6 originally specified is unnecessary. Phase 7 split the test suite along the library/app seam: 43 library-candidate tests moved to `tests_lib/` with their own Flask-free conftest, and `./run-tests.sh vtscore-clean` runs them with a meta-path import hook that refuses `flask` / `werkzeug` / `flask_smorest`. Only Phase 9 (release plumbing — independent semver, CHANGELOG, README quickstart, PyPI cadence) remains.

Goal: split VTSearch into two distributions in one repo:

- **`vtscore`** — reusable Python library: dataset origins, MediaSources, clippers/croppers, embedders, MLP/detector training and scoring, evaluation. No Flask, no Angular, no auto-writing JSON configs.
- **`vtsearch`** — the Flask + Angular application that wraps `vtscore`. Owns user-facing concerns: HTTP routes, auth, persistent user preferences, the SPA.

The expensive work is **introducing seams in the current monolith**. Once `vtsearch` itself runs cleanly through those seams with no behaviour change, the actual `git mv` is mechanical.

**Path-name drift note.** This plan was written when the embedder/detector code lived under `vtsearch/models/`. That package has since been split into three: `vtscore/embedding/` (embedder loaders, matrix cache), `vtscore/detectors/` (detector lifecycle, workflow, store, training), and `vtscore/training/` (media-agnostic MLP / threshold primitives). Path references below have been updated to the current layout; the underlying refactor work is unchanged.

## Naming

- Library import name: `vtscore` (recommended). Alternatives considered and rejected: `VTSearchLib` (non-idiomatic camelCase), `vtsearch.lib` (inverts the dependency direction — the app depends on the library, not vice versa), `vtsearch.core` namespace package (works but adds packaging complexity for marginal benefit).
- App import name: stays `vtsearch`.
- PyPI distribution names if/when published: `vtscore` and `vtsearch`.

## Phase 0 — Preparation (non-blocking, do anytime)

- [x] Inventory the actual public surface that external consumers would call. Captured in [`docs/vtscore-api.md`](../vtscore-api.md) as a docstring-only API sketch. This is the contract the refactor must preserve.
- [ ] Add a CI job that runs the full test suite with `flask` *uninstalled* against a candidate library subset, to prove import-cleanliness as the seams land. *(Caveat: VTSearch has no GitHub Actions workflows today — `./run-tests.sh` is the source of truth per CLAUDE.md. The equivalent gate here is a new `./run-tests.sh vtscore-clean` mode that runs library-candidate tests in a venv with Flask absent; revisit when Phase 5 is ready.)*

## Phase 1 — Cut the Flask seam

The library cannot import Flask. Today the leakage outside `routes/` is small:

- [x] `vtscore/state/core.py` — `_request_*_context()` helpers replaced with module-level resolver callables (`_dataset_context_resolver`, `_detector_context_resolver`) installable via `register_dataset_context_resolver()` / `register_detector_context_resolver()`. Default = `lambda: None`. The Flask-aware versions live in the new `vtsearch/shim/` package and are installed once at `app.py` startup via `register_flask_context_resolvers()`.
- [x] `vtscore/media/base.py` — formerly imported Flask; now Flask-free. The route layer converts the abstract `MediaResponse` into a `flask.Response` via `media_response_to_flask`. Module-level docstring still *mentions* Flask for orientation, but no `import flask`.
- [x] `vtscore/detectors/workflow.py` (`apply_and_retrain`) — formerly wrote `g._detector_context = det_ctx` to swap the request-scoped context. Now uses `override_detector_context()` from `vtscore.state.core`, a new context manager that sits at the top of `get_active_detector_context()`'s resolution chain (above the resolver and the thread-local). Works in both Flask requests and background threads.

Audit command (run before declaring this phase done):

```sh
grep -rn "^import flask\|^from flask\|^\s*import flask\|^\s*from flask" \
  vtsearch/{datasets,detectors,embedding,training,media,converters,exporters,labels,eval,plugins,concurrency,state,sync,utils,security}
```

**Exit criteria** ✅ that command returns zero hits as of this commit. The remaining Flask imports live in `vtsearch/auth/`, `vtsearch/logging_config.py`, `vtsearch/routes/`, and `app.py` — all unambiguously app-layer.

## Phase 2 — Cut the settings seam

Each library-candidate call site that pulls from `vtsearch.settings` needs to accept its config as an argument instead.

Files converted:

- [x] `vtscore/cli.py` — was `set_settings_path` + `get_autorun_detectors`. Now builds a `CoreConfig` once at the top of `autodetect()`; the optional `settings_path` flows through `CoreConfig.from_settings(settings_path=…)`, and `autorun_detectors` becomes `config.autorun_detectors`.
- [x] `vtscore/cli_pipeline.py` — same treatment.
- [x] `vtscore/datasets/load_pipeline.py` — `ConcurrencyGate` caps now read through `CoreConfig.from_settings()`. The `set_last_embedder_for_media_type` write becomes an opt-in app-installed hook (`register_last_embedder_persistence_hook`) wired by `vtsearch/shim/`.
- [x] `vtscore/datasets/registry.py` — routes through `CoreConfig.from_settings().saved_datasets_dir`.
- [x] `vtscore/detectors/labeling_progress.py` — routes through `CoreConfig.from_settings().autopilot_goal_diversity`.
- [x] `vtscore/detectors/store.py` — routes through `CoreConfig.from_settings().detectors_dir`.
- [x] `vtsearch/state/__init__.py` — read wrappers (`get_inclusion`, `get_calibrate_count`, …) read from `CoreConfig.from_settings()`. Write wrappers (`set_inclusion`, …) delegate persistence to `register_setting_persister(key, fn)`, which `vtsearch/shim/` wires to the matching `vtsearch.settings.set_*` at app startup.

Already clean (no `vtsearch.settings` imports today):

- `vtscore/embedding/loader.py` (was `vtsearch/models/loader.py`).
- `vtscore/sync/__init__.py` (mentions settings only in docstrings).

Approach:

1. [x] Define a `vtscore.config.CoreConfig` dataclass with the knobs library code actually consumes (`safe_thresholds`, `calibrate_count`, `calibration_fraction`, `enrich_descriptions`, `data_dir`, `saved_datasets_dir`, `detectors_dir`, `autopilot_goal_diversity`, `max_concurrent_dataset_downloads`, `max_concurrent_dataset_embeddings`, `inclusion`). **Shipped** as `vtscore.config.CoreConfig` (will move to `vtscore.config.CoreConfig` in Phase 8). `CoreConfig.from_settings()` is the app-side bridge.
2. Replace direct `from vtsearch.settings import get_X` calls with reading from a `CoreConfig` argument or attribute of an enclosing context object (`DatasetContext`/`DetectorContext`). **In progress**: the two easy-win files (`detectors/store.py`, `datasets/registry.py`) now route through `CoreConfig.from_settings()`.
3. App-side: at request boundary, build a `CoreConfig` from `vtsearch.settings` and pass it down. Settings auto-save behaviour stays in the app.
4. The `vtsearch/state/__init__.py` setter wrappers (e.g. `set_inclusion`, `set_calibrate_count`) currently both update the in-memory cache **and** call `vtsearch.settings.set_X` to persist. Library callers should only do the in-memory half; the app-side persistence half moves into a thin shim that the app installs.

**Exit criteria** ✅ as of this commit: the same `grep` from Phase 1 (extended to look for `vtsearch.settings` instead of `flask`) returns **zero** hits in library-candidate modules. The Phase 8 bridge relocation (see Open follow-ups #6) moved the body of `CoreConfig.from_settings()` into `vtsearch/shim/__init__.py:build_core_config()`; the classmethod in `vtscore/config.py` now delegates to a builder installed by the app at startup via `register_core_config_builder()`. Library-only consumers without an app construct `CoreConfig(...)` directly.

## Phase 3 — Cut the global-state seam

Currently `medias`, `good_votes`, `label_history`, etc. are module-level proxies. Library consumers should be able to pass `DatasetContext`/`DetectorContext` explicitly.

- [x] Audit every public library function and ensure it accepts a context object as a parameter. The state submodules (`vtscore/state/votes.py`, `clicks.py`, `media_lookup.py`, `diversity.py`) and `vtscore/labels/sync.py` now resolve the active context internally via `get_active_*_context()` and operate on `ctx.attr` instead of importing the module-level proxy names — the dependency on "the active context" is explicit in each function body.
- [x] Keep the proxy module in the *app* layer, not the library. The proxy classes (`_ProxyDict`, `_ProxyList`) and module-level instances (`medias`, `good_votes`, `bad_votes`, `label_history`, `vote_click_times`, `vote_region_boxes`, `last_learned_scores`, `textsort_suggestions`) now live in `vtsearch/shim/state_proxies.py`. `vtsearch/state/__init__.py` re-exports them so `from vtsearch.state import medias` still works for routes/tests; the library subpackage submodules never import them.
- [x] Move `autorun_extractors`, `autorun_localizers` off the library-candidate `vtsearch/state/` package onto an app-side singleton (`vtsearch/autorun_processors.py`). The dicts + CRUD now live in a single file outside `state/`; `state/processors.py` is gone, and `state/__init__.py` no longer re-exports any autorun names. Route + test imports were updated to point at the new module. `autorun_detectors` is already an app-side server-tier setting in `vtsearch/settings.py`, so the relocation is complete.

**Exit criteria** ✅ — `grep -rn "^medias\|^good_votes\|^_ProxyDict\|^_ProxyList" vtsearch/{datasets,detectors,embedding,training,media,converters,exporters,labels,eval,plugins,concurrency,state,sync,utils,security}` returns zero hits as of this commit. The only file that defines or instantiates the proxy names is `vtsearch/shim/state_proxies.py` (app-side).

## Phase 4 — Cut the filesystem seam

The library should never assume a `data/` directory exists at CWD.

- [x] All `data/` path resolution flows through `CoreConfig.data_dir` or a passed-in `Path`. Every library-candidate module derives data paths from `vtscore.config.DATA_DIR` (a module-level constant resolved at import time from `VTSEARCH_DATA_DIR` or `<repo>/data`), which is snapshotted into `CoreConfig.data_dir` by `build_core_config()`. No library file hardcodes a `"data/"` string literal.
- [x] Embedding cache, model cache, ingestion staging — all parameterised. `EMBEDDINGS_DIR`, `MODELS_CACHE_DIR`, and `STAGING_DIR` are all derived from `DATA_DIR` (the first two in `vtscore/config.py`, `STAGING_DIR` in `vtscore/datasets/load_pipeline.py`).
- [x] App default: `data/` relative to the app's run directory (today's behaviour, preserved). `DATA_DIR` defaults to `_REPO_ROOT / "data"` so existing installations keep working; `VTSEARCH_DATA_DIR` overrides.

**Exit criteria** ✅ — `grep -rn '"data/\|"data\|\x27data/\|Path(\x27data\x27)\|Path("data")' vtsearch/{datasets,detectors,embedding,training,media,converters,exporters,labels,eval,plugins,concurrency,state,sync,utils,security,cli.py,cli_pipeline.py,cli_progress.py,config.py}` returns zero hardcoded path hits as of this commit. The five plugin form-field placeholders that previously hardcoded `"data/..."` (server_csv_file + server_json_file exporters, server_csv_file + server_json_file label importers, server_json_file labelset source) now interpolate `DATA_DIR` so they always honor `VTSEARCH_DATA_DIR`. Placeholders are absolute paths instead of relative.

## Phase 5 — Plugin discovery

The sentinel-based registry (`IMPORTER`, `EXPORTER`, `SETTINGS_SOURCE`, `LABELSET_SOURCE`, `PROCESSOR_IMPORTER`, `LABEL_IMPORTER`, `SETTINGS_IMPORTER`, `SETTINGS_EXPORTER`) walks packages by name. After the split, plugins live in two distributions.

- [x] Library exposes a generic `PluginRegistry` (will move to `vtscore.plugins.PluginRegistry` in Phase 8; the class is already generic and re-exported from `vtscore.plugins`).
- [x] Library auto-discovers its own plugins (importers, exporters, label sources, etc.) at registry creation. `PluginRegistry.__init__` now calls `_ensure_discovered()` by default (`eager=True`), so each plugin family's `__init__.py` returns with the registry already populated. The lazy path is preserved as an opt-in (`eager=False`) for tests that inspect the pre-discovery state or simulate concurrent first access. Required deferring `vtscore/converters/runner.py`'s `from vtscore.converters import get_converter` import to a function-local import — the top-level import collided with eager discovery's `discover_modules=True` scan of the converters package, since `runner.py` is in the package directory but does not declare a `CONVERTER` sentinel.
- [x] App registers app-only plugins (`settings_io/`, settings sources) on top of the library's registries at startup. `vtscore/plugins/inventory.py` now exposes `FamilyProvider` + `register_plugin_family()`; library-tier families self-register at module import, and the app installs `settings_importers` / `settings_exporters` / `settings_sources` via `vtsearch/shim/register_app_plugin_families()` (called from `app.py` startup alongside the other shim hooks). Closes the last cross-boundary import in library-candidate code (`grep -rn "from vtsearch\.settings_io" vtsearch/{datasets,detectors,embedding,training,media,converters,exporters,labels,eval,plugins,concurrency,state,sync,utils,security}` returns zero hits as of this commit).
- [x] Add an `importlib.metadata` entry-point hook so third-party packages can register plugins without monkey-patching. (Landed ahead of the library split as feature-brainstorm §12.11 — `PluginRegistry(entry_point_group=…)` scans `vtsearch.<family>` groups. Built-ins win on name clashes; broken entry points warn and are skipped.)

## Phase 6 — Pickle compatibility

**Audit complete — no compatibility shim required.** This phase was scoped on the assumption that saved pickles or weight files would reference `vtsearch.<subpackage>.<Class>` import paths that would need remapping after the move to `vtscore.*`. The audit shows that assumption doesn't hold:

- [x] **Audit:** dataset pickles round-trip through `vtscore/datasets/loader.py` (`pickle.dump`) and `vtscore.security.pickle.safe_pickle_load`. The `safe_pickle_load` allowlist (`_PICKLE_SAFE_CLASSES` in `vtscore/security/pickle.py`) only permits `builtins`, `collections.OrderedDict`, and `numpy` reconstruction helpers — by construction a saved pickle cannot contain a `vtsearch.*` class reference, or it would be rejected on load. `Origin` is stored as `origin.to_dict()` (a plain `dict`), so it round-trips as data; the `Origin` dataclass is rebuilt at load time via `Origin.from_dict()`. Detector labelsets are JSON (`vtscore/detectors/store.py`'s `json.dump`), and MLP weights live only in memory (CLAUDE.md "No Persisted Vectors or MLPs" rule). There is no `weights_compat.py` to use as precedent because there are no persisted weights in the first place.
- [x] **`Unpickler.find_class` remap:** unnecessary. The allowlist already encodes the contract.
- [x] **App-side compat shim re-exporting old paths:** unnecessary for the same reason. The only `vtsearch.*` references in any persisted artifact are in dataset *origin dicts* (`{"importer": "server_folder", "params": ...}`), which use plugin `name` strings (not import paths), so the `vtsearch → vtscore` rename does not affect them.

## Phase 7 — Test split

- [x] Identify tests that don't use the Flask `client` fixture and don't reach into `vtsearch.routes`, `vtsearch.settings`, `vtsearch.auth`, `vtsearch.shim`, `vtsearch.autorun_processors`, or `vtsearch.settings_io`. The 43 files that passed the grep were moved out of `tests/` into `tests_lib/`. Two MIXED files were salvaged: `test_audio.py` (only needed `app_module` for `generate_wav` / `SAMPLE_RATE`, both of which live in `vtscore.media.audio.audio_generator`) and `test_input_spec.py` (imported `app` purely to populate plugin registries, which Phase 5 eager discovery now handles).
- [x] Create `tests_lib/` for library-only tests; keep `tests/` for app tests. Both run from `./run-tests.sh`. `pyproject.toml` adds `tests_lib` to both `testpaths` and `pythonpath`, so `from helpers import …` keeps working in the library tier (a sibling `tests_lib/helpers.py` mirrors `tests/helpers.py`).
- [x] Library tests must pass with Flask unavailable. *Implementation*: `./run-tests.sh vtscore-clean` runs `scripts/check-vtscore-clean.py`, which installs a `MetaPathFinder` that refuses `flask`, `werkzeug`, and `flask_smorest` before pytest collection. All 927 `tests_lib/` tests pass with the hook active. The actual Flask-less virtualenv variant lands in Phase 8 with the real `vtscore` package.
- [x] Conftest fixtures split: `reset_state` and `client` stay in `tests/conftest.py`; `tests_lib/conftest.py` provides `reset_contexts` (smaller — no login provider, no autorun-processor reset, no settings file isolation, no `register_app_config_builder` reinstall) plus the same `_allow_test_tmp_paths` + `_stub_embedding_models` autouse stubs. The library conftest also installs a default library-only `CoreConfig` builder so `CoreConfig.from_settings()` works without the app shim.

## Phase 8 — Physical move

**Shipped.** What landed:

- [x] `vtscore/` created at repo root with its own `__init__.py` (`__version__ = "0.1.0"` placeholder until Phase 9 wires up real semver).
- [x] `git mv` of every library subpackage out of `vtsearch/`: `datasets/`, `embedding/`, `detectors/`, `training/`, `media/`, `converters/`, `exporters/`, `labels/`, `eval/`, `plugins/`, `concurrency/`, `sync/`, `security/`, `utils/`.
- [x] `git mv` of the four library-tier single files: `cli.py`, `cli_pipeline.py`, `cli_progress.py`, `config.py`.
- [x] `state/` split between the two distributions: the six library submodules (`core.py`, `clicks.py`, `votes.py`, `diversity.py`, `diversity_tree.py`, `media_lookup.py`) moved under `vtscore/state/` with a new library-tier `vtscore/state/__init__.py` carrying the public surface; `vtsearch/state/__init__.py` was rewritten as a thin app-tier shim that `from vtscore.state import *`s the library names and layers the proxy view (`medias`, `good_votes`, …) from `vtsearch.shim.state_proxies` on top. Phase 7's plan note about leaving the library submodules under `vtscore/state/` was followed verbatim.
- [x] Repo-wide import rewrite: 3463 `vtsearch.<lib>` occurrences across 352 files → `vtscore.<lib>`, including state submodule references (`vtsearch.state.core` → `vtscore.state.core`, etc.) and the 8 `from vtsearch import config|cli|cli_pipeline|cli_progress` sites that don't go through dotted access. Library plugin entry-point group names renamed to match the new package: `vtsearch.importers` → `vtscore.importers`, `vtsearch.label_importers` → `vtscore.label_importers`, `vtsearch.labelset_sources` → `vtscore.labelset_sources`, `vtsearch.media_sources` → `vtscore.media_sources` (the `vtsearch.settings_*` groups stay vtsearch.* — they remain in `vtsearch/settings_io/`).
- [x] `pyproject.toml`: include both packages in `setuptools.packages.find`, add `vtscore` to coverage source, retarget moved-file ruff per-file ignores; `pyrightconfig.json` adds `vtscore` to `include`. The OpenAPI snapshot was regenerated so docstring references resolve cleanly under the new paths.
- [x] Test result: `./run-tests.sh` → all 3939 tests pass; `./run-tests.sh vtscore-clean` → all 927 library-only tests pass with the Flask-blocking import hook active. Behaviour is identical to pre-move.

Open follow-ups (deferred, captured in the §"Open follow-ups (Phase 8)" section below):

- A separate `vtscore/pyproject.toml` distribution is not yet split out — both packages ship from the single root `pyproject.toml` (declared as the `vtsearch` distribution with both `vtsearch*` and `vtscore*` packages included). That's enough for an internal monorepo workflow; the actual two-distribution split waits for Phase 9.

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
3. ~~**Phase 3, audit: explicit context parameters.**~~ **Shipped.** The state submodules (`vtscore/state/votes.py`, `clicks.py`, `media_lookup.py`, `diversity.py`) and `vtscore/labels/sync.py` now resolve the active context internally via `get_active_*_context()` and operate on `ctx.attr` directly — no library-candidate code imports the module-level proxy names anymore.
4. ~~**Phase 3, finish: move the proxies to the app layer.**~~ **Shipped.** Proxy classes (`_ProxyDict`, `_ProxyList`) and instances (`medias`, `good_votes`, …) moved out of `vtscore/state/core.py` into `vtsearch/shim/state_proxies.py`. `vtsearch/state/__init__.py` re-exports them so app-tier imports (`from vtsearch.state import medias`) keep working. Test imports that reached into `state.core` for proxies were updated to use `vtsearch.state` or `vtsearch.shim.state_proxies`.
5. ~~**Phase 4, filesystem seam.**~~ **Shipped.** All `data/` references now route through `vtscore.config.DATA_DIR` (which is what `CoreConfig.data_dir` snapshots). The five plugin form-field placeholders that previously hardcoded `"data/..."` strings (server_csv_file + server_json_file exporters, server_csv_file + server_json_file label importers, server_json_file labelset source) now build their placeholder/default from `DATA_DIR` via f-strings. Tests in `tests/io/test_csv_webhook_exporters.py` were updated to assert the dynamic value. Side effect: placeholders are now absolute paths (e.g. `/home/user/VTSearch/data/...`) instead of relative — this is more correct in a library context because relative-from-CWD was always brittle.
6. ~~**Phase 8 bridge relocation.**~~ **Shipped.** The body of `CoreConfig.from_settings()` moved out of `vtscore/config.py` into `vtsearch/shim/__init__.py:build_core_config()`. The classmethod is now a thin wrapper that delegates to whatever the app installed via `vtscore.config.register_core_config_builder()`; `vtsearch/shim/register_app_config_builder()` is the app-side wiring (called from `app.py` startup alongside the other shim hooks). Library-only consumers without an app skip `from_settings()` entirely and construct `CoreConfig` directly — they get a clear `RuntimeError` if they try otherwise. `vtscore/config.py` no longer imports `vtsearch.settings`.
7. ~~**Phase 5 plugin DI seam.**~~ **Shipped.** `vtscore/plugins/inventory.py` no longer imports `vtsearch.settings_io.*`. The hardcoded `gather_plugins()` body became a `FamilyProvider` + `register_plugin_family()` registry; library-tier families (importers, exporters, label_importers, labelset_sources, converters, media_sources, media_types, embedders, clippers) self-register at module import, and the app installs the three settings_io families (`settings_importers`, `settings_exporters`, `settings_sources`) at startup via `vtsearch/shim/register_app_plugin_families()`. `FAMILIES` is now a module-level `__getattr__` returning a tuple snapshot of the live registry, so callers see every family that's been registered by the time they import it. `register_family_shortcuts(parser)` iterates the registry directly. No behaviour change — `python app.py --list-plugins` still produces the same 12 families because the shim is wired in `app.py` before the argparse parser is built.
8. ~~**Phase 5 eager registry discovery.**~~ **Shipped.** `PluginRegistry.__init__` now triggers `_ensure_discovered()` by default (`eager=True`), so plugin packages are populated by the time their `__init__.py` returns. Tests that probe the deferred path opt in via `eager=False`. Side fix: `vtscore/converters/runner.py` had a top-level `from vtscore.converters import get_converter` that collided with eager discovery's `discover_modules=True` scan (runner.py sits in the converters directory but isn't a converter plugin); moved that import inside the two functions that use it. Tests that monkeypatched `vtscore.converters.runner.get_converter` were retargeted at `vtscore.converters.get_converter` so the in-function deferred import sees the mock.
9. ~~**Phase 6 pickle compatibility audit.**~~ **Shipped (as no-op).** `safe_pickle_load`'s allowlist (builtins + `collections.OrderedDict` + numpy reconstruction helpers) proves dataset pickles never contain `vtsearch.*` class references; `Origin` round-trips as a dict; detector labelsets are JSON; MLP weights are in-memory only. No `find_class` remap, no compat re-exports — there's nothing to be backwards-compatible with on the persistence side. Plan section updated to reflect the audit findings.
10. ~~**Phase 7 test split.**~~ **Shipped.** 43 library-candidate test files moved from `tests/` to `tests_lib/` via `git mv` (history preserved). New `tests_lib/conftest.py` is app-free / settings-free / Flask-free and installs a library-only `CoreConfig.from_settings()` builder. `pyproject.toml` adds `tests_lib` to both `testpaths` and `pythonpath`; `run-tests.sh` passes both trees to pytest and adds a new `vtscore-clean` subcommand that runs `tests_lib/` under a meta-path import hook (in `scripts/check-vtscore-clean.py`) that refuses `flask` / `werkzeug` / `flask_smorest`. 927 library tests pass with the hook active; the full `tests/ tests_lib/` suite runs 3860 tests in ~70s.
11. ~~**Phase 8 physical move.**~~ **Shipped.** `git mv` of every library subpackage + the four library-tier single files out of `vtsearch/` into a new `vtscore/` root, plus the six state submodules under `vtscore/state/`. The 3463 cross-cutting `vtsearch.<lib>` import / string occurrences across 352 files were rewritten to `vtscore.<lib>` (including the 8 `from vtsearch import config|cli|cli_pipeline|cli_progress` sites the dotted-form rewrite didn't catch). `vtsearch/state/__init__.py` was rewritten as a thin app-tier shim that re-exports every public name from `vtscore.state` and layers the proxy view on top — `from vtsearch.state import medias` keeps working. Library plugin entry-point group names renamed to `vtscore.<family>` (settings_io families stay `vtsearch.<family>` since they remain app-tier). `pyproject.toml` adds `vtscore*` to the package include list; `pyrightconfig.json` adds `vtscore` to `include`. All 3939 app+library tests pass; all 927 library-only tests pass with the Flask-blocking hook active.

## Open follow-ups (Phase 7)

- The library-tier and app-tier conftests each generate their own copy of the test medias (~20 audio files) and stub every embedder. That's intentional — each tier should be self-contained — but it means a single `./run-tests.sh` invocation calls `init_medias()` twice and pays the audio-generation cost in both worker pools.  Cheap (~1s per pool, cached on disk between runs) but worth knowing.
- `tests_lib/conftest.py:pytest_unconfigure` and `tests/conftest.py:pytest_unconfigure` both want to call `os._exit()` at the end of a session.  Today only the app-tier hook fires (because pytest auto-discovers `tests/conftest.py` first when both trees are present), which is fine for the combined run.  Standalone `./run-tests.sh vtscore-clean` triggers the library-tier hook instead; both paths print one summary block.  Watch out if `tests_lib/` ever grows into multiple package roots.
- `tests/api/test_ssrf_validation.py` was moved to `tests_lib/core/test_ssrf_validation.py` because it tests `vtscore.security.url_validation` (library code) and doesn't use the Flask client.  The `core` group placement is a judgment call — security is its own concern, but creating a `security` group for one file isn't worth it.  Revisit if more security tests land.

## Open follow-ups (Phase 8)

- Both packages currently ship from the single root `pyproject.toml` declared as the `vtsearch` distribution (with `setuptools.packages.find` configured to include both `vtsearch*` and `vtscore*`). When/if `vtscore` ever ships as its own PyPI package, that needs a `vtscore/pyproject.toml` of its own and the root config has to slim down to just the app's deps. Defer until a real external consumer asks for it — see Phase 9.
- `vtscore/__init__.py` carries a hardcoded `__version__ = "0.1.0"` placeholder. The real plan (Phase 9) is independent semver — once that lands, replace the placeholder with the resolved value (probably git-derived like `vtsearch.__init__`, or a baked `_version.txt`).
- `vtscore/plugins/inventory.py:374:5` triggers a pyright warning `"FAMILIES" is specified in __all__ but is not present in module (reportUnsupportedDunderAll)`. The name is exposed via module `__getattr__` (intentional — returns a live snapshot of the registry), but pyright doesn't recognise the dynamic export. The existing `# noqa: F822` covers ruff; a `# pyright: ignore[reportUnsupportedDunderAll]` on the `__all__` entry would silence the warning if it becomes noisy. Pre-existing, not a Phase 8 regression.

When you take one of these on, check the box and add a one-line "shipped in #PR" note so the plan tracks reality.
