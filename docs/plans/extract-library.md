# Extract `vtscore` Library Plan

Status: **Phase 0 shipped; Phase 1 mostly landed** — public API surface captured in [`docs/vtscore-api.md`](../vtscore-api.md). `vtsearch/media/` and `vtsearch/state/` are Flask-free; the latter now uses a pluggable resolver hook installed at app startup by `vtsearch/shim/`. Only `vtsearch/detectors/workflow.py` (renamed from `vtsearch/models/training_workflow.py`) still reads `flask.g`. Phase 2 not yet started. Phase 5's entry-point hook landed ahead of the split.

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
- [ ] Add a CI job that runs the full test suite with `flask` *uninstalled* against a candidate library subset, to prove import-cleanliness as the seams land. Initially this job will fail; it becomes the green light for Phase 5.

## Phase 1 — Cut the Flask seam

The library cannot import Flask. Today the leakage outside `routes/` is small:

- [x] `vtsearch/state/core.py` — `_request_*_context()` helpers replaced with module-level resolver callables (`_dataset_context_resolver`, `_detector_context_resolver`) installable via `register_dataset_context_resolver()` / `register_detector_context_resolver()`. Default = `lambda: None`. The Flask-aware versions live in the new `vtsearch/shim/` package and are installed once at `app.py` startup via `register_flask_context_resolvers()`.
- [x] `vtsearch/media/base.py` — formerly imported Flask; now Flask-free. The route layer converts the abstract `MediaResponse` into a `flask.Response` via `media_response_to_flask`. Module-level docstring still *mentions* Flask for orientation, but no `import flask`.
- [ ] `vtsearch/detectors/workflow.py` (`apply_and_retrain`) — reads `flask.g` to override the request-scoped detector context. **Fix**: same approach as `state/core.py`; the context-override should go through the pluggable resolver so library callers can swap contexts without importing Flask.

Audit command (run before declaring this phase done):

```sh
grep -rn "^import flask\|^from flask\|^\s*import flask\|^\s*from flask" \
  vtsearch/{datasets,detectors,embedding,training,media,converters,processors,exporters,labels,eval,plugins,concurrency,state,sync,utils,security}
```

**Exit criteria**: that command returns zero hits.

## Phase 2 — Cut the settings seam

Each library-candidate call site that pulls from `vtsearch.settings` needs to accept its config as an argument instead.

Files to convert (current paths; verified by `grep` over the library-candidate packages):

- `vtsearch/cli.py` — calls `set_settings_path`, `get_autorun_detectors`.
- `vtsearch/cli_pipeline.py` — calls `set_settings_path` (new since the plan was written; same treatment as `cli.py`).
- `vtsearch/datasets/load_pipeline.py` — module-level `from vtsearch.settings import (...)` plus `set_last_embedder_for_media_type` at line 671.
- `vtsearch/datasets/registry.py` — reads `get_saved_datasets_dir`.
- `vtsearch/detectors/labeling_progress.py` — reads `get_autopilot_goal_diversity`.
- `vtsearch/detectors/store.py` — reads `get_detectors_dir`.
- `vtsearch/state/__init__.py` — 8 lazy imports of `vtsearch.settings` (inclusion, calibrate_count, calibration_fraction, safe_thresholds, etc.) routed through state wrappers. These wrappers should become thin delegates to the per-context config object.

Already clean (no `vtsearch.settings` imports today):

- `vtsearch/embedding/loader.py` (was `vtsearch/models/loader.py`).
- `vtsearch/sync/__init__.py` (mentions settings only in docstrings).

Approach:

1. Define a `vtscore.config.CoreConfig` dataclass with the knobs library code actually consumes (`safe_thresholds`, `calibrate_count`, `calibration_fraction`, `enrich_descriptions`, `data_dir`, `saved_datasets_dir`, `detectors_dir`, `autopilot_goal_diversity`, `max_concurrent_dataset_downloads`, `max_concurrent_dataset_embeddings`, etc.).
2. Replace direct `from vtsearch.settings import get_X` calls with reading from a `CoreConfig` argument or attribute of an enclosing context object (`DatasetContext`/`DetectorContext`).
3. App-side: at request boundary, build a `CoreConfig` from `vtsearch.settings` and pass it down. Settings auto-save behaviour stays in the app.
4. The `vtsearch/state/__init__.py` setter wrappers (e.g. `set_inclusion`, `set_calibrate_count`) currently both update the in-memory cache **and** call `vtsearch.settings.set_X` to persist. Library callers should only do the in-memory half; the app-side persistence half moves into a thin shim that the app installs.

**Exit criteria**: the same `grep` from Phase 1 (extended to look for `vtsearch.settings` instead of `flask`) returns zero hits across the library-candidate packages.

## Phase 3 — Cut the global-state seam

Currently `medias`, `good_votes`, `label_history`, etc. are module-level proxies. Library consumers should be able to pass `DatasetContext`/`DetectorContext` explicitly.

- [ ] Audit every public library function and ensure it accepts a context object as a parameter (most already do via the proxy delegation; some implicitly read globals — make those explicit).
- [ ] Keep the proxy module in the *app* layer, not the library. The library exports the context classes; the app exports the proxies that delegate to them via Flask `g` / thread-local.
- [ ] Move `autorun_extractors`, `autorun_localizers` (currently defined as module-level dicts in `vtsearch/state/core.py` and mutated through `vtsearch/state/processors.py`) onto a context-or-config object the app owns. `autorun_detectors` is already an app-side server-tier setting in `vtsearch/settings.py`, so only the two `state/core.py` globals need relocating.

**Exit criteria**: `grep -n "^medias\|^good_votes" vtscore-candidate-paths/` returns zero hits — those names exist only in the app shim.

## Phase 4 — Cut the filesystem seam

The library should never assume a `data/` directory exists at CWD.

- [ ] All `data/` path resolution flows through `CoreConfig.data_dir` or a passed-in `Path`.
- [ ] Embedding cache, model cache, ingestion staging — all parameterised.
- [ ] App default: `data/` relative to the app's run directory (today's behaviour, preserved).

**Exit criteria**: `grep -rn '"data/' vtscore-candidate-paths/` and `grep -rn "Path('data')" vtscore-candidate-paths/` are clean (only constants in `CoreConfig`).

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
3. Search-and-replace `vtsearch.datasets` → `vtscore.datasets` etc., across the codebase.
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
- `state/processors.py`'s autorun-extractor/localizer CRUD operates on globals defined in `state/core.py` — Phase 3 needs to relocate those before `state/` can move into `vtscore/`.
- `plugins/inventory.py` currently imports from `vtsearch.settings_io.*` to enumerate app-side plugin families. After the split, inventory either lives app-side or accepts the app-side registries via DI.

## Risks and open questions

- **Settings sources / labelset sources straddle the boundary.** `SettingsSource` is inherently a user-pref concern (app-side), but `LabelsetSource` is library-side (detector training writes labels). The shared `SyncSource[L,S]` ABC stays in the library; the settings-source registry stays app-side; the labelset-source registry moves to the library. Confirm before Phase 5.
- **Test-media generator** already lives at `tests/fixtures/medias.py` (loaded by `tests/conftest.py`), so it is *not* a runtime concern of either distribution — when Phase 7 splits the test tree, the fixtures move with `tests_lib/` (or stay shared via a `pythonpath` entry).
- **`eval/visualize.py`** pulls in matplotlib — keep as an optional extra (`vtscore[viz]`) to avoid forcing the dep on lean consumers.
- **Heavy ML deps** (torch/transformers/CLAP/CLIP) — declare as required for now; revisit extras (`vtscore[embedders]`) if a consumer asks for a leaner install.

## Order of operations recap

Phases 1 → 4 are independent and can land in parallel PRs. Phase 5 depends on 1–4. Phase 6 can land any time after Phase 0. Phase 7 should land before Phase 8. Phase 8 is one big PR; Phase 9 is post-merge cleanup.

## Open follow-ups (what's left)

Next concrete pieces of work, smallest-first, so a contributor can grab one without reading the whole plan:

1. ~~**Phase 0 inventory doc.** Stub `docs/vtscore-api.md` with the docstring-only sketch of the public surface.~~ **Shipped.** See [`docs/vtscore-api.md`](../vtscore-api.md).
2. ~~**Phase 1, file 1 of 2: `vtsearch/state/core.py`.**~~ **Shipped.** Resolver hook installed at app startup via `vtsearch/shim/register_flask_context_resolvers()`; `state/core.py` now imports nothing from Flask. Full test suite (3662) green.
3. **Phase 1, file 2 of 2: `vtsearch/detectors/workflow.py::apply_and_retrain`.** Currently writes `g._detector_context = det_ctx` to override the request-scoped context. Replace with a context-manager helper from `state/core.py` (e.g. `override_detector_context(ctx)`) that does the override through the same resolver hook. Removes the `from flask import g` line.
4. **Phase 2 scaffold: `vtscore.config.CoreConfig`.** Add the dataclass under `vtsearch/config.py` (or a new `vtsearch/core_config.py`) with the knobs listed in Phase 2 above and a `CoreConfig.from_settings()` classmethod that reads `vtsearch.settings`. No call-site changes yet — landing the type lets follow-up PRs convert one file at a time.
5. **Phase 2, easy wins.** `vtsearch/datasets/registry.py` and `vtsearch/detectors/store.py` each read exactly one settings value (`get_saved_datasets_dir`, `get_detectors_dir`). Convert those two first as the proof-of-pattern, then `detectors/labeling_progress.py`, then the heavier `datasets/load_pipeline.py` and `state/__init__.py`.

When you take one of these on, check the box and add a one-line "shipped in #PR" note so the plan tracks reality.
