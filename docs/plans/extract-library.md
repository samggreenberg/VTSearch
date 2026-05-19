# Extract `vtscore` Library Plan

Goal: split VTSearch into two distributions in one repo:

- **`vtscore`** — reusable Python library: dataset origins, MediaSources, clippers/croppers, embedders, MLP/detector training and scoring, evaluation. No Flask, no Angular, no auto-writing JSON configs.
- **`vtsearch`** — the Flask + Angular application that wraps `vtscore`. Owns user-facing concerns: HTTP routes, auth, persistent user preferences, the SPA.

The expensive work is **introducing seams in the current monolith**. Once `vtsearch` itself runs cleanly through those seams with no behaviour change, the actual `git mv` is mechanical.

## Naming

- Library import name: `vtscore` (recommended). Alternatives considered and rejected: `VTSearchLib` (non-idiomatic camelCase), `vtsearch.lib` (inverts the dependency direction — the app depends on the library, not vice versa), `vtsearch.core` namespace package (works but adds packaging complexity for marginal benefit).
- App import name: stays `vtsearch`.
- PyPI distribution names if/when published: `vtscore` and `vtsearch`.

## Phase 0 — Preparation (non-blocking, do anytime)

- [x] Inventory the actual public surface that external consumers would call. Capture as a docstring-only API sketch in `docs/vtscore-api.md`. This is the contract the refactor must preserve. *(Landed: `docs/vtscore-api.md`. Also flagged that the codebase has moved on from `models/` → `detectors/`/`embedding/`/`training/` and that `concurrency/` and `security/` have graduated; the "final shape" section below should be updated during Phase 1 to match.)*
- [ ] Add a CI job that runs the full test suite with `flask` *uninstalled* against a candidate library subset, to prove import-cleanliness as the seams land. Initially this job will fail; it becomes the green light for Phase 5. *(Caveat: VTSearch has no GitHub Actions workflows today — `./run-tests.sh` is the source of truth per CLAUDE.md. The equivalent gate here is a new `./run-tests.sh vtscore-clean` mode that runs library-candidate tests in a venv with Flask absent; revisit when Phase 5 is ready.)*

## Phase 1 — Cut the Flask seam

The library cannot import Flask. Today the leakage outside `routes/` is small:

- `vtsearch/state/core.py` — proxy dicts read `flask.g`. **Fix**: keep the proxy class itself but parameterise the "current context" lookup via a pluggable resolver. Default resolver = thread-local. App registers a Flask-aware resolver at startup that reads `g`.
- `vtsearch/media/base.py` — uses Flask. **Fix**: identify what it actually needs (likely a URL builder or request URL); accept it as a constructor arg or move that concern to the route layer.
- `vtsearch/models/training_workflow.py` — uses Flask. **Fix**: same approach; lift Flask-aware bits to a thin app-side wrapper.

**Exit criteria**: `grep -rn "flask" vtscore-candidate-paths/` returns zero hits.

## Phase 2 — Cut the settings seam

Today 8 non-route modules import `vtsearch.settings` directly. Each call site needs to accept its config as an argument instead of pulling from the global settings module.

Files to convert:

- `vtsearch/cli.py`
- `vtsearch/datasets/load_pipeline.py`
- `vtsearch/datasets/registry.py`
- `vtsearch/models/loader.py`
- `vtsearch/models/labeling_progress.py`
- `vtsearch/models/detector_store.py`
- `vtsearch/state/__init__.py`
- `vtsearch/sync/__init__.py`

Approach:

1. Define a `vtscore.config.CoreConfig` dataclass with the knobs library code actually consumes (`safe_thresholds`, `calibrate_count`, `calibration_fraction`, `enrich_descriptions`, `data_dir`, `max_concurrent_dataset_downloads`, `max_concurrent_dataset_embeddings`, etc.).
2. Replace direct `from vtsearch.settings import get_X` calls with reading from a `CoreConfig` argument or attribute of an enclosing context object (`DatasetContext`/`DetectorContext`).
3. App-side: at request boundary, build a `CoreConfig` from `vtsearch.settings` and pass it down. Settings auto-save behaviour stays in the app.

**Exit criteria**: library candidate paths import nothing from `vtsearch.settings`.

## Phase 3 — Cut the global-state seam

Currently `medias`, `good_votes`, `label_history`, etc. are module-level proxies. Library consumers should be able to pass `DatasetContext`/`DetectorContext` explicitly.

- [ ] Audit every public library function and ensure it accepts a context object as a parameter (most already do via the proxy delegation; some implicitly read globals — make those explicit).
- [ ] Keep the proxy module in the *app* layer, not the library. The library exports the context classes; the app exports the proxies that delegate to them via Flask `g` / thread-local.
- [ ] Move `autorun_detectors`, `autorun_extractors`, `autorun_localizers` (currently global module state in `vtsearch/state/__init__.py`) onto a context-or-config object the app owns.

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
2. `git mv` library subpackages into it (`datasets/`, `models/`, `media/`, `converters/`, `processors/`, `exporters/`, `labels/`, `eval/`, `cli.py`, `config.py`, plus the relevant `utils/` modules).
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
├── datasets/              # origins, labelsets, loaders, importers, sources, ingestion, split
├── models/                # embeddings, MLP, detector, diversity_tree, resolver, weights_compat
├── media/                 # audio/image/text/video/document type plugins
├── converters/            # document→image, video→audio, etc.
├── processors/            # processor importers
├── exporters/             # results exporters (file/CSV/webhook/email)
├── labels/                # label importers + labelset sync sources
├── eval/                  # evaluation runner, metrics, visualisation
└── utils/                 # contexts, progress, paths, ffmpeg, hits, registry, sync_source ABC,
                           # audio_generator, synthetic, url_validation
```

### `vtsearch/` (application)

```
vtsearch/
├── app.py                 # Flask entry point
├── settings.py            # auto-saving JSON user prefs
├── settings_factory.py
├── settings_io/           # settings import/export plugins
├── auth/                  # LoginProvider ABC + default impl
├── routes/                # all Flask blueprints
├── medias.py              # startup test-media generator
├── shim/                  # NEW: glue between Flask g and vtscore contexts;
│                          #      builds CoreConfig from settings on each request
frontend/                  # Angular SPA source
static/                    # built Angular output
```

## Risks and open questions

- **Settings sources / labelset sources straddle the boundary.** *Resolved (Phase 0 review):* `SettingsSource` is a user-pref concern and stays app-side. `LabelsetSource` ships in `vtscore.labels` — pulling labels in from external systems is a core library affordance, and the registry (`get_labelset_source` / `list_labelset_sources`) ships with it. The shared `SyncSource[L,S]` ABC remains in `vtscore.sync`.
- **`medias.py` test-media generator** is used both at startup (app concern) and by `conftest.py`. The generator already moved to `tests/fixtures/medias.py` and is not part of the runtime app — no Phase action needed.
- **`eval/visualisation.py`** *Resolved (Phase 0 review):* plotting (`plot_eval_results`, `plot_voting_iterations`) is presentation, not computation, and stays in `vtsearch/`. The library exports the data (`DatasetResult`, `QueryMetrics`, `LearnedSortMetrics`, `format_results_json`); the app renders it. No `vtscore[viz]` extra needed.
- **`autorun_*` global state.** *Resolved (Phase 0 review):* the registry of which processors to autorun (`autorun_detectors`, `autorun_extractors`, `autorun_localizers` and their CRUD) is *policy* and stays app-side. The library keeps the `Processor` / `Detector` / `Localizer` / `Extractor` ABCs plus the code that applies them; the app passes in the resolved list. Phase 3 wires this as an explicit argument.
- **Per-thread progress callback on `vtscore.media`.** *Resolved (Phase 0 review):* mirror `vtscore.concurrency.progress.set_thread_progress` — add `set_thread_progress_callback` that takes priority when set on the calling thread, keep `set_progress_callback` as the global default. Library consumers running multi-threaded ingestion won't have one thread clobber another's callback.
- **Heavy ML deps** (torch/transformers/CLAP/CLIP) — declare as required for now; revisit extras (`vtscore[embedders]`) if a consumer asks for a leaner install.

## Order of operations recap

Phases 1 → 4 are independent and can land in parallel PRs. Phase 5 depends on 1–4. Phase 6 can land any time after Phase 0. Phase 7 should land before Phase 8. Phase 8 is one big PR; Phase 9 is post-merge cleanup.
