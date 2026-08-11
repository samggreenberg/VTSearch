# vtscore Architecture

This document explains how `vtscore` is structured, why each piece exists,
and how the boundary between library and application is drawn. Read
[concepts.md](concepts.md) first if the words *Media*, *Origin*, *LabelSet*,
or *Context* are unfamiliar - this doc assumes them.

## Contents

1. [System overview](#system-overview)
2. [The seven seams](#the-seven-seams)
3. [Resolution chain for "active context"](#resolution-chain-for-active-context)
4. [The CoreConfig bridge](#the-coreconfig-bridge)
5. [Plugin discovery](#plugin-discovery)
6. [Threading model](#threading-model)
7. [The no-persisted-vectors rule](#the-no-persisted-vectors-rule)
8. [Directory map](#directory-map)
9. [Import paths](#import-paths-read-before-copy-pasting)
10. [Dependency direction](#dependency-direction)

---

## System overview

`vtscore` does three things and only three things:

1. **Turn media into embeddings.** Audio, images, text, video, documents go
   in via a `MediaEmbedder`; fixed-dimensional `(D,)` numpy vectors come
   out.
2. **Train a linear (logistic) head on user-labelled embeddings.** Voted-good
   and voted-bad media become `(X, y)`; a `Linear(D, 1)` head plus a calibrated
   threshold comes out. See [`docs/ML.md`](../../docs/ML.md) for why the head is
   linear and where the older MLP path survives.
3. **Score new media against a trained detector.** A new dataset is loaded,
   embedded, and ranked by the detector; the top results are exported.

Everything else in the library exists to make those three flows reliable,
extensible, and reproducible:

- `vtscore.datasets` holds the data formats and loaders.
- `vtscore.embedding` and `vtscore.media` hold the per-format embedder
  implementations.
- `vtscore.training` holds the classifier-head / threshold primitives.
- `vtscore.detectors` orchestrates the full train→score→persist cycle.
- `vtscore.plugins` plus the per-family registries make every component
  swappable and third-party-extensible.
- `vtscore.state`, `vtscore.concurrency`, `vtscore.security` and
  `vtscore.config` are the runtime substrate that keeps the rest honest.

The companion `vtsearch` app wraps all of this with Flask routes, an Angular
SPA, a settings system, and per-user authentication. The library knows
nothing about any of those.

## The seven seams

Splitting `vtsearch` into a library plus an application meant cutting seven
distinct couplings between core code and app code. Every seam below was
detangled before `vtscore` could ship as a standalone package; together
they define the library's interface boundary.

| # | Seam | What used to leak | How the library decouples it |
|---|------|-------------------|-----|
| 1 | **Flask** | `flask.g` reads in `state.core` and `detectors.workflow` | Pluggable context resolvers (`register_dataset_context_resolver`, `register_detector_context_resolver`) plus the current-user resolver (`register_request_user_resolver`, backing `vtscore.state.current_user.get_current_user`); the Flask wiring lives in `vtsearch/shim/` and `vtsearch/auth/`. |
| 2 | **Settings** | Library code read `vtsearch.settings.get_*` directly | `CoreConfig` dataclass carries every knob library code consumes; `CoreConfig.from_settings()` is the bridge the app installs via `register_core_config_builder()`. |
| 3 | **Global state** | Library code imported the `medias`, `good_votes`, … proxies | `DatasetContext` / `DetectorContext` are the library primitives; the proxies stayed app-side in `vtsearch/state_proxies.py`. |
| 4 | **Filesystem** | Hardcoded `"data/"` paths scattered around | Every path routes through `vtscore.config.DATA_DIR` (honouring `$VTSEARCH_DATA_DIR`), snapshotted into `CoreConfig.data_dir`. |
| 5 | **Plugin discovery** | Module scan over `vtsearch.<family>` package paths | Generic `PluginRegistry[T]` walks any package by name + sentinel; library families register under `vtscore.<family>` entry-point groups, app families stay `vtsearch.<family>`. |
| 6 | **Pickle compatibility** | Risk that old pickles referenced `vtsearch.*` classes | Audit confirmed `safe_pickle_load`'s allowlist already prevents app-class references in any saved artefact. No shim needed. |
| 7 | **Test suite** | `tests/` reached into Flask, settings, auth | `tests_lib/` mirrors the tree with Flask-free fixtures; `./run-tests.sh vtscore-clean` runs them under a meta-path hook that refuses `flask` / `werkzeug` / `flask_smorest`. The hook (`tests_lib/flask_blocker.py`) is re-installed at the top of `tests_lib/conftest.py` in every xdist worker, since `sys.meta_path` is per-process and the workers are what import the code under test. |

If you find code in `vtscore/` that violates one of these seams, it's a bug.
The grep commands that enforce each seam live in the git history under the
extract-library commits.

## Resolution chain for "active context"

The library is multi-context: many `DatasetContext`s and `DetectorContext`s
can be loaded at once, and any given call needs to know which one to
operate on. There is **no implicit global default** - that's the point of
the resolution chain.

For each of dataset and detector context, the resolution order is (highest
precedence first):

1. **Explicit override** via `override_detector_context(ctx)`. Used by the
   detector workflow (`vtscore/detectors/workflow.py:apply_and_retrain`)
   when it needs to swap the active detector mid-request without touching
   the request-scoped state. This step exists for detectors only - there is
   no `override_dataset_context`, because nothing has needed to swap the
   active *dataset* inside a call it doesn't own.
2. **Installed resolver hook** via `register_dataset_context_resolver(fn)`
   / `register_detector_context_resolver(fn)`. The Flask shim installs
   resolvers that read from `flask.g` - these are how route handlers see
   the dataset / detector implied by the `X-Dataset-Id` / `X-Detector-Id`
   request headers.
3. **Thread-local** set via `set_thread_dataset_context(ctx)` /
   `set_thread_detector_context(ctx)` (or their save-and-restore
   context-manager forms `thread_dataset_context` / `thread_detector_context`).
   Used by background threads spawned from a request handler (so per-thread
   state doesn't collide).
4. **The request-missing sentinel**, when a host app has registered a
   request-context predicate and the current request named no dataset /
   detector. Reads see an empty context; writes raise
   `RequestMissingContextError` rather than silently polluting shared state.
5. **A process-wide empty fallback context**, for CLI and library callers
   outside any request.

Note what step 5 means: `get_active_context()` and
`get_active_detector_context()` always return a context object, never
`None`. The fallback is deliberately *empty* rather than "the first loaded
one" - but it is shared, so a library caller that skips step 3 is writing
into the same object as every other caller that skipped it. Bind a context
explicitly.

```python
# Library consumer with no Flask, no threads - just call the API directly.
from vtscore.state import (
    DatasetContext,
    register_context,
    set_thread_dataset_context,
)

ctx = DatasetContext(dataset_id="my-dataset")
register_context(ctx)
set_thread_dataset_context(ctx)   # makes ctx the active context
# … library calls now resolve to ctx
```

```python
# Inside a Flask request, the app's resolver hook does this for you.
@app.before_request
def _resolve_context() -> None:
    dataset_id = request.headers.get("X-Dataset-Id")
    g._dataset_context = get_context(dataset_id) if dataset_id else None
```

```python
# Override scope - beats both resolver and thread-local for the duration
# of the block.
from vtscore.state import override_detector_context

with override_detector_context(other_ctx):
    apply_and_retrain(...)  # operates on other_ctx
```

## The CoreConfig bridge

`vtscore.config.CoreConfig` is a dataclass holding every settings knob the
library actually reads:

- ML knobs: `calibrate_count`, `calibration_fraction`,
  `enrich_descriptions`, `inclusion`, `autopilot_goal_diversity`.
- Filesystem knobs: `data_dir`, `saved_datasets_dir`, `detectors_dir`.
- Concurrency knobs: `max_concurrent_dataset_downloads`,
  `max_concurrent_dataset_embeddings`.

Library code calls `CoreConfig.from_settings()` to get a populated config
for the current request / thread. That classmethod is a thin wrapper that
delegates to whatever builder the app installed via
`register_core_config_builder()`. The actual implementation lives at
`vtsearch/shim/__init__.py:build_core_config()` and reads from
`vtsearch.settings`.

**Library-only consumers don't need a builder.** They construct `CoreConfig`
directly and pass it where it's needed:

```python
from vtscore.config import CoreConfig

config = CoreConfig(
    data_dir=Path("/var/lib/myapp/data"),
    saved_datasets_dir=Path("/var/lib/myapp/datasets"),
    detectors_dir=Path("/var/lib/myapp/detectors"),
    max_concurrent_dataset_downloads=2,
    max_concurrent_dataset_embeddings=1,
    autofind_detectors=(),
    dataset_max_age_days=None,
    calibrate_count=2,
    calibration_fraction=0.5,
    enrich_descriptions=False,
    autopilot_goal_diversity=8,
    inclusion=0,
)
```

If a library consumer calls `CoreConfig.from_settings()` without first
installing a builder, the method raises a clear `RuntimeError`. That's
intentional - silent fallback to "some default" would mask integration
bugs.

## Plugin discovery

Every plugin family in `vtscore` follows the same shape:

1. A base ABC (`DatasetImporter`, `LabelsetExporter`, `MediaEmbedder`, …)
2. A sentinel attribute name (`IMPORTER`, `EXPORTER`, `EMBEDDER`, …)
3. A `PluginRegistry[T]` constructed with that sentinel, eager by default
4. An optional `importlib.metadata` entry-point group (`vtscore.<family>`)
   that third-party packages can register under

Discovery happens at registry-construction time - by the time
`vtscore.datasets.importers.__init__` returns, every importer module in
the package has been imported, every `IMPORTER` sentinel harvested, and
every `vtscore.importers` entry point loaded. Built-ins win on name clash;
broken entry points warn and are skipped.

See [packages/plugins.md](packages/plugins.md) for the full mechanics and
[extending/README.md](extending/README.md) for plugin-authoring guides.

## Threading model

The library is designed to be used from multiple threads, with these
ground rules:

- **All mutable state is RLock-protected.** The single `_state_lock` in
  `vtscore/state/core.py` covers `DatasetContext` and `DetectorContext`
  mutation. Using an RLock (not a plain Lock) lets one public function
  call another safely - for example `clear_all()` calls `clear_medias()`
  and `clear_votes()` while holding the lock.
- **Thread-local progress callbacks.** Both `vtscore.media` (per-thread
  via `set_thread_progress_callback`) and `vtscore.concurrency.progress`
  (per-thread via `set_thread_progress`) let parallel ingestion threads
  report progress without clobbering each other.
- **Per-thread context binding.** `set_thread_dataset_context()` /
  `set_thread_detector_context()` are how background threads tell the
  library which context they're operating on. The dataset-load and
  learned-sort job managers do this automatically when they spawn workers.
- **Deterministic training.** `train_model` uses a local
  `torch.Generator` seeded with the caller-supplied seed (default 42) and
  wraps `nn.Dropout` initialisation in `torch.random.fork_rng()`, so
  parallel training calls don't race on the global RNG.

What the library **doesn't** do is take responsibility for serialising
your job queue. If you want to train two detectors at once, you can - but
you allocate the threads and supply the contexts.

## The no-persisted-vectors rule

Read this once and the persistence story makes sense.

**Trained model weights and embeddings live in memory only.** They are never
serialised to disk, never written to `data/settings.json`, never embedded
inside a detector JSON file, never persisted anywhere.

The reasoning:

- **Embedder version drift** would silently break everything. If a saved
  pickle held an `nn.Sequential` with weights, and the embedder model
  version changed between save and load, the dimensions would still match
  but the semantics wouldn't. By forcing every load to re-derive embeddings
  from origins against the *active* embedder, drift is impossible by
  construction.
- **Cross-machine portability** is automatic. An origin like
  `{"importer": "server_folder", "params": {"path": "/data/audio"}}` is a
  reference, not a binary. Move the labelset to another machine that has
  the same files, point it at a compatible dataset, and the detector
  retrains itself.
- **Disk usage stays small.** Detectors are JSON files of a few KB. The
  expensive embeddings are the consumer's responsibility (or they're
  rebuilt on demand).

The single exception is **dataset pickle files**, which are by design a
`(medias, embeddings)` snapshot - they *are* the dataset, not a cache.
They round-trip through `pickle.dump` / `safe_pickle_load`, with the
unpickler's allowlist preventing any non-numpy class reference.

## Directory map

```
vtscore/
├── __init__.py                         # __version__ (manual semver)
├── config.py                           # CoreConfig + DATA_DIR + model IDs
├── cli.py                              # autodetect entry points
├── cli_pipeline.py                     # YAML pipeline parser
├── cli_progress.py                     # text / NDJSON progress emit
├── docs/                               # this directory
├── datasets/                           # origins, labelsets, loaders, importers
│   ├── importers/                      # IMPORTER-sentinel auto-discovery
│   │   ├── server_folder/              # local filesystem importer
│   │   ├── http_archive/               # URL-fetched archive importer
│   │   ├── combine_datasets/           # union of saved datasets
│   │   ├── synthetic/                  # deterministic synthetic media
│   │   ├── demo/                       # bundled demo datasets
│   │   └── …                           # see packages/datasets.md
│   ├── sources/                        # MediaSource resolvers (local_folder, http_archive, pullwrest)
│   ├── origin.py                       # Origin dataclass
│   ├── labelset.py                     # LabelSet + LabeledElement
│   ├── loader.py                       # façade re-exporting loader_folder/loader_pickle/loader_demo
│   ├── load_pipeline.py                # ConcurrencyGate, post-load fix-ups
│   ├── registry.py                     # on-disk dataset registry (saved_datasets_dir)
│   ├── split.py                        # train/test split
│   └── …
├── media/                              # MediaType / MediaEmbedder / MediaClipper registries
│   ├── base.py                         # the ABCs
│   ├── audio/                          # MEDIA_TYPE + EMBEDDER + CLIPPERS sentinels
│   ├── image/
│   ├── text/
│   ├── video/
│   └── document/
├── embedding/                          # embedder façade + cached matrix
├── training/                           # classifier head / thresholds / SVM / region-similarity
├── detectors/                          # full detector lifecycle
│   ├── registry.py                     # in-memory detector registry
│   ├── store.py                        # JSON labelset persistence
│   ├── training.py                     # train_and_threshold, train_and_score
│   ├── workflow.py                     # apply_and_retrain
│   ├── resolver.py                     # origin → file → embedding
│   ├── label_sync.py / label_restoration.py / dataset_sync.py / media_seeding.py
│   ├── labelset_elements.py / labelset_training.py
│   └── labeling_progress.py            # per-step model cache + stopping conditions
├── eval/                               # offline evaluation runner + metrics
├── converters/                         # audio↔image/text, video→audio/image, etc.
├── exporters/                          # EXPORTER-sentinel auto-discovery
├── labels/                             # LabelImporter + LabelsetSource families
├── plugins/                            # PluginRegistry + sentinel scanner + entry-points
├── state/                              # DatasetContext, DetectorContext, ops
├── sync/                               # SyncSource[L,S] ABC
├── concurrency/                        # AsyncJob / JobManager / progress trackers
├── security/                           # path / URL validation, safe pickle
└── utils/                              # build_media_hit, synthetic media generators
```

## Import paths (read before copy-pasting)

The package guides under [`packages/`](README.md#package-reference) group
symbols **by intent**, not by import path. A package's `__init__.py` is not
a re-export contract: several packages expose nothing at their root, and two
have no `__init__.py` at all (PEP 420 namespace packages). When an example
elsewhere in these docs shows a symbol without a full dotted path, this table
is the authority on where to import it from.

| Package | How to import |
|---------|---------------|
| `vtscore.config` | Plain module (`vtscore/config.py`): `from vtscore.config import CoreConfig, DATA_DIR`. |
| `vtscore.datasets` | Re-exports the loader / importer-registry / `Origin` / `LabelSet` surface. Per-dataset demo metadata helpers live in their own submodules. |
| `vtscore.media` | Re-exports the ABCs (`MediaType`, `MediaEmbedder`, `MediaClipper`, the processor ABCs) and the registry helpers (`get`, `get_embedder`, `get_clipper`, `set_progress_callback`, …). |
| `vtscore.embedding` | Re-exports the embed / loader / matrix helpers. |
| `vtscore.training` | Re-exports `build_model` / `train_model` and the threshold helpers. `SVMClassifier` is at `vtscore.training.svm`; region helpers at `vtscore.training.region_similarity`. |
| `vtscore.detectors` | **`__init__` is a docstring only - no re-exports.** Always use the submodule: `vtscore.detectors.registry.list_detectors`, `vtscore.detectors.workflow.apply_and_retrain`, `vtscore.detectors.store`, … |
| `vtscore.eval` | Re-exports the top-level runners (`run_eval`, `run_voting_iterations_eval`, `simulate_voting_iterations`) plus `compute_metrics` and `EvalQuery`. The per-sort runners are **not** re-exported: `vtscore.eval.runner.eval_learned_sort` / `eval_text_sort`. Metric dataclasses live at `vtscore.eval.metrics`. |
| `vtscore.converters` | `get_converter`, `list_converters`, plus the built-in converter classes. |
| `vtscore.exporters` | `get_exporter`, `list_exporters`. |
| `vtscore.labels` | **Empty `__init__` - no re-exports.** Importers: `vtscore.labels.importers.get_label_importer` / `list_label_importers`. Sources: `vtscore.labels.sources.get_labelset_source` / `list_labelset_sources`. Sync: `vtscore.labels.sync`. |
| `vtscore.plugins` | `PluginBase`, `PluginField`, `PluginRegistry`, `make_plugin_registry`, the field-type enums. |
| `vtscore.state` | Contexts (`DatasetContext`, `DetectorContext`), registries, and the vote / click ops. The `medias` / `good_votes` proxies are **app-side**, in `vtsearch.state_proxies`. |
| `vtscore.sync` | `SyncSource`. |
| `vtscore.concurrency` | **Namespace package (no `__init__.py`).** Always use the submodule: `vtscore.concurrency.progress.ProgressTracker`, `vtscore.concurrency.async_jobs.JobManager`, `vtscore.concurrency.memory_budget.cap_workers_by_memory`. |
| `vtscore.security` | **Namespace package (no `__init__.py`).** Always use the submodule: `vtscore.security.pickle.safe_pickle_load` / `RestrictedUnpickler`, `vtscore.security.path_validation.validate_server_filepath`, `vtscore.security.url_validation.validate_url`. |
| `vtscore.utils` | **`__init__` is a docstring only - no re-exports.** Always use the submodule: `vtscore.utils.hits.build_media_hit`, `vtscore.utils.hashing`, `vtscore.utils.scores`, `vtscore.utils.synthetic`. |
| `vtscore.cli` | Plain modules: `vtscore.cli`, `vtscore.cli_pipeline`, `vtscore.cli_progress`. |

`vtsearch.state` is an app-tier shim that re-exports `vtscore.state` plus the
request-scoped proxy views. Library code should import `vtscore.state`
directly; app code may use either.

## Dependency direction

The dependency graph is strictly one-way: `vtsearch` depends on `vtscore`,
never the reverse.

```
┌─────────────────────────────────────────────┐
│  vtsearch/  (Flask app)                     │
│  ├── routes/                                │
│  ├── settings.py                            │
│  ├── auth/                                  │
│  ├── shim/   ← installs hooks into vtscore  │
│  └── …                                      │
└──────────────────┬──────────────────────────┘
                   │ imports
                   ▼
┌─────────────────────────────────────────────┐
│  vtscore/  (library)                        │
│  ├── config / state / plugins / sync        │
│  ├── datasets / media / embedding           │
│  ├── training / detectors / eval            │
│  ├── converters / exporters / labels        │
│  ├── concurrency / security / utils         │
│  └── cli / cli_pipeline / cli_progress      │
└─────────────────────────────────────────────┘
```

The app uses five categories of hooks to inject app-side behaviour into
the library without the library knowing:

1. **Context resolvers** - `register_dataset_context_resolver`,
   `register_detector_context_resolver`. Wired in `vtsearch/shim/`.
2. **`CoreConfig` builder** - `register_core_config_builder`. Maps
   `vtsearch.settings` to `CoreConfig`. Wired in `vtsearch/shim/`.
3. **Plugin families** - `register_plugin_family(name, provider)`. Lets
   the app contribute `settings_importers`, `settings_exporters`,
   `settings_sources` to the registry that library tools (like
   `python app.py --list-plugins`) enumerate. Wired in `vtsearch/shim/`.
4. **Current-user resolver** - `register_request_user_resolver`. Tells
   `vtscore.state.current_user.get_current_user()` how to read the
   *request-scoped* user; `vtsearch/auth/` wires it to `flask.g.user` at
   import time. Below it sit the thread-local (`thread_user`) and the
   `"default"` fallback, both Flask-free.
5. **Achievement recorders** -
   `vtscore.achievements_hooks.register_achievement_recorder(event, fn)`
   for the `vote` / `dataset_load` / `detector_import` / `find` events the
   library raises. The counters are per-user settings state, so the
   recording itself lives in `vtsearch.achievements`. Wired in
   `vtsearch/shim/`.

If you're embedding `vtscore` in your own application, you'll typically
install your own variants of these hooks. None of them is required -
the library has working defaults for all five (no context, no
`from_settings()` builder, no app-side plugin families, every user is
`"default"`, and achievement events that no-op).

**The dependency direction is enforced by a test.**
`tests_lib/core/test_library_layering.py` walks the AST of every
`vtscore` module and fails on any `import vtsearch`, at any nesting
depth. Lazy function-level imports were how the rule kept breaking: the
module still imported cleanly and the inverted dependency only bit at
call time, in exactly the Flask-free deployment the tier exists for. A
short allowlist there carries the remaining imports with their rationale
(two optional `try`/`except`-guarded ones, plus `security/path_validation.py`,
which still reaches for the `LoginProvider` abstraction); add a hook
rather than a sixth category.
