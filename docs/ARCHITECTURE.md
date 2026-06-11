# VTSearch Architecture

This document is for developers who want to **understand, evaluate, and
selectively extract** components from VTSearch.  It maps the module
structure, dependency graph, and public APIs so you can quickly identify
which pieces you need and how to pull them out.

## Table of Contents

1. [What VTSearch does](#what-vtsearch-does)
2. [Directory map](#directory-map)
3. [Dependency graph](#dependency-graph)
4. [Extractability matrix](#extractability-matrix)
5. [How to extract specific components](#how-to-extract-specific-components)
6. [Plugin architecture details](#plugin-architecture-details)
7. [State management](#state-management) (includes [Multi-dataset support](#multi-dataset-support))
8. [Authentication and multi-user support](#authentication-and-multi-user-support)
9. [Element-level origin tracking](#element-level-origin-tracking)

---

## What VTSearch does

VTSearch is a trainable media search tool. The thing the user searches
*with* is a **detector**: a small ranker that scores every item in a
dataset by how well it matches. Detectors come from two places;
either trained in the UI from good/bad votes in a labeling pass, or
imported/loaded from disk and applied as-is. The architecture combines:

- **Detectors (learned search)**: a small MLP trained on user votes
  to predict good/bad labels. This is the primary search mechanism.
  Detectors are persisted as **labelsets** (origin info + labels; never
  weights; weights are an in-memory artifact, re-derived on demand from
  origins and the active embedder).
- **Semantic sort (text-similarity search)**: LAION-CLAP (audio),
  SigLIP (images), X-CLIP (video), E5-base-v2 (text) for
  embedding-based similarity search, with alternative embedders
  available (CLAP Music, BGE). Used to seed a detector during the
  training loop, or as a quick stand-alone search.
- **Flask web UI**: Angular SPA frontend with a REST API.
- **Plugin systems**: nine auto-discovered plugin families: dataset
  importers, results exporters, label importers, settings importers/
  exporters/sources, labelset sources, media converters, and media
  sources.

---

## Directory map

The codebase is split into two top-level Python packages: **`vtscore/`**
(the Flask-free library tier; everything ML, media, dataset, and
plugin-related) and **`vtsearch/`** (the Flask app tier; routes,
settings, auth, and the app-side state shim).

```
VTSearch/
├── app.py                          Flask entry point, CLI arg parsing, startup
│
├── vtscore/                        Library tier; no Flask dependency
│   ├── config.py                   Constants (sample rates, paths, model IDs)
│   ├── cli.py                      CLI autodetect workflow
│   ├── cli_pipeline.py             Pipeline YAML loader
│   ├── cli_progress.py             CLI progress bars
│   │
│   ├── media/                      Media type, embedder, clipper + processor ABCs
│   │   ├── base.py                 MediaType, MediaEmbedder, MediaClipper ABCs
│   │   ├── processors.py           Processor, Detector, Localizer, Extractor ABCs
│   │   ├── embedder.py             MediaEmbedder shared helpers (media_from_path, etc.)
│   │   ├── clipper.py              Shared clipper logic
│   │   ├── audio/                  Audio media type, embedders (CLAP, CLAP-Music, CLAP-General,
│   │   │                           AST, Whisper), clippers, SpeechExtractor
│   │   ├── image/                  Image media type, embedders (SigLIP default; SigLIP2, CLIP,
│   │   │                           DINOv2, DINOv3, EUPE; each with single + patch variants),
│   │   │                           clippers, ImageClassExtractor, FaceLocalizer, OCRExtractor
│   │   ├── text/                   Text media type, embedders (E5 default, BGE), clippers
│   │   ├── video/                  Video media type, embedders (X-CLIP default, LanguageBind,
│   │   │                           VideoMAE), clippers
│   │   └── document/               Document media type (no embedder; convert first), clipper
│   │
│   ├── converters/                 Media type converters (auto-discovered via CONVERTER sentinel)
│   │   ├── audio2image.py          Mel/CQT spectrogram rendering
│   │   ├── audio2text.py           Whisper ASR transcription
│   │   ├── document2image.py       PDF page rendering
│   │   ├── document2text.py        Text extraction from documents
│   │   ├── image2text.py           OCR (PaddleOCR)
│   │   ├── video2audio.py          Audio track extraction
│   │   └── video2image.py          Frame sampling
│   │
│   ├── training/                   Generic learned-sort primitives (no Flask, no state)
│   │   ├── mlp.py                  build_model, train_model (pure PyTorch)
│   │   ├── thresholds.py           GMM / cross-calibration / safe-threshold helpers
│   │   ├── svm.py                  SVM trainer prototype
│   │   └── region_similarity.py    Region-aware cosine similarity scoring
│   │
│   ├── embedding/                  Embedder façades and torch runtime
│   │   ├── helpers.py              embed_audio_file / embed_image_file / embed_text_query / …
│   │   ├── matrix.py               Cached contiguous (N, D) embedding matrix on DatasetContext
│   │   └── loader.py               initialize_models, smart_preload_in_background
│   │
│   ├── detectors/                  Detector lifecycle; resolve→embed→train pipeline
│   │   ├── registry.py             In-memory detector registry
│   │   ├── store.py                On-disk labelset/query store
│   │   ├── training.py             Vote-aware training, origin-based training
│   │   ├── workflow.py             apply-labels-and-retrain orchestration (uses flask.g)
│   │   ├── resolver.py             Origin → file + embedding resolution
│   │   ├── label_sync.py           Sync labels to loaded detector
│   │   ├── label_restoration.py    Label restoration
│   │   ├── labelset_elements.py    Labelset element materialisation
│   │   ├── labelset_training.py    Cross-dataset MLP training
│   │   ├── dataset_sync.py         Sync detectors when a dataset loads
│   │   ├── media_seeding.py        Media seeding utilities
│   │   └── labeling_progress.py    Per-step MLP cache + stability analysis
│   │
│   ├── datasets/                   Dataset loading, downloading, ingestion
│   │   ├── origin.py               Origin dataclass (per-element provenance)
│   │   ├── labelset.py             LabelSet / LabeledElement (labeled data with origins)
│   │   ├── loader.py               Public façade + re-exports
│   │   ├── loader_folder.py        load_dataset_from_folder + chunked variant
│   │   ├── loader_pickle.py        load_dataset_from_pickle + chunked + sidecars
│   │   ├── loader_demo.py          load_demo_dataset, _stamp_demo_origin
│   │   ├── load_pipeline.py        Background-task orchestration, ConcurrencyGate, clip fix-up
│   │   ├── registry.py             Persistent dataset registry (data/dataset_registry.json)
│   │   ├── downloader/             Demo dataset downloaders (audio, image, video, text, docs)
│   │   ├── sources/                MediaSource abstraction (local_folder, http_archive, pullwrest);
│   │   │                           all fetch/resolve ops return FetchedItem (path + optional
│   │   │                           embedding, embedder_name, extra metadata)
│   │   └── importers/              Plugin importers (server_folder, server_files, local_folder,
│   │                               local_files, pickle, http_archive, combine_datasets,
│   │                               demo, synthetic, recaller)
│   │
│   ├── exporters/                  Results exporters (server_json_file, server_csv_file,
│   │                               email_smtp, webhook, gui, holder)
│   │
│   ├── labels/                     Label importers, sync sources, sync utilities
│   │   ├── importers/              server_json_file, server_csv_file, holder
│   │   ├── sources/                server_json_file (bidirectional label sync)
│   │   └── sync.py                 sync_to/from_labelset_source utilities
│   │
│   ├── eval/                       Evaluation framework
│   │   ├── __main__.py             CLI entry point (python -m vtscore.eval)
│   │   ├── runner.py               run_eval() orchestrator
│   │   ├── metrics.py              mAP, P@k, R@k, F1 calculations
│   │   ├── visualize.py            Matplotlib chart generation
│   │   └── voting_iterations.py    Voting-iteration simulation
│   │
│   ├── concurrency/                Async jobs, memory budgeting, progress tracking
│   │   ├── async_jobs.py           AsyncJob, JobManager, eval_jobs, learned_sort_jobs
│   │   ├── memory_budget.py        cap_workers_by_memory
│   │   └── progress.py             ProgressTracker, update_progress, cancel_dataset_progress
│   │
│   ├── state/                      Multi-dataset / multi-detector global state (library tier)
│   │   ├── core.py                 DatasetContext, DetectorContext, _state_lock, context registries
│   │   ├── votes.py                toggle_vote / apply_label / clear_votes
│   │   ├── clicks.py               Vote click-time tracking
│   │   ├── diversity.py            Diversity tree construction and sampling
│   │   └── media_lookup.py         Origin-keyed lookup, collapse_duplicates
│   │
│   ├── plugins/                    PluginBase, PluginField, PluginRegistry (shared plugin infra)
│   ├── security/                   Path/URL/pickle safety (path_validation, url_validation, pickle)
│   ├── sync/                       SyncSource[LoadT, SaveT] generic base class
│   └── utils/                      Shared helpers: hits.py (build_media_hit), synthetic/
│
├── vtsearch/                       Flask app tier (imports Flask; not library-safe)
│   ├── settings.py                 Persistent settings (server tier + per-user tier)
│   ├── settings_factory.py         Accessor factories for settings.py
│   ├── settings_models.py          Marshmallow schema helpers for settings
│   ├── achievements.py             Achievement state management
│   ├── autorun_processors.py       autorun_extractors / autorun_localizers CRUD
│   ├── logging_config.py           Logging setup
│   ├── openapi_postprocess.py      OpenAPI schema post-processing
│   │
│   ├── auth/                       LoginProvider ABC, DefaultLoginProvider, get_current_user(),
│   │                               get_user_data_dir()
│   │
│   ├── state/                      App-tier state shim; re-exports vtscore.state.* and adds
│   │                               proxy view (medias, good_votes, bad_votes, …) from shim/
│   │
│   ├── shim/                       state_proxies.py; _ProxyDict / _ProxyList per-request
│   │                               resolution (checks flask.g, falls back to thread-local)
│   │
│   ├── schemas/                    Marshmallow schemas for API serialisation
│   │
│   ├── settings_io/                Settings import/export/sync plugins (vtsearch-tier)
│   │   ├── importers/              local_json_file, server_json_file
│   │   ├── exporters/              local_json_file, server_json_file
│   │   └── sources/                server_json_file (bidirectional settings sync)
│   │
│   └── routes/                     Flask blueprints; all HTTP request handling
│       ├── _shared.py              Shared route helpers (request parsing, JSON safety)
│       ├── auth.py                 /api/auth/status
│       ├── main.py                 Root route, favicon, logo
│       ├── sorting.py              Text/learned/example sort, diversity
│       ├── eval.py                 Evaluation and labeling progress routes
│       ├── events.py               SSE event stream (/api/events)
│       ├── file_browser.py         File browser API (/api/file-browser/*)
│       ├── health.py               Health check (/api/health)
│       ├── jobs.py                 Job management (/api/jobs/*)
│       ├── sessions.py             Session management (/api/sessions/*)
│       ├── achievements.py         Achievement routes (/api/achievements/*)
│       ├── datasets/               Dataset routes; listings, load, staging, registry, status, ui
│       ├── detectors/              Detector routes; crud, labels, registry, scoring, find
│       ├── processors/             Processor routes; crud, scoring (extractors/localizers)
│       ├── media/                  Media routes; list, server, embed
│       ├── labels/                 Label routes; vote, importers, exporters
│       └── settings/               Settings routes; api, io, sources
│
├── static/                         Angular build output (HTML + CSS + JS)
├── frontend/                       Angular SPA source (components, services, SCSS)
├── tests/                          App-tier test suite (uses Flask client, vtsearch.*)
└── tests_lib/                      Library-tier test suite (Flask-import-clean, vtscore.*)
```

---

## Dependency graph

Arrows point from dependent → dependency.  Modules on the left import
modules on the right.

```
┌────────────────────────────────────────────────────────────┐
│                     Flask / HTTP layer                     │
│                                                            │
│  app.py ──► vtsearch/auth, routes/* ──► vtscore/state,     │
│                │                         vtscore/concurrency │
│                ├──► vtscore/embedding/, training/, detectors/ │
│                ├──► vtscore/datasets/loader, load_pipeline  │
│                ├──► vtscore/exporters (registry)            │
│                ├──► vtscore/labels/importers (registry)     │
│                └──► vtsearch/settings                       │
└────────────────────────────────────────────────────────────┘
        │               │                   │
        ▼               ▼                   ▼
┌──────────────┐ ┌────────────┐ ┌────────────────────┐
│ vtscore/     │ │ vtscore/   │ │ vtscore/datasets/  │
│ media/*      │ │ training/  │ │                    │
│              │ │ embedding/ │ │ loader ──► media/* │
│ audio    ─┐  │ │ detectors/ │ │ downloader/        │
│ image    ─┤  │ │            │ │ importers/*        │
│ text     ─┤  │ │ mlp        │ │                    │
│ video    ─┤  │ │ thresholds │ └────────────────────┘
│ document ─┘  │ │ loader     │
│   │          │ │   │        │
│   ▼          │ │   ▼        │
│ config.py    │ │ media/*    │
│ torch/HF     │ │ config.py  │
│ (NO Flask)   │ │ (NO Flask) │
└──────────────┘ └────────────┘

┌──────────────────────────┐  ┌────────────────────────┐
│ vtscore/exporters/*      │  │ vtscore/labels/        │
│                          │  │ importers/*            │
│ base.py (ABC)            │  │                        │
│ server_json, server_csv  │  │ base.py (ABC)          │
│ email_smtp, webhook, gui │  │ server_json, server_csv│
│                          │  │                        │
│ (NO Flask, NO state,     │  │ (NO Flask, NO state,   │
│  pure data in/out)       │  │  pure data processing) │
└──────────────────────────┘  └────────────────────────┘

                          ┌──────────────────────────┐
                          │ vtscore/sync/            │
                          │                          │
                          │ SyncSource[LoadT, SaveT] │
                          │  (generic ABC,           │
                          │   shared load() / save() │
                          │   signature)             │
                          └────────────┬─────────────┘
                                       │
              ┌────────────────────────┴────────────────────────┐
              ▼                                                 ▼
┌─────────────────────────────┐            ┌─────────────────────────────┐
│ vtsearch/settings_io/       │            │ vtscore/labels/sources/*    │
│ sources/*                   │            │ vtscore/labels/sync.py      │
│                             │            │                             │
│ base.py (SettingsSource)    │            │ base.py (LabelsetSource)    │
│ server_json_file            │            │ server_json_file            │
│                             │            │                             │
│ (NO Flask; reads/writes     │            │ (NO Flask; reads/writes     │
│  settings via file I/O)     │            │  labelsets via file I/O)    │
└─────────────────────────────┘            └─────────────────────────────┘
```

### Key observations

- **media types do NOT import Flask.**  They return a `MediaResponse`
  dataclass; the route layer converts it to a Flask response.
- **Most of training/ and embedding/ do NOT import Flask or global state**
 ; core functions in `training/mlp.py`, `training/thresholds.py`,
  `embedding/helpers.py` accept parameters only.  The exception is
  `detectors/workflow.py`, which imports `flask.g` for request-scoped
  context resolution.
- **exporters, label importers, and sync sources are fully standalone.**
  They receive a plain dict and return a plain dict/list.  Zero framework
  coupling.  Sync sources (`SettingsSource` for settings, `LabelsetSource`
  for detector labels) both inherit from the generic
  `SyncSource[LoadT, SaveT]` in `vtscore/sync/`, which captures the
  shared `load()`/`save()` shape; each concrete source plugin is still
  pure I/O.
- **datasets/ functions accept an optional `on_progress` callback.**
  When `None`, they lazily resolve the app's `update_progress`; when
  provided, they use the caller's callback.
- **`routes/*` is the primary consumer of global state** from
  `vtsearch.state`.  A few non-route modules also import specific
  helpers (e.g. `update_progress`, `next_media_id`) for progress
  reporting and ID generation during dataset loading.

---

## Extractability matrix

| Module | Flask? | Global state? | Can extract standalone? |
|--------|--------|---------------|-------------------------|
| `vtscore/training/mlp.py` + `thresholds.py` | No | No (params) | **Yes**: pure PyTorch/sklearn |
| `vtscore/detectors/labeling_progress.py` | No | No (params) | **Yes**: pure torch/numpy |
| `vtscore/exporters/` (base + all) | No | No | **Yes**: pure data processing |
| `vtscore/labels/importers/` (base + all) | No | No | **Yes**: pure data processing |
| `vtsearch/settings_io/sources/` | No | No | **Yes**: pure file I/O |
| `vtscore/labels/sources/` | No | No | **Yes**: pure file I/O |
| `vtscore/labels/sync.py` | No | Yes (reads votes) | Partially: needs state for vote export |
| `vtscore/datasets/downloader/` | No | No (callback) | **Yes**: requests only |
| `vtscore/datasets/loader.py` | No | No (callback + params) | **Yes**: needs media registry |
| `vtscore/datasets/importers/` (base + all) | No | No (callback) | **Yes**: each self-contained |
| `vtscore/eval/` | No | No | **Yes**: needs media + datasets |
| `vtsearch/settings.py` | No | No | **Yes**: JSON file I/O |
| `vtscore/media/base.py` | No | No | **Yes**: abstract only |
| `vtscore/media/{audio,image,text,video,document}` | No | No | **Yes**: torch + HF models |
| `vtscore/converters/` | No | No | **Yes**: pure media conversion |
| `vtscore/concurrency/progress.py` | No | No | **Yes**: threading only |
| `vtscore/state/` | No | N/A (IS the state) | **Yes**: plain Python dicts |
| `vtscore/config.py` | No | No | **Yes**: just constants |
| `vtsearch/auth/` | No | No | **Yes**: ABC + default provider |
| `vtsearch/routes/` | **Yes** | **Yes** | No: Flask-specific |
| `app.py` | **Yes** | **Yes** | No: application entry point |

---

## How to extract specific components

### The ML training pipeline

**Files:** `vtscore/training/mlp.py` / `vtscore/training/thresholds.py`, `vtscore/config.py` (for `TRAIN_EPOCHS`)

**Dependencies:** `torch`, `sklearn`, `numpy`

**What you get:** `train_model()` trains a 2-layer MLP classifier on
embeddings + binary labels.  `find_optimal_threshold()` iterates over
candidate thresholds to minimize weighted FPR+FNR with configurable
trade-off via an `inclusion` parameter.  A separate
`calculate_gmm_threshold()` fits a 2-component GMM for semantic sort
thresholds.

```python
from vtscore.training.mlp import train_model
from vtscore.training.thresholds import find_optimal_threshold

model = train_model(X_train, y_train, input_dim=512, inclusion_value=0)
threshold = find_optimal_threshold(scores, labels, inclusion_value=0)
```

### Embedding models (CLAP, SigLIP, E5, X-CLIP, and more)

**Files:** `vtscore/media/{audio,image,text,video}/embedder_*.py`

**Dependencies:** `torch`, `transformers`, `librosa` (audio), `PIL`
(image/video), `sentence-transformers` (text)

Each embedder is a self-contained class (separate from the `MediaType`).
Instantiate it, call `load_models()`, then use `embed_media()` /
`embed_text()`.  `embed_media()` takes a **media dict** (the same shape the
dataset loader builds); for ad-hoc files, use the `media_from_path` helper:

```python
from vtscore.media.audio.embedder_clap import AudioClapEmbedder
from vtscore.media.embedder import media_from_path

embedder = AudioClapEmbedder()
embedder.load_models()                                            # loads CLAP (cached)
embedding = embedder.embed_media(media_from_path("example.wav"))  # → numpy array
text_vec  = embedder.embed_text("birdsong")                       # same space
```

Because the embedder sees the whole media dict (not just a `Path`), a
service-based embedder can resolve content via `media["origin"]` /
`media.get("custom_metadata")` without touching local disk; e.g. a
remote lookup by `origin["params"]["content_id"]`.  The loader always
routes every pending file through `embed_media_bulk(medias)`; the ABC's
default implementation loops per item (with progress).  Services that
natively accept many items per request override `_embed_media_bulk_impl`
and batch internally.

No Flask, no global state, no progress dependency (silent no-op by
default).  To get progress reporting, set a callback before loading:

```python
embedder._on_progress = lambda status, msg, cur, tot: print(f"{msg} ({cur}/{tot})")
embedder.load_models()
```

### The plugin systems

**Pattern:** Each of the ten plugin systems uses the same architecture:
1. An abstract base class with `fields` (form descriptors) and a
   `run()`/`export()`/`load()`/`save()` method.
2. Auto-discovery via `PluginRegistry` using direct filesystem scanning
   (`Path.iterdir()`) for a sentinel attribute (`EXPORTER`, `IMPORTER`,
   `LABEL_IMPORTER`, `SETTINGS_IMPORTER`, `SETTINGS_EXPORTER`,
   `SETTINGS_SOURCE`, `LABELSET_SOURCE`, `CONVERTER`, `SOURCE`).
3. CLI support auto-derived from field definitions.

To use an exporter standalone:

```python
from vtscore.exporters.server_json_file import EXPORTER

result = EXPORTER.export(
    results={"media_type": "audio", "results": {...}},
    field_values={"filepath": "/tmp/output.json"},
)
```

### Dataset loading (without Flask)

```python
from vtscore.datasets.loader import load_dataset_from_folder

medias = {}
load_dataset_from_folder(
    Path("my_audio_folder"),
    media_type="audio",
    medias=medias,
    on_progress=lambda s, m, c, t: print(f"{m} {c}/{t}"),
)
# medias is now {1: {"id": 1, "embedding": ..., "media_bytes": ..., ...}, ...}
```

### Progress tracking

**Files:** `vtscore/concurrency/progress.py`

A thread-safe progress tracker with no framework dependencies.  Uses
`threading.Lock` and module-level dicts.  Can be dropped into any
application as-is.

---

## Plugin architecture details

### Auto-discovered plugins (importers / exporters)

All nine plugin systems (dataset importers, exporters, label importers,
settings importers/exporters/sources, labelset sources, media converters,
and media sources) share a common `PluginBase` / `PluginField` /
`PluginRegistry` architecture in `vtscore/plugins/__init__.py`:

1. **Base class** (`PluginBase`) defines `name`, `display_name`, `fields`,
   and an abstract `run()`/`export()`/`load()`/`save()` method.
2. **Field dataclass** (`PluginField`, aliased as `ImporterField`,
   `ExporterField`, `LabelImporterField`, `SettingsImporterField`, etc.)
   describes each user-configurable input with type, label, default,
   validation, and placeholder.
3. **Auto-discovery** via `PluginRegistry` scans sub-packages using
   direct filesystem scanning for a sentinel attribute (`IMPORTER`,
   `EXPORTER`, `LABEL_IMPORTER`, `SETTINGS_IMPORTER`, `SETTINGS_EXPORTER`,
   `SETTINGS_SOURCE`, `LABELSET_SOURCE`, `CONVERTER`, `SOURCE`) and
   registers them lazily on first access.
4. **CLI support** auto-generates `argparse` flags from field
   definitions.  Override `add_cli_arguments()` for custom handling.
5. **Graceful degradation**; if a plugin's optional dependency is
   missing, a warning is emitted but the app continues.

### Explicitly registered plugins (media types / embedders / clippers)

Media types, embedders, and clippers use three separate dict-based
registries in `vtscore/media/__init__.py`:

| Registry | Registration function | Lookup functions |
|----------|----------------------|------------------|
| Media types | `register(media_type)` | `get(type_id)`, `all_types()`, `get_by_folder_name()`, `get_by_extension()` |
| Embedders | `register_embedder(embedder)` | `get_embedder(name)`, `all_embedders()`, `embedders_for_type(type_id)` |
| Clippers | `register_clipper(clipper)` | `get_clipper(name)`, `all_clippers()`, `clippers_for_type(type_id)` |

**`type_id` and `folder_import_name`:** Each media type has a `type_id`
(e.g. `"audio"`, `"image"`) and a `folder_import_name` which is the
same value. Both `get(type_id)` and `get_by_folder_name(name)` accept
the canonical type ID.

Media converters use the same `PluginRegistry` auto-discovery pattern
(sentinel: `CONVERTER`) in `vtscore/converters/__init__.py`, with
`list_converters()`, `get_converter(name)`,
`list_converters_for_source()`, and `list_converters_for_target()`.

To add a new extension, create the class, import it, and call the
register function.  See `EXTENDING.md` (in this directory) for full examples.

---

## State management

Application state is split across two packages: **`vtscore/state/`**
owns `DatasetContext`, `DetectorContext`, `_state_lock`, and all context
operations; **`vtsearch/state/__init__.py`** is the app-tier shim that
re-exports everything from `vtscore.state` and adds the proxy view.  The
module-level names below are **proxy objects** (from `vtsearch/shim/state_proxies.py`)
that delegate to a per-request `DatasetContext` or `DetectorContext`;
see [Multi-dataset support](#multi-dataset-support). All mutable access
is protected by `_state_lock` (a `threading.RLock`):

| Variable | Type | Purpose |
|----------|------|---------|
| `medias` | `dict[int, dict]` | All loaded media items with embeddings |
| `good_votes` | `dict[int, None]` | Media IDs voted "good" |
| `bad_votes` | `dict[int, None]` | Media IDs voted "bad" |
| `label_history` | `list[tuple[int, str, float]]` | Ordered labelling events `(media_id, label, timestamp)` |
| `vote_click_times` | `dict[int, int]` | Media ID → click order (1-indexed); tracks voting sequence |
| `last_learned_scores` | `dict[int, float]` | Media ID → score from the most recent learned sort |
| `inclusion` | `int \| None` | FPR/FNR trade-off parameter; lazy-loaded from settings |
| `textsort_suggestions` | `list[str]` | Text queries that received a Good vote (MRU order) |
| `autorun_extractors` | `dict` | Saved extractor configurations |
| `autorun_localizers` | `dict` | Saved localizer configurations |
| `_diversity_tree` | `DiversityTree \| None` | Hierarchical k-means tree for diverse sampling |
| `_dataset_display_name` | `str \| None` | Custom display name for the loaded dataset |

Of these, only `autorun_extractors` and `autorun_localizers` are truly
global (shared across all loaded datasets). The rest are per-dataset
(`medias`, `_diversity_tree`, `_dataset_display_name`) or per-detector
(votes, label history, click times, learned scores, inclusion, textsort
suggestions) and resolve via the active `DatasetContext` /
`DetectorContext`.

Persistent settings live in `vtsearch/settings.py`, split across two
tiers.  **Server tier** (shared, `data/settings.json`): `saved_datasets_dir`,
`detectors_dir`, `max_concurrent_*`, `hidden_plugins`.
**Per-user tier** (`<user_data_dir>/user_settings.json`): everything else;
`volume`, `theme`, `inclusion`, `enrich_descriptions`, `safe_thresholds`,
`calibrate_count`, `calibration_fraction`, `audio_playing`, `swipe_animation`,
`show_metadata`, `view_mode_*`, `grid_icon_size_*`, `focus_mode_*`,
`panel_pct_*`, `autopilot_*`, `solo_media_type`, `settings_source`,
`achievement_state`, and the **Auto-Find** keys `autorun_detectors`,
`autofind_exporter`, `autofind_exporter_field_values`.  The Auto-Find keys read
through to the server file for the built-in `default` user (CLI / single-user
back-compat); see `_DEFAULT_USER_FALLBACK_KEYS`.  Theme supports three modes:
`dark`, `light`, and `highviz` (high-contrast).

Detectors are persisted as JSON files in `data/detectors/`
via the `detectors_crud_bp` / `detectors_labels_bp` route blueprints.
Each stores a name, text query, media type, examples list, and labelset.

**Primarily Flask routes mutate this state.**  Most ML and dataset
functions accept state as parameters; so you can use the ML code in a
script or notebook by passing your own dicts. A few modules (notably
`vtscore/detectors/workflow.py` and `vtscore/labels/sync.py`) import
specific helpers and resolve the active context via Flask's `g` or
thread-local storage, but these are the exceptions rather than the rule.

### State submodule organisation

`vtscore/state/` is split into focused submodules; `vtsearch/state/__init__.py`
re-exports all of them for app-tier call-sites:

| Submodule (`vtscore/state/`) | Responsibility |
|------------------------------|----------------|
| `core.py` | `DatasetContext`, `DetectorContext`, context registries, `_state_lock` |
| `votes.py` | Vote operations, label history, text-sort suggestions, learned scores |
| `clicks.py` | Click-time tracking for vote sequence analysis |
| `diversity.py` | Diversity tree construction and sampling |
| `media_lookup.py` | Media ID resolution, duplicate collapsing, origin tracking |

Global (non-per-context) state lives in `vtsearch/autorun_processors.py`:
`autorun_extractors` and `autorun_localizers` dicts and their CRUD.

### Multi-dataset support

Multiple datasets can be loaded simultaneously. Per-dataset state is
bundled in `DatasetContext` objects (`vtscore/state/core.py`), and
per-detector state in `DetectorContext` objects:

| Context | Key state |
|---------|-----------|
| `DatasetContext` | `medias`, `diversity_tree`, `dataset_display_name` |
| `DetectorContext` | `good_votes`, `bad_votes`, `label_history`, `vote_click_times`, `click_counter`, `last_learned_scores`, `textsort_suggestions`, `find_initial_labels`, `inclusion`, `training_medias`, `model`, `threshold`, `labelset_source` |

The module-level names (`medias`, `good_votes`, etc.) are **proxy
objects** (`_ProxyDict` / `_ProxyList`) that delegate to the context
resolved per-request:

1. **Inside a Flask request**; the `before_request` handler reads
   `X-Dataset-Id` and `X-Detector-Id` headers, resolves the matching
   contexts, and stashes them on Flask's `g`. Proxies check `g` first.
2. **Outside a request** (background threads, CLI, tests); proxies
   fall back to a thread-local context scoped via the
   `thread_dataset_context()` / `thread_detector_context()` context
   managers (which snapshot and restore the prior value automatically).
   The bare `set_thread_dataset_context()` / `set_thread_detector_context()`
   setters remain available for tests and for the rare call site that
   wants the unscoped form.

There is no single global "active" pointer. Key functions:
`register_context()`, `unregister_context()`, `get_context()`,
`list_loaded_dataset_ids()`.

**Dataset registry** (`vtscore/datasets/registry.py`) maintains a persistent
JSON manifest at `data/dataset_registry.json` tracking which datasets
are available and which are currently loaded in memory (`_loaded_ids`).

API endpoints: `POST /api/datasets/registry/<id>/load` (load from pkl),
`POST /api/datasets/registry/<id>/unload` (free RAM).

The Angular frontend's `ActiveContextService` tracks which dataset/model
the user selected, and `activeContextInterceptor` attaches
`X-Dataset-Id` / `X-Detector-Id` headers to every API request.

---

## Authentication and multi-user support

VTSearch uses a pluggable authentication system to support both single-user
and multi-user deployments.

### LoginProvider abstraction (`vtsearch/auth/`)

The `LoginProvider` ABC defines the interface:

| Method | Purpose |
|--------|---------|
| `get_user(request)` | Return the username for the current request |
| `is_authenticated(request)` | Check if the request carries valid credentials |
| `login_required()` | Whether the frontend should show a login screen |
| `get_user_data_dir(username, base)` | Return per-user data directory |
| `status_dict(request)` | JSON dict for `/api/auth/status` |

`DefaultLoginProvider` is the built-in default: single-user, always
authenticated, shared data directory. Custom providers can implement PKI,
OAuth, LDAP, or any auth scheme without modifying route code.

### Per-request user context

The `before_request` middleware in `app.py` populates `g.user` on every
request via the active provider's `get_user()`. Routes access the current
user via `get_current_user()`. Outside a Flask request context (CLI,
background threads) it falls back to `"default"`.

### Ownership tracking

Routes that create detectors, datasets, or detectors record
`created_by = get_current_user()` for provenance. The auth endpoint
`GET /api/auth/status` returns the provider name, current user,
authentication state, and login-required flag.

### Current scope

Per-dataset and per-detector runtime state is isolated via
`DatasetContext` and `DetectorContext` proxy objects (see
[Multi-dataset support](#multi-dataset-support)). The auth
infrastructure provides per-user data directories and ownership
tracking. Settings remain global (shared across all users/datasets).

---

## Element-level origin tracking

Every data element (clip) carries its own provenance so that:

- Data from multiple sources can coexist in the same dataset.
- Exported label sets can be re-imported and matched back to their
  original source.
- Origins are preserved through pickle save/load round-trips.

### Per-clip fields

Each clip dict includes two provenance fields:

| Field | Type | Description |
|-------|------|-------------|
| `origin` | `dict \| None` | Serialised `Origin` (e.g. `{"importer": "server_folder", "params": {"path": "/data"}}`) |
| `origin_name` | `str` | Unique name within the origin (typically the filename) |
| `media_url` | `str \| None` | Remote URL for lazy-fetching media bytes (e.g. PullWrest URL). Used as fallback when `media_bytes` and `media_path` are both absent |

### Origin class (`vtscore/datasets/origin.py`)

```python
from vtscore.datasets.origin import Origin

o = Origin("server_folder", {"path": "/data/audio", "media_type": "audio"})
o.display()   # "server_folder(/data/audio)"
o.to_dict()   # {"importer": "server_folder", "params": {"path": "/data/audio", ...}}
```

Origins are set automatically when data is loaded:

- **Importers** produce an `Origin` from their field values via
  `DatasetImporter.build_origin(field_values)`.
- **Demo datasets** get `Origin("demo", {"name": dataset_name})`.
- **Pickle loads** preserve the per-element origins stored in the file.
  Old pickles without origins fall back to the legacy `creation_info` stored in the pickle (if any).

### LabelSet (`vtscore/datasets/labelset.py`)

A `LabelSet` extends the dataset concept: each element carries its origin,
its name within that origin, its label (`"good"` / `"bad"`), and optional
`metadata` (arbitrary key-value dict for round-tripping extra data like
`contentID`, `mediaID`, etc.).

```python
from vtscore.datasets.labelset import LabelSet

# Build from current state
ls = LabelSet.from_clips_and_votes(medias, good_votes, bad_votes)

# Build from auto-detect results
ls = LabelSet.from_results(results_dict, medias=medias)

# Serialise / deserialise (superset of legacy label format)
data = ls.to_dict()   # {"labels": [{"md5": ..., "label": ..., "origin": ..., ...}]}
ls2 = LabelSet.from_dict(data)
```

The `GET /api/labels/export` endpoint returns a `LabelSet` serialised
as JSON.  The format is backward-compatible: old consumers that only
read `md5` + `label` continue to work.  With `enrich=true`, each entry
gains `custom_metadata` (merged from media type display metadata, the
media's `custom_metadata`, and the flattened `origin.params`) and the
response includes an `available_columns` list.
