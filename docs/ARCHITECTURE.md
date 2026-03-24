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
7. [State management](#state-management)
8. [Authentication and multi-user support](#authentication-and-multi-user-support)
9. [Element-level origin tracking](#element-level-origin-tracking)

---

## What VTSearch does

VTSearch is a media-explorer web app for browsing, voting on, and
semantically sorting collections of audio, images, text, video, or
documents.  It combines:

- **Semantic sorting** — LAION-CLAP (audio), CLIP (images), X-CLIP
  (video), E5-base-v2 (text) for embedding-based similarity search,
  with alternative embedders available (CLAP Music, SigLIP, BGE).
- **Learned sorting** — a small MLP trained on user votes to predict
  good/bad labels.
- **Flask web UI** — Angular SPA frontend with a REST API.
- **Plugin systems** — auto-discovered dataset importers, results
  exporters, label importers, and processor importers.

---

## Directory map

```
VTSearch/
├── app.py                          Flask entry point, CLI, startup
│
├── vtsearch/
│   ├── config.py                   Constants (paths, model IDs, rates)
│   ├── medias.py                   Test media generation & embedding cache
│   ├── cli.py                      CLI autodetect workflow
│   ├── settings.py                 Persistent settings & autorun processors
│   │
│   ├── auth/                       Authentication & user management
│   │   └── __init__.py             LoginProvider ABC, DefaultLoginProvider, get_current_user(), get_user_data_dir()
│   │
│   ├── media/                      Media type, embedder, and clipper registries + plugins
│   │   ├── base.py                 MediaType, MediaEmbedder, MediaClipper, Processor, Detector, Localizer, Extractor ABCs
│   │   ├── __init__.py             Three registries: register/register_embedder/register_clipper
│   │   ├── audio/media_type.py     Audio media type (WAV serving, folder import)
│   │   ├── audio/embedder.py       AudioClapEmbedder (LAION CLAP, 512-d)
│   │   ├── audio/embedder_clap_music.py  AudioClapMusicEmbedder (CLAP Music, 512-d)
│   │   ├── audio/clipper.py        SoundDefaultClipper, SoundTilingClipper
│   │   ├── image/media_type.py     Image media type (JPEG/PNG serving)
│   │   ├── image/embedder.py       ImageClipEmbedder (OpenAI CLIP, 768-d)
│   │   ├── image/embedder_siglip.py  ImageSiglipEmbedder (SigLIP, 768-d)
│   │   ├── image/clipper.py        ImageDefaultClipper, ImageTilingClipper
│   │   ├── text/media_type.py      Text/paragraph media type (JSON serving)
│   │   ├── text/embedder.py        TextE5Embedder (E5-base-v2, 768-d)
│   │   ├── text/embedder_bge.py    TextBGEEmbedder (BGE-base-en-v1.5, 768-d)
│   │   ├── text/clipper.py         TextDefaultClipper, TextSentenceClipper
│   │   ├── video/media_type.py     Video media type (MP4/WebM serving)
│   │   ├── video/embedder.py       VideoXClipEmbedder (X-CLIP, 768-d)
│   │   ├── video/clipper.py        VideoDefaultClipper, VideoTilingClipper
│   │   ├── document/media_type.py  Document handling (no embedder; convert first)
│   │   └── document/clipper.py     DocumentDefaultClipper
│   │
│   ├── converters/                 Media type converters
│   │   ├── base.py                 MediaConverter ABC
│   │   ├── document2image.py       Render document pages as images
│   │   ├── document2text.py        Extract text from documents
│   │   ├── video2audio.py          Extract audio track from video
│   │   └── video2image.py          Sample frames from video as images
│   │
│   ├── models/                     ML model wrappers
│   │   ├── training.py             MLP training, GMM thresholds (pure PyTorch)
│   │   ├── progress.py             Labelling-progress cache & analysis
│   │   ├── embeddings.py           Thin wrappers around media-type embed()
│   │   ├── loader.py               Model initialisation (delegates to media)
│   │   └── diversity_tree.py       Hierarchical k-means tree for diverse sampling
│   │
│   ├── datasets/                   Dataset loading & downloading
│   │   ├── origin.py               Origin dataclass (per-element provenance)
│   │   ├── labelset.py             LabelSet / LabeledElement (labeled data with origins)
│   │   ├── loader.py               load_dataset_from_folder/pickle/demo
│   │   ├── downloader/             HTTP download + demo dataset downloaders
│   │   │   ├── __init__.py         Re-exports all symbols for backward compat
│   │   │   ├── core.py             URLs, sizes, progress, archive validation/extraction
│   │   │   ├── audio.py            ESC-50, GTZAN, Speech Commands v2, UrbanSound8K
│   │   │   ├── images.py           CIFAR-10, Caltech-101/256, Flowers, Food-101, EuroSAT, Dogs
│   │   │   ├── video.py            UCF-101 subset
│   │   │   ├── text.py             20 Newsgroups, BBC News, AG News, IMDB
│   │   │   └── documents.py        UCSF Industry Documents
│   │   ├── ingest.py               Clip ingestion (file → clip dict)
│   │   ├── config.py               Demo dataset catalogue
│   │   ├── split.py                Train/test splitting
│   │   └── importers/              Plugin system for data sources
│   │       ├── base.py             DatasetImporter ABC + ImporterField
│   │       ├── folder/             Local directory importer
│   │       ├── pickle/             .pkl file importer
│   │       ├── http_zip/           HTTP archive importer (API name: http_archive)
│   │       ├── combine_datasets/   Merge multiple pickle datasets
│   │       └── demo/              Demo dataset importer (pre-configured catalogues)
│   │
│   ├── exporters/                  Plugin system for output destinations
│   │   ├── base.py                 LabelsetExporter ABC + ExporterField
│   │   ├── server_json_file/       JSON file on server
│   │   ├── server_csv_file/        CSV file on server
│   │   ├── email_smtp/             SMTP email sender
│   │   ├── webhook/                HTTP POST webhook
│   │   └── gui/                    In-browser / console display
│   │
│   ├── labels/importers/           Plugin system for label sources
│   │   ├── base.py                 LabelImporter ABC + LabelImporterField
│   │   ├── server_json_file/       JSON label file on server
│   │   └── server_csv_file/        CSV label file on server
│   │
│   ├── processors/importers/       Plugin system for processor sources
│   │   ├── base.py                 ProcessorImporter ABC + ProcessorImporterField
│   │   └── server_detector_file/   Import detector from server JSON file
│   │
│   ├── eval/                       Evaluation framework
│   │   ├── __main__.py             CLI entry point (python -m vtsearch.eval)
│   │   ├── config.py               Eval dataset catalogue
│   │   ├── runner.py               run_eval() orchestrator
│   │   ├── metrics.py              mAP, P@k, R@k, F1 calculations
│   │   ├── visualize.py            Matplotlib chart generation
│   │   └── voting_iterations.py    Voting-iteration simulation
│   │
│   ├── routes/                     Flask blueprints (HTTP layer)
│   │   ├── auth.py                 Authentication status endpoint (/api/auth/status)
│   │   ├── helpers.py              Shared route helpers (get_json_or_400)
│   │   ├── main.py                 Root route, favicon, logo
│   │   ├── medias.py               Media listing, serving, voting
│   │   ├── sorting.py              Text/learned/example sort, labels, diversity
│   │   ├── detectors.py            Detector/extractor/localizer management, autodetect
│   │   ├── detectors_crud.py       Detector CRUD operations (create, rename, delete)
│   │   ├── detectors_scoring.py    Detector scoring and autodetect execution
│   │   ├── detectors_training.py   Detector training from votes
│   │   ├── datasets.py             Dataset loading, demos, dashboard
│   │   ├── datasets_loading.py     Dataset loading and import orchestration
│   │   ├── datasets_ui.py          Dataset UI helpers and demo listing
│   │   ├── exporters.py            Exporter registry & execution
│   │   ├── label_importers.py      Label importer registry & execution
│   │   ├── processor_importers.py  Processor importer registry & execution
│   │   ├── settings.py             Settings persistence (volume, theme, etc.)
│   │   └── trainable_models.py     Persistent trainable model definitions (CRUD)
│   │
│   ├── utils/
│   │   ├── state.py                Global state (medias, votes, autorun config, history)
│   │   ├── progress.py             Thread-safe progress tracking
│   │   └── registry.py             PluginBase, PluginField, PluginRegistry (shared plugin infra)
│   │
│   └── audio/                      WAV/tone generation utilities
│
├── static/                         Angular build output (HTML + CSS + JS)
└── tests/                          Comprehensive test suite
```

---

## Dependency graph

Arrows point from dependent → dependency.  Modules on the left import
modules on the right.

```
┌──────────────────────────────────────────────────────────┐
│                    Flask / HTTP layer                      │
│                                                          │
│  app.py ──► auth, routes/* ──► utils/state, utils/progress │
│                │                                          │
│                ├──► models/embeddings, models/training    │
│                ├──► datasets/loader                       │
│                ├──► exporters (registry)                  │
│                ├──► labels/importers (registry)           │
│                ├──► processors/importers (registry)       │
│                └──► settings                              │
└──────────────────────────────────────────────────────────┘
        │               │                │
        ▼               ▼                ▼
┌──────────────┐ ┌────────────┐ ┌───────────────────┐
│ media/*      │ │ models/    │ │ datasets/          │
│              │ │            │ │                     │
│ audio    ─┐ │ │ training   │ │ loader ──► media/*  │
│ image    ─┤ │ │ progress   │ │ downloader/          │
│ text     ─┤ │ │ embeddings │ │ importers/*         │
│ video    ─┤ │ │ loader     │ │                     │
│ document ─┘ │ │            │ │                     │
│   │         │ │   │        │ │                     │
│   ▼         │ │   ▼        │ └───────────────────┘
│ config.py   │ │ media/*    │
│ torch/HF    │ │ config.py  │
│ (NO Flask)  │ │ (NO Flask) │
└──────────────┘ └────────────┘

┌──────────────────────────┐  ┌────────────────────────┐  ┌──────────────────────────┐
│ exporters/*              │  │ labels/importers/*      │  │ processors/importers/*   │
│                          │  │                          │  │                          │
│ base.py (ABC)            │  │ base.py (ABC)           │  │ base.py (ABC)            │
│ server_json, server_csv  │  │ server_json, server_csv │  │ server_detector_file     │
│ email_smtp, webhook, gui │  │                          │  │                          │
│                          │  │ (NO Flask, NO state,     │  │ (NO Flask, NO state,     │
│ (NO Flask, NO state,     │  │  pure data processing)   │  │  pure data processing)   │
│  pure data in/out)       │  │                          │  │                          │
└──────────────────────────┘  └────────────────────────┘  └──────────────────────────┘
```

### Key observations

- **media types do NOT import Flask.**  They return a `MediaResponse`
  dataclass; the route layer converts it to a Flask response.
- **models/ do NOT import Flask or global state.**  Functions accept
  `clips_dict`, `good_votes`, `bad_votes` etc. as parameters.
- **exporters, label importers, and processor importers are fully
  standalone.**  They receive a plain dict and return a plain dict/list.
  Zero framework coupling.
- **datasets/ functions accept an optional `on_progress` callback.**
  When `None`, they lazily resolve the app's `update_progress`; when
  provided, they use the caller's callback.
- **`routes/*` is the primary consumer of global state** from
  `vtsearch.utils.state`.  A few non-route modules also import specific
  helpers (e.g. `update_progress`, `next_media_id`) for progress
  reporting and ID generation during dataset loading.

---

## Extractability matrix

| Module | Flask? | Global state? | Can extract standalone? |
|--------|--------|---------------|-------------------------|
| `models/training.py` | No | No (params) | **Yes** — pure PyTorch/sklearn |
| `models/progress.py` | No | No (params) | **Yes** — pure torch/numpy |
| `exporters/base.py` + all exporters | No | No | **Yes** — pure data processing |
| `labels/importers/base.py` + all importers | No | No | **Yes** — pure data processing |
| `processors/importers/base.py` + all importers | No | No | **Yes** — pure data processing |
| `datasets/downloader/` | No | No (callback) | **Yes** — requests only |
| `datasets/loader.py` | No | No (callback + params) | **Yes** — needs media registry |
| `datasets/importers/base.py` + all importers | No | No (callback) | **Yes** — each self-contained |
| `eval/*` | No | No | **Yes** — needs media + datasets |
| `settings.py` | No | No | **Yes** — JSON file I/O |
| `media/base.py` | No | No | **Yes** — abstract only |
| `media/audio,image,text,video,document` | No | No | **Yes** — torch + HF models |
| `converters/*` | No | No | **Yes** — pure media conversion |
| `utils/progress.py` | No | No | **Yes** — threading only |
| `utils/state.py` | No | N/A (IS the state) | **Yes** — plain Python dicts |
| `config.py` | No | No | **Yes** — just constants |
| `auth/` | No | No | **Yes** — ABC + default provider |
| `routes/*` | **Yes** | **Yes** | No — Flask-specific |
| `app.py` | **Yes** | **Yes** | No — application entry point |

---

## How to extract specific components

### The ML training pipeline

**Files:** `vtsearch/models/training.py`, `vtsearch/config.py` (for `TRAIN_EPOCHS`)

**Dependencies:** `torch`, `sklearn`, `numpy`

**What you get:** `train_model()` trains a 2-layer MLP classifier on
embeddings + binary labels.  `find_optimal_threshold()` uses a GMM to
pick a decision boundary with configurable FPR/FNR trade-off via an
`inclusion` parameter.

```python
from vtsearch.models.training import train_model, find_optimal_threshold

model = train_model(X_train, y_train, input_dim=512, inclusion_value=0)
threshold = find_optimal_threshold(scores, labels, inclusion_value=0)
```

### Embedding models (CLAP, CLIP, E5, X-CLIP)

**Files:** `vtsearch/media/{audio,image,text,video}/media_type.py`

**Dependencies:** `torch`, `transformers`, `librosa` (audio), `PIL`
(image/video), `sentence-transformers` (text)

Each media type is a self-contained class.  Instantiate it, call
`load_models()`, then use `embed_media()` / `embed_text()`:

```python
from vtsearch.media.audio.media_type import AudioMediaType

audio = AudioMediaType()
audio.load_models()                    # loads CLAP (cached)
embedding = audio.embed_media(Path("example.wav"))  # → numpy array
text_vec  = audio.embed_text("birdsong")            # same space
```

No Flask, no global state, no progress dependency (silent no-op by
default).  To get progress reporting, set a callback before loading:

```python
audio._on_progress = lambda status, msg, cur, tot: print(f"{msg} ({cur}/{tot})")
audio.load_models()
```

### The plugin systems

**Pattern:** Each of the four plugin systems uses the same architecture:
1. An abstract base class with `fields` (form descriptors) and a
   `run()`/`export()` method.
2. Auto-discovery via `pkgutil.iter_modules` scanning for a sentinel
   attribute (`EXPORTER`, `IMPORTER`, `LABEL_IMPORTER`, `PROCESSOR_IMPORTER`).
3. CLI support auto-derived from field definitions.

To use an exporter standalone:

```python
from vtsearch.exporters.server_json_file import EXPORTER

result = EXPORTER.export(
    results={"media_type": "audio", "results": {...}},
    field_values={"filepath": "/tmp/output.json"},
)
```

### Dataset loading (without Flask)

```python
from vtsearch.datasets.loader import load_dataset_from_folder

medias = {}
load_dataset_from_folder(
    Path("my_audio_folder"),
    media_type="sounds",
    medias=medias,
    on_progress=lambda s, m, c, t: print(f"{m} {c}/{t}"),
)
# medias is now {1: {"id": 1, "embedding": ..., "media_bytes": ..., ...}, ...}
```

### Progress tracking

**Files:** `vtsearch/utils/progress.py`

A thread-safe progress tracker with no framework dependencies.  Uses
`threading.Lock` and module-level dicts.  Can be dropped into any
application as-is.

---

## Plugin architecture details

### Auto-discovered plugins (importers / exporters)

All four plugin systems (dataset importers, exporters, label importers,
processor importers) share a common `PluginBase` / `PluginField` /
`PluginRegistry` architecture in `vtsearch/utils/registry.py`:

1. **Base class** (`PluginBase`) defines `name`, `display_name`, `fields`,
   and an abstract `run()`/`export()` method.
2. **Field dataclass** (`PluginField`, aliased as `ImporterField`,
   `ExporterField`, `LabelImporterField`, `ProcessorImporterField`)
   describes each user-configurable input with type, label, default,
   validation, and placeholder.
3. **Auto-discovery** via `PluginRegistry` scans sub-packages for a
   sentinel attribute (`IMPORTER`, `EXPORTER`, `LABEL_IMPORTER`,
   `PROCESSOR_IMPORTER`) and registers them lazily on first access.
4. **CLI support** auto-generates `argparse` flags from field
   definitions.  Override `add_cli_arguments()` for custom handling.
5. **Graceful degradation** — if a plugin's optional dependency is
   missing, a warning is emitted but the app continues.

### Explicitly registered plugins (media types / embedders / clippers)

Media types, embedders, and clippers use three separate dict-based
registries in `vtsearch/media/__init__.py`:

| Registry | Registration function | Lookup functions |
|----------|----------------------|------------------|
| Media types | `register(media_type)` | `get(type_id)`, `all_types()`, `get_by_folder_name()`, `get_by_extension()` |
| Embedders | `register_embedder(embedder)` | `get_embedder(name)`, `all_embedders()`, `embedders_for_type(type_id)` |
| Clippers | `register_clipper(clipper)` | `get_clipper(name)`, `all_clippers()`, `clippers_for_type(type_id)` |

Media converters use a hardcoded list in `vtsearch/converters/__init__.py`
with `list_converters()`, `get_converter(name)`,
`list_converters_for_source()`, and `list_converters_for_target()`.

To add a new extension, create the class, import it, and call the
register function.  See `EXTENDING.md` (in this directory) for full examples.

---

## State management

Application state lives in `vtsearch/utils/state.py` as module-level
dicts, all protected by `_state_lock` (a `threading.RLock`):

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
| `autorun_detectors` | `dict` | Saved detector configurations (with `autodetect` flag, `examples`, `num_labels`) |
| `autorun_extractors` | `dict` | Saved extractor configurations |
| `autorun_localizers` | `dict` | Saved localizer configurations |
| `_diversity_tree` | `DiversityTree \| None` | Hierarchical k-means tree for diverse sampling |
| `_dataset_display_name` | `str \| None` | Custom display name for the loaded dataset |

Persistent settings live separately in `vtsearch/settings.py` and are
auto-saved to `data/settings.json`.  Keys include: `volume`, `theme`,
`inclusion`, `enrich_descriptions`, `safe_thresholds`, `calibrate_count`,
`calibration_fraction`, `audio_playing`, `swipe_animation`,
`show_metadata`, `view_mode_*`, `grid_columns_*`, `focus_mode_*`,
`panel_pct_*` (per-media-type layout), `autoload_media_types`,
`autoload_media_embedders`, `autopilot_enabled`, `hide_autopilot`,
`autopilot_top_greens`, `autopilot_hard_reds`, `autopilot_goal_diversity`,
autorun processor recipes, and infrastructure directories
`saved_datasets_dir`, `detectors_dir`, `trainable_models_dir`.
See `_DEFAULTS` in `settings.py` for the full list.
Theme supports three modes: `dark`, `light`, and `highviz` (high-contrast).

Trainable models are persisted as JSON files in `data/trainable_models/`
via the `trainable_models_bp` route blueprint.  Each stores a name,
text query, media type, examples list, and labelset.

**Only Flask routes mutate this state.**  All ML and dataset functions
accept state as parameters — they never import it directly.  This means
you can use the ML code in a script or notebook by passing your own
dicts.

### State submodule organisation

`state.py` is a re-export facade over split-out submodules:

| Submodule | Responsibility |
|-----------|----------------|
| `state_core.py` | Core variables (medias, votes, inclusion, diversity tree, display name) and `_state_lock` |
| `state_votes.py` | Vote operations, label history, text-sort suggestions, learned scores |
| `state_clicks.py` | Click-time tracking for vote sequence analysis |
| `state_processors.py` | Autorun detector/extractor/localizer configuration CRUD |
| `state_diversity.py` | Diversity tree construction and sampling |
| `state_media_lookup.py` | Media ID resolution, duplicate collapsing, origin tracking |

All are imported and re-exported by `state.py` so call-sites remain unchanged.

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

Routes that create detectors, datasets, or trainable models record
`created_by = get_current_user()` for provenance. The auth endpoint
`GET /api/auth/status` returns the provider name, current user,
authentication state, and login-required flag.

### Current limitations

In-memory state (votes, medias, labels, settings) is still global — the
auth infrastructure enables metadata tracking and per-user data
directories, but runtime state isolation is not yet implemented.

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
| `origin` | `dict \| None` | Serialised `Origin` (e.g. `{"importer": "folder", "params": {"path": "/data"}}`) |
| `origin_name` | `str` | Unique name within the origin (typically the filename) |

### Origin class (`vtsearch/datasets/origin.py`)

```python
from vtsearch.datasets.origin import Origin

o = Origin("folder", {"path": "/data/audio", "media_type": "sounds"})
o.display()   # "folder(/data/audio)"
o.to_dict()   # {"importer": "folder", "params": {"path": "/data/audio", ...}}
```

Origins are set automatically when data is loaded:

- **Importers** produce an `Origin` from their field values via
  `DatasetImporter.build_origin(field_values)`.
- **Demo datasets** get `Origin("demo", {"name": dataset_name})`.
- **Pickle loads** preserve the per-element origins stored in the file.
  Old pickles without origins fall back to the legacy `creation_info` stored in the pickle (if any).

### LabelSet (`vtsearch/datasets/labelset.py`)

A `LabelSet` extends the dataset concept: each element carries its origin,
its name within that origin, **and** its label (`"good"` / `"bad"`).

```python
from vtsearch.datasets.labelset import LabelSet

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
read `md5` + `label` continue to work.
