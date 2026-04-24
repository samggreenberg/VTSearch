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

VTSearch is a media-explorer web app for browsing, voting on, and
semantically sorting collections of audio, images, text, video, or
documents.  It combines:

- **Semantic sorting** — LAION-CLAP (audio), SigLIP (images), X-CLIP
  (video), E5-base-v2 (text) for embedding-based similarity search,
  with alternative embedders available (CLAP Music, OpenAI CLIP, BGE).
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
│   │   ├── audio/speech_extractor.py  SpeechExtractor processor
│   │   ├── image/media_type.py     Image media type (JPEG/PNG serving)
│   │   ├── image/embedder.py       ImageClipEmbedder (OpenAI CLIP, 768-d)
│   │   ├── image/embedder_siglip.py  ImageSiglipEmbedder (SigLIP, 768-d, default)
│   │   ├── image/clipper.py        ImageDefaultClipper, ImageTilingClipper
│   │   ├── image/extractor.py      ImageClassExtractor (YOLO-based)
│   │   ├── image/face_localizer.py FaceLocalizer (MediaPipe-based)
│   │   ├── image/ocr_extractor.py  OCRExtractor
│   │   ├── text/media_type.py      Text media type (JSON serving, type_id="text")
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
│   │   ├── runner.py               Converter orchestration (run_converters_on_folder)
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
│   │   ├── diversity_tree.py       Hierarchical k-means tree for diverse sampling
│   │   ├── registry.py             Persistent model registry/manifest management
│   │   ├── resolver.py             Label resolution by following origin trails
│   │   ├── media_seeding.py        Media seeding utilities
│   │   ├── label_restoration.py    Label restoration functionality
│   │   ├── training_workflow.py    Training workflow orchestration
│   │   └── weights_compat.py       Origin-based detector weight normalization
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
│   │   ├── registry.py             Persistent dataset registry (data/dataset_registry.json)
│   │   ├── pickle_security.py      Restricted pickle unpickler (RCE prevention)
│   │   ├── metadata.py             Metadata extraction (CSV, MAT, CIFAR, folders)
│   │   ├── pdf.py                  PDF rendering (render_pdf_pages)
│   │   ├── ingest.py               Clip ingestion (file → clip dict)
│   │   ├── config.py               Demo dataset catalogue
│   │   ├── split.py                Train/test splitting
│   │   ├── sources/                MediaSource abstraction for resolving media files
│   │   │   ├── base.py             MediaSource ABC (list_items, fetch_item, resolve_path)
│   │   │   ├── local_folder.py     LocalFolderSource implementation
│   │   │   ├── http_archive.py     HTTP archive source implementation
│   │   │   └── pullwrest.py        PullWrestSource — fetch media via PullWrest (scaffold)
│   │   └── importers/              Plugin system for data sources
│   │       ├── base.py             DatasetImporter ABC + ImporterField
│   │       ├── folder/             Local directory importer
│   │       ├── pickle/             .pkl file importer
│   │       ├── http_zip/           HTTP archive importer (API name: http_archive)
│   │       ├── combine_datasets/   Merge multiple pickle datasets
│   │       ├── demo/               Demo dataset importer (pre-configured catalogues)
│   │       └── recaller/           ReCaller importer (scaffold — hidden from picker)
│   │
│   ├── exporters/                  Plugin system for output destinations
│   │   ├── base.py                 LabelsetExporter ABC + ExporterField
│   │   ├── server_json_file/       JSON file on server
│   │   ├── server_csv_file/        CSV file on server
│   │   ├── email_smtp/             SMTP email sender
│   │   ├── webhook/                HTTP POST webhook
│   │   ├── gui/                    In-browser / console display
│   │   └── holder/                 Holder labelset exporter (scaffold — hidden from picker)
│   │
│   ├── settings_io/                Settings import/export/sync plugins
│   │   ├── importers/              One-shot settings importers
│   │   │   ├── base.py             SettingsImporter ABC
│   │   │   ├── local_json_file/    Browser file upload
│   │   │   └── server_json_file/   Server filesystem JSON
│   │   ├── exporters/              One-shot settings exporters
│   │   │   ├── base.py             SettingsExporter ABC
│   │   │   ├── local_json_file/    Browser file download
│   │   │   └── server_json_file/   Server filesystem JSON
│   │   └── sources/                Bidirectional settings sync sources
│   │       ├── base.py             SettingsSource ABC
│   │       └── server_json_file/   Sync with server JSON file ({username} template)
│   │
│   ├── labels/                     Label importers and sync sources
│   │   ├── importers/              Plugin system for one-shot label import
│   │   │   ├── base.py             LabelImporter ABC + LabelImporterField
│   │   │   ├── server_json_file/   JSON label file on server
│   │   │   ├── server_csv_file/    CSV label file on server
│   │   │   └── holder/             Holder label importer (scaffold — hidden from picker)
│   │   ├── sources/                Bidirectional label sync sources
│   │   │   ├── base.py             LabelsetSource ABC + LabelsetSourceField
│   │   │   └── server_json_file/   Sync labels with server JSON file
│   │   └── sync.py                 sync_to/from_labelset_source utilities
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
│   │   ├── labels.py               Label export, import, fill-from-sort
│   │   ├── eval.py                 Evaluation and labeling progress routes
│   │   ├── media_server.py         Server media file management, example-sort by origin
│   │   ├── exporters.py            Exporter registry & execution
│   │   ├── label_importers.py      Label importer registry & execution
│   │   ├── processor_importers.py  Processor importer registry & execution
│   │   ├── settings.py             Settings persistence (volume, theme, etc.)
│   │   ├── settings_io.py          Settings import/export plugin routes
│   │   ├── sync_sources.py         Sync source management (settings + labelset sources)
│   │   ├── file_browser.py         File browser API for directory navigation
│   │   └── trainable_models.py     Persistent trainable model definitions (CRUD)
│   │
│   ├── utils/
│   │   ├── state.py                Re-export facade over state_*.py submodules
│   │   ├── state_core.py           Core variables (medias, votes, inclusion) and _state_lock
│   │   ├── state_votes.py          Vote operations, label history, learned scores
│   │   ├── state_clicks.py         Click-time tracking for vote sequence analysis
│   │   ├── state_processors.py     Autorun detector/extractor/localizer CRUD
│   │   ├── state_diversity.py      Diversity tree construction and sampling
│   │   ├── state_media_lookup.py   Media ID resolution, duplicate collapsing, origins
│   │   ├── progress.py             Thread-safe progress tracking
│   │   ├── registry.py             PluginBase, PluginField, PluginRegistry (shared plugin infra)
│   │   ├── hits.py                 Helpers for building media hit dicts
│   │   ├── paths.py                Path utilities
│   │   └── url_validation.py       SSRF URL validation
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
┌────────────────────────────────────────────────────────────┐
│                     Flask / HTTP layer                     │
│                                                            │
│  app.py ──► auth, routes/* ──► utils/state, utils/progress │
│                │                                           │
│                ├──► models/embeddings, models/training     │
│                ├──► datasets/loader                        │
│                ├──► exporters (registry)                   │
│                ├──► labels/importers (registry)            │
│                ├──► processors/importers (registry)        │
│                └──► settings                               │
└────────────────────────────────────────────────────────────┘
        │               │                   │
        ▼               ▼                   ▼
┌──────────────┐ ┌────────────┐ ┌────────────────────┐
│ media/*      │ │ models/    │ │ datasets/          │
│              │ │            │ │                    │
│ audio    ─┐  │ │ training   │ │ loader ──► media/* │
│ image    ─┤  │ │ progress   │ │ downloader/        │
│ text     ─┤  │ │ embeddings │ │ importers/*        │
│ video    ─┤  │ │ loader     │ │                    │
│ document ─┘  │ │            │ │                    │
│   │          │ │   │        │ │                    │
│   ▼          │ │   ▼        │ └────────────────────┘
│ config.py    │ │ media/*    │
│ torch/HF     │ │ config.py  │
│ (NO Flask)   │ │ (NO Flask) │
└──────────────┘ └────────────┘

┌──────────────────────────┐  ┌────────────────────────┐  ┌──────────────────────────┐
│ exporters/*              │  │ labels/importers/*     │  │ processors/importers/*   │
│                          │  │                        │  │                          │
│ base.py (ABC)            │  │ base.py (ABC)          │  │ base.py (ABC)            │
│ server_json, server_csv  │  │ server_json, server_csv│  │ server_detector_file     │
│ email_smtp, webhook, gui │  │                        │  │                          │
│                          │  │ (NO Flask, NO state,   │  │ (NO Flask, NO state,     │
│ (NO Flask, NO state,     │  │  pure data processing) │  │  pure data processing)   │
│  pure data in/out)       │  │                        │  │                          │
│                          │  │                        │  │                          │
└──────────────────────────┘  └────────────────────────┘  └──────────────────────────┘

┌─────────────────────────────┐  ┌─────────────────────────────┐
│ settings_io/sources/*       │  │ labels/sources/*            │
│                             │  │ labels/sync.py              │
│ base.py (SettingsSource ABC)│  │                             │
│ server_json_file            │  │ base.py (LabelsetSource ABC)│
│                             │  │ server_json_file            │
│ (NO Flask; reads/writes     │  │                             │
│  settings via file I/O)     │  │ (NO Flask; reads/writes     │
│                             │  │  labelsets via file I/O)    │
└─────────────────────────────┘  └─────────────────────────────┘
```

### Key observations

- **media types do NOT import Flask.**  They return a `MediaResponse`
  dataclass; the route layer converts it to a Flask response.
- **Most of models/ does NOT import Flask or global state** — core
  functions in `training.py`, `progress.py`, `embeddings.py` accept
  parameters only.  The exception is `training_workflow.py`, which
  imports `flask.g` for request-scoped context resolution.
- **exporters, label importers, processor importers, and sync sources
  are fully standalone.**  They receive a plain dict and return a plain
  dict/list.  Zero framework coupling.  Sync sources (`SettingsSource`,
  `LabelsetSource`) pair an import and export behind a single abstraction
  for bidirectional sync — but each source plugin is still pure I/O.
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
| `settings_io/sources/base.py` + all sources | No | No | **Yes** — pure file I/O |
| `labels/sources/base.py` + all sources | No | No | **Yes** — pure file I/O |
| `labels/sync.py` | No | Yes (reads votes) | Partially — needs state for vote export |
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
embeddings + binary labels.  `find_optimal_threshold()` iterates over
candidate thresholds to minimize weighted FPR+FNR with configurable
trade-off via an `inclusion` parameter.  A separate
`calculate_gmm_threshold()` fits a 2-component GMM for semantic sort
thresholds.

```python
from vtsearch.models.training import train_model, find_optimal_threshold

model = train_model(X_train, y_train, input_dim=512, inclusion_value=0)
threshold = find_optimal_threshold(scores, labels, inclusion_value=0)
```

### Embedding models (CLAP, CLIP, E5, X-CLIP)

**Files:** `vtsearch/media/{audio,image,text,video}/embedder.py`

**Dependencies:** `torch`, `transformers`, `librosa` (audio), `PIL`
(image/video), `sentence-transformers` (text)

Each embedder is a self-contained class (separate from the `MediaType`).
Instantiate it, call `load_models()`, then use `embed_media()` /
`embed_text()`.  `embed_media()` takes a **media dict** (the same shape the
dataset loader builds); for ad-hoc files, use the `media_from_path` helper:

```python
from vtsearch.media.audio.embedder import AudioClapEmbedder
from vtsearch.media.embedder import media_from_path

embedder = AudioClapEmbedder()
embedder.load_models()                                            # loads CLAP (cached)
embedding = embedder.embed_media(media_from_path("example.wav"))  # → numpy array
text_vec  = embedder.embed_text("birdsong")                       # same space
```

Because the embedder sees the whole media dict (not just a `Path`), a
service-based embedder can resolve content via `media["origin"]` /
`media.get("custom_metadata")` without touching local disk — e.g. a
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
   `LABEL_IMPORTER`, `PROCESSOR_IMPORTER`, `SETTINGS_IMPORTER`,
   `SETTINGS_EXPORTER`, `SETTINGS_SOURCE`, `LABELSET_SOURCE`, `CONVERTER`,
   `SOURCE`).
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
    media_type="audio",
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

All plugin systems (dataset importers, exporters, label importers,
processor importers, settings importers/exporters/sources, labelset
sources, media converters, and media sources) share a common
`PluginBase` / `PluginField` / `PluginRegistry` architecture in
`vtsearch/utils/registry.py`:

1. **Base class** (`PluginBase`) defines `name`, `display_name`, `fields`,
   and an abstract `run()`/`export()`/`load()`/`save()` method.
2. **Field dataclass** (`PluginField`, aliased as `ImporterField`,
   `ExporterField`, `LabelImporterField`, `ProcessorImporterField`)
   describes each user-configurable input with type, label, default,
   validation, and placeholder.
3. **Auto-discovery** via `PluginRegistry` scans sub-packages using
   direct filesystem scanning for a sentinel attribute (`IMPORTER`,
   `EXPORTER`, `LABEL_IMPORTER`, `PROCESSOR_IMPORTER`,
   `SETTINGS_IMPORTER`, `SETTINGS_EXPORTER`, `SETTINGS_SOURCE`,
   `LABELSET_SOURCE`, `CONVERTER`, `SOURCE`) and registers them lazily
   on first access.
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

**`type_id` and `folder_import_name`:** Each media type has a `type_id`
(e.g. `"audio"`, `"image"`) and a `folder_import_name` which is the
same value. Both `get(type_id)` and `get_by_folder_name(name)` accept
the canonical type ID.

Media converters use the same `PluginRegistry` auto-discovery pattern
(sentinel: `CONVERTER`) in `vtsearch/converters/__init__.py`, with
`list_converters()`, `get_converter(name)`,
`list_converters_for_source()`, and `list_converters_for_target()`.

To add a new extension, create the class, import it, and call the
register function.  See `EXTENDING.md` (in this directory) for full examples.

---

## State management

Application state is exposed through `vtsearch/utils/state.py` (a
re-export facade over the `state_*.py` submodules). The module-level
names below are **proxy objects** that delegate to a per-request
`DatasetContext` or `DetectorContext` — see
[Multi-dataset support](#multi-dataset-support). All mutable access is
protected by `_state_lock` (a `threading.RLock`):

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

Of these, only `autorun_detectors`, `autorun_extractors`, and
`autorun_localizers` are truly global (shared across all loaded
datasets). The rest are per-dataset (`medias`, `_diversity_tree`,
`_dataset_display_name`) or per-detector (votes, label history, click
times, learned scores, inclusion, textsort suggestions) and resolve via
the active `DatasetContext` / `DetectorContext`.

Persistent settings live separately in `vtsearch/settings.py` and are
auto-saved to `data/settings.json`.  Keys include: `volume`, `theme`,
`inclusion`, `enrich_descriptions`, `safe_thresholds`, `calibrate_count`,
`calibration_fraction`, `audio_playing`, `swipe_animation`,
`show_metadata`, `view_mode_*`, `grid_icon_size_*`, `focus_mode_*`,
`panel_pct_*` (per-media-type layout), `autoload_media_embedders`,
`autopilot_enabled`, `hide_autopilot`,
`autopilot_top_greens`, `autopilot_hard_reds`, `autopilot_goal_diversity`,
autorun processor recipes, and infrastructure directories
`saved_datasets_dir`, `detectors_dir`, `trainable_models_dir`.
See `_DEFAULTS` in `settings.py` for the full list.
Theme supports three modes: `dark`, `light`, and `highviz` (high-contrast).

Trainable models are persisted as JSON files in `data/trainable_models/`
via the `trainable_models_bp` route blueprint.  Each stores a name,
text query, media type, examples list, and labelset.

**Primarily Flask routes mutate this state.**  Most ML and dataset
functions accept state as parameters — so you can use the ML code in a
script or notebook by passing your own dicts. A few modules (notably
`training_workflow.py` and `labels/sync.py`) import specific helpers
and resolve the active context via Flask's `g` or thread-local storage,
but these are the exceptions rather than the rule.

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

### Multi-dataset support

Multiple datasets can be loaded simultaneously. Per-dataset state is
bundled in `DatasetContext` objects (`state_core.py`), and per-detector
state in `DetectorContext` objects:

| Context | Key state |
|---------|-----------|
| `DatasetContext` | `medias`, `diversity_tree`, `dataset_display_name` |
| `DetectorContext` | `good_votes`, `bad_votes`, `label_history`, `vote_click_times`, `click_counter`, `last_learned_scores`, `textsort_suggestions`, `find_initial_labels`, `inclusion`, `training_medias`, `model`, `threshold`, `labelset_source` |

The module-level names (`medias`, `good_votes`, etc.) are **proxy
objects** (`_ProxyDict` / `_ProxyList`) that delegate to the context
resolved per-request:

1. **Inside a Flask request** — the `before_request` handler reads
   `X-Dataset-Id` and `X-Model-Id` headers, resolves the matching
   contexts, and stashes them on Flask's `g`. Proxies check `g` first.
2. **Outside a request** (background threads, CLI, tests) — proxies
   fall back to a thread-local context set via
   `set_thread_dataset_context()` / `set_thread_detector_context()`.

There is no single global "active" pointer. Key functions:
`register_context()`, `unregister_context()`, `get_context()`,
`list_loaded_dataset_ids()`.

**Dataset registry** (`datasets/registry.py`) maintains a persistent
JSON manifest at `data/dataset_registry.json` tracking which datasets
are available and which are currently loaded in memory (`_loaded_ids`).

API endpoints: `POST /api/datasets/registry/<id>/load` (load from pkl),
`POST /api/datasets/registry/<id>/unload` (free RAM).

The Angular frontend's `ActiveContextService` tracks which dataset/model
the user selected, and `activeContextInterceptor` attaches
`X-Dataset-Id` / `X-Model-Id` headers to every API request.

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
| `origin` | `dict \| None` | Serialised `Origin` (e.g. `{"importer": "folder", "params": {"path": "/data"}}`) |
| `origin_name` | `str` | Unique name within the origin (typically the filename) |
| `media_url` | `str \| None` | Remote URL for lazy-fetching media bytes (e.g. PullWrest URL). Used as fallback when `media_bytes` and `media_path` are both absent |

### Origin class (`vtsearch/datasets/origin.py`)

```python
from vtsearch.datasets.origin import Origin

o = Origin("folder", {"path": "/data/audio", "media_type": "audio"})
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
its name within that origin, its label (`"good"` / `"bad"`), and optional
`metadata` (arbitrary key-value dict for round-tripping extra data like
`contentID`, `mediaID`, etc.).

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
read `md5` + `label` continue to work.  With `enrich=true`, each entry
gains `custom_metadata` (merged from media type display metadata, the
media's `custom_metadata`, and the flattened `origin.params`) and the
response includes an `available_columns` list.
