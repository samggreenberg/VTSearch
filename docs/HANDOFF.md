# Project Handoff

This document provides a concise orientation for anyone picking up VTSearch
for the first time — whether to operate it, extend it, or evaluate it. It
ties together the full documentation set and highlights the things you need
to know first.

## Table of Contents

1. [What is VTSearch?](#what-is-vtsearch)
2. [Documentation map](#documentation-map)
3. [Quick start](#quick-start)
4. [Key concepts](#key-concepts)
5. [Codebase orientation](#codebase-orientation)
6. [Running the test suite](#running-the-test-suite)
7. [Deployment checklist](#deployment-checklist)
8. [Common workflows](#common-workflows)
9. [Known constraints and trade-offs](#known-constraints-and-trade-offs)
10. [Feature ideas and future direction](#feature-ideas-and-future-direction)

---

## What is VTSearch?

VTSearch is a media explorer web app for browsing and voting on collections
of audio, images, text, video, or documents. It supports two sorting
strategies:

- **Semantic sorting** — uses pretrained embedding models (CLAP, CLIP,
  X-CLIP, E5) to rank items by similarity to a text query.
- **Learned sorting** — trains a small MLP neural network on user votes to
  predict good/bad labels.

Built with Flask + vanilla JavaScript + PyTorch. Single-user, no auth,
runs locally or in Docker.

---

## Documentation map

| Document | Purpose |
|----------|---------|
| [SETUP.md](SETUP.md) | Installation, prerequisites, getting started, basic Docker usage |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Production deployment, offline mode, network deps, env vars, data directory, troubleshooting |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Module structure, dependency graph, extractability matrix, state management |
| [CLI.md](CLI.md) | Command-line interface reference (autodetect, importers, exporters) |
| [ML.md](ML.md) | MLP architecture, training config, embedding models, threshold calibration |
| [EVAL.md](EVAL.md) | Evaluation framework (metrics, runner, visualisation) |
| [EXTENDING.md](EXTENDING.md) | Plugin authoring guide (importers, exporters, media types) |
| [FEATURE_IDEAS.md](FEATURE_IDEAS.md) | 132 brainstormed feature ideas |
| [demos.md](demos.md) | Available demo datasets |
| [old_io.md](old_io.md) | Retired IO module reference implementations |

---

## Quick start

### Local (fastest for development)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-cpu.txt
python app.py --local        # lazy model loading, faster startup
```

### Docker (recommended for deployment)

```bash
docker compose up -d         # CPU
# or
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d  # GPU
```

Open `http://localhost:5000`, then use the hamburger menu to load a demo
dataset or import your own data.

---

## Key concepts

### Clips

A **clip** is the fundamental data unit — a dict with an `id`, media bytes,
an embedding vector, and optional metadata (`origin`, `origin_name`, `md5`).
All clips live in a global dict keyed by integer ID.

### Votes

Users vote clips as **good** or **bad**. Votes are stored as
`dict[int, None]` (not sets) in `vtsearch/utils/state.py`. Votes drive
both the learned-sort training and the label export.

### Media types

Five media types are supported:

| Media type | Embedding model | Model ID |
|-----------|----------------|----------|
| Audio | CLAP | `laion/clap-htsat-unfused` |
| Image | CLIP | `openai/clip-vit-base-patch32` |
| Video | X-CLIP | `microsoft/xclip-base-patch32` |
| Text | E5 | `intfloat/e5-base-v2` |
| Document | None (convert first) | N/A — use converters to transform to image/text |

Models are loaded lazily on first use. Total download size: ~3.1 GB.

### Converters

Media converters transform content between types (e.g. document pages to
images, video to audio). Available converters: `Document2ImageConverter`,
`Document2TextConverter`, `Video2AudioConverter`, `Video2ImageConverter`.
See `vtsearch/converters/`.

### Detectors and processors

A **detector** is a trained MLP that classifies clips as positive/negative.
An **extractor** groups clips into categories. A **localizer** identifies
regions of interest within clips (e.g. face detection). All three are types
of **processors** and can be exported/imported as JSON files.

### Plugin systems

Four auto-discovered plugin systems share the same architecture:

- **Dataset importers** — load data from folders, pickles, HTTP archives,
  or combined datasets.
- **Results exporters** — write autodetect results to server files (JSON/CSV),
  email, webhooks, or the GUI.
- **Label importers** — import labels from server-side JSON or CSV files.
- **Processor importers** — import detectors from server-side JSON files.

See [EXTENDING.md](EXTENDING.md) for how to add new plugins.

### Origin tracking

Every clip carries per-element provenance via `origin` (dict) and
`origin_name` (str). This enables datasets from multiple sources to
coexist, and exported labels can be matched back to their source.

---

## Codebase orientation

### Entry point

`app.py` — Flask app setup, blueprint registration, startup logic, CLI
argument parsing. Key startup sequence:

1. Create `data/` directory structure
2. Initialize model cache directory
3. Load persistent settings from `data/settings.json`
4. Preload models for `autoload_media_types` (if configured)
5. Start Flask server (or run CLI autodetect workflow)

### Where things live

| What | Where |
|------|-------|
| Flask routes (REST API) | `vtsearch/routes/` |
| Global state (medias, votes) | `vtsearch/utils/state.py` |
| Persistent settings | `vtsearch/settings.py` → `data/settings.json` |
| ML training and inference | `vtsearch/models/training.py` |
| Embedding models | `vtsearch/media/{audio,image,text,video,document}/media_type.py` |
| Media converters | `vtsearch/converters/` |
| Trainable model definitions | `vtsearch/routes/trainable_models.py` → `data/trainable_models/` |
| Dataset loading and downloading | `vtsearch/datasets/` |
| Plugin registries | `vtsearch/datasets/importers/`, `vtsearch/exporters/`, `vtsearch/labels/importers/`, `vtsearch/processors/importers/` |
| Constants and model IDs | `vtsearch/config.py` |
| Frontend | `static/index.html`, `static/app.js`, `static/styles.css` |
| Tests | `tests/` (see test list in CLAUDE.md) |

### Architectural boundaries

- **Media types, models, exporters, and importers do NOT import Flask.**
  They are standalone and can be used in scripts or notebooks.
- **Only `vtsearch/routes/` imports global state.** All ML and dataset
  functions accept state as parameters.
- **Each plugin is self-contained** in its own subdirectory with its own
  `requirements.txt`.

---

## Running the test suite

```bash
pip install -r requirements-dev.txt

# Fast CPU tests (~35s)
python -m pytest tests/ -v

# Full CPU tests including slow CLI subprocess tests (~5 min)
python -m pytest tests/ -v -m 'not gpu'

# GPU tests only (requires CUDA)
python -m pytest tests/test_gpu.py -v -m gpu

# All tests
python -m pytest tests/ -v -m ''
```

### Test markers

- Default (no marker flag): fast CPU tests only, excludes `gpu` and `slow`
- `slow`: CLI subprocess tests that spawn `python app.py --autodetect`
  (~16 seconds each)
- `gpu`: CUDA-only tests

### Linting and formatting

```bash
ruff check .       # lint
ruff format .      # format
```

Configuration is in `pyproject.toml` (E402 ignored, line-length 120,
target Python 3.10).

---

## Deployment checklist

Use this checklist when setting up VTSearch for a new environment.

### Minimal deployment

- [ ] Python 3.10+ available (or Docker installed)
- [ ] System packages: `libsndfile1`, `ffmpeg`, `libgl1`, `libglib2.0-0`
- [ ] `pip install -r requirements-cpu.txt` (or build Docker image)
- [ ] `data/` directory writable (models, embeddings, settings stored here)
- [ ] Port 5000 available (or configure as needed)
- [ ] Run `python app.py` or `docker compose up`

### For offline / air-gapped environments

- [ ] Pre-download models: `./download_models.sh /path/to/models`
- [ ] Set `HF_HUB_OFFLINE=1` and `VTSEARCH_MODELS_DIR=/path/to/models`
- [ ] Prepare datasets locally (folder or pickle files)
- [ ] If using Docker, bake models into the image or mount as a volume

### For GPU acceleration

- [ ] NVIDIA GPU with CUDA support available
- [ ] NVIDIA Container Toolkit installed (for Docker)
- [ ] Use `Dockerfile.gpu` or `requirements-gpu.txt`

See [DEPLOYMENT.md](DEPLOYMENT.md) for full details.

---

## Common workflows

### Load and explore a dataset

1. Start the app
2. Click the hamburger menu (top-left)
3. Select a demo dataset or use an importer
4. Browse, listen/view, and vote

### Run autodetect from CLI

```bash
python app.py --autodetect \
  --dataset data.pkl \
  --settings settings.json \
  --exporter server_json_file --filepath results.json
```

This loads the dataset, runs all detectors from the settings file, and
exports results without starting the web server.

### Train a detector through the UI

1. Load a dataset
2. Vote several clips as good/bad (minimum ~10 votes recommended)
3. Click "Learned Sort" to train the MLP on your votes
4. Export the detector via the detectors panel

### Import pre-trained processors

Use the processor importers panel or CLI to load detectors from
server-side JSON files.

### Evaluate sorting quality

```bash
python -m vtsearch.eval --plot-dir eval_output
```

See [EVAL.md](EVAL.md) for the full evaluation framework guide.

---

## Known constraints and trade-offs

### Single-user design

VTSearch is designed for single-user operation. There is no authentication,
no session isolation, and global state is shared. Running multiple
simultaneous users against the same instance will cause vote conflicts.

### Memory usage

All clips and embeddings are held in memory. Large datasets (10k+ clips)
may require significant RAM. Models add 500 MB–1.5 GB each when loaded.

### Thread settings

`OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` are set at startup to reduce
per-operation memory. This is a deliberate trade-off: lower peak memory at
the cost of slower individual operations.

### Votes are `dict[int, None]`, not sets

This is an intentional design choice throughout the codebase. Always use
`votes[id] = None` syntax, not `votes.add(id)`.

### numpy < 2 constraint

The `numpy<2` pin avoids breaking changes in numpy 2.x that affect
PyTorch and other dependencies. Upgrading requires testing across the full
dependency chain.

---

## Feature ideas and future direction

See [FEATURE_IDEAS.md](FEATURE_IDEAS.md) for a brainstorm of 132 potential
features spanning sorting, active learning, UI/UX, datasets, export,
evaluation, media types, infrastructure, and accessibility. Highlights
include:

- Multi-query and negative-query text sorting
- Uncertainty sampling and active learning
- Grid view, keyboard shortcuts, and responsive layout
- S3 / cloud storage importers
- ONNX / TorchScript model export
- Distributed computation and job queues
- Internationalization and accessibility improvements
