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

---

## What is VTSearch?

VTSearch is a media explorer web app for browsing and voting on collections
of audio, images, text, video, or documents. It supports two sorting
strategies:

- **Semantic sorting** — uses pretrained embedding models (CLAP, CLIP,
  X-CLIP, E5) to rank items by similarity to a text query.
- **Learned sorting** — trains a small MLP neural network on user votes to
  predict good/bad labels.

Built with Flask + Angular + PyTorch. Single-user by default;
pluggable authentication via `LoginProvider` ABC supports multi-user
deployments. Runs locally or in Docker.

---

## Documentation map

| Document | Purpose |
|----------|---------|
| [SETUP.md](SETUP.md) | Installation, prerequisites, getting started, basic Docker usage |
| [USER_GUIDE.md](USER_GUIDE.md) | End-user walkthrough — Autopilot labeling, manual mode, sort modes, dashboard, exporting |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Production deployment, offline mode, network deps, env vars, data directory, troubleshooting |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Module structure, dependency graph, extractability matrix, state management |
| [API.md](API.md) | HTTP API reference (all REST endpoints, request/response formats) |
| [CLI.md](CLI.md) | Command-line interface reference (autodetect, importers, exporters) |
| [ML.md](ML.md) | MLP architecture, training config, embedding models, threshold calibration |
| [EVAL.md](EVAL.md) | Evaluation framework (metrics, runner, visualisation) |
| [EXTENDING.md](EXTENDING.md) | Plugin authoring index — splits into **[EXTENDING-plugins.md](EXTENDING-plugins.md)** (importers/exporters/sources), **[EXTENDING-media.md](EXTENDING-media.md)** (media types/embedders/clippers/converters), **[EXTENDING-processors.md](EXTENDING-processors.md)** (detectors/localizers/extractors). EXTENDING.md itself holds auth, dependencies, and checklists. |
| [demos.md](demos.md) | Available demo datasets |
| [plans/README.md](plans/README.md) | Index of open design plans (codebase reorg, multi-media import, patch embedders, extract-library, etc.) |
| [design/cli-detector-converter.md](design/cli-detector-converter.md) | CLI autodetect with converters/clippers (**Design proposal**, not yet implemented) |

---

## Quick start

### Local (fastest for development)

```bash
python3 -m venv venv && source venv/bin/activate
bash install-cpu.sh
python app.py                # Flask dev server on 0.0.0.0:5000
```

For production, run under gunicorn instead:
`VTSEARCH_SERVER_INIT=1 gunicorn -c gunicorn.conf.py app:app`.

### Docker (recommended for deployment)

```bash
docker compose up -d         # CPU (full feature set)
# or
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d  # GPU
# or
docker compose -f docker-compose.labbench.yml up -d  # LabBench (SigLIP-only image search; smallest, weights baked in)
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
`dict[int, None]` (not sets) per-detector in `DetectorContext` objects
(defined in `vtsearch/state/core.py`, re-exported via
`vtsearch/state/__init__.py`). Votes drive both the learned-sort training
and the label export.

### Media types

Five media types are supported:

| Media type | Default embedder | Model ID | Alternatives |
|-----------|-----------------|----------|-------------|
| Audio | CLAP | `laion/clap-htsat-unfused` | CLAP Music (`laion/larger_clap_music_and_speech`) |
| Image | SigLIP | `google/siglip-base-patch16-224` | CLIP (`openai/clip-vit-base-patch32`) |
| Video | X-CLIP | `microsoft/xclip-base-patch32` | — |
| Text | E5 | `intfloat/e5-base-v2` | BGE (`BAAI/bge-base-en-v1.5`) |
| Document | None (convert first) | N/A — use converters to transform to image/text | — |

Models are loaded lazily on first use. Default models total ~3.1 GB.

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

Ten auto-discovered plugin families share a common `PluginRegistry`
architecture: dataset importers, results exporters, label importers,
processor importers, settings importers/exporters, settings sources,
labelset sources, media converters, and media sources.

See [EXTENDING.md](EXTENDING.md) (and its split child docs) for how
each family works and how to add new plugins.

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
4. Preload embedders listed in `autoload_media_embedders` (if configured)
5. Start Flask server (or run CLI autodetect workflow)

### Where things live (quick lookup)

| What | Where |
|------|-------|
| Flask routes (REST API) | `vtsearch/routes/` |
| Global state (medias, votes) | `vtsearch/state/core.py` |
| Persistent settings | `vtsearch/settings.py` → `data/settings.json` |
| ML training, embedding models | `vtsearch/models/`, `vtsearch/media/*/embedder.py` |
| Dataset loading, demo downloads | `vtsearch/datasets/` |
| Frontend (Angular source / build output) | `frontend/` → `static/` |

For the full module-by-module map — extractability matrix, dependency
graph, plugin directories — see
[ARCHITECTURE.md](ARCHITECTURE.md#directory-map).

### Architectural boundaries

- **Media types, models, exporters, and importers do NOT import Flask.**
  They are standalone and can be used in scripts or notebooks.
- **`vtsearch/routes/` is the primary consumer of global state.** ML
  and dataset functions generally accept state as parameters, though
  some modules import specific helpers (e.g. `update_progress`,
  `next_media_id`) for progress reporting and ID generation.
- **Each plugin is self-contained** in its own subdirectory. Dependencies
  are declared in per-plugin `requirements.txt` files, auto-discovered
  by `install-plugin-deps.sh`.

---

## Running the test suite

```bash
pip install -r requirements.txt

# Fast CPU tests (~35s)
python -m pytest tests/ -v

# Full CPU tests including slow CLI subprocess tests (~5 min)
python -m pytest tests/ -v -m 'not gpu'

# GPU tests only (requires CUDA)
python -m pytest tests/test_gpu.py -v -m gpu

# All tests
python -m pytest tests/ -v -m ''
```

### Test groups

For faster iteration, run tests by area instead of the full suite:

```bash
./run-tests.sh core          # basic app functionality
./run-tests.sh api           # API contracts and security
./run-tests.sh sorting       # sort algorithms and diversity
./run-tests.sh datasets      # dataset loading and management
./run-tests.sh io            # import/export and sync
```

See `CLAUDE.md` for the complete group-to-file mapping.

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
- [ ] `bash install-cpu.sh` (or build Docker image)
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
- [ ] Use `Dockerfile.gpu` or `bash install-gpu.sh`

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

### Single-user default, pluggable multi-user

VTSearch ships with `DefaultLoginProvider` (single-user, no auth).
A pluggable `LoginProvider` ABC in `vtsearch/auth/` allows custom
providers (e.g. PKI, OAuth, username/password) without changing route
code. The `created_by` field on detectors, datasets, and models tracks
ownership, and `get_user_data_dir()` supports per-user data directories.

**Current scope:** Per-dataset and per-detector runtime state is isolated
via `DatasetContext` and `DetectorContext` proxy objects (see
[ARCHITECTURE.md](ARCHITECTURE.md#multi-dataset-support)). Each user
can work with different datasets/models simultaneously via
`X-Dataset-Id`/`X-Detector-Id` headers. Settings remain global (shared
across all users).

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


