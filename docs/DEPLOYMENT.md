# Deployment & Operations Guide

This document covers production deployment, offline operation, network
dependencies, environment variables, and data management. For basic
installation and getting started, see [SETUP.md](SETUP.md).

## Table of Contents

1. [Environment variables](#environment-variables)
2. [Running under gunicorn](#running-under-gunicorn)
3. [Network dependencies](#network-dependencies)
4. [Offline deployment](#offline-deployment)
5. [Data directory layout](#data-directory-layout)
6. [Docker production notes](#docker-production-notes)
7. [Requirements file structure](#requirements-file-structure)
8. [Troubleshooting](#troubleshooting)

---

## Environment variables

### Application

| Variable | Default | Description |
|----------|---------|-------------|
| `VTSEARCH_SECRET_KEY` | `vtsearch-dev-key-change-in-production` | Flask session secret key (**set this to a random value in production**) |
| `VTSEARCH_LOG_LEVEL` | `WARNING` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). `INFO`/`DEBUG` also turn on the per-request access log. |
| `VTSEARCH_MODELS_DIR` | `data/models` | Directory for HuggingFace model cache |

### Dataset-ingest concurrency

How many datasets the server downloads / embeds in parallel. Both knobs **autodetect from hardware** on every startup (no config needed): downloads scale with CPU count (cap 4), and embeddings scale with the scarcer of cores and RAM (~1 job per 4 cores, ~1 per 4 GiB; cap 4), flooring to 1 on small/RAM-starved boxes so a laptop stays constrained. The env vars below override the autodetect **without** persisting to `data/settings.json` — so the same `python app.py` launch picks a small default on a laptop and a bigger one on a fat node that exports the var (e.g. a single-GPU SLURM allocation, which the autodetect alone would otherwise throttle to one embed worker since embedders currently run on CPU). Values are clamped to `[1, 16]`; a non-integer value is ignored (logged, falls back to autodetect). An explicit value set via the settings UI still wins over both.

| Variable | Default | Description |
|----------|---------|-------------|
| `VTSEARCH_MAX_CONCURRENT_DOWNLOADS` | autodetect (CPU count, cap 4) | Max datasets downloaded in parallel |
| `VTSEARCH_MAX_CONCURRENT_EMBEDDINGS` | autodetect (min of cores/4 and RAM/4 GiB, cap 4) | Max dataset embedding jobs run in parallel |

### Gunicorn / WSGI (production)

| Variable | Default | Description |
|----------|---------|-------------|
| `VTSEARCH_SERVER_INIT` | unset | Set to `1` when running under gunicorn. Triggers model init / autoload / settings-source sync at import time (the Flask `__main__` block is skipped under WSGI). Set automatically in the Dockerfiles. |
| `VTSEARCH_BIND` | `0.0.0.0:5000` | Gunicorn bind address (`host:port`) |
| `VTSEARCH_THREADS` | `8` | Threads per gunicorn worker |
| `VTSEARCH_TIMEOUT` | `0` | Worker request timeout in seconds; `0` (default) disables. Long imports, training, and evaluation runs routinely exceed any short timeout; overriding to anything below ~1800 risks SIGKILL mid-operation. |

### HuggingFace / PyTorch

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_HUB_OFFLINE` | unset | Set to `1` to prevent any HuggingFace Hub downloads |
| `HF_HUB_DISABLE_IMPLICIT_TOKEN` | `1` (set by app) | Disables HuggingFace auth tokens (all models are public) |
| `TRANSFORMERS_NO_ADVISORY_WARNINGS` | `1` (set by app) | Suppresses advisory warnings from `transformers` |
| `OMP_NUM_THREADS` | `1` (set by app) | OpenMP thread count; kept at 1 for memory optimization |
| `MKL_NUM_THREADS` | `1` (set by app) | Intel MKL thread count; kept at 1 for memory optimization |

### Docker / GPU

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHONUNBUFFERED` | `1` (set in Dockerfiles) | Disables Python output buffering for real-time logs |
| `NVIDIA_VISIBLE_DEVICES` | `all` (GPU Dockerfile) | GPU visibility for NVIDIA Container Toolkit |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility` (GPU Dockerfile) | GPU driver capabilities |

---

## Running under gunicorn

For production, run the app under [gunicorn](https://gunicorn.org/) using
the bundled config:

```bash
VTSEARCH_SERVER_INIT=1 gunicorn -c gunicorn.conf.py app:app
```

`python app.py` runs Flask's built-in dev server and is intended for
development only. The Docker images already use gunicorn via the CMD in
`docker/Dockerfile`, `docker/Dockerfile.gpu`, and
`docker/Dockerfile.labbench`, so this only applies if you are running
outside Docker.

### Why `VTSEARCH_SERVER_INIT=1`?

`app.py` runs its startup sequence (model init, embedder preload,
settings-source sync) from its `if __name__ == "__main__":` block.
Gunicorn **imports** `app.py` rather than executing it, so that block
never runs. Setting `VTSEARCH_SERVER_INIT=1` tells `app.py` to also run
`initialize_server()` at import time. The Dockerfiles set this env var
automatically.

### `gunicorn.conf.py`

The bundled config pins a single worker with 8 gthread threads:

```python
workers = 1
worker_class = "gthread"
threads = 8
timeout = 0  # disabled; long imports / training would otherwise SIGKILL the worker
```

**Why one worker?** VTSearch keeps all dataset/model state in-process
(multi-dataset context, global registries, RLock-protected mutable
state). Multiple worker processes would each hold their own independent
copy, which wastes memory and breaks cross-request state continuity.
Concurrency comes from threads within the single worker, matching the
Flask dev server's `threaded=True` behaviour.

### Tuning

Override the relevant config via environment variables:

| Env var | Default | Notes |
|---------|---------|-------|
| `VTSEARCH_BIND` | `0.0.0.0:5000` | `host:port` |
| `VTSEARCH_THREADS` | `8` | Threads per worker; raise for more concurrent requests |
| `VTSEARCH_TIMEOUT` | `0` | Worker timeout in seconds; `0` (default) disables. Long imports / training routinely exceed short timeouts. |
| `VTSEARCH_LOG_LEVEL` | `warning` | Gunicorn log level; `info`/`debug` also enable the gunicorn access log (streamed to stdout) |

For larger tuning changes, edit `gunicorn.conf.py` directly.

### Reverse proxy

For public deployments, put nginx / Caddy / Traefik in front of gunicorn
to handle TLS, gzip, and static-asset caching. Flask serves the Angular
build from `static/` directly, but a dedicated reverse proxy is much
more efficient for that traffic.

---

## Network dependencies

### Embedding models (HuggingFace Hub)

VTSearch downloads four embedding models on first use. Each model is
lazy-loaded when a dataset of the corresponding media type is opened for
the first time. At startup, VTSearch also runs a smart-preload pass that
warms every embedder referenced by the dataset and detector registries
(see `predict_embedders_to_preload()` in `vtscore/embedding/loader.py`),
so the first request that uses each embedder doesn't pay the cold-load
cost. On an empty registry, nothing is preloaded.

| Model | Media type | HuggingFace ID | Approx. size |
|-------|-----------|----------------|-------------|
| CLAP | Audio (default) | `laion/clap-htsat-unfused` | ~1.1 GB |
| SigLIP | Image (default) | `google/siglip-base-patch16-224` | ~400 MB |
| X-CLIP | Video (default) | `microsoft/xclip-base-patch32` | ~1.2 GB |
| E5 | Text (default) | `intfloat/e5-base-v2` | ~440 MB |

**Total: ~3.2 GB** for the four default models.

#### Alternative embedders

Two additional embedder models are available as alternatives to the defaults.
These are only downloaded if explicitly selected:

| Model | Media type | HuggingFace ID | Approx. size |
|-------|-----------|----------------|-------------|
| CLAP Music & Speech | Audio | `laion/larger_clap_music_and_speech` | ~1.3 GB |
| ParaSpeechCLAP (speech style) | Audio | `microsoft/wavlm-large` + `ibm-granite/granite-embedding-278m-multilingual` + `ajd12342/paraspeechclap-combined` | ~4.5 GB |
| BGE | Text | `BAAI/bge-base-en-v1.5` | ~440 MB |

All model downloads use `token=False`; no HuggingFace account or API
token is required.

### Demo dataset downloads

Demo datasets are downloaded only when a user selects them through the web
UI. They are **not** downloaded at startup.

| Dataset | Media type | Source | Approx. size |
|---------|-----------|--------|-------------|
| ESC-50 | Audio | GitHub | ~600 MB |
| GTZAN Music Genre | Audio | Internet Archive | ~600 MB |
| Speech Commands v2 | Audio | Google | ~600 MB |
| UrbanSound8K | Audio | Zenodo | ~600 MB |
| Caltech-101 | Image | Caltech Data | ~131 MB |
| Caltech-256 | Image | Caltech Data | ~1200 MB |
| Oxford Flowers 102 | Image | Oxford | ~131 MB |
| Food-101 | Image | ETH Zurich | ~170 MB |
| EuroSAT | Image | Zenodo | ~170 MB |
| Stanford Dogs | Image | Stanford | ~170 MB |
| UCSF Industry Documents | Document | UCSF IDL | ~50 MB |
| 20 Newsgroups | Text | scikit-learn | ~14 MB |
| AG News | Text | HuggingFace | ~15 MB |
| BBC News | Text | Internet | ~15 MB |
| IMDB Movie Reviews | Text | Stanford | ~15 MB |
| UCF-101 subset | Video | HuggingFace Datasets | ~171 MB |

### User-triggered network operations

These features require network access only when explicitly used:

| Feature | Network target | Fallback |
|---------|---------------|----------|
| HTTP archive importer | User-provided URL | Use folder/pickle importer |
| Webhook exporter | User-provided endpoint | Use server file exporter |

### pip install during build

CPU builds fetch PyTorch wheels from `https://download.pytorch.org/whl/cpu`.
GPU builds use the default PyPI + NVIDIA indexes.

---

## Offline deployment

VTSearch can run fully offline once models are cached. Follow these steps:

### 1. Pre-download models

Run the provided script on a machine with internet access:

```bash
./scripts/download_models.sh [CACHE_DIR]
```

This downloads all four embedding models to `CACHE_DIR` (defaults to
`data/models`). The script prints instructions for offline use when
finished.

### 2. Set offline environment variables

```bash
export HF_HUB_OFFLINE=1
export VTSEARCH_MODELS_DIR=/path/to/cached/models
```

With `HF_HUB_OFFLINE=1`, the HuggingFace `transformers` library will never
attempt network requests; it only loads from the local cache.

### 3. Provide datasets locally

In offline mode, demo datasets cannot be downloaded. Use one of these
local-only importers instead:

- **Folder importer:** point at a directory of media files
- **Pickle importer:** load a pre-built `.pkl` dataset file
- **Combine-datasets importer:** merge multiple pickle files

### Docker offline deployment

For Docker, pre-download models and either bake them into the image or
mount them as a volume:

**Option A - Bake into the image** (larger image, simpler deployment):

```dockerfile
# Add to the end of Dockerfile, before CMD
COPY ./pre-downloaded-models/ /app/data/models/
ENV HF_HUB_OFFLINE=1
```

**Option B - Mount as a volume** (smaller image; models shared across
containers):

```bash
# Pre-download on host
./scripts/download_models.sh /opt/vtsearch-models

# Run with mount
docker run -p 5000:5000 \
  -v vtsearch-data:/app/data \
  -v /opt/vtsearch-models:/app/data/models:ro \
  -e HF_HUB_OFFLINE=1 \
  vtsearch
```

### What breaks without network access

| Component | Impact | Symptom |
|-----------|--------|---------|
| Models not cached | **Cannot load any dataset** | Error during model initialization |
| Demo datasets | Cannot use demo datasets | Download error in web UI |
| HTTP archive importer | That importer fails | Network error in importer |
| Webhook exporter | That exporter fails | Connection error on export |

Everything else (folder/pickle import, server file export, ML training,
evaluation, sorting, voting) works fully offline.

---

## Data directory layout

The `data/` directory (or `/app/data` in Docker) holds all runtime state.
It is created automatically on first startup.

```
data/
├── models/                           # HuggingFace model cache (~3.2 GB total)
│   ├── models--laion--clap-htsat-unfused/
│   ├── models--google--siglip-base-patch16-224/
│   ├── models--microsoft--xclip-base-patch32/
│   └── models--intfloat--e5-base-v2/
├── embeddings/                       # Cached dataset embeddings (.pkl files)
├── detectors/                 # Persistent detector definitions (.json)
├── settings.json                     # User preferences, Auto-Find config, thresholds
├── audio/                            # Audio media files
├── video/                            # Video media files
├── images/                           # Image media files
├── paragraphs/                       # Text media files
└── documents/                        # Document media files (PDF, DOC, PPT)
```

### What to preserve vs. what's safe to delete

| Path | Preserve? | Why |
|------|-----------|-----|
| `data/models/` | **Yes** | Re-downloading is slow (~3.2 GB) |
| `data/embeddings/` | **Yes** | Contains cached embeddings; losing them means recomputing |
| `data/settings.json` | **Yes** | User preferences, trained detectors, autorun processors |
| `data/detectors/` | **Yes** | Persistent detector definitions with labelsets |
| `data/audio/`, `video/`, `images/`, `paragraphs/`, `documents/` | Depends | Media files from imported datasets; re-import if lost |
| Demo dataset archives (`.zip`, `.tar.gz`) | Safe to delete | Can be re-downloaded |
| Extracted demo folders (`ESC-50-master/`, etc.) | Safe to delete | Can be re-extracted from archives |

### Settings file schema

The settings file at `data/settings.json` is auto-created on first startup
and auto-saved on every change. Schema:

```json
{
  "volume": 1.0,
  "inclusion": 0,
  "theme": "dark",
  "enrich_descriptions": false,
  "safe_thresholds": false,
  "calibrate_count": 2,
  "calibration_fraction": 0.5,
  "audio_playing": true,
  "swipe_animation": true,
  "show_metadata": true,
  "view_mode_left": {},
  "view_mode_right": {},
  "focus_mode_left": {},
  "focus_mode_right": {},
  "grid_icon_size_left": {},
  "grid_icon_size_right": {},
  "panel_pct_left": {},
  "panel_pct_right": {},
  "autofind_detectors": [],
  "autopilot_enabled": true,
  "hide_autopilot": false,
  "autopilot_top_greens": 3,
  "autopilot_hard_reds": 4,
  "autopilot_resort_interval": 10,
  "autopilot_goal_diversity": 40,
  "saved_datasets_dir": "data/saved_datasets",
  "detectors_dir": "data/detectors",
  "max_concurrent_dataset_downloads": 1,
  "max_concurrent_dataset_embeddings": 1
}
```

Notable fields:

- `autofind_detectors`: list of registered detector names
  to run during `/api/auto-detect` and the CLI `--autodetect` flow.
  This is a **per-user** setting (each user curates their own list in
  Settings → Auto-Find or via `PUT /api/detectors/registry/<id>/autofind`).
  A list placed here in `data/settings.json` applies to the built-in
  `default` user — i.e. single-user deployments and the CLI's default
  `--autodetect` run, which read through to this file.
- `saved_datasets_dir`, `detectors_dir`:
  infrastructure directories (overridable for custom data layouts)
- `max_concurrent_dataset_downloads` /
  `max_concurrent_dataset_embeddings`: concurrency gates for dataset
  loading. The download gate covers the bandwidth/disk-bound import
  phase; the embed gate covers CPU/GPU-bound embedding plus post-load
  clipping, dedup, and diversity-tree construction. Changes take
  effect on queued and future loads (running tasks are never
  preempted). Defaults derive from hardware on first read (and are
  **not** persisted to disk): downloads = `min(4, cpu_count)`,
  embeddings = `1` on CPU-only hosts else `min(2, gpu_count)`. An
  explicit value in `settings.json` always wins.
- `settings_source` (not shown above; excluded from defaults): opt-in
  bidirectional sync. Set to a plugin name + field values to auto-export
  every settings change and auto-import at startup. See `settings_io/sources/`.
- `theme`: `"dark"`, `"light"`, or `"highviz"`
- `view_mode_*`, `grid_icon_size_*`, `focus_mode_*`, `panel_pct_*`:
  per-media-type UI layout preferences (keyed by media type ID)
- `autopilot_enabled`: whether the autopilot feature is active

---

## Docker production notes

Both images run the app under gunicorn with the bundled `gunicorn.conf.py`
(single worker + gthread threads, not Flask's dev server) and set
`VTSEARCH_SERVER_INIT=1` so the startup sequence runs at WSGI import
time. See [Running under gunicorn](#running-under-gunicorn) for tuning.

### CPU deployment

```bash
docker compose -f docker/compose/docker-compose.yml up -d
```

Uses `docker/Dockerfile` (base: `python:3.10-slim`). System packages installed:
`libsndfile1`, `ffmpeg`, `libgl1`, `libglib2.0-0`.

### GPU deployment

```bash
docker compose \
  -f docker/compose/docker-compose.yml \
  -f docker/compose/docker-compose.gpu.yml up -d
```

Uses `docker/Dockerfile.gpu` (base: `nvidia/cuda:12.1.1-runtime-ubuntu22.04`).
Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
on the host.

### LabBench deployment (SigLIP-only image search)

```bash
docker compose -f docker/compose/docker-compose.labbench.yml up -d
```

Uses `docker/Dockerfile.labbench` (base: `python:3.10-slim`) and
`requirements/labbench.txt`, a pared-down dependency set that skips audio,
video, document, text, and extractor plugins. The SigLIP model weights
(`google/siglip-base-patch16-224`) are pre-downloaded **at build time**,
so the container is ready to serve immediately on first run with no
Hugging Face round-trip.

The model cache is baked into `/opt/vtsearch/models` (set via
`VTSEARCH_MODELS_DIR` in the Dockerfile) so the weights survive volume
mounts on `/app/data`. No system packages beyond the Python slim base
are required (no `ffmpeg`, `libsndfile1`, `libgl1`, ...).

This is the recommended variant when you only need image search (LabBench).

### Data persistence

The Docker volume `vtsearch-data` is mounted at `/app/data`. This persists
models, embeddings, settings, and media files across container restarts.

To use a host directory instead of a named volume:

```bash
docker run -p 5000:5000 -v /path/on/host:/app/data vtsearch
```

### Resource considerations

- **Memory**: Each embedding model uses ~500 MB–1.5 GB of RAM when loaded.
  With all four loaded simultaneously, expect ~4–6 GB total application
  memory. Models are loaded lazily; only the media types actually used are
  loaded.
- **Disk**: The `data/models/` directory uses ~3.2 GB. Dataset embeddings
  and media files vary by dataset size.
- **CPU**: `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` are set to reduce
  per-operation memory. This trades single-operation throughput for lower
  memory usage.
- **GPU (optional)**: GPU mode accelerates embedding computation and model
  training. Not required for basic operation.

### Health check

The app serves the web UI at `/`. A simple health check:

```bash
curl -f http://localhost:5000/ || exit 1
```

### Rebuilding after code changes

```bash
docker compose -f docker/compose/docker-compose.yml build           # CPU
docker compose \
  -f docker/compose/docker-compose.yml \
  -f docker/compose/docker-compose.gpu.yml build                    # GPU
```

Add `--no-cache` after dependency changes to force a full rebuild.

---

## Dependency structure

Runtime + dev dependencies are declared in `pyproject.toml` (under
`[project.dependencies]` and `[project.optional-dependencies].dev`).
The top-level `requirements/base.txt` and `requirements/gpu.txt` just
forward to it via `-e .[dev]`, so pyproject is the single source of
truth and deptry verifies every imported package is declared there.

The labbench / image-embedders requirements files are deliberately
standalone (curated minimal subsets pinned for size-constrained Docker
images) and do **not** flow through pyproject.

```
pyproject.toml                       ← [project.dependencies] + [project.optional-dependencies].dev
requirements/base.txt                ← --extra-index-url <cpu wheel index> + `-e .[dev]`
requirements/gpu.txt                 ← `-e .[dev]` (install-gpu.sh / Dockerfile.gpu set --extra-index-url)
requirements/labbench.txt            ← LabBench (SigLIP-only) image deps (standalone)
requirements/image-embedders.txt     ← All-image-embedders image deps (CPU, standalone)
requirements/image-embedders-gpu.txt ← All-image-embedders image deps (GPU, standalone)
```

Install commands:

```bash
# CPU with all features + dev tools
bash scripts/install-cpu.sh

# GPU
bash scripts/install-gpu.sh
```

### Key dependencies

| Package | Version constraint | Purpose |
|---------|-------------------|---------|
| `torch` | `>=2.0.0` | Neural network training and inference |
| `transformers` | latest | HuggingFace model loading (CLAP, CLIP, X-CLIP) |
| `sentence-transformers` | latest | E5 text embeddings |
| `numpy` | latest | Numeric arrays |
| `flask` | latest | Web server |
| `opencv-python-headless` | `<4.10` | Video frame extraction |
| `ultralytics` | latest | YOLO-based image processing |
| `laion_clap` | latest | Audio embedding preprocessing |
| `librosa` | latest | Audio file loading and processing |
| `PyMuPDF` | latest | Document (PDF/DOC/PPT) rendering and text extraction |

---

## Troubleshooting

### Models fail to download

**Symptom**: Error during startup or dataset loading mentioning
`ConnectionError` or `OSError: We couldn't connect to`.

**Fix**: Ensure the machine has internet access to `huggingface.co`, or
pre-download models with `./scripts/download_models.sh` and set
`HF_HUB_OFFLINE=1`.

### Out of memory

**Symptom**: Process killed or `torch.cuda.OutOfMemoryError`.

**Fix**: Load fewer media types simultaneously. The smart-preload pass
at startup warms every embedder referenced by the dataset and detector
registries; unregister datasets you no longer use so their embedders
aren't preloaded. For GPU, ensure adequate VRAM (4+ GB recommended).

### Docker build fails on pip install

**Symptom**: Network timeout downloading PyTorch wheels.

**Fix**: Retry the build (transient network issue), or use a local PyTorch
wheel mirror. For air-gapped environments, pre-download all wheels and
`COPY` them into the build context.

### Settings not persisting across container restarts

**Symptom**: Settings reset to defaults after `docker compose down && up`.

**Fix**: Ensure you're mounting a persistent volume at `/app/data`. Check
with `docker volume ls` and verify the `vtsearch-data` volume exists.

### Demo dataset download fails

**Symptom**: Error in web UI when selecting a demo dataset.

**Fix**: Check internet connectivity. For offline use, import data locally
via the folder or pickle importer instead.

### "No module named" errors

**Symptom**: `ModuleNotFoundError` for a media type or importer package.

**Fix**: Reinstall to ensure all dependencies are present:
```bash
bash scripts/install-cpu.sh
```
