# Deployment & Operations Guide

This document covers production deployment, offline operation, network
dependencies, environment variables, and data management. For basic
installation and getting started, see [SETUP.md](SETUP.md).

## Table of Contents

1. [Environment variables](#environment-variables)
2. [Network dependencies](#network-dependencies)
3. [Offline deployment](#offline-deployment)
4. [Data directory layout](#data-directory-layout)
5. [Docker production notes](#docker-production-notes)
6. [Requirements file structure](#requirements-file-structure)
7. [Troubleshooting](#troubleshooting)

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VTSEARCH_MODELS_DIR` | `data/models` | Directory for HuggingFace model cache |
| `HF_HUB_OFFLINE` | unset | Set to `1` to prevent any HuggingFace Hub downloads |
| `HF_HUB_DISABLE_IMPLICIT_TOKEN` | `1` (set by app) | Disables HuggingFace auth tokens (all models are public) |
| `OMP_NUM_THREADS` | `1` (set by app) | OpenMP thread count; kept at 1 for memory optimization |
| `MKL_NUM_THREADS` | `1` (set by app) | Intel MKL thread count; kept at 1 for memory optimization |
| `PYTHONUNBUFFERED` | `1` (set in Dockerfiles) | Disables Python output buffering for real-time logs |
| `NVIDIA_VISIBLE_DEVICES` | `all` (GPU Dockerfile) | GPU visibility for NVIDIA Container Toolkit |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility` (GPU Dockerfile) | GPU driver capabilities |

---

## Network dependencies

### Embedding models (HuggingFace Hub)

VTSearch downloads four embedding models on first use. Each model is
lazy-loaded when a dataset of the corresponding media type is opened for the
first time (or at startup if `autoload_media_types` is configured in
settings).

| Model | Media type | HuggingFace ID | Approx. size |
|-------|-----------|----------------|-------------|
| CLAP | Audio | `laion/clap-htsat-unfused` | ~1.1 GB |
| CLIP | Image | `openai/clip-vit-base-patch32` | ~350 MB |
| X-CLIP | Video | `microsoft/xclip-base-patch32` | ~1.2 GB |
| E5 | Text | `intfloat/e5-base-v2` | ~440 MB |

**Total: ~3.1 GB** for all four models.

All model downloads use `token=False` — no HuggingFace account or API
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
| 20 Newsgroups | Text | scikit-learn | ~14 MB |
| AG News | Text | HuggingFace | ~15 MB |
| BBC News | Text | Internet | ~15 MB |
| IMDB Movie Reviews | Text | Stanford | ~15 MB |
| UCF-101 subset | Video | HuggingFace Datasets | ~171 MB |

### User-triggered network operations

These features require network access only when explicitly used:

| Feature | Network target | Fallback |
|---------|---------------|----------|
| YouTube playlist importer | YouTube (via yt-dlp) | Use folder/pickle importer |
| RSS feed importer | RSS/Atom feed server | Use folder/pickle importer |
| HTTP archive importer | User-provided URL | Use folder/pickle importer |
| Webhook exporter | User-provided endpoint | Use file/CSV exporter |

### pip install during build

CPU builds fetch PyTorch wheels from `https://download.pytorch.org/whl/cpu`.
GPU builds use the default PyPI + NVIDIA indexes.

---

## Offline deployment

VTSearch can run fully offline once models are cached. Follow these steps:

### 1. Pre-download models

Run the provided script on a machine with internet access:

```bash
./download_models.sh [CACHE_DIR]
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
attempt network requests — it only loads from the local cache.

### 3. Provide datasets locally

In offline mode, demo datasets cannot be downloaded. Use one of these
local-only importers instead:

- **Folder importer** — point at a directory of media files
- **Pickle importer** — load a pre-built `.pkl` dataset file
- **Combine-datasets importer** — merge multiple pickle files

### Docker offline deployment

For Docker, pre-download models and either bake them into the image or
mount them as a volume:

**Option A — Bake into the image** (larger image, simpler deployment):

```dockerfile
# Add to the end of Dockerfile, before CMD
COPY ./pre-downloaded-models/ /app/data/models/
ENV HF_HUB_OFFLINE=1
```

**Option B — Mount as a volume** (smaller image, models shared across
containers):

```bash
# Pre-download on host
./download_models.sh /opt/vtsearch-models

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
| YouTube/RSS/HTTP importers | Those importers fail | Network error in importer |
| Webhook exporter | That exporter fails | Connection error on export |

Everything else (folder/pickle import, file/CSV export, ML training,
evaluation, sorting, voting) works fully offline.

---

## Data directory layout

The `data/` directory (or `/app/data` in Docker) holds all runtime state.
It is created automatically on first startup.

```
data/
├── models/                           # HuggingFace model cache (~3.1 GB total)
│   ├── models--laion--clap-htsat-unfused/
│   ├── models--openai--clip-vit-base-patch32/
│   ├── models--microsoft--xclip-base-patch32/
│   └── models--sentence-transformers--e5-base-v2/
├── embeddings/                       # Cached dataset embeddings (.pkl files)
├── settings.json                     # User preferences, favorites, thresholds
├── audio/                            # Audio media files
├── video/                            # Video media files
├── images/                           # Image media files
└── paragraphs/                       # Text media files
```

### What to preserve vs. what's safe to delete

| Path | Preserve? | Why |
|------|-----------|-----|
| `data/models/` | **Yes** | Re-downloading is slow (~3.1 GB) |
| `data/embeddings/` | **Yes** | Contains cached embeddings; losing them means recomputing |
| `data/settings.json` | **Yes** | User preferences, trained detectors, autorun processors |
| `data/audio/`, `video/`, `images/`, `paragraphs/` | Depends | Media files from imported datasets; re-import if lost |
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
  "swipe_animation": true,
  "show_thumbnails_left": false,
  "show_thumbnails_right": true,
  "autoload_media_types": [],
  "autorun_processors": []
}
```

Notable fields:

- `autoload_media_types` — media types to preload at startup (triggers
  model downloads if models aren't cached)
- `autorun_processors` — saved detector/extractor configurations with
  importer name, processor name, and field values
- `theme` — `"dark"`, `"light"`, or `"highviz"`

---

## Docker production notes

### CPU deployment

```bash
docker compose up -d
```

Uses `Dockerfile` (base: `python:3.10-slim`). System packages installed:
`libsndfile1`, `ffmpeg`, `libgl1`, `libglib2.0-0`.

### GPU deployment

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

Uses `Dockerfile.gpu` (base: `nvidia/cuda:12.1.1-runtime-ubuntu22.04`).
Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
on the host.

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
  memory. Models are loaded lazily — only the media types actually used are
  loaded.
- **Disk**: The `data/models/` directory uses ~3.1 GB. Dataset embeddings
  and media files vary by dataset size.
- **CPU**: `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` are set to reduce
  per-operation memory. This trades single-operation throughput for lower
  memory usage — appropriate for a single-user application.
- **GPU (optional)**: GPU mode accelerates embedding computation and model
  training. Not required for basic operation.

### Health check

The app serves the web UI at `/`. A simple health check:

```bash
curl -f http://localhost:5000/ || exit 1
```

### Rebuilding after code changes

```bash
docker compose build                  # CPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build  # GPU
```

Add `--no-cache` after dependency changes to force a full rebuild.

---

## Requirements file structure

Dependencies are organized in a layered structure:

```
requirements-cpu.txt          ← Main CPU deps (Flask, PyTorch CPU, all media types)
  └── requirements-importers.txt  ← All importer deps (aggregates per-importer files)
  └── (inline) per-media-type packages

requirements-gpu.txt          ← Main GPU deps (PyTorch GPU, all media types)
  └── requirements-importers.txt

requirements-exporters.txt    ← All exporter deps (aggregates per-exporter files)

requirements-dev.txt          ← Dev tools (pytest)

requirements.txt              ← Generic (includes all)
```

Each plugin has its own `requirements.txt` in its subdirectory:

```
vtsearch/media/{audio,image,text,video}/requirements.txt
vtsearch/datasets/importers/{folder,pickle,http_zip,rss_feed,youtube_playlist,combine_datasets}/requirements.txt
vtsearch/exporters/{file,csv_file,gui,email_smtp,webhook}/requirements.txt
```

### Key dependencies

| Package | Version constraint | Purpose |
|---------|-------------------|---------|
| `torch` | `>=2.0.0` | Neural network training and inference |
| `transformers` | latest | HuggingFace model loading (CLAP, CLIP, X-CLIP) |
| `sentence-transformers` | latest | E5 text embeddings |
| `numpy` | `<2` | Numeric arrays (numpy 2.x has breaking changes) |
| `flask` | latest | Web server |
| `opencv-python-headless` | `<4.10` | Video frame extraction |
| `ultralytics` | latest | YOLO-based image processing |
| `laion_clap` | latest | Audio embedding preprocessing |
| `librosa` | latest | Audio file loading and processing |
| `feedparser` | latest | RSS feed importer |
| `yt-dlp` | latest | YouTube playlist importer |

---

## Troubleshooting

### Models fail to download

**Symptom**: Error during startup or dataset loading mentioning
`ConnectionError` or `OSError: We couldn't connect to`.

**Fix**: Ensure the machine has internet access to `huggingface.co`, or
pre-download models with `./download_models.sh` and set
`HF_HUB_OFFLINE=1`.

### Out of memory

**Symptom**: Process killed or `torch.cuda.OutOfMemoryError`.

**Fix**: Load fewer media types simultaneously. Set
`autoload_media_types` in settings to only the types you need, so unused
models aren't preloaded. For GPU, ensure adequate VRAM (4+ GB
recommended).

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

**Fix**: Install the full requirements:
```bash
pip install -r requirements-cpu.txt    # or requirements-gpu.txt
pip install -r requirements-exporters.txt
```

For specific importers (RSS, YouTube), install their individual
`requirements.txt` as well.
