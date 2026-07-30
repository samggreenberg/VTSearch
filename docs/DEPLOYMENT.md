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
6. [Progress-bar timing profile](#progress-bar-timing-profile)
7. [Docker production notes](#docker-production-notes)
8. [Dependency structure](#dependency-structure)
9. [Troubleshooting](#troubleshooting)

---

## Environment variables

### Application

| Variable | Default | Description |
|----------|---------|-------------|
| `VTSEARCH_SECRET_KEY` | `vtsearch-dev-key-change-in-production` | Flask session secret key (**set this to a random value in production**) |
| `VTSEARCH_LOG_LEVEL` | `WARNING` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). `INFO`/`DEBUG` also turn on the per-request access log. |
| `VTSEARCH_MODELS_DIR` | `data/models` | Directory for HuggingFace model cache |
| `VTSEARCH_SUPPORT_EMAIL` | built-in project address | Recipient for the Help modal's "Email us" link. Overrides the persisted `support_email` setting for the process lifetime (all users; not editable via the API). Equivalent to the `--support-email` CLI flag, for the gunicorn images that never parse `argv`; an explicit flag wins. |

### Progress-bar timing

| Variable | Default | Description |
|----------|---------|-------------|
| `VTSEARCH_TIMING_PROFILE` | unset | Path to a timing-profile JSON measured on this environment's hardware. Tells every instance how long each step of each long-running task takes here, so progress bars pace and predict against reality instead of the shipped defaults. See [Progress-bar timing profile](#progress-bar-timing-profile). |
| `VTSEARCH_TIMING_RECORD` | unset | Path to a JSONL sink. When set, every long-running task appends one row per step as it finishes. This is how you gather the measurements the profile is fit from; leave it unset in steady state. |

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

A few VTSearch-specific points matter when configuring the proxy:

- **TLS termination.** Terminate HTTPS at the proxy and forward plain HTTP to
  gunicorn on the loopback / private network. Gunicorn itself is not configured
  for TLS.
- **Long-running requests vs. proxy read-timeouts.** Imports, embedding,
  detector training, and evaluation routinely run for minutes. The bundled
  `gunicorn.conf.py` sets `timeout = 0` (no worker timeout) precisely so these
  don't get SIGKILLed — but the **proxy** has its own read timeout that will sever
  the connection independently. nginx's default `proxy_read_timeout` is **60s**,
  which will cut off any import or training run that takes longer. Raise it to
  cover your longest operation (e.g. `proxy_read_timeout 1800s;`), matching the
  same reasoning behind `VTSEARCH_TIMEOUT=0`.
- **Server-Sent Events (`/api/events`).** VTSearch streams live progress over an
  SSE endpoint. Proxy response buffering breaks SSE (events arrive only when the
  buffer flushes), so disable it for the stream — `proxy_buffering off;` in nginx
  (and the equivalent elsewhere) — and ensure the same long read-timeout applies,
  since the connection stays open for the life of the page. Each open connection
  pins a gthread worker thread for its lifetime, so the server caps concurrent
  connections at `VTSEARCH_THREADS - 2` (override with
  `VTSEARCH_SSE_MAX_CONNECTIONS`) and returns `503` once saturated instead of
  starving the pool that serves ordinary requests; the frontend's `EventSource`
  wrapper retries on a timer when that happens. Raising `VTSEARCH_THREADS` raises
  the SSE cap along with it.
- **Forwarded headers.** Pass `X-Forwarded-For`, `X-Forwarded-Proto`, and
  `X-Forwarded-Host` (and `Host`) so the app sees the real client and external
  scheme. For SSE/WebSocket-style streams also forward `Connection`/`Upgrade` if
  your proxy requires it.

Minimal nginx `location` for the API (illustrative):

```nginx
location /api/ {
    proxy_pass http://127.0.0.1:5000;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_read_timeout 1800s;   # cover long imports / training (gunicorn timeout is 0)
    proxy_buffering off;        # required for /api/events SSE streaming
}
```

---

## Network dependencies

### Embedding models (HuggingFace Hub)

VTSearch downloads five embedding models on first use. Each model is
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
| CLIP | Image (alternative) | `openai/clip-vit-base-patch32` | ~600 MB |
| X-CLIP | Video (default) | `microsoft/xclip-base-patch32` | ~1.2 GB |
| E5 | Text (default) | `intfloat/e5-base-v2` | ~440 MB |

**Total: ~3.8 GB** for the five models `download_models.sh` fetches (four
defaults plus CLIP, the image alternative). At runtime, only the embedders a
dataset or detector actually uses are loaded.

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
| TUT Sound Events 2017 | Audio | Zenodo | ~1700 MB |
| Caltech-101 | Image | Caltech Data | ~131 MB |
| Caltech-256 | Image | Caltech Data | ~1200 MB |
| Oxford Flowers 102 | Image | Oxford | ~131 MB |
| Food-101 | Image | ETH Zurich | ~170 MB |
| EuroSAT | Image | Zenodo | ~170 MB |
| UCSF Industry Documents | Document | UCSF IDL | ~50 MB |
| 20 Newsgroups | Text | scikit-learn | ~14 MB |
| AG News | Text | HuggingFace | ~15 MB |
| BBC News | Text | Internet | ~15 MB |
| IMDB Movie Reviews | Text | Stanford | ~15 MB |
| UCF-101 subset | Video | HuggingFace Datasets | ~171 MB |

### Sharing demo downloads between data dirs (multi-user servers)

Each data dir normally downloads its own copy of every demo dataset. On a
shared server — or any machine with several VTSearch checkouts — the multi-GB
demo sources only need to exist once: the downloaders skip a download whenever
the dataset's extraction path already exists under the data dir, so a communal
cache directory whose entries mirror that extraction layout can be symlinked
into each data dir. [`scripts/link-demo-cache.sh`](../scripts/link-demo-cache.sh)
does the wiring:

```bash
scripts/link-demo-cache.sh /shared/vtsearch-demos ./data            # link cached demos in
scripts/link-demo-cache.sh /shared/vtsearch-demos ./data --harvest  # donate demos this
                                                                    # data dir downloaded, then link
```

- The cache's entries are the extraction dirs the downloaders create under
  `DATA_DIR` (`ESC-50-master/`, `visual_genome/`, `UrbanSound8K/`, …); the
  script knows the full list and links whichever entries are populated.
- `--harvest` moves demos a data dir already downloaded into the cache and
  replaces them with symlinks, so the cache grows as users pull new demos.
  On multi-user hosts run with a cooperative umask (e.g. `umask 002`) so
  harvested files stay group-writable.
- A demo downloaded through an existing symlink writes into the cache
  through the link — no harvest needed for those.
- **Never hand-create an empty dataset dir in the cache.** The downloaders
  treat an existing extraction path as "download complete", so an empty dir
  makes that demo silently load zero items. (The script only links populated
  entries for the same reason.)
- Temp archives still spool onto `DATA_DIR`'s own volume during a fresh
  download, so that volume needs headroom for the largest archive even when
  the extraction dirs are symlinked elsewhere (#2605).

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

This downloads all five embedding models (CLAP, SigLIP, CLIP, X-CLIP, E5) to `CACHE_DIR` (defaults to
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

Demo datasets *can* still load offline if their sources were downloaded (or
copied) beforehand — see
[Sharing demo downloads between data dirs](#sharing-demo-downloads-between-data-dirs-multi-user-servers)
for the shared-cache/symlink pattern that provides them without any network
access.

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
├── models/                           # HuggingFace model cache (~3.8 GB total)
│   ├── models--laion--clap-htsat-unfused/
│   ├── models--google--siglip-base-patch16-224/
│   ├── models--openai--clip-vit-base-patch32/
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
| `data/models/` | **Yes** | Re-downloading is slow (~3.8 GB) |
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
  "calibrate_count": 1,
  "calibration_fraction": 0.5,
  "audio_playing": true,
  "show_animations": "show",
  "show_metadata": true,
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
  "max_concurrent_dataset_embeddings": 1,
  "support_email": "sam.greenberg@gmail.com"
}
```

Notable fields:

- `autofind_detectors`: list of registered detector names
  to run during `/api/auto-detect` and the CLI `--autodetect` flow.
  This is a **per-user** setting (each user curates their own list on the
  Dashboard's AutoRun detector tab, backed by
  `PUT /api/detectors/registry/<id>/autofind`).
  A list placed here in `data/settings.json` applies to the built-in
  `default` user — i.e. single-user deployments and the CLI's default
  `--autodetect` run, which read through to this file.
- `saved_datasets_dir`, `detectors_dir`:
  infrastructure directories (overridable for custom data layouts)
- `max_concurrent_dataset_downloads` /
  `max_concurrent_dataset_embeddings`: concurrency gates for dataset
  loading. The download gate covers the bandwidth/disk-bound import
  phase; the embed gate covers CPU/GPU-bound embedding plus post-load
  clipping, dedup, and coverage-atlas construction. Changes take
  effect on queued and future loads (running tasks are never
  preempted). Defaults derive from hardware on first read (and are
  **not** persisted to disk): downloads = `max(1, min(4, cpu_count))`;
  embeddings = `1` on an accelerator (CUDA/MPS — embed jobs share the
  one device and serialise on it, so extra workers only add VRAM
  pressure), and on a CPU host the scarcer of `cores/4` and
  `RAM/4 GiB` (cap 4, floor 1). See `default_concurrent_downloads` /
  `default_concurrent_embeddings` in `vtscore/embedding/loader.py`.
  These match the autodetect described under
  [Dataset-ingest concurrency](#dataset-ingest-concurrency) above. An
  explicit value in `settings.json` always wins.
- `support_email`: recipient for the Help modal's "Email us" contact
  link. Server-tier (shared across all users); defaults to the built-in
  project address. Override per instance by editing this key, or at
  startup with the `--support-email` CLI flag / `VTSEARCH_SUPPORT_EMAIL`
  env var (either applies process-wide and wins over the persisted
  value). Surfaced read-only at `GET /api/settings`; not editable via
  `PUT`.
- `settings_source` (not shown above; excluded from defaults): opt-in
  bidirectional sync. Set to a plugin name + field values to auto-export
  every settings change and auto-import at startup. See `settings_io/sources/`.
- `theme`: `"dark"`, `"light"`, or `"highviz"`
- `grid_icon_size_*`, `focus_mode_*`, `panel_pct_*`:
  per-media-type UI layout preferences (keyed by media type ID)
- `autopilot_enabled`: whether the autopilot feature is active

---

## Progress-bar timing profile

Every long-running operation — importing a dataset, opening one, loading a
detector, a text search, a Find, a train-and-score, a promote — shows one
progress bar that fills across several steps and reports a remaining-time
estimate. To pace that bar the server needs a prior for what each step *costs*,
and to keep the estimate from drifting it needs that prior to be roughly right.

VTSearch ships defaults for those costs, measured on one GPU cluster. Your
hardware is not that cluster. A profile replaces the shipped numbers with ones
measured here.

### What a profile changes (and what it can't)

A profile only affects **pacing and prediction**. A wrong or missing profile
makes a bar race one phase and crawl the next, and makes its ETA converge slowly;
it never affects correctness, results, or what gets stored. Deploying without one
is entirely supported — that is what the shipped defaults are for.

What it buys you is worth having, though. The cost of each step is affine in the
job's size:

```
seconds ≈ a + b·n + per_mb·archive_mb
```

The fixed part `a` matters enormously at small `n` (an 8-second encoder load *is*
a 1000-item text search) and not at all at large `n`. A single fixed weight
vector cannot be right at both ends, which is exactly the failure that makes a
bar's estimate climb: the job is paced as if the expensive phase were nearly
over.

### Gathering the measurements

**Recommended: observe real usage.** Point the recorder at a file and run
normally:

```bash
VTSEARCH_TIMING_RECORD=/var/lib/vtsearch/timings.jsonl \
VTSEARCH_SERVER_INIT=1 gunicorn ...
```

Each task appends one row per step as it completes. This has no side effects and
no performance cost worth measuring (a file append per task), and it captures the
datasets your users actually load at the sizes they actually are — a mix no
synthetic sweep reproduces. Let it run for a day or a week, then fit:

```bash
python scripts/profiling/tune_timing_profile.py --fit-only \
    --out /etc/vtsearch/timing-profile.json \
    /var/lib/vtsearch/timings.jsonl
```

**Alternative: drive the workloads.** When you want numbers immediately —
commissioning a node, or after a hardware change — the script can exercise the
tasks itself against datasets and detectors you name:

```bash
python scripts/profiling/tune_timing_profile.py --drive \
    --out timing-profile.json --datasets ds-a,ds-b,ds-c --reps 3
```

By default `--drive` runs only the read-only families (opening a dataset, text
search, Find). The others mutate state — loading a detector seeds example votes,
**train-and-score overwrites the active dataset's labels**, promote creates a
dataset, an import writes one — so they require `--allow-mutating` and should be
pointed at a scratch `--data-dir`, never at live user data.

Either way the script ends by printing a coverage report naming which task
families got measured and which fell back to the defaults, so a thin sweep is
visible rather than silently half-effective.

### Deploying it

```bash
VTSEARCH_TIMING_PROFILE=/etc/vtsearch/timing-profile.json
```

The file is read once per process at startup. It is plain JSON and safe to
hand-edit; a malformed one logs a warning and falls back to the defaults rather
than failing the server. Coefficients are keyed by a *cell* —
`device|media_type|embedder` — with `*` or an empty component as a wildcard, and
lookup walks from the most specific key to the least:

```json
{
  "schema": "vtsearch-timing-profile",
  "version": 1,
  "host": "prod-gpu-01",
  "tasks": {
    "text_sort": {
      "cells": {
        "cuda+cuml|image|siglip": {
          "samples": 12,
          "steps": {
            "load_model":  {"a": 8.2},
            "embed_query": {"a": 0.04},
            "score":       {"a": 0.1, "b": 0.00012}
          }
        }
      }
    }
  }
}
```

The fit emits rollup cells (`cuda||`) alongside precise ones, so measuring three
exemplar datasets still improves pacing for every dataset that host will see.

Re-run the tuning whenever the hardware, storage, or GPU stack changes. Nothing
expires a profile automatically; a stale one costs accuracy, never correctness.

### A note on the ETA itself

Even a perfectly tuned profile cannot make a remaining-time estimate exact — it
is an extrapolation from a rate that is still changing. So the server never
publishes its raw estimate: it snaps to a coarse ladder and holds each rung until
the underlying estimate moves decisively, which is why the UI says
"About 10 min left" and keeps saying it rather than counting through every
revision. A genuinely slowing job still reports the increase; what it no longer
does is twitch.

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
- **Disk**: The `data/models/` directory uses ~3.8 GB. Dataset embeddings
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

> **Note on the baked version string.** `vtsearch.__version__` is normally
> derived from git at import time, but the Docker build context excludes `.git`,
> so the Dockerfile reads the version from a build arg instead. None of the
> `docker/compose/*.yml` files pass it, so a plain `docker compose build` bakes
> the fallback `0.0.0-unknown` into the image. To stamp the real version, pass it
> explicitly, e.g.:
>
> ```bash
> docker compose -f docker/compose/docker-compose.yml build \
>   --build-arg VTSEARCH_VERSION="$(TZ=UTC git log -1 --format=%cd --date=format:%Y-%m-%dT%H:%M:%SZ HEAD)"
> ```
>
> The Dockerfile writes this into `vtsearch/_version.txt`; `__init__.py` reads it
> when git is unavailable.

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
requirements/gpu.txt                 ← `-e .[dev]` (install.sh / Dockerfile.gpu set --extra-index-url)
requirements/labbench.txt            ← LabBench (SigLIP-only) image deps (standalone)
requirements/image-embedders.txt     ← All-image-embedders image deps (CPU, standalone)
requirements/image-embedders-gpu.txt ← All-image-embedders image deps (GPU, standalone)
```

Install commands:

```bash
# Auto-detect CPU vs GPU (installs all features + dev tools)
bash scripts/install.sh

# Force one or the other
bash scripts/install.sh cpu
bash scripts/install.sh gpu
```

### Key dependencies

| Package | Version constraint | Purpose |
|---------|-------------------|---------|
| `torch` | `>=2.0.0` | Neural network training and inference |
| `transformers` | latest | HuggingFace model loading (CLAP, CLIP, X-CLIP) |
| `sentence-transformers` | latest | E5 text embeddings |
| `numpy` | latest | Numeric arrays |
| `flask` | latest | Web server |
| `opencv-python-headless` | latest | Image processing (structural embedder, YOLO). **Not** used for video decoding — see [FIPS](#video-import-crashes-with-fatal-fips-selftest-failure) |
| `ultralytics` | latest | YOLO-based image processing |
| `laion_clap` | latest | Audio embedding preprocessing |
| `librosa` | latest | Audio analysis (spectrograms, silence splitting) |
| `soundfile` / `soxr` | latest | Audio decoding and resampling (`vtscore.media.audio.decode`) |
| `imageio-ffmpeg` | latest | Bundled ffmpeg binary; audio codecs libsndfile can't read (AAC/M4A/MP4), and all video frame decoding (`vtscore.media.video.decode`) |
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

### Video import crashes with `FATAL FIPS SELFTEST FAILURE`

**Symptom**: on a FIPS-enabled host, importing a video kills the process
outright — no traceback, just a core dump and:

```
crypto/fips/fips.c:154 OpenSSL internal error: FATAL FIPS SELFTEST FAILURE
```

**Cause**: the `opencv-python-headless` wheel vendors its own OpenSSL.
`cv2.abi3.so` needs `libavformat`, which needs the `libssl`/`libcrypto`
1.1.1 pair shipped inside `opencv_python_headless.libs`, so merely
importing `cv2` maps a second, non-FIPS OpenSSL into a process that
already holds the system's FIPS-validated one. The duplicate trips the
FIPS self-test and OpenSSL responds with `abort()`. Downgrading does not
help — every wheel from 4.9 through 5.0 vendors the same OpenSSL 1.1.1w.

**Fix**: none needed for video. All video decoding goes through ffmpeg
(`vtscore.media.video.decode`), which runs out-of-process and so never
loads OpenSSL into the interpreter. Confirm the host resolved to it:

```bash
python -c "from vtscore.media.video import decode; print(decode.backend())"
```

`ffmpeg` is the expected answer. If it prints `opencv`, this install has
no ffmpeg at all and video decoding will fall back to `cv2` and crash —
install one (`dnf install ffmpeg`, or `pip install imageio-ffmpeg` for
the bundled static binary).

**Still applies to image features.** The structural image embedder and
the YOLO-based image processors import `cv2` directly, so those remain
unusable on a FIPS host. To use them, replace the wheel with a build
that links the system OpenSSL instead of vendoring one — a distro
package (`dnf install python3-opencv`), or `opencv-python` built from
source. Everything else, including the whole video workflow, works with
the stock wheel installed.

### "No module named" errors

**Symptom**: `ModuleNotFoundError` for a media type or importer package.

**Fix**: Reinstall to ensure all dependencies are present:
```bash
bash scripts/install.sh
```

### `install.sh` installs CPU torch on a machine that has a GPU

**Symptom**: On a GPU host (e.g. an AWS `g4dn` with a Tesla T4), `scripts/install.sh`
detects the card has no driver and offers to fix it:

```
NOTICE: An NVIDIA GPU is physically present, but no usable driver was found ...

What would you like to do?
  [i] Install the NVIDIA driver now (needs sudo; may require a reboot) -- recommended
  [c] Install CPU-only torch instead (no GPU acceleration)
  [s] Stop and fix it yourself
Choice [I/c/s]:
```

…or, on older versions, silently installed the CPU dependency set.

**Cause**: The script's CPU-vs-GPU decision asks `nvidia-smi` whether a GPU is
usable. On a fresh cloud GPU instance booted from a **base AMI** (anything but
the AWS Deep Learning AMI), the card is attached but the **NVIDIA kernel driver
isn't installed**, so `nvidia-smi` is absent and CUDA can't run. `pip` cannot
fix this — the driver is a system package, not a Python wheel.

**Fix**: Pick `[i]` (the default) and the installer installs the driver for you:
a distro-aware, best-effort `sudo` install (`ubuntu-drivers` / `apt` on
Debian-family, `cuda-drivers` via `dnf`/`yum` on RHEL-family), then it re-checks
`nvidia-smi` and proceeds straight into the GPU install if the GPU came online.
If the kernel module needs a **reboot** to load (common), it tells you to reboot
and re-run `bash scripts/install.sh`.

On the RHEL family, `cuda-drivers` lives in **NVIDIA's CUDA repo**, which a base
AMI does not have enabled — so a bare `dnf install cuda-drivers` fails with
`No match for argument: cuda-drivers`. The installer handles this: it tries the
install, and on that failure it drops NVIDIA's `cuda-<slug>.repo` into
`/etc/yum.repos.d` (keyed to the distro + major version, e.g. `rhel9`, plus the
CPU arch) and retries. It also enables **EPEL** best-effort first, since `dkms`
(used to build the kernel module) ships there rather than in the base RHEL repos.
Note this is *not* something a reboot fixes — until the repo is enabled the
package simply doesn't exist, so nothing got installed to take effect on boot.

There is a **second** RHEL failure mode, hit on RHEL 8/9 (and Rocky / Alma /
CentOS Stream) once the repo *is* enabled: the CUDA repo packages the driver as
a **DNF module**, so `dnf install cuda-drivers` is rejected with `All matches
were filtered out by modular filtering for argument: cuda-drivers`. The package
exists but is hidden behind a module stream that has to be enabled first. The
installer handles this too: after the plain `cuda-drivers` install is filtered
out, it falls back to `dnf module install nvidia-driver:latest-dkms`
(proprietary DKMS — covers Turing/Ampere/Ada like the g4dn's T4), and if that
stream is unavailable it retries with `nvidia-driver:open-dkms` (the open kernel
module, which newer datacenter GPUs such as Hopper and Blackwell require).

There is a **third** RHEL failure mode, hit on a fresh box that is **not
registered** with a subscription server (common on a bare RHEL 9 AMI): the
`*-dkms` streams above are rejected with `nothing provides dkms >= 3.1.8 needed
by kmod-nvidia-latest-dkms`. DKMS builds the kernel module from source, so the
stream needs the `dkms` package — which on the RHEL family ships from **EPEL**,
not the base repos. A plain `dnf install epel-release` finds nothing on an
unregistered RHEL box (EPEL isn't in any enabled repo), so `dkms` never installs
and every `*-dkms` stream fails dependency resolution. The installer handles this
two ways: (1) before trying the DKMS streams it makes sure `dkms` is actually
installed — trying the packaged `epel-release` first, then **bootstrapping EPEL
from its canonical URL** (`epel-release-latest-$(rpm -E %rhel).noarch.rpm`) when
that fails; and (2) if `dkms` still can't be had, it skips the DKMS streams and
installs the **precompiled, kABI-tracking** streams `nvidia-driver:latest` /
`nvidia-driver:open` instead, which ship a prebuilt module (no DKMS, no `dkms`
package) and only need a kernel whose kABI matches.

There is a **fourth** RHEL failure mode, seen on a bare, unregistered RHEL 9
g4dn where even the precompiled streams dead-end: `nvidia-driver:latest`
resolves only to the `*-dkms` kmod (itself `filtered out by modular filtering`
and still wanting `dkms >= 3.1.8`), and `nvidia-driver:open` reports `missing
groups or modules: nvidia-driver:open`. At that point **every** dnf path is
exhausted, so the installer falls back to **NVIDIA's self-contained `.run`
installer** — the route [AWS itself documents](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html)
for EC2. It needs no CUDA repo, no DNF module, no EPEL, and no `dkms`: it
compiles the kernel module in place against the running kernel's source, so it
only needs a C toolchain + kernel headers (installed best-effort first). By
default the installer fetches the latest driver from AWS's public,
credential-free S3 bucket (`ec2-linux-nvidia-drivers`, served over plain
HTTPS); set `VTSEARCH_NVIDIA_RUNFILE_URL` to pin a version or point at the
public Tesla compute driver instead. A reboot is usually required afterward so
the freshly built `nvidia` module loads.

You can also do it by hand:

```bash
# Ubuntu / Debian:
sudo apt-get update && sudo apt-get install -y nvidia-driver-535   # or newer

# RHEL 9 family (Rocky / Alma / CentOS Stream): enable the CUDA repo first.
# `dkms` lives in EPEL. On an unregistered RHEL box `dnf install epel-release`
# finds nothing, so bootstrap EPEL from its URL instead:
sudo dnf install -y \
  "https://dl.fedoraproject.org/pub/epel/epel-release-latest-$(rpm -E %rhel).noarch.rpm"
sudo dnf install -y dkms                                           # for the DKMS streams
sudo dnf config-manager --add-repo \
  https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
# The driver is a DNF module on RHEL 8/9, so `dnf install cuda-drivers` is
# rejected with "filtered out by modular filtering" -- install the module:
sudo dnf module install -y nvidia-driver:latest-dkms              # builds via DKMS
# ...or, if you can't get dkms, the precompiled (no-DKMS) stream instead:
#   sudo dnf module install -y nvidia-driver:latest
# ...or, if EVERY dnf path dead-ends (bare unregistered RHEL), use NVIDIA's
# self-contained .run installer (needs only gcc/make + kernel-devel):
#   sudo dnf install -y "kernel-devel-$(uname -r)" kernel-headers gcc make
#   curl -fSL -o nvidia.run \
#     https://us.download.nvidia.com/tesla/<ver>/NVIDIA-Linux-x86_64-<ver>.run
#   sudo sh nvidia.run --silent --disable-nouveau

sudo reboot                                                        # if needed

nvidia-smi              # should list the GPU and a CUDA version
bash scripts/install.sh # now auto-detects the GPU
```

See also [AWS's GPU-driver guide](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html),
or use the **AWS Deep Learning AMI**, which ships the driver preinstalled.

The installer detects the physical card via its PCI vendor ID (so it knows the
GPU is there even without a driver). The other prompt choices: `[c]` installs
CPU-only torch (same as `bash scripts/install.sh cpu`); `[s]` stops. On a
**non-interactive** shell (CI, `curl … | bash`, Docker build) it can't prompt,
so it stops — unless you set `VTSEARCH_AUTO_DRIVER=1` (auto-install the driver)
or `VTSEARCH_ASSUME_CPU=1` (proceed CPU-only) to choose unattended.

**About the output.** The driver install is a *try-until-one-works* cascade, and
on a bare cloud GPU box its early attempts are *expected* to fail (the dnf module
is filtered, `dkms` is unreachable, etc.) before a later approach succeeds. By
default each attempt runs under a **live heartbeat** with its output captured to
a log, so instead of the raw `Problem 1..6 / nothing provides dkms` /
`filtered out by modular filtering` / subscription-manager walls you see a moving
`Installing … ` line followed by either a green `✓` or a dim *"not available
here; trying another approach…"*. The captured output is shown only if a step
genuinely fails. The heartbeat also keeps long, otherwise-silent steps (metadata
refresh, the kernel-module compile, the torch import in the smoke test) visibly
alive so they don't look frozen. To watch every command's raw, unfiltered output
live (e.g. to debug a genuinely stuck install), re-run with
`VTSEARCH_VERBOSE=1 bash scripts/install.sh`.

### Making the GPU driver survive reboots and kernel updates

**Symptom**: The GPU worked yesterday, but after a stop/start (or overnight) you
have to re-run `scripts/install.sh` to get GPU support back. `nvidia-smi` is gone
or `torch.cuda.is_available()` is `False`, even though the CUDA `torch` wheel is
still installed.

**Cause**: This is almost never the Python side. The CUDA `torch`/`torchvision`/
`torchaudio` wheels live in your environment and **persist** across reboots — a
`git pull` or a frontend build never touches them. What resets is the **NVIDIA
kernel driver**, which is a *kernel module*, not a Python package, so a virtualenv
fundamentally cannot hold it. On a cloud GPU box the driver **files** survive a
stop/start on the EBS root volume, but if the OS pulled an **automatic kernel
upgrade** while it was running (Ubuntu `unattended-upgrades`, Amazon Linux/`dnf`
automatic updates), you boot into a *new kernel* and a **pre-built** `nvidia`
module no longer matches it — so `nvidia-smi` goes dark until the module is
rebuilt. Re-running the installer rebuilds it against the new kernel; that's why
it "comes back every time."

**Which half is resetting** — run these to confirm:

```bash
nvidia-smi                                                   # driver visible now?
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
#   2.x+cu124  True   -> both fine
#   2.x+cu124  False  -> CUDA torch OK, the kernel MODULE broke (kernel update)
#   2.x+cpu           -> the env got a CPU torch (env-recreation, a separate issue)
dkms status | grep -i nvidia || echo "NOT dkms-managed -> kernel updates WILL break it"
uname -r; ls /lib/modules/                                   # >1 kernel = an update happened
```

**What the installer now does for you**: `scripts/install.sh` makes the driver
set-and-forget wherever it can — it registers the module with **DKMS** when `dkms`
is available (including on the self-contained `.run` fallback path, via `--dkms`),
so the module **auto-rebuilds for each new kernel**; it enables NVIDIA's
**persistence daemon** (`nvidia-persistenced`) so the driver initializes at every
boot; and every GPU install prints whether the result is DKMS-managed
(kernel-update-proof) or still kernel-pinned. On the Debian/Ubuntu path the
distro's `nvidia-driver-*` packages are already DKMS builds, so they're covered
too.

**If the driver is already up but reported NON-DKMS**, the installer *offers to
convert it in place* — it reinstalls the driver via NVIDIA's `.run` installer
with `--dkms` (bootstrapping EPEL + `dkms` first), then recommends a reboot so the
freshly built module loads. Answer `Y` at the prompt, or drive it unattended with
`VTSEARCH_AUTO_DKMS=1 bash scripts/install.sh` (`VTSEARCH_SKIP_DKMS=1` skips the
offer). Your GPU keeps working either way; declining just leaves it kernel-pinned.

**To make a NON-DKMS driver durable by hand** — e.g. on a bare, unregistered RHEL
box where `dkms` can't be reached and only the precompiled kABI-stream or a plain
`.run` module could be installed — use one of:

- **Bake a custom AMI** (or use the **AWS Deep Learning AMI**, which ships the
  driver preinstalled and maintained). A stop/start then starts from a known-good
  driver. This is the most robust option and the recommendation for a fleet.
- **Install `dkms` first, then re-run the installer** so the module becomes
  DKMS-managed. On RHEL, `dkms` lives in EPEL — and on a **hardened or unregistered**
  box `dnf install <epel-url.rpm>` fails the GPG check (`Public key ... is not
  installed`) unless you import EPEL's key first:
  ```bash
  sudo rpm --import "https://dl.fedoraproject.org/pub/epel/RPM-GPG-KEY-EPEL-$(rpm -E %rhel)"
  sudo dnf install -y \
    "https://dl.fedoraproject.org/pub/epel/epel-release-latest-$(rpm -E %rhel).noarch.rpm"
  sudo dnf install -y dkms
  bash scripts/install.sh
  ```
  If the driver is **already loaded** (nvidia-smi works), the `.run` reinstall aborts
  with `nvidia-modeset appears to be already loaded` — stop the driver and unload the
  modules first, then reinstall with `--dkms`:
  ```bash
  sudo systemctl stop nvidia-persistenced
  sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia   # dependents before the base module
  lsmod | grep nvidia || echo "modules unloaded"           # confirm none remain
  bash scripts/install.sh                                   # or: sudo sh nvidia.run --silent --dkms
  sudo reboot
  ```
  (The installer's built-in DKMS conversion, above, does this stop/unload for you.)
- **Pin the kernel** so it stops changing under the module (blocks kernel security
  updates until you unpin — a tradeoff):
  ```bash
  # Ubuntu/Debian:
  sudo apt-mark hold "linux-image-$(uname -r)" "linux-headers-$(uname -r)"
  # RHEL/Amazon Linux:
  sudo dnf install -y 'dnf-command(versionlock)'
  sudo dnf versionlock add kernel kernel-core kernel-modules
  ```

### GPU install's cuML step (the "dependency conflicts" report, captured by default)

**Cause**: cuML (RAPIDS 25.x) depends on **newer** `nvidia-*-cu12` runtime
wheels than the torch build pins **exactly** (e.g. `cu124` torch pins
`==12.4.x`). Installing cuML upgrades those libs, so pip's post-install
consistency check emits a red `ERROR: pip's dependency resolver ...` report
flagging torch's now-unsatisfied `==` pins:

```
ERROR: pip's dependency resolver does not currently take into account all the
packages that are installed. ... torch 2.6.0+cu124 requires
nvidia-cublas-cu12==12.4.5.8 ... but you have nvidia-cublas-cu12 12.9.2.10
which is incompatible. ... (one line per nvidia-*-cu12 lib)
```

**This is cosmetic and non-fatal**: pip completes the install and rolls nothing
back, and CUDA 12.x minor runtimes are ABI-compatible across versions, so torch
keeps working on the bumped libraries.

**What you actually see**: by default `scripts/install.sh` runs the cuML step
under a heartbeat with its output **captured to a log**, so that red wall does
**not** scroll past — you see a live `Installing cuML/RAPIDS …` line and then a
green `✓`. The raw report only surfaces if the step actually fails, or if you
re-run with `VTSEARCH_VERBOSE=1` (which streams every step's raw output live).

**What to do**: nothing. The installer runs a **GPU smoke test** at the end (a
tiny torch CUDA matmul + a cuML import) that confirms the stack actually works.
Only act if the smoke test *fails*, or if your error is the **fatal**
`cuda_fp8.hpp` / nvjitlink variant below (that one names `nvidia-nvjitlink-cu12
>= 12.9` and `cuml-cu12 >= 26`, and the install does **not** succeed) — that is
a different problem with a real fix.

### cuML crashes compiling a kernel (`cuda_fp8.hpp` / nvrtc errors)

**Symptom**: A VTSBrowse projection or coverage-atlas build dies with an
nvrtc compile error like:

```
.../nvidia/cu13/include/cuda_fp8.hpp(...): error: this declaration has no
storage class or type specifier  __NV_SILENCE_DEPRECATION_BEGIN
... N errors detected in the compilation of ".../<hash>.cubin.cu".
```

**Cause**: a **RAPIDS release newer than your torch's CUDA**. cuML compiles
its cuVS/raft kernels with nvrtc lazily, on the first UMAP/k-means `fit`.
RAPIDS **26.x** (`cuml-cu12 >= 26`) raised its CUDA floor to require
`nvidia-nvjitlink-cu12 >= 12.9`, but the torch CUDA wheels VTSearch pins
(`cu124` = CUDA 12.4 ... `cu128` = CUDA 12.8) cap the CUDA libraries at 12.8 —
and no torch wheel ships CUDA 12.9 yet. An **unpinned** `cuml-cu12` therefore
floats up to 26.x, which fails the pip resolver:

```
cuml-cu12 26.6.0 requires nvidia-nvjitlink-cu12<13,>=12.9, but you have
nvidia-nvjitlink-cu12 12.4.127 which is incompatible.
```

...and, if mismatched libraries land anyway, makes cupy's nvrtc compile the
wrong-version fp8/fp6/fp4 headers (the `__NV_SILENCE_DEPRECATION_BEGIN` macro
the resident nvrtc doesn't define), producing the `cuda_fp8.hpp` crash above.

`scripts/install.sh` and `docker/Dockerfile.gpu` now **cap the wheel at
`cuml-cu12<26`** (RAPIDS 25.x declares `cuda-toolkit==12.*` and resolves
cleanly against the pinned torch), so fresh installs are unaffected. A venv
that already pulled 26.x just needs the matching downgrade.

VTSearch also now **degrades to the CPU UMAP/k-means path** whenever a cuML
fit fails for any reason (logging a one-time warning) instead of crashing, so
the run still completes — just slower, without GPU acceleration. To restore
the GPU path:

```bash
# 1. Inspect the installed CUDA stack (look for cuml-cu12 / libcuml-cu12 26.x,
#    and any stray *-cu13 wheels from an out-of-band install).
pip list | grep -iE 'cu13|cupy|cuml|cuvs|libraft|pylibraft|nvidia-nvjitlink'

# 2. Reinstall the RAPIDS stack capped below 26 so it matches the pinned torch.
pip install --extra-index-url https://pypi.nvidia.com --prefer-binary "cuml-cu12<26"

# 3. If step 1 showed stray CUDA-13 wheels, remove them and reinstall cleanly:
pip uninstall -y $(pip list --format=freeze | grep -iE '(-cu13|cupy-cuda13x)' | cut -d= -f1)
bash scripts/install.sh            # or: bash scripts/install.sh cu124
```

If you don't need GPU UMAP/k-means at all, set `VTSEARCH_SKIP_CUML=1`
(skips the cuML install) and the CPU `umap-learn`/`scikit-learn` paths run
without any toolchain risk.
