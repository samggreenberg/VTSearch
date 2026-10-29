# Deployment & Operations Guide

This document covers running VTSearch as a service: security, gunicorn and
Docker, the data directory, network dependencies and offline operation, then
the environment-variable reference. For installation and a first local run,
see [SETUP.md](SETUP.md); for every `python app.py` flag, see [CLI.md](CLI.md);
for every key in the two settings files, see [SETTINGS.md](SETTINGS.md); and
when something goes wrong, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

**Deploying for the first time?** In order:

1. Read [Security](#security): a default deployment has no authentication and
   no filesystem confinement.
2. Pick a [Docker image](#choosing-an-image), or run [under
   gunicorn](#running-under-gunicorn) on the host. Keep state on persistent
   storage: a volume at `/app/data` in Docker, `VTSEARCH_DATA_DIR` outside
   the checkout on a host install.
3. Set `VTSEARCH_SECRET_KEY`, bind to loopback, and put an authenticating
   [reverse proxy](#reverse-proxy) in front, with a long read timeout and
   buffering off for `/api/events`.
4. Apply any operator restrictions, either as environment variables
   (`VTSEARCH_SOLO_MEDIA_TYPE`, `VTSEARCH_HIDE_PLUGINS`,
   `VTSEARCH_DATASET_MAX_AGE_DAYS`, …; see [Application](#application)) or in
   the [server settings file](SETTINGS.md#server-tier--datasettingsjson), which is also
   where `default_labelset_source` backs every detector's labels up off the
   server.
5. Size the disk for the model cache (~4.1 GB) and the demos your users will
   pull ([Network dependencies](#network-dependencies)); air-gapped hosts
   follow [Offline deployment](#offline-deployment).
6. Point your orchestrator at the [health probes](#health-check).

## Table of Contents

1. [Security](#security)
2. [Running under gunicorn](#running-under-gunicorn)
3. [Docker production notes](#docker-production-notes)
4. [Data directory layout](#data-directory-layout)
5. [Network dependencies](#network-dependencies)
6. [Offline deployment](#offline-deployment)
7. [Environment variables](#environment-variables)
8. [Progress-bar estimates](#progress-bar-estimates)
9. [Dependency structure](#dependency-structure)

---

## Security

> **A default VTSearch deployment has no authentication and no filesystem
> confinement. Anyone who can reach the port gets full use of the app and,
> through it, read and write access to every path the server process can
> reach. Do not put one on an untrusted network — including the public
> internet — without the measures in [Before you expose
> it](#before-you-expose-it) below.**

This is a deliberate design for the deployment VTSearch is built around: one
trusted user, on their own workstation or a private host. It is *not* a
hardened multi-tenant server, and nothing in the app assumes it is.

### What the default deployment grants

With no `--login` flag the app runs `DefaultLoginProvider`: every request is
treated as the single user `default`, always authenticated. There is no
password, token, session, or login step to obtain — the request lifecycle
identifies the caller but never challenges them. Concretely, an unauthenticated
caller can:

- **Read any server-readable path.** `GET /api/browse` roots at the filesystem
  root (`/`) in single-user mode, and server-path importer fields accept any
  absolute or relative path. `get_file_access_base_dir()`
  (`vtscore/security/path_validation.py`) returns `None` for the default
  provider, which tells `validate_server_filepath()` to apply **no**
  containment check at all. Imported file contents are then served back
  through the media routes.
- **Write any server-writable path.** Server-file exporters
  (`server_json_file`, `server_csv_file`, portable-detector export, the
  server-JSON label source) take a destination path as a free-text field and
  create parent directories as needed, under the same `None` confinement.
- **Make the server issue outbound requests.** The HTTP-archive importer, the
  URL datasource importer and the webhook exporter fetch/post to a
  user-supplied URL (these go through the SSRF guard in
  `vtscore/security/url_validation.py`, which rejects non-HTTP(S) schemes and
  private/internal addresses), and the email exporter connects to the
  recipient's mail server.
- **Read and change every setting and every dataset/detector on the instance.**
  There is one shared `data/` tree with no per-user boundary.

The bound host makes this reachable by default: `python app.py` listens on
`0.0.0.0:5000`, and gunicorn's `VTSEARCH_BIND` defaults to `0.0.0.0:5000`. On a
cloud or multi-homed host that is every interface, not just loopback.

### Before you expose it

- **Put an authenticating proxy in front, and bind VTSearch to loopback.** Set
  `VTSEARCH_BIND=127.0.0.1:5000` (or publish only to a private Docker network)
  and terminate authentication — mTLS, OIDC/SSO, or HTTP basic auth — in the
  reverse proxy, which then forwards to gunicorn. **Today this is the only way
  to actually require authentication**; see the next bullet for why the
  built-in providers are not a substitute.
- **Understand what `--login` does and does not do.** Selecting a non-default
  provider (`--login trivial`, `--login api_key`) makes
  `get_file_access_base_dir()` return `data/<username>/`, which *does* confine
  the browse root and every server-path importer/exporter to that subtree.
  Whether it also *gates* access depends on the provider: the `_enforce_auth`
  `before_request` hook (`vtsearch/runtime/request_hooks.py`) rejects unauthenticated `/api/*`
  requests with 401 only when the provider's `enforce_auth()` is true.
  `--login api_key` enforces — a request without a valid Bearer token (keys in
  `data/api_keys.json`) gets 401, except `/api/auth/status`, `/api/auth/login`
  and `/api/auth/logout`. `--login trivial` deliberately does **not**: it has no password,
  so its login screen is a way to pick an identity, not an access control, and
  a request with no cookie is served as `anonymous`. So `api_key` is real
  authentication for headless clients; for browser users, the proxy above is
  still the only authentication. `--login` is a dev-server flag: under
  gunicorn (every Docker image) there is no flag or environment variable for
  it, so a gunicorn deployment runs the default single-user provider unless
  you install another in code (`vtsearch.auth.set_login_provider`).
- **Set `VTSEARCH_SECRET_KEY` to a random value.** The default is a constant
  visible in the source, and the signed session cookie is only
  integrity-protected by that key. Two features trust what the cookie says:
  `--login trivial` keeps the username in it, and Sign in with HuggingFace
  (live whenever `HF_OAUTH_CLIENT_ID` is set, under any login provider) keeps
  its OAuth `state` and PKCE verifier there. When either is active and the
  variable is unset, startup replaces the default with a random key for that
  process and logs a warning, so nobody can forge a cookie from the source —
  but sessions then end at every restart. That is safe with the single worker
  `gunicorn.conf.py` runs; a deployment with more than one worker **must** set
  the variable, or each worker rejects the cookies the others signed.
  Single-user mode without HuggingFace sign-in trusts no cookie and keeps the
  default.
- **Let the OS be the real boundary.** In single-user mode the only limit on
  file access is what the process account can read and write, so run VTSearch
  as an unprivileged, dedicated user (or in a container) whose only writable
  mount is the data directory. Do not run it as root; the Docker images
  already run as the unprivileged UID 1000.
- **Keep the media you import in mind.** Dataset pickles are loaded through a
  restricted unpickler (`vtscore/security/pickle.py`) that permits only plain
  types and numpy arrays, so a `.pkl` cannot execute code — but everything a
  user can point an importer at is content the server will read and serve.

---

## Running under gunicorn

For production, run the app under [gunicorn](https://gunicorn.org/) using
the bundled config:

```bash
VTSEARCH_SERVER_INIT=1 gunicorn -c gunicorn.conf.py app:app
```

`python app.py` runs Flask's built-in dev server and is intended for
development only. Every Docker image under `docker/` already uses gunicorn
via its CMD, so this only applies if you are running outside Docker.

Gunicorn never parses `python app.py`'s flags. The operator restrictions
(`--solo-media-type`, `--hide-plugin`, `--semantic-only`, …) have
environment-variable forms for this case (see [Application](#application));
`--login` and `--port` do not (bind with `VTSEARCH_BIND`; for authentication
see [Before you expose it](#before-you-expose-it)).

### Why `VTSEARCH_SERVER_INIT=1`?

`app.py` runs its startup sequence (model init, embedder preload) from
its `if __name__ == "__main__":` block.
Gunicorn **imports** `app.py` rather than executing it, so that block
never runs. Setting `VTSEARCH_SERVER_INIT=1` tells `app.py` to also run
`initialize_server()` at import time. The Dockerfiles set this env var
automatically.

It applies only when `app.py` is imported. Under `python app.py` the
variable is ignored, because the `__main__` block already decides whether a
server starts. So a `python app.py --autodetect` run inside a container
stays headless even though the image sets the variable for every process.

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

`gunicorn.conf.py` reads `VTSEARCH_BIND`, `VTSEARCH_THREADS` and `VTSEARCH_TIMEOUT` (see [Gunicorn / WSGI](#gunicorn--wsgi-production) in the environment-variable reference), plus `VTSEARCH_LOG_LEVEL` as its own log level — at `info`/`debug` that also turns on gunicorn's access log, streamed to stdout. Raising `VTSEARCH_THREADS` raises the concurrent-request ceiling and, with it, the SSE connection cap.

For larger tuning changes, edit `gunicorn.conf.py` directly.

### Reverse proxy

For public deployments, put nginx / Caddy / Traefik in front of gunicorn
to handle TLS, gzip, and static-asset caching. Flask serves the Angular
build from `static/` directly, but a dedicated reverse proxy is much
more efficient for that traffic.

> **Read [Security](#security) first.** VTSearch itself does not
> authenticate anyone and, in the default single-user mode, does not confine
> file access. A proxy that only terminates TLS publishes an unauthenticated
> filesystem browser. The proxy is also where you add authentication — and
> gunicorn should be bound to loopback (`VTSEARCH_BIND=127.0.0.1:5000`) so it
> can't be reached around it.

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
  wrapper retries on a timer when that happens (a cap rejection proves the
  backend is alive, so it never counts toward the client's offline circuit
  breaker). Raising `VTSEARCH_THREADS` raises the SSE cap along with it. The
  cap only applies under gunicorn: the dev server (`python app.py`) spawns a
  thread per connection and runs uncapped unless `VTSEARCH_SSE_MAX_CONNECTIONS`
  is set explicitly.
- **Forwarded headers.** Forward `Host` and `X-Forwarded-Proto`. Gunicorn
  trusts `X-Forwarded-Proto` from loopback (its default
  `forwarded_allow_ips`), which is how the app learns the external `https`
  scheme; the app installs no `ProxyFix`, so it does not read
  `X-Forwarded-For` or `X-Forwarded-Host`. The one URL the app builds from the
  request is the HuggingFace OAuth callback: set `HF_OAUTH_REDIRECT_URI` when
  the public URL differs from what the app sees.

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

## Docker production notes

Build and run commands for every image live in [SETUP.md § Docker](SETUP.md#docker);
this section covers what matters when you operate one. Every image runs the app
under gunicorn with the bundled `gunicorn.conf.py` (single worker + gthread
threads, not Flask's dev server) and sets `VTSEARCH_SERVER_INIT=1` so the
startup sequence runs at WSGI import time. See
[Running under gunicorn](#running-under-gunicorn) for tuning.

### Choosing an image

| Dockerfile | Compose file (`docker/compose/`) | Base | What it is for |
|---|---|---|---|
| `docker/Dockerfile` | `docker-compose.yml` | `python:3.12-slim` | Full CPU build (all media types). Installs `libsndfile1`, `ffmpeg`, `libgl1`, `libglib2.0-0`. |
| `docker/Dockerfile.gpu` | `docker-compose.yml` + `docker-compose.gpu.yml` | `nvidia/cuda:12.9.1-runtime-ubuntu24.04` | Full GPU build; needs the NVIDIA Container Toolkit on the host. |
| `docker/Dockerfile.labbench` | `docker-compose.labbench.yml`, `docker-compose.labbench-audio.yml` | `python:3.12-slim` | The slim build for image and audio, or for one of them (`MEDIA_TYPE` build arg: unset, the default, for both; `image` or `audio` for one, locked to it): only those types' dependencies, each type's embedder locked, the embedders' weights baked under `/opt/vtsearch/models`, and every dataset expiring after 14 days. What it locks and hides, and why each LabBench variant needs its own data volume: [SETUP.md § LabBench](SETUP.md#labbench-slim-image-or-audio). |
| `docker/Dockerfile.solo` | `docker-compose.audio.yml`, `docker-compose.video.yml` | `python:3.12-slim` | The full CPU build locked to the media type given as the required `MEDIA_TYPE` build arg, with that type's default embedder weights baked under `/opt/vtsearch/models`; see [SETUP.md § Single media type](SETUP.md#single-media-type-audio-only-video-only). |
| `docker/Dockerfile.image-embedders` | — (build directly) | `python:3.12-slim` | Every image embedder, with SigLIP, SigLIP 2, CLIP, DINOv2, DINOv3 and EUPE weights baked in (the SO400M models download lazily). |
| `docker/Dockerfile.image-embedders.gpu` | `docker-compose.image-embedders.gpu.yml` | `nvidia/cuda:12.9.1-runtime-ubuntu24.04` | The same on CUDA. DINOv3 is gated: populate the build cache first with `HF_TOKEN=… scripts/cache_gated_models.sh`. |

All of them persist state in the volume mounted at `/app/data` (named
`vtsearch-data` in the compose files); a container without that mount loses its
settings, datasets and detectors on restart.

The two CUDA images start only on a host whose NVIDIA driver the base image
accepts: one new enough for CUDA 12.9 (driver 575 or later), or one from the
R535, R550, R560, R565 or R570 branches. Anything else is refused at
`docker run` with an `unsatisfied condition: cuda>=12.9` error. Check with
`nvidia-smi`. The images ship torch's `cu129` build, which carries no Volta
(`sm_70`) kernels: on a V100 host use `scripts/install.sh cu126` instead (see
[SETUP.md § Picking the CUDA tag](SETUP.md#picking-the-cuda-tag)).

### Resource considerations

- **Memory**: Each embedding model uses ~500 MB–1.5 GB of RAM when loaded
  (the heaviest alternatives more). With the five
  [prefetched models](#embedding-models-huggingface-hub) loaded at once,
  expect ~4–6 GB total application memory. Models are loaded lazily; only the
  embedders a dataset or detector uses are loaded.
- **Disk**: The model cache (`data/models/`, or `VTSEARCH_MODELS_DIR`) holds
  ~4.1 GB for the prefetched set, plus any alternative embedder someone picks.
  Dataset embeddings and media files vary by dataset size.
- **CPU**: the server sizes its torch/OpenMP/MKL thread pool to the
  container's CPU allocation (the `OMP_NUM_THREADS=1` in the Dockerfiles is
  overwritten at startup). Cap it with `VTSEARCH_TORCH_THREADS` (e.g.
  `docker run -e VTSEARCH_TORCH_THREADS=2 …`) if memory matters more than
  throughput, or limit the container's CPUs.
- **GPU (optional)**: GPU mode accelerates embedding computation and model
  training. Not required for basic operation.

### Health check

The app serves two probes at the root of the URL space (`vtsearch/routes/health.py`):
`/healthz` (liveness — 200 whenever the process is serving) and `/readyz`
(readiness — 503 until every readiness check passes, e.g. while embedders are
still warming up). None of the images declares a `HEALTHCHECK`; add one in your
orchestrator:

```bash
curl -fsS http://localhost:5000/healthz || exit 1   # liveness
curl -fsS http://localhost:5000/readyz  || exit 1   # readiness
```

### Stamping the version into an image

`vtsearch.__version__` is normally
derived from git at import time, but the Docker build context excludes `.git`,
so the Dockerfile reads the version from a build arg instead. None of the
`docker/compose/*.yml` files pass it, so a plain `docker compose build` bakes
the fallback `0.0.0-unknown` into the image. To stamp the real version, pass it
explicitly, e.g.:

```bash
docker compose -f docker/compose/docker-compose.yml build \
  --build-arg VTSEARCH_VERSION="$(TZ=UTC git log -1 --format=%cd --date=format-local:%Y-%m-%dT%H:%M:%SZ HEAD)"
```

The Dockerfile writes this into `vtsearch/_version.txt`; `__init__.py` reads it
when git is unavailable.

---

## Data directory layout

The `data/` directory (or `/app/data` in Docker, or whatever
`VTSEARCH_DATA_DIR` points at) holds all runtime state. It is created
automatically on first startup.

```
data/
├── models/                           # HuggingFace model cache (VTSEARCH_MODELS_DIR; ~4.1 GB for the prefetched set)
├── saved_datasets/                   # Every registered dataset's pickle (ds_<id>.pkl) and its derived sidecars
├── dataset_registry.json             # Index of registered datasets (names, pickle paths, expiry)
├── detectors/                        # Detector definitions (.json, labelsets with origins)
├── detector_registry.json            # Index of registered detectors
├── embeddings/                       # Cache of embedded demo datasets (.pkl), keyed by demo + embedder
├── settings.json                     # SERVER settings only (see below)
├── user_settings.json                # Preferences of the single-user "default" user
├── api_keys.json                     # Only with `--login api_key`: hashed API keys
├── <username>/                       # Multi-user only: one subtree per user
│   ├── user_settings.json            #   that user's preferences
│   └── ...                           #   their example media; the root their server paths are confined to
├── local_uploads/                    # Files uploaded through the browser
├── staging/                          # Scratch space for in-flight imports
├── logs/                             # App logs written by the SLURM launcher (VTSEARCH_LOG_FILE)
├── images/, video/, synthetic/       # Demo/synthetic media materialised by the downloaders
└── ESC-50-master/, gtzan/, ...       # Extracted demo dataset sources
```

The per-user subtree only appears under a multi-user login provider: the
single-user default keeps `user_settings.json` directly in the data dir. See
[SETTINGS.md](SETTINGS.md) for what goes in which file.

### What to preserve vs. what's safe to delete

| Path | Preserve? | Why |
|------|-----------|-----|
| `data/models/` | **Yes** | Re-downloading is slow (~4.1 GB) |
| `data/embeddings/` | Optional | Embedded demo-dataset cache; losing it means re-embedding a demo on its next load |
| `data/settings.json` | **Yes** | Server settings: directories, concurrency limits, deployment locks |
| `data/user_settings.json`, `data/<username>/user_settings.json` | **Yes** | Every user preference, including each user's AutoFind detector list |
| `data/detectors/`, `data/detector_registry.json` | **Yes** | Persistent detector definitions with labelsets |
| `data/saved_datasets/`, `data/dataset_registry.json` | **Yes** | Every registered dataset (pickle + registry entry); losing them means re-importing |
| `data/api_keys.json` | **Yes** (if used) | API keys for `--login api_key` |
| `data/images/`, `video/`, `synthetic/` | Safe to delete | Demo/synthetic media; re-materialised on the next demo load |
| `data/staging/` | Safe to delete | Scratch space for in-flight imports |
| Demo dataset archives (`.zip`, `.tar.gz`) | Safe to delete | Can be re-downloaded |
| Extracted demo folders (`ESC-50-master/`, etc.) | Safe to delete | Can be re-extracted from archives |

---

## Network dependencies

### Embedding models (HuggingFace Hub)

VTSearch downloads embedding models from the HuggingFace Hub on first
use. Each model is lazy-loaded when a dataset of the corresponding media
type is opened for the first time. At startup, VTSearch also runs a smart-preload pass that
warms every embedder referenced by the dataset and detector registries
(see `predict_embedders_to_preload()` in `vtscore/embedding/loader.py`),
so the first request that uses each embedder doesn't pay the cold-load
cost. On an empty registry, nothing is preloaded.

`scripts/download_models.sh` prefetches the most commonly used models for
offline deployments:

| Model | Media type | HuggingFace ID | Approx. size |
|-------|-----------|----------------|-------------|
| CLAP General | Audio (default) | `laion/larger_clap_general` | ~1.4 GB |
| SigLIP | Image (default) | `google/siglip-base-patch16-224` | ~400 MB |
| CLIP | Image (alternative) | `openai/clip-vit-base-patch32` | ~600 MB |
| X-CLIP | Video (default) | `microsoft/xclip-base-patch32` | ~1.2 GB |
| E5 | Text (default) | `intfloat/e5-base-v2` | ~440 MB |

**Total: ~4.1 GB** for the five models `download_models.sh` fetches (four
defaults plus CLIP, the image alternative). At runtime, only the embedders a
dataset or detector actually uses are loaded. The smaller
`laion/clap-htsat-unfused` checkpoint (the `clap` embedder, ~1.1 GB) is *not*
prefetched: it is an alternative, so it downloads on first use like any
other non-default embedder.

#### Alternative embedders

Every other embedder in the registry is downloaded only when explicitly
selected. The authoritative roster — every embedder with its model ID —
is the generated table in [docs/ML.md § Embedding
Models](ML.md#embedding-models). Model sizes vary from a few hundred MB
to several GB (ParaSpeechCLAP, which chains WavLM and Granite encoders,
is the largest at ~4.5 GB across its three checkpoints).

No HuggingFace account or token is required for the public models. DINOv3 is
gated: it needs an accepted license plus a token, from `HF_TOKEN` or a
server-wide **Sign in with HuggingFace** (see
[HuggingFace / PyTorch](#huggingface--pytorch)).

### Face models

Face (the `image2face` converter, the `face` media type and its embedder) runs on face models
the deployment supplies. **The app never downloads them**, so a fresh install has no Face, and
Face is on exactly when the server finds them:

1. the pipeline config `IMAGEKIT_FACE_CONFIG` names, if it is set;
2. otherwise `face.json` in the models folder's `imagekit` directory. The
   models folder is `VTSEARCH_MODELS_DIR`: `data/models/` by default, and
   `/opt/vtsearch/models` in the solo, labbench and image-embedders images.

The config names a face detector and a face embedder, each relative to the config's own
directory, so a models directory moves as one unit:

```
data/models/imagekit/
├── face.json
├── det_500m.onnx
├── w600k_mbf.onnx
└── arcface_mbf.json
```

```json
{
  "detector": {"model_bundle": "det_500m.onnx"},
  "embedder": {"model": "w600k_mbf.onnx", "config": "arcface_mbf.json"}
}
```

Where the weights come from, which ones work, and the full config schema are in
[`imagekit/README.md`](../imagekit/README.md#models-where-to-get-them). **Check the licence of
any weights you deploy**: the InsightFace model-zoo weights (`buffalo_s`, above, the GRID's
default; `buffalo_l` and `buffalo_m` likewise) are for non-commercial research only.
`arcface_mbf.json` is a copy of `imagekit/face/extraction/arcface_mbf.json`; `buffalo_l`'s
ResNet-50 recogniser takes `arcface_r50.json` instead, a different embedding space (compat tag),
so changing packs means re-importing every face dataset. `IMAGEKIT_FACE_PROVIDER` picks the backend, else the
config's own `provider` key, else `oss`; the `ext` provider also needs its backend module installed
(`IMAGEKIT_EXT_BACKEND` names it, else the config's `ext_backend` key, else `ext_backend`).

The check reads the config and looks for each model file it names; it loads nothing. It runs
again every couple of seconds, so models put in place while the server runs turn Face on with no
restart. The startup log says which way it went:

```
🙂 Face: on (oss provider, /app/data/models/imagekit/face.json)
😶 Face: off, no face models: put a pipeline config at /app/data/models/imagekit/face.json, or set IMAGEKIT_FACE_CONFIG to one ...
```

While Face is off, the Face output, its media type and its embedder are left out of every
picker, and an import that names `image2face` anyway fails on its first photo with the same
reason. Face datasets saved earlier still open: their vectors are in the pickle. In Docker,
mount the models directory (e.g. `-v ./face-models:/app/data/models/imagekit:ro` for the main
images), or mount it anywhere and set `IMAGEKIT_FACE_CONFIG`.

The face models run on the device `VTSEARCH_DEVICE` resolves to, which needs `onnxruntime-gpu` for
CUDA. `docker/Dockerfile.gpu` and a CUDA `scripts/install.sh` install it in place of the CPU
`onnxruntime`, so there Face uses the GPU. The image-embedders images install neither build, nor
OpenCV, so they have no Face.

### Demo dataset downloads

Demo datasets are downloaded only when a user selects them through the web
UI. They are **not** downloaded at startup.

**[docs/demos.md](demos.md) is the catalogue** — every demo, grouped by media
type, with the size variants (S / M / L / A) each one offers and per-demo
download sizes, generated from the demo-dataset registry. It is not
duplicated here: a second copy of the roster would rot the moment a demo is
added or its upstream source moves.

What matters for capacity planning: the underlying sources range from a few MB
(the text corpora) to several GB (the largest audio and image archives), they
land under `DATA_DIR` as both a temporary archive and an extracted directory,
and every size variant of a demo shares one download. Provision the data volume
for the demos your users will actually pull rather than the whole catalogue, and
see [Sharing demo downloads between data dirs](#sharing-demo-downloads-between-data-dirs-multi-user-servers)
below when several instances live on one host.

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
| URL datasource importer (one example item) | User-provided URL | Use the server-file datasource importer, or upload the file |
| Webhook exporter | User-provided endpoint | Use server file exporter |
| Email exporter | Recipient's mail server (DNS MX lookup, then SMTP) | Use server file exporter |

### pip install during build

Besides PyPI, CPU builds fetch PyTorch wheels from `https://download.pytorch.org/whl/cpu`.
GPU builds fetch them from the CUDA-tagged PyTorch index (`…/whl/cu129` in
`docker/Dockerfile.gpu`; `install.sh` picks the tag per host) and the RAPIDS/cuML
wheels from `https://pypi.nvidia.com`. Mirror these for an air-gapped build.

---

## Offline deployment

VTSearch can run fully offline once models are cached. Follow these steps:

### 1. Pre-download models

Run the provided script on a machine with internet access:

```bash
./scripts/download_models.sh [CACHE_DIR]
```

This downloads all five embedding models (CLAP General, SigLIP, CLIP, X-CLIP, E5) to `CACHE_DIR` (defaults to
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
# Add to the end of Dockerfile, before CMD. Keep the cache OUTSIDE /app/data:
# a volume mounted there would hide anything baked under it.
COPY ./pre-downloaded-models/ /opt/vtsearch/models/
ENV VTSEARCH_MODELS_DIR=/opt/vtsearch/models \
    HF_HUB_OFFLINE=1
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
| HTTP archive / URL importers | Those importers fail | Network error in importer |
| Webhook / email exporters | Those exporters fail | Connection error on export |

Everything else (folder/pickle import, server file export, ML training,
evaluation, sorting, voting) works fully offline.

---

## Environment variables

### Application

| Variable | Default | Description |
|----------|---------|-------------|
| `VTSEARCH_DATA_DIR` | `<repo root>/data` | **Relocates the entire data directory** (settings, per-user dirs, model cache, embeddings, detectors, media, demo downloads). The default is anchored to the *repository root*, not the current working directory, so a systemd unit or cron job started from elsewhere still finds existing state. This is the one variable most production deployments need: point it at a persistent volume outside the checkout so a redeploy never touches user state. |
| `VTSEARCH_MODELS_DIR` | `$VTSEARCH_DATA_DIR/models` | HuggingFace model cache. Split it out from the data dir when several instances should share one ~4.1 GB model download. Read independently of `VTSEARCH_DATA_DIR`, so setting only this variable works even with the data dir unset. |
| `VTSEARCH_SECRET_KEY` | `vtsearch-dev-key-change-in-production` | Flask session secret key (**set this to a random value in production**). Left unset with `--login trivial` or `HF_OAUTH_CLIENT_ID` in force, startup swaps in a random per-process key and warns; see [Security](#security). |
| `VTSEARCH_PORT` | `5000` | Bind port for the **dev server** (`python app.py`); `--port` wins over it. Gunicorn ignores this — use `VTSEARCH_BIND` below. |
| `VTSEARCH_LOG_LEVEL` | `WARNING` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). `INFO`/`DEBUG` also turn on the per-request access log. |
| `VTSEARCH_LOG_FORMAT` | `json` | Log record format: `json` (one JSON object per line, for log aggregators) or `text` (bracketed-tag human-readable form, for local dev). Every record carries the active user, `dataset_id`, `detector_id`, and `request_id`. |
| `VTSEARCH_LOG_FILE` | unset | Also append every log record to this file (the terminal stream stays). The SLURM launcher sets it to `data/logs/app-<node>-<timestamp>.log` so a stall nobody was watching still leaves a trace; see [Diagnosing a stall](TROUBLESHOOTING.md#the-app-freezes-for-seconds-during-labeling-diagnosing-a-stall). |
| `VTSEARCH_DIAGNOSE` | unset | Truthy turns on the whole diagnostic bar set at once: `VTSEARCH_LOG_LEVEL=INFO`, `VTSEARCH_SLOW_REQUEST_MS=400`, `VTSEARCH_SLOW_PHASE_MS=150` (and, by the coupling below, a 75 ms GC bar). Each is a default, so any variable you set yourself still wins. It deliberately does **not** pin `VTSEARCH_GC_WARN_MS`, because pinning it would bypass that coupling. |
| `VTSEARCH_SLOW_REQUEST_MS` | `1000` | A request whose handler takes at least this long is logged at WARNING with its method, path, status, duration, thread CPU time, GC time and `request_id` (the same id the browser sees as `X-Request-Id`). Below the bar, and only at `VTSEARCH_LOG_LEVEL=INFO`, the same figures are logged as `request trace:` so a diagnostic run has the whole chain to add up. |
| `VTSEARCH_STALL_WATCHDOG_MS` | `1000` | Heartbeat-miss threshold for the stall watchdog: when the interpreter cannot run the heartbeat thread for this long, a WARNING names the thread that burned the wall clock (or reports that none did), just after every thread's stack, taken the moment the heartbeat wakes. `0` disables the watchdog. |
| `VTSEARCH_STALL_STARTUP_MS` | `5000` | The watchdog's threshold while the server starts, from the ML imports until **VTSearch is ready**: cold-cache disk reads during the imports hold the GIL long enough to look exactly like a stall, and at the normal bar a routine start would dump every thread's stack. Never lower than `VTSEARCH_STALL_WATCHDOG_MS`; `0` keeps that bar throughout. A `stall:` line from the startup window says `bar 5000ms`. |
| `VTSEARCH_STALL_DUMP_FILE` | `VTSEARCH_LOG_FILE`, else stderr | Where the watchdog writes the thread stacks. |
| `VTSEARCH_STALL_LIVE_DUMP` | unset | **Can crash the app; off by default** (#4345). `1` also has `faulthandler` dump every thread's frames *during* a stall. That dump reads other threads' frames without the GIL while they run, and has segfaulted the app. Use it only for a diagnostic session chasing a GIL hold that the stacks taken at wake do not explain. |
| `VTSEARCH_GC_WARN_MS` | half `VTSEARCH_SLOW_PHASE_MS`, capped at `200` | A garbage-collection pause at least this long is logged at WARNING with its generation and duration. Unset it tracks the phase threshold, so a collection can never be too small to report while still being large enough to inflate the phase it lands in. |
| `VTSEARCH_GC_FREEZE` | `1` | After the model preload, `gc.freeze()` moves the imported ML libraries and the loaded embedders into the permanent generation, which full collections skip, removing the periodic multi-hundred-ms gen-2 pauses that froze every in-flight request (#3870). Datasets and detectors load lazily afterwards and stay collectable. Set falsey to skip it. |
| `VTSEARCH_SLOW_PHASE_MS` | `500` | Threshold for the internal phase breakdowns (learned-sort retrain, per-vote labelset rewrite, labeling-status replay, vote rehydrate) and for waits on the locks those paths share; each logs one WARNING line at or above it. |
| `VTSEARCH_DETECTOR_WRITE_MODE` | `async` | Where the per-vote labelset rewrite runs. `async` (default) composes the merged labelset on the request thread and hands the `fsync` + rename of the detector JSON to a background writer, so a slow filesystem (a shared NFS export, say) never holds `POST /api/medias/<id>/vote`. Every in-process reader sees the queued text before it lands, a burst of votes coalesces into one write of the newest text, and a failed write is raised as a 500 from the *next* vote. `sync` writes inline on the request thread; the test suite runs that way. |
| `VTSEARCH_MAX_UPLOAD_MB` | `2048` | Maximum size of a single HTTP request body, in MB (Flask's `MAX_CONTENT_LENGTH`). Oversize uploads are rejected with HTTP 413 before they consume disk. Set to `0` to disable the cap entirely for genuinely large-archive uploads. |
| `VTSEARCH_SSE_MAX_CONNECTIONS` | `VTSEARCH_THREADS - 2` (i.e. `6`) | Hard cap on concurrent `/api/events` streams. Each stream holds a gunicorn worker thread for its lifetime, so the default reserves headroom for ordinary REST requests. The Flask dev server spawns a thread per connection and therefore uncaps this automatically unless you set it explicitly. |
| `VTSEARCH_RUNDIR` | system temp dir | Directory for the single-instance port lockfiles. Set it when several users run VTSearch on one host and a shared `/tmp` lockfile would collide. |
| `VTSEARCH_SUPPORT_EMAIL` | unset (persisted `support_email`, else the built-in project address) | Env-var form of `--support-email`: the Help modal's "Email us" recipient. |
| `VTSEARCH_ON_DATASET_IMPORTED` | unset | Env-var form of `--on-dataset-imported`: comma-separated `module:function` hooks called when a web import succeeds or fails ([EXTENDING.md § Dataset-Import Hooks](EXTENDING.md#dataset-import-hooks)). A spec that fails to load is reported on stdout and none of the variable's hooks are installed. Deliberately not a setting: nothing in the settings file can name code to run. |
| `VTSEARCH_SEMANTIC_ONLY` | unset | Env-var form of `--semantic-only` (`1`/`true`/`yes`/`on`): only Semantic embedders are offered or accepted. |
| `VTSEARCH_AUTOPILOT_ONLY` | unset | Env-var form of `--autopilot-only` (`1`/`true`/`yes`/`on`): Train and Tune show only their Autopilot tabs. |
| `VTSEARCH_DATASET_MAX_AGE_DAYS` | unset (datasets never expire) | Env-var form of `--dataset-max-age-days`: new datasets expire this many days out. Positive integers only. |
| `VTSEARCH_SOLO_MEDIA_TYPE` | unset | Env-var form of `--solo-media-type`: lock the instance to one registered media type (`python app.py --list-media-types`). |
| `VTSEARCH_SOLO_EMBEDDERS` | unset | Env-var form of the repeatable `--solo-embedder`, as comma-separated `TYPE=EMBEDDER` pairs (`image=siglip,audio=clap`). A per-process fallback that a user's own choice overrides, not a lock. |
| `VTSEARCH_HIDE_PLUGINS` | unset | Env-var form of the repeatable `--hide-plugin`, as comma-separated `FAMILY:NAME` pairs (`embedders:e5,importers:synthetic`). Unioned with the persisted `hidden_plugins` setting; a UI declutter, not a security boundary. |

The eight `Env-var form of` variables above exist because gunicorn never
parses `argv`, so the Docker images cannot take `python app.py`'s operator
flags. What each one does, and how it combines with any persisted server
setting, is in [CLI.md § Web server modes](CLI.md#web-server-modes). In every case an explicit
flag wins over the variable, and a malformed value is printed as a warning and
ignored rather than stopping the server.

### Progress-bar timing

| Variable | Default | Description |
|----------|---------|-------------|
| `VTSEARCH_PROFILE_LOAD` | unset | Path to a JSONL sink. When set, every dataset import appends one row per phase (download, load, embed, finalize and its sub-slots), marking cold vs warm model loads and cold vs cached downloads. Developer tooling: `scripts/profiling/` fits these rows into the load cost model the app ships with. Leave it unset in production. See [Progress-bar estimates](#progress-bar-estimates). |

### Dataset-ingest concurrency

How many datasets the server downloads / embeds in parallel. Both knobs **autodetect from hardware** on every startup (no config needed): downloads scale with CPU count (cap 4); embeddings are `1` on an accelerator host (CUDA/MPS — embed jobs share the one device and serialise on it) and on a CPU host scale with the scarcer of cores and RAM (~1 job per 4 cores, ~1 per 4 GiB; cap 4), flooring to 1 on small/RAM-starved boxes so a laptop stays constrained. The env vars below override the autodetect **without** persisting to `data/settings.json` — so the same `python app.py` launch picks a small default on a laptop and a bigger one on a node that exports the var. Values are clamped to `[1, 16]`; a non-integer value is ignored (logged, falls back to autodetect). An explicit value in `data/settings.json` still wins over both; the settings UI shows these read-only.

| Variable | Default | Description |
|----------|---------|-------------|
| `VTSEARCH_MAX_CONCURRENT_DOWNLOADS` | autodetect (CPU count, cap 4) | Max datasets downloaded in parallel |
| `VTSEARCH_MAX_CONCURRENT_EMBEDDINGS` | autodetect (`1` on CUDA/MPS; else min of cores/4 and RAM/4 GiB, cap 4) | Max dataset embedding jobs run in parallel |

### Compute and model runtime

| Variable | Default | Description |
|----------|---------|-------------|
| `VTSEARCH_DEVICE` | `auto` | Preferred compute device for embedding, training, and scoring. `auto` resolves to `cuda` when a usable GPU is visible, then `mps`, then `cpu`; explicit values (`cuda`, `cuda:1`, `cpu`, `mps`) pass through unchanged. Resolution is lazy — the var records intent, torch is only imported when a device is actually needed. Pin it to `cpu` to keep a shared GPU free, or to `cuda:N` to place the process on one card of a multi-GPU box. |
| `VTSEARCH_TORCH_THREADS` | server: the process's CPU allocation; library/batch use: `1` | Thread count for `torch` and the native math libraries. When the server starts (`python app.py` **or** gunicorn importing `app.py`), `vtsearch/runtime/torch_threads.py` resolves it — the variable if set, otherwise the CPUs this process may run on (`os.sched_getaffinity`, i.e. the SLURM/cgroup/container allocation) — and exports the result as `OMP_NUM_THREADS` / `MKL_NUM_THREADS` before torch is imported, **overwriting** any value already in the environment (including the `OMP_NUM_THREADS=1` the Dockerfiles set). Code that imports `vtscore` without `app.py` (scripts, tests) defaults to `1`. Each thread allocates its own scratch buffers, so lower it on a memory-constrained box. |
| `IMAGEKIT_FACE_PROVIDER` | the config's `provider`, else `oss` | Which face backend runs Face (the `image2face` converter and the `face` embedder): `oss` (open SCRFD + ArcFace/MagFace ONNX weights on ONNX Runtime) or `ext` (an external backend module behind the same interface, which must be installed; see `IMAGEKIT_EXT_BACKEND`), or a third-party provider registered under its own name. See [Face models](#face-models) and [`imagekit/README.md`](../imagekit/README.md). |
| `IMAGEKIT_EXT_BACKEND` | the config's `ext_backend`, else `ext_backend` | The module the `ext` face provider imports when its pipeline is built (with `importlib.import_module`, so a dotted path works). It must expose `face.FaceDetector`, `face.FaceEmbeddingExtractor` and `RuntimeOptions`; the compat tag its extractor reports is stamped on every face verbatim. Face stays off, naming the module, while it is not installed. See [`imagekit/README.md`](../imagekit/README.md#provider-architecture). |
| `IMAGEKIT_FACE_CONFIG` | `data/models/imagekit/face.json` (under `VTSEARCH_MODELS_DIR`) | Path to the pipeline config JSON naming the face models (and optionally a `runtime` block). Relative model paths resolve against the file's directory. With no config whose models exist, Face is off: see [Face models](#face-models). The runtime follows `VTSEARCH_DEVICE` (install `onnxruntime-gpu` for CUDA) and `VTSEARCH_TORCH_THREADS`. |
| `VTSEARCH_EMBED_BATCH_SIZE` | `32` | Items per GPU embedding batch. Lower it if a large model OOMs on a small card; non-positive or unparseable values fall back to the default. Some embedders (e.g. video models, whose per-clip frame stacks are much larger) ship a smaller default of their own. |
| `VTSEARCH_DECODE_WORKERS` | allocated CPUs − 1, capped at 8 | Threads used to decode images *ahead* of the GPU forward during bulk image embedding. Decoding a batch inline, forwarding it, then decoding the next leaves the GPU idle for the whole decode — measured at 82% idle for base SigLIP on a V100 — so the decode runs on a pool, one batch ahead. The default is sized from the CPUs this process may actually run on (`os.sched_getaffinity`, which reflects a SLURM/cgroup allocation rather than the node's core count), leaving one for the calling thread. Set `0` to decode inline on the calling thread. Results are unaffected either way; the only cost of the pool is holding two batches of decoded images instead of one. |
| `VTSEARCH_EMBED_PRECISION` | `fp32` | Compute precision for the **image** embedding forward pass. `fp32` is full precision and the shipped default. `fp16` / `bf16` cast the weights (fastest; `bf16` needs sm_80+, so not a V100); `autocast_fp16` / `autocast_bf16` keep fp32 weights and wrap the forward in `torch.autocast`, which holds softmax and layer norm in fp32 — numerically safer, slower; `auto` picks `bf16` where supported, else `fp16`. Every half mode degrades to `fp32` off CUDA, where half is emulated and *slower* than the fp32 it replaces. Stored vectors are always fp32 — only the compute is half. **Half precision changes the vectors** (cosine similarities shift ~1e-3, against a ~1e-7 fp32 kernel-selection noise floor), so a dataset embedded in one mode is not interchangeable with one embedded in another; re-embed a collection wholesale rather than in halves. **The speedup is only on the heavy encoders.** End to end it is 2.0x (L40S) to 2.5x (V100) on `siglip2_l`, but **0.99x on the default `siglip`** on both cards — with decode overlapped, base SigLIP's forward is not the bottleneck, so the knob buys the shipped default nothing. **EmbeddingGemma 2** (`embeddinggemma2`, `embeddinggemma2_audio`) honours only the `bf16` modes: its activations overflow fp16 (NaN or silently degraded vectors, per its model card), so the `fp16` modes run it in `fp32`. This is also the escape hatch back to `fp32`. |
| `VTSEARCH_TEXT_SORT_CUT` | `guarded_tail` | Which rule draws a **typed-query** sort's display line (what is painted green, and what Find's verdicts read). `guarded_tail` keeps a two-Gaussian fit only when its components are separated, and otherwise cuts at the median plus 3 robust sigmas; `gmm_midpoint` is that fit's midpoint, the older line. Autopilot's opening samples around the midpoint under either rule, so the flag moves only the display line. Example and label-file sorts are unaffected. An unrecognised value falls back to the default. Why the guarded rule ships, and its known failure on majority-class queries: [ML.md § Threshold Calibration](ML.md#threshold-calibration). |
| `VTSEARCH_IMAGE_PROCESSOR_BACKEND` | `torchvision` | Which implementation resizes and normalises an image before the **image** encoder sees it: `torchvision` (the fast tensor path, default), `pil` (the legacy PIL/numpy one), or `auto` (whatever the installed `transformers` defaults to: PIL below 5, torchvision at 5+). **The backends produce different vectors** (a median `1 − cos` of ~1.5e-04 on `siglip2_l`, 50× the fp16 perturbation), so the default is pinned rather than `auto`: every host then agrees with the torchvision-built pre-embedded pile whichever `transformers` it resolved, and on a 4.x host that is a change of behaviour. Not every architecture ships both (DINOv3 has no PIL implementation), and `transformers` *warns and falls back* rather than raising when asked for one that does not exist, so the app reads the loaded class back and logs a warning naming the embedder when it differs. |
| `VTSEARCH_IMAGE_PROCESSOR_DEVICE` | `auto` | Where the resize/normalise above runs: `auto` (pass nothing; CPU tensors), `cpu`, or `cuda` (the `torchvision` backend's `device=` kwarg; degrades to `auto` off CUDA rather than raising). It stays at `auto` because it is **not** free numerically — GPU resampling differs from CPU torchvision by *more* than CPU torchvision differs from PIL — and the end-to-end speedup is small: ~1.09× per pile cell on `siglip`, ~1.02× on `siglip2_l`. |
| `VTSEARCH_MAX_DECODE_PIXELS` | `64000000` (64 MP) | Pixel budget for a single image decode. Sources above it are downsampled (aspect preserved) before reaching a thumbnail, embedder, or converter — all of which resize to a few hundred pixels anyway — so gigapixel panoramas and whole-slide scans import instead of exhausting memory. Ordinary photographs are never touched; crop/clip paths deliberately bypass this and decode at native size. Set to `0` to disable bounding entirely. |
| `VTSEARCH_MAX_STRUCTURAL_DETECT_PIXELS` | `2000000` (2 MP) | Resolution budget for local-feature detection in the structural (instance-matching) embedders. SIFT detection cost scales with pixel count while the keypoint set is capped regardless, so an uncapped high-resolution source pays many times over for the same descriptors — and spends them on fine texture that does not survive a rescale, so it matches worse as well as slower. Keypoints are stored in normalised coordinates and SIFT is scale-invariant, so features detected under different budgets still match each other. Set to `0` to detect at native size. |
| `VTSEARCH_TRAIN_EPOCHS` | `200` | Upper bound on epochs for the BCE gradient loop. **Does not affect the shipped detector head**, which is a linear SVM fitted by liblinear (see [ML.md](ML.md#training-configuration)); only the eval-harness head arms run that loop. |
| `VTSEARCH_TRAIN_PATIENCE` | `10` | Epochs that loop's loss may fail to improve before early-stop fires; `0` disables early-stop. Same scope as `VTSEARCH_TRAIN_EPOCHS`. |
| `VTSEARCH_SVM_HEAD_C` | `1.0` | Inverse regularisation strength of the shipped linear SVM detector head ([ML.md](ML.md#training-configuration)). Lower it to regularise harder on very few votes; raise it to trust the labels more. |
| `VTSEARCH_CALIBRATE_COUNT` | `2` | Default `calibrate_count` baked into a fresh user's settings. Each unit adds one fold-training pass per learned sort, and buys resolution on the conformal inclusion rule (a quantile rule over pooled held-out fold scores). Lower to `1` to trade calibration quality for sort latency. |
| `VTSEARCH_PROJECTION_SEED` | unset | An integer seeds the Browse map's UMAP fit, and the clustering behind its signposts, so a dataset gets the same layout and signs every time it is projected. Unset, the fit is unseeded, which keeps UMAP's parallelism on; the layout is persisted after its one fit, so it does not change between visits either way. Seeding is for reproducing a map from scratch — the user-docs screenshot harness sets it — and costs a single-threaded fit. A layout persisted under another seed (or none) is refit once a seed is set. |
| `VTSEARCH_SPOT_CHECK_SEED` | unset | An integer seeds the line's spot check (the balance's walk), so a check over the same candidate deals the same picks every time. Unset, every check draws fresh uniform picks, which is what its bound rests on; seeding is for reproducing a check from scratch — the user-docs screenshot harness sets it so the spot-check shot frames the same pick — and is not for a real session. |
| `VTSEARCH_DISABLE_CUML` | unset | Set to any non-empty, non-`0` value to force the CPU clustering libraries for UMAP / k-means even when cuML is installed and the GPU is usable. Runtime opt-out — distinct from the install-time `VTSEARCH_SKIP_CUML` below. Useful when a RAPIDS install is present but misbehaving. |

### Install-time (`scripts/install.sh`)

These are read by the installer, not the running app.

| Variable | Default | Description |
|----------|---------|-------------|
| `VTSEARCH_SKIP_CUML` | `0` | Set to `1` to skip the cuML/RAPIDS install step on a GPU box (e.g. air-gapped installs that can't reach the NVIDIA index). GPU UMAP / k-means then use the CPU fallback. The step also skips itself, with a message, on every CUDA tag but `cu126`, `cu129` (and `cu13x`); see [GPU install's cuML step](TROUBLESHOOTING.md#gpu-installs-cuml-step-skipped-on-cu118-and-cu128). |
| `VTS_CUML_CU12_SPEC` | unset | Replace the pip requirement the cuML step installs (`cuml-cu12>=26.8` on `cu126` and `cu129`, `cuml-cu13>=26.8` on `cu13x`), and run the step on a tag it would otherwise skip. For assembling a different RAPIDS on your own terms, e.g. `VTS_CUML_CU12_SPEC='cuml-cu12<26'` on `cu128`, which costs pandas dropping to 2.x. |
| `VTSEARCH_NO_AGPL` | `0` | Set to `1` to install the `*-no-agpl.txt` requirement set, omitting the AGPL-licensed dependencies. See [Installing without the AGPL dependencies](#installing-without-the-agpl-dependencies). |
| `VTSEARCH_AUTO_DRIVER` | `0` | On a box with NVIDIA hardware but no working driver, `1` installs the driver unattended instead of prompting. |
| `VTSEARCH_ASSUME_CPU` | `0` | The opposite unattended answer: `1` skips the driver and installs the CPU stack. On a non-interactive shell with neither set, the installer takes neither privileged action. |
| `VTSEARCH_AUTO_DKMS` | `0` | When a working driver is *not* DKMS-managed (so a kernel update will break it), `1` converts it in place unattended. See [Making the GPU driver survive reboots and kernel updates](TROUBLESHOOTING.md#making-the-gpu-driver-survive-reboots-and-kernel-updates). |
| `VTSEARCH_SKIP_DKMS` | `0` | The opposite unattended answer: `1` declines the DKMS conversion offer. |
| `VTSEARCH_NVIDIA_RUNFILE_URL` | unset (auto-resolved) | Pin a specific NVIDIA driver `.run` installer, or point at a locally mirrored copy for an air-gapped install. |
| `VTSEARCH_VERBOSE` | `0` | Set to `1` to stream every command's raw output instead of only surfacing it when a step fails. |

### Gunicorn / WSGI (production)

| Variable | Default | Description |
|----------|---------|-------------|
| `VTSEARCH_SERVER_INIT` | unset | Set to `1` when running under gunicorn. Triggers model init / embedder preload at import time (the Flask `__main__` block is skipped under WSGI). Settings-source sync is *not* part of it — that is per-user and lazy, on each user's first settings access. Set automatically in the Dockerfiles. |
| `VTSEARCH_BIND` | `0.0.0.0:5000` | Gunicorn bind address (`host:port`) |
| `VTSEARCH_THREADS` | `8` | Threads per gunicorn worker |
| `VTSEARCH_TIMEOUT` | `0` | Worker request timeout in seconds; `0` (default) disables. Long imports, training, and evaluation runs routinely exceed any short timeout; overriding to anything below ~1800 risks SIGKILL mid-operation. |

### HuggingFace / PyTorch

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_HUB_OFFLINE` | unset | Set to `1` to prevent any HuggingFace Hub downloads |
| `HF_TOKEN` | unset | Hub token for model and demo downloads when nobody has used **Sign in with HuggingFace** (that token wins). Needed only for gated weights (DINOv3) and gated demos, or to escape the Hub's per-IP anonymous rate limits behind shared egress such as a cluster NAT. |
| `HF_OAUTH_CLIENT_ID` | unset | Enables **Sign in with HuggingFace**, which hands the *server* a Hub token held in memory; it does not change who the VTSearch user is. Optional companions: `HF_OAUTH_CLIENT_SECRET` (confidential client), `HF_OAUTH_REDIRECT_URI` (when the public URL differs from what the app sees, e.g. behind a TLS-terminating proxy), `HF_OAUTH_SCOPES` (default `openid profile read-repos`). Setup: [api/auth.md § HuggingFace OAuth](api/auth.md#huggingface-oauth-sign-in-with-huggingface). |
| `HF_HUB_DISABLE_IMPLICIT_TOKEN` | `1` (set by app) | Stops the HF libraries picking up a token on their own (e.g. a cached `huggingface-cli login`); the app passes the token above explicitly, or none |
| `TRANSFORMERS_NO_ADVISORY_WARNINGS` | `1` (set by app) | Suppresses advisory warnings from `transformers` |
| `OMP_NUM_THREADS` | set by app | OpenMP thread count; overwritten at startup with the resolved `VTSEARCH_TORCH_THREADS` — set that instead |
| `MKL_NUM_THREADS` | set by app | Intel MKL thread count; same as above |
| `OPENBLAS_CORETYPE` | `Haswell` on x86-64 (set by app unless already set) | One set of OpenBLAS kernels on every node, so a document ranking does not depend on the CPU the server runs on (#4481): AVX-512 and AVX2 kernels round float32 differently, which moved the VLAD and tile vectors. |
| `OPENBLAS_L2_SIZE` | `2048` with the `Haswell` core (set by app unless already set) | Keeps the Haswell pin meaning the same blocking on every CPU (#4938). OpenBLAS 0.3.34, bundled with numpy 2.5.2 and later, re-tunes a forced core on an AMD Zen 4/5 CPU with a 1 MB L2: float64 matrix products and SVDs then crash, and float32 products round differently. The Haswell kernels read the L2 size nowhere else. Export it empty to let OpenBLAS detect the cache. |

### Docker / GPU

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHONUNBUFFERED` | `1` (set in Dockerfiles) | Disables Python output buffering for real-time logs |
| `NVIDIA_VISIBLE_DEVICES` | `all` (GPU Dockerfile) | GPU visibility for NVIDIA Container Toolkit |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility` (GPU Dockerfile) | GPU driver capabilities |

---

## Progress-bar estimates

Every long-running operation — importing a dataset, opening one, loading a
detector, a text search, a Find, a train-and-score, a promote — shows one
progress bar that fills across several steps. To pace that bar the server
splits it between the steps by a weight per step, shipped with the app: the
dataset import's from a cost model measured on one GPU cluster
(`vtscore/datasets/stages/_load_cost_model.py`), the rest from the hand-tuned
vectors in `vtscore/timing/tasks.py`. A deployment on very different hardware
may see a bar race one phase and crawl the next; it never affects correctness,
results, or what gets stored. There is nothing to configure.

**Imports show no remaining-time estimate.** An import's rate is set by the
network, the source's disks and the files themselves — a shared filesystem whose
throughput depends on who else is on it, a remote archive that stalls and
resumes, files that range from kilobytes to gigabytes — and its estimate swung
from "About 10 sec left" to "About 45 min left" within one job. So the
dataset-import, staging-import and labelset missing-media bars publish none
(`eta_seconds` is always `null` on their progress events); they still fill,
count and name their step, so users can see the import is moving.

**Every other bar** (opening a dataset, loading a detector, sorts, Find,
train-and-score, promote) shows an estimate, extrapolated from the job's own
elapsed time and progress. It is never exact, so the server does not publish its
raw estimate: it snaps to a coarse ladder and holds each rung until the
underlying estimate moves decisively, which is why the UI says "About 10 min
left" and keeps saying it rather than counting through every revision. A
genuinely slowing job still reports the increase; it just does not twitch.

---

## Dependency structure

`pyproject.toml` is the single source of truth for runtime and dev dependencies; how
`scripts/install.sh` picks CPU vs GPU and the CUDA wheel tag is in
[SETUP.md § Installing dependencies](SETUP.md#installing-dependencies). The
requirements files an operator may meet:

```
pyproject.toml                       ← [project.dependencies] + [project.optional-dependencies] (dev, agpl)
requirements/base.txt                ← --extra-index-url <cpu wheel index> + `-e .[dev,agpl]`
requirements/gpu.txt                 ← `-e .[dev,agpl]` (install.sh / Dockerfile.gpu set --extra-index-url)
requirements/base-no-agpl.txt        ← base.txt without the `agpl` extra (see below)
requirements/gpu-no-agpl.txt         ← gpu.txt without the `agpl` extra (see below)
requirements/labbench.txt            ← LabBench core deps, shared by every LabBench media type (standalone)
requirements/labbench-<type>.txt     ← One LabBench media type: `-r labbench.txt` + that type's deps
                                       (labbench-image.txt for SigLIP, labbench-audio.txt for CLAP General)
requirements/image-embedders.txt     ← All-image-embedders image deps (standalone; shared by the CPU
                                       and GPU Dockerfiles, which each pass their own --extra-index-url)
```

### Installing without the AGPL dependencies

Two runtime dependencies are AGPL-3.0-or-later: **`ultralytics`** (YOLO)
and **`PyMuPDF`**. Every default install path includes them — the install
script, both `requirements/{base,gpu}.txt`, and the full Docker images
(`Dockerfile`, `Dockerfile.gpu`, `Dockerfile.solo`; the LabBench and
image-embedders images leave them out) — so the features that need them work
out of the box. That is a deliberate
choice for turnkey deployments, not an accident of the requirements
fan-out: VTSearch's own Apache-2.0 grant is unaffected either way, but
AGPL terms may attach to a combined work you redistribute or operate as a
network service, and a service you host publicly is exactly the case
AGPL's network clause is about.

If your deployment cannot take copyleft code, install without them:

```bash
# Install script (CPU or GPU; picks the matching *-no-agpl.txt file)
VTSEARCH_NO_AGPL=1 bash scripts/install.sh

# pip, directly
pip install -r requirements/base-no-agpl.txt      # CPU
pip install -r requirements/gpu-no-agpl.txt       # CUDA hosts

# Docker (same for Dockerfile.solo; Dockerfile.gpu takes gpu-no-agpl.txt)
docker build --build-arg REQUIREMENTS=base-no-agpl.txt -f docker/Dockerfile -t vtsearch .
```

What you give up: the YOLO image clipper, PDF import, and the
document → image / document → text converters. Those fail with a message
naming the missing package and how to get it
(`vtscore/utils/optional_deps.py`) rather than erroring opaquely.
Everything else — every embedder, every other media type, training,
scoring, Browse — is unaffected. See [`NOTICE`](../NOTICE) for the full
licensing statement.

### Key dependencies

| Package | Version constraint | Purpose |
|---------|-------------------|---------|
| `torch` | `>=2.7` | Neural network training and inference |
| `transformers` | `>=5.18` | HuggingFace model loading (CLAP, SigLIP, CLIP, X-CLIP, …) |
| `sentence-transformers` | latest | E5 text embeddings |
| `numpy` | latest | Numeric arrays |
| `flask` | latest | Web server |
| `opencv-python-headless` | latest | Image processing (structural embedder, YOLO). **Not** used for video decoding — see [FIPS](TROUBLESHOOTING.md#video-import-crashes-with-fatal-fips-selftest-failure) |
| `ultralytics` | latest | YOLO-based image processing. AGPL-3.0; in the `agpl` extra, installed by default — see [Installing without the AGPL dependencies](#installing-without-the-agpl-dependencies) |
| `librosa` | latest | Audio analysis (spectrograms, silence splitting) |
| `soundfile` / `soxr` | latest | Audio decoding and resampling (`vtscore.media.audio.decode`) |
| `imageio-ffmpeg` | latest | Bundled ffmpeg binary; audio codecs libsndfile can't read (AAC/M4A/MP4), and all video frame decoding (`vtscore.media.video.decode`) |
| `PyMuPDF` | latest | Document (PDF/DOC/PPT) rendering and text extraction. AGPL-3.0; in the `agpl` extra, installed by default — see [Installing without the AGPL dependencies](#installing-without-the-agpl-dependencies) |
