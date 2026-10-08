# Deployment & Operations Guide

This document covers production deployment, offline operation, network
dependencies, environment variables, and data management. For basic
installation and getting started, see [SETUP.md](SETUP.md).

## Table of Contents

1. [Security](#security)
2. [Environment variables](#environment-variables)
3. [Running under gunicorn](#running-under-gunicorn)
4. [Network dependencies](#network-dependencies)
5. [Offline deployment](#offline-deployment)
6. [Data directory layout](#data-directory-layout)
7. [Progress-bar estimates](#progress-bar-estimates)
8. [Docker production notes](#docker-production-notes)
9. [Dependency structure](#dependency-structure)
10. [Troubleshooting](#troubleshooting)

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
- **Make the server issue outbound requests.** The HTTP-archive importer and
  the webhook exporter fetch/post to a user-supplied URL (these go through the
  SSRF guard in `vtscore/security/url_validation.py`, which rejects non-HTTP(S)
  schemes and private/internal addresses).
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
  `before_request` hook (`vtsearch/hooks.py`) rejects unauthenticated `/api/*`
  requests with 401 only when the provider's `enforce_auth()` is true.
  `--login api_key` enforces — a request without a valid Bearer token (keys in
  `data/api_keys.json`) gets 401, except `/api/auth/status`, `/login` and
  `/logout`. `--login trivial` deliberately does **not**: it has no password,
  so its login screen is a way to pick an identity, not an access control, and
  a request with no cookie is served as `anonymous`. So `api_key` is real
  authentication for headless clients; for browser users, the proxy above is
  still the only authentication.
- **Set `VTSEARCH_SECRET_KEY` to a random value.** The default is a constant
  visible in the source. `TrivialLoginProvider`'s session cookie is only
  integrity-protected by that key, so with the default anyone can forge a
  cookie naming any user — and therefore read that user's `data/<username>/`
  subtree.
- **Let the OS be the real boundary.** In single-user mode the only limit on
  file access is what the process account can read and write, so run VTSearch
  as an unprivileged, dedicated user (or in a container) whose only writable
  mount is the data directory. Do not run it as root.
- **Keep the media you import in mind.** Dataset pickles are loaded through a
  restricted unpickler (`vtscore/security/pickle.py`) that permits only plain
  types and numpy arrays, so a `.pkl` cannot execute code — but everything a
  user can point an importer at is content the server will read and serve.

Behaviour changes to close these gaps are tracked as open issues rather than
documented workarounds; this section describes the code as it stands.

---

## Environment variables

### Application

| Variable | Default | Description |
|----------|---------|-------------|
| `VTSEARCH_DATA_DIR` | `<repo root>/data` | **Relocates the entire data directory** (settings, per-user dirs, model cache, embeddings, detectors, media, demo downloads). The default is anchored to the *repository root*, not the current working directory, so a systemd unit or cron job started from elsewhere still finds existing state. This is the one variable most production deployments need: point it at a persistent volume outside the checkout so a redeploy never touches user state. |
| `VTSEARCH_MODELS_DIR` | `$VTSEARCH_DATA_DIR/models` | HuggingFace model cache. Split it out from the data dir when several instances should share one ~4.1 GB model download. Read independently of `VTSEARCH_DATA_DIR`, so setting only this variable works even with the data dir unset. |
| `VTSEARCH_SECRET_KEY` | `vtsearch-dev-key-change-in-production` | Flask session secret key (**set this to a random value in production**) |
| `VTSEARCH_PORT` | `5000` | Bind port for the **dev server** (`python app.py`); `--port` wins over it. Gunicorn ignores this — use `VTSEARCH_BIND` below. |
| `VTSEARCH_LOG_LEVEL` | `WARNING` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). `INFO`/`DEBUG` also turn on the per-request access log. |
| `VTSEARCH_LOG_FORMAT` | `json` | Log record format: `json` (one JSON object per line, for log aggregators) or `text` (bracketed-tag human-readable form, for local dev). Every record carries the active user, `dataset_id`, `detector_id`, and `request_id`. |
| `VTSEARCH_LOG_FILE` | unset | Also append every log record to this file (the terminal stream stays). The SLURM launcher sets it to `data/logs/app-<node>-<timestamp>.log` so a stall nobody was watching still leaves a trace; see [Diagnosing a stall](#the-app-freezes-for-seconds-during-labeling-diagnosing-a-stall). |
| `VTSEARCH_DIAGNOSE` | unset | Truthy turns on the whole diagnostic bar set at once: `VTSEARCH_LOG_LEVEL=INFO`, `VTSEARCH_SLOW_REQUEST_MS=400`, `VTSEARCH_SLOW_PHASE_MS=150` (and, by the coupling below, a 75 ms GC bar). Each is a default, so any variable you set yourself still wins. It deliberately does **not** pin `VTSEARCH_GC_WARN_MS`, because pinning it would bypass that coupling. |
| `VTSEARCH_SLOW_REQUEST_MS` | `1000` | A request whose handler takes at least this long is logged at WARNING with its method, path, status, duration, thread CPU time, GC time and `request_id` (the same id the browser sees as `X-Request-Id`). Below the bar, and only at `VTSEARCH_LOG_LEVEL=INFO`, the same figures are logged as `request trace:` so a diagnostic run has the whole chain to add up. |
| `VTSEARCH_STALL_WATCHDOG_MS` | `1000` | Heartbeat-miss threshold for the stall watchdog: when the interpreter cannot run the heartbeat thread for this long, a WARNING names the thread that burned the wall clock (or reports that none did), just after every thread's stack, taken the moment the heartbeat wakes. `0` disables the watchdog. |
| `VTSEARCH_STALL_DUMP_FILE` | `VTSEARCH_LOG_FILE`, else stderr | Where the watchdog writes the thread stacks. |
| `VTSEARCH_STALL_LIVE_DUMP` | unset | **Can crash the app; off by default** (issue #4345). Set to `1` to also have `faulthandler` dump every thread's frames *during* a stall. That dump reads other threads' frames without the GIL while they run, and has segfaulted the app mid-import. Use it only for a diagnostic session chasing a GIL hold that the stacks taken at wake do not explain. |
| `VTSEARCH_GC_WARN_MS` | half `VTSEARCH_SLOW_PHASE_MS`, capped at `200` | A garbage-collection pause at least this long is logged at WARNING with its generation and duration. Unset it tracks the phase threshold, so a collection can never be too small to report while still being large enough to inflate the phase it lands in. |
| `VTSEARCH_GC_FREEZE` | `1` | After the model preload, `gc.freeze()` moves the imported ML libraries and the loaded embedders into the permanent generation, which full collections skip (issue #3870: gen-2 pauses of ~300 ms every ~2 minutes, each freezing every in-flight request, measured to zero with this on). Datasets and detectors load lazily afterwards and stay collectable. Set falsey to skip it. |
| `VTSEARCH_SLOW_PHASE_MS` | `500` | Threshold for the internal phase breakdowns (learned-sort retrain, per-vote labelset rewrite, labeling-status replay, vote rehydrate) and for waits on the locks those paths share; each logs one WARNING line at or above it. |
| `VTSEARCH_DETECTOR_WRITE_MODE` | `async` | Where the per-vote labelset rewrite runs. `async` (default) composes the merged labelset on the request thread and hands the `fsync` + rename of the detector JSON to a background writer, so a slow filesystem never holds `POST /api/medias/<id>/vote` (issue #3853: on a shared NFS export that write's tail ran to seconds and the panel stayed black for it). Every in-process reader sees the queued text before it lands, a burst of votes coalesces into one write of the newest text, and a failed write is raised as a 500 from the *next* vote. `sync` writes inline on the request thread, as before; the test suite runs that way. |
| `VTSEARCH_MAX_UPLOAD_MB` | `2048` | Maximum size of a single HTTP request body, in MB (Flask's `MAX_CONTENT_LENGTH`). Oversize uploads are rejected with HTTP 413 before they consume disk. Set to `0` to disable the cap entirely for genuinely large-archive uploads. |
| `VTSEARCH_SSE_MAX_CONNECTIONS` | `VTSEARCH_THREADS - 2` (i.e. `6`) | Hard cap on concurrent `/api/events` streams. Each stream holds a gunicorn worker thread for its lifetime, so the default reserves headroom for ordinary REST requests. The Flask dev server spawns a thread per connection and therefore uncaps this automatically unless you set it explicitly. |
| `VTSEARCH_RUNDIR` | system temp dir | Directory for the single-instance port lockfiles. Set it when several users run VTSearch on one host and a shared `/tmp` lockfile would collide. |
| `VTSEARCH_SUPPORT_EMAIL` | built-in project address | Recipient for the Help modal's "Email us" link. Overrides the persisted `support_email` setting for the process lifetime (all users; not editable via the API). Equivalent to the `--support-email` CLI flag, for the gunicorn images that never parse `argv`; an explicit flag wins. |
| `VTSEARCH_ON_DATASET_IMPORTED` | unset | Comma-separated `module:function` specs to call when a user's dataset import from the web app succeeds or fails, e.g. to email them (see [EXTENDING.md § Dataset-Import Hooks](EXTENDING.md#dataset-import-hooks)). Each module must be importable on the server. A spec that fails to load is reported on stdout and none of the variable's hooks are installed. Env-var equivalent of `--on-dataset-imported`, for the gunicorn images; an explicit flag wins. Deliberately not a setting: nothing in the settings file can name code to run. |
| `VTSEARCH_SEMANTIC_ONLY` | unset | Set to `1`/`true`/`yes`/`on` to lock the deployment to **Semantic** embedders, hiding the prototype Patch Semantic and Structural types from every picker and rejecting them at the dataset-load / detector-create routes. Env-var equivalent of `--semantic-only`, for the gunicorn images; an explicit flag wins, and either beats the persisted `semantic_only` server setting. |
| `VTSEARCH_AUTOPILOT_ONLY` | unset | Set to `1`/`true`/`yes`/`on` to lock **Train** and **Test** to their Autopilot tabs: Train's Manual tab and Test's Review tab are removed, and neither view shows a tab bar. Env-var equivalent of `--autopilot-only`, for the gunicorn images; an explicit flag wins, and either beats the persisted `autopilot_only` server setting. |
| `VTSEARCH_DATASET_MAX_AGE_DAYS` | unset (datasets never expire) | Stamps every newly created dataset with an expiry this many days out. Positive integers only; anything else is ignored with a warning on stdout. Env-var equivalent of `--dataset-max-age-days`, for the gunicorn images; an explicit flag wins. |
| `VTSEARCH_SOLO_MEDIA_TYPE` | unset | Lock the whole instance to one mediaType: the importer and new-detector flows hide their mediaType pickers, converter offerings are filtered to converters that output this type, and that type's default embedder is preloaded at startup. Must be a registered media-type id (`audio`, `image`, `video`, `text`, `document`). Env-var equivalent of `--solo-media-type`, for the gunicorn images; an explicit flag wins, and either beats the persisted `solo_media_type` server setting. |
| `VTSEARCH_SOLO_EMBEDDERS` | unset | Comma-separated `TYPE=EMBEDDER` pairs (e.g. `image=siglip,audio=clap`) locking the embedder for those mediaTypes, so the importer modal hides its embedder picker for each. A per-process *fallback*: any user who picks their own embedder in the settings UI overrides it for themselves. Env-var equivalent of the repeatable `--solo-embedder`; an explicit flag wins. |
| `VTSEARCH_HIDE_PLUGINS` | unset | Comma-separated `FAMILY:NAME` pairs (e.g. `embedders:e5,importers:synthetic`) hidden from picker and listing API responses. Hidden plugins stay importable and callable by name — this declutters the UI, it is not a security boundary. Env-var equivalent of the repeatable `--hide-plugin`; an explicit flag wins, and the result is *unioned* with the persisted `hidden_plugins` setting (either source can add a hide; neither can un-hide). |

### Progress-bar timing

| Variable | Default | Description |
|----------|---------|-------------|
| `VTSEARCH_PROFILE_LOAD` | unset | Path to a JSONL sink. When set, every dataset import appends one row per phase (download, load, embed, finalize and its sub-slots), marking cold vs warm model loads and cold vs cached downloads. Developer tooling: `scripts/profiling/` fits these rows into the load cost model the app ships with. Leave it unset in production. See [Progress-bar estimates](#progress-bar-estimates). |

### Dataset-ingest concurrency

How many datasets the server downloads / embeds in parallel. Both knobs **autodetect from hardware** on every startup (no config needed): downloads scale with CPU count (cap 4); embeddings are `1` on an accelerator host (CUDA/MPS — embed jobs share the one device and serialise on it) and on a CPU host scale with the scarcer of cores and RAM (~1 job per 4 cores, ~1 per 4 GiB; cap 4), flooring to 1 on small/RAM-starved boxes so a laptop stays constrained. The env vars below override the autodetect **without** persisting to `data/settings.json` — so the same `python app.py` launch picks a small default on a laptop and a bigger one on a node that exports the var. Values are clamped to `[1, 16]`; a non-integer value is ignored (logged, falls back to autodetect). An explicit value set via the settings UI still wins over both.

| Variable | Default | Description |
|----------|---------|-------------|
| `VTSEARCH_MAX_CONCURRENT_DOWNLOADS` | autodetect (CPU count, cap 4) | Max datasets downloaded in parallel |
| `VTSEARCH_MAX_CONCURRENT_EMBEDDINGS` | autodetect (`1` on CUDA/MPS; else min of cores/4 and RAM/4 GiB, cap 4) | Max dataset embedding jobs run in parallel |

### Compute and model runtime

| Variable | Default | Description |
|----------|---------|-------------|
| `VTSEARCH_DEVICE` | `auto` | Preferred compute device for embedding, training, and scoring. `auto` resolves to `cuda` when a usable GPU is visible, then `mps`, then `cpu`; explicit values (`cuda`, `cuda:1`, `cpu`, `mps`) pass through unchanged. Resolution is lazy — the var records intent, torch is only imported when a device is actually needed. Pin it to `cpu` to keep a shared GPU free, or to `cuda:N` to place the process on one card of a multi-GPU box. |
| `VTSEARCH_TORCH_THREADS` | server: the process's CPU allocation; library/batch use: `1` | Thread count for `torch` and the native math libraries. When the server starts (`python app.py` **or** gunicorn importing `app.py`), `vtsearch/torch_threads.py` resolves it — the variable if set, otherwise the CPUs this process may run on (`os.sched_getaffinity`, i.e. the SLURM/cgroup/container allocation) — and exports the result as `OMP_NUM_THREADS` / `MKL_NUM_THREADS` before torch is imported, **overwriting** any value already in the environment (including the `OMP_NUM_THREADS=1` the Dockerfiles set). Code that imports `vtscore` without `app.py` (scripts, tests) defaults to `1`. Each thread allocates its own scratch buffers, so lower it on a memory-constrained box. |
| `VTSEARCH_EMBED_BATCH_SIZE` | `32` | Items per GPU embedding batch. Lower it if a large model OOMs on a small card; non-positive or unparseable values fall back to the default. Some embedders (e.g. video models, whose per-clip frame stacks are much larger) ship a smaller default of their own. |
| `VTSEARCH_DECODE_WORKERS` | allocated CPUs − 1, capped at 8 | Threads used to decode images *ahead* of the GPU forward during bulk image embedding. Decoding a batch inline, forwarding it, then decoding the next leaves the GPU idle for the whole decode — measured at 82% idle for base SigLIP on a V100 — so the decode runs on a pool, one batch ahead. The default is sized from the CPUs this process may actually run on (`os.sched_getaffinity`, which reflects a SLURM/cgroup allocation rather than the node's core count), leaving one for the calling thread. Set `0` to decode inline on the calling thread. Results are unaffected either way; the only cost of the pool is holding two batches of decoded images instead of one. |
| `VTSEARCH_EMBED_PRECISION` | `fp32` | Compute precision for the **image** embedding forward pass. `fp32` is full precision and the shipped default. `fp16` / `bf16` cast the weights (fastest; `bf16` needs sm_80+, so not a V100); `autocast_fp16` / `autocast_bf16` keep fp32 weights and wrap the forward in `torch.autocast`, which holds softmax and layer norm in fp32 — numerically safer, slower; `auto` picks `bf16` where supported, else `fp16`. Every half mode degrades to `fp32` off CUDA, where half is emulated and *slower* than the fp32 it replaces. Stored vectors are always fp32 — only the compute is half. **Half precision changes the vectors** (cosine similarities shift ~1e-3, against a ~1e-7 fp32 kernel-selection noise floor), so a dataset embedded in one mode is not interchangeable with one embedded in another; re-embed a collection wholesale rather than in halves. **The speedup is only on the heavy encoders.** End to end it is 2.0x (L40S) to 2.5x (V100) on `siglip2_l`, but **0.99x on the default `siglip`** on both cards — base SigLIP's forward stopped being the bottleneck once decode was overlapped with it, so the knob buys the shipped default nothing. (The 4.2x in #3143 was the `siglip2_l` forward measured in isolation, before that overlap landed.) This is also the escape hatch back to `fp32`. |
| `VTSEARCH_TEXT_SORT_CUT` | `gmm_midpoint` | Where a **typed-query** sort draws its green/red line, and where Autopilot's opening Bad phase votes. `gmm_midpoint` is the midpoint of a two-Gaussian fit, the shipped default. `guarded_tail` keeps that mixture (fitted to convergence) only when its components are separated, and otherwise cuts at the median plus 3 robust sigmas. On 1,120 labelled text sorts (#3826) the midpoint painted a median 43% of the collection green. The guarded line paints about as many as truly match (F1 0.17 → 0.39) and moves 4× less when the collection is resampled. It is worse on the Inclusion-0 rate cost, and it fails on queries whose matches are the majority of the collection (e.g. "a person" in COCO). Example and label-file sorts are unaffected. **It is off by default because it makes Autopilot's opening worse.** The Bad phase votes near this line, so under `guarded_tail` it samples the top of the ranking instead of its middle, and the first detectors see only near-miss negatives: Δcost +0.016 ± 0.005 over 100 clicks and +0.071 ± 0.013 at 6-20 votes in the trajectory A/B (`docs/experiments/2026-09-23-text-cut-ab-3826/REPORT.md`). Shipping it as a display-only line is #4136. An unrecognised value falls back to the default. |
| `VTSEARCH_IMAGE_PROCESSOR_BACKEND` | `torchvision` | Which implementation resizes and normalises an image before the **image** encoder sees it. `torchvision` is the fast tensor path and the shipped default; `pil` is the legacy PIL/numpy one; `auto` passes nothing and inherits whatever the installed `transformers` defaults to. The default is `torchvision` rather than `auto` because `auto` is not a property of this repository: `transformers` 5 removed the `Fast` suffix, so the bare `SiglipImageProcessor` means PIL below 5 and torchvision at 5+, and `requirements/image-embedders.txt` pins a range (`>=4.49`) that spans the flip. **The two backends produce different vectors** — they disagree on 53–59% of pixel elements and by a median `1 − cos` of ~1.5e-04 on `siglip2_l`, 50× the fp16 perturbation — so two hosts resolving different wheels used to embed differently from identical code and weights. Naming the backend removes that axis. The pre-embedded pile is torchvision-built, so on a `transformers` 5 host this is a no-op; **on a 4.x host it changes behaviour**, bringing that host into agreement with the pile instead of quietly disagreeing with it. Not every architecture ships both (DINOv3 has no PIL implementation), and `transformers` *warns and falls back* rather than raising when asked for one that does not exist — so a request is not a guarantee, and the app reads the loaded class back and logs a warning naming the embedder when it differs. Set `auto` for the pre-#3173 behaviour. |
| `VTSEARCH_IMAGE_PROCESSOR_DEVICE` | `auto` | Where the resize/normalise above runs. `auto` passes nothing (CPU tensors, today's behaviour); `cpu` is explicit; `cuda` hands the work to the GPU, which the `torchvision` backend supports through a `device=` call kwarg. `cuda` degrades to `auto` off CUDA rather than raising — an escape hatch that crashes on a laptop is not an escape hatch. It stays at `auto` because it is **not** free numerically: GPU resampling differs from CPU torchvision by *more* than CPU torchvision differs from PIL. The speedup is also smaller than it looks — 1.68× on `siglip`'s embed path in isolation but ~1.09× per pile cell, and ~1.02× for `siglip2_l`. |
| `VTSEARCH_MAX_DECODE_PIXELS` | `64000000` (64 MP) | Pixel budget for a single image decode. Sources above it are downsampled (aspect preserved) before reaching a thumbnail, embedder, extractor, or converter — all of which resize to a few hundred pixels anyway — so gigapixel panoramas and whole-slide scans import instead of exhausting memory. Ordinary photographs are never touched; crop/clip paths deliberately bypass this and decode at native size. Set to `0` to disable bounding entirely. |
| `VTSEARCH_MAX_STRUCTURAL_DETECT_PIXELS` | `2000000` (2 MP) | Resolution budget for local-feature detection in the structural (instance-matching) embedders. SIFT detection cost scales with pixel count while the keypoint set is capped regardless, so an uncapped high-resolution source pays many times over for the same descriptors — and spends them on fine texture that does not survive a rescale, so it matches worse as well as slower. Keypoints are stored in normalised coordinates and SIFT is scale-invariant, so features detected under different budgets still match each other. Set to `0` to detect at native size. |
| `VTSEARCH_TRAIN_EPOCHS` | `200` | Upper bound on epochs for the BCE gradient loop. **Does not affect the shipped detector head**, which is a linear SVM fitted by liblinear (see [ML.md](ML.md#training-configuration)); only the eval-harness head arms run that loop. |
| `VTSEARCH_TRAIN_PATIENCE` | `10` | Epochs that loop's loss may fail to improve before early-stop fires; `0` disables early-stop. Same scope as `VTSEARCH_TRAIN_EPOCHS`. |
| `VTSEARCH_CALIBRATE_COUNT` | `2` | Default `calibrate_count` baked into a fresh user's settings. Each unit adds one fold-training pass per learned sort, and buys resolution on the conformal inclusion rule (a quantile rule over pooled held-out fold scores). Lower to `1` to trade calibration quality for sort latency. |
| `VTSEARCH_PROJECTION_SEED` | unset | An integer seeds the Browse map's UMAP fit, and the clustering behind its signposts, so a dataset gets the same layout and signs every time it is projected. Unset, the fit is unseeded, which keeps UMAP's parallelism on; the layout is persisted after its one fit, so it does not change between visits either way. Seeding is for reproducing a map from scratch — the user-docs screenshot harness sets it — and costs a single-threaded fit. A layout persisted under another seed (or none) is refit once a seed is set. |
| `VTSEARCH_SPOT_CHECK_SEED` | unset | An integer seeds the line's spot check (the balance's walk), so a check over the same candidate deals the same picks every time. Unset, every check draws fresh uniform picks, which is what its bound rests on; seeding is for reproducing a check from scratch — the user-docs screenshot harness sets it so the spot-check shot frames the same pick — and is not for a real session. |
| `VTSEARCH_DISABLE_CUML` | unset | Set to any non-empty, non-`0` value to force the CPU clustering libraries for UMAP / k-means even when cuML is installed and the GPU is usable. Runtime opt-out — distinct from the install-time `VTSEARCH_SKIP_CUML` below. Useful when a RAPIDS install is present but misbehaving. |

### Install-time (`scripts/install.sh`)

These are read by the installer, not the running app.

| Variable | Default | Description |
|----------|---------|-------------|
| `VTSEARCH_SKIP_CUML` | `0` | Set to `1` to skip the cuML/RAPIDS install step on a GPU box (e.g. air-gapped installs that can't reach the NVIDIA index). GPU UMAP / k-means then use the CPU fallback. The step also skips itself, with a message, on every CUDA tag but `cu129` (and `cu13x`): RAPIDS ≥ 26.8, the first whose cudf takes the pandas 3 pinned in `pyproject.toml`, needs the CUDA 12.9 libraries only the `cu129` torch wheel pins. See [cuML crashes compiling a kernel](#cuml-crashes-compiling-a-kernel-cuda_fp8hpp--nvrtc-errors). |
| `VTS_CUML_CU12_SPEC` | unset | Replace the pip requirement the cuML step installs (`cuml-cu12>=26.8` on `cu129`, `cuml-cu13>=26.8` on `cu13x`), and run the step on a tag it would otherwise skip. For assembling a different RAPIDS on your own terms, e.g. `VTS_CUML_CU12_SPEC='cuml-cu12<26'` on a Volta card, which costs pandas dropping to 2.x. |
| `VTSEARCH_NO_AGPL` | `0` | Set to `1` to install the `*-no-agpl.txt` requirement set, omitting the AGPL-licensed dependencies. See [Installing without the AGPL dependencies](#installing-without-the-agpl-dependencies). |
| `VTSEARCH_AUTO_DRIVER` | `0` | On a box with NVIDIA hardware but no working driver, `1` installs the driver unattended instead of prompting. |
| `VTSEARCH_ASSUME_CPU` | `0` | The opposite unattended answer: `1` skips the driver and installs the CPU stack. On a non-interactive shell with neither set, the installer takes neither privileged action. |
| `VTSEARCH_AUTO_DKMS` | `0` | When a working driver is *not* DKMS-managed (so a kernel update will break it), `1` converts it in place unattended. See [Making the GPU driver survive reboots and kernel updates](#making-the-gpu-driver-survive-reboots-and-kernel-updates). |
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
| `HF_HUB_DISABLE_IMPLICIT_TOKEN` | `1` (set by app) | Disables HuggingFace auth tokens (all models are public) |
| `TRANSFORMERS_NO_ADVISORY_WARNINGS` | `1` (set by app) | Suppresses advisory warnings from `transformers` |
| `OMP_NUM_THREADS` | set by app | OpenMP thread count; overwritten at startup with the resolved `VTSEARCH_TORCH_THREADS` — set that instead |
| `MKL_NUM_THREADS` | set by app | Intel MKL thread count; same as above |
| `OPENBLAS_CORETYPE` | `Haswell` on x86-64 (set by app unless already set) | One set of OpenBLAS kernels on every node, so a document ranking does not depend on the CPU the server runs on (#4481): AVX-512 and AVX2 kernels round float32 differently, which moved the VLAD and tile vectors. |

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

`app.py` runs its startup sequence (model init, embedder preload) from
its `if __name__ == "__main__":` block.
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

`gunicorn.conf.py` reads `VTSEARCH_BIND`, `VTSEARCH_THREADS` and `VTSEARCH_TIMEOUT` (see [Gunicorn / WSGI](#gunicorn--wsgi-production) above), plus `VTSEARCH_LOG_LEVEL` as its own log level — at `info`/`debug` that also turns on gunicorn's access log, streamed to stdout. Raising `VTSEARCH_THREADS` raises the concurrent-request ceiling and, with it, the SSE connection cap.

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
prefetched: it is an alternative now, so it downloads on first use like any
other non-default embedder.

#### Alternative embedders

Every other embedder in the registry is downloaded only when explicitly
selected. The authoritative roster — every embedder with its model ID —
is the generated table in [docs/ML.md § Embedding
Models](ML.md#embedding-models). Model sizes vary from a few hundred MB
to several GB (ParaSpeechCLAP, which chains WavLM and Granite encoders,
is the largest at ~4.5 GB across its three checkpoints).

Most model downloads use `token=False`, so no HuggingFace account or API
token is required; the exception is DINOv3, which is gated and needs an
accepted license plus a token.

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
| Webhook exporter | User-provided endpoint | Use server file exporter |

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
| HTTP archive importer | That importer fails | Network error in importer |
| Webhook exporter | That exporter fails | Connection error on export |

Everything else (folder/pickle import, server file export, ML training,
evaluation, sorting, voting) works fully offline.

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
│   └── ...                           #   their datasets, detectors, media
├── local_uploads/                    # Files uploaded through the browser
├── staging/                          # Scratch space for in-flight imports
├── logs/                             # App logs written by the SLURM launcher (VTSEARCH_LOG_FILE)
├── images/, video/, synthetic/       # Demo/synthetic media materialised by the downloaders
└── ESC-50-master/, gtzan/, ...       # Extracted demo dataset sources
```

The per-user subtree only appears under a multi-user login provider: the
single-user default keeps `user_settings.json` directly in the data dir. See
[Settings file schema](#settings-file-schema) below for what goes in which file.

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

### Settings file schema

Settings live in **two tiers, in two different files**. Editing the wrong one
is the most common settings mistake on a fresh deployment, so start here:

| Tier | File | Holds | Model |
|------|------|-------|-------|
| **Server** | `data/settings.json` | Deployment-wide infrastructure and operator locks. Loaded once at startup, before any user logs in. | `ServerSettings` (`vtsearch/settings_models.py`) |
| **Per-user** | `<user data dir>/user_settings.json` | Every user preference — theme, volume, autopilot, AutoFind, panel layout, browse prefs. Resolved per request from the logged-in user. | `UserSettings` (same module) |

The per-user file lives at `data/user_settings.json` in a single-user
deployment and at `data/<username>/user_settings.json` under a multi-user login
provider. Both files are auto-created on first use and auto-saved on every
change. Both models accept extra keys, so an unrecognised key is preserved
rather than dropped.

A key the model *does* know but whose value it rejects — a hand-edit out of
range, or a value written before the field changed shape — is **ignored on
load**: the field reads as unset, so its default applies. Nothing is rewritten,
so the file keeps whatever you put in it; fix the value and it takes effect on
the next start. VTSearch does not migrate old settings shapes forward.

**If you are changing a user preference, edit `user_settings.json`, not
`settings.json`.** A `theme` or `autopilot_enabled` key placed in
`data/settings.json` is simply ignored. The one deliberate exception is the
CLI-facing AutoFind keys (`autofind_detectors`, `autofind_exporter`,
`autofind_exporter_field_values`, `autofind_cli_delete_dataset`): for the
built-in `default` user only, a read
that misses in `user_settings.json` falls through to `data/settings.json`, which
is what lets the CLI's `--settings` flat file and single-user deployments keep
working.

#### Server tier — `data/settings.json`

```json
{
  "saved_datasets_dir": "data/saved_datasets",
  "detectors_dir": "data/detectors",
  "max_concurrent_dataset_downloads": 4,
  "max_concurrent_dataset_embeddings": 2,
  "hidden_plugins": {},
  "dataset_max_age_days": null,
  "support_email": "ops@example.org",
  "docs_links": [
    {"label": "Acme plugin guide", "url": "https://docs.example.org/acme-plugin"},
    {"label": "Lab data policy", "url": "/wiki/data-policy"}
  ],
  "semantic_only": false,
  "autopilot_only": false,
  "solo_media_type": null,
  "projection_n_neighbors": 15,
  "projection_min_dist": 0.1,
  "browse_signpost_vocab": {},
  "default_settings_source": null
}
```

- `saved_datasets_dir`, `detectors_dir`: infrastructure directories
  (overridable for custom data layouts).
- `max_concurrent_dataset_downloads` / `max_concurrent_dataset_embeddings`:
  concurrency gates for dataset loading. The download gate covers the
  bandwidth/disk-bound import phase; the embed gate covers CPU/GPU-bound
  embedding plus post-load clipping, dedup, and coverage-atlas construction.
  Changes take effect on queued and future loads (running tasks are never
  preempted). Defaults derive from hardware on first read and are **not**
  persisted to disk: downloads = `max(1, min(4, cpu_count))`; embeddings = `1`
  on an accelerator (CUDA/MPS — embed jobs share the one device and serialise
  on it, so extra workers only add VRAM pressure), and on a CPU host the
  scarcer of `cores/4` and `RAM/4 GiB` (cap 4, floor 1). See
  `default_concurrent_downloads` / `default_concurrent_embeddings` in
  `vtscore/embedding/loader.py`. These match the autodetect described under
  [Dataset-ingest concurrency](#dataset-ingest-concurrency) above. Values are
  clamped to `[1, 16]`; an explicit value here always wins over both the
  autodetect and the env vars.
- `hidden_plugins`: admin-side plugin hiding, mapping a plugin-family id
  (`"converters"`, `"embedders"`, `"importers"`, … — the keys used by
  `vtscore.plugins.inventory`) to a list of plugin names to omit from picker
  and listing responses. Merged at read time with any `--hide-plugin
  family:name` flags / `VTSEARCH_HIDE_PLUGINS` entries — the merge is a union,
  so either source can add a hide and neither can un-hide. This is a
  UI-declutter setting, **not a security boundary**: hidden plugins remain
  importable and callable by name.
- `dataset_max_age_days`: stamps new datasets with an expiry this many days
  out. `null` (the default) means datasets never expire. Also settable with
  `--dataset-max-age-days` / `VTSEARCH_DATASET_MAX_AGE_DAYS`, either of which
  wins for the process lifetime.
- `support_email`: recipient for the Help modal's "Email us" contact link.
  Shared across all users; defaults to the built-in project address. Override
  per instance by editing this key, or at startup with `--support-email` /
  `VTSEARCH_SUPPORT_EMAIL` (either applies process-wide and wins over the
  persisted value). Surfaced read-only at `GET /api/settings`; not editable via
  `PUT`.
- `docs_links`: this deployment's own documentation, for when you add plugins
  or extensions users need to read up on. An ordered list of
  `{"label": ..., "url": ...}` objects; the Help modal lists them, in this
  order, under **Docs for this server** at the bottom of every tab, and each
  opens in a new browser tab. `url` must be an absolute `http://` / `https://`
  URL or a root-relative `/path` on the VTSearch host; an entry whose URL is
  anything else, or whose `label` is blank, is dropped, and the startup log
  prints an `Ignoring docs_links entry …` line for each one (a good entry is
  listed on a `Help-modal docs: …` line). Empty (the default) shows nothing.
  Settings-file only (there is no flag or env var); surfaced read-only at
  `GET /api/settings`, not editable via `PUT`. Read at startup, so restart
  after editing it.
- `semantic_only`: locks the deployment to Semantic embedders, hiding the
  prototype Patch Semantic and Structural types from every picker and rejecting
  them at the dataset-load / detector-create routes. Also settable with
  `--semantic-only` / `VTSEARCH_SEMANTIC_ONLY`.
- `autopilot_only`: keeps **Train** and **Test** on their Autopilot tabs. Train
  loses its Manual tab and Test its Review tab, and neither view shows a tab
  bar, so users only see the guided flow. Train starts in Autopilot whatever a
  user's own `autopilot_enabled` preference says. With Test's Review tab gone,
  its Browse / To Dataset / Export actions for unverified matches are gone too.
  Autopilot needs a first sort to start from, and on a dataset that can't search by
  text that has to be an example item, which the New Detector dialog
  requires. A detector that reaches Train without one (opened on such a
  dataset with only a text description and too few labels to train) shows a
  note instead of the Autopilot panel. A UI simplification, not a permission
  boundary. Also settable with `--autopilot-only` / `VTSEARCH_AUTOPILOT_ONLY`.
- `solo_media_type`: narrows the whole instance to one media type. The importer
  and new-detector flows hide their media-type pickers and lock to it, the
  converter picker filters to converters that output it, and media-type steps
  in tabbed UIs are skipped. `null` shows everything. Users cannot change it;
  also settable with `--solo-media-type` / `VTSEARCH_SOLO_MEDIA_TYPE`, either of
  which wins for the process lifetime.
- `projection_n_neighbors`, `projection_min_dist`: UMAP knobs for the VTSBrowse
  map. The persisted projection is keyed on these, so changing one forces a
  recompute rather than serving a layout fit under the old params. Clamped to
  `[2, 200]` and `[0.0, 0.99]`.
- `browse_signpost_vocab`: per-media-type zero-shot tag vocabulary used to name
  map regions in Tags mode, replacing the built-in AudioSet-527 /
  OpenImages-600 lists — the hook for a domain-specific taxonomy (bird species,
  machine faults, product categories). Server-tier and read-only over the API,
  since every user of an instance should read the same region names. Normalized
  on write (trimmed, de-duplicated, capped at 2000 terms); a media type with an
  empty or absent list falls back to the shipped vocabulary. Takes effect on the
  next projection build / Re-project, since signpost texts are cached.
- `default_settings_source`: deployment-wide default for settings sync, same
  `{"source_name": ..., "field_values": ...}` shape as the per-user
  `settings_source`. A user with no explicit source of their own inherits this;
  a user whose `settings_source` is `{"source_name": "none"}` explicitly opts
  out. Template the `field_values` (e.g.
  `{"filepath": "data/user-settings/{username}.json"}`) to personalise it per
  user.

#### Per-user tier — `user_settings.json`

An abridged example; the full field list is `UserSettings` in
`vtsearch/settings_models.py`.

```json
{
  "volume": 1.0,
  "beta": 1.0,
  "theme": "system",
  "enrich_descriptions": false,
  "calibrate_count": 2,
  "calibration_fraction": null,
  "audio_playing": true,
  "show_animations": "show",
  "show_usage_bars": "default",
  "show_metadata": false,
  "label_hint_dismissed": false,
  "hide_all_hints": false,
  "hidden_hints": [],
  "enable_achievements": true,
  "autopilot_enabled": true,
  "hide_autopilot": false,
  "autopilot_top_greens": 3,
  "autopilot_hard_reds": 4,
  "autopilot_resort_interval": 10,
  "autopilot_goal_diversity": 40,
  "autofind_detectors": [],
  "autofind_exporter": "",
  "autofind_exporter_field_values": {},
  "autofind_on_import": true,
  "autofind_cli_delete_dataset": false,
  "focus_mode_left": {},
  "focus_mode_right": {},
  "grid_icon_size_left": {},
  "grid_icon_size_right": {},
  "panel_pct_left": {},
  "panel_pct_right": {},
  "browse_graphics": "auto",
  "browse_panel_width": 360,
  "browse_colormap": {},
  "browse_icon_size": {},
  "import_defaults_by_media_type": {},
  "recent_sessions": []
}
```

- `theme`: `"system"` (the default — follows the OS `prefers-color-scheme`),
  `"dark"`, `"light"`, or `"highviz"`.
- `autofind_detectors`: detector names to run on each web import, from a
  dataset's **Run AutoFind**, during `/api/auto-detect`, and in the CLI
  `--autodetect` flow, each mapping to a JSON file under `data/detectors/`.
  Every user curates their own list on the Dashboard's AutoFind detector tab
  (`PUT /api/detectors/registry/<id>/autofind`). `autofind_exporter` names the
  results exporter run afterwards (`""` = no auto-export; the CLI then falls
  back to the `gui` exporter), and `autofind_exporter_field_values` keeps each
  exporter's configuration around when the picker switches between them.
  `autofind_cli_delete_dataset` (default `false`) makes a CLI `--autodetect`
  run delete the dataset it imported once its detectors have run (see
  [`docs/CLI.md`](CLI.md#saving-the-dataset-to-the-dashboard---tempimport)).
  These four read through to `data/settings.json` for the `default` user.
  `autofind_on_import` (default `true`) is whether a web import runs them: the
  Add Dataset dialog's **Run AutoFind** checkbox starts from it and each import
  writes it back.
- `grid_icon_size_*`, `focus_mode_*`, `panel_pct_*`, and the `browse_*` maps:
  per-media-type UI preferences, keyed by media-type id, so a user can tune
  audio and image datasets independently. Empty entries fall back to the
  frontend's per-type default.
- `import_defaults_by_media_type`: the embedder, clipper (+ params), and
  converter rows to auto-fill into the Add Dataset advanced panel for each
  output media type. Set from Settings → Data Imports.
- `settings_source` (not shown; excluded from the defaults endpoint): opt-in
  bidirectional sync. Set it to a plugin name + field values to auto-export
  every settings change and auto-import at startup. See `settings_io/sources/`.

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
genuinely slowing job still reports the increase; what it no longer does is
twitch.

Earlier versions let an operator hide the import estimate
(`--hide-ingest-eta`, `VTSEARCH_HIDE_INGEST_ETA`, `"hide_ingest_eta"`) and tune
the per-step weights to their own hardware (`VTSEARCH_TIMING_PROFILE`,
`VTSEARCH_TIMING_RECORD`, `scripts/profiling/tune_timing_profile.py`). All of
those were removed (#4667): the flag is no longer accepted, and the variables
and the settings key are ignored.

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
| `docker/Dockerfile.labbench` | `docker-compose.labbench.yml` | `python:3.12-slim` | SigLIP-only image search from `requirements/labbench.txt`; SigLIP weights baked in at build time under `/opt/vtsearch/models` (`VTSEARCH_MODELS_DIR`), so they survive a volume mounted on `/app/data`. |
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
(`sm_70`) kernels: on a V100 host use `scripts/install.sh cu124` instead (see
[SETUP.md § Picking the CUDA tag](SETUP.md#picking-the-cuda-tag)).

### Resource considerations

- **Memory**: Each embedding model uses ~500 MB–1.5 GB of RAM when loaded.
  With all five loaded simultaneously, expect ~4–6 GB total application
  memory. Models are loaded lazily; only the media types actually used are
  loaded.
- **Disk**: The `data/models/` directory uses ~4.1 GB. Dataset embeddings
  and media files vary by dataset size.
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
requirements/labbench.txt            ← LabBench (SigLIP-only) image deps (standalone)
requirements/image-embedders.txt     ← All-image-embedders image deps (standalone; shared by the CPU
                                       and GPU Dockerfiles, which each pass their own --extra-index-url)
```

### Installing without the AGPL dependencies

Two runtime dependencies are AGPL-3.0-or-later: **`ultralytics`** (YOLO)
and **`PyMuPDF`**. Every default install path includes them — the install
scripts, both `requirements/{base,gpu}.txt`, and both Docker images — so
the features that need them work out of the box. That is a deliberate
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

# Docker (same for Dockerfile.gpu with gpu-no-agpl.txt)
docker build --build-arg REQUIREMENTS=base-no-agpl.txt -f docker/Dockerfile -t vtsearch .
```

What you give up: the YOLO image extractor and the YOLO image clipper,
PDF import, and the document → image / document → text converters. Those
fail with a message naming the missing package and how to get it
(`vtscore/utils/optional_deps.py`) rather than erroring opaquely.
Everything else — every embedder, every other media type, training,
scoring, Browse — is unaffected. See [`NOTICE`](../NOTICE) for the full
licensing statement.

### Key dependencies

| Package | Version constraint | Purpose |
|---------|-------------------|---------|
| `torch` | `>=2.0.0` | Neural network training and inference |
| `transformers` | latest | HuggingFace model loading (CLAP, CLIP, X-CLIP) |
| `sentence-transformers` | latest | E5 text embeddings |
| `numpy` | latest | Numeric arrays |
| `flask` | latest | Web server |
| `opencv-python-headless` | latest | Image processing (structural embedder, YOLO). **Not** used for video decoding — see [FIPS](#video-import-crashes-with-fatal-fips-selftest-failure) |
| `ultralytics` | latest | YOLO-based image processing. AGPL-3.0; in the `agpl` extra, installed by default — see [Installing without the AGPL dependencies](#installing-without-the-agpl-dependencies) |
| `librosa` | latest | Audio analysis (spectrograms, silence splitting) |
| `soundfile` / `soxr` | latest | Audio decoding and resampling (`vtscore.media.audio.decode`) |
| `imageio-ffmpeg` | latest | Bundled ffmpeg binary; audio codecs libsndfile can't read (AAC/M4A/MP4), and all video frame decoding (`vtscore.media.video.decode`) |
| `PyMuPDF` | latest | Document (PDF/DOC/PPT) rendering and text extraction. AGPL-3.0; in the `agpl` extra, installed by default — see [Installing without the AGPL dependencies](#installing-without-the-agpl-dependencies) |

---

## Troubleshooting

### The app freezes for seconds during labeling (diagnosing a stall)

A rare multi-second pause in which *every* in-flight request finishes at the
same instant (issue #3853) is not one slow endpoint. It is either a thread
holding the GIL, a lock convoy, or the process not being scheduled at all,
and the per-request timer cannot tell those apart. The app carries three
instruments that can, all on at the default log level:

- **The stall watchdog** (`VTSEARCH_STALL_WATCHDOG_MS`). A heartbeat thread
  measures how late it wakes. When it misses by more than the threshold it
  logs `stall: heartbeat late by …ms` with the process's CPU time over the
  gap, the threads that consumed it, major page faults, RSS, the cgroup memory
  counters and GC activity. Read it like this:
  - *process cpu ≈ wall, one thread on top* → that thread held the GIL.
    Just above the `stall:` line, the watchdog writes every thread's Python
    stack (`Stall snapshot …`, to `VTSEARCH_STALL_DUMP_FILE`, which defaults
    to the log file), with the thread that burned the most CPU first. Read
    that thread's top frames. They are taken the moment the heartbeat wakes,
    and a thread holding the GIL in C code can only give it up at the call
    that stalled, so that is where it still is. `VTSEARCH_STALL_LIVE_DUMP=1`
    adds `faulthandler`'s dump from *during* the stall, which can crash the
    app (#4345).
  - *no thread consumed cpu* → the process was not running: look at `majflt`
    and the cgroup `limit hits` (memory pressure), or at the node.
  - *cpu spread across threads* → contention rather than one holder; the lock
    and phase lines below say where.
- **GC pauses** (`VTSEARCH_GC_WARN_MS`): `gc pause: generation 2 took …ms`. A
  full collection holds the GIL and shows in a thread dump only as an
  arbitrary allocation site, so it is named separately. Left unset this bar
  **tracks `VTSEARCH_SLOW_PHASE_MS`**, so lowering the phase bar for a
  diagnostic session cannot leave collections below it invisible — they would
  otherwise still inflate every phase they land in, with nothing saying why.
- **Phase and lock timers** (`VTSEARCH_SLOW_PHASE_MS`): `slow phase: <name>
  total …ms cpu=…ms gc=…ms (phase=…ms, …)` for the learned-sort retrain, the
  per-vote labelset rewrite, the labeling-status replay and a vote rehydrate;
  and `lock wait: <lock> waited …ms` when a vote, the sort thread or the status
  poll queued behind another holder.

**Read the `cpu=` column first.** Every request and phase line carries the
thread's own CPU time beside its wall time, and the two together say what
kind of slow it was:

| | reading |
|---|---|
| `cpu ≈ wall` | it did the work — a real cost that grows with the data |
| `cpu ≪ wall`, `gc` small | it blocked (a lock, an `fsync`, a slow filesystem) or was descheduled — an 8-thread torch process in an 8-CPU cgroup on a shared node is descheduled routinely |
| `gc ≈ wall` | a collection froze every thread; the phase is innocent |

`gc` is not additive with `cpu` — a collection running on that thread burned
its CPU too, so `total 1137ms cpu=1133ms gc=139ms` reads "on the CPU
throughout, 139 ms of it collecting".

To capture one on the GRID: launch with **`VTSEARCH_DIAGNOSE=1`** (one switch
for every bar — see the env table; it reaches the app through the launcher's
environment, so no launcher change is needed), label until a stall is felt,
then read the log around the `stall:` line.
`scripts/experiments/stall_3853/analyze_app_log.py` prints that window for
every stall in a log.

**Every run records the bars it is using**, as a `diagnostics config:` line at
startup, at WARNING so a stock deployment has it too. Without it a log cannot
be read honestly after the fact: "no slow requests" means *nothing was slow*
and *the bar was a second* equally well, and one whole session in #3853 was
mis-read that way. The analyzer prints that line first, and says so when a log
predates it.

For what the app cannot see — NFS latency on the mount every vote writes to,
and major faults on the app process — run
`scripts/experiments/stall_3853/sample_host.py` alongside the session; it
writes JSONL that lines up with the app log by timestamp, and summarises
itself with `--summarize`.

**A felt pause is often a sum, not an outlier.** The residue of #3853 is a
vote cycle whose chain of requests, retrain and collection each cost less than
any threshold while adding up to the half-second the reviewer notices — which
a per-request bar cannot see by construction. At `VTSEARCH_LOG_LEVEL=INFO`
every request is logged as `request trace: …` with the same wall/CPU/GC
figures, and `analyze_app_log.py` adds them up into a **per-vote budget**:
each cycle's span from keypress to keypress, how much of it the server had a
request in flight (`busy`), and how much it did not (`gap` — client work,
browser queueing and think time). A 2 s cycle with 200 ms of `busy` was not
the server.

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

### A feature you just pulled isn't in the UI ("out-of-date build")

**Symptom**: a toast reading *"This page is running an out-of-date build"*, or a
`⚠ bundle v …` chip beside the version in the Settings footer. Alternatively no
warning at all, but a control that the docs describe is missing, does nothing,
or behaves like an older release.

**Cause**: `static/` is a build artifact and is not tracked by git, so pulling
new code and restarting the server upgrades the *Python* side only. The browser
keeps loading whichever bundle was last built. The version shown in Settings is
the server's, so it looks current while the JavaScript is not.

**Fix**: rebuild the frontend and hard-reload the page (Ctrl/Cmd-Shift-R — a
normal reload can serve the old bundle from cache):
```bash
cd frontend && npm install && npm run build:prod
```

### "No module named" errors

**Symptom**: `ModuleNotFoundError` for a media type or importer package.

**Fix**: Reinstall to ensure all dependencies are present:
```bash
bash scripts/install.sh
```

### Browse maps render unlettered ("VTSBrowse signposts are disabled")

**Symptom**: the Browse canvas draws the map but never names any region, and
the server log carries a one-time

```
VTSBrowse signposts are disabled: the required 'toponymy' library is not importable.
```

**Cause**: `toponymy` is installed in a dedicated `--no-deps` step rather than
declared as an ordinary dependency (its `transformers<5.0.0` pin would
downgrade the transformers stack every embedder sits on). Any environment that
skipped that step has no labeler.

**Fix**: on a host install, re-run the installer, which performs the step (and
installs `apricot-select` alongside it):

```bash
bash scripts/install.sh
```

In a container, rebuild the image from a current checkout: every Dockerfile
under `docker/` carries the step as of issue #3852. Images built before that
fix are missing it, and `pip install toponymy` inside a running container is a
workaround rather than a repair -- pass `--no-deps` if you try it, or pip will
downgrade transformers and break the embedders.

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

### GPU install's cuML step (skipped on every tag but `cu129`)

**What you see**: on a `cu118`, `cu121`, `cu124` or `cu128` install the step
prints `(skipped: cuML needs RAPIDS >= 26.8 …)` and the smoke test at the end
reports `cuML not importable -> GPU UMAP/k-means will use the CPU fallback`.

**Cause**: pandas is held at `>=3` everywhere (`pyproject.toml`), and cudf,
which cuML depends on, is the first to take pandas 3 at RAPIDS **26.8**; an
older RAPIDS would downgrade pandas to 2.x mid-install. RAPIDS 26.x in turn
requires `nvidia-nvjitlink-cu12 >= 12.9`, which only the `cu129` torch wheel
pins (`cu124` pins `==12.4.x`, `cu128` `==12.8.x`), so on those tags the two
cannot share a venv and the installer does not try.

**What to do**: if the GPU is Turing or newer (`sm_75+`; every card since the
RTX 20 series and the T4) and `nvidia-smi` reports a driver CUDA version of
12.9 or later, re-run as `bash scripts/install.sh cu129`; the auto-detect
already picks that tag on such hosts, so an explicit older tag is usually why
you got here. A Volta card (V100) has no `cu129` torch, and a driver below CUDA
12.9 can't run one: those hosts run UMAP / k-means on the CPU. RAPIDS 25 can
still be assembled on them by hand (`VTS_CUML_CU12_SPEC='cuml-cu12<26'`), at
the cost of pandas dropping to 2.x in that venv.

**On `cu129`** the step runs, and torch and RAPIDS share one set of
`nvidia-*-cu12` wheels (torch pins them at 12.9.x, RAPIDS 26.8 is built on
12.9), so pip should install it without a "dependency conflicts" report. If
one appears, the venv holds a torch from another tag (one built before the
`cu129` move, or a `VTS_CUML_CU12_SPEC` override); the installer captures the
report to a log and the **GPU smoke test** at the end (a tiny torch CUDA matmul
plus a cuML import) is what decides whether the stack works. Re-run with
`VTSEARCH_VERBOSE=1` to stream the raw output.

### cuML crashes compiling a kernel (`cuda_fp8.hpp` / nvrtc errors)

**Symptom**: A VTSBrowse projection or coverage-atlas build dies with an
nvrtc compile error like:

```
.../nvidia/cu13/include/cuda_fp8.hpp(...): error: this declaration has no
storage class or type specifier  __NV_SILENCE_DEPRECATION_BEGIN
... N errors detected in the compilation of ".../<hash>.cubin.cu".
```

**Cause**: a **RAPIDS built on a newer CUDA than the libraries your torch
pinned**. cuML compiles its cuVS/raft kernels with nvrtc lazily, on the first
UMAP/k-means `fit`. RAPIDS **26.x** requires `nvidia-nvjitlink-cu12 >= 12.9`
and is built on CUDA 12.9; a torch from an older tag pins the other CUDA
libraries lower (`cu124` = 12.4.x, `cu128` = 12.8.x), so pip reports the
conflict:

```
cuml-cu12 26.8.0 requires nvidia-nvjitlink-cu12<13,>=12.9, but you have
nvidia-nvjitlink-cu12 12.4.127 which is incompatible.
```

...and, if mismatched libraries land anyway, makes cupy's nvrtc compile the
wrong-version fp8/fp6/fp4 headers (the `__NV_SILENCE_DEPRECATION_BEGIN` macro
the resident nvrtc doesn't define), producing the `cuda_fp8.hpp` crash above.

`scripts/install.sh` and `docker/Dockerfile.gpu` install RAPIDS **only
alongside the `cu129` torch**, whose pins are the 12.9.x libraries RAPIDS is
built on, so fresh installs are unaffected. A venv built before that move
(torch `cu124` plus `cuml-cu12<26` and pandas 2) is consistent with itself and
keeps working; the crash comes from mixing the two eras in one venv, e.g. by
upgrading cuML there by hand.

VTSearch also **degrades to the CPU UMAP/k-means path** whenever a cuML fit
fails for any reason (logging a one-time warning) instead of crashing, so the
run still completes — just slower, without GPU acceleration. To restore the
GPU path, rebuild the venv on one CUDA minor:

```bash
# 1. Inspect the installed CUDA stack (look for a torch tag other than cu129
#    next to cuml-cu12 26.x, and any stray *-cu13 wheels from an out-of-band install).
pip list | grep -iE '^torch |cu13|cupy|cuml|cuvs|libraft|pylibraft|nvidia-nvjitlink'

# 2. If step 1 showed stray CUDA-13 wheels, remove them first:
pip uninstall -y $(pip list --format=freeze | grep -iE '(-cu13|cupy-cuda13x)' | cut -d= -f1)

# 3. Reinstall on the cu129 tag (Turing or newer, driver CUDA >= 12.9), which
#    pins torch and RAPIDS to the same 12.9.x libraries:
bash scripts/install.sh cu129
```

If you don't need GPU UMAP/k-means at all, set `VTSEARCH_SKIP_CUML=1`
(skips the cuML install) and the CPU `umap-learn`/`scikit-learn` paths run
without any toolchain risk.
