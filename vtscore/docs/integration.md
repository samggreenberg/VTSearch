# Embedding `vtscore` in Your Application

Most readers reach for `vtscore` because they want to add VTSearch's
detector workflow to an existing application — a Django app, a FastAPI
backend, a Jupyter notebook pipeline, a CLI tool. This guide walks
through what that looks like, what hooks the library expects you to
install, and what the resulting code shape looks like.

The companion `vtsearch` Flask app is one such integration; this guide
describes the general pattern in less code than `vtsearch` does.

## Contents

1. [What you need to install](#what-you-need-to-install)
2. [The three integration hooks](#the-three-integration-hooks)
3. [Minimal integration: a script](#minimal-integration-a-script)
4. [Multi-request integration: a web app](#multi-request-integration-a-web-app)
5. [Multi-thread integration: a worker pool](#multi-thread-integration-a-worker-pool)
6. [Persistent storage](#persistent-storage)
7. [Authentication and per-user data](#authentication-and-per-user-data)
8. [Things you don't need to install](#things-you-dont-need-to-install)

---

## What you need to install

`vtscore` ships with reasonable defaults — out of the box, you can do
this from any Python process:

```python
from vtscore.media import audio
from vtscore.datasets.loader import load_dataset_from_folder
from vtscore.training import train_model

medias = {}
load_dataset_from_folder("/path", media_type="audio", medias=medias)
# train_model(...) works on raw arrays — no setup needed
```

The library only requires explicit setup when you want to:

- Read **configuration** (`CoreConfig.from_settings()`)
- Resolve the **active context** from a request
- **Persist** detector settings (`set_inclusion`, `set_safe_thresholds`,
  etc.) back to your settings store
- Register **app-side plugins** alongside the library's built-ins

Each of these has a hook. Install only the ones you need.

## The three integration hooks

### Hook 1: `register_core_config_builder`

By default, `CoreConfig.from_settings()` raises `RuntimeError` — the
library refuses to silently fall back to a guess. Install a builder that
constructs a `CoreConfig` from your config source:

```python
from pathlib import Path
from vtscore.config import CoreConfig, register_core_config_builder

def _build_core_config() -> CoreConfig:
    # Your config source — env vars, settings JSON, database row, …
    return CoreConfig(
        data_dir=Path("/var/lib/myapp/data"),
        saved_datasets_dir=Path("/var/lib/myapp/data/datasets"),
        detectors_dir=Path("/var/lib/myapp/data/detectors"),
        inclusion=0,
        safe_thresholds=False,
        calibrate_count=2,
        calibration_fraction=0.5,
        enrich_descriptions=False,
        autopilot_goal_diversity=0.5,
        max_concurrent_dataset_downloads=2,
        max_concurrent_dataset_embeddings=1,
    )

register_core_config_builder(_build_core_config)
```

Call this **once at startup**, before any library code that calls
`CoreConfig.from_settings()`. The builder is process-wide.

### Hook 2: `register_*_context_resolver`

If your application is request-oriented (web app, RPC server), install
resolvers that map "the current request" → the relevant
`DatasetContext` / `DetectorContext`:

```python
from vtscore.state import (
    register_dataset_context_resolver,
    register_detector_context_resolver,
    get_context,
    get_detector_context,
)

# Example: read dataset/detector IDs from a contextvars.ContextVar
from contextvars import ContextVar

current_dataset_id: ContextVar[str | None] = ContextVar("current_dataset_id", default=None)
current_detector_id: ContextVar[str | None] = ContextVar("current_detector_id", default=None)

def _resolve_dataset_context():
    did = current_dataset_id.get()
    return get_context(did) if did else None

def _resolve_detector_context():
    did = current_detector_id.get()
    return get_detector_context(did) if did else None

register_dataset_context_resolver(_resolve_dataset_context)
register_detector_context_resolver(_resolve_detector_context)
```

Then your request middleware sets the `ContextVar`s on each incoming
request, and library calls in handlers see the right context
automatically.

If your application **isn't** request-oriented — say it's a worker that
processes one dataset at a time — skip this hook entirely and use
`set_thread_dataset_context()` / `set_thread_detector_context()` (or
the `override_*_context()` context managers) directly.

### Hook 3: `register_setting_persister`

The `vtscore.state` package exposes setter functions like
`set_inclusion(value)` and `set_safe_thresholds(value)`. By default,
those update only the in-memory cache. If you want them to persist to
your settings store, install a persister per key:

```python
from vtscore.state import register_setting_persister

def _persist_inclusion(value: int) -> None:
    my_settings_store["inclusion"] = value

def _persist_safe_thresholds(value: bool) -> None:
    my_settings_store["safe_thresholds"] = value

register_setting_persister("inclusion", _persist_inclusion)
register_setting_persister("safe_thresholds", _persist_safe_thresholds)
```

If you don't install persisters, library code can still call
`set_inclusion(5)` — the value just won't survive a process restart.
That's a fine choice for many apps.

## Minimal integration: a script

The shortest possible integration runs in a single thread with no web
framework:

```python
from pathlib import Path
import numpy as np, torch

from vtscore.config import CoreConfig, register_core_config_builder
from vtscore.media import audio  # noqa: F401 — registers MediaType + embedders
from vtscore.datasets.loader import load_dataset_from_folder
from vtscore.training import train_model, calculate_cross_calibration_threshold


# Hook 1: where does data live?
register_core_config_builder(lambda: CoreConfig(
    data_dir=Path("/tmp/myapp"),
    saved_datasets_dir=Path("/tmp/myapp/datasets"),
    detectors_dir=Path("/tmp/myapp/detectors"),
    inclusion=0, safe_thresholds=False,
    calibrate_count=2, calibration_fraction=0.5,
    enrich_descriptions=False, autopilot_goal_diversity=0.5,
    max_concurrent_dataset_downloads=1,
    max_concurrent_dataset_embeddings=1,
))

# That's it. Now use the library.
medias: dict[int, dict] = {}
load_dataset_from_folder(Path("/data/audio"), media_type="audio", medias=medias)
# train, score, export, …
```

No context resolver, no persister — for a one-shot script there's
nothing to persist between processes anyway.

## Multi-request integration: a web app

For a request-oriented app, install all three hooks. Pseudocode using
FastAPI:

```python
from contextvars import ContextVar
from fastapi import FastAPI, Header

from vtscore.config import CoreConfig, register_core_config_builder
from vtscore.state import (
    DatasetContext, DetectorContext,
    register_context, register_detector_context, get_context, get_detector_context,
    register_dataset_context_resolver, register_detector_context_resolver,
    register_setting_persister,
)


# Request-scoped state.
current_dataset_id: ContextVar[str | None] = ContextVar("current_dataset_id", default=None)
current_detector_id: ContextVar[str | None] = ContextVar("current_detector_id", default=None)


# Hook 1: settings → CoreConfig
register_core_config_builder(lambda: CoreConfig(
    data_dir=settings.data_dir,
    saved_datasets_dir=settings.saved_datasets_dir,
    detectors_dir=settings.detectors_dir,
    inclusion=settings.inclusion,
    safe_thresholds=settings.safe_thresholds,
    calibrate_count=settings.calibrate_count,
    calibration_fraction=settings.calibration_fraction,
    enrich_descriptions=settings.enrich_descriptions,
    autopilot_goal_diversity=settings.autopilot_goal_diversity,
    max_concurrent_dataset_downloads=settings.max_concurrent_dataset_downloads,
    max_concurrent_dataset_embeddings=settings.max_concurrent_dataset_embeddings,
))


# Hook 2: context resolvers
register_dataset_context_resolver(
    lambda: get_context(current_dataset_id.get()) if current_dataset_id.get() else None
)
register_detector_context_resolver(
    lambda: get_detector_context(current_detector_id.get()) if current_detector_id.get() else None
)


# Hook 3: per-key persisters
register_setting_persister("inclusion", lambda v: settings.update("inclusion", v))
register_setting_persister("safe_thresholds", lambda v: settings.update("safe_thresholds", v))


app = FastAPI()


@app.middleware("http")
async def attach_context(request, call_next):
    dataset_token = current_dataset_id.set(request.headers.get("X-Dataset-Id"))
    detector_token = current_detector_id.set(request.headers.get("X-Detector-Id"))
    try:
        return await call_next(request)
    finally:
        current_dataset_id.reset(dataset_token)
        current_detector_id.reset(detector_token)


@app.get("/medias")
def list_medias():
    from vtscore.state import snapshot_medias
    return snapshot_medias()        # operates on the resolved DatasetContext
```

The pattern is the same for Django, Flask, Starlette, aiohttp, etc. —
only the middleware mechanics differ.

## Multi-thread integration: a worker pool

If you're running multiple training / scoring jobs in parallel,
each thread needs to declare its context once and the library handles
the rest:

```python
import concurrent.futures
from vtscore.state import (
    DatasetContext, DetectorContext,
    set_thread_dataset_context, set_thread_detector_context,
)
from vtscore.detectors.training import train_detector_from_origins


def score_one(dataset_ctx: DatasetContext, detector_ctx: DetectorContext) -> dict:
    set_thread_dataset_context(dataset_ctx)
    set_thread_detector_context(detector_ctx)
    # Re-derive the detector's MLP from its saved origins. Pass
    # detector_ctx.embedder so the re-embedded vectors match the embedder
    # the detector was originally trained with — never the media type's
    # default, which may have changed since save.
    good_origins, bad_origins = origins_from_labelset(detector_ctx.labelset)
    train_detector_from_origins(
        good_origins, bad_origins,
        inclusion=0,
        media_type=detector_ctx.media_type,
        embedder_name=detector_ctx.embedder,
    )
    # … score the dataset against the detector, return hits
    return {"detector_id": detector_ctx.detector_id, "n_hits": ...}


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(score_one, ds, dt) for ds, dt in jobs]
    results = [f.result() for f in futures]
```

Notes on the multi-thread story:

- `train_model` is thread-safe — it uses `torch.random.fork_rng()` to
  isolate its RNG and a local `torch.Generator` for the model weights.
- The state lock (`_state_lock` in `vtscore/state/core.py`) is an
  `RLock`, so one public function can call another while holding the
  lock without deadlock.
- Per-thread progress callbacks are available via
  `vtscore.concurrency.progress.set_thread_progress(cb)` — set one in
  each worker and the long-running ops report to that callback only,
  not a shared global.

What the library does **not** do is supply a job manager. If you want a
single-slot pending-job system, look at `vtscore.concurrency.JobManager`
(used by the app's learned-sort and eval routes). If you want
something fancier — a Celery integration, a Kubernetes operator —
that's your code.

## Persistent storage

`vtscore` writes to disk in three places. All three are
`CoreConfig`-driven; nothing is hardcoded.

| What | Where | Format |
|------|-------|--------|
| Saved dataset pickles | `CoreConfig.saved_datasets_dir` | `pickle.dump((medias, embeddings))` — the **only** sanctioned vector store |
| Detector labelsets | `CoreConfig.detectors_dir / "<name>.json"` | JSON; origins + labels only, never weights |
| Embedder model cache | `CoreConfig.data_dir / "models"` | HuggingFace / torch cache layout |

If you want a different layout — say, store detectors in your database
instead of on disk — you have two options:

1. **Implement a `LabelsetSource` plugin.** Sources sync detector
   labels bidirectionally with an external store (see
   [extending/labelset-sources.md](extending/labelset-sources.md)).
   The on-disk JSON still exists as a cache, but your store is the
   source of truth.
2. **Replace `vtscore.detectors.store` calls.** The store module is
   small (a few `json.load` / `json.dump` calls); replacing it with
   your DB equivalent is straightforward. This is more invasive but
   gives full control.

For datasets, the analogous plugin is `MediaSource` — implement one
that resolves an `Origin` back to a file pulled from your storage layer
(S3, GCS, HDFS, …). See [extending/media-types.md](extending/media-types.md).

## Authentication and per-user data

`vtscore` is **single-tenant by design.** It does not know about users,
ACLs, or per-user data directories. If your app is multi-user, the
`vtsearch` companion app's `vtsearch.auth` package is one reference
implementation:

- A `LoginProvider` ABC plus a `DefaultLoginProvider` (single-user,
  no-op).
- A `get_current_user()` accessor that reads the resolved user from
  `flask.g`.
- A `get_user_data_dir(user)` helper that returns
  `data/<username>/...`.

To port this idea to your own app:

- Replace `flask.g` with your request-local store (a `ContextVar`,
  request-scoped dependency injection, etc.).
- Build a per-user `CoreConfig` in your `register_core_config_builder`
  callback — point `data_dir` / `saved_datasets_dir` / `detectors_dir`
  at the per-user paths.
- Build a per-user `DatasetContext` / `DetectorContext` in your
  resolver hooks.

The library doesn't care that the contexts are per-user — they're just
keyed by ID strings.

## Things you don't need to install

The library has no required hooks. If you're writing a one-shot script,
you can ignore all of the above and just import what you need:

- `from vtscore.training import train_model` — works directly on numpy
  arrays, no setup required.
- `from vtscore.embedding.helpers import embed_text_query, embed_image_file` —
  works as long as the relevant `MediaType` has been imported (which
  registers the default embedder).
- `from vtscore.datasets.loader import load_dataset_from_folder` —
  works if you don't call `CoreConfig.from_settings()` anywhere.

The library is designed to scale from "one-line numpy operation" to "a
full Flask + Angular app" without any code path being mandatory in
between.
