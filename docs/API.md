# HTTP API Reference

VTSearch exposes a REST-style JSON API. All endpoints accept and return JSON
unless otherwise noted. File uploads use `multipart/form-data`.

## Sub-pages

| Section | Description |
|---------|-------------|
| [Authentication & UI](api/auth.md) | Auth status, login/logout, static assets |
| [Medias & Sorting](api/medias.md) | Media listing/streaming, text/learned/example sort, votes & labels, pile upload |
| [Labeling & Diversity](api/labeling.md) | Inclusion, thresholds, labeling progress, coverage atlas |
| [Detectors](api/detectors.md) | Detector CRUD, detector registry, Auto-Find toggle, loading SSE |
| [Datasets](api/datasets.md) | Loading, importers, demos, staging, registry, media types, embedders, clippers, converters, file browsing |
| [Import & Export](api/io.md) | Result exporters, label importers, pregen processors, autorun extractors/localizers |
| [Settings](api/settings.md) | App settings, autorun processors, settings sources, labelset sources |
| [Dashboard](api/dashboard.md) | Dashboard dataset info/rename, disk/RAM usage probes |
| [Find, Auto-Detect & Scoring](api/find.md) | Multi-dataset find (+ cancel/check-labels), Find Label, Auto-Detect, find stats/corrections, find progress |
| [File Browser](api/file-browser.md) | Server filesystem browsing |

### Not yet documented in depth

The OpenAPI snapshot (`frontend/openapi.json`) is the ground truth and covers
several endpoint families that the hand-written pages above don't yet describe
in full. Use Swagger (`GET /api/docs`) for these:

- **Achievements** — gamification unlock/state endpoints (see `vtsearch/achievements.py`).
- **Projection / VTSBrowse** — `GET/POST /api/projection/*` (UMAP projection + hex-tile pyramid).
- **Sessions** — session lifecycle endpoints.
- **Jobs** — async job listing/status (backed by `vtscore.concurrency.async_jobs`).
- **Health probes** — `GET /healthz`, `GET /readyz` (liveness / readiness).
- **Version** — `GET /api/version` (returns the running build's version string).

## Context headers (`X-Dataset-Id` / `X-Detector-Id`)

VTSearch holds **multiple loaded datasets and detectors at once**, one context
each. Most endpoints operate on "the active context", and the active context is
chosen **per request** by two HTTP headers:

| Header | Selects | Sent by |
|--------|---------|---------|
| `X-Dataset-Id` | Which loaded `DatasetContext` the request's `medias` / coverage / dataset-scoped votes resolve to | Angular's `HttpClient` interceptor on every API call |
| `X-Detector-Id` | Which loaded `DetectorContext` the request's `good_votes` / `bad_votes` / model / labelset resolve to | Same interceptor |

Key semantics (`app.py` `before_request`, `vtsearch/routes/_shared.py`):

- **Per-request, not global.** The headers stash the chosen context on
  `flask.g` for the lifetime of the request; they do **not** mutate any global
  "currently active" state, so concurrent requests for different datasets don't
  interfere.
- **Query-param fallback.** Browser-native requests that bypass the Angular
  interceptor (`<img src>`, `<audio src>`, `<video src>`) may pass
  `?dataset_id=` / `?detector_id=` instead; the header takes priority when both
  are present.
- **Required on context-mutating endpoints.** Endpoints that mutate dataset or
  detector state (votes, label imports, media insertion, learned sort, …) are
  guarded by `require_dataset_header` / `require_detector_header` and return
  **400** with a message like `X-Dataset-Id header (or ?dataset_id= query param)
  is required for this endpoint` when the id can't be determined. Pure reads,
  registry listings, auth, and file-browser routes don't require them.
- **Unloaded id → 409.** If a header names an id that isn't currently loaded,
  the request proceeds until it touches a context proxy, then fails **409**
  (`DatasetNotLoadedError` / `DetectorNotLoadedError`) rather than silently
  falling back to stale data. Routes that never touch the proxies still respond
  normally.

Per-endpoint pages note where these headers are required; when in doubt, send
both for any dataset- or detector-scoped call.

## Conventions

| Pattern | Meaning |
|---------|---------|
| `{param}` | URL path parameter |
| **Body** | JSON request body (Content-Type `application/json`) |
| **Form** | `multipart/form-data` |
| `→` | Response body |
| `X-Dataset-Id` / `X-Detector-Id` | [Context headers](#context-headers-x-dataset-id--x-detector-id) selecting the active dataset / detector |
| Async endpoints | Return immediately; subscribe to the `dataset` channel on `GET /api/events` (SSE) for progress |

**Common error shape:**

```json
{"error": "Human-readable message"}
```

Status codes follow standard HTTP semantics: 200 OK, 201 Created, 204 No
Content, 400 Bad Request, 404 Not Found, 409 Conflict, 500 Internal Server
Error.

## Machine-readable schema

`GET /api/openapi.json` returns an OpenAPI 3.0 document describing every
route on the running app, with real request/response schemas declared
via `flask-smorest` decorators. A browsable Swagger UI is served at
`GET /api/docs`. Use it to:

- Browse / try endpoints live via Swagger UI.
- Generate a TypeScript / Python client.
- Diff against the snapshot at `frontend/openapi.json` to catch unintended API surface changes. `./run-tests.sh` regenerates the spec and fails on drift.

---

*Readme Reader code phrase:* `json all the way down`

