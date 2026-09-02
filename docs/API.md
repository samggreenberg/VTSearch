<!-- This file is served raw at GET /api/achievements/docs/api/raw and its
     footer phrase is hash-matched in vtsearch/achievements.py. Don't remove
     or reword the "Readme Reader code phrase" line without updating
     achievements.py to match. See CLAUDE.md. -->

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
| [Dashboard](api/dashboard.md) | Dashboard disk/RAM usage probes |
| [Find, Auto-Detect & Scoring](api/find.md) | Multi-dataset find (+ cancel/check-labels), Find Label, Auto-Detect, find stats/corrections, find progress |
| [File Browser](api/file-browser.md) | Server filesystem browsing |
| [Progress events (SSE)](api/events.md) | `GET /api/events`: the single Server-Sent Events stream carrying progress for every long-running operation |

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

**Common error shape.** Every JSON error the API returns — from a route, from
a global handler, or from schema validation — carries the same envelope
(`vtsearch/errors.py`, documented in the OpenAPI spec as the `Error` schema):

```json
{
  "code": 404,
  "status": "Not Found",
  "message": "Human-readable message",
  "request_id": "ab12cd34ef56"
}
```

`code` is the numeric HTTP status and `status` its name. `request_id` is
present on every error raised inside a request (it also comes back in the
`X-Request-Id` header), so a user can quote it in a bug report and an
operator can grep the structured logs for it.

Three optional fields appear when they apply, plus any endpoint-specific
extras (e.g. `available`, `missing_fields`, `dataset_id`):

| Field | When |
|-------|------|
| `errors` | Schema validation failed; maps location → field → messages (see the 422 example in [file-browser.md](api/file-browser.md)) |
| `detail` | 500s; the exception type and its first line, e.g. `"RuntimeError: embedder X not loaded"` |
| `error_code` | A machine-readable slug where the client branches on the *kind* of failure: `auth_required`, `dataset_not_loaded`, `detector_not_loaded` |

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

### Routes absent from the spec

Most routes are `flask_smorest`-decorated and appear in `/api/openapi.json`.
A route is left undecorated — plain Flask, still registered on the same
`Blueprint` and reachable normally, just missing from the spec — for one of
these reasons:

- **Plugin-field bodies.** Import/export/labelset-seed endpoints whose body
  shape depends on which plugin is named in the URL (e.g.
  `POST /api/detectors/registry/from-labelset/<importer>`,
  `POST /api/detectors/<name>/import-labels/<importer_name>`, the settings
  importer/exporter run routes) take fields the plugin declares dynamically
  (`creation_questions` / `fields`), which doesn't fit a single static
  marshmallow schema. Runtime validation still goes through
  `validate_plugin_args` (per-plugin schema built from the importer's
  `fields`), so a missing required field or an invalid `select` value still
  raises 422 with the standard `errors` envelope — the enforcement is real,
  it's just not statically declared for Swagger/codegen.
- **Binary-streaming routes.** Audio/video/image/media/thumbnail/preview
  routes serve raw bytes (or a tiny content-only JSON for text media), not a
  JSON response body; the sibling JSON-shaped routes in the same module are
  migrated normally.
- **Dual-mode dispatchers.** `POST /api/embed` accepts either
  `multipart/form-data` (file upload) or `application/json` (text), decided
  at request time — no single schema describes both.
- **Non-JSON content types.** `GET /api/achievements/docs/<doc_id>/raw`
  streams `text/plain`, not JSON.
- **SPA-serving routes.** The Angular static-asset / deep-link catch-all
  routes serve HTML, not JSON.

---

*Readme Reader code phrase:* `json all the way down`

