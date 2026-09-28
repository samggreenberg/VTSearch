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
| [Authentication & UI](api/auth.md) | Auth status, login/logout, HuggingFace OAuth, SPA routes, static assets, health probes, version |
| [Medias & Sorting](api/medias.md) | Media listing/streaming, text/learned/example sort, votes & labels, pile upload, example-media files, seed and datasource importers |
| [Labeling & Diversity](api/labeling.md) | Inclusion, labeling status/progress, indicator history, train-and-score, coverage atlas |
| [Detectors](api/detectors.md) | Detector CRUD, saved-label review, detector registry, Auto-Find toggle, loading, labelset-file moves |
| [Datasets](api/datasets.md) | Loading, importers, demos, staging, registry, media types, embedders, clippers, cleaners, converters, media-file browsing, VTSBrowse projection |
| [Import & Export](api/io.md) | Result exporters, label importers, pregen processors, autorun extractors/localizers and running them, settings importers/exporters |
| [Settings](api/settings.md) | App settings, settings sources, labelset sources |
| [Dashboard](api/dashboard.md) | Dashboard disk/RAM usage probes |
| [Find, Auto-Detect & Scoring](api/find.md) | Multi-dataset find (+ cancel/check-labels), Find Label, Auto-Detect, find stats/corrections/queues/evidence coverage |
| [File Browser](api/file-browser.md) | Server filesystem browsing |
| [Progress events (SSE)](api/events.md) | `GET /api/events`: the single Server-Sent Events stream carrying progress for every long-running operation |

These pages explain **purpose and non-obvious semantics** — ordering,
idempotency, async hand-offs, which errors mean what. For the exhaustive field
list of any request or response, the generated spec is authoritative (see
[Machine-readable schema](#machine-readable-schema)); where a page and the spec
disagree on a field, trust the spec and fix the page.

### Other endpoints

Small families with no page of their own. Field lists are in the spec.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/achievements` | Full achievement state: `tier_names`, `achievements` (per-category `counter`, `tier_idx` (`-1` = locked), `next_threshold`), `pending_announcements`, `pending_toasts`, `docs`, `media_types`, `hours`. Counters are zeroed when the `enable_achievements` setting is off. |
| POST | `/api/achievements/{category_id}/acknowledge` | Body `{"tier_idx": N}`. Marks that tier as announced in the panel. → `{"ok", "changed"}` |
| POST | `/api/achievements/{category_id}/mark-toasted` | Body `{"tier_idx": N}`. Marks the tier's unlock toast as shown so it fires once. → `{"ok", "changed"}` |
| POST | `/api/achievements/check-phrase` | Body `{"phrase": "..."}`. Checks a Readme Reader code phrase. → `{"matched", "doc_id", "doc_name", "already_read"}` |
| GET | `/api/achievements/docs/{doc_id}/raw` | Raw markdown of a Readme Reader doc, `text/plain` (this file is one). 404 for an unknown id. |
| GET | `/api/sessions/recent` | The current user's recently used `(dataset, detector)` pairs, newest first: `{"sessions": [{dataset_id, detector_id, dataset_name, detector_name, last_activity}]}`. Pairs whose dataset or detector was deleted are dropped on read. |
| POST | `/api/sessions/recent` | Body `{"dataset_id", "detector_id"}`. Records activity on a pair; returns the updated list. |
| GET | `/api/jobs/active` | `{"busy_pairs": [{dataset_id, detector_id, job_types}]}` — every pair with a running or pending background job (learned sort, eval, …). Cheap and lock-free; safe to poll every few seconds. |

Health probes and `GET /api/version` are in [auth.md](api/auth.md#health--version);
the VTSBrowse projection endpoints are in [datasets.md](api/datasets.md#vtsbrowse-projection).

## Context headers (`X-Dataset-Id` / `X-Detector-Id`)

VTSearch holds **multiple loaded datasets and detectors at once**, one context
each. Most endpoints operate on "the active context", and the active context is
chosen **per request** by two HTTP headers:

| Header | Selects | Sent by |
|--------|---------|---------|
| `X-Dataset-Id` | Which loaded `DatasetContext` the request's `medias` / coverage / dataset-scoped votes resolve to | Angular's `HttpClient` interceptor on every API call |
| `X-Detector-Id` | Which loaded `DetectorContext` the request's `good_votes` / `bad_votes` / model / labelset resolve to | Same interceptor |

Key semantics (`app.py` `before_request`, `vtsearch/routes/_context.py`):

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
| Async endpoints | Return immediately (usually with a `task_id` or `job_id`); follow progress on [`GET /api/events`](api/events.md) (SSE) — dataset loads on the `loading-tasks` channel, detector loads/ingests on `detector-loading-tasks` |

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
| `errors` | Schema validation failed (**422**); maps location → field → messages, e.g. `{"json": {"volume": ["Not a valid number."]}}` or `{"query": {...}}` |
| `detail` | 500s from the catch-all handler; the exception type and its first line, e.g. `"RuntimeError: embedder X not loaded"` |
| `error_code` | A machine-readable slug where the client branches on the *kind* of failure: `auth_required` (401), `dataset_not_loaded` / `detector_not_loaded` (409: the id is registered but not in memory yet, so a load cures it), `dataset_not_found` / `detector_not_found` (404: the registry no longer lists the id — it was deleted) |

A schema-validation failure therefore looks like:

```json
{
  "code": 422,
  "status": "Unprocessable Entity",
  "message": "Unprocessable Entity",
  "errors": {"json": {"volume": ["Not a valid number."]}},
  "request_id": "45159ab44b01"
}
```

(`status` is the reason phrase as the running Python spells it —
`"Unprocessable Content"` from 3.13 on. Match on `code` / `errors`.) Unknown
`/api/*` paths and wrong methods also get this envelope (JSON 404 / 405), never
the SPA's HTML.

Status codes follow standard HTTP semantics: 200 OK, 201 Created, 400 Bad
Request (handler-level validation), 401 (auth, see
[auth.md](api/auth.md#server-side-enforcement)), 403 (per-user access to a
registered dataset/detector), 404 Not Found, 409 Conflict, 422 (schema
validation), 500, plus 501/502 from plugin-backed routes (plugin not
implemented / upstream failure) and 503 from `GET /api/events` and `/readyz`.

## Machine-readable schema

`GET /api/openapi.json` returns an OpenAPI 3.0 document describing the
running app, generated by `flask-smorest` from the marshmallow schemas in
`vtsearch/schemas/`. A browsable Swagger UI is served at `GET /api/docs`
(its assets load from the jsDelivr CDN, so it needs internet access). The
committed snapshot `frontend/openapi.json` is the same document; the Angular
client is generated from it, and `./run-tests.sh` fails when the live spec
drifts from it. Use it to:

- Look up the exhaustive field list, types, enums, and defaults of any
  request or response.
- Browse / try endpoints live via Swagger UI.
- Generate a TypeScript / Python client.

### Routes with no typed schema

Every route on a `flask_smorest` blueprint appears in the spec — path,
parameters, summary, and error responses — even when it declares no
request/response schema. A route is left without one for one of these reasons,
so for these the pages here (or the route's spec `description`) are the
contract:

- **Plugin-field bodies.** Endpoints whose body is the fields of the plugin
  named in the URL — `POST /api/dataset/import/{importer_name}`,
  `POST /api/dataset/stage-import/{importer_name}`,
  `POST /api/label-importers/import/{importer_name}`,
  `POST /api/detectors/{name}/import-labels/{importer_name}`,
  `POST /api/detectors/registry/from-labelset/{importer_name}`,
  `POST /api/settings-importers/import/{importer_name}`,
  `POST /api/seed-import/{importer_name}`,
  `POST /api/datasource-import/{importer_name}`. The accepted keys are the
  plugin's declared `fields` (fetch them from the family's list endpoint), sent
  as JSON, or multipart when the plugin declares a `file` field. Runtime
  validation still goes through `validate_plugin_args`, so a missing required
  field or an invalid `select` value still returns 422 with the standard
  `errors` envelope.
- **Binary responses.** Audio/video/image/media/thumbnail/preview routes, the
  dataset and portable-bundle downloads, and `GET
  /api/achievements/docs/{doc_id}/raw` (`text/plain`).
- **Dual-mode dispatchers.** `POST /api/embed` accepts either
  `multipart/form-data` (file upload) or `application/json` (text), decided
  at request time.
- **SPA-serving routes.** `/`, `/label`, `/dashboard`, favicons, `/logo.svg`,
  and the `/{path}` catch-all serve HTML or static files.

### Routes absent from the spec

A few routes live on plain Flask blueprints and are not in the spec at all:

- `GET /api/events` — the SSE stream ([events.md](api/events.md)).
- `/api/auth/huggingface/*` — the OAuth browser-redirect flow
  ([auth.md](api/auth.md#huggingface-oauth-sign-in-with-huggingface)).
- `GET /api/openapi.json` and `GET /api/docs` themselves.

---

*Readme Reader code phrase:* `json all the way down`

