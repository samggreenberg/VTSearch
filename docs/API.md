# HTTP API Reference

VTSearch exposes a REST-style JSON API. All endpoints accept and return JSON
unless otherwise noted. File uploads use `multipart/form-data`.

## Sub-pages

| Section | Description |
|---------|-------------|
| [Authentication & UI](api/auth.md) | Auth status, login/logout, static assets |
| [Medias & Sorting](api/medias.md) | Media listing/streaming, text/learned/example sort, votes & labels, pile upload |
| [Labeling & Diversity](api/labeling.md) | Inclusion, thresholds, labeling progress, diversity tree |
| [Detectors](api/detectors.md) | Detector CRUD, detector registry, Auto-Find toggle, loading SSE |
| [Datasets](api/datasets.md) | Loading, importers, demos, staging, registry, media types, embedders, clippers, converters, file browsing |
| [Import & Export](api/io.md) | Result exporters, label importers, pregen processors, autorun extractors/localizers |
| [Settings](api/settings.md) | App settings, autorun processors, settings sources, labelset sources |
| [Dashboard & Lookup](api/dashboard.md) | Dashboard info (incl. disk/RAM usage), multi-dataset find (incl. cancel/stats/corrections), find progress |
| [File Browser](api/file-browser.md) | Server filesystem browsing |

### Not yet documented in depth

The OpenAPI snapshot (`frontend/openapi.json`) is the ground truth and covers
several endpoint families that the hand-written pages above don't yet describe
in full. Use Swagger (`GET /api/docs`) for these:

- **Achievements** — gamification unlock/state endpoints (see `vtsearch/achievements.py`).
- **Projection / VTSBrowse** — `GET/POST /api/projection/*` (UMAP projection + hex-tile pyramid).
- **Sessions** — session lifecycle endpoints.
- **Jobs** — async job listing/status (backed by `vtscore.concurrency.async_jobs`).
- **Find** — `cancel`, `stats`, and `corrections` companions to multi-dataset find.
- **Dashboard usage** — disk-usage / RAM-usage probes.
- **Health probes** — `GET /healthz`, `GET /readyz` (liveness / readiness).
- **Version** — `GET /api/version` (returns the running build's version string).

## Conventions

| Pattern | Meaning |
|---------|---------|
| `{param}` | URL path parameter |
| **Body** | JSON request body (Content-Type `application/json`) |
| **Form** | `multipart/form-data` |
| `→` | Response body |
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

