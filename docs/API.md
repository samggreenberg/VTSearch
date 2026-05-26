# HTTP API Reference

VTSearch exposes a REST-style JSON API. All endpoints accept and return JSON
unless otherwise noted. File uploads use `multipart/form-data`.

## Sub-pages

| Section | Description |
|---------|-------------|
| [Authentication & UI](api/auth.md) | Auth status, login/logout, static assets |
| [Medias & Sorting](api/medias.md) | Media listing/streaming, text/learned/example sort, votes & labels, pile upload |
| [Labeling & Diversity](api/labeling.md) | Inclusion, thresholds, labeling progress, diversity tree |
| [Detectors](api/detectors.md) | Detector CRUD, detector registry, autorun toggle, loading SSE |
| [Datasets](api/datasets.md) | Loading, importers, demos, staging, registry, media types, embedders, clippers, converters, file browsing |
| [Import & Export](api/io.md) | Result exporters, label importers, processor importers |
| [Settings](api/settings.md) | App settings, autorun processors, settings sources, labelset sources |
| [Dashboard & Lookup](api/dashboard.md) | Dashboard info, multi-dataset find, find progress |
| [File Browser](api/file-browser.md) | Server filesystem browsing |

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
- Diff against the snapshot at `frontend/openapi.json` to catch unintended API surface changes: `./run-tests.sh` regenerates the spec and fails on drift.

---

*Readme Reader code phrase:* `json all the way down`

