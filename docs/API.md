# HTTP API Reference

VTSearch exposes a REST-style JSON API. All endpoints accept and return JSON
unless otherwise noted. File uploads use `multipart/form-data`.

## Sub-pages

| Section | Description |
|---------|-------------|
| [Authentication & UI](api/auth.md) | Auth status, login/logout, static assets |
| [Medias & Sorting](api/medias.md) | Media listing/streaming, text/learned/example sort, votes & labels, pile upload |
| [Labeling & Diversity](api/labeling.md) | Inclusion, thresholds, labeling progress, diversity tree |
| [Detectors & Processors](api/detectors.md) | Detectors, extractors, localizers, pre-generated processors |
| [Datasets](api/datasets.md) | Loading, importers, demos, staging, registry, media types, embedders, clippers, converters, file browsing |
| [Import & Export](api/io.md) | Result exporters, label importers, processor importers |
| [Settings](api/settings.md) | App settings, autorun processors, settings sources, labelset sources |
| [Models](api/models.md) | Trainable models, model registry |
| [Dashboard & Lookup](api/dashboard.md) | Dashboard info, multi-dataset find, find progress |
| [File Browser](api/file-browser.md) | Server filesystem browsing |

## Conventions

| Pattern | Meaning |
|---------|---------|
| `{param}` | URL path parameter |
| **Body** | JSON request body (Content-Type `application/json`) |
| **Form** | `multipart/form-data` |
| `→` | Response body |
| Async endpoints | Return immediately; poll `GET /api/dataset/progress` |

**Common error shape:**

```json
{"error": "Human-readable message"}
```

Status codes follow standard HTTP semantics: 200 OK, 201 Created, 204 No
Content, 400 Bad Request, 404 Not Found, 409 Conflict, 500 Internal Server
Error.

---

*Readme Reader code phrase:* `json all the way down`

