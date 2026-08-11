# File Browser

[← Back to API index](../API.md)

---

## General File Browser

### Browse files

```
GET /api/browse
```

**Query:**
- `path` (optional): path relative to the browse root (default `""`, the
  root itself).
- `extensions` (optional): comma-separated list of file extensions to show,
  e.g. `".csv,.json"`. When omitted, all files are listed.

Lists directories and files at a path within the browse root. What that
root is depends on the active login provider (see
[Authentication](auth.md#login-providers)):

- **Single-user mode** (`DefaultLoginProvider`, the default — no `--login`
  flag): the root is the **filesystem root**, `/`. There is no confinement:
  the lone trusted user may browse any directory the server process can
  read, matching the unrestricted server-path validation applied to
  importers and exporters (`get_file_access_base_dir()` returns `None`).
  Paths in the response are absolute (`/data`, `/data/sounds`), so the value
  can be handed straight to a server-path field.
- **Multi-user mode** (any other provider): the root is the current user's
  data directory, `data/<username>/`, and the browser is confined to that
  subtree. Paths in the response are relative to it, so the server's
  absolute layout never leaks.

In both modes the server's own root path is omitted from the response, and
`path` is interpreted relative to the root. Hidden files (names starting
with `.`) are excluded, as are symlinks whose target resolves outside the
root.

> **Deploying this publicly?** In the default single-user mode this endpoint
> exposes the whole server filesystem to anyone who can reach the port, and
> nothing authenticates the caller. See
> [Security](../DEPLOYMENT.md#security) before putting VTSearch on an
> untrusted network.

→
```json
{
  "directories": [
    {"name": "subdir", "path": "subdir", "modified_at": "2025-03-31T10:15:00"}
  ],
  "files": [
    {"name": "labels.csv", "path": "labels.csv", "size_bytes": 1234, "modified_at": "2025-03-31T10:15:00"}
  ],
  "current_path": "data/labels"
}
```

(Shown with a confined root; in single-user mode the same listing carries
absolute paths — `"path": "/data/labels/labels.csv"`,
`"current_path": "/data/labels"`.)

400 if the path escapes the browse root (path traversal prevention) and
403 if permission is denied reading the directory; both use the standard
`{"message": "..."}` envelope.

404 if the directory does not exist. This one is intercepted by the
app-level `NotFound` handler, so it keeps the legacy shape
`{"error": "Not Found", "request_id": "..."}` rather than the `message`
envelope.

422 if a query parameter fails schema validation, with the standard
per-field `errors` envelope:

```json
{
  "code": 422,
  "status": "Unprocessable Content",
  "errors": {"query": {"path": ["Not a valid string."]}}
}
```

(`status` is the HTTP reason phrase as the running interpreter spells it —
`"Unprocessable Entity"` before Python 3.13, `"Unprocessable Content"` from
3.13 on. Match on `code` / `errors`, not on that string.)

Both query params are optional strings, so in practice this endpoint
rarely produces a 422; it is declared (and appears in
`/api/openapi.json`) because the route is schema-validated like every
other `flask-smorest` endpoint.

---

The media-specific file browser endpoints (`/api/browse-media-files` and
`/api/browse-media-files/select`) are documented in [Datasets](datasets.md#file-browsing).
