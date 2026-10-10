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

→ `{directories, files, current_path}`; each entry carries `name`, `path` and
`modified_at`, and files also `size_bytes`. `current_path` is `""` at the root.

400 if the path escapes the browse root, 403 if reading the directory is denied,
404 if it does not exist. A 422 (standard `errors` envelope, see
[Conventions](../API.md#conventions)) is declared because the route is
schema-validated, but both parameters are optional strings, so it is rare.

---

The media-specific file browser endpoints (`/api/browse-media-files` and
`/api/browse-media-files/select`) are documented in [Datasets](datasets.md#file-browsing).
