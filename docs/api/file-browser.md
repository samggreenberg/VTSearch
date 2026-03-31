# File Browser

[← Back to API index](../API.md)

---

## General File Browser

### Browse files

```
GET /api/browse
```

**Query:**
- `path` (optional): relative path within the allowed root (default `""`).
- `extensions` (optional): comma-separated list of file extensions to show,
  e.g. `".csv,.json"`. When omitted, all files are listed.

Lists directories and files at a relative path within the allowed root
directory. In single-user mode the root is the current working directory;
in multi-user mode it is the current user's data directory. Hidden files
(names starting with `.`) are excluded.

→ ```json
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

400 if the path escapes the allowed root (path traversal prevention).
403 if permission is denied. 404 if the directory does not exist.

---

The media-specific file browser endpoints (`/api/browse-media-files` and
`/api/browse-media-files/select`) are documented in [Datasets](datasets.md#file-browsing).
