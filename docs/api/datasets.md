# Datasets

[← Back to API index](../API.md)

---

## Media Types, Embedders, Clippers & Converters

### Media types

```
GET /api/media-types
```

→ ```json
{
  "media_types": [
    {
      "type_id": "audio",
      "name": "Audio",
      "icon": "🔊",
      "tab_title": "Sounds",
      "folder_import_name": "audio",
      "loops": true,
      "file_extensions": ["*.wav", "*.mp3"]
    }
  ]
}
```

### Embedders

```
GET /api/embedders
```

**Query:** `?media_type=image` — optional, filter by `type_id` or
`folder_import_name`.

→ `{"embedders": [{"name": "siglip", "display_name": "SigLIP", "media_type_id": "image", ...}]}`

### Embed on demand

```
POST /api/embed
```

Embed a single media file or text snippet with a chosen embedder, without
loading a dataset.  Two input modes share the endpoint — the embedder
declares its own `media_type_id`, so the caller does not pass a media
type separately.

**Media upload (`multipart/form-data`)** — form fields:

| Field | Description |
|-------|-------------|
| `embedder` | Required. Embedder name from `GET /api/embedders`. |
| `file` | Required. Binary upload. The extension is checked against the embedder's `media_type_id` before any model load. |

**Text (`application/json`)** — body:

```json
{"embedder": "e5", "text": "a cat on a mat"}
```

Calls `embed_text(...)` — only embedders whose `supports_text` is `true`
accept this mode (image/audio cross-modal embedders like CLIP, SigLIP, CLAP
support it; vision-only embedders like DINOv3 do not).

→
```json
{
  "embedding": [0.123, -0.045, ...],
  "dim": 512,
  "norm": 1.0,
  "embedder": "clip",
  "media_type": "image"
}
```

**Errors:**

| Status | Cause |
|--------|-------|
| `400` | Missing `embedder`; missing `file` (multipart) or `text` (JSON); text sent to an embedder where `supports_text == false`; file extension does not match the embedder's media type; embedder returned no vector for the upload. |
| `404` | Unknown embedder name. Response body lists every registered embedder's `name`. |
| `500` | Model load failure, or text embedding returned `None`. |

### Clippers

```
GET /api/clippers
```

**Query:** `?media_type=audio` — optional, filter by `type_id` or
`folder_import_name`.

→ `{"clippers": [{"name": "sound_default", "media_type": "audio", "display_name": "Sound Default", "description": "Import each audio file as-is, without splitting.", ...}]}`

Each clipper object includes `name`, `display_name`, `media_type`, and
an optional `description` (short tooltip text). Clippers with
configurable settings also include `parameters` and
`creation_questions` arrays, where each parameter has `key`, `label`,
`type`, `default`, and an optional `description` for hover tooltips.

### Converters

```
GET /api/converters
```

**Query (mutually exclusive):** `?source=video` or `?target=image` — filter by
`type_id` or `folder_import_name`. Omit both to list all converters.

→ `{"converters": [{"name": "video2image", "source_type": "video", "target_type": "image", ...}]}`

---

## Dataset Status & Progress

### Dataset status

```
GET /api/dataset/status
```

→ `{"loaded": true, "num_medias": 500, "has_votes": true, "media_type": "audio", "display_name": "ESC-50", "num_dupes": 3}`

### Dataset progress

```
GET /api/dataset/progress
```

→ Progress object for long-running operations (loading, embedding, etc.):

```json
{"status": "loading", "message": "Embedding medias…", "cur": 50, "total": 500}
```

### Loading tasks

```
GET /api/dataset/loading-tasks
```

→ All active dataset loading tasks with their progress:

```json
{"tasks": [{"id": "task_abc", "name": "ESC-50", "status": "loading", "message": "...", "cur": 50, "total": 500}]}
```

### Cancel loading

```
POST /api/dataset/cancel
```

Cancels all active loading tasks.

→ `{"ok": true}`

```
POST /api/dataset/cancel/{task_id}
```

Cancels a specific loading task.

→ `{"ok": true}` or `{"error": "Task not found"}` (404)

---

## Importers

### List importers

```
GET /api/dataset/all-importers
```

Returns all registered importers including built-in ones (pickle, combine_datasets, demo).

→ `{"importers": [{"name": "...", "display_name": "...", "description": "...", "fields": [...]}]}`

```
GET /api/dataset/importers
```

Returns only importers with `ui_mode == "form"` (excludes pickle, combine_datasets, demo, and any other importers with non-form UI modes). Used by the frontend's generic form-based import dialog.

→ `{"importers": [...]}`

### Available dataset files

```
GET /api/dataset/available-files
```

Lists `.pkl` files in the embeddings directory.

→ `{"files": [{"name": "esc50", "path": "/abs/path/esc50.pkl", "size_mb": 12.3}]}`

---

## Loading Datasets

**From uploaded pickle:**

```
POST /api/dataset/load-file
```

**Form:** `file` — `.pkl` file.

→ `{"ok": true, "message": "Loading started"}`

**From folder:**

```
POST /api/dataset/load-folder
```

**Body:** `{"path": "/data/sounds", "media_type": "audio"}`

→ `{"ok": true, "message": "Loading started"}`

**From demo:**

```
POST /api/dataset/load-demo
```

**Body:** `{"name": "esc50_animals"}`

→ `{"ok": true, "message": "Loading started"}`

**From importer:**

```
POST /api/dataset/import/{importer_name}
```

**Form or Body:** importer-specific fields.

→ `{"ok": true, "message": "Loading started"}`

**From source origin:**

```
POST /api/dataset/load-source
```

**Body:** `{"source": {"importer": "demo", "params": {"name": "esc50"}}}`

→ `{"ok": true, "message": "Loading started"}`

All load endpoints are async — poll `GET /api/dataset/progress`.

### Demo datasets

```
GET /api/dataset/demo-list
```

→ ```json
{
  "datasets": [
    {
      "name": "esc50_animals",
      "label": "ESC-50 Animals",
      "status": "ready",
      "ready": true,
      "num_files": 200,
      "download_size_mb": 45.2,
      "description": "...",
      "media_type": "audio",
      "num_categories": 5
    }
  ]
}
```

`status`: `"ready"`, `"needs_embedding"`, or `"needs_download"`.

### Demo categories

```
GET /api/dataset/demo-categories/{name}
```

Lists the categories within a specific demo dataset.

→ `{"categories": ["dog", "cat", "bird", "traffic"]}`
404 if the demo name is not recognized.

---

## File Browsing

### Browse media files

```
GET /api/browse-media-files
```

**Query:**
- `source` (required): `"demo:<name>"` (a demo dataset) or `"folder"` (the
  configured `saved_datasets_dir`).
- `path` (optional): relative sub-path within the root (default `""`).

Lists files and subdirectories within an allowed root, filtered to only
media files with recognized extensions.

→ ```json
{
  "directories": [{"name": "dog", "path": "dog", "modified_at": "2025-03-31T10:15:00"}],
  "files": [{"name": "bark.wav", "path": "dog/bark.wav", "size_bytes": 12345, "modified_at": "2025-03-31T10:15:00"}],
  "root_path": "/absolute/path/to/root"
}
```

### Select browsed file

```
POST /api/browse-media-files/select
```

**Body:** `{"source": "demo:esc50_s", "path": "dog/1-100032-A-0.wav"}`

Copies a file from a browse source into `data/example_media/` with a unique
prefix to avoid collisions.

→ `{"filename": "abc123_bark.wav", "original_name": "bark.wav"}` (201)

---

## Staging (for combine-datasets flow)

```
POST /api/dataset/stage-file
```

**Form:** `file` — `.pkl` file.

→ `{"path": "/abs/staging/path.pkl", "name": "uploaded.pkl", "count": 500, "media_type": "audio"}`

```
POST /api/dataset/stage-import/{importer_name}
```

→ `{"ok": true, "message": "Staging started"}`

```
POST /api/dataset/stage-demo/{name}
```

→ `{"ok": true, "message": "Staging demo dataset..."}`

```
DELETE /api/dataset/staging
```

→ `{"ok": true}`

### Combine datasets

```
POST /api/dataset/combine
```

**Body:** `{"datasets": ["/path/to/a.pkl", "/path/to/b.pkl"]}`

Requires at least two paths.

→ `{"ok": true, "message": "Combining datasets..."}`

---

## Export & Clear

### Export dataset

```
GET /api/dataset/export
```

→ Binary `.pkl` file download.

### Clear dataset

```
POST /api/dataset/clear
```

→ `{"ok": true}`

---

## Dataset Registry

### List registered datasets

```
GET /api/datasets/registry
```

→ ```json
{
  "datasets": [
    {
      "id": "abc123",
      "name": "ESC-50",
      "media_type": "audio",
      "num_items": 500,
      "loaded": true,
      "origin": "demo:esc50",
      "source": {"importer": "demo", "params": {"name": "esc50"}},
      "created_at": 1234567890.0
    }
  ]
}
```

### Load registered dataset

```
POST /api/datasets/registry/{dataset_id}/load
```

→ `{"ok": true, "message": "Loading started"}`

### Unload registered dataset

```
POST /api/datasets/registry/{dataset_id}/unload
```

→ `{"ok": true}`

### Delete registered dataset

```
DELETE /api/datasets/registry/{dataset_id}
```

→ `{"ok": true}`

### Rename registered dataset

```
PUT /api/datasets/registry/{dataset_id}/rename
```

**Body:** `{"name": "New Name"}`

→ `{"ok": true, "name": "New Name"}`

### Dataset statistics

```
GET /api/datasets/registry/{dataset_id}/stats
```

Returns ingest statistics for a registered dataset.

→ ```json
{
  "num_items": 1250,
  "num_dupes": 45,
  "file_type_counts": {"audio/wav": 800, "audio/mp3": 450},
  "ingest_started_at": "2025-03-31T10:15:00",
  "ingest_finished_at": "2025-03-31T10:45:00"
}
```

404 if the dataset does not exist.

### Set dataset readers

```
PUT /api/datasets/registry/{dataset_id}/readers
```

**Body:** `{"readers": ["user1", "user2"]}`

Sets which users can access a dataset (multi-user deployments). Only the
dataset owner or an admin can modify readers.

→ `{"ok": true, "readers": ["user1", "user2"]}`
