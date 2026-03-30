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
      "folder_import_name": "sounds",
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

→ `{"embedders": [{"name": "clip", "display_name": "CLIP", "media_type_id": "image", ...}]}`

### Clippers

```
GET /api/clippers
```

**Query:** `?media_type=audio` — optional, filter by `type_id` or
`folder_import_name`.

→ `{"clippers": [{"name": "sound_default", "media_type": "audio", ...}]}`

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

→ `{"loaded": true, "num_medias": 500, "has_votes": true, "media_type": "audio", "num_dupes": 3}`

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

**Body:** `{"path": "/data/sounds", "media_type": "sounds"}`

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

### Activate registered dataset

```
POST /api/datasets/registry/{dataset_id}/activate
```

Switches the active dataset context to the specified dataset (must already
be loaded). This is an instant operation — no re-embedding.

→ `{"ok": true}`

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

### Set dataset readers

```
PUT /api/datasets/registry/{dataset_id}/readers
```

**Body:** `{"readers": ["user1", "user2"]}`

Sets which users can access a dataset (multi-user deployments). Only the
dataset owner or an admin can modify readers.

→ `{"ok": true, "readers": ["user1", "user2"]}`
