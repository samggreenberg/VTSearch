# Models

[← Back to API index](../API.md)

---

## Trainable Models

### List trainable models

```
GET /api/trainable-models
```

→ `{"models": [{"name": "Dog Barks", "text_query": "dog barking", "media_type": "audio", "examples": [...], "num_labels": 50, "created_at": 1234567890.0}]}`

### Create trainable model

```
POST /api/trainable-models
```

**Body:** `{"name": "Dog Barks", "text_query": "dog barking sounds", "media_type": "audio"}`

Or with examples: `{"name": "Dog Barks", "examples": [{"type": "text", "value": "dog barking"}]}`

→ `{"success": true, "name": "...", "text_query": "...", "media_type": "audio", "examples": [...], "num_labels": 0}` (201)

409 if name already exists.

### Get trainable model

```
GET /api/trainable-models/{name}
```

→ Full model object including `labelset`.

### Delete trainable model

```
DELETE /api/trainable-models/{name}
```

→ `{"success": true, "name": "..."}`

### Rename trainable model

```
PUT /api/trainable-models/{name}/rename
```

**Body:** `{"new_name": "Cat Meows"}`

→ `{"success": true, "old_name": "...", "new_name": "Cat Meows"}`

409 if new name already exists.

### Set examples

```
PUT /api/trainable-models/{name}/examples
```

**Body:** `{"examples": [{"type": "text", "value": "dog barking"}]}`

→ `{"success": true, "name": "...", "examples": [...]}`

### Save labels

```
POST /api/trainable-models/{name}/labels
```

Saves the current good/bad votes as the model's labelset.

→ `{"success": true, "name": "...", "num_labels": 50}`

### Import labels into model

```
POST /api/trainable-models/{name}/import-labels/{importer_name}
```

Run a label importer and merge results into this model's persisted labelset.
Unlike `/api/label-importers/import/`, this does **not** require a dataset to
be loaded. Field values are passed as JSON body or multipart form (same as
regular label import).

When the model's detector context **is** loaded, the new labels are also
resolved against the loaded dataset's medias, applied to the detector's votes,
and a fresh MLP is trained with a cross-validated threshold.

→ `{"success": true, "applied": 12, "skipped": 3, "num_labels": 62, "message": "..."}`

404 if model or importer not found. 400 on validation errors.

---

## Model Registry

### List registered models

```
GET /api/models/registry
```

→ ```json
{
  "models": [
    {
      "id": "abc123",
      "name": "Dog Barks",
      "media_type": "audio",
      "trainable": true,
      "text_query": "dog barking",
      "detector_name": "",
      "trainable_model_name": "Dog Barks",
      "num_training": 50,
      "loaded": true
    }
  ]
}
```

### Register model

```
POST /api/models/registry
```

**Body:**

```json
{
  "name": "Dog Barks",
  "media_type": "audio",
  "trainable": true,
  "text_query": "dog barking sounds",
  "detector_name": "",
  "trainable_model_name": ""
}
```

→ `{"ok": true, "model": {...}}` (201)

### Load / unload model

```
POST /api/models/registry/load
```

**Body:** `{"model_id": "abc123"}` — pass `null` to unload.

→ `{"ok": true, "message": "Loading started", "task_id": "..."}`

Loading is async — poll `GET /api/models/loading-tasks` for progress.
404 if model not found.

### Unload model

```
POST /api/models/registry/{model_id}/unload
```

→ `{"ok": true}`

### Model loading tasks

```
GET /api/models/loading-tasks
```

→ `{"tasks": [{"id": "...", "name": "...", "status": "loading", "message": "...", "cur": 50, "total": 100}]}`

### Cancel model loading

```
POST /api/models/cancel/{task_id}
```

→ `{"ok": true}`

### Delete registered model

```
DELETE /api/models/registry/{model_id}
```

Also cleans up any associated trainable model file and autorun detector.

→ `{"ok": true}`

### Rename registered model

```
PUT /api/models/registry/{model_id}/rename
```

**Body:** `{"name": "New Name"}`

→ `{"ok": true, "name": "New Name"}`
