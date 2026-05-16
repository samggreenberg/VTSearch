# Models

[← Back to API index](../API.md)

---

## Trainable Models

### List detectors

```
GET /api/detectors
```

→ `{"models": [{"name": "Dog Barks", "text_query": "dog barking", "media_type": "audio", "examples": [...], "num_labels": 50, "created_at": 1234567890.0}]}`

### Create detector

```
POST /api/detectors
```

**Body:** `{"name": "Dog Barks", "text_query": "dog barking sounds", "media_type": "audio"}`

Or with examples: `{"name": "Dog Barks", "examples": [{"type": "text", "value": "dog barking"}]}`

→ `{"success": true, "name": "...", "text_query": "...", "media_type": "audio", "examples": [...], "num_labels": 0}` (201)

409 if name already exists.

### Get detector

```
GET /api/detectors/{name}
```

→ Full model object including `labelset`.

### Delete detector

```
DELETE /api/detectors/{name}
```

→ `{"success": true, "name": "..."}`

### Rename detector

```
PUT /api/detectors/{name}/rename
```

**Body:** `{"new_name": "Cat Meows"}`

→ `{"success": true, "old_name": "...", "new_name": "Cat Meows"}`

409 if new name already exists.

### Set examples

```
PUT /api/detectors/{name}/examples
```

**Body:** `{"examples": [{"type": "text", "value": "dog barking"}]}`

→ `{"success": true, "name": "...", "examples": [...]}`

### Save labels

```
POST /api/detectors/{name}/labels
```

Saves the current good/bad votes as the model's labelset.

→ `{"success": true, "name": "...", "num_labels": 50}`

### Import labels into model

```
POST /api/detectors/{name}/import-labels/{importer_name}
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
GET /api/detectors/registry
```

→ ```json
{
  "models": [
    {
      "id": "abc123",
      "name": "Dog Barks",
      "media_type": "audio",
      "text_query": "dog barking",
      "num_training": 50,
      "loaded": true,
      "detector_loaded": true,
      "autorun": false
    }
  ]
}
```

`name` is the slug used to look up the on-disk labelset file at
`data/detectors/<name>.json`.  Every registered model is a
detector — the MLP is trained on demand from the labelset and
lives only in RAM.  `autorun` mirrors whether the model's name appears
in `autorun_detectors` settings (toggle it with the route below).

### Register model

```
POST /api/detectors/registry
```

**Body:**

```json
{
  "name": "Dog Barks",
  "media_type": "audio",
  "text_query": "dog barking sounds"
}
```

→ `{"ok": true, "model": {...}}` (201)

### Toggle autorun flag

```
PUT /api/detectors/registry/{model_id}/autorun
```

**Body:** `{"autorun": true}`

→ `{"ok": true, "autorun": true}` — writes the model's name into
`autorun_detectors` so `/api/auto-detect` and the CLI
`--autodetect` flow pick it up.

### Load / unload model

```
POST /api/detectors/registry/load
```

**Body:** `{"model_id": "abc123"}` — pass `null` to unload.

→ `{"ok": true, "message": "Loading started", "task_id": "..."}`

Loading is async — subscribe to the `detector-loading-tasks` channel
on [`/api/events`](events.md) (SSE) for progress. 404 if model not found.

### Unload model

```
POST /api/detectors/registry/{model_id}/unload
```

→ `{"ok": true}`

### Model loading tasks (SSE)

Active model loading tasks are streamed on the `detector-loading-tasks`
channel of [`/api/events`](events.md):

```json
[{"task_id": "...", "name": "...", "status": "loading", "message": "...", "current": 50, "total": 100}]
```

### Cancel model loading

```
POST /api/detectors/cancel/{task_id}
```

→ `{"ok": true}`

### Delete registered model

```
DELETE /api/detectors/registry/{model_id}
```

Also cleans up any associated detector file and autorun detector.

→ `{"ok": true}`

### Rename registered model

```
PUT /api/detectors/registry/{model_id}/rename
```

**Body:** `{"name": "New Name"}`

→ `{"ok": true, "name": "New Name"}`
