# Detectors

[← Back to API index](../API.md)

---

## Detectors

A detector is a named labelset plus a text-sort query, persisted as a
JSON file at `data/detectors/<name>.json`. The MLP is trained on demand
from the labelset and lives only in `DetectorContext` once the user
loads the detector into memory.

### List detectors

```
GET /api/detectors
```

→ `{"detectors": [{"name": "Dog Barks", "text_query": "dog barking", "media_type": "audio", "examples": [...], "num_labels": 50, "created_at": 1234567890.0}]}`

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

→ Full detector object including `labelset`.

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

Saves the current good/bad votes as the detector's labelset.

→ `{"success": true, "name": "...", "num_labels": 50}`

### Import labels into detector

```
POST /api/detectors/{name}/import-labels/{importer_name}
```

Run a label importer and merge results into this detector's persisted labelset.
Unlike `/api/label-importers/import/`, this does **not** require a dataset to
be loaded. Field values are passed as JSON body or multipart form (same as
regular label import).

When the detector context **is** loaded, the new labels are also resolved
against the loaded dataset's medias, applied to the detector's votes, and
a fresh MLP is trained with a cross-validated threshold.

→ `{"success": true, "applied": 12, "skipped": 3, "num_labels": 62, "message": "..."}`

404 if detector or importer not found. 400 on validation errors.

### Combine detectors

```
POST /api/detectors/combine
```

**Body:** `{"names": ["A", "B"], "new_name": "A+B", "conflict_policy": "drop"}`

Merges the labelsets of two or more detectors into a new detector. All
sources must share a `media_type`. `conflict_policy="drop"` (the only
supported policy) removes any element that appears with disagreeing
labels across sources.

→ `{"success": true, "name": "A+B", "media_type": "audio", "num_labels": 73, "combined_from": ["A", "B"], "source_label_counts": [50, 30], "examples": [...]}` (201)

---

## Detector Registry

### List registered detectors

```
GET /api/detectors/registry
```

→ ```json
{
  "detectors": [
    {
      "id": "abc123",
      "name": "Dog Barks",
      "media_type": "audio",
      "text_query": "dog barking",
      "num_training": 50,
      "loaded": true,
      "detector_loaded": true,
      "autorun": false,
      "last_trained_at": 1234567890.0
    }
  ]
}
```

`name` is the slug used to look up the on-disk labelset file at
`data/detectors/<name>.json`. The MLP is trained on demand from the
labelset and lives only in RAM. `autorun` mirrors whether the
detector's name appears in `autorun_detectors` settings (toggle it with
the route below).

### Register detector

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

→ `{"ok": true, "detector": {...}}` (201)

### Toggle autorun flag

```
PUT /api/detectors/registry/{detector_id}/autorun
```

**Body:** `{"autorun": true}`

→ `{"ok": true, "autorun": true}` - writes the detector's name into
`autorun_detectors` so `/api/auto-detect` and the CLI
`--autodetect` flow pick it up.

### Load / unload detector

```
POST /api/detectors/registry/load
```

**Body:** `{"detector_id": "abc123"}` - pass `null` (or omit the field)
to unload the active detector without loading another one.

→ `{"ok": true, "message": "Loading started", "task_id": "..."}` when
loading; `{"ok": true, "labels_restored": 0, "examples_seeded": 0}`
when unloading.

Loading is async - subscribe to the `detector-loading-tasks` channel
on [`/api/events`](events.md) (SSE) for progress. 404 if the detector
is not in the registry.

### Unload detector

```
POST /api/detectors/registry/{detector_id}/unload
```

→ `{"ok": true}`

### Detector loading tasks (SSE)

Active detector loading tasks are streamed on the `detector-loading-tasks`
channel of [`/api/events`](events.md):

```json
[{"task_id": "...", "name": "...", "status": "loading", "message": "...", "current": 50, "total": 100}]
```

### Cancel detector loading

```
POST /api/detectors/cancel/{task_id}
```

→ `{"ok": true}`

### Delete registered detector

```
DELETE /api/detectors/registry/{detector_id}
```

Also cleans up the on-disk labelset file and clears the autorun flag.

→ `{"ok": true}`

### Rename registered detector

```
PUT /api/detectors/registry/{detector_id}/rename
```

**Body:** `{"name": "New Name"}`

→ `{"ok": true, "name": "New Name"}`
