# Dashboard & Lookup

[← Back to API index](../API.md)

---

## Dashboard

### Dataset info

```
GET /api/dashboard/dataset-info
```

→ ```json
{
  "name": "ESC-50 Animals",
  "num_medias": 500,
  "num_dupes": 3,
  "media_type": "audio",
  "origin": "demo:esc50",
  "source": {"importer": "demo", "params": {"name": "esc50"}}
}
```

### Rename dataset

```
PUT /api/dashboard/dataset-rename
```

**Body:** `{"name": "My Custom Name"}`

→ `{"success": true, "name": "My Custom Name"}`

---

## Multi-dataset Find

### Check label resolution (pre-flight)

```
POST /api/find/check-labels
```

**Body:** `{"dataset_ids": ["id1", "id2"], "model_ids": ["m1"]}`

Pre-flight check that reports how many trainable-model labels can be resolved
for the given models and datasets. Call this before starting a Find to warn
the user about unresolved labels.

→ ```json
{
  "warnings": [
    {
      "model_name": "Mammals",
      "total_labels": 82,
      "resolved_labels": 60,
      "failed_labels": 22
    }
  ]
}
```

`warnings` only contains entries for models with at least one unresolved label.
An empty `warnings` list means everything is fine.

### Run find

```
POST /api/find
```

**Body:** `{"dataset_ids": ["id1", "id2"], "model_ids": ["m1"]}`

→ ```json
{
  "results": [...],
  "negative_results": [...],
  "datasets": ["ESC-50", "Speech Commands"],
  "models": ["Dog Barks"],
  "media_type": "audio",
  "multiple_datasets": true,
  "multiple_models": false,
  "total_hits": 42
}
```

### Find progress

```
GET /api/find/progress
```

→ ```json
{
  "status": "running",
  "message": "Scoring with \"ModelName\" on \"DatasetName\"...",
  "cur": 150,
  "total": 300,
  "step": 2,
  "total_steps": 3,
  "error": null
}
```

`status` is `"idle"` or `"running"`. `step` / `total_steps` track the
high-level Find phases (prepare models, load data, score).

### Apply labels from model (Find Label)

```
POST /api/find-label
```

**Body:** `{"model_id": "abc123"}` — optionally include `"dataset_id"` to
override the request-scoped dataset context.

Resolves the model from the registry, scores every loaded media using the
model's weights, and applies Good/Bad labels for **all** elements based on
the threshold. If no pre-trained weights are available, trains on-the-fly
from the trainable model's labelset (resolving label origins as needed).

→ ```json
{
  "results": [{"id": 0, "score": 0.9812}, ...],
  "threshold": 0.5,
  "applied": 42,
  "total_scored": 500
}
```

404 if model not found. 400 if `model_id` is missing.
