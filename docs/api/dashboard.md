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

```
POST /api/find
```

**Body:** `{"dataset_ids": ["id1", "id2"], "model_ids": ["m1"]}`

→ ```json
{
  "results": [...],
  "datasets": [...],
  "models": [...],
  "multiple_datasets": true,
  "multiple_models": false,
  "total_hits": 42
}
```
