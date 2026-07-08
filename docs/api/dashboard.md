# Dashboard

[← Back to API index](../API.md)

Metadata and resource-usage probes for the Dashboard view. For running
detectors against data (multi-dataset Find, Find Label, Auto-Detect), see
[Find, Auto-Detect & Scoring](find.md).

---

### Dataset info

```
GET /api/dashboard/dataset-info
```

Metadata about the currently loaded dataset.

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

**404** if no dataset is loaded.

### Rename dataset

```
PUT /api/dashboard/dataset-rename
```

**Body:** `{"name": "My Custom Name"}`

Sets a custom display name for the currently loaded dataset.

→ `{"success": true, "name": "My Custom Name"}`

**400** if the name is empty after trimming.

### Disk usage

```
GET /api/dashboard/disk-usage
```

Free/used/total bytes for the partition holding `DATA_DIR`.

→ `{"total": 500107862016, "used": 210000000000, "free": 290107862016, "path": "/app/data"}`

### RAM usage

```
GET /api/dashboard/ram-usage
```

System RAM total/used/free in bytes, read from `/proc/meminfo` (Linux). `free`
is `MemAvailable`; `used` is `total − free`. (No `path` key, unlike disk usage.)

→ `{"total": 16777216000, "used": 8388608000, "free": 8388608000}`
