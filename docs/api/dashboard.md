# Dashboard

[← Back to API index](../API.md)

Resource-usage probes for the Dashboard view. Dataset metadata and
renaming live on the registry endpoints in [Datasets](datasets.md); for
running detectors against data (multi-dataset Find, Find Label,
Auto-Detect), see [Find, Auto-Detect & Scoring](find.md).

---

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
