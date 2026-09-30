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

Free/used/total bytes for the partition holding `DATA_DIR`, plus whether that
free space is running low (see [Headroom](#headroom) below).

→ `{"total": 500107862016, "used": 210000000000, "free": 290107862016, "path": "/app/data", "dataset_bytes": 2147483648, "dataset_bytes_source": "largest", "low": false}`

### RAM usage

```
GET /api/dashboard/ram-usage
```

System RAM total/used/free in bytes, read from `/proc/meminfo` (Linux). `free`
is `MemAvailable`; `used` is `total − free`. (No `path` key, unlike disk usage.)

→ `{"total": 16777216000, "used": 8388608000, "free": 8388608000, "dataset_bytes": 2147483648, "dataset_bytes_source": "largest", "low": false}`

### Headroom

Both probes judge their `free` bytes in **datasets**, not as a fraction of the
machine: a disk that is 99% full may still hold dozens more datasets, and one
that is half empty may not hold one. `dataset_bytes` is the unit, the on-disk
size of the largest registered dataset (its `.pkl` plus same-stem sidecars;
a container is stored uncompressed, so it also stands in for the RAM a load
takes). `dataset_bytes_source` is `largest` when it was measured, or `default`
with a 1 GiB stand-in while no registered dataset has a file on disk.

`low` is `true` when `free` holds fewer than **3** more datasets of
`dataset_bytes`, counting any dataset smaller than 512 MiB as 512 MiB so a
registry of tiny datasets never lets the machine run nearly dry unannounced.
An unreadable RAM probe (`total` of 0, off Linux) is never `low`. The
Dashboard's RAM / Disk bars show on their **Default** setting only while their
probe is `low`.
