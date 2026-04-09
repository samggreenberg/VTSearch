# Detectors & Processors

[← Back to API index](../API.md)

---

## Detectors

### Export detector to server file

```
POST /api/detector/export-server
```

**Body:** `{"name": "my_detector", "overwrite": false}`

→ `{"success": true, "name": "my_detector", "media_type": "audio"}`

409 if name exists and `overwrite` is `false`.

### List server detector files

```
GET /api/detector/server-files
```

→ `{"files": [{"name": "my_detector", "filename": "my_detector.json", "size_bytes": 1234}]}`

### Get server detector file

```
GET /api/detector/server-files/{name}
```

→ JSON detector data (origins, inclusion, media type, name)

### Score with detector

```
POST /api/detector-sort
```

**Body:** `{"detector": {"weights": {...}, "threshold": 0.5}}`

→ `{"results": [{"id": 0, "score": 0.92}], "threshold": 0.5}`

### Autorun detectors

```
GET /api/autorun-detectors
```

→ `{"detectors": [...]}`

```
POST /api/autorun-detectors
```

**Body:** `{"name": "bark_detector", "media_type": "audio"}` (required)

Optional fields: `weights` (dict), `threshold` (number), `autodetect`
(bool, default false), `examples` (list), `num_labels` (int, default 0).

→ `{"success": true, "name": "bark_detector"}`

```
DELETE /api/autorun-detectors/{name}
```

→ `{"success": true}` or 404.

```
PUT /api/autorun-detectors/{name}/rename
```

**Body:** `{"new_name": "new_name"}`

→ `{"success": true, "new_name": "new_name"}`

```
PUT /api/autorun-detectors/{name}/autodetect
```

**Body:** `{"autodetect": true}`

→ `{"success": true, "autodetect": true}`

```
GET /api/autorun-detectors/{name}/export
```

→ `{"weights": {...}, "threshold": 0.5, "media_type": "audio", "name": "..."}`

```
POST /api/autorun-detectors/{name}/export-server
```

**Body:** `{"filename": "detector.json", "overwrite": false}`

→ `{"success": true, "name": "..."}`

```
GET /api/autorun-detectors/{name}/examples
```

→ `{"name": "...", "examples": [...]}`

```
PUT /api/autorun-detectors/{name}/examples
```

**Body:** `{"examples": [{"type": "text", "value": "dog barking"}]}`

→ `{"success": true, "name": "...", "examples": [...]}`

### Import detector from file

```
POST /api/autorun-detectors/import-pkl
```

**Form:** `file` — JSON detector file. Optional form field: `name`.

→ `{"success": true, "name": "...", "media_type": "audio"}`

### Import detector from labeled files

```
POST /api/autorun-detectors/import-labels
```

**Form:** `file` — JSON file with labeled audio paths. Optional: `name`,
`media_type`.

→ `{"success": true, "name": "...", "media_type": "audio", "loaded": 50, "skipped": 3}`

### Train detector from label importer

```
POST /api/autorun-detectors/from-label-import/{importer_name}
```

**Form:** label importer fields + `name`.

→ `{"success": true, "name": "...", "media_type": "audio", "loaded": 50, "skipped": 3}`

### Run auto-detect

```
POST /api/auto-detect
```

**Body (optional):** `{"detector_name": "bark_detector"}`

Runs all autorun detectors that match the current media type (or a specific one
if `detector_name` is provided).

→ ```json
{
  "media_type": "audio",
  "detectors_run": 2,
  "results": {
    "bark_detector": {
      "hits": [...],
      "negative_hits": [...],
      "threshold": 0.5
    }
  }
}
```

---

## Extractors

```
GET /api/autorun-extractors
```

→ `{"extractors": [...]}`

```
POST /api/autorun-extractors
```

**Body:** `{"name": "...", "extractor_type": "...", "media_type": "...", "config": {...}}`

→ `{"success": true, "name": "..."}`

```
DELETE /api/autorun-extractors/{name}
```

→ `{"success": true}` or 404.

```
PUT /api/autorun-extractors/{name}/rename
```

**Body:** `{"new_name": "..."}`

→ `{"success": true, "new_name": "..."}`

### Run extractor

```
POST /api/extract
```

**Body:** `{"name": "...", "extractor_type": "...", "config": {...}}`

→ `{"extractor_name": "...", "media_type": "...", "total_medias_with_hits": 5, "results": [...]}`

### Run all autorun extractors

```
POST /api/auto-extract
```

→ `{"media_type": "...", "extractors_run": 2, "results": {"extractor_name": {...}}}`

---

## Localizers

```
GET /api/autorun-localizers
```

→ `{"localizers": [...]}`

```
POST /api/autorun-localizers
```

**Body:** `{"name": "...", "localizer_type": "...", "media_type": "...", "config": {...}}`

→ `{"success": true, "name": "..."}`

```
DELETE /api/autorun-localizers/{name}
```

→ `{"success": true}` or 404.

```
PUT /api/autorun-localizers/{name}/rename
```

**Body:** `{"new_name": "..."}`

→ `{"success": true, "new_name": "..."}`

### Run localizer

```
POST /api/localize
```

**Body:** `{"name": "...", "localizer_type": "...", "config": {...}}`

→ `{"localizer_name": "...", "media_type": "...", "total_medias_with_hits": 3, "results": [...]}`

### Run all autorun localizers

```
POST /api/auto-localize
```

→ `{"media_type": "...", "localizers_run": 1, "results": {"localizer_name": {...}}}`

---

## Pre-generated Processors

### List available pregen processors

```
GET /api/pregen-processors
```

→ `{"processors": [...]}`

Lists predefined processor recipes (OCR, Speech, Face detection, etc.).

### Add all pregen processors

```
POST /api/pregen-processors/add
```

→ `{"success": true, "added": ["OCR", "Speech"]}`

Registers all predefined processors as autorun entries.
