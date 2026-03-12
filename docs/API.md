# HTTP API Reference

VTSearch exposes a REST-style JSON API. All endpoints accept and return JSON
unless otherwise noted. File uploads use `multipart/form-data`.

## Table of Contents

1.  [Conventions](#conventions)
2.  [Authentication](#authentication)
3.  [Static / UI](#static--ui)
4.  [Medias](#medias)
5.  [Sorting](#sorting)
6.  [Votes & Labels](#votes--labels)
7.  [Inclusion & Thresholds](#inclusion--thresholds)
8.  [Labeling Progress](#labeling-progress)
9.  [Diversity Tree](#diversity-tree)
10. [Detectors](#detectors)
11. [Extractors](#extractors)
12. [Localizers](#localizers)
13. [Pre-generated Processors](#pre-generated-processors)
14. [Datasets](#datasets)
15. [Dataset Registry](#dataset-registry)
16. [Exporters](#exporters)
17. [Label Importers](#label-importers)
18. [Processor Importers](#processor-importers)
19. [Settings](#settings)
20. [Trainable Models](#trainable-models)
21. [Model Registry](#model-registry)
22. [Dashboard](#dashboard)
23. [Media Lookup](#media-lookup)
24. [Multi-dataset Find](#multi-dataset-find)

---

## Conventions

| Pattern | Meaning |
|---------|---------|
| `{param}` | URL path parameter |
| **Body** | JSON request body (Content-Type `application/json`) |
| **Form** | `multipart/form-data` |
| `→` | Response body |
| Async endpoints | Return immediately; poll `GET /api/dataset/progress` |

**Common error shape:**

```json
{"error": "Human-readable message"}
```

Status codes follow standard HTTP semantics: 200 OK, 201 Created, 204 No
Content, 400 Bad Request, 404 Not Found, 409 Conflict, 500 Internal Server
Error.

---

## Authentication

### Auth status

```
GET /api/auth/status
```

→ ```json
{
  "provider": "default",
  "user": "default",
  "authenticated": true,
  "login_required": false
}
```

Returns the active login provider name, current user, whether the request
is authenticated, and whether the frontend should show a login screen.
With `DefaultLoginProvider`, every request is authenticated as `"default"`.

---

## Static / UI

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serve the single-page application (`index.html`) |
| GET | `/favicon.ico` | Site favicon (204 if missing) |
| GET | `/favicon-{variant}.ico` | Favicon variant: `smile`, `frown`, or `surprised` (404 for unknown variant, 204 if file missing) |
| GET | `/logo.svg` | Site logo (204 if missing) |

---

## Medias

### List medias

```
GET /api/medias
```

→ JSON array of media metadata objects:

```json
[
  {
    "id": 0,
    "type": "audio",
    "filename": "media_0.wav",
    "md5": "abc123...",
    "custom_metadata": {
      "duration": 5.0,
      "file_size": 160044,
      "category": "sine",
      "frequency": 440
    },
    "origin_name": "media_0.wav",
    "description": "A 440 Hz sine wave"
  }
]
```

Every item contains `id`, `type`, `filename`, `md5`, and `custom_metadata`.
The `custom_metadata` dict holds media-type-specific display fields (e.g.
`duration`/`frequency` for audio, `width`/`height` for images, `word_count`
for text). `origin_name` and `description` are included when present.

### Stream audio

```
GET /api/medias/{media_id}/audio
```

→ `audio/wav` binary stream.
404 if media not found.

### Stream video

```
GET /api/medias/{media_id}/video
```

→ Video binary stream (`video/mp4`, `video/webm`, `video/quicktime`, or
`video/x-msvideo` based on filename extension).
400 if not a video. 404 if not found.

### Stream image

```
GET /api/medias/{media_id}/image
```

→ Image binary stream (`image/jpeg`, `image/png`, `image/gif`, `image/webp`, or
`image/bmp` based on filename extension).
400 if not an image. 404 if not found.

### Get paragraph text

```
GET /api/medias/{media_id}/paragraph
```

→ `{"content": "...", "word_count": 150, "character_count": 900}`
400 if not a paragraph. 404 if not found.

### Generic media endpoint

```
GET /api/medias/{media_id}/media
```

Delegates to the registered media type's handler. Works for all media types.
400 for unsupported type. 404 if not found.

### Vote on a media

```
POST /api/medias/{media_id}/vote
```

**Body:** `{"vote": "good"}` or `{"vote": "bad"}`

Toggle semantics: voting the same direction again removes the vote. Voting the
opposite direction switches sides.

→ `{"ok": true}`
400 for invalid vote value. 404 if not found.

---

## Sorting

### Text sort

```
POST /api/sort
```

**Body:** `{"text": "dog barking"}`

Embeds the text query using the media type's embedding model, then sorts all
medias by cosine similarity. Includes a GMM-based threshold.

→ `{"results": [{"id": 0, "similarity": 0.8234}], "threshold": 0.5123}`

### Text sort progress

```
GET /api/sort/progress
```

→ `{"status": "sorting", "message": "Computing similarities…", "cur": 50, "total": 100}`

Status is `"idle"` or `"sorting"`.

### Learned sort

```
POST /api/learned-sort
```

Trains an MLP on the current good/bad votes and scores all medias. Requires at
least one good and one bad vote.

→ `{"results": [{"id": 0, "score": 0.9234}], "threshold": 0.5123}`

### Example sort (upload)

```
POST /api/example-sort
```

**Form:** `file` — media file to use as the query example.

Embeds the uploaded file and sorts by cosine similarity.

→ `{"results": [{"id": 0, "similarity": 0.8234}], "threshold": 0.5123}`

### Example sort (server file)

```
POST /api/example-sort-server
```

**Body:** `{"filename": "example.wav"}`

Same as example sort but uses a file already on the server in
`data/example_media/`.

→ `{"results": [...], "threshold": 0.5123}`

### List server media files

```
GET /api/server-media-files
```

→ `{"files": [{"name": "example", "filename": "example.wav", "path": "/abs/path", "size_bytes": 160044}]}`

### Label-file sort

```
POST /api/label-file-sort
```

**Form:** `file` — JSON file with a `labels` array. Each entry has `label`
(`"good"` / `"bad"`) and a `path`/`file`/`filename` pointing to an audio file.

Trains an MLP on the labeled files, then scores all loaded medias.

→ `{"results": [...], "threshold": 0.5123, "loaded": 10, "skipped": 2}`

---

## Votes & Labels

### Get votes

```
GET /api/votes
```

→ ```json
{
  "good": [0, 3, 7],
  "bad": [1, 5],
  "click_times": {"0": 1234567890.123},
  "learned_scores": {"0": 0.9234}
}
```

### Clear votes

```
POST /api/votes/clear
```

Clears all good/bad votes without clearing the loaded dataset. Used by the
Label flow to reset votes before importing a model's labelset.

→ `{"ok": true}`

### Text-sort suggestions

```
GET /api/textsort-suggestions
```

→ `{"suggestions": ["dog barking", "cat meowing"]}`

```
POST /api/textsort-suggestions
```

**Body:** `{"text": "dog barking"}`

→ `{"ok": true}`

### Export labels

```
GET /api/labels/export
```

**Query:** `?goods_only=1` — optional, export only good labels.

→ LabelSet JSON with per-element origin and MD5 info:

```json
{
  "labels": [
    {
      "origin": {"importer": "demo", "params": {"name": "esc50"}},
      "origin_name": "dog_bark_001.wav",
      "md5": "abc123...",
      "label": "good"
    }
  ]
}
```

### Import labels

```
POST /api/labels/import
```

**Body:** `{"labels": [{"origin": {...}, "origin_name": "...", "md5": "...", "label": "good"}]}`

Matches by origin+origin_name first, falls back to MD5.

→ `{"applied": 8, "skipped": 2}`

### Fill labels from sort results

```
POST /api/labels/fill-from-sort
```

**Body:**

```json
{
  "sort_results": [{"id": 0, "score": 0.8}],
  "threshold": 0.5,
  "sides": "good",
  "confirm": false
}
```

`sides`: `"good"`, `"bad"`, or `"both"`.

When `confirm` is `false` (dry run):

→ `{"good_count": 15, "bad_count": 10}`

When `confirm` is `true`:

→ `{"good_applied": 15, "bad_applied": 10, "results": {...}}`

---

## Inclusion & Thresholds

### Get / set inclusion

```
GET /api/inclusion
```

→ `{"inclusion": 0}`

```
POST /api/inclusion
```

**Body:** `{"inclusion": 3}`

Value is clamped to the range -10 to +10.

→ `{"inclusion": 3}`

### Get / set safe thresholds

```
GET /api/safe-thresholds
```

→ `{"safe_thresholds": false}`

```
POST /api/safe-thresholds
```

**Body:** `{"safe_thresholds": true}`

→ `{"safe_thresholds": true}`

---

## Labeling Progress

### Analyze progress

```
POST /api/labeling-progress
```

Requires at least one good vote, one bad vote, and label history.

→ Analysis object with progress metrics (structure depends on internal
implementation).

### Labeling status indicators

```
GET /api/labeling-status
```

→ ```json
{
  "smart": {"status": "green"},
  "stable": {"status": "yellow"},
  "span": {"status": "red"}
}
```

Each metric has a `status` of `"red"`, `"yellow"`, or `"green"`.

### Indicator score history

```
GET /api/indicator-score-history
```

**Query:** `?metric=smart` — one of `smart`, `stable`, or `diverse`.

→ `{"metric": "smart", "history": [...]}`

Returns cached per-step data without retraining.

### Compute indicator score history

```
POST /api/eval/train-and-score
```

**Body:** `{"metric": "smart"}` — one of `smart`, `stable`, or `diverse`.

→ `{"error_cost": [...]}` or `{"stability": [...]}` or `{"diversity": [...]}`

### Eval computation progress

```
GET /api/eval/voting-iterations
```

→ `{"progress": 50, "total": 100, "done": false}`

---

## Diversity Tree

### Get next diverse sample

```
GET /api/diversity-tree/next
POST /api/diversity-tree/next
```

POST accepts an optional body with sort scores to influence selection:

**Body:** `{"scores": {"0": 0.9, "1": 0.2}}`

→ `{"id": 42, "diversity_level": 3, "exhausted": false}`

`id` is `null` when the tree is not built or exhausted. `exhausted` is `true`
when every node has been seen.

---

## Detectors

### Export detector (train from votes)

```
POST /api/detector/export
```

Trains an MLP on current votes and exports weights.

→ `{"weights": {...}, "threshold": 0.5}`

### Export detector to server file

```
POST /api/detector/export-server
```

**Body:** `{"name": "my_detector", "overwrite": false}`

→ `{"success": true, "name": "my_detector", "path": "/abs/path.json", "threshold": 0.5, "media_type": "audio"}`

409 if name exists and `overwrite` is `false`.

### List server detector files

```
GET /api/detector/server-files
```

→ `{"files": [{"name": "my_detector", "path": "/abs/path.json", "size_bytes": 1234}]}`

### Get server detector file

```
GET /api/detector/server-files/{name}
```

→ JSON detector data (weights, threshold, etc.)

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

→ `{"success": true, "name": "...", "path": "..."}`

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

### Multi-dataset find

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

---

## Datasets

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

### List importers

```
GET /api/dataset/importers
```

Returns importers excluding built-in ones (pickle, combine_datasets).

→ `{"importers": [{"name": "...", "display_name": "...", "description": "...", "fields": [...]}]}`

```
GET /api/dataset/all-importers
```

Returns all importers including built-in ones.

→ `{"importers": [...]}`

### Available dataset files

```
GET /api/dataset/available-files
```

Lists `.pkl` files in the embeddings directory.

→ `{"files": [{"name": "esc50", "path": "/abs/path/esc50.pkl", "size_mb": 12.3}]}`

### Load dataset

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

### Staging (for combine-datasets flow)

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

---

## Exporters

### List exporters

```
GET /api/exporters
```

→ JSON array of exporter objects, each with `name`, `display_name`,
`description`, and `fields`.

### Run export

```
POST /api/exporters/export
```

**Body:**

```json
{
  "exporter_name": "server_json_file",
  "field_values": {"filepath": "/home/user/results.json"},
  "results": {}
}
```

→ `{"success": true, "message": "...", ...}`

Available built-in exporters: `server_json_file`, `server_csv_file`, `webhook`,
`email_smtp`, `gui`.

---

## Label Importers

### List label importers

```
GET /api/label-importers
```

→ JSON array of label importer objects.

### Run label import

```
POST /api/label-importers/import/{importer_name}
```

**Form or Body:** importer-specific fields.

→ ```json
{
  "applied": 8,
  "skipped": 2,
  "missing_count": 3,
  "missing": [...],
  "message": "Applied 8 label(s), skipped 2. 3 element(s) not found in dataset."
}
```

When `missing_count > 0`, the frontend can call `ingest-missing` to pull those
medias from their origins.

### Ingest missing medias

```
POST /api/label-importers/ingest-missing
```

**Body:** `{"entries": [...]}`

Re-ingests medias from their recorded origins and applies labels.

→ `{"ingested": 3, "applied": 3, "message": "Ingested 3 media(s), applied 3 label(s)."}`

---

## Processor Importers

### List processor importers

```
GET /api/processor-importers
```

→ JSON array of processor importer objects.

### Run processor import

```
POST /api/processor-importers/import/{importer_name}
```

**Form or Body:** importer-specific fields. `name` is required.

Runs the importer and saves the result as an autorun detector.

→ `{"success": true, "name": "...", "media_type": "audio"}`

---

## Settings

### Get all settings

```
GET /api/settings
```

→ ```json
{
  "volume": 1.0,
  "theme": "dark",
  "inclusion": 0,
  "enrich_descriptions": false,
  "safe_thresholds": false,
  "calibrate_count": 2,
  "calibration_fraction": 0.5,
  "audio_playing": true,
  "swipe_animation": true,
  "show_metadata": true,
  "view_mode_left": {},
  "view_mode_right": {},
  "focus_mode_left": {},
  "focus_mode_right": {},
  "grid_columns_left": {},
  "grid_columns_right": {},
  "panel_pct_left": {},
  "panel_pct_right": {},
  "autoload_media_types": [],
  "autoload_media_embedders": [],
  "autorun_processors": [],
  "autopilot_enabled": true,
  "hide_autopilot": false,
  "autopilot_top_greens": 3,
  "autopilot_hard_reds": 4,
  "saved_datasets_dir": "data/saved_datasets",
  "detectors_dir": "data/detectors",
  "trainable_models_dir": "data/trainable_models"
}
```

Per-media-type settings (`view_mode_*`, `focus_mode_*`, `grid_columns_*`,
`panel_pct_*`) use dicts keyed by media type ID (e.g. `{"audio": "list"}`).

### Update settings

```
PUT /api/settings
```

**Body:** partial object with any settings keys to update.

```json
{"volume": 0.5, "theme": "light"}
```

→ Full settings object.

Supported keys: `volume` (number), `theme` (`"dark"` / `"light"` /
`"highviz"`), `inclusion` (int, -10 to +10), `enrich_descriptions` (bool),
`safe_thresholds` (bool), `calibrate_count` (int), `calibration_fraction`
(number), `audio_playing` (bool), `swipe_animation` (bool),
`show_metadata` (bool), `view_mode_left` (dict), `view_mode_right` (dict),
`focus_mode_left` (dict), `focus_mode_right` (dict), `grid_columns_left`
(dict), `grid_columns_right` (dict), `panel_pct_left` (dict),
`panel_pct_right` (dict), `autoload_media_types` (list of strings),
`autoload_media_embedders` (list of strings), `autopilot_enabled` (bool),
`hide_autopilot` (bool), `autopilot_top_greens` (int),
`autopilot_hard_reds` (int), `saved_datasets_dir` (string path),
`detectors_dir` (string path), `trainable_models_dir` (string path).

### Get default settings

```
GET /api/settings/defaults
```

→ Default values for all settings (excluding `autorun_processors`).

### Autorun processors

```
GET /api/settings/autorun-processors
```

→ `{"autorun_processors": [{"processor_name": "...", "processor_importer": "...", "field_values": {...}, "settings_json": "..."}]}`

```
POST /api/settings/autorun-processors
```

**Body:** `{"processor_name": "my_detector", "processor_importer": "server_detector_file", "field_values": {"filepath": "/path/to/detector.json"}}`

→ `{"success": true, "processor_name": "...", "processor_importer": "...", "field_values": {...}, "settings_json": "..."}`

```
DELETE /api/settings/autorun-processors/{name}
```

→ `{"success": true}` or 404.

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

→ `{"ok": true}`

404 if model not found.

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

## Media Lookup

### List embedders

```
GET /api/embedders
```

**Query:** `?media_type=image` — optional, filter by `type_id` or `folder_import_name`.

→ `{"embedders": [{"name": "clip", "display_name": "CLIP", "media_type_id": "image", ...}]}`

### List clippers

```
GET /api/clippers
```

**Query:** `?media_type=audio` — optional, same filtering as embedders.

→ `{"clippers": [{"name": "sound_default", "media_type": "audio", ...}]}`

### List converters

```
GET /api/converters
```

**Query:** `?target=image` and/or `?source=video` — optional, filter by
`type_id` or `folder_import_name`.

→ `{"converters": [{"name": "video2image", "source_type": "video", "target_type": "image", ...}]}`
