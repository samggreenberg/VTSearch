# Datasets

[← Back to API index](../API.md)

> Endpoints that act on "the loaded dataset" resolve it via the
> [`X-Dataset-Id` context header](../API.md#context-headers-x-dataset-id--x-detector-id)
> (registry routes take the id in the path instead).

---

## Media Types, Embedders, Clippers & Converters

### Media types

```
GET /api/media-types
```

→
```json
{
  "media_types": [
    {
      "type_id": "audio",
      "name": "Audio",
      "icon": "🔊",
      "folder_import_name": "audio",
      "loops": true,
      "file_extensions": ["*.wav", "*.mp3"]
    }
  ]
}
```

### Embedders

```
GET /api/embedders
```

**Query:** `?media_type=image` (optional): filter by `type_id` or
`folder_import_name`.

→ `{"embedders": [{"name": "siglip", "display_name": "SigLIP", "model_id": "google/siglip-base-patch16-224", "media_type_id": "image", ...}]}`

### Embed on demand

```
POST /api/embed
```

Embed a single media file or text snippet with a chosen embedder, without
loading a dataset. Two input modes share the endpoint; the embedder
declares its own `media_type_id`, so the caller does not pass a media
type separately.

**Media upload (`multipart/form-data`)** - form fields:

| Field | Description |
|-------|-------------|
| `embedder` | Required. Embedder name from `GET /api/embedders`. |
| `file` | Required. Binary upload. The extension is checked against the embedder's `media_type_id` before any model load. |

**Text (`application/json`)** - request body:

```json
{"embedder": "e5", "text": "a cat on a mat"}
```

Calls `embed_text(...)`; only embedders whose `supports_text` is `true`
accept this mode (image/audio cross-modal embedders like CLIP, SigLIP, CLAP
support it; vision-only embedders like DINOv3 do not).

→
```json
{
  "embedding": [0.123, -0.045, ...],
  "dim": 512,
  "norm": 1.0,
  "embedder": "clip",
  "media_type": "image"
}
```

**Errors:**

| Status | Cause |
|--------|-------|
| `400` | Missing `embedder`; missing `file` (multipart) or `text` (JSON); text sent to an embedder where `supports_text == false`; file extension does not match the embedder's media type; embedder returned no vector for the upload. |
| `404` | Unknown embedder name. Response body lists every registered embedder's `name`. |
| `500` | Model load failure, or text embedding returned `None`. |

### Clippers

```
GET /api/clippers
```

**Query:** `?media_type=audio` (optional): filter by `type_id` or
`folder_import_name`.

→ `{"clippers": [{"name": "sound_default", "media_type": "audio", "display_name": "Sound Default", "description": "Import each audio file as-is, without splitting.", ...}]}`

Each clipper object includes `name`, `display_name`, `media_type`, and
an optional `description` (short tooltip text). Clippers with
configurable settings also include `parameters` and
`creation_questions` arrays, where each parameter has `key`, `label`,
`type`, `default`, and an optional `description` for hover tooltips.

### Cleaners

```
GET /api/cleaners
```

**Query:** `?media_type=image` (optional): filter by `type_id` or
`folder_import_name`.

→ `{"cleaners": [{"name": "image_exif_orient", "media_type": "image", "display_name": "EXIF Orientation", "default_enabled": true, "description": "Rotate photos to their EXIF display orientation…"}]}`

*Cleaners* are the optional 1-to-1 cleanup gates every imported item of a media
type can pass through before it is embedded (see
[EXTENDING-media.md § Adding a Media Cleaner](../EXTENDING-media.md#adding-a-media-cleaner)).
They are listed separately from clippers because the UI treats them
differently: a clipper is a radio choice, cleaners are a checkbox list where
any combination can be enabled. An entry may also carry `parameters` (same
descriptor shape as a clipper's); the import form renders those inputs beneath
the cleaner's checkbox once it is ticked, and the chosen values ride along in
the `cleaners` field's per-entry `params`.

Each cleaner object carries the same fields as a clipper (`name`,
`display_name`, `media_type`, optional `description` / `parameters` /
`creation_questions`) plus `default_enabled`, which tells the import form
whether to pre-check the box.

To enable cleanup gates for an import, send a `cleaners` field alongside
`clipper` / `clipper_params` — a JSON array of names or of
`{"name": ..., "params": {...}}` objects. Order is ignored: cleaners always run
**last**, after the whole clipper/converter chain, on the units that will
actually be embedded. The field is accepted by every import entry point
(`POST /api/dataset/import/<importer>`, `/api/dataset/load-demo`,
`/api/dataset/import-local-folder`, `/api/dataset/import-local-files`).

### Converters

```
GET /api/converters
```

**Query (mutually exclusive):** `?source=video` or `?target=image`: filter by
`type_id` or `folder_import_name`. Omit both to list all converters.

→ `{"converters": [{"name": "video2image", "source_type": "video", "target_type": "image", ...}]}`

---

## Dataset Status & Progress

### Dataset status

```
GET /api/dataset/status
```

→ `{"loaded": true, "num_medias": 500, "has_votes": true, "media_type": "audio", "display_name": "ESC-50", "num_dupes": 3}`

### Dataset progress (SSE)

Progress for dataset operations is streamed through the unified
[`/api/events`](events.md) Server-Sent Events endpoint. Two channels
carry dataset state:

- `dataset`: the singleton dataset progress tracker (used by staging,
  embedding, and other one-at-a-time operations):

  ```json
  {"status": "loading", "message": "Embedding medias…", "current": 50, "total": 500}
  ```

- `loading-tasks`: array of all active dataset loading tasks:

  ```json
  [{"task_id": "task_abc", "name": "ESC-50", "status": "loading", "message": "...", "current": 50, "total": 500}]
  ```

Connect with `new EventSource('/api/events')` and listen for the
`dataset` and `loading-tasks` events. The first frame on each channel
is the current snapshot; no separate bootstrap call is needed.

### Cancel loading

```
POST /api/dataset/cancel
```

Cancels all active loading tasks and the legacy global tracker, then waits
briefly (~2 s) for one of them to act on the flag.

Cancellation is **cooperative**: the endpoint sets an event that a running
worker has to observe. `ok` therefore reports whether the cancel actually
reached something, not merely that the flag was set — a flag set with no
worker left to see it stops nothing.

→ `200 {"ok": true, "message": "...", "targets": [...], "acknowledged": [...], "pending": [...], "unresponsive": [...]}`

| Field | Meaning |
|---|---|
| `targets` | Everything that claimed to be working when the cancel arrived (loading-task ids, plus `"dataset_progress"` for the global tracker). |
| `acknowledged` | Targets that reached a terminal state within the grace period. |
| `pending` | Targets still running, whose live worker will observe the flag. |
| `unresponsive` | Targets whose progress claimed work no live thread was doing — stale trackers, now cleared. Not operations that were stopped. |

`ok` is `true` when at least one target acknowledged or is pending. When the
cancel reached nothing — no operation was running, or every target's progress
was stale — the response is `409` with `ok: false`, and the stale progress it
found is cleared on the way out.

```
POST /api/dataset/cancel/{task_id}
```

Cancels a specific loading task, with the same response shape and the same
`409` contract when the task was not running or its worker is gone.

→ `200 {"ok": true, ...}`, `409 {"ok": false, ...}`, or `{"error": "Task not found"}` (404)

---

## Importers

### List importers

```
GET /api/dataset/all-importers
```

Returns all registered importers including built-in ones (pickle, combine_datasets, demo).

→ `{"importers": [{"name": "...", "display_name": "...", "description": "...", "fields": [...]}]}`

```
GET /api/dataset/importers
```

Returns only importers with `ui_mode == "form"` (excludes pickle, combine_datasets, demo, and any other importers with non-form UI modes). Used by the frontend's generic form-based import dialog.

→ `{"importers": [...]}`

### Available dataset files

```
GET /api/dataset/available-files
```

Lists `.pkl` files in the embeddings directory.

→ `{"files": [{"name": "esc50", "path": "/abs/path/esc50.pkl", "size_mb": 12.3}]}`

### Importer field options (dynamic dropdowns)

```
POST /api/dataset/import/{importer_name}/options
```

**Body:** `{"field_key": "collection", "values": {...}}` (`values` is the
current form snapshot, optional).

Calls the importer's `get_field_options(field_key, values)` and returns
dropdown options for a [dynamic-options field](../EXTENDING-plugins.md#dynamic-field-options).

→ `{"options": [{"value": "a", "label": "A"}, {"value": "b", "label": "b"}]}`
(each option carries a `value` to submit and a `label` to display; the two
coincide for plain-string options and differ for `(value, label)` tuples).

400 (unknown / non-dynamic field), 404 (unknown importer), 501 (not
implemented), 502 (remote error).

### Importer suggested dataset name

```
POST /api/dataset/import/{importer_name}/suggested-name
```

**Body:** `{"values": {...}}` (the current form snapshot, optional).

Calls the importer's `default_display_name(values)` and returns the name it
would give a dataset built from those values, so the import modal can
prefill its Dataset Name box — including a label the importer resolved from
an opaque selection. See
[Naming the imported dataset](../EXTENDING-plugins.md#naming-the-imported-dataset).

Any `dataset_name` in `values` is dropped before the importer sees it: the
route reports what the importer *would* pick, and the caller decides
whether to overwrite what the user typed.

→ `{"dataset_name": "Q1 Field Survey"}`

404 (unknown importer), 502 (the importer raised).

### Detect media type

```
GET /api/dataset/detect-media-type
```

**Query:** `source` (default `"folder"`), `path` (default `""`), `recursive`
(default `true`), `limit` (default 50, clamped to 1–500).

Samples a folder's files by extension to pre-fill the import modal's
media-type dropdown.

→
```json
{
  "sample_size": 50,
  "counts_by_type": {"audio": 48, "image": 2},
  "extensions": {".wav": 48, ".png": 2},
  "dominant": "audio",
  "truncated": false
}
```

400 (path escapes root), 404 (source/dir not found).

---

## Loading Datasets

**From uploaded pickle:**

```
POST /api/dataset/load-file
```

**Form:** `file`: `.pkl` file.

→ `{"ok": true, "message": "Loading started"}`

**From folder:**

```
POST /api/dataset/load-folder
```

**Body:** `{"path": "/data/sounds", "media_type": "audio"}`

→ `{"ok": true, "message": "Loading started"}`

**From demo:**

```
POST /api/dataset/load-demo
```

**Body:** `{"name": "esc50_animals"}`

Optional fields: `embedder`, `clipper`, `clipper_params`, `converter`,
`dataset_name`, `build_projection`. When `clipper` names a real (non-default)
clipper, every loaded media is split into sub-clips at load time and the clips
are re-embedded; `clipper_params` (e.g. `{"duration": 5.0}`) overrides the
clipper's defaults. The pre-selected default clipper for a media type is a
no-op. Clipped clips inherit their parent media's category. Example:
`{"name": "tut_sound_events_2017_a", "clipper": "sound_tiling", "clipper_params": {"duration": 5.0}}`.

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

**From a browser folder upload:**

```
POST /api/dataset/import-local-folder
```

**Form (`multipart/form-data`):** `files` (repeated — one file field per file,
each multipart filename set to its `webkitRelativePath`), `media_type`
(required); optional `dataset_name` (default `"Local folder upload"`),
`embedder`, `clipper_params` (JSON string), `build_projection`,
`merge_near_duplicates`, and clipper-chain fields. Streams the uploaded folder
to a server temp dir and runs the `server_folder` importer in the background.

→ `{"ok": true, "message": "...", "task_id": "..."}`

**From a browser paths-file upload:**

```
POST /api/dataset/import-local-files
```

**Form (`multipart/form-data`):** `paths_file` (a single `.txt` / `.list` /
`.npz` of media paths, required), `media_type` (required); optional
`dataset_name` (default `"Local files upload"`), `embedder`, `clipper`,
`clipper_params` (JSON string), `source_specs`. Runs the `server_files`
importer in the background. Same response shape as `import-local-folder`.

→ `{"ok": true, "message": "...", "task_id": "..."}`

All load endpoints are async; subscribe to the `dataset` and
`loading-tasks` channels on [`/api/events`](events.md) (SSE) for progress.

### Demo datasets

```
GET /api/dataset/demo-list
```

→
```json
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

### Demo categories

```
GET /api/dataset/demo-categories/{name}
```

Lists the categories within a specific demo dataset.

→ `{"categories": ["dog", "cat", "bird", "traffic"]}`
404 if the demo name is not recognized.

---

## File Browsing

### Browse media files

```
GET /api/browse-media-files
```

**Query:**
- `source` (required): `"demo:<name>"` (a demo dataset) or `"folder"` (the
  configured `saved_datasets_dir`).
- `path` (optional): relative sub-path within the root (default `""`).

Lists files and subdirectories within an allowed root, filtered to only
media files with recognized extensions.

→
```json
{
  "directories": [{"name": "dog", "path": "dog", "modified_at": "2025-03-31T10:15:00"}],
  "files": [{"name": "bark.wav", "path": "dog/bark.wav", "size_bytes": 12345, "modified_at": "2025-03-31T10:15:00"}],
  "root_path": "/absolute/path/to/root"
}
```

### Select browsed file

```
POST /api/browse-media-files/select
```

**Body:** `{"source": "demo:esc50_s", "path": "dog/1-100032-A-0.wav"}`

Copies a file from a browse source into the user's `example_media/` directory
with a unique prefix to avoid collisions.

→ `{"filename": "abc123_bark.wav", "original_name": "bark.wav"}` (201)

---

## Staging (for combine-datasets flow)

```
POST /api/dataset/stage-file
```

**Form:** `file` - `.pkl` file.

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

### Promote a selection to a new dataset

```
POST /api/dataset/promote
```

**Body:** `{"name": "My subset", "media_ids": [0, 3, 7]}` — ids from the active
dataset (both fields required, non-empty).

Snapshots the selected media (preserving origins and embeddings) into a brand-
new saved, registered dataset. The snapshot happens synchronously (a bad
request still 400s at request time); the coverage-atlas build, pickle write,
and registry insert run in a background task reported on the `loading-tasks`
SSE channel. The finished task's `dataset_id` association carries the new
dataset's id.

→ `{"ok": true, "message": "Promoting to dataset...", "task_id": "_promote_ab12cd34"}`

---

## Export & Clear

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

→
```json
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

### Dataset statistics

```
GET /api/datasets/registry/{dataset_id}/stats
```

Returns registry and ingest statistics for a registered dataset. The
response is a superset of the Dashboard grid's row (`name`, `media_type`,
`num_items`, `created_at`, `expires_at`, `created_by`, `readers`), so the
Stats window can show everything the grid does while it covers the grid up.

→
```json
{
  "name": "Field recordings",
  "media_type": "audio",
  "num_items": 1250,
  "num_dupes": 45,
  "file_type_counts": {"wav": 800, "mp3": 450},
  "created_at": 1743415500.0,
  "expires_at": null,
  "created_by": "alice",
  "readers": ["bob"],
  "ingest_started_at": 1743413700.0,
  "ingest_finished_at": 1743415500.0,
  "origin": "server_folder",
  "source": {"importer": "server_folder", "params": {"path": "/data/sounds"}},
  "clipper": "5 seconds",
  "embedder": "clap"
}
```

`expires_at` is `null` when the dataset never ages off.

`file_type_counts` keys are file types, not necessarily filename extensions:
each item is typed by its filename extension when it has one, and otherwise by
the format sniffed from its bytes, so a service importer that names items after
an opaque content id still reports `{"jpg": 437}` instead of one useless bucket.
Items that no signal could type land in a parenthesised `"(unknown)"` key (the
parenthesis is what tells a sentinel from a real extension). When the stored
histogram is uninformative — every item unknown — and the dataset happens to be
loaded, the endpoint recounts it from the in-memory medias and writes the repair
back to the registry.

404 if the dataset does not exist.

### Dataset duplicates

```
GET /api/datasets/registry/{dataset_id}/duplicates
```

Returns the collapsed duplicate sets of a **loaded** dataset, expanded to
their full membership so the caller can see which items were collapsed
together and where each one came from. Each set corresponds to one
`dupe_set` representative in the dataset's in-memory context; exact-dupe
members share the representative's MD5, near-dupe members keep their own.

→
```json
{
  "duplicate_sets": [
    {
      "name": "a.wav",
      "members": [
        {"md5": "abc123", "filename": "a.wav", "category": "dogs",
         "origin_name": "a.wav", "importer": "server_folder"},
        {"md5": "abc123", "filename": "b.wav", "category": "pets",
         "origin_name": "b.wav", "importer": "http_archive"}
      ]
    }
  ]
}
```

400 if the dataset isn't loaded (duplicate provenance lives only in memory);
403 if access is denied; 404 if the dataset does not exist.

### Set dataset readers

```
PUT /api/datasets/registry/{dataset_id}/readers
```

**Body:** `{"readers": ["user1", "user2"]}`

Sets which users can access a dataset (multi-user deployments). Only the
dataset owner or an admin can modify readers.

→ `{"ok": true, "readers": ["user1", "user2"]}`

### Preload dataset embedder

```
POST /api/datasets/registry/{dataset_id}/preload-embedder
```

Warms the dataset's embedder in a background daemon thread (idempotent) so it's
ready before training. No body.

→ `{"ok": true, "embedder": "clap"}` (`embedder` is `""` if the dataset has none)

403 if access is denied; 404 if the dataset does not exist.

### Build coverage atlas

```
POST /api/datasets/registry/{dataset_id}/coverage-atlas
```

Kicks off a cancellable background build of the coverage atlas for an
already-loaded dataset. Progress streams on the `loading-tasks` channel of
[`GET /api/events`](events.md#task-object-shape-loading-tasks--detector-loading-tasks),
under the returned `task_id`. No body.

→ `{"ok": true, "message": "...", "task_id": "_atlas_abc12345"}`

400 if the dataset isn't loaded; 403 if access is denied; 404 if it doesn't exist.

### Domain-shift report

```
GET /api/datasets/registry/{dataset_id}/domain-shift
```

Reports how typical the **active** dataset's items (the `X-Dataset-Id`
header) look under `{dataset_id}`'s coverage atlas — `{dataset_id}` is the
*reference*, i.e. the dataset a detector was trained on. Use it before
trusting a detector trained on the reference against the active dataset.

→ `{"reference_dataset_id": "…", "n_items": 40000, "alpha": 0.05,
"frac_atypical": 0.31, "expected_atypical": 0.05, "z_score": 24.1,
"median_pvalue": 0.18, "shifted": true}`

`frac_atypical` is the fraction of active-dataset items whose calibrated
typicality p-value falls below `alpha` — roughly the shifted proportion
(it stays near `expected_atypical` when there is no shift). `shifted` is
the headline verdict (statistically clear **and** practically large
excess).

400 if either dataset isn't loaded, the reference has no coverage atlas,
the two datasets use different embedders, or the active dataset equals the
reference; 403 if access is denied; 404 if the reference doesn't exist.
