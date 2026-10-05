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

→ `{"cleaners": [{"name": "image_exif_orient", "media_type": "image", "display_name": "EXIF Orientation", "default_enabled": false, "description": "Rewrite photos upright in their stored bytes…"}]}`

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

With no dataset loaded: `loaded: false`, counts `0`, `media_type` and
`display_name` `null`.

### Dataset progress (SSE)

Every dataset load, import, staging, combine, promote, and atlas build runs as
a **task** with its own `task_id` (returned by the endpoint that started it),
and progress streams on the `loading-tasks` channel of the unified
[`/api/events`](events.md) Server-Sent Events endpoint — an array of every
active task:

```json
[{"task_id": "task_abc", "name": "ESC-50", "status": "loading", "message": "...", "current": 50, "total": 500}]
```

See [events.md](events.md#task-object-shape-loading-tasks--detector-loading-tasks)
for the full task shape. The first frame is the current snapshot; no separate
bootstrap call is needed. (There is no process-wide `dataset` channel any
more; work that isn't bound to a task reports nowhere.)

### Cancel loading

```
POST /api/dataset/cancel
```

Cancels every active loading task, then waits briefly (~2 s) for one of them
to act on the flag.

Cancellation is **cooperative**: the endpoint sets an event that a running
worker has to observe. `ok` therefore reports whether the cancel actually
reached something, not merely that the flag was set — a flag set with no
worker left to see it stops nothing.

→ `200 {"ok": true, "message": "...", "targets": [...], "acknowledged": [...], "pending": [...], "unresponsive": [...]}`

| Field | Meaning |
|---|---|
| `targets` | The loading-task ids that claimed to be working when the cancel arrived. |
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

→ `200 {"ok": true, ...}`, `409 {"ok": false, ...}`, or `404 {"message": "Task not found", ...}`

---

## Importers

### List importers

```
GET /api/dataset/all-importers
```

Returns all registered importers including built-in ones (pickle, combine_datasets, demo),
plus the picker-tab declarations the Add-Dataset modal groups them under.

→ `{"importers": [{"name": "...", "display_name": "...", "description": "...", "fields": [...], "ui_mode": "form", "category": "...", ...}], "tabs": [{"id": "...", "label": "...", "icon": "...", "order": 0}]}`

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

→ `{"ok": true, "message": "Loading started", "task_id": "..."}`

**From demo:**

```
POST /api/dataset/load-demo
```

**Body:** `{"name": "esc50_animals"}`

Optional fields: `embedder` (or `embedders`, a list of create-time picks —
one per embedder kind — producing a multi-embedder dataset), `clipper`,
`clipper_params`, `converter`, `cleaners`, `dataset_name`, and the
string flags `build_projection` / `merge_near_duplicates` (`"true"` /
`"false"`, default `"false"`). When `clipper` names a real (non-default)
clipper, every loaded media is split into sub-clips at load time and the clips
are re-embedded; `clipper_params` (e.g. `{"duration": 5.0}`) overrides the
clipper's defaults. The pre-selected default clipper for a media type is a
no-op. Clipped clips inherit their parent media's category. Example:
`{"name": "tut_sound_events_2017_a", "clipper": "sound_tiling", "clipper_params": {"duration": 5.0}}`.

→ `{"ok": true, "message": "Loading started", "task_id": "..."}`

**From importer:**

```
POST /api/dataset/import/{importer_name}
```

**Form or Body:** the importer's declared `fields` (from
`GET /api/dataset/all-importers`), plus the shared load options (`embedder`,
`clipper`, `clipper_params`, `cleaners`, `dataset_name`, …).

→ `{"ok": true, "message": "Loading started", "task_id": "..."}`

**From source origin:**

```
POST /api/dataset/load-source
```

**Body:** `{"source": {"importer": "demo", "params": {"name": "esc50"}}}`

Re-runs a recorded origin (e.g. a registry entry's `source`).

→ `{"ok": true, "message": "Loading started", "task_id": "..."}`

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

All load endpoints are async: they return as soon as the task is started, and
the returned `task_id` names it on the `loading-tasks` channel of
[`/api/events`](events.md) (SSE). `task_id` can be `""` in the rare case no task
was registered. Cancel with `POST /api/dataset/cancel/{task_id}`.

**AutoRun after import.** `load-file`, `load-demo`, `import/{importer_name}`,
`import-local-folder` and `import-local-files` accept an optional `autorun`
flag (`"true"` / `"false"`): whether to run the caller's AutoRun detectors on
the dataset once it is saved. A sent flag is also remembered as the caller's
`autorun_on_import` [setting](settings.md), which decides an import that
sends none (default `true`). The run starts only after the import finished
successfully, as its own task on the `loading-tasks` channel keyed to the new
dataset (see [Run AutoRun](#run-autorun-on-a-registered-dataset)). When none
of the caller's AutoRun detectors applies (another media type, or an embedder
type the dataset lacks) nothing runs; if they have any AutoRun detectors at
all, a row that is already idle reports it, its `autorun.skipped` holding the
reason.

### Demo datasets

```
GET /api/dataset/demo-list
```

**Query (optional):** `embedder`, `clipper`, `converter` — the choices the
user is about to load with; each entry's `status` is computed against them
(a cached pickle built with a different embedder/clipper is not `ready`).

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
      "num_categories": 5,
      "pkl_embedder": "clap",
      "pkl_clipper": "",
      "available_converters": []
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
  "root_path": "/absolute/path/to/root",
  "default_path": ""
}
```

`default_path` (optional) is the sub-path the picker should open at.

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

**Form or Body:** the importer's fields, as for `POST /api/dataset/import/{importer_name}`.

→ `{"ok": true, "message": "Staging started", "task_id": "..."}`

```
POST /api/dataset/stage-demo/{name}
```

**Body (optional):** `{"converter": "...", "dataset_name": "..."}`

→ `{"ok": true, "message": "Staging demo dataset...", "task_id": "..."}`

Staging tasks report on the `loading-tasks` channel; the finished task's
`staging_result` carries the staged file (same shape as `stage-file`'s
response).

```
DELETE /api/dataset/staging
```

→ `{"ok": true}`

### Combine datasets

```
POST /api/dataset/combine
```

**Body:** `{"datasets": ["/path/to/a.pkl", "/path/to/b.pkl"], "name": "Merged", "resolutions": {}}`

Requires at least two paths. When the sources bind conflicting embedders of
the same kind (say one `siglip` and one `clip` dataset, both semantic), each
conflict must be settled in `resolutions`, keyed by embedder type:
`{"action": "reembed", "embedder": "siglip"}` re-embeds every source to that
embedder, `{"action": "drop"}` leaves that embedder type out. An unresolved
conflict is a 400 rather than a silently mixed vector space.

→ `{"ok": true, "message": "Combining datasets...", "task_id": "..."}`

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

Entries also carry provenance and embedder fields (`created_by`, `readers`,
`expires_at`, `embedder`, `bound_embedders`, `embedders_by_type`, `clipper`,
`num_dupes`, `file_type_counts`, …); see the `DatasetsRegistryListResponse` schema in
the spec for the full list.

### Load registered dataset

```
POST /api/datasets/registry/{dataset_id}/load
```

→ `{"ok": true, "message": "Loading started", "task_id": "..."}`

403 if access is denied; 404 if the dataset or its saved pickle is missing.

### Run AutoRun on a registered dataset

```
POST /api/datasets/registry/{dataset_id}/autorun
Content-Type: application/json

{"detector_ids": ["<detector id>", "..."]}
```

→ `{"ok": true, "message": "AutoRun started", "task_id": "_autorun_…"}`

Runs detectors that apply to the (loaded) dataset in the background. The body
is optional. Without it (or with `detector_ids` omitted or `null`), the run
uses the caller's AutoRun detectors - the Dashboard's ⋯ **Run AutoRun**. With
`detector_ids`, it uses exactly the detectors those registry ids name, drafts
included, and leaves the caller's AutoRun list alone - the Dashboard's big
**AutoRun** button, which sends the ticked detectors once per ticked dataset.
Detectors of another media type, or of an embedder type the dataset lacks, are
skipped as for the AutoRun list. The task reports on the
`loading-tasks` channel with the dataset's `dataset_id`, so it renders on the
dataset's row, and carries an `autorun` block: `{run_id, owner, trigger,
dataset_id, dataset_name}`, plus `detectors_run`, `total_hits`,
`missing_detectors` and `auto_export` once it finishes. The results go to the
caller's Auto-Find exporter when one is set, and are served by
[`GET /api/autorun/runs/{run_id}`](find.md#autorun-results). Cancel with
`POST /api/dataset/cancel/{task_id}`.

400 when none of the detectors to run applies (wrong media type, or an
embedder type the dataset lacks), or `detector_ids` is empty; 403 if access to
the dataset is denied; 404 if the dataset is unknown, or a `detector_ids` entry
names no detector the caller can access; 409 if the dataset is not loaded; 422
if `detector_ids` is not a list of strings.

### Unload registered dataset

```
POST /api/datasets/registry/{dataset_id}/unload
```

→ `{"ok": true}`

400 if not loaded; 403 if the caller isn't the creator.

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

`frac_atypical` is the fraction of active-dataset items whose typicality
score falls below `alpha` — roughly the shifted proportion (it stays near
`expected_atypical` when there is no shift). `shifted` is the headline
verdict (statistically clear **and** practically large excess).

**Not supported for patch embedders** (`dinov3_patch`, `dinov2_patch`,
`eupe_patch`), which are refused with 400. In a patch space the guard does
not separate in-domain from out-of-domain data: on `dinov3_patch` it fires
on 80 % of the reference's own held-out data against 93 % for a different
corpus. See [`docs/ML.md`](../ML.md#how-good-are-these-p-values) for what
the scores are and are not.

400 if either dataset isn't loaded, the reference has no coverage atlas,
the two datasets use different embedders, the reference uses a patch
embedder, or the active dataset equals the reference; 403 if access is
denied; 404 if the reference doesn't exist.

---

## VTSBrowse projection

The Browse view's 2-D map of the **active** dataset (`X-Dataset-Id`): a UMAP
layout binned into a multi-level tile pyramid. The bin shape (squares for
image/video/document, hexagons otherwise) is fixed by the media type and never
sent by the client; `meta` reports it as `bin_shape`. Every read endpoint takes
`?subset=true` to address the ephemeral subset layout instead of the
full-dataset one. Nothing here persists vectors: the layout is derived from the
dataset's own embeddings (see `vtscore.projection.service`).

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/projection/build` | Body (optional) `{"ids": [...], "force": false}`. Returns at once: `{"status": "ready", "projection_id"}` when a layout is cached or persisted, otherwise starts the background fit and returns `{"status": "building", "job_id"}`; poll `meta` until ready. `ids` fits a subset layout over just those items (e.g. a Find run's positives); `force` discards the existing layout and re-fits ("Re-project"). 409 if the dataset is empty or has no embeddings. |
| GET | `/api/projection/meta` | Build status (`status`, plus `current` / `total` / `step` / `overall` / `eta_seconds` / `error` while building) and, once ready, `projection_id`, `bounds`, `levels` (`{level, n_cells, radius}`), `base_radius`, `tile_span`, `point_count`, `media_type`, `bin_shape`, `has_labels`, `content_version`. |
| GET | `/api/projection/tiles/{level}/{tx}/{ty}` | One tile's cells: `{"level", "tx", "ty", "cells": [{q, r, cx, cy, count, rep_id, member_ids?}]}`. Tile coordinates may be negative. Immutable for a given layout, so the response is browser-cacheable (`Vary: X-Dataset-Id`). 404 until the projection is built. |
| GET | `/api/projection/labels` | Region signpost labels: `{"status", "projection_id", "labels": [{text, x, y, level, score?, source?, has_coarser?, has_finer?}]}`. An empty list (not an error) when labeling hasn't run; `status` is `"idle"` only when no projection exists. |
| POST | `/api/projection/subset/remove` | Body `{"ids": [...]}`. Culls items from the current subset layout **without re-fitting**: positions are kept, `projection_id` is unchanged, `content_version` bumps (busting the tile cache), and `bounds` shrink to the survivors. Returns the updated subset `meta`. 409 if no subset layout exists. |
