# Datasets

[← Back to API index](../API.md)

> Endpoints that act on "the loaded dataset" resolve it via the
> [`X-Dataset-Id` context header](../API.md#context-headers-x-dataset-id--x-detector-id)
> (registry routes take the id in the path instead).
>
> This page covers purpose and non-obvious semantics. Exact request and
> response fields are in the [OpenAPI spec](../API.md#machine-readable-schema)
> (`frontend/openapi.json`).

---

## Media Types, Embedders, Clippers & Converters

### Media types

```
GET /api/media-types
```

→ `{"media_types": [...]}`: every registered media type that is not hidden by
the admin, with its `type_id`, display `name`, accepted `file_extensions` and
capability flags (`importable` is `false` for a convert-in type such as
`face`; a non-embeddable type such as `document` lists in `converts_to` the
embeddable types it can convert into).

### Embedders

```
GET /api/embedders
```

**Query:** `?media_type=image` (optional): filter by `type_id` or
`folder_import_name`.

→ `{"embedders": [{"name": "siglip", "display_name": "SigLIP", "model_id": "google/siglip-base-patch16-224", "media_type_id": "image", ...}]}`

Embedders the admin hides (`hidden_plugins`) are left out, and so is every
patch / structural embedder under the `semantic_only` lock.

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
| `400` | Missing `embedder`; missing `file` (multipart) or `text` (JSON); text sent to an embedder where `supports_text == false` (the body carries `supports_text: false`); a file extension that belongs to another media type than the embedder's (the body carries `expected_media_type` / `detected_media_type`); embedder returned no vector for the upload. |
| `404` | Unknown embedder name. The body's `available` lists every registered embedder's `name`. |
| `500` | Model load or embedding raised, or text embedding returned no vector. |

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
`type_id` or `folder_import_name` (`target` wins if both are sent). Omit both
to list all converters.

→ `{"converters": [{"name": "video2image", "source_type": "video", "target_type": "image", ...}]}`;
converters hidden by the admin or flagged `hidden_from_picker` are left out.

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
active task. See
[events.md](events.md#task-object-shape-loading-tasks--detector-loading-tasks)
for the task shape. The first frame is the current snapshot; no separate
bootstrap call is needed. Work that isn't bound to a task reports nowhere.

### Cancel loading

```
POST /api/dataset/cancel
```

Cancels every active loading task, then waits up to ~2 s
(`CANCEL_ACK_GRACE_SECONDS`) for them to act on the flag.

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

```
POST /api/dataset/dismiss/{task_id}
```

Removes a finished task from the `loading-tasks` channel. A failed import is
not pruned on a timer: it stays on the channel until this is called, so the
dashboard can keep showing it (#4973). That failure is visible only to the user
who started the import, so for anyone else it does not exist. Other finished
tasks (and a cancelled import) lapse on their own, but can be dismissed early.

→ `200 {"ok": true}`, `404 {"message": "Task not found", ...}` (unknown, or another
user's failure), or `409 {"message": "Task is still running; cancel it instead", ...}`

---

## Importers

### List importers

```
GET /api/dataset/all-importers
```

Returns all registered importers including built-in ones (pickle, combine_datasets, demo),
plus the picker-tab declarations the Add-Dataset modal groups them under.

→ `{"importers": [{"name": "...", "display_name": "...", "description": "...", "fields": [...], "ui_mode": "form", "category": "...", ...}], "tabs": [{"id": "...", "label": "...", "icon": "...", "order": 0}]}`

The `combine_datasets` entry also carries `enabled`, `true` only when at least
two registered datasets share a media type.

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

→ `{sample_size, counts_by_type, extensions, dominant, truncated}`: `dominant`
is the media type with the most matching files (`null` when none match), and
`truncated` says the walk stopped early on its directory or time budget, so the
sample may be partial. `source` takes the same values as
[`/api/browse-media-files`](#browse-media-files).

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

→ `{"ok": true, "message": "Loading started", "task_id": "...", "task_ids": ["..."]}`

**Multi-dataset demo load.** An `outputs` list loads the demo as several
linked datasets, one per entry, in the shape the importer route below takes.
Each entry's `source_specs` is one row of the demo's own media type, direct or
through one converter (which runs with its default parameters); any other row
is a 400. With `outputs`, the single-dataset `embedder`, `embedders`,
`clipper`, `clipper_params`, `cleaners` and `converter` are ignored, and
`task_ids` lists one task per entry. Example, the photos and the faces cut out
of them:
`{"name": "celebrity_together_s", "outputs": [{"media_type": "image", "source_specs": [{"source_type": "image"}]}, {"media_type": "face", "source_specs": [{"source_type": "image", "converter": "image2face"}]}]}`.

**From importer:**

```
POST /api/dataset/import/{importer_name}
```

**Form or Body:** the importer's declared `fields` (from
`GET /api/dataset/all-importers`), plus the shared load options (`embedder`,
`clipper`, `clipper_params`, `cleaners`, `dataset_name`, …).

→ `{"ok": true, "message": "Loading started", "task_id": "...", "task_ids": ["..."]}`

**Multi-dataset import.** An `outputs` list turns one run of the importer
into several datasets, one per entry (the Add Dataset dialog's
**Multi-Dataset** mode). Each entry carries one dataset's `media_type`
(the type it ends up holding), its `source_specs` (the same rows a
single-dataset import sends, every converter producing that type), an
optional `category` (the ingestion category it stands for, when it is reached
only through a converter: `{"media_type": "image", "category": "document",
"source_specs": [{"source_type": "document", "converter":
"document2image", …}]}` is the *Document* dataset of rendered pages), and
its own `embedder` / `embedders` / `clipper` / `clipper_params` /
`cleaners` / `dataset_name`. The body's top-level `dataset_name` is the base
name; an entry without its own is named `"<base> – <Category>"`.
`build_projection`, `merge_near_duplicates`, `reference_files` and
`autofind` stay top-level and apply to every dataset. The importer acquires
its source once (the URL-archive importer downloads and extracts once) and
every dataset gets its own row, task id (listed in `task_ids`, in `outputs`
order), import-finished event and AutoFind run; a cancel on one row stops
that dataset alone. The registry entries they land as share one
`import_group` id (see [List registered datasets](#list-registered-datasets)).
An importer that produces one dataset per run (its `multi_output` is off)
refuses it with 400; the form offers the mode only where the importer's
`GET /api/dataset/all-importers` entry reports `supports_multi_output: true`.

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
(required, unless `outputs` is sent); optional `dataset_name` (default
`"Local folder upload"`), `embedder` / `embedders` (a JSON array),
`clipper`, `clipper_params` (JSON string), `source_specs`, `cleaners`,
`recursive` (default `"true"`), `build_projection`, `merge_near_duplicates`,
and `outputs` (the multi-dataset list above, JSON-encoded, walking the one
upload into several datasets). Streams the uploaded folder to a server temp dir and
runs the `server_folder` importer in the background.

→ `{"ok": true, "message": "...", "task_id": "...", "task_ids": ["..."]}`

**From a browser paths-file upload:**

```
POST /api/dataset/import-local-files
```

**Form (`multipart/form-data`):** `paths_file` (a single `.txt` / `.list` /
`.npz` of media paths, required), `media_type` (required, unless `outputs` is
sent); optional `dataset_name` (default `"Local files upload"`), `embedder` /
`embedders`, `clipper`, `clipper_params` (JSON string), `source_specs`,
`cleaners`, `build_projection`, `merge_near_duplicates`, `outputs` (as for
`import-local-folder`). Runs the `server_files` importer in the background.
Same response shape as `import-local-folder`.

→ `{"ok": true, "message": "...", "task_id": "...", "task_ids": ["..."]}`

All load endpoints are async: they return as soon as the task is started, and
the returned `task_id` names it on the `loading-tasks` channel of
[`/api/events`](events.md) (SSE). `task_id` can be `""` in the rare case no task
was registered. Cancel with `POST /api/dataset/cancel/{task_id}`. The three
routes that accept `outputs` also return `task_ids`, one per dataset started
(a single-dataset request lists its one `task_id`), each cancellable on its own.

**AutoFind after import.** `load-file`, `load-demo`, `import/{importer_name}`,
`import-local-folder` and `import-local-files` accept an optional `autofind`
flag (`"true"` / `"false"`): whether to run the caller's AutoFind detectors on
the dataset once it is saved. A sent flag is also remembered as the caller's
`autofind_on_import` [setting](settings.md), which decides an import that
sends none (default `true`). The run starts only after the import finished
successfully, as its own task on the `loading-tasks` channel keyed to the new
dataset (see [Run AutoFind](#run-autofind-on-a-registered-dataset)). When none
of the caller's AutoFind detectors applies (another media type, or an embedder
type the dataset lacks) nothing runs; if they have any AutoFind detectors at
all, a row that is already idle reports it, its `autofind.skipped` holding the
reason.

### Demo datasets

```
GET /api/dataset/demo-list
```

**Query (optional):** `embedder`, `clipper`, `converter` — the choices the
user is about to load with; each entry's `status` is computed against them
(a cached pickle built with a different embedder/clipper is not `ready`).

→ `{"datasets": [...]}`, one entry per demo whose media type is registered.
`status` is `"ready"` (a cached pickle matches the requested choices),
`"needs_embedding"` (the media is on disk but the pickle is missing or was
built with another embedder / clipper) or `"needs_download"`; `pkl_embedder`
/ `pkl_clipper` name what the cached pickle was built with, and
`available_converters` the converters that apply to the demo's media type.

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
- `source` (required): `"demo:<name>"` (a demo dataset's folder), `"folder"`
  (the configured `saved_datasets_dir`) or `"server_fs"` (the user's data
  directory in multi-user mode, the filesystem root otherwise: the scope the
  `server_folder` / `server_files` importers accept).
- `path` (optional): relative sub-path within the root (default `""`).

Lists files and subdirectories within an allowed root, filtered to only
media files with recognized extensions (dot-files skipped).

→ `{directories, files, root_path, default_path}`: `path`s are relative to
`root_path`, and `default_path` is the sub-path the picker should open at.
400 if `path` escapes the root, 404 if the source or directory is missing.

### Select browsed file

```
POST /api/browse-media-files/select
```

**Body:** `{"source": "demo:esc50_s", "path": "dog/1-100032-A-0.wav"}`

Copies a file from a browse source into the user's `example_media/` directory
with a unique prefix to avoid collisions. 400 if `path` escapes the root, 404
if the source or file is missing.

→ `{"filename": "abc123_bark.wav", "original_name": "bark.wav"}` (201)

---

## Staging (for combine-datasets flow)

```
POST /api/dataset/stage-file
```

**Form:** `file` - `.pkl` file.

→ `{"path": "/abs/staging/path.pkl", "name": "uploaded.pkl", "count": 500, "media_type": "audio", "error": ""}`:
the staged file, ready to pass to [combine](#combine-datasets). Always 200 once
the file is saved, so the client keeps the path to clean up; a pickle the
server cannot read reports it in `error` (with `count` 0 and `media_type`
`"unknown"`).

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

→ Binary `.pkl` file download (`vtsearch_dataset.pkl`) of the active
dataset. 400 if no dataset is loaded.

### Clear dataset

```
POST /api/dataset/clear
```

Drops the `X-Dataset-Id` dataset's in-memory context (required: 400
without it); the registry entry stays.

→ `{"ok": true}`

---

## Dataset Registry

### List registered datasets

```
GET /api/datasets/registry
```

→ `{"datasets": [...]}`: every registered dataset the caller can read, with
its `id`, `name`, `media_type`, `num_items`, `loaded` flag, recorded `source`
origin, ownership (`created_by`, `readers`, `expires_at`) and embedder fields;
the full entry is the `DatasetsRegistryListResponse` schema in the spec. A
dataset past its `expires_at` is unregistered on the way out rather than
listed. `bound_embedders` names every embedder the dataset binds and
`embedders_by_type` the one of each embedder type, which is what the Combine
Datasets dialog checks for conflicts; `clipper` is the clipper's display name
(`"-"` for a media type's default clipper).

`import_group` and `output_category` link the datasets of one
[multi-dataset import](#loading-datasets): every dataset the run produced
carries the same `import_group` id and the `category` of the output it stands
for (`"face"`, `"document"`…, which can differ from `media_type`). A
single-dataset import has `import_group: null` and `output_category: ""`.
Deleting one sibling leaves the others as they are.

### Load registered dataset

```
POST /api/datasets/registry/{dataset_id}/load
```

→ `{"ok": true, "message": "Loading started", "task_id": "..."}`. A dataset
already in memory answers at once with `task_id` `""`; a load already under
way answers with that load's `task_id`.

403 if access is denied; 404 if the dataset or its saved pickle is missing;
410 if it has expired (it is unregistered on the spot).

### Run AutoFind on a registered dataset

```
POST /api/datasets/registry/{dataset_id}/autofind
Content-Type: application/json

{"detector_ids": ["<detector id>", "..."]}
```

→ `{"ok": true, "message": "AutoFind started", "task_id": "_autofind_…"}`

Runs detectors that apply to the (loaded) dataset in the background. The body
is optional. Without it (or with `detector_ids` omitted or `null`), the run
uses the caller's AutoFind detectors - the Dashboard's ⋯ **Run AutoFind**. With
`detector_ids`, it uses exactly the detectors those registry ids name, drafts
included, and leaves the caller's AutoFind list alone - the Dashboard's big
**Find** button, which sends the ticked detectors once per ticked dataset.
Detectors of another media type, or of an embedder type the dataset lacks, are
skipped as for the AutoFind list. The task reports on the
`loading-tasks` channel with the dataset's `dataset_id`, so it renders on the
dataset's row, and carries an `autofind` block: `{run_id, owner, trigger,
dataset_id, dataset_name}` (`trigger` is `"manual"` without `detector_ids` and
`"find"` with them), plus `detectors_run`, `total_hits`,
`missing_detectors` and `auto_export` once it finishes. The results go to the
caller's AutoFind exporter when one is set, and are served by
[`GET /api/autofind/runs/{run_id}`](find.md#autofind-results). Cancel with
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

Drops the dataset's in-memory context; it stays registered.

→ `{"ok": true}`

400 if not loaded; 403 if the caller isn't the creator.

### Delete registered dataset

```
DELETE /api/datasets/registry/{dataset_id}
```

Unloads the dataset if needed, removes it from the registry and deletes its
pickle and any sidecar files.

→ `{"ok": true}`. 403 if the caller isn't the creator.

### Rename registered dataset

```
PUT /api/datasets/registry/{dataset_id}/rename
```

**Body:** `{"name": "New Name"}`

→ `{"ok": true, "name": "New Name"}`. Also renames the loaded context. 400 for
a blank name; 403 if the caller isn't the creator.

### Dataset statistics

```
GET /api/datasets/registry/{dataset_id}/stats
```

Returns registry and ingest statistics for a registered dataset
(`num_dupes`, `file_type_counts`, ingest start / finish times, `origin`,
`source`, `clipper`, `embedder`). The response is a superset of the Dashboard
grid's row (`name`, `media_type`, `num_items`, `created_at`, `expires_at`,
`created_by`, `readers`), so the Stats window can show everything the grid
does while it covers the grid up.

`expires_at` is `null` when the dataset never ages off. `clipper` is the
clipper's display name, `""` for a media type's default clipper.

`file_type_counts` keys are file types, not necessarily filename extensions:
each item is typed by its filename extension when it has one, and otherwise by
the format sniffed from its bytes, so a service importer that names items after
an opaque content id still reports `{"jpg": 437}` instead of one useless bucket.
Items that no signal could type land in a parenthesised `"(unknown)"` key (the
parenthesis is what tells a sentinel from a real extension). When the stored
histogram is uninformative — every item unknown — and the dataset happens to be
loaded, the endpoint recounts it from the in-memory medias and writes the repair
back to the registry.

`coverage_atlas` is `null` when the dataset is not loaded (the atlas lives only
in memory), else `{"built", "building", "auto_build_threshold"}`: whether the
loaded dataset has a coverage atlas, whether an on-demand
[build](#build-coverage-atlas) is running, and the item count above which
loading leaves the atlas unbuilt. The Stats window offers the build from it.

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

→ `{"duplicate_sets": [{"name", "members": [{"md5", "filename", "category", "origin_name", "importer"}]}]}`,
`importer` being the importer each member came through.

400 if the dataset isn't loaded (duplicate provenance lives only in memory);
403 if access is denied; 404 if the dataset does not exist.

### Set dataset readers

```
PUT /api/datasets/registry/{dataset_id}/readers
```

**Body:** `{"readers": ["user1", "user2"]}`

Sets which users can access a dataset (multi-user deployments); `["*"]`
makes it readable by every user. Only the dataset's creator can change it.

→ `{"ok": true, "readers": ["user1", "user2"]}`. 403 if the caller isn't the
creator; 404 if the dataset does not exist.

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

A request while a build is already running returns that build's `task_id` and
starts nothing. The [stats](#dataset-statistics) response's `coverage_atlas`
reports whether the dataset has an atlas, and the Stats window's **Build**
button calls this route.

400 if the dataset isn't loaded; 403 if access is denied; 404 if it doesn't exist.

### Domain-shift report

```
GET /api/datasets/registry/{dataset_id}/domain-shift
```

Reports how typical the **active** dataset's items (the `X-Dataset-Id`
header) look against `{dataset_id}`'s own embeddings — `{dataset_id}` is the
*reference*, i.e. the dataset a detector was trained on. Use it before
trusting a detector trained on the reference against the active dataset.

→ `{reference_dataset_id, n_items, alpha, frac_atypical, expected_atypical,
z_score, median_pvalue, shifted}`. Each active item gets a split-conformal
kNN p-value against the reference's vectors, uniform for an item drawn from
the reference, on every embedder including patch embedders (#3348).
`frac_atypical` is the fraction of active-dataset items whose p-value falls
below `alpha`, roughly the shifted proportion; it stays near
`expected_atypical` when there is no shift. `shifted` is the headline verdict
(statistically clear **and** practically large excess). `n_items` is the
number scored: at most 4 000, sampled. Needs no coverage atlas. See
[`docs/ML.md`](../ML.md#domain-shift).

400 if the reference isn't loaded or has fewer than 40 embeddings, the two
datasets use different embedders, the reference is the active dataset, or
the active dataset has no embeddings; 403 if access is denied; 404 if the
reference doesn't exist.

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
| POST | `/api/projection/build` | Body (optional) `{"ids": [...], "force": false}`. Returns at once: `{"status": "ready", "projection_id"}` when a layout is cached or persisted, otherwise starts the background fit and returns `{"status": "building", "job_id"}`; poll `meta` until ready. `ids` fits a subset layout over just those items (e.g. a Find run's positives); `force` discards the existing layout and re-fits ("Re-project"). 422 if `ids` is not a list of integers; 409 if the dataset (or the `ids` selection) is empty or has no embeddings. |
| GET | `/api/projection/meta` | Build status (`status`, plus `current` / `total` / `step` / `overall` / `eta_seconds` / `error` while building) and, once ready, `projection_id`, `bounds`, `levels` (`{level, n_cells, radius}`), `base_radius`, `tile_span`, `point_count`, `media_type`, `bin_shape`, `has_labels`, `content_version`. |
| GET | `/api/projection/tiles/{level}/{tx}/{ty}` | One tile's cells: `{"level", "tx", "ty", "cells": [{q, r, cx, cy, count, rep_id, member_ids?}]}`. Tile coordinates may be negative. Immutable for a given layout, so the response is browser-cacheable (`Vary: X-Dataset-Id`). 404 until the projection is built. |
| GET | `/api/projection/labels` | Region signpost labels: `{"status", "projection_id", "labels": [{text, x, y, level, score?, source?, has_coarser?, has_finer?}]}`. An empty list (not an error) when labeling hasn't run; `status` is `"idle"` only when no projection exists. |
| POST | `/api/projection/subset/remove` | Body `{"ids": [...]}`. Culls items from the current subset layout **without re-fitting**: positions are kept, `projection_id` is unchanged, `content_version` bumps (busting the tile cache), and `bounds` shrink to the survivors. Returns the updated subset `meta`. 409 if no subset layout exists. |
