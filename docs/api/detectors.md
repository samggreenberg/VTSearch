# Detectors

[← Back to API index](../API.md)

> Detector-scoped endpoints resolve the active detector (and, where scoring is
> involved, the active dataset) via the
> [`X-Dataset-Id` / `X-Detector-Id` context headers](../API.md#context-headers-x-dataset-id--x-detector-id).
> Requirements are noted per endpoint.

---

## Detectors

A detector is a named labelset plus a text-sort query, persisted as a
JSON file at `data/detectors/<name>.json`. The head is trained on demand
from the labelset and lives only in `DetectorContext` once the user
loads the detector into memory.

### List detectors

```
GET /api/detectors
```

→ `{"detectors": [{"name": "Dog Barks", "text_query": "dog barking", "media_type": "audio", "examples": [...], "num_labels": 50, "created_at": 1234567890.0}]}`

### Create detector

```
POST /api/detectors
```

**Body:** `{"name": "Dog Barks", "text_query": "dog barking sounds", "media_type": "audio"}`

Or with examples: `{"name": "Dog Barks", "examples": [{"type": "text", "value": "dog barking"}]}`

→ `{"success": true, "name": "...", "text_query": "...", "media_type": "audio", "examples": [...], "num_labels": 0}` (201)

`num_labels` counts the **media** examples, which are written into the new
detector's labelset as `good` labels (see *Register detector* below); a
text-only detector reports 0.

409 if name already exists.

### Get detector

```
GET /api/detectors/{name}
```

→ Full detector object including `labelset`.

### Delete detector

```
DELETE /api/detectors/{name}
```

→ `{"success": true, "name": "..."}`

### Rename detector

```
PUT /api/detectors/{name}/rename
```

**Body:** `{"new_name": "Cat Meows"}`

→ `{"success": true, "old_name": "...", "new_name": "Cat Meows"}`

409 if new name already exists.

### Set examples

```
PUT /api/detectors/{name}/examples
```

**Body:** `{"examples": [{"type": "text", "value": "dog barking"}]}`

→ `{"success": true, "name": "...", "examples": [...]}`

Replaces the `examples` list (on the detector JSON and the registry entry) and
**adds** a `good` label for each media example not already in the labelset.
The labelset edit is additive only: no existing label is dropped, and an
exemplar the user has since voted Bad keeps that label.

### Save labels

```
POST /api/detectors/{name}/labels
```

Saves the current good/bad votes as the detector's labelset.

→ `{"success": true, "name": "...", "num_labels": 50}`

### Import labels into detector

```
POST /api/detectors/{name}/import-labels/{importer_name}
```

Run a label importer and merge results into this detector's persisted labelset.
Unlike `/api/label-importers/import/`, this does **not** require a dataset to
be loaded. Field values are passed as JSON body or multipart form (same as
regular label import).

When the detector context **is** loaded, the new labels are also resolved
against the loaded dataset's medias, applied to the detector's votes, and
a fresh head is trained with a cross-validated threshold.

→ `{"applied": 12, "skipped": 3, "resolved": 12, "trained": true, "num_labels": 62, "message": "..."}`

`resolved` counts labels resolved into the loaded detector context (0 when no
context is loaded); `trained` is `true` when a fresh head was retrained.

404 if detector or importer not found. 400 on validation errors.

### Combine detectors

```
POST /api/detectors/combine
```

**Body:** `{"names": ["A", "B"], "new_name": "A+B", "conflict_policy": "drop"}`

Merges the labelsets of two or more detectors into a new detector. All
sources must share a `media_type`. `conflict_policy="drop"` (the only
supported policy) removes any element that appears with disagreeing
labels across sources.

→ `{"success": true, "name": "A+B", "media_type": "audio", "num_labels": 73, "combined_from": ["A", "B"], "source_label_counts": [50, 30], "examples": [...]}` (201)

### Labels detail

```
GET /api/detectors/{name}/labels-detail
```

Returns the detector's saved labelset split into good/bad lists with right-pane
render data. Not gated on a loaded dataset (but when one is loaded, each item's
`cid` / `time` / `score` resolve against it).

→
```json
{
  "media_type": "audio",
  "good": [
    {"id": "...", "label": "good", "media_type": "audio", "name": "...",
     "filename": "dog.wav", "origin_name": "...", "md5": "...", "cid": 12,
     "time": 1234567890.0, "score": 0.97, "region_box": null}
  ],
  "bad": [...]
}
```

404 if the detector is not found.

### Label preview / thumbnail

```
GET /api/detectors/{name}/labels/{element_id}/preview
GET /api/detectors/{name}/labels/{element_id}/thumbnail
```

Serve one saved labelset element, resolved via its origin:

- **`/preview`** — the full underlying media file bytes (mimetype by type), or,
  for text, JSON `{"content", "word_count", "character_count"}`.
- **`/thumbnail`** — a small image: resized image (cropped to `region_box` for
  region votes), audio waveform PNG, or video mid-frame PNG. Much smaller than
  `/preview`.

404 if the detector, element, or file is missing; 500 if a thumbnail can't be
generated.

### Export portable bundle

```
POST /api/detectors/{detector_id}/portable-bundle
```

**Requires** [`X-Dataset-Id`](../API.md#context-headers-x-dataset-id--x-detector-id).
No body.

Retrains the detector from its on-disk labelset in the **active dataset's**
embedder space and streams a zipped, standalone scoring bundle (ONNX model +
manifest + README).

→ Binary `.zip` download (`<detector>-detector.zip`).

400 (no medias loaded, or no labels to score), 404 (detector not found), 409
(active dataset can't supply the detector's embedder type).

---

## Detector Registry

### List registered detectors

```
GET /api/detectors/registry
```

→
```json
{
  "detectors": [
    {
      "id": "abc123",
      "name": "Dog Barks",
      "media_type": "audio",
      "text_query": "dog barking",
      "num_training": 50,
      "loaded": true,
      "detector_loaded": true,
      "autofind": false,
      "last_trained_at": 1234567890.0
    }
  ]
}
```

`name` is the slug used to look up the on-disk labelset file at
`data/detectors/<name>.json`. The head is trained on demand from the
labelset and lives only in RAM. `autofind` mirrors whether the
detector's name appears in `autofind_detectors` settings (toggle it with
the route below).

### Register detector

```
POST /api/detectors/registry
```

**Body:**

```json
{
  "name": "Dog Barks",
  "media_type": "audio",
  "text_query": "dog barking sounds"
}
```

Or seeded with media examples (each `value` a filename previously saved in
`example_media/`, e.g. via `/api/server-media-files/upload`):

```json
{
  "name": "Red Cars",
  "media_type": "image",
  "media_example": "a1b2.jpg",
  "examples": [
    {"type": "media", "value": "a1b2.jpg"},
    {"type": "media", "value": "c3d4.jpg"}
  ]
}
```

The full `examples` list is persisted on both the detector JSON and the
registry entry. Every **media** example additionally becomes a `good`
`LabeledElement` in the detector's labelset right away (so `num_training`
is the example count, not 0): a supplied exemplar is a vote the user
already cast. Each label carries the example's durable `origin` when it has
one (`url_download`, `server_file`, …), else the `example_media` sentinel;
because labels are origin-keyed and dataset-agnostic, an `https://` exemplar
is kept verbatim even when the dataset it is used against holds only local
files. Text examples are queries, not media, and produce no labels.

On detector load every media example is *also* seeded as a Good vote against
the active dataset (matched by MD5, or embedded and inserted when absent), and
Autopilot's Good phase sorts against the embedding centroid of all of them.

→ `{"ok": true, "detector": {...}}` (201)

409 if the name is already taken — by another registry entry, or by a
detector file created through `POST /api/detectors`. Names are compared by the
labelset *slug* (lowercased, punctuation collapsed), so "My Cat" and "my cat"
collide.

### Register detector from a label importer

```
POST /api/detectors/registry/from-labelset/{importer_name}
```

**Form or Body:** `name`, optional `embedder_type`, plus the importer's own
fields (plugin-dependent, so not described in `/api/openapi.json`).

Runs the label importer and creates a detector seeded with the labels it
returns. The media type is inferred from the labels' origins; labels spanning
more than one media type are rejected (400).

→
```json
{
  "ok": true,
  "detector": {...},
  "applied": 12,
  "skipped": 0,
  "num_labels": 12,
  "ingest_task_id": "_detingest_<detector_id>"
}
```

An imported labelset usually references media the active dataset doesn't have,
which must be pulled in from their origins for the labels to be visible and
exportable. That fetch + embed runs on a **background task** streamed on the
`detector-loading-tasks` channel of [`/api/events`](events.md), so the request
returns immediately; `ingest_task_id` names it, and is `""` when there is
nothing to ingest (no active dataset, or every label already resolves).

**Wait for that task before loading the detector.** Loading restores the
labelset into votes by resolving each label against the active dataset's
media, so a load that starts mid-ingest silently drops the labels whose media
haven't landed yet. The task's terminal frame carries
`"ingest_result": {"ingested": 12}`; cancel it with
`POST /api/detectors/cancel/{task_id}`.

### Toggle Auto-Find flag

```
PUT /api/detectors/registry/{detector_id}/autofind
```

**Body:** `{"autofind": true}`

→ `{"ok": true, "autofind": true}` (writes the detector's name into
`autofind_detectors` so `/api/auto-detect` and the CLI
`--autodetect` flow pick it up). In the GUI this is the Dashboard's
Drafts ↔ AutoRun detector-tab move: `autofind: true` detectors sit on
the frozen AutoRun tab, everything else on Drafts.

### Load / unload detector

```
POST /api/detectors/registry/load
```

**Body:** `{"detector_id": "abc123"}` (pass `null` or omit the field
to unload the active detector without loading another one).

→ `{"ok": true, "message": "Loading started", "task_id": "..."}` when
loading; `{"ok": true, "labels_restored": 0, "examples_seeded": 0}`
when unloading.

Loading is async; subscribe to the `detector-loading-tasks` channel
on [`/api/events`](events.md) (SSE) for progress. 404 if the detector
is not in the registry.

### Unload detector

```
POST /api/detectors/registry/{detector_id}/unload
```

→ `{"ok": true}`

### Detector loading tasks (SSE)

Active detector loading tasks are streamed on the `detector-loading-tasks`
channel of [`/api/events`](events.md):

```json
[{"task_id": "...", "name": "...", "status": "loading", "message": "...", "current": 50, "total": 100}]
```

### Cancel detector loading

```
POST /api/detectors/cancel/{task_id}
```

→ `{"ok": true}`

### Delete registered detector

```
DELETE /api/detectors/registry/{detector_id}
```

Also cleans up the on-disk labelset file and clears the Auto-Find flag.

→ `{"ok": true}`

### Rename registered detector

```
PUT /api/detectors/registry/{detector_id}/rename
```

**Body:** `{"name": "New Name"}`

→ `{"ok": true, "name": "New Name"}`

409 if the new name is already taken. Names are compared by the labelset
*slug* (lowercased, punctuation collapsed), so "My Cat" and "my cat" collide;
re-spelling a detector's own name that way is allowed.

### Set detector readers

```
PUT /api/detectors/registry/{detector_id}/readers
```

**Body:** `{"readers": ["user1", "user2"]}` (`["*"]` makes it public).

Replaces the detector's reader access list (multi-user deployments). Only the
detector's creator may call it.

→ `{"ok": true, "readers": ["user1", "user2"]}`

403 if the caller is not the creator; 404 if the detector does not exist.

### Detector statistics

```
GET /api/detectors/registry/{detector_id}/stats
```

Returns labelset composition and provenance for a registered detector.
Counts and metadata only — never embeddings or model weights.
`num_positive_resolved` / `active_dataset_name` report how many of the
detector's positive labels currently resolve into the loaded dataset (the
set the dashboard's Browse button projects).

→
```json
{
  "name": "cat-sounds",
  "media_type": "audio",
  "num_positive": 24,
  "num_negative": 18,
  "num_total": 42,
  "num_positive_resolved": 20,
  "active_dataset_name": "ESC-50",
  "embedder": "laion_clap",
  "text_query": "cat meowing",
  "media_example": "",
  "clipper": "",
  "created_at": 1743412500.0,
  "last_trained_at": 1743419700.0,
  "created_by": "default",
  "readers": [],
  "autofind": false
}
```

403 if the caller cannot access the detector; 404 if it does not exist.

### Browse a detector's positives

```
POST /api/detectors/registry/{detector_id}/browse-positives
```

Prepare an in-memory VTSBrowse map of just this detector's positive labels.
Each positive's origin is resolved to its file and embedded with the
**detector's own** embedder (not whatever dataset is selected) — so
mixed-source detectors work and no dataset need be loaded. The resulting
throwaway context (vectors + preview bytes, never persisted) is registered
under a synthetic `dataset_id` the browse view opens.

→
```json
{
  "ok": true,
  "dataset_id": "__detpos__<detector_id>",
  "task_id": "_detbrowse_<id>",
  "media_type": "audio"
}
```

The build runs in the background; its progress rides the detector-loading
task channel (the dashboard row shows it). 409 if the detector has no
positive labels; 403/404 as above.

```
POST /api/detectors/registry/{detector_id}/browse-positives/release
```

Free the ephemeral positives-browse context (called when leaving the view).
Idempotent.

→ `{"ok": true, "released": true}`
