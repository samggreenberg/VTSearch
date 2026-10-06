# Detectors

[← Back to API index](../API.md)

> Detector-scoped endpoints resolve the active detector (and, where scoring is
> involved, the active dataset) via the
> [`X-Dataset-Id` / `X-Detector-Id` context headers](../API.md#context-headers-x-dataset-id--x-detector-id).
> Requirements are noted per endpoint.

---

## Detectors

A detector is a named labelset plus a text-sort query, persisted as a
JSON file under `data/detectors/`, named after a slug of the detector name
rather than the name itself (`Dog Barks` → `dog_barks.json`; see
[CLI.md § Detector file names](../CLI.md#detector-file-names)). The head is trained on demand
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

Or with examples: `{"name": "Dog Barks", "media_type": "audio", "examples": [{"type": "text", "value": "dog barking"}]}`

`name` and `media_type` are required (`media_type: "any"` is rejected with
400), plus at least one of `text_query`, `media_example`, or `examples`.
Optional `embedder_type` pins which kind of embedder the detector learns in.

→ `{"success": true, "name": "...", "text_query": "...", "media_type": "audio", "examples": [...], "num_labels": 0}` (201)

`num_labels` counts the **media** examples, which are written into the new
detector's labelset as `good` labels (see *Register detector* below); a
text-only detector reports 0.

409 if name already exists.

This writes the detector file only; `POST /api/detectors/registry` (below) is
what the GUI uses, and also creates the registry entry.

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

→ `{"success": true, "old_name": "...", "new_name": "Cat Meows", "pending_labelset_move": null}`

409 if new name already exists. `pending_labelset_move` is
`{"old_path", "new_path"}` when the detector has a
[labelset source](settings.md#labelset-sources-sync) whose file path is
templated on `{detector_name}` / `{detector_id}` and the rename left the old
file behind; offer the user a
[move](#move-an-orphaned-labelset-file), otherwise it is `null`.

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

**Requires** `X-Dataset-Id` **and** `X-Detector-Id`.

Saves the current good/bad votes as the detector's labelset.

→ `{"success": true, "name": "...", "num_labels": 50}`

409 if the detector's vote state isn't aligned with the active dataset.

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

### Vote on a saved label

```
POST /api/detectors/{name}/labels/{element_id}/vote
```

**Body:** `{"target": "good"}`, `{"target": "bad"}`, or `{"target": "remove"}`,
plus an optional `provenance` block (same shape as
[`POST /api/medias/{media_id}/vote`](medias.md#vote-on-a-media); defaults to
`{"flow": "labelset_review"}`).

Edits one element of the detector's **saved** labelset directly — the
Dashboard's label-review surface, which works whether or not the detector is
loaded. Absolute-target semantics: `good` / `bad` set the label (re-asserting
the current label is a no-op), `remove` drops the element. When the element
resolves into the active dataset, the loaded detector's in-memory votes are
updated to match so retraining and learned sort see the change. Provenance is
recorded only when the label actually flips.

→ `{"ok": true, "action": "flipped"}` — `action` is `"flipped"`,
`"removed"`, or `"unchanged"`.

400 (malformed `provenance`), 404 (detector or element not found), 422
(`target` outside the three values).

### Export portable bundle

```
POST /api/detectors/{detector_id}/portable-bundle
```

**Requires** [`X-Dataset-Id`](../API.md#context-headers-x-dataset-id--x-detector-id).
No body.

Retrains the detector from its on-disk labelset in the **active dataset's**
embedder space and streams a zipped, standalone scoring bundle (ONNX model +
manifest + README).

This route has no GUI affordance and is meant to be called directly: the bundle
is an expert artifact most users never need, and as a dashboard menu item it sat
beside **Export labels** reading as a confusing second "export". The headless
equivalent is the `portable_detector` CLI exporter (see
[`docs/CLI.md`](../CLI.md#auto-detect-run-detectors-on-a-dataset)).

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

Entries also carry `examples`, `media_example`, `embedder`, `embedder_type`,
`created_at`, `created_by`, `readers`, and `is_owner`; see the
`DetectorRegistryListResponse` schema in the spec. An AutoRun detector
(`autofind: true`) also carries `test_verdict`: the newest test verdict it
keeps, in the shape of the [stats](#detector-statistics)' `test_verdicts`
entries, or `null` when it was never tested. Drafts leave it out.

`name` is what the on-disk labelset file is looked up by; the file itself is
`data/detectors/<slug-of-name>.json`. The head is trained on demand from the
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
already cast. The exception is an example marked `"labeled": false` — an
unlabeled *seed* from a [seed importer](medias.md#seed-importers), i.e. media that is
close to the target without being it. A seed is persisted and steers the
first sort like any other example, but produces no label and casts no vote,
so a detector built from seeds alone starts at `num_training: 0`.
Each label carries the example's durable `origin` when it has
one (`url_download`, `server_file`, …), else the `example_media` sentinel;
because labels are origin-keyed and dataset-agnostic, an `https://` exemplar
is kept verbatim even when the dataset it is used against holds only local
files. Text examples are queries, not media, and produce no labels.

On detector load every labeled media example is *also* seeded as a Good vote
against the active dataset (matched by MD5, or embedded and inserted when
absent); seeds are skipped there too. Autopilot's Good phase sorts against
the embedding centroid of *all* the media examples, seeds included — that is
the one thing a seed does do.

Optional `embedder_type` locks the embedder kind the detector learns in
(`semantic`, `patch_semantic`, or `structural`; a concrete embedder name is
also accepted and classified). Empty lets the server pick the sole kind the
dataset supplies. (The schema also accepts a `trainable` flag, which nothing
reads.)

→ `{"ok": true, "detector": {...}}` (201) — `detector` is the new registry entry.

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

The detector's labels are resolved into the active dataset (`X-Dataset-Id`);
a header naming a registered-but-unloaded dataset is a 409.

→ `{"ok": true, "message": "Loading started", "task_id": "..."}` when
loading; `{"ok": true, "labels_restored": 0, "examples_seeded": 0}`
when unloading.

Loading is async; subscribe to the `detector-loading-tasks` channel
on [`/api/events`](events.md) (SSE) for progress. 403 if access is denied; 404
if the detector is not in the registry.

### Unload detector

```
POST /api/detectors/registry/{detector_id}/unload
```

Drops the detector's in-memory context (votes, trained head).

→ `{"ok": true, "message": "..."}`

400 if the detector isn't loaded; 404 if it doesn't exist.

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

Cancels any task on the `detector-loading-tasks` channel: a detector load, a
labelset-media ingest, or a positives-browse build.

→ `{"ok": true}`; 404 if the task is unknown.

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

→ `{"ok": true, "name": "New Name", "pending_labelset_move": null}`

`pending_labelset_move` has the same meaning as on
[`PUT /api/detectors/{name}/rename`](#rename-detector). 403 if the caller isn't
the creator. 409 if the new name is already taken. Names are compared by the labelset
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
set the dashboard's Browse button projects). `test_verdicts` is every test
verdict the detector keeps, one per tested dataset, newest first: what a test
measured its line to ship there, as ranges, with `stale` once the
detector has been retrained since (see
[the kept verdict](find.md#the-kept-verdict)). The Stats dialog's *Tested on*
section reads it.

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
  "embedder_type": "semantic",
  "text_query": "cat meowing",
  "media_example": "",
  "clipper": "",
  "created_at": 1743412500.0,
  "last_trained_at": 1743419700.0,
  "created_by": "default",
  "readers": [],
  "autofind": false,
  "test_verdicts": [
    {
      "dataset_id": "4f2c…", "dataset_name": "esc50-new", "tested_at": 1790000000.0,
      "beta": 1.0, "line_count": 64, "size": 1200, "labelled": 34,
      "precision": {"point": 0.78, "lo": 0.70, "hi": 0.85},
      "recall": {"point": 0.52, "lo": 0.38, "hi": 0.66},
      "fbeta": {"point": 0.62, "lo": 0.51, "hi": 0.72},
      "found": "about half of them found",
      "stale": false
    }
  ]
}
```

403 if the caller cannot access the detector; 404 if it does not exist.

### Move an orphaned labelset file

```
POST /api/detectors/registry/{detector_id}/labelset-source/move-file
```

**Body:** `{"old_path": "...", "new_path": "..."}` — normally the
`pending_labelset_move` pair a rename returned.

Moves the detector's labelset-source file after a rename left it at the old
template-resolved path (the *Move existing labelset file?* prompt).

→ `{"ok": true, "moved": true, "old_path": "...", "new_path": "..."}`
(`moved: false` when there was nothing at `old_path`).

400 (path outside the allowed base), 404 (detector not found), 409
(`new_path` already exists).

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
