# Detectors

[← Back to API index](../API.md)

> Detector-scoped endpoints resolve the active detector (and, where scoring is
> involved, the active dataset) via the
> [`X-Dataset-Id` / `X-Detector-Id` context headers](../API.md#context-headers-x-dataset-id--x-detector-id).
> Requirements are noted per endpoint.
>
> This page covers purpose and non-obvious semantics. Exact request and
> response fields are in the [OpenAPI spec](../API.md#machine-readable-schema)
> (`frontend/openapi.json`).

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

→ `{"detectors": [...]}`: a summary of every detector file (`name`,
`text_query`, `media_example`, `media_type`, `examples`, `num_labels`,
`created_at`, `embedder_type`), whether or not it has a registry entry.

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

→ Full detector object including `labelset`. 404 if not found.

### Delete detector

```
DELETE /api/detectors/{name}
```

Deletes the detector file only; a registry entry for it is left in place
([`DELETE /api/detectors/registry/{detector_id}`](#delete-registered-detector)
removes both).

→ `{"success": true, "name": "..."}`. 404 if not found.

### Rename detector

```
PUT /api/detectors/{name}/rename
```

**Body:** `{"new_name": "Cat Meows"}`

→ `{"success": true, "old_name": "...", "new_name": "Cat Meows"}`

Also renames the detector's registry entry (and its loaded context) and its
place on the AutoFind list. 400 for a blank `new_name`; 404 if not found; 409
if the new name already exists. When the detector syncs to a
[labelset source](settings.md#labelset-sources-sync) whose server file path
is templated on `{detector_name}`, the rename moves that file to the new
name's path before the detector's next push can write there (never over an
existing file).

### Set examples

```
PUT /api/detectors/{name}/examples
```

**Body:** `{"examples": [{"type": "text", "value": "dog barking"}]}`

→ `{"success": true, "name": "...", "examples": [...]}`

Replaces the `examples` list (on the detector JSON and the registry entry) and
**adds** a `good` label for each media example not already in the labelset.
The labelset edit is additive only: no existing label is dropped, and an
exemplar the user has since voted Bad keeps that label. An unlabeled seed
(`"labeled": false`) adds none.

`text_query` and `media_example` follow the new list, on both records: each is
the value of its first text / media example, or `""` when there is none. So
replacing a typed query with a media example drops the query. Under the label
quota a typed-query detector gives its query's own set, so this is the query
Tune, AutoFind and Find use. The Train view's re-sort prompt saves its new
example here, as a text example or as an unlabeled media seed. The Manual sort
bar saves nothing: it is a browse.

### Save labels

```
POST /api/detectors/{name}/labels
```

**Requires** `X-Dataset-Id` **and** `X-Detector-Id`.

Saves the current good/bad votes into the detector's labelset. The save
merges rather than replaces: entries that resolve into the active dataset are
reconciled against the votes, and entries gathered on other datasets are kept
as they are.

→ `{"success": true, "name": "...", "num_labels": 50}`

404 if the detector is not found; 409 if the detector's vote state isn't
aligned with the active dataset.

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

404 if detector or importer not found. 422 when a field fails validation;
400 when the importer rejects its input.

### Combine detectors

```
POST /api/detectors/combine
```

**Body:** `{"names": ["A", "B"], "new_name": "A+B", "conflict_policy": "drop"}`

Merges the labelsets of two or more detectors into a new detector. All
sources must share a `media_type`. `conflict_policy="drop"` (the only
supported policy) removes any element that appears with disagreeing
labels across sources. The new detector keeps the sources' balance
(`beta`, #4665) when they all keep the same one; otherwise it keeps none and
takes the user's balance when it is loaded.

The new detector also gets a registry entry, so it shows on the Dashboard; it
keeps the sources' embedder type only when they all agree, and no source's
labelset-source binding.

→ `{"success": true, "name": "A+B", "media_type": "audio", "num_labels": 73, "combined_from": ["A", "B"], "source_label_counts": [50, 30], "examples": [...]}` (201)

400 if the sources' media types differ (or are not specific); 404 if a source
is missing; 409 if `new_name` is taken; 422 for fewer than two `names`, or a
merge that drops every element.

### Labels detail

```
GET /api/detectors/{name}/labels-detail
```

Returns the detector's saved labelset split into good/bad lists with right-pane
render data. Not gated on a loaded dataset (but when one is loaded, each item's
`cid` / `time` / `score` resolve against it).

→ `{"media_type", "good": [...], "bad": [...]}`. Each element carries a stable
`id` (derived from its origin / MD5 identity: the `element_id` the routes
below take), its `label`, display `name`, `origin_name`, `md5` and
`region_box`, plus the dataset-resolved `cid` / `time` / `score`.

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
(`target` outside the three values, or an unrecognised `provenance` value).

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
is an expert artifact most users never need. The headless
equivalent is the `portable_detector` CLI exporter (see
[`docs/CLI.md`](../CLI.md#auto-detect-run-detectors-on-a-dataset)).

→ Binary `.zip` download (`<slug-of-name>-detector.zip`).

400 (no medias loaded, or no labels to score), 404 (detector not found), 409
(active dataset can't supply the detector's embedder type).

---

## Detector Registry

### List registered detectors

```
GET /api/detectors/registry
```

→ `{"detectors": [...]}`: every registered detector the caller can read (its
own, plus those whose `readers` name the caller or `"*"`), with its `id`,
`name`, `media_type`, `num_training`, load state (`loaded` and
`detector_loaded`, the same flag), `autofind` flag, examples, embedder and
ownership fields (`created_by`, `readers`, `is_owner`); the full entry is the
`DetectorRegistryListResponse` schema in the spec. `embedder` is the loaded
context's live embedder when the detector is loaded, else the one it last
trained in. An AutoFind detector
(`autofind: true`) also carries `test_verdict`: the newest test verdict it
keeps, in the shape of the [stats](#detector-statistics)' `test_verdicts`
entries, or `null` when it was never tested. Drafts leave it out.
`labelset_source_status` says where the detector's labels are auto-exported:
`{"source_name", "display_name", "inherited", "missing_fields"}`, or `null`
when they sync nowhere (see [Labelset Sources](settings.md#labelset-sources-sync));
a non-empty `missing_fields` means the detector is not synced until those
values are filled in. The binding's field values are never listed.

`name` is what the on-disk labelset file is looked up by; the file itself is
`data/detectors/<slug-of-name>.json`. The head is trained on demand from the
labelset and lives only in RAM. `autofind` mirrors whether the
detector's name appears in the caller's `autofind_detectors` setting (toggle
it with the route below).

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
dataset supplies.

Optional `beta` is the detector's **balance** (#4665): F-beta's beta, which
way its line leans between false positives and false negatives, clamped to
`[0.25, 4]` as [`POST /api/balance`](labeling.md#the-balance) clamps it. The New
Detector form asks for it on its Threshold control, because Autopilot has no
control of its own and runs at the detector's balance. It is kept on the
detector JSON (top-level `beta`) and also becomes the user's `beta` setting,
their last pick, which the next form starts on. Omitted or `null`, the
detector keeps none and takes the user's balance when it is loaded. A
non-number is a 400 (422 when the JSON body's type is wrong).

→ `{"ok": true, "detector": {...}}` (201) — `detector` is the new registry entry.

409 if the name is already taken — by another registry entry, or by a
detector file created through `POST /api/detectors`. Names are compared by the
labelset *slug* (lowercased, punctuation collapsed), so "My Cat" and "my cat"
collide.

### Register detector from a label importer

```
POST /api/detectors/registry/from-labelset/{importer_name}
```

**Form or Body:** `name`, optional `embedder_type`, optional `beta` (the
balance, as for `POST /api/detectors/registry` above; a form sends it as a
string, and an empty one means none), plus the importer's own fields
(plugin-dependent, so not described in `/api/openapi.json`).

Runs the label importer and creates a detector seeded with the labels it
returns (entries labelled neither `good` nor `bad` are counted in `skipped`).
The media type is inferred from the labels' origins; labels spanning more than
one media type, or none whose origin names one, are rejected (400). A missing
`name` is a 422; a taken one a 409. The new registry entry keeps how it was
imported as `imported` (the importer, its fields, the time), which
[the detector's stats](#detector-statistics) return.

→ (201)
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

### Toggle AutoFind flag

```
PUT /api/detectors/registry/{detector_id}/autofind
```

**Body:** `{"autofind": true}`

→ `{"ok": true, "autofind": true}` (writes the detector's name into the
calling user's `autofind_detectors` setting, so each user curates their own
list, and `/api/auto-detect` and the CLI `--autodetect` flow pick it up). In
the GUI this is the Dashboard's Drafts ↔ AutoFind detector-tab move:
`autofind: true` detectors sit on the frozen AutoFind tab, everything else on
Drafts. 403 if the caller cannot access the detector; 404 if it does not
exist.

### Load (or unload) the active detector

```
POST /api/detectors/registry/load
```

**Body:** `{"detector_id": "abc123"}` (pass `null` or omit the field
to unload the active detector without loading another one).

The detector's labels are resolved into the active dataset (`X-Dataset-Id`);
a header naming a registered-but-unloaded dataset is a 409.

→ `{"ok": true, "message": "Loading started", "task_id": "..."}` when
loading; `{"ok": true, "labels_restored": 0, "examples_seeded": 0}`
when unloading. A load already under way answers with its `task_id`. A
detector already in memory answers the unloading shape, unless its labels were
embedded with another embedder than the active dataset's: it then re-embeds
them, answering `{"ok": true, "message": "Re-embedding labels", "task_id": ...}`.

Loading is async; subscribe to the `detector-loading-tasks` channel
on [`/api/events`](events.md) (SSE) for progress. 403 if access is denied; 404
if the detector is not in the registry.

### Unload a registered detector

```
POST /api/detectors/registry/{detector_id}/unload
```

Drops the detector's in-memory context (votes, trained head).

→ `{"ok": true, "message": "..."}`

400 if the detector isn't loaded; 404 if it doesn't exist.

### Detector loading tasks (SSE)

Active detector loading tasks are streamed on the `detector-loading-tasks`
channel of [`/api/events`](events.md), in the
[task shape](events.md#task-object-shape-loading-tasks--detector-loading-tasks).

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

Also cleans up the on-disk labelset file, unloads the detector if it is
loaded, and clears the AutoFind flag.

→ `{"ok": true}`. 403 if the caller isn't the creator; 404 if it does not
exist.

### Rename registered detector

```
PUT /api/detectors/registry/{detector_id}/rename
```

**Body:** `{"name": "New Name"}`

→ `{"ok": true, "name": "New Name"}`

A templated labelset-source file follows the rename, as on
[`PUT /api/detectors/{name}/rename`](#rename-detector), as does the
detector's place on the AutoFind list. 400 for a blank name; 403 if the caller
isn't the creator; 404 if it does not exist. 409 if the new name is already
taken. Names are compared by the labelset
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

→ The positive / negative / total label counts, `num_positive_resolved` and
`active_dataset_name`, the detector's `embedder` (the loaded context's live
one, else the one it last trained in), `embedder_type`, query and example,
`clipper`, creation and training times, ownership, its `autofind` flag, and
`test_verdicts`. Each verdict carries the tested dataset, `tested_at`, the
`beta` it was tested at, `line_count` (the line kept the top `line_count` of
the `size` items scored), `labelled` (the picks the ranges rest on),
`precision` / `recall` / `fbeta` as `{point, lo, hi}` ranges, a plain-language
`found` line, and `stale`.

`imported` says how a label importer created the detector (through
[`from-labelset`](#register-detector-from-a-label-importer) or the CLI's
`--import-labels-into … --create-detector`): `{"importer": "server_json_file",
"params": {"filepath": "/data/labels/cats.json"}, "imported_at": 1761580800.0}`.
`params` holds the fields the importer ran with, minus `file` and `password`
fields, the policy a dataset origin follows. It is `null` for a detector made
any other way. The Stats dialog's *Import* section reads it.

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

→ `{"ok": true, "dataset_id": "__detpos__<detector_id>", "task_id": "...", "media_type": "audio"}`

The build runs in the background; its progress rides the detector-loading
task channel (the dashboard row shows it). 409 if the detector has no
positive labels; 403/404 as above.

```
POST /api/detectors/registry/{detector_id}/browse-positives/release
```

Free the ephemeral positives-browse context (called when leaving the view).
Idempotent.

→ `{"ok": true, "released": true}` (`released` is `false` when there was
nothing to free).
