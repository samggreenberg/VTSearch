# Medias & Sorting

[← Back to API index](../API.md)

> Media, vote, and sort endpoints are scoped to the active dataset/detector via
> the [`X-Dataset-Id` / `X-Detector-Id` context headers](../API.md#context-headers-x-dataset-id--x-detector-id).
> Vote- and label-mutating routes **require** them (400 otherwise).
>
> This page covers purpose and non-obvious semantics. Exact request/response
> fields are in the OpenAPI spec (`frontend/openapi.json`, or live at
> `GET /api/docs`); see [Machine-readable schema](../API.md#machine-readable-schema).

---

## Medias

### List media IDs

```
GET /api/medias/ids
```

→ A JSON array of lightweight stubs, one per loaded media: `id`, `media_type`,
and `embedder` / `embedders` (plural when a media was embedded by more than one
embedder, e.g. a semantic + region-patch pair) when present. Display metadata is
deliberately left out and fetched on demand for the ids a client actually shows,
via [Batch fetch](#batch-fetch-metadata), so the listing stays small on datasets
with tens of thousands of items.

### Batch fetch metadata

```
POST /api/medias/batch          Body: {"ids": [0, 1, 2]}
```

→ A JSON array of full metadata objects for the requested ids. Unknown ids are
silently omitted.

`has_original: true` marks an item a
[MediaCleaner](../EXTENDING-media.md#adding-a-media-cleaner) rewrote at load
time, whose pre-clean payload was kept beside the canonical (cleaned) one. Those
items accept [`?variant=original`](#payload-variants-variantoriginal) on every
payload route below, and the detail viewer offers a Clean/Original toggle. The
key is absent otherwise.

`custom_metadata` holds the media type's display fields (`duration` /
`frequency` for audio, `width` / `height` for images, `word_count` for text),
with any importer-supplied `custom_metadata` layered on top. It also carries up
to three curated **provenance** lines distilled from the media's
`origin.params`:

| Field | Present on | Example |
|-------|-----------|---------|
| `Source` | Converter / clipper output | `/data/videos/movie.mp4` |
| `Derived Via` | Converter / clipper output | `Video → Images (n_clips=2)` |
| `Imported Via` | Any media whose origin names an importer | `Manifest (paths_file=/data/list.txt)` |

`Source` is the file the item was cut from (the video behind an extracted frame,
the recording behind an audio clip); a plainly imported file has neither
`Source` nor `Derived Via`. Each is one line rather than a key per
`origin.params` entry, because a dataset-level import knob (`size=60`) is not a
fact about one item, and the machine-only replay recipe
(`converter_content_hash`, `clipper_chain`, `converter_param_*`, …) is folded
in rather than listed raw. The enriched label export
(`GET /api/labels/export?enrich=true`) does flatten the full `origin.params`
key by key, because an export is a machine-facing artifact.

### Payload variants (`?variant=original`)

Every per-media payload route below (`/audio`, `/video`, `/image`,
`/thumbnail`, `/text`, `/paragraph`, `/media`) accepts an optional `variant`
query:

| `variant` | Serves |
|-----------|--------|
| omitted / `""` | The **canonical** payload: the cleaned bytes that were actually hashed, embedded, and scored. |
| `original` | The pre-clean payload of an item a cleaner rewrote at load time. |

`?variant=original` on an item with no snapshot (`has_original` absent) falls
back to the canonical payload rather than 404ing, so a stale link still shows
the item. Any other value is rejected with `422`. Derived metadata is recomputed
from what is actually served: the `original` variant regenerates the thumbnail,
recounts `word_count` / `character_count` for text, and hashes the served bytes
for its `ETag`.

### Stream audio

```
GET /api/medias/{media_id}/audio
```

→ `audio/wav`. 404 if the media is not found.

### Stream video

```
GET /api/medias/{media_id}/video
```

→ `video/mp4`, `video/webm` or `video/ogg` by filename extension; formats a
browser can't play are transcoded to MP4. 400 if not a video, 404 if not found,
415 if transcoding is needed but neither ffmpeg nor OpenCV is available.

### Stream image

```
GET /api/medias/{media_id}/image
```

→ The image bytes, typed by filename extension. For a non-image media type the
route delegates to that type's `image_response` hook, so audio serves its
waveform PNG, video its mid-frame, and a PDF document its first page. 400 if
the hook yields nothing, 404 if not found.

### Get text content

```
GET /api/medias/{media_id}/paragraph
GET /api/medias/{media_id}/text
```

Both paths serve the same handler. → `{content, word_count, character_count}`.
400 if not a text media, 404 if not found.

### Generic media endpoint

```
GET /api/medias/{media_id}/media
```

Delegates to the registered media type's handler, so it works for every media
type. 400 for an unsupported type, 404 if not found.

### Vote on a media

```
POST /api/medias/{media_id}/vote          Body: {"target": "good" | "bad" | "none"}
```

**Absolute-target semantics**, not toggle semantics: `target` is the state the
media should be in *after* the call, so un-voting is an explicit
`{"target": "none"}`. The call is **idempotent**: sending the state the media is
already in is a no-op that appends no label history, credits no achievements,
and returns the existing click time, so two stale tabs clicking the same item
cannot flip it back and forth.

→ `{ok, state, click_time}`. `state` is the vote state after the call, so a
client can reconcile its optimistic view without re-fetching `GET /api/votes`;
`click_time` is `null` when the target was `"none"` or the call was a no-op.
Unknown body fields are silently dropped.

**Optional `region_box`** (`"good"` targets only): `[x0, y0, x1, y1]` in
normalised, pre-rotation image coordinates (`0..1`), marking *which region* the
user is voting good on. It is persisted with the vote and used by region-aware
head training. The box is dropped when the vote is removed or switched to bad.

```json
{"target": "good", "region_box": [0.2, 0.3, 0.55, 0.7]}
```

**Optional `provenance`**: how the item came to be in front of the user. It is
recorded because that context cannot be re-derived later (the ranking is
client-side state and the scoring model is overwritten by the next retrain). It
is stored in the element's labelset `metadata` under `"vt:provenance"` and
round-trips through label export/import; nothing in the app reads it. The four
categorical fields are independent axes rather than one fused enum, because how
an item was *drawn* and who was *driving* come apart (a user can pick the
`hard` select mode by hand):

| Field | Values | Meaning |
|-------|--------|---------|
| `flow` | `autopilot`, `list_review`, `find_verify`, `labelset_review`, `seed_example`, `import`, `bulk`, `undo`, `check`, `test`, `unknown` | Which UI flow drove the vote (`check` is a spot-check pick, `test` a Test-mode pick that never trains). |
| `phase` | `good`, `bad`, `more`, `hard`, `new` | Autopilot phase; ignored unless `flow` is `autopilot`. |
| `select_mode` | `top`, `hard`, `new` | How the item was drawn off the ranking. |
| `sort_kind` | `learned`, `text`, `load` | Which ranking the user was looking at. |
| `rank_at_vote` | integer ≥ 0 | The item's position in that ranking. |
| `score_at_vote` | float | The item's model score when it was surfaced. |

Provenance is recorded **only when the call changes the vote state**, so an
idempotent re-send cannot overwrite what the original click recorded. A payload
carrying nothing beyond `{"flow": "unknown"}` is dropped rather than stored.

| Status | Cause |
|--------|-------|
| `400` | `region_box` on a `"bad"` / `"none"` target, or a malformed box (not 4 numbers in `[0, 1]`). |
| `404` | Media not found. |
| `422` | Missing or unknown `target`, or an unrecognised `provenance` value. |
| `500` | The vote was applied in memory but the detector labelset could not be persisted. |

### Bulk vote

```
POST /api/medias/vote-bulk          Body: {"ids": [1, 2, 3], "target": "good", "provenance": {...}}
```

Applies one absolute `target` to many medias with the per-media vote's
idempotent semantics (including Find-mode verification: a good/bad target marks
the item verified), persisting the detector labelset once rather than per id.
`provenance` applies to every id and defaults to `{"flow": "bulk"}`. Bulk votes
are image-level (no region boxes) and do not build the Marathoner streak. Powers
the Browser's "Verified Good" / "Verified Bad" actions.

→ `{ok, changed, missing}`: `changed` counts only ids whose state actually
moved; `missing` lists ids not in the loaded dataset. 400 if no ids, 422 on an
unrecognised `provenance` value.

### Thumbnail

```
GET /api/medias/{media_id}/thumbnail?region=x0,y0,x1,y1
```

A downscaled image bounded to a fixed longest side, independent of zoom, with an
`ETag` so the browser reuses it across scrolls. Grid and list tiles use it
instead of `/image` so a gallery doesn't decode every full-size bitmap.
Optional `region` (normalised fractions) crops to a sub-region, so the Good pile can show
a region-voted item's crop. 400 if the media is not an image and its
`image_response` hook yields nothing; 404 if not found or bytes unavailable.

---

## Sorting

### Sort response shape (windowing)

The sorts that rank the whole dataset ([text sort](#text-sort),
[learned sort](#learned-sort), [example sort (upload)](#example-sort-upload)
and [label-file sort](#label-file-sort)) return a **windowed envelope**, not a
bare `{results, threshold}` pair:

| Field | Meaning |
|-------|---------|
| `results` | Ranking rows, **descending by score**; possibly only a *head window* of the full ranking. |
| `threshold` | The decision line (see [learned sort](#learned-sort) for `threshold` vs `acq_threshold`). |
| `acq_threshold` | The acquisition cut Autopilot's Hard / New picks sample around. Set by the learned and text sorts; `null` on the example and label-file sorts, where a client falls back to `threshold`. |
| `sort_token` | Opaque handle for [`GET /api/sort/page`](#sort-page); a re-sort mints a new one. |
| `total` | Length of the **full** ranking (`>= results.length`). |
| `above_threshold` | Rows at or above `threshold` across the full ranking. |
| `has_more_below` | `true` when `results` is a head window and more rows follow. |

**Windowing only engages on large sorts.** Below `SORT_WINDOW_THRESHOLD`
(20 000 rows, `vtscore/state/sort_results_cache.py`) the full ranking is sent
and `has_more_below` is `false`. At or above it the response carries up to
`SORT_WINDOW_HEAD` (500) above-threshold rows plus `SORT_WINDOW_TAIL` (200) rows
just past the boundary, and the client pages the rest. A client that ignores
`has_more_below` therefore gets a **silently truncated ranking** on large
datasets.

The full ranking is held server-side in a process-global LRU of the 8 most
recent sorts. It holds only `{id, score}` / `{id, similarity}` rows, never
embeddings or model weights, and nothing is persisted.

### Sort page

```
GET /api/sort/page?token=<sort_token>&offset=0&limit=200
```

Returns one window of a cached ranking. `offset` ≥ 0 (default 0); `limit`
1–2000 (default 200). → `{results, offset, limit, total, threshold, has_more}`;
page until `has_more` is `false` to get the whole order.

404 when the token is unknown, evicted, or belongs to a dataset other than the
active `X-Dataset-Id`; in every case, re-run the sort and use the new token.

### Text sort

```
POST /api/sort          Body: {"text": "dog barking"}
```

Embeds the query with the dataset's embedder and ranks every media by cosine
similarity. → A [windowed response](#sort-response-shape-windowing) with rows
`{id, similarity}` and two lines: `threshold`, the **display** line (the
mixture midpoint only when the two fitted components are well separated, else
the bulk's median + 3 robust sigmas; `VTSEARCH_TEXT_SORT_CUT=gmm_midpoint`
restores the plain midpoint), and `acq_threshold`, always the mixture midpoint,
which is where Autopilot's Bad phase samples. The guarded line is the better
one to paint and the worse place to sample negatives; see
[`docs/ML.md`](../ML.md#threshold-calibration).

On a patch-region-aware embedder (e.g. `dinov3_patch`) each row also carries
`best_region: [x0, y0, x1, y1]`, the normalised box of the best-matching
region; the gallery outlines it unless it is the whole image.

400 with `{"supports_text": false, ...}` when the dataset's embedder can't
embed text.

### Text sort progress (SSE)

Text-sort progress streams on the `sort` channel of
[`/api/events`](events.md): `{status, message, current, total}`, with `status`
`"idle"` or `"sorting"`.

### Learned sort

```
POST /api/learned-sort          Body: {"wait": false}
```

Trains the detector head on the current good/bad votes and scores every media.
Needs at least one good and one bad vote (400 otherwise).

**Asynchronous by default.** Training is GIL-bound, so the call hands it to a
background thread and returns `{job_id, status: "running", current, total}`.
Poll [`GET /api/learned-sort/result`](#learned-sort-result-poll) until
`status == "done"`. A call whose votes, detector, balance and threshold settings
are unchanged since the last successful run short-circuits to the cached `done`
payload. `{"wait": true}` blocks and returns the result inline (tests use it;
the frontend does not).

The `done` payload is the [windowed envelope](#sort-response-shape-windowing)
with rows `{id, score}`, plus `balance`: the
[line state](labeling.md#the-line-state) of `threshold` (the set the line keeps
and what the spot check found on it). This is the only sort with a detector
behind it, so the only one whose `balance` is non-`null`.

`threshold` is the **decision line**: what the user sees, what
`above_threshold` counts against, and what Find calls a match. `acq_threshold`
is the **acquisition cut**: Autopilot's Hard and New picks read a threshold as
a *rank position*, so they sample around a cut taken four inclusion steps below
the reporting one (`ACQUISITION_INCLUSION_OFFSET`), higher in the ranking.
Nothing shown to the user reads it. See
[`docs/ML.md`](../ML.md#threshold-calibration) for the mechanism and the
measurement behind the offset.

On patch datasets the head is max-pooled over each image's score rows (the
image-level vector plus every patch), and each result carries `best_region` for
the winning row. Region-annotated Good votes train on the patch nearest the
user's box; Bad votes cover the whole stack. Design:
[`docs/plans/patch-embedder.md`](../plans/patch-embedder.md).

#### Learned sort result (poll)

```
GET /api/learned-sort/result?job_id=<id>
```

- Running: `{job_id, status: "running", current, total}`
- Done: the windowed envelope plus `job_id` and `status`.
- Cancelled: `{job_id, status: "cancelled"}`
- Job failed: 500. Unknown `job_id`: 404.

#### Cancel learned sort

```
POST /api/learned-sort/cancel/{job_id}
```

Sets the job's cancel flag; the training loop polls it cooperatively. Returns
`{"ok": true}` even when the job has already finished (the contract is "make
sure it's no longer running"). Unknown `job_id`: 404.

### Example sort (upload)

```
POST /api/example-sort          Form: file
```

Embeds the uploaded file and ranks by cosine similarity. → A
[windowed response](#sort-response-shape-windowing) with rows `{id, similarity}`
(plus `best_region` on patch-region-aware datasets).

### Example sort (by loaded media id)

```
POST /api/example-sort-by-id          Body: {"media_id": 42, "crop_params": {...}}
```

Ranks by similarity to an already-loaded media. Without `crop_params` the
media's stored embedding is reused; with it, the bytes are materialised,
cropped and re-embedded. Powers the right-click "sort by similarity" / "crop
then sort" actions. 400 if nothing is loaded or `media_id` isn't in the loaded
snapshot; 404 if the bytes are unavailable when cropping.

The three `example-sort-{by-id,server,origin}` routes are the exception to the
windowed envelope: they return the plain `{results, threshold}` pair with the
full ranking and mint no `sort_token`.

### Example sort (server files)

```
POST /api/example-sort-server          Body: {"filenames": ["example.wav"], "crop_params": {...}}
```

Like example sort, but the query is one or more files already in the user's
`example_media/` directory. With several files the dataset is ranked against
the centroid of their L2-normalised embeddings (how Autopilot's Good phase sorts
for a detector seeded with several examples). On a structural (SIFT/VLAD)
dataset that order is then geometrically re-ranked against *every* example as a
template, max over templates, so crops of different marks each surface their
own instances. `crop_params` describes a single example, so it is rejected
(400) with more than one filename.

### List server media files

```
GET /api/server-media-files
```

→ `{files: [{name, filename, size_bytes}]}`: the files in the user's
`example_media/` directory.

### Example sort (origin)

```
POST /api/example-sort-origin          Body: {"origin": {"importer": "server_folder", "params": {...}}, "key": "subdir/a.wav"}
```

Ranks by similarity to a file resolved from an origin dict (optionally cropped
with `crop_params`).

### Upload server media file

```
POST /api/server-media-files/upload          Form: file, crop_params?, media_type?
```

Saves a file into the user's `example_media/` directory. With `crop_params`
(audio `{"start", "end"}`, image `{"box": [...]}`) the file is cropped
server-side first, so the saved example *is* the sub-region; `media_type`
(`"audio"` / `"image"`) is then required to pick the clipper.

→ 201 `{filename, original_name}`: `filename` is the server-generated UUID name
(the persistence key), `original_name` the user's name for display. 400 if the
file is missing or `crop_params` is invalid for the media type.

### Save loaded media as a server example file

```
POST /api/server-media-files/from-media-id          Body: {"media_id": 42, "crop_params": {...}}
```

Materialises a loaded media's bytes (optionally cropped) into `example_media/`
so the new-detector form can use it as a seed. → 201 `{filename, original_name}`.
400 if the media isn't loaded or `crop_params` is invalid; 404 if the bytes are
unavailable.

### Server media file thumbnail

```
GET /api/server-media-files/{filename}/thumbnail
```

A preview of an `example_media/` file (the image itself, an audio waveform PNG,
or a video mid-frame PNG). 400 if the filename escapes the directory, 404 if not
found or the type has no thumbnail, 500 if generation failed.

### Seed importers

A **seed importer** contributes a *batch* of unlabeled seed media ("close but
not quite" what the user is hunting for) to a new blank detector. None ships
in-tree; the family is an extension point (see
[`EXTENDING-plugins.md` § Adding a Seed Importer](../EXTENDING-plugins.md#adding-a-seed-importer)).

```
GET /api/seed-importers
```

→ `{importers: [...]}`, each with its `fields` and `max_items` cap; empty on a
vanilla install.

```
POST /api/seed-import/{importer_name}
```

**Body:** the named importer's declared fields (JSON, or multipart when it
declares a `file` field); see
[Routes with no typed schema](../API.md#routes-with-no-typed-schema). Runs the
importer and saves each item's bytes into `example_media/`.

→ `{items: [{filename, original_name, origin}], count, truncated}`. Each item
plugs into the detector-example model as
`{"type": "media", "value": <filename>, "labeled": false}`; `labeled: false` is
what keeps a seed a query rather than a Good vote (see
[Register detector](detectors.md#register-detector)). `truncated` is `true` when
the importer exceeded `max_items` and the tail was dropped. `count: 0` is a
valid "nothing matched", not an error.

400 (bad user input), 404 (unknown importer), 422 (missing/invalid field), 501
(`run` not implemented), 502 (source failure).

```
POST /api/seed-import/{importer_name}/options
```

Dynamic select options, same contract as the dataset-importer variant.

### Datasource importers

A **datasource importer** is the single-item sibling of a dataset importer: it
fetches one media item (from a URL, a server path, a third-party service) into
`example_media/`. It powers the example-media picker in the New Detector modal.
Plugins live in `vtscore.datasource_importers`.

```
GET /api/datasource-importers
```

→ `{importers, tabs}`. `tabs` are the dataset-importer picker tabs; datasource
importers use the same category ids so both families share one tab bar.

```
POST /api/datasource-import/{importer_name}
```

**Body:** the importer's declared fields (JSON, or multipart with a `file`
field); see [Routes with no typed schema](../API.md#routes-with-no-typed-schema).

→ 201 `{filename, original_name, origin}`, the same contract as
[upload](#upload-server-media-file), so the result plugs into a
`{"type": "media", "value": <filename>}` detector example unchanged. `origin`
is the item's durable origin when the importer reports one (`null` otherwise);
store it on the example so the item stays re-fetchable after the cached file is
gone.

400 (bad user input), 404 (unknown importer), 422 (missing/invalid field), 501
(`fetch` not implemented), 502 (source failure, or no data returned).

```
POST /api/datasource-import/{importer_name}/options
```

Dynamic select options, same contract as the dataset-importer variant.

### Label-file sort

```
POST /api/label-file-sort          Form: file
```

`file` is a JSON file with a `labels` array; each entry has `label` (`"good"` /
`"bad"`) and a `path` / `file` / `filename` naming a media file on the server.
Embeds those files with the dataset's embedder, trains the detector head on
them, and scores every loaded media.

→ A [windowed response](#sort-response-shape-windowing) plus `loaded` and
`skipped` counts for the label file. 400 when fewer than two entries load or
the good/bad split is missing.

---

## Votes & Labels

### Get votes

```
GET /api/votes
```

→ `{good, bad, verified, click_times, learned_scores, labelset_good_count,
labelset_bad_count, good_region_boxes}`; every key is always present.

- `verified`: ids the human explicitly verified this session (Find mode splits
  them from the unverified work queue); empty outside Find mode.
- `labelset_good_count` / `labelset_bad_count`: the **active detector's**
  persisted label counts, which include elements that don't resolve into the
  loaded dataset, so they can exceed `good.length` / `bad.length`. They fall back
  to the session vote counts when no detector is active.
- `good_region_boxes`: media id (string) → normalised box, for good votes cast by
  drawing on the image only.

### Clear votes

```
POST /api/votes/clear
```

Clears all good/bad votes without unloading the dataset (the Label flow does
this before importing a model's labelset). → `{"ok": true}`

### Text-sort suggestions

```
GET  /api/textsort-suggestions          → {suggestions: [...]}
POST /api/textsort-suggestions          Body: {"text": "dog barking"} → {"ok": true}
```

Stores and lists past text-sort queries, offered as suggested names for
detectors and labelsets.

### Export labels

```
GET /api/labels/export
```

**Query (all optional):**

- `goods_only=1`: export only good labels.
- `label_filter=<mode>`: `good`, `bad`, `both` (default), `corrections`
  (entries where the user changed the detector's original label), `unverified`
  (Find work-queue items the human hasn't acted on), or `verified`. Overrides
  `goods_only`. The session-scoped modes (`corrections` / `unverified` /
  `verified`) never include `origin_only` entries.
- `enrich=1`: add per-entry `custom_metadata` and a top-level
  `available_columns` list (with `origin.params` flattened; see
  [Batch fetch](#batch-fetch-metadata)).
- `detector_name=<name>`: export **that** detector's persisted labelset from its
  JSON file instead of the active pair's live labels, independent of the context
  headers and any Find session (the Dashboard's row action uses it). The
  session-scoped `label_filter` modes are refused with 400 here; an unknown name
  is 404.
- `format=ndjson`: stream `application/x-ndjson`, one entry per line, instead of
  the buffered `{"labels": [...]}` object, for exports too large to build in
  memory. `available_columns` is omitted in this mode.

→ A LabelSet: `{labels: [...]}`. Only `md5` and `label` are guaranteed on an
entry; `origin`, `origin_name`, `filename`, `category`, `metadata` and
`region_box` appear when the element has them. The export renders the
**detector's** labelset, not just the session's votes: elements that don't
resolve into the active dataset are appended with `"origin_only": true`, and
entries the user corrected carry `"is_correction": true` (never under
`detector_name`, whose rows belong to a detector the live session says nothing
about).

### Import labels

```
POST /api/labels/import          Body: {"labels": [{"origin": {...}, "origin_name": "...", "md5": "...", "label": "good"}]}
```

Matches by origin + `origin_name` first, then falls back to MD5. An entry whose
`label` is neither `"good"` nor `"bad"`, or that resolves to no loaded media, is
counted in `skipped` rather than failing the request. A `region_box` on a good
entry is restored onto the vote. → `{applied, skipped}`

### Upload media to pile

```
POST /api/medias/add-to-pile          Form: file, label ("good" | "bad")
```

Adds an uploaded file to the Good or Bad pile. If a media with the same MD5 is
already loaded it is just voted; otherwise the file is embedded with the
dataset's embedder, inserted as a new media, and voted.

→ `{ok, media_id, is_new}`: 201 when new, 200 when it already existed. 400 for
a missing/empty file, an invalid label, no dataset loaded, no embedder for the
dataset, or an embedding failure.

### Fill labels from sort results

```
POST /api/labels/fill-from-sort          Body: {"sort_results": [{"id": 0, "score": 0.8}], "threshold": 0.5, "sides": "good", "confirm": false}
```

Labels the currently **unlabeled** items on one or both sides of `threshold`
(`sides`: `"good"`, `"bad"` or `"both"`); existing votes are left alone. With
`confirm: false` (the default) it is a dry run returning
`{good_count, bad_count}`; with `confirm: true` it applies the labels and
returns `{good_applied, bad_applied, results}`, where `results` is a dict any
[exporter](io.md) accepts. A persistence failure rolls the votes back and
returns 500.
