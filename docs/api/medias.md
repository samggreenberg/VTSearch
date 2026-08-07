# Medias & Sorting

[← Back to API index](../API.md)

> Media, vote, and sort endpoints are scoped to the active dataset/detector via
> the [`X-Dataset-Id` / `X-Detector-Id` context headers](../API.md#context-headers-x-dataset-id--x-detector-id).
> Vote- and label-mutating routes **require** them (400 otherwise).

---

## Medias

### List media IDs

```
GET /api/medias/ids
```

→ Lightweight JSON array of stubs, one per media in the loaded dataset:

```json
[
  { "id": 0, "media_type": "audio", "embedder": "clap-fused" },
  { "id": 1, "media_type": "audio", "embedder": "clap-fused" }
]
```

Every stub carries `id` and `media_type`; `embedder` (and the plural
`embedders` array, when a media was embedded by more than one embedder, e.g.
a semantic + region-patch pair) is included when present.  Display-worthy
metadata (`filename`, `md5`, `custom_metadata`,
`origin_name`, `description`, `clip_*`) is fetched on demand for the IDs
the client actually needs via [Batch fetch](#batch-fetch-metadata); this
keeps the listing payload bounded even for datasets with tens of
thousands of items.

### Batch fetch metadata

```
POST /api/medias/batch
Content-Type: application/json

{ "ids": [0, 1, 2] }
```

→ JSON array of full metadata objects for the requested IDs (unknown IDs
are silently omitted):

```json
[
  {
    "id": 0,
    "media_type": "audio",
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

Every returned item contains `id`, `media_type`, `filename`, `md5`, and
`custom_metadata`.  `origin_name`, `description`, `embedder`, `embedders`
(plural array, when present), `has_original`, and `clip_*` keys are included
when present.

`has_original: true` marks an item a
[MediaCleaner](../EXTENDING-media.md#adding-a-media-cleaner) rewrote at load
time, whose pre-clean payload was kept alongside the canonical (cleaned) one.
Those items accept `?variant=original` on every payload route below, and the
detail viewer offers a Clean/Original toggle. The key is absent otherwise.

The `custom_metadata` dict is the media type's display fields — e.g.
`duration`/`frequency` for audio, `width`/`height` for images, `word_count`
for text — with any importer-supplied `custom_metadata` layered on top.

It also carries up to three curated **provenance** lines distilled from the
media's `origin.params`:

| Field | Present on | Example |
|-------|-----------|---------|
| `Source` | Converter / clipper output | `/data/videos/movie.mp4` |
| `Derived Via` | Converter / clipper output | `Video → Images (n_clips=2)` |
| `Imported Via` | Any media whose origin names an importer | `Manifest (paths_file=/data/list.txt)` |

`Source` is the original file the item came from — the video an extracted
frame was cut from, the recording an audio clip was sliced out of.  A plainly
imported file gets neither `Source` nor `Derived Via`; it is its own source.

Each is one line rather than a key-per-`origin.params`-entry, because a
dataset-level import knob (`size=60`) is not a fact about one item and reads
wrong in a per-item grid.  The machine-only replay recipe
(`converter_content_hash`, `converter_out_index`, `clipper_chain`,
`converter_param_*`, …) is folded into these lines rather than listed raw.
The enriched label export (`GET /api/labels/export?enrich=true`) does
flatten the *full* `origin.params` key-by-key — an export is a machine-facing
artifact with opt-in columns, where the raw recipe is the point.

### Payload variants (`?variant=original`)

Every per-media payload route below — `/audio`, `/video`, `/image`,
`/thumbnail`, `/text`, `/paragraph`, `/media` — accepts an optional
`variant` query:

| `variant` | Serves |
|-----------|--------|
| omitted / `""` | The **canonical** payload: the cleaned bytes that were actually hashed, embedded, and scored. |
| `original` | The pre-clean payload of an item a cleaner rewrote at load time. |

`?variant=original` on an item with no snapshot (`has_original` absent) falls
back to the canonical payload rather than 404ing, so a stale link still shows
the item. Any other value is rejected with `422`.

Derived metadata is recomputed from what is actually served rather than reused
from the canonical item: the `original` variant regenerates the thumbnail
(the stored one describes the cleaned bytes), recounts `word_count` /
`character_count` for text, and hashes the served bytes for its `ETag`.

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

→ Video binary stream (`video/mp4`, `video/webm`, or `video/ogg` based on
filename extension). Non-browser-playable formats are transcoded to MP4.
400 if not a video. 404 if not found.

### Stream image

```
GET /api/medias/{media_id}/image
```

→ Image binary stream (`image/jpeg`, `image/png`, `image/gif`, `image/webp`, or
`image/bmp` based on filename extension).
400 if not an image. 404 if not found.

### Get text content

```
GET /api/medias/{media_id}/paragraph
GET /api/medias/{media_id}/text
```

Both paths serve the same handler. Returns the text content and statistics
for a text media item.

→ `{"content": "...", "word_count": 150, "character_count": 900}`
400 if not a text media. 404 if not found.

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

**Optional `region_box`** (yes-votes only): a 4-float array
`[x0, y0, x1, y1]` in normalised image coordinates (`0..1`,
pre-rotation) that annotates *which region of the image* the user
is voting good on. Persisted alongside the vote and consumed by
region-aware MLP training (the trainer pools the box's patch-grid
cells on the fly). The box is dropped when the vote is toggled off
or switched good → bad.

```json
{"vote": "good", "region_box": [0.2, 0.3, 0.55, 0.7]}
```

→ `{"ok": true}`

| Status | Cause |
|--------|-------|
| `400` | Invalid `vote` value; `region_box` on a `bad` vote; `region_box` outside `[0, 1]` or not a 4-tuple. |
| `404` | Media not found. |

### Bulk vote

```
POST /api/medias/vote-bulk
```

**Body:** `{"ids": [1, 2, 3], "target": "good"}`

Applies one absolute vote `target` (`"good"` / `"bad"`) to many medias in a
single request, with the same idempotent semantics as the per-media vote
(including Find-mode verification: a good/bad target marks the item verified).
The detector labelset is persisted once rather than per id. Bulk votes are
image-level (no region boxes). Powers the Browser's "Verified Good" /
"Verified Bad" actions.

→ `{"changed": 2, "missing": [3]}` — `changed` counts only ids whose state
actually moved; ids not in the loaded dataset are reported in `missing`.
400 if no ids supplied.

### Thumbnail

```
GET /api/medias/{media_id}/thumbnail
```

**Query (optional):** `region=x0,y0,x1,y1` (normalised fractions in `[0, 1]`)
crops the thumbnail to a sub-region (used so the Good pile shows a
region-voted item's crop rather than the whole frame).

Streams a downscaled thumbnail bounded to a fixed longest-side length, the
same regardless of zoom level (an `ETag` lets the browser reuse it across
scrolls/zoom). Grid and list tiles use this instead of `/image` so a gallery
of high-resolution items doesn't decode every full-size bitmap at once.
400 if the media is not an image and has no `image_response` delegate. 404 if
not found or bytes unavailable.

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

When the dataset's embedder is patch-region-aware (e.g.
`dinov3_patch`), each result additionally carries
`"best_region": [x0, y0, x1, y1]`: the normalised box of the
region whose vector matched best against the query, used by the
gallery card to draw a faint outline. Boxes that cover the full
image (the single-vector fallback `[0, 0, 1, 1]`) are suppressed by
the frontend.

Returns HTTP 400 + `{"supports_text": false, ...}` when the dataset's
embedder doesn't support text queries.

### Text sort progress (SSE)

Text-sort progress streams on the `sort` channel of
[`/api/events`](events.md):

```json
{"status": "sorting", "message": "Computing similarities…", "current": 50, "total": 100}
```

Status is `"idle"` or `"sorting"`.

### Learned sort

```
POST /api/learned-sort
```

Trains an MLP on the current good/bad votes and scores all medias. Requires at
least one good and one bad vote.

**Asynchronous by default.** Training is GIL-bound, so the endpoint hands the
work to a background thread and returns immediately:

→ `{"job_id": "…", "status": "running", "current": 0, "total": 1}`

Poll [`GET /api/learned-sort/result`](#learned-sort-result-poll) with that
`job_id` until `status == "done"` to receive the results. A no-op call (votes,
detector, inclusion, and threshold settings unchanged from the most recent
successful run) short-circuits and returns the cached `done` payload directly.

Pass `{"wait": true}` in the body to block until the job finishes and receive
the result inline (used by tests; the frontend leaves it `false`):

→ `{"results": [{"id": 0, "score": 0.9234}], "threshold": 0.5123, "acq_threshold": 0.6180}`

The `done` payload — whether returned inline (`wait=true`) or via the result
poll — carries `results`, `threshold` and `acq_threshold`.

`threshold` is the **decision line**: the cutoff shown to the user, what
`above_threshold` counts against, and what Find calls a match. `acq_threshold`
is the **acquisition cut**, and it is a different number — Autopilot's Hard and
New picks read a threshold as a *rank position* rather than a boundary, so they
sample around a cut taken three inclusion steps below the reporting one, which
places it higher in the ranking. Nothing shown to the user reads it. It is
`null` on sorts with no detector behind them (`/api/sort`, `/api/example-sort`,
`/api/label-file-sort`), where a client should fall back to `threshold`. See
[`docs/ML.md`](../ML.md#threshold-calibration) for the mechanism and the
measurement behind the offset.

#### Learned sort result (poll)

```
GET /api/learned-sort/result?job_id=<id>
```

Polls a background learned-sort job.

- Running: `{"job_id": "…", "status": "running", "current": N, "total": M}`
- Done: `{"results": [{"id": 0, "score": 0.9234}], "threshold": 0.5123, "acq_threshold": 0.6180}`
- Cancelled: `{"job_id": "…", "status": "cancelled"}`
- Job failed: HTTP 500.
- Unknown `job_id`: HTTP 404.

#### Cancel learned sort

```
POST /api/learned-sort/cancel/<job_id>
```

Sets the cancel flag on the job; the training loop polls it cooperatively.
Returns `{"ok": true}` (HTTP 200) even when the job has already finished — the
contract is "make sure it's no longer running". Unknown `job_id`: HTTP 404.

On patch datasets the MLP is max-pooled over each image's score-row
stack (the image-level vector plus every raw patch of its
`patch_grid`), and each result carries `"best_region": [x0, y0, x1,
y1]` for the row whose score won - the whole image when the
image-level row wins, otherwise the single winning grid cell.
Region-annotated Good votes (`region_box` on `LabeledElement`) train
on the raw patch nearest the user's box; Bad votes flood the whole
stack (a region-aware asymmetric loss). See [`docs/plans/patch-embedder.md`](../plans/patch-embedder.md)
for the design.

### Example sort (upload)

```
POST /api/example-sort
```

**Form:** `file`: media file to use as the query example.

Embeds the uploaded file and sorts by cosine similarity.

→ `{"results": [{"id": 0, "similarity": 0.8234}], "threshold": 0.5123}`

`best_region` is included per-result on patch-region-aware datasets,
same shape as text sort.

### Example sort (by loaded media id)

```
POST /api/example-sort-by-id
```

**Body:** `{"media_id": 42}` (optionally `{"media_id": 42, "crop_params": {...}}`)

Sorts all medias by similarity to an already-loaded media item. When
`crop_params` is absent the media's existing embedding vector is reused (no
fetch, no re-embed); when set, the media's bytes are materialised, cropped,
and re-embedded before sorting. Powers the right-click "sort by similarity" /
"crop then sort" context-menu actions.

→ `{"results": [...], "threshold": 0.5123}`

400 if no medias loaded or `media_id` not in the loaded snapshot. 404 if the
media's bytes are unavailable when cropping is requested.

### Example sort (server files)

```
POST /api/example-sort-server
```

**Body:** `{"filenames": ["example.wav"]}` (optionally with `"crop_params"`)

Same as example sort but uses one or more files already on the server in
`data/example_media/`. With multiple filenames the haystack is ranked
against the centroid (mean of the L2-normalised embeddings) of all
examples — this is how Autopilot's Good phase sorts for a detector seeded
with several media examples. `crop_params` describes a single example, so
it is rejected (400) when more than one filename is given.

→ `{"results": [...], "threshold": 0.5123}`

### List server media files

```
GET /api/server-media-files
```

→ `{"files": [{"name": "example", "filename": "example.wav", "size_bytes": 160044}]}`

### Example sort (origin)

```
POST /api/example-sort-origin
```

**Body:** `{"origin": {"importer": "server_folder", "params": {"path": "/data/sounds"}}, "key": "subdir/audio123.wav"}`

Sorts by similarity to a file resolved from an origin dict.

→ `{"results": [...], "threshold": 0.5123}`

### Upload server media file

```
POST /api/server-media-files/upload
```

**Form:** `file`: media file to upload.

→ `{"filename": "abc123.wav", "original_name": "dog_bark.wav"}` (201)

### Save loaded media as a server example file

```
POST /api/server-media-files/from-media-id
```

**Body:** `{"media_id": 42}` (optionally `{"media_id": 42, "crop_params": {...}}`
— e.g. audio `{"start", "end"}` or image `{"box": [...]}`).

Materialises a loaded media's bytes (optionally cropped) into the per-user
`example_media/` dir so the new-detector form can reference it as a seed.

→ `{"filename": "abc123.wav", "original_name": "dog_bark.wav"}` (201)

400 (media not loaded, or invalid `crop_params`), 404 (media bytes unavailable).

### Server media file thumbnail

```
GET /api/server-media-files/{filename}/thumbnail
```

Small preview image of an example file in the user's `example_media/` dir:
image bytes, an audio waveform PNG, or a video mid-frame PNG (binary, not JSON).

400 (filename escapes the media dir), 404 (not found / no thumbnail for the
type), 500 (generation failed).

### Seed votes from examples

```
POST /api/votes/seed-from-examples
```

Seeds good votes from the active model's media examples.

→ `{"ok": true, "seeded": 5}`

### Label-file sort

```
POST /api/label-file-sort
```

**Form:** `file`: JSON file with a `labels` array. Each entry has `label`
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

**Query:**
- `?goods_only=1` (optional): export only good labels.
- `?format=ndjson` (optional): stream the response as newline-delimited JSON
  (`application/x-ndjson`), one label entry per line, instead of the buffered
  `{"labels": [...]}` object. Use for large exports that shouldn't be
  materialised in memory server-side. The top-level `available_columns` list
  (see `enrich`) is omitted in this mode.

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

### Upload media to pile

```
POST /api/medias/add-to-pile
```

**Form:**
- `file`: the media file to upload.
- `label`: `"good"` or `"bad"`.

Uploads a media file and adds it to the Good or Bad pile. If a media with
the same MD5 already exists, the existing media is voted accordingly.
Otherwise, the file is embedded using the dataset's embedder, inserted as
a new media item, and then voted.

→ `{"ok": true, "media_id": 123, "is_new": true}` (201 if new, 200 if existing)
400 if no file, empty file, or invalid label. 400 if no dataset loaded.

---

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
