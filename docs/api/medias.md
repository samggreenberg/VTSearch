# Medias & Sorting

[← Back to API index](../API.md)

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

Every stub carries `id` and `type`; `embedder` is included when the media
has one.  Display-worthy metadata (`filename`, `md5`, `custom_metadata`,
`origin_name`, `description`, `clip_*`) is fetched on demand for the IDs
the client actually needs via [Batch fetch](#batch-fetch-metadata) — this
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

Every returned item contains `id`, `type`, `filename`, `md5`, and
`custom_metadata`.  The `custom_metadata` dict holds media-type-specific
display fields (e.g.  `duration`/`frequency` for audio, `width`/`height`
for images, `word_count` for text).  `origin_name`, `description`,
`embedder`, and `clip_*` keys are included when present.

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
`"best_region": [x0, y0, x1, y1]` — the normalised box of the
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

→ `{"results": [{"id": 0, "score": 0.9234}], "threshold": 0.5123}`

On patch-region-aware datasets the MLP is max-pooled over each
image's region tree, and each result carries `"best_region": [x0,
y0, x1, y1]` for the region whose score won. Region-annotated Good
votes (`region_box` on `LabeledElement`) pool the user's box from
the patch grid at training time; Bad votes use a region-aware
asymmetric loss. See [`docs/plans/patch-embedder.md`](../plans/patch-embedder.md)
for the design.

### Example sort (upload)

```
POST /api/example-sort
```

**Form:** `file` — media file to use as the query example.

Embeds the uploaded file and sorts by cosine similarity.

→ `{"results": [{"id": 0, "similarity": 0.8234}], "threshold": 0.5123}`

`best_region` is included per-result on patch-region-aware datasets,
same shape as text sort.

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

**Form:** `file` — media file to upload.

→ `{"filename": "abc123.wav", "original_name": "dog_bark.wav"}` (201)

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

### Upload media to pile

```
POST /api/medias/add-to-pile
```

**Form:**
- `file` — the media file to upload.
- `label` — `"good"` or `"bad"`.

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
