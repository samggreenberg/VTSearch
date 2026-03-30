# Medias & Sorting

[← Back to API index](../API.md)

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

→ `{"files": [{"name": "example", "filename": "example.wav", "size_bytes": 160044}]}`

### Example sort (origin)

```
POST /api/example-sort-origin
```

**Body:** `{"origin": {"importer": "folder", "params": {"path": "/data/sounds"}}, "key": "subdir/audio123.wav"}`

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

### Batch media metadata

```
POST /api/medias/batch
```

**Body:** `{"ids": [0, 1, 2], "offset": 0, "limit": 50}`

Paginated media metadata retrieval.

→ `{"medias": [...], "total": 500}`

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
