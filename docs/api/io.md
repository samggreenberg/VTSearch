# Import & Export

[← Back to API index](../API.md)

---

## Exporters

### List exporters

```
GET /api/exporters
```

→ JSON array of exporter objects, each with `name`, `display_name`,
`description`, `fields`, and `opens_url` (see below).

### Run export

```
POST /api/exporters/export
```

**Body:**

```json
{
  "exporter_name": "server_json_file",
  "field_values": {"filepath": "/home/user/results.json"},
  "results": {}
}
```

→ `{"success": true, "message": "...", ...}`

Available built-in exporters: `server_json_file`, `server_csv_file`, `webhook`,
`email_smtp`, `gui`, `open_url`.

**`open_url`** — an exporter may return an `open_url` key, an `http(s)` URL the
frontend opens in a new browser tab. It is how a third-party site with no ingest
API receives a labelset: the exporter formats the selection into that site's own
URL. The handler re-validates the URL against a scheme allowlist
(`vtscore.security.url_validation.validate_browser_url`) and returns 500 if it
fails, so no plugin can push a `javascript:` URL to the browser. An exporter that
*always* returns one sets `opens_url: true` in its `GET /api/exporters` entry, so
the UI can label the button before the export runs.

The same key reaches the client from the Auto-Find auto-export block
(`auto_export.open_url` on `POST /api/auto-detect`), where the Auto-Detect
Results modal offers it as an **Open** button instead of opening a tab on
arrival. That path runs the same `validate_browser_url` check, but drops an
unusable URL and notes it in `message` rather than failing the request: the
export already happened and the scored results must survive it.

### Dynamic field options

```
POST /api/exporters/field-options/{exporter_name}
```

**Body:** `{"field_key": "...", "values": {...}}` (`values` is a snapshot of
the form's current field values)

Returns the dropdown options for a `dynamic_options` field on a results
exporter, for an exporter whose destinations are only knowable at runtime.
Both surfaces that render exporter fields use it: the Export modal and the
Settings › Auto-Find results exporter.

→ `{"options": [{"value": "...", "label": "..."}, ...]}` (same shape as the
label-importer route below).

Errors: 400 (unknown/non-dynamic field key), 404 (unknown exporter),
501 (exporter does not implement `get_field_options`),
502 (remote service backing dynamic options failed).

**Streaming support** (CLI `--autodetect --stream-results` for sources larger
than RAM): `server_json_file` (NDJSON), `server_csv_file`, and `gui` write hits
incrementally; `webhook` and `email_smtp` deliver in `batch_size`-sized batches
(one POST / one email per batch) so they too stay bounded. This applies to the
CLI streaming path only — the `POST /api/exporters/export` route above always
receives a fully-materialised results dict. See [CLI.md](../CLI.md) and
`docs/plans/cli-stream-massive-images.md`.

---

## Label Importers

### List label importers

```
GET /api/label-importers
```

→ JSON array of label importer objects.

### Dynamic field options

```
POST /api/label-importers/field-options/{importer_name}
```

**Body:** `{"field_key": "...", "values": {...}}` (`values` is a snapshot of
the form's current field values)

Returns the dropdown options for a dynamic-options field on a label importer
(used to populate dependent selects in the importer form).

→ `{"options": [{"value": "...", "label": "..."}, ...]}` (each option carries a
`value` to submit and a `label` to display; they coincide for plain-string
options and differ for `(value, label)` tuples).

Errors: 400 (unknown/non-dynamic field key), 404 (unknown importer),
501 (importer does not implement `get_field_options`),
502 (remote service backing dynamic options failed).

### Run label import

```
POST /api/label-importers/import/{importer_name}
```

**Form or Body:** importer-specific fields.

→
```json
{
  "applied": 8,
  "skipped": 2,
  "missing_count": 0,
  "missing": [],
  "ingest_task_id": "_labelingest_<detector_id>",
  "ingest_pending_count": 3,
  "failed_count": 0,
  "failed": [],
  "message": "Applied 8 label(s), skipped 2. Resolving 3 missing element(s) from their sources in the background…"
}
```

`applied` / `skipped` describe only the entries that matched media already in
the active dataset. Entries that matched nothing are auto-resolved from their
origins on a **background task** — one fetch + embed per entry is far too slow
to run inside the request. `ingest_pending_count` is how many were handed off
and `ingest_task_id` names the task on the `detector-loading-tasks` SSE
channel; `missing_count` / `missing` stay empty because nothing is known to be
unresolvable until that task finishes. Both are `""` / `0` when every entry
matched.

Watch the task on `/api/events`; its terminal frame carries

```json
"ingest_result": {"ingested": 3, "applied": 3, "unresolved": 0, "failed": 0}
```

The task re-applies the labels of whatever it ingested and re-syncs the loaded
detector, so no follow-up call is needed. Cancel it with
`POST /api/detectors/cancel/{task_id}`.

### Ingest missing medias

```
POST /api/label-importers/ingest-missing
```

**Body:** `{"entries": [...]}`

Re-ingests medias from their recorded origins and applies labels.

→ `{"ingested": 3, "applied": 3, "message": "Ingested 3 media(s), applied 3 label(s)."}`

---

## Pregen Processors

Pregen processors are the built-in autorun processors VTSearch ships with: an
OCR extractor (PaddleOCR), a Speech extractor (Whisper Tiny), and a Face
localizer (MTCNN). Adding them registers each into the autorun
extractors / localizers stores, from which `POST /api/auto-extract` and
`POST /api/auto-localize` run them over the loaded medias. They do **not**
create detectors.

### List pregen processors

```
GET /api/pregen-processors
```

→ `{"processors": [{"name": "OCR (PaddleOCR)", "kind": "extractor", "processor_type": "ocr", "media_type": "image", "config": {...}}, ...]}`

### Add all pregen processors

```
POST /api/pregen-processors/add
```

Registers every bundled pregen processor (OCR extractor, Speech extractor,
Face localizer) into the autorun extractor / localizer stores.

→ `{"success": true, "added": ["OCR (PaddleOCR)", "Speech (Whisper Tiny)", "Face (MTCNN)"]}`

---

## Autorun Extractors

Autorun extractors pull free-form metadata records out of each media (e.g. OCR
text, speech transcripts, image classes). Registering one stores its type and
config; `POST /api/auto-extract` is what builds and runs every stored extractor
matching the loaded medias' type. The store is in-memory and does not survive
a restart.

Which extractor types exist is fixed in app code (`ocr`, `speech`,
`image_class`); adding a new one means editing the factory dict in
`vtsearch/routes/processors/crud.py` — see
[EXTENDING-processors.md](../EXTENDING-processors.md#registering-a-processor-with-the-app).

### List autorun extractors

```
GET /api/autorun-extractors
```

→ `{"extractors": [...]}`

### Add an autorun extractor

```
POST /api/autorun-extractors
```

**Body:** `{"name": "...", "extractor_type": "ocr"|"speech"|"image_class", "media_type": "image", "config": {...}}`

→ `{"success": true, "name": "..."}` (400 if the config can't be built).

### Delete an autorun extractor

```
DELETE /api/autorun-extractors/{name}
```

→ `{"success": true}` (404 if not found).

### Rename an autorun extractor

```
PUT /api/autorun-extractors/{name}/rename
```

**Body:** `{"new_name": "..."}`

→ `{"success": true, "new_name": "..."}` (400 if not found or name taken).

---

## Autorun Localizers

Autorun localizers find `(box, confidence)` regions inside each media (e.g.
face detection). As with extractors, registering one only stores its type and
config in an in-memory store; `POST /api/auto-localize` builds and runs them.
`face` is the only localizer type in the factory dict.

### List autorun localizers

```
GET /api/autorun-localizers
```

→ `{"localizers": [...]}`

### Add an autorun localizer

```
POST /api/autorun-localizers
```

**Body:** `{"name": "...", "localizer_type": "face", "media_type": "image", "config": {...}}`

→ `{"success": true, "name": "..."}` (400 if the config can't be built).

### Delete an autorun localizer

```
DELETE /api/autorun-localizers/{name}
```

→ `{"success": true}` (404 if not found).

### Rename an autorun localizer

```
PUT /api/autorun-localizers/{name}/rename
```

**Body:** `{"new_name": "..."}`

→ `{"success": true, "new_name": "..."}` (400 if not found or name taken).

---

## Settings Importers & Exporters

### List settings importers

```
GET /api/settings-importers
```

→ JSON array of settings importer objects.

### Run settings import

```
POST /api/settings-importers/import/{importer_name}
```

**Form or Body:** importer-specific fields.

→ `{"ok": true, "message": "..."}`

### List settings exporters

```
GET /api/settings-exporters
```

→ JSON array of settings exporter objects.

### Run settings export

```
POST /api/settings-exporters/export
```

**Body:** `{"exporter_name": "...", "field_values": {...}}`

→ `{"ok": true, "message": "..."}`
