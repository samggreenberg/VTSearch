# Import & Export

[← Back to API index](../API.md)

---

## Exporters

### List exporters

```
GET /api/exporters
```

→ JSON array of exporter objects, each with `name`, `display_name`,
`description`, and `fields`.

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
`email_smtp`, `gui`.

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

**Body:** `{"field_key": "...", "field_values": {...}}`

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

→ ```json
{
  "applied": 8,
  "skipped": 2,
  "missing_count": 3,
  "missing": [...],
  "ingested": 1,
  "message": "Applied 8 label(s), skipped 2. Auto-resolved 1 missing element(s) from their sources. 3 element(s) could not be resolved."
}
```

When `missing_count > 0`, the frontend can call `ingest-missing` to pull those
medias from their origins.

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
localizer (MediaPipe). Adding them registers each into the autorun
extractors / localizers stores so they run automatically after a dataset
loads. They do **not** create detectors.

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

→ `{"success": true, "added": ["OCR (PaddleOCR)", "Speech (Whisper Tiny)", "Face (MediaPipe)"]}`

---

## Autorun Extractors

Autorun extractors run after a dataset loads and attach free-form metadata
records to each media (e.g. OCR text, speech transcripts, image classes).

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

Autorun localizers run after a dataset loads and attach a list of
`(box, confidence)` regions to each media (e.g. face detection).

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
