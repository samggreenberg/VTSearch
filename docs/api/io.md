# Import & Export

[← Back to API index](../API.md)

> Every `field-options` route here takes `{"field_key": "...", "values": {...}}`
> (`values` is a snapshot of the form's current field values) and answers
> `{"options": [{"value", "label"}, ...]}`: a `value` to submit and a `label`
> to display, which coincide for plain-string options and differ for
> `(value, label)` tuples. It fails 400 (unknown or non-dynamic field key), 404
> (unknown plugin), 500 (the plugin's answer is not a list), 501 (the plugin
> does not implement `get_field_options`) or 502 (the remote service behind the
> options failed).

---

## Exporters

### List exporters

```
GET /api/exporters
```

→ JSON array of exporter objects, each with `name`, `display_name`,
`description`, `icon`, `fields`, `ui_mode`, `hidden_from_picker`, `opens_url`
(see below), and `supported_payloads` — which of `find_results`, `labelset`,
`detector_bundles` the exporter can take. Operator-hidden plugins (the
`hidden_plugins` setting) are omitted; `hidden_from_picker` ones are listed
with the flag set.

### Run export

```
POST /api/exporters/export
```

**Body:** `{"exporter_name": "server_json_file", "field_values": {"filepath": "/data/results.json"}, "payload_kind": "find_results", "results": {...}}`

`payload_kind` says what `results` is: `find_results` (a scored run, e.g. the
`POST /api/auto-detect` response) or `labelset` (a serialised labelset, e.g.
the `GET /api/labels/export` response). When omitted it is inferred: a dict
with a top-level `labels` key is a labelset. A kind the exporter doesn't list in `supported_payloads` is a 400.
`detector_bundles` is refused with a 422: it is the trained classifiers
themselves, which only the CLI pipeline has, so this route cannot build one.
A `labelset` reaches the exporter without what this server's embedder and head
measured: each entry's `metadata["vt:provenance"]` loses `rank_at_vote` and
`score_at_vote`, and a `detector_meta` loses `threshold`. An exported labelset
is references to labels, so it can rebuild the detector under another embedder.

→ `{success, message, open_url, display_results}`: `success` always, the rest
only when the exporter returns them
(`display_results` is what the `gui` exporter hands back for in-app display).
400 (missing plugin field, invalid file path, unsupported payload kind, or the
exporter rejected its input), 404 (unknown exporter), 422 (unknown or
CLI-only `payload_kind`), 500 (exporter error).

Built-in exporters: `server_json_file`, `server_csv_file`, `webhook`,
`email_smtp`, `open_url`, `gui` (picker-hidden; returns results for display
rather than writing them), and `portable_detector` (picker-hidden; supports
only `detector_bundles`, which this route cannot produce — a `find_results`
or `labelset` request is a 400, a `detector_bundles` one a 422. Use
it from the CLI as `--autodetect --exporter portable_detector`, or export one
saved detector with
[`POST /api/detectors/{detector_id}/portable-bundle`](detectors.md#export-portable-bundle)).

**`open_url`** — an exporter may return an `open_url` key, an `http(s)` URL the
frontend opens in a new browser tab. It is how a third-party site with no ingest
API receives a labelset: the exporter formats the selection into that site's own
URL. The handler re-validates the URL against a scheme allowlist
(`vtscore.security.url_validation.validate_browser_url`) and returns 500 if it
fails, so no plugin can push a `javascript:` URL to the browser. An exporter that
*always* returns one sets `opens_url: true` in its `GET /api/exporters` entry, so
the UI can label the button before the export runs.

The same key reaches the client from the AutoFind auto-export block
(`auto_export.open_url` on `POST /api/auto-detect`), where the Auto-Detect
Results modal offers it as an **Open** button instead of opening a tab on
arrival. That path runs the same `validate_browser_url` check, but drops an
unusable URL and notes it in `message` rather than failing the request: the
export already happened and the scored results must survive it.

### Dynamic field options

```
POST /api/exporters/field-options/{exporter_name}
```

Options for a `dynamic_options` field on a results exporter, for an exporter
whose destinations are only knowable at runtime (contract in the note at the
top). Both surfaces that render exporter fields use it: the Export modal and
the Settings › AutoFind results exporter.

**Streaming support** (CLI `--autodetect --tempimport --stream-results` for sources larger
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

→ JSON array of label importer objects (`name`, `display_name`, `description`,
`icon`, `fields`, `ui_mode`, `hidden_from_picker`). `fields` is what the run
route below accepts.

### Dynamic field options

```
POST /api/label-importers/field-options/{importer_name}
```

Options for a dynamic-options field on a label importer, for the dependent
selects in the importer form (contract in the note at the top).

### Run label import

```
POST /api/label-importers/import/{importer_name}
```

**Requires** `X-Dataset-Id` **and** `X-Detector-Id`.
**Form or Body:** importer-specific fields.

→ `{applied, skipped, missing_count, missing, ingest_task_id,
ingest_pending_count, failed_count, failed, message}`

`applied` / `skipped` describe only the entries that matched media already in
the active dataset. Entries that matched nothing are auto-resolved from their
origins on a **background task** — one fetch + embed per entry is far too slow
to run inside the request. `ingest_pending_count` is how many were handed off
and `ingest_task_id` names the task (`_labelingest_<detector_id>`) on the
`detector-loading-tasks` SSE channel; the two are `""` / `0` when every entry
matched. `missing_count` / `missing` stay empty because nothing is known to be
unresolvable until that task finishes.

Watch the task on `/api/events`; its terminal frame carries `ingest_result`,
`{ingested, applied, unresolved, failed}`. The task re-applies the labels of whatever it ingested and re-syncs the loaded
detector, so no follow-up call is needed. Cancel it with
`POST /api/detectors/cancel/{task_id}`.

### Ingest missing medias

```
POST /api/label-importers/ingest-missing
```

**Body:** `{"entries": [...]}`

**Requires** `X-Dataset-Id` **and** `X-Detector-Id`.

Synchronously re-ingests medias from their recorded origins into the active
dataset and applies their labels. Entries are label-export entries (`origin`,
`origin_name`, `md5`, `label`, …).

→ `{ingested, applied, failed_count, failed, message}`

---

## Settings Importers & Exporters

### List settings importers

```
GET /api/settings-importers
```

→ JSON array of settings importer objects.

### Dynamic field options (settings importers)

```
POST /api/settings-importers/field-options/{importer_name}
```

Options for a `dynamic_options` field on a settings importer, for the Import
Settings modal (contract in the note at the top).

### Run settings import

```
POST /api/settings-importers/import/{importer_name}
```

**Form or Body:** importer-specific fields.

Applies the imported settings that are the user's own, each through the
checks `PUT /api/settings` runs (see [Settings](settings.md#update-settings)),
as a settings-source sync does. A server-tier key (`detectors_dir`,
`solo_media_type`, …), an unknown key, or a value that fails validation is
skipped rather than failing the import.

→ `{"success": true, "message": "Imported 5 setting(s) via ...", "keys": ["volume", ...], "skipped": ["detectors_dir"]}`

`keys` lists the settings actually applied; `skipped` lists the rest of the
file's keys, which `message` also names.

### List settings exporters

```
GET /api/settings-exporters
```

→ JSON array of settings exporter objects.

### Dynamic field options (settings exporters)

```
POST /api/settings-exporters/field-options/{exporter_name}
```

The same, for the Export Settings modal's field form. Note that the exporter is named in the **path** here even
though `POST /api/settings-exporters/export` names it in the body.

### Run settings export

```
POST /api/settings-exporters/export
```

**Body:** `{"exporter_name": "...", "field_values": {...}}`

→ `{success, message}` plus whatever the exporter returns —
e.g. `filepath` for a server-file exporter, or `data` + `filename` +
`download: true` when the browser should save the file. 400 (missing field or
invalid path), 404 (unknown exporter), 500 (exporter error).
