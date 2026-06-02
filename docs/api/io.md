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

---

## Label Importers

### List label importers

```
GET /api/label-importers
```

→ JSON array of label importer objects.

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

## Processor Importers

### List processor importers

```
GET /api/processor-importers
```

→ JSON array of processor importer objects.

### Run processor import

```
POST /api/processor-importers/import/{importer_name}
```

**Form or Body:** importer-specific fields. `name` is required.

Runs the importer and saves the result as an autorun detector.

→ `{"success": true, "name": "...", "media_type": "audio"}`

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
