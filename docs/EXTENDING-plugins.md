# Extending VTSearch — Plugin Systems

Eight auto-discovered plugin families share a common registry-based
architecture. Subclass the relevant base class, expose a sentinel
attribute, drop the module in the right directory, and the route/UI
wiring happens automatically.

**Related docs:** [EXTENDING.md](EXTENDING.md) (index, checklists, auth,
dependencies) · [EXTENDING-media.md](EXTENDING-media.md) (media types,
embedders, clippers, converters, sources) ·
[EXTENDING-processors.md](EXTENDING-processors.md) (detectors, localizers,
extractors).

## Contents

- [Shared Plugin Architecture](#shared-plugin-architecture) — PluginField,
  PluginRegistry, discovery, route generation
- [Adding a Data Importer](#adding-a-data-importer)
- [Adding a Results Exporter](#adding-a-results-exporter)
- [Adding a Label Importer](#adding-a-label-importer)
- [Adding a Processor Importer](#adding-a-processor-importer)
- [Adding a Settings Importer](#adding-a-settings-importer)
- [Adding a Settings Exporter](#adding-a-settings-exporter)
- [Adding a Settings Source](#adding-a-settings-source)
- [Adding a Labelset Source](#adding-a-labelset-source)

---

## Shared Plugin Architecture

Ten plugin systems — data importers, results exporters, label
importers, processor importers, settings importers, settings exporters,
settings sources, labelset sources, media converters, and media
sources — share the same architecture built on two base classes in
`vtsearch/utils/registry.py`:

### PluginField

A dataclass describing a single user-configurable input. All plugin
families use the same field type (aliased as `ImporterField`,
`ExporterField`, `LabelImporterField`, `ProcessorImporterField`, etc.).

| Parameter     | Type        | Default  | Description                                             |
|---------------|-------------|----------|---------------------------------------------------------|
| `key`         | `str`       | —        | Field identifier (dict key in `field_values`)           |
| `label`       | `str`       | —        | Display label in the UI                                 |
| `field_type`  | `FieldType` | —        | `"text"`, `"url"`, `"folder"`, `"file"`, `"password"`, `"email"`, `"select"`, or `"server_path"` |
| `description` | `str`       | `""`     | Helper text shown below the field                       |
| `accept`      | `str`       | `""`     | For `"file"` fields: comma-separated extensions (e.g. `".pkl"`) |
| `options`     | `list[str]` | `[]`     | For `"select"` fields: allowed dropdown values          |
| `default`     | `str`       | `""`     | Pre-filled value                                        |
| `required`    | `bool`      | `True`   | Whether the field must be filled before submitting      |
| `placeholder` | `str`       | `""`     | Hint shown as placeholder text in the input widget      |
| `dynamic_options` | `bool`  | `False`  | When `True`, options for this `"select"` field are fetched at runtime from the plugin's `get_field_options()` method (see [Dynamic field options](#dynamic-field-options)) |
| `depends_on`  | `list[str]` | `[]`     | Other field keys whose values this field's options depend on; the frontend re-fetches whenever any depended-on field changes |

### PluginBase

Shared base class providing CLI-argument derivation, validation, and
serialisation. All plugin base classes inherit from it.

**Required class attributes (set on your subclass):**

| Attribute      | Type                | Required | Description                                   |
|----------------|---------------------|----------|-----------------------------------------------|
| `name`         | `str`               | Yes      | Snake_case identifier, used in API URL path   |
| `display_name` | `str`               | Yes      | Human-readable label for the UI               |
| `description`  | `str`               | Yes      | One-sentence subtitle                         |
| `icon`         | `str`               | No       | Emoji/icon string (each family has a default) |
| `fields`       | `list[PluginField]` | Yes      | Ordered list of user-facing input fields      |

**Optional class attributes:**

| Attribute            | Type   | Default  | Description                                     |
|----------------------|--------|----------|-------------------------------------------------|
| `ui_mode`            | `str`  | `"form"` | `"form"`, `"file_upload"`, `"custom"`, `"none"` |
| `hidden_from_picker` | `bool` | `False`  | Exclude from generic picker list in frontend     |

**Inherited methods (available on all plugins):**

| Method                          | Description                                              |
|---------------------------------|----------------------------------------------------------|
| `add_cli_arguments(parser)`     | Auto-generates `argparse` flags from `fields`            |
| `validate_cli_field_values(fv)` | Raises `ValueError` if any required field is missing     |
| `to_dict()`                     | JSON-serialisable plugin metadata for API responses      |

### PluginRegistry (Auto-Discovery)

All plugin families use `PluginRegistry` for auto-discovery. The
registry uses direct filesystem scanning (`Path.iterdir()`) under the
plugin package directory. It discovers both **sub-packages**
(directories with `__init__.py`) and, for registries created with
`discover_modules=True`, **flat `.py` modules** (excluding `__init__.py`
and `base.py`). In each module it looks for a module-level sentinel
attribute; if found, the plugin is registered by its `name`.

Most plugin families use sub-packages, which pair well with per-plugin
`requirements.txt` files. The exception is **media sources**
(`vtsearch.datasets.sources`), which use flat `.py` modules
(`local_folder.py`, `http_archive.py`, `pullwrest.py`).

| Plugin Family       | Package                            | Sentinel              | Base Class          |
|---------------------|------------------------------------|-----------------------|---------------------|
| Data Importers      | `vtsearch.datasets.importers`      | `IMPORTER`            | `DatasetImporter`   |
| Results Exporters   | `vtsearch.exporters`               | `EXPORTER`            | `LabelsetExporter`  |
| Label Importers     | `vtsearch.labels.importers`        | `LABEL_IMPORTER`      | `LabelImporter`     |
| Processor Importers | `vtsearch.processors.importers`    | `PROCESSOR_IMPORTER`  | `ProcessorImporter` |
| Settings Importers  | `vtsearch.settings_io.importers`   | `SETTINGS_IMPORTER`   | `SettingsImporter`  |
| Settings Exporters  | `vtsearch.settings_io.exporters`   | `SETTINGS_EXPORTER`   | `SettingsExporter`  |
| Settings Sources    | `vtsearch.settings_io.sources`     | `SETTINGS_SOURCE`     | `SettingsSource`    |
| Labelset Sources    | `vtsearch.labels.sources`          | `LABELSET_SOURCE`     | `LabelsetSource`    |
| Media Converters    | `vtsearch.converters`              | `CONVERTER`           | `MediaConverter`    |
| Media Sources       | `vtsearch.datasets.sources`        | `SOURCE`              | `MediaSource`       |

Failed imports emit a warning but do not break the application — a missing
optional dependency gracefully disables that plugin.

---

## Adding a Data Importer

Data importers let users load datasets from new sources (S3 buckets,
databases, APIs, etc.). The system auto-discovers importers at runtime — no
changes to routes or core code are needed.

### File structure

```
vtsearch/datasets/importers/<your_importer>/
└── __init__.py       # Importer class + IMPORTER instance (required)
```

### What to implement

Subclass `DatasetImporter` from `vtsearch.datasets.importers.base`.
Set the required class attributes and implement the `run()` method.
Expose a module-level `IMPORTER` instance.

```python
# vtsearch/datasets/importers/s3/__init__.py

from vtsearch.datasets.importers.base import DatasetImporter, ImporterField
from vtsearch.media import all_folder_names


class S3Importer(DatasetImporter):
    name = "s3"
    display_name = "AWS S3 Bucket"
    description = "Download media files from an S3 bucket."
    icon = "☁️"

    fields = [
        ImporterField(
            key="bucket",
            label="Bucket Name",
            field_type="text",
            description="The S3 bucket name.",
            required=True,
        ),
        ImporterField(
            key="prefix",
            label="Key Prefix",
            field_type="text",
            description="Optional prefix to filter objects.",
            required=False,
            default="",
        ),
        ImporterField(
            key="media_type",
            label="Media Type",
            field_type="select",
            options=all_folder_names(),
            default="audio",
        ),
    ]

    def run(self, field_values: dict, medias: dict, thin: bool = False) -> None:
        """Download files from S3, then load them into the dataset.

        Args:
            field_values: Maps each ImporterField.key to the user's input.
                - "file" fields arrive as werkzeug FileStorage objects.
                - All other fields arrive as plain strings.
            medias: The global medias dict.  Populate it **in-place**; do not
                replace the reference.
            thin: When True, store media_path references instead of loading
                media bytes into memory (for CLI workflows).
        """
        import boto3
        from pathlib import Path
        from vtsearch.config import DATA_DIR
        from vtsearch.utils import update_progress

        bucket = field_values["bucket"]
        prefix = field_values.get("prefix", "")
        media_type = field_values.get("media_type", "audio")

        download_dir = DATA_DIR / "s3_import"
        download_dir.mkdir(parents=True, exist_ok=True)

        s3 = boto3.client("s3")
        objects = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        keys = [o["Key"] for o in objects.get("Contents", [])]

        for i, key in enumerate(keys):
            update_progress("downloading", f"Downloading {key}", i, len(keys))
            local_path = download_dir / Path(key).name
            s3.download_file(bucket, key, str(local_path))

        # Delegate to the standard folder loader
        from vtsearch.datasets.loader import load_dataset_from_folder
        load_dataset_from_folder(download_dir, media_type, medias, thin=thin)


# This module-level instance is what the registry discovers.
IMPORTER = S3Importer()
```

### DatasetImporter class reference

**Required to implement (pick one approach):**

| Member | Signature | Description |
|--------|-----------|-------------|
| `run()` | `(field_values: dict, medias: dict, thin: bool = False) -> None` | Populate `medias` in-place with loaded data |
| `list_records()` + `fetch_record()` | see [Bulk-record hooks](#bulk-record-hooks) | Per-record / bulk-record split that mirrors `MediaEmbedder`. The default `run()` lists records, hands them all to `fetch_records_bulk()`, and assigns IDs |

**Optional overrides:**

| Member | Signature | Description |
|--------|-----------|-------------|
| `_fetch_records_bulk_impl()` | `(records: list, field_values: dict, thin: bool) -> list[dict | None]` | Batched fetch hook. Default loops `fetch_record`. Override to issue concurrent / batched I/O |
| `run_cli()` | `(field_values: dict, medias: dict, thin: bool = False) -> None` | CLI variant; default delegates to `run()`. Override when `run()` expects FileStorage objects |
| `get_field_options()` | `(field_key: str, current_values: dict) -> list[str]` | Compute dropdown options for fields declared with `dynamic_options=True`. See [Dynamic field options](#dynamic-field-options) |
| `run_chunked()` | `(field_values, chunk_size, thin) -> Iterator[dict]` | Yield chunks of medias for piecewise processing. Set `supports_chunked = True` |
| `run_chunked_cli()` | `(field_values, chunk_size, thin) -> Iterator[dict]` | CLI variant of `run_chunked()` |
| `build_origin()` | `(field_values: dict) -> dict` | Build an origin dict for provenance tracking. Default uses importer name + string field values |
| `build_cli_args()` | `(field_values: dict) -> str` | Reconstruct CLI arguments from field values |
| `origin_display()` | `(origin: dict) -> str` | Human-readable string for an origin dict |
| `can_reload_from_origin()` | `(origin: dict) -> bool` | Whether data can be re-loaded from an origin. Default: `True` |
| `reload_from_origin()` | `(origin: dict) -> dict | None` | Extract field_values from an origin for re-import |
| `resolve_file()` | `(origin, origin_name, filename) -> Path | None` | Resolve a media file from origin info. Default: `None` |

**Instance attributes (set during `run()`):**

| Attribute | Type | Description |
|-----------|------|-------------|
| `content_vectors` | `dict[str, np.ndarray]` | Pre-computed embeddings keyed by filename; skips embedding model |
| `content_md5s` | `dict[str, str]` | Pre-computed MD5 hashes keyed by filename; skips hash computation |

### Element-level origin tracking

Every clip produced by an importer is automatically tagged with an
**origin** — a dict identifying the importer and its parameters. This
happens in `_run_importer_in_background()` after `run()` completes:

```python
clip["origin"]      = {"importer": "s3", "params": {"bucket": "my-data"}}
clip["origin_name"] = "clip_001.wav"  # defaults to clip["filename"]
```

If your importer pre-populates `clip["origin"]` in `run()`, those values
are preserved. Otherwise the system calls `build_origin(field_values)` on
your importer class and applies the result to all clips that lack an origin.

### Custom metadata

Importers can attach arbitrary per-media display metadata by setting
`media["custom_metadata"]` to a `dict[str, Any]`. For example:
`{"Uploaded By": "alice", "Bucket": "my-data"}`. These fields are merged
with the media type's built-in display fields and rendered in the labeling
UI.  When `enrich=true` is used on `GET /api/labels/export`, both
`custom_metadata` **and** `origin.params` are flattened into the per-entry
`custom_metadata` and `available_columns`, making fields like `contentID`
or `mediaID` selectable export columns.

### URL-backed media (`media_url`)

For importers that fetch media from a remote service (e.g. PullWrest),
set `media["media_url"]` to the URL of the media file.  The lazy-loading
system (`_resolve_media_bytes` / `_resolve_media_string`) resolves media
in this priority order:

1. `media_bytes` / `media_string` — already in memory.
2. `media_path` — local file on disk (thin mode with local files).
3. `media_url` — remote URL (fetched on demand).

In thin mode, URL-backed importers can skip downloading entirely: set
`media_bytes=None`, `media_path=None`, and `media_url="https://..."`.
Embeddings and MD5 can come from external services, so sorting and scoring
work without ever downloading the actual media.  Bytes are fetched lazily
only when the UI needs to display or play the media.

### Direct media dict construction

Most importers delegate to `load_dataset_from_folder()` after downloading
files to a local directory.  However, importers whose data comes from
API calls (not files on disk) can build media dicts directly in `run()`:

```python
def run(self, field_values, medias, thin=False):
    for i, item in enumerate(api_results, start=1):
        medias[i] = {
            "id": i,
            "type": "audio",
            "filename": item["id"],
            "md5": item["md5"],                  # pre-computed by the service
            "embedding": item["embedding"],      # pre-computed by the service
            "embedder": item["embedder_name"],   # must match a VTSearch embedder
            "media_bytes": data if not thin else None,
            "media_path": None,
            "media_url": item["url"],            # URL-based lazy-fetch fallback
            "media_string": None,
            "file_size": len(data) if data else 0,
            "duration": 0,
            "category": "",
            "origin": {                          # per-media origin (not dataset-level)
                "importer": self.name,
                "params": {"contentID": item["id"], ...},
            },
            "origin_name": item["id"],
            "custom_metadata": {"contentID": item["id"], ...},
        }
```

When building dicts directly, the importer should also override
`build_origin()` to return an empty origin (since the default
implementation captures dataset-level field values like query IDs that
are not useful per-media).  The post-processing step only backfills
`origin` on media that have `origin=None`, so per-media origins set
in `run()` are preserved.

### Bulk-record hooks

For service-style importers — those that fetch records from a remote
source rather than scanning a local folder — override the per-record /
bulk-record hooks instead of writing `run()` from scratch.  The split
mirrors the embedder's `embed_media` / `embed_media_bulk` pattern: a
working baseline comes from the per-item method, and you can opt into
batched / concurrent I/O by overriding the bulk hook.

```python
class MyServiceImporter(DatasetImporter):
    def list_records(self, field_values):
        # Return whatever shape you want — opaque to the framework.
        return _api.list(query=field_values["query"])

    def fetch_record(self, record, field_values, thin=False):
        # Default per-item path. The framework loops this when no
        # bulk override is provided. Return None to skip a record.
        return {
            "type": "audio",
            "filename": record["id"],
            "embedding": _api.get_embedding(record["id"]),
            "media_bytes": None if thin else _api.fetch_bytes(record["url"]),
            "media_url": record["url"],
            # origin / origin_name auto-filled from build_origin if omitted
        }

    def _fetch_records_bulk_impl(self, records, field_values, thin=False):
        # Optional: replace the per-item loop with a single bulk request,
        # a thread/async pool, or whatever the source supports.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=16) as pool:
            embeddings = list(pool.map(lambda r: _api.get_embedding(r["id"]), records))
            blobs = ([] if thin else
                     list(pool.map(lambda r: _api.fetch_bytes(r["url"]), records)))
        return [
            {"type": "audio", "filename": r["id"], "embedding": e,
             "media_bytes": (None if thin else b), "media_url": r["url"]}
            for r, e, b in zip(records, embeddings, blobs or [None] * len(records))
        ]
```

The default `run()`:

1. Calls `list_records(field_values)` to get the work list.
2. Calls `fetch_records_bulk(records, field_values, thin)` once with all
   records (which dispatches to `_fetch_records_bulk_impl`; the default
   impl loops `fetch_record` and emits per-item progress).
3. Assigns sequential integer IDs starting at 1 and stores the resulting
   media dicts in *medias*.
4. Backfills `media["origin"]` from `build_origin(field_values)` and
   `media["origin_name"]` from `media["filename"]` for any record that
   didn't set them.

Records returning `None` are skipped (gaps are squeezed out, IDs stay
sequential).

For an end-to-end example see
`vtsearch/datasets/importers/recaller/__init__.py`, which overrides the
bulk hook to issue DataWrest embedding lookups and PullWrest downloads
concurrently via a thread pool.

### Dynamic field options

When a `"select"` field's options must be computed at runtime — for
example, populating a list of remote queries after the user picks a
media type — declare it with `dynamic_options=True` and list the parent
fields it depends on in `depends_on`.  Then implement
`get_field_options(field_key, current_values)` on your importer:

```python
class ReCallerImporter(DatasetImporter):
    name = "recaller"
    fields = [
        ImporterField("media_type", "Media Type", "select",
                      options=all_folder_names(), default="audio"),
        ImporterField(
            "query_id", "Query ID", "select",
            dynamic_options=True,
            depends_on=["media_type"],   # re-fetch when media_type changes
        ),
    ]

    def get_field_options(self, field_key, current_values):
        if field_key == "query_id":
            return _list_recent_queries(current_values.get("media_type", ""))
        return super().get_field_options(field_key, current_values)
```

The frontend wiring is fully automatic:

1. When the user opens your importer, the modal pre-fetches options for
   every `dynamic_options=True` field.
2. Whenever a field listed in another field's `depends_on` changes, the
   modal clears the dependent field's value and re-fetches its options.
3. While a fetch is in-flight the dropdown is disabled and shows
   `Loading…`.  Errors raised by `get_field_options()` are surfaced
   inline next to the field.

API contract: `POST /api/dataset/import/<name>/options` with body
`{"field_key": "...", "values": {...}}` returns `{"options": [...]}`.
Any exception your `get_field_options()` raises is returned as a 502
with the exception message — perfect for surfacing remote-service
errors directly to the user.

### How it gets invoked

1. `GET /api/dataset/all-importers` returns all registered importers (your
   importer appears automatically). Note: `GET /api/dataset/importers` only
   returns importers with `ui_mode == "form"`.
2. `POST /api/dataset/import/<name>` invokes `run()` in a background
   daemon thread.
3. `POST /api/dataset/import/<name>/options` invokes `get_field_options()`
   to populate dynamic-options dropdowns (see above).
4. `GET /api/dataset/progress` provides progress bar data.

### Progress reporting

```python
from vtsearch.utils import update_progress
update_progress("downloading", "Downloading file 3/10", 3, 10)
```

### CLI usage

Importers are automatically usable from the command line:

```bash
python app.py --autodetect --importer s3 --bucket my-data --prefix audio/ \
    --media-type audio --settings settings.json
```

CLI arguments are auto-generated from `fields`. Override `run_cli()` if
your `run()` expects non-string values (e.g. FileStorage objects).

### Wiring up dependencies

Add any extra packages to a `requirements.txt` inside the plugin directory,
then run `bash install-plugin-deps.sh` to regenerate the dependency tree.
The next `pip install -r requirements.txt` will pick them up.

---

## Adding a Results Exporter

Results exporters deliver autodetect results **or labels** to a destination
(file, webhook, email, Holder, etc.).  Auto-discovered — no changes to
routes needed.

Exporters receive **two possible result formats** and should detect which:

- **Auto-detect results**: `{"media_type": "audio", "results": {...}}`
- **Labels**: `{"labels": [...], "selected_columns": [...]}` (from the
  label export flow with `enrich=true`)

Check `if "labels" in results` to distinguish them.  The built-in
CSV/JSON/webhook exporters handle both formats.

### File structure

```
vtsearch/exporters/<your_exporter>/
└── __init__.py       # Exporter class + EXPORTER instance (required)
```

### What to implement

Subclass `LabelsetExporter` from `vtsearch.exporters.base`.

```python
# vtsearch/exporters/sftp/__init__.py

from vtsearch.exporters.base import LabelsetExporter, ExporterField


class SftpLabelsetExporter(LabelsetExporter):
    name = "sftp"
    display_name = "SFTP Upload"
    description = "Upload results JSON to a remote SFTP server."
    icon = "📡"
    fields = [
        ExporterField("host", "Hostname", "text"),
        ExporterField("user", "Username", "text"),
        ExporterField("password", "Password", "password"),
        ExporterField(
            "path", "Remote Path", "text",
            default="/results/autodetect.json",
        ),
    ]

    def export(self, results: dict, field_values: dict) -> dict:
        """Export results to an SFTP server.

        Args:
            results: The full auto-detect results dict.  Shape:
                {
                    "media_type": "audio",
                    "detectors_run": 2,
                    "results": {
                        "detector_name": {
                            "detector_name": "...",
                            "threshold": 0.5,
                            "total_hits": 15,
                            "hits": [{...}, ...]
                        }
                    }
                }
            field_values: Mapping of ExporterField.key to user-supplied value.

        Returns:
            A dict with a "message" key (shown as confirmation to the user).
        """
        import json
        import paramiko

        host = field_values["host"]
        path = field_values["path"]

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(host, username=field_values["user"],
                    password=field_values["password"])
        sftp = ssh.open_sftp()
        with sftp.open(path, "w") as f:
            f.write(json.dumps(results, indent=2))
        sftp.close()
        ssh.close()

        return {"message": f"Uploaded to {host}:{path}"}


EXPORTER = SftpLabelsetExporter()
```

### LabelsetExporter class reference

**Required to implement:**

| Member | Signature | Description |
|--------|-----------|-------------|
| `export()` | `(results: dict, field_values: dict) -> dict` | Perform the export; return dict with `"message"` key |

**Optional overrides:**

| Member | Signature | Description |
|--------|-----------|-------------|
| `export_cli()` | `(results: dict, field_values: dict) -> dict` | CLI variant; default delegates to `export()` |

**Default class attributes:**

| Attribute | Default | Description |
|-----------|---------|-------------|
| `icon` | `"📤"` | Emoji shown in the UI |

### How it gets invoked

1. `GET /api/exporters` returns available exporters.
2. `POST /api/exporters/export` with `exporter_name` and `field_values`.

### CLI usage

```bash
python app.py --autodetect --dataset data.pkl --settings settings.json \
    --exporter sftp --host example.com --user admin --password secret \
    --path /results.json
```

### Built-in export endpoints

In addition to the exporter plugin system, VTSearch has built-in export
endpoints:

| Endpoint                       | Method | What it exports                           | Format          |
|--------------------------------|--------|-------------------------------------------|-----------------|
| `/api/dataset/export`          | GET    | Full dataset (clips + embeddings + media)  | Pickle (`.pkl`) |
| `/api/labels/export`           | GET    | LabelSet — labels with per-element origin  | JSON            |
| `/api/detectors/{name}` | GET    | Trainable model labelset + examples        | JSON            |

### Wiring up dependencies

Add any extra packages to a `requirements.txt` inside the plugin directory,
then run `bash install-plugin-deps.sh` to regenerate the dependency tree.
The next `pip install -r requirements.txt` will pick them up.

---

## Adding a Label Importer

Label importers let users import pre-existing labels (good/bad votes) from
external sources. Auto-discovered at runtime.

### File structure

```
vtsearch/labels/importers/<your_importer>/
└── __init__.py       # Importer class + LABEL_IMPORTER instance (required)
```

### What to implement

Subclass `LabelImporter` from `vtsearch.labels.importers.base`. The
`run()` method must return a list of label dicts.

```python
# vtsearch/labels/importers/postgres/__init__.py

from vtsearch.labels.importers.base import LabelImporter, LabelImporterField


class PostgresLabelImporter(LabelImporter):
    name = "postgres"
    display_name = "PostgreSQL Query"
    description = "Import labels from a PostgreSQL database query."
    icon = "🐘"
    fields = [
        LabelImporterField("host", "Hostname", "text"),
        LabelImporterField("database", "Database", "text"),
        LabelImporterField(
            "query", "SQL Query", "text",
            description="Must return md5 and label columns.",
        ),
    ]

    def run(self, field_values: dict) -> list[dict]:
        """Return a list of label dicts.

        Each dict must have "md5" and "label" keys.  Labels must be
        "good" or "bad"; any other value is skipped by the route handler.
        """
        import psycopg2

        conn = psycopg2.connect(
            host=field_values["host"],
            database=field_values["database"],
        )
        cur = conn.cursor()
        cur.execute(field_values["query"])
        return [{"md5": row[0], "label": row[1]} for row in cur.fetchall()]


LABEL_IMPORTER = PostgresLabelImporter()
```

### LabelImporter class reference

**Required to implement:**

| Member | Signature | Description |
|--------|-----------|-------------|
| `run()` | `(field_values: dict) -> list[dict[str, str]]` | Return list of `{"md5": ..., "label": "good"|"bad"}` dicts |

**Optional overrides:**

| Member | Signature | Description |
|--------|-----------|-------------|
| `run_cli()` | `(field_values: dict) -> list[dict[str, str]]` | CLI variant; default delegates to `run()` |

**Default class attributes:**

| Attribute | Default | Description |
|-----------|---------|-------------|
| `icon` | `"🏷️"` | Emoji shown in the UI |

### How it gets invoked

1. `GET /api/label-importers` returns available label importers.
2. `POST /api/label-importers/import/<name>` invokes `run()` and applies
   returned labels by matching clip MD5 hashes.

---

## Adding a detector from external labels

The detector and processor-importer plugin systems were removed.  To
publish or share a classifier:

1. Use `POST /api/detectors` (or
   `POST /api/detectors/registry/from-labelset/<importer>`) to create a
   detector file under `data/detectors/<name>.json`.
2. Toggle its autorun flag with
   `PUT /api/detectors/registry/<id>/autorun` so it runs from
   `/api/auto-detect` and the CLI's `--autodetect` flow.
3. The MLP itself lives only in RAM — it's trained on demand from the
   labelset's origins each time the model is loaded or scored.

For ready-made classifiers without labels (e.g. an OCR or face-detector
heuristic), build an Extractor or Localizer plugin instead — see
[EXTENDING-processors.md](EXTENDING-processors.md).

---

## Adding a Settings Importer

Settings importers let users import settings from external sources via
a one-shot operation (as opposed to settings *sources*, which provide
ongoing bidirectional sync). Auto-discovered at runtime.

### File structure

```
vtsearch/settings_io/importers/<your_importer>/
└── __init__.py       # Importer class + SETTINGS_IMPORTER instance (required)
```

### What to implement

Subclass `SettingsImporter` from `vtsearch.settings_io.importers.base`.
The `run()` method must return a dict of settings key-value pairs.

```python
# vtsearch/settings_io/importers/s3/__init__.py

from vtsearch.settings_io.importers.base import SettingsImporter, SettingsImporterField


class S3SettingsImporter(SettingsImporter):
    name = "s3"
    display_name = "S3 Settings File"
    description = "Import settings from an S3 object."
    icon = "☁️"
    fields = [
        SettingsImporterField("bucket", "S3 Bucket", "text"),
        SettingsImporterField("key", "Object Key", "text"),
    ]

    def run(self, field_values: dict) -> dict:
        """Import settings from S3. Return a settings dict.

        The returned dict is applied via the settings module's update
        mechanism — only keys present in the dict are changed.
        """
        import boto3, json
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=field_values["bucket"], Key=field_values["key"])
        return json.loads(obj["Body"].read())


SETTINGS_IMPORTER = S3SettingsImporter()
```

### SettingsImporter class reference

**Required to implement:**

| Member | Signature | Description |
|--------|-----------|-------------|
| `run()` | `(field_values: dict) -> dict[str, Any]` | Return settings key-value pairs to apply |

**Default class attributes:**

| Attribute | Default | Description |
|-----------|---------|-------------|
| `icon` | `"⚙️"` | Emoji shown in the UI |

### How it gets invoked

1. `GET /api/settings-importers` returns available importers.
2. `POST /api/settings-importers/import/<name>` invokes `run()` and applies
   the returned settings.

---

## Adding a Settings Exporter

Settings exporters let users export the current app settings to an
external destination (file download, remote server, etc.). Auto-discovered
at runtime.

### File structure

```
vtsearch/settings_io/exporters/<your_exporter>/
└── __init__.py       # Exporter class + SETTINGS_EXPORTER instance (required)
```

### What to implement

Subclass `SettingsExporter` from `vtsearch.settings_io.exporters.base`.
The `export()` method receives the full settings dict and must return a
dict with a `"message"` key.

```python
# vtsearch/settings_io/exporters/s3/__init__.py

from vtsearch.settings_io.exporters.base import SettingsExporter, SettingsExporterField


class S3SettingsExporter(SettingsExporter):
    name = "s3"
    display_name = "S3 Settings File"
    description = "Export settings to an S3 object."
    icon = "☁️"
    fields = [
        SettingsExporterField("bucket", "S3 Bucket", "text"),
        SettingsExporterField("key", "Object Key", "text"),
    ]

    def export(self, settings_data: dict, field_values: dict) -> dict:
        """Export settings to S3.

        Args:
            settings_data: The full settings dict from settings.get_all().
            field_values: User-supplied field values.

        Returns:
            A dict with a "message" key (shown as confirmation).
        """
        import boto3, json
        s3 = boto3.client("s3")
        s3.put_object(
            Bucket=field_values["bucket"],
            Key=field_values["key"],
            Body=json.dumps(settings_data, indent=2),
        )
        return {"message": f"Settings exported to s3://{field_values['bucket']}/{field_values['key']}"}


SETTINGS_EXPORTER = S3SettingsExporter()
```

### SettingsExporter class reference

**Required to implement:**

| Member | Signature | Description |
|--------|-----------|-------------|
| `export()` | `(settings_data: dict, field_values: dict) -> dict` | Perform export; return dict with `"message"` key |

**Default class attributes:**

| Attribute | Default | Description |
|-----------|---------|-------------|
| `icon` | `"📤"` | Emoji shown in the UI |

### How it gets invoked

1. `GET /api/settings-exporters` returns available exporters.
2. `POST /api/settings-exporters/export/<name>` invokes `export()` with
   the current settings and user-supplied field values.

---

## Adding a Settings Source

Settings sources provide **bidirectional sync** — combining the roles of
a settings importer and exporter into a single plugin that stays
connected. When a source is active, changing any setting auto-exports to
the source, and syncing pulls from the source back into the app.

Use a **Settings Importer** or **Settings Exporter** (above) for
one-shot operations. Use a **Settings Source** when you want ongoing
automatic sync.

### File structure

```
vtsearch/settings_io/sources/<your_source>/
└── __init__.py       # Source class + SETTINGS_SOURCE instance (required)
```

### What to implement

Subclass `SettingsSource` from `vtsearch.settings_io.sources.base`.

```python
# vtsearch/settings_io/sources/s3/__init__.py

from vtsearch.settings_io.sources.base import SettingsSource, PluginField


class S3SettingsSource(SettingsSource):
    name = "s3"
    display_name = "S3 Settings File"
    description = "Sync settings with an S3 object."
    icon = "☁️"
    fields = [
        PluginField("bucket", "S3 Bucket", "text"),
        PluginField("key", "Object Key", "text"),
    ]

    def load(self, field_values: dict) -> dict:
        """Read settings from S3. Return a settings dict."""
        import boto3, json
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=field_values["bucket"], Key=field_values["key"])
        return json.loads(obj["Body"].read())

    def save(self, settings_data: dict, field_values: dict) -> None:
        """Write settings to S3."""
        import boto3, json
        s3 = boto3.client("s3")
        s3.put_object(
            Bucket=field_values["bucket"],
            Key=field_values["key"],
            Body=json.dumps(settings_data, indent=2),
        )


SETTINGS_SOURCE = S3SettingsSource()
```

The sentinel `SETTINGS_SOURCE` at module level is required for auto-discovery.

### Template variables

Field values support `{username}` — resolved at runtime from
`get_current_user()`. This enables per-user settings files
(e.g. `data/{username}.settings.json`).

### How it gets invoked

1. `GET /api/settings-sources` lists all discovered sources.
2. `PUT /api/settings-sources/active` sets which source is active.
3. **At startup**, `sync_from_settings_source()` auto-imports settings
   from the active source (so the source takes precedence over the local
   `data/settings.json` file).
4. When settings change, `settings._save()` auto-calls `source.save()`.
5. `POST /api/settings-sources/sync` manually re-imports from the source.

### Making your source the active auto-import

To use a custom settings source for auto-import at startup, set it as
the active source in `data/settings.json`:

```json
{
  "settings_source": {
    "source_name": "s3",
    "field_values": {
      "bucket": "my-settings-bucket",
      "key": "vtsearch/{username}.json"
    }
  }
}
```

Or set it via the API:

```bash
curl -X PUT http://localhost:5000/api/settings-sources/active \
  -H 'Content-Type: application/json' \
  -d '{"source_name": "s3", "field_values": {"bucket": "my-bucket", "key": "settings.json"}}'
```

On the next app startup, VTSearch will call `source.load()` and apply
the returned settings before starting the server. The built-in
`server_json_file` source does this with a local file path — your
custom source can fetch from S3, a database, a remote API, etc.

If the source is unavailable at startup (file missing, network error),
the import is silently skipped and the local settings file is used as
fallback.

---

## Adding a Labelset Source

Labelset sources provide **bidirectional sync** for detector labels —
combining the roles of a label importer and exporter into a single
plugin. Each detector can link to a source that auto-exports labels on
change and imports them on sync.

Use a **Label Importer** (above) for one-shot label import. Use a
**Labelset Source** when you want ongoing automatic sync per-detector.

### File structure

```
vtsearch/labels/sources/<your_source>/
└── __init__.py       # Source class + LABELSET_SOURCE instance (required)
```

### What to implement

Subclass `LabelsetSource` from `vtsearch.labels.sources.base`.

```python
# vtsearch/labels/sources/database/__init__.py

from vtsearch.labels.sources.base import LabelsetSource, PluginField
from vtsearch.datasets.labelset import LabelSet


class DatabaseLabelsetSource(LabelsetSource):
    name = "database"
    display_name = "Database Labels"
    description = "Sync labels with a database table."
    icon = "🗄️"
    fields = [
        PluginField("connection_string", "Connection String", "text"),
        PluginField("table", "Table Name", "text", default="labels"),
    ]

    def load(self, field_values: dict) -> list[dict]:
        """Read labels from database. Return list of label dicts."""
        # Each dict should have: "name" (media name/hash), "label" ("Good"/"Bad")
        ...

    def save(self, labelset: LabelSet, field_values: dict) -> None:
        """Write labelset to database."""
        for elem in labelset.elements:
            # Upsert each element...
            ...


LABELSET_SOURCE = DatabaseLabelsetSource()
```

The sentinel `LABELSET_SOURCE` at module level is required for auto-discovery.

### Template variables

Field values support `{detector_id}` and `{detector_name}` — resolved
at runtime from the active `DetectorContext`.

### How it gets invoked

1. `GET /api/labelset-sources` lists all discovered sources.
2. `PUT /api/detectors/<name>/labelset-source` links a source to a detector.
3. When votes change or labels are imported, `sync_to_labelset_source()` auto-calls `source.save()`.
4. `POST /api/detectors/<name>/labelset-source/sync` manually calls `source.load()`.

### Circular trigger prevention

When a source imports labels (via `sync_from_labelset_source()`), each
applied label would normally trigger a re-export back. A thread-local
`_syncing` guard in `vtsearch/labels/sync.py` suppresses this:

```python
with _sync_guard():
    # apply_label() calls during import won't trigger source.save()
    ...
```

The same pattern is used in `vtsearch/settings.py` for settings sources.

---
