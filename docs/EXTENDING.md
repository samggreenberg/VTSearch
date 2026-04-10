# Extending VTSearch

This guide explains how to add new plugins and extensions to VTSearch.
Each section describes the interface contract, where files go, how
discovery/registration works, and includes a complete example.

**Extension types covered:**

- [Plugin systems](#shared-plugin-architecture) (shared base for importers/exporters)
  - [Data Importers](#adding-a-data-importer)
  - [Results Exporters](#adding-a-results-exporter)
  - [Label Importers](#adding-a-label-importer)
  - [Processor Importers](#adding-a-processor-importer)
  - [Settings Importers](#adding-a-settings-importer)
  - [Settings Exporters](#adding-a-settings-exporter)
  - [Settings Sources](#adding-a-settings-source)
  - [Labelset Sources](#adding-a-labelset-source)
- [Media system](#media-system) (explicit registration)
  - [Media Types](#adding-a-media-type)
  - [Media Embedders](#adding-a-media-embedder)
  - [Media Clippers](#adding-a-media-clipper)
  - [Media Converters](#adding-a-media-converter)
  - [Media Sources](#adding-a-media-source)
- [Processor system](#processor-system) (Detectors, Localizers, Extractors)
  - [Detectors](#adding-a-detector)
  - [Localizers](#adding-a-localizer)
  - [Extractors](#adding-an-extractor)
- [Authentication Providers](#authentication-providers)
- [Dependency Management](#dependency-management)
- [Quick Reference Checklists](#quick-reference-checklist-for-each-extension-type)

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
registry uses direct filesystem scanning (`Path.iterdir()`) to find
**sub-packages** (directories with `__init__.py`) under the plugin
directory. For each sub-package, it imports the module and looks for a
module-level sentinel attribute. If found, the plugin is registered by
its `name`.

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

**Required to implement:**

| Member | Signature | Description |
|--------|-----------|-------------|
| `run()` | `(field_values: dict, medias: dict, thin: bool = False) -> None` | Populate `medias` in-place with loaded data |

**Optional overrides:**

| Member | Signature | Description |
|--------|-----------|-------------|
| `run_cli()` | `(field_values: dict, medias: dict, thin: bool = False) -> None` | CLI variant; default delegates to `run()`. Override when `run()` expects FileStorage objects |
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

### How it gets invoked

1. `GET /api/dataset/all-importers` returns all registered importers (your
   importer appears automatically). Note: `GET /api/dataset/importers` only
   returns importers with `ui_mode == "form"`.
2. `POST /api/dataset/import/<name>` invokes `run()` in a background
   daemon thread.
3. `GET /api/dataset/progress` provides progress bar data.

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
| `/api/detector/export-server`  | POST   | Detector origins + inclusion to server file| JSON            |

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

## Adding a Processor Importer

Processor importers let users import processors (detectors/extractors) from
external sources. A processor importer takes input (a JSON detector file,
etc.) and returns a dict containing model weights and a threshold — which
is then saved as an autorun detector.

### File structure

```
vtsearch/processors/importers/<your_importer>/
└── __init__.py       # Importer class + PROCESSOR_IMPORTER instance (required)
```

### What to implement

Subclass `ProcessorImporter` from `vtsearch.processors.importers.base`.
The `run()` method must return a dict with model data.

```python
# vtsearch/processors/importers/s3/__init__.py

from vtsearch.processors.importers.base import ProcessorImporter, ProcessorImporterField


class S3ProcessorImporter(ProcessorImporter):
    name = "s3"
    display_name = "S3 Detector File"
    description = "Download a detector JSON file from an S3 bucket."
    icon = "☁️"
    fields = [
        ProcessorImporterField("bucket", "S3 Bucket", "text"),
        ProcessorImporterField("key", "Object Key", "text"),
    ]

    def run(self, field_values: dict) -> dict:
        """Download and parse a detector JSON from S3.

        Must return a dict with at minimum:
            - "good_origins" (list): origin dicts for Good-labeled media
            - "bad_origins" (list): origin dicts for Bad-labeled media
            - "media_type" (str): e.g. "audio", "image"
        May also include:
            - "inclusion" (int): inclusion bias from training
            - "weights" (dict): pre-computed MLP weights (fallback)
            - "threshold" (float): decision boundary in [0, 1]
            - "name" (str): suggested default name
        """
        import json
        import boto3

        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=field_values["bucket"], Key=field_values["key"])
        data = json.loads(obj["Body"].read())

        from vtsearch.models.weights_compat import normalize_detector_weights

        nw = normalize_detector_weights(data)
        return {
            "media_type": data.get("media_type", "audio"),
            "good_origins": nw.good_origins,
            "bad_origins": nw.bad_origins,
            "inclusion": nw.inclusion,
            "weights": nw.weights,
            "threshold": nw.threshold,
        }


PROCESSOR_IMPORTER = S3ProcessorImporter()
```

### ProcessorImporter class reference

**Required to implement:**

| Member | Signature | Description |
|--------|-----------|-------------|
| `run()` | `(field_values: dict) -> dict` | Return dict with `good_origins`, `bad_origins`, `media_type`, `weights`, `threshold` |

**Optional overrides:**

| Member | Signature | Description |
|--------|-----------|-------------|
| `run_cli()` | `(field_values: dict) -> dict` | CLI variant; default delegates to `run()` |

**Default class attributes:**

| Attribute | Default | Description |
|-----------|---------|-------------|
| `icon` | `"🧩"` | Emoji shown in the UI |

### How it gets invoked

1. `GET /api/processor-importers` returns available importers.
2. `POST /api/processor-importers/import/<name>` invokes `run()`, combines
   with user-supplied name, and saves as an autorun detector.

### CLI usage

Processor importers are used from the CLI via the settings file. Add a
processor recipe to `autorun_processors` in `settings.json`:

```json
{
    "autorun_processors": [
        {
            "processor_name": "my detector",
            "processor_importer": "server_detector_file",
            "field_values": {"filepath": "/path/to/detector.json"}
        }
    ]
}
```

Then run autodetect:

```bash
python app.py --autodetect --dataset data.pkl --settings settings.json
```

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

## Media System

Media types, embedders, and clippers are **auto-discovered** at import
time. The `_discover_media_plugins()` function in
`vtsearch/media/__init__.py` scans sub-packages of `vtsearch/media/` for
module-level sentinel attributes:

| Sentinel     | Type                  | Description                          |
|--------------|-----------------------|--------------------------------------|
| `MEDIA_TYPE` | `MediaType`           | A single media type instance         |
| `EMBEDDERS`  | `list[MediaEmbedder]` | Embedder instances (may be empty)    |
| `CLIPPERS`   | `list[MediaClipper]`  | Clipper instances (may be empty)     |

To add a new built-in media type, create a sub-package under
`vtsearch/media/` with an `__init__.py` that exposes the relevant
sentinels. Symlinked directories are supported.

Third-party or project-specific types can still be registered manually
via `register()`, `register_embedder()`, and `register_clipper()`.

---

## Adding a Media Type

Media types define how VTSearch handles a particular kind of content: how
to serve clips over HTTP, what file extensions to scan for, what demo
datasets are available, and how to load media-specific fields from files.

### File structure

```
vtsearch/media/<your_type>/
├── __init__.py       # Must expose MEDIA_TYPE, EMBEDDERS, CLIPPERS sentinels
└── media_type.py     # Your MediaType subclass (required)
```

### What to implement

Subclass `MediaType` from `vtsearch.media.base` and implement all abstract
properties and methods.

```python
# vtsearch/media/code/media_type.py

from __future__ import annotations

from pathlib import Path
from typing import Any

from vtsearch.media.base import DemoDataset, MediaResponse, MediaType


class CodeMediaType(MediaType):
    """Source code files."""

    # --- Identity (required abstract properties) ---

    @property
    def type_id(self) -> str:
        return "code"

    @property
    def name(self) -> str:
        return "Source Code"

    @property
    def icon(self) -> str:
        return "code"  # SVG icon type name for the UI

    # --- File import (required abstract property) ---

    @property
    def file_extensions(self) -> list:
        return ["*.py", "*.js", "*.ts", "*.go", "*.rs"]

    # --- Viewer behaviour (required abstract property) ---

    @property
    def loops(self) -> bool:
        return False

    # --- Demo datasets (required abstract property) ---

    @property
    def demo_datasets(self) -> list:
        return []  # No demos yet

    # --- Media data (required abstract method) ---

    def load_media_data(self, file_path: Path) -> dict:
        """Return media-specific fields to merge into the media dict.

        The base media dict already contains: id, type, file_size, md5,
        embedding, filename, category.  You MUST include a "duration" key
        (use 0 for non-temporal media).
        """
        content = file_path.read_text(errors="replace")
        return {
            "media_string": content,
            "duration": 0,
            "line_count": content.count("\n") + 1,
        }

    # --- HTTP serving (required abstract method) ---

    def media_response(self, media: dict) -> MediaResponse:
        """Return a MediaResponse for HTTP serving.

        Use _resolve_media_bytes() for binary media or
        _resolve_media_string() for text media to support both
        preloaded and thin (lazy-loaded) modes.
        """
        content = self._resolve_media_string(media)
        return MediaResponse(
            data={"content": content, "line_count": media.get("line_count", 0)},
            mimetype="application/json",
        )
```

### Register the new type

Expose the sentinels in your sub-package's `__init__.py`:

```python
# vtsearch/media/code/__init__.py

from vtsearch.media.code.media_type import CodeMediaType
from vtsearch.media.code.embedder import CodeBertEmbedder

MEDIA_TYPE = CodeMediaType()
EMBEDDERS = [CodeBertEmbedder()]
CLIPPERS = []  # No clippers yet — add when needed
```

The auto-discovery system finds these sentinels at import time. No
changes to `vtsearch/media/__init__.py` are needed.

### MediaType abstract interface reference

**Required abstract properties:**

| Property          | Returns     | Example                              |
|-------------------|-------------|--------------------------------------|
| `type_id`         | `str`       | `"audio"`, `"image"`, `"code"`       |
| `name`            | `str`       | `"Audio"`, `"Source Code"`           |
| `icon`            | `str`       | `"audio"`, `"code"` (SVG icon type name) |
| `file_extensions` | `list[str]` | `["*.wav", "*.mp3"]`                 |
| `loops`           | `bool`      | `True` for audio/video, else `False` |
| `demo_datasets`   | `list[DemoDataset]` | See example above              |

**Required abstract methods:**

| Method                      | Signature                      | Description                              |
|-----------------------------|--------------------------------|------------------------------------------|
| `load_media_data(file_path)`| `(Path) -> dict`               | Must include `"duration"` key            |
| `media_response(media)`     | `(dict) -> MediaResponse`      | HTTP response for a media item           |

**Optional overridable properties (with defaults):**

| Property             | Returns     | Default            | Purpose                                   |
|----------------------|-------------|--------------------|-------------------------------------------|
| `folder_import_name` | `str`       | `type_id`          | Alias for folder imports (matches `type_id`) |
| `tab_title`          | `str`       | `name + "s"`       | Plural name for UI tabs                    |
| `dir_key`            | `str`       | `type_id + "_dir"` | Key in pickle files for external dir       |
| `legacy_bytes_keys`  | `list[str]` | `[]`               | Legacy keys for inline bytes in old pickles |
| `pickle_extra_fields`| `list[str]` | `[]`               | Extra fields to preserve in pickle round-trips (e.g. `["width", "height"]`) |

**Optional overridable methods:**

| Method                        | Signature                          | Description                        |
|-------------------------------|------------------------------------|------------------------------------|
| `display_metadata(media)`     | `(dict) -> dict[str, Any]`         | Metadata for the labeling UI       |
| `load_models()`               | `() -> None`                       | Load inline embedding models (legacy) |
| `embed_media(file_path)`      | `(Path) -> Optional[np.ndarray]`   | Inline embedding (legacy, prefer MediaEmbedder) |
| `embed_text(text)`            | `(str) -> Optional[np.ndarray]`    | Inline text embedding (legacy)     |
| `load_demo_source(...)`       | See docstring                      | Download and embed a demo dataset  |

### What happens automatically after registration

| Subsystem              | What happens                                                  |
|------------------------|---------------------------------------------------------------|
| **Folder import**      | Files matching your `file_extensions` are found and embedded  |
| **Generic media route**| `GET /api/medias/<id>/media` delegates to your `media_response()`|
| **Demo listing**       | Your `demo_datasets` appear in `GET /api/dataset/demo-list`   |
| **Dataset export**     | Clip data is serialized to pickle (including custom fields)   |
| **Media types API**    | `GET /api/media-types` includes your type's metadata          |

### Making dataset export aware of custom clip fields

If your media type stores clip data under non-standard keys, override
`pickle_extra_fields` to return those key names so they survive pickle
export/import. For example:

```python
@property
def pickle_extra_fields(self) -> list[str]:
    return ["line_count"]
```

---

## Adding a Media Embedder

Media embedders produce fixed-size vector embeddings from media files and
text queries. Each embedder is associated with exactly one media type but a
media type may have multiple embedders.

### File structure

```
vtsearch/media/<type>/
├── embedder.py              # Default embedder (required for new media types)
└── embedder_<variant>.py    # Alternative embedder (optional, e.g. embedder_siglip.py)
```

Each media type has one default embedder in `embedder.py`. To add an
**alternative** embedder for an existing media type, create a new file named
`embedder_<variant>.py` (e.g. `embedder_clap_music.py`, `embedder_siglip.py`,
`embedder_bge.py`) and register it the same way. Existing alternatives:

| File | Embedder | Media type |
|------|----------|-----------|
| `audio/embedder_clap_music.py` | `AudioClapMusicEmbedder` | audio |
| `image/embedder_siglip.py` | `ImageSiglipEmbedder` | image |
| `text/embedder_bge.py` | `TextBGEEmbedder` | text |

### What to implement

Subclass `MediaEmbedder` from `vtsearch.media.base`.

```python
# vtsearch/media/code/embedder.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from vtsearch.media.base import MediaEmbedder


class CodeBertEmbedder(MediaEmbedder):
    """Embeds source code using CodeBERT."""

    def __init__(self) -> None:
        super().__init__()
        self._model = None

    # --- Identity (required abstract properties) ---

    @property
    def name(self) -> str:
        """Unique identifier — also the registry key."""
        return "codebert"

    @property
    def media_type_id(self) -> str:
        """The type_id of the media type this embedder works with."""
        return "code"

    # --- Model lifecycle (required abstract method) ---

    def _load_models_impl(self) -> None:
        """Load the embedding model. Must be idempotent.

        Override ``_load_models_impl`` (not ``load_models``).
        The public ``load_models()`` wrapper handles locking and
        ImportError wrapping automatically.
        """
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._on_progress("loading", "Loading CodeBERT…", 0, 0)
        self._model = SentenceTransformer("microsoft/codebert-base")

    # --- Embedding (required abstract method) ---

    def _embed_media_impl(self, file_path: Path) -> Optional[np.ndarray]:
        """Return a fixed-size embedding vector for the file.

        Override ``_embed_media_impl`` (not ``embed_media``).
        The public ``embed_media()`` wrapper acquires a global lock
        so that only one forward pass runs at a time.

        Returns None if embedding fails. The vector dimensionality
        must be consistent and must match embed_text().
        """
        if self._model is None:
            self.load_models()
        try:
            text = file_path.read_text(errors="replace")[:8000]
            return self._model.encode(text, normalize_embeddings=True)
        except Exception:
            return None

    # --- Optional: text embedding ---

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """Embed a text query into the SAME vector space as embed_media().

        Used for text-query sorting. Default returns None (no text sort).
        """
        if self._model is None:
            self.load_models()
        try:
            return self._model.encode(text, normalize_embeddings=True)
        except Exception:
            return None
```

### Register the embedder

Add the embedder to the `EMBEDDERS` sentinel list in your media type's
`__init__.py`:

```python
# vtsearch/media/code/__init__.py

from vtsearch.media.code.embedder import CodeBertEmbedder
# ...
EMBEDDERS = [CodeBertEmbedder()]
```

For an alternative embedder on an **existing** media type, add it to that
type's `EMBEDDERS` list (e.g. in `vtsearch/media/image/__init__.py`).

### MediaEmbedder abstract interface reference

**Required abstract properties:**

| Property        | Returns | Description                              |
|-----------------|---------|------------------------------------------|
| `name`          | `str`   | Unique identifier (e.g. `"clap"`, `"clip"`) |
| `media_type_id` | `str`  | Which media type this embedder works with |

**Required abstract methods:**

| Method                      | Signature                        | Description                    |
|-----------------------------|----------------------------------|--------------------------------|
| `_load_models_impl()`       | `() -> None`                     | Load model; must be idempotent. Override this, not `load_models()` |
| `embed_media(file_path)`    | `(Path) -> Optional[np.ndarray]` | Embed a media file             |

**Optional overridable methods:**

| Method                          | Signature                         | Description                          |
|---------------------------------|-----------------------------------|--------------------------------------|
| `embed_text(text)`              | `(str) -> Optional[np.ndarray]`   | Embed a text query (default: `None`) |
| `embed_text_enriched(text)`     | `(str) -> Optional[np.ndarray]`   | Average over `description_wrappers`  |

**Optional overridable properties:**

| Property               | Returns     | Description                                |
|------------------------|-------------|--------------------------------------------|
| `description_wrappers` | `list[str]` | Templates with `{text}` for enriched embedding (e.g. `["the sound of {text}"]`) |

**Instance attributes:**

| Attribute       | Type               | Description                         |
|-----------------|--------------------|-------------------------------------|
| `_on_progress`  | `ProgressCallback` | Progress callback (default: no-op). Set via `set_progress_callback()` |

### Built-in embedders

| Embedder | Name | Media Type | Model | Dimensions |
|----------|------|------------|-------|------------|
| `AudioClapEmbedder` | `clap` | `audio` | LAION CLAP (laion/clap-htsat-unfused) | 512 |
| `AudioClapMusicEmbedder` | `clap_music` | `audio` | CLAP Music & Speech (laion/larger_clap_music_and_speech) | 512 |
| `ImageSiglipEmbedder` | `siglip` | `image` | SigLIP (google/siglip-base-patch16-224) | 768 |
| `ImageClipEmbedder` | `clip` | `image` | OpenAI CLIP (openai/clip-vit-base-patch32) | 768 |
| `TextE5Embedder` | `e5` | `text` | E5-base-v2 (intfloat/e5-base-v2) | 768 |
| `TextBGEEmbedder` | `bge` | `text` | BGE-base-en-v1.5 (BAAI/bge-base-en-v1.5) | 768 |
| `VideoXClipEmbedder` | `xclip` | `video` | X-CLIP (microsoft/xclip-base-patch32) | 768 |

---

## Adding a Media Clipper

Media clippers split a single media item into one or more items of the
**same** type. Unlike processors which return metadata about media,
clippers return **new media dicts** that can replace the original.

### Built-in clippers

| Clipper | Name | Media Type | Description |
|---------|------|------------|-------------|
| `SoundDefaultClipper` | `sound_default` | `audio` | Import each audio file as-is, without splitting |
| `SoundTilingClipper` | `sound_tiling_2.0s` | `audio` | Split each audio file into fixed-length overlapping segments |
| `ImageDefaultClipper` | `image_default` | `image` | Import each image as-is, without splitting |
| `ImageTilingClipper` | `image_tiling` | `image` | Tile each image into equidistant square crops along the longer axis |
| `TextDefaultClipper` | `text_default` | `text` | Import each text entry as-is, without splitting |
| `TextSentenceClipper` | `text_sentence` | `text` | Split each text entry into individual sentences |
| `VideoDefaultClipper` | `video_default` | `video` | Import each video as-is, without splitting |
| `VideoTilingClipper` | `video_tiling_2.0s` | `video` | Split each video into fixed-length overlapping segments |
| `VideoSceneClipper` | `video_scene` | `video` | Automatically split each video at detected scene changes |
| `DocumentDefaultClipper` | `document_default` | `document` | Import each document as-is, without splitting |

### What to implement

Subclass `MediaClipper` from `vtsearch.media.base`.

```python
# vtsearch/media/audio/clipper.py  (or a new file)

from vtsearch.media.base import MediaClipper
from typing import Any


class SoundOverlapClipper(MediaClipper):
    """Tile audio with 50% overlap between segments."""

    def __init__(self, duration: float) -> None:
        self._duration = duration

    @property
    def name(self) -> str:
        """Unique identifier for this clipper."""
        return f"sound_overlap_{self._duration}s"

    @property
    def media_type(self) -> str:
        """The type_id this clipper operates on."""
        return "audio"

    @property
    def description(self) -> str:
        """Short tooltip shown on hover in the clipper chooser UI."""
        return "Tile audio with 50% overlap between consecutive segments."

    def clip(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        """Split media into one or more media dicts of the same type.

        Each returned dict preserves the original structure (id, type,
        category, origin, etc.) but with updated content.
        Returns a list with at least one element.
        """
        wav_bytes = media.get("media_bytes")
        if wav_bytes is None:
            return [media]

        # ... implement overlapping tiling logic ...
        # Return list of new media dicts with updated media_bytes, duration, etc.
        return [media]  # placeholder
```

### Register the clipper

Add the clipper to the `CLIPPERS` sentinel list in your media type's
`__init__.py`:

```python
# vtsearch/media/audio/__init__.py

from vtsearch.media.audio.clipper import SoundOverlapClipper
# ...
CLIPPERS = [SoundDefaultClipper(), SoundTilingClipper(2.0), SoundOverlapClipper(2.0)]
```

### MediaClipper abstract interface reference

**Required abstract properties:**

| Property     | Returns | Description                                    |
|--------------|---------|------------------------------------------------|
| `name`       | `str`   | Unique identifier (e.g. `"sound_tiling_2.0s"`) |
| `media_type` | `str`   | The `type_id` this clipper operates on          |

**Required abstract methods:**

| Method          | Signature              | Description                        |
|-----------------|------------------------|------------------------------------|
| `clip(media)`   | `(dict) -> list[dict]` | Split one media into one or more   |

**Optional overridable methods/properties:**

| Method/Property      | Signature / Returns      | Description                                                       |
|----------------------|--------------------------|-------------------------------------------------------------------|
| `display_name`       | `str`                    | Human-readable name for UI tabs (default: title-cased `name`)     |
| `description`        | `str`                    | Short tooltip text shown on hover in the clipper chooser UI       |
| `to_dict()`          | `() -> dict`             | JSON-serialisable metadata (default: name + media_type)           |
| `parameters`         | `list[dict[str, Any]]`   | Configurable parameters (key, label, type, default, description)  |
| `creation_questions` | `list[dict[str, Any]]`   | Questions shown at creation time (defaults to `parameters`)       |
| `with_params(p)`     | `(dict) -> MediaClipper` | Return new clipper with overridden parameters                     |

Parameter dicts support an optional `description` key alongside `label`
— this is shown as a tooltip when the user hovers over the setting in
the clipper chooser dialog.

### Clip method contract

Each dict in the returned list must:
- Preserve the structure of the original (`id`, `type`, `category`,
  `origin`, `origin_name`, etc.)
- Contain the clipped content (updated `media_bytes`/`media_string`,
  `duration`, and any type-specific fields)
- Default clippers return `[media]` unchanged

---

## Adding a Media Converter

Media converters transform content from one media type to another (e.g.
document pages to images, video to audio). Converters are
**auto-discovered** via `PluginRegistry` with the `CONVERTER` sentinel,
just like other plugin families.

### Built-in converters

| Converter | Source → Target | Description |
|-----------|----------------|-------------|
| `Document2ImageMediaConverter` | document → image | Render document pages as images |
| `Document2TextMediaConverter` | document → text | Extract embedded text from documents |
| `Video2AudioMediaConverter` | video → audio | Extract the audio track from a video |
| `Video2ImageMediaConverter` | video → image | Sample frames from a video as images |

### File structure

```
vtsearch/converters/<source>2<target>.py   # Your converter class
```

### What to implement

Subclass `MediaConverter` from `vtsearch.converters.base`.

```python
# vtsearch/converters/audio2text.py

from vtsearch.converters.base import MediaConverter
from typing import Any


class Audio2TextMediaConverter(MediaConverter):

    display_name = "Audio → Text"
    converter_description = "Transcribe audio to text using a speech model."

    @property
    def source_type(self) -> str:
        """The type_id of the input media type."""
        return "audio"

    @property
    def target_type(self) -> str:
        """The type_id of the output media type."""
        return "text"

    def convert(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert one media dict into one or more target-type media dicts.

        Each returned dict must include:
        - "filename": a descriptive filename
        - Data fields expected by the target type (e.g. "media_string"
          for text, "media_bytes" for images)

        Returns empty list if conversion fails.
        Does NOT include "id", "embedding", or "md5" — caller handles those.
        """
        # ... transcription logic ...
        return [{"filename": "transcript.txt", "media_string": transcript}]
```

### Register the converter

Expose a `CONVERTER` sentinel at module level in your converter file:

```python
# At the bottom of vtsearch/converters/audio2text.py

CONVERTER = Audio2TextMediaConverter()
```

The `PluginRegistry` auto-discovers `.py` files in `vtsearch/converters/`
that expose a `CONVERTER` attribute. No manual registration in
`__init__.py` is needed.

<!--
   Old explicit registration (no longer needed):
   ```python
   __all__ = [
       # ... existing entries ...
       "Audio2TextMediaConverter",
   ]
   ```
-->

### MediaConverter abstract interface reference

**Required abstract properties:**

| Property      | Returns | Description                                |
|---------------|---------|--------------------------------------------|
| `source_type` | `str`   | The `type_id` of the input media type      |
| `target_type` | `str`   | The `type_id` of the output media type     |

**Required abstract methods:**

| Method             | Signature              | Description                              |
|--------------------|------------------------|------------------------------------------|
| `convert(media)`   | `(dict) -> list[dict]` | Convert one media into target-type dicts  |

**Optional class attributes:**

| Attribute               | Type  | Default | Description                              |
|-------------------------|-------|---------|------------------------------------------|
| `display_name`          | `str` | `""`    | Human-readable label (auto-derived if empty) |
| `converter_description` | `str` | `""`    | Short description of the conversion      |

**Derived property (not overridable):**

| Property | Returns | Description |
|----------|---------|-------------|
| `name`   | `str`   | Auto-generated as `"{source_type}2{target_type}"` |

---

## Adding a Media Source

Media sources provide low-level access to media files at a location
(local folder, HTTP archive, S3 bucket, etc.). They sit *below* dataset
importers — importers that access file-like storage compose a
`MediaSource` for single-file resolution and cross-dataset label
re-ingestion.

Sources are **stateful** (e.g. an archive source may download and extract
on first access), so each call to `get_source_for_origin()` returns a
fresh instance. Callers should call `cleanup()` when done.

### File structure

```
vtsearch/datasets/sources/<your_source>/
└── __init__.py       # Source factory + SOURCE instance (required)
```

### What to implement

Unlike other plugin families, media sources use a **factory pattern**.
The `SOURCE` sentinel is a factory object with a `create_from_origin()`
method that returns a `MediaSource` instance.

```python
# vtsearch/datasets/sources/s3/__init__.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from vtsearch.datasets.sources.base import MediaItem, MediaSource


class S3MediaSource(MediaSource):
    """Access media files in an S3 bucket."""

    name = "s3"

    def __init__(self, bucket: str, prefix: str = "") -> None:
        self._bucket = bucket
        self._prefix = prefix

    def list_items(self, extensions: list[str] | None = None) -> Iterator[MediaItem]:
        """Yield all media items in the bucket (optionally filtered by extension)."""
        import boto3
        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=self._prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                filename = key.rsplit("/", 1)[-1]
                if extensions and not any(filename.lower().endswith(e) for e in extensions):
                    continue
                yield MediaItem(key=key, filename=filename, source_name=self.name)

    def fetch_item(self, key: str) -> Path | None:
        """Download an item to a temp directory and return the local path."""
        import boto3, tempfile
        local = Path(tempfile.gettempdir()) / "vtsearch_s3" / key
        local.parent.mkdir(parents=True, exist_ok=True)
        if not local.exists():
            boto3.client("s3").download_file(self._bucket, key, str(local))
        return local

    def resolve_path(self, origin_name: str = "", filename: str = "") -> Path | None:
        """Resolve a media file by origin_name or filename."""
        for candidate in (origin_name, filename):
            if candidate:
                key = f"{self._prefix}{candidate}" if self._prefix else candidate
                path = self.fetch_item(key)
                if path and path.exists():
                    return path
        return None


class _S3SourceFactory:
    """Factory that creates S3MediaSource instances from origin dicts."""

    name = "s3"

    def create_from_origin(self, origin: dict[str, Any]) -> S3MediaSource | None:
        params = origin.get("params", {})
        bucket = params.get("bucket", "")
        if not bucket:
            return None
        return S3MediaSource(bucket, params.get("prefix", ""))


SOURCE = _S3SourceFactory()
```

### MediaSource abstract interface reference

**Required abstract methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `list_items()` | `(extensions: list[str] \| None) -> Iterator[MediaItem]` | Yield all media items, optionally filtered |
| `fetch_item()` | `(key: str) -> Path \| None` | Return local path for item (may download on demand) |
| `resolve_path()` | `(origin_name: str, filename: str) -> Path \| None` | Find a file by origin_name or filename |

**Optional methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `cleanup()` | `() -> None` | Release temporary resources (default: no-op) |

**Data types:**

| Type | Fields | Description |
|------|--------|-------------|
| `MediaItem` | `key`, `filename`, `source_name` | A discoverable file within a source |

### How it gets invoked

`get_source_for_origin(origin_dict)` looks up the factory by matching
`origin["importer"]` to the factory's `name`, then calls
`factory.create_from_origin(origin)`.

---

## Processor System

Processors analyze media items. The hierarchy has a common base
(`Processor`) with three concrete subtypes. All are defined in
`vtsearch/media/base.py`.

```
Processor (ABC)
├── Detector      — "does this media match?"      → bool
├── Localizer     — "where in this media?"        → list[dict] (bounding boxes)
└── Extractor     — "what details are inside?"    → list[dict] (structured results)
```

Each processor operates on exactly one media type.

### Adding a Detector

A Detector answers "is this media Good?" with a boolean.

```python
from vtsearch.media.base import Detector
from typing import Any


class LoudnessDetector(Detector):

    @property
    def name(self) -> str:
        return "loud_audio"

    @property
    def media_type(self) -> str:
        return "audio"

    def load_model(self) -> None:
        """Optional: load heavyweight resources once before first use."""
        pass

    def detect(self, media: dict[str, Any]) -> bool:
        """Return True if the media matches, False otherwise."""
        return media.get("duration", 0) > 5.0
```

### Adding a Localizer

A Localizer returns bounding boxes with confidence scores.

```python
from vtsearch.media.base import Localizer
from typing import Any


class FaceLocalizer(Localizer):

    @property
    def name(self) -> str:
        return "face_localizer"

    @property
    def media_type(self) -> str:
        return "image"

    def localize(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        """Return bounding boxes. Each dict must include:
        - "confidence": float in [0, 1]
        - "bbox": bounding box (format is media-specific)

        Returns empty list when nothing is found.
        """
        # ... face detection logic ...
        return [
            {"confidence": 0.95, "bbox": [10, 20, 200, 300]},
            {"confidence": 0.73, "bbox": [400, 50, 600, 250]},
        ]
```

### Adding an Extractor

An Extractor returns structured details for each occurrence found.

```python
from vtsearch.media.base import Extractor
from typing import Any


class ObjectExtractor(Extractor):

    @property
    def name(self) -> str:
        return "object_extractor"

    @property
    def media_type(self) -> str:
        return "image"

    def extract(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        """Return a list of result dicts. Each dict must include
        a "confidence" key (float in [0, 1]).

        Returns empty list when nothing is found.
        """
        # ... object detection logic ...
        return [
            {"confidence": 0.92, "bbox": [10, 20, 200, 300], "label": "car"},
            {"confidence": 0.87, "bbox": [400, 50, 600, 250], "label": "tree"},
        ]
```

### Processor abstract interface reference

**Processor (base class) — required:**

| Member | Type | Description |
|--------|------|-------------|
| `name` | `str` (property) | Unique identifier |
| `media_type` | `str` (property) | Which media type it operates on |
| `process(media)` | `(dict) -> Any` | Run the processor (delegates to subclass) |

**Processor (base class) — optional:**

| Member | Type | Description |
|--------|------|-------------|
| `load_model()` | `() -> None` | One-time model loading (default: no-op) |

**Detector:**

| Member | Signature | Description |
|--------|-----------|-------------|
| `detect(media)` | `(dict) -> bool` | Return True if media matches |

**Localizer:**

| Member | Signature | Description |
|--------|-----------|-------------|
| `localize(media)` | `(dict) -> list[dict]` | Return bounding boxes with `confidence` and `bbox` |

**Extractor:**

| Member | Signature | Description |
|--------|-----------|-------------|
| `extract(media)` | `(dict) -> list[dict]` | Return result dicts with `confidence` key |

---

## Dependency Management

Runtime dependencies are managed via **per-plugin `requirements.txt`
files**, auto-discovered by `install-plugin-deps.sh`. Each plugin
sub-package (media type, importer, exporter, etc.) can include its own
`requirements.txt` in its directory.

```
vtsearch/media/image/requirements.txt     # Pillow, ultralytics, …
vtsearch/media/audio/requirements.txt     # librosa, soundfile, …
vtsearch/exporters/webhook/requirements.txt
```

Running `bash install-plugin-deps.sh` regenerates
`requirements-plugins.txt` with `-r` references to each plugin's file.
The top-level `requirements.txt` includes this, so `pip install -r
requirements.txt` installs everything.

`pyproject.toml` is kept minimal — only `cpu`/`gpu` (for choosing the
right PyTorch build) and `dev` (for test/lint tools) live there.

### For a new media type, importer, or exporter

Add a `requirements.txt` inside your plugin's directory, then run
`bash install-plugin-deps.sh` to regenerate the dependency tree. Failed
imports of a plugin's sub-package emit a warning rather than crashing, so
missing dependencies degrade gracefully.

---

## Quick Reference: Checklist for Each Extension Type

### New Data Importer Checklist

- [ ] Create `vtsearch/datasets/importers/<name>/__init__.py`
- [ ] Subclass `DatasetImporter`, set `name`, `display_name`, `description`, `fields`
- [ ] Implement `run(self, field_values, medias, thin=False)` — populate `medias` in-place
- [ ] Expose `IMPORTER = YourImporter()` at module level
- [ ] Add a `requirements.txt` in the plugin directory and run `bash install-plugin-deps.sh`
- [ ] Test: start the app and check `GET /api/dataset/all-importers` includes your importer

### New Results Exporter Checklist

- [ ] Create `vtsearch/exporters/<name>/__init__.py`
- [ ] Subclass `LabelsetExporter`, set `name`, `display_name`, `description`, `fields`
- [ ] Implement `export(self, results, field_values)` — return a dict with a `"message"` key
- [ ] Expose `EXPORTER = YourExporter()` at module level
- [ ] Add a `requirements.txt` in the plugin directory and run `bash install-plugin-deps.sh`
- [ ] Test: start the app and check `GET /api/exporters` includes your exporter

### New Label Importer Checklist

- [ ] Create `vtsearch/labels/importers/<name>/__init__.py`
- [ ] Subclass `LabelImporter`, set `name`, `display_name`, `description`, `fields`
- [ ] Implement `run(self, field_values)` — return a list of `{"md5": ..., "label": ...}` dicts
- [ ] Expose `LABEL_IMPORTER = YourImporter()` at module level
- [ ] Add a `requirements.txt` in the plugin directory and run `bash install-plugin-deps.sh`
- [ ] Test: start the app and check `GET /api/label-importers` includes your importer

### New Processor Importer Checklist

- [ ] Create `vtsearch/processors/importers/<name>/__init__.py`
- [ ] Subclass `ProcessorImporter`, set `name`, `display_name`, `description`, `fields`
- [ ] Implement `run(self, field_values)` — return a dict with `media_type`, `weights`, `threshold`
- [ ] Expose `PROCESSOR_IMPORTER = YourImporter()` at module level
- [ ] Add a `requirements.txt` in the plugin directory and run `bash install-plugin-deps.sh`
- [ ] Test: start the app and check `GET /api/processor-importers` includes your importer

### New Settings Source Checklist

- [ ] Create `vtsearch/settings_io/sources/<name>/__init__.py`
- [ ] Subclass `SettingsSource`, set `name`, `display_name`, `description`, `fields`
- [ ] Implement `load(self, field_values)` — return a settings dict
- [ ] Implement `save(self, settings_data, field_values)` — persist settings
- [ ] Expose `SETTINGS_SOURCE = YourSource()` at module level
- [ ] Add a `requirements.txt` in the plugin directory and run `bash install-plugin-deps.sh`
- [ ] Test: start the app and check `GET /api/settings-sources` includes your source

### New Labelset Source Checklist

- [ ] Create `vtsearch/labels/sources/<name>/__init__.py`
- [ ] Subclass `LabelsetSource`, set `name`, `display_name`, `description`, `fields`
- [ ] Implement `load(self, field_values)` — return a list of label dicts
- [ ] Implement `save(self, labelset, field_values)` — persist a `LabelSet`
- [ ] Expose `LABELSET_SOURCE = YourSource()` at module level
- [ ] Add a `requirements.txt` in the plugin directory and run `bash install-plugin-deps.sh`
- [ ] Test: start the app and check `GET /api/labelset-sources` includes your source

### New Settings Importer Checklist

- [ ] Create `vtsearch/settings_io/importers/<name>/__init__.py`
- [ ] Subclass `SettingsImporter`, set `name`, `display_name`, `description`, `fields`
- [ ] Implement `run(self, field_values)` — return a settings dict
- [ ] Expose `SETTINGS_IMPORTER = YourImporter()` at module level
- [ ] Test: start the app and check `GET /api/settings-importers` includes your importer

### New Settings Exporter Checklist

- [ ] Create `vtsearch/settings_io/exporters/<name>/__init__.py`
- [ ] Subclass `SettingsExporter`, set `name`, `display_name`, `description`, `fields`
- [ ] Implement `export(self, settings_data, field_values)` — return dict with `"message"` key
- [ ] Expose `SETTINGS_EXPORTER = YourExporter()` at module level
- [ ] Test: start the app and check `GET /api/settings-exporters` includes your exporter

### New Media Source Checklist

- [ ] Create `vtsearch/datasets/sources/<name>/__init__.py`
- [ ] Create a `MediaSource` subclass with `list_items()`, `fetch_item()`, `resolve_path()`
- [ ] Create a factory class with `name` and `create_from_origin(origin)` method
- [ ] Expose `SOURCE = YourFactory()` at module level
- [ ] Test: create an origin dict for your source and verify `get_source_for_origin()` returns it

### New Media Type Checklist

- [ ] Create `vtsearch/media/<type>/` directory with `__init__.py`, `media_type.py`
- [ ] Subclass `MediaType` and implement all abstract properties and methods
- [ ] Expose `MEDIA_TYPE`, `EMBEDDERS`, and `CLIPPERS` sentinels in `__init__.py`
- [ ] Add a `requirements.txt` in the plugin directory and run `bash install-plugin-deps.sh`
- [ ] Override `pickle_extra_fields` if you use custom clip keys
- [ ] Test: import a folder of your media type, verify clips appear and are sortable

### New Media Embedder Checklist

- [ ] Create `vtsearch/media/<type>/embedder.py` (or `embedder_<variant>.py` for alternatives)
- [ ] Subclass `MediaEmbedder`, implement `name`, `media_type_id`, `_load_models_impl()`, `embed_media()`
- [ ] Optionally implement `embed_text()` for text-query sorting
- [ ] Optionally set `description_wrappers` for enriched text embedding
- [ ] Add to the `EMBEDDERS` list in the media type's `__init__.py`
- [ ] Test: load a dataset and verify embeddings are generated

### New Media Clipper Checklist

- [ ] Create or add to `vtsearch/media/<type>/clipper.py`
- [ ] Subclass `MediaClipper`, implement `name`, `media_type`, `clip()`
- [ ] Override `description` with a short tooltip string for the chooser UI
- [ ] If adding `parameters`, include a `description` key in each param dict
- [ ] Add to the `CLIPPERS` list in the media type's `__init__.py`
- [ ] Test: verify `clip()` returns valid media dicts

### New Media Converter Checklist

- [ ] Create `vtsearch/converters/<source>2<target>.py`
- [ ] Subclass `MediaConverter`, implement `source_type`, `target_type`, and `convert()`
- [ ] Expose `CONVERTER = YourConverter()` at module level
- [ ] Test: convert a source-type media and verify output dicts are valid

### New Detector / Localizer / Extractor Checklist

- [ ] Subclass `Detector`, `Localizer`, or `Extractor` from `vtsearch.media.base`
- [ ] Implement `name`, `media_type`, and the type-specific method (`detect`, `localize`, or `extract`)
- [ ] Optionally override `load_model()` for one-time resource loading
- [ ] Register as autorun via `POST /api/autorun-detectors` (or extractors/localizers)

### New Login Provider Checklist

- [ ] Create a new module (e.g. `vtsearch/auth/my_provider.py`)
- [ ] Subclass `LoginProvider` from `vtsearch.auth`
- [ ] Implement `get_user(request)` and `is_authenticated(request)`
- [ ] Override `login_required()` if the frontend should show a login screen
- [ ] Override `get_user_data_dir(username, base_data_dir)` for per-user isolation
- [ ] Call `set_login_provider(MyProvider())` at app startup (in `app.py`)

---

## Authentication Providers

VTSearch uses a pluggable `LoginProvider` ABC (`vtsearch/auth/__init__.py`)
so that multi-user deployments can be supported without modifying routes.

### Interface

```python
from vtsearch.auth import LoginProvider

class MyProvider(LoginProvider):
    name = "my_provider"

    def get_user(self, request) -> str:
        """Return username from the request (e.g. header, cookie, cert)."""
        return request.headers.get("X-User", "anonymous")

    def is_authenticated(self, request) -> bool:
        """Return True if the request is authenticated."""
        return "X-User" in request.headers

    def login_required(self) -> bool:
        """Return True to show a login screen in the frontend."""
        return True

    def get_user_data_dir(self, username: str, base_data_dir: Path) -> Path:
        """Return a per-user data directory for isolated storage."""
        return base_data_dir / username
```

### How it works

1. `set_login_provider(provider)` is called once at startup (in `app.py`).
2. The `before_request` middleware calls `provider.get_user(request)` and
   stores the result in `g.user`.
3. Routes call `get_current_user()` to read `g.user`.
4. `GET /api/auth/status` calls `provider.status_dict(request)`.

### Built-in provider

`DefaultLoginProvider` (the default) returns `"default"` for every request,
is always authenticated, and uses the shared `data/` directory.

### Current limitations

In-memory state (votes, medias, labels, settings) is globally shared.
The auth infrastructure supports ownership tracking (`created_by`) and
per-user data directories, but full runtime state isolation is not yet
implemented. Custom providers should be aware that votes and loaded
datasets are shared across all users.
