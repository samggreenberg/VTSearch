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
- [Media system](#media-system) (explicit registration)
  - [Media Types](#adding-a-media-type)
  - [Media Embedders](#adding-a-media-embedder)
  - [Media Clippers](#adding-a-media-clipper)
  - [Media Converters](#adding-a-media-converter)
- [Processor system](#processor-system) (Detectors, Localizers, Extractors)
  - [Detectors](#adding-a-detector)
  - [Localizers](#adding-a-localizer)
  - [Extractors](#adding-an-extractor)
- [Dependency Management](#dependency-management)
- [Quick Reference Checklists](#quick-reference-checklist-for-each-extension-type)

---

## Shared Plugin Architecture

The four plugin systems — data importers, results exporters, label
importers, and processor importers — share the same architecture built on
two base classes in `vtsearch/utils/registry.py`:

### PluginField

A dataclass describing a single user-configurable input. All four plugin
families use the same field type (aliased as `ImporterField`,
`ExporterField`, `LabelImporterField`, `ProcessorImporterField`).

| Parameter     | Type        | Default  | Description                                             |
|---------------|-------------|----------|---------------------------------------------------------|
| `key`         | `str`       | —        | Field identifier (dict key in `field_values`)           |
| `label`       | `str`       | —        | Display label in the UI                                 |
| `field_type`  | `FieldType` | —        | `"text"`, `"url"`, `"folder"`, `"file"`, `"password"`, `"email"`, or `"select"` |
| `description` | `str`       | `""`     | Helper text shown below the field                       |
| `accept`      | `str`       | `""`     | For `"file"` fields: comma-separated extensions (e.g. `".pkl"`) |
| `options`     | `list[str]` | `[]`     | For `"select"` fields: allowed dropdown values          |
| `default`     | `str`       | `""`     | Pre-filled value                                        |
| `required`    | `bool`      | `True`   | Whether the field must be filled before submitting      |
| `placeholder` | `str`       | `""`     | Hint shown as placeholder text in the input widget      |

### PluginBase

Shared base class providing CLI-argument derivation, validation, and
serialisation. All four plugin base classes inherit from it.

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

All four plugin families use `PluginRegistry` for auto-discovery. The
registry uses `pkgutil.iter_modules` to scan for **sub-packages**
(directories with `__init__.py`) under the plugin directory. For each
sub-package, it imports the module and looks for a module-level sentinel
attribute. If found, the plugin is registered by its `name`.

| Plugin Family      | Package                            | Sentinel             | Base Class          |
|--------------------|------------------------------------|----------------------|---------------------|
| Data Importers     | `vtsearch.datasets.importers`      | `IMPORTER`           | `DatasetImporter`   |
| Results Exporters  | `vtsearch.exporters`               | `EXPORTER`           | `LabelsetExporter`  |
| Label Importers    | `vtsearch.labels.importers`        | `LABEL_IMPORTER`     | `LabelImporter`     |
| Processor Importers| `vtsearch.processors.importers`    | `PROCESSOR_IMPORTER` | `ProcessorImporter` |

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
├── __init__.py       # Importer class + IMPORTER instance (required)
└── requirements.txt  # Pip dependencies, even if empty (required)
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
            default="sounds",
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
        media_type = field_values.get("media_type", "sounds")

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
UI.

### How it gets invoked

1. `GET /api/dataset/importers` returns available importers (your importer
   appears automatically).
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
    --media-type sounds --settings settings.json
```

CLI arguments are auto-generated from `fields`. Override `run_cli()` if
your `run()` expects non-string values (e.g. FileStorage objects).

### Wiring up dependencies

1. Create `vtsearch/datasets/importers/<name>/requirements.txt`
2. Add `-r vtsearch/datasets/importers/<name>/requirements.txt` to
   `requirements-importers.txt`
3. Add packages inline to `requirements-cpu.txt` if needed

---

## Adding a Results Exporter

Results exporters deliver autodetect results to a destination (file,
webhook, email, etc.). Auto-discovered — no changes to routes needed.

### File structure

```
vtsearch/exporters/<your_exporter>/
├── __init__.py       # Exporter class + EXPORTER instance (required)
└── requirements.txt  # Pip dependencies, even if empty (required)
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

| Endpoint                  | Method | What it exports                           | Format          |
|---------------------------|--------|-------------------------------------------|-----------------|
| `/api/dataset/export`     | GET    | Full dataset (clips + embeddings + media)  | Pickle (`.pkl`) |
| `/api/labels/export`      | GET    | LabelSet — labels with per-element origin  | JSON            |
| `/api/detector/export`    | POST   | Trained MLP weights + threshold            | JSON            |

### Wiring up dependencies

1. Create `vtsearch/exporters/<name>/requirements.txt`
2. Add `-r vtsearch/exporters/<name>/requirements.txt` to
   `requirements-exporters.txt`
3. Add packages inline to `requirements-cpu.txt` if needed

---

## Adding a Label Importer

Label importers let users import pre-existing labels (good/bad votes) from
external sources. Auto-discovered at runtime.

### File structure

```
vtsearch/labels/importers/<your_importer>/
├── __init__.py       # Importer class + LABEL_IMPORTER instance (required)
└── requirements.txt  # Pip dependencies, even if empty (required)
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
├── __init__.py       # Importer class + PROCESSOR_IMPORTER instance (required)
└── requirements.txt  # Pip dependencies, even if empty (required)
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
            - "media_type" (str): e.g. "audio", "image"
            - "weights" (dict): MLP state dict as nested lists
            - "threshold" (float): decision boundary in [0, 1]
        May also include:
            - "name" (str): suggested default name
        """
        import json
        import boto3

        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=field_values["bucket"], Key=field_values["key"])
        data = json.loads(obj["Body"].read())
        return {
            "media_type": data.get("media_type", "audio"),
            "weights": data["weights"],
            "threshold": data.get("threshold", 0.5),
        }


PROCESSOR_IMPORTER = S3ProcessorImporter()
```

### ProcessorImporter class reference

**Required to implement:**

| Member | Signature | Description |
|--------|-----------|-------------|
| `run()` | `(field_values: dict) -> dict` | Return dict with `media_type`, `weights`, `threshold` |

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

## Media System

Media types, embedders, and clippers use **explicit registration** in
`vtsearch/media/__init__.py`. Unlike the plugin systems above, they are
not auto-discovered — you import your class and call the appropriate
`register*()` function.

The three registries:

| Registry | Function | Keyed by |
|----------|----------|----------|
| Media types | `register(media_type)` | `MediaType.type_id` |
| Embedders | `register_embedder(embedder)` | `MediaEmbedder.name` |
| Clippers | `register_clipper(clipper)` | `MediaClipper.name` |

---

## Adding a Media Type

Media types define how VTSearch handles a particular kind of content: how
to serve clips over HTTP, what file extensions to scan for, what demo
datasets are available, and how to load media-specific fields from files.

### File structure

```
vtsearch/media/<your_type>/
├── __init__.py       # Can be empty
├── media_type.py     # Your MediaType subclass (required)
└── requirements.txt  # Pip dependencies (required, even if empty)
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
        return "💻"

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

Add two lines to `vtsearch/media/__init__.py`, alongside the existing
registrations at the bottom of the file:

```python
from vtsearch.media.code.media_type import CodeMediaType   # noqa: E402
register(CodeMediaType())
```

### MediaType abstract interface reference

**Required abstract properties:**

| Property          | Returns     | Example                              |
|-------------------|-------------|--------------------------------------|
| `type_id`         | `str`       | `"audio"`, `"image"`, `"code"`       |
| `name`            | `str`       | `"Audio"`, `"Source Code"`           |
| `icon`            | `str`       | `"🔊"`, `"💻"`                       |
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
| `folder_import_name` | `str`       | `type_id`          | Alias for folder imports (e.g. `"sounds"`) |
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
├── embedder.py      # Your MediaEmbedder subclass (required)
└── requirements.txt # Dependencies (already exists for the media type)
```

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

    def load_models(self) -> None:
        """Load the embedding model. Must be idempotent."""
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._on_progress("loading", "Loading CodeBERT…", 0, 0)
        self._model = SentenceTransformer("microsoft/codebert-base")

    # --- Embedding (required abstract method) ---

    def embed_media(self, file_path: Path) -> Optional[np.ndarray]:
        """Return a fixed-size embedding vector for the file.

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

Add to `vtsearch/media/__init__.py`:

```python
from vtsearch.media.code.embedder import CodeBertEmbedder  # noqa: E402
register_embedder(CodeBertEmbedder())
```

### MediaEmbedder abstract interface reference

**Required abstract properties:**

| Property        | Returns | Description                              |
|-----------------|---------|------------------------------------------|
| `name`          | `str`   | Unique identifier (e.g. `"clap"`, `"clip"`) |
| `media_type_id` | `str`  | Which media type this embedder works with |

**Required abstract methods:**

| Method                    | Signature                        | Description                    |
|---------------------------|----------------------------------|--------------------------------|
| `load_models()`           | `() -> None`                     | Load model; must be idempotent |
| `embed_media(file_path)`  | `(Path) -> Optional[np.ndarray]` | Embed a media file             |

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
| `ImageClipEmbedder` | `clip` | `image` | OpenAI CLIP (openai/clip-vit-base-patch32) | 512 |
| `TextE5Embedder` | `e5` | `paragraph` | E5-base-v2 (intfloat/e5-base-v2) | 768 |
| `VideoXClipEmbedder` | `xclip` | `video` | X-CLIP (microsoft/xclip-base-patch32) | 512 |

---

## Adding a Media Clipper

Media clippers split a single media item into one or more items of the
**same** type. Unlike processors which return metadata about media,
clippers return **new media dicts** that can replace the original.

### Built-in clippers

| Clipper | Name | Media Type | Description |
|---------|------|------------|-------------|
| `SoundDefaultClipper` | `sound_default` | `audio` | Returns audio unchanged |
| `SoundTilingClipper` | `sound_tiling_2.0s` | `audio` | Tiles into 2s segments |
| `ImageDefaultClipper` | `image_default` | `image` | Returns image unchanged |
| `ImageTilingClipper` | `image_tiling` | `image` | Tiles tall images into squares |
| `TextDefaultClipper` | `text_default` | `paragraph` | Returns text unchanged |
| `TextSentenceClipper` | `text_sentence` | `paragraph` | Splits into individual sentences |
| `VideoDefaultClipper` | `video_default` | `video` | Returns video unchanged |
| `VideoTilingClipper` | `video_tiling_2.0s` | `video` | Tiles into 2s segments |
| `DocumentDefaultClipper` | `document_default` | `document` | Returns document unchanged |

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

Add to `vtsearch/media/__init__.py`:

```python
from vtsearch.media.audio.clipper import SoundOverlapClipper  # noqa: E402
register_clipper(SoundOverlapClipper(2.0))
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

**Optional overridable methods:**

| Method       | Signature      | Description                                              |
|--------------|----------------|----------------------------------------------------------|
| `to_dict()`  | `() -> dict`   | JSON-serialisable metadata (default: name + media_type)  |

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
document pages to images, video to audio). Converters use **explicit
registration** in `vtsearch/converters/__init__.py` — they are not
auto-discovered.

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
        return "paragraph"

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

Edit `vtsearch/converters/__init__.py`:

1. Add the import at the top:
   ```python
   from vtsearch.converters.audio2text import Audio2TextMediaConverter
   ```

2. Add to `_ALL_CONVERTERS`:
   ```python
   _ALL_CONVERTERS: list[MediaConverter] = [
       # ... existing converters ...
       Audio2TextMediaConverter(),
   ]
   ```

3. Add to `__all__`:
   ```python
   __all__ = [
       # ... existing entries ...
       "Audio2TextMediaConverter",
   ]
   ```

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

VTSearch uses a layered requirements file structure:

```
requirements.txt              # Core deps + includes per-media + per-importer + per-exporter
├── vtsearch/media/audio/requirements.txt
├── vtsearch/media/video/requirements.txt
├── vtsearch/media/image/requirements.txt
├── vtsearch/media/text/requirements.txt
├── vtsearch/media/document/requirements.txt
├── requirements-importers.txt          # Aggregates all data importer deps
│   ├── vtsearch/datasets/importers/pickle/requirements.txt
│   ├── vtsearch/datasets/importers/folder/requirements.txt
│   ├── vtsearch/datasets/importers/http_zip/requirements.txt
│   └── vtsearch/datasets/importers/combine_datasets/requirements.txt
├── requirements-exporters.txt          # Aggregates all exporter deps
│   ├── vtsearch/exporters/gui/requirements.txt
│   ├── vtsearch/exporters/server_json_file/requirements.txt
│   ├── vtsearch/exporters/server_csv_file/requirements.txt
│   ├── vtsearch/exporters/email_smtp/requirements.txt
│   └── vtsearch/exporters/webhook/requirements.txt
├── vtsearch/labels/importers/server_json_file/requirements.txt
├── vtsearch/labels/importers/server_csv_file/requirements.txt
├── vtsearch/processors/importers/server_detector_file/requirements.txt
requirements-cpu.txt          # CPU-specific pins (lists packages INLINE)
requirements-gpu.txt          # GPU-specific (minimal, includes importers)
requirements-dev.txt          # Dev tools (pytest)
```

### For a new media type

1. Create `vtsearch/media/<type>/requirements.txt`
2. Add `-r vtsearch/media/<type>/requirements.txt` to `requirements.txt`
3. Add packages inline to `requirements-cpu.txt`

### For a new data importer

1. Create `vtsearch/datasets/importers/<name>/requirements.txt`
2. Add `-r` line to `requirements-importers.txt`
3. Add packages inline to `requirements-cpu.txt` if needed

### For a new exporter

1. Create `vtsearch/exporters/<name>/requirements.txt`
2. Add `-r` line to `requirements-exporters.txt`
3. Add packages inline to `requirements-cpu.txt` if needed

### Why the layered structure?

- Each component owns its own `requirements.txt` so it's obvious which
  packages belong to which feature.
- The aggregator files tie everything together for `pip install -r
  requirements.txt`.
- `requirements-cpu.txt` duplicates packages inline with version pins
  because CPU-only PyTorch wheels require a special `--extra-index-url`.
- Failed imports of a plugin's sub-package emit a warning rather than
  crashing, so missing optional dependencies degrade gracefully.

---

## Quick Reference: Checklist for Each Extension Type

### New Data Importer Checklist

- [ ] Create `vtsearch/datasets/importers/<name>/__init__.py`
- [ ] Subclass `DatasetImporter`, set `name`, `display_name`, `description`, `fields`
- [ ] Implement `run(self, field_values, medias, thin=False)` — populate `medias` in-place
- [ ] Expose `IMPORTER = YourImporter()` at module level
- [ ] Create `vtsearch/datasets/importers/<name>/requirements.txt`
- [ ] Add `-r` line to `requirements-importers.txt`
- [ ] Add inline deps to `requirements-cpu.txt` if needed
- [ ] Test: start the app and check `GET /api/dataset/importers` includes your importer

### New Results Exporter Checklist

- [ ] Create `vtsearch/exporters/<name>/__init__.py`
- [ ] Subclass `LabelsetExporter`, set `name`, `display_name`, `description`, `fields`
- [ ] Implement `export(self, results, field_values)` — return a dict with a `"message"` key
- [ ] Expose `EXPORTER = YourExporter()` at module level
- [ ] Create `vtsearch/exporters/<name>/requirements.txt`
- [ ] Add `-r` line to `requirements-exporters.txt`
- [ ] Add inline deps to `requirements-cpu.txt` if needed
- [ ] Test: start the app and check `GET /api/exporters` includes your exporter

### New Label Importer Checklist

- [ ] Create `vtsearch/labels/importers/<name>/__init__.py`
- [ ] Subclass `LabelImporter`, set `name`, `display_name`, `description`, `fields`
- [ ] Implement `run(self, field_values)` — return a list of `{"md5": ..., "label": ...}` dicts
- [ ] Expose `LABEL_IMPORTER = YourImporter()` at module level
- [ ] Create `vtsearch/labels/importers/<name>/requirements.txt`
- [ ] Test: start the app and check `GET /api/label-importers` includes your importer

### New Processor Importer Checklist

- [ ] Create `vtsearch/processors/importers/<name>/__init__.py`
- [ ] Subclass `ProcessorImporter`, set `name`, `display_name`, `description`, `fields`
- [ ] Implement `run(self, field_values)` — return a dict with `media_type`, `weights`, `threshold`
- [ ] Expose `PROCESSOR_IMPORTER = YourImporter()` at module level
- [ ] Create `vtsearch/processors/importers/<name>/requirements.txt`
- [ ] Test: start the app and check `GET /api/processor-importers` includes your importer

### New Media Type Checklist

- [ ] Create `vtsearch/media/<type>/` directory with `__init__.py`, `media_type.py`, `requirements.txt`
- [ ] Subclass `MediaType` and implement all abstract properties and methods
- [ ] Register in `vtsearch/media/__init__.py` with `register(YourType())`
- [ ] Add `-r` line to `requirements.txt` pointing to your `requirements.txt`
- [ ] Add inline deps to `requirements-cpu.txt`
- [ ] Override `pickle_extra_fields` if you use custom clip keys
- [ ] Test: import a folder of your media type, verify clips appear and are sortable

### New Media Embedder Checklist

- [ ] Create `vtsearch/media/<type>/embedder.py`
- [ ] Subclass `MediaEmbedder`, implement `name`, `media_type_id`, `load_models()`, `embed_media()`
- [ ] Optionally implement `embed_text()` for text-query sorting
- [ ] Optionally set `description_wrappers` for enriched text embedding
- [ ] Register in `vtsearch/media/__init__.py` with `register_embedder(YourEmbedder())`
- [ ] Test: load a dataset and verify embeddings are generated

### New Media Clipper Checklist

- [ ] Create or add to `vtsearch/media/<type>/clipper.py`
- [ ] Subclass `MediaClipper`, implement `name`, `media_type`, `clip()`
- [ ] Register in `vtsearch/media/__init__.py` with `register_clipper(YourClipper())`
- [ ] Test: verify `clip()` returns valid media dicts

### New Media Converter Checklist

- [ ] Create `vtsearch/converters/<source>2<target>.py`
- [ ] Subclass `MediaConverter`, implement `source_type`, `target_type`, and `convert()`
- [ ] Add import and list entry in `vtsearch/converters/__init__.py`
- [ ] Add to `__all__` in `vtsearch/converters/__init__.py`
- [ ] Test: convert a source-type media and verify output dicts are valid

### New Detector / Localizer / Extractor Checklist

- [ ] Subclass `Detector`, `Localizer`, or `Extractor` from `vtsearch.media.base`
- [ ] Implement `name`, `media_type`, and the type-specific method (`detect`, `localize`, or `extract`)
- [ ] Optionally override `load_model()` for one-time resource loading
- [ ] Register as autorun via `POST /api/autorun-detectors` (or extractors/localizers)
