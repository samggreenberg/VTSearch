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
- [Adding a Media Converter](#adding-a-media-converter)
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
`vtscore/plugins/__init__.py`:

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
(`vtscore.datasets.sources`), which use flat `.py` modules
(`local_folder.py`, `http_archive.py`, `pullwrest.py`).

| Plugin Family       | Package                            | Sentinel              | Base Class          | Entry-point group              |
|---------------------|------------------------------------|-----------------------|---------------------|--------------------------------|
| Data Importers      | `vtscore.datasets.importers`      | `IMPORTER`            | `DatasetImporter`   | `vtscore.importers`            |
| Results Exporters   | `vtscore.exporters`               | `EXPORTER`            | `LabelsetExporter`  | `vtscore.exporters`            |
| Label Importers     | `vtscore.labels.importers`        | `LABEL_IMPORTER`      | `LabelImporter`     | `vtscore.label_importers`      |
| Processor Importers | `vtsearch.processors.importers`    | `PROCESSOR_IMPORTER`  | `ProcessorImporter` | —                              |
| Settings Importers  | `vtsearch.settings_io.importers`   | `SETTINGS_IMPORTER`   | `SettingsImporter`  | `vtsearch.settings_importers`  |
| Settings Exporters  | `vtsearch.settings_io.exporters`   | `SETTINGS_EXPORTER`   | `SettingsExporter`  | `vtsearch.settings_exporters`  |
| Settings Sources    | `vtsearch.settings_io.sources`     | `SETTINGS_SOURCE`     | `SettingsSource`    | `vtsearch.settings_sources`    |
| Labelset Sources    | `vtscore.labels.sources`          | `LABELSET_SOURCE`     | `LabelsetSource`    | `vtscore.labelset_sources`     |
| Media Converters    | `vtscore.converters`              | `CONVERTER`           | `MediaConverter`    | `vtscore.converters`           |
| Media Sources       | `vtscore.datasets.sources`        | `SOURCE`              | `MediaSource`       | `vtscore.media_sources`        |

Failed imports emit a warning but do not break the application — a missing
optional dependency gracefully disables that plugin.

### Third-party plugins via `importlib.metadata` entry points

Plugins don't have to live inside the `vtsearch` source tree. Any installed
Python distribution can register a plugin by declaring an entry point in
the family's group (see the rightmost column above). For example, a
third-party importer would add this to its `pyproject.toml`:

```toml
[project.entry-points."vtscore.importers"]
my_importer = "my_pkg.importer:IMPORTER"
```

The value (`my_pkg.importer:IMPORTER`) must resolve to an already-instantiated
plugin object — the same shape that the in-tree sentinel attribute holds.
After `pip install` of the third-party package, the plugin appears in
`list_importers()`, the relevant `/api/...` endpoint, and `python app.py
--list-plugins` without any changes to the core repo.

Built-in plugins take precedence: if an entry point's `name` clashes with
a name already registered by the package scan, it is skipped and a warning
is emitted. A broken entry point (import error, missing `name` attribute)
warns and is skipped — it cannot block discovery of other plugins.

### Listing every registered plugin

`python app.py --list-plugins` prints every auto-discovered plugin
across all families — useful both for humans (`--format plain`, the
default) and for shell completion scripts (`--format names`). Add
`--plugin-family <name>` to scope the output, e.g.
`python app.py --list-plugins --plugin-family importers --format names`
emits one importer name per line.

Output formats:

| Flag value      | Use case                                                |
|-----------------|---------------------------------------------------------|
| `plain` (default) | Human-readable grouped table                          |
| `json`          | Machine-readable; full `{name, display_name, description}` for each plugin |
| `names`         | One name per line; with a family, bare names — without one, `family:name` pairs |

The same inventory is also available programmatically as
`vtscore.plugins.inventory.gather_plugins()`.

---

## Adding a Data Importer

Data importers let users load datasets from new sources (S3 buckets,
databases, APIs, etc.). The system auto-discovers importers at runtime — no
changes to routes or core code are needed.

### File structure

```
vtscore/datasets/importers/<your_importer>/
└── __init__.py       # Importer class + IMPORTER instance (required)
```

### What to implement

Subclass `DatasetImporter` from `vtscore.datasets.importers.base`.
Set the required class attributes and implement the `run()` method.
Expose a module-level `IMPORTER` instance.

```python
# vtscore/datasets/importers/s3/__init__.py

from vtscore.datasets.importers.base import DatasetImporter, ImporterField
from vtscore.media import all_folder_names


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
        from vtscore.config import DATA_DIR
        from vtscore.concurrency.progress import update_progress

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
        from vtscore.datasets.loader import load_dataset_from_folder
        load_dataset_from_folder(download_dir, media_type, medias, thin=thin)


# This module-level instance is what the registry discovers.
IMPORTER = S3Importer()
```

### Choosing your override point

`DatasetImporter` offers four override points, from simplest to
fullest-control.  **Hooks 1–3 leave conversion and ingestion to the
framework** — you never call `get_converter()`, never invoke
`converter.convert()`, never assign media IDs, never set default
origins.  Only hook 4 takes that responsibility back.

```
Does your backend serve media one record at a time, with one media
type per query?
│
├─ Yes, and the importer only ever pulls one media type per import.
│  └─→ Hook 1: list_records() + fetch_record()
│       Simplest split — return opaque record handles, then convert
│       each to a media dict. Default fetch_source_media() delegates
│       here, so the spec list is invisible to you. Best for
│       single-source-type service importers (e.g. "pull all rows
│       from this table").
│
├─ Yes, and the importer can pull different media types per spec
│  (e.g. one query for images, another for videos).
│  └─→ Hook 2: fetch_source_media(spec, ...)
│       Framework loops effective_source_specs() and calls you once
│       per spec. You yield raw media dicts of spec.source_type;
│       framework runs spec.converter on each. Best for service-style
│       multi-media importers (e.g. ReCaller).
│
├─ No — one upstream call returns mixed source types in a single
│  response, and you want to make it only once.
│  └─→ Hook 3: fetch_all_source_media(specs, ...)
│       Framework calls you once with the full spec list. You make
│       the one upstream call and yield (spec, raw_media) pairs,
│       tagging each record with the spec it satisfies. Framework
│       still runs converters and ingests.
│
└─ Neither — the data is folder-shaped (already on disk, or staged
   there after download) and you want to delegate to the folder
   loader / converter runner.
   └─→ Hook 4: run()
        Full control. You own the medias dict, ID assignment,
        origin, and conversion. Typical body: stage files to a temp
        dir, call load_dataset_from_folder() for direct specs, call
        run_converters_on_folder() for converter specs. The four
        in-tree folder importers (server_folder, server_files,
        local_folder, local_files) all use this hook.
```

**Single-spec rule of thumb:** if your importer always pulls exactly
one media type per import (no per-source-type fan-out), use hook 1 and
ignore `SourceSpec` entirely.  The default `fetch_source_media()`
threads through to it without you knowing the spec exists.

**Multi-spec rule of thumb:** if the user can pick multiple source
types in one import, use hook 2 unless making N upstream calls is
actively wasteful — in which case use hook 3.  Both leave conversion
to the framework.

### DatasetImporter class reference

**Required to implement (pick one — see [Choosing your override point](#choosing-your-override-point)):**

| Member | Signature | Description |
|--------|-----------|-------------|
| `list_records()` + `fetch_record()` | see [Bulk-record hooks](#bulk-record-hooks) | **Hook 1.** Per-record / bulk-record split for **single-source-type** service importers. Default `fetch_source_media()` delegates to these |
| `fetch_source_media()` | `(spec: SourceSpec, field_values: dict, thin: bool = False) -> Iterator[dict]` | **Hook 2.** Per-spec fetch for **multi-source-type** service importers. Yield raw media dicts of `spec.source_type`; framework runs `spec.converter` on each. See [Multi-media imports](#multi-media-imports) |
| `fetch_all_source_media()` | `(specs: list[SourceSpec], field_values: dict, thin: bool = False) -> Iterator[tuple[SourceSpec, dict]]` | **Hook 3.** Bulk fetch for service-style importers whose backend returns mixed source types in one call. Yield `(spec, raw_media)` pairs; framework runs converters and ingests. Default delegates to `fetch_source_media()` per spec. See [Multi-media imports](#multi-media-imports) |
| `run()` | `(field_values: dict, medias: dict, thin: bool = False) -> None` | **Hook 4.** Full control — populate `medias` in-place yourself. Used by folder-shaped importers that delegate to `load_dataset_from_folder()` and `run_converters_on_folder()` |

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
| `effective_source_specs()` | `(field_values: dict) -> list[SourceSpec]` | Resolve the user's form values into a flat list of `(source_type, converter, params)` rows for multi-media imports. See [Multi-media imports](#multi-media-imports) |

**Class attributes:**

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `multi_media` | `bool` | `False` | When `True`, the importer participates in the new multi-media flow (output type + per-source-type converter rows). See [Multi-media imports](#multi-media-imports) |

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
API calls (not files on disk) can build media dicts directly in `run()`.

**Importers do not embed.** Set `embedding=None` and `embedder=""`; the
framework `embed_missing` stage embeds every item still at `None` after
`run()` returns, using the user-selected embedder (or the default for
the media type).  Only set `embedding` to a real vector when your data
source ships pre-computed vectors that are dimension-compatible with the
embedder the user picked — in that case also set `embedder` to the name
of that embedder.

```python
def run(self, field_values, medias, thin=False):
    for i, item in enumerate(api_results, start=1):
        medias[i] = {
            "id": i,
            "media_type": "audio",
            "filename": item["id"],
            "md5": item["md5"],                  # pre-computed by the service
            "embedding": None,                   # framework embed stage fills this in
            "embedder": "",                      # framework embed stage stamps this
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

If the source ships pre-computed vectors, prefer
`self.content_vectors[filename] = vec` or
`self.custom_metadata_map[filename] = {"embedding": vec}` — the framework
treats those as already-embedded and skips them.

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
            "media_type": "audio",
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
            {"media_type": "audio", "filename": r["id"], "embedding": e,
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
`vtscore/datasets/importers/recaller/__init__.py`, which overrides the
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
4. The `dataset` channel on `GET /api/events` (SSE) streams progress
   bar data.

### Progress reporting

```python
from vtscore.concurrency.progress import update_progress
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

Add any extra packages to `[project.dependencies]` in the repo's
`pyproject.toml` — that's the single source of truth and deptry verifies
every import is declared there. They are picked up the next time you run
`bash scripts/install-cpu.sh` (or any editable install).

### Multi-media imports

Importers that want to pull in **multiple source media types** (e.g.
"images, plus videos converted to images, plus documents converted to
images") set the class attribute `multi_media = True`.

A `SourceSpec` (defined in `vtscore.datasets.importers.base`) is:

```python
SourceSpec(
    source_type="video",            # type_id of the source media
    converter="video2image",        # converter name, or None to include directly
    params={"n_clips": "30"},       # user-supplied converter param values
)
```

`effective_source_specs()` reads the user's `source_specs` form value
(either a Python list or a JSON-encoded string), validates it against
the converter registry, and returns the typed list.  Each spec where
`converter is None` is a "include directly" row — the framework ingests
files of `source_type` straight into the dataset.  Each spec where
`converter` is set asks the framework to take files of `source_type`
from the importer and pass them through that converter (with
`spec.params`) to produce media of the dataset's output type.

**The framework drives the conversion, not your importer.**  Your job
is just to **yield raw source-type media**; the framework runs each
spec's converter and ingests the result.  Subclasses **never** call
`get_converter()` or `converter.convert()` directly.

You pick one of two fetch hooks depending on how your backend is
shaped.

#### Hook 2 — per-spec fetch (`fetch_source_media`)

When the backend serves one media type per query.  The framework
loops `effective_source_specs()` and calls you once per spec.

```python
class DXImporter(DatasetImporter):
    name = "dx"
    multi_media = True
    fields = [
        ImporterField(key="media_type", label="Output Media Type", field_type="select", ...),
        ImporterField(key="dataset_id", ..., required=True),
    ]

    def fetch_source_media(self, spec, field_values, thin=False):
        """Yield raw media dicts of spec.source_type — one per upstream record.

        Called once per spec. When spec.converter is set the framework
        runs converter.convert(raw, spec.params) on every yielded dict
        before storing it.
        """
        for record in self._dx_list(spec.source_type, field_values):
            yield self._dx_fetch(record, spec.source_type, field_values)
```

That's the entire integration — no `run()`, no converter calls, no
spec loop.

#### Hook 3 — bulk fetch (`fetch_all_source_media`)

When one upstream call returns mixed source types in a single
response and you want to make it only once.  The framework calls you
once with the full spec list; you yield `(spec, raw_media)` pairs.

```python
class DXImporter(DatasetImporter):
    name = "dx"
    multi_media = True
    fields = [...]

    def fetch_all_source_media(self, specs, field_values, thin=False):
        """One upstream call covers every spec; tag each record with
        the spec it satisfies. Framework still owns converter dispatch
        and ingestion.
        """
        wanted_types = {spec.source_type for spec in specs}
        records = self._dx_fetch_everything(field_values, types=wanted_types)

        # Bucket by type, then walk specs to preserve user-submitted order.
        by_type: dict[str, list[dict]] = {}
        for rec in records:
            by_type.setdefault(rec["media_type"], []).append(rec)

        for spec in specs:
            for rec in by_type.get(spec.source_type, []):
                yield spec, self._dx_to_media_dict(rec, spec.source_type)
```

The default `fetch_all_source_media()` just loops
`fetch_source_media()` per spec, so importers using hook 2 see no
behavioural change.

> **Heads-up:** hooks 2 and 3 only run when
> `effective_source_specs()` resolves to at least one spec.  That
> requires either a `media_type` field on a legacy importer
> (`multi_media = False`) or a `source_specs` value on a multi-media
> importer (`multi_media = True`, with `media_type` declaring the
> output type).  If your importer declares neither, `run()` falls
> through to the hook-1 path and raises `NotImplementedError` from
> `list_records()` — even when you've overridden `fetch_source_media`
> or `fetch_all_source_media`.

**Legacy / shim path.** Importers that have **not** flipped
`multi_media` still work as before: they declare a single `media_type`
field and (optionally) accept a comma-separated `converters` field that
post-processes the imported folder through
`run_converters_on_folder()`.  Legacy importers can also call
`effective_source_specs()` — it synthesises an equivalent list from the
classic `media_type` + `converters` fields, so a legacy importer can
migrate to the new iteration style before changing its form schema.

See [`docs/plans/multi-media-import.md`](plans/multi-media-import.md) for
the full design and migration checklist.

---

## Adding a Media Converter

A `MediaConverter` takes media of one type and produces one or more
media dicts of a *different* type.  Built-in examples:
`video2image`, `video2audio`, `document2image`, `document2text`,
`audio2image` (spectrogram), `audio2text` (Whisper ASR), `image2text` (OCR).
Converters are auto-discovered from `vtscore.converters` via the
`CONVERTER` sentinel.

### File structure

```
vtscore/converters/<your_converter>.py    # flat module — single file per converter
```

### What to implement

Subclass `MediaConverter` from `vtscore.converters.base`.  Implement
`source_type`, `target_type`, and `convert()`.  Optionally declare
user-configurable parameters as a list of `PluginField`s on the class.

```python
from vtscore.converters.base import MediaConverter
from vtscore.plugins import PluginField


class Image2TextMediaConverter(MediaConverter):
    display_name = "Image → Text (OCR)"
    description = "Run OCR on image files"
    fields = [
        PluginField(
            key="lang",
            label="OCR Language",
            field_type="text",
            default="eng",
            description="Tesseract language code (e.g. 'eng', 'spa').",
        ),
    ]

    @property
    def source_type(self) -> str:
        return "image"

    @property
    def target_type(self) -> str:
        return "text"

    def convert(self, media: dict, params: dict | None = None) -> list[dict]:
        lang = self.get_param(params, "lang")
        import pytesseract
        from PIL import Image

        img = Image.open(io.BytesIO(media["media_bytes"]))
        text = pytesseract.image_to_string(img, lang=lang)
        if not text.strip():
            return []
        return [{"filename": Path(media["filename"]).stem + ".txt", "media_string": text}]


CONVERTER = Image2TextMediaConverter()
```

### MediaConverter class reference

| Member | Description |
|--------|-------------|
| `source_type` (property) | The `type_id` of the input media type (e.g. `"image"`) |
| `target_type` (property) | The `type_id` of the output media type (e.g. `"text"`) |
| `convert(media, params=None)` | Convert a single source media dict; return a list of target dicts |
| `fields` | Class-level list of `PluginField`s for user-configurable params |
| `get_param(params, key)` | Helper: read a param value with field-default fallback |
| `name` (property) | Auto-derived as `f"{source_type}2{target_type}"` |
| `display_name` | Human-readable label shown in the picker |
| `description` | One-line description |

Each returned dict must include a `filename` and the target type's data
fields (`media_bytes` and `duration` for image/audio/video,
`media_string` for text).  The caller assigns IDs and embeds the
outputs.

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
vtscore/exporters/<your_exporter>/
└── __init__.py       # Exporter class + EXPORTER instance (required)
```

### What to implement

Subclass `LabelsetExporter` from `vtscore.exporters.base`.

```python
# vtscore/exporters/sftp/__init__.py

from vtscore.exporters.base import LabelsetExporter, ExporterField


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
| `/api/detectors/{name}` | GET    | Detector labelset + examples              | JSON            |

### Wiring up dependencies

Add any extra packages to `[project.dependencies]` in the repo's
`pyproject.toml` — that's the single source of truth and deptry verifies
every import is declared there. They are picked up the next time you run
`bash scripts/install-cpu.sh` (or any editable install).

---

## Adding a Label Importer

Label importers let users import pre-existing labels (good/bad votes) from
external sources. Auto-discovered at runtime.

### File structure

```
vtscore/labels/importers/<your_importer>/
└── __init__.py       # Importer class + LABEL_IMPORTER instance (required)
```

### What to implement

Subclass `LabelImporter` from `vtscore.labels.importers.base`. The
`run()` method must return a list of label dicts.

```python
# vtscore/labels/importers/postgres/__init__.py

from vtscore.labels.importers.base import LabelImporter, LabelImporterField


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
vtscore/labels/sources/<your_source>/
└── __init__.py       # Source class + LABELSET_SOURCE instance (required)
```

### What to implement

Subclass `LabelsetSource` from `vtscore.labels.sources.base`.

```python
# vtscore/labels/sources/database/__init__.py

from vtscore.labels.sources.base import LabelsetSource, PluginField
from vtscore.datasets.labelset import LabelSet


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
`_syncing` guard in `vtscore/labels/sync.py` suppresses this:

```python
with _sync_guard():
    # apply_label() calls during import won't trigger source.save()
    ...
```

The same pattern is used in `vtsearch/settings.py` for settings sources.

---
