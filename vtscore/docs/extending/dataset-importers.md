# Writing a `DatasetImporter`

A dataset importer pulls media into a VTSearch dataset from a source —
a local folder, an HTTP archive, an S3 bucket, a database, a remote
embedding service, anything. The library auto-discovers importers
under `vtscore.datasets.importers` (sentinel `IMPORTER`) and also walks
the `vtscore.importers` entry-point group, so a third-party
distribution can ship one by `pip install`-ing a package with the
right `pyproject.toml` block. Subclass
[`DatasetImporter`](../../datasets/importers/base.py) ([`vtscore/datasets/importers/base.py:210`](../../datasets/importers/base.py)),
declare your `fields`, and either implement `run()` (full control) or
the `list_records()` / `fetch_record()` hooks (default `run()` does
the rest).

**App-side counterpart:** [`docs/EXTENDING-plugins.md § Adding a Data
Importer`](../../../docs/EXTENDING-plugins.md#adding-a-data-importer)
covers the form / picker / multi-media UI wiring. This guide focuses
on the library API and third-party packaging.

## Contents

- [The minimum contract](#the-minimum-contract)
- [Two ways to implement: `run()` vs. per-record hooks](#two-ways-to-implement-run-vs-per-record-hooks)
- [Building media dicts](#building-media-dicts)
- [Pre-computed embeddings, MD5s, and metadata](#pre-computed-embeddings-md5s-and-metadata)
- [Multi-media imports](#multi-media-imports)
- [Reporting progress](#reporting-progress)
- [Entry-point registration](#entry-point-registration)
- [Testing pattern](#testing-pattern)
- [Worked example](#worked-example)

## The minimum contract

A `DatasetImporter` subclass must set four class attributes and
implement one of two flow shapes:

| Attribute | Purpose |
|-----------|---------|
| `name: str` | Snake-case identifier — registry key, CLI subcommand, API path segment |
| `display_name: str` | Human-readable label |
| `description: str` | One-sentence subtitle |
| `fields: list[PluginField]` | User-configurable inputs (rendered into a form / CLI flags / validation schema) |

Then either:

- Override `run(field_values, medias, thin=False)` to populate `medias`
  in place — full control of the import flow, used by every
  folder-style importer; or
- Implement `list_records(field_values) → list` plus
  `fetch_record(record, field_values, thin) → dict | None`, optionally
  overriding `_fetch_records_bulk_impl()` for batched / concurrent
  fetches — used by service-style importers (see
  [`vtscore/datasets/importers/recaller/__init__.py`](../../datasets/importers/recaller/__init__.py)
  for a working example that issues concurrent thread-pool fetches).

Expose a module-level `IMPORTER = YourImporter()` so the registry picks
it up. The sentinel must be an already-instantiated plugin object, not
the class.

## Two ways to implement: `run()` vs. per-record hooks

Both paths end up with the same shape: `medias[id] = {...}`. The split
mirrors `MediaEmbedder.embed_media` / `embed_media_bulk`.

**Use `run()` when** the import is fundamentally a single pass over a
local resource — typically files in a folder. You control the entire
flow, including how IDs are assigned.

**Use the per-record hooks when** records come from a remote service.
`list_records()` returns whatever opaque shape you want;
`fetch_record(record, field_values, thin)` turns one record into one
media dict (returning `None` to skip), and the default `run()` assigns
sequential integer IDs starting at 1, fills in `origin` from
`build_origin(field_values)` where the record didn't set its own, and
populates `medias`.

When the per-record API supports batching or you want to issue
concurrent I/O, override `_fetch_records_bulk_impl(records,
field_values, thin)` and return a same-length list. The framework
calls `fetch_records_bulk()` once with every record, so a single
bulk-request or `ThreadPoolExecutor.map` covers the entire dataset.

## Building media dicts

Each entry in `medias` is a dict with these keys (most are required;
see [`vtscore/datasets/importers/base.py:340`](../../datasets/importers/base.py)
for the importer-side instance attributes that feed into the loader):

| Key | Type | Required | Notes |
|-----|------|----------|-------|
| `id` | `int` | Yes (set by framework when using the hook path) | 1-based |
| `type` | `str` | Yes | The media type's `type_id` (`"audio"`, `"image"`, …) |
| `filename` | `str` | Yes | Display name and origin-resolution key |
| `md5` | `str` | Recommended | Hex digest; computed by loader if absent |
| `embedding` | `np.ndarray` | Recommended | Skips the embedding model when present |
| `embedder` | `str` | Yes when `embedding` is set | Name of the embedder that produced the vector |
| `media_bytes` | `bytes \| None` | Conditional | The actual content; `None` in thin mode |
| `media_path` | `str \| None` | Conditional | Local path; required for thin mode if no `media_url` |
| `media_url` | `str \| None` | Optional | Remote URL for lazy-fetch (PullWrest-style) |
| `media_string` | `str \| None` | Conditional | Text content (text-type media only) |
| `duration` | `float` | Yes | Seconds; `0` for non-temporal media |
| `file_size` | `int` | Yes | Bytes |
| `category` | `str` | Yes | Free-form; usually `""` |
| `origin` | `dict \| None` | Recommended | Per-media provenance (see below) |
| `origin_name` | `str` | Recommended | Stable identifier within the origin |
| `custom_metadata` | `dict` | Optional | Per-media display fields surfaced in the labeling UI |

Folder-style importers usually delegate everything after the download
to `vtscore.datasets.loader.load_dataset_from_folder` — it walks the
folder, embeds files (skipping any whose name appears in
`self.content_vectors`), and assigns IDs. Service-style importers
build the dicts directly (see the recaller importer for a real
example).

### Origins

Every media should carry an `origin` dict that identifies the importer
and parameters needed to refetch the same content. The framework
automatically calls `build_origin(field_values)` after `run()` and
applies the result to every media whose `origin` is `None`, so the
common case requires no work. Override `build_origin()` only when the
default (importer name + stringified field values) is too coarse —
e.g. when the importer fans out across multiple sources within a
single dataset.

`origin` flows through the registry, through dataset pickle export /
import, and is what `vtscore.datasets.sources.get_source_for_origin`
keys on to resolve a media file later. If your importer should be
reloadable (i.e. someone with the labelset can recreate the dataset),
keep `origin.params` round-trippable through your `fields`.

## Pre-computed embeddings, MD5s, and metadata

Importers that already have vectors or hashes shouldn't waste compute
re-deriving them. The base class provides three instance attributes to
populate during `run()`:

| Attribute | Purpose |
|-----------|---------|
| `content_vectors: dict[str, np.ndarray]` | filename → embedding; loader skips the embedding model |
| `content_md5s: dict[str, str]` | filename → hex MD5; loader skips its own hashing |
| `custom_metadata_map: dict[str, dict[str, Any]]` | filename → per-file metadata dict; entries with an `"md5"` or `"embedding"` key override the above |

Lookup tries the relative path first (for files in subdirectories),
then the basename. When both keys exist for the same file with
different values, the loader logs a warning and keeps the
relative-path entry. When a bare basename would match multiple files
in the folder (e.g. `class_a/foo.wav` and `class_b/foo.wav` with a
single `"foo.wav"` key) without an explicit relative-path entry for
every match, the loader raises `ValueError` — supply the full
relative path for each file to disambiguate. Don't persist vectors
or MLP weights to disk on your own — the library re-derives them
from origins.

## Multi-media imports

Every importer can pull in multiple source media types in one shot —
e.g. images, plus videos converted to images, plus documents
converted to images.  The user submits a list of `source_specs` in
the dataset modal; the framework iterates them and dispatches
converters.  Each `SourceSpec` ([`vtscore/datasets/importers/base.py:67`](../../datasets/importers/base.py))
is `(source_type, converter, params)`:

- `converter is None` means "include directly" — fetch files of
  `source_type` straight into the dataset.
- `converter is set` means "fetch files of `source_type`, then pass
  them through the named converter with `params`".

**The framework owns conversion and ingestion.** Subclasses never
call `get_converter()` or `converter.convert()` themselves — they
just yield raw source-type media, and the framework runs each spec's
converter on it before assigning IDs and storing the result.

### Choosing your override point

`DatasetImporter` exposes four override points. Pick the simplest
one that fits your backend:

| When your backend looks like… | Override |
|------------------------------|----------|
| One query, one media type per import (no per-source-type fan-out) | `list_records()` + `fetch_record()` |
| Different query per media type — framework loops specs for you | `fetch_source_media(spec, ...)` |
| One query that returns mixed types in one response | `fetch_all_source_media(specs, ...)` |
| Folder-shaped (already on disk; delegates to `load_dataset_from_folder()` / `run_converters_on_folder()`) | `run()` directly |

The first three hooks all hand back raw source-type media; the
framework runs converters and ingests.  Only `run()` gives up that
help in exchange for full control.  The built-in `server_folder`
importer is the canonical `run()`-shaped example; `recaller` is the
canonical `fetch_source_media()`-shaped example.

The default `fetch_source_media()` delegates to `list_records()` +
`fetch_record()`, and the default `fetch_all_source_media()`
delegates to `fetch_source_media()` per spec — so each hook is a
strict generalisation of the one above it.  You can also call
`self.effective_source_specs(field_values)` directly from any of
these (or from a custom `run()`) to inspect the resolved spec list.

> **Heads-up:** `fetch_source_media()` and `fetch_all_source_media()`
> only run when `effective_source_specs()` resolves to at least one
> spec.  An importer that overrides one of those hooks but does
> **not** declare a `media_type` field (declaring the output type)
> falls through to the `list_records()` path and raises
> `NotImplementedError` at runtime.

## Reporting progress

```python
from vtscore.concurrency.progress import update_progress

update_progress("downloading", "Downloading file 3/10", 3, 10)
update_progress("embedding", "Embedding 7/10", 7, 10)
```

Status strings are conventional, not enforced: `"downloading"`,
`"embedding"`, `"importing"`. The dataset-loading pipeline keys on the
first `"embedding"` status to swap between the download
concurrency-gate and the embed concurrency-gate ([`vtscore/datasets/load_pipeline.py`](../../datasets/load_pipeline.py)),
so service-style importers that interleave both phases should emit the
right status string at each step.

## Entry-point registration

In-tree, drop the importer in
`vtscore/datasets/importers/<your_name>/__init__.py` and expose
`IMPORTER = YourImporter()`. Out-of-tree, ship a Python package whose
`pyproject.toml` declares the entry point:

```toml
[project]
name = "vtsearch-myimporter"
version = "0.1.0"
dependencies = ["vtsearch"]  # or vtscore once it's published standalone

[project.entry-points."vtscore.importers"]
my_importer = "my_pkg.importer:IMPORTER"
```

The value `my_pkg.importer:IMPORTER` must resolve to an
already-instantiated `DatasetImporter` — typically you point straight
at the module's `IMPORTER` sentinel. After `pip install`, the importer
appears in `list_importers()`, the `/api/dataset/all-importers`
endpoint, and `python app.py --list-plugins`.

Built-in plugins win on name clashes; a broken third-party entry point
emits a warning and is skipped (it can't block other plugins).

## Testing pattern

Library-tier importer tests live in `tests_lib/io/` or
`tests_lib/datasets/` and don't touch Flask. Use the autouse fixtures
in [`tests_lib/conftest.py`](../../../tests_lib/conftest.py) (which
reset contexts and stub all embedders) and assert the importer is
discoverable, validates its fields correctly, and populates `medias`
with the right shape.

```python
# tests_lib/io/test_my_importer.py
from vtscore.datasets.importers import get_importer, list_importers
from my_pkg.importer import MyImporter


class TestMyImporterRegistration:
    def test_is_discovered(self):
        # Works whether you installed via entry point or dropped into
        # vtscore/datasets/importers/<name>/.
        names = [imp.name for imp in list_importers()]
        assert "my_importer" in names

    def test_metadata(self):
        imp = get_importer("my_importer")
        assert isinstance(imp, MyImporter)
        assert imp.display_name
        assert imp.fields


class TestMyImporterRun:
    def test_populates_medias(self, tmp_path):
        imp = MyImporter()
        medias: dict[int, dict] = {}
        imp.run({"path": str(tmp_path), "media_type": "audio"}, medias)
        assert medias
        for m in medias.values():
            assert m["media_type"] == "audio"
            assert m["origin"]["importer"] == "my_importer"
```

See [`tests_lib/datasets/test_synthetic_importer.py`](../../../tests_lib/datasets/test_synthetic_importer.py)
for a complete reference covering registration, field validation, and
run behaviour without any app dependency.

## Worked example

A minimal third-party importer that pulls records from a hypothetical
JSON-line catalogue API. Each line of the catalogue has an `id`, a
public `url`, and a pre-computed embedding — perfect for the
service-style hook path because no per-file download or embedding is
needed.

```python
# my_pkg/catalogue_importer.py
from __future__ import annotations

import json
import urllib.request
from typing import Any

import numpy as np

from vtscore.datasets.importers.base import DatasetImporter, ImporterField
from vtscore.security.url_validation import validate_url


class CatalogueImporter(DatasetImporter):
    name = "catalogue"
    display_name = "Catalogue API"
    description = "Import audio clips from a remote JSON-line catalogue."
    icon = "\U0001f5c3"  # card-index
    fields = [
        ImporterField(
            key="catalogue_url",
            label="Catalogue URL",
            field_type="url",
            description="HTTPS URL of the JSON-line catalogue.",
            required=True,
        ),
    ]

    def list_records(self, field_values: dict[str, Any]) -> list[dict]:
        url = validate_url(field_values["catalogue_url"])
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
            return [json.loads(line) for line in resp if line.strip()]

    def fetch_record(
        self,
        record: dict,
        field_values: dict[str, Any],
        thin: bool = False,
    ) -> dict | None:
        embedding = np.asarray(record["embedding"], dtype=np.float32)
        return {
            "media_type": "audio",
            "filename": record["id"] + ".wav",
            "md5": record["md5"],
            "embedding": embedding,
            "embedder": "clap",
            "media_bytes": None,
            "media_path": None,
            "media_url": record["url"],
            "duration": float(record.get("duration", 0)),
            "file_size": int(record.get("size", 0)),
            "category": "",
            "custom_metadata": {"Catalogue ID": record["id"]},
        }


IMPORTER = CatalogueImporter()
```

And the `pyproject.toml` snippet:

```toml
[project.entry-points."vtscore.importers"]
catalogue = "my_pkg.catalogue_importer:IMPORTER"
```

After `pip install`, the importer is discoverable via
`vtscore.datasets.importers.get_importer("catalogue")`, runnable from
the CLI as `python app.py --autodetect --importer catalogue
--catalogue-url https://… --settings settings.json`, and the dataset
loads with one media per catalogue line — no downloads, no embedding,
just the pre-computed vectors. The actual bytes are fetched lazily
when the UI needs to play a clip.
