# MediaSource Abstraction — Implementation Plan

> **Status: Proposal — not yet implemented.** This document describes a
> planned design that has not been built. The current codebase uses
> `resolve_file_from_origin()` in `vtsearch/models/resolver.py` for
> individual file resolution. See [ARCHITECTURE.md](ARCHITECTURE.md) for
> the current architecture.

## Problem

Users want to supply **individual media examples** for sorting during labeling (not just text descriptions). Today, the only way to get media into VTSearch is through dataset importers, which are designed for bulk ingestion — there's no standard way to say "give me just `audio123.wav` from that folder" or "fetch this one file from that S3 bucket."

Related need: when importing labels from another dataset (e.g. a LabelSet for a trainable detector), we need to resolve individual media files from their origins to get embeddings for training. The current `ingest_missing_medias()` approach re-runs the **entire importer** just to cherry-pick a few files.

## Design

### Key Insight

`MediaSource` is an **optional composition layer** that sits *below* `DatasetImporter`, not above it. Importers that access file-like sources (folder, http_archive, future S3/SFTP) compose a `MediaSource`. Importers that don't deal with file-like sources (pickle, combine_datasets) continue working as-is.

### Architecture

```
                     ┌──────────────────────┐
                     │   DatasetImporter     │  (unchanged ABC)
                     │  run(), resolve_file  │
                     └─────────┬────────────┘
                               │ composes (optional)
                     ┌─────────▼────────────┐
                     │     MediaSource       │  (NEW ABC)
                     │  list_items()         │
                     │  fetch_item()         │
                     │  resolve_path()       │
                     └─────────┬────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
     ┌────────▼──────┐ ┌──────▼────────┐ ┌─────▼──────────┐
     │ LocalFolder   │ │ HttpArchive   │ │ (future: S3,   │
     │   Source      │ │   Source      │ │  SFTP, etc.)   │
     └───────────────┘ └───────────────┘ └────────────────┘
```

### Core Types

**`MediaItem`** (dataclass) — represents a single discoverable media file within a source:

```python
@dataclass
class MediaItem:
    key: str            # unique identifier within the source (relative path)
    filename: str       # basename, e.g. "audio123.wav"
    source_name: str    # source type, e.g. "local_folder", "http_archive"
```

**`MediaSource`** (ABC) — how to access media from a location:

```python
class MediaSource(ABC):
    name: str           # "local_folder", "http_archive"

    @abstractmethod
    def list_items(self, extensions: list[str] | None = None) -> Iterator[MediaItem]:
        """Yield all media items in this source, optionally filtered by extension."""

    @abstractmethod
    def fetch_item(self, key: str) -> Path | None:
        """Return a local file path for the item identified by *key*.
        May download/extract on demand. Returns None if not found."""

    @abstractmethod
    def resolve_path(self, origin_name: str = "", filename: str = "") -> Path | None:
        """Resolve a media file by origin_name or filename. Used by resolver.py."""

    def cleanup(self) -> None:
        """Release any temporary resources (extracted archives, etc.)."""
```

## Implementation Steps

### Step 1: Define MediaSource ABC + MediaItem

**File:** `vtsearch/datasets/sources/__init__.py`
**File:** `vtsearch/datasets/sources/base.py`

- Define `MediaItem` dataclass with `key`, `filename`, `source_name` fields
- Define `MediaSource` ABC with `list_items()`, `fetch_item()`, `resolve_path()`, `cleanup()`
- Sources are **instantiated per-use**, not singletons — each call creates a source bound to specific params (a folder path, a URL, etc.)

### Step 2: Implement LocalFolderSource

**File:** `vtsearch/datasets/sources/local_folder.py`

- Wraps the folder-scanning logic currently in `loader.py` and `FolderDatasetImporter.resolve_file()`
- Constructor takes `folder_path: Path`
- `list_items(extensions)` — `rglob` for matching files, yields `MediaItem` with `key = relative_path`
- `fetch_item(key)` — returns `folder_path / key` if it exists
- `resolve_path(origin_name, filename)` — tries `folder_path / origin_name`, then `folder_path / filename`

### Step 3: Implement HttpArchiveSource

**File:** `vtsearch/datasets/sources/http_archive.py`

- Wraps the download+extract logic from `HttpArchiveDatasetImporter`
- Constructor takes `url: str`; lazily downloads and extracts on first access
- Internally creates a `LocalFolderSource` over the extraction directory
- `list_items()` / `fetch_item()` / `resolve_path()` all delegate to the inner `LocalFolderSource`
- `cleanup()` — removes the extraction directory
- Reuses `_extract_archive()` from the http_zip importer (moved or re-exported to a shared location)
- Reuses `validate_url()` for SSRF protection

### Step 4: Add source registry — `get_source_for_origin()`

**File:** `vtsearch/datasets/sources/__init__.py`

```python
def get_source_for_origin(origin: dict) -> MediaSource | None:
    """Look up and instantiate the appropriate MediaSource for an origin dict."""
```

Mapping:
- `origin["importer"] == "folder"` → `LocalFolderSource(Path(origin["params"]["path"]))`
- `origin["importer"] == "http_archive"` → `HttpArchiveSource(origin["params"]["url"])`
- Anything else → `None` (importer doesn't use the source abstraction)

This is a **factory function**, not a registry singleton — it creates a fresh source each time, because sources are stateful (extraction dirs, etc.).

### Step 5: Refactor folder importer to compose LocalFolderSource

**File:** `vtsearch/datasets/importers/folder/__init__.py`

- `resolve_file()` → delegates to `LocalFolderSource(folder_path).resolve_path(origin_name, filename)`
- The `run()` method and `load_dataset_from_folder` integration stay as-is for now — `LocalFolderSource.list_items()` could eventually replace the file-scanning loop in `loader.py`, but that's a bigger refactor for later.

### Step 6: Refactor http_archive importer to compose HttpArchiveSource

**File:** `vtsearch/datasets/importers/http_zip/__init__.py`

- `resolve_file()` → delegates to `HttpArchiveSource(url).resolve_path(origin_name, filename)`
- Extract `_extract_archive()` to a shared location (or import from sources module)
- The `run()` method stays as-is for now — the source is composed for resolution, not yet for bulk loading

### Step 7: Refactor ingest_missing_medias to use source.fetch_item()

**File:** `vtsearch/datasets/ingest.py`

Currently, `ingest_missing_medias` runs the **entire importer** (`importer.run_cli(params, temp_medias)`) to get all media, then cherry-picks matching ones. With `MediaSource`, we can:

1. Call `get_source_for_origin(origin_dict)` to get a source
2. If a source is available, use `source.fetch_item(origin_name)` for each missing entry — fetching only what we need
3. Embed each fetched file individually
4. Fall back to the current full-import approach if `get_source_for_origin()` returns `None`

This is a **major efficiency win** — instead of re-ingesting thousands of files from a folder to get 3, we just grab those 3.

### Step 8: Refactor resolve_file_from_origin to use source.resolve_path()

**File:** `vtsearch/models/resolver.py`

- Before the current registry-based dispatch, try `get_source_for_origin(origin)`
- If a source is returned, call `source.resolve_path(origin_name, filename)`
- Fall back to current importer-based resolution if no source available
- This keeps synthetic origins (dupe_set, converter) handled inline

### Step 9: Wire up example-sort API to accept origin+key

**File:** `vtsearch/routes/sorting.py`

Add a new endpoint or extend the existing one:

**POST `/api/example-sort-origin`**
```json
{
    "origin": {"importer": "folder", "params": {"path": "/data/sounds"}},
    "key": "subdir/audio123.wav"
}
```

Implementation:
1. Call `get_source_for_origin(origin)` to get a source
2. Call `source.fetch_item(key)` to get the file path
3. Call `_example_sort_from_path(file_path)` (existing helper)
4. Clean up source if needed

This enables the UI to say "sort by similarity to this specific item from that dataset" without uploading or having the file on `SERVER_MEDIA_DIR`.

### Step 10: Tests

**File:** `tests/test_media_sources.py`

Test coverage:
- `MediaItem` construction and field access
- `LocalFolderSource`:
  - `list_items()` with/without extension filtering
  - `fetch_item()` success and missing-key cases
  - `resolve_path()` by origin_name and filename
- `HttpArchiveSource`:
  - Construction with URL, lazy download behavior (mocked)
  - Delegation to inner `LocalFolderSource`
  - `cleanup()` removes temp directories
- `get_source_for_origin()`:
  - Returns `LocalFolderSource` for folder origins
  - Returns `HttpArchiveSource` for http_archive origins
  - Returns `None` for pickle/combine_datasets origins
- Refactored `resolve_file_from_origin` still works via source path
- Refactored `ingest_missing_medias` fetches individual items when source available
- New `/api/example-sort-origin` endpoint works end-to-end

## What Does NOT Change

- `DatasetImporter` base class — no API changes
- Pickle importer and combine_datasets importer — no source composition
- `load_dataset_from_folder` / `load_dataset_from_folder_chunked` — not refactored yet (future work to compose `LocalFolderSource` into the scanning loop)
- Existing test files — all existing tests continue to pass
- Origin format — `{"importer": "...", "params": {...}}` is unchanged

## Future Extensions

Once this foundation is in place, adding new sources is straightforward:

- **S3Source** — constructor takes bucket + prefix; `fetch_item()` downloads to a local cache
- **SFTPSource** — constructor takes host + path; `fetch_item()` downloads on demand
- **SingleFileSource** — wraps a single file path/URL for the "sort by this one example" use case
- Bulk loading via sources — eventually `load_dataset_from_folder` could iterate `source.list_items()` instead of doing its own `rglob`
