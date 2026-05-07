# RCDatasetImporter Extension Plan

This document describes everything needed to complete the ReCaller / Holder /
PullWrest integration.  VTSearch core changes and plugin scaffolds are already
in place — the developer's job is to implement the API client stubs.

## Overview

Three external services are involved:

| Service | What it does | VTSearch talks to it via |
|---------|-------------|------------------------|
| **ReCaller (RC)** | Browse tool. Given a `queryID`, returns a collection of results: `{contentID, mediaID, media_type, media_url, md5, ...}`.  Also exposes a list of recent queryIDs filtered by `media_type`, used to populate the importer's dynamic `query_id` dropdown | `_rc_fetch_results()` and `_rc_list_queries()` in the RC importer |
| **DataWrest (DW)** | Given a `mediaID`, returns `{embedding, embedder}`. All media of a given type share the same embedder | `_dw_get_embedding()` in the RC importer |
| **PullWrest (PW)** | Given a `media_url`, returns the raw media bytes | `_pw_fetch_media()` in the RC importer + PW media source |
| **Holder** | ContentID storage. Create packages, add folders, write/read contentIDs with metadata | `_holder_*()` functions in the Holder exporter and importer |

Four plugins are scaffolded (all `hidden_from_picker = True` until ready):

| Plugin | File | Type | Sentinel |
|--------|------|------|----------|
| ReCaller Dataset Importer | `vtsearch/datasets/importers/recaller/__init__.py` | `DatasetImporter` | `IMPORTER` |
| Holder Labelset Exporter | `vtsearch/exporters/holder/__init__.py` | `LabelsetExporter` | `EXPORTER` |
| Holder Label Importer | `vtsearch/labels/importers/holder/__init__.py` | `LabelImporter` | `LABEL_IMPORTER` |
| PullWrest Media Source | `vtsearch/datasets/sources/pullwrest.py` | `MediaSource` factory | `SOURCE` |

## What's already done (VTSearch core)

1. **`LabeledElement.metadata`** — Optional `dict[str, Any]` on each label
   element that round-trips through `to_dict()` / `from_dict()`.  When
   building a LabelSet from votes, `custom_metadata` from the media
   automatically flows into `metadata`.

2. **`media_url` lazy-fetch** — `_resolve_media_bytes()` and
   `_resolve_media_string()` in `MediaType` fall back to fetching from
   `media["media_url"]` when `media_bytes` and `media_path` are both
   absent.

3. **Origin params in enriched export** — `GET /api/labels/export?enrich=true`
   flattens `origin.params` (e.g. `contentID`, `mediaID`) into
   `custom_metadata` and `available_columns`.  `custom_metadata` values
   override same-named `origin.params` if both are present.

4. **36 tests** in `tests/test_extension_scaffolds.py` covering metadata
   round-trip, media_url resolution, plugin discovery, helper functions,
   and enriched export.

---

## Step 1: Implement API clients

Create a shared client module (suggested: `vtsearch/ext/rc_clients.py` or
just inline in each plugin) with HTTP functions for each service.

### 1a. ReCaller client

```python
def _rc_list_queries(media_type: str) -> list[str]:
    """GET /api/rc/queries?media_type=<type> → list of recent queryIDs.

    Powers the importer's ``query_id`` dropdown (declared with
    ``dynamic_options=True, depends_on=["media_type"]``).  The frontend
    re-fetches whenever the user picks a different media type, so the
    user never has to copy-paste a queryID — they pick from a list.
    """


def _rc_fetch_results(query_id: str) -> list[dict[str, Any]]:
    """GET /api/rc/query/{query_id} → list of result dicts.

    Each result must have at least:
        contentID: str
        mediaID: str
        media_type: str   (e.g. "audio", "image")
        media_url: str    (PullWrest-resolvable URL)
        md5: str          (hex digest)
    """
```

Replace the `NotImplementedError` stubs in
`vtsearch/datasets/importers/recaller/__init__.py`.

### 1b. DataWrest client

```python
def _dw_get_embedding(media_id: str) -> dict[str, Any]:
    """GET /api/dw/embedding/{media_id} → {embedding: np.ndarray, embedder: str}.

    The embedder name MUST match a VTSearch-registered embedder:
    "clap" (audio), "clip" (image), "xclip" (video), "e5" (text),
    or any alternative embedder registered in the media type.
    """
```

Replace the stub in the same file.

### 1c. PullWrest client

```python
def _pw_fetch_media(media_url: str) -> bytes:
    """GET {media_url} → raw binary media content."""
```

This function appears in **two** places — DRY it up:
- `vtsearch/datasets/importers/recaller/__init__.py`
- `vtsearch/datasets/sources/pullwrest.py`

Consider extracting to a shared module that both import from.

### 1d. Holder client

```python
def _holder_create_package() -> str:
    """POST /api/holder/packages → new holderID string."""

def _holder_create_folder(holder_id: str, folder_name: str) -> None:
    """POST /api/holder/packages/{holder_id}/folders → create named folder."""

def _holder_write_entry(holder_id, folder_name, content_id, metadata=None):
    """POST /api/holder/packages/{holder_id}/folders/{folder_name}/entries"""

def _holder_read_folder(holder_id: str, folder_name: str) -> list[dict]:
    """GET /api/holder/packages/{holder_id}/folders/{folder_name}/entries
    → list of {contentID, mediaID, md5, media_url, media_type}"""
```

Replace stubs in:
- `vtsearch/exporters/holder/__init__.py` (create_package, create_folder,
  write_entry)
- `vtsearch/labels/importers/holder/__init__.py` (read_folder)

---

## Step 2: Add `requirements.txt` for each plugin

If the API clients use `requests` or `httpx`, create a `requirements.txt`
in each plugin directory:

```
# vtsearch/datasets/importers/recaller/requirements.txt
requests>=2.28
```

```
# vtsearch/exporters/holder/requirements.txt
requests>=2.28
```

```
# vtsearch/labels/importers/holder/requirements.txt
requests>=2.28
```

Then run `bash install-plugin-deps.sh` to regenerate
`requirements-plugins.txt`.

---

## Step 3: Test with real services

### 3a. Unit tests with mocked API clients

Write tests that mock `_rc_fetch_results`, `_dw_get_embedding`, etc. and
verify end-to-end flow:

```python
from unittest.mock import patch
import numpy as np

def test_rc_importer_builds_media_dicts():
    from vtsearch.datasets.importers.recaller import ReCallerDatasetImporter

    rc_results = [{
        "contentID": "C1", "mediaID": "M1",
        "media_type": "audio", "media_url": "http://pw/M1", "md5": "abc123",
    }]
    dw_result = {"embedding": np.zeros(512, dtype=np.float32), "embedder": "clap"}

    imp = ReCallerDatasetImporter()
    medias = {}
    with patch("vtsearch.datasets.importers.recaller._rc_fetch_results", return_value=rc_results), \
         patch("vtsearch.datasets.importers.recaller._dw_get_embedding", return_value=dw_result), \
         patch("vtsearch.datasets.importers.recaller._pw_fetch_media", return_value=b"\x00"):
        imp.run({"query_id": "Q1", "media_type": "audio"}, medias)

    assert len(medias) == 1
    assert medias[1]["origin"]["params"]["contentID"] == "C1"
    assert medias[1]["md5"] == "abc123"
    assert medias[1]["media_url"] == "http://pw/M1"
```

### 3b. Integration test with live services

Once API clients work, test the full round-trip:

1. Import dataset from RC query
2. Vote on some media (good/bad)
3. Export labels to Holder → capture `holder_id`
4. Import labels from Holder using the same `holder_id`
5. Verify votes match

---

## Step 4: Flip `hidden_from_picker`

When the plugins are ready for production:

1. Set `hidden_from_picker = False` in each plugin class
2. The plugins will immediately appear in the frontend UI

---

## Data flow diagrams

### Import flow

```
RC API  ──queryID──>  _rc_fetch_results()
                           │
                    filter by mediaType
                           │
                    ┌──────┴──────┐
                    │ for each    │
                    │ result:     │
                    ├─────────────┤
                    │ DW API ─────> _dw_get_embedding(mediaID) → {embedding, embedder}
                    │ PW API ─────> _pw_fetch_media(media_url)  → bytes (skip if thin)
                    │ RC result ──> md5 (pre-computed)
                    └─────────────┘
                           │
                    build media dict:
                      origin = {importer: "recaller", params: {contentID, mediaID, media_url, media_type}}
                      custom_metadata = {contentID, mediaID, media_url}
                      media_url = pw_url  (for lazy-fetch)
                           │
                    medias[i] = {...}
```

### Export flow (labels → Holder)

```
GET /api/labels/export?enrich=true
    │
    ├── builds LabelSet from votes
    ├── enriches with custom_metadata + origin.params
    │   (contentID, mediaID, media_url now in custom_metadata)
    │
POST /api/exporters/export  {exporter_name: "holder", results: labels}
    │
    ├── _holder_create_package() → holderID
    ├── _holder_create_folder(holderID, "Good")
    ├── _holder_create_folder(holderID, "Bad")
    │
    └── for each label with contentID:
        _holder_write_entry(holderID, folder, contentID, {mediaID, md5, media_url, media_type})
    │
    └── return {holder_id: "H123", exported: N}
```

### Import flow (Holder → labels)

```
POST /api/label-importers/import/holder  {holder_id: "H123"}
    │
    ├── _holder_read_folder("H123", "Good") → [{contentID, mediaID, md5, ...}, ...]
    ├── _holder_read_folder("H123", "Bad")  → [...]
    │
    └── for each entry, build label dict:
        {
            md5: entry.md5,
            label: "good"/"bad",
            origin: {importer: "recaller", params: {contentID, mediaID, media_url, media_type}},
            origin_name: contentID,
            metadata: {contentID, mediaID, md5, media_url, media_type},
        }
    │
    └── VTSearch matches to existing media by origin+origin_name or md5
```

---

## Key design decisions (already baked in)

### Per-media origin (not queryID)

The RC importer stores each media's `contentID` in its origin, NOT the
ephemeral `queryID`.  The `build_origin()` override returns an empty
origin, and `run()` sets per-media origins directly.  This means:

- Labels can be exported/imported without knowing the original query
- Origin matching works across datasets (different queries, same media)
- The queryID is only used during the import call, never persisted

### Thin mode skips PullWrest

When `thin=True`, the importer skips `_pw_fetch_media()` entirely.
`media_bytes=None`, `media_path=None`, `media_url="https://..."`.
Embeddings come from DataWrest, MD5 from ReCaller.  Sorting and scoring
work without downloading.  Bytes are fetched lazily by
`_resolve_media_bytes()` only when the UI needs to display/play media.

### Holder exporter uses LabelsetExporter

No new plugin type needed.  The Holder exporter is a standard
`LabelsetExporter` whose `export()` returns `{"holder_id": "..."}` in
the response dict alongside the standard `"message"` key.

### ContentID lookup order in exporter

The Holder exporter looks for `contentID` in three places (first wins):

1. `entry["metadata"]["contentID"]` — from a Holder import round-trip
2. `entry["custom_metadata"]["contentID"]` — from enriched RC media
3. `entry["origin"]["params"]["contentID"]` — from RC origin

### Origin reconstruction in importer

The Holder label importer reconstructs the **exact same** origin dict
format that the RC importer creates:

```python
{"importer": "recaller", "params": {"contentID": "...", "mediaID": "...", "media_url": "...", "media_type": "..."}}
```

This enables origin-based matching when importing into an RC-loaded dataset.

---

## Checklist

- [ ] Implement `_rc_fetch_results()` in `recaller/__init__.py`
- [ ] Implement `_dw_get_embedding()` in `recaller/__init__.py`
- [ ] Implement `_pw_fetch_media()` — shared between `recaller/__init__.py`
      and `sources/pullwrest.py`
- [ ] Implement `_holder_create_package()` in `exporters/holder/__init__.py`
- [ ] Implement `_holder_create_folder()` in `exporters/holder/__init__.py`
- [ ] Implement `_holder_write_entry()` in `exporters/holder/__init__.py`
- [ ] Implement `_holder_read_folder()` in `labels/importers/holder/__init__.py`
- [ ] Add `requirements.txt` to each plugin directory
- [ ] Run `bash install-plugin-deps.sh`
- [ ] Write unit tests with mocked API clients
- [ ] Test with live services (import → vote → export → re-import)
- [ ] Set `hidden_from_picker = False` on all four plugins
- [ ] Run `./run-tests.sh` — all tests must pass
