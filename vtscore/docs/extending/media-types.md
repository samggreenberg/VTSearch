# Adding a `MediaType`

A media type defines how `vtscore` handles a kind of content: which
file extensions to scan for during folder imports, how to serve a clip
over HTTP, which fields to surface in the labeling UI, and which
demo datasets ship with it. Media types live as sub-packages under
`vtscore.media.<type>/` and are auto-discovered by a sentinel scan at
import time - drop a directory in, expose a `MEDIA_TYPE` sentinel from
its `__init__.py`, and the rest of the system picks it up. Embedders
and clippers for the type live alongside in the same sub-package.

**App-side counterpart:** [`docs/EXTENDING-media.md § Adding a Media
Type`](../../../docs/EXTENDING-media.md#adding-a-media-type) covers
the same machinery from the app side; this guide focuses on the
library API and the per-type sub-package convention.

## Contents

- [Why add a media type vs. a converter](#why-add-a-media-type-vs-a-converter)
- [Sub-package layout](#sub-package-layout)
- [The `MediaType` contract](#the-mediatype-contract)
- [What happens automatically](#what-happens-automatically)
- [Worked example](#worked-example)
- [Testing pattern](#testing-pattern)

## Why add a media type vs. a converter

Add a media type when the content is genuinely new - point clouds, 3D
meshes, source code, MIDI - and you need its own embedder, its own
HTTP serving, and its own UI viewer. Add a [converter](converters.md)
instead when the content is just a different surface over an existing
type: OCR text from an image, a spectrogram view of audio, a thumbnail
of a PDF page. Converters compose with existing embedders; new media
types stand alone.

A new media type is a contract on five things: identity (`type_id`,
`name`, `icon`), file import (`file_extensions`,
`folder_import_name`), HTTP serving (`media_response`), viewer
behaviour (`loops`), and content extraction
(`load_media_data`). Everything else has a useful default.

## Sub-package layout

```
vtscore/media/<your_type>/
├── __init__.py            # exposes MEDIA_TYPE and CLIPPERS sentinels
├── media_type.py          # MediaType subclass
├── clipper.py             # MediaClipper subclasses (the "default" + any tiling variants)
├── embedder_<name>.py     # one file per embedder, each exposing EMBEDDER
└── embedder_<other>.py    # additional embedders
```

The discovery scan ([`vtscore/media/__init__.py:260`](../../media/__init__.py))
walks every sub-package of `vtscore.media`, imports the package, and:

- registers `MEDIA_TYPE` (a single `MediaType` instance) via
  `register()`;
- registers every clipper in the `CLIPPERS` list via
  `register_clipper()`;
- scans the sub-package for any `embedder*.py` file or `embedder*/`
  sub-package, imports it, and registers its `EMBEDDER` sentinel via
  `register_embedder()`.

Symlinked files and symlinked directories are loaded via
`importlib.util.spec_from_file_location`, so a custom embedder can
live outside the source tree and be wired in by symlinking one file.

## The `MediaType` contract

`vtscore.media.base.MediaType` ([`vtscore/media/base.py`](../../media/base.py))
is an ABC. Required overrides:

| Member | Type | Purpose |
|--------|------|---------|
| `type_id` (property) | `str` | Canonical identifier - `"audio"`, `"image"`, `"text"`, `"mesh3d"` |
| `name` (property) | `str` | Human-readable label |
| `icon` (property) | `str` | SVG icon-type name resolved by the frontend |
| `file_extensions` (property) | `list[str]` | Glob patterns like `["*.obj", "*.stl"]` |
| `loops` (property) | `bool` | True for content the viewer should auto-loop (audio, video) |
| `demo_datasets` (property) | `list[DemoDataset]` | Demo entries surfaced in `/api/dataset/demo-list` |
| `load_media_data(file_path)` | `(Path) -> dict` | Type-specific fields to merge into the media dict; must include `"duration"` |
| `media_response(media)` | `(dict) -> MediaResponse` | Framework-agnostic HTTP response payload |

Optional overrides:

| Member | Default | Purpose |
|--------|---------|---------|
| `folder_import_name` | `type_id` | Alias used by folder-style importers |
| `dir_key` | `type_id + "_dir"` | Key in pickle files for external directories |
| `legacy_bytes_keys` | `[]` | Legacy keys to honour when unpickling old datasets |
| `pickle_extra_fields` | `[]` | Custom clip-dict keys to preserve through pickle round-trip |
| `display_metadata(media)` | base fields | Extra fields surfaced in the labeling UI |

`MediaResponse` is a small dataclass (`data`, `mimetype`,
`download_name`) that decouples the library from Flask
([`vtscore/media/base.py:68`](../../media/base.py)). The app-tier
route layer converts it into a `flask.Response`; library callers can
use the same value directly.

For text-type media, use the `_resolve_media_string(media)` helper to
honour `media_bytes` / `media_path` / `media_url` in priority order;
for binary media, use `_resolve_media_bytes(media)`. Both are inherited
from `MediaType`.

## What happens automatically

After registration:

| Subsystem | Behaviour |
|-----------|-----------|
| Folder import | Files matching `file_extensions` are scanned and embedded by the default embedder for `type_id` |
| Generic media route | `GET /api/medias/<id>/media` calls your `media_response()` |
| Demo listing | Entries in `demo_datasets` appear in `/api/dataset/demo-list` |
| Pickle round-trip | Standard fields + anything in `pickle_extra_fields` survive export/import |
| Inventory | The type appears under the `media_types` family in `gather_plugins()` |

## Worked example

A minimal `mesh3d` type that handles `.obj` and `.stl` files. No
embedder yet - that lives in its own [embedder
guide](embedders.md) - just enough to load files into a dataset and
serve them over HTTP.

```python
# vtscore/media/mesh3d/media_type.py
from __future__ import annotations

from pathlib import Path

from vtscore.media.base import MediaResponse, MediaType


class Mesh3DMediaType(MediaType):
    """3D meshes (.obj, .stl)."""

    @property
    def type_id(self) -> str:
        return "mesh3d"

    @property
    def name(self) -> str:
        return "3D Mesh"

    @property
    def icon(self) -> str:
        return "cube"

    @property
    def file_extensions(self) -> list[str]:
        return ["*.obj", "*.stl"]

    @property
    def loops(self) -> bool:
        return False

    @property
    def demo_datasets(self) -> list:
        return []

    @property
    def pickle_extra_fields(self) -> list[str]:
        # Preserve our custom vertex / face counts across pickle round-trip.
        return ["vertex_count", "face_count"]

    def load_media_data(self, file_path: Path) -> dict:
        raw = file_path.read_bytes()
        # Toy counts; a real impl would parse properly.
        text = raw.decode("ascii", errors="replace")
        vertex_count = text.count("\nv ") + text.count("\nvertex ")
        face_count = text.count("\nf ") + text.count("\nfacet ")
        return {
            "media_bytes": raw,
            "duration": 0,
            "vertex_count": vertex_count,
            "face_count": face_count,
        }

    def media_response(self, media: dict) -> MediaResponse:
        data = self._resolve_media_bytes(media)
        return MediaResponse(
            data=data or b"",
            mimetype="model/obj" if media.get("filename", "").endswith(".obj") else "model/stl",
            download_name=media.get("filename", "mesh"),
        )
```

The package wiring:

```python
# vtscore/media/mesh3d/__init__.py
from vtscore.media.mesh3d.media_type import Mesh3DMediaType

MEDIA_TYPE = Mesh3DMediaType()
CLIPPERS = []  # default to no clipping; add later if useful
```

That's the full library-tier contract for a new media type. The next
time `vtscore.media` is imported, the type is registered, files
matching `*.obj` / `*.stl` are picked up by folder importers, and
`/api/medias/<id>/media` serves the raw mesh bytes with the right
MIME type. To make it useful you also need at least one
[embedder](embedders.md) and probably a default [clipper](clippers.md)
that returns the mesh unchanged.

## Testing pattern

```python
# tests_lib/core/test_mesh3d_media_type.py
from pathlib import Path

from vtscore.media import get, get_by_folder_name


class TestMesh3DRegistration:
    def test_type_is_registered(self):
        mt = get("mesh3d")
        assert mt.type_id == "mesh3d"
        assert mt.file_extensions == ["*.obj", "*.stl"]

    def test_folder_import_name(self):
        # Default: folder_import_name == type_id
        mt = get_by_folder_name("mesh3d")
        assert mt.type_id == "mesh3d"


class TestMesh3DLoad:
    def test_load_media_data_counts_vertices(self, tmp_path: Path):
        mt = get("mesh3d")
        obj = tmp_path / "cube.obj"
        obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        data = mt.load_media_data(obj)
        assert data["vertex_count"] == 3
        assert data["face_count"] == 1
        assert data["duration"] == 0
```

Follow the pattern in [`tests_lib/core/`](../../../tests_lib/core/)
for media-type tests - the autouse fixtures stub every embedder, so
your test only needs to exercise the `MediaType` itself. If your type
needs a custom test medium, add a fixture under `tests_lib/fixtures/`
mirroring the audio / image / video pattern in
[`tests_lib/fixtures/medias.py`](../../../tests_lib/fixtures/medias.py).
