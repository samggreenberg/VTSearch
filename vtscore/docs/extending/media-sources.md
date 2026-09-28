# Writing a `MediaSource`

A media source gives the library file-level access to media at a
location: a local folder, an HTTP archive, one member of a tar/zip, an S3
bucket. It sits *below* dataset importers. Its main job is the reverse
direction - turning an `origin` saved in a detector's labelset back into a
local file, so the detector can be re-embedded and retrained on a machine
that never loaded the original dataset
([concepts.md § Origin](../concepts.md#3-origin)).

Sources live in `vtscore.datasets.sources` (sentinel `SOURCE`) and are
also discovered from the `vtscore.media_sources` entry-point group. Unlike
every other family, the sentinel is a **factory**, not the plugin itself:
sources are stateful (an archive source may download and extract on first
use), so each lookup builds a fresh instance.

**App-side counterpart:** [`docs/EXTENDING-media.md § Adding a Media
Source`](../../../docs/EXTENDING-media.md#adding-a-media-source) walks a
longer S3 example with a parallel `resolve_paths()` override. This guide
is the library contract.

## Contents

- [How a source is found](#how-a-source-is-found)
- [`MediaSource` members](#mediasource-members)
- [`FetchedItem`: returning more than a path](#fetcheditem-returning-more-than-a-path)
- [Where the file goes](#where-the-file-goes)
- [Worked example](#worked-example)
- [Testing pattern](#testing-pattern)

## How a source is found

`vtscore.datasets.sources.get_source_for_origin(origin)` reads
`origin["importer"]`, looks up the factory whose `name` equals it, and
returns `factory.create_from_origin(origin)` - or `None` when no factory
matches (pickle, `combine_datasets` and other non-file origins). So a
factory's `name` is the **importer name whose origins it resolves**, not
the source's own name: the built-in `LocalFolderSource` (`name =
"local_folder"`) is reached through a factory named `server_folder`,
because that is the importer that stamps those origins.

Two call sites use the result:

- **Detector re-derivation.** `vtscore.detectors.resolver` calls
  `source.resolve_path(origin_name, filename)` for each saved label and
  registers `source.cleanup()` to run when the caller's
  `resolve_file_context()` block exits, so a file in a temporary
  directory stays valid while it is embedded.
- **Label re-ingestion.** `vtscore.datasets.ingest` calls
  `source.resolve_paths(pairs)` once for a whole group of missing
  entries, then embeds each resolved file - unless the `FetchedItem`
  already carries a vector (see below).

The built-in factories: `server_folder`, `http_archive`, `local_archive`,
`local_archive_member`, `server_files`, `url_download`
(`list_media_sources()` returns them).

## `MediaSource` members

`MediaSource` ([`vtscore/datasets/sources/base.py`](../../datasets/sources/base.py))
is an ABC.

| Member | Kind | Purpose |
|--------|------|---------|
| `name` | class attr | Short identifier for the source type, stamped on each `MediaItem` it lists |
| `list_items(extensions=None)` | abstract | Yield a `MediaItem(key, filename, source_name)` per file; `extensions` is a list of dotted lowercase suffixes |
| `fetch_item(key)` | abstract | Return a `FetchedItem` for one `MediaItem.key`; `path=None` when absent |
| `resolve_path(origin_name="", filename="")` | abstract | Try `origin_name`, then `filename`; return a `FetchedItem` (`path=None` when neither resolves) |
| `fetch_items(keys)` | optional | Bulk `fetch_item`; default loops. Returns `{key: FetchedItem}` |
| `resolve_paths(entries)` | optional | Bulk `resolve_path` over `(origin_name, filename)` pairs, result aligned with the input; default loops. This is the one re-ingestion calls, so override it to parallelise |
| `cleanup()` | optional | Release temp files or directories; default no-op |

The factory needs only two things: a `name` and a
`create_from_origin(origin) -> MediaSource | None` method. Return `None`
when the origin lacks what you need (the built-ins do this for an empty
`path` or `url`), so resolution falls through to the importer's own
`resolve_file()`.

Validate anything you build from `origin["params"]` before touching the
filesystem or the network: origins come back from labelset files, which
may have been written elsewhere. The built-ins confine keys to the
source root (`LocalFolderSource.fetch_item` rejects a key that escapes
it); a network source should run URLs through
`vtscore.security.url_validation.validate_url`.

## `FetchedItem`: returning more than a path

Every fetch and resolve method returns
`vtscore.datasets.sources.base.FetchedItem`, not a bare path:

| Field | Type | Meaning |
|-------|------|---------|
| `path` | `Path \| None` | Local file, or `None` when not found |
| `embedding` | `np.ndarray \| None` | A pre-computed vector. Re-ingestion skips the embed step when this is set, unless the origin carries clip params |
| `embedder_name` | `str` | Required alongside `embedding`: the embedder whose space the vector is in |
| `extra` | `dict` | Source-authoritative media fields (`file_size`, `duration`, …) merged into the media record |

`FetchedItem(path=local_path)` is enough for a source that only
downloads files. Fill the other fields when your backend already returns
them in the same round trip. A vector returned here is used in memory for
the current dataset only; never cache it to disk yourself.

## Where the file goes

In-tree, a source is a **flat module**:
`vtscore/datasets/sources/<your_source>.py`, exposing
`SOURCE = YourSourceFactory()`. The registry scans module files, not
sub-packages, so a source defined in a sub-directory's `__init__.py` is
never seen.

Out-of-tree, point an entry point at the factory instance:

```toml
[project.entry-points."vtscore.media_sources"]
bucket = "my_pkg.bucket_source:SOURCE"
```

The entry-point *name* is only a label; the registry key is the
factory's `name`. A factory whose `name` clashes with a built-in is
skipped with a warning, and one whose import raises is registered as a
tombstone that re-raises on use (see
[architecture.md § Plugin discovery](../architecture.md#plugin-discovery)).

A source only helps if some importer stamps origins it can read. For a
new storage backend, ship the source with a
[dataset importer](dataset-importers.md) whose `name` matches the
factory's, and keep everything the source needs in the origin's `params`.

## Worked example

A read-only source for a directory served by a plain HTTP file server,
resolving origins stamped by a hypothetical `http_mirror` importer whose
`params` carry a `base_url`.

```python
# my_pkg/mirror_source.py
from __future__ import annotations

import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote

from vtscore.datasets.sources.base import FetchedItem, MediaItem, MediaSource
from vtscore.security.url_validation import validate_url


class HttpMirrorSource(MediaSource):
    """Fetch individual files from an HTTP directory listing."""

    name = "http_mirror"

    def __init__(self, base_url: str) -> None:
        self._base = base_url.rstrip("/") + "/"
        self._tmp = tempfile.TemporaryDirectory(prefix="http_mirror_")

    def list_items(self, extensions: list[str] | None = None) -> Iterator[MediaItem]:
        # A bare file server has no machine-readable listing; this source
        # only resolves known names. Sources that can enumerate should.
        return iter(())

    def fetch_item(self, key: str) -> FetchedItem:
        if not key or key.startswith("/") or ".." in Path(key).parts:
            return FetchedItem(path=None)
        url = validate_url(self._base + quote(key))
        dest = Path(self._tmp.name) / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - validated above
                dest.write_bytes(resp.read())
        except OSError:
            return FetchedItem(path=None)
        return FetchedItem(path=dest)

    def resolve_path(self, origin_name: str = "", filename: str = "") -> FetchedItem:
        for name in (origin_name, filename):
            if name:
                item = self.fetch_item(name)
                if item.path is not None:
                    return item
        return FetchedItem(path=None)

    def cleanup(self) -> None:
        self._tmp.cleanup()


class _HttpMirrorSourceFactory:
    name = "http_mirror"  # the importer whose origins this resolves

    def create_from_origin(self, origin: dict[str, Any]) -> HttpMirrorSource | None:
        base_url = origin.get("params", {}).get("base_url", "")
        return HttpMirrorSource(base_url) if base_url else None


SOURCE = _HttpMirrorSourceFactory()
```

Because `fetch_item` writes into a temporary directory the source owns,
`cleanup()` deletes it - which is why callers hold the path only inside
`resolve_file_context()`, never past it.

## Testing pattern

Test the factory and the source directly; there is no need to install
the entry point to exercise the contract. Stub the network.

```python
# tests_lib/io/test_http_mirror_source.py
from unittest.mock import patch

from my_pkg.mirror_source import SOURCE, HttpMirrorSource


class _Resp:
    def __enter__(self): return self
    def __exit__(self, *exc): return False
    def read(self): return b"RIFF...."


def test_factory_needs_a_base_url():
    assert SOURCE.create_from_origin({"importer": "http_mirror", "params": {}}) is None
    src = SOURCE.create_from_origin(
        {"importer": "http_mirror", "params": {"base_url": "https://example.com/audio"}}
    )
    assert isinstance(src, HttpMirrorSource)
    src.cleanup()


def test_resolve_path_falls_back_to_filename():
    src = HttpMirrorSource("https://example.com/audio")
    try:
        with patch("urllib.request.urlopen", return_value=_Resp()), \
             patch("my_pkg.mirror_source.validate_url", side_effect=lambda u: u):
            item = src.resolve_path(origin_name="", filename="bark.wav")
        assert item.path is not None and item.path.read_bytes() == b"RIFF...."
    finally:
        src.cleanup()


def test_rejects_traversal():
    src = HttpMirrorSource("https://example.com/audio")
    try:
        assert src.fetch_item("../etc/passwd").path is None
    finally:
        src.cleanup()
```

`validate_url` resolves the hostname to screen private addresses, which
is why the test stubs it alongside `urlopen`. The built-in sources'
tests under `tests_lib/datasets/` and `tests_lib/io/` cover archive
extraction and the resolver's cleanup timing.
