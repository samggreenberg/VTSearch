"""Server-files media source – re-ingests from a ``.npz`` paths archive.

When the ``server_files`` importer is given a ``.npz`` paths file the
origin params record both the path to that archive and the embedder name
stored inside it.  On re-ingestion this source reads the archive and
supplies the pre-computed embedding for each file so the embed stage is
skipped, provided the origin carries no clip params.

Plain ``.txt`` / ``.list`` paths files produce no pre-computed vectors
so the factory returns ``None`` for those origins; normal embedding
proceeds as usual.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from vtscore.datasets.sources.base import FetchedItem, MediaItem, MediaSource

__all__ = ["ServerFilesSource"]


class ServerFilesSource(MediaSource):
    """A media source backed by a ``.npz`` paths-and-vectors archive.

    On re-ingestion the source supplies pre-computed embedding vectors
    read directly from the archive, so files skip re-embedding as long
    as the origin carries no clip params.

    Args:
        npz_path: Path to the ``.npz`` archive.
        embedder_name: Name of the embedder that produced the vectors
            stored in the archive.  Passed through to every returned
            :class:`FetchedItem`.
    """

    name = "server_files"

    def __init__(self, npz_path: str | Path, embedder_name: str = "") -> None:
        self._npz_path = Path(npz_path)
        self._embedder_name = embedder_name
        self._path_to_vector: dict[str, np.ndarray] | None = None

    def _ensure_loaded(self) -> dict[str, np.ndarray]:
        if self._path_to_vector is None:
            from vtscore.datasets.importers._npz_vectors import read_npz_filenames_and_vectors  # noqa: PLC0415

            base_dir = self._npz_path.resolve().parent
            name_to_vec = read_npz_filenames_and_vectors(self._npz_path)
            mapping: dict[str, np.ndarray] = {}
            for raw_name, vec in name_to_vec.items():
                candidate = Path(raw_name.strip())
                if not candidate.is_absolute():
                    candidate = (base_dir / candidate).resolve()
                mapping[str(candidate)] = vec
            self._path_to_vector = mapping
        return self._path_to_vector

    def list_items(self, extensions: list[str] | None = None) -> Iterator[MediaItem]:
        """Yield items for every path recorded in the NPZ archive."""
        ext_set = {e.lower() for e in extensions} if extensions else None
        for abs_path_str in self._ensure_loaded():
            p = Path(abs_path_str)
            if ext_set is not None and p.suffix.lower() not in ext_set:
                continue
            if not p.is_file():
                continue
            yield MediaItem(key=abs_path_str, filename=p.name, source_name=self.name)

    def fetch_item(self, key: str) -> FetchedItem:
        """Fetch item by absolute path string; supplies pre-computed embedding when available."""
        path_to_vec = self._ensure_loaded()
        resolved = str(Path(key).resolve())
        p = Path(resolved)
        if not p.is_file():
            return FetchedItem(path=None)
        vec = path_to_vec.get(resolved) or path_to_vec.get(key)
        if vec is not None:
            return FetchedItem(path=p, embedding=vec, embedder_name=self._embedder_name)
        return FetchedItem(path=p)

    def resolve_path(self, origin_name: str = "", filename: str = "") -> FetchedItem:
        """Resolve by trying *origin_name* then *filename* as absolute paths."""
        for candidate_str in (origin_name, filename):
            if not candidate_str:
                continue
            item = self.fetch_item(candidate_str)
            if item.path is not None:
                return item
        return FetchedItem(path=None)


class _ServerFilesSourceFactory:
    """Factory for auto-discovery by :class:`~vtscore.plugins.PluginRegistry`.

    Resolves origins emitted by the :mod:`server_files` dataset importer
    when the paths file is a ``.npz`` archive.  Origins from plain ``.txt``
    or ``.list`` paths files return ``None`` (no pre-computed vectors).
    """

    name = "server_files"

    def create_from_origin(self, origin: dict) -> ServerFilesSource | None:
        params = origin.get("params", {})
        paths_file = params.get("paths_file", "")
        if not paths_file:
            return None
        p = Path(paths_file)
        if p.suffix.lower() != ".npz" or not p.is_file():
            return None
        embedder_name = params.get("embedder_name", "")
        return ServerFilesSource(p, embedder_name)


SOURCE = _ServerFilesSourceFactory()
