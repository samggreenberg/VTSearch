"""Manifest-backed media source for no-extraction archive members.

The ``local_archive_member`` importer records only ``{archive path, member}``
(plus an optional clip window) and re-derives a media's *bytes* through
:mod:`vtscore.datasets.archive_stream`.  That covers playback, but the
:class:`~vtscore.datasets.sources.base.MediaSource` contract the cross-dataset
resolver / example-sort uses is **path-based**, and an archive member has no
on-disk path -- so without a source factory, per-media "Find from origin"
(:func:`vtsearch.routes.media.server.example_sort_origin`) returns 400.

This source closes that gap without re-deriving bytes: the embeddings are
already precomputed in the ``.npz`` manifest, so it re-supplies a member's (or a
specific *window's*) vector straight out of the manifest.  A
:class:`~vtscore.datasets.sources.base.FetchedItem` returned here has
``path=None`` and ``embedding`` set; the route sorts directly on that vector
(no fetch, no re-embed), exactly as :func:`example_sort_by_id` reuses an
in-memory embedding.  Per the no-persisted-vectors rule nothing here writes the
vectors anywhere; they are read from the manifest on demand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from vtscore.datasets.importers._npz_vectors import (
    read_npz_archive_member_rows,
    read_npz_embedder_name,
    window_suffix,
)
from vtscore.datasets.sources.base import FetchedItem, MediaItem, MediaSource

__all__ = ["LocalArchiveMemberSource"]


class LocalArchiveMemberSource(MediaSource):
    """A media source that re-supplies precomputed vectors from an NPZ manifest.

    Each manifest row is indexed under its **window key** -- the member name plus
    the same per-window suffix the importer folds into a media's identity (see
    :func:`~vtscore.datasets.importers._npz_vectors.window_suffix`) -- and, as a
    fallback, under its bare member name (resolving to that member's first
    window).  Fetches return the row's embedding, never a file path.

    Args:
        manifest: Path to the ``.npz`` manifest the import used.
        embedder_name: The embedder that produced the vectors; falls back to the
            name recorded in the manifest when blank.
    """

    name = "local_archive_member"

    def __init__(self, manifest: str | Path, embedder_name: str = "") -> None:
        self._manifest = Path(manifest)
        self._embedder_name = embedder_name or read_npz_embedder_name(self._manifest)
        self._index: dict[str, dict] | None = None

    def _rows_by_key(self) -> dict[str, dict]:
        """Build (once) the ``{window_key | member: row}`` lookup for the manifest."""
        if self._index is not None:
            return self._index
        index: dict[str, dict] = {}
        for row in read_npz_archive_member_rows(self._manifest):
            member = row["member"]
            key = member + window_suffix(row.get("window_id"), row.get("clip_start"))
            index[key] = row
            # Bare-member fallback resolves to the member's first window.
            index.setdefault(member, row)
        self._index = index
        return index

    def _lookup(self, key: str) -> dict | None:
        """Resolve *key* to a manifest row, tolerating an ``archive::member`` form."""
        key = (key or "").strip()
        if not key:
            return None
        index = self._rows_by_key()
        row = index.get(key)
        if row is not None:
            return row
        # ``origin_name`` is stored as ``"<archive>::<member><suffix>"``; accept it.
        if "::" in key:
            return index.get(key.split("::", 1)[1])
        return None

    def _fetched(self, row: dict | None) -> FetchedItem:
        if row is None:
            return FetchedItem(path=None)
        return FetchedItem(
            path=None,
            embedding=np.asarray(row["vector"]),
            embedder_name=self._embedder_name,
        )

    def list_items(self, extensions: list[str] | None = None) -> Iterator[MediaItem]:
        exts = {e.lower() for e in extensions} if extensions else None
        seen: set[str] = set()
        for row in read_npz_archive_member_rows(self._manifest):
            member = row["member"]
            key = member + window_suffix(row.get("window_id"), row.get("clip_start"))
            if key in seen:
                continue
            seen.add(key)
            if exts is not None and Path(member).suffix.lower() not in exts:
                continue
            yield MediaItem(key=key, filename=row["filename"], source_name=self.name)

    def fetch_item(self, key: str) -> FetchedItem:
        return self._fetched(self._lookup(key))

    def resolve_path(self, origin_name: str = "", filename: str = "") -> FetchedItem:
        row = self._lookup(origin_name)
        if row is None and filename:
            target = Path(filename).name
            row = next(
                (r for r in read_npz_archive_member_rows(self._manifest) if r["filename"] == target),
                None,
            )
        return self._fetched(row)


class _LocalArchiveMemberSourceFactory:
    """Factory for auto-discovery by :class:`~vtscore.plugins.PluginRegistry`.

    Resolves the ``local_archive_member`` origins emitted by
    :class:`~vtscore.datasets.importers.local_archive_member.LocalArchiveMemberImporter`.
    """

    name = "local_archive_member"

    def create_from_origin(self, origin: dict) -> LocalArchiveMemberSource | None:
        params = origin.get("params", {})
        manifest = params.get("manifest", "")
        if not manifest:
            return None
        return LocalArchiveMemberSource(manifest, embedder_name=params.get("embedder_name", ""))


SOURCE = _LocalArchiveMemberSourceFactory()
