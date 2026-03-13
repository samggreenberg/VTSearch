"""Resolve label entries to embeddings by following their origin trails.

When a detector's training labels don't match a target dataset (cross-dataset
scenario), we need to find the original media files, embed them, and use those
embeddings for training.  This module handles that resolution:

1. Given a label entry's origin info, resolve to an actual file on disk.
2. Embed the file using the appropriate embedder for the media type.
3. Return resolved embeddings with availability stats.

File resolution is **not** hardcoded to specific importers.  Instead,
:func:`resolve_file_from_origin` looks up the importer by name from the
auto-discovered registry and calls its
:meth:`~vtsearch.datasets.importers.base.DatasetImporter.resolve_file` method.
Adding a new ``DatasetImporter`` with a ``resolve_file`` override automatically
extends the resolution capability — no changes to this module required.

Two synthetic origin types (``dupe_set`` and ``converter``) are handled inline
because they are not importers in the registry — they delegate to real
importers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ResolvedLabels:
    """Result of resolving label entries to embeddings."""

    embeddings: list[np.ndarray] = field(default_factory=list)
    labels: list[float] = field(default_factory=list)
    resolved_count: int = 0
    total_count: int = 0
    missing_entries: list[dict[str, Any]] = field(default_factory=list)

    @property
    def available_fraction(self) -> float:
        return self.resolved_count / self.total_count if self.total_count else 0.0

    @property
    def has_good_and_bad(self) -> bool:
        return any(v == 1.0 for v in self.labels) and any(v == 0.0 for v in self.labels)


def resolve_file_from_origin(
    origin: dict[str, Any] | None,
    origin_name: str = "",
    filename: str = "",
) -> Path | None:
    """Resolve a media file from its origin information.

    Looks up the importer named in ``origin["importer"]`` from the
    auto-discovered importer registry and calls its ``resolve_file()``
    method.  Two synthetic origin types are handled inline:

    - ``dupe_set``: tries each member until one resolves.
    - ``converter``: reconstructs the parent origin and delegates.

    Returns the file path if found, or ``None``.
    """
    if origin is None:
        return None

    importer_name = origin.get("importer", "")

    # -- Synthetic origins that delegate to real importers --

    if importer_name == "dupe_set":
        return _resolve_dupe_set(origin)

    if importer_name == "converter":
        return _resolve_converter(origin.get("params", {}))

    # -- Source-based dispatch (preferred) --

    from vtsearch.datasets.sources import get_source_for_origin

    source = get_source_for_origin(origin)
    if source is not None:
        result = source.resolve_path(origin_name, filename)
        if result is not None:
            return result

    # -- Registry-based dispatch (fallback for importers without a source) --

    from vtsearch.datasets.importers import get_importer

    importer = get_importer(importer_name)
    if importer is not None:
        result = importer.resolve_file(origin, origin_name, filename)
        if result is not None:
            return result

    # -- Generic fallback for unregistered origins with a path param --
    # Handles synthetic origins like "pdf" that store a direct file path.
    path = origin.get("params", {}).get("path", "")
    if path:
        p = Path(path)
        if p.is_file():
            return p

    return None


def _resolve_dupe_set(origin: dict[str, Any]) -> Path | None:
    """Try each member of a dupe_set until one resolves."""
    for m in origin.get("members", []):
        result = resolve_file_from_origin(
            m.get("origin"),
            m.get("origin_name", ""),
            m.get("filename", ""),
        )
        if result is not None:
            return result
    return None


def _resolve_converter(params: dict[str, str]) -> Path | None:
    """Resolve a converter origin by rebuilding its parent origin."""
    source_file = params.get("source_file", "")
    parent_importer = params.get("parent_importer", "")
    if not source_file or not parent_importer:
        return None

    # Reconstruct a parent origin dict from the converter's stored params
    parent_params: dict[str, str] = {}
    if params.get("parent_path"):
        parent_params["path"] = params["parent_path"]
    if params.get("parent_url"):
        parent_params["url"] = params["parent_url"]

    parent_origin = {"importer": parent_importer, "params": parent_params}
    return resolve_file_from_origin(parent_origin, origin_name=source_file)


def embed_file(file_path: Path, media_type: str) -> np.ndarray | None:
    """Embed a media file using the appropriate embedder for the media type."""
    from vtsearch.media import embedders_for_type

    avail = embedders_for_type(media_type)
    if not avail:
        return None
    return avail[0].embed_media(file_path)


def resolve_label_embeddings(
    labels: list[dict[str, Any]],
    media_type: str,
) -> ResolvedLabels:
    """Resolve label entries to embeddings by following their origin trails.

    For each label entry, attempts to:
    1. Resolve the original media file from its origin info
    2. Embed it using the appropriate embedder for *media_type*
    3. Collect the embedding and label value

    Args:
        labels: List of label dicts (with origin, origin_name, filename, label keys).
        media_type: The media type for embedding (e.g. "audio", "image").

    Returns:
        A :class:`ResolvedLabels` with resolved embeddings, stats, and missing entries.
    """
    result = ResolvedLabels()

    for entry in labels:
        label_val = entry.get("label", "")
        if label_val not in ("good", "bad"):
            continue

        result.total_count += 1

        origin = entry.get("origin")
        origin_name = entry.get("origin_name", "")
        filename = entry.get("filename", "")

        file_path = resolve_file_from_origin(origin, origin_name, filename)
        if file_path is None:
            result.missing_entries.append(entry)
            continue

        embedding = embed_file(file_path, media_type)
        if embedding is None:
            result.missing_entries.append(entry)
            continue

        result.embeddings.append(embedding)
        result.labels.append(1.0 if label_val == "good" else 0.0)
        result.resolved_count += 1

    return result
