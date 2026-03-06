"""Resolve label entries to embeddings by following their origin trails.

When a detector's training labels don't match a target dataset (cross-dataset
scenario), we need to find the original media files, embed them, and use those
embeddings for training.  This module handles that resolution:

1. Given a label entry's origin info, resolve to an actual file on disk.
2. Embed the file using the appropriate embedder for the media type.
3. Return resolved embeddings with availability stats.
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

    Follows the origin trail to find the actual file on disk:

    - ``folder``: ``origin.params.path / origin_name``
    - ``pdf``: ``origin.params.path`` (the PDF file itself)
    - ``http_archive``: re-downloads and extracts if needed
    - ``demo``: checks demo cache directory
    - ``converter``: resolves parent origin, finds source file
    - ``dupe_set``: tries each member until one resolves

    Returns the file path if found, or ``None``.
    """
    if origin is None:
        return None

    importer = origin.get("importer", "")
    params = origin.get("params", {})

    if importer == "folder":
        return _resolve_folder(params, origin_name, filename)

    if importer == "pdf":
        return _resolve_pdf(params)

    if importer == "http_archive":
        return _resolve_http_archive(params, origin_name, filename)

    if importer == "demo":
        return _resolve_demo(params, origin_name, filename)

    if importer == "converter":
        return _resolve_converter(params, origin_name, filename)

    if importer == "dupe_set":
        return _resolve_dupe_set(origin, origin_name, filename)

    return None


def _resolve_folder(
    params: dict[str, str], origin_name: str, filename: str
) -> Path | None:
    folder = params.get("path", "")
    if not folder:
        return None
    folder_path = Path(folder)
    for name in [origin_name, filename]:
        if name:
            candidate = folder_path / name
            if candidate.is_file():
                return candidate
    return None


def _resolve_pdf(params: dict[str, str]) -> Path | None:
    path = params.get("path", "")
    if path:
        p = Path(path)
        if p.is_file():
            return p
    return None


def _resolve_http_archive(
    params: dict[str, str], origin_name: str, filename: str
) -> Path | None:
    """Resolve a file from an http_archive origin.

    Checks if the archive has already been downloaded to the data directory.
    If found, extracts and searches for the file.  If not found, downloads
    the archive first.
    """
    url = params.get("url", "")
    if not url:
        return None

    from vtsearch.config import DATA_DIR

    url_filename = url.rsplit("/", 1)[-1].split("?")[0]
    download_path = DATA_DIR / f"http_archive_download_{url_filename}"
    extract_dir = DATA_DIR / f"http_archive_resolve_{url_filename}"

    # If already extracted, search there
    if extract_dir.is_dir():
        found = _search_dir_for_file(extract_dir, origin_name, filename)
        if found:
            return found

    # If archive downloaded but not extracted (or extraction didn't find file)
    if download_path.is_file() and not extract_dir.is_dir():
        try:
            from vtsearch.datasets.importers.http_zip import _extract_archive

            extract_dir.mkdir(parents=True, exist_ok=True)
            _extract_archive(download_path, extract_dir)
            found = _search_dir_for_file(extract_dir, origin_name, filename)
            if found:
                return found
        except Exception:
            log.warning("Failed to extract %s", download_path, exc_info=True)

    # Download the archive if not present
    if not download_path.is_file():
        try:
            log.info("Downloading %s for label resolution...", url)
            _download_url(url, download_path)
            if download_path.is_file():
                from vtsearch.datasets.importers.http_zip import _extract_archive

                extract_dir.mkdir(parents=True, exist_ok=True)
                _extract_archive(download_path, extract_dir)
                found = _search_dir_for_file(extract_dir, origin_name, filename)
                if found:
                    return found
        except Exception:
            log.warning("Failed to download %s for label resolution", url, exc_info=True)

    return None


def _resolve_demo(
    params: dict[str, str], origin_name: str, filename: str
) -> Path | None:
    """Resolve a file from a demo dataset origin.

    Demo datasets may store files on disk (audio, video, document types keep
    files in a type-specific directory).  Text and image demos often store
    data in-memory only, so resolution may fail for those.
    """
    demo_name = params.get("name", "")
    if not demo_name:
        return None

    from vtsearch.datasets.config import DEMO_DATASETS

    demo_info = DEMO_DATASETS.get(demo_name)
    if demo_info is None:
        return None

    media_type = demo_info.get("media_type", "")
    source = demo_info.get("source", "")

    # For demo datasets that download to a known directory,
    # check common locations.  Audio demos (ESC-50, GTZAN, etc.)
    # download to DATA_DIR / <source_folder>.
    from vtsearch.config import DATA_DIR

    # Try DATA_DIR-based paths (common for extracted archives)
    for name in [origin_name, filename]:
        if not name:
            continue
        # Direct path under data/
        candidate = DATA_DIR / name
        if candidate.is_file():
            return candidate

    # For audio/video/document types that use external directories,
    # check the embeddings pkl for the stored dir_key path.
    # But we can't open pkls per the design — instead check known
    # download locations.
    if source:
        # Many demo sources download to DATA_DIR / <extracted_folder>
        # Try to find the file by searching DATA_DIR recursively
        # (bounded to avoid scanning huge trees)
        for name in [origin_name, filename]:
            if not name:
                continue
            basename = Path(name).name
            for candidate_dir in DATA_DIR.iterdir():
                if not candidate_dir.is_dir():
                    continue
                candidate = candidate_dir / name
                if candidate.is_file():
                    return candidate
                # One level of rglob
                matches = list(candidate_dir.rglob(basename))
                if matches:
                    return matches[0]

    return None


def _resolve_converter(
    params: dict[str, str], origin_name: str, filename: str
) -> Path | None:
    """Resolve a file from a converter origin.

    Converter origins track a parent importer and a source file within it.
    """
    source_file = params.get("source_file", "")
    parent_importer = params.get("parent_importer", "")
    parent_path = params.get("parent_path", "")
    parent_url = params.get("parent_url", "")

    # Try resolving via parent
    if parent_importer == "folder" and parent_path and source_file:
        candidate = Path(parent_path) / source_file
        if candidate.is_file():
            return candidate

    if parent_importer == "http_archive" and parent_url:
        return _resolve_http_archive({"url": parent_url}, source_file, "")

    return None


def _resolve_dupe_set(
    origin: dict[str, Any], origin_name: str, filename: str
) -> Path | None:
    members = origin.get("members", [])
    for m in members:
        result = resolve_file_from_origin(
            m.get("origin"),
            m.get("origin_name", ""),
            m.get("filename", ""),
        )
        if result is not None:
            return result
    return None


def _search_dir_for_file(
    directory: Path, origin_name: str, filename: str
) -> Path | None:
    """Search a directory for a file matching origin_name or filename."""
    for name in [origin_name, filename]:
        if not name:
            continue
        candidate = directory / name
        if candidate.is_file():
            return candidate
        # Search recursively by basename
        matches = list(directory.rglob(Path(name).name))
        if matches:
            return matches[0]
    return None


def _download_url(url: str, dest: Path) -> None:
    """Download a URL to a local file."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(dest))  # noqa: S310


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
