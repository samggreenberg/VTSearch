"""Pickle-based dataset loaders.

Loads datasets from a ``.pkl`` file (with optional companion media-file
directory), plus the small image-embedding helper used by importers and
the embedder/clipper sidecar utilities.  Split out from
:mod:`vtsearch.datasets.loader` for navigability.
"""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from vtsearch.datasets.loader import (
    ProgressCallback,
)
from vtsearch.security.pickle import safe_pickle_load


def load_dataset_from_pickle(  # noqa: C901
    file_path: Path,
    medias: dict[int, dict[str, Any]],
    thin: bool = False,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any] | None:
    """Load a dataset from a pickle file into the medias dict in-place.

    The pickle must contain a dict with a ``"medias"`` key mapping media IDs
    to media data dicts.  It may also include ``"audio_dir"``, ``"video_dir"``,
    ``"image_dir"``, or ``"text_dir"`` keys pointing to directories containing
    raw media files when the bytes are not stored inline.

    If media bytes are not stored inline in the pickle, the function attempts to
    load them from the companion directory entry in the pickle. Medias for which
    no bytes can be resolved are silently skipped (a warning is printed to
    stdout after loading).

    The ``medias`` dict is cleared before loading begins.

    Args:
        file_path: Path to a ``.pkl`` file previously created by
            :func:`export_dataset_to_file` or :func:`load_demo_dataset`.
        medias: Dict to populate in-place. Existing entries are removed before
            loading. Keys are media IDs (int); values are media data dicts.
        thin: When ``True``, skip loading media bytes into memory.  Inline
            bytes from the pickle are discarded and external-dir files are
            referenced by ``media_path`` instead of read.  Useful for CLI
            workflows that only need embeddings for scoring.

    Returns:
        ``None``.
    """
    if on_progress is not None:
        on_progress("loading", f"Reading {file_path.name}…", 0, 0)

    try:
        with open(file_path, "rb") as f:
            data = safe_pickle_load(f)
    except MemoryError:
        gc.collect()
        raise MemoryError(
            f"Out of memory while reading {file_path.name}. The pickle file is too large for available RAM."
        )

    medias.clear()

    if not isinstance(data, dict) or "medias" not in data:
        raise ValueError(f"Invalid pickle format in {file_path.name}: expected a dict with a 'medias' key.")
    medias_data = data["medias"]

    # Build lookup tables dynamically from the media type registry.
    from vtsearch.media import all_types

    _dir_keys: dict[str, str] = {}
    _extra_fields: dict[str, list[str]] = {}
    for mt in all_types():
        _dir_keys[mt.type_id] = mt.dir_key
        _extra_fields[mt.type_id] = mt.pickle_extra_fields

    # Convert to the app's media format
    missing_media = 0
    loaded_count = 0
    total_count = len(medias_data)
    # Report progress every ~2% or at least every 50 items
    _progress_interval = max(1, min(50, total_count // 50)) if total_count > 0 else 1
    if on_progress is not None:
        on_progress("loading", f"Processing 0 of {total_count} items…", 0, total_count)
    try:
        for media_id, media_info in medias_data.items():
            media_type = media_info.get("type", "audio")

            if thin:
                # ── Thin mode: skip bytes, store media_path if available ──
                media_path: str | None = media_info.get("media_path")

                # Try to resolve a media_path from the external directory
                if not media_path:
                    dir_key = _dir_keys.get(media_type)
                    if dir_key and dir_key in data and "filename" in media_info:
                        candidate = Path(data[dir_key]) / media_info["filename"]
                        if candidate.exists():
                            media_path = str(candidate.resolve())

                # We still need the embedding to be useful
                if "embedding" not in media_info:
                    missing_media += 1
                    continue

                fname = media_info.get("filename", f"media_{media_id}.{media_type}")
                media_data: dict[str, Any] = {
                    "id": media_id,
                    "type": media_type,
                    "embedder": media_info.get("embedder", ""),
                    "duration": media_info.get("duration", 0),
                    "file_size": media_info.get("file_size", 0),
                    "md5": media_info.get("md5", ""),
                    "embedding": np.array(media_info["embedding"]),
                    "media_bytes": None,
                    "media_string": None,
                    "media_path": media_path,
                    "filename": fname,
                    "category": media_info.get("category", "unknown"),
                    "origin": media_info.get("origin"),
                    "origin_name": media_info.get("origin_name", fname),
                }
                for field in _extra_fields.get(media_type, []):
                    media_data[field] = media_info.get(field)
                cm = media_info.get("custom_metadata")
                if cm:
                    media_data["custom_metadata"] = cm

                medias[media_id] = media_data
                loaded_count += 1
                if on_progress is not None and loaded_count % _progress_interval == 0:
                    on_progress(
                        "loading", f"Processing {loaded_count} of {total_count} items…", loaded_count, total_count
                    )
                continue

            # ── Full mode ──
            # Load the actual media content.
            media_bytes = None
            media_string = None
            media_path = None

            bytes_val = media_info.get("media_bytes")
            string_val = media_info.get("media_string")

            if bytes_val is not None:
                media_bytes = bytes_val
            elif string_val is not None:
                media_string = string_val
                media_bytes = string_val.encode("utf-8")
            else:
                # Try loading from the external directory via registry dir_key
                dir_key = _dir_keys.get(media_type)
                if dir_key and dir_key in data and "filename" in media_info:
                    ext_path = Path(data[dir_key]) / media_info["filename"]
                    if ext_path.exists():
                        if media_string is None and ext_path.suffix in (".txt", ".md"):
                            with open(ext_path, "r", encoding="utf-8") as f:
                                media_string = f.read()
                                media_bytes = media_string.encode("utf-8")
                        else:
                            with open(ext_path, "rb") as f:
                                media_bytes = f.read()
                        media_path = str(ext_path.resolve())
                    else:
                        missing_media += 1

            if media_bytes is not None:
                fname = media_info.get("filename", f"media_{media_id}.{media_type}")
                media_data = {
                    "id": media_id,
                    "type": media_type,
                    "embedder": media_info.get("embedder", ""),
                    "duration": media_info.get("duration", 0),
                    "file_size": media_info.get("file_size", len(media_bytes)),
                    "md5": media_info.get("md5") or hashlib.md5(media_bytes).hexdigest(),
                    "embedding": np.array(media_info["embedding"]),
                    "media_bytes": media_bytes,
                    "media_string": media_string,
                    "media_path": media_path or media_info.get("media_path"),
                    "filename": fname,
                    "category": media_info.get("category", "unknown"),
                    "origin": media_info.get("origin"),
                    "origin_name": media_info.get("origin_name", fname),
                }
                for field in _extra_fields.get(media_type, []):
                    media_data[field] = media_info.get(field)
                cm = media_info.get("custom_metadata")
                if cm:
                    media_data["custom_metadata"] = cm

                medias[media_id] = media_data
                loaded_count += 1
                if on_progress is not None and loaded_count % _progress_interval == 0:
                    on_progress(
                        "loading", f"Processing {loaded_count} of {total_count} items…", loaded_count, total_count
                    )
    except MemoryError:
        medias.clear()
        del data
        gc.collect()
        raise MemoryError(
            f"Out of memory after loading {loaded_count} of {total_count} medias from "
            f"{file_path.name}. Try a smaller dataset or free up system RAM."
        )

    # Release the raw pickle data now that medias are built
    del data  # noqa: F821 — ruff cannot see past `del data` in the except branch (which always re-raises)
    gc.collect()

    if missing_media > 0:
        print(f"WARNING: {missing_media} media files missing from {file_path}", flush=True)

    return None


def load_dataset_from_pickle_chunked(
    file_path: Path,
    chunk_size: int,
    thin: bool = False,
) -> Iterator[dict[int, dict[str, Any]]]:
    """Yield chunks of medias from a pickle dataset file.

    Works identically to :func:`load_dataset_from_pickle` but yields the
    medias in groups of at most *chunk_size*.  Each yielded dict is a
    self-contained medias dict with IDs starting at 1.

    The entire pickle is deserialized once (unavoidable for ``.pkl``
    format), but media bytes are dropped or skipped per-chunk so that
    only one chunk's worth of media data is alive at a time.

    Args:
        file_path: Path to a ``.pkl`` dataset file.
        chunk_size: Maximum number of medias per yielded chunk.
        thin: When ``True``, skip loading media bytes into memory.

    Yields:
        A dict mapping int media IDs (starting at 1) to media data dicts.
    """
    with open(file_path, "rb") as f:
        data = safe_pickle_load(f)

    if not isinstance(data, dict) or "medias" not in data:
        raise ValueError(f"Invalid pickle format in {file_path.name}: expected a dict with a 'medias' key.")
    medias_data = data["medias"]

    # Build lookup tables dynamically from the media type registry.
    from vtsearch.media import all_types

    _dir_keys: dict[str, str] = {}
    _extra_fields: dict[str, list[str]] = {}
    for mt in all_types():
        _dir_keys[mt.type_id] = mt.dir_key
        _extra_fields[mt.type_id] = mt.pickle_extra_fields

    all_media_ids = sorted(medias_data.keys())

    for start in range(0, len(all_media_ids), chunk_size):
        batch_ids = all_media_ids[start : start + chunk_size]
        chunk_medias: dict[int, dict[str, Any]] = {}
        new_id = 1

        for media_id in batch_ids:
            media_info = medias_data[media_id]
            media_type = media_info.get("type", "audio")

            if thin:
                media_path: str | None = media_info.get("media_path")
                if not media_path:
                    dir_key = _dir_keys.get(media_type)
                    if dir_key and dir_key in data and "filename" in media_info:
                        candidate = Path(data[dir_key]) / media_info["filename"]
                        if candidate.exists():
                            media_path = str(candidate.resolve())

                if "embedding" not in media_info:
                    continue

                fname = media_info.get("filename", f"media_{media_id}.{media_type}")
                media_data: dict[str, Any] = {
                    "id": new_id,
                    "type": media_type,
                    "embedder": media_info.get("embedder", ""),
                    "duration": media_info.get("duration", 0),
                    "file_size": media_info.get("file_size", 0),
                    "md5": media_info.get("md5", ""),
                    "embedding": np.array(media_info["embedding"]),
                    "media_bytes": None,
                    "media_string": None,
                    "media_path": media_path,
                    "filename": fname,
                    "category": media_info.get("category", "unknown"),
                    "origin": media_info.get("origin"),
                    "origin_name": media_info.get("origin_name", fname),
                }
                for field in _extra_fields.get(media_type, []):
                    media_data[field] = media_info.get(field)
                cm = media_info.get("custom_metadata")
                if cm:
                    media_data["custom_metadata"] = cm

                chunk_medias[new_id] = media_data
                new_id += 1
                continue

            # Full mode — same logic as load_dataset_from_pickle
            media_bytes = None
            media_string = None
            media_path = None

            bytes_val = media_info.get("media_bytes")
            string_val = media_info.get("media_string")

            if bytes_val is not None:
                media_bytes = bytes_val
            elif string_val is not None:
                media_string = string_val
                media_bytes = string_val.encode("utf-8")
            else:
                # Try loading from the external directory via registry dir_key
                dir_key = _dir_keys.get(media_type)
                if dir_key and dir_key in data and "filename" in media_info:
                    ext_path = Path(data[dir_key]) / media_info["filename"]
                    if ext_path.exists():
                        if media_string is None and ext_path.suffix in (".txt", ".md"):
                            with open(ext_path, "r", encoding="utf-8") as f:
                                media_string = f.read()
                                media_bytes = media_string.encode("utf-8")
                        else:
                            with open(ext_path, "rb") as f:
                                media_bytes = f.read()
                        media_path = str(ext_path.resolve())

            if media_bytes is not None:
                fname = media_info.get("filename", f"media_{media_id}.{media_type}")
                media_data = {
                    "id": new_id,
                    "type": media_type,
                    "embedder": media_info.get("embedder", ""),
                    "duration": media_info.get("duration", 0),
                    "file_size": media_info.get("file_size", len(media_bytes)),
                    "md5": media_info.get("md5") or hashlib.md5(media_bytes).hexdigest(),
                    "embedding": np.array(media_info["embedding"]),
                    "media_bytes": media_bytes,
                    "media_string": media_string,
                    "media_path": media_path or media_info.get("media_path"),
                    "filename": fname,
                    "category": media_info.get("category", "unknown"),
                    "origin": media_info.get("origin"),
                    "origin_name": media_info.get("origin_name", fname),
                }
                for field in _extra_fields.get(media_type, []):
                    media_data[field] = media_info.get(field)
                cm = media_info.get("custom_metadata")
                if cm:
                    media_data["custom_metadata"] = cm

                chunk_medias[new_id] = media_data
                new_id += 1

        if chunk_medias:
            yield chunk_medias


def _write_embedder_sidecar(pkl_path: Path, embedder_name: str) -> None:
    """Write a small ``<name>.embedder`` file next to *pkl_path*.

    The file stores the embedder name used to produce the pickle so that
    ``demo_dataset_list`` can cheaply check whether the cached embeddings
    match the user's current embedder selection without loading the full pkl.
    """
    sidecar = pkl_path.with_suffix(".embedder")
    sidecar.write_text(embedder_name, encoding="utf-8")


def _write_clipper_sidecar(pkl_path: Path, clipper_name: str) -> None:
    """Write a small ``<name>.clipper`` file next to *pkl_path*."""
    sidecar = pkl_path.with_suffix(".clipper")
    sidecar.write_text(clipper_name, encoding="utf-8")


def read_pkl_clipper(pkl_path: Path) -> str | None:
    """Return the clipper name stored for *pkl_path*, or ``None`` if unknown."""
    sidecar = pkl_path.with_suffix(".clipper")
    if sidecar.exists():
        return sidecar.read_text(encoding="utf-8").strip()
    return None


def read_pkl_embedder(pkl_path: Path) -> str | None:
    """Return the embedder name stored for *pkl_path*, or ``None`` if unknown."""
    sidecar = pkl_path.with_suffix(".embedder")
    if sidecar.exists():
        return sidecar.read_text(encoding="utf-8").strip()
    return None
