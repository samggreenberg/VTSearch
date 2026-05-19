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


def _read_pickle_dataset(file_path: Path) -> dict[str, Any]:
    """Load a dataset pickle and assert the ``"medias"`` envelope.

    Translates :class:`MemoryError` into a contextual message and raises
    :class:`ValueError` when the file does not contain a dict with a
    ``"medias"`` key.
    """
    try:
        with open(file_path, "rb") as f:
            data = safe_pickle_load(f)
    except MemoryError:
        gc.collect()
        raise MemoryError(
            f"Out of memory while reading {file_path.name}. The pickle file is too large for available RAM."
        )
    if not isinstance(data, dict) or "medias" not in data:
        raise ValueError(f"Invalid pickle format in {file_path.name}: expected a dict with a 'medias' key.")
    return data


def _build_pickle_dir_maps() -> tuple[dict[str, str], dict[str, list[str]]]:
    """Build (dir_keys, extra_fields) maps keyed by media type id."""
    from vtsearch.media import all_types  # noqa: PLC0415

    dir_keys: dict[str, str] = {}
    extra_fields: dict[str, list[str]] = {}
    for mt in all_types():
        dir_keys[mt.type_id] = mt.dir_key
        extra_fields[mt.type_id] = mt.pickle_extra_fields
    return dir_keys, extra_fields


def _resolve_thin_media_path(
    media_type: str,
    media_info: dict[str, Any],
    data: dict[str, Any],
    dir_keys: dict[str, str],
) -> str | None:
    """Resolve ``media_path`` for thin-mode pickle loads.

    Returns the path stored on the media (if any) or a probe of the
    pickle's external directory entry for the media's type.
    """
    media_path = media_info.get("media_path")
    if media_path:
        return media_path
    dir_key = dir_keys.get(media_type)
    if not dir_key or dir_key not in data or "filename" not in media_info:
        return None
    candidate = Path(data[dir_key]) / media_info["filename"]
    if candidate.exists():
        return str(candidate.resolve())
    return None


def _load_pickle_media_payload(
    media_type: str,
    media_info: dict[str, Any],
    data: dict[str, Any],
    dir_keys: dict[str, str],
) -> tuple[bytes | None, str | None, str | None, bool]:
    """Resolve ``(media_bytes, media_string, media_path, missing)`` for full mode.

    Returns ``missing=True`` when the media references an external file
    that does not exist on disk.  ``media_bytes`` may still be ``None``
    when neither inline nor external content was found.
    """
    bytes_val = media_info.get("media_bytes")
    string_val = media_info.get("media_string")
    if bytes_val is not None:
        return bytes_val, None, None, False
    if string_val is not None:
        return string_val.encode("utf-8"), string_val, None, False

    dir_key = dir_keys.get(media_type)
    if not dir_key or dir_key not in data or "filename" not in media_info:
        return None, None, None, False
    ext_path = Path(data[dir_key]) / media_info["filename"]
    if not ext_path.exists():
        return None, None, None, True
    if ext_path.suffix in (".txt", ".md"):
        with open(ext_path, "r", encoding="utf-8") as f:
            txt = f.read()
        return txt.encode("utf-8"), txt, str(ext_path.resolve()), False
    with open(ext_path, "rb") as f:
        return f.read(), None, str(ext_path.resolve()), False


def _build_pickle_thin_media(
    new_id: int,
    media_info: dict[str, Any],
    media_type: str,
    media_path: str | None,
    extra_fields: list[str],
) -> dict[str, Any]:
    """Build a thin-mode media dict from a pickle entry (no bytes loaded)."""
    fname = media_info.get("filename", f"media_{new_id}.{media_type}")
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
    for field in extra_fields:
        media_data[field] = media_info.get(field)
    cm = media_info.get("custom_metadata")
    if cm:
        media_data["custom_metadata"] = cm
    return media_data


def _build_pickle_full_media(
    new_id: int,
    media_info: dict[str, Any],
    media_type: str,
    media_bytes: bytes,
    media_string: str | None,
    media_path: str | None,
    extra_fields: list[str],
) -> dict[str, Any]:
    """Build a full-mode media dict from a pickle entry."""
    fname = media_info.get("filename", f"media_{new_id}.{media_type}")
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
    for field in extra_fields:
        media_data[field] = media_info.get(field)
    cm = media_info.get("custom_metadata")
    if cm:
        media_data["custom_metadata"] = cm
    return media_data


def _convert_one_pickle_media(
    new_id: int,
    media_info: dict[str, Any],
    thin: bool,
    data: dict[str, Any],
    dir_keys: dict[str, str],
    extra_fields_map: dict[str, list[str]],
) -> tuple[dict[str, Any] | None, bool]:
    """Convert one pickle media entry to the app's media format.

    Returns ``(media_data, missing)``.  ``media_data`` is ``None`` when
    the entry is unusable (thin without embedding, full without bytes);
    ``missing`` is ``True`` only when an external file reference failed
    to resolve (used to bump the "missing media" warning counter).
    """
    media_type = media_info.get("type", "audio")
    extra_fields = extra_fields_map.get(media_type, [])

    if thin:
        if "embedding" not in media_info:
            return None, True
        media_path = _resolve_thin_media_path(media_type, media_info, data, dir_keys)
        return _build_pickle_thin_media(new_id, media_info, media_type, media_path, extra_fields), False

    media_bytes, media_string, media_path, missing = _load_pickle_media_payload(
        media_type,
        media_info,
        data,
        dir_keys,
    )
    if media_bytes is None:
        return None, missing
    return (
        _build_pickle_full_media(
            new_id,
            media_info,
            media_type,
            media_bytes,
            media_string,
            media_path,
            extra_fields,
        ),
        False,
    )


def load_dataset_from_pickle(
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

    data = _read_pickle_dataset(file_path)
    medias.clear()
    medias_data = data["medias"]
    dir_keys, extra_fields_map = _build_pickle_dir_maps()

    missing_media = 0
    loaded_count = 0
    total_count = len(medias_data)
    _progress_interval = max(1, min(50, total_count // 50)) if total_count > 0 else 1
    if on_progress is not None:
        on_progress("loading", f"Processing 0 of {total_count} items…", 0, total_count)

    try:
        for media_id, media_info in medias_data.items():
            media_data, missing = _convert_one_pickle_media(
                media_id,
                media_info,
                thin,
                data,
                dir_keys,
                extra_fields_map,
            )
            if media_data is None:
                if missing:
                    missing_media += 1
                continue
            medias[media_id] = media_data
            loaded_count += 1
            if on_progress is not None and loaded_count % _progress_interval == 0:
                on_progress("loading", f"Processing {loaded_count} of {total_count} items…", loaded_count, total_count)
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
    data = _read_pickle_dataset(file_path)
    medias_data = data["medias"]
    dir_keys, extra_fields_map = _build_pickle_dir_maps()

    all_media_ids = sorted(medias_data.keys())

    for start in range(0, len(all_media_ids), chunk_size):
        batch_ids = all_media_ids[start : start + chunk_size]
        chunk_medias: dict[int, dict[str, Any]] = {}
        new_id = 1

        for media_id in batch_ids:
            media_info = medias_data[media_id]
            media_data, _missing = _convert_one_pickle_media(
                new_id,
                media_info,
                thin,
                data,
                dir_keys,
                extra_fields_map,
            )
            if media_data is None:
                continue
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
