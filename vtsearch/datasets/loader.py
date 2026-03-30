"""Dataset loading and management utilities.

All public functions that perform I/O accept an optional ``on_progress``
callback with the signature
``(status: str, message: str, current: int, total: int) -> None``.
When omitted the functions fall back to the application-wide
:func:`~vtsearch.utils.update_progress` reporter; pass an explicit callback
to use these functions outside the Flask app.
"""

from __future__ import annotations

import gc
import hashlib
import io
import pickle
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import numpy as np
from PIL import Image

from vtsearch.config import EMBEDDINGS_DIR
from vtsearch.datasets.config import DEMO_DATASETS
from vtsearch.datasets.metadata import (  # noqa: F401  — re-exported for consumers
    load_audio_metadata_from_folders,
    load_cifar10_batch,
    load_esc50_metadata,
    load_image_metadata_from_folders,
    load_oxford_flowers_metadata,
    load_paragraph_metadata_from_folders,
    load_urbansound8k_metadata,
    load_video_metadata_from_folders,
)
from vtsearch.datasets.pickle_security import (  # noqa: F401  — re-exported for consumers
    RestrictedUnpickler,
    _PICKLE_SAFE_CLASSES,
    safe_pickle_load,
)
from vtsearch.utils.paths import rglob_follow_symlinks

ProgressCallback = Callable[[str, str, int, int], None]


def _default_progress() -> ProgressCallback:
    """Lazily resolve the progress callback for the current thread.

    Checks for a per-thread callback first (set during parallel dataset
    loading) and falls back to the global singleton.
    """
    from vtsearch.utils.progress import get_thread_progress

    cb = get_thread_progress()
    if cb is not None:
        return cb
    from vtsearch.utils import update_progress

    return update_progress


def _pop_md5_key(d: dict[str, Any]) -> str:
    """Pop and return the MD5 value from *d*, trying both ``"md5"`` and ``"MD5"`` keys.

    Returns the value (or ``""`` if neither key is present) and removes the
    matched key from *d* so it doesn't leak into downstream metadata.
    """
    for key in ("md5", "MD5"):
        val = d.get(key)
        if val:
            del d[key]
            return val
    return ""


def _get_md5_value(d: dict[str, Any]) -> str:
    """Return the MD5 value from *d*, trying both ``"md5"`` and ``"MD5"`` keys.

    Unlike :func:`_pop_md5_key` this does **not** mutate *d*.
    """
    return d.get("md5") or d.get("MD5") or ""


def _streaming_md5(file_path: Path) -> str:
    """Compute MD5 hash of a file using constant memory."""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset_from_folder(
    folder_path: Path,
    media_type: str,
    medias: dict[int, dict[str, Any]],
    content_vectors: dict[str, Any] | None = None,
    content_md5s: dict[str, str] | None = None,
    on_progress: Optional[ProgressCallback] = None,
    origin: dict[str, Any] | None = None,
    thin: bool = False,
    embedder_name: str = "",
    custom_metadata_map: dict[str, dict[str, Any]] | None = None,
    skip_embedding: bool = False,
) -> None:
    """Generate a dataset in-place from a flat folder of media files.

    Scans ``folder_path`` for all files matching the extensions for ``media_type``,
    embeds each file using the appropriate model, and populates ``medias`` with
    the resulting media dicts. Progress is reported via :func:`update_progress`.

    Files whose basename appears in ``content_vectors`` will use the supplied
    embedding instead of running the embedding model.  This allows importers
    that already provide content vectors to avoid redundant computation.

    Similarly, files whose basename appears in ``content_md5s`` will use the
    supplied hash instead of computing it from the file contents.

    The ``medias`` dict is cleared before loading begins.

    ``media_type`` is looked up in the media type registry by
    :attr:`~vtsearch.media.base.MediaType.folder_import_name`.  Adding a
    new media type to the registry automatically makes it available here
    without any changes to this function.

    Args:
        folder_path: Path to the root directory containing media files.
            Subdirectories are scanned recursively.
        media_type: Folder-import alias for the media type (e.g. ``"sounds"``).
        medias: Dict to populate in-place. Existing entries are removed before
            loading. Keys are sequential integer media IDs starting from 1.
        content_vectors: Optional mapping of filename to a pre-computed
            embedding ``numpy.ndarray``.  Keys may be relative paths
            (``"subdir/file.wav"``) or basenames (``"file.wav"``); relative
            paths are checked first for an exact match, then basenames as a
            fallback.
        content_md5s: Optional mapping of filename to a pre-computed MD5 hex
            digest string.  Keys follow the same lookup logic as
            ``content_vectors`` (relative path first, then basename).
        origin: Optional serialised
            :class:`~vtsearch.datasets.origin.Origin` dict to attach to each
            media (as ``media["origin"]``).  When ``None`` no origin is set
            and the caller is expected to set it afterwards.
        thin: When ``True``, store a ``media_path`` reference to the file on
            disk instead of reading all bytes into ``media_bytes``.  This saves
            memory for CLI workflows that only need embeddings for scoring.
            MD5 is still computed via streaming (constant memory).
        embedder_name: Optional name of a registered embedder to use.
            When empty, the first registered embedder for the media type
            is used.
        custom_metadata_map: Optional mapping of filename to a metadata dict.
            Keys follow the same lookup logic as ``content_vectors`` (relative
            path first, then basename).  When a metadata dict contains a
            non-empty ``"md5"`` key, that value is used as the media's MD5
            instead of computing it from the file contents.  The metadata dict
            is also attached to the media as ``custom_metadata``.
        skip_embedding: When ``True``, skip embedder resolution and model
            loading entirely.  Files with pre-computed vectors in
            ``content_vectors`` use those; files without are included with
            ``embedding=None``.  Useful when vectors have already been
            downloaded or computed externally.

    Raises:
        ValueError: If ``media_type`` is not recognised, or if no matching
            files are found in ``folder_path``.
    """
    from vtsearch.media import embedders_for_type, get_by_folder_name, get_embedder

    if on_progress is None:
        on_progress = _default_progress()

    on_progress("loading", "Scanning media files...", 0, 0)

    try:
        mt = get_by_folder_name(media_type)
    except KeyError:
        raise ValueError(f"Invalid media type: {media_type}")

    # Resolve the embedder (skipped entirely when skip_embedding=True).
    emb = None
    if not skip_embedding:
        if embedder_name:
            try:
                emb = get_embedder(embedder_name)
            except KeyError:
                raise ValueError(f"Unknown embedder: {embedder_name}")
        else:
            avail = embedders_for_type(mt.type_id)
            if avail:
                emb = avail[0]

        # Eagerly load models before starting the embedding timer so that
        # download / weight-loading time does not pollute the progress bar.
        if emb is not None and getattr(emb, "_model", None) is None:
            on_progress("loading", "Loading embedding model…", 0, 0)
            original_cb = emb._on_progress
            emb._on_progress = on_progress
            try:
                emb.load_models()
            finally:
                emb._on_progress = original_cb

    # Find all files of the specified media type (recursive so that
    # subdirectory structures are preserved).
    media_files = []
    for ext in mt.file_extensions:
        media_files.extend(rglob_follow_symlinks(folder_path, ext))

    if not media_files:
        raise ValueError(f"No {media_type} files found in folder")

    medias.clear()
    media_id = 1
    total_files = len(media_files)

    try:
        for i, file_path in enumerate(media_files):
            # Preserve relative path from the import root so that files in
            # different subdirectories with the same basename stay distinct.
            rel_path = file_path.relative_to(folder_path).as_posix()

            phase = "loading" if skip_embedding else "embedding"
            on_progress(
                phase,
                f"{'Loading' if skip_embedding else 'Embedding'} {media_type} {rel_path}...",
                i + 1,
                total_files,
            )

            # Look up pre-computed vectors by relative path first, then
            # fall back to basename for backward compatibility.
            if content_vectors and rel_path in content_vectors:
                embedding = content_vectors[rel_path]
            elif content_vectors and file_path.name in content_vectors:
                embedding = content_vectors[file_path.name]
            elif skip_embedding:
                embedding = None
            else:
                if emb is None:
                    continue
                embedding = emb.embed_media(file_path)
                if embedding is None:
                    continue

            embedder_id = emb.name if emb else ""

            # Look up per-file custom metadata (relative path first, then
            # basename fallback — same lookup order as content_vectors).
            file_cm: dict[str, Any] | None = None
            if custom_metadata_map:
                if rel_path in custom_metadata_map:
                    file_cm = custom_metadata_map[rel_path]
                elif file_path.name in custom_metadata_map:
                    file_cm = custom_metadata_map[file_path.name]

            # Resolve MD5: custom_metadata > content_md5s > computed
            cm_md5 = _get_md5_value(file_cm) if file_cm else ""

            if thin:
                # Thin mode: store file path reference, skip loading bytes.
                # Use stat for file_size and streaming hash for MD5.
                if cm_md5:
                    md5 = cm_md5
                elif content_md5s and rel_path in content_md5s:
                    md5 = content_md5s[rel_path]
                elif content_md5s and file_path.name in content_md5s:
                    md5 = content_md5s[file_path.name]
                else:
                    md5 = _streaming_md5(file_path)
                media_data: dict[str, Any] = {
                    "id": media_id,
                    "type": mt.type_id,
                    "embedder": embedder_id,
                    "file_size": file_path.stat().st_size,
                    "md5": md5,
                    "embedding": embedding,
                    "filename": rel_path,
                    "category": "custom",
                    "origin": origin,
                    "origin_name": rel_path,
                    "media_bytes": None,
                    "media_string": None,
                    "media_path": str(file_path.resolve()),
                    "duration": 0,
                }
            else:
                with open(file_path, "rb") as f:
                    file_bytes = f.read()

                if cm_md5:
                    md5 = cm_md5
                elif content_md5s and rel_path in content_md5s:
                    md5 = content_md5s[rel_path]
                elif content_md5s and file_path.name in content_md5s:
                    md5 = content_md5s[file_path.name]
                else:
                    md5 = hashlib.md5(file_bytes).hexdigest()

                # Build the base media dict
                media_data = {
                    "id": media_id,
                    "type": mt.type_id,
                    "embedder": embedder_id,
                    "file_size": len(file_bytes),
                    "md5": md5,
                    "embedding": embedding,
                    "filename": rel_path,
                    "category": "custom",
                    "origin": origin,
                    "origin_name": rel_path,
                    # Null-out optional media fields so medias from different types
                    # stored in the same dict have consistent keys.
                    "media_bytes": None,
                    "media_string": None,
                    "media_path": str(file_path.resolve()),
                    "duration": 0,
                }

                # Merge in media-specific fields from the media type
                media_data.update(mt.load_media_data(file_path))

            if file_cm:
                media_data["custom_metadata"] = file_cm

            medias[media_id] = media_data
            media_id += 1
    except MemoryError:
        medias.clear()
        gc.collect()
        raise MemoryError(
            f"Out of memory after loading {media_id - 1} of {total_files} files. "
            "Try a smaller dataset or free up system RAM."
        )

    on_progress("idle", f"Loaded {len(medias)} {media_type} medias from folder")


def apply_custom_metadata_md5(media_dict: dict[int, dict[str, Any]]) -> int:
    """Use MD5 hashes from custom_metadata when available.

    If a media item's ``custom_metadata`` contains a non-empty ``"md5"`` (or
    ``"MD5"``) key, that value is used as the item's ``"md5"`` instead of
    whatever was calculated during loading.  This lets importers supply
    authoritative hashes from their data source without recalculation.

    Args:
        media_dict: The mutable medias dict.  Modified in place.

    Returns:
        The number of media items whose MD5 was replaced.
    """
    count = 0
    for media in media_dict.values():
        cm = media.get("custom_metadata")
        if not cm:
            continue
        cm_md5 = _pop_md5_key(cm)
        if cm_md5:
            media["md5"] = cm_md5
            count += 1
    return count


def load_dataset_from_folder_chunked(
    folder_path: Path,
    media_type: str,
    chunk_size: int,
    content_vectors: dict[str, Any] | None = None,
    content_md5s: dict[str, str] | None = None,
    on_progress: Optional[ProgressCallback] = None,
    origin: dict[str, Any] | None = None,
    thin: bool = False,
    embedder_name: str = "",
    custom_metadata_map: dict[str, dict[str, Any]] | None = None,
    skip_embedding: bool = False,
) -> Iterator[dict[int, dict[str, Any]]]:
    """Yield chunks of medias from a folder of media files.

    Works identically to :func:`load_dataset_from_folder` but yields the
    medias in groups of at most *chunk_size*.  Each yielded dict is a
    self-contained medias dict with IDs starting at 1.  After the caller
    has processed a chunk, the dict can be discarded to free memory.

    Args:
        folder_path: Path to the root directory containing media files.
            Subdirectories are scanned recursively.
        media_type: Folder-import alias (e.g. ``"sounds"``).
        chunk_size: Maximum number of medias per chunk.
        content_vectors: Optional pre-computed embeddings keyed by filename
            (relative path or basename; relative path is tried first).
        content_md5s: Optional pre-computed MD5s keyed by filename (same
            lookup logic as *content_vectors*).
        origin: Optional origin dict to attach to each media.
        thin: When ``True``, store ``media_path`` instead of ``media_bytes``.
        embedder_name: Optional name of a registered embedder to use.
            When empty, the first registered embedder for the media type
            is used.
        custom_metadata_map: Optional mapping of filename to a metadata dict.
            Same semantics as in :func:`load_dataset_from_folder`.
        skip_embedding: Same semantics as in :func:`load_dataset_from_folder`.

    Yields:
        A dict mapping int media IDs (starting at 1) to media data dicts.
        Each yielded dict contains at most *chunk_size* medias.

    Raises:
        ValueError: If ``media_type`` is not recognised, or if no matching
            files are found in ``folder_path``.
    """
    from vtsearch.media import embedders_for_type, get_by_folder_name, get_embedder

    if on_progress is None:
        on_progress = _default_progress()

    on_progress("loading", "Scanning media files...", 0, 0)

    try:
        mt = get_by_folder_name(media_type)
    except KeyError:
        raise ValueError(f"Invalid media type: {media_type}")

    # Resolve the embedder (skipped entirely when skip_embedding=True).
    emb = None
    if not skip_embedding:
        if embedder_name:
            try:
                emb = get_embedder(embedder_name)
            except KeyError:
                raise ValueError(f"Unknown embedder: {embedder_name}")
        else:
            avail = embedders_for_type(mt.type_id)
            if avail:
                emb = avail[0]

        # Eagerly load models before starting the embedding timer so that
        # download / weight-loading time does not pollute the progress bar.
        if emb is not None and getattr(emb, "_model", None) is None:
            on_progress("loading", "Loading embedding model…", 0, 0)
            original_cb = emb._on_progress
            emb._on_progress = on_progress
            try:
                emb.load_models()
            finally:
                emb._on_progress = original_cb

    # Find all files of the specified media type (recursive so that
    # subdirectory structures are preserved).
    media_files: list[Path] = []
    for ext in mt.file_extensions:
        media_files.extend(rglob_follow_symlinks(folder_path, ext))

    if not media_files:
        raise ValueError(f"No {media_type} files found in folder")

    total_files = len(media_files)
    embedder_id = emb.name if emb else ""

    # Process in groups of chunk_size
    for start in range(0, total_files, chunk_size):
        batch = media_files[start : start + chunk_size]
        chunk_medias: dict[int, dict[str, Any]] = {}
        media_id = 1

        for i, file_path in enumerate(batch):
            global_idx = start + i
            rel_path = file_path.relative_to(folder_path).as_posix()

            phase = "loading" if skip_embedding else "embedding"
            on_progress(
                phase,
                f"{'Loading' if skip_embedding else 'Embedding'} {media_type} {rel_path} (chunk {start // chunk_size + 1})...",
                global_idx + 1,
                total_files,
            )

            if content_vectors and rel_path in content_vectors:
                embedding = content_vectors[rel_path]
            elif content_vectors and file_path.name in content_vectors:
                embedding = content_vectors[file_path.name]
            elif skip_embedding:
                embedding = None
            else:
                if emb is None:
                    continue
                embedding = emb.embed_media(file_path)
                if embedding is None:
                    continue

            # Look up per-file custom metadata (same logic as non-chunked).
            file_cm: dict[str, Any] | None = None
            if custom_metadata_map:
                if rel_path in custom_metadata_map:
                    file_cm = custom_metadata_map[rel_path]
                elif file_path.name in custom_metadata_map:
                    file_cm = custom_metadata_map[file_path.name]

            cm_md5 = _get_md5_value(file_cm) if file_cm else ""

            if thin:
                if cm_md5:
                    md5 = cm_md5
                elif content_md5s and rel_path in content_md5s:
                    md5 = content_md5s[rel_path]
                elif content_md5s and file_path.name in content_md5s:
                    md5 = content_md5s[file_path.name]
                else:
                    md5 = _streaming_md5(file_path)
                media_data: dict[str, Any] = {
                    "id": media_id,
                    "type": mt.type_id,
                    "embedder": embedder_id,
                    "file_size": file_path.stat().st_size,
                    "md5": md5,
                    "embedding": embedding,
                    "filename": rel_path,
                    "category": "custom",
                    "origin": origin,
                    "origin_name": rel_path,
                    "media_bytes": None,
                    "media_string": None,
                    "media_path": str(file_path.resolve()),
                    "duration": 0,
                }
            else:
                with open(file_path, "rb") as f:
                    file_bytes = f.read()

                if cm_md5:
                    md5 = cm_md5
                elif content_md5s and rel_path in content_md5s:
                    md5 = content_md5s[rel_path]
                elif content_md5s and file_path.name in content_md5s:
                    md5 = content_md5s[file_path.name]
                else:
                    md5 = hashlib.md5(file_bytes).hexdigest()

                media_data = {
                    "id": media_id,
                    "type": mt.type_id,
                    "embedder": embedder_id,
                    "file_size": len(file_bytes),
                    "md5": md5,
                    "embedding": embedding,
                    "filename": rel_path,
                    "category": "custom",
                    "origin": origin,
                    "origin_name": rel_path,
                    "media_bytes": None,
                    "media_string": None,
                    "media_path": str(file_path.resolve()),
                    "duration": 0,
                }
                media_data.update(mt.load_media_data(file_path))

            if file_cm:
                media_data["custom_metadata"] = file_cm

            chunk_medias[media_id] = media_data
            media_id += 1

        if chunk_medias:
            yield chunk_medias

    on_progress("idle", f"Finished chunked loading of {total_files} {media_type} files")


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


def embed_image_file_from_pil(image: Image.Image, embedder_name: str = "") -> Optional[np.ndarray]:
    """Generate a CLIP embedding vector for a PIL Image object.

    A convenience wrapper for cases where the image is already in memory
    (e.g. reconstructed from a NumPy array during CIFAR-10 loading).

    Delegates to the image embedder's ``embed_pil_image`` method.

    Args:
        image: A PIL Image in any mode.
        embedder_name: Optional name of a registered embedder.  When empty,
            the first image embedder is used.

    Returns:
        A 1-D ``numpy.ndarray`` of shape ``(embedding_dim,)``, or ``None`` if
        no image embedder is available or an exception occurs.
    """
    from vtsearch.media import embedders_for_type, get_embedder

    if embedder_name:
        emb = get_embedder(embedder_name)
    else:
        avail = embedders_for_type("image")
        if not avail:
            return None
        emb = avail[0]
    return emb.embed_pil_image(image)


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


def _stamp_demo_origin(
    medias: dict[int, dict[str, Any]],
    dataset_name: str,
    converter_name: str = "",
) -> None:
    """Stamp the demo origin on all medias (fresh dict per media).

    Ensures every media has ``origin = {"importer": "demo", "params": {"name": ...}}``.
    """
    demo_origin_params: dict[str, str] = {"name": dataset_name}
    if converter_name:
        demo_origin_params["converter"] = converter_name
    for media in medias.values():
        media["origin"] = {"importer": "demo", "params": dict(demo_origin_params)}


def load_demo_dataset(
    dataset_name: str,
    medias: dict[int, dict[str, Any]],
    on_progress: Optional[ProgressCallback] = None,
    embedder_name: str = "",
    converter_name: str = "",
    clipper_name: str = "",
) -> None:
    """Load a named demo dataset into the medias dict, downloading and embedding as needed.

    Checks for a cached ``.pkl`` file in ``EMBEDDINGS_DIR``; if found, loads
    from that file. If the cache is missing or the media bytes it references can
    no longer be found on disk, the raw data is re-downloaded and re-embedded.

    Each media type implements its own
    :meth:`~vtsearch.media.base.MediaType.load_demo_source` method that
    handles downloading, embedding, and populating clips for its demo sources.
    This function simply orchestrates pickle caching around that delegation.

    When *converter_name* is given (e.g. ``"video2image"``), the demo data is
    loaded using its original media type, then each media is converted via the
    named converter.  The resulting dataset contains the *target* type and
    is cached under a separate pickle key.

    Progress throughout the operation is reported via :func:`update_progress`.

    Args:
        dataset_name: Key into ``DEMO_DATASETS`` identifying which demo dataset
            to load.  Raises ``ValueError`` if the key is not found.
        medias: Dict to populate in-place. Existing entries are removed before
            loading. Keys are integer media IDs; values are media data dicts.
        embedder_name: Optional name of a registered embedder to use.
            When empty, the first registered embedder for the media type
            is used.
        converter_name: Optional name of a converter (e.g. ``"video2image"``).
            When given, the demo is loaded in its native type and then
            converted.
        clipper_name: Optional name of a registered clipper.  Recorded in
            a ``.clipper`` sidecar next to the pickle for status tracking.

    Raises:
        ValueError: If ``dataset_name`` is not in ``DEMO_DATASETS``, or if the
            media type does not support the requested demo source.
    """
    if on_progress is None:
        on_progress = _default_progress()

    if dataset_name not in DEMO_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_info = DEMO_DATASETS[dataset_name]
    media_type_id = dataset_info.get("media_type", "audio")

    # When a converter is specified, use a separate pickle cache key.
    cache_key = f"{dataset_name}__{converter_name}" if converter_name else dataset_name

    # Check if already embedded
    pkl_file = EMBEDDINGS_DIR / f"{cache_key}.pkl"
    if pkl_file.exists():
        on_progress("loading", f"Loading {dataset_name} dataset...", 0, 0)
        load_dataset_from_pickle(pkl_file, medias)

        # Check if any medias were actually loaded
        if len(medias) == 0:
            # Pickle file exists but media files are missing, delete and re-embed
            on_progress("loading", f"Media files missing, re-embedding {dataset_name}...", 0, 0)
            pkl_file.unlink()
            pkl_file.with_suffix(".embedder").unlink(missing_ok=True)
            pkl_file.with_suffix(".clipper").unlink(missing_ok=True)
        else:
            # Stamp demo origin on cached medias so that cross-dataset
            # resolution always has the dataset name in the origin params.
            # Old pickles (created before origin stamping) may have empty
            # params — this ensures they are corrected on load.
            _stamp_demo_origin(medias, dataset_name, converter_name)
            on_progress("idle", f"Loaded {dataset_name} dataset")
            return

    # Resolve the embedder
    from vtsearch.media import embedders_for_type, get as media_get, get_embedder

    embedder = None
    if embedder_name:
        try:
            embedder = get_embedder(embedder_name)
        except KeyError:
            raise ValueError(f"Unknown embedder: {embedder_name}")
    else:
        avail = embedders_for_type(media_type_id)
        if avail:
            embedder = avail[0]

    mt = media_get(media_type_id)

    source = dataset_info.get("source", "")
    categories = dataset_info["categories"]
    slice_start = dataset_info.get("slice_start", 0)
    slice_end = dataset_info.get("slice_end")

    medias.clear()
    external_dir = mt.load_demo_source(
        source=source,
        categories=categories,
        slice_start=slice_start,
        slice_end=slice_end,
        clips=medias,
        on_progress=on_progress,
        embedder=embedder,
    )

    # Stamp the demo origin on all medias
    _stamp_demo_origin(medias, dataset_name, converter_name)

    # --- Apply converter if requested ---
    if converter_name:
        _apply_converter_to_demo(
            converter_name=converter_name,
            dataset_name=dataset_name,
            medias=medias,
            embedder_name=embedder_name,
            on_progress=on_progress,
        )

    # Build the pickle cache payload
    # For types with external media dirs (audio, video), exclude media_bytes
    # from the pickle and store the dir path so reloading can find the files.
    # When a converter was applied, external_dir is no longer relevant (the
    # converted medias carry their own bytes/strings).
    if external_dir is not None and not converter_name:
        pkl_data: dict[str, Any] = {
            "name": dataset_name,
            "medias": {
                cid: {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in media.items() if k not in ("media_bytes", "thumbnail_bytes")}
                for cid, media in medias.items()
            },
            mt.dir_key: external_dir,
        }
    else:
        pkl_data = {
            "name": dataset_name,
            "medias": {
                cid: {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in media.items()}
                for cid, media in medias.items()
            },
        }

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(pkl_file, "wb") as f:
        pickle.dump(pkl_data, f)

    # Write a lightweight sidecar that records which embedder produced this pkl.
    resolved_name = getattr(embedder, "name", "") if embedder is not None else ""
    _write_embedder_sidecar(pkl_file, resolved_name)

    # Write a clipper sidecar so the demo list can check readiness.
    _write_clipper_sidecar(pkl_file, clipper_name)

    on_progress("idle", f"Loaded {dataset_name} dataset")


# Backward-compat alias — canonical location is vtsearch.converters.runner
from vtsearch.converters.runner import apply_converter_to_demo as _apply_converter_to_demo  # noqa: F401, E402


def export_dataset_to_file(
    medias: dict[int, dict[str, Any]],
) -> bytes:
    """Serialise the current media dataset to a pickle-formatted byte string.

    Converts the in-memory ``medias`` dict to a portable format (converting any
    ``numpy.ndarray`` embeddings to plain Python lists) and returns it as bytes
    suitable for writing to a ``.pkl`` file or sending as an HTTP response.

    The resulting bytes can be reloaded with :func:`load_dataset_from_pickle`.

    Args:
        medias: Mapping of media ID to media data dict.

    Returns:
        Raw bytes of the pickled dataset dict.
    """
    data: dict[str, Any] = {
        "medias": {
            cid: {
                "id": media["id"],
                "type": media.get("type", "audio"),
                "duration": media["duration"],
                "file_size": media["file_size"],
                "md5": media["md5"],
                "embedder": media.get("embedder", ""),
                "embedding": media["embedding"].tolist()
                if isinstance(media["embedding"], np.ndarray)
                else media["embedding"],
                "filename": media.get("filename", f"media_{cid}.wav"),
                "category": media.get("category", "unknown"),
                "origin": media.get("origin"),
                "origin_name": media.get("origin_name", media.get("filename", "")),
                "media_bytes": media.get("media_bytes"),
                "media_string": media.get("media_string"),
                "media_path": media.get("media_path"),
                "word_count": media.get("word_count"),
                "character_count": media.get("character_count"),
                "width": media.get("width"),
                "height": media.get("height"),
                "custom_metadata": media.get("custom_metadata"),
            }
            for cid, media in medias.items()
        }
    }

    buf = io.BytesIO()
    pickle.dump(data, buf)
    buf.seek(0)
    return buf.getvalue()
