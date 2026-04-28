"""Folder-based dataset loaders.

Loads datasets from a directory tree of media files, with optional
pre-computed embeddings/MD5s, custom metadata, and chunked iteration.
Split out from :mod:`vtsearch.datasets.loader` for navigability.
"""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path
from typing import Any, Iterator, Optional

from vtsearch.datasets.loader import (
    ProgressCallback,
    _default_progress,
    _get_embedding_value,
    _get_md5_value,
    _pop_md5_key,
    _streaming_md5,
)
from vtsearch.utils.paths import rglob_follow_symlinks


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
        media_type: Media type identifier (e.g. ``"audio"``).
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
            instead of computing it from the file contents.  When it contains
            an ``"embedding"`` key, that value is used as the media's
            embedding vector (highest priority, above ``content_vectors``
            and the embedding model).  The metadata dict is also attached
            to the media as ``custom_metadata``.
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

            # Look up per-file custom metadata (relative path first, then
            # basename fallback — same lookup order as content_vectors).
            file_cm: dict[str, Any] | None = None
            if custom_metadata_map:
                if rel_path in custom_metadata_map:
                    file_cm = custom_metadata_map[rel_path]
                elif file_path.name in custom_metadata_map:
                    file_cm = custom_metadata_map[file_path.name]

            # Resolve embedding: custom_metadata > content_vectors > model
            cm_embedding = _get_embedding_value(file_cm) if file_cm else None
            if cm_embedding is not None:
                embedding = cm_embedding
            elif content_vectors and rel_path in content_vectors:
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
        media_type: Media type identifier (e.g. ``"audio"``).
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

            # Look up per-file custom metadata (same logic as non-chunked).
            file_cm: dict[str, Any] | None = None
            if custom_metadata_map:
                if rel_path in custom_metadata_map:
                    file_cm = custom_metadata_map[rel_path]
                elif file_path.name in custom_metadata_map:
                    file_cm = custom_metadata_map[file_path.name]

            # Resolve embedding: custom_metadata > content_vectors > model
            cm_embedding = _get_embedding_value(file_cm) if file_cm else None
            if cm_embedding is not None:
                embedding = cm_embedding
            elif content_vectors and rel_path in content_vectors:
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
