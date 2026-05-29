"""Folder-based dataset loaders.

Loads datasets from a directory tree of media files, with optional
pre-computed embeddings/MD5s and custom metadata.  Loaders build the
media dicts but do **not** call the embedder - items without a
pre-computed embedding leave ``embedding=None`` for the framework
embed stage (:func:`vtscore.datasets.load_pipeline.embed_missing`) to
fill in.

Split out from :mod:`vtscore.datasets.loader` for navigability.
"""

from __future__ import annotations

import gc
import hashlib
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Optional

from vtscore.datasets.loader import (
    ProgressCallback,
    _default_progress,
    _get_embedding_value,
    _get_md5_value,
    _pop_md5_key,
    _streaming_md5,
)
from vtscore.security.path_validation import glob_top_level, rglob_follow_symlinks

logger = logging.getLogger(__name__)


def _scan_files(folder: Path, pattern: str, recursive: bool) -> list[Path]:
    """Find files in *folder* matching *pattern*, optionally recursing."""
    if recursive:
        return rglob_follow_symlinks(folder, pattern)
    return glob_top_level(folder, pattern)


def _resolve_folder_load_inputs(
    folder_path: Path,
    media_type: str,
    recursive: bool,
    content_vectors: dict[str, Any] | None,
    content_md5s: dict[str, str] | None,
    custom_metadata_map: dict[str, dict[str, Any]] | None,
) -> tuple[Any, list[Path], int]:
    """Shared front-matter for the folder loaders.

    Resolves the media type, scans the folder, and validates the
    override maps against the scanned file list.  Returns
    ``(mt, media_files, total_files)``.  Raises ``ValueError`` for
    unknown media types, empty folders, or ambiguous basename keys in
    any override map.
    """
    from vtscore.media import get_by_folder_name  # noqa: PLC0415

    try:
        mt = get_by_folder_name(media_type)
    except KeyError:
        raise ValueError(f"Invalid media type: {media_type}")

    media_files: list[Path] = []
    for ext in mt.file_extensions:
        media_files.extend(_scan_files(folder_path, ext, recursive))

    if not media_files:
        raise ValueError(f"No {media_type} files found in folder")

    _validate_override_keys(
        media_files,
        folder_path,
        content_vectors,
        content_md5s,
        custom_metadata_map,
    )

    return mt, media_files, len(media_files)


def _embeddings_equal(a: Any, b: Any) -> bool:
    """Return True if two override-map values represent the same embedding.

    Numpy arrays compare via ``np.array_equal`` (shape + element equality,
    no NaN-vs-NaN special case).  Anything else uses ``==``; a falsy
    result or a ``ValueError`` from the comparison means "different".
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        import numpy as np  # noqa: PLC0415

        return bool(np.array_equal(a, b))
    except Exception:
        try:
            return bool(a == b)
        except Exception:
            return False


def _validate_override_keys(
    media_files: list[Path],
    folder_path: Path,
    content_vectors: dict[str, Any] | None,
    content_md5s: dict[str, str] | None,
    custom_metadata_map: dict[str, dict[str, Any]] | None,
) -> None:
    """Audit the override maps for ambiguity before the loader uses them.

    Two failure modes are covered:

    1. **Per-file rel_path vs basename conflict.**  If a file's relative
       path *and* its basename are both present as keys in the same
       override map with **different** values, the resolution chain
       silently picks the rel_path entry.  Log a warning so the caller
       knows their data is inconsistent.

    2. **Ambiguous basename key spanning multiple files.**  When several
       files share a basename (typical for recursive imports with files
       like ``class_a/foo.wav`` and ``class_b/foo.wav``) *and* that
       basename appears in an override map *without* a rel_path entry
       for every affected file, the basename fallback would silently
       assign the same vector/md5/metadata to all of them.  That is
       genuinely wrong data, so raise ``ValueError``.

    The validation only warns when there is at least one true conflict;
    it does not flag redundant keys that resolve to the same value, and
    it does not flag unused keys (e.g. an NPZ that ships more entries
    than the folder contains).
    """
    rel_paths_by_basename: dict[str, list[str]] = defaultdict(list)
    for file_path in media_files:
        rel_path = file_path.relative_to(folder_path).as_posix()
        rel_paths_by_basename[file_path.name].append(rel_path)

    _check_per_file_conflicts(
        rel_paths_by_basename,
        content_vectors,
        content_md5s,
        custom_metadata_map,
    )
    _check_ambiguous_basename_keys(
        rel_paths_by_basename,
        content_vectors,
        content_md5s,
        custom_metadata_map,
    )


def _check_per_file_conflicts(
    rel_paths_by_basename: dict[str, list[str]],
    content_vectors: dict[str, Any] | None,
    content_md5s: dict[str, str] | None,
    custom_metadata_map: dict[str, dict[str, Any]] | None,
) -> None:
    """Log a warning when a file's rel_path and basename keys disagree."""
    for basename, rel_paths in rel_paths_by_basename.items():
        for rel_path in rel_paths:
            if rel_path == basename:
                continue
            _warn_if_content_vectors_conflict(rel_path, basename, content_vectors)
            _warn_if_content_md5s_conflict(rel_path, basename, content_md5s)
            _warn_if_custom_metadata_conflict(rel_path, basename, custom_metadata_map)


def _warn_if_content_vectors_conflict(
    rel_path: str,
    basename: str,
    content_vectors: dict[str, Any] | None,
) -> None:
    if not content_vectors or rel_path not in content_vectors or basename not in content_vectors:
        return
    if _embeddings_equal(content_vectors[rel_path], content_vectors[basename]):
        return
    logger.warning(
        "content_vectors has conflicting entries for %r and %r; using the relative-path entry.",
        rel_path,
        basename,
    )


def _warn_if_content_md5s_conflict(
    rel_path: str,
    basename: str,
    content_md5s: dict[str, str] | None,
) -> None:
    if not content_md5s or rel_path not in content_md5s or basename not in content_md5s:
        return
    if content_md5s[rel_path] == content_md5s[basename]:
        return
    logger.warning(
        "content_md5s has conflicting entries for %r and %r (%r vs %r); using the relative-path entry.",
        rel_path,
        basename,
        content_md5s[rel_path],
        content_md5s[basename],
    )


def _warn_if_custom_metadata_conflict(
    rel_path: str,
    basename: str,
    custom_metadata_map: dict[str, dict[str, Any]] | None,
) -> None:
    if not custom_metadata_map or rel_path not in custom_metadata_map or basename not in custom_metadata_map:
        return
    rp_cm = custom_metadata_map[rel_path] or {}
    bn_cm = custom_metadata_map[basename] or {}
    rp_emb = _get_embedding_value(rp_cm)
    bn_emb = _get_embedding_value(bn_cm)
    if not _embeddings_equal(rp_emb, bn_emb):
        logger.warning(
            "custom_metadata_map has conflicting embeddings for %r and %r; using the relative-path entry.",
            rel_path,
            basename,
        )
    rp_md5 = _get_md5_value(rp_cm)
    bn_md5 = _get_md5_value(bn_cm)
    if rp_md5 and bn_md5 and rp_md5 != bn_md5:
        logger.warning(
            "custom_metadata_map has conflicting md5s for %r and %r (%r vs %r); using the relative-path entry.",
            rel_path,
            basename,
            rp_md5,
            bn_md5,
        )


def _check_ambiguous_basename_keys(
    rel_paths_by_basename: dict[str, list[str]],
    content_vectors: dict[str, Any] | None,
    content_md5s: dict[str, str] | None,
    custom_metadata_map: dict[str, dict[str, Any]] | None,
) -> None:
    """Raise when a basename override would silently fan out to multiple files."""
    for basename, rel_paths in rel_paths_by_basename.items():
        if len(rel_paths) < 2:
            continue
        for label, override_map in (
            ("content_vectors", content_vectors),
            ("content_md5s", content_md5s),
            ("custom_metadata_map", custom_metadata_map),
        ):
            if not override_map or basename not in override_map:
                continue
            unresolved = [rp for rp in rel_paths if rp != basename and rp not in override_map]
            if not unresolved:
                continue
            raise ValueError(
                f"{label} has a bare-basename key {basename!r} that would be applied to "
                f"multiple files via the basename fallback ({', '.join(sorted(unresolved))}). "
                "Use the full relative path for each file (e.g. 'subdir/file.ext') "
                "to disambiguate."
            )


def _lookup_file_custom_metadata(
    rel_path: str,
    file_name: str,
    custom_metadata_map: dict[str, dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Return the custom-metadata dict for a file (relative path first, then basename)."""
    if not custom_metadata_map:
        return None
    return custom_metadata_map.get(rel_path) or custom_metadata_map.get(file_name)


def _resolve_file_embedding(
    rel_path: str,
    file_name: str,
    file_cm: dict[str, Any] | None,
    content_vectors: dict[str, Any] | None,
    content_embedder_name: str = "",
) -> tuple[Any, str]:
    """Pick a pre-computed embedding for *file_path* if available.

    Resolution order: custom_metadata embedding → content_vectors[rel_path]
    → content_vectors[basename] → ``(None, "")``.  Custom-metadata vectors
    come back with ``embedder_id == ""``.  Content-vectors hits use
    *content_embedder_name* so the embedder that produced the NPZ archive
    is recorded on the media; ``""`` is returned when the caller doesn't
    know the embedder (the framework embed stage will stamp its own name
    when embedding is ``None``).
    """
    cm_embedding = _get_embedding_value(file_cm) if file_cm else None
    if cm_embedding is not None:
        return cm_embedding, ""
    if content_vectors and rel_path in content_vectors:
        return content_vectors[rel_path], content_embedder_name
    if content_vectors and file_name in content_vectors:
        return content_vectors[file_name], content_embedder_name
    return None, ""


def _resolve_file_md5(
    file_path: Path,
    rel_path: str,
    cm_md5: str,
    content_md5s: dict[str, str] | None,
    file_bytes: bytes | None,
) -> str:
    """Resolve a file's MD5 with the standard precedence.

    custom_metadata MD5 → content_md5s[rel_path] → content_md5s[basename]
    → streaming hash (thin mode, ``file_bytes`` is None) → ``hashlib.md5``
    over ``file_bytes`` (full mode).
    """
    if cm_md5:
        return cm_md5
    if content_md5s and rel_path in content_md5s:
        return content_md5s[rel_path]
    if content_md5s and file_path.name in content_md5s:
        return content_md5s[file_path.name]
    if file_bytes is None:
        return _streaming_md5(file_path)
    return hashlib.md5(file_bytes).hexdigest()


def _build_folder_media_data(
    media_id: int,
    type_id: str,
    embedder_id: str,
    embedding: Any,
    md5: str,
    rel_path: str,
    file_path: Path,
    file_size: int,
    origin: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the per-file media dict shared by the thin and full branches.

    Type-specific fields (and ``media_bytes`` for full mode) are merged
    by the caller after this base is constructed.

    The *origin* dict is copied per-media so siblings never share the
    same nested ``params`` dict - a later mutation on one media would
    otherwise propagate to every other media built from the same call,
    and that aliasing also survives pickle round-trips via backrefs.
    """
    media_origin: dict[str, Any] | None
    if origin is None:
        media_origin = None
    else:
        media_origin = {
            "importer": origin.get("importer", ""),
            "params": dict(origin.get("params", {})),
        }
    return {
        "id": media_id,
        "media_type": type_id,
        "embedder": embedder_id,
        "file_size": file_size,
        "md5": md5,
        "embedding": embedding,
        "filename": rel_path,
        "category": "custom",
        "origin": media_origin,
        "origin_name": rel_path,
        "media_bytes": None,
        "media_string": None,
        "media_path": str(file_path.resolve()),
        "duration": 0,
    }


def _emit_per_file_progress(
    on_progress: ProgressCallback,
    media_type: str,
    rel_path: str,
    current: int,
    total: int,
    chunk_label: str,
) -> None:
    """Emit a per-file progress update (folder loaders)."""
    msg = f"Loading {media_type} {rel_path}{chunk_label}..."
    on_progress("loading", msg, current, total)


def _build_per_file_media(
    *,
    media_id: int,
    file_path: Path,
    rel_path: str,
    mt: Any,
    content_vectors: dict[str, Any] | None,
    content_md5s: dict[str, str] | None,
    custom_metadata_map: dict[str, dict[str, Any]] | None,
    thin: bool,
    origin: dict[str, Any] | None,
    content_embedder_name: str = "",
) -> dict[str, Any]:
    """Resolve any pre-computed embedding + md5 and build the per-file media dict.

    Files without a pre-computed embedding are returned with
    ``embedding=None``; the framework embed stage fills those in.
    """
    file_cm = _lookup_file_custom_metadata(rel_path, file_path.name, custom_metadata_map)

    embedding, embedder_id = _resolve_file_embedding(
        rel_path, file_path.name, file_cm, content_vectors, content_embedder_name
    )

    cm_md5 = _get_md5_value(file_cm) if file_cm else ""

    if thin:
        md5 = _resolve_file_md5(file_path, rel_path, cm_md5, content_md5s, file_bytes=None)
        media_data = _build_folder_media_data(
            media_id,
            mt.type_id,
            embedder_id,
            embedding,
            md5,
            rel_path,
            file_path,
            file_path.stat().st_size,
            origin,
        )
    else:
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        md5 = _resolve_file_md5(file_path, rel_path, cm_md5, content_md5s, file_bytes)
        media_data = _build_folder_media_data(
            media_id,
            mt.type_id,
            embedder_id,
            embedding,
            md5,
            rel_path,
            file_path,
            len(file_bytes),
            origin,
        )
        # Merge in media-specific fields without re-reading the file.
        media_data.update(mt.load_media_data(file_path, media_bytes=file_bytes))

    if file_cm:
        media_data["custom_metadata"] = file_cm

    return media_data


# ---------------------------------------------------------------------------
# Public folder loaders
# ---------------------------------------------------------------------------


def load_dataset_from_folder(
    folder_path: Path,
    media_type: str,
    medias: dict[int, dict[str, Any]],
    content_vectors: dict[str, Any] | None = None,
    content_md5s: dict[str, str] | None = None,
    on_progress: Optional[ProgressCallback] = None,
    origin: dict[str, Any] | None = None,
    thin: bool = False,
    custom_metadata_map: dict[str, dict[str, Any]] | None = None,
    recursive: bool = True,
    content_embedder_name: str = "",
) -> None:
    """Generate a dataset in-place from a flat folder of media files.

    Scans ``folder_path`` for all files matching the extensions for
    ``media_type`` and populates ``medias`` with one media dict per
    file.  Loaders do **not** call the embedder - items leave with
    ``embedding=None`` unless a pre-computed vector is supplied through
    ``content_vectors`` or ``custom_metadata_map``.  The framework
    embed stage (:func:`vtscore.datasets.load_pipeline.embed_missing`)
    fills the rest in after the importer returns.

    The ``medias`` dict is cleared before loading begins.

    ``media_type`` is looked up in the media type registry by
    :attr:`~vtscore.media.base.MediaType.folder_import_name`.

    Args:
        folder_path: Path to the root directory containing media files.
        media_type: Media type identifier (e.g. ``"audio"``).
        medias: Dict to populate in-place. Existing entries are removed
            before loading. Keys are sequential integer media IDs
            starting from 1.
        content_vectors: Optional mapping of filename to a pre-computed
            embedding ``numpy.ndarray``.  Keys may be relative paths
            (``"subdir/file.wav"``) or basenames (``"file.wav"``);
            relative paths are checked first.  When a bare basename
            would match more than one file in the folder without a
            disambiguating relative-path entry, the load fails.
        content_md5s: Optional mapping of filename to a pre-computed
            MD5 hex digest string.  Same lookup logic as
            ``content_vectors``.
        origin: Optional serialised
            :class:`~vtscore.datasets.origin.Origin` dict.
        thin: When ``True``, store ``media_path`` instead of reading
            all bytes into ``media_bytes``.  MD5 is computed via
            streaming.
        custom_metadata_map: Optional mapping of filename to a metadata
            dict.  Same lookup logic as ``content_vectors``.  An
            ``"md5"`` key overrides the computed hash; an
            ``"embedding"`` key supplies a pre-computed vector
            (highest priority).  The dict is attached as
            ``media["custom_metadata"]``.
        recursive: When ``True`` (default), scan subdirectories.
        content_embedder_name: Name of the embedder that produced the
            vectors in *content_vectors*.  Stored as ``media["embedder"]``
            for every file whose vector comes from *content_vectors*.

    Raises:
        ValueError: If ``media_type`` is not recognised, if no matching
            files are found in ``folder_path``, or if an override map
            has a bare-basename key that would silently fan out to
            multiple files in the folder.
    """
    if on_progress is None:
        on_progress = _default_progress()

    on_progress("loading", "Scanning media files...", 0, 0)

    mt, media_files, total_files = _resolve_folder_load_inputs(
        folder_path,
        media_type,
        recursive,
        content_vectors,
        content_md5s,
        custom_metadata_map,
    )

    medias.clear()
    media_id = 1
    _progress_interval = max(1, min(50, total_files // 50)) if total_files > 0 else 1

    try:
        for i, file_path in enumerate(media_files):
            rel_path = file_path.relative_to(folder_path).as_posix()

            if i % _progress_interval == 0 or i + 1 == total_files:
                _emit_per_file_progress(
                    on_progress,
                    media_type,
                    rel_path,
                    i + 1,
                    total_files,
                    chunk_label="",
                )

            built = _build_per_file_media(
                media_id=media_id,
                file_path=file_path,
                rel_path=rel_path,
                mt=mt,
                content_vectors=content_vectors,
                content_md5s=content_md5s,
                custom_metadata_map=custom_metadata_map,
                thin=thin,
                origin=origin,
                content_embedder_name=content_embedder_name,
            )
            medias[media_id] = built
            media_id += 1
    except MemoryError:
        medias.clear()
        gc.collect()
        raise MemoryError(
            f"Out of memory after loading {media_id - 1} of {total_files} files. "
            "Try a smaller dataset or free up system RAM."
        )

    on_progress("idle", f"Loaded {len(medias)} {media_type} medias from folder", 0, 0)


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
    custom_metadata_map: dict[str, dict[str, Any]] | None = None,
    recursive: bool = True,
    content_embedder_name: str = "",
) -> Iterator[dict[int, dict[str, Any]]]:
    """Yield chunks of medias from a folder of media files.

    Works identically to :func:`load_dataset_from_folder` but yields the
    medias in groups of at most *chunk_size*.  Each yielded dict is
    self-contained with IDs starting at 1.  Embedding is left to the
    framework embed stage; items without a pre-computed vector come out
    with ``embedding=None``.

    Raises:
        ValueError: If ``media_type`` is not recognised, if no matching
            files are found in ``folder_path``, or if an override map
            has a bare-basename key that would silently fan out.
    """
    if on_progress is None:
        on_progress = _default_progress()

    on_progress("loading", "Scanning media files...", 0, 0)

    mt, media_files, total_files = _resolve_folder_load_inputs(
        folder_path,
        media_type,
        recursive,
        content_vectors,
        content_md5s,
        custom_metadata_map,
    )

    _progress_interval = max(1, min(50, total_files // 50)) if total_files > 0 else 1

    for start in range(0, total_files, chunk_size):
        batch = media_files[start : start + chunk_size]
        chunk_medias: dict[int, dict[str, Any]] = {}
        media_id = 1

        chunk_label = f" (chunk {start // chunk_size + 1})"

        for i, file_path in enumerate(batch):
            global_idx = start + i
            rel_path = file_path.relative_to(folder_path).as_posix()

            if global_idx % _progress_interval == 0 or global_idx + 1 == total_files:
                _emit_per_file_progress(
                    on_progress,
                    media_type,
                    rel_path,
                    global_idx + 1,
                    total_files,
                    chunk_label=chunk_label,
                )

            built = _build_per_file_media(
                media_id=media_id,
                file_path=file_path,
                rel_path=rel_path,
                mt=mt,
                content_vectors=content_vectors,
                content_md5s=content_md5s,
                custom_metadata_map=custom_metadata_map,
                thin=thin,
                origin=origin,
                content_embedder_name=content_embedder_name,
            )
            chunk_medias[media_id] = built
            media_id += 1

        if chunk_medias:
            yield chunk_medias

    on_progress("idle", f"Finished chunked loading of {total_files} {media_type} files", 0, 0)
