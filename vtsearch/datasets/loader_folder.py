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
from vtsearch.security.path_validation import glob_top_level, rglob_follow_symlinks


def _scan_files(folder: Path, pattern: str, recursive: bool) -> list[Path]:
    """Find files in *folder* matching *pattern*, optionally recursing."""
    if recursive:
        return rglob_follow_symlinks(folder, pattern)
    return glob_top_level(folder, pattern)


# ---------------------------------------------------------------------------
# Bulk-embedding helpers used by the folder loaders below.
# ---------------------------------------------------------------------------


def _has_override(
    rel_path: str,
    file_name: str,
    content_vectors: dict[str, Any] | None,
    custom_metadata_map: dict[str, dict[str, Any]] | None,
) -> bool:
    """Return True if the file's embedding is already resolved by overrides.

    An override is either a ``custom_metadata`` entry with an ``"embedding"``
    key or a ``content_vectors`` entry.  Files with overrides do not need to
    be sent to the embedding model.
    """
    if custom_metadata_map:
        cm = custom_metadata_map.get(rel_path) or custom_metadata_map.get(file_name)
        if cm and _get_embedding_value(cm) is not None:
            return True
    if content_vectors and (rel_path in content_vectors or file_name in content_vectors):
        return True
    return False


def _make_embed_input(
    file_path: Path,
    folder_path: Path,
    origin: dict[str, Any] | None,
    custom_metadata_map: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    """Build a minimal media dict suitable for :meth:`MediaEmbedder.embed_media`.

    File-based embedders pull ``media["media_path"]`` from this dict;
    service-based embedders can use ``media["origin"]``, ``media["origin_name"]``,
    and ``media.get("custom_metadata")`` to resolve the content without
    touching local disk.  The full media dict (with bytes, md5, duration,
    type-specific fields) is constructed after the embedding succeeds.
    """
    rel_path = file_path.relative_to(folder_path).as_posix()
    file_cm: dict[str, Any] | None = None
    if custom_metadata_map:
        file_cm = custom_metadata_map.get(rel_path) or custom_metadata_map.get(file_path.name)
    return {
        "media_path": str(file_path.resolve()),
        "origin": origin,
        "origin_name": rel_path,
        "filename": rel_path,
        "custom_metadata": file_cm,
    }


def _bulk_embed_files(
    emb: Any,
    media_files: list[Path],
    folder_path: Path,
    content_vectors: dict[str, Any] | None,
    custom_metadata_map: dict[str, dict[str, Any]] | None,
    on_progress: ProgressCallback,
    media_type: str,
    origin: dict[str, Any] | None = None,
) -> dict[Path, Any]:
    """Pre-compute embeddings for *media_files* via :meth:`embed_media_bulk`.

    Files already resolved by ``custom_metadata`` or ``content_vectors``
    are skipped; the rest are packaged into minimal media dicts (via
    :func:`_make_embed_input`) and handed to the embedder in a single
    call.  The embedder's ``_on_progress`` callback is routed through
    *on_progress* for the duration of the call so progress updates
    (whether from the default per-item loop or a subclass's custom
    batching) reach the UI.

    Returns a dict mapping ``Path`` → embedding vector.  Paths whose
    bulk call returned ``None`` are omitted, matching the per-file
    behaviour of skipping files that fail to embed.
    """
    pending_paths: list[Path] = []
    pending_medias: list[dict[str, Any]] = []
    for file_path in media_files:
        rel_path = file_path.relative_to(folder_path).as_posix()
        if _has_override(rel_path, file_path.name, content_vectors, custom_metadata_map):
            continue
        pending_paths.append(file_path)
        pending_medias.append(_make_embed_input(file_path, folder_path, origin, custom_metadata_map))

    if not pending_paths:
        return {}

    on_progress("embedding", f"Embedding {len(pending_paths)} {media_type} files...", 0, len(pending_paths))

    original_cb = emb._on_progress
    emb._on_progress = on_progress
    try:
        vectors = emb.embed_media_bulk(pending_medias)
    finally:
        emb._on_progress = original_cb

    return {fp: vec for fp, vec in zip(pending_paths, vectors) if vec is not None}


def _bulk_patch_forward_files(
    emb: Any,
    media_files: list[Path],
    folder_path: Path,
    on_progress: ProgressCallback,
    media_type: str,
    origin: dict[str, Any] | None = None,
    custom_metadata_map: dict[str, dict[str, Any]] | None = None,
) -> dict[Path, Any]:
    """Run :meth:`MediaEmbedder.patch_forward_bulk` on every file in *media_files*.

    Only called when the active embedder reports
    ``supports_patch_regions == True``.  Returns a ``Path → PatchEmbedOutput``
    mapping; files whose patch forward returned ``None`` are omitted.

    The embedder may batch the forward internally (DINOv2 / DINOv3 / EUPE
    patch variants do) or fall back to per-item via the default loop
    contract on :class:`MediaEmbedder`.
    """
    pending_paths: list[Path] = []
    pending_medias: list[dict[str, Any]] = []
    for file_path in media_files:
        pending_paths.append(file_path)
        pending_medias.append(_make_embed_input(file_path, folder_path, origin, custom_metadata_map))

    if not pending_paths:
        return {}

    on_progress("embedding", f"Patch-embedding {len(pending_paths)} {media_type} files...", 0, len(pending_paths))

    original_cb = emb._on_progress
    emb._on_progress = on_progress
    try:
        outputs = emb.patch_forward_bulk(pending_medias)
    finally:
        emb._on_progress = original_cb

    return {fp: out for fp, out in zip(pending_paths, outputs) if out is not None}


def _scan_media_files(folder_path: Path, mt: Any, recursive: bool) -> list[Path]:
    """Collect every file under *folder_path* matching *mt*'s file extensions."""
    media_files: list[Path] = []
    for ext in mt.file_extensions:
        media_files.extend(_scan_files(folder_path, ext, recursive))
    return media_files


def _report_per_file_progress(
    on_progress: ProgressCallback,
    i: int,
    total_files: int,
    progress_interval: int,
    media_type: str,
    rel_path: str,
    skip_embedding: bool,
    *,
    chunk_suffix: str = "",
) -> None:
    """Emit a per-file progress callback every *progress_interval* items.

    The phase string switches between ``"loading"`` and ``"embedding"``
    depending on *skip_embedding*; *chunk_suffix* lets the chunked loader
    append a ``" (chunk N)"`` tag.
    """
    if i % progress_interval != 0 and i + 1 != total_files:
        return
    phase = "loading" if skip_embedding else "embedding"
    verb = "Loading" if skip_embedding else "Embedding"
    on_progress(phase, f"{verb} {media_type} {rel_path}{chunk_suffix}...", i + 1, total_files)


def _lookup_with_rel_or_basename(mapping: dict[str, Any] | None, rel_path: str, file_name: str) -> Any | None:
    """Look up *rel_path* in *mapping* first, then *file_name*; ``None`` if absent.

    Used for ``content_vectors`` / ``content_md5s`` / ``custom_metadata_map``,
    which all share the same "relative path first, basename fallback" lookup
    contract.
    """
    if not mapping:
        return None
    if rel_path in mapping:
        return mapping[rel_path]
    if file_name in mapping:
        return mapping[file_name]
    return None


def _resolve_file_embedding(
    rel_path: str,
    file_path: Path,
    file_cm: dict[str, Any] | None,
    content_vectors: dict[str, Any] | None,
    bulk_embeddings: dict[Path, Any],
    emb: Any,
    skip_embedding: bool,
) -> tuple[Any, str] | None:
    """Resolve the embedding for one file.

    Order of precedence (matches the legacy in-line staircase):
        1. ``custom_metadata`` ``"embedding"`` (externally supplied — ``embedder_id`` blank)
        2. ``content_vectors`` entry (externally supplied — ``embedder_id`` blank)
        3. ``skip_embedding`` mode (no embedding at all — ``embedder_id`` blank)
        4. Vector pre-computed by the bulk embed pass (``embedder_id`` = ``emb.name``)

    Returns ``(embedding, embedder_id)`` or ``None`` to signal "skip this file"
    (the embedder is missing or the bulk pass dropped it).
    """
    cm_embedding = _get_embedding_value(file_cm) if file_cm else None
    if cm_embedding is not None:
        return cm_embedding, ""

    cv = _lookup_with_rel_or_basename(content_vectors, rel_path, file_path.name)
    if cv is not None:
        return cv, ""

    if skip_embedding:
        return None, ""

    if emb is None:
        return None
    embedding = bulk_embeddings.get(file_path)
    if embedding is None:
        return None
    return embedding, emb.name


def _resolve_file_md5(
    file_cm: dict[str, Any] | None,
    content_md5s: dict[str, str] | None,
    rel_path: str,
    file_name: str,
    compute_fallback: Any,
) -> str:
    """Resolve the MD5 for one file.

    Order of precedence: ``custom_metadata`` MD5 → ``content_md5s`` (rel-path
    or basename) → caller-supplied *compute_fallback* (zero-arg callable).
    """
    cm_md5 = _get_md5_value(file_cm) if file_cm else ""
    if cm_md5:
        return cm_md5
    cs = _lookup_with_rel_or_basename(content_md5s, rel_path, file_name)
    if cs is not None:
        return cs
    return compute_fallback()


def _resolve_target_and_embedder(media_type: str, embedder_name: str, skip_embedding: bool) -> tuple[Any, Any]:
    """Resolve the :class:`MediaType` and embedder for a folder load.

    Raises :class:`ValueError` on unknown media type or embedder.
    """
    from vtsearch.media import embedders_for_type, get_by_folder_name, get_embedder

    try:
        mt = get_by_folder_name(media_type)
    except KeyError:
        raise ValueError(f"Invalid media type: {media_type}")

    if skip_embedding:
        return mt, None
    if embedder_name:
        try:
            return mt, get_embedder(embedder_name)
        except KeyError:
            raise ValueError(f"Unknown embedder: {embedder_name}")
    avail = embedders_for_type(mt.type_id)
    return mt, (avail[0] if avail else None)


def _maybe_eager_load_model(
    emb: Any,
    media_files: list[Path],
    folder_path: Path,
    content_vectors: dict[str, Any] | None,
    custom_metadata_map: dict[str, dict[str, Any]] | None,
    on_progress: ProgressCallback,
    skip_embedding: bool,
) -> None:
    """Eagerly load *emb*'s weights when any file actually needs them.

    Skipping the load when every file has an override (custom_metadata
    embedding or content_vectors entry) keeps NPZ-only imports off the
    weight-loading critical path.  The load happens before any progress
    timer starts so download / weight-loading does not pollute the bar.
    """
    if skip_embedding or emb is None or getattr(emb, "_model", None) is not None:
        return
    needs_model = any(
        not _has_override(
            fp.relative_to(folder_path).as_posix(),
            fp.name,
            content_vectors,
            custom_metadata_map,
        )
        for fp in media_files
    )
    if not needs_model:
        return

    on_progress("loading", "Loading embedding model…", 0, 0)
    original_cb = emb._on_progress
    emb._on_progress = on_progress
    try:
        emb.load_models()
    finally:
        emb._on_progress = original_cb


def _run_bulk_passes(
    emb: Any,
    media_files: list[Path],
    folder_path: Path,
    content_vectors: dict[str, Any] | None,
    custom_metadata_map: dict[str, dict[str, Any]] | None,
    on_progress: ProgressCallback,
    media_type: str,
    origin: dict[str, Any] | None,
    skip_embedding: bool,
) -> tuple[dict[Path, Any], dict[Path, Any]]:
    """Pre-compute embeddings (and patch outputs if supported) for *media_files*.

    Returns ``(bulk_embeddings, bulk_patch_outputs)`` — both empty when
    *emb* is ``None`` or *skip_embedding* is true.
    """
    if emb is None or skip_embedding:
        return {}, {}

    bulk_embeddings = _bulk_embed_files(
        emb, media_files, folder_path, content_vectors, custom_metadata_map, on_progress, media_type, origin=origin
    )
    if getattr(emb, "supports_patch_regions", False) is True:
        bulk_patch_outputs = _bulk_patch_forward_files(
            emb,
            media_files,
            folder_path,
            on_progress,
            media_type,
            origin=origin,
            custom_metadata_map=custom_metadata_map,
        )
    else:
        bulk_patch_outputs = {}
    return bulk_embeddings, bulk_patch_outputs


def _build_media_for_file(
    media_id: int,
    file_path: Path,
    rel_path: str,
    folder_path: Path,
    mt: Any,
    origin: dict[str, Any] | None,
    thin: bool,
    content_vectors: dict[str, Any] | None,
    content_md5s: dict[str, str] | None,
    custom_metadata_map: dict[str, dict[str, Any]] | None,
    emb: Any,
    bulk_embeddings: dict[Path, Any],
    bulk_patch_outputs: dict[Path, Any],
    skip_embedding: bool,
) -> dict[str, Any] | None:
    """Construct one media dict, or ``None`` to skip the file.

    ``None`` matches the legacy behaviour: when no embedder is configured
    and no override exists, or when the bulk pass dropped this file, the
    legacy loops did ``continue`` and the file was excluded from the result.
    """
    file_cm = _lookup_with_rel_or_basename(custom_metadata_map, rel_path, file_path.name)

    resolved = _resolve_file_embedding(
        rel_path, file_path, file_cm, content_vectors, bulk_embeddings, emb, skip_embedding
    )
    if resolved is None:
        return None
    embedding, embedder_id = resolved

    if thin:
        md5 = _resolve_file_md5(file_cm, content_md5s, rel_path, file_path.name, lambda: _streaming_md5(file_path))
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
        md5 = _resolve_file_md5(
            file_cm, content_md5s, rel_path, file_path.name, lambda: hashlib.md5(file_bytes).hexdigest()
        )
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
        media_data.update(mt.load_media_data(file_path, media_bytes=file_bytes))

    if file_cm:
        media_data["custom_metadata"] = file_cm

    patch_out = bulk_patch_outputs.get(file_path)
    if patch_out is not None:
        _attach_patch_regions(media_data, patch_out)

    return media_data


def _attach_patch_regions(media_data: dict[str, Any], patch_out: Any) -> None:
    """Build the HAC region tree from *patch_out* and attach it to *media_data*.

    Converts the float32 region vectors and patch grid to **float16**
    for pickling — vectors are rehydrated to float32 on read by callers
    that score them.  Sets ``media_data["patch_regions"]`` (the
    ``2K - 1 + 1`` region nodes including the CLS full-image node) and
    ``media_data["patch_grid"]`` (the raw H × W × 768 patch tokens
    needed by phase-2 region voting).

    Both ``K`` and the HAC affinity ``alpha`` are pinned to the design
    doc's defaults; the caltech101_s sweep is the prescribed way to
    revisit them.
    """
    import numpy as np  # noqa: PLC0415

    from vtsearch.media.patch_embed import build_region_tree, to_fp16  # noqa: PLC0415

    regions = build_region_tree(patch_out, k=12, alpha=0.5)
    media_data["patch_regions"] = to_fp16(regions)
    media_data["patch_grid"] = patch_out.patch_grid.astype(np.float16, copy=False)


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
    embedder_name: str = "",
    custom_metadata_map: dict[str, dict[str, Any]] | None = None,
    skip_embedding: bool = False,
    recursive: bool = True,
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
    if on_progress is None:
        on_progress = _default_progress()
    on_progress("loading", "Scanning media files...", 0, 0)

    mt, emb = _resolve_target_and_embedder(media_type, embedder_name, skip_embedding)

    media_files = _scan_media_files(folder_path, mt, recursive)
    if not media_files:
        raise ValueError(f"No {media_type} files found in folder")

    _maybe_eager_load_model(
        emb, media_files, folder_path, content_vectors, custom_metadata_map, on_progress, skip_embedding
    )

    medias.clear()
    total_files = len(media_files)
    # Report progress every ~2% or at most every 50 items (mirrors
    # loader_pickle.py).  At 100k files this is ~2k callback invocations
    # instead of 100k.
    progress_interval = max(1, min(50, total_files // 50)) if total_files > 0 else 1

    media_id = 1
    try:
        bulk_embeddings, bulk_patch_outputs = _run_bulk_passes(
            emb,
            media_files,
            folder_path,
            content_vectors,
            custom_metadata_map,
            on_progress,
            media_type,
            origin,
            skip_embedding,
        )

        for i, file_path in enumerate(media_files):
            rel_path = file_path.relative_to(folder_path).as_posix()
            _report_per_file_progress(
                on_progress, i, total_files, progress_interval, media_type, rel_path, skip_embedding
            )

            media_data = _build_media_for_file(
                media_id,
                file_path,
                rel_path,
                folder_path,
                mt,
                origin,
                thin,
                content_vectors,
                content_md5s,
                custom_metadata_map,
                emb,
                bulk_embeddings,
                bulk_patch_outputs,
                skip_embedding,
            )
            if media_data is None:
                continue
            medias[media_id] = media_data
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
    embedder_name: str = "",
    custom_metadata_map: dict[str, dict[str, Any]] | None = None,
    skip_embedding: bool = False,
    recursive: bool = True,
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
    if on_progress is None:
        on_progress = _default_progress()
    on_progress("loading", "Scanning media files...", 0, 0)

    mt, emb = _resolve_target_and_embedder(media_type, embedder_name, skip_embedding)

    media_files = _scan_media_files(folder_path, mt, recursive)
    if not media_files:
        raise ValueError(f"No {media_type} files found in folder")

    _maybe_eager_load_model(
        emb, media_files, folder_path, content_vectors, custom_metadata_map, on_progress, skip_embedding
    )

    total_files = len(media_files)
    progress_interval = max(1, min(50, total_files // 50)) if total_files > 0 else 1

    for start in range(0, total_files, chunk_size):
        batch = media_files[start : start + chunk_size]
        chunk_bulk_embeddings, chunk_bulk_patch_outputs = _run_bulk_passes(
            emb,
            batch,
            folder_path,
            content_vectors,
            custom_metadata_map,
            on_progress,
            media_type,
            origin,
            skip_embedding,
        )

        chunk_medias: dict[int, dict[str, Any]] = {}
        media_id = 1
        chunk_suffix = f" (chunk {start // chunk_size + 1})"
        for i, file_path in enumerate(batch):
            rel_path = file_path.relative_to(folder_path).as_posix()
            _report_per_file_progress(
                on_progress,
                start + i,
                total_files,
                progress_interval,
                media_type,
                rel_path,
                skip_embedding,
                chunk_suffix=chunk_suffix,
            )

            media_data = _build_media_for_file(
                media_id,
                file_path,
                rel_path,
                folder_path,
                mt,
                origin,
                thin,
                content_vectors,
                content_md5s,
                custom_metadata_map,
                emb,
                chunk_bulk_embeddings,
                chunk_bulk_patch_outputs,
                skip_embedding,
            )
            if media_data is None:
                continue
            chunk_medias[media_id] = media_data
            media_id += 1

        if chunk_medias:
            yield chunk_medias

    on_progress("idle", f"Finished chunked loading of {total_files} {media_type} files", 0, 0)
