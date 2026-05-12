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
from vtsearch.utils.paths import glob_top_level, rglob_follow_symlinks


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
    """Run :meth:`MediaEmbedder.patch_forward` on every file in *media_files*.

    Only called when the active embedder reports
    ``supports_patch_regions == True``.  Returns a ``Path → PatchEmbedOutput``
    mapping; files whose patch forward returned ``None`` are omitted.

    Per-image, not bulk-batched — there is no patch-forward batch API on
    the embedder yet.  For v1 this means patch-capable embedders run two
    forward passes per image (one in ``embed_media_bulk`` for the CLS
    vector, one here for the full patch grid).  Acceptable for v1; a
    follow-up can fuse the two passes when latency matters.
    """
    pending: list[tuple[Path, dict[str, Any]]] = []
    for file_path in media_files:
        pending.append((file_path, _make_embed_input(file_path, folder_path, origin, custom_metadata_map)))

    if not pending:
        return {}

    on_progress("embedding", f"Patch-embedding {len(pending)} {media_type} files...", 0, len(pending))

    out: dict[Path, Any] = {}
    original_cb = emb._on_progress
    emb._on_progress = on_progress
    try:
        for i, (file_path, media_dict) in enumerate(pending):
            patch_out = emb.patch_forward(media_dict)
            if patch_out is not None:
                out[file_path] = patch_out
            on_progress(
                "embedding",
                f"Patch-embedding {i + 1}/{len(pending)} {media_type} files...",
                i + 1,
                len(pending),
            )
    finally:
        emb._on_progress = original_cb

    return out


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

    from vtsearch.models.patch_regions import build_region_tree, to_fp16  # noqa: PLC0415

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

    # Find all files of the specified media type.  Recursion descends into
    # subdirectories (default); when disabled, only files directly inside
    # ``folder_path`` are included.
    media_files = []
    for ext in mt.file_extensions:
        media_files.extend(_scan_files(folder_path, ext, recursive))

    if not media_files:
        raise ValueError(f"No {media_type} files found in folder")

    medias.clear()
    media_id = 1
    total_files = len(media_files)

    # Flush everything that still needs embedding through the embedder's
    # bulk entrypoint up front; the per-file loop below just looks up the
    # pre-computed vector.  Subclasses that override ``_embed_media_bulk_impl``
    # can batch internally; the default impl loops per item and emits
    # per-item progress so the UI stays responsive.
    try:
        bulk_embeddings: dict[Path, Any] = {}
        bulk_patch_outputs: dict[Path, Any] = {}
        if emb is not None and not skip_embedding:
            bulk_embeddings = _bulk_embed_files(
                emb,
                media_files,
                folder_path,
                content_vectors,
                custom_metadata_map,
                on_progress,
                media_type,
                origin=origin,
            )
            # If the active embedder produces patch regions (DINOv2,
            # DINOv3, EUPE), run a second per-image pass to harvest the
            # CLS / patch grid / saliency for each file. Skipped entirely
            # for single-vector embedders (SigLIP etc.) so legacy
            # datasets are byte-identical to the pre-patch behaviour.
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
                embedding = bulk_embeddings.get(file_path)
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

            patch_out = bulk_patch_outputs.get(file_path)
            if patch_out is not None:
                _attach_patch_regions(media_data, patch_out)

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

    # Find all files of the specified media type.  Recursion descends into
    # subdirectories (default); when disabled, only files directly inside
    # ``folder_path`` are included.
    media_files: list[Path] = []
    for ext in mt.file_extensions:
        media_files.extend(_scan_files(folder_path, ext, recursive))

    if not media_files:
        raise ValueError(f"No {media_type} files found in folder")

    total_files = len(media_files)
    embedder_id = emb.name if emb else ""

    # Process in groups of chunk_size
    for start in range(0, total_files, chunk_size):
        batch = media_files[start : start + chunk_size]
        chunk_medias: dict[int, dict[str, Any]] = {}
        media_id = 1

        # Flush the whole chunk through ``embed_media_bulk`` up front, then
        # the per-file loop below just looks up the pre-computed vector.
        # Scoping the bulk call to a single chunk preserves chunked loading's
        # memory story — only one chunk's worth of embeddings lives in
        # memory at a time.
        chunk_bulk_embeddings: dict[Path, Any] = {}
        chunk_bulk_patch_outputs: dict[Path, Any] = {}
        if emb is not None and not skip_embedding:
            chunk_bulk_embeddings = _bulk_embed_files(
                emb,
                batch,
                folder_path,
                content_vectors,
                custom_metadata_map,
                on_progress,
                media_type,
                origin=origin,
            )
            if getattr(emb, "supports_patch_regions", False) is True:
                chunk_bulk_patch_outputs = _bulk_patch_forward_files(
                    emb,
                    batch,
                    folder_path,
                    on_progress,
                    media_type,
                    origin=origin,
                    custom_metadata_map=custom_metadata_map,
                )

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
                embedding = chunk_bulk_embeddings.get(file_path)
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

            patch_out = chunk_bulk_patch_outputs.get(file_path)
            if patch_out is not None:
                _attach_patch_regions(media_data, patch_out)

            chunk_medias[media_id] = media_data
            media_id += 1

        if chunk_medias:
            yield chunk_medias

    on_progress("idle", f"Finished chunked loading of {total_files} {media_type} files")
