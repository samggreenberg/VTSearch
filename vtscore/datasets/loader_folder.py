"""Folder-based dataset loaders.

Loads datasets from a directory tree of media files, with optional
pre-computed embeddings/MD5s, custom metadata, and chunked iteration.
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


def _resolve_folder_embedder(media_type: str, embedder_name: str, skip_embedding: bool) -> tuple[Any, Any]:
    """Look up the media type and pick an embedder.

    Returns ``(mt, emb)``.  ``emb`` is ``None`` when ``skip_embedding`` is
    set or when no embedder is registered for the media type.
    """
    from vtscore.media import embedders_for_type, get_by_folder_name, get_embedder  # noqa: PLC0415

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


def _eager_load_embedder_models(
    emb: Any,
    media_files: list[Path],
    folder_path: Path,
    content_vectors: dict[str, Any] | None,
    custom_metadata_map: dict[str, dict[str, Any]] | None,
    on_progress: ProgressCallback,
) -> None:
    """Eagerly load model weights when at least one file needs the embedder.

    Skipped when every file already has an override (NPZ-only imports
    shouldn't pay the weight-load cost).  Loading happens here so that
    download / weight-loading time does not pollute the per-file
    embedding progress bar that the caller starts immediately after.
    """
    if getattr(emb, "_model", None) is not None:
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


def _resolve_folder_load_inputs(
    folder_path: Path,
    media_type: str,
    embedder_name: str,
    skip_embedding: bool,
    recursive: bool,
    content_vectors: dict[str, Any] | None,
    content_md5s: dict[str, str] | None,
    custom_metadata_map: dict[str, dict[str, Any]] | None,
    on_progress: ProgressCallback,
) -> tuple[Any, Any, list[Path], int]:
    """Shared front-matter for the folder loaders.

    Resolves the media type + embedder, scans the folder, validates the
    override maps against the scanned file list, and eagerly loads model
    weights when needed.  Returns ``(mt, emb, media_files, total_files)``.
    Raises ``ValueError`` for unknown media types, unknown embedder names,
    empty folders, or ambiguous basename keys in any override map.
    """
    mt, emb = _resolve_folder_embedder(media_type, embedder_name, skip_embedding)

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

    if not skip_embedding and emb is not None:
        _eager_load_embedder_models(
            emb,
            media_files,
            folder_path,
            content_vectors,
            custom_metadata_map,
            on_progress,
        )

    return mt, emb, media_files, len(media_files)


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
            if content_vectors and rel_path in content_vectors and basename in content_vectors:
                if not _embeddings_equal(content_vectors[rel_path], content_vectors[basename]):
                    logger.warning(
                        "content_vectors has conflicting entries for %r and %r; "
                        "using the relative-path entry.",
                        rel_path,
                        basename,
                    )
            if content_md5s and rel_path in content_md5s and basename in content_md5s:
                if content_md5s[rel_path] != content_md5s[basename]:
                    logger.warning(
                        "content_md5s has conflicting entries for %r and %r "
                        "(%r vs %r); using the relative-path entry.",
                        rel_path,
                        basename,
                        content_md5s[rel_path],
                        content_md5s[basename],
                    )
            if custom_metadata_map and rel_path in custom_metadata_map and basename in custom_metadata_map:
                rp_cm = custom_metadata_map[rel_path] or {}
                bn_cm = custom_metadata_map[basename] or {}
                rp_emb = _get_embedding_value(rp_cm)
                bn_emb = _get_embedding_value(bn_cm)
                if not _embeddings_equal(rp_emb, bn_emb):
                    logger.warning(
                        "custom_metadata_map has conflicting embeddings for %r and %r; "
                        "using the relative-path entry.",
                        rel_path,
                        basename,
                    )
                rp_md5 = _get_md5_value(rp_cm)
                bn_md5 = _get_md5_value(bn_cm)
                if rp_md5 and bn_md5 and rp_md5 != bn_md5:
                    logger.warning(
                        "custom_metadata_map has conflicting md5s for %r and %r "
                        "(%r vs %r); using the relative-path entry.",
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
    skip_embedding: bool,
    emb: Any,
    bulk_embeddings: dict[Path, Any],
    file_path: Path,
) -> tuple[Any, str] | None:
    """Pick the embedding + embedder_id for *file_path*.

    Resolution order matches the original inline logic: custom_metadata
    embedding → content_vectors[rel_path] → content_vectors[basename] →
    ``None`` when ``skip_embedding`` → bulk-embedded vector.  Returns
    ``None`` when the file should be skipped entirely (no embedder
    available, or bulk embedding failed for this path).  External
    vectors come back with ``embedder_id == ""`` so downstream code does
    not re-embed against a mismatched model.
    """
    cm_embedding = _get_embedding_value(file_cm) if file_cm else None
    if cm_embedding is not None:
        return cm_embedding, ""
    if content_vectors and rel_path in content_vectors:
        return content_vectors[rel_path], ""
    if content_vectors and file_name in content_vectors:
        return content_vectors[file_name], ""
    if skip_embedding:
        return None, ""
    if emb is None:
        return None
    embedding = bulk_embeddings.get(file_path)
    if embedding is None:
        return None
    return embedding, emb.name


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
    same nested ``params`` dict — a later mutation on one media would
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
        "type": type_id,
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


def _run_bulk_embedders(
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
    """Run :func:`_bulk_embed_files` and (when supported) the patch-forward pass.

    Returns ``(bulk_embeddings, bulk_patch_outputs)``.  Both default to
    empty dicts when ``emb`` is missing or ``skip_embedding`` is set.
    """
    if emb is None or skip_embedding:
        return {}, {}
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
    bulk_patch_outputs: dict[Path, Any] = {}
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
    return bulk_embeddings, bulk_patch_outputs


def _emit_per_file_progress(
    on_progress: ProgressCallback,
    skip_embedding: bool,
    media_type: str,
    rel_path: str,
    current: int,
    total: int,
    chunk_label: str,
) -> None:
    """Emit a per-file progress update (folder loaders)."""
    phase = "loading" if skip_embedding else "embedding"
    verb = "Loading" if skip_embedding else "Embedding"
    msg = f"{verb} {media_type} {rel_path}{chunk_label}..."
    on_progress(phase, msg, current, total)


def _build_per_file_media(
    *,
    media_id: int,
    file_path: Path,
    rel_path: str,
    mt: Any,
    emb: Any,
    bulk_embeddings: dict[Path, Any],
    bulk_patch_outputs: dict[Path, Any],
    content_vectors: dict[str, Any] | None,
    content_md5s: dict[str, str] | None,
    custom_metadata_map: dict[str, dict[str, Any]] | None,
    skip_embedding: bool,
    thin: bool,
    origin: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Resolve embedding + md5 and build the per-file media dict.

    Returns ``None`` when the file should be skipped (no embedder / bulk
    embedding failed).  Handles both thin and full modes, attaches
    custom_metadata and patch regions when available, and merges in
    type-specific fields in full mode.
    """
    file_cm = _lookup_file_custom_metadata(rel_path, file_path.name, custom_metadata_map)

    resolved = _resolve_file_embedding(
        rel_path,
        file_path.name,
        file_cm,
        content_vectors,
        skip_embedding,
        emb,
        bulk_embeddings,
        file_path,
    )
    if resolved is None:
        return None
    embedding, embedder_id = resolved

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

    from vtscore.media.patch_embed import build_region_tree, to_fp16  # noqa: PLC0415

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
    :attr:`~vtscore.media.base.MediaType.folder_import_name`.  Adding a
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
            fallback.  When a bare basename would match more than one file
            in the folder (without a disambiguating relative-path entry for
            every match) the load fails with ``ValueError``; when both a
            relative-path entry and a basename entry exist for the same
            file with different values, the loader logs a warning and
            keeps the relative-path entry.
        content_md5s: Optional mapping of filename to a pre-computed MD5 hex
            digest string.  Keys follow the same lookup logic as
            ``content_vectors`` (relative path first, then basename), and
            the same ambiguity checks apply.
        origin: Optional serialised
            :class:`~vtscore.datasets.origin.Origin` dict to attach to each
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
            path first, then basename), and the same ambiguity checks apply.
            When a metadata dict contains a non-empty ``"md5"`` key, that
            value is used as the media's MD5 instead of computing it from
            the file contents.  When it contains an ``"embedding"`` key,
            that value is used as the media's embedding vector (highest
            priority, above ``content_vectors`` and the embedding model).
            The metadata dict is also attached to the media as
            ``custom_metadata``.
        skip_embedding: When ``True``, skip embedder resolution and model
            loading entirely.  Files with pre-computed vectors in
            ``content_vectors`` use those; files without are included with
            ``embedding=None``.  Useful when vectors have already been
            downloaded or computed externally.

    Raises:
        ValueError: If ``media_type`` is not recognised, if no matching
            files are found in ``folder_path``, or if an override map has
            a bare-basename key that would silently fan out to multiple
            files in the folder.
    """
    if on_progress is None:
        on_progress = _default_progress()

    on_progress("loading", "Scanning media files...", 0, 0)

    mt, emb, media_files, total_files = _resolve_folder_load_inputs(
        folder_path,
        media_type,
        embedder_name,
        skip_embedding,
        recursive,
        content_vectors,
        content_md5s,
        custom_metadata_map,
        on_progress,
    )

    medias.clear()
    media_id = 1
    _progress_interval = max(1, min(50, total_files // 50)) if total_files > 0 else 1

    # Flush everything that still needs embedding through the embedder's
    # bulk entrypoint up front; the per-file loop below just looks up the
    # pre-computed vector.  Subclasses that override ``_embed_media_bulk_impl``
    # can batch internally; the default impl loops per item and emits
    # per-item progress so the UI stays responsive.
    try:
        bulk_embeddings, bulk_patch_outputs = _run_bulk_embedders(
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

            if i % _progress_interval == 0 or i + 1 == total_files:
                _emit_per_file_progress(
                    on_progress,
                    skip_embedding,
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
                emb=emb,
                bulk_embeddings=bulk_embeddings,
                bulk_patch_outputs=bulk_patch_outputs,
                content_vectors=content_vectors,
                content_md5s=content_md5s,
                custom_metadata_map=custom_metadata_map,
                skip_embedding=skip_embedding,
                thin=thin,
                origin=origin,
            )
            if built is None:
                continue
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
        ValueError: If ``media_type`` is not recognised, if no matching
            files are found in ``folder_path``, or if an override map has
            a bare-basename key that would silently fan out to multiple
            files in the folder.
    """
    if on_progress is None:
        on_progress = _default_progress()

    on_progress("loading", "Scanning media files...", 0, 0)

    mt, emb, media_files, total_files = _resolve_folder_load_inputs(
        folder_path,
        media_type,
        embedder_name,
        skip_embedding,
        recursive,
        content_vectors,
        content_md5s,
        custom_metadata_map,
        on_progress,
    )

    _progress_interval = max(1, min(50, total_files // 50)) if total_files > 0 else 1

    for start in range(0, total_files, chunk_size):
        batch = media_files[start : start + chunk_size]
        chunk_medias: dict[int, dict[str, Any]] = {}
        media_id = 1

        # Scope bulk embedding to one chunk so memory stays bounded.
        chunk_bulk_embeddings, chunk_bulk_patch_outputs = _run_bulk_embedders(
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

        chunk_label = f" (chunk {start // chunk_size + 1})"

        for i, file_path in enumerate(batch):
            global_idx = start + i
            rel_path = file_path.relative_to(folder_path).as_posix()

            if global_idx % _progress_interval == 0 or global_idx + 1 == total_files:
                _emit_per_file_progress(
                    on_progress,
                    skip_embedding,
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
                emb=emb,
                bulk_embeddings=chunk_bulk_embeddings,
                bulk_patch_outputs=chunk_bulk_patch_outputs,
                content_vectors=content_vectors,
                content_md5s=content_md5s,
                custom_metadata_map=custom_metadata_map,
                skip_embedding=skip_embedding,
                thin=thin,
                origin=origin,
            )
            if built is None:
                continue
            chunk_medias[media_id] = built
            media_id += 1

        if chunk_medias:
            yield chunk_medias

    on_progress("idle", f"Finished chunked loading of {total_files} {media_type} files", 0, 0)
