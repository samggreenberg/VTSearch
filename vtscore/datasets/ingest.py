"""Ingest missing medias into an existing dataset by re-running their origins.

When a label-set references elements that are not present in the current
dataset, the user may choose to pull them in from their original sources.
This module groups the missing entries by origin, runs the appropriate
dataset importer for each origin, and cherry-picks only the medias that
match the missing entries (by ``origin_name``).  The recovered medias are
assigned fresh IDs that do not collide with existing medias and are
appended to the in-memory dataset.

When a :class:`~vtscore.datasets.sources.base.MediaSource` is available
for an origin (e.g. folder, http_archive), individual files are fetched
directly via :meth:`~MediaSource.fetch_item` instead of re-running the
full importer - this is much faster when only a few medias are missing.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Optional

from vtsearch.state import next_media_id
from vtscore.state.core import _state_lock

ProgressCallback = Callable[[str, str, int, int], None]


def _default_progress() -> ProgressCallback:
    from vtscore.concurrency.progress import get_thread_progress

    cb = get_thread_progress()
    if cb is not None:
        return cb
    from vtscore.concurrency.progress import update_progress

    return update_progress


def _group_by_origin(
    entries: list[dict[str, Any]],
) -> dict[str, tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Group label entries by their serialised origin.

    Returns a dict mapping ``json.dumps(origin, sort_keys=True)`` to a tuple
    of ``(origin_dict, [entries_with_that_origin])``.  Entries without an
    origin are silently skipped (they cannot be re-ingested).
    """
    groups: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for entry in entries:
        origin = entry.get("origin")
        if origin is None:
            continue
        key = json.dumps(origin, sort_keys=True)
        if key not in groups:
            groups[key] = (origin, [])
        groups[key][1].append(entry)
    return groups


def _media_type_from_origin(origin_dict: dict[str, Any]) -> str:
    """Determine the media type ID from an origin dict.

    Checks origin params (``media_type`` for folder origins) and falls back
    to the ``DEMO_DATASETS`` config for demo origins.  Returns an empty
    string if the media type cannot be determined.
    """
    params = origin_dict.get("params", {})
    importer_name = origin_dict.get("importer", "")

    # Folder origins store the media type (e.g. "audio")
    folder_import_name = params.get("media_type", "")
    if folder_import_name:
        try:
            from vtscore.media import get_by_folder_name

            return get_by_folder_name(folder_import_name).type_id
        except (KeyError, ValueError):
            pass

    # Demo origins: look up from DEMO_DATASETS config
    if importer_name == "demo":
        from vtscore.datasets.config import DEMO_DATASETS

        demo_name = params.get("name", "")
        if demo_name in DEMO_DATASETS:
            return DEMO_DATASETS[demo_name].get("media_type", "")

    return ""


def _embedder_name_for_type(medias: dict[int, dict[str, Any]], media_type_id: str) -> str:
    """Return the embedder name used by existing medias of the given type.

    Scans the live dataset for the first media whose ``type`` matches
    *media_type_id* and returns its ``embedder`` field.  Returns an empty
    string when no matching media is found (caller falls back to the default).
    """
    for m in medias.values():
        if m.get("media_type") == media_type_id:
            name = m.get("embedder", "")
            if name:
                return name
    return ""


def _build_media_data(
    *,
    origin_dict: dict[str, Any],
    entry: dict[str, Any],
    media_type_id: str,
    origin_name: str,
    file_path: Any,
    file_bytes: bytes,
    md5: str,
    embedding: Any,
    embedder_name: str = "",
) -> dict[str, Any]:
    """Build a media dict matching the shape produced by folder loading.

    Mirrors the field set in ``loader.py`` so re-ingested medias carry
    their media ``type`` (without it the frontend falls back to audio)
    and any ``custom_metadata`` supplied by the label entry (via
    :class:`~vtscore.datasets.labelset.LabeledElement.metadata`).
    """
    name = origin_name or file_path.name
    media_data: dict[str, Any] = {
        "media_type": media_type_id,
        "file_size": len(file_bytes),
        "md5": md5,
        "embedding": embedding,
        "embedder": embedder_name,
        "filename": entry.get("filename") or name,
        "category": entry.get("category", ""),
        "origin": origin_dict,
        "origin_name": name,
        "media_bytes": file_bytes,
        "media_string": None,
        "media_path": str(file_path),
        "duration": 0,
    }
    custom_metadata = entry.get("metadata")
    if custom_metadata:
        media_data["custom_metadata"] = custom_metadata
    return media_data


def _has_clip_params(origin_dict: dict[str, Any]) -> bool:
    """Return True if the origin carries any clip-aware params."""
    params = origin_dict.get("params", {})
    return bool(params.get("clipper") or params.get("clipper_chain"))


def _boundary_tag(params: dict[str, Any]) -> str:
    """Build a stable tag from clip metadata to disambiguate clips that share parent bytes.

    Used for video metadata-only clips (and other fall-through paths where
    the embedding step couldn't produce distinct content bytes) so each
    clip still gets a unique MD5 instead of colliding on the parent's hash.
    Mirrors the encoding in
    :func:`vtscore.datasets.load_pipeline._fixup_clip_md5_and_embeddings`.
    """
    parts: list[str] = []
    for key in ("clipper", "clipper_chain", "clip_start", "clip_end", "clip_box", "clip_index"):
        if key in params:
            parts.append(f"{key}={params[key]}")
    return "|" + "|".join(parts) if parts else ""


def _resolve_clip_content_and_embedding(
    file_path: Any,
    media_type_id: str,
    origin_dict: dict[str, Any],
    embedder_name: str,
) -> tuple[Any, bytes, str] | None:
    """Resolve embedding + ``media_bytes`` + ``md5`` for a (possibly clipped) origin.

    Returns ``(embedding, content_bytes, md5)`` where:

    * ``embedding`` is the clipped embedding when the origin has clip
      params, otherwise the parent's embedding.
    * ``content_bytes`` is the clip's bytes (audio / image / text clips)
      OR the parent's bytes when the clip is metadata-only (e.g. video) or
      when there are no clip params. The frontend can still play video
      clips because ``clip_start`` / ``clip_end`` live in the origin.
    * ``md5`` is the hash of ``content_bytes`` for true clips and
      non-clipped medias. For metadata-only clips, the MD5 mixes the
      parent's hash with a clip boundary tag so distinct clips of the same
      parent don't collide under :func:`collapse_duplicates`.

    Returns ``None`` when embedding fails.
    """
    from vtscore.detectors.resolver import _apply_clip_and_embed, embed_file

    if _has_clip_params(origin_dict):
        result = _apply_clip_and_embed(file_path, media_type_id, origin_dict, embedder_name)
        if result is None:
            return None
        embedding, clip_bytes = result
        if clip_bytes is not None:
            return embedding, clip_bytes, hashlib.md5(clip_bytes).hexdigest()

        # Metadata-only clip (video) or fall-through: use parent bytes,
        # but disambiguate the MD5 via a boundary tag so distinct clips of
        # the same parent don't collapse together.
        parent_bytes = file_path.read_bytes()
        tag = _boundary_tag(origin_dict.get("params", {}))
        if tag:
            combined = hashlib.md5(parent_bytes).hexdigest() + tag
            md5 = hashlib.md5(combined.encode()).hexdigest()
        else:
            md5 = hashlib.md5(parent_bytes).hexdigest()
        return embedding, parent_bytes, md5

    embedding = embed_file(file_path, media_type_id, embedder_name) if media_type_id else None
    if embedding is None:
        return None
    parent_bytes = file_path.read_bytes()
    return embedding, parent_bytes, hashlib.md5(parent_bytes).hexdigest()


def _ingest_via_source(
    origin_dict: dict[str, Any],
    entries: list[dict[str, Any]],
    medias: dict[int, dict[str, Any]],
    on_progress: ProgressCallback,
) -> int:
    """Try to ingest missing entries using a MediaSource.

    Resolves all entry paths in a single bulk call so sources that
    override :meth:`~vtscore.datasets.sources.base.MediaSource.resolve_paths`
    can parallelise network I/O before the sequential embed step.

    Returns the number of successfully ingested medias, or -1 if no
    MediaSource is available for this origin (caller should fall back to
    the full-importer path).
    """
    from vtscore.datasets.sources import get_source_for_origin

    source = get_source_for_origin(origin_dict)
    if source is None:
        return -1

    media_type_id = _media_type_from_origin(origin_dict)
    embedder_name = _embedder_name_for_type(medias, media_type_id) if media_type_id else ""

    ingested = 0

    try:
        pairs = [(e.get("origin_name", ""), e.get("filename", "")) for e in entries]
        paths = source.resolve_paths(pairs)

        for entry, file_path in zip(entries, paths):
            origin_name = entry.get("origin_name", "")
            filename = entry.get("filename", "")

            if file_path is None:
                continue

            # Resolve embedding + clip-aware bytes/md5 - if any step fails,
            # fall back to the legacy full-import path so we don't produce
            # medias with mismatched embedding/MD5/bytes triples.
            if not media_type_id:
                source.cleanup()
                return -1
            resolved = _resolve_clip_content_and_embedding(file_path, media_type_id, origin_dict, embedder_name)
            if resolved is None:
                source.cleanup()
                return -1  # Signal caller to use legacy full-import path
            embedding, file_bytes, md5 = resolved

            media_data = _build_media_data(
                origin_dict=origin_dict,
                entry=entry,
                media_type_id=media_type_id,
                origin_name=origin_name,
                file_path=file_path,
                file_bytes=file_bytes,
                md5=md5,
                embedding=embedding,
                embedder_name=embedder_name,
            )

            # Allocate the ID and insert atomically, so two concurrent
            # ingests cannot collide on the same next_media_id.
            with _state_lock:
                cid = next_media_id(medias)
                media_data["id"] = cid
                medias[cid] = media_data
            ingested += 1

            on_progress(
                "ingesting",
                f"Fetched {origin_name or filename}",
                ingested,
                len(entries),
            )
    finally:
        source.cleanup()

    return ingested


def _ingest_via_resolver(
    origin_dict: dict[str, Any],
    entries: list[dict[str, Any]],
    medias: dict[int, dict[str, Any]],
    on_progress: ProgressCallback,
) -> int:
    """Try to ingest missing entries using resolve_file_from_origin (item-by-item).

    Uses the importer-registry-based file resolver, which handles demo,
    folder, converter, dupe_set, and any other origin type that has a
    ``resolve_file()`` method on its importer.

    Returns the number of successfully ingested medias, or -1 if the
    media type cannot be determined (caller should fall back to the
    full-importer path).
    """
    media_type_id = _media_type_from_origin(origin_dict)
    if not media_type_id:
        return -1

    embedder_name = _embedder_name_for_type(medias, media_type_id)

    from vtscore.detectors.resolver import resolve_file_context

    ingested = 0

    for entry in entries:
        origin_name = entry.get("origin_name", "")
        filename = entry.get("filename", "")

        # Hold the source alive through both embed and read_bytes - some
        # MediaSources materialise the file inside a TemporaryDirectory
        # they own, and dropping the source before we touch the path
        # finalises that temp dir under GC and leaves us with a stale path.
        with resolve_file_context(origin_dict, origin_name, filename) as file_path:
            if file_path is None:
                continue

            # Clip-aware: produces the clip embedding and the clip's bytes
            # so the stored ``media_bytes`` / ``md5`` / ``file_size`` all
            # describe the clip, not the parent.
            resolved = _resolve_clip_content_and_embedding(file_path, media_type_id, origin_dict, embedder_name)
            if resolved is None:
                continue
            embedding, file_bytes, md5 = resolved

            media_data = _build_media_data(
                origin_dict=origin_dict,
                entry=entry,
                media_type_id=media_type_id,
                origin_name=origin_name,
                file_path=file_path,
                file_bytes=file_bytes,
                md5=md5,
                embedding=embedding,
                embedder_name=embedder_name,
            )

        # Atomic id allocation + insert, see _ingest_via_source above.
        with _state_lock:
            cid = next_media_id(medias)
            media_data["id"] = cid
            medias[cid] = media_data
        ingested += 1

        on_progress(
            "ingesting",
            f"Resolved {origin_name or filename}",
            ingested,
            len(entries),
        )

    return ingested


def ingest_missing_medias(  # noqa: C901
    missing_entries: list[dict[str, Any]],
    medias: dict[int, dict[str, Any]],
    on_progress: Optional[ProgressCallback] = None,
) -> int:
    """Re-ingest missing medias from their origins into *medias*.

    Groups *missing_entries* by origin, then for each origin group:

    1. If a :class:`~vtscore.datasets.sources.base.MediaSource` is
       available, fetch only the needed files individually (fast path).
    2. If :func:`~vtscore.detectors.resolver.resolve_file_from_origin` can
       locate individual files (e.g. demo datasets with files on disk),
       embed and ingest them one-by-one (medium path).
    3. Otherwise, fall back to running the full dataset importer and
       cherry-picking matching medias (legacy path).

    Args:
        missing_entries: Label entries (dicts with ``origin``,
            ``origin_name``, ``md5``, ``label``, etc.) that were not found
            in the current dataset.
        medias: The live dataset dict to extend in-place.
        on_progress: Optional progress callback.

    Returns:
        The number of medias successfully ingested.
    """
    from vtscore.datasets.importers import get_importer

    if on_progress is None:
        on_progress = _default_progress()

    groups = _group_by_origin(missing_entries)
    if not groups:
        return 0

    total_ingested = 0

    for _origin_key, (origin_dict, entries) in groups.items():
        importer_name = origin_dict.get("importer", "")

        on_progress(
            "ingesting",
            f"Re-ingesting from {importer_name} ({len(entries)} medias)...",
            0,
            0,
        )

        # Fast path: use MediaSource for item-by-item fetching
        source_result = _ingest_via_source(origin_dict, entries, medias, on_progress)
        if source_result >= 0:
            total_ingested += source_result
            continue

        # Medium path: use resolve_file_from_origin for item-by-item resolution
        # (handles demo datasets, importers with resolve_file, etc.)
        resolver_result = _ingest_via_resolver(origin_dict, entries, medias, on_progress)
        if resolver_result >= 0:
            total_ingested += resolver_result
            continue

        # Legacy path: run the full importer
        importer = get_importer(importer_name)
        if importer is None:
            continue

        params = origin_dict.get("params", {})

        # Build a set of origin_names we're looking for
        wanted_names: set[str] = set()
        wanted_md5s: set[str] = set()
        for entry in entries:
            name = entry.get("origin_name", "")
            if name:
                wanted_names.add(name)
            md5 = entry.get("md5", "")
            if md5:
                wanted_md5s.add(md5)

        # Run the importer into a temporary medias dict
        temp_medias: dict[int, dict[str, Any]] = {}
        try:
            importer.run_cli(params, temp_medias)
        except Exception:
            # If the importer fails (e.g. folder not found), skip this origin
            continue

        # Set origin on temp medias that don't have one.  Each media gets a
        # fresh copy so a later mutation of one media's ``origin.params``
        # cannot leak across siblings (and across pickle round-trips).
        for media in temp_medias.values():
            if media.get("origin") is None:
                media["origin"] = {
                    "importer": origin_dict.get("importer", ""),
                    "params": dict(origin_dict.get("params", {})),
                }
            if not media.get("origin_name"):
                media["origin_name"] = media.get("filename", "")

        # Cherry-pick matching medias.  Allocate IDs and insert atomically
        # under _state_lock so a concurrent ingest can't reuse the same ID.
        # Snapshot the values up front: the importer may retain a reference to
        # temp_medias and mutate it from a background thread.
        for temp_clip in list(temp_medias.values()):
            clip_origin_name = temp_clip.get("origin_name", "")
            clip_md5 = temp_clip.get("md5", "")
            if clip_origin_name in wanted_names or clip_md5 in wanted_md5s:
                with _state_lock:
                    cid = next_media_id(medias)
                    temp_clip["id"] = cid
                    medias[cid] = temp_clip
                total_ingested += 1

    on_progress("idle", f"Ingested {total_ingested} media(s) from origins.", 0, 0)
    return total_ingested
