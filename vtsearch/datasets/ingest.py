"""Ingest missing medias into an existing dataset by re-running their origins.

When a label-set references elements that are not present in the current
dataset, the user may choose to pull them in from their original sources.
This module groups the missing entries by origin, runs the appropriate
dataset importer for each origin, and cherry-picks only the medias that
match the missing entries (by ``origin_name``).  The recovered medias are
assigned fresh IDs that do not collide with existing medias and are
appended to the in-memory dataset.

When a :class:`~vtsearch.datasets.sources.base.MediaSource` is available
for an origin (e.g. folder, http_archive), individual files are fetched
directly via :meth:`~MediaSource.fetch_item` instead of re-running the
full importer — this is much faster when only a few medias are missing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Optional

from vtsearch.utils.state import next_media_id

ProgressCallback = Callable[[str, str, int, int], None]


def _default_progress() -> ProgressCallback:
    from vtsearch.utils import update_progress

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


def _ingest_via_source(
    origin_dict: dict[str, Any],
    entries: list[dict[str, Any]],
    medias: dict[int, dict[str, Any]],
    on_progress: ProgressCallback,
) -> int:
    """Try to ingest missing entries using a MediaSource (item-by-item).

    Returns the number of successfully ingested medias, or -1 if no
    MediaSource is available for this origin (caller should fall back to
    the full-importer path).
    """
    from vtsearch.datasets.sources import get_source_for_origin

    source = get_source_for_origin(origin_dict)
    if source is None:
        return -1

    # Resolve the media type from the folder_import_name (e.g. "sounds" → "audio").
    folder_import_name = origin_dict.get("params", {}).get("media_type", "")
    media_type_id = ""
    if folder_import_name:
        try:
            from vtsearch.media import get_by_folder_name

            mt = get_by_folder_name(folder_import_name)
            media_type_id = mt.type_id
        except (KeyError, ValueError):
            pass

    from vtsearch.models.resolver import embed_file

    ingested = 0
    cid = next_media_id(medias)

    try:
        for entry in entries:
            origin_name = entry.get("origin_name", "")
            filename = entry.get("filename", "")

            file_path = source.resolve_path(origin_name, filename)
            if file_path is None:
                continue

            # Embed the file — if embedding fails, fall back to legacy path
            # so we don't produce medias without embeddings.
            embedding = embed_file(file_path, media_type_id) if media_type_id else None
            if embedding is None:
                source.cleanup()
                return -1  # Signal caller to use legacy full-import path

            # Read file bytes and compute MD5
            file_bytes = file_path.read_bytes()
            md5 = hashlib.md5(file_bytes).hexdigest()

            media_data: dict[str, Any] = {
                "id": cid,
                "filename": origin_name or file_path.name,
                "origin": origin_dict,
                "origin_name": origin_name or file_path.name,
                "md5": md5,
                "embedding": embedding,
                "media_bytes": file_bytes,
                "media_path": str(file_path),
            }

            medias[cid] = media_data
            cid += 1
            ingested += 1

            on_progress(
                "ingesting",
                f"Fetched {origin_name or filename} ({ingested}/{len(entries)})...",
                ingested,
                len(entries),
            )
    finally:
        source.cleanup()

    return ingested


def ingest_missing_medias(
    missing_entries: list[dict[str, Any]],
    medias: dict[int, dict[str, Any]],
    on_progress: Optional[ProgressCallback] = None,
) -> int:
    """Re-ingest missing medias from their origins into *medias*.

    Groups *missing_entries* by origin, then for each origin group:

    1. If a :class:`~vtsearch.datasets.sources.base.MediaSource` is
       available, fetch only the needed files individually (fast path).
    2. Otherwise, fall back to running the full dataset importer and
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
    from vtsearch.datasets.importers import get_importer

    if on_progress is None:
        on_progress = _default_progress()

    groups = _group_by_origin(missing_entries)
    if not groups:
        return 0

    total_ingested = 0

    for origin_key, (origin_dict, entries) in groups.items():
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

        # Set origin on temp medias that don't have one
        for media in temp_medias.values():
            if media.get("origin") is None:
                media["origin"] = origin_dict
            if not media.get("origin_name"):
                media["origin_name"] = media.get("filename", "")

        # Cherry-pick matching medias
        cid = next_media_id(medias)
        for temp_clip in temp_medias.values():
            clip_origin_name = temp_clip.get("origin_name", "")
            clip_md5 = temp_clip.get("md5", "")
            if clip_origin_name in wanted_names or clip_md5 in wanted_md5s:
                temp_clip["id"] = cid
                medias[cid] = temp_clip
                cid += 1
                total_ingested += 1

    on_progress("idle", f"Ingested {total_ingested} media(s) from origins.", 0, 0)
    return total_ingested
