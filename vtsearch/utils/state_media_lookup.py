"""Media lookup helpers (origin+origin_name union with MD5) and duplicate collapsing."""

from __future__ import annotations

import json
from typing import Any

from vtsearch.utils.state_core import _state_lock, medias


def _origin_key(origin: dict[str, Any], origin_name: str) -> str:
    """Return a hashable string key for an (origin, origin_name) pair."""
    return json.dumps(origin, sort_keys=True) + "\0" + origin_name


def build_media_lookup(
    media_dict: dict[int, dict[str, Any]],
) -> tuple[dict[str, list[int]], dict[str, list[int]], dict[str, list[int]]]:
    """Build lookup tables for matching label entries to medias.

    Returns ``(origin_lookup, md5_lookup, name_lookup)`` where:

    * **origin_lookup** maps ``_origin_key(origin, origin_name)`` to a list of
      media IDs that share that origin+name pair.
    * **md5_lookup** maps an MD5 hex string to a list of media IDs whose
      content hash matches.
    * **name_lookup** maps an ``origin_name`` string to a list of media IDs
      that share that name.  Used as a fallback when the full origin dict is
      not available (e.g. CSV label imports across datasets).

    All lookups map to *lists* because the same key can match multiple medias
    (e.g. duplicate files with the same MD5).
    """
    origin_lookup: dict[str, list[int]] = {}
    md5_lookup: dict[str, list[int]] = {}
    name_lookup: dict[str, list[int]] = {}

    for media in media_dict.values():
        cid = media["id"]

        origin = media.get("origin")
        origin_name = media.get("origin_name", "")
        if origin is not None and origin_name:
            key = _origin_key(origin, origin_name)
            origin_lookup.setdefault(key, []).append(cid)

        if origin_name:
            name_lookup.setdefault(origin_name, []).append(cid)

        md5 = media.get("md5", "")
        if md5:
            md5_lookup.setdefault(md5, []).append(cid)

    return origin_lookup, md5_lookup, name_lookup


def resolve_media_ids(
    entry: dict[str, Any],
    origin_lookup: dict[str, list[int]],
    md5_lookup: dict[str, list[int]],
    name_lookup: dict[str, list[int]] | None = None,
) -> list[int]:
    """Resolve a label entry to matching media ID(s).

    Returns the **union** of medias matched by ``origin`` + ``origin_name``
    and medias matched by ``md5``.  Both lookups are always attempted so that
    a label is applied to every element in the dataset that corresponds to
    the entry, regardless of whether it was matched by provenance or by
    content hash.  Duplicate IDs are removed.

    When *name_lookup* is provided and neither origin+name nor MD5 produced a
    match, ``origin_name`` (or ``filename``) is used as a last-resort
    fallback.  This enables cross-dataset label transfer from CSV files that
    lack the full origin dict.
    """
    matched: dict[int, None] = {}

    origin = entry.get("origin")
    origin_name = entry.get("origin_name", "")

    if origin is not None and origin_name:
        key = _origin_key(origin, origin_name)
        for cid in origin_lookup.get(key, []):
            matched[cid] = None

    md5 = entry.get("md5", "")
    if md5:
        for cid in md5_lookup.get(md5, []):
            matched[cid] = None

    # Fallback: match by origin_name alone (or filename) when the full origin
    # dict is not available, e.g. CSV label imports across datasets.
    if not matched and name_lookup is not None:
        name = origin_name or entry.get("filename", "")
        if name:
            for cid in name_lookup.get(name, []):
                matched[cid] = None

    return list(matched)


def find_missing_entries(
    label_entries: list[dict[str, Any]],
    origin_lookup: dict[str, list[int]],
    md5_lookup: dict[str, list[int]],
    name_lookup: dict[str, list[int]] | None = None,
) -> list[dict[str, Any]]:
    """Return label entries that do not match any media by origin+name or md5.

    Only entries with a valid label (``"good"`` or ``"bad"``) are considered;
    entries with invalid labels are silently excluded (they are already counted
    as "skipped" by the caller).
    """
    missing: list[dict[str, Any]] = []
    for entry in label_entries:
        label = entry.get("label", "")
        if label not in ("good", "bad"):
            continue
        cids = resolve_media_ids(entry, origin_lookup, md5_lookup, name_lookup)
        if not cids:
            missing.append(entry)
    return missing


def collapse_duplicates(media_dict: dict[int, dict[str, Any]]) -> int:
    """Collapse duplicate medias (same MD5) into single representative items.

    For each group of medias sharing the same MD5, the first media becomes
    the representative.  Its ``"origin"`` is replaced with a ``"dupe_set"``
    origin whose ``"members"`` list records the original provenance of every
    duplicate (including the representative itself).  All other medias in the
    group are removed from *media_dict*.

    Args:
        media_dict: The mutable medias dict.  Modified in place.

    Returns:
        The number of duplicate groups collapsed (i.e. groups of size >= 2).
    """
    md5_groups: dict[str, list[int]] = {}
    for cid, media in media_dict.items():
        md5 = media.get("md5", "")
        if md5:
            md5_groups.setdefault(md5, []).append(cid)

    dupe_count = 0
    for md5, cids in md5_groups.items():
        if len(cids) < 2:
            continue
        dupe_count += 1

        rep_id = cids[0]
        rep = media_dict[rep_id]

        # Build members list with each duplicate's provenance
        members = []
        for cid in cids:
            media = media_dict[cid]
            members.append(
                {
                    "origin": media.get("origin"),
                    "origin_name": media.get("origin_name", ""),
                    "filename": media.get("filename", ""),
                    "category": media.get("category", ""),
                }
            )

        first_name = rep.get("origin_name", rep.get("filename", ""))
        rep["origin"] = {
            "importer": "dupe_set",
            "params": {"name": first_name},
            "members": members,
        }
        rep["origin_name"] = first_name

        # Remove the other duplicates
        for cid in cids[1:]:
            del media_dict[cid]

    return dupe_count


def get_dupe_count(media_dict: dict[int, dict[str, Any]] | None = None) -> int:
    """Return the number of duplicate groups in the media dict.

    Each media whose origin is ``"dupe_set"`` represents one group.
    """
    with _state_lock:
        source = media_dict if media_dict is not None else medias
        return sum(
            1
            for m in source.values()
            if isinstance(m.get("origin"), dict) and m["origin"].get("importer") == "dupe_set"
        )


def next_media_id(media_dict: dict[int, dict[str, Any]]) -> int:
    """Return the next available media ID (one past the current maximum)."""
    if not media_dict:
        return 1
    return max(media_dict) + 1
