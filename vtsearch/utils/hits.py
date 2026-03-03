"""Helpers for building media hit dicts used by CLI scoring and route responses."""

from __future__ import annotations

from typing import Any


def build_media_hit(
    cid: int,
    media: dict[str, Any],
    score: float,
    **extra: Any,
) -> dict[str, Any]:
    """Build a hit dict for a scored media item.

    This is the single source of truth for the hit format used by CLI
    autodetect results and the ``/api/labels/fill-from-sort`` route.

    Args:
        cid: The media id.
        media: The media dict (from ``medias[cid]``).
        score: The prediction or similarity score.
        **extra: Additional keys merged into the hit (e.g. ``label="good"``).

    Returns:
        A dict with ``id``, ``filename``, ``category``, ``score`` and,
        when present on the media, ``origin``, ``origin_name``, ``md5``.
    """
    hit: dict[str, Any] = {
        "id": cid,
        "filename": media.get("filename", f"media_{cid}"),
        "category": media.get("category", "unknown"),
        "score": round(score, 4),
    }
    if media.get("origin") is not None:
        hit["origin"] = media["origin"]
    if media.get("origin_name"):
        hit["origin_name"] = media["origin_name"]
    if media.get("md5"):
        hit["md5"] = media["md5"]
    if extra:
        hit.update(extra)
    return hit
