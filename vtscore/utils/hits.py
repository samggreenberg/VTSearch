"""Helpers for building media hit dicts used by CLI scoring and route responses."""

from __future__ import annotations

from typing import Any

#: ``custom_metadata`` keys stripped before the dict reaches a caller.
#:
#: ``embedding`` is the pre-computed-vector channel an importer uses to ship a
#: vector alongside a file (``custom_metadata_map``, see
#: :func:`vtscore.datasets.loader_folder.load_dataset_from_folder`).  It is
#: consumed at load time and has no business in anything served back out: it
#: is a numpy array, so it would break ``json.dumps`` in every JSON exporter
#: and in Flask's response encoder alike, and writing a vector into an export
#: is exactly the persistence the no-persisted-vectors rule forbids.
#: ``_HitSchema`` relies on its media fields being an allowlist for the same
#: reason, and ``custom_metadata`` is a free-form ``Dict`` that would wave the
#: vector straight through.
_CUSTOM_METADATA_EXCLUDED_KEYS = frozenset({"embedding"})


def hit_custom_metadata(media: dict[str, Any]) -> dict[str, Any]:
    """Return *media*'s importer metadata, safe to serve, or ``{}`` for none.

    Public because every surface that hands a media's ``custom_metadata`` to
    an outside caller — hits, the detector and processor scoring routes,
    ``POST /api/medias/batch`` — has to agree on what it strips.  A top-level
    key filter cannot do the job: the vector rides *inside* ``custom_metadata``.

    Always a fresh dict, so a consumer that mutates the result cannot reach
    back into the loaded media.
    """
    custom = media.get("custom_metadata")
    if not isinstance(custom, dict):
        return {}
    return {k: v for k, v in custom.items() if k not in _CUSTOM_METADATA_EXCLUDED_KEYS}


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
        when present on the media, ``custom_metadata``, ``origin``,
        ``origin_name``, ``md5``.
    """
    hit: dict[str, Any] = {
        "id": cid,
        "filename": media.get("filename", f"media_{cid}"),
        "category": media.get("category", "unknown"),
        "score": round(score, 4),
    }
    # Importer-supplied metadata is how an exporter correlates a hit back to
    # the caller's own system (asset ids, catalogue rows, …), so it travels
    # with the hit whenever the media carries any.
    custom = hit_custom_metadata(media)
    if custom:
        hit["custom_metadata"] = custom
    if media.get("origin") is not None:
        hit["origin"] = media["origin"]
    if media.get("origin_name"):
        hit["origin_name"] = media["origin_name"]
    if media.get("md5"):
        hit["md5"] = media["md5"]
    # Include clip boundary fields when present (clipped sub-medias).
    if media.get("clip_start") is not None:
        hit["clip_start"] = media["clip_start"]
    if media.get("clip_end") is not None:
        hit["clip_end"] = media["clip_end"]
    if media.get("clip_box") is not None:
        hit["clip_box"] = media["clip_box"]
    if media.get("clip_index") is not None:
        hit["clip_index"] = media["clip_index"]
    if extra:
        hit.update(extra)
    return hit
