"""Vote routes: set one media's vote, or apply one target to many medias.

The routes register on :data:`vtsearch.routes.media.list.medias_bp`.
"""

from __future__ import annotations

from flask_smorest import abort

from vtscore.datasets.vote_provenance import normalize_provenance
from vtsearch.routes._context import require_dataset_header, require_detector_header
from vtsearch.routes.media._serving import _parse_region_box
from vtsearch.routes.media.list import medias_bp
from vtsearch.schemas.media import (
    MediaVoteBulkRequestSchema,
    MediaVoteBulkResponseSchema,
    MediaVoteRequestSchema,
    MediaVoteResponseSchema,
)
from vtsearch.state import get_media, set_vote


@medias_bp.route("/api/medias/<int:media_id>/vote", methods=["POST"])
@medias_bp.arguments(MediaVoteRequestSchema)
@medias_bp.response(200, MediaVoteResponseSchema)
@medias_bp.alt_response(
    400, description="region_box or provenance is malformed, or a non-good target carries a region_box."
)
@medias_bp.alt_response(404, description="Media not found.")
@medias_bp.alt_response(500, description="The vote could not be written to the detector store.")
@require_dataset_header
@require_detector_header
def vote_media(body: dict, media_id: int):
    """Set a single media's vote to an **absolute target state**.

    ``target`` is one of:

    - ``"good"``: set to good (overrides ``bad`` if present).
    - ``"bad"``: set to bad (overrides ``good`` if present).
    - ``"none"``: un-vote (remove any existing vote).

    Behaviour is **idempotent**: sending the current state is a no-op
    that does not append to ``label_history``, does not credit
    achievements, and returns the existing click-time, so two stale-view
    tabs that race the same media cannot alternate ADD/REMOVE on the
    server or inflate the achievement counter beyond the number of real
    labeling decisions (logical-bug-audit H1).

    Yes-targets may carry ``"region_box": [x0, y0, x1, y1]`` (normalised
    image coords in ``[0, 1]``) to designate the good region.
    ``"bad"`` and ``"none"`` targets must not include ``region_box``; by
    design no-votes are image-level always (``docs/ARCHITECTURE.md``).

    An optional ``provenance`` block records how the item was surfaced
    (flow / autopilot phase / select mode / sort / rank / score).  It is
    stored only when the call actually changes the vote state, so an
    idempotent re-send from a stale tab cannot overwrite what the original
    click recorded.  See :mod:`vtscore.datasets.vote_provenance`.

    Returns ``{"ok": true, "state": <new state>, "click_time": <int|null>}``
    so the client can reconcile its optimistic view directly from the
    response.
    """
    if get_media(media_id) is None:
        abort(404, message="not found")

    target = body["target"]

    try:
        region_box = _parse_region_box(body.get("region_box"))
    except ValueError as exc:
        abort(400, message=str(exc))
    if target != "good" and region_box is not None:
        abort(400, message="region_box is only valid on 'good' targets")

    try:
        provenance = normalize_provenance(body.get("provenance"))
    except ValueError as exc:
        abort(400, message=str(exc))

    _old, new_state, click_time = set_vote(media_id, target, region_box=region_box, provenance=provenance)

    # Persist the resulting labelset.  A failure here (e.g. ``os.replace``
    # EBUSY/ENOSPC under ``_write_detector``) leaves the in-memory vote
    # committed while the on-disk labelset stays at its prior value (audit
    # finding H30), so it is surfaced as a 500 with a clear message; the
    # in-memory mutation is rare enough to be acceptable since the next vote
    # re-merges and retries.  The write also schedules
    # the push to the detector's labelset source, if it has one.
    from vtscore.detectors.label_sync import sync_labels_to_loaded_detector

    try:
        sync_labels_to_loaded_detector()
    except Exception as exc:
        import logging

        logging.getLogger(__name__).exception("vote_media: detector label sync failed")
        abort(500, message=f"Failed to persist vote to detector store: {exc}")

    return {"ok": True, "state": new_state, "click_time": click_time}


@medias_bp.route("/api/medias/vote-bulk", methods=["POST"])
@medias_bp.arguments(MediaVoteBulkRequestSchema)
@medias_bp.response(200, MediaVoteBulkResponseSchema)
@medias_bp.alt_response(400, description="No ids supplied, or provenance is malformed.")
@medias_bp.alt_response(500, description="The votes could not be written to the detector store.")
@require_dataset_header
@require_detector_header
def vote_media_bulk(body: dict):
    """Apply one absolute vote target to many medias in a single request.

    Mirrors ``/api/medias/<id>/vote`` for a batch: each id is set to
    ``target`` with the same idempotent semantics (including Find-mode
    verification — a good/bad target marks the item verified), then the
    detector labelset is persisted **once** rather than per id.  Bulk votes
    are image-level (no region boxes).  Powers the Browser's "Verified Good" /
    "Verified Bad" actions, which mark a hand-selected set good/bad, verify
    them, and drop them from the browse.

    Ids that aren't in the loaded dataset are skipped and reported back in
    ``missing``; ``changed`` counts only the ids whose state actually moved
    (idempotent re-applies don't count).
    """
    target = body["target"]
    ids = body["ids"]
    if not ids:
        abort(400, message="No ids supplied")

    # A batch action over a hand-selected set is its own surfacing flow; the
    # client may refine it (with the sort it was selecting from), but never
    # has to.
    try:
        provenance = normalize_provenance(body.get("provenance") or {"flow": "bulk"})
    except ValueError as exc:
        abort(400, message=str(exc))

    changed = 0
    missing: list[int] = []
    for media_id in ids:
        if get_media(media_id) is None:
            missing.append(media_id)
            continue
        # Bulk "Verified Good/Bad" over a hand-selected set is not an
        # individual hand-click, so it must not build a Marathoner streak;
        # count_streak=False still credits the other vote achievements.
        old, new, _click_time = set_vote(media_id, target, count_streak=False, provenance=provenance)
        if old != new:
            changed += 1

    # Persist the resulting labelset once for the whole batch.  Mirrors the
    # single-vote route's H30 handling: a write failure surfaces as a 500
    # rather than leaving the in-memory votes silently un-persisted.
    from vtscore.detectors.label_sync import sync_labels_to_loaded_detector

    try:
        sync_labels_to_loaded_detector()
    except Exception as exc:
        import logging

        logging.getLogger(__name__).exception("vote_media_bulk: detector label sync failed")
        abort(500, message=f"Failed to persist votes to detector store: {exc}")

    return {"ok": True, "changed": changed, "missing": missing}
