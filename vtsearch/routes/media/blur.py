"""Blur routes: which of a dataset's items show blurred, and the Blur / Unblur override.

The state is per dataset, shared by everyone who can read it (see
:mod:`vtscore.datasets.blur_state`): an item's override if someone set one,
else whether any of the server's ``blur_rules`` matches its ingest metadata.
Both routes are confined to a dataset the current user may read.

The routes register on :data:`vtsearch.routes.media.list.medias_bp`.
"""

from __future__ import annotations

from flask import request
from flask_smorest import abort

from vtscore.datasets.blur_state import blurred_ids, set_blur_override
from vtscore.state.dataset_context import DatasetContext
from vtsearch.routes._context import MISSING_DATASET_HEADER, require_dataset_header
from vtsearch.routes.media.list import medias_bp
from vtsearch.schemas.media import MediaBlurredIdsResponseSchema, MediaBlurRequestSchema, MediaBlurResponseSchema

_NO_ACCESS = "The current user may not read this dataset."
_NOT_LOADED = "The dataset in ``X-Dataset-Id`` is not loaded."


def _readable_dataset() -> DatasetContext:
    """The requesting dataset's context, once the current user is known to be allowed to read it.

    Access is checked against the registry before the context is touched, so
    a user who may not read a dataset learns nothing about it, not even
    whether it is loaded. A context with no registry entry (one built in
    memory) has no owner to check against.
    """
    from vtscore.datasets.registry import can_user_access, get_dataset  # noqa: PLC0415
    from vtsearch.auth import get_current_user  # noqa: PLC0415
    from vtsearch.state import get_active_context  # noqa: PLC0415

    dataset_id = request.headers.get("X-Dataset-Id") or request.args.get("dataset_id") or ""
    if get_dataset(dataset_id) is not None and not can_user_access(dataset_id, get_current_user()):
        abort(403, message="Access denied")
    return get_active_context()


@medias_bp.route("/api/medias/blurred")
@medias_bp.response(200, MediaBlurredIdsResponseSchema)
@medias_bp.alt_response(400, description=MISSING_DATASET_HEADER)
@medias_bp.alt_response(403, description=_NO_ACCESS)
@medias_bp.alt_response(409, description=_NOT_LOADED)
@require_dataset_header
def list_blurred_media():
    """List the ids of the dataset's items that show blurred.

    One compact list for the whole dataset in ``X-Dataset-Id``, since most
    places an image is drawn hold a bare id and no metadata. An item is in it
    when its Blur / Unblur override says so, or, with no override, when any
    ``blur_rules`` rule matches its raw ingest metadata. Changing the rules
    re-decides every item without an override at once.
    """
    from vtsearch import settings  # noqa: PLC0415

    ctx = _readable_dataset()
    return {"ids": sorted(blurred_ids(ctx, settings.get_blur_rules()))}


@medias_bp.route("/api/medias/<int:media_id>/blur", methods=["PUT"])
@medias_bp.arguments(MediaBlurRequestSchema)
@medias_bp.response(200, MediaBlurResponseSchema)
@medias_bp.alt_response(400, description=MISSING_DATASET_HEADER)
@medias_bp.alt_response(403, description=_NO_ACCESS)
@medias_bp.alt_response(404, description="Media not found.")
@medias_bp.alt_response(409, description=_NOT_LOADED)
@require_dataset_header
def set_media_blur(body: dict, media_id: int):
    """Set one item's blur state for good: the Blur / Unblur button.

    Records ``blurred`` as the item's override, which wins over the
    ``blur_rules`` from then on in either direction, so it corrects a wrong
    "sensitive" ruling and a missed one alike. There is one state per dataset,
    shared by everyone who reads it, kept in a sidecar beside the dataset's
    file so it survives a restart. Sending the state the item already has is a
    no-op.
    """
    ctx = _readable_dataset()
    if media_id not in ctx.medias:
        abort(404, message="not found")
    blurred = body["blurred"]
    set_blur_override(ctx, media_id, blurred)
    return {"ok": True, "id": media_id, "blurred": blurred}
