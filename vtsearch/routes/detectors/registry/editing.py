"""Detector-registry routes that edit a registered detector's entry.

Endpoints
---------
PUT    /api/detectors/registry/<id>/rename            Rename a registered detector.
PUT    /api/detectors/registry/<id>/autofind          Toggle the detector's AutoFind flag.
PUT    /api/detectors/registry/<id>/readers           Set who can see the detector.
"""

from __future__ import annotations

import logging

from flask_smorest import abort

from vtsearch.auth import get_current_user
from vtscore.detectors.store import (
    _detector_path,
    _read_detector,
    _write_detector,
)
from vtsearch.routes.detectors.registry._common import _abort_if_name_taken, detectors_registry_bp
from vtsearch.schemas.detectors import (
    DetectorRegistryAutofindRequestSchema,
    DetectorRegistryAutofindResponseSchema,
    DetectorRegistryReadersRequestSchema,
    DetectorRegistryReadersResponseSchema,
    DetectorRegistryRenameRequestSchema,
    DetectorRegistryRenameResponseSchema,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PUT /api/detectors/registry/<detector_id>/rename
# ---------------------------------------------------------------------------


@detectors_registry_bp.route("/api/detectors/registry/<detector_id>/rename", methods=["PUT"])
@detectors_registry_bp.arguments(DetectorRegistryRenameRequestSchema)
@detectors_registry_bp.response(200, DetectorRegistryRenameResponseSchema)
@detectors_registry_bp.alt_response(400, description="Empty name after stripping.")
@detectors_registry_bp.alt_response(403, description="Only the detector creator can rename it.")
@detectors_registry_bp.alt_response(404, description="Detector not found.")
@detectors_registry_bp.alt_response(409, description="A detector with the new name already exists.")
def rename_registered_detector(body: dict, detector_id: str):
    """Rename a registered detector and its on-disk labelset file.

    A ``{detector_name}``-templated labelset-source file follows the rename
    (see :func:`vtscore.detectors.labelset_ops.follow_rename_with_labelset_file`).
    """
    from vtscore.detectors.labelset_ops import follow_rename_with_labelset_file
    from vtscore.detectors.registry import get_detector, is_detector_owner, rename_detector
    from vtscore.state.core import get_detector_context

    new_name = body["name"].strip()
    if not new_name:
        abort(400, message="name is required")

    entry = get_detector(detector_id)
    if entry is None:
        abort(404, message="Detector not found")
    if not is_detector_owner(detector_id, get_current_user()):
        abort(403, message="Only the detector creator can rename it")

    old_name = entry.get("name", "")
    # Without this guard the write below would overwrite the target detector's
    # labelset file and leave both registry entries resolving to it.  A rename
    # that only re-spells its own name keeps its own file, so it is allowed.
    _abort_if_name_taken(new_name, own_path=_detector_path(old_name), exclude_id=detector_id)

    if old_name and old_name != new_name:
        # Before the write below, which schedules a push to the new name's
        # path: a push landing first would leave the old file behind.
        follow_rename_with_labelset_file(detector_id, old_name=old_name, new_name=new_name)

        old_path = _detector_path(old_name)
        det_data = _read_detector(old_path)
        if det_data:
            new_path = _detector_path(new_name)
            det_data["name"] = new_name
            _write_detector(new_path, det_data)
            if new_path != old_path:
                old_path.unlink(missing_ok=True)

        # Rename AutoFind entry if present.
        try:
            from vtsearch.settings import get_autofind_detectors, set_autofind_detectors

            current = get_autofind_detectors()
            if old_name in current:
                current = [new_name if n == old_name else n for n in current]
                set_autofind_detectors(current)
        except Exception:
            logger.exception("Failed to rename AutoFind entry for %s", detector_id)

        # Update the loaded in-memory context so future syncs use the new
        # name in {detector_name} template substitution.
        ctx = get_detector_context(detector_id)
        if ctx is not None:
            ctx.name = new_name

    rename_detector(detector_id, new_name)
    # The write above scheduled its push before the registry knew the new
    # name, so it could not see an explicit binding; schedule it again now.
    from vtscore.labels.sync import schedule_labelset_push

    schedule_labelset_push(new_name, detector_id=detector_id)
    return {"ok": True, "name": new_name}


# ---------------------------------------------------------------------------
# PUT /api/detectors/registry/<detector_id>/autofind
# ---------------------------------------------------------------------------


@detectors_registry_bp.route("/api/detectors/registry/<detector_id>/autofind", methods=["PUT"])
@detectors_registry_bp.arguments(DetectorRegistryAutofindRequestSchema)
@detectors_registry_bp.response(200, DetectorRegistryAutofindResponseSchema)
@detectors_registry_bp.alt_response(403, description="Access denied for the current user.")
@detectors_registry_bp.alt_response(404, description="Detector not found.")
@detectors_registry_bp.alt_response(500, description="Detector has no associated name.")
def set_detector_autofind(body: dict, detector_id: str):
    """Toggle the calling user's AutoFind flag for a registered detector.

    The flag is stored per-user under ``autofind_detectors`` (see
    :func:`vtsearch.settings.add_autofind_detector`), so each user curates their
    own AutoFind list. The CLI's ``--autodetect`` flow reads the running
    user's list (the built-in ``default`` user falls back to the server
    settings file). The user must be able to access the detector to flag it.
    """
    from vtscore.detectors.registry import can_user_access_detector, get_detector
    from vtsearch.settings import (
        add_autofind_detector,
        remove_autofind_detector,
    )

    entry = get_detector(detector_id)
    if entry is None:
        abort(404, message="Detector not found")
    if not can_user_access_detector(detector_id, get_current_user()):
        abort(403, message="You do not have access to this detector")

    flag = body["autofind"]

    name = entry.get("name", "")
    if not name:
        abort(500, message="Detector has no name")

    if flag:
        add_autofind_detector(name)
    else:
        remove_autofind_detector(name)
    return {"ok": True, "autofind": flag}


# ---------------------------------------------------------------------------
# PUT /api/detectors/registry/<detector_id>/readers
# ---------------------------------------------------------------------------


@detectors_registry_bp.route("/api/detectors/registry/<detector_id>/readers", methods=["PUT"])
@detectors_registry_bp.arguments(DetectorRegistryReadersRequestSchema)
@detectors_registry_bp.response(200, DetectorRegistryReadersResponseSchema)
@detectors_registry_bp.alt_response(403, description="Only the detector creator can update readers.")
@detectors_registry_bp.alt_response(404, description="Detector not found.")
def update_detector_readers(body: dict, detector_id: str):
    """Update a detector's access list. Only the creator may call this.

    Body: ``{"readers": ["alice", "bob"]}``. Use ``["*"]`` to make the
    detector visible to all users. Mirrors the dataset readers endpoint.
    """
    from vtscore.detectors.registry import update_detector_readers as _reg_update_readers
    from vtscore.security.readers import ReadersUpdate

    readers = body["readers"]
    outcome = _reg_update_readers(detector_id, readers, get_current_user())
    if outcome is ReadersUpdate.NOT_CREATOR:
        abort(403, message=outcome.message("detector"))
    if outcome is ReadersUpdate.NOT_FOUND:
        abort(404, message=outcome.message("detector"))
    return {"ok": True, "readers": readers}
