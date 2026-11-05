"""Detector-sharing routes: the HTTP surface over a labelset source's sharing capability.

Endpoints
---------
GET    /api/detectors/sharing                     Whether sharing is on, and through which source.
GET    /api/detectors/sharing/received            Containers shared with the caller.
POST   /api/detectors/sharing/receive             Receive one as a read-only (linked) detector.
POST   /api/detectors/sharing/search-recipients   People and groups to share with.
GET    /api/detectors/registry/<id>/sharing       A detector's container: its state, recipients, public flag.
POST   /api/detectors/registry/<id>/share         Share a detector (creating its container); change who sees it.
DELETE /api/detectors/registry/<id>/share         Stop sharing it; it becomes a plain local detector.
POST   /api/detectors/registry/<id>/refresh       Bring a received detector level with its container.
POST   /api/detectors/registry/<id>/fork          An editable copy, owned by the caller.

Every route goes through the active sharing source
(:func:`vtscore.labels.sharing.active_sharing_source`), the one the
``detector_sharing_source`` server setting names.  While that setting is null
every route answers 404 ``error_code: "sharing_disabled"``, except two: the
capability route, which says sharing is off, and fork, which needs only the
local copy and is how a detector received before sharing was switched off is
made editable again.

Receive, refresh and fork delegate to the library's operations
(:func:`~vtscore.labels.sharing.receive_container`,
:func:`~vtscore.labels.sharing.refresh_linked_detector`,
:func:`~vtscore.labels.sharing.fork_detector`); share and unshare call the
source's ``share_*`` wrappers and keep the entry's binding.  Whatever a source
raises is answered by :func:`_abort_sharing_error`, the one mapping of its
typed errors to statuses.  The page for these routes is
``docs/api/detectors.md`` ("Detector sharing").
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, NoReturn

from flask_smorest import Blueprint, abort

from vtsearch.auth import get_current_user
from vtsearch.errors import (
    ALREADY_RECEIVED,
    DETECTOR_NOT_LINKED,
    DETECTOR_NOT_SHARED,
    REMOTE_NOT_FOUND,
    REMOTE_PERMISSION_DENIED,
    SHARING_DISABLED,
    SHARING_FAILED,
    SHARING_UNAVAILABLE,
    SHARING_UNSUPPORTED,
)
from vtsearch.routes._read_only import abort_read_only
from vtsearch.schemas.detectors import (
    DetectorForkRequestSchema,
    DetectorForkResponseSchema,
    DetectorShareRequestSchema,
    DetectorShareResponseSchema,
    DetectorSharingCapabilitySchema,
    DetectorSharingReceivedResponseSchema,
    DetectorSharingReceiveRequestSchema,
    DetectorSharingReceiveResponseSchema,
    DetectorSharingRefreshRequestSchema,
    DetectorSharingRefreshResponseSchema,
    DetectorSharingSearchRequestSchema,
    DetectorSharingSearchResponseSchema,
    DetectorSharingViewSchema,
    DetectorUnshareResponseSchema,
)

if TYPE_CHECKING:
    from vtscore.labels.sharing import ActiveSharingSource, SharedContainer, SharingError

logger = logging.getLogger(__name__)

detectors_sharing_bp = Blueprint(
    "detectors_sharing",
    __name__,
    description="Share detectors through the active sharing source: share, receive, refresh and fork.",
)

# What the source errors every route that reaches the source can answer mean
# (see ``_abort_sharing_error``); each route's ``alt_response`` names them
# beside its own meaning of the same status.
_E403 = "The sharing service refused the caller (``error_code: remote_permission_denied``)."
_E404 = "The sharing service has no such container for the caller (``error_code: remote_not_found``)."
_E409 = (
    "The detector's sharing source is not this server's active one (``error_code: sharing_unavailable``, "
    "the message saying why)."
)
_E501 = "The sharing source does not implement this operation (``error_code: sharing_unsupported``)."
_E502 = "The sharing service failed (``error_code: sharing_failed``)."
_OFF = "Detector sharing is off on this server (``error_code: sharing_disabled``)."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _abort_disabled() -> NoReturn:
    abort(404, message="Detector sharing is off on this server", error_code=SHARING_DISABLED)


def _require_active() -> ActiveSharingSource:
    """The active sharing source, or 404 ``sharing_disabled``."""
    from vtscore.labels.sharing import active_sharing_source  # noqa: PLC0415

    active = active_sharing_source()
    if active is None:
        _abort_disabled()
    return active


def _abort_sharing_error(exc: SharingError) -> NoReturn:
    """Answer the sharing error *exc*: the one mapping of a source's typed errors to statuses.

    ``RemoteNotFound`` is 404, ``RemotePermissionDenied`` 403,
    ``SharingUnavailable`` (the detector's binding names a source other than
    the active one; the message is its ``sharing_note``) 409,
    ``SharingUnsupported`` 501, and any other ``SharingError`` - the service
    is unreachable or answered nonsense - 502.
    """
    from vtscore.labels.sharing import (  # noqa: PLC0415
        RemoteNotFound,
        RemotePermissionDenied,
        SharingUnavailable,
        SharingUnsupported,
    )

    message = str(exc) or type(exc).__name__
    if isinstance(exc, RemoteNotFound):
        abort(404, message=message, error_code=REMOTE_NOT_FOUND)
    if isinstance(exc, RemotePermissionDenied):
        abort(403, message=message, error_code=REMOTE_PERMISSION_DENIED)
    if isinstance(exc, SharingUnavailable):
        abort(409, message=message, error_code=SHARING_UNAVAILABLE)
    if isinstance(exc, SharingUnsupported):
        abort(501, message=message, error_code=SHARING_UNSUPPORTED)
    abort(502, message=f"The sharing service failed: {message}", error_code=SHARING_FAILED)


def _entry_or_404(detector_id: str) -> dict[str, Any]:
    from vtscore.detectors.registry import get_detector  # noqa: PLC0415

    entry = get_detector(detector_id)
    if entry is None:
        abort(404, message="Detector not found")
    return entry


def _require_access(entry: dict[str, Any]) -> None:
    from vtscore.detectors.registry import can_user_access_detector  # noqa: PLC0415

    if not can_user_access_detector(entry["id"], get_current_user()):
        abort(403, message="You do not have access to this detector")


def _require_creator(entry: dict[str, Any], action: str) -> None:
    from vtscore.detectors.registry import is_detector_owner  # noqa: PLC0415

    if not is_detector_owner(entry["id"], get_current_user()):
        abort(403, message=f"Only the detector's creator can {action}")


def _require_matching_binding(entry: dict[str, Any]) -> dict[str, Any]:
    """The entry's sharing binding when it belongs to the active source; 404 when it shares nothing, else 409."""
    from vtscore.labels.binding import binding_matches_active_source, sharing_binding, sharing_note  # noqa: PLC0415

    binding = sharing_binding(entry)
    if binding is None:
        abort(404, message=f"Detector '{entry.get('name', '')}' is not shared", error_code=DETECTOR_NOT_SHARED)
    if not binding_matches_active_source(entry):
        abort(
            409,
            message=sharing_note(entry) or "This detector's sharing source is not available on this server.",
            error_code=SHARING_UNAVAILABLE,
            detector_id=entry["id"],
        )
    return binding


def _field_values(binding: dict[str, Any] | None, active: ActiveSharingSource) -> dict[str, Any]:
    """The values a source is handed for *binding*: its own, with the setting's fixed ones laid over."""
    from vtscore.labels.sharing import _field_values as _laid_over  # noqa: PLC0415

    return _laid_over(binding or {}, active)


@contextmanager
def _as_detector(entry: dict[str, Any]) -> Iterator[None]:
    """Resolve the source's ``{detector_*}`` templates to *entry*, as the push engine does for it."""
    from vtscore.labels.sync import _template_context  # noqa: PLC0415
    from vtscore.state.core import override_detector_context  # noqa: PLC0415

    with override_detector_context(_template_context(entry.get("name", ""), entry["id"])):
        yield


def _entry_view(entry: dict[str, Any]) -> dict[str, Any]:
    """*entry* with its sharing state as ``GET /api/detectors/registry`` lists it, never the binding's field values."""
    from vtscore.detectors.registry import sync_mode_of  # noqa: PLC0415
    from vtscore.labels.binding import sharing_binding, sharing_note  # noqa: PLC0415
    from vtsearch.routes.detectors.registry.listing import _sharing_state  # noqa: PLC0415

    view = {k: v for k, v in entry.items() if k != "labelset_source"}
    view["sync_mode"] = sync_mode_of(entry)
    view["sharing"] = _sharing_state(sharing_binding(entry))
    view["sharing_note"] = sharing_note(entry)
    view["forked_from"] = entry.get("forked_from") or None
    view.setdefault("created_by", "default")
    view.setdefault("readers", [])
    view["is_owner"] = view["created_by"] == get_current_user()
    return view


def _held_by_caller(active: ActiveSharingSource) -> dict[str, str]:
    """``{remote_ref: detector id}`` for the caller's detectors that hold a container of the active source.

    The rule :func:`~vtscore.labels.sharing.receive_container` refuses a
    second receive by, so the listing's ``detector_id`` and the 409 agree.
    """
    from vtscore.detectors.registry import list_detectors  # noqa: PLC0415
    from vtscore.labels.binding import sharing_binding  # noqa: PLC0415

    user = get_current_user()
    held: dict[str, str] = {}
    for entry in list_detectors():
        binding = sharing_binding(entry)
        if (
            binding is not None
            and entry.get("created_by", "default") == user
            and binding["source_name"].strip() == active.source_name
        ):
            held.setdefault(str(binding["remote_ref"]), entry["id"])
    return held


def _integrity_message(name: str, owner: str, integrity: dict[str, Any]) -> str:
    return (
        f"'{name}' has {integrity['count']} change(s) not made by its owner, {owner}. "
        "Nothing was changed: choose whether to import those changes or skip them."
    )


def _sharing_view(entry: dict[str, Any], container: SharedContainer | None) -> dict[str, Any]:
    """The ``GET .../sharing`` body for *entry*, with *container* (``None`` when it was not read)."""
    from vtscore.detectors.registry import is_linked, sync_mode_of  # noqa: PLC0415
    from vtscore.labels.binding import sharing_binding, sharing_note  # noqa: PLC0415
    from vtsearch.routes.detectors.registry.listing import _sharing_state  # noqa: PLC0415

    return {
        "detector_id": entry["id"],
        "sync_mode": sync_mode_of(entry),
        "is_owner": not is_linked(entry) and entry.get("created_by", "default") == get_current_user(),
        "sharing": _sharing_state(sharing_binding(entry)),
        "sharing_note": sharing_note(entry),
        "owner": container.owner if container is not None else None,
        "recipients": list(container.recipients) if container is not None else None,
        "public": container.public if container is not None else None,
    }


def _read_container(entry: dict[str, Any], active: ActiveSharingSource, binding: dict[str, Any]) -> SharedContainer:
    """Refresh *entry*'s container through *active*, as the caller; a source error is answered."""
    from vtscore.labels.sharing import SharingError  # noqa: PLC0415

    try:
        with _as_detector(entry):
            return active.source.share_refresh(str(binding["remote_ref"]), _field_values(binding, active))
    except SharingError as exc:
        _abort_sharing_error(exc)


# ---------------------------------------------------------------------------
# GET /api/detectors/sharing
# ---------------------------------------------------------------------------


@detectors_sharing_bp.route("/api/detectors/sharing", methods=["GET"])
@detectors_sharing_bp.response(200, DetectorSharingCapabilitySchema)
def get_sharing_capability():
    """Whether detector sharing is on here, and through which source.

    Always 200: ``enabled`` is false while the ``detector_sharing_source``
    setting is null (or names a source that is not installed or cannot
    share), and the dashboard hides its sharing surfaces.
    """
    from vtscore.labels.sharing import active_sharing_source  # noqa: PLC0415

    active = active_sharing_source()
    if active is None:
        return {
            "enabled": False,
            "name": None,
            "display_name": None,
            "icon": None,
            "supports_recipient_search": False,
            "missing_fields": [],
        }
    source = active.source
    return {
        "enabled": True,
        "name": active.source_name,
        "display_name": source.display_name or active.source_name,
        "icon": source.icon or "",
        "supports_recipient_search": source.supports_recipient_search,
        "missing_fields": list(active.missing_fields),
    }


# ---------------------------------------------------------------------------
# GET /api/detectors/sharing/received
# ---------------------------------------------------------------------------


@detectors_sharing_bp.route("/api/detectors/sharing/received", methods=["GET"])
@detectors_sharing_bp.response(200, DetectorSharingReceivedResponseSchema)
@detectors_sharing_bp.alt_response(403, description=_E403)
@detectors_sharing_bp.alt_response(404, description=f"{_OFF} {_E404}")
@detectors_sharing_bp.alt_response(409, description=_E409)
@detectors_sharing_bp.alt_response(501, description=_E501)
@detectors_sharing_bp.alt_response(502, description=_E502)
def list_received_detectors():
    """The containers shared with the caller (directly, through a group, or publicly) that they do not own.

    Each carries ``detector_id``: the caller's detector that already holds it,
    or null when they have not received it yet.
    """
    from vtscore.labels.sharing import SharingError  # noqa: PLC0415

    active = _require_active()
    try:
        summaries = active.source.share_list_received(_field_values(None, active))
    except SharingError as exc:
        _abort_sharing_error(exc)
    held = _held_by_caller(active)
    return {
        "received": [
            {
                "remote_ref": s.remote_ref,
                "name": s.name,
                "owner": s.owner,
                "media_type": s.media_type,
                "record_count": s.record_count,
                "updated_at": s.updated_at,
                "shared_via": s.shared_via,
                "detector_id": held.get(s.remote_ref),
            }
            for s in summaries
        ]
    }


# ---------------------------------------------------------------------------
# POST /api/detectors/sharing/receive
# ---------------------------------------------------------------------------


@detectors_sharing_bp.route("/api/detectors/sharing/receive", methods=["POST"])
@detectors_sharing_bp.arguments(DetectorSharingReceiveRequestSchema)
@detectors_sharing_bp.response(201, DetectorSharingReceiveResponseSchema)
@detectors_sharing_bp.alt_response(
    200,
    schema=DetectorSharingReceiveResponseSchema,
    description=(
        "The ``abort`` policy met records someone other than the owner wrote: ``applied: false``, nothing "
        "was created, and ``integrity`` says how many, so the client can ask whether to import or skip them."
    ),
)
@detectors_sharing_bp.alt_response(403, description=_E403)
@detectors_sharing_bp.alt_response(404, description=f"{_OFF} {_E404}")
@detectors_sharing_bp.alt_response(
    409,
    description=(
        "The caller already holds this container (``error_code: already_received``; ``detector_id`` names "
        f"their copy). {_E409}"
    ),
)
@detectors_sharing_bp.alt_response(501, description=_E501)
@detectors_sharing_bp.alt_response(502, description=_E502)
def receive_shared_detector(body: dict):
    """Make a container shared with the caller a local, read-only (``linked``) detector of theirs.

    Its labels are the container's records, under the ``foreign`` policy; its
    name is the container's, suffixed when another detector holds it.  A
    received detector changes only by refresh, and can be forked.
    """
    from vtscore.labels.sharing import ForeignModifications, SharingError, SharingUnavailable  # noqa: PLC0415
    from vtscore.labels.sharing import receive_container  # noqa: PLC0415

    active = _require_active()
    remote_ref = body["remote_ref"]
    held = _held_by_caller(active).get(remote_ref)
    if held is not None:
        abort(
            409,
            message="You already have this shared detector",
            error_code=ALREADY_RECEIVED,
            detector_id=held,
        )
    try:
        entry = receive_container(remote_ref, foreign=body["foreign"])
    except ForeignModifications as exc:
        container = exc.container
        return {
            "ok": True,
            "applied": False,
            "message": _integrity_message(container.name, container.owner, exc.integrity),
            "integrity": exc.integrity,
            "remote_ref": remote_ref,
            "name": container.name,
            "owner": container.owner,
            "detector": None,
        }, 200
    except SharingUnavailable:
        _abort_disabled()  # sharing was switched off since the check above
    except SharingError as exc:
        _abort_sharing_error(exc)
    except ValueError as exc:
        # A receive that raced this one: the check above missed it.
        abort(409, message=str(exc), error_code=ALREADY_RECEIVED, detector_id=_held_by_caller(active).get(remote_ref))
    binding = entry["labelset_source"]
    return {
        "ok": True,
        "applied": True,
        "message": f"Received '{entry['name']}' from {binding.get('remote_owner') or 'its owner'}.",
        "integrity": binding["integrity"],
        "remote_ref": remote_ref,
        "name": entry["name"],
        "owner": binding.get("remote_owner") or "",
        "detector": _entry_view(entry),
    }


# ---------------------------------------------------------------------------
# POST /api/detectors/sharing/search-recipients
# ---------------------------------------------------------------------------


@detectors_sharing_bp.route("/api/detectors/sharing/search-recipients", methods=["POST"])
@detectors_sharing_bp.arguments(DetectorSharingSearchRequestSchema)
@detectors_sharing_bp.response(200, DetectorSharingSearchResponseSchema)
@detectors_sharing_bp.alt_response(403, description=_E403)
@detectors_sharing_bp.alt_response(404, description=f"{_OFF} {_E404}")
@detectors_sharing_bp.alt_response(409, description=_E409)
@detectors_sharing_bp.alt_response(
    501,
    description=(
        "The active sharing source has no recipient search (``error_code: sharing_unsupported``): "
        "``supports_recipient_search`` is false on ``GET /api/detectors/sharing``."
    ),
)
@detectors_sharing_bp.alt_response(502, description=_E502)
def search_sharing_recipients(body: dict):
    """People and groups matching ``query``, for a share dialog's recipient picker.

    An empty query returns no suggestions without asking the source.
    """
    from vtscore.labels.sharing import SharingError  # noqa: PLC0415

    active = _require_active()
    if not active.source.supports_recipient_search:
        abort(
            501,
            message=f"{active.source.display_name or active.source_name} cannot search for people to share with",
            error_code=SHARING_UNSUPPORTED,
        )
    try:
        found = active.source.share_search_recipients(body["query"], _field_values(None, active))
    except SharingError as exc:
        _abort_sharing_error(exc)
    return {"suggestions": [{"id": s.id, "display_name": s.display_name, "kind": s.kind} for s in found]}


# ---------------------------------------------------------------------------
# GET /api/detectors/registry/<detector_id>/sharing
# ---------------------------------------------------------------------------


@detectors_sharing_bp.route("/api/detectors/registry/<detector_id>/sharing", methods=["GET"])
@detectors_sharing_bp.response(200, DetectorSharingViewSchema)
@detectors_sharing_bp.alt_response(403, description=f"The caller cannot access this detector. {_E403}")
@detectors_sharing_bp.alt_response(
    404,
    description=(
        f"{_OFF} Or the detector does not exist, or shares nothing (``error_code: detector_not_shared``). {_E404}"
    ),
)
@detectors_sharing_bp.alt_response(409, description=_E409)
@detectors_sharing_bp.alt_response(501, description=_E501)
@detectors_sharing_bp.alt_response(502, description=_E502)
def get_detector_sharing(detector_id: str):
    """A shared or received detector's container: its state, who it is shared with, and whether publicly.

    ``recipients`` and ``public`` are read from the container now.  When the
    detector's sharing source is not the active one, ``sharing_note`` says why
    and they are null: the container is not read.
    """
    from vtscore.labels.binding import binding_matches_active_source, sharing_binding  # noqa: PLC0415

    active = _require_active()
    entry = _entry_or_404(detector_id)
    _require_access(entry)
    binding = sharing_binding(entry)
    if binding is None:
        abort(404, message=f"Detector '{entry.get('name', '')}' is not shared", error_code=DETECTOR_NOT_SHARED)
    if not binding_matches_active_source(entry):
        return _sharing_view(entry, None)
    return _sharing_view(entry, _read_container(entry, active, binding))


# ---------------------------------------------------------------------------
# POST /api/detectors/registry/<detector_id>/share
# ---------------------------------------------------------------------------


def _create_container(entry: dict[str, Any], active: ActiveSharingSource) -> tuple[dict[str, Any], str]:
    """Create *entry*'s container from its file and bind it: ``(the entry as stored, what that replaced)``.

    The second item names the plain auto-export the binding replaces (an own
    binding or the deployment default), or is empty.  The push engine keeps
    the container level from here on: the detector's role is now owner.
    """
    from vtscore.detectors.registry import get_detector, update_detector  # noqa: PLC0415
    from vtscore.detectors.store import _detector_path, _read_detector, flush_detector_writes  # noqa: PLC0415
    from vtscore.labels.binding import resolve_labelset_source  # noqa: PLC0415
    from vtscore.labels.sharing import SharingError, share_payload  # noqa: PLC0415
    from vtscore.labels.sources import get_labelset_source  # noqa: PLC0415

    path = _detector_path(entry["name"])
    flush_detector_writes(path)
    data = _read_detector(path) or {"name": entry["name"], "media_type": entry.get("media_type", "")}
    data["name"] = entry["name"]
    labelset, meta = share_payload(data)
    try:
        with _as_detector(entry):
            remote_ref = active.source.share_create(labelset, meta, _field_values(None, active))
    except SharingError as exc:
        _abort_sharing_error(exc)

    replaced = ""
    plain = resolve_labelset_source(entry)
    if plain is not None:
        source = get_labelset_source(plain.source_name)
        replaced = (source.display_name if source is not None else "") or plain.source_name
    update_detector(
        entry["id"],
        labelset_source={
            "source_name": active.source_name,
            "field_values": {},
            "remote_ref": remote_ref,
            "last_synced": time.time(),
            "sync_error": None,
        },
    )
    return get_detector(entry["id"]) or entry, replaced


@detectors_sharing_bp.route("/api/detectors/registry/<detector_id>/share", methods=["POST"])
@detectors_sharing_bp.arguments(DetectorShareRequestSchema)
@detectors_sharing_bp.response(200, DetectorShareResponseSchema)
@detectors_sharing_bp.alt_response(403, description=f"Only the detector's creator can share it. {_E403}")
@detectors_sharing_bp.alt_response(404, description=f"{_OFF} Or the detector does not exist. {_E404}")
@detectors_sharing_bp.alt_response(
    409,
    description=(
        f"The detector is a received, read-only copy (``error_code: detector_read_only``); fork it to share it. {_E409}"
    ),
)
@detectors_sharing_bp.alt_response(501, description=_E501)
@detectors_sharing_bp.alt_response(502, description=_E502)
def share_detector(body: dict, detector_id: str):
    """Share a detector of the caller's, and change who it is shared with.

    The first call creates its container from the detector's labels and binds
    the detector to it; a plain auto-export binding (its own, or the
    deployment default it inherited) is replaced, and the message says so.
    From then on every change to its labels is pushed to the container.  Then
    ``remove``, ``add`` and ``public`` are applied, in that order.  The
    response is the container as ``GET .../sharing`` shows it, read afterwards.
    """
    from vtscore.detectors.registry import is_linked  # noqa: PLC0415
    from vtscore.labels.binding import sharing_binding  # noqa: PLC0415
    from vtscore.labels.sharing import SharingError  # noqa: PLC0415

    active = _require_active()
    entry = _entry_or_404(detector_id)
    _require_creator(entry, "share it")
    if is_linked(entry):
        abort_read_only(entry)

    created, replaced = False, ""
    if sharing_binding(entry) is None:
        entry, replaced = _create_container(entry, active)
        created = True
    binding = _require_matching_binding(entry)
    remote_ref = str(binding["remote_ref"])
    values = _field_values(binding, active)
    changes = []
    try:
        with _as_detector(entry):
            if body["remove"]:
                active.source.share_remove_recipients(remote_ref, body["remove"], values)
                changes.append(f"removed {len(body['remove'])} recipient(s)")
            if body["add"]:
                active.source.share_add_recipients(remote_ref, body["add"], values)
                changes.append(f"added {len(body['add'])} recipient(s)")
            if body["public"] is not None:
                active.source.share_set_public(remote_ref, body["public"], values)
                changes.append("made it public" if body["public"] else "made it private")
    except SharingError as exc:
        _abort_sharing_error(exc)

    label = active.source.display_name or active.source_name
    parts = [f"Shared '{entry['name']}' through {label}." if created else f"'{entry['name']}' is shared."]
    if replaced:
        parts.append(f"Its labels are no longer auto-exported to {replaced}.")
    if changes:
        summary = ", ".join(changes)
        parts.append(f"{summary[0].upper()}{summary[1:]}.")
    view = _sharing_view(entry, _read_container(entry, active, binding))
    return {**view, "ok": True, "created": created, "message": " ".join(parts)}


# ---------------------------------------------------------------------------
# DELETE /api/detectors/registry/<detector_id>/share
# ---------------------------------------------------------------------------


@detectors_sharing_bp.route("/api/detectors/registry/<detector_id>/share", methods=["DELETE"])
@detectors_sharing_bp.response(200, DetectorUnshareResponseSchema)
@detectors_sharing_bp.alt_response(403, description=f"Only the detector's creator can stop sharing it. {_E403}")
@detectors_sharing_bp.alt_response(
    404,
    description=f"{_OFF} Or the detector does not exist, or shares nothing (``error_code: detector_not_shared``).",
)
@detectors_sharing_bp.alt_response(
    409,
    description=(
        "The detector is a received copy (``error_code: detector_read_only``): delete it instead, which "
        f"removes only the local copy. {_E409}"
    ),
)
@detectors_sharing_bp.alt_response(501, description=_E501)
@detectors_sharing_bp.alt_response(502, description=_E502)
def unshare_detector(detector_id: str):
    """Stop sharing a detector: archive its container and unbind it.

    The detector keeps its labels and becomes a plain local one; recipients'
    copies stop receiving updates.  A container the service no longer has is
    unbound all the same.
    """
    from vtscore.detectors.registry import get_detector, is_linked, update_detector  # noqa: PLC0415
    from vtscore.labels.sharing import RemoteNotFound, SharingError  # noqa: PLC0415

    active = _require_active()
    entry = _entry_or_404(detector_id)
    _require_creator(entry, "stop sharing it")
    if is_linked(entry):
        abort_read_only(entry)
    binding = _require_matching_binding(entry)
    label = active.source.display_name or active.source_name
    message = f"Stopped sharing '{entry['name']}' through {label}."
    try:
        with _as_detector(entry):
            active.source.share_archive(str(binding["remote_ref"]), _field_values(binding, active))
    except RemoteNotFound:
        message = f"'{entry['name']}' was no longer shared through {label}; it is now local only."
    except SharingError as exc:
        _abort_sharing_error(exc)
    update_detector(entry["id"], labelset_source=None)
    return {"ok": True, "message": message, "detector": _entry_view(get_detector(entry["id"]) or entry)}


# ---------------------------------------------------------------------------
# POST /api/detectors/registry/<detector_id>/refresh
# ---------------------------------------------------------------------------


@detectors_sharing_bp.route("/api/detectors/registry/<detector_id>/refresh", methods=["POST"])
@detectors_sharing_bp.arguments(DetectorSharingRefreshRequestSchema)
@detectors_sharing_bp.response(200, DetectorSharingRefreshResponseSchema)
@detectors_sharing_bp.alt_response(403, description=f"Only the detector's creator can refresh it. {_E403}")
@detectors_sharing_bp.alt_response(404, description=f"{_OFF} Or the detector does not exist. {_E404}")
@detectors_sharing_bp.alt_response(
    409,
    description=f"The detector is not a received copy (``error_code: detector_not_linked``). {_E409}",
)
@detectors_sharing_bp.alt_response(501, description=_E501)
@detectors_sharing_bp.alt_response(502, description=_E502)
def refresh_detector(body: dict, detector_id: str):
    """Bring a received detector level with its container ("Check for updates").

    The local labels are replaced by the container's, under the ``foreign``
    policy; ``abort`` (the default) changes nothing when someone other than the
    owner wrote to it, and says so (``applied: false``), recording the
    integrity on the detector so the listing can warn.
    """
    from vtscore.detectors.registry import get_detector, is_linked  # noqa: PLC0415
    from vtscore.labels.sharing import SharingError, refresh_linked_detector  # noqa: PLC0415

    _require_active()
    entry = _entry_or_404(detector_id)
    _require_creator(entry, "refresh it")
    if not is_linked(entry):
        abort(
            409,
            message=f"Detector '{entry.get('name', '')}' is not a received detector",
            error_code=DETECTOR_NOT_LINKED,
        )
    try:
        result = refresh_linked_detector(detector_id, foreign=body["foreign"])
    except LookupError:
        abort(404, message="Detector not found")
    except SharingError as exc:
        _abort_sharing_error(exc)

    fresh = get_detector(detector_id) or entry
    owner = (fresh.get("labelset_source") or {}).get("remote_owner") or "its owner"
    integrity = dict(result.integrity)
    if not result.applied:
        message = _integrity_message(fresh["name"], owner, integrity)
    elif result.added or result.changed or result.removed:
        message = (
            f"Updated '{fresh['name']}': {result.added} label(s) added, {result.changed} changed, "
            f"{result.removed} removed."
        )
    else:
        message = f"'{fresh['name']}' is up to date."
    return {
        "ok": True,
        "applied": result.applied,
        "message": message,
        "integrity": integrity,
        "added": result.added,
        "changed": result.changed,
        "removed": result.removed,
        "detector": _entry_view(fresh),
    }


# ---------------------------------------------------------------------------
# POST /api/detectors/registry/<detector_id>/fork
# ---------------------------------------------------------------------------


@detectors_sharing_bp.route("/api/detectors/registry/<detector_id>/fork", methods=["POST"])
@detectors_sharing_bp.arguments(DetectorForkRequestSchema)
@detectors_sharing_bp.response(201, DetectorForkResponseSchema)
@detectors_sharing_bp.alt_response(403, description="The caller cannot access this detector.")
@detectors_sharing_bp.alt_response(404, description="The detector does not exist, or has no labelset file.")
@detectors_sharing_bp.alt_response(409, description="A detector with that name already exists.")
def fork_registered_detector(body: dict, detector_id: str):
    """Copy a detector into a new, editable one owned by the caller.

    Works on any detector the caller can see, and is how a received one is
    made editable: its local copy is what is forked.  The fork is ``forked``,
    with ``forked_from`` provenance.  When sharing is on, a container is made
    for it too, so it is shared from the start (with nobody yet); when that is
    not possible - sharing is off, or creating the container failed -
    ``remote_sync`` says why and the fork is local only.  Works while sharing
    is off.
    """
    from vtscore.detectors.store import _detector_path, flush_detector_writes  # noqa: PLC0415
    from vtscore.labels.sharing import active_sharing_source, fork_detector  # noqa: PLC0415

    entry = _entry_or_404(detector_id)
    _require_access(entry)
    name = (body.get("name") or "").strip() or None
    flush_detector_writes(_detector_path(entry["name"]))
    try:
        forked = fork_detector(detector_id, name=name)
    except LookupError as exc:
        abort(404, message=str(exc) or "Detector not found")
    except ValueError as exc:
        abort(409, message=str(exc))

    active = active_sharing_source()
    share_error = forked.get("share_error")
    if active is None:
        remote_sync = "Detector sharing is off on this server, so this fork is local only."
    elif share_error:
        label = active.source.display_name or active.source_name
        remote_sync = f"This fork could not be shared through {label} ({share_error}), so it is local only."
    else:
        remote_sync = None
    return {
        "ok": True,
        "message": f"Forked '{entry['name']}' as '{forked['name']}'.",
        "detector": _entry_view(forked),
        "remote_sync": remote_sync,
    }
