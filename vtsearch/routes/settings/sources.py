"""Flask routes for the Sync Sources API.

Migrated to ``flask_smorest`` so these routes appear in
``/api/openapi.json``.

Endpoints
---------
Settings sources:
    GET  /api/settings-sources              (list available source plugins)
    GET  /api/settings-sources/active       (get the active source config)
    PUT  /api/settings-sources/active       (set or clear the active source)
    POST /api/settings-sources/sync         (force import from source)

Labelset sources:
    GET  /api/labelset-sources                       (list available source plugins)
    GET  /api/labelset-sources/default               (the deployment-wide default, or null)
    GET  /api/detectors/<name>/labelset-source       (get detector's source)
    PUT  /api/detectors/<name>/labelset-source       (set or clear source)
    POST /api/detectors/<name>/labelset-source/sync  (force import from source)

Schema-level validation failures surface as 422 with the standard
``errors`` envelope; handler-level rejects (unknown source, unknown
detector) keep their HTTP codes (404) with the standard ``message``
envelope.  Sending ``{}`` (empty body) clears the active source.
"""

from __future__ import annotations

from flask import jsonify
from flask_smorest import Blueprint, abort

from vtsearch.schemas.settings_io import (
    DefaultLabelsetSourceSchema,
    DetectorLabelsetSourceSchema,
    OkMessageSchema,
    SetSyncSourceRequestSchema,
    SyncFromLabelsetSourceResponseSchema,
    SyncFromSourceResponseSchema,
    SyncSourceConfigSchema,
    SyncSourceEntrySchema,
)

sync_sources_bp = Blueprint(
    "sync_sources",
    __name__,
    description="Manage the active settings / labelset sync sources.",
)


# ---------------------------------------------------------------------------
# Settings sources
# ---------------------------------------------------------------------------


@sync_sources_bp.route("/api/settings-sources", methods=["GET"])
@sync_sources_bp.response(200, SyncSourceEntrySchema(many=True))
def list_settings_sources_route():
    """Return a list of all registered settings source plugins."""
    from vtsearch.settings import filter_visible_plugins
    from vtsearch.settings_io.sources import list_settings_sources

    return [src.to_dict() for src in filter_visible_plugins("settings_sources", list_settings_sources())]


@sync_sources_bp.route("/api/settings-sources/active", methods=["GET"])
@sync_sources_bp.response(200, SyncSourceConfigSchema)
def get_active_settings_source():
    """Return the active settings source config (with ``inherited``), or null.

    ``inherited`` is ``true`` when the effective source is the deployment-wide
    ``default_settings_source`` rather than the user's own key. A user who has
    opted out (``{"source_name": "none"}``) or for whom no source resolves
    gets a literal ``null``.
    """
    from vtsearch import settings

    cfg, inherited = settings.get_settings_source_config_resolved()
    # ``jsonify(None)`` short-circuits flask-smorest's schema.dump, which
    # would otherwise turn ``None`` into ``{}``. The frontend expects a
    # literal ``null`` when no source is active.
    if cfg is None:
        return jsonify(None)
    return {**cfg, "inherited": inherited}


@sync_sources_bp.route("/api/settings-sources/active", methods=["PUT"])
@sync_sources_bp.arguments(SetSyncSourceRequestSchema)
@sync_sources_bp.response(200, OkMessageSchema)
@sync_sources_bp.alt_response(404, description="Unknown settings source name.")
def set_active_settings_source(body: dict):
    """Set or clear the active settings source.

    A body of ``{}`` or one with empty ``source_name`` clears the user's
    explicit source so they inherit the deployment-wide default (if any).
    ``source_name: "none"`` records an explicit opt-out (no source even when
    a default exists). Otherwise both fields specify the new source.
    """
    from vtsearch import settings
    from vtsearch.settings_io.sources import get_settings_source

    source_name = (body.get("source_name") or "").strip()
    if not source_name:
        settings.set_settings_source_config(None)
        return {"ok": True, "message": "Settings source cleared."}

    if source_name == "none":
        # Explicit opt-out: store the sentinel so the precedence resolver
        # returns no source for this user even if a deployment default exists.
        settings.set_settings_source_config({"source_name": "none"})
        return {"ok": True, "message": "Settings source opted out."}

    source = get_settings_source(source_name)
    if source is None:
        abort(404, message=f"Unknown settings source: {source_name!r}")

    config = {
        "source_name": source_name,
        "field_values": body.get("field_values") or {},
    }
    settings.set_settings_source_config(config)
    return {"ok": True, "message": f"Settings source set to {source.display_name}."}


@sync_sources_bp.route("/api/settings-sources/sync", methods=["POST"])
@sync_sources_bp.response(200, SyncFromSourceResponseSchema)
def sync_settings_from_source():
    """Force a manual import from the active settings source."""
    from vtsearch import settings

    imported = settings.sync_from_settings_source()
    if imported is None:
        # Always include ``keys`` (even empty) because marshmallow's
        # fallback ``getattr(dict, "keys")`` returns ``dict.keys`` (the
        # built-in method), which then fails to serialize.
        return {"ok": False, "message": "No settings source configured or source is empty.", "keys": []}

    return {
        "ok": True,
        "message": f"Imported {len(imported)} setting(s) from source.",
        "keys": list(imported.keys()),
    }


# ---------------------------------------------------------------------------
# Labelset sources
#
# A detector's binding is persisted on its registry entry and resolved by
# :func:`vtscore.labels.binding.resolve_labelset_source` (explicit binding >
# deployment default > none), so these routes work on any registered
# detector, loaded or not.  The ``<detector_name>`` path segment takes the
# detector's id (what the routes always resolved it by) or its name.
# ---------------------------------------------------------------------------


@sync_sources_bp.route("/api/labelset-sources", methods=["GET"])
@sync_sources_bp.response(200, SyncSourceEntrySchema(many=True))
def list_labelset_sources_route():
    """Return a list of all registered labelset source plugins."""
    from vtscore.labels.sources import list_labelset_sources
    from vtsearch.settings import filter_visible_plugins

    return [src.to_dict() for src in filter_visible_plugins("labelset_sources", list_labelset_sources())]


@sync_sources_bp.route("/api/labelset-sources/default", methods=["GET"])
@sync_sources_bp.response(200, DefaultLabelsetSourceSchema)
def get_default_labelset_source():
    """Return the deployment-wide labelset source, or null.

    The source every detector on this instance syncs its labels to unless it
    has a binding of its own (the ``default_labelset_source`` server setting).
    Its ``fields`` are only those the admin left for users to fill in, which
    is what the New Detector form asks for; the fixed values stay here.
    """
    from vtscore.labels.binding import deployment_default_labelset_source, fixed_field_values, unfixed_fields
    from vtscore.labels.sources import get_labelset_source
    from vtsearch import settings

    default = deployment_default_labelset_source()
    source = get_labelset_source(default["source_name"].strip()) if default else None
    if default is None or source is None:
        # ``jsonify(None)`` short-circuits flask-smorest's schema.dump (see
        # ``get_active_settings_source``).
        return jsonify(None)
    entry = source.to_dict()
    entry["fields"] = [f.to_dict() for f in unfixed_fields(source, default)]
    entry["fixed_keys"] = sorted(k for k in fixed_field_values(default) if any(f.key == k for f in source.fields))
    entry["push_interval_seconds"] = settings.get_labelset_source_push_interval_seconds()
    return entry


def _registered_detector(detector_ref: str) -> dict:
    """The registry entry *detector_ref* names (by id, else by name), or 404."""
    from vtscore.detectors.registry import find_by_name, get_detector

    entry = get_detector(detector_ref) or find_by_name(detector_ref)
    if entry is None:
        abort(404, message=f"Detector not found: {detector_ref!r}")
    return entry


@sync_sources_bp.route("/api/detectors/<detector_name>/labelset-source", methods=["GET"])
@sync_sources_bp.response(200, DetectorLabelsetSourceSchema)
@sync_sources_bp.alt_response(403, description="Access denied for the current user.")
@sync_sources_bp.alt_response(404, description="Detector not found.")
def get_detector_labelset_source(detector_name: str):
    """Return the labelset source a detector syncs with, or null.

    ``field_values`` carries the detector's own values, and only for its
    owner; values the deployment default fixes are never echoed.
    """
    from vtscore.detectors.registry import can_user_access_detector
    from vtscore.labels.binding import resolve_labelset_source, stored_labelset_source
    from vtscore.state.core import get_detector_context
    from vtsearch.auth import get_current_user

    entry = _registered_detector(detector_name)
    user = get_current_user()
    if not can_user_access_detector(entry["id"], user):
        abort(403, message="You do not have access to this detector")
    resolved = resolve_labelset_source(entry)
    if resolved is None:
        return jsonify(None)
    if resolved.inherited:
        own: dict = {}
    else:
        stored = stored_labelset_source(entry)
        if stored is None:
            ctx = get_detector_context(entry["id"])
            stored = ctx.labelset_source if ctx is not None else None
        own = dict((stored or {}).get("field_values") or {})
    own = {k: v for k, v in own.items() if k not in resolved.fixed_keys}
    source = resolved.source
    return {
        "source_name": resolved.source_name,
        "display_name": (source.display_name if source is not None else "") or resolved.source_name,
        "field_values": own if entry.get("created_by", "default") == user else {},
        "inherited": resolved.inherited,
        "fixed_keys": sorted(resolved.fixed_keys),
        "missing_fields": list(resolved.missing_fields),
    }


def _validated_binding(entry: dict, source, field_values: dict) -> dict:
    """Clean *field_values* for *source* on the detector *entry*, or abort 400.

    Keeps only the source's own, non-fixed fields (as strings), then checks
    the values the source would actually be handed - the fixed ones laid over
    them - the way a push will: required fields present, templates resolvable,
    paths and URLs allowed.
    """
    from vtscore.labels.binding import deployment_default_labelset_source, fixed_field_values
    from vtscore.plugins.normalize import normalize_field_values
    from vtscore.state.core import DetectorContext, override_detector_context

    default = deployment_default_labelset_source()
    fixed = fixed_field_values(default) if default and default["source_name"].strip() == source.name else {}
    keys = {f.key for f in source.fields}
    own = {
        str(k): "" if v is None else str(v)
        for k, v in (field_values or {}).items()
        if str(k) in keys and str(k) not in fixed
    }
    with override_detector_context(DetectorContext(entry["id"], name=entry.get("name", ""))):
        try:
            normalize_field_values(source, {**own, **fixed})
        except ValueError as exc:
            abort(400, message=str(exc))
    return own


@sync_sources_bp.route("/api/detectors/<detector_name>/labelset-source", methods=["PUT"])
@sync_sources_bp.arguments(SetSyncSourceRequestSchema)
@sync_sources_bp.response(200, OkMessageSchema)
@sync_sources_bp.alt_response(400, description="A field value is missing or not allowed.")
@sync_sources_bp.alt_response(403, description="Only the detector's creator can change where it syncs.")
@sync_sources_bp.alt_response(404, description="Unknown detector or labelset source.")
def set_detector_labelset_source(body: dict, detector_name: str):
    """Set, clear, or opt a detector out of its labelset source.

    A body of ``{}`` (or an empty ``source_name``) clears the detector's own
    binding, so it follows the deployment default again, if any.
    ``source_name: "none"`` opts it out even when a default exists.
    Otherwise the detector syncs with the named source using ``field_values``
    (any values the default fixes for that source are dropped: they always
    come from the server).  Setting a source pulls what it already holds into
    the detector, then pushes the detector's labels back to it.
    """
    from vtscore.detectors.registry import is_detector_owner, update_detector
    from vtscore.labels.binding import OPT_OUT
    from vtscore.labels.sources import get_labelset_source
    from vtscore.labels.sync import reconcile_labelset_source
    from vtsearch.auth import get_current_user

    entry = _registered_detector(detector_name)
    if not is_detector_owner(entry["id"], get_current_user()):
        abort(403, message="Only the detector's creator can change where it syncs")

    source_name = (body.get("source_name") or "").strip()
    if not source_name:
        update_detector(entry["id"], labelset_source=None)
        reconcile_labelset_source(entry["name"], detector_id=entry["id"])
        return {"ok": True, "message": "Labelset source cleared."}

    if source_name == OPT_OUT:
        update_detector(entry["id"], labelset_source={"source_name": OPT_OUT})
        return {"ok": True, "message": "Labelset source opted out."}

    source = get_labelset_source(source_name)
    if source is None:
        abort(404, message=f"Unknown labelset source: {source_name!r}")

    own = _validated_binding(entry, source, body.get("field_values") or {})
    update_detector(entry["id"], labelset_source={"source_name": source_name, "field_values": own})
    pulled = reconcile_labelset_source(entry["name"], detector_id=entry["id"])
    message = f"Labels will be auto-exported to {source.display_name}."
    if pulled:
        message += f" Restored {pulled} label(s) it already held."
    return {"ok": True, "message": message}


@sync_sources_bp.route("/api/detectors/<detector_name>/labelset-source/sync", methods=["POST"])
@sync_sources_bp.response(200, SyncFromLabelsetSourceResponseSchema)
@sync_sources_bp.alt_response(403, description="Access denied for the current user.")
@sync_sources_bp.alt_response(404, description="Detector not found.")
def sync_detector_labelset_from_source(detector_name: str):
    """Pull the labels at the detector's labelset source into it now.

    Labels the source holds and the detector lacks are added, and where the
    two disagree the source's label wins; nothing is removed.  The merged
    labels are then pushed back to the source.
    """
    from vtscore.detectors.registry import can_user_access_detector
    from vtscore.labels.sync import sync_from_labelset_source
    from vtsearch.auth import get_current_user

    entry = _registered_detector(detector_name)
    if not can_user_access_detector(entry["id"], get_current_user()):
        abort(403, message="You do not have access to this detector")

    from vtscore.detectors.store import _detector_path, _read_detector

    before = _labels_of(_read_detector(_detector_path(entry["name"])))
    labels = sync_from_labelset_source(entry["id"])
    if labels is None:
        return {"ok": False, "message": "No labelset source configured or source is empty."}
    changed = _count_changed(before, _labels_of(_read_detector(_detector_path(entry["name"]))))
    return {
        "ok": True,
        "message": f"Imported {len(labels)} label(s) from source ({changed} new or changed).",
        "labels": len(labels),
        "changed": changed,
    }


def _labels_of(data: dict | None) -> dict:
    """``{element key: label}`` for a detector file's labelset."""
    from vtscore.datasets.labelset import LabelSet, element_key

    labelset = LabelSet.from_dict((data or {}).get("labelset") or {})
    return {element_key(el): el.label for el in labelset.elements}


def _count_changed(before: dict, after: dict) -> int:
    return sum(1 for key, label in after.items() if before.get(key) != label)
