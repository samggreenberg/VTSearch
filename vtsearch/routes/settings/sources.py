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
    GET  /api/detectors/<name>/labelset-source       (get detector's source)
    PUT  /api/detectors/<name>/labelset-source       (set or clear source)
    POST /api/detectors/<name>/labelset-source/sync  (force import from source)

Schema-level validation failures surface as 422 with the standard
``errors`` envelope; handler-level rejects (unknown source, detector not
loaded) keep their HTTP codes (404) with the standard ``message``
envelope.  Sending ``{}`` (empty body) clears the active source.
"""

from __future__ import annotations

from flask import jsonify
from flask_smorest import Blueprint, abort

from vtsearch.schemas.settings_io import (
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
# ---------------------------------------------------------------------------


@sync_sources_bp.route("/api/labelset-sources", methods=["GET"])
@sync_sources_bp.response(200, SyncSourceEntrySchema(many=True))
def list_labelset_sources_route():
    """Return a list of all registered labelset source plugins."""
    from vtscore.labels.sources import list_labelset_sources
    from vtsearch.settings import filter_visible_plugins

    return [src.to_dict() for src in filter_visible_plugins("labelset_sources", list_labelset_sources())]


@sync_sources_bp.route("/api/detectors/<detector_name>/labelset-source", methods=["GET"])
@sync_sources_bp.response(200, SyncSourceConfigSchema)
def get_detector_labelset_source(detector_name: str):
    """Return the labelset source config for a loaded detector, or null."""
    from vtscore.state.core import get_detector_context

    ctx = get_detector_context(detector_name)
    # ``jsonify(None)`` short-circuits flask-smorest's schema.dump (see
    # ``get_active_settings_source``).
    if ctx is None or ctx.labelset_source is None:
        return jsonify(None)
    return ctx.labelset_source


@sync_sources_bp.route("/api/detectors/<detector_name>/labelset-source", methods=["PUT"])
@sync_sources_bp.arguments(SetSyncSourceRequestSchema)
@sync_sources_bp.response(200, OkMessageSchema)
@sync_sources_bp.alt_response(404, description="Detector not loaded or unknown labelset source.")
def set_detector_labelset_source(body: dict, detector_name: str):
    """Set or clear the labelset source for a detector.

    A body of ``{}`` or one with empty ``source_name`` clears the source.
    """
    from vtscore.labels.sources import get_labelset_source
    from vtscore.state.core import get_detector_context

    ctx = get_detector_context(detector_name)
    if ctx is None:
        abort(404, message=f"Detector not loaded: {detector_name!r}")

    source_name = (body.get("source_name") or "").strip()
    if not source_name:
        ctx.labelset_source = None
        return {"ok": True, "message": "Labelset source cleared."}

    source = get_labelset_source(source_name)
    if source is None:
        abort(404, message=f"Unknown labelset source: {source_name!r}")

    ctx.labelset_source = {
        "source_name": source_name,
        "field_values": body.get("field_values") or {},
    }
    return {"ok": True, "message": f"Labelset source set to {source.display_name}."}


@sync_sources_bp.route("/api/detectors/<detector_name>/labelset-source/sync", methods=["POST"])
@sync_sources_bp.response(200, SyncFromLabelsetSourceResponseSchema)
@sync_sources_bp.alt_response(404, description="Detector not loaded.")
def sync_detector_labelset_from_source(detector_name: str):
    """Force a manual import from the detector's labelset source."""
    from vtscore.labels.sync import sync_from_labelset_source
    from vtscore.state.core import get_detector_context

    ctx = get_detector_context(detector_name)
    if ctx is None:
        abort(404, message=f"Detector not loaded: {detector_name!r}")

    labels = sync_from_labelset_source(detector_name)
    if labels is None:
        return {"ok": False, "message": "No labelset source configured or source is empty."}

    return {
        "ok": True,
        "message": f"Imported {len(labels)} label(s) from source.",
    }
