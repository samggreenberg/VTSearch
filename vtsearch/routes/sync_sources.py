"""Flask routes for the Sync Sources API.

Endpoints
---------
Settings sources:
    GET  /api/settings-sources              — list available source plugins
    GET  /api/settings-sources/active       — get the active source config
    PUT  /api/settings-sources/active       — set or clear the active source
    POST /api/settings-sources/sync         — force import from source

Labelset sources:
    GET  /api/labelset-sources              — list available source plugins
    GET  /api/detectors/<name>/labelset-source       — get detector's source
    PUT  /api/detectors/<name>/labelset-source       — set or clear source
    POST /api/detectors/<name>/labelset-source/sync  — force import from source
"""

from __future__ import annotations

from flask import Blueprint, jsonify

from vtsearch.routes.helpers import get_json_or_400

sync_sources_bp = Blueprint("sync_sources", __name__)


# ---------------------------------------------------------------------------
# Settings sources
# ---------------------------------------------------------------------------


@sync_sources_bp.route("/api/settings-sources", methods=["GET"])
def list_settings_sources_route():
    """Return a list of all registered settings source plugins."""
    from vtsearch.settings_io.sources import list_settings_sources

    return jsonify([src.to_dict() for src in list_settings_sources()])


@sync_sources_bp.route("/api/settings-sources/active", methods=["GET"])
def get_active_settings_source():
    """Return the active settings source config, or null."""
    from vtsearch import settings

    cfg = settings.get_settings_source_config()
    return jsonify(cfg)


@sync_sources_bp.route("/api/settings-sources/active", methods=["PUT"])
def set_active_settings_source():
    """Set or clear the active settings source.

    Request body (JSON)::

        {"source_name": "server_json_file", "field_values": {"filepath": "..."}}

    Send ``null`` or ``{}`` to clear the active source.
    """
    from vtsearch import settings
    from vtsearch.settings_io.sources import get_settings_source

    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    if not data or not data.get("source_name"):
        settings.set_settings_source_config(None)
        return jsonify({"ok": True, "message": "Settings source cleared."})

    source_name = data["source_name"]
    source = get_settings_source(source_name)
    if source is None:
        return jsonify({"error": f"Unknown settings source: {source_name!r}"}), 404

    config = {
        "source_name": source_name,
        "field_values": data.get("field_values", {}),
    }
    settings.set_settings_source_config(config)
    return jsonify({"ok": True, "message": f"Settings source set to {source.display_name}."})


@sync_sources_bp.route("/api/settings-sources/sync", methods=["POST"])
def sync_settings_from_source():
    """Force a manual import from the active settings source."""
    from vtsearch import settings

    imported = settings.sync_from_settings_source()
    if imported is None:
        return jsonify({"ok": False, "message": "No settings source configured or source is empty."})

    return jsonify({
        "ok": True,
        "message": f"Imported {len(imported)} setting(s) from source.",
        "keys": list(imported.keys()),
    })


# ---------------------------------------------------------------------------
# Labelset sources
# ---------------------------------------------------------------------------


@sync_sources_bp.route("/api/labelset-sources", methods=["GET"])
def list_labelset_sources_route():
    """Return a list of all registered labelset source plugins."""
    from vtsearch.labels.sources import list_labelset_sources

    return jsonify([src.to_dict() for src in list_labelset_sources()])


@sync_sources_bp.route("/api/detectors/<detector_name>/labelset-source", methods=["GET"])
def get_detector_labelset_source(detector_name: str):
    """Return the labelset source config for a detector, or null."""
    from vtsearch.utils import get_autorun_detectors

    detectors = get_autorun_detectors()
    if detector_name not in detectors:
        return jsonify({"error": f"Detector not found: {detector_name!r}"}), 404

    from vtsearch.utils.state_core import get_detector_context

    ctx = get_detector_context(detector_name)
    if ctx is None:
        return jsonify(None)

    return jsonify(ctx.labelset_source)


@sync_sources_bp.route("/api/detectors/<detector_name>/labelset-source", methods=["PUT"])
def set_detector_labelset_source(detector_name: str):
    """Set or clear the labelset source for a detector.

    Request body (JSON)::

        {"source_name": "server_json_file", "field_values": {"filepath": "..."}}

    Send ``null`` or ``{}`` to clear the source.
    """
    from vtsearch.labels.sources import get_labelset_source
    from vtsearch.utils.state_core import get_detector_context

    ctx = get_detector_context(detector_name)
    if ctx is None:
        return jsonify({"error": f"Detector not loaded: {detector_name!r}"}), 404

    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    if not data or not data.get("source_name"):
        ctx.labelset_source = None
        return jsonify({"ok": True, "message": "Labelset source cleared."})

    source_name = data["source_name"]
    source = get_labelset_source(source_name)
    if source is None:
        return jsonify({"error": f"Unknown labelset source: {source_name!r}"}), 404

    ctx.labelset_source = {
        "source_name": source_name,
        "field_values": data.get("field_values", {}),
    }
    return jsonify({"ok": True, "message": f"Labelset source set to {source.display_name}."})


@sync_sources_bp.route("/api/detectors/<detector_name>/labelset-source/sync", methods=["POST"])
def sync_detector_labelset_from_source(detector_name: str):
    """Force a manual import from the detector's labelset source."""
    from vtsearch.labels.sync import sync_from_labelset_source
    from vtsearch.utils.state_core import get_detector_context

    ctx = get_detector_context(detector_name)
    if ctx is None:
        return jsonify({"error": f"Detector not loaded: {detector_name!r}"}), 404

    labels = sync_from_labelset_source(detector_name)
    if labels is None:
        return jsonify({"ok": False, "message": "No labelset source configured or source is empty."})

    return jsonify({
        "ok": True,
        "message": f"Imported {len(labels)} label(s) from source.",
    })
