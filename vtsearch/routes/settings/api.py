"""Flask routes for the Settings API.

Endpoints
---------
GET  /api/settings
    Return all persisted settings.

PUT  /api/settings
    Update one or more settings fields.  Only supplied keys are changed.

GET  /api/settings/defaults
    Return the default values for all settings.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from vtsearch import settings

settings_bp = Blueprint("settings", __name__)


@settings_bp.route("/api/settings", methods=["GET"])
def get_settings():
    """Return all settings."""
    return jsonify(settings.get_all())


@settings_bp.route("/api/settings", methods=["PUT"])
def update_settings():
    """Update settings.  Only supplied keys are changed."""
    body = request.get_json(force=True, silent=True)
    if not body or not isinstance(body, dict):
        return jsonify({"error": "Invalid request body"}), 400

    if "volume" in body:
        try:
            settings.set_volume(float(body["volume"]))
        except (TypeError, ValueError):
            return jsonify({"error": "volume must be a number"}), 400

    if "theme" in body:
        try:
            settings.set_theme(str(body["theme"]))
        except ValueError:
            return jsonify({"error": "theme must be 'dark', 'light', or 'highviz'"}), 400

    if "inclusion" in body:
        try:
            val = body["inclusion"]
            if not isinstance(val, (int, float)):
                return jsonify({"error": "inclusion must be a number"}), 400
            clamped = int(max(-10, min(10, int(val))))
            # Update runtime state (which also persists to settings file)
            from vtsearch.state import set_inclusion

            set_inclusion(clamped)
        except (TypeError, ValueError):
            return jsonify({"error": "inclusion must be a number"}), 400

    if "enrich_descriptions" in body:
        settings.set_enrich_descriptions(bool(body["enrich_descriptions"]))

    if "safe_thresholds" in body:
        settings.set_safe_thresholds(bool(body["safe_thresholds"]))

    if "calibration_fraction" in body:
        try:
            val = body["calibration_fraction"]
            if not isinstance(val, (int, float)):
                return jsonify({"error": "calibration_fraction must be a number"}), 400
            settings.set_calibration_fraction(float(val))
        except (TypeError, ValueError):
            return jsonify({"error": "calibration_fraction must be a number"}), 400

    if "audio_playing" in body:
        settings.set_audio_playing(bool(body["audio_playing"]))

    if "swipe_animation" in body:
        settings.set_swipe_animation(bool(body["swipe_animation"]))

    if "calibrate_count" in body:
        try:
            val = body["calibrate_count"]
            if not isinstance(val, (int, float)):
                return jsonify({"error": "calibrate_count must be a number"}), 400
            settings.set_calibrate_count(int(val))
        except (TypeError, ValueError):
            return jsonify({"error": "calibrate_count must be a number"}), 400

    if "show_metadata" in body:
        settings.set_show_metadata(bool(body["show_metadata"]))

    if "view_mode_left" in body:
        try:
            settings.set_view_mode_left(body["view_mode_left"])
        except (ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400

    if "view_mode_right" in body:
        try:
            settings.set_view_mode_right(body["view_mode_right"])
        except (ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400

    if "grid_icon_size_left" in body:
        try:
            settings.set_grid_icon_size_left(body["grid_icon_size_left"])
        except (ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400

    if "grid_icon_size_right" in body:
        try:
            settings.set_grid_icon_size_right(body["grid_icon_size_right"])
        except (ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400

    if "focus_mode_left" in body:
        try:
            settings.set_focus_mode_left(body["focus_mode_left"])
        except (ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400

    if "focus_mode_right" in body:
        try:
            settings.set_focus_mode_right(body["focus_mode_right"])
        except (ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400

    if "panel_pct_left" in body:
        try:
            settings.set_panel_pct_left(body["panel_pct_left"])
        except (ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400

    if "panel_pct_right" in body:
        try:
            settings.set_panel_pct_right(body["panel_pct_right"])
        except (ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400

    if "autopilot_enabled" in body:
        settings.set_autopilot_enabled(bool(body["autopilot_enabled"]))

    if "hide_autopilot" in body:
        settings.set_hide_autopilot(bool(body["hide_autopilot"]))

    if "autopilot_top_greens" in body:
        try:
            val = body["autopilot_top_greens"]
            if not isinstance(val, (int, float)):
                return jsonify({"error": "autopilot_top_greens must be a number"}), 400
            settings.set_autopilot_top_greens(int(val))
        except (TypeError, ValueError):
            return jsonify({"error": "autopilot_top_greens must be a number"}), 400

    if "autopilot_hard_reds" in body:
        try:
            val = body["autopilot_hard_reds"]
            if not isinstance(val, (int, float)):
                return jsonify({"error": "autopilot_hard_reds must be a number"}), 400
            settings.set_autopilot_hard_reds(int(val))
        except (TypeError, ValueError):
            return jsonify({"error": "autopilot_hard_reds must be a number"}), 400

    if "autopilot_resort_interval" in body:
        try:
            val = body["autopilot_resort_interval"]
            if not isinstance(val, (int, float)):
                return jsonify({"error": "autopilot_resort_interval must be a number"}), 400
            settings.set_autopilot_resort_interval(int(val))
        except (TypeError, ValueError):
            return jsonify({"error": "autopilot_resort_interval must be a number"}), 400

    if "autopilot_goal_diversity" in body:
        try:
            val = body["autopilot_goal_diversity"]
            if not isinstance(val, (int, float)):
                return jsonify({"error": "autopilot_goal_diversity must be a number"}), 400
            settings.set_autopilot_goal_diversity(int(val))
        except (TypeError, ValueError):
            return jsonify({"error": "autopilot_goal_diversity must be a number"}), 400

    if "autoload_media_embedders" in body:
        val = body["autoload_media_embedders"]
        if not isinstance(val, list) or not all(isinstance(v, str) for v in val):
            return jsonify({"error": "autoload_media_embedders must be a list of strings"}), 400
        try:
            settings.set_autoload_media_embedders(val)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    if "autorun_detectors" in body:
        val = body["autorun_detectors"]
        if not isinstance(val, list) or not all(isinstance(v, str) for v in val):
            return jsonify({"error": "autorun_detectors must be a list of strings"}), 400
        settings.set_autorun_detectors(val)

    # Directory path settings
    import vtsearch.security.path_validation as _paths

    _dir_base = _paths.get_file_access_base_dir()
    for dir_key, setter in (
        ("saved_datasets_dir", settings.set_saved_datasets_dir),
        ("detectors_dir", settings.set_detectors_dir),
    ):
        if dir_key in body:
            val = body[dir_key]
            if not isinstance(val, str) or not val.strip():
                return jsonify({"error": f"{dir_key} must be a non-empty string"}), 400
            # In multi-user mode, restrict directory paths to user's data dir
            if _dir_base is not None:
                try:
                    _paths.validate_server_filepath(val.strip(), base_dir=_dir_base)
                except ValueError as exc:
                    return jsonify({"error": str(exc)}), 400
            setter(val.strip())

    return jsonify(settings.get_all())


@settings_bp.route("/api/settings/defaults", methods=["GET"])
def get_defaults():
    """Return the default values for all settings."""
    return jsonify(settings.get_defaults())
