"""Flask routes for the Settings API.

Endpoints
---------
GET  /api/settings
    Return all persisted settings for the current user (merged
    server-tier + per-user-tier).

PUT  /api/settings
    Update one or more settings fields.  Only supplied keys are changed.

GET  /api/settings/defaults
    Return the default values for all settings.

This module is the **OpenAPI pilot** for the migration described in
``docs/plans/openapi-schema.md``: schemas in
``vtsearch/schemas/settings.py`` are the source of truth, validation
runs through marshmallow, and the OpenAPI spec is generated
automatically by flask-smorest.
"""

from __future__ import annotations

from flask_smorest import Blueprint, abort

from vtsearch import settings
from vtsearch.schemas.settings import AppSettingsSchema, SettingsUpdateSchema

settings_bp = Blueprint(
    "settings",
    __name__,
    description="Read and modify persisted user / server settings.",
)


# Map of update-body key → setter callable. Setters live in
# ``vtsearch.settings`` and enforce range clamping / value validation,
# so this module's only job is to dispatch — marshmallow already
# validated the *types*.
_SCALAR_SETTERS: dict[str, callable] = {
    "volume": settings.set_volume,
    "theme": settings.set_theme,
    "enrich_descriptions": settings.set_enrich_descriptions,
    "safe_thresholds": settings.set_safe_thresholds,
    "calibrate_count": settings.set_calibrate_count,
    "calibration_fraction": settings.set_calibration_fraction,
    "audio_playing": settings.set_audio_playing,
    "swipe_animation": settings.set_swipe_animation,
    "show_metadata": settings.set_show_metadata,
    "view_mode_left": settings.set_view_mode_left,
    "view_mode_right": settings.set_view_mode_right,
    "grid_icon_size_left": settings.set_grid_icon_size_left,
    "grid_icon_size_right": settings.set_grid_icon_size_right,
    "focus_mode_left": settings.set_focus_mode_left,
    "focus_mode_right": settings.set_focus_mode_right,
    "panel_pct_left": settings.set_panel_pct_left,
    "panel_pct_right": settings.set_panel_pct_right,
    "autopilot_enabled": settings.set_autopilot_enabled,
    "hide_autopilot": settings.set_hide_autopilot,
    "autopilot_top_greens": settings.set_autopilot_top_greens,
    "autopilot_hard_reds": settings.set_autopilot_hard_reds,
    "autopilot_resort_interval": settings.set_autopilot_resort_interval,
    "autopilot_goal_diversity": settings.set_autopilot_goal_diversity,
    "autorun_detectors": settings.set_autorun_detectors,
}


def _apply_inclusion(value) -> None:
    """``inclusion`` is set via :mod:`vtsearch.state`, not :mod:`settings`."""
    from vtsearch.state import set_inclusion

    clamped = int(max(-10, min(10, int(value))))
    set_inclusion(clamped)


def _apply_dir(key: str, value: str, setter) -> None:
    """Validate and apply a directory-path setting."""
    import vtsearch.security.path_validation as _paths

    if not value or not value.strip():
        abort(400, message=f"{key} must be a non-empty string")

    base = _paths.get_file_access_base_dir()
    if base is not None:
        try:
            _paths.validate_server_filepath(value.strip(), base_dir=base)
        except ValueError as exc:
            abort(400, message=str(exc))
    setter(value.strip())


@settings_bp.route("/api/settings", methods=["GET"])
@settings_bp.response(200, AppSettingsSchema)
def get_settings():
    """Return the merged server + per-user settings dict."""
    return settings.get_all()


@settings_bp.route("/api/settings", methods=["PUT"])
@settings_bp.arguments(SettingsUpdateSchema)
@settings_bp.response(200, AppSettingsSchema)
@settings_bp.alt_response(400, description="Setter-level validation failure (range, one-of, path traversal).")
def update_settings(body: dict):
    """Update one or more settings fields.

    Only keys present in *body* are applied. Unknown keys are silently
    dropped (per the schema's ``unknown = "exclude"`` policy); type
    errors raise 422 with the standard error envelope; setter-level
    validation failures (range / one-of / path traversal) raise 400.
    """
    for key, value in body.items():
        if key == "inclusion":
            try:
                _apply_inclusion(value)
            except (TypeError, ValueError) as exc:
                abort(400, message=str(exc))
            continue

        if key in ("saved_datasets_dir", "detectors_dir"):
            setter = (
                settings.set_saved_datasets_dir
                if key == "saved_datasets_dir"
                else settings.set_detectors_dir
            )
            _apply_dir(key, value, setter)
            continue

        setter = _SCALAR_SETTERS.get(key)
        if setter is None:
            continue
        try:
            setter(value)
        except (TypeError, ValueError) as exc:
            abort(400, message=str(exc))

    return settings.get_all()


@settings_bp.route("/api/settings/defaults", methods=["GET"])
@settings_bp.response(200, AppSettingsSchema)
def get_defaults():
    """Return the default values for all settings."""
    return settings.get_defaults()
