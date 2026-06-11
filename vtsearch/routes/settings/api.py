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

from typing import Any, Callable

from flask_smorest import Blueprint, abort
from marshmallow import fields

from vtsearch import settings
from vtsearch.schemas.settings import AppSettingsSchema, SettingsUpdateSchema
from vtsearch.state import (
    set_calibrate_count as _state_set_calibrate_count,
    set_calibration_fraction as _state_set_calibration_fraction,
    set_safe_thresholds as _state_set_safe_thresholds,
)

settings_bp = Blueprint(
    "settings",
    __name__,
    description="Read and modify persisted user / server settings.",
)


# Map of update-body key → setter callable. Setters live in
# ``vtsearch.settings`` and enforce range clamping / value validation,
# so this module's only job is to dispatch; marshmallow already
# validated the *types*.
#
# The training-relevant settings (``safe_thresholds``, ``calibrate_count``,
# ``calibration_fraction``) route through ``vtsearch.state`` rather than
# ``vtsearch.settings`` so the state setter's side-effect
# (``invalidate_loaded_detector_models``) fires and the cached MLP /
# threshold on every loaded detector context is dropped; otherwise
# ``/api/find-label`` / ``/api/find`` / ``/api/auto-detect`` would keep
# scoring with a threshold computed under the prior setting (M7).
_SCALAR_SETTERS: dict[str, Callable[[Any], Any]] = {
    "volume": settings.set_volume,
    "theme": settings.set_theme,
    "enrich_descriptions": settings.set_enrich_descriptions,
    "safe_thresholds": _state_set_safe_thresholds,
    "calibrate_count": _state_set_calibrate_count,
    "calibration_fraction": _state_set_calibration_fraction,
    "audio_playing": settings.set_audio_playing,
    "swipe_animation": settings.set_swipe_animation,
    "show_metadata": settings.set_show_metadata,
    "label_hint_dismissed": settings.set_label_hint_dismissed,
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
    "import_defaults_by_media_type": settings.set_import_defaults_by_media_type,
    "browse_panel_width": settings.set_browse_panel_width,
    "browse_bin_shape": settings.set_browse_bin_shape,
    "browse_colormap": settings.set_browse_colormap,
    "browse_icon_size": settings.set_browse_icon_size,
    "browse_thumbnail_border": settings.set_browse_thumbnail_border,
    "browse_compact": settings.set_browse_compact,
}


def _apply_enable_achievements(value: bool) -> None:
    """Persist the toggle and wipe stored counters when flipping it off.

    The user-visible promise is that turning the feature off zeroes the
    achievement counters and keeps them there. Wiping on the True→False
    transition (rather than every set) makes the on→off→on cycle
    deterministic: counters reset on opt-out and start fresh if the user
    ever opts back in.
    """
    prev = bool(settings.get_enable_achievements())
    coerced = bool(value)
    settings.set_enable_achievements(coerced)
    if prev and not coerced:
        from vtsearch import achievements

        achievements.wipe_state()


def _apply_inclusion(value) -> None:
    """``inclusion`` is set via :mod:`vtsearch.state`, not :mod:`settings`."""
    from vtsearch.state import set_inclusion

    clamped = int(max(-10, min(10, int(value))))
    set_inclusion(clamped)


def _apply_solo_media_type(value) -> None:
    """Validate *value* against the registry and apply via the combined setter.

    ``None`` / ``""`` clears solo mode (still marks the choice explicit so
    the user's "show everything" opt-out is preserved against the CLI
    fallback). Any other string must match a registered media-type id.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        settings.apply_user_solo_media_type(None)
        return
    if not isinstance(value, str):
        abort(400, message="solo_media_type must be a string or null")
    from vtscore.media import all_type_ids

    valid = set(all_type_ids())
    if value not in valid:
        abort(400, message=f"Unknown media type: {value!r}. Valid: {sorted(valid)}")
    settings.apply_user_solo_media_type(value)


def _apply_solo_embedder_per_media_type(value) -> None:
    """Validate the ``{media_type: embedder}`` map and persist it.

    ``None`` clears every per-type lock. Otherwise *value* must be a dict
    mapping registered media-type ids to embedder names that exist for
    that type (per :func:`vtscore.media.embedders_for_type`). An empty
    string value is preserved as a **per-type opt-out sentinel**; it
    overrides the ``--solo-embedder`` CLI fallback for that type
    (analog of setting ``solo_media_type=null`` to override
    ``--solo-media-type``). Any other invalid pairing raises 400.
    """
    if value is None:
        settings.apply_user_solo_embedder_per_media_type(None)
        return
    if not isinstance(value, dict):
        abort(400, message="solo_embedder_per_media_type must be a dict or null")

    from vtscore.media import all_type_ids, embedders_for_type

    valid_types = set(all_type_ids())
    cleaned: dict[str, str] = {}
    for raw_type, raw_emb in value.items():
        if not isinstance(raw_type, str) or not raw_type.strip():
            abort(400, message="solo_embedder_per_media_type keys must be non-empty media-type ids")
        mt = raw_type.strip()
        if mt not in valid_types:
            abort(400, message=f"Unknown media type: {mt!r}. Valid: {sorted(valid_types)}")
        if raw_emb is None or (isinstance(raw_emb, str) and not raw_emb.strip()):
            # Per-type opt-out sentinel: preserve so it overrides the
            # CLI fallback. ``None`` is normalised to "" here.
            cleaned[mt] = ""
            continue
        if not isinstance(raw_emb, str):
            abort(400, message=f"solo_embedder_per_media_type[{mt!r}] must be a string")
        emb_name = raw_emb.strip()
        valid_embedders = {e.name for e in embedders_for_type(mt)}
        if emb_name not in valid_embedders:
            abort(
                400,
                message=(f"Unknown embedder {emb_name!r} for media type {mt!r}. Valid: {sorted(valid_embedders)}"),
            )
        cleaned[mt] = emb_name
    settings.apply_user_solo_embedder_per_media_type(cleaned)


def _apply_dir(key: str, value: str, setter) -> None:
    """Validate and apply a directory-path setting."""
    import vtscore.security.path_validation as _paths

    if not value or not value.strip():
        abort(400, message=f"{key} must be a non-empty string")

    base = _paths.get_file_access_base_dir()
    if base is not None:
        try:
            _paths.validate_server_filepath(value.strip(), base_dir=base)
        except ValueError as exc:
            abort(400, message=str(exc))
    setter(value.strip())


#: Schema fields declared as dicts (``browse_*``, the per-media-type
#: embedder maps, ``import_defaults_by_media_type`` …). A persisted or
#: hand-edited settings file from an older build can carry a stale scalar
#: where one of these is now expected (e.g. a pre-per-media-type
#: ``browse_bin_shape: "hex"``). Marshmallow's ``Dict`` field calls
#: ``.items()`` while dumping, so a bare string there would 500 the entire
#: settings endpoint. Derived from the schema so it stays in sync as
#: fields are added.
_DICT_FIELD_NAMES = frozenset(
    name for name, field in AppSettingsSchema().fields.items() if isinstance(field, fields.Dict)
)


def _coerce_dict_fields(data: dict) -> dict:
    """Replace any non-dict value in a schema dict field with ``{}``.

    Mirrors how the per-side getters (``view_mode`` etc.) already coerce a
    junk persisted value back to its default on read: a stale scalar left
    over from an older settings file resolves to "never set" rather than
    crashing the serializer. Mutates and returns *data*.
    """
    for name in _DICT_FIELD_NAMES:
        if name in data and not isinstance(data[name], dict):
            data[name] = {}
    return data


def _with_effective(data: dict) -> dict:
    """Overlay the resolver-computed (read-only) views onto *data*.

    Centralises the augmentation shared by the GET and PUT responses:

    * ``effective_solo_media_type`` / ``effective_solo_embedder_per_media_type``
      - the per-user values layered over their CLI fallbacks; the frontend
      reads these to decide whether to hide mediaType / embedder pickers.
    * ``hidden_plugins`` - the persisted server setting unioned with any
      ``--hide-plugin`` CLI flags, normalised to sorted lists so the
      "Server" settings tab can render what's actually in force.

    Stale dict fields are coerced to ``{}`` first so a corrupt persisted
    value can't 500 the endpoint (see :func:`_coerce_dict_fields`).
    """
    _coerce_dict_fields(data)
    data["effective_solo_media_type"] = settings.get_effective_solo_media_type()
    data["effective_solo_embedder_per_media_type"] = settings.get_effective_solo_embedders()
    data["hidden_plugins"] = {
        family: sorted(names) for family, names in settings.get_effective_hidden_plugins().items()
    }
    return data


@settings_bp.route("/api/settings", methods=["GET"])
@settings_bp.response(200, AppSettingsSchema)
def get_settings():
    """Return the merged server + per-user settings dict.

    Augments the persisted dict with the resolver-computed read-only views
    (effective solo mediaType / embedder, effective hidden plugins) via
    :func:`_with_effective`. The frontend reads the ``effective_*`` keys
    when deciding whether to hide pickers; the raw ``solo_media_type`` /
    ``solo_media_type_explicit`` pair is still exposed for the settings UI
    to render the current state.
    """
    return _with_effective(settings.get_all())


#: Keys whose value is computed on read and silently ignored on write
#: (the raw fields they're derived from go through their own dispatch
#: entry below).
_READ_ONLY_KEYS = frozenset(
    {
        "effective_solo_media_type",
        "effective_solo_embedder_per_media_type",
    }
)


def _apply_inclusion_guarded(value) -> None:
    try:
        _apply_inclusion(value)
    except (TypeError, ValueError) as exc:
        abort(400, message=str(exc))


def _apply_enable_achievements_guarded(value) -> None:
    try:
        _apply_enable_achievements(value)
    except (TypeError, ValueError) as exc:
        abort(400, message=str(exc))


def _apply_saved_datasets_dir(value) -> None:
    _apply_dir("saved_datasets_dir", value, settings.set_saved_datasets_dir)


def _apply_detectors_dir(value) -> None:
    _apply_dir("detectors_dir", value, settings.set_detectors_dir)


#: Keys with bespoke side-effects (validation against a registry, path
#: traversal checks, counter wipes, etc.). Each handler raises 400 on
#: invalid input itself, so the dispatcher just calls and returns.
_CUSTOM_SETTERS: dict[str, Callable[[Any], None]] = {
    "inclusion": _apply_inclusion_guarded,
    "saved_datasets_dir": _apply_saved_datasets_dir,
    "detectors_dir": _apply_detectors_dir,
    "enable_achievements": _apply_enable_achievements_guarded,
    "solo_media_type": _apply_solo_media_type,
    "solo_embedder_per_media_type": _apply_solo_embedder_per_media_type,
}


def _apply_one_key(key: str, value) -> None:
    """Dispatch a single settings update body entry to the right setter.

    Keeps :func:`update_settings` simple by hosting the per-key
    branching here. Side effects (path validation, achievement wipe,
    state-tier setter) are isolated to their helper functions.
    """
    if key in _READ_ONLY_KEYS:
        return
    custom = _CUSTOM_SETTERS.get(key)
    if custom is not None:
        custom(value)
        return
    setter = _SCALAR_SETTERS.get(key)
    if setter is None:
        return
    try:
        setter(value)
    except (TypeError, ValueError) as exc:
        abort(400, message=str(exc))


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
        _apply_one_key(key, value)

    return _with_effective(settings.get_all())


@settings_bp.route("/api/settings/defaults", methods=["GET"])
@settings_bp.response(200, AppSettingsSchema)
def get_defaults():
    """Return the default values for all settings."""
    return settings.get_defaults()
