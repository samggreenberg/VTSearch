"""Schemas for the Settings API (``/api/settings`` and friends).

The response shape (``AppSettingsSchema``) mirrors ``settings.get_all()``
and is the source of truth for the frontend's settings DTO.
``SettingsUpdateSchema`` is the partial-update shape accepted by
``PUT /api/settings`` — every field is optional; only present keys are
modified.

Server-tier vs. per-user routing is handled inside ``vtsearch.settings``;
the schema is flat to match what the frontend sends and receives.
"""

from __future__ import annotations

from marshmallow import Schema, fields, validate

from vtsearch.settings import VALID_FOCUS_MODES, VALID_GRID_ICON_SIZES, VALID_THEMES, VALID_VIEW_MODES


class _PerMediaTypeStringDict(fields.Dict):
    """``{media_type_id: value}`` dict where keys and values are strings."""

    def __init__(self, **kwargs):
        super().__init__(keys=fields.String(), values=fields.String(), **kwargs)


class _PerMediaTypeNumberDict(fields.Dict):
    """``{media_type_id: number}`` dict (used for panel-percent settings)."""

    def __init__(self, **kwargs):
        super().__init__(keys=fields.String(), values=fields.Float(), **kwargs)


class AppSettingsSchema(Schema):
    """Full settings dict returned by ``GET /api/settings`` and ``/defaults``.

    This is the schema the frontend's ``AppSettings`` interface is
    generated from. Every key is optional on read because the
    server/user split means a request can carry either tier alone.
    """

    # Per-user, scalar
    volume = fields.Float()
    inclusion = fields.Integer()
    theme = fields.String(validate=validate.OneOf(VALID_THEMES))
    enrich_descriptions = fields.Boolean()
    safe_thresholds = fields.Boolean()
    calibrate_count = fields.Integer()
    calibration_fraction = fields.Float()
    audio_playing = fields.Boolean()
    swipe_animation = fields.Boolean()
    show_metadata = fields.Boolean()
    autopilot_enabled = fields.Boolean()
    hide_autopilot = fields.Boolean()
    autopilot_top_greens = fields.Integer()
    autopilot_hard_reds = fields.Integer()
    autopilot_resort_interval = fields.Integer()
    autopilot_goal_diversity = fields.Integer()

    # Per-user, per-media-type
    view_mode_left = _PerMediaTypeStringDict()
    view_mode_right = _PerMediaTypeStringDict()
    grid_icon_size_left = _PerMediaTypeStringDict()
    grid_icon_size_right = _PerMediaTypeStringDict()
    focus_mode_left = _PerMediaTypeStringDict()
    focus_mode_right = _PerMediaTypeStringDict()
    panel_pct_left = _PerMediaTypeNumberDict()
    panel_pct_right = _PerMediaTypeNumberDict()

    # Server-tier
    saved_datasets_dir = fields.String()
    detectors_dir = fields.String()
    max_concurrent_dataset_downloads = fields.Integer()
    max_concurrent_dataset_embeddings = fields.Integer()
    autorun_detectors = fields.List(fields.String())

    class Meta:
        # Allow extra keys on dump so transitional fields (e.g.
        # ``settings_source`` config blobs, ``achievement_state``) flow
        # through without dropping. They'll be tightened as the rest of
        # the API is migrated.
        unknown = "include"


class SettingsUpdateSchema(Schema):
    """Partial-update body for ``PUT /api/settings``.

    Every field is optional; only keys present in the request body are
    applied. Field-level validation runs here (range clamps stay in the
    setters so the truth lives in one place).
    """

    volume = fields.Float()
    inclusion = fields.Integer()
    theme = fields.String(validate=validate.OneOf(VALID_THEMES))
    enrich_descriptions = fields.Boolean()
    safe_thresholds = fields.Boolean()
    calibrate_count = fields.Integer()
    calibration_fraction = fields.Float()
    audio_playing = fields.Boolean()
    swipe_animation = fields.Boolean()
    show_metadata = fields.Boolean()

    view_mode_left = fields.Raw()
    view_mode_right = fields.Raw()
    grid_icon_size_left = fields.Raw()
    grid_icon_size_right = fields.Raw()
    focus_mode_left = fields.Raw()
    focus_mode_right = fields.Raw()
    panel_pct_left = fields.Raw()
    panel_pct_right = fields.Raw()

    autopilot_enabled = fields.Boolean()
    hide_autopilot = fields.Boolean()
    autopilot_top_greens = fields.Integer()
    autopilot_hard_reds = fields.Integer()
    autopilot_resort_interval = fields.Integer()
    autopilot_goal_diversity = fields.Integer()

    autorun_detectors = fields.List(fields.String())

    saved_datasets_dir = fields.String()
    detectors_dir = fields.String()

    class Meta:
        # Reject keys we don't know — the frontend should never send
        # settings the backend can't apply.
        unknown = "exclude"


# Note: per-side dict fields (view_mode_left etc.) are declared as
# ``fields.Raw`` on the *update* schema because the existing setters
# accept either ``"grid"``/``{media_type: "grid"}`` and the validators
# inside ``settings.py`` are the source of truth. Tightening these to
# ``fields.Dict(...)`` is a follow-up once the per-side setters are
# unified.

# View modes / grid sizes / focus modes are re-exported for the few
# call sites that import them from here when checking PUT body values
# directly (test helpers).
__all__ = [
    "AppSettingsSchema",
    "SettingsUpdateSchema",
    "VALID_FOCUS_MODES",
    "VALID_GRID_ICON_SIZES",
    "VALID_THEMES",
    "VALID_VIEW_MODES",
]
