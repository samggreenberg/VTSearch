"""Schemas for the Settings API (``/api/settings`` and friends).

The response shape (``AppSettingsSchema``) mirrors ``settings.get_all()``
and is the source of truth for the frontend's settings DTO.
``SettingsUpdateSchema`` is the partial-update shape accepted by
``PUT /api/settings``; every field is optional and only present keys are
modified.

Server-tier vs. per-user routing is handled inside ``vtsearch.settings``;
the schema is flat to match what the frontend sends and receives.
"""

from __future__ import annotations

from marshmallow import Schema, fields, validate

from vtsearch.settings_models import VALID_FOCUS_MODES, VALID_GRID_ICON_SIZES, VALID_THEMES, VALID_VIEW_MODES


class _PerMediaTypeStringDict(fields.Dict):
    """``{media_type_id: value}`` dict where keys and values are strings."""

    def __init__(self, **kwargs):
        super().__init__(keys=fields.String(), values=fields.String(), **kwargs)


class _PerMediaTypeNumberDict(fields.Dict):
    """``{media_type_id: number}`` dict (used for panel-percent settings)."""

    def __init__(self, **kwargs):
        super().__init__(keys=fields.String(), values=fields.Float(), **kwargs)


class _PerMediaTypeIntDict(fields.Dict):
    """``{media_type_id: int}`` dict (used for pixel-width settings)."""

    def __init__(self, **kwargs):
        super().__init__(keys=fields.String(), values=fields.Integer(), **kwargs)


class _PerMediaTypeBooleanDict(fields.Dict):
    """``{media_type_id: bool}`` dict (used for on/off browse toggles)."""

    def __init__(self, **kwargs):
        super().__init__(keys=fields.String(), values=fields.Boolean(), **kwargs)


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
    label_hint_dismissed = fields.Boolean()
    autopilot_enabled = fields.Boolean()
    hide_autopilot = fields.Boolean()
    autopilot_top_greens = fields.Integer()
    autopilot_hard_reds = fields.Integer()
    autopilot_resort_interval = fields.Integer()
    autopilot_goal_diversity = fields.Integer()
    enable_achievements = fields.Boolean()

    # VTSBrowse side-panel width (CSS px). Persisted but not shown as a
    # Settings-modal widget; the panel's draggable divider drives it.
    browse_panel_width = fields.Integer()

    # VTSBrowse per-media-type display prefs (bin shape, density colormap,
    # on-screen cell size). ``{media_type_id: value}`` dicts driven by the
    # browse-canvas toolbar and the Settings → Browser tab.
    browse_bin_shape = _PerMediaTypeStringDict()
    browse_colormap = _PerMediaTypeStringDict()
    browse_icon_size = _PerMediaTypeStringDict()
    # Pile-thumbnail border width in CSS px, per media type (0 disables).
    browse_thumbnail_border = _PerMediaTypeIntDict()
    # Whether (re)building the projection compacts the layout, per media type.
    browse_compact = _PerMediaTypeBooleanDict()

    # Per-user, per-media-type
    view_mode_left = _PerMediaTypeStringDict()
    view_mode_right = _PerMediaTypeStringDict()
    grid_icon_size_left = _PerMediaTypeStringDict()
    grid_icon_size_right = _PerMediaTypeStringDict()
    focus_mode_left = _PerMediaTypeStringDict()
    focus_mode_right = _PerMediaTypeStringDict()
    panel_pct_left = _PerMediaTypeNumberDict()
    panel_pct_right = _PerMediaTypeNumberDict()
    # VTSBrowse bin-popup view mode + thumbnail size, per media type. Driven by
    # the popup's own view controls and the Settings → Browser tab.
    view_mode_popup = _PerMediaTypeStringDict()
    grid_icon_size_popup = _PerMediaTypeStringDict()

    # Server-tier. These are fixed at server start (config file /
    # environment / CLI flags) and shared across all users; the frontend
    # surfaces them read-only in the "Server" settings tab. They are not
    # in ``SettingsUpdateSchema``, so the API rejects attempts to change
    # them.
    saved_datasets_dir = fields.String(dump_only=True)
    detectors_dir = fields.String(dump_only=True)
    max_concurrent_dataset_downloads = fields.Integer(dump_only=True)
    max_concurrent_dataset_embeddings = fields.Integer(dump_only=True)
    autorun_detectors = fields.List(fields.String(), dump_only=True)
    dataset_max_age_days = fields.Integer(load_default=None, allow_none=True)
    # Effective ``{plugin_family: [name, ...]}`` hide map (the persisted
    # ``hidden_plugins`` server setting unioned with any ``--hide-plugin``
    # CLI flags). Populated by the route from
    # ``settings.get_effective_hidden_plugins()``; read-only.
    hidden_plugins = fields.Dict(keys=fields.String(), values=fields.List(fields.String()), dump_only=True)

    # Per-user, ``{media_type_id: embedder_name}``; the dataset-importer
    # modal pre-selects the last embedder the user picked for each media
    # type from this map when no loaded dataset is around to supply the
    # same hint via ``guessedMediaEmbedder``.
    last_embedder_per_media_type = _PerMediaTypeStringDict()

    # Per-user, ``{media_type_id: {embedder, clipper, clipper_params,
    # source_specs}}``, which defaults the Add Dataset modal auto-fills when
    # the user picks an importer whose output is the matching mediaType.
    # See ``UserSettings.import_defaults_by_media_type`` for the value
    # shape; the field is declared as a free-form dict here because the
    # nested values are heterogeneous (string, dict, list).
    import_defaults_by_media_type = fields.Dict(keys=fields.String(), values=fields.Raw())

    # Solo mediaType streamlining. ``solo_media_type`` is the user's raw
    # value (None or a media-type id); ``solo_media_type_explicit`` is True
    # once the user has changed it from settings (so the CLI fallback no
    # longer applies). ``effective_solo_media_type`` is the resolver's
    # output the frontend should read to decide whether to hide mediaType
    # pickers; it accounts for the CLI fallback and the explicit flag.
    solo_media_type = fields.String(allow_none=True)
    solo_media_type_explicit = fields.Boolean()
    effective_solo_media_type = fields.String(allow_none=True, dump_only=True)

    # Solo mediaEmbedder per mediaType. ``solo_embedder_per_media_type`` is
    # the user's raw map (``{media_type_id: embedder_name}``);
    # ``effective_solo_embedder_per_media_type`` is the resolver's view
    # (user map layered over the ``--solo-embedder`` CLI fallback), and is
    # what the frontend reads to decide whether to hide the embedder
    # picker for a given type.
    solo_embedder_per_media_type = _PerMediaTypeStringDict()
    effective_solo_embedder_per_media_type = _PerMediaTypeStringDict(dump_only=True)

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
    label_hint_dismissed = fields.Boolean()

    view_mode_left = fields.Raw()
    view_mode_right = fields.Raw()
    grid_icon_size_left = fields.Raw()
    grid_icon_size_right = fields.Raw()
    focus_mode_left = fields.Raw()
    focus_mode_right = fields.Raw()
    panel_pct_left = fields.Raw()
    panel_pct_right = fields.Raw()
    view_mode_popup = fields.Raw()
    grid_icon_size_popup = fields.Raw()

    autopilot_enabled = fields.Boolean()
    hide_autopilot = fields.Boolean()
    autopilot_top_greens = fields.Integer()
    autopilot_hard_reds = fields.Integer()
    autopilot_resort_interval = fields.Integer()
    autopilot_goal_diversity = fields.Integer()
    enable_achievements = fields.Boolean()

    browse_panel_width = fields.Integer()

    # Per-media-type dicts; the setters in ``settings.py`` validate each
    # value against its enum (BinShape / BrowseColormap / BrowseIconSize),
    # so these are declared ``Raw`` here like the other per-media settings.
    browse_bin_shape = fields.Raw()
    browse_colormap = fields.Raw()
    browse_icon_size = fields.Raw()
    browse_thumbnail_border = fields.Raw()
    browse_compact = fields.Raw()

    autorun_detectors = fields.List(fields.String())

    saved_datasets_dir = fields.String()
    detectors_dir = fields.String()
    dataset_max_age_days = fields.Integer(allow_none=True)

    last_embedder_per_media_type = fields.Raw()
    import_defaults_by_media_type = fields.Raw()

    # The route layer validates ``solo_media_type`` against the media-type
    # registry and applies it via ``apply_user_solo_media_type`` so the
    # ``solo_media_type_explicit`` flag flips automatically. Accept either
    # a string id or ``null`` for "show everything".
    solo_media_type = fields.String(allow_none=True)

    # ``{media_type_id: embedder_name}`` map. The route layer validates
    # each entry against the embedder registry and rejects unknown
    # type/embedder pairs with a 400. Sending ``null`` or an empty dict
    # clears every per-type lock. Sending ``{"image": ""}`` clears just
    # the image lock while leaving the rest in place.
    solo_embedder_per_media_type = fields.Raw(allow_none=True)

    class Meta:
        # Reject keys we don't know; the frontend should never send
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
