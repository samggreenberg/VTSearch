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

from marshmallow import Schema, fields, pre_load, validate

from vtsearch.settings_models import (
    VALID_ANIMATION_MODES,
    coerce_animation_mode,
    VALID_FOCUS_MODES,
    VALID_GRID_ICON_SIZES,
    VALID_THEMES,
)


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


class _PerMediaTypeStringListDict(fields.Dict):
    """``{media_type_id: [str, ...]}`` dict (custom signpost vocabularies)."""

    def __init__(self, **kwargs):
        super().__init__(keys=fields.String(), values=fields.List(fields.String()), **kwargs)


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
    show_animations = fields.String(validate=validate.OneOf(VALID_ANIMATION_MODES))
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

    # VTSBrowse docked bin-details panel width (CSS px). Persisted but not
    # shown as a Settings-modal widget; the panel's draggable divider drives it.
    browse_details_panel_width = fields.Integer()

    # VTSBrowse docked bin-details metadata-column width (CSS px). Persisted but
    # not shown as a Settings-modal widget; the divider between the metadata
    # column and the large item drives it.
    browse_details_metadata_width = fields.Integer()

    # VTSBrowse per-media-type display prefs (density colormap, on-screen cell
    # size). ``{media_type_id: value}`` dicts driven by the browse-canvas
    # toolbar and the Settings → Browser tab. (Bin shape is not stored — it is
    # fixed by media type; see ``vtscore.projection.bin_shape_for_media_type``.)
    browse_colormap = _PerMediaTypeStringDict()
    browse_icon_size = _PerMediaTypeStringDict()
    # Pile-thumbnail border width in CSS px, per media type (0 disables).
    browse_thumbnail_border = _PerMediaTypeIntDict()
    # Whether (re)building the projection compacts the layout, per media type.
    browse_compact = _PerMediaTypeBooleanDict()
    # Wheel notches / +/- clicks per pyramid level (1..3), per media type.
    browse_mouse_zooms_per_level = _PerMediaTypeIntDict()
    # Whether the canvas draws region signposts (named "street sign" labels
    # over the map), per media type; unset falls back to on.
    browse_signposts = _PerMediaTypeBooleanDict()
    # Whether a media type's signpost texts come from the generative captioner
    # (image VLM / audio captioner) instead of the zero-shot tags; unset =tags.
    browse_signpost_captioner = _PerMediaTypeBooleanDict()
    # Per-media-type custom zero-shot tag vocabulary replacing the built-in
    # AudioSet-527 / OpenImages-600 lists; unset/empty = the shipped vocabulary.
    browse_signpost_vocab = _PerMediaTypeStringListDict()

    # Per-user, per-media-type
    grid_icon_size_left = _PerMediaTypeStringDict()
    grid_icon_size_right = _PerMediaTypeStringDict()
    focus_mode_left = _PerMediaTypeStringDict()
    focus_mode_right = _PerMediaTypeStringDict()
    panel_pct_left = _PerMediaTypeNumberDict()
    panel_pct_right = _PerMediaTypeNumberDict()
    # VTSBrowse bin-popup thumbnail size, per media type. Driven by the popup's
    # own size buttons and the Settings → Browser tab. (The popup is grid-only.)
    grid_icon_size_popup = _PerMediaTypeStringDict()
    # VTSBrowse bin-popup detail-canvas (large single-item preview) size in CSS
    # px, per media type. Driven by the popup's own top-left size buttons.
    popup_preview_size = _PerMediaTypeIntDict()
    # VTSBrowse bin-popup metadata column visibility, per media type. Driven by
    # the popup's own metadata toggle button; when shown, a column left of the
    # detail preview carries the focused item's Train/Find metadata fields.
    popup_metadata_shown = _PerMediaTypeBooleanDict()
    # VTSBrowse bin-details presentation, per media type: true = docked left
    # panel, false/unset = floating right-click popup window. Driven by the
    # dock button on the floating window and the pop-out button on the panel.
    bin_details_docked = _PerMediaTypeBooleanDict()

    # Server-tier. These are fixed at server start (config file /
    # environment / CLI flags) and shared across all users; the frontend
    # surfaces them read-only in the "Server" settings tab. They are not
    # in ``SettingsUpdateSchema``, so the API rejects attempts to change
    # them.
    saved_datasets_dir = fields.String(dump_only=True)
    detectors_dir = fields.String(dump_only=True)
    max_concurrent_dataset_downloads = fields.Integer(dump_only=True)
    max_concurrent_dataset_embeddings = fields.Integer(dump_only=True)
    # Server-tier dataset retention policy. Set via the
    # ``--dataset-max-age-days`` CLI flag (process-wide, all users) or the
    # persisted settings file; surfaced read-only here so the dashboard can
    # gate its Age-Off column. Not in ``SettingsUpdateSchema`` - not
    # editable via PUT.
    dataset_max_age_days = fields.Integer(dump_only=True, allow_none=True)
    # Server-tier "Email us" recipient. Set via the ``--support-email`` CLI
    # flag / ``VTSEARCH_SUPPORT_EMAIL`` env var (process-wide, all users) or
    # the persisted settings file; surfaced read-only here so the Help modal
    # can build a pre-addressed ``mailto:`` link. Not in
    # ``SettingsUpdateSchema`` - not editable via PUT.
    support_email = fields.String(dump_only=True)
    # Server-tier "Semantic embedders only" lock. Set via the
    # ``--semantic-only`` CLI flag / ``VTSEARCH_SEMANTIC_ONLY`` env var
    # (process-wide, all users) or the persisted settings file; surfaced
    # read-only here so the New-detector modal can hide its embedder-type
    # picker and the Server settings tab can report the restriction. Not in
    # ``SettingsUpdateSchema`` - not editable via PUT.
    semantic_only = fields.Boolean(dump_only=True)

    # Auto-Find (per-user, editable from the Auto-Find settings tab).
    # ``autofind_detectors`` is each user's own list of detectors that auto-run
    # on import; ``autofind_exporter`` is the chosen results-exporter name
    # ("" = none); ``autofind_exporter_field_values`` keeps each exporter's
    # field values under its own name so switching the picker preserves config.
    autofind_detectors = fields.List(fields.String())
    autofind_exporter = fields.String()
    autofind_exporter_field_values = fields.Dict(
        keys=fields.String(),
        values=fields.Dict(keys=fields.String(), values=fields.String()),
    )
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
    show_animations = fields.String(validate=validate.OneOf(VALID_ANIMATION_MODES))
    show_metadata = fields.Boolean()
    label_hint_dismissed = fields.Boolean()

    @pre_load
    def _migrate_legacy_show_animations(self, data, **kwargs):
        """Fold pre-enum boolean ``show_animations`` values into the enum.

        Settings files written before the 3-way pulldown (a3a37106) stored a
        boolean; a client that fetched such a value re-sends it verbatim on
        the next save, and without this hook the ``OneOf`` validator rejects
        the entire update (every settings save 422s until the value is fixed
        by hand).  See ``vtsearch.settings_models.coerce_animation_mode``.
        """
        if isinstance(data, dict) and "show_animations" in data:
            data = dict(data)
            data["show_animations"] = coerce_animation_mode(data["show_animations"])
        return data

    grid_icon_size_left = fields.Raw()
    grid_icon_size_right = fields.Raw()
    focus_mode_left = fields.Raw()
    focus_mode_right = fields.Raw()
    panel_pct_left = fields.Raw()
    panel_pct_right = fields.Raw()
    grid_icon_size_popup = fields.Raw()
    popup_preview_size = fields.Raw()
    popup_metadata_shown = fields.Raw()
    bin_details_docked = fields.Raw()

    autopilot_enabled = fields.Boolean()
    hide_autopilot = fields.Boolean()
    autopilot_top_greens = fields.Integer()
    autopilot_hard_reds = fields.Integer()
    autopilot_resort_interval = fields.Integer()
    autopilot_goal_diversity = fields.Integer()
    enable_achievements = fields.Boolean()

    browse_panel_width = fields.Integer()
    browse_details_panel_width = fields.Integer()
    browse_details_metadata_width = fields.Integer()

    # Per-media-type dicts; the setters in ``settings.py`` validate each
    # value against its enum (BrowseColormap / BrowseIconSize), so these are
    # declared ``Raw`` here like the other per-media settings.
    browse_colormap = fields.Raw()
    browse_icon_size = fields.Raw()
    browse_thumbnail_border = fields.Raw()
    browse_compact = fields.Raw()
    browse_mouse_zooms_per_level = fields.Raw()
    browse_signposts = fields.Raw()
    browse_signpost_captioner = fields.Raw()
    browse_signpost_vocab = fields.Raw()

    autofind_detectors = fields.List(fields.String())
    # Auto-Find results exporter. ``autofind_exporter`` is validated against the
    # exporter registry in the route layer; ``autofind_exporter_field_values``
    # is a free-form ``{exporter_name: {field_key: value}}`` map (per-field
    # validation runs at export time against the chosen plugin's schema).
    autofind_exporter = fields.String()
    autofind_exporter_field_values = fields.Raw()

    saved_datasets_dir = fields.String()
    detectors_dir = fields.String()
    # NB: dataset_max_age_days is intentionally absent - it is a server-tier
    # retention policy set via --dataset-max-age-days (or the settings file),
    # not editable via PUT /api/settings. It is dump_only in AppSettingsSchema.

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


# Note: per-side dict fields (grid_icon_size_left etc.) are declared as
# ``fields.Raw`` on the *update* schema because the existing setters
# accept either ``"M"``/``{media_type: "M"}`` and the validators
# inside ``settings.py`` are the source of truth. Tightening these to
# ``fields.Dict(...)`` is a follow-up once the per-side setters are
# unified.

# Grid sizes / focus modes are re-exported for the few call sites that import
# them from here when checking PUT body values directly (test helpers).
__all__ = [
    "AppSettingsSchema",
    "SettingsUpdateSchema",
    "VALID_FOCUS_MODES",
    "VALID_GRID_ICON_SIZES",
    "VALID_THEMES",
]
