"""Pydantic models for VTSearch settings.

These models are the source of truth for setting types, defaults, ranges,
and enum membership. :mod:`vtsearch.settings` reads ``model_fields`` to
generate the ``get_<key>`` / ``set_<key>`` accessor pairs, and the JSON
schema is available for free via :meth:`pydantic.BaseModel.model_json_schema`.

Two models, mirroring the two storage tiers:

* :class:`ServerSettings` - keys persisted to ``data/settings.json``
  (shared across users; loaded once at startup).
* :class:`UserSettings` - keys persisted to
  ``<get_user_data_dir(user)>/user_settings.json`` (per-user; resolved
  per-request via :func:`vtsearch.auth.get_current_user`).

The settings module keeps storage as plain ``dict[str, Any]`` so
free-form sub-objects (``achievement_state``, ``settings_source``) can
live alongside the typed keys. The Pydantic models only validate the
typed keys; ``extra = "allow"`` is set so extra keys round-trip without
loss.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from vtscore.config import DATA_DIR, DEFAULT_CALIBRATE_COUNT

__all__ = [
    "BROWSE_THUMBNAIL_BORDER_PX",
    "BinShape",
    "BrowseColormap",
    "BrowseIconSize",
    "FocusMode",
    "GridIconSize",
    "ServerSettings",
    "Theme",
    "UserSettings",
    "VALID_BIN_SHAPES",
    "VALID_BROWSE_COLORMAPS",
    "VALID_BROWSE_ICON_SIZES",
    "VALID_FOCUS_MODES",
    "VALID_GRID_ICON_SIZES",
    "VALID_PANEL_PX",
    "VALID_THEMES",
    "VALID_VIEW_MODES",
    "ViewMode",
]


Theme = Literal["dark", "light", "highviz", "system"]
ViewMode = Literal["grid", "list"]
GridIconSize = Literal["XS", "S", "M", "L", "XL"]
FocusMode = Literal["click", "hover"]
BinShape = Literal["hex", "square"]
# VTSBrowse density colormap preset. ``auto`` follows the active theme (Ocean
# in light mode, Heat in dark/high-viz); the rest lock to a specific map.
BrowseColormap = Literal["auto", "heat", "ocean", "gray"]
# VTSBrowse on-screen cell size. Extends the grid icon size label set with four
# larger steps (2XL..5XL): the browse canvas's bigger/smaller buttons walk nine
# zoom levels, index-aligned with the frontend ``ICON_SIZES`` array. The largest
# steps render a cell close to the full media, so the canvas serves the original
# image rather than a low-res thumbnail at those sizes.
BrowseIconSize = Literal["XS", "S", "M", "L", "XL", "2XL", "3XL", "4XL", "5XL"]

VALID_THEMES: tuple[str, ...] = ("dark", "light", "highviz", "system")
VALID_VIEW_MODES: tuple[str, ...] = ("grid", "list")
VALID_GRID_ICON_SIZES: tuple[str, ...] = ("XS", "S", "M", "L", "XL")
VALID_FOCUS_MODES: tuple[str, ...] = ("click", "hover")
VALID_BIN_SHAPES: tuple[str, ...] = ("hex", "square")
VALID_BROWSE_COLORMAPS: tuple[str, ...] = ("auto", "heat", "ocean", "gray")
VALID_BROWSE_ICON_SIZES: tuple[str, ...] = ("XS", "S", "M", "L", "XL", "2XL", "3XL", "4XL", "5XL")
# Allowed range (CSS px) for the saved left/right panel widths. The floor keeps
# a panel usable; the ceiling is a sanity bound only. The frontend resize logic
# already constrains a panel to the available layout space (viewport minus the
# dividers, the center column, and the opposite panel), so legitimate widths
# never approach this ceiling even on very large displays. Out-of-range values
# raise on write and clamp on read.
VALID_PANEL_PX: tuple[int, int] = (150, 10000)
# Allowed range (CSS px) for the VTSBrowse pile-thumbnail border width. ``0``
# disables the border; values are clamped into this range on read/write.
BROWSE_THUMBNAIL_BORDER_PX: tuple[int, int] = (0, 8)


def _clamp(lo: float, hi: float):
    """Return a :class:`BeforeValidator` that clamps numeric input to ``[lo, hi]``."""

    def _coerce(v: Any) -> Any:
        try:
            return max(lo, min(hi, float(v)))
        except (TypeError, ValueError):
            return v

    return BeforeValidator(_coerce)


def _clamp_min(lo: float):
    """Return a :class:`BeforeValidator` that clamps numeric input to ``>= lo``."""

    def _coerce(v: Any) -> Any:
        try:
            return max(lo, float(v))
        except (TypeError, ValueError):
            return v

    return BeforeValidator(_coerce)


def _upper(v: Any) -> Any:
    """Uppercase incoming strings (used for case-insensitive grid icon sizes)."""
    return v.upper() if isinstance(v, str) else v


def _default_concurrent_downloads() -> int:
    """Lazily resolve the hardware-derived default for parallel downloads.

    Imported lazily to avoid pulling :mod:`vtscore.embedding.loader` (and
    transitively torch) at settings-model import time.
    """
    from vtscore.embedding.loader import default_concurrent_downloads

    return default_concurrent_downloads()


def _default_concurrent_embeddings() -> int:
    """Lazily resolve the hardware-derived default for parallel embeddings."""
    from vtscore.embedding.loader import default_concurrent_embeddings

    return default_concurrent_embeddings()


class ServerSettings(BaseModel):
    """Server-tier (shared) settings persisted in ``data/settings.json``."""

    model_config = ConfigDict(extra="allow")

    saved_datasets_dir: str = Field(default_factory=lambda: str(DATA_DIR / "saved_datasets"))
    detectors_dir: str = Field(default_factory=lambda: str(DATA_DIR / "detectors"))
    # Defaults derive from cores/GPUs at first read (not persisted to disk);
    # a manual override in ``data/settings.json`` always wins.
    max_concurrent_dataset_downloads: Annotated[int, _clamp(1, 16)] = Field(
        default_factory=_default_concurrent_downloads
    )
    max_concurrent_dataset_embeddings: Annotated[int, _clamp(1, 16)] = Field(
        default_factory=_default_concurrent_embeddings
    )
    # Admin-side plugin hiding. Maps a plugin-family id (e.g.
    # ``"converters"``, ``"embedders"``, ``"importers"`` - the keys used by
    # :mod:`vtscore.plugins.inventory`) to a list of plugin ``name``s that
    # should be omitted from picker / listing API responses for this
    # deployment. Hidden plugins remain importable and callable by name
    # via execution endpoints; this is a UI-declutter setting, not a
    # security boundary. Merged at read time with any
    # ``--hide-plugin family:name`` flags passed on the CLI - see
    # :func:`vtsearch.settings.get_effective_hidden_plugins`.
    hidden_plugins: dict[str, list[str]] = Field(default_factory=dict)

    # Maximum age (in days) for datasets created on this server.  New
    # datasets are stamped with an ``expires_at`` timestamp based on this
    # value.  ``None`` (the default) means datasets never expire.
    dataset_max_age_days: int | None = None


class UserSettings(BaseModel):
    """Per-user settings persisted in ``<user_data_dir>/user_settings.json``."""

    model_config = ConfigDict(extra="allow")

    volume: Annotated[float, _clamp(0.0, 1.0)] = 1.0
    inclusion: Annotated[int, _clamp(-10, 10)] = 0
    # ``"system"`` resolves to the OS ``prefers-color-scheme`` value
    # (dark or light) at render time on the frontend. Users can pick a
    # concrete theme to opt out and return to "system" to opt back in.
    theme: Theme = "system"
    enrich_descriptions: bool = False
    safe_thresholds: bool = False
    calibrate_count: Annotated[int, _clamp(1, 100)] = DEFAULT_CALIBRATE_COUNT
    calibration_fraction: Annotated[float, _clamp(0.0, 1.0)] = 0.5
    audio_playing: bool = True
    # Master switch for decorative motion (vote swipe, icon spins/waggles/tilts,
    # toast/banner slide-ins, smooth scrolling, projection-browser zoom tweens).
    # When False the frontend mirrors the OS "reduce motion" behavior. See the
    # "Show Animations" checkbox in the appearance settings.
    show_animations: bool = True
    show_metadata: bool = True
    # Set to True once the user dismisses the zero-votes "Use ← / → or click"
    # hint that overlays the Good/Bad buttons when a fresh labeling session
    # has no votes yet. Persisting it keeps the hint from re-appearing every
    # time the same user starts a new session.
    label_hint_dismissed: bool = False
    autopilot_enabled: bool = True
    hide_autopilot: bool = False
    # When False, the Achievements tab/button and unlock pop-ups are
    # hidden, every ``record_*`` hook is a no-op, and ``get_full_state``
    # returns zeroed counters with no pending announcements. Flipping it
    # off also wipes any stored ``achievement_state`` so the counters
    # start at zero if the user ever turns the feature back on. Defaults
    # to True (achievements on). See the ``enable_achievements`` route
    # handler in ``vtsearch/routes/settings/api.py``.
    enable_achievements: bool = True
    autopilot_top_greens: Annotated[int, _clamp_min(1)] = 3
    autopilot_hard_reds: Annotated[int, _clamp_min(1)] = 4
    autopilot_resort_interval: Annotated[int, _clamp_min(1)] = 10
    autopilot_goal_diversity: Annotated[int, _clamp_min(1)] = 40

    # Auto-Find: each user's own list of detectors that auto-run against a newly
    # imported dataset, plus what to do with the results. Per-user (everyone
    # curates their own favorites from the shared detector pool). For the
    # built-in "default" user, reads fall back to the server settings file when
    # absent here, so the CLI ``--settings`` flat file and single-user
    # deployments keep working (see ``_DEFAULT_USER_FALLBACK_KEYS`` and the
    # read-through in ``vtsearch.settings._read_value``).
    #
    # - ``autofind_detectors``: detector names flagged for Auto-Find (each maps
    #   to a JSON file under ``data/detectors/``).
    # - ``autofind_exporter``: results-exporter name run after an Auto-Find
    #   (``""`` = no auto-export; CLI then falls back to the ``gui`` exporter).
    # - ``autofind_exporter_field_values``: per-exporter field values
    #   (``{exporter_name: {field_key: value}}``) so switching the picker
    #   preserves each exporter's configuration.
    # See ``docs/plans/auto-find-settings-tab.md``.
    autofind_detectors: list[str] = Field(default_factory=list)
    autofind_exporter: str = ""
    autofind_exporter_field_values: dict[str, dict[str, str]] = Field(default_factory=dict)

    # VTSBrowse side-panel width (CSS px). The browse view docks a
    # selection panel (selected-item grid + the legend and overview
    # minimap) to the right of the canvas, separated by a draggable
    # divider; this persists the width the user dragged it to so the
    # panel comes back the same way on the next visit. Not surfaced as a
    # Settings-modal widget - the divider drives it. Clamped to a sane
    # on-screen range.
    browse_panel_width: Annotated[int, _clamp(260, 800)] = 360

    # VTSBrowse per-media-type display preferences. Each is a
    # ``{media_type_id: value}`` dict so a user can tune the projection
    # browser independently for, say, audio vs. image datasets. Empty
    # entries fall back to the per-type default on the frontend (hex,
    # ``auto`` colormap, ``M`` size). Driven by the toolbar toggles on the
    # browse canvas AND the Settings → Browser tab; both write the same
    # maps keyed by the active dataset's media type.
    #
    # - ``browse_bin_shape``: tile the projection as hexagons (default) or
    #   squares, per media type.
    # - ``browse_colormap``: density colormap preset (``auto`` follows the
    #   theme — Ocean in light mode, Heat in dark/high-viz).
    # - ``browse_icon_size``: on-screen cell size (XS…XL), the named form of
    #   the canvas's bigger/smaller buttons.
    # - ``browse_thumbnail_border``: width in CSS px of the colormap-coloured
    #   border drawn around multi-item ("pile") thumbnails. The band's colour is
    #   the density colour for the pile's item count, so its hue/brightness reads
    #   as the stack height under the tile. Only affects media types that paint
    #   thumbnails (image, video); ``0`` disables it. Clamped to 0..8 px.
    # - ``browse_compact``: whether the UMAP layout is compacted — clusters slid
    #   together as rigid bodies to close the empty "oceans" between islands —
    #   when the projection is (re)built. Unlike the others this affects the
    #   Stage-1 coordinates, which are computed once and frozen, so a change
    #   takes effect on the next fresh build or the Browser's Re-project action,
    #   not retroactively on an already-built layout. Defaults to on per type.
    browse_bin_shape: dict[str, BinShape] = Field(default_factory=dict)
    browse_colormap: dict[str, BrowseColormap] = Field(default_factory=dict)
    browse_icon_size: dict[str, Annotated[BrowseIconSize, BeforeValidator(_upper)]] = Field(default_factory=dict)
    browse_thumbnail_border: dict[str, Annotated[int, _clamp(*BROWSE_THUMBNAIL_BORDER_PX)]] = Field(
        default_factory=dict
    )
    browse_compact: dict[str, bool] = Field(default_factory=dict)

    view_mode_left: dict[str, ViewMode] = Field(default_factory=dict)
    view_mode_right: dict[str, ViewMode] = Field(default_factory=dict)
    grid_icon_size_left: dict[str, Annotated[GridIconSize, BeforeValidator(_upper)]] = Field(default_factory=dict)
    grid_icon_size_right: dict[str, Annotated[GridIconSize, BeforeValidator(_upper)]] = Field(default_factory=dict)
    focus_mode_left: dict[str, FocusMode] = Field(default_factory=dict)
    focus_mode_right: dict[str, FocusMode] = Field(default_factory=dict)

    # VTSBrowse bin-popup display prefs, per media type. The right-click bin
    # popup renders the bin's members like a mini Find panel with its own
    # List/Grid + thumbnail-size controls, independent of the left/right
    # panels. Empty entries fall back on the frontend to grid + ``M``.
    # Driven by the popup's own view-controls AND the Settings → Browser tab;
    # both write the same maps keyed by the active dataset's media type, so
    # tuning the popup while browsing one bin becomes the default for every
    # future popup of that media type. Unlike ``view_mode_{left,right}`` these
    # are plain per-media-type dicts (no per-side machinery): the popup is a
    # single, third context, so it uses the generic Pydantic-driven accessors.
    view_mode_popup: dict[str, ViewMode] = Field(default_factory=dict)
    grid_icon_size_popup: dict[str, Annotated[GridIconSize, BeforeValidator(_upper)]] = Field(default_factory=dict)
    panel_pct_left: dict[str, int] = Field(default_factory=dict)
    panel_pct_right: dict[str, int] = Field(default_factory=dict)

    # Per-media-type memory of the last embedder the user picked, used by the
    # dataset importer modal to pre-select a sensible default when no loaded
    # dataset is around to supply the same hint via ``guessedMediaEmbedder``.
    # Keys are canonical media-type ids (e.g. ``"image"``, ``"audio"``).
    last_embedder_per_media_type: dict[str, str] = Field(default_factory=dict)

    # Per-media-type default settings for the Add Dataset advanced panel -
    # the embedder, clipper (+ params), and source-spec converter rows the
    # user wants applied automatically every time they import a dataset of
    # that output mediaType. Set from the Settings > Data Imports tab and
    # silently auto-filled into the importer form when an importer is
    # selected. Keys are canonical media-type ids; each value is a
    # free-form dict shaped like:
    #     {
    #       "embedder": "<embedder name>" | "",
    #       "clipper": "<clipper name>" | "",
    #       "clipper_params": {"<key>": <value>, ...},
    #       "source_specs": [{"source_type": "...", "converter": "..."|null,
    #                         "params": {...}}, ...],
    #     }
    # Any missing sub-key falls back to the importer's existing default.
    import_defaults_by_media_type: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # Solo-mediaType streamlining. When set, the importer and new-detector
    # flows hide their mediaType pickers and lock to this type, the converter
    # picker filters to converters whose output is this type, and the
    # mediaType-picking step in tabbed UIs is skipped. ``None`` means "show
    # everything" (the default, non-streamlined experience).
    #
    # Resolution at read time is ``solo_media_type`` if
    # ``solo_media_type_explicit`` is True, else the process-level CLI
    # fallback (``settings.set_cli_solo_media_type``), else None. The
    # ``explicit`` flag is set to True whenever the user changes the value
    # through the settings UI - so once a user opts out (sets it to None),
    # the CLI flag no longer reapplies on future launches for that user.
    solo_media_type: str | None = None
    solo_media_type_explicit: bool = False

    # Solo mediaEmbedder streamlining (per mediaType). When a media-type id
    # appears as a key here, the dataset-importer modal hides its embedder
    # picker for that type and locks it to the named embedder. Types not
    # present in the dict keep the normal embedder dropdown. ``None`` /
    # empty-string values are treated as "not set" (no lock).
    #
    # Resolution at read time layers ``solo_embedder_per_media_type``
    # (user explicit) over the process-level CLI fallback set by
    # :func:`vtsearch.settings.set_cli_solo_embedder` - user entries win
    # per-key, missing keys fall through to the CLI value. The full merged
    # view is exposed at ``/api/settings`` as
    # ``effective_solo_embedder_per_media_type``. A stored embedder id
    # that no longer matches the registry is treated as absent by the
    # frontend (it falls back to the normal picker), so renaming or
    # removing an embedder never strands the user.
    solo_embedder_per_media_type: dict[str, str] = Field(default_factory=dict)

    # Rolling list of recent (dataset_id, detector_id, last_activity)
    # entries, capped at MAX_RECENT_SESSIONS by the route handler. Most
    # recent first. last_activity is epoch seconds (float). The
    # "Recent sessions" burger-menu submenu reads this and filters out
    # entries whose ids no longer resolve in the registries.
    recent_sessions: list[dict[str, Any]] = Field(default_factory=list)
