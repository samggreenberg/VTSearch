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

from vtscore.config import DATA_DIR, DEFAULT_CALIBRATE_COUNT, PROJECTION_MIN_DIST, PROJECTION_N_NEIGHBORS

__all__ = [
    "BROWSE_MOUSE_ZOOMS_PER_LEVEL",
    "BROWSE_THUMBNAIL_BORDER_PX",
    "POPUP_PREVIEW_SIZE_PX",
    "AnimationMode",
    "BrowseColormap",
    "BrowseGraphics",
    "BrowseIconSize",
    "FocusMode",
    "GridIconSize",
    "ServerSettings",
    "Theme",
    "UserSettings",
    "VALID_ANIMATION_MODES",
    "VALID_BROWSE_COLORMAPS",
    "VALID_BROWSE_GRAPHICS",
    "VALID_BROWSE_ICON_SIZES",
    "VALID_FOCUS_MODES",
    "VALID_GRID_ICON_SIZES",
    "VALID_PANEL_PX",
    "VALID_THEMES",
]


Theme = Literal["dark", "light", "highviz", "system"]


# Decorative-motion master switch. ``"show"`` forces animations on even when the
# OS asks for reduced motion; ``"hide"`` always suppresses them; ``"os"`` defers
# to the platform ``prefers-reduced-motion`` preference.
AnimationMode = Literal["show", "hide", "os"]
GridIconSize = Literal["XS", "S", "M", "L", "XL"]
FocusMode = Literal["click", "hover"]
# VTSBrowse density colormap preset. ``auto`` follows the active theme (Ocean
# in light mode, Heat in dark/high-viz); the rest lock to a specific map.
BrowseColormap = Literal["auto", "heat", "ocean", "gray"]
# VTSBrowse on-screen cell size. Extends the grid icon size label set with four
# larger steps (2XL..5XL): the browse canvas's bigger/smaller buttons walk nine
# zoom levels, index-aligned with the frontend ``ICON_SIZES`` array. The largest
# steps render a cell close to the full media, so the canvas serves the original
# image rather than a low-res thumbnail at those sizes.
BrowseIconSize = Literal["XS", "S", "M", "L", "XL", "2XL", "3XL", "4XL", "5XL"]
# VTSBrowse canvas rendering effort. The browse canvas's pan/zoom animations
# lean on operations a GPU does for free but a software rasterizer does not
# (smoothed full-canvas blits, overscanned snapshot buffers, shadow blurs), so
# a browser running without hardware acceleration pans and zooms badly.
# ``full`` always runs the rich pipeline, ``reduced`` always runs the cheap one
# (same animations, minus the effects a CPU rasterizer can't afford), and
# ``auto`` (the default) picks per client — probing the WebGL renderer for a
# software rasterizer and, failing that, latching on measured frame costs.
BrowseGraphics = Literal["auto", "full", "reduced"]

VALID_THEMES: tuple[str, ...] = ("dark", "light", "highviz", "system")
VALID_ANIMATION_MODES: tuple[str, ...] = ("show", "hide", "os")
VALID_GRID_ICON_SIZES: tuple[str, ...] = ("XS", "S", "M", "L", "XL")
VALID_FOCUS_MODES: tuple[str, ...] = ("click", "hover")
VALID_BROWSE_COLORMAPS: tuple[str, ...] = ("auto", "heat", "ocean", "gray")
VALID_BROWSE_GRAPHICS: tuple[str, ...] = ("auto", "full", "reduced")
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
# Allowed range for the VTSBrowse "mouse-zooms per pyramid level" setting: how
# many wheel notches / +/- button clicks it takes to cross one pyramid level
# (a full 2x of zoom). The per-click/per-notch width factor is ``2 ** (1 / n)``,
# so ``1`` ⇒ 2x per step (one step = one level), ``2`` ⇒ √2 (the default, two
# steps per level), ``3`` ⇒ ∛2 (three steps per level). Clamped on read/write.
BROWSE_MOUSE_ZOOMS_PER_LEVEL: tuple[int, int] = (1, 3)
# Allowed range (CSS px) for the VTSBrowse bin-popup detail-canvas (the large
# single-item preview) side. The popup's own size buttons drive it; values are
# clamped into this range on read/write. The floor keeps a usable preview; the
# ceiling matches the frontend's absolute preview cap.
POPUP_PREVIEW_SIZE_PX: tuple[int, int] = (96, 720)

# Default recipient for the Help modal's "Email us" affordance. A per-instance
# operator can override it with the ``--support-email`` CLI flag or the
# persisted ``support_email`` server setting (see
# ``vtsearch.settings.get_effective_support_email``).
DEFAULT_SUPPORT_EMAIL: str = "sam.greenberg@gmail.com"


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


#: Cap the per-media-type custom signpost vocabulary. Each term costs one text
#: embed at build time, so an unbounded list would let a single setting write
#: stall every projection build; AudioSet-527 / OpenImages-600 sit well under it.
SIGNPOST_VOCAB_MAX_TERMS = 2000


def _normalize_signpost_vocab(v: Any) -> Any:
    """Clean a ``{media_type: [term, ...]}`` custom-vocabulary map.

    Strips whitespace, drops blank/duplicate terms (order-preserving), caps each
    list at :data:`SIGNPOST_VOCAB_MAX_TERMS`, and drops media types left with no
    terms so an empty list reads as "use the built-in vocabulary". Non-dict or
    non-list input is passed through untouched for Pydantic to reject.
    """
    if not isinstance(v, dict):
        return v
    out: dict[str, list[str]] = {}
    for media_type, terms in v.items():
        if not isinstance(terms, list):
            return v
        seen: set[str] = set()
        clean: list[str] = []
        for term in terms:
            if not isinstance(term, str):
                continue
            stripped = term.strip()
            if not stripped or stripped in seen:
                continue
            seen.add(stripped)
            clean.append(stripped)
        if clean:
            out[str(media_type)] = clean[:SIGNPOST_VOCAB_MAX_TERMS]
    return out


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

    # Recipient address for the Help modal's "Email us" contact affordance.
    # Shared across all users (single-file server tier); a per-instance
    # operator overrides it with the ``--support-email`` CLI flag (which wins
    # for the process lifetime) or by editing this key in the settings file.
    # The frontend reads the effective value from ``/api/settings`` to build
    # the ``mailto:`` link. See
    # :func:`vtsearch.settings.get_effective_support_email`.
    support_email: str = DEFAULT_SUPPORT_EMAIL

    # Lock this deployment to **Semantic** embedders only.  The Patch Semantic
    # and Structural embedder types are still prototypes; an operator running a
    # production instance can hide them wholesale rather than naming each
    # prototype embedder in ``hidden_plugins``.  When true, ``GET
    # /api/embedders`` drops every patch/structural embedder (so the Add Dataset
    # "Advanced" block shows no Region / Instance embedder pickers), the
    # New-detector modal offers no embedder-type choice, and the dataset-load /
    # detector-create routes reject a non-semantic type outright.  Set with the
    # ``--semantic-only`` CLI flag / ``VTSEARCH_SEMANTIC_ONLY`` env var
    # (process-wide, wins for the process lifetime) or by editing this key in
    # the settings file.  See
    # :func:`vtsearch.settings.get_effective_semantic_only`.
    semantic_only: bool = False

    # Solo-mediaType streamlining. An admin-set restriction: when set, the
    # importer and new-detector flows hide their mediaType pickers and lock to
    # this type, the converter picker filters to converters whose output is
    # this type, and the mediaType-picking step in tabbed UIs is skipped.
    # ``None`` (the default) means "show everything" (the non-streamlined
    # experience). Users cannot change it - it exists so an operator can
    # narrow the app down to the one media type their deployment is for,
    # both to show less and to ask fewer questions. Set with the
    # ``--solo-media-type`` CLI flag (process-wide, wins for the process
    # lifetime) or by editing this key in the settings file. See
    # :func:`vtsearch.settings.get_effective_solo_media_type`.
    solo_media_type: str | None = None

    # Browse UMAP projection knobs (Stage 1).  They change the map layout, so
    # a per-deployment operator may want to tune them.  The persisted
    # projection is keyed on these values (see
    # ``vtscore.projection.store.projection_params_match``), so a change
    # forces a recompute instead of serving a layout fit under the old params.
    projection_n_neighbors: Annotated[int, _clamp(2, 200)] = PROJECTION_N_NEIGHBORS
    projection_min_dist: Annotated[float, _clamp(0.0, 0.99)] = PROJECTION_MIN_DIST

    # Per-media-type zero-shot tag vocabulary used to name map regions in Tags
    # mode, replacing the built-in AudioSet-527 / OpenImages-600 lists (one
    # entry per media type, a list of terms). Server-tier and read-only over
    # the API: a term list is a deployment-level choice an operator makes for
    # the whole instance (a domain-specific taxonomy — bird species, machine
    # faults, product categories), not a preference an individual arrives
    # with, and every user of an instance should read the same region names.
    # Set it by editing this key in the settings file. Normalized on write
    # (trimmed, de-duplicated, capped at
    # :data:`SIGNPOST_VOCAB_MAX_TERMS`); a media type with an empty or absent
    # list falls back to the shipped vocabulary. Takes effect on the next
    # projection build / Re-project, since signpost texts are cached.
    browse_signpost_vocab: Annotated[dict[str, list[str]], BeforeValidator(_normalize_signpost_vocab)] = Field(
        default_factory=dict
    )

    # Deployment-wide default settings sync source. Same
    # ``{"source_name": ..., "field_values": ...}`` shape as the per-user
    # ``settings_source`` key, but shared across the whole instance. A user
    # with no explicit ``settings_source`` of their own inherits this; a user
    # whose ``settings_source`` is ``{"source_name": "none"}`` explicitly opts
    # out (no source, even when a default exists). ``None`` (the default)
    # means there is no deployment-wide source. The precedence
    # (user-explicit > deployment-default > none) is resolved in one place -
    # :meth:`vtsearch.settings_store.UserSettingsStore.resolve_settings_source` -
    # so every sync path agrees. To personalise per user, template the
    # ``field_values`` (e.g. ``{"filepath": "data/user-settings/{username}.json"}``)
    # just as with the per-user key.
    default_settings_source: dict[str, Any] | None = None


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
    calibrate_count: Annotated[int, _clamp(1, 100)] = DEFAULT_CALIBRATE_COUNT
    # ``None`` (the default) means "no explicit split": training resolves it
    # to the per-embedder production default - 0.3 when the detector learns in
    # a single-vector space, 0.5 on a patch grid (issue #3287).  A stored
    # float is an explicit user choice and always wins; clearing the field in
    # the GUI writes ``None`` and returns to automatic.
    calibration_fraction: Annotated[float | None, _clamp(0.0, 1.0)] = None
    audio_playing: bool = True
    # Master switch for decorative motion (vote swipe, icon spins/waggles/tilts,
    # toast/banner slide-ins, smooth scrolling, projection-browser zoom tweens).
    # ``"show"`` (the default) forces motion on even against an OS reduce-motion
    # request, ``"hide"`` always suppresses it, and ``"os"`` follows the
    # platform ``prefers-reduced-motion`` preference. See the "Show Animations"
    # pulldown in the appearance settings.
    show_animations: AnimationMode = "show"
    show_metadata: bool = False
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

    # VTSBrowse docked bin-details panel width (CSS px). When the bin-details
    # window is docked (``bin_details_docked``), it renders as a left panel
    # beside the canvas, separated by a draggable divider; this persists the
    # width the user dragged it to. Not surfaced as a Settings-modal widget -
    # the divider drives it. Clamped to a sane on-screen range.
    browse_details_panel_width: Annotated[int, _clamp(220, 800)] = 340

    # VTSBrowse docked bin-details metadata-column width (CSS px). When the
    # docked bin-details panel shows its metadata column, a draggable divider
    # separates that column from the large item beside it; this persists the
    # width the user dragged the metadata column to. The large item takes
    # whatever the panel leaves after the metadata column, so it always grows
    # to fill the space (there is no per-item size control in the docked panel).
    # Defaults to just enough for an MD5 digest to wrap onto two rows. Not
    # surfaced as a Settings-modal widget - the divider drives it. Clamped to a
    # sane on-screen range.
    browse_details_metadata_width: Annotated[int, _clamp(120, 600)] = 150

    # VTSBrowse canvas rendering effort. Unlike the per-media-type prefs below
    # this is a single scalar, because it describes the *client's* rendering
    # capability rather than anything about the data: a browser without
    # hardware acceleration rasterizes every canvas paint on the CPU, where the
    # animation pipeline's smoothed full-canvas blits, overscanned snapshots and
    # shadow blurs cost tens of ms per frame and make pan/zoom lag. ``reduced``
    # keeps every animation but strips those effects; ``full`` always runs the
    # rich pipeline; ``auto`` (the default) detects per client. Surfaced as the
    # "Graphics" pulldown at the top of the Settings -> Browser tab.
    browse_graphics: BrowseGraphics = "auto"

    # VTSBrowse per-media-type display preferences. Each is a
    # ``{media_type_id: value}`` dict so a user can tune the projection
    # browser independently for, say, audio vs. image datasets. Empty
    # entries fall back to the per-type default on the frontend
    # (``auto`` colormap, ``M`` size). Driven by the toolbar toggles on the
    # browse canvas AND the Settings → Browser tab; both write the same
    # maps keyed by the active dataset's media type.
    #
    # (The bin shape — hexagons vs squares — is *not* a stored preference: it
    # is fixed by the media type, squares for browsable-thumbnail media
    # (image/video/document) and hexes otherwise. See
    # ``vtscore.projection.bin_shape_for_media_type``.)
    #
    # - ``browse_colormap``: density colormap preset (``auto`` follows the
    #   theme — Ocean in light mode, Heat in dark/high-viz).
    # - ``browse_icon_size``: on-screen cell size (XS…XL), the named form of
    #   the canvas's bigger/smaller buttons.
    # - ``browse_thumbnail_border``: width in CSS px of the colormap-coloured
    #   border drawn around multi-item ("pile") thumbnails. The band's colour is
    #   the density colour for the pile's item count, so its hue/brightness reads
    #   as the stack height under the tile. Only affects media types that paint
    #   thumbnails (image, video); ``0`` disables it. Clamped to 0..8 px.
    # - ``browse_mouse_zooms_per_level``: how many wheel notches / +/- button
    #   clicks cross one pyramid level (a full 2x). The per-step width factor is
    #   ``2 ** (1 / n)``, so 1 ⇒ 2x, 2 ⇒ √2 (default), 3 ⇒ ∛2. Clamped to 1..3;
    #   empty entries fall back to 2 on the frontend.
    # - ``browse_signposts``: whether the canvas draws region signposts — the
    #   named "street sign" labels lettered over the map (see
    #   docs/plans/vtsbrowse-toponymy.md) — when the projection has labels.
    #   Toggled by the signpost button on the browse canvas; empty entries
    #   fall back to on (true) on the frontend.
    browse_colormap: dict[str, BrowseColormap] = Field(default_factory=dict)
    browse_icon_size: dict[str, Annotated[BrowseIconSize, BeforeValidator(_upper)]] = Field(default_factory=dict)
    browse_thumbnail_border: dict[str, Annotated[int, _clamp(*BROWSE_THUMBNAIL_BORDER_PX)]] = Field(
        default_factory=dict
    )
    browse_mouse_zooms_per_level: dict[str, Annotated[int, _clamp(*BROWSE_MOUSE_ZOOMS_PER_LEVEL)]] = Field(
        default_factory=dict
    )
    browse_signposts: dict[str, bool] = Field(default_factory=dict)
    # - ``browse_signpost_captioner``: whether a media type's signpost texts come
    #   from the generative captioner (image VLM / audio captioner) instead of
    #   the default zero-shot tags. Empty entries fall back to tags (false).
    browse_signpost_captioner: dict[str, bool] = Field(default_factory=dict)

    grid_icon_size_left: dict[str, Annotated[GridIconSize, BeforeValidator(_upper)]] = Field(default_factory=dict)
    grid_icon_size_right: dict[str, Annotated[GridIconSize, BeforeValidator(_upper)]] = Field(default_factory=dict)
    focus_mode_left: dict[str, FocusMode] = Field(default_factory=dict)
    focus_mode_right: dict[str, FocusMode] = Field(default_factory=dict)

    # VTSBrowse bin-popup thumbnail size, per media type. The right-click bin
    # popup renders the bin's members as a thumbnail grid (always grid; there is
    # no list mode) beside a large hover-preview pane, with its own size control
    # independent of the left/right panels. Empty entries fall back on the
    # frontend to ``M``. Driven by the popup's own size buttons AND the
    # Settings → Browser tab; both write the same map keyed by the active
    # dataset's media type, so tuning the popup while browsing one bin becomes
    # the default for every future popup of that media type. Unlike
    # ``grid_icon_size_{left,right}`` this is a plain per-media-type dict (no
    # per-side machinery): the popup is a single, third context, so it uses the
    # generic Pydantic-driven accessors.
    grid_icon_size_popup: dict[str, Annotated[GridIconSize, BeforeValidator(_upper)]] = Field(default_factory=dict)
    # VTSBrowse bin-popup detail-canvas size, per media type (CSS px). The popup
    # opens its hovered/representative item in a large square preview pane beside
    # the member grid; this is the user's chosen side for that pane, driven by the
    # popup's own top-left size buttons (independent of the grid thumbnail size in
    # ``grid_icon_size_popup``). Empty entries fall back on the frontend to a size
    # derived from the main-canvas thumbnail radius. Clamped to
    # ``POPUP_PREVIEW_SIZE_PX``.
    popup_preview_size: dict[str, Annotated[int, _clamp(*POPUP_PREVIEW_SIZE_PX)]] = Field(default_factory=dict)
    # VTSBrowse bin-popup metadata column, per media type. The popup can show a
    # column to the left of the detail-canvas preview carrying the same
    # name/media-type/custom-metadata/MD5 fields the Train/Find center panel
    # shows for the focused item; this remembers whether that column is shown for
    # each media type. Empty entries fall back on the frontend to shown (mirroring
    # the center panel's default). Only image/video popups have a preview pane and
    # thus a focused item to attach the column to, but the flag is stored per media
    # type generically.
    popup_metadata_shown: dict[str, bool] = Field(default_factory=dict)
    # VTSBrowse bin-details presentation, per media type. When true, the bin
    # details open docked as a left panel beside the canvas (large item +
    # metadata on top, the bin's member grid below) instead of the floating
    # right-click popup window. Driven by the dock button on the floating
    # window and the pop-out button on the docked panel; empty entries fall
    # back to the docked left panel (true) — the pop-out button persists an
    # explicit ``false`` for a media type the user chose to float instead.
    bin_details_docked: dict[str, bool] = Field(default_factory=dict)
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
