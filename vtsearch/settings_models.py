"""Pydantic models for VTSearch settings.

These models are the source of truth for setting types, defaults, ranges,
and enum membership. :mod:`vtsearch.settings` reads ``model_fields`` to
generate the ``get_<key>`` / ``set_<key>`` accessor pairs, and the JSON
schema is available for free via :meth:`pydantic.BaseModel.model_json_schema`.

Two models, mirroring the two storage tiers:

* :class:`ServerSettings` — keys persisted to ``data/settings.json``
  (shared across users; loaded once at startup).
* :class:`UserSettings` — keys persisted to
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
    "FocusMode",
    "GridIconSize",
    "ServerSettings",
    "Theme",
    "UserSettings",
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

VALID_THEMES: tuple[str, ...] = ("dark", "light", "highviz", "system")
VALID_VIEW_MODES: tuple[str, ...] = ("grid", "list")
VALID_GRID_ICON_SIZES: tuple[str, ...] = ("XS", "S", "M", "L", "XL")
VALID_FOCUS_MODES: tuple[str, ...] = ("click", "hover")
VALID_PANEL_PX: tuple[int, int] = (150, 500)


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
    autorun_detectors: list[str] = Field(default_factory=list)


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
    swipe_animation: bool = True
    show_metadata: bool = True
    # Set to True once the user dismisses the zero-votes "Use ← / → or click"
    # hint that overlays the Good/Bad buttons when a fresh labeling session
    # has no votes yet. Persisting it keeps the hint from re-appearing every
    # time the same user starts a new session.
    label_hint_dismissed: bool = False
    autopilot_enabled: bool = True
    hide_autopilot: bool = False
    # When True, the Achievements tab/button and unlock pop-ups are
    # hidden, every ``record_*`` hook is a no-op, and ``get_full_state``
    # returns zeroed counters with no pending announcements. Flipping it
    # on also wipes any stored ``achievement_state`` so the counters
    # start at zero if the user ever turns the feature back on. See the
    # ``disable_achievements`` route handler in
    # ``vtsearch/routes/settings/api.py``.
    disable_achievements: bool = False
    autopilot_top_greens: Annotated[int, _clamp_min(1)] = 3
    autopilot_hard_reds: Annotated[int, _clamp_min(1)] = 4
    autopilot_resort_interval: Annotated[int, _clamp_min(1)] = 10
    autopilot_goal_diversity: Annotated[int, _clamp_min(1)] = 40

    view_mode_left: dict[str, ViewMode] = Field(default_factory=dict)
    view_mode_right: dict[str, ViewMode] = Field(default_factory=dict)
    grid_icon_size_left: dict[str, Annotated[GridIconSize, BeforeValidator(_upper)]] = Field(default_factory=dict)
    grid_icon_size_right: dict[str, Annotated[GridIconSize, BeforeValidator(_upper)]] = Field(default_factory=dict)
    focus_mode_left: dict[str, FocusMode] = Field(default_factory=dict)
    focus_mode_right: dict[str, FocusMode] = Field(default_factory=dict)
    panel_pct_left: dict[str, int] = Field(default_factory=dict)
    panel_pct_right: dict[str, int] = Field(default_factory=dict)

    # Per-media-type memory of the last embedder the user picked, used by the
    # dataset importer modal to pre-select a sensible default when no loaded
    # dataset is around to supply the same hint via ``guessedMediaEmbedder``.
    # Keys are canonical media-type ids (e.g. ``"image"``, ``"audio"``).
    last_embedder_per_media_type: dict[str, str] = Field(default_factory=dict)

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
    # through the settings UI — so once a user opts out (sets it to None),
    # the CLI flag no longer reapplies on future launches for that user.
    solo_media_type: str | None = None
    solo_media_type_explicit: bool = False

    # Rolling list of recent (dataset_id, detector_id, last_activity)
    # entries, capped at MAX_RECENT_SESSIONS by the route handler. Most
    # recent first. last_activity is epoch seconds (float). The
    # "Recent sessions" burger-menu submenu reads this and filters out
    # entries whose ids no longer resolve in the registries.
    recent_sessions: list[dict[str, Any]] = Field(default_factory=list)
