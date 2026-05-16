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

from vtsearch.config import DATA_DIR, DEFAULT_CALIBRATE_COUNT

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


Theme = Literal["dark", "light", "highviz"]
ViewMode = Literal["grid", "list"]
GridIconSize = Literal["XS", "S", "M", "L", "XL"]
FocusMode = Literal["click", "hover"]

VALID_THEMES: tuple[str, ...] = ("dark", "light", "highviz")
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

    Imported lazily to avoid pulling :mod:`vtsearch.embedding.loader` (and
    transitively torch) at settings-model import time.
    """
    from vtsearch.embedding.loader import default_concurrent_downloads

    return default_concurrent_downloads()


def _default_concurrent_embeddings() -> int:
    """Lazily resolve the hardware-derived default for parallel embeddings."""
    from vtsearch.embedding.loader import default_concurrent_embeddings

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
    theme: Theme = "dark"
    enrich_descriptions: bool = False
    safe_thresholds: bool = False
    calibrate_count: Annotated[int, _clamp(1, 100)] = DEFAULT_CALIBRATE_COUNT
    calibration_fraction: Annotated[float, _clamp(0.0, 1.0)] = 0.5
    audio_playing: bool = True
    swipe_animation: bool = True
    show_metadata: bool = True
    autopilot_enabled: bool = True
    hide_autopilot: bool = False
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
