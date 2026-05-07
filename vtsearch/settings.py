"""Persistent settings for VTSearch.

Settings are stored as a JSON file in ``data/settings.json``.  The module
exposes simple get/set helpers and auto-saves on every mutation.

Schema (all keys optional, missing keys use defaults)::

    {
        "volume": 1.0,
        "inclusion": 0,
        "autorun_processors": [
            {
                "processor_name": "my detector",
                "processor_importer": "server_detector_file",
                "field_values": {"filepath": "/path/to/detector.json"}
            }
        ]
    }

Autorun processors store the *recipe* for importing a processor (the importer
name, field values, and desired detector name).  They are only materialised
into autorun detectors on demand — during autodetect.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from vtsearch.config import DATA_DIR

logger = logging.getLogger(__name__)

SETTINGS_PATH: Path = DATA_DIR / "settings.json"

_DEFAULTS: dict[str, Any] = {
    "volume": 1.0,
    "inclusion": 0,
    "theme": "dark",
    "enrich_descriptions": False,
    "safe_thresholds": False,
    "calibrate_count": 2,
    "calibration_fraction": 0.5,
    "audio_playing": True,
    "swipe_animation": True,
    "show_metadata": True,
    "view_mode_left": {},
    "view_mode_right": {},
    "grid_icon_size_left": {},
    "grid_icon_size_right": {},
    "focus_mode_left": {},
    "focus_mode_right": {},
    "panel_pct_left": {},
    "panel_pct_right": {},
    "autoload_media_embedders": [],
    "autorun_processors": [],
    "autorun_detector_names": [],
    "autorun_trainable_models": [],
    "autopilot_enabled": True,
    "hide_autopilot": False,
    "autopilot_top_greens": 3,
    "autopilot_hard_reds": 4,
    "autopilot_resort_interval": 10,
    "autopilot_goal_diversity": 40,
    "saved_datasets_dir": str(DATA_DIR / "saved_datasets"),
    "detectors_dir": str(DATA_DIR / "detectors"),
    "trainable_models_dir": str(DATA_DIR / "trainable_models"),
    "max_concurrent_dataset_downloads": 1,
    "max_concurrent_dataset_embeddings": 1,
}

#: Keys excluded from the "defaults" endpoint (infrastructure settings that
#: should not be reset by the Default button).
_EXCLUDE_FROM_DEFAULTS = {
    "autorun_processors",
    "autorun_detector_names",
    "autorun_trainable_models",
    "saved_datasets_dir",
    "detectors_dir",
    "trainable_models_dir",
    "settings_source",
}

#: Keys excluded from source sync export (to avoid circular config).
_EXCLUDE_FROM_SOURCE_EXPORT = {
    "settings_source",
}

# In-memory cache — loaded once, written on every mutation.
_settings: dict[str, Any] | None = None

# Reentrant lock protecting the in-memory cache.  RLock is used because
# some public functions (e.g. toggle_autoload_media_embedder) call other
# public functions that also acquire the lock.
_settings_lock = threading.RLock()

# Module-level (NOT thread-local) guard that prevents re-exporting to the
# source during an import-from-source pass.  A thread-local flag would not
# block a concurrent ``set_*()`` call running on a different thread from
# triggering a ``_sync_to_source(...)`` while we are mid-import — racing
# with and potentially overwriting the data we just imported.  Always
# read/written while holding ``_settings_lock``.
_syncing: bool = False


def _load() -> dict[str, Any]:
    """Read settings from disk, returning defaults on any failure."""
    if SETTINGS_PATH.exists():
        try:
            text = SETTINGS_PATH.read_text(encoding="utf-8")
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            logger.warning("Failed to read settings file: %s", exc)
    return {}


def _save(data: dict[str, Any]) -> None:
    """Write *data* to the settings file (creating parent dirs if needed).

    Uses atomic write (write to temp file, then rename) to prevent
    data loss if the process crashes or is killed mid-write.  When an
    active settings source is configured and we are not already syncing,
    the source is also updated.
    """
    import os

    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, SETTINGS_PATH)

    # Auto-export to active settings source (skip during import-from-source).
    # ``_syncing`` is process-wide, so a parallel set_*() on another thread
    # correctly skips the export while sync_from_settings_source() is running.
    if not _syncing:
        _sync_to_source(data)


def _ensure_loaded() -> dict[str, Any]:
    global _settings
    with _settings_lock:
        if _settings is None:
            _settings = _load()
        return _settings


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------


def get_defaults() -> dict[str, Any]:
    """Return a copy of the default settings (excluding infrastructure keys)."""
    result = {k: v for k, v in _DEFAULTS.items() if k not in _EXCLUDE_FROM_DEFAULTS}
    # Expand view mode defaults to per-media-type dicts
    valid_types = _valid_media_types()
    result["view_mode_left"] = {tid: _VIEW_MODE_DEFAULTS["left"] for tid in valid_types}
    result["view_mode_right"] = {tid: _VIEW_MODE_DEFAULTS["right"] for tid in valid_types}
    result["grid_icon_size_left"] = {tid: _GRID_ICON_SIZE_DEFAULT for tid in valid_types}
    result["grid_icon_size_right"] = {tid: _GRID_ICON_SIZE_DEFAULT for tid in valid_types}
    # Expand focus mode defaults to per-media-type dicts
    result["focus_mode_left"] = {tid: _FOCUS_MODE_DEFAULTS["left"] for tid in valid_types}
    result["focus_mode_right"] = {tid: _FOCUS_MODE_DEFAULTS["right"] for tid in valid_types}
    # Expand panel percentage defaults to per-media-type dicts
    result["panel_pct_left"] = {tid: _PANEL_PX_DEFAULTS["left"] for tid in valid_types}
    result["panel_pct_right"] = {tid: _PANEL_PX_DEFAULTS["right"] for tid in valid_types}
    return result


def get_all() -> dict[str, Any]:
    """Return the full settings dict (with defaults filled in)."""
    with _settings_lock:
        s = _ensure_loaded()
        result = dict(_DEFAULTS)
        result.update(s)
        # Always return expanded per-media-type view mode dicts
        result["view_mode_left"] = get_view_mode_left()
        result["view_mode_right"] = get_view_mode_right()
        result["grid_icon_size_left"] = get_grid_icon_size_left()
        result["grid_icon_size_right"] = get_grid_icon_size_right()
        # Always return expanded per-media-type focus mode dicts
        result["focus_mode_left"] = get_focus_mode_left()
        result["focus_mode_right"] = get_focus_mode_right()
        # Always return expanded per-media-type panel percentage dicts
        result["panel_pct_left"] = get_panel_pct_left()
        result["panel_pct_right"] = get_panel_pct_right()
        return result


VALID_THEMES = ("dark", "light", "highviz")
VALID_VIEW_MODES = ("grid", "list")
VALID_GRID_ICON_SIZES = ("XS", "S", "M", "L", "XL")
VALID_FOCUS_MODES = ("click", "hover")


# -------------------------------------------------------------------
# Spec-driven generation of simple get_<key> / set_<key> pairs.
# Factories live in :mod:`vtsearch.settings_factory`.
# -------------------------------------------------------------------

from vtsearch.settings_factory import (  # noqa: E402
    clamp as _clamp,
    clamp_min as _clamp_min,
    make_accessors as _make_accessors,
    make_per_side_setting as _make_per_side_setting_impl,
    one_of as _one_of,
)


# (key, cast, coerce_or_None)
_SETTING_SPECS: list[tuple] = [
    ("volume", float, _clamp(float, 0.0, 1.0)),
    ("inclusion", int, _clamp(int, -10, 10)),
    ("theme", str, _one_of("theme", VALID_THEMES)),
    ("enrich_descriptions", bool, None),
    ("safe_thresholds", bool, None),
    ("calibrate_count", int, _clamp(int, 1, 100)),
    ("calibration_fraction", float, _clamp(float, 0.0, 1.0)),
    ("audio_playing", bool, None),
    ("swipe_animation", bool, None),
    ("show_metadata", bool, None),
    ("autopilot_enabled", bool, None),
    ("hide_autopilot", bool, None),
    ("autopilot_top_greens", int, _clamp_min(int, 1)),
    ("autopilot_hard_reds", int, _clamp_min(int, 1)),
    ("autopilot_resort_interval", int, _clamp_min(int, 1)),
    ("autopilot_goal_diversity", int, _clamp_min(int, 1)),
    ("max_concurrent_dataset_downloads", int, _clamp(int, 1, 16)),
    ("max_concurrent_dataset_embeddings", int, _clamp(int, 1, 16)),
]

for _key, _cast, _coerce in _SETTING_SPECS:
    _g, _s = _make_accessors(_key, _cast, _coerce)
    globals()[f"get_{_key}"] = _g
    globals()[f"set_{_key}"] = _s

del _key, _cast, _coerce, _g, _s


# -------------------------------------------------------------------
# Per-media-type per-side settings factory
# -------------------------------------------------------------------

_VIEW_MODE_DEFAULTS = {"left": "list", "right": "grid"}
_GRID_ICON_SIZE_DEFAULT = "M"
_FOCUS_MODE_DEFAULTS = {"left": "click", "right": "click"}
_PANEL_PX_DEFAULTS: dict[str, int] = {"left": 260, "right": 300}
VALID_PANEL_PX = (150, 500)  # pixel range matching frontend LEFT/RIGHT_MIN/MAX


def _valid_media_types() -> tuple[str, ...]:
    """Return valid media type IDs from the media registry."""
    from vtsearch.media import all_type_ids

    return tuple(all_type_ids())


def _make_per_side_setting(
    key_base: str,
    defaults: dict[str, Any],
    valid_values: tuple[str, ...] | None = None,
    *,
    normalize=None,
    value_type: str = "str",
):
    """Thin wrapper around :func:`settings_factory.make_per_side_setting`
    that injects this module's ``VALID_PANEL_PX`` constant."""
    return _make_per_side_setting_impl(
        key_base,
        defaults,
        valid_values,
        valid_panel_px=VALID_PANEL_PX,
        normalize=normalize,
        value_type=value_type,
    )


# Generate all four per-side settings

get_view_mode_left, get_view_mode_right, set_view_mode_left, set_view_mode_right = _make_per_side_setting(
    "view_mode",
    _VIEW_MODE_DEFAULTS,
    VALID_VIEW_MODES,
)

get_grid_icon_size_left, get_grid_icon_size_right, set_grid_icon_size_left, set_grid_icon_size_right = (
    _make_per_side_setting(
        "grid_icon_size",
        {"left": _GRID_ICON_SIZE_DEFAULT, "right": _GRID_ICON_SIZE_DEFAULT},
        VALID_GRID_ICON_SIZES,
        normalize=str.upper,
    )
)

get_focus_mode_left, get_focus_mode_right, set_focus_mode_left, set_focus_mode_right = _make_per_side_setting(
    "focus_mode",
    _FOCUS_MODE_DEFAULTS,
    VALID_FOCUS_MODES,
)

get_panel_pct_left, get_panel_pct_right, set_panel_pct_left, set_panel_pct_right = _make_per_side_setting(
    "panel_pct",
    _PANEL_PX_DEFAULTS,
    None,
    value_type="int",
)


def _valid_embedder_names() -> tuple[str, ...]:
    """Return the names of all registered embedders (lazy import to avoid circular deps)."""
    from vtsearch.media import all_embedders

    return tuple(e.name for e in all_embedders())


def get_autoload_media_embedders() -> list[str]:
    """Return the list of autoload embedder names (empty list if none set)."""
    with _settings_lock:
        raw = _ensure_loaded().get("autoload_media_embedders", _DEFAULTS["autoload_media_embedders"])
        if isinstance(raw, list):
            valid = _valid_embedder_names()
            return [v for v in raw if v in valid]
        return []


def set_autoload_media_embedders(value: list[str]) -> None:
    """Set and persist the full list of autoload embedder names."""
    valid = _valid_embedder_names()
    for v in value:
        if v not in valid:
            raise ValueError(f"Invalid embedder: {v!r}")
    with _settings_lock:
        s = _ensure_loaded()
        s["autoload_media_embedders"] = list(dict.fromkeys(value))
        _save(s)


def toggle_autoload_media_embedder(embedder_name: str) -> list[str]:
    """Toggle a single embedder's autoload status.  Returns the updated list."""
    valid = _valid_embedder_names()
    if embedder_name not in valid:
        raise ValueError(f"Invalid embedder: {embedder_name!r}")
    with _settings_lock:
        current = get_autoload_media_embedders()
        if embedder_name in current:
            current.remove(embedder_name)
        else:
            current.append(embedder_name)
        set_autoload_media_embedders(current)
        return current


def get_autorun_processors() -> list[dict[str, Any]]:
    """Return the list of autorun processor recipes."""
    with _settings_lock:
        return list(_ensure_loaded().get("autorun_processors", []))


def add_autorun_processor(
    processor_name: str,
    processor_importer: str,
    field_values: dict[str, Any],
) -> None:
    """Add a autorun processor recipe (or overwrite one with the same name)."""
    with _settings_lock:
        s = _ensure_loaded()
        procs: list[dict[str, Any]] = s.setdefault("autorun_processors", [])
        # Remove existing entry with same name
        procs[:] = [p for p in procs if p.get("processor_name") != processor_name]
        procs.append(
            {
                "processor_name": processor_name,
                "processor_importer": processor_importer,
                "field_values": field_values,
            }
        )
        _save(s)


def remove_autorun_processor(processor_name: str) -> bool:
    """Remove a autorun processor by name.  Returns True if found."""
    with _settings_lock:
        s = _ensure_loaded()
        procs: list[dict[str, Any]] = s.get("autorun_processors", [])
        before = len(procs)
        procs[:] = [p for p in procs if p.get("processor_name") != processor_name]
        if len(procs) < before:
            s["autorun_processors"] = procs
            _save(s)
            return True
        return False


def to_settings_json(entry: dict[str, Any]) -> str:
    """Build the JSON snippet for a autorun processor entry.

    Returns the JSON object that would appear inside the
    ``autorun_processors`` array in a settings file.  Useful for showing
    users how to recreate this processor configuration.

    Example output::

        {"processor_name": "my detector", "processor_importer": "server_detector_file",
         "field_values": {"filepath": "detector.json"}}
    """
    import json

    snippet = {
        "processor_name": entry["processor_name"],
        "processor_importer": entry["processor_importer"],
        "field_values": entry.get("field_values", {}),
    }
    return json.dumps(snippet)


def get_autorun_detector_names() -> list[str]:
    """Return the list of detector names flagged for autorun."""
    with _settings_lock:
        raw = _ensure_loaded().get("autorun_detector_names", [])
        if isinstance(raw, list):
            return list(raw)
        return []


def set_autorun_detector_names(value: list[str]) -> None:
    """Set and persist the full list of autorun detector names."""
    with _settings_lock:
        s = _ensure_loaded()
        s["autorun_detector_names"] = list(dict.fromkeys(value))  # deduplicate, preserve order
        _save(s)


def add_autorun_detector_name(name: str) -> None:
    """Add a detector name to the autorun list (idempotent)."""
    with _settings_lock:
        current = get_autorun_detector_names()
        if name not in current:
            current.append(name)
            set_autorun_detector_names(current)


def remove_autorun_detector_name(name: str) -> bool:
    """Remove a detector name from the autorun list. Returns True if found."""
    with _settings_lock:
        current = get_autorun_detector_names()
        if name in current:
            current.remove(name)
            set_autorun_detector_names(current)
            return True
        return False


def get_autorun_trainable_models() -> list[str]:
    """Return the list of trainable-model names flagged for CLI autodetect.

    These are scored alongside :func:`get_autorun_detector_names` when the
    CLI runs ``--autodetect``.  Each name maps to a JSON file under
    ``data/trainable_models/``; scoring resolves the labelset's origins,
    re-embeds, trains an MLP, and applies it to the loaded dataset.
    """
    with _settings_lock:
        raw = _ensure_loaded().get("autorun_trainable_models", [])
        if isinstance(raw, list):
            return list(raw)
        return []


def set_autorun_trainable_models(value: list[str]) -> None:
    """Set and persist the full list of autorun trainable-model names."""
    with _settings_lock:
        s = _ensure_loaded()
        s["autorun_trainable_models"] = list(dict.fromkeys(value))  # dedupe, preserve order
        _save(s)


def is_autorun_detector(name: str) -> bool:
    """Check whether a detector name is in the autorun list."""
    return name in get_autorun_detector_names()


def ensure_autorun_processors_imported() -> list[str]:
    """Import any autorun processors that are not already loaded as autorun detectors.

    This is the lazy-load mechanism: autorun processor recipes are materialised
    into real autorun detectors only when this function is called (typically
    right before autodetect).

    Returns:
        A list of processor names that were newly imported.
    """
    from vtsearch.processors.importers import get_processor_importer
    from vtsearch.utils import add_autorun_detector, get_autorun_detectors

    existing = get_autorun_detectors()
    imported: list[str] = []

    for entry in get_autorun_processors():
        name = entry.get("processor_name", "")
        if not name or name in existing:
            continue

        importer_name = entry.get("processor_importer", "")
        importer = get_processor_importer(importer_name)
        if importer is None:
            logger.warning("Autorun processor '%s': unknown importer '%s'", name, importer_name)
            continue

        field_values = dict(entry.get("field_values", {}))

        try:
            importer.validate_cli_field_values(field_values)
            result = importer.run_cli(field_values)

            if not isinstance(result, dict) or not result.get("weights"):
                logger.warning("Autorun processor '%s': importer returned invalid result", name)
                continue

            add_autorun_detector(
                name,
                result.get("media_type", "audio"),
                result["weights"],
                result.get("threshold", 0.5),
                autodetect=True,
                good_origins=result.get("good_origins"),
                bad_origins=result.get("bad_origins"),
                inclusion=result.get("inclusion", 0),
            )
            imported.append(name)
        except Exception as exc:
            logger.warning("Autorun processor '%s': import failed: %s", name, exc)

    return imported


# -------------------------------------------------------------------
# Directory path settings
# -------------------------------------------------------------------


def _get_dir(key: str) -> Path:
    """Return a directory path setting as a :class:`~pathlib.Path`."""
    with _settings_lock:
        raw = _ensure_loaded().get(key, _DEFAULTS[key])
    return Path(raw)


def _set_dir(key: str, value: str | Path) -> None:
    """Persist a directory path setting."""
    with _settings_lock:
        s = _ensure_loaded()
        s[key] = str(value)
        _save(s)


def get_saved_datasets_dir() -> Path:
    """Return the configured saved-datasets directory."""
    return _get_dir("saved_datasets_dir")


def set_saved_datasets_dir(value: str | Path) -> None:
    """Set the saved-datasets directory."""
    _set_dir("saved_datasets_dir", value)


def get_detectors_dir() -> Path:
    """Return the configured detectors directory."""
    return _get_dir("detectors_dir")


def set_detectors_dir(value: str | Path) -> None:
    """Set the detectors directory."""
    _set_dir("detectors_dir", value)


def get_trainable_models_dir() -> Path:
    """Return the configured trainable-models directory."""
    return _get_dir("trainable_models_dir")


def set_trainable_models_dir(value: str | Path) -> None:
    """Set the trainable-models directory."""
    _set_dir("trainable_models_dir", value)


def set_settings_path(path: str | Path) -> None:
    """Override the settings file path and reset the in-memory cache.

    Call this before :func:`ensure_autorun_processors_imported` to load
    autorun processors from a custom settings file (e.g. the ``--settings``
    CLI flag).
    """
    global SETTINGS_PATH, _settings
    with _settings_lock:
        SETTINGS_PATH = Path(path)
        _settings = None  # force reload on next access


def reset() -> None:
    """Reset the in-memory cache (for testing)."""
    global _settings
    with _settings_lock:
        _settings = None


# -------------------------------------------------------------------
# Settings source (bidirectional sync)
# -------------------------------------------------------------------


def get_settings_source_config() -> dict[str, Any] | None:
    """Return the active settings source config, or ``None`` if unset.

    Config shape::

        {
            "source_name": "server_json_file",
            "field_values": {"filepath": "data/{username}.settings.json"}
        }
    """
    with _settings_lock:
        cfg = _ensure_loaded().get("settings_source")
    if isinstance(cfg, dict) and cfg.get("source_name"):
        return cfg
    return None


def set_settings_source_config(config: dict[str, Any] | None) -> None:
    """Set or clear the active settings source config."""
    with _settings_lock:
        s = _ensure_loaded()
        if config is None:
            s.pop("settings_source", None)
        else:
            s["settings_source"] = config
        _save(s)


# Map of setting key → setter function (generated dynamically).
_SETTER_MAP: dict[str, Any] | None = None


def _get_setter_map() -> dict:
    """Build a map of setting-key → setter-function by introspecting this module."""
    global _SETTER_MAP
    if _SETTER_MAP is not None:
        return _SETTER_MAP
    import vtsearch.settings as _self

    _SETTER_MAP = {}
    for attr_name in dir(_self):
        if attr_name.startswith("set_") and callable(getattr(_self, attr_name)):
            _SETTER_MAP[attr_name[4:]] = getattr(_self, attr_name)
    return _SETTER_MAP


def _apply_settings(imported: dict) -> None:
    """Apply a dict of settings via this module's ``set_*`` functions.

    Unknown keys or values that fail validation are silently skipped.
    Used by :func:`sync_from_settings_source` and by the settings-import
    route in ``routes/settings_io.py``.
    """
    setter_map = _get_setter_map()
    for key, value in imported.items():
        setter = setter_map.get(key)
        if setter is not None:
            try:
                setter(value)
            except (TypeError, ValueError):
                pass  # Skip invalid values silently


def sync_from_settings_source() -> dict[str, Any] | None:
    """Pull settings from the active source and apply them.

    Returns the imported settings dict, or ``None`` if no source is
    configured or the source file doesn't exist yet.

    This is called:
    - At app startup (auto-import), so settings from the source take
      precedence over the local settings file.
    - Manually via ``POST /api/settings-sources/sync``.
    """
    cfg = get_settings_source_config()
    if cfg is None:
        return None

    from vtsearch.settings_io.sources import get_settings_source

    source = get_settings_source(cfg["source_name"])
    if source is None:
        logger.warning("Unknown settings source: %s", cfg["source_name"])
        return None

    field_values = cfg.get("field_values", {})
    try:
        imported = source.load(field_values)
    except Exception as exc:
        logger.exception("Failed to load from settings source: %s", exc)
        return None

    if not imported:
        return None

    # Hold _settings_lock for the entire apply pass so that:
    #   * concurrent set_*() calls on other threads block until we finish,
    #     instead of racing and re-exporting partially-imported state
    #   * the _syncing flag (also guarded by _settings_lock) is observed
    #     consistently by every _save() that runs during the import
    global _syncing
    with _settings_lock:
        _syncing = True
        try:
            _apply_settings(imported)
        finally:
            _syncing = False

    return imported


def _sync_to_source(data: dict[str, Any]) -> None:
    """Push current settings to the active source (if any).

    Called from :func:`_save` after the local file is written.  Strips
    the ``settings_source`` key itself to avoid circular config.
    """
    cfg = data.get("settings_source")
    if not isinstance(cfg, dict) or not cfg.get("source_name"):
        return

    from vtsearch.settings_io.sources import get_settings_source

    source = get_settings_source(cfg["source_name"])
    if source is None:
        return

    field_values = cfg.get("field_values", {})
    export_data = {k: v for k, v in data.items() if k not in _EXCLUDE_FROM_SOURCE_EXPORT}

    try:
        source.save(export_data, field_values)
    except Exception as exc:
        logger.exception("Failed to sync settings to source: %s", exc)
