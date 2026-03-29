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
    "autopilot_enabled": True,
    "hide_autopilot": False,
    "autopilot_top_greens": 3,
    "autopilot_hard_reds": 4,
    "autopilot_resort_interval": 10,
    "autopilot_goal_diversity": 40,
    "saved_datasets_dir": str(DATA_DIR / "saved_datasets"),
    "detectors_dir": str(DATA_DIR / "detectors"),
    "trainable_models_dir": str(DATA_DIR / "trainable_models"),
}

#: Keys excluded from the "defaults" endpoint (infrastructure settings that
#: should not be reset by the Default button).
_EXCLUDE_FROM_DEFAULTS = {
    "autorun_processors",
    "autorun_detector_names",
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

# Thread-local guard to prevent re-exporting to the source during an
# import-from-source pass.
_syncing = threading.local()


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
    if not getattr(_syncing, "active", False):
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
# Factory for simple get_<key> / set_<key> pairs
# -------------------------------------------------------------------


def _make_accessors(key: str, cast: type, coerce=None):
    """Create a ``get_<key>`` / ``set_<key>`` pair for a simple setting.

    *cast* converts the stored value on read (e.g. ``float``, ``int``, ``bool``).
    *coerce* normalises the value on write (e.g. clamping); defaults to *cast*.
    """
    if coerce is None:
        coerce = cast

    def getter():
        with _settings_lock:
            return cast(_ensure_loaded().get(key, _DEFAULTS[key]))

    def setter(value):
        with _settings_lock:
            s = _ensure_loaded()
            s[key] = coerce(value)
            _save(s)

    getter.__name__ = f"get_{key}"
    setter.__name__ = f"set_{key}"
    return getter, setter


def _clamp(cast, lo, hi):
    """Return a coercion that casts then clamps to ``[lo, hi]``."""
    return lambda v: cast(max(lo, min(hi, cast(v))))


def _clamp_min(cast, lo):
    """Return a coercion that casts then clamps to ``>= lo``."""
    return lambda v: cast(max(lo, cast(v)))


def _one_of(key, valid):
    """Return a coercion that validates membership in *valid*."""

    def _coerce(v):
        v = str(v)
        if v not in valid:
            raise ValueError(f"Invalid {key}: {v!r}")
        return v

    return _coerce


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
    """Factory for per-media-type per-side settings.

    Generates ``get_<key_base>_left()``, ``get_<key_base>_right()``,
    ``set_<key_base>_left()``, ``set_<key_base>_right()``.

    Parameters
    ----------
    key_base:
        Setting name without the side suffix, e.g. ``"view_mode"``.
    defaults:
        ``{"left": default_value, "right": default_value}``.
    valid_values:
        Tuple of allowed string values (for enum-like settings).
        ``None`` skips membership validation (for numeric settings).
    normalize:
        Optional ``str -> str`` applied to string values on read and
        write (e.g. ``str.upper`` for grid_icon_size).
    value_type:
        ``"str"`` or ``"int"`` — controls validation and coercion.
    """
    lo_hi = VALID_PANEL_PX if value_type == "int" else None

    def _get_dict(key: str) -> dict[str, Any]:
        side = key[len(key_base) + 1:]  # strip "<key_base>_" prefix
        default_val = defaults.get(side, next(iter(defaults.values())))
        with _settings_lock:
            raw = _ensure_loaded().get(key, _DEFAULTS[key])

        types = _valid_media_types()

        if not isinstance(raw, dict):
            return {tid: default_val for tid in types}

        result: dict[str, Any] = {}
        for tid in types:
            v = raw.get(tid, default_val)
            if normalize is not None and isinstance(v, str):
                v = normalize(v)
            if valid_values is not None and v not in valid_values:
                v = default_val
            elif value_type == "int":
                lo, hi = lo_hi  # type: ignore[misc]
                try:
                    v = max(lo, min(hi, int(round(float(v)))))
                except (ValueError, TypeError):
                    v = default_val
            result[tid] = v
        return result

    def _validate_str(v, key, tid=None):
        if normalize is not None and isinstance(v, str):
            v = normalize(v)
        if valid_values is not None and v not in valid_values:
            label = f"{key} value for {tid}" if tid else key
            raise ValueError(f"Invalid {label}: {v!r}")
        return v

    def _validate_int(v, key, tid=None):
        lo, hi = lo_hi  # type: ignore[misc]
        try:
            v = int(round(float(v)))
        except (ValueError, TypeError):
            label = f"{key} value for {tid}" if tid else key
            raise ValueError(f"Invalid {label}: {v!r}")
        if not (lo <= v <= hi):
            label = f"{key} value for {tid}" if tid else key
            raise ValueError(f"Invalid {label}: {v} (must be between {lo} and {hi})")
        return v

    _validate_entry = _validate_int if value_type == "int" else _validate_str

    def _set_dict(key: str, value) -> None:
        valid_types = _valid_media_types()

        # Scalar expansion: "grid" → {"audio": "grid", "image": "grid", ...}
        if value_type == "str" and isinstance(value, str):
            value = {tid: _validate_entry(value, key) for tid in valid_types}
        elif value_type == "int" and isinstance(value, (int, float)):
            value = {tid: _validate_entry(value, key) for tid in valid_types}

        if not isinstance(value, dict):
            expected = "dict or string" if value_type == "str" else "dict or number"
            raise ValueError(f"{key} must be a {expected}")

        coerced: dict[str, Any] = {}
        for tid, v in value.items():
            if tid not in valid_types:
                raise ValueError(f"Invalid media type: {tid!r}")
            coerced[tid] = _validate_entry(v, key, tid)

        with _settings_lock:
            s = _ensure_loaded()
            s[key] = coerced
            _save(s)

    def get_left():
        return _get_dict(f"{key_base}_left")

    def get_right():
        return _get_dict(f"{key_base}_right")

    def set_left(value):
        _set_dict(f"{key_base}_left", value)

    def set_right(value):
        _set_dict(f"{key_base}_right", value)

    get_left.__name__ = f"get_{key_base}_left"
    get_right.__name__ = f"get_{key_base}_right"
    set_left.__name__ = f"set_{key_base}_left"
    set_right.__name__ = f"set_{key_base}_right"
    return get_left, get_right, set_left, set_right


# Generate all four per-side settings

get_view_mode_left, get_view_mode_right, set_view_mode_left, set_view_mode_right = _make_per_side_setting(
    "view_mode", _VIEW_MODE_DEFAULTS, VALID_VIEW_MODES,
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
    "focus_mode", _FOCUS_MODE_DEFAULTS, VALID_FOCUS_MODES,
)

get_panel_pct_left, get_panel_pct_right, set_panel_pct_left, set_panel_pct_right = _make_per_side_setting(
    "panel_pct", _PANEL_PX_DEFAULTS, None, value_type="int",
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


def sync_from_settings_source() -> dict[str, Any] | None:
    """Pull settings from the active source and apply them.

    Returns the imported settings dict, or ``None`` if no source is
    configured or the source file doesn't exist yet.
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
        logger.warning("Failed to load from settings source: %s", exc)
        return None

    if not imported:
        return None

    # Apply under sync guard to prevent re-exporting back to source.
    _syncing.active = True
    try:
        from vtsearch.routes.settings_io import _apply_settings

        _apply_settings(imported)
    finally:
        _syncing.active = False

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
        logger.warning("Failed to sync settings to source: %s", exc)
