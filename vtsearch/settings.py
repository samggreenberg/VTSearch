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
    "swipe_animation": True,
    "show_metadata": True,
    "view_mode_left": {},
    "view_mode_right": {},
    "grid_item_size_left": {},
    "grid_item_size_right": {},
    "focus_mode_left": "click",
    "focus_mode_right": "click",
    "autoload_media_types": [],
    "autoload_media_embedders": [],
    "autorun_processors": [],
    "hide_autopilot": False,
    "autopilot_top_greens": 3,
    "autopilot_hard_reds": 4,
    "saved_datasets_dir": str(DATA_DIR / "saved_datasets"),
    "detectors_dir": str(DATA_DIR / "detectors"),
    "trainable_models_dir": str(DATA_DIR / "trainable_models"),
}

#: Keys excluded from the "defaults" endpoint (infrastructure settings that
#: should not be reset by the Default button).
_EXCLUDE_FROM_DEFAULTS = {"autorun_processors", "saved_datasets_dir", "detectors_dir", "trainable_models_dir"}

# In-memory cache — loaded once, written on every mutation.
_settings: dict[str, Any] | None = None

# Reentrant lock protecting the in-memory cache.  RLock is used because
# some public functions (e.g. toggle_autoload_media_type) call other public
# functions that also acquire the lock.
_settings_lock = threading.RLock()


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
    data loss if the process crashes or is killed mid-write.
    """
    import os

    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, SETTINGS_PATH)


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
    result["grid_item_size_left"] = {tid: _GRID_ITEM_SIZE_DEFAULT for tid in valid_types}
    result["grid_item_size_right"] = {tid: _GRID_ITEM_SIZE_DEFAULT for tid in valid_types}
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
        result["grid_item_size_left"] = get_grid_item_size_left()
        result["grid_item_size_right"] = get_grid_item_size_right()
        return result


VALID_THEMES = ("dark", "light", "highviz")
VALID_VIEW_MODES = ("grid", "list")
VALID_GRID_ITEM_SIZES = ("small", "medium", "large")
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
    ("swipe_animation", bool, None),
    ("show_metadata", bool, None),
    ("focus_mode_left", str, _one_of("focus_mode_left", VALID_FOCUS_MODES)),
    ("focus_mode_right", str, _one_of("focus_mode_right", VALID_FOCUS_MODES)),
    ("hide_autopilot", bool, None),
    ("autopilot_top_greens", int, _clamp_min(int, 1)),
    ("autopilot_hard_reds", int, _clamp_min(int, 1)),
]

for _key, _cast, _coerce in _SETTING_SPECS:
    _g, _s = _make_accessors(_key, _cast, _coerce)
    globals()[f"get_{_key}"] = _g
    globals()[f"set_{_key}"] = _s

del _key, _cast, _coerce, _g, _s


# -------------------------------------------------------------------
# Per-media-type view mode settings
# -------------------------------------------------------------------

_VIEW_MODE_DEFAULTS = {"left": "list", "right": "grid"}
_GRID_ITEM_SIZE_DEFAULT = "medium"


def _get_view_mode_dict(key: str) -> dict[str, str]:
    """Return the per-media-type view mode dict for *key* (``view_mode_left`` or ``view_mode_right``).

    Handles backward compatibility: if the stored value is a plain string
    (old format), it is treated as the mode for all known media types.
    """
    side = key.replace("view_mode_", "")
    default_mode = _VIEW_MODE_DEFAULTS.get(side, "list")
    with _settings_lock:
        raw = _ensure_loaded().get(key, _DEFAULTS[key])
    if isinstance(raw, str):
        # Legacy scalar value — expand to dict for all known types
        return {tid: raw for tid in _valid_media_types()}
    if isinstance(raw, dict):
        # Fill in missing types with the side default
        result = {}
        for tid in _valid_media_types():
            val = raw.get(tid, default_mode)
            result[tid] = val if val in VALID_VIEW_MODES else default_mode
        return result
    return {tid: default_mode for tid in _valid_media_types()}


def get_view_mode_left() -> dict[str, str]:
    """Return a dict mapping media type ID -> view mode for the left panel."""
    return _get_view_mode_dict("view_mode_left")


def get_view_mode_right() -> dict[str, str]:
    """Return a dict mapping media type ID -> view mode for the right panel."""
    return _get_view_mode_dict("view_mode_right")


def set_view_mode_left(value: dict[str, str] | str) -> None:
    """Set the left panel view mode (per-media-type dict or legacy scalar)."""
    _set_view_mode_dict("view_mode_left", value)


def set_view_mode_right(value: dict[str, str] | str) -> None:
    """Set the right panel view mode (per-media-type dict or legacy scalar)."""
    _set_view_mode_dict("view_mode_right", value)


def _set_view_mode_dict(key: str, value: dict[str, str] | str) -> None:
    """Persist the per-media-type view mode dict for *key*."""
    valid_types = _valid_media_types()
    if isinstance(value, str):
        # Legacy scalar — expand to all types
        if value not in VALID_VIEW_MODES:
            raise ValueError(f"Invalid {key}: {value!r}")
        value = {tid: value for tid in valid_types}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a dict or string")
    for tid, mode in value.items():
        if tid not in valid_types:
            raise ValueError(f"Invalid media type: {tid!r}")
        if mode not in VALID_VIEW_MODES:
            raise ValueError(f"Invalid {key} value for {tid}: {mode!r}")
    with _settings_lock:
        s = _ensure_loaded()
        s[key] = dict(value)
        _save(s)


def _get_grid_item_size_dict(key: str) -> dict[str, str]:
    """Return the per-media-type grid item size dict for *key*.

    Handles backward compatibility: if the stored value is a plain string,
    it is treated as the size for all known media types.
    """
    with _settings_lock:
        raw = _ensure_loaded().get(key, _DEFAULTS[key])
    if isinstance(raw, str):
        return {tid: raw for tid in _valid_media_types()}
    if isinstance(raw, dict):
        result = {}
        for tid in _valid_media_types():
            val = raw.get(tid, _GRID_ITEM_SIZE_DEFAULT)
            result[tid] = val if val in VALID_GRID_ITEM_SIZES else _GRID_ITEM_SIZE_DEFAULT
        return result
    return {tid: _GRID_ITEM_SIZE_DEFAULT for tid in _valid_media_types()}


def get_grid_item_size_left() -> dict[str, str]:
    """Return a dict mapping media type ID -> grid item size for the left panel."""
    return _get_grid_item_size_dict("grid_item_size_left")


def get_grid_item_size_right() -> dict[str, str]:
    """Return a dict mapping media type ID -> grid item size for the right panel."""
    return _get_grid_item_size_dict("grid_item_size_right")


def set_grid_item_size_left(value: dict[str, str] | str) -> None:
    """Set the left panel grid item size (per-media-type dict or scalar)."""
    _set_grid_item_size_dict("grid_item_size_left", value)


def set_grid_item_size_right(value: dict[str, str] | str) -> None:
    """Set the right panel grid item size (per-media-type dict or scalar)."""
    _set_grid_item_size_dict("grid_item_size_right", value)


def _set_grid_item_size_dict(key: str, value: dict[str, str] | str) -> None:
    """Persist the per-media-type grid item size dict for *key*."""
    valid_types = _valid_media_types()
    if isinstance(value, str):
        if value not in VALID_GRID_ITEM_SIZES:
            raise ValueError(f"Invalid {key}: {value!r}")
        value = {tid: value for tid in valid_types}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a dict or string")
    for tid, size in value.items():
        if tid not in valid_types:
            raise ValueError(f"Invalid media type: {tid!r}")
        if size not in VALID_GRID_ITEM_SIZES:
            raise ValueError(f"Invalid {key} value for {tid}: {size!r}")
    with _settings_lock:
        s = _ensure_loaded()
        s[key] = dict(value)
        _save(s)


def _valid_media_types() -> tuple[str, ...]:
    """Return valid media type IDs from the media registry."""
    from vtsearch.media import all_type_ids

    return tuple(all_type_ids())


def get_autoload_media_types() -> list[str]:
    """Return the list of autoload media type IDs (empty list if none set).

    .. deprecated:: Use :func:`get_autoload_media_embedders` instead.
    """
    valid = _valid_media_types()
    with _settings_lock:
        raw = _ensure_loaded().get("autoload_media_types", _DEFAULTS["autoload_media_types"])
        if isinstance(raw, list):
            return [v for v in raw if v in valid]
        return []


def set_autoload_media_types(value: list[str]) -> None:
    """Set and persist the full list of autoload media types.

    .. deprecated:: Use :func:`set_autoload_media_embedders` instead.
    """
    valid = _valid_media_types()
    for v in value:
        if v not in valid:
            raise ValueError(f"Invalid media type: {v!r}")
    with _settings_lock:
        s = _ensure_loaded()
        s["autoload_media_types"] = list(dict.fromkeys(value))  # deduplicate, preserve order
        _save(s)


def toggle_autoload_media_type(type_id: str) -> list[str]:
    """Toggle a single media type's autoload status.  Returns the updated list.

    .. deprecated:: Use :func:`toggle_autoload_media_embedder` instead.
    """
    if type_id not in _valid_media_types():
        raise ValueError(f"Invalid media type: {type_id!r}")
    with _settings_lock:
        current = get_autoload_media_types()
        if type_id in current:
            current.remove(type_id)
        else:
            current.append(type_id)
        set_autoload_media_types(current)
        return current


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
