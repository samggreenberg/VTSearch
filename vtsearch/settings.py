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
    "show_thumbnails_left": False,
    "show_thumbnails_right": True,
    "autoload_media_types": [],
    "autorun_processors": [],
    "autopilot_top_greens": 3,
    "autopilot_hard_reds": 4,
}

# In-memory cache — loaded once, written on every mutation.
_settings: dict[str, Any] | None = None


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
    if _settings is None:
        _settings = _load()
    return _settings


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------


def get_defaults() -> dict[str, Any]:
    """Return a copy of the default settings (excluding autorun_processors)."""
    return {k: v for k, v in _DEFAULTS.items() if k != "autorun_processors"}


def get_all() -> dict[str, Any]:
    """Return the full settings dict (with defaults filled in)."""
    s = _ensure_loaded()
    result = dict(_DEFAULTS)
    result.update(s)
    return result


VALID_THEMES = ("dark", "light", "highviz")


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
        return cast(_ensure_loaded().get(key, _DEFAULTS[key]))

    def setter(value):
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
    ("show_thumbnails_left", bool, None),
    ("show_thumbnails_right", bool, None),
    ("autopilot_top_greens", int, _clamp_min(int, 1)),
    ("autopilot_hard_reds", int, _clamp_min(int, 1)),
]

for _key, _cast, _coerce in _SETTING_SPECS:
    _g, _s = _make_accessors(_key, _cast, _coerce)
    globals()[f"get_{_key}"] = _g
    globals()[f"set_{_key}"] = _s

del _key, _cast, _coerce, _g, _s


VALID_MEDIA_TYPES = ("audio", "document", "image", "paragraph", "video")


def get_autoload_media_types() -> list[str]:
    """Return the list of autoload media type IDs (empty list if none set)."""
    raw = _ensure_loaded().get("autoload_media_types", _DEFAULTS["autoload_media_types"])
    if isinstance(raw, list):
        return [v for v in raw if v in VALID_MEDIA_TYPES]
    return []


def set_autoload_media_types(value: list[str]) -> None:
    """Set and persist the full list of autoload media types."""
    for v in value:
        if v not in VALID_MEDIA_TYPES:
            raise ValueError(f"Invalid media type: {v!r}")
    s = _ensure_loaded()
    s["autoload_media_types"] = list(dict.fromkeys(value))  # deduplicate, preserve order
    _save(s)


def toggle_autoload_media_type(type_id: str) -> list[str]:
    """Toggle a single media type's autoload status.  Returns the updated list."""
    if type_id not in VALID_MEDIA_TYPES:
        raise ValueError(f"Invalid media type: {type_id!r}")
    current = get_autoload_media_types()
    if type_id in current:
        current.remove(type_id)
    else:
        current.append(type_id)
    set_autoload_media_types(current)
    return current


def get_autorun_processors() -> list[dict[str, Any]]:
    """Return the list of autorun processor recipes."""
    return list(_ensure_loaded().get("autorun_processors", []))


def add_autorun_processor(
    processor_name: str,
    processor_importer: str,
    field_values: dict[str, Any],
) -> None:
    """Add a autorun processor recipe (or overwrite one with the same name)."""
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


def set_settings_path(path: str | Path) -> None:
    """Override the settings file path and reset the in-memory cache.

    Call this before :func:`ensure_autorun_processors_imported` to load
    autorun processors from a custom settings file (e.g. the ``--settings``
    CLI flag).
    """
    global SETTINGS_PATH, _settings
    SETTINGS_PATH = Path(path)
    _settings = None  # force reload on next access


def reset() -> None:
    """Reset the in-memory cache (for testing)."""
    global _settings
    _settings = None
