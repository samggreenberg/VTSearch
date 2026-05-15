"""Persistent settings for VTSearch.

Settings are split across two tiers:

* **Server tier** — shared, single-file settings used to bring the
  process up: dataset/detector directories, concurrency limits, the
  autoload-embedder / autorun-detector lists. Stored in
  ``data/settings.json`` (path = :data:`SETTINGS_PATH`). Loaded once at
  startup, before any user has logged in.
* **Per-user tier** — every other key (preferences, autopilot config,
  per-side view modes, the ``settings_source`` sync target). Stored in
  ``<get_user_data_dir(user)>/user_settings.json``. Resolved per-request
  via :func:`~vtsearch.auth.get_current_user`.

The module exposes the same ``get_<key>`` / ``set_<key>`` accessors as
before; routing between tiers is internal. Auto-save on mutation and
auto-sync to the configured ``settings_source`` are unchanged in shape;
the sync source is per-user and triggers only on per-user writes.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from vtsearch.config import DATA_DIR, DEFAULT_CALIBRATE_COUNT

logger = logging.getLogger(__name__)

#: Path to the server-tier settings file. Tests monkey-patch this; the CLI
#: ``set_settings_path()`` helper also points it at a different file.
SETTINGS_PATH: Path = DATA_DIR / "settings.json"

#: Filename used for the per-user settings file inside
#: ``get_user_data_dir(user)``.
USER_SETTINGS_FILENAME: str = "user_settings.json"

#: Optional override for the directory layout used by per-user settings
#: files. When set, the per-user file resolves to
#: ``_USER_DATA_DIR_OVERRIDE / <username> / USER_SETTINGS_FILENAME``
#: instead of using :func:`vtsearch.auth.get_user_data_dir`. Tests use
#: this to redirect both tiers under a single ``tmp_path`` without
#: monkey-patching the auth module.
_USER_DATA_DIR_OVERRIDE: Path | None = None

# ---------------------------------------------------------------------------
# Defaults, partitioned by tier
# ---------------------------------------------------------------------------

#: Server-tier defaults (keys that live in :data:`SETTINGS_PATH`).
_SERVER_DEFAULTS: dict[str, Any] = {
    "saved_datasets_dir": str(DATA_DIR / "saved_datasets"),
    "detectors_dir": str(DATA_DIR / "detectors"),
    "max_concurrent_dataset_downloads": 1,
    "max_concurrent_dataset_embeddings": 1,
    "autorun_detectors": [],
}

#: Per-user defaults (keys that live in
#: ``<get_user_data_dir(user)>/user_settings.json``).
_USER_DEFAULTS: dict[str, Any] = {
    "volume": 1.0,
    "inclusion": 0,
    "theme": "dark",
    "enrich_descriptions": False,
    "safe_thresholds": False,
    "calibrate_count": DEFAULT_CALIBRATE_COUNT,
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
    "autopilot_enabled": True,
    "hide_autopilot": False,
    "autopilot_top_greens": 3,
    "autopilot_hard_reds": 4,
    "autopilot_resort_interval": 10,
    "autopilot_goal_diversity": 40,
}

#: Combined defaults for callers that want a flat view of every key.
_DEFAULTS: dict[str, Any] = {**_SERVER_DEFAULTS, **_USER_DEFAULTS}

#: Set of keys that belong to the server tier (everything else is per-user).
_SERVER_KEYS: frozenset[str] = frozenset(_SERVER_DEFAULTS.keys())

#: Keys excluded from the "defaults" endpoint (infrastructure settings that
#: should not be reset by the Default button).
_EXCLUDE_FROM_DEFAULTS = {
    "autorun_detectors",
    "saved_datasets_dir",
    "detectors_dir",
    "settings_source",
}

#: Keys excluded from source sync export (to avoid circular config).
_EXCLUDE_FROM_SOURCE_EXPORT = {
    "settings_source",
}

# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------

# Server-tier cache (one process-wide dict).
_server_cache: dict[str, Any] | None = None

# Per-user caches keyed by username.
_user_caches: dict[str, dict[str, Any]] = {}

# Set of usernames whose sync-from-source has run this process lifetime;
# guards lazy startup-equivalent sync per user.
_synced_users: set[str] = set()

# One-shot legacy-migration guard.
_legacy_migrated: bool = False

# Reentrant lock covering both tiers' caches and the syncing flag.
_settings_lock = threading.RLock()

# Set of usernames currently being imported from their settings_source.
# A per-user save that fires while the user is in this set skips
# ``_sync_to_source`` so the import isn't immediately re-exported.
_syncing: set[str] = set()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _server_settings_path() -> Path:
    return SETTINGS_PATH


def _user_settings_path(username: str) -> Path:
    """Return the per-user settings file path for *username*."""
    if _USER_DATA_DIR_OVERRIDE is not None:
        return _USER_DATA_DIR_OVERRIDE / username / USER_SETTINGS_FILENAME
    from vtsearch.auth import get_user_data_dir

    return get_user_data_dir(username) / USER_SETTINGS_FILENAME


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def _load_path(path: Path) -> dict[str, Any]:
    """Read settings from *path*, returning ``{}`` on any failure."""
    if path.exists():
        try:
            text = path.read_text(encoding="utf-8")
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            logger.warning("Failed to read settings file %s: %s", path, exc)
    return {}


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    """Write *data* to *path* via a temp-file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, indent=2) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Cache loaders
# ---------------------------------------------------------------------------


def _ensure_server_loaded() -> dict[str, Any]:
    """Load the server-tier cache on first access and migrate legacy keys."""
    global _server_cache
    with _settings_lock:
        if _server_cache is None:
            _server_cache = _load_path(_server_settings_path())
            _maybe_migrate_legacy_settings_locked()
        return _server_cache


def _ensure_user_loaded(username: str) -> dict[str, Any]:
    """Load *username*'s per-user cache on first access."""
    with _settings_lock:
        cache = _user_caches.get(username)
        if cache is None:
            # Make sure server tier is loaded (and legacy migration has run)
            # before we materialise a fresh user cache, otherwise the legacy
            # migration step might not see this user yet.
            _ensure_server_loaded()
            cache = _user_caches.get(username)
            if cache is None:
                cache = _load_path(_user_settings_path(username))
                _user_caches[username] = cache
            # Lazy sync-from-source on first load for this user this process.
            _maybe_sync_from_source_locked(username)
        return _user_caches[username]


def _maybe_migrate_legacy_settings_locked() -> None:
    """Move per-user keys from a legacy ``data/settings.json`` into the
    default user's per-user file (one-shot, idempotent).

    Called from :func:`_ensure_server_loaded` with the settings lock held.
    """
    global _legacy_migrated
    if _legacy_migrated:
        return
    _legacy_migrated = True
    assert _server_cache is not None
    legacy_user_entries = {k: v for k, v in _server_cache.items() if k not in _SERVER_KEYS}
    if not legacy_user_entries:
        return

    # Migrate into the "default" user's file. The default user is the one
    # the single-user provider returns, and is also the safe target for
    # multi-user upgrades (admins can copy it into other users' files).
    default_user = "default"
    user_path = _user_settings_path(default_user)
    if user_path.exists():
        existing = _load_path(user_path)
        # Existing per-user values win — never clobber a real user file.
        merged: dict[str, Any] = {**legacy_user_entries, **existing}
    else:
        merged = dict(legacy_user_entries)
    try:
        _atomic_write(user_path, merged)
    except Exception as exc:
        logger.warning("Legacy settings migration to %s failed: %s", user_path, exc)
        return

    # Rewrite the server file with only server-tier keys.
    for k in list(_server_cache.keys()):
        if k not in _SERVER_KEYS:
            _server_cache.pop(k, None)
    try:
        _atomic_write(_server_settings_path(), _server_cache)
    except Exception as exc:
        logger.warning("Failed to rewrite server settings after legacy migration: %s", exc)

    # Refresh the default user's cache if it was already materialised
    # (unlikely, since this runs from _ensure_server_loaded, but safe).
    _user_caches[default_user] = _load_path(user_path)
    logger.info(
        "Migrated %d legacy per-user setting(s) from %s into %s",
        len(legacy_user_entries),
        _server_settings_path(),
        user_path,
    )


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------


def _save_server() -> None:
    """Write the server-tier cache to disk."""
    assert _server_cache is not None
    _atomic_write(_server_settings_path(), _server_cache)


def _save_user(username: str) -> None:
    """Write *username*'s cache to disk and (if configured) sync to source."""
    cache = _user_caches.get(username)
    if cache is None:
        return
    _atomic_write(_user_settings_path(username), cache)
    if username not in _syncing:
        _sync_to_source(username, cache)


def _save_for_key(key: str) -> None:
    """Persist whichever tier *key* belongs to."""
    if key in _SERVER_KEYS:
        _save_server()
    else:
        from vtsearch.auth import get_current_user

        _save_user(get_current_user())


# ---------------------------------------------------------------------------
# Low-level value get/set used by accessor factories
# ---------------------------------------------------------------------------


def _read_value(key: str) -> Any:
    """Return the raw stored value for *key* (or its default).

    Routes to the server tier or the current user's tier based on *key*.
    The caller is responsible for casting/coercion.
    """
    if key in _SERVER_KEYS:
        return _ensure_server_loaded().get(key, _SERVER_DEFAULTS[key])
    from vtsearch.auth import get_current_user

    username = get_current_user()
    return _ensure_user_loaded(username).get(key, _USER_DEFAULTS.get(key))


def _write_value(key: str, value: Any) -> None:
    """Persist *value* for *key*, routing to the correct tier."""
    if key in _SERVER_KEYS:
        cache = _ensure_server_loaded()
        cache[key] = value
        _save_server()
        return
    from vtsearch.auth import get_current_user

    username = get_current_user()
    cache = _ensure_user_loaded(username)
    cache[key] = value
    _save_user(username)


def _get_active_cache_for_key(key: str) -> dict[str, Any]:
    """Return the underlying cache dict that owns *key* for the active user.

    Exposed for backwards compatibility with accessor factories that
    previously called ``_ensure_loaded()`` and mutated the returned dict
    directly.
    """
    if key in _SERVER_KEYS:
        return _ensure_server_loaded()
    from vtsearch.auth import get_current_user

    return _ensure_user_loaded(get_current_user())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_defaults() -> dict[str, Any]:
    """Return a copy of the default settings (excluding infrastructure keys)."""
    result = {k: v for k, v in _DEFAULTS.items() if k not in _EXCLUDE_FROM_DEFAULTS}
    valid_types = _valid_media_types()
    result["view_mode_left"] = {tid: _VIEW_MODE_DEFAULTS["left"] for tid in valid_types}
    result["view_mode_right"] = {tid: _VIEW_MODE_DEFAULTS["right"] for tid in valid_types}
    result["grid_icon_size_left"] = {tid: _GRID_ICON_SIZE_DEFAULT for tid in valid_types}
    result["grid_icon_size_right"] = {tid: _GRID_ICON_SIZE_DEFAULT for tid in valid_types}
    result["focus_mode_left"] = {tid: _FOCUS_MODE_DEFAULTS["left"] for tid in valid_types}
    result["focus_mode_right"] = {tid: _FOCUS_MODE_DEFAULTS["right"] for tid in valid_types}
    result["panel_pct_left"] = {tid: _PANEL_PX_DEFAULTS["left"] for tid in valid_types}
    result["panel_pct_right"] = {tid: _PANEL_PX_DEFAULTS["right"] for tid in valid_types}
    return result


def get_user_settings() -> dict[str, Any]:
    """Return the current user's per-user settings (with defaults filled in).

    Used by the settings-export route and the per-user sync source so an
    export only carries this user's preferences — not the shared
    server-tier infrastructure keys (``saved_datasets_dir``,
    ``autoload_media_embedders`` etc.) and not other users' files.
    """
    from vtsearch.auth import get_current_user

    username = get_current_user()
    with _settings_lock:
        user = _ensure_user_loaded(username)
        result = dict(_USER_DEFAULTS)
        result.update(user)
        result["view_mode_left"] = get_view_mode_left()
        result["view_mode_right"] = get_view_mode_right()
        result["grid_icon_size_left"] = get_grid_icon_size_left()
        result["grid_icon_size_right"] = get_grid_icon_size_right()
        result["focus_mode_left"] = get_focus_mode_left()
        result["focus_mode_right"] = get_focus_mode_right()
        result["panel_pct_left"] = get_panel_pct_left()
        result["panel_pct_right"] = get_panel_pct_right()
        return result


def get_all() -> dict[str, Any]:
    """Return the merged settings dict for the current user.

    Combines the server-tier defaults+overrides with the current user's
    per-user defaults+overrides. The per-side view-mode dicts are
    returned in expanded (per-media-type) form.
    """
    from vtsearch.auth import get_current_user

    username = get_current_user()
    with _settings_lock:
        server = _ensure_server_loaded()
        user = _ensure_user_loaded(username)
        result = dict(_DEFAULTS)
        result.update(server)
        result.update(user)
        # Always return expanded per-media-type view/focus/panel dicts.
        result["view_mode_left"] = get_view_mode_left()
        result["view_mode_right"] = get_view_mode_right()
        result["grid_icon_size_left"] = get_grid_icon_size_left()
        result["grid_icon_size_right"] = get_grid_icon_size_right()
        result["focus_mode_left"] = get_focus_mode_left()
        result["focus_mode_right"] = get_focus_mode_right()
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


def get_autorun_detectors() -> list[str]:
    """Return the list of detector names flagged for autorun.

    Each name maps to a JSON file under ``data/detectors/``; scoring resolves
    the labelset's origins, re-embeds, trains an MLP, and applies it to the
    loaded dataset.
    """
    with _settings_lock:
        raw = _ensure_server_loaded().get("autorun_detectors", [])
        if isinstance(raw, list):
            return list(raw)
        return []


def set_autorun_detectors(value: list[str]) -> None:
    """Set and persist the full list of autorun detector names."""
    with _settings_lock:
        cache = _ensure_server_loaded()
        cache["autorun_detectors"] = list(dict.fromkeys(value))  # dedupe, preserve order
        _save_server()


def add_autorun_detector(name: str) -> None:
    """Add a detector name to the autorun list (idempotent)."""
    with _settings_lock:
        current = get_autorun_detectors()
        if name not in current:
            current.append(name)
            set_autorun_detectors(current)


def remove_autorun_detector(name: str) -> bool:
    """Remove a detector name from the autorun list. Returns True if found."""
    with _settings_lock:
        current = get_autorun_detectors()
        if name in current:
            current.remove(name)
            set_autorun_detectors(current)
            return True
        return False


def is_autorun_detector(name: str) -> bool:
    """Check whether a detector name is in the autorun list."""
    return name in get_autorun_detectors()


# -------------------------------------------------------------------
# Directory path settings (server tier)
# -------------------------------------------------------------------


def _get_dir(key: str) -> Path:
    """Return a server-tier directory path setting as a :class:`~pathlib.Path`."""
    with _settings_lock:
        raw = _ensure_server_loaded().get(key, _SERVER_DEFAULTS[key])
    return Path(raw)


def _set_dir(key: str, value: str | Path) -> None:
    """Persist a server-tier directory path setting."""
    with _settings_lock:
        cache = _ensure_server_loaded()
        cache[key] = str(value)
        _save_server()


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


def set_settings_path(path: str | Path) -> None:
    """Override the server-tier settings file path and reset its cache.

    Used by the CLI to point at a project- or run-specific settings file.
    Does not affect per-user settings files — those still live under
    :func:`vtsearch.auth.get_user_data_dir`.
    """
    global SETTINGS_PATH, _server_cache, _legacy_migrated
    with _settings_lock:
        SETTINGS_PATH = Path(path)
        _server_cache = None
        _legacy_migrated = False


def set_user_data_dir_override(path: Path | None) -> None:
    """Redirect per-user settings files under *path* (tests only).

    When *path* is set, every per-user settings file resolves to
    ``path / <username> / user_settings.json``, bypassing
    :func:`vtsearch.auth.get_user_data_dir`. Pass ``None`` to clear.
    """
    global _USER_DATA_DIR_OVERRIDE
    with _settings_lock:
        _USER_DATA_DIR_OVERRIDE = Path(path) if path is not None else None


def reset() -> None:
    """Reset every in-memory cache (for testing)."""
    global _server_cache, _legacy_migrated
    with _settings_lock:
        _server_cache = None
        _user_caches.clear()
        _synced_users.clear()
        _syncing.clear()
        _legacy_migrated = False


# -------------------------------------------------------------------
# Settings source (bidirectional sync, per-user)
# -------------------------------------------------------------------


def get_settings_source_config() -> dict[str, Any] | None:
    """Return the active user's settings source config, or ``None`` if unset.

    Config shape::

        {
            "source_name": "server_json_file",
            "field_values": {"filepath": "data/{username}.settings.json"}
        }
    """
    from vtsearch.auth import get_current_user

    with _settings_lock:
        cache = _ensure_user_loaded(get_current_user())
        cfg = cache.get("settings_source")
    if isinstance(cfg, dict) and cfg.get("source_name"):
        return cfg
    return None


def set_settings_source_config(config: dict[str, Any] | None) -> None:
    """Set or clear the active user's settings source config."""
    from vtsearch.auth import get_current_user

    username = get_current_user()
    with _settings_lock:
        cache = _ensure_user_loaded(username)
        if config is None:
            cache.pop("settings_source", None)
        else:
            cache["settings_source"] = config
        _save_user(username)


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
    """Pull settings from the active user's source and apply them.

    Returns the imported settings dict, or ``None`` if no source is
    configured or the source file doesn't exist yet.

    Triggered:

    - Lazily on first per-user cache load each process lifetime (see
      :func:`_maybe_sync_from_source_locked`).
    - Manually via ``POST /api/settings-sources/sync``.
    """
    cfg = get_settings_source_config()
    if cfg is None:
        return None

    from vtsearch.auth import get_current_user
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

    username = get_current_user()
    with _settings_lock:
        _syncing.add(username)
        try:
            _apply_settings(imported)
        finally:
            _syncing.discard(username)

    return imported


def _maybe_sync_from_source_locked(username: str) -> None:
    """Once-per-process lazy sync-from-source for *username*.

    Called from :func:`_ensure_user_loaded` with the settings lock held.
    Safe to call repeatedly: noops after the first successful run for the
    user. Errors are logged and swallowed so a misconfigured source never
    blocks ordinary settings access.
    """
    if username in _synced_users:
        return
    _synced_users.add(username)

    cache = _user_caches.get(username) or {}
    cfg = cache.get("settings_source")
    if not isinstance(cfg, dict) or not cfg.get("source_name"):
        return

    from vtsearch.settings_io.sources import get_settings_source

    source = get_settings_source(cfg["source_name"])
    if source is None:
        logger.warning("Unknown settings source: %s", cfg["source_name"])
        return

    field_values = cfg.get("field_values", {})
    try:
        imported = source.load(field_values)
    except Exception as exc:
        logger.exception("Failed to load from settings source for %s: %s", username, exc)
        return
    if not imported:
        return

    _syncing.add(username)
    try:
        _apply_settings(imported)
    finally:
        _syncing.discard(username)


def _sync_to_source(username: str, data: dict[str, Any]) -> None:
    """Push *username*'s current settings to their active source (if any).

    Called from :func:`_save_user` after the per-user file is written.
    Strips the ``settings_source`` key itself to avoid circular config.
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
        logger.exception("Failed to sync settings to source for %s: %s", username, exc)
