"""Persistent settings for VTSearch.

Settings are split across two tiers:

* **Server tier** - shared, single-file settings used to bring the
  process up: dataset/detector directories, concurrency limits, the
  autoload-embedder / Auto-Find detector lists. Stored in
  ``data/settings.json`` (path = :data:`SETTINGS_PATH`). Loaded once at
  startup, before any user has logged in.
* **Per-user tier** - every other key (preferences, autopilot config,
  per-side view modes, the ``settings_source`` sync target). Stored in
  ``<get_user_data_dir(user)>/user_settings.json``. Resolved per-request
  via :func:`~vtsearch.auth.get_current_user`.

The module exposes the same ``get_<key>`` / ``set_<key>`` accessors as
before; routing between tiers is internal. Auto-save on mutation and
auto-sync to the configured ``settings_source`` are unchanged in shape;
the sync source is per-user and triggers only on per-user writes.
"""

from __future__ import annotations

import threading
from pathlib import Path
from collections.abc import Iterable
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import TypeAdapter, ValidationError

from vtscore.config import DATA_DIR
from vtsearch.settings_models import (
    VALID_FOCUS_MODES,
    VALID_GRID_ICON_SIZES,
    VALID_PANEL_PX,
    ServerSettings,
    UserSettings,
)
from vtsearch.settings_store import UserSettingsStore, UserSyncState as _UserSyncState

if TYPE_CHECKING:
    # The accessors below are generated dynamically by the loop at the
    # bottom of this module (``_SETTING_SPECS`` + ``make_accessors``).
    # Declare their signatures here so static type-checkers can resolve
    # ``vtsearch.settings.get_<key>()`` / ``set_<key>()`` calls from
    # other modules. Keep this list in sync with ``_SETTING_SPECS`` and
    # the per-side factory invocations below.

    def get_volume() -> float: ...
    def set_volume(value: float) -> None: ...
    def get_inclusion() -> int: ...
    def set_inclusion(value: int) -> None: ...
    def get_theme() -> str: ...
    def set_theme(value: str) -> None: ...
    def get_enrich_descriptions() -> bool: ...
    def set_enrich_descriptions(value: bool) -> None: ...
    def get_safe_thresholds() -> bool: ...
    def set_safe_thresholds(value: bool) -> None: ...
    def get_calibrate_count() -> int: ...
    def set_calibrate_count(value: int) -> None: ...
    def get_calibration_fraction() -> float: ...
    def set_calibration_fraction(value: float) -> None: ...
    def get_audio_playing() -> bool: ...
    def set_audio_playing(value: bool) -> None: ...
    def get_show_animations() -> str: ...
    def set_show_animations(value: str) -> None: ...
    def get_show_metadata() -> bool: ...
    def set_show_metadata(value: bool) -> None: ...
    def get_label_hint_dismissed() -> bool: ...
    def set_label_hint_dismissed(value: bool) -> None: ...
    def get_autopilot_enabled() -> bool: ...
    def set_autopilot_enabled(value: bool) -> None: ...
    def get_hide_autopilot() -> bool: ...
    def set_hide_autopilot(value: bool) -> None: ...
    def get_enable_achievements() -> bool: ...
    def set_enable_achievements(value: bool) -> None: ...
    def get_browse_panel_width() -> int: ...
    def set_browse_panel_width(value: int) -> None: ...
    def get_browse_graphics() -> str: ...
    def set_browse_graphics(value: str) -> None: ...
    def get_browse_colormap() -> dict[str, str]: ...
    def set_browse_colormap(value: dict[str, str]) -> None: ...
    def get_browse_icon_size() -> dict[str, str]: ...
    def set_browse_icon_size(value: dict[str, str]) -> None: ...
    def get_browse_thumbnail_border() -> dict[str, int]: ...
    def set_browse_thumbnail_border(value: dict[str, int]) -> None: ...
    def get_browse_compact() -> dict[str, bool]: ...
    def set_browse_compact(value: dict[str, bool]) -> None: ...
    def get_browse_mouse_zooms_per_level() -> dict[str, int]: ...
    def set_browse_mouse_zooms_per_level(value: dict[str, int]) -> None: ...
    def get_browse_signposts() -> dict[str, bool]: ...
    def set_browse_signposts(value: dict[str, bool]) -> None: ...
    def get_browse_signpost_captioner() -> dict[str, bool]: ...
    def set_browse_signpost_captioner(value: dict[str, bool]) -> None: ...
    def get_browse_signpost_vocab() -> dict[str, list[str]]: ...
    def set_browse_signpost_vocab(value: dict[str, list[str]]) -> None: ...
    def get_autopilot_top_greens() -> int: ...
    def set_autopilot_top_greens(value: int) -> None: ...
    def get_autopilot_hard_reds() -> int: ...
    def set_autopilot_hard_reds(value: int) -> None: ...
    def get_autopilot_resort_interval() -> int: ...
    def set_autopilot_resort_interval(value: int) -> None: ...
    def get_autopilot_goal_diversity() -> int: ...
    def set_autopilot_goal_diversity(value: int) -> None: ...
    def get_max_concurrent_dataset_downloads() -> int: ...
    def set_max_concurrent_dataset_downloads(value: int) -> None: ...
    def get_max_concurrent_dataset_embeddings() -> int: ...
    def set_max_concurrent_dataset_embeddings(value: int) -> None: ...
    def get_dataset_max_age_days() -> int | None: ...
    def set_dataset_max_age_days(value: int | None) -> None: ...
    def get_support_email() -> str: ...
    def set_support_email(value: str) -> None: ...
    def get_semantic_only() -> bool: ...
    def set_semantic_only(value: bool) -> None: ...
    def get_projection_n_neighbors() -> int: ...
    def set_projection_n_neighbors(value: int) -> None: ...
    def get_projection_min_dist() -> float: ...
    def set_projection_min_dist(value: float) -> None: ...
    def get_default_settings_source() -> dict[str, Any] | None: ...
    def set_default_settings_source(value: dict[str, Any] | None) -> None: ...

    # Per-side accessors generated by ``_make_per_side_setting``.
    def get_grid_icon_size_left() -> dict[str, str]: ...
    def get_grid_icon_size_right() -> dict[str, str]: ...
    def set_grid_icon_size_left(value: dict[str, str] | str) -> None: ...
    def set_grid_icon_size_right(value: dict[str, str] | str) -> None: ...
    def get_focus_mode_left() -> dict[str, str]: ...
    def get_focus_mode_right() -> dict[str, str]: ...
    def set_focus_mode_left(value: dict[str, str] | str) -> None: ...
    def set_focus_mode_right(value: dict[str, str] | str) -> None: ...
    def get_panel_pct_left() -> dict[str, int]: ...
    def get_panel_pct_right() -> dict[str, int]: ...
    def set_panel_pct_left(value: dict[str, int] | int | float) -> None: ...
    def set_panel_pct_right(value: dict[str, int] | int | float) -> None: ...

    def get_last_embedder_per_media_type() -> dict[str, str]: ...
    def set_last_embedder_per_media_type(value: dict[str, str]) -> None: ...

    def get_import_defaults_by_media_type() -> dict[str, dict[str, Any]]: ...
    def set_import_defaults_by_media_type(value: dict[str, dict[str, Any]]) -> None: ...

    def get_solo_media_type() -> str | None: ...
    def set_solo_media_type(value: str | None) -> None: ...
    def get_solo_media_type_explicit() -> bool: ...
    def set_solo_media_type_explicit(value: bool) -> None: ...

    def get_solo_embedder_per_media_type() -> dict[str, str]: ...
    def set_solo_embedder_per_media_type(value: dict[str, str]) -> None: ...

    def get_recent_sessions() -> list[dict[str, Any]]: ...
    def set_recent_sessions(value: list[dict[str, Any]]) -> None: ...

    def get_hidden_plugins() -> dict[str, list[str]]: ...
    def set_hidden_plugins(value: dict[str, list[str]]) -> None: ...


#: Path to the server-tier settings file. Tests monkey-patch this; the CLI
#: ``set_settings_path()`` helper also points it at a different file.
SETTINGS_PATH: Path = DATA_DIR / "settings.json"

#: Process-level fallback for the per-user ``solo_media_type`` setting, set
#: by :func:`set_cli_solo_media_type` from the ``--solo-media-type`` flag
#: in :mod:`app`. ``None`` means "no CLI default"; a user with
#: ``solo_media_type_explicit=False`` will see this value (or ``None``) as
#: their effective solo mediaType. A user who has explicitly set their own
#: value (including explicitly ``None`` for "show everything") overrides
#: this fallback - see :func:`get_effective_solo_media_type`.
_cli_solo_media_type: str | None = None

#: Process-level fallback for the ``hidden_plugins`` server setting, set
#: by :func:`set_cli_hidden_plugins` / :func:`add_cli_hidden_plugin` from
#: the repeatable ``--hide-plugin family:name`` flag in :mod:`app`. Maps
#: a plugin-family id to the set of plugin ``name``s to hide. Merged with
#: the persisted ``hidden_plugins`` server setting at read time - see
#: :func:`get_effective_hidden_plugins`. Empty dict means "no CLI hides".
_cli_hidden_plugins: dict[str, set[str]] = {}

#: Process-level fallback for the per-user
#: ``solo_embedder_per_media_type`` setting, set by
#: :func:`set_cli_solo_embedder` from the (repeatable) ``--solo-embedder``
#: flag in :mod:`app`. Maps ``media_type_id`` → embedder name. Empty means
#: "no CLI default". The resolver :func:`get_effective_solo_embedders`
#: layers per-user entries over this dict (per-key), so a user can override
#: individual mediaTypes without losing the others.
_cli_solo_embedders: dict[str, str] = {}

#: Process-level override for the server-tier ``dataset_max_age_days``
#: setting, set by :func:`set_cli_dataset_max_age_days` from the
#: ``--dataset-max-age-days`` flag in :mod:`app`. ``None`` means "no CLI
#: override" - reads fall through to the persisted server setting. When a
#: value is set it applies to every user and wins over the persisted file
#: for the lifetime of the process; the setting is not user-editable via
#: the API (see :func:`get_effective_dataset_max_age_days`).
_cli_dataset_max_age_days: int | None = None

#: Process-level override for the server-tier ``support_email`` setting, set
#: by :func:`set_cli_support_email` from the ``--support-email`` flag in
#: :mod:`app`. ``None`` means "no CLI override" - reads fall through to the
#: persisted server setting (and then the model default). When a value is set
#: it applies to every user and wins over the persisted file for the lifetime
#: of the process; the setting is not user-editable via the API (see
#: :func:`get_effective_support_email`).
_cli_support_email: str | None = None

#: Process-level override for the server-tier ``semantic_only`` setting, set by
#: :func:`set_cli_semantic_only` from the ``--semantic-only`` flag (or the
#: ``VTSEARCH_SEMANTIC_ONLY`` env var) in :mod:`app`. ``None`` means "no CLI
#: override" - reads fall through to the persisted server setting. ``True``
#: locks the instance to Semantic embedders for the lifetime of the process;
#: the setting is not user-editable via the API (see
#: :func:`get_effective_semantic_only`).
_cli_semantic_only: bool | None = None

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

#: Lazy caches for the model-derived defaults dicts.  Instantiating
#: :class:`ServerSettings` fires its pydantic ``default_factory`` callbacks
#: (notably the GPU-aware default for ``max_concurrent_dataset_embeddings``),
#: and that probe imports torch - ~870ms at startup just to pick an int.
#: Defer the instantiation to first read so ``import vtsearch.settings``
#: stays torch-free.
_SERVER_DEFAULTS_CACHE: dict[str, Any] | None = None
_USER_DEFAULTS_CACHE: dict[str, Any] | None = None


def _server_defaults() -> dict[str, Any]:
    global _SERVER_DEFAULTS_CACHE
    if _SERVER_DEFAULTS_CACHE is None:
        _SERVER_DEFAULTS_CACHE = ServerSettings().model_dump()
    return _SERVER_DEFAULTS_CACHE


def _user_defaults() -> dict[str, Any]:
    global _USER_DEFAULTS_CACHE
    if _USER_DEFAULTS_CACHE is None:
        _USER_DEFAULTS_CACHE = UserSettings().model_dump()
    return _USER_DEFAULTS_CACHE


def _all_defaults() -> dict[str, Any]:
    return {**_server_defaults(), **_user_defaults()}


def __getattr__(name: str) -> Any:
    # PEP 562 - expose ``_SERVER_DEFAULTS`` / ``_USER_DEFAULTS`` / ``_DEFAULTS``
    # as attributes for external readers (tests, future callers) without
    # forcing the eager pydantic instantiation at module load.
    if name == "_SERVER_DEFAULTS":
        return _server_defaults()
    if name == "_USER_DEFAULTS":
        return _user_defaults()
    if name == "_DEFAULTS":
        return _all_defaults()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


#: Set of keys that belong to the server tier (everything else is per-user).
#: Reading ``model_fields`` doesn't instantiate the model, so this stays eager.
_SERVER_KEYS: frozenset[str] = frozenset(ServerSettings.model_fields.keys())

#: Per-user keys for which the built-in "default" user reads through to the
#: *server* settings file when the key is absent from its own per-user file.
#: These are the Auto-Find knobs: per-user in multi-user deployments, but the
#: single-user GUI (everyone is "default") and the CLI ``--settings`` flat file
#: still expect a value placed in ``settings.json`` to take effect. The
#: read-through (see :func:`_read_value`) makes that work without the
#: destructive legacy migration moving them out of the server file (see
#: ``UserSettingsStore._maybe_migrate_legacy_settings``, which skips
#: these keys).
_DEFAULT_USER_FALLBACK_KEYS: frozenset[str] = frozenset(
    {"autofind_detectors", "autofind_exporter", "autofind_exporter_field_values"}
)

#: Keys excluded from the "defaults" endpoint (infrastructure settings that
#: should not be reset by the Default button).
_EXCLUDE_FROM_DEFAULTS = {
    "autofind_detectors",
    "autofind_exporter",
    "autofind_exporter_field_values",
    "saved_datasets_dir",
    "detectors_dir",
    "settings_source",
    "default_settings_source",
}

#: Keys excluded from source sync export (to avoid circular config). Both the
#: per-user ``settings_source`` and the deployment-wide
#: ``default_settings_source`` name a sync target, so neither is exported into
#: the target itself.
_EXCLUDE_FROM_SOURCE_EXPORT = {
    "settings_source",
    "default_settings_source",
}

# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------

# The mutable engine state below lives here (not on the store) because
# other modules import these containers by name: ``vtsearch.achievements``
# reads ``_user_caches`` / ``_settings_lock``, and the sync-source tests
# poke ``_sync_state``. The :class:`~vtsearch.settings_store.UserSettingsStore`
# instance (created at the bottom of this module) is handed these same
# objects by reference, so both views mutate one set of containers. The
# reassignable scalars (``server_cache``, ``legacy_migrated``) and the
# per-user sync locks live solely on the store.

# Per-user caches keyed by username.
_user_caches: dict[str, dict[str, Any]] = {}

# Per-user sync bookkeeping.  Replaces the old ``_synced_users: set[str]``,
# which couldn't distinguish "never synced" from "synced ages ago" or
# "tried once and failed."
_sync_state: dict[str, _UserSyncState] = {}

# Reentrant lock covering both tiers' caches, the syncing flag, and
# ``_sync_state``.  Not held while running the actual sync I/O.
_settings_lock = threading.RLock()

# Rate-limit window for the ``peek_version`` freshness probe.  Settings
# reads happen in hot paths (``before_request``, every accessor); we
# don't want to stat the source file on every one.  The first read in a
# window does the probe; subsequent reads inside the window short-circuit
# back to the local cache.
_FRESHNESS_CHECK_INTERVAL = 1.0

# Set of usernames currently being imported from their settings_source.
# A per-user save that fires while the user is in this set skips
# ``_sync_to_source`` so the import isn't immediately re-exported.  Also
# acts as a recursion guard for ``_ensure_user_loaded``: a setter called
# from inside ``_apply_settings`` re-enters the function but must skip
# the sync path.
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
# Cache loaders (thin delegators onto the store)
#
# ``_ensure_server_loaded`` / ``_ensure_user_loaded`` keep their names and
# module-level identity because ``vtsearch.achievements`` imports
# ``_ensure_user_loaded`` directly, and the rest of this module's read/write
# paths call both by name. The actual two-tier cache + sync machinery lives
# on ``_store`` (see :mod:`vtsearch.settings_store`).
# ---------------------------------------------------------------------------


def _ensure_server_loaded() -> dict[str, Any]:
    """Load the server-tier cache on first access and migrate legacy keys."""
    return _store.ensure_server_loaded()


def _ensure_user_loaded(username: str) -> dict[str, Any]:
    """Load *username*'s per-user cache and reconcile with its source."""
    return _store.ensure_user_loaded(username)


# ---------------------------------------------------------------------------
# Save helpers (thin delegators onto the store)
# ---------------------------------------------------------------------------


def _mutate_server_locked(mutator) -> None:
    """Atomically read-modify-write the server-tier settings file."""
    _store.mutate_server_locked(mutator)


def _mutate_user_locked(username: str, mutator) -> dict[str, Any] | None:
    """Atomically read-modify-write *username*'s per-user settings file.

    Returns a post-mutation snapshot for ``_sync_to_source`` (invoked
    outside the file lock), or ``None`` when a sync-from-source import is
    in progress for this user.
    """
    return _store.mutate_user_locked(username, mutator)


def mutate_user(mutator) -> None:
    """Atomically read-modify-write the current user's settings file.

    The on-disk file is locked across processes, re-read fresh,
    *mutator(cache)* runs to mutate the loaded dict in place, and the
    result is written back atomically. Use this whenever you need to
    update a nested structure (counters, list appends, dict merges) -
    a plain ``set_*`` call only round-trips the top-level key correctly
    if you replace it wholesale, but ``mutate_user`` is correct for
    any in-place change.

    Triggers ``_sync_to_source`` after the lock is released (so a slow
    sync target can't block other settings writes).
    """
    from vtsearch.auth import get_current_user

    username = get_current_user()
    _ensure_user_loaded(username)
    sync_data = _mutate_user_locked(username, mutator)
    if sync_data is not None:
        # ``mutate_user`` can change arbitrary top-level keys; mark
        # every exportable key dirty so an auto re-sync between now
        # and the ``_sync_to_source`` push doesn't clobber the mutation.
        _mark_user_keys_dirty(username, [k for k in sync_data if k not in _EXCLUDE_FROM_SOURCE_EXPORT])
        _sync_to_source(username, sync_data)


# ---------------------------------------------------------------------------
# Low-level value get/set used by accessor factories
# ---------------------------------------------------------------------------


def _read_value(key: str) -> Any:
    """Return the raw stored value for *key* (or its default).

    Routes to the server tier or the current user's tier based on *key*.
    The caller is responsible for casting/coercion.
    """
    if key in _SERVER_KEYS:
        return _ensure_server_loaded().get(key, _server_defaults()[key])
    from vtsearch.auth import get_current_user

    username = get_current_user()
    user_cache = _ensure_user_loaded(username)
    if key in user_cache:
        return user_cache[key]
    # Default-user read-through: the single-user GUI and the CLI ``--settings``
    # flat file carry these Auto-Find keys in the server settings file. Honor
    # them for the built-in "default" user when not set in its own file, so
    # those workflows keep working without making the setting truly server-wide.
    if username == "default" and key in _DEFAULT_USER_FALLBACK_KEYS:
        server_cache = _ensure_server_loaded()
        if key in server_cache:
            return server_cache[key]
    return _user_defaults().get(key)


def _write_value(key: str, value: Any) -> None:
    """Persist *value* for *key*, routing to the correct tier.

    Each call is a single read-modify-write under the cross-process
    file lock: the on-disk file is re-read, *key* is replaced, the
    merged dict is written back, then the in-memory cache is refreshed
    to match. This means two processes setting *different* keys on the
    same user no longer clobber each other.
    """
    if key in _SERVER_KEYS:
        _ensure_server_loaded()  # one-shot legacy migration
        _mutate_server_locked(lambda c: c.__setitem__(key, value))
        return
    from vtsearch.auth import get_current_user

    username = get_current_user()
    _ensure_user_loaded(username)  # reconciles with source if due
    sync_data = _mutate_user_locked(username, lambda c: c.__setitem__(key, value))
    if sync_data is not None:
        # User-initiated write (not from an in-progress import).  Mark
        # this key dirty so an auto re-sync running between now and the
        # ``_sync_to_source`` push leaves it alone - the source push
        # below clears the dirty flag again once it succeeds.
        if key not in _EXCLUDE_FROM_SOURCE_EXPORT:
            _mark_user_keys_dirty(username, [key])
        _sync_to_source(username, sync_data)


def _mark_user_keys_dirty(username: str, keys) -> None:
    """Add *keys* to the user's ``dirty_keys`` set."""
    _store.mark_user_keys_dirty(username, keys)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_defaults() -> dict[str, Any]:
    """Return a copy of the default settings (excluding infrastructure keys)."""
    result = {k: v for k, v in _all_defaults().items() if k not in _EXCLUDE_FROM_DEFAULTS}
    valid_types = _valid_media_types()
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
    export only carries this user's preferences - not the shared
    server-tier infrastructure keys (``saved_datasets_dir``,
    ``autoload_media_embedders`` etc.) and not other users' files.
    """
    from vtsearch.auth import get_current_user

    username = get_current_user()
    # ``_ensure_user_loaded`` may trigger sync-from-source which calls
    # setters that acquire ``_file_lock``. Calling it outside
    # ``_settings_lock`` keeps the canonical lock order
    # (file_lock → settings_lock) and avoids an AB-BA deadlock with
    # concurrent writers.
    _ensure_user_loaded(username)
    with _settings_lock:
        user_copy = dict(_user_caches.get(username, {}))
    result = dict(_user_defaults())
    result.update(user_copy)
    # Same legacy migration as ``get_all`` so settings exports carry a value
    # a fresh import will accept.
    result["show_animations"] = get_show_animations()  # type: ignore[name-defined]  # noqa: F821
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
    _ensure_server_loaded()
    _ensure_user_loaded(username)
    with _settings_lock:
        server_copy = dict(_store.server_cache or {})
        user_copy = dict(_user_caches.get(username, {}))
    result = dict(_all_defaults())
    result.update(server_copy)
    result.update(user_copy)
    # Always return expanded per-media-type size/focus/panel dicts.
    result["grid_icon_size_left"] = get_grid_icon_size_left()
    result["grid_icon_size_right"] = get_grid_icon_size_right()
    result["focus_mode_left"] = get_focus_mode_left()
    result["focus_mode_right"] = get_focus_mode_right()
    result["panel_pct_left"] = get_panel_pct_left()
    result["panel_pct_right"] = get_panel_pct_right()
    # Auto-Find keys go through their accessors so the default-user read-through
    # (and per-user isolation for named users) is applied consistently: the
    # plain ``result.update(server_copy)`` above would otherwise leak a legacy
    # server-file Auto-Find list to every named user.
    # ``show_animations`` goes through its accessor so pre-enum settings
    # files (boolean ``True``/``"True"``; see ``coerce_animation_mode``) are
    # migrated on read - the raw merge above would leak the legacy value to
    # GET, and the frontend would echo it into a PUT that 422s.
    result["show_animations"] = get_show_animations()  # type: ignore[name-defined]  # noqa: F821
    result["autofind_detectors"] = get_autofind_detectors()
    result["autofind_exporter"] = get_autofind_exporter()  # type: ignore[name-defined]  # noqa: F821
    result["autofind_exporter_field_values"] = get_autofind_exporter_field_values()  # type: ignore[name-defined]  # noqa: F821
    return result


# -------------------------------------------------------------------
# Pydantic-driven generation of simple get_<key> / set_<key> pairs.
#
# Each :class:`UserSettings` / :class:`ServerSettings` field becomes a
# pair of module-level accessors. Validation (clamping, one-of) lives
# on the model; the shims below only marshal values between the cache
# dict and ``model_validate({key: value})``.
# -------------------------------------------------------------------


#: Cache of per-field validators, keyed by ``(model, key)``. Built lazily on
#: first use; a ``TypeAdapter`` compiles a core-schema once, so caching it
#: makes ``_validate_field`` O(one field) instead of constructing and dumping
#: the whole ~40-field model (which also fires every absent field's
#: ``default_factory``) on every settings read/write. ``None`` marks a field
#: that must keep the whole-model path (see :func:`_build_field_adapter`).
_FIELD_ADAPTERS: dict[tuple[type, str], TypeAdapter[Any] | None] = {}


def _build_field_adapter(model: type, key: str) -> TypeAdapter[Any] | None:
    """Build a validator for a single field of *model*, or ``None``.

    ``None`` means "validate through the whole model": a ``TypeAdapter``
    only sees the field's annotation, so any ``@field_validator`` /
    ``@model_validator`` hook would be silently skipped. Neither settings
    model currently defines such hooks (all validation lives in
    ``Annotated[..., BeforeValidator(...)]`` metadata), so today every
    field gets a fast per-field adapter; the guard keeps a future hook
    from being dropped without anyone noticing.
    """
    decorators = model.__pydantic_decorators__  # type: ignore[attr-defined]
    hooked = {f for dec in decorators.field_validators.values() for f in dec.info.fields}
    if decorators.model_validators or "*" in hooked or key in hooked:
        return None
    field_info = model.model_fields[key]  # type: ignore[attr-defined]
    annotation = field_info.annotation
    if field_info.metadata:
        # Pydantic unpacks ``Annotated[T, meta...]`` into ``annotation=T``
        # + ``metadata=[meta...]``; re-wrap so the ``BeforeValidator``
        # clamps and case-folds still run under the adapter.
        annotation = Annotated[tuple([annotation, *field_info.metadata])]
    return TypeAdapter(annotation)


def _validate_field(model: type, key: str, value: Any) -> Any:
    """Run *value* through *model*'s validator for *key* and return the result.

    Uses a cached per-field :class:`TypeAdapter` built from the field's
    annotation (metadata re-attached) so per-field ``BeforeValidator``
    clamps and enum checks fire as if the value had been loaded from disk,
    without paying a whole-model construct-and-dump on every read/write.
    A :class:`pydantic.ValidationError` is surfaced as :class:`ValueError`
    with a compact message - matches the error shape callers expect from
    the legacy spec-driven setters.
    """
    cache_key = (model, key)
    if cache_key not in _FIELD_ADAPTERS:
        _FIELD_ADAPTERS[cache_key] = _build_field_adapter(model, key)
    adapter = _FIELD_ADAPTERS[cache_key]
    try:
        if adapter is None:
            return model.model_validate({key: value}).model_dump()[key]  # type: ignore[attr-defined]
        return adapter.validate_python(value)
    except ValidationError as exc:
        first = exc.errors()[0] if exc.errors() else {"msg": str(exc)}
        raise ValueError(f"Invalid {key}: {first.get('msg', exc)}") from None


def _make_scalar_accessors(model: type, key: str):
    def getter():
        with _settings_lock:
            raw = _read_value(key)
        try:
            return _validate_field(model, key, raw)
        except ValueError:
            # Corrupt disk value - fall back to the default so callers
            # never see partially-typed garbage.
            return model.model_fields[key].get_default(call_default_factory=True)

    def setter(value):
        # Lock acquisition is delegated to ``_write_value`` →
        # ``_mutate_*_locked``, which takes ``_file_lock`` first and
        # then ``_settings_lock``. Acquiring ``_settings_lock`` here
        # would invert the order and risk an AB-BA deadlock with paths
        # that enter ``_mutate_*_locked`` directly.
        coerced = _validate_field(model, key, value)
        _write_value(key, coerced)

    getter.__name__ = f"get_{key}"
    setter.__name__ = f"set_{key}"
    return getter, setter


# Generate accessors for every field in both models. The ``autofind_detectors``
# key on ServerSettings is excluded because it has hand-written accessors
# below with extra semantics (dedupe, add/remove/is_).
_SKIP_AUTOGEN = {"autofind_detectors", "saved_datasets_dir", "detectors_dir"}
_PER_SIDE_KEYS = {
    "grid_icon_size_left",
    "grid_icon_size_right",
    "focus_mode_left",
    "focus_mode_right",
    "panel_pct_left",
    "panel_pct_right",
}

for _model in (ServerSettings, UserSettings):
    for _field_name in _model.model_fields:
        if _field_name in _SKIP_AUTOGEN or _field_name in _PER_SIDE_KEYS:
            continue
        _g, _s = _make_scalar_accessors(_model, _field_name)
        globals()[f"get_{_field_name}"] = _g
        globals()[f"set_{_field_name}"] = _s

del _model, _field_name, _g, _s


# -------------------------------------------------------------------
# Per-media-type per-side settings
#
# These six settings (``grid_icon_size_{left,right}``,
# ``focus_mode_{left,right}``, ``panel_pct_{left,right}``) store a
# ``{media_type_id: value}`` dict.
# Their validation rules differ from the simple settings in two ways:
#
# 1. The set of valid media-type IDs is resolved dynamically from
#    :func:`vtscore.media.all_type_ids` (plugins may register more).
# 2. ``panel_pct_*`` raises on out-of-range writes but clamps on reads
#    (the disk value can become invalid via direct file edits).
#
# We keep a small shim layer that delegates per-element validation to
# the Pydantic models defined in :mod:`vtsearch.settings_models`.
# -------------------------------------------------------------------

_GRID_ICON_SIZE_DEFAULT: str = "M"
_FOCUS_MODE_DEFAULTS: dict[str, str] = {"left": "click", "right": "click"}
_PANEL_PX_DEFAULTS: dict[str, int] = {"left": 260, "right": 300}


def _valid_media_types() -> tuple[str, ...]:
    """Return valid media type IDs from the media registry."""
    from vtscore.media import all_type_ids

    return tuple(all_type_ids())


def _make_per_side_setting(  # noqa: C901
    key_base: str,
    defaults: dict[str, Any],
    *,
    normalize=None,
    value_type: str = "str",
    valid_values: tuple[str, ...] | None = None,
):
    """Build ``get_{base}_left``/``_right`` and ``set_{base}_left``/``_right``.

    *normalize* is applied to string values on read and write (e.g.
    :func:`str.upper`). *valid_values* enables enum membership checks
    for string types. For ``value_type="int"`` the value is clamped to
    :data:`VALID_PANEL_PX` on read and raises on out-of-range writes.
    """
    lo_hi = VALID_PANEL_PX if value_type == "int" else None

    def _validate_entry(v, key: str, tid: str | None = None):
        if value_type == "str":
            if normalize is not None and isinstance(v, str):
                v = normalize(v)
            if valid_values is not None and v not in valid_values:
                label = f"{key} value for {tid}" if tid else key
                raise ValueError(f"Invalid {label}: {v!r}")
            return v
        # int (panel_pct): raise on out-of-range writes
        lo, hi = lo_hi  # type: ignore[misc]
        try:
            v = int(round(float(v)))
        except (ValueError, TypeError):
            label = f"{key} value for {tid}" if tid else key
            raise ValueError(f"Invalid {label}: {v!r}") from None
        if not (lo <= v <= hi):
            label = f"{key} value for {tid}" if tid else key
            raise ValueError(f"Invalid {label}: {v} (must be between {lo} and {hi})")
        return v

    def _get_dict(key: str) -> dict[str, Any]:
        side = key[len(key_base) + 1 :]
        default_val = defaults.get(side, next(iter(defaults.values())))
        with _settings_lock:
            raw = _read_value(key)
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

        # Locks are taken inside ``_write_value`` in the canonical
        # order (file_lock → settings_lock); see ``_make_scalar_accessors.setter``.
        _write_value(key, coerced)

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


get_grid_icon_size_left, get_grid_icon_size_right, set_grid_icon_size_left, set_grid_icon_size_right = (
    _make_per_side_setting(
        "grid_icon_size",
        {"left": _GRID_ICON_SIZE_DEFAULT, "right": _GRID_ICON_SIZE_DEFAULT},
        valid_values=VALID_GRID_ICON_SIZES,
        normalize=str.upper,
    )
)

get_focus_mode_left, get_focus_mode_right, set_focus_mode_left, set_focus_mode_right = _make_per_side_setting(
    "focus_mode",
    _FOCUS_MODE_DEFAULTS,
    valid_values=VALID_FOCUS_MODES,
)

get_panel_pct_left, get_panel_pct_right, set_panel_pct_left, set_panel_pct_right = _make_per_side_setting(
    "panel_pct",
    _PANEL_PX_DEFAULTS,
    value_type="int",
)


def get_last_embedder_for_media_type(media_type: str) -> str:
    """Return the user's last picked embedder for *media_type*, or ``""``."""
    if not media_type:
        return ""
    raw = get_last_embedder_per_media_type()
    if isinstance(raw, dict):
        value = raw.get(media_type, "")
        if isinstance(value, str):
            return value
    return ""


def set_last_embedder_for_media_type(media_type: str, embedder: str) -> None:
    """Record *embedder* as the user's last pick for *media_type*."""
    if not media_type or not embedder:
        return
    current = get_last_embedder_per_media_type()
    updated = dict(current) if isinstance(current, dict) else {}
    if updated.get(media_type) == embedder:
        return
    updated[media_type] = embedder
    set_last_embedder_per_media_type(updated)


def set_cli_solo_media_type(value: str | None) -> None:
    """Set the process-level fallback for the per-user ``solo_media_type`` setting.

    Called once from ``app.py`` startup when ``--solo-media-type`` is
    passed on the command line. The value is consulted by
    :func:`get_effective_solo_media_type` for any user who has not
    explicitly set their own ``solo_media_type`` via the settings UI.
    Pass ``None`` (or call from a process where ``--solo-media-type`` was
    not passed) to disable the fallback.
    """
    global _cli_solo_media_type
    if value is not None:
        value = value.strip() or None
    _cli_solo_media_type = value


def get_cli_solo_media_type() -> str | None:
    """Return the process-level CLI fallback (``None`` if unset)."""
    return _cli_solo_media_type


def get_effective_solo_media_type() -> str | None:
    """Return the effective solo mediaType for the current user.

    Resolution order:

    1. The user's explicit choice (``solo_media_type`` when
       ``solo_media_type_explicit`` is True), including an explicit
       ``None`` for "show everything".
    2. The process-level CLI fallback set by
       :func:`set_cli_solo_media_type`.
    3. ``None`` (no streamlining - show every mediaType).

    Returns ``None`` to mean "no solo mode active"; any other return
    value is a mediaType id (e.g. ``"image"``) that the UI should lock
    its pickers to.
    """
    if get_solo_media_type_explicit():  # type: ignore[name-defined]  # autogen'd accessor
        explicit = get_solo_media_type()  # type: ignore[name-defined]  # autogen'd accessor
        # Empty string from JSON drift normalises to None.
        if isinstance(explicit, str) and not explicit.strip():
            return None
        return explicit
    return _cli_solo_media_type


def apply_user_solo_media_type(value: str | None) -> None:
    """Persist *value* as the user's solo mediaType choice and flip ``explicit``.

    Used by the settings PUT route so a single UI change updates both
    fields atomically. ``value=None`` (or an empty string) means "show
    everything"; any other string is validated against the media-type
    registry by the route layer before this is called.
    """
    if isinstance(value, str) and not value.strip():
        value = None
    set_solo_media_type(value)  # type: ignore[name-defined]  # autogen'd accessor
    set_solo_media_type_explicit(True)  # type: ignore[name-defined]  # autogen'd accessor


def set_cli_dataset_max_age_days(value: int | None) -> None:
    """Set the process-level override for the ``dataset_max_age_days`` setting.

    Called once from ``app.py`` startup when ``--dataset-max-age-days`` is
    passed on the command line. The value applies server-wide (every user)
    and is fixed for the process lifetime;
    :func:`get_effective_dataset_max_age_days` returns it in preference to
    the persisted server setting. Pass ``None`` to clear the override
    (reads fall back to the persisted file value).
    """
    global _cli_dataset_max_age_days
    _cli_dataset_max_age_days = value


def get_cli_dataset_max_age_days() -> int | None:
    """Return the process-level CLI override (``None`` if unset)."""
    return _cli_dataset_max_age_days


def get_effective_dataset_max_age_days() -> int | None:
    """Return the dataset max age (in days) in force.

    Resolution order:

    1. The process-level CLI override set by
       :func:`set_cli_dataset_max_age_days` (``--dataset-max-age-days``),
       which applies to every user for the lifetime of the process.
    2. The persisted server-tier setting (``data/settings.json``).

    ``None`` means datasets never expire. This is the value the dataset
    creation pipeline stamps ``expires_at`` from, and the value surfaced
    at ``/api/settings`` so the dashboard's Age-Off column reflects what is
    actually in force.
    """
    if _cli_dataset_max_age_days is not None:
        return _cli_dataset_max_age_days
    return get_dataset_max_age_days()  # type: ignore[name-defined]  # autogen'd accessor


def set_cli_support_email(value: str | None) -> None:
    """Set the process-level override for the ``support_email`` setting.

    Called once from ``app.py`` startup when ``--support-email`` is passed on
    the command line. The value applies server-wide (every user) and is fixed
    for the process lifetime; :func:`get_effective_support_email` returns it in
    preference to the persisted server setting. Pass ``None`` (or an empty /
    whitespace-only string) to clear the override so reads fall back to the
    persisted file value.
    """
    global _cli_support_email
    if value is not None:
        value = value.strip() or None
    _cli_support_email = value


def get_cli_support_email() -> str | None:
    """Return the process-level CLI override (``None`` if unset)."""
    return _cli_support_email


def get_effective_support_email() -> str:
    """Return the "Email us" support address in force.

    Resolution order:

    1. The process-level CLI override set by :func:`set_cli_support_email`
       (``--support-email``), which applies to every user for the lifetime of
       the process.
    2. The persisted server-tier setting (``data/settings.json``), which
       defaults to :data:`~vtsearch.settings_models.DEFAULT_SUPPORT_EMAIL`.

    This is the value surfaced at ``/api/settings`` so the Help modal's
    "Email us" link opens a pre-addressed compose window.
    """
    if _cli_support_email is not None:
        return _cli_support_email
    return get_support_email()  # type: ignore[name-defined]  # autogen'd accessor


def set_cli_semantic_only(value: bool | None) -> None:
    """Set the process-level override for the ``semantic_only`` setting.

    Called once from ``app.py`` startup when ``--semantic-only`` is passed (or
    ``VTSEARCH_SEMANTIC_ONLY`` is set). The value applies server-wide (every
    user) and is fixed for the process lifetime;
    :func:`get_effective_semantic_only` returns it in preference to the
    persisted server setting. Pass ``None`` to clear the override so reads fall
    back to the persisted file value.
    """
    global _cli_semantic_only
    _cli_semantic_only = None if value is None else bool(value)


def get_cli_semantic_only() -> bool | None:
    """Return the process-level CLI override (``None`` if unset)."""
    return _cli_semantic_only


def get_effective_semantic_only() -> bool:
    """Return whether this instance is locked to Semantic embedders.

    Resolution order:

    1. The process-level CLI override set by :func:`set_cli_semantic_only`
       (``--semantic-only`` / ``VTSEARCH_SEMANTIC_ONLY``), which applies to
       every user for the lifetime of the process.
    2. The persisted server-tier setting (``data/settings.json``), which
       defaults to ``False``.

    When true, the prototype Patch Semantic / Structural embedder types are
    hidden from every picker (``GET /api/embedders`` filters them out) and
    rejected by the dataset-load and detector-create routes.
    """
    if _cli_semantic_only is not None:
        return _cli_semantic_only
    return bool(get_semantic_only())  # type: ignore[name-defined]  # autogen'd accessor


def set_cli_solo_embedder(media_type: str, embedder: str | None) -> None:
    """Set or clear a process-level solo-embedder fallback for *media_type*.

    Called from ``app.py`` startup for each ``--solo-embedder TYPE=EMB``
    pair on the command line. Pass ``embedder=None`` (or an empty
    string) to clear an entry. Both arguments are stripped; an empty
    *media_type* is silently ignored (the CLI parser already validates).
    """
    mt = (media_type or "").strip()
    if not mt:
        return
    emb = (embedder or "").strip() if embedder else ""
    if not emb:
        _cli_solo_embedders.pop(mt, None)
        return
    _cli_solo_embedders[mt] = emb


def get_cli_solo_embedders() -> dict[str, str]:
    """Return a copy of the process-level solo-embedder CLI fallbacks."""
    return dict(_cli_solo_embedders)


def get_effective_solo_embedders() -> dict[str, str]:
    """Return the merged ``{media_type: embedder}`` dict for the current user.

    Combines the per-user :func:`get_solo_embedder_per_media_type` map
    (user explicit) with the process-level
    :data:`_cli_solo_embedders` (CLI fallback). User entries win per-key;
    missing user keys fall through to the CLI value. An **empty-string
    value** in the user map is a per-type opt-out sentinel - it removes
    that type from the merged map even if the CLI fallback has a
    value for it. This is the analog of setting ``solo_media_type=null``
    with ``solo_media_type_explicit=True`` to override
    ``--solo-media-type``.

    Validity (does the embedder still exist for this type?) is *not*
    checked here - the frontend resolves it against the live embedder
    registry on its end and falls back to the normal picker for any
    entry that no longer matches. Keeping validation client-side means a
    rename or removal never blocks the settings UI from rendering.
    """
    merged: dict[str, str] = {}
    for mt, emb in _cli_solo_embedders.items():
        if mt and isinstance(emb, str) and emb.strip():
            merged[mt] = emb.strip()
    user_map = get_solo_embedder_per_media_type()  # type: ignore[name-defined]  # autogen'd accessor
    if isinstance(user_map, dict):
        for mt, emb in user_map.items():
            if not isinstance(mt, str) or not mt:
                continue
            if isinstance(emb, str) and emb.strip():
                merged[mt] = emb.strip()
            else:
                # Empty-string sentinel - user explicitly opted out for
                # this type, so drop the CLI fallback too.
                merged.pop(mt, None)
    return merged


def get_effective_solo_embedder(media_type: str) -> str | None:
    """Return the effective solo embedder for *media_type*, or ``None``."""
    if not media_type:
        return None
    return get_effective_solo_embedders().get(media_type)


def apply_user_solo_embedder_per_media_type(value: dict[str, str] | None) -> None:
    """Replace the per-user ``solo_embedder_per_media_type`` map.

    Used by the settings PUT route. ``None`` clears every entry. Keys
    are stripped and skipped if empty. Values are stripped; an
    empty-string value is preserved as a **per-type opt-out sentinel**
    (overrides the CLI fallback for that type - see
    :func:`get_effective_solo_embedders`). The route layer is
    responsible for validating non-empty ``(media_type, embedder)``
    pairs against the live registries before calling this.
    """
    if value is None:
        set_solo_embedder_per_media_type({})  # type: ignore[name-defined]  # autogen'd accessor
        return
    cleaned: dict[str, str] = {}
    for mt, emb in value.items():
        if not isinstance(mt, str) or not mt.strip():
            continue
        if isinstance(emb, str) and emb.strip():
            cleaned[mt.strip()] = emb.strip()
        elif isinstance(emb, str):
            # Empty-string sentinel - preserve as opt-out marker.
            cleaned[mt.strip()] = ""
    set_solo_embedder_per_media_type(cleaned)  # type: ignore[name-defined]  # autogen'd accessor


def get_autofind_detectors() -> list[str]:
    """Return the current user's list of detector names flagged for Auto-Find.

    Each name maps to a JSON file under ``data/detectors/``; scoring resolves
    the labelset's origins, re-embeds, trains an MLP, and applies it to the
    loaded dataset. Per-user (with a read-through to the server file for the
    built-in "default" user; see :func:`_read_value`).
    """
    raw = _read_value("autofind_detectors")
    return list(raw) if isinstance(raw, list) else []


def set_autofind_detectors(value: list[str]) -> None:
    """Set and persist the current user's full Auto-Find detector list."""
    deduped = list(dict.fromkeys(value))  # dedupe, preserve order
    _write_value("autofind_detectors", deduped)


def add_autofind_detector(name: str) -> None:
    """Add a detector name to the current user's Auto-Find list (idempotent).

    Atomic per-user read-modify-write: ``mutate_user`` re-reads the on-disk
    file under the cross-process lock, so a concurrent writer's entry is
    merged rather than clobbered. The ``seed`` captures the effective value
    (including the default-user read-through) for the first write, before the
    user has any entry of its own.
    """
    seed = get_autofind_detectors()

    def _add(cache: dict[str, Any]) -> None:
        base = cache["autofind_detectors"] if "autofind_detectors" in cache else seed
        cache["autofind_detectors"] = list(dict.fromkeys([*base, name]))

    mutate_user(_add)


def remove_autofind_detector(name: str) -> bool:
    """Remove a detector name from the current user's Auto-Find list.

    Returns ``True`` if the name was present (pre-read). Atomic per-user RMW,
    same merge semantics as :func:`add_autofind_detector`.
    """
    seed = get_autofind_detectors()
    found = name in seed

    def _remove(cache: dict[str, Any]) -> None:
        base = cache["autofind_detectors"] if "autofind_detectors" in cache else seed
        cache["autofind_detectors"] = [n for n in base if n != name]

    mutate_user(_remove)
    return found


def is_autofind_detector(name: str) -> bool:
    """Check whether a detector name is in the Auto-Find list."""
    return name in get_autofind_detectors()


# -------------------------------------------------------------------
# Plugin hiding (admin-side picker declutter)
# -------------------------------------------------------------------


def _normalize_hidden_plugins(value: Any) -> dict[str, set[str]]:
    """Coerce arbitrary input to ``{family: {name, ...}}`` form.

    Accepts ``dict[str, Iterable[str]]`` shapes and drops empty entries.
    Non-string keys / names and ``None`` values are silently skipped so a
    corrupt settings file doesn't crash plugin listings.
    """
    out: dict[str, set[str]] = {}
    if not isinstance(value, dict):
        return out
    for family, names in value.items():
        if not isinstance(family, str) or not family:
            continue
        if isinstance(names, (str, bytes)):
            continue
        try:
            members = {n for n in names if isinstance(n, str) and n}
        except TypeError:
            continue
        if members:
            out[family] = members
    return out


def set_cli_hidden_plugins(value: dict[str, Any] | None) -> None:
    """Replace the process-level ``--hide-plugin`` fallback in one shot.

    Called by ``app.py`` after parsing ``--hide-plugin family:name`` flags
    (or by tests that want to seed a known CLI hide list). ``None`` or an
    empty dict clears the fallback.
    """
    global _cli_hidden_plugins
    _cli_hidden_plugins = _normalize_hidden_plugins(value or {})


def add_cli_hidden_plugin(family: str, name: str) -> None:
    """Append ``(family, name)`` to the CLI hide list (idempotent).

    Used by the ``--hide-plugin`` argparse hook to accumulate one entry
    per flag occurrence.
    """
    if not family or not name:
        return
    _cli_hidden_plugins.setdefault(family, set()).add(name)


def get_cli_hidden_plugins() -> dict[str, set[str]]:
    """Return a defensive copy of the CLI-only hide map."""
    return {family: set(names) for family, names in _cli_hidden_plugins.items()}


def get_effective_hidden_plugins() -> dict[str, set[str]]:
    """Return the merged ``{family: {name, ...}}`` hide map.

    Combines the persisted ``hidden_plugins`` server setting with the
    process-level ``--hide-plugin`` fallback. The union semantics matter:
    a plugin is hidden if either source asks for it, so the CLI flag can
    only add hides (never un-hide something the settings file marks
    hidden).
    """
    persisted = _normalize_hidden_plugins(get_hidden_plugins())  # type: ignore[name-defined]
    merged: dict[str, set[str]] = {family: set(names) for family, names in persisted.items()}
    for family, names in _cli_hidden_plugins.items():
        merged.setdefault(family, set()).update(names)
    return merged


def is_plugin_hidden(family: str, name: str) -> bool:
    """Return True if *name* is hidden in *family* by the effective hide map.

    Static ``hidden_from_picker=True`` on a plugin class is **not**
    consulted here - that flag is already serialised on the plugin's
    ``to_dict()`` and filtered client-side. This function only answers
    "is the admin hiding this in this deployment?"
    """
    hidden = _cli_hidden_plugins.get(family)
    if hidden and name in hidden:
        return True
    persisted = get_hidden_plugins()  # type: ignore[name-defined]
    if not isinstance(persisted, dict):
        return False
    family_names = persisted.get(family)
    if not isinstance(family_names, (list, set, tuple)):
        return False
    return name in family_names


def filter_visible_plugins(family: str, plugins: Iterable[Any], *, id_attr: str = "name") -> list[Any]:
    """Drop plugins whose id-attribute is hidden in *family*.

    The single chokepoint for the listing routes. *id_attr* names the
    attribute that carries the plugin's registry id - ``"name"`` for the
    PluginBase families, ``"type_id"`` for :class:`MediaType` (whose
    registry key is the type id, not its human-readable name).
    """
    hidden = get_effective_hidden_plugins().get(family)
    if not hidden:
        return list(plugins)
    return [p for p in plugins if getattr(p, id_attr, None) not in hidden]


def filter_visible_plugin_dicts(
    family: str, plugin_dicts: Iterable[dict[str, Any]], *, id_key: str = "name"
) -> list[dict[str, Any]]:
    """Same as :func:`filter_visible_plugins` but for pre-serialised dicts.

    Reads ``dict[id_key]`` to compare against the hide set. Entries
    without that key are kept.
    """
    hidden = get_effective_hidden_plugins().get(family)
    if not hidden:
        return list(plugin_dicts)
    return [d for d in plugin_dicts if d.get(id_key) not in hidden]


def filter_semantic_only_embedder_dicts(embedder_dicts: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop patch / structural embedders when the instance is Semantic-locked.

    The single chokepoint for ``GET /api/embedders``: with
    :func:`get_effective_semantic_only` true, an embedder that advertises
    ``supports_patch_regions`` or ``supports_geometric_verification`` is
    withheld from the listing, so every picker fed by that endpoint (the Add
    Dataset "Advanced" block's Embedder / Region embedder / Instance embedder
    selects, the Import Defaults tab) offers Semantic embedders only.

    A no-op when the lock is off, so the ordinary deployment pays one boolean.
    """
    dicts = list(embedder_dicts)
    if not get_effective_semantic_only():
        return dicts
    return [d for d in dicts if not d.get("supports_patch_regions") and not d.get("supports_geometric_verification")]


# -------------------------------------------------------------------
# Directory path settings (server tier)
# -------------------------------------------------------------------


def _get_dir(key: str) -> Path:
    """Return a server-tier directory path setting as a :class:`~pathlib.Path`."""
    with _settings_lock:
        raw = _ensure_server_loaded().get(key, _server_defaults()[key])
    return Path(raw)


def _set_dir(key: str, value: str | Path) -> None:
    """Persist a server-tier directory path setting."""
    _ensure_server_loaded()
    coerced = str(value)
    _mutate_server_locked(lambda c: c.__setitem__(key, coerced))


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
    Does not affect per-user settings files - those still live under
    :func:`vtsearch.auth.get_user_data_dir`.
    """
    global SETTINGS_PATH
    with _settings_lock:
        SETTINGS_PATH = Path(path)
        _store.invalidate_server_cache()


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
    _store.reset()


# -------------------------------------------------------------------
# Settings source (bidirectional sync, per-user)
# -------------------------------------------------------------------


def _server_default_settings_source() -> dict[str, Any] | None:
    """Return the deployment-wide ``default_settings_source`` server setting.

    A thin wrapper over the autogenerated server-tier accessor, injected into
    :class:`~vtsearch.settings_store.UserSettingsStore` so its precedence
    resolver can read the deployment default without importing this module.
    """
    return get_default_settings_source()  # type: ignore[name-defined]  # autogen'd accessor


def get_settings_source_config() -> dict[str, Any] | None:
    """Return the active user's effective settings source config, or ``None``.

    Resolves the per-user / deployment-default precedence via the store's
    single resolver (see
    :meth:`~vtsearch.settings_store.UserSettingsStore.resolve_settings_source`):
    a user's own ``settings_source`` wins, an explicit ``{"source_name":
    "none"}`` opts out, and otherwise the user inherits the deployment-wide
    ``default_settings_source``. Config shape::

        {
            "source_name": "server_json_file",
            "field_values": {"filepath": "data/{username}.settings.json"}
        }
    """
    return get_settings_source_config_resolved()[0]


def get_settings_source_config_resolved() -> tuple[dict[str, Any] | None, bool]:
    """Return ``(config, inherited)`` for the active user's effective source.

    ``inherited`` is ``True`` when the effective config comes from the
    deployment-wide ``default_settings_source`` rather than the user's own
    ``settings_source`` key. When no source is active (user opted out, or no
    default configured) the config is ``None`` and ``inherited`` is ``False``.
    """
    from vtsearch.auth import get_current_user

    username = get_current_user()
    _ensure_user_loaded(username)
    return _store.resolve_settings_source(username)


def set_settings_source_config(config: dict[str, Any] | None) -> None:
    """Set or clear the active user's settings source config.

    ``config`` semantics:

    * ``None`` clears the user's explicit key so they *inherit* the
      deployment-wide ``default_settings_source`` again (or, if none is
      configured, end up with no source).
    * ``{"source_name": "none"}`` records an explicit opt-out: the user has no
      source even when a deployment default exists.
    * Any other config binds the user to that named source.
    """
    from vtsearch.auth import get_current_user

    username = get_current_user()
    _ensure_user_loaded(username)

    def _apply(cache: dict[str, Any]) -> None:
        if config is None:
            cache.pop("settings_source", None)
        else:
            cache["settings_source"] = config

    sync_data = _mutate_user_locked(username, _apply)

    # Resolve the *effective* source after the write: clearing the explicit
    # key may leave the user inheriting the deployment default, and ``"none"``
    # resolves to no source at all.
    resolved, _inherited = _store.resolve_settings_source(username)
    if config is None or resolved is None:
        # Cleared to inherit (let the next read pull whatever is now
        # resolved) or opted out (no active source): forget stale sync
        # bookkeeping pointing at the previously-configured target.
        _store.drop_sync_state(username)
        return
    if sync_data is not None:
        # ``_sync_to_source`` exports the full local dict to the resolved
        # source location and stamps the user as freshly synced (so the next
        # ``_ensure_user_loaded`` doesn't immediately pull what we just
        # pushed).
        _sync_to_source(username, sync_data)


# Map of setting key → setter function (generated dynamically).
_SETTER_MAP: dict[str, Any] | None = None


def _get_setter_map() -> dict:
    """Build a map of setting-key → setter-function by introspecting this module.

    Restricted to actual settings-schema keys (the server + user tier
    pydantic fields).  The module also exposes process-level ``set_*``
    helpers - ``set_settings_path``, ``set_user_data_dir_override``, the
    CLI knobs - and an unrestricted scan would let an imported or synced
    settings dict (file content, not trusted code) invoke them, e.g.
    repointing the server settings file or every user's data dir for the
    rest of the process lifetime.
    """
    global _SETTER_MAP
    if _SETTER_MAP is not None:
        return _SETTER_MAP
    import vtsearch.settings as _self

    schema_keys = set(_all_defaults())
    _SETTER_MAP = {}
    for attr_name in dir(_self):
        key = attr_name[4:]
        if attr_name.startswith("set_") and key in schema_keys and callable(getattr(_self, attr_name)):
            _SETTER_MAP[key] = getattr(_self, attr_name)
    return _SETTER_MAP


def _apply_settings(imported: dict, skip_keys: set[str] | None = None) -> None:
    """Apply a dict of settings via this module's ``set_*`` functions.

    Unknown keys or values that fail validation are silently skipped.
    Keys in *skip_keys* are also skipped - used by the auto re-sync
    path so an upstream value doesn't silently overwrite a key the
    user has just edited locally.  Used by :func:`sync_from_settings_source`
    and by the settings-import route in ``routes/settings/io.py``.
    """
    setter_map = _get_setter_map()
    skip = skip_keys or set()
    for key, value in imported.items():
        if key in skip:
            continue
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

    - Automatically from :func:`_ensure_user_loaded` whenever the
      source's :meth:`SettingsSource.peek_version` token differs from
      the cached one (or no successful sync has happened yet this
      process lifetime).
    - Manually via ``POST /api/settings-sources/sync``.

    This (explicit) path **ignores the local ``dirty_keys`` set**: the
    user clicked "Sync now" precisely to get the source values, so a
    locally-edited key is overwritten and the dirty marker is cleared.
    """
    cfg = get_settings_source_config()
    if cfg is None:
        return None

    from vtsearch.auth import get_current_user

    return _store.sync_from_source_now(cfg, get_current_user())


def _sync_to_source(username: str, data: dict[str, Any]) -> None:
    """Push *username*'s current settings to their active source (if any).

    Called from :func:`_write_value` / :func:`mutate_user` /
    :func:`set_settings_source_config` after the per-user file is
    written, **outside** the cross-process file lock so a slow sync
    target can't block other settings writes.
    """
    _store.sync_to_source(username, data)


# ---------------------------------------------------------------------------
# The persistence engine
#
# Instantiated last so every injected dependency (path resolvers, the
# ``_apply_settings`` callback, the tier key sets) is already defined. The
# shared mutable containers (``_settings_lock`` / ``_user_caches`` /
# ``_sync_state`` / ``_syncing``) are passed by reference so the store and
# the module-level names mutate one set of objects; the delegators above
# reference ``_store`` lazily (at call time), so its position at the bottom
# of the module is fine.
# ---------------------------------------------------------------------------

_store = UserSettingsStore(
    settings_lock=_settings_lock,
    user_caches=_user_caches,
    sync_state=_sync_state,
    syncing=_syncing,
    server_path=_server_settings_path,
    user_path=_user_settings_path,
    apply_settings=_apply_settings,
    server_default_source=_server_default_settings_source,
    server_keys=_SERVER_KEYS,
    fallback_keys=_DEFAULT_USER_FALLBACK_KEYS,
    exclude_from_source_export=_EXCLUDE_FROM_SOURCE_EXPORT,
    freshness_check_interval=_FRESHNESS_CHECK_INTERVAL,
)
