"""Where each tier's settings live, and reading or writing one key.

Settings are stored in two tiers (see :mod:`vtsearch.settings`): the server file
at :data:`SETTINGS_PATH` and one file per user, :data:`USER_SETTINGS_FILENAME`
under :func:`vtsearch.auth.get_user_data_dir`.  This module owns their paths,
the in-memory caches and the lock over them, and the
:class:`~vtsearch.settings.store.UserSettingsStore` instance (:data:`_store`) that
loads, writes and syncs both.  On top of those sit the per-key read and write
every accessor goes through (:func:`_read_value`, :func:`_write_value`) and the
whole-tier ones (:func:`mutate_user`, :func:`snapshot_user`).

:data:`SETTINGS_PATH` and :data:`_USER_DATA_DIR_OVERRIDE` are rebound at run time
(by :func:`set_settings_path` / :func:`set_user_data_dir_override`, and by
tests), so the package does not re-export them: a copy there would go stale,
and a stub installed on it would be silently ignored.  Read and patch them here.
"""

from __future__ import annotations

import threading
from collections.abc import Container
from pathlib import Path
from typing import Any

from vtscore.config import DATA_DIR
from vtsearch.settings import schema
from vtsearch.settings.schema import (
    _DEFAULT_USER_FALLBACK_KEYS,
    _EXCLUDE_FROM_SOURCE_EXPORT,
    _SERVER_KEYS,
    _sanitize_tier,
)
from vtsearch.settings.store import AppliedSettings, UserSettingsStore, UserSyncState as _UserSyncState

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
# Caches
# ---------------------------------------------------------------------------

# The mutable engine state below lives here (not on the store) because the
# test suite imports these containers by name: the sync-source tests poke
# ``_sync_state`` and ``_user_caches``, and the thread-safety test asserts on
# ``_settings_lock``. No *production* module reads them any more - app code
# that wants a multi-key read of a user's tier calls :func:`snapshot_user`.
# The :class:`~vtsearch.settings.store.UserSettingsStore` instance (created at
# the bottom of this module) is handed these same objects by reference, so
# both views mutate one set of containers. The reassignable scalars
# (``server_cache``, ``legacy_migrated``) and the per-user sync locks live
# solely on the store.

# Per-user caches keyed by username.
_user_caches: dict[str, dict[str, Any]] = {}

# Per-user sync bookkeeping, which tells "never synced" from "synced ages
# ago" from "tried once and failed."
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
# The package's read/write paths call ``_ensure_server_loaded`` /
# ``_ensure_user_loaded`` by name. The actual two-tier cache + sync machinery
# lives on ``_store`` (see :mod:`vtsearch.settings.store`).
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


def _mutate_user_locked(username: str, mutator, *, keys=None) -> dict[str, Any] | None:
    """Atomically read-modify-write *username*'s per-user settings file.

    Returns a post-mutation snapshot for ``_sync_to_source`` (invoked
    outside the file lock), or ``None`` when a sync-from-source import is
    in progress for this user (or the write was skipped to preserve a
    locally-edited key).

    *keys* names the top-level keys the mutation touches (literally, or as
    a callable resolving them from the post-mutation dict); the store marks
    them dirty under the file lock so a concurrent auto re-sync can't
    overwrite them in the gap between the write and the marking.
    """
    return _store.mutate_user_locked(username, mutator, keys=keys)


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
    # ``mutate_user`` can change arbitrary top-level keys, so the dirty set
    # is resolved from the post-mutation dict: every exportable key is
    # marked (under the file lock) so an auto re-sync between now and the
    # ``_sync_to_source`` push doesn't clobber the mutation.
    sync_data = _mutate_user_locked(username, mutator, keys=lambda cache: list(cache))
    if sync_data is not None:
        _sync_to_source(username, sync_data)


# ---------------------------------------------------------------------------
# Low-level value get/set used by every accessor
# ---------------------------------------------------------------------------


def _read_value(key: str) -> Any:
    """Return the raw stored value for *key* (or its default).

    Routes to the server tier or the current user's tier based on *key*.
    The caller is responsible for casting/coercion.

    Takes ``_settings_lock`` itself, and *only* around the cache-dict
    reads. The ``_ensure_*_loaded`` calls run outside it because they take
    the cross-process ``file_lock`` on a cold cache (and the per-user sync
    RLock plus, via ``_apply_settings``'s setters, ``file_lock`` again on a
    due sync), and the canonical order is file_lock → settings_lock.
    Callers must therefore **not** hold ``_settings_lock`` when calling
    this - doing so inverts the order and deadlocks against any concurrent
    writer (see ``ensure_server_loaded``'s docstring). Keeping the lock out
    of the ensure calls also means a slow ``source.load()`` no longer
    stalls every other settings read.

    The caches are re-read from the store *after* the ensure calls rather
    than using their return values, because a concurrent writer replaces
    the cache dict wholesale (``mutate_*_locked``) - the returned dict can
    be the pre-write one.
    """
    if key in _SERVER_KEYS:
        _ensure_server_loaded()
        with _settings_lock:
            server_cache = _store.server_cache or {}
            if key in server_cache:
                return server_cache[key]
        return schema._server_defaults()[key]
    from vtsearch.auth import get_current_user

    username = get_current_user()
    _ensure_user_loaded(username)
    # Default-user read-through: the single-user GUI and the CLI ``--settings``
    # flat file carry these AutoFind keys in the server settings file. Honor
    # them for the built-in "default" user when not set in its own file, so
    # those workflows keep working without making the setting truly server-wide.
    # ``_ensure_user_loaded`` already loaded the server tier, so this is a
    # cache hit - it stays outside the lock anyway so the cold path can never
    # invert the order.
    fallback_to_server = username == "default" and key in _DEFAULT_USER_FALLBACK_KEYS
    if fallback_to_server:
        _ensure_server_loaded()
    with _settings_lock:
        user_cache = _user_caches.get(username, {})
        if key in user_cache:
            return user_cache[key]
        if fallback_to_server:
            server_cache = _store.server_cache or {}
            if key in server_cache:
                return server_cache[key]
    return schema._user_defaults().get(key)


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
    # ``keys=[key]`` makes the store mark this key dirty *under the file
    # lock*, so an auto re-sync running between now and the
    # ``_sync_to_source`` push leaves it alone - the push below clears the
    # dirty flag again once it succeeds.  On the import path the same
    # argument makes the store skip the write entirely when the key is
    # already dirty.
    sync_data = _mutate_user_locked(username, lambda c: c.__setitem__(key, value), keys=[key])
    if sync_data is not None:
        _sync_to_source(username, sync_data)


def snapshot_user(username: str) -> dict[str, Any]:
    """Return a consistent copy of *username*'s raw per-user settings cache.

    This is the public read counterpart to :func:`mutate_user`: it loads the
    user's tier if cold, then copies the cache under ``_settings_lock`` so the
    caller walks a view that no concurrent writer can tear.  Callers that need
    to read several related keys at once (rather than one key via an accessor)
    should use this instead of reaching for ``_user_caches`` directly.

    The returned dict holds only what is *stored* for the user - no defaults
    are filled in, and no server-tier read-through is applied.  Use
    :func:`~vtsearch.settings.get_user_settings` when you want the defaults merged in.

    ``_ensure_user_loaded`` runs outside ``_settings_lock`` because a cold
    cache takes the cross-process file lock (and, via a due sync,
    ``_apply_settings``'s setters take it again); the canonical order is
    file_lock -> settings_lock, so holding the settings lock across it would
    deadlock against a concurrent writer.
    """
    _ensure_user_loaded(username)
    with _settings_lock:
        return dict(_user_caches.get(username, {}))


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


def _sync_to_source(username: str, data: dict[str, Any]) -> None:
    """Push *username*'s current settings to their active source (if any).

    Called from :func:`_write_value` / :func:`mutate_user` /
    :func:`~vtsearch.settings.sync.set_settings_source_config` after the per-user file is
    written, **outside** the cross-process file lock so a slow sync
    target can't block other settings writes.
    """
    _store.sync_to_source(username, data)


# ---------------------------------------------------------------------------
# The engine's calls back up the package
#
# The store applies a pulled settings dict through the per-user write rule,
# carries a training-setting change on to the loaded detectors, and resolves a
# user's source against the deployment-wide ``default_settings_source``. All
# three are :mod:`vtsearch.settings.sync`'s, which sits above this module and
# imports it, so these look them up when the store calls.
# ---------------------------------------------------------------------------


def _apply_settings(imported: dict[str, Any], skip_keys: Container[str] | None) -> AppliedSettings:
    from vtsearch.settings import sync  # noqa: PLC0415

    return sync._apply_settings(imported, skip_keys)


def _refresh_loaded_detectors(applied: AppliedSettings) -> None:
    from vtsearch.settings import sync  # noqa: PLC0415

    sync._refresh_loaded_detectors(applied)


def _server_default_settings_source() -> dict[str, Any] | None:
    from vtsearch.settings import sync  # noqa: PLC0415

    return sync._server_default_settings_source()


# ---------------------------------------------------------------------------
# The persistence engine
#
# Instantiated last so every injected dependency (path resolvers, the
# ``_apply_settings`` callback and its ``_refresh_loaded_detectors`` follow-up,
# the tier key sets) is already defined. The
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
    after_apply=_refresh_loaded_detectors,
    server_default_source=_server_default_settings_source,
    sanitize=_sanitize_tier,
    exclude_from_source_export=_EXCLUDE_FROM_SOURCE_EXPORT,
    freshness_check_interval=_FRESHNESS_CHECK_INTERVAL,
)
