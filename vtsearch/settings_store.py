"""Two-tier settings persistence engine for :mod:`vtsearch.settings`.

This module holds the lock-ordering-sensitive machinery that the settings
accessors in :mod:`vtsearch.settings` delegate to: cross-process file
locking, the server/per-user in-memory caches, the one-shot legacy
migration, and the bidirectional sync-from/to-source state machine.

The split keeps :mod:`vtsearch.settings` focused on the *schema* (the
Pydantic-driven ``get_<key>`` / ``set_<key>`` surface, tier routing, CLI
fallbacks, effective-value resolvers) while this module owns the *engine*.

Why a class with injected dependencies rather than a second flat module:
several mutable containers (``_settings_lock``, ``_user_caches``,
``_sync_state``) are imported by name from :mod:`vtsearch.settings` by other
modules (``vtsearch.achievements``, the sync-source tests), so they must
stay as module globals there. The store receives those same objects *by
reference* in its constructor, so both views mutate one set of containers.
The reassignable scalars (``server_cache``, ``legacy_migrated``) and the
per-user sync locks live solely on the store. Path resolution and the
"apply an imported settings dict" callback are injected so this module
never imports :mod:`vtsearch.settings` (no import cycle).

The canonical lock order is **file lock → settings lock**: file I/O runs
under the cross-process :func:`file_lock` only, and the in-process
``settings_lock`` is taken briefly afterwards just to swap the in-memory
cache, so a slow fsync can't stall unrelated settings reads.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import fcntl as _fcntl  # POSIX-only; falls back to in-process locking on Windows.
except ImportError:  # pragma: no cover - Windows
    _fcntl = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File I/O primitives (path-keyed, no settings state)
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
    """Write *data* to *path* via a per-writer temp file + rename.

    The temp filename embeds PID + a UUID so two processes writing to the
    same target can't truncate each other's in-flight temp file. The
    final ``os.replace`` is atomic on POSIX, so a concurrent reader
    always sees either the pre- or post-write content, never a partial
    write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


# Per-path in-process locks. Used in two ways:
# 1. As a fallback when ``fcntl`` is unavailable (Windows).
# 2. Held in addition to the cross-process flock so that, within a single
#    process, multiple threads serialise on the same path without
#    repeatedly re-entering the kernel.
_path_locks: dict[str, threading.Lock] = {}
_path_locks_guard = threading.Lock()


def _path_lock_for(path: Path) -> threading.Lock:
    key = str(path.resolve()) if path.exists() else str(path)
    with _path_locks_guard:
        lock = _path_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _path_locks[key] = lock
        return lock


@contextlib.contextmanager
def file_lock(path: Path):
    """Acquire an exclusive cross-process lock for *path*.

    The lock is taken on a sibling ``<path>.lock`` file rather than on
    the data file itself, because ``_atomic_write`` replaces the data
    file's inode via ``os.replace`` - any fd held against the old inode
    would be useless. The sibling lock file's inode is stable.

    The lock is released automatically if the process exits (POSIX
    flock semantics), so there are no zombie locks after crashes.

    On Windows (``fcntl`` unavailable) the in-process lock alone is
    used; cross-process protection degrades silently. VTSearch is
    deployed on Linux containers, so this only affects the rare
    Windows-dev case where multiple processes shouldn't be writing
    the same settings file anyway.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    in_proc = _path_lock_for(path)
    in_proc.acquire()
    fd: int | None = None
    fcntl_mod = _fcntl  # snapshot so the narrowed binding survives the yield
    if fcntl_mod is not None:
        lock_path = path.with_name(path.name + ".lock")
        try:
            fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT, 0o600)
            fcntl_mod.flock(fd, fcntl_mod.LOCK_EX)
        except OSError as exc:
            logger.warning("Could not acquire file lock on %s: %s", lock_path, exc)
            if fd is not None:
                with contextlib.suppress(OSError):
                    os.close(fd)
                fd = None
    try:
        yield
    finally:
        if fd is not None and fcntl_mod is not None:
            try:
                fcntl_mod.flock(fd, fcntl_mod.LOCK_UN)
            finally:
                with contextlib.suppress(OSError):
                    os.close(fd)
        in_proc.release()


# ---------------------------------------------------------------------------
# Per-user sync bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class UserSyncState:
    """Per-user sync-from-source bookkeeping.

    - ``last_version``: opaque token from :meth:`SettingsSource.peek_version`
      stashed at the last successful sync.  Compared on every read to
      decide whether a re-sync is due.  ``None`` if the source can't
      cheaply check freshness.
    - ``last_check_monotonic``: :func:`time.monotonic` of the last peek.
      Used to rate-limit ``peek_version`` calls so a hot read path doesn't
      stat the source file on every ``get_volume()``.
    - ``last_sync_succeeded``: ``False`` means the user has never been
      successfully synced this process lifetime (or the last attempt
      failed).  A transient source failure no longer permanently locks
      the user out of sync - the slow-path retries (rate-limited) until
      it succeeds.
    - ``dirty_keys``: keys the user has set locally since the last
      successful :func:`_sync_to_source`.  An auto re-sync (from a
      version-bump on the source) skips these keys so a freshly clicked
      local toggle isn't silently overwritten by an upstream value.
      Cleared whenever ``_sync_to_source`` succeeds (because the source
      then matches local) or when a manual :func:`sync_from_settings_source`
      runs (explicit user pull).
    """

    last_version: Any = None
    last_check_monotonic: float = 0.0
    last_sync_succeeded: bool = False
    dirty_keys: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


class UserSettingsStore:
    """Two-tier (server + per-user) settings persistence engine.

    Owns the lock-ordering-sensitive load / mutate / sync machinery.  The
    mutable containers passed in (``settings_lock``, ``user_caches``,
    ``sync_state``, ``syncing``) are shared by reference with
    :mod:`vtsearch.settings` so external importers see one set of objects.
    """

    def __init__(
        self,
        *,
        settings_lock: threading.RLock,
        user_caches: dict[str, dict[str, Any]],
        sync_state: dict[str, UserSyncState],
        syncing: set[str],
        server_path: Callable[[], Path],
        user_path: Callable[[str], Path],
        apply_settings: Callable[[dict[str, Any], set[str] | None], None],
        server_keys: frozenset[str],
        fallback_keys: frozenset[str],
        exclude_from_source_export: set[str],
        freshness_check_interval: float = 1.0,
    ) -> None:
        # Shared-by-reference containers (also module globals in vtsearch.settings).
        self._lock = settings_lock
        self._user_caches = user_caches
        self._sync_state = sync_state
        self._syncing = syncing

        # Injected schema/policy + path resolution + apply callback.
        self._server_path = server_path
        self._user_path = user_path
        self._apply_settings = apply_settings
        self._server_keys = server_keys
        self._fallback_keys = fallback_keys
        self._exclude_from_source_export = exclude_from_source_export
        self._freshness_check_interval = freshness_check_interval

        # Store-only state (no external references).
        self._server_cache: dict[str, Any] | None = None
        self._legacy_migrated: bool = False
        self._user_sync_locks: dict[str, threading.RLock] = {}
        self._user_sync_locks_guard = threading.Lock()

    # -- introspection used by the settings module's read paths ----------

    @property
    def server_cache(self) -> dict[str, Any] | None:
        """The server-tier cache (``None`` until first load)."""
        return self._server_cache

    def per_user_sync_lock(self, username: str) -> threading.RLock:
        """Return (creating if needed) the sync RLock for *username*."""
        with self._user_sync_locks_guard:
            lock = self._user_sync_locks.get(username)
            if lock is None:
                lock = threading.RLock()
                self._user_sync_locks[username] = lock
            return lock

    # -- cache loaders ---------------------------------------------------

    def ensure_server_loaded(self) -> dict[str, Any]:
        """Load the server-tier cache on first access and migrate legacy keys."""
        with self._lock:
            if self._server_cache is None:
                self._server_cache = _load_path(self._server_path())
                self._maybe_migrate_legacy_settings_locked()
            return self._server_cache

    def ensure_user_loaded(self, username: str) -> dict[str, Any]:
        """Load *username*'s per-user cache on first access and reconcile with the source.

        Fast path: take ``settings_lock`` briefly, hydrate the local cache
        if needed, and return immediately when no sync work is due (no
        source configured, freshness window still valid, or we're already
        inside an import for this user).

        Slow path: take the per-user sync RLock so the actual ``peek_version``
        + ``load`` + ``_apply_settings`` work happens serially for the same
        user, and concurrent readers see the post-sync cache when they
        acquire the lock.  Without this lock the previous design had a
        TOCTOU race: thread A set the "synced" marker before running the
        sync, so a concurrent thread B saw the marker and returned the
        pre-sync local cache.
        """
        with self._lock:
            cache = self._user_caches.get(username)
            if cache is None:
                # Make sure server tier is loaded (and legacy migration has run)
                # before we materialise a fresh user cache, otherwise the legacy
                # migration step might not see this user yet.
                self.ensure_server_loaded()
                cache = self._user_caches.get(username)
                if cache is None:
                    cache = _load_path(self._user_path(username))
                    self._user_caches[username] = cache
            # Re-entrance guard: a setter inside ``_apply_settings`` re-enters
            # this function while the outer call holds the sync lock.  Skip
            # the sync path so we don't recurse or fire another peek probe.
            if username in self._syncing:
                return cache
            cfg = cache.get("settings_source")
            has_source = isinstance(cfg, dict) and bool(cfg.get("source_name"))
            if not has_source:
                return cache
            state = self._sync_state.get(username)
            if state is not None and state.last_sync_succeeded:
                now = time.monotonic()
                if now - state.last_check_monotonic < self._freshness_check_interval:
                    # Hot path: a recent successful sync exists and we're
                    # inside the rate-limit window.  No probe, no lock.
                    return cache

        sync_lock = self.per_user_sync_lock(username)
        with sync_lock:
            if self._needs_sync_from_source(username):
                self._run_sync_from_source(username)

        return self._user_caches[username]

    def _needs_sync_from_source(self, username: str) -> bool:
        """Decide whether a sync-from-source pass is due for *username*.

        Called with the per-user sync lock held (so the decision and the
        subsequent sync are atomic with respect to other threads on the
        same user).  A sync is due when:

        1. We've never attempted one this process lifetime, OR
        2. The last attempt failed and the rate-limit window has elapsed
           (so a transient source outage no longer permanently locks
           the user out of sync - old ``_synced_users`` did exactly that), OR
        3. A previous attempt succeeded but the source's
           :meth:`SettingsSource.peek_version` token has changed since
           (source file rewritten by another process, hand-edited, etc.).

        Refreshes ``last_check_monotonic`` on the no-sync paths so the
        freshness probe is rate-limited even when state is being mutated.
        """
        with self._lock:
            state = self._sync_state.get(username)
            cache_cfg = self._user_caches.get(username, {}).get("settings_source")
        if not isinstance(cache_cfg, dict) or not cache_cfg.get("source_name"):
            return False

        now = time.monotonic()

        if state is None or state.last_check_monotonic == 0.0:
            # No state, or state exists only because a setter populated it
            # via ``mark_user_keys_dirty`` before we'd ever talked to the
            # source.  Either way: first sync attempt is due.
            return True

        if not state.last_sync_succeeded:
            # Rate-limited retry after a failure.
            return (now - state.last_check_monotonic) >= self._freshness_check_interval

        # Last attempt succeeded.  If we're still inside the freshness
        # window, skip (the fast path in ``ensure_user_loaded`` usually
        # handles this; keep the check here in case the slow path was
        # entered via a different code route).
        if (now - state.last_check_monotonic) < self._freshness_check_interval:
            return False

        # Window elapsed - cheap probe to detect upstream changes.
        from vtsearch.settings_io.sources import get_settings_source

        source = get_settings_source(cache_cfg["source_name"])
        if source is None:
            return False
        field_values = cache_cfg.get("field_values", {})
        try:
            current_version = source.peek_version(field_values)
        except Exception:
            # Transient peek failure - back off until next window, keep
            # serving the local cache.
            with self._lock:
                s = self._sync_state.get(username)
                if s is not None:
                    s.last_check_monotonic = now
            return False

        if current_version is None or current_version == state.last_version:
            # Source can't cheaply check, or unchanged since last sync.
            with self._lock:
                s = self._sync_state.get(username)
                if s is not None:
                    s.last_check_monotonic = now
            return False

        return True

    def _run_sync_from_source(self, username: str) -> None:
        """Pull settings from *username*'s configured source and apply them.

        Called with the per-user sync lock held.  Respects ``dirty_keys``:
        keys the user has set locally since the last successful
        ``_sync_to_source`` are skipped so a clicked-toggle isn't silently
        overwritten by an upstream value.

        On success: stash the new ``peek_version`` token, mark the user as
        successfully synced, and refresh the freshness-check timestamp.
        On failure: log the error and refresh only the check timestamp
        (``last_sync_succeeded`` stays ``False``) so the slow path retries
        once the rate-limit window elapses.
        """
        with self._lock:
            cache_cfg = self._user_caches.get(username, {}).get("settings_source")
            state = self._sync_state.setdefault(username, UserSyncState())
            dirty_snapshot = set(state.dirty_keys)
        if not isinstance(cache_cfg, dict) or not cache_cfg.get("source_name"):
            return

        from vtsearch.settings_io.sources import get_settings_source

        source = get_settings_source(cache_cfg["source_name"])
        if source is None:
            logger.warning("Unknown settings source: %s", cache_cfg["source_name"])
            return

        field_values = cache_cfg.get("field_values", {})
        new_version: Any = None
        try:
            new_version = source.peek_version(field_values)
        except Exception:
            new_version = None

        try:
            imported = source.load(field_values)
        except Exception as exc:
            logger.exception("Failed to load from settings source for %s: %s", username, exc)
            with self._lock:
                state = self._sync_state.setdefault(username, UserSyncState())
                state.last_check_monotonic = time.monotonic()
                # Leave last_sync_succeeded as-is: a transient failure after
                # a previous success must not erase the success flag (the
                # local cache is still valid).
            return

        if imported:
            with self._lock:
                self._syncing.add(username)
            try:
                self._apply_settings(imported, dirty_snapshot)
            finally:
                with self._lock:
                    self._syncing.discard(username)

        with self._lock:
            state = self._sync_state.setdefault(username, UserSyncState())
            state.last_version = new_version
            state.last_check_monotonic = time.monotonic()
            state.last_sync_succeeded = True
            # dirty_keys preserved across an auto re-sync - the user's local
            # edits stay protected until an explicit ``_sync_to_source`` push
            # confirms the source matches local (which clears them) or a
            # manual POST sync clears them on purpose.

    def _maybe_migrate_legacy_settings_locked(self) -> None:
        """Move per-user keys from a legacy ``data/settings.json`` into the
        default user's per-user file (one-shot, idempotent).

        Called from :meth:`ensure_server_loaded` with the settings lock held.
        """
        if self._legacy_migrated:
            return
        self._legacy_migrated = True
        assert self._server_cache is not None
        # ``fallback_keys`` legitimately live in the server file (the
        # default user reads through to them), so they are NOT "orphaned per-user"
        # keys; leave them in place rather than moving them into the default user's
        # file (which would also rewrite a CLI ``--settings`` file under the user).
        legacy_user_entries = {
            k: v for k, v in self._server_cache.items() if k not in self._server_keys and k not in self._fallback_keys
        }
        if not legacy_user_entries:
            return

        # Migrate into the "default" user's file. The default user is the one
        # the single-user provider returns, and is also the safe target for
        # multi-user upgrades (admins can copy it into other users' files).
        default_user = "default"
        user_path = self._user_path(default_user)
        if user_path.exists():
            existing = _load_path(user_path)
            # Existing per-user values win - never clobber a real user file.
            merged: dict[str, Any] = {**legacy_user_entries, **existing}
        else:
            merged = dict(legacy_user_entries)
        try:
            _atomic_write(user_path, merged)
        except Exception as exc:
            logger.warning("Legacy settings migration to %s failed: %s", user_path, exc)
            return

        # Build the server-tier shape first; a failure here must not pop the
        # in-memory cache, or server_cache and disk would silently diverge. Keep
        # the default-user fallback keys in the server file (they are read through
        # there, not migrated out).
        new_server = {k: v for k, v in self._server_cache.items() if k in self._server_keys or k in self._fallback_keys}
        try:
            _atomic_write(self._server_path(), new_server)
        except Exception as exc:
            logger.warning("Failed to rewrite server settings after legacy migration: %s", exc)
            return
        for k in list(self._server_cache.keys()):
            if k not in self._server_keys and k not in self._fallback_keys:
                self._server_cache.pop(k, None)

        # Refresh the default user's cache if it was already materialised
        # (unlikely, since this runs from ensure_server_loaded, but safe).
        self._user_caches[default_user] = _load_path(user_path)
        logger.info(
            "Migrated %d legacy per-user setting(s) from %s into %s",
            len(legacy_user_entries),
            self._server_path(),
            user_path,
        )

    # -- save helpers ----------------------------------------------------

    def mutate_server_locked(self, mutator: Callable[[dict[str, Any]], None]) -> None:
        """Apply *mutator* to a fresh-from-disk server cache, atomically.

        Acquires the cross-process file lock, re-reads ``data/settings.json``
        so any changes a sibling process made since this process loaded are
        picked up, runs ``mutator(cache)`` to mutate the dict in place, then
        atomically writes the result back. This is the only correct way to
        mutate a multi-writer settings file from a Python process.

        The legacy migration is left to ``ensure_server_loaded`` and never
        fires from inside the lock - by the time any setter runs the cache
        has been loaded at least once.

        File I/O (``_load_path``, ``_atomic_write``) runs under the
        cross-process file lock only; ``settings_lock`` is acquired briefly
        at the end just to swap the in-memory cache, so a slow local fsync
        (NFS, full disk, hung disk controller) can't stall unrelated
        settings reads - see H29 in ``docs/plans/logical-bug-audit.md``.
        """
        path = self._server_path()
        with file_lock(path):
            fresh = _load_path(path)
            mutator(fresh)
            _atomic_write(path, fresh)
            with self._lock:
                self._server_cache = fresh

    def mutate_user_locked(self, username: str, mutator: Callable[[dict[str, Any]], None]) -> dict[str, Any] | None:
        """Apply *mutator* to a fresh-from-disk per-user cache, atomically.

        Returns a snapshot of the cache after the mutation, intended for
        ``sync_to_source`` which is invoked **outside** the file lock so a
        slow sync target (NFS, webhook) can't block other settings writes.
        Returns ``None`` if the cache should not be synced (i.e. we are
        currently importing from the source).

        Like :meth:`mutate_server_locked`, file I/O runs under the
        cross-process file lock only; ``settings_lock`` is acquired briefly
        at the end just to swap the in-memory cache and read the ``syncing``
        flag.
        """
        path = self._user_path(username)
        with file_lock(path):
            fresh = _load_path(path)
            mutator(fresh)
            _atomic_write(path, fresh)
            with self._lock:
                self._user_caches[username] = fresh
                if username in self._syncing:
                    return None
                return dict(fresh)

    def mark_user_keys_dirty(self, username: str, keys) -> None:
        """Add *keys* to the user's ``dirty_keys`` set."""
        if not keys:
            return
        with self._lock:
            state = self._sync_state.setdefault(username, UserSyncState())
            state.dirty_keys.update(keys)

    # -- sync to source --------------------------------------------------

    def sync_to_source(self, username: str, data: dict[str, Any]) -> None:
        """Push *username*'s current settings to their active source (if any).

        Called from ``_write_value`` / ``mutate_user`` /
        ``set_settings_source_config`` after the per-user file is
        written, **outside** the cross-process file lock so a slow sync
        target can't block other settings writes.  Strips the
        ``settings_source`` key itself to avoid circular config.

        On successful save the user is stamped as freshly synced
        (``last_version`` refreshed from the post-save ``peek_version``,
        ``dirty_keys`` cleared) - source now matches local, so the next
        :meth:`ensure_user_loaded` short-circuits via the fast path
        instead of pulling back the values we just pushed.
        """
        cfg = data.get("settings_source")
        if not isinstance(cfg, dict) or not cfg.get("source_name"):
            return

        from vtsearch.settings_io.sources import get_settings_source

        source = get_settings_source(cfg["source_name"])
        if source is None:
            return

        field_values = cfg.get("field_values", {})
        export_data = {k: v for k, v in data.items() if k not in self._exclude_from_source_export}

        try:
            source.save(export_data, field_values)
        except Exception as exc:
            logger.exception("Failed to sync settings to source for %s: %s", username, exc)
            return

        # Source now matches local for every exported key - clear the dirty
        # set and refresh the version token so an auto re-sync doesn't fire
        # for the change we just exported.
        new_version: Any = None
        try:
            new_version = source.peek_version(field_values)
        except Exception:
            new_version = None
        with self._lock:
            self._sync_state[username] = UserSyncState(
                last_version=new_version,
                last_check_monotonic=time.monotonic(),
                last_sync_succeeded=True,
                dirty_keys=set(),
            )

    def sync_from_source_now(self, cfg: dict[str, Any], username: str) -> dict[str, Any] | None:
        """Pull settings from *cfg*'s source and apply them (explicit/manual).

        ``cfg`` is the active user's resolved ``settings_source`` config.
        Returns the imported settings dict, or ``None`` if the source is
        unknown, errors, or yields nothing.

        This (explicit) path **ignores the local ``dirty_keys`` set**: the
        user clicked "Sync now" precisely to get the source values, so a
        locally-edited key is overwritten and the dirty marker is cleared.
        """
        from vtsearch.settings_io.sources import get_settings_source

        source = get_settings_source(cfg["source_name"])
        if source is None:
            logger.warning("Unknown settings source: %s", cfg["source_name"])
            return None

        field_values = cfg.get("field_values", {})
        new_version: Any = None
        try:
            new_version = source.peek_version(field_values)
        except Exception:
            new_version = None

        try:
            imported = source.load(field_values)
        except Exception as exc:
            logger.exception("Failed to load from settings source: %s", exc)
            return None

        if not imported:
            return None

        # The setters invoked by ``_apply_settings`` acquire ``file_lock``
        # and then ``settings_lock``. Holding ``settings_lock`` here would
        # invert the canonical order - take the lock only to mutate
        # ``syncing``, then drop it for the actual apply.
        with self._lock:
            self._syncing.add(username)
        try:
            self._apply_settings(imported, None)
        finally:
            with self._lock:
                self._syncing.discard(username)

        with self._lock:
            self._sync_state[username] = UserSyncState(
                last_version=new_version,
                last_check_monotonic=time.monotonic(),
                last_sync_succeeded=True,
                dirty_keys=set(),
            )

        return imported

    # -- lifecycle -------------------------------------------------------

    def invalidate_server_cache(self) -> None:
        """Drop the server cache and re-arm the one-shot legacy migration.

        Used when the server settings path is repointed (CLI ``--settings``).
        Caller already holds ``settings_lock``.
        """
        self._server_cache = None
        self._legacy_migrated = False

    def drop_sync_state(self, username: str) -> None:
        """Forget *username*'s sync bookkeeping (e.g. when the source is cleared)."""
        with self._lock:
            self._sync_state.pop(username, None)

    def reset(self) -> None:
        """Reset every in-memory cache and sync state (for testing).

        Clears the shared containers in place (so the module-level aliases
        in :mod:`vtsearch.settings` stay valid) and the store-only state.
        """
        with self._lock:
            self._server_cache = None
            self._user_caches.clear()
            self._sync_state.clear()
            self._syncing.clear()
            self._legacy_migrated = False
        with self._user_sync_locks_guard:
            self._user_sync_locks.clear()
