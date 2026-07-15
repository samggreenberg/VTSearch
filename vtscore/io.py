"""Shared file I/O helpers for plugins that read or write server-side files.

Two recurring patterns lived inline across importers / exporters / sources
before this module existed:

* Reading a JSON file from a server path: every label / settings importer
  and labelset / settings source repeated the same ``Path.exists()`` /
  ``is_file()`` / ``read_bytes()`` / ``json.loads()`` dance with slightly
  different error messages.
* Writing a file atomically: every server-side exporter and source
  re-implemented the tmp-file + ``fsync`` + ``os.replace`` ritual.  Forget
  one piece and a process crash mid-write leaves a half-written file behind.

The helpers here are deliberately small.  They standardise the
:class:`ValueError` text the framework surfaces to users, and they make
"future file-writing plugin that forgets `fsync`" impossible without
explicitly working around the helper.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

try:
    import fcntl as _fcntl  # POSIX-only; falls back to in-process locking on Windows.
except ImportError:  # pragma: no cover - Windows
    _fcntl = None

logger = logging.getLogger(__name__)

__all__ = [
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "file_lock",
    "read_server_json",
]


def read_server_json(path: Path | str, *, missing_ok: bool = False) -> Any:
    """Read and parse a JSON file from the server filesystem.

    Args:
        path: The file to read.
        missing_ok: When ``True``, return ``None`` if *path* doesn't
            exist.  Otherwise raise :class:`ValueError`.  Sources whose
            "no file yet" state is normal (sync sources on first load,
            etc.) pass ``missing_ok=True``; importers that demand a
            specific user-supplied file leave the default.

    Raises:
        ValueError: If the path exists but is not a regular file, the
            file is not valid UTF-8 JSON, or the file doesn't exist
            (only when ``missing_ok=False``).
    """
    p = Path(path)
    if not p.exists():
        if missing_ok:
            return None
        raise ValueError(f"File not found: {p}")
    if not p.is_file():
        raise ValueError(f"Not a file: {p}")
    raw = p.read_bytes()
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc


def atomic_write_text(path: Path | str, text: str) -> None:
    """Write *text* to *path* atomically.

    The implementation writes to a sibling ``.tmp`` file, ``fsync``\\ s it,
    then renames into place via :func:`os.replace`, which is atomic on
    POSIX, so a concurrent reader always sees either the pre- or
    post-write content, never a partial write.  Parent directories are
    created if needed.

    The file is opened with ``newline=""`` so any ``\\r\\n`` or ``\\n``
    sequences already present in *text* are preserved exactly.  Callers
    that built their content with :mod:`csv` (which emits ``\\r\\n``)
    therefore don't end up with doubled ``\\r``\\ s on Windows.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Per-writer unique tmp suffix so two threads (or two processes)
    # racing to overwrite the same destination can't truncate each
    # other's in-flight tmp file or chase one that was already renamed
    # away.  Mirrors the pattern in ``vtsearch.settings_store._atomic_write``.
    tmp = p.with_name(f"{p.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except Exception:
        # Best-effort tmp cleanup so a failed write doesn't leak a
        # half-written tmp file next to the destination.
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def atomic_write_bytes(path: Path | str, data: bytes) -> None:
    """Write *data* to *path* atomically (binary twin of :func:`atomic_write_text`).

    Writes to a per-writer-unique sibling ``.tmp`` file, ``fsync``\\ s it, then
    renames into place via :func:`os.replace`, so a concurrent reader always
    sees either the pre- or post-write content, never a partial write.  Parent
    directories are created if needed.  Used for binary artifacts such as the
    portable-detector zip bundle.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path | str, obj: Any, *, indent: int = 2) -> None:
    """Serialise *obj* to JSON and write atomically.

    Thin wrapper around :func:`atomic_write_text`; the trailing newline
    keeps the file POSIX-friendly (text-mode tools that expect a final
    newline don't complain).
    """
    atomic_write_text(path, json.dumps(obj, indent=indent) + "\n")


# ---------------------------------------------------------------------------
# Cross-process file locking
# ---------------------------------------------------------------------------
#
# Library-tier twin of ``vtsearch.settings_store.file_lock``.  It lives here
# (not in the app tier) so library-tier stores - the dataset/detector
# registries - can serialise their read-modify-write cycles across processes
# without importing ``vtsearch`` (which would break the library-clean layering).
#
# Pair it with a fresh ``_load()`` inside the ``with file_lock(...)`` block and
# an :func:`atomic_write_json` at the end: holding the flock while re-reading
# means a writer always merges into the *current* on-disk state instead of
# clobbering entries a sibling process committed since this process last read.

# Per-path in-process locks, held in addition to the cross-process flock so
# threads within one process serialise on the same path without repeatedly
# re-entering the kernel (and as the sole guard when ``fcntl`` is unavailable).
_path_locks: dict[str, threading.Lock] = {}
_path_locks_guard = threading.Lock()


def _path_lock_for(path: Path) -> threading.Lock:
    # resolve() is non-strict (works for not-yet-created files), so the same
    # target maps to one lock across its whole lifetime regardless of whether
    # the file exists yet or is reached via a relative/symlinked path.
    key = str(path.resolve())
    with _path_locks_guard:
        lock = _path_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _path_locks[key] = lock
        return lock


@contextlib.contextmanager
def file_lock(path: Path | str) -> Iterator[None]:
    """Acquire an exclusive cross-process lock for *path*.

    The lock is taken on a sibling ``<path>.lock`` file rather than on the
    data file itself, because atomic writes replace the data file's inode via
    :func:`os.replace` - any fd held against the old inode would be useless.
    The sibling lock file's inode is stable.

    An in-process lock is always taken first (so threads in one process
    serialise even on Windows), then the POSIX ``flock``.  ``flock`` releases
    automatically when the process exits, so a crash never leaves a stale lock
    behind.  On Windows (``fcntl`` unavailable) only the in-process lock is
    used and cross-process protection degrades silently; VTSearch is deployed
    on Linux, so this affects only the rare Windows-dev case.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    in_proc = _path_lock_for(p)
    in_proc.acquire()
    fd: int | None = None
    fcntl_mod = _fcntl  # snapshot so the narrowed binding survives the yield
    if fcntl_mod is not None:
        lock_path = p.with_name(p.name + ".lock")
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
