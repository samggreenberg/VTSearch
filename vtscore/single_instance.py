"""Process-level single-instance lock.

Used by the VTSearch server (``app.py``) to refuse a second ``python app.py``
on the same port. Running the server twice in one allocation reloads the model
stack (~17 GB) and OOM-kills the SLURM job. Uses ``flock``, so the lock
releases automatically when the holding process exits -- a crash never leaves a
stale lock behind.

Linux/POSIX only (``fcntl``), consistent with the rest of the server, which
already relies on ``/proc`` and POSIX semantics.
"""
from __future__ import annotations

import errno
import fcntl
import os
import tempfile
from typing import IO


class AlreadyRunningError(RuntimeError):
    """Raised when another live process already holds the lock for this port."""

    def __init__(self, port: int, holder_pid: str) -> None:
        self.port = port
        self.holder_pid = holder_pid
        super().__init__(f"VTSearch already running (PID {holder_pid}) on port {port}")


def lock_path_for(port: int) -> str:
    """Path of the lockfile for ``port`` (override the dir via ``VTSEARCH_RUNDIR``)."""
    rundir = os.environ.get("VTSEARCH_RUNDIR") or tempfile.gettempdir()
    return os.path.join(rundir, f"vtsearch-{port}.lock")


def acquire(port: int) -> "IO[str]":
    """Take an exclusive lock for ``port`` and return the open handle to hold.

    The caller must keep the returned handle open for the process lifetime
    (``flock`` is released when it closes, or when the process exits). Raises
    :class:`AlreadyRunningError` if another live process already holds it.
    """
    handle = open(lock_path_for(port), "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno not in (errno.EACCES, errno.EAGAIN):
            handle.close()
            raise
        handle.seek(0)
        holder = handle.read().strip() or "unknown"
        handle.close()
        raise AlreadyRunningError(port, holder) from None
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle
