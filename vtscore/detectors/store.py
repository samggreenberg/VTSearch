"""Low-level file I/O helpers for detector JSON files.

Provides path resolution, read, and write utilities used by the route layer
(``vtsearch.routes.detectors``) and by model-layer modules that need to
persist or inspect detector data without importing the full route blueprint.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from vtscore.config import DATA_DIR

#: Cap on the slug length so ``<slug>.json`` and its longer atomic-write
#: sibling ``<slug>.json.<pid>.<uuid>.tmp`` (worst case ~50 extra chars) stay
#: under the common filesystem ``NAME_MAX`` of 255.  The schema layer already
#: caps user-supplied names well below this (see
#: ``vtsearch.schemas.detectors.MAX_NAME_LENGTH``); this is the last-line
#: backstop for names that reach the store by other paths (combine, CLI,
#: direct library use) so a write can never raise ``[Errno 36] File name too
#: long`` — an uncaught ``OSError`` whose message leaks the absolute path.
_MAX_SLUG_LENGTH = 190


def get_detectors_dir() -> Path:
    """Return the configured detectors directory.

    Reads from ``CoreConfig.from_settings()`` rather than ``vtsearch.settings``
    directly so this module stays library-clean (see Phase 2 of
    ``../docs/architecture.md``).  The classmethod still consults the
    app's settings layer today; after Phase 8 it moves to an app-side shim
    and library callers pass a ``CoreConfig`` explicitly.
    """
    from vtscore.config import CoreConfig  # noqa: PLC0415

    return CoreConfig.from_settings().detectors_dir


#: Default location used by tests that bypass settings.
DETECTORS_DIR = DATA_DIR / "detectors"


def _slug(name: str) -> str:
    """Turn a human-readable name into a filesystem-safe slug.

    Truncated to ``_MAX_SLUG_LENGTH`` so the derived filename stays under the
    filesystem ``NAME_MAX``.  When truncation drops characters, an 8-hex
    content hash of the full name is appended so two long names sharing a
    prefix don't collide onto the same file.
    """
    slug = re.sub(r"[^a-z0-9_-]+", "_", name.lower()).strip("_") or "detector"
    if len(slug) > _MAX_SLUG_LENGTH:
        digest = hashlib.blake2b(name.encode("utf-8"), digest_size=4).hexdigest()
        slug = f"{slug[: _MAX_SLUG_LENGTH - len(digest) - 1]}_{digest}"
    return slug


def _detector_path(name: str) -> Path:
    return get_detectors_dir() / f"{_slug(name)}.json"


def _read_detector(path: Path) -> dict | None:
    """Parse the detector JSON at *path*, or ``None`` if absent or malformed.

    A write queued through :func:`queue_detector_write` but not yet on disk is
    what the file *will* contain, so the reader returns that text instead of
    the stale file: a vote is visible to every in-process reader the moment
    it is accepted, even while the background writer is still waiting on the
    filesystem.  Each call parses afresh, so callers get their own copy and
    cannot corrupt the queued text by mutating the result.
    """
    with _queue_lock:
        pend = _pending.get(path)
    if pend is not None:
        return json.loads(pend.text)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _atomic_write_text(path: Path, text: str) -> None:
    get_detectors_dir().mkdir(parents=True, exist_ok=True)
    # Per-writer unique tmp suffix so two threads (or two processes) racing to
    # overwrite the same detector file can't truncate each other's in-flight
    # tmp file or chase one that was already renamed away (which surfaced as
    # ``FileNotFoundError: '<name>.json.tmp' -> '<name>.json'`` from
    # ``os.replace``).  Mirrors ``vtsearch.settings_store._atomic_write`` and
    # ``vtscore.io.atomic_write_text``.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        # Best-effort tmp cleanup so a failed write doesn't leak a
        # half-written tmp file next to the destination.
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def _write_detector(path: Path, data: dict) -> None:
    """Write *data* to *path* now, on the calling thread, atomically.

    Any write still queued for *path* lands first (in order), so a caller
    that read the queued state through :func:`_read_detector`, modified it
    and writes it back can never be overtaken by the older queued text.
    """
    text = json.dumps(data, indent=2)
    with _writer_lock:
        _drain_pending(path)
        _atomic_write_text(path, text)


# ---------------------------------------------------------------------------
# Background write queue for the per-vote labelset rewrite (issue #3853)
#
# Every vote rewrites the whole detector JSON - 650 KB at 800 labels - and
# ``os.fsync`` + ``os.replace`` of that file went to a shared NFS export in
# the deployment where the stall was felt.  A WRITE RPC there averaged 75 ms
# of queue+service time and its tail ran to seconds, and the write sat
# inline in ``POST /api/medias/<id>/vote``, so the panel stayed black until
# the filesystem answered.  Persisting off the request thread makes the vote
# return as soon as the labelset is composed; the file follows within the
# filesystem's latency, and a burst of votes that outruns it coalesces into
# one write of the newest text.
#
# Correctness contract (what keeps "the file is the truth" true enough):
#   * ``_read_detector`` returns queued text, so in-process readers never see
#     the file lag its votes.
#   * ``_write_detector`` drains the queue for its path before writing, so a
#     direct writer that read queued state cannot be overtaken by it.
#   * ``flush_detector_writes`` lands everything inline; it is registered
#     with ``atexit`` and called wherever another *process* is about to read
#     the file (detector unload).  ``discard_pending_detector_write`` is for
#     the delete paths, where landing the write would resurrect the file.
#   * A failed background write is logged and raised from the *next*
#     ``queue_detector_write`` for that path, so the vote route still turns
#     a persistence failure into a visible 500 - one vote late.
# ``VTSEARCH_DETECTOR_WRITE_MODE=sync`` restores the inline write (the test
# suite runs that way, since it asserts on file contents after each call).
# ---------------------------------------------------------------------------

#: Env var: ``async`` (default) queues the labelset rewrite to a background
#: thread; ``sync`` writes it inline on the caller's thread.
WRITE_MODE_ENV = "VTSEARCH_DETECTOR_WRITE_MODE"

log = logging.getLogger(__name__)


class DetectorWriteError(RuntimeError):
    """A queued detector write failed; raised from the next queue call for that path."""


class _PendingWrite:
    __slots__ = ("after", "queued_at", "text")

    def __init__(self, text: str, after: Callable[[], None] | None) -> None:
        self.text = text
        self.after = after
        self.queued_at = time.monotonic()


_queue_lock = threading.Lock()
_queue_cv = threading.Condition(_queue_lock)
_pending: dict[Path, _PendingWrite] = {}
_failed: dict[Path, Exception] = {}
#: Serialises physical writes (background thread, flushers, direct writers).
_writer_lock = threading.RLock()
_writer_thread: threading.Thread | None = None
_write_mode_override: str | None = None
#: Tests set this False to keep queued writes parked until an explicit flush.
_writer_autostart = True


def detector_write_mode() -> str:
    """``"sync"`` or ``"async"``: the override set for tests, else the env var."""
    mode = _write_mode_override or os.environ.get(WRITE_MODE_ENV, "async")
    return "sync" if mode.strip().lower() == "sync" else "async"


def set_detector_write_mode(mode: str | None) -> None:
    """Override :func:`detector_write_mode` (``None`` restores the env var)."""
    global _write_mode_override
    _write_mode_override = mode


def queue_detector_write(path: Path, data: dict, *, after: Callable[[], None] | None = None) -> None:
    """Persist *data* to *path*: inline in ``sync`` mode, else on the writer thread.

    *after* runs once the write has landed (on whichever thread wrote it),
    for the cache refreshes that must see the file's new mtime.  Serialising
    happens here, on the caller's thread, so *data* may be mutated freely
    after the call returns.

    Raises :class:`DetectorWriteError` when the previous queued write for
    *path* failed, after queueing this one (whose text supersedes the lost
    one, so a later success repairs the file).
    """
    text = json.dumps(data, indent=2)
    if detector_write_mode() == "sync":
        with _writer_lock:
            _drain_pending(path)
            _atomic_write_text(path, text)
        if after is not None:
            after()
        return
    with _queue_cv:
        failed = _failed.pop(path, None)
        _pending[path] = _PendingWrite(text, after)
        _ensure_writer_thread_locked()
        _queue_cv.notify()
    if failed is not None:
        raise DetectorWriteError(f"the previous write of {path.name} failed: {failed}") from failed


def pending_detector_writes() -> list[Path]:
    """Paths with a queued write not yet on disk (newest queue order not implied)."""
    with _queue_lock:
        return list(_pending)


def flush_detector_writes(path: Path | None = None) -> None:
    """Land every queued write (or just *path*'s) now, on the calling thread."""
    with _writer_lock:
        _drain_pending(path)


def discard_pending_detector_write(path: Path) -> None:
    """Drop a queued write for *path* without landing it.

    For the delete paths: waits for any in-flight write of *path* to finish
    first, so nothing lands after the caller has unlinked the file.
    """
    with _writer_lock, _queue_lock:
        _pending.pop(path, None)
        _failed.pop(path, None)


def reset_detector_write_queue_for_tests() -> None:
    """Drop queued writes and failures; tests call this between cases."""
    with _writer_lock, _queue_lock:
        _pending.clear()
        _failed.clear()


def _ensure_writer_thread_locked() -> None:
    global _writer_thread
    if not _writer_autostart or (_writer_thread is not None and _writer_thread.is_alive()):
        return
    _writer_thread = threading.Thread(target=_writer_loop, name="detector-writer", daemon=True)
    _writer_thread.start()


def _writer_loop() -> None:
    while True:
        with _queue_cv:
            while not _pending:
                _queue_cv.wait()
        with _writer_lock:
            _drain_pending(None)


def _drain_pending(only: Path | None) -> None:
    """Write queued entries (all, or *only*) until none are left.  Caller holds ``_writer_lock``.

    An entry stays in ``_pending`` while its text is being written, so a
    concurrent :func:`_read_detector` keeps returning the queued state
    rather than the half-old file; it is removed only if no newer text was
    queued for the same path meanwhile, in which case the loop writes again.
    """
    while True:
        with _queue_lock:
            if only is not None:
                pend = _pending.get(only)
                item = (only, pend) if pend is not None else None
            else:
                item = next(iter(_pending.items()), None)
        if item is None:
            return
        path, pend = item
        # Keep the filesystem's latency visible now that it no longer holds
        # a request: one WARNING per slow landing, with how long the text sat
        # queued before the writer got to it.
        from vtscore.concurrency.stalls import PhaseClock

        clock = PhaseClock(
            "detector_write", file=path.name, queued_ms=round((time.monotonic() - pend.queued_at) * 1000)
        )
        try:
            _atomic_write_text(path, pend.text)
            clock.mark("write")
            if pend.after is not None:
                pend.after()
            clock.finish(bytes=len(pend.text))
        except Exception as exc:  # noqa: BLE001 - reported on the next queue call
            log.exception("queued detector write failed for %s", path)
            with _queue_lock:
                _failed[path] = exc
                if _pending.get(path) is pend:
                    del _pending[path]
            if only is not None:
                return
            continue
        with _queue_lock:
            if _pending.get(path) is pend:
                del _pending[path]


atexit.register(flush_detector_writes)


def save_detector(
    name: str,
    labelset: Any,
    *,
    media_type: str = "",
    embedder_type: str = "",
    extra: dict | None = None,
) -> Path:
    """Write *labelset* to ``<detectors_dir>/<slug>.json`` and return the path.

    The library-facing entry point for detector persistence: a caller that
    has built a :class:`~vtscore.datasets.labelset.LabelSet` (from votes, from
    an external label store, from its own pipeline) hands it here and gets a
    file the app, the CLI, and :func:`load_detector` all read.

    Only origins and labels are written — never embeddings and never model
    weights (see the "No Persisted Vectors or MLPs" invariant).  An existing
    file for the same slug is replaced wholesale; use :func:`load_detector`
    plus ``LabelSet.merge`` first when you mean to add to one.

    Args:
        name: Human-readable detector name.  Slugified for the filename.
        labelset: The :class:`~vtscore.datasets.labelset.LabelSet` to persist.
        media_type: Media type the detector scores (``"audio"``, ``"image"``, …).
        embedder_type: Locked embedder type (``"semantic"`` /
            ``"patch_semantic"`` / ``"structural"``); ``""`` leaves the
            detector typeless, resolved from its labels on load.
        extra: Additional top-level keys to merge into the file (e.g.
            ``{"text_query": "dog barking"}``).  Never overrides the keys
            above.

    Returns:
        The path written.
    """
    data: dict = dict(extra or {})
    data.update(
        {
            "name": name,
            "media_type": media_type,
            "labelset": labelset.to_dict(),
        }
    )
    if embedder_type:
        data["embedder_type"] = embedder_type
    path = _detector_path(name)
    _write_detector(path, data)
    return path


def load_detector(name: str) -> dict | None:
    """Return the parsed detector JSON for *name*, or ``None`` if absent.

    The counterpart of :func:`save_detector`.  Rebuild the labelset with
    ``LabelSet.from_dict(data["labelset"])``; the head itself is not stored
    and is re-derived from the labelset's origins (see
    :func:`vtscore.detectors.training.train_detector_from_origins`).
    """
    return _read_detector(_detector_path(name))
