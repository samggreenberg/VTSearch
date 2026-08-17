"""User-facing notifications ("toasts") published from server-side code.

The progress trackers in :mod:`vtscore.concurrency.progress` publish *state*:
a channel has one current snapshot, clients re-read it whenever they like, and
a lost frame heals on the next heartbeat. This module publishes the other kind
of thing a backend has to say — a one-off **message**: "3 files had no readable
text and were skipped", "the remote API rate-limited us, so results are
partial". Those have no snapshot to re-read; they happen once and are shown
once.

The motivating case is plugin code that hits a recoverable problem mid-run.
Raising turns a partial success into a total failure, and logging buries the
news in a terminal the user is not looking at. :func:`notify` is the third
option: keep going, and put a toast in front of the user.

Usage from anywhere in the library or the app::

    from vtscore.concurrency.notifications import notify

    notify(
        "Skipped 3 unreadable files",
        level="warning",
        detail="page_2.pdf, page_9.pdf, notes.pdf could not be decoded.",
        source="Server Folder",
    )

Plugin subclasses get the same thing with the ``source`` filled in for them —
see :meth:`vtscore.plugins.PluginBase.notify`.

Delivery semantics — read these before relying on a notification:

- **Live broadcast, no replay.** Every client with an open ``/api/events``
  stream at the moment of the call gets the frame; a client that connects a
  second later does not. Notifications are for narrating work the user is
  currently watching, not for durable records. Anything that must survive a
  page reload belongs in a task's terminal payload or in persisted state.
- **Every open client sees it,** including a second browser tab. There is no
  per-session routing: the backend has no notion of which request belongs to
  which stream.
- **Never raises.** A bad level is coerced to ``"info"``, an over-long message
  is truncated, and a subscriber that blows up is swallowed — a call that is
  supposed to *avoid* interrupting the caller must not become the interruption.
- **Always logged**, at a severity matching the level, so a headless run (CLI,
  SLURM, tests) still has a record when no browser is attached.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, get_args

logger = logging.getLogger(__name__)

#: Severities a notification can carry. They map onto the frontend's toast
#: levels one-for-one: ``error`` and ``warning`` stay up until the user
#: dismisses them, ``success`` and ``info`` auto-dismiss.
NotificationLevel = Literal["info", "success", "warning", "error"]

#: Tuple form of :data:`NotificationLevel`, for validation and tests.
LEVELS: tuple[str, ...] = get_args(NotificationLevel)

#: Fallback for a level the caller got wrong (see the "never raises" contract).
DEFAULT_LEVEL: NotificationLevel = "info"

_LOG_LEVEL_FOR: dict[str, int] = {
    "info": logging.INFO,
    "success": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

#: Caps on what one notification may push into every connected client's queue.
#: A toast is a headline, not a log file; a plugin that hands us a stack trace
#: or a 10k-entry filename list gets it cut off rather than wedging the stream.
MAX_MESSAGE_CHARS = 300
MAX_DETAIL_CHARS = 2000

#: Per-process id prefix. Ids only need to be unique among the notifications a
#: client can hold at once, but a restarted backend re-using ``note_1`` while a
#: stale toast is still on screen would collide in the frontend's dedup map, so
#: the counter is namespaced per process.
_ID_PREFIX = uuid.uuid4().hex[:8]
_id_counter = itertools.count(1)


def _truncate(text: str, limit: int) -> str:
    """Clip *text* to *limit* characters, marking that something was cut."""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


@dataclass(frozen=True)
class Notification:
    """One user-facing message, as broadcast to connected clients."""

    #: Unique within this process; the frontend keys its toast on it.
    id: str
    level: NotificationLevel
    message: str
    #: Secondary line, shown smaller under the message.
    detail: Optional[str] = None
    #: Who is talking — a plugin's display name, a subsystem. Rendered next
    #: to the detail so the user knows which part of the app spoke up.
    source: Optional[str] = None
    #: Unix seconds, filled in at construction.
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form; the SSE ``notification`` frame payload."""
        return {
            "id": self.id,
            "level": self.level,
            "message": self.message,
            "detail": self.detail,
            "source": self.source,
            "timestamp": self.timestamp,
        }


class NotificationBroker:
    """Fan-out point for :class:`Notification`\\ s.

    Mirrors :meth:`~vtscore.concurrency.progress.ProgressTracker.subscribe` /
    ``unsubscribe`` so the SSE stream registers for notifications exactly the
    way it registers for progress. The difference is what is delivered: a
    tracker replays its current snapshot to every new subscriber, this broker
    has no snapshot to replay and delivers only what happens next.
    """

    def __init__(self) -> None:
        self._subscribers: list[Callable[[Notification], None]] = []
        self._lock = threading.Lock()

    def subscribe(self, callback: Callable[[Notification], None]) -> None:
        """Register *callback*, fired once per published notification.

        The callback runs synchronously on the publishing thread, outside the
        broker's lock. Subscribers must be non-blocking and exception-safe;
        exceptions are swallowed so one bad subscriber cannot break a publish
        for the others.
        """
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Notification], None]) -> None:
        """Remove a previously-registered subscriber. No-op if not present."""
        with self._lock:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass

    def subscriber_count(self) -> int:
        """Number of live subscribers (roughly: connected clients)."""
        with self._lock:
            return len(self._subscribers)

    def clear_subscribers(self) -> None:
        """Drop every subscriber. For test isolation, not production use."""
        with self._lock:
            self._subscribers.clear()

    def publish(self, notification: Notification) -> None:
        """Deliver *notification* to every current subscriber."""
        with self._lock:
            subs = list(self._subscribers)
        for callback in subs:
            try:
                callback(notification)
            except Exception:
                logger.exception("Notification subscriber raised; continuing")


#: Process-wide broker. The SSE endpoint subscribes one handler per connected
#: client; the CLI subscribes a printer (see ``vtsearch.cli_main``).
notifications = NotificationBroker()


def notify(
    message: str,
    *,
    level: str = DEFAULT_LEVEL,
    detail: Optional[str] = None,
    source: Optional[str] = None,
) -> Notification:
    """Show *message* to every connected user, and log it.

    This is the whole public API. It is deliberately impossible to make it
    fail: it is called from paths that chose *not* to raise, so it must not
    reintroduce the exception it was used to avoid.

    Args:
        message: Headline, one short sentence. Truncated at
            :data:`MAX_MESSAGE_CHARS`.
        level: One of :data:`LEVELS`. ``"warning"`` / ``"error"`` stay on
            screen until dismissed; ``"info"`` / ``"success"`` fade on their
            own. An unrecognised value is coerced to :data:`DEFAULT_LEVEL`
            (and logged) rather than raising.
        detail: Optional second line with the specifics — which files, which
            endpoint, how many. Truncated at :data:`MAX_DETAIL_CHARS`.
        source: Who is speaking, e.g. a plugin's ``display_name``. Plugin
            subclasses should use :meth:`vtscore.plugins.PluginBase.notify`,
            which fills this in.

    Returns:
        The :class:`Notification` that was published. A blank *message*
        produces a notification that is returned but **not** broadcast (an
        empty toast is worse than none), which is also the only case where
        nothing reaches the user.
    """
    if level not in LEVELS:
        logger.warning("Unknown notification level %r; falling back to %r", level, DEFAULT_LEVEL)
        level = DEFAULT_LEVEL

    text = _truncate(str(message).strip(), MAX_MESSAGE_CHARS)
    detail_text = _truncate(str(detail).strip(), MAX_DETAIL_CHARS) if detail else None
    source_text = str(source).strip() or None if source else None

    notification = Notification(
        id=f"note_{_ID_PREFIX}_{next(_id_counter)}",
        level=level,  # type: ignore[arg-type]  # narrowed by the LEVELS check above
        message=text,
        detail=detail_text or None,
        source=source_text,
    )

    prefix = f"[{source_text}] " if source_text else ""
    logger.log(
        _LOG_LEVEL_FOR[level],
        "%s%s%s",
        prefix,
        text or "(empty notification)",
        f" - {detail_text}" if detail_text else "",
    )

    if text:
        notifications.publish(notification)
    return notification
