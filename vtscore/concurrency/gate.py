"""A semaphore-like gate whose limit is re-read on every acquisition."""

from __future__ import annotations

import threading
import time
from typing import Callable


class ConcurrencyGate:
    """A semaphore-like gate whose limit is read fresh on every acquisition.

    Unlike :class:`threading.Semaphore`, the cap is a callable evaluated at
    each ``acquire()`` attempt, so changes to the underlying setting take
    effect immediately for queued and future tasks (already-running tasks
    are never preempted).
    """

    def __init__(self, get_limit: Callable[[], int]) -> None:
        self._get_limit = get_limit
        self._cv = threading.Condition()
        self._active = 0
        #: Test-visible hook: set every time a blocking ``acquire()`` actually
        #: parks because the gate is full.  Lets tests wait for "a waiter is
        #: queued" deterministically instead of sleeping a fixed race window.
        #: Tests clear it before triggering the contended acquire they care
        #: about; production code never reads it.
        self.waiter_parked = threading.Event()

    def _limit(self) -> int:
        return max(1, int(self._get_limit()))

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        with self._cv:
            if not blocking:
                if self._active >= self._limit():
                    return False
                self._active += 1
                return True

            if timeout is None:
                while self._active >= self._limit():
                    self.waiter_parked.set()
                    self._cv.wait()
                self._active += 1
                return True

            deadline = time.monotonic() + timeout
            while self._active >= self._limit():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.waiter_parked.set()
                self._cv.wait(timeout=remaining)
            self._active += 1
            return True

    def release(self) -> None:
        with self._cv:
            self._active -= 1
            self._cv.notify_all()

    @property
    def active(self) -> int:
        with self._cv:
            return self._active
