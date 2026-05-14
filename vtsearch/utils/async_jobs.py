"""Single-slot async job manager for long-running training operations.

The Flask app runs with ``gthread`` workers and a small thread pool.  Endpoints
that do heavy GIL-bound Python work (training loops, list/array churn) starve
unrelated requests like ``/api/votes`` polls and thumbnail fetches.  This
module lets those endpoints push the heavy work onto a background daemon
thread so the request handler can return immediately with a job id, freeing
the worker thread to serve other requests while training runs.

Each :class:`JobManager` is a *single-slot* runner: starting a new job
cancels the in-flight one (callers can re-derive at any time).  Results from
the most recent successful run are kept in a tiny ring so a follow-up
request with the same signature can short-circuit and return immediately —
the "re-sort without new votes is free" fast path.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class AsyncJob:
    """State container for a single background training job."""

    job_id: str
    signature: Any = None
    status: str = "running"  # "running" | "done" | "error" | "cancelled"
    result: Any = None
    error: str | None = None
    current: int = 0
    total: int = 0
    message: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    cancel_event: threading.Event = field(default_factory=threading.Event)
    done_event: threading.Event = field(default_factory=threading.Event)

    @property
    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def cancel(self) -> None:
        self.cancel_event.set()

    def update_progress(self, current: int, total: int, message: str = "") -> None:
        """Update progress counters atomically (single writer per job)."""
        self.current = current
        self.total = total
        if message:
            self.message = message


class JobManager:
    """Single-slot, signature-keyed background job runner.

    Starting a new job cancels any in-flight one.  Results from the most
    recently completed job are cached by *signature* so that a follow-up
    call with an unchanged signature reuses the previous result instead of
    re-running.
    """

    def __init__(self, name: str, max_history: int = 8) -> None:
        self._name = name
        self._lock = threading.RLock()
        self._jobs: dict[str, AsyncJob] = {}
        self._current_id: str | None = None
        self._last_done: AsyncJob | None = None
        self._max_history = max_history

    # ------------------------------------------------------------------ #
    # Lookups
    # ------------------------------------------------------------------ #

    def get(self, job_id: str) -> Optional[AsyncJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def current(self) -> Optional[AsyncJob]:
        with self._lock:
            if self._current_id is None:
                return None
            return self._jobs.get(self._current_id)

    def cached_for(self, signature: Any) -> Optional[AsyncJob]:
        """Return the most-recent completed job iff its signature matches."""
        with self._lock:
            j = self._last_done
            if j is not None and j.status == "done" and j.signature == signature:
                return j
            return None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(
        self,
        signature: Any,
        target: Callable[[AsyncJob], Any],
    ) -> AsyncJob:
        """Start a new job.  Cancels the in-flight one if any.

        *target* receives the :class:`AsyncJob` and should assign
        ``job.result`` before returning.  Raising propagates as
        ``status = "error"``.
        """
        with self._lock:
            prev = self._jobs.get(self._current_id) if self._current_id else None
            if prev is not None and prev.status == "running":
                prev.cancel()
            job = AsyncJob(
                job_id=uuid.uuid4().hex,
                signature=signature,
                status="running",
                started_at=time.time(),
            )
            self._jobs[job.job_id] = job
            self._current_id = job.job_id
            self._prune_locked()

        threading.Thread(
            target=self._run,
            args=(job, target),
            name=f"{self._name}-{job.job_id[:8]}",
            daemon=True,
        ).start()
        return job

    def _run(self, job: AsyncJob, target: Callable[[AsyncJob], Any]) -> None:
        try:
            target(job)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).exception("%s job failed", self._name)
            with self._lock:
                job.status = "error"
                job.error = str(exc) or exc.__class__.__name__
            return
        finally:
            job.finished_at = time.time()

        with self._lock:
            if job.is_cancelled and job.status == "running":
                job.status = "cancelled"
            elif job.status == "running":
                job.status = "done"
                self._last_done = job
            job.done_event.set()

    def _prune_locked(self) -> None:
        if len(self._jobs) <= self._max_history:
            return
        keep: set[str] = set()
        if self._current_id:
            keep.add(self._current_id)
        if self._last_done is not None:
            keep.add(self._last_done.job_id)
        # Keep the most recent N by start time
        ordered = sorted(self._jobs.values(), key=lambda j: j.started_at, reverse=True)
        for j in ordered[: self._max_history]:
            keep.add(j.job_id)
        self._jobs = {jid: j for jid, j in self._jobs.items() if jid in keep}

    # ------------------------------------------------------------------ #
    # Test helpers
    # ------------------------------------------------------------------ #

    def reset_for_tests(self) -> None:
        """Cancel any running job and clear all stored state."""
        with self._lock:
            for j in self._jobs.values():
                if j.status == "running":
                    j.cancel()
            self._jobs.clear()
            self._current_id = None
            self._last_done = None


# ---------------------------------------------------------------------- #
# Application-wide singletons
# ---------------------------------------------------------------------- #

#: Background runner for ``/api/learned-sort``.
learned_sort_jobs = JobManager("learned-sort")

#: Background runner for ``/api/eval/train-and-score``.
eval_jobs = JobManager("eval-train-score")


def reset_all_async_jobs_for_tests() -> None:
    """Reset every singleton job manager.  Called from the autouse fixture."""
    learned_sort_jobs.reset_for_tests()
    eval_jobs.reset_for_tests()


def serialize_job(job: AsyncJob) -> dict[str, Any]:
    """Return a JSON-safe snapshot of *job* (without ``result``).

    The result payload differs per endpoint and is added by the route.
    """
    return {
        "job_id": job.job_id,
        "status": job.status,
        "current": job.current,
        "total": job.total,
        "message": job.message,
        "error": job.error,
    }
