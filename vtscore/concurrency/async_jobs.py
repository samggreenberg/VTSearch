"""Single-slot async job manager for long-running training operations.

The Flask app runs with ``gthread`` workers and a small thread pool.  Endpoints
that do heavy GIL-bound Python work (training loops, list/array churn) starve
unrelated requests like ``/api/votes`` polls and thumbnail fetches.  This
module lets those endpoints push the heavy work onto a background daemon
thread so the request handler can return immediately with a job id, freeing
the worker thread to serve other requests while training runs.

Each :class:`JobManager` runs **one** background job at a time and keeps a
single coalescing pending slot.  When a new ``start()`` arrives while a job
is running, its target+signature are stashed as *pending*; if another
``start()`` from the **same requester context** (user + dataset + detector)
arrives before the runner picks the pending slot up, the pending slot is
updated in place (latest wins) and the same job object is returned.  A
``start()`` from a *different* requester context instead supersedes the
parked pending (marking it ``cancelled`` so its pollers see a terminal
state rather than another request's results) and parks a fresh job.  When
the running job finishes, the pending job is promoted and spawned
automatically - unless it was cancelled while parked, in which case it is
terminated without running.  This avoids the previous design's failure mode
where rapid-fire requests spawned parallel training threads that fought
for CPU/GPU - cancellation was cooperative-only and never honoured inside
the training loop.

Results from the most recent successful run are kept by *signature* so a
follow-up request with an unchanged signature can short-circuit and return
the previous result - the "re-sort without new votes is free" fast path.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

from vtscore.concurrency.progress import CancelledError


@dataclass
class AsyncJob:
    """State container for a single background training job."""

    job_id: str
    signature: Any = None
    status: str = "running"  # "pending" | "running" | "done" | "error" | "cancelled"
    result: Any = None
    error: str | None = None
    current: int = 0
    total: int = 0
    message: str = ""
    started_at: float = 0.0
    cancel_event: threading.Event = field(default_factory=threading.Event)
    done_event: threading.Event = field(default_factory=threading.Event)
    # Username of the request that spawned this job, captured at start()
    # time so the worker thread can resolve per-user settings correctly.
    user: str | None = None
    # (dataset_id, detector_id) the job is operating against, captured at
    # start() time so /api/jobs/active can list which (ds, det) pairs have
    # background work in flight without parsing per-manager signatures.
    dataset_id: str = ""
    detector_id: str = ""

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


# ---------------------------------------------------------------------- #
# Cooperative cancellation of *running* jobs
# ---------------------------------------------------------------------- #
#
# The heavy work a job performs (MLP epoch loops in ``train_model``, eval
# retraining over label history) is GIL-bound Python/torch, so cancelling
# the parked pending slot - which this module already does - does nothing
# for a job that is *already running*.  To make cancel real, the running
# job is bound to its worker thread here and the deep compute loops poll
# :func:`check_job_cancelled` at their natural boundaries (epoch, eval
# step).  When the job's ``cancel_event`` is set, the next poll raises
# :class:`CancelledError`, which unwinds the training/eval stack and is
# caught in :meth:`JobManager._run_inner` to mark the job ``cancelled``.
#
# The binding is thread-local and restored on scope exit, and every job
# runs on its own fresh daemon thread, so no cancellation state leaks
# across jobs or into the synchronous request handlers, tests, and CLI
# paths that share the same training code.  Outside a bound job
# :func:`check_job_cancelled` is a no-op.
_thread_state = threading.local()


@contextmanager
def bind_job_cancellation(job: AsyncJob) -> Iterator[None]:
    """Bind *job* as the current worker thread's cancellable job.

    Deep compute loops call :func:`check_job_cancelled` to observe the
    binding.  The previous binding (normally ``None``) is restored on exit.
    """
    prev = getattr(_thread_state, "job", None)
    _thread_state.job = job
    try:
        yield
    finally:
        _thread_state.job = prev


def current_job() -> Optional[AsyncJob]:
    """Return the :class:`AsyncJob` bound to the current thread, or ``None``."""
    return getattr(_thread_state, "job", None)


def check_job_cancelled() -> None:
    """Raise :class:`CancelledError` if the current thread's job was cancelled.

    A no-op when no job is bound to the calling thread (synchronous request
    handlers, tests, the CLI), so shared library code that runs both inside
    and outside a background job can call it unconditionally at loop
    boundaries.
    """
    job = getattr(_thread_state, "job", None)
    if job is not None and job.is_cancelled:
        raise CancelledError("Job cancelled by user")


class JobManager:
    """Single-runner, signature-keyed background job manager with one pending slot.

    Only one job runs at a time.  A second ``start()`` while a job is in
    flight stashes the new target+signature in a pending slot; further
    ``start()`` calls from the same requester context (user + dataset +
    detector pair) update the pending slot in place (latest wins) and
    return the same pending job object, while a call from a different
    requester context supersedes the parked pending with a fresh job.
    When the running job finishes, the pending job is promoted and spawned
    (unless it was cancelled while parked).

    Results from the most recently completed job are cached by *signature*
    so a follow-up call with an unchanged signature reuses the previous
    result instead of re-running.
    """

    def __init__(self, name: str, max_history: int = 8) -> None:
        self._name = name
        self._lock = threading.RLock()
        self._jobs: dict[str, AsyncJob] = {}
        self._current_id: str | None = None
        self._pending: AsyncJob | None = None
        self._pending_target: Callable[[AsyncJob], Any] | None = None
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
        *,
        dataset_id: str = "",
        detector_id: str = "",
    ) -> AsyncJob:
        """Start (or coalesce into pending) a job.

        If no job is running, *target* spawns immediately on a daemon
        thread.  If a job is already running, the new (signature, target)
        is stashed in the pending slot - overwriting any existing pending
        - and the same pending :class:`AsyncJob` is returned on every
        subsequent call until the runner picks it up.

        *target* receives the :class:`AsyncJob` and should assign
        ``job.result`` before returning.  Raising propagates as
        ``status = "error"``.

        ``dataset_id`` / ``detector_id`` identify the (dataset, detector)
        pair the job is running against; the values are stashed on the
        :class:`AsyncJob` so ``/api/jobs/active`` can enumerate busy pairs
        for the top-bar pulldown's spinner glyph.
        """
        # Capture the user that triggered this start() so background work
        # touching per-user settings resolves to the right user.
        from vtsearch.auth import get_current_user

        current_user = get_current_user()

        spawn_job: AsyncJob | None = None
        with self._lock:
            prev = self._jobs.get(self._current_id) if self._current_id else None
            if prev is not None and prev.status == "running":
                pend = self._pending
                if pend is not None:
                    if (
                        pend.user == current_user
                        and pend.dataset_id == dataset_id
                        and pend.detector_id == detector_id
                        and not pend.is_cancelled
                    ):
                        # Coalesce: same requester context (user + pair), so
                        # update the existing pending in place and every
                        # caller waiting on this job_id receives the latest
                        # run.  This is the rapid vote→re-sort burst path.
                        pend.signature = signature
                        self._pending_target = target
                        return pend
                    # Different (user, dataset, detector) - or a pending that
                    # was cancelled while parked.  Updating it in place would
                    # serve its pollers *another request's* results (wrong
                    # dataset/detector, wrong user) under their job_id, so
                    # supersede: terminate the old pending visibly and park a
                    # fresh job for the new requester.
                    pend.status = "cancelled"
                    pend.error = pend.error or "superseded by a newer request"
                    pend.done_event.set()
                    self._pending = None
                    self._pending_target = None
                job = AsyncJob(
                    job_id=uuid.uuid4().hex,
                    signature=signature,
                    status="pending",
                    started_at=time.time(),
                    user=current_user,
                    dataset_id=dataset_id,
                    detector_id=detector_id,
                )
                self._jobs[job.job_id] = job
                self._pending = job
                self._pending_target = target
                self._prune_locked()
                return job

            job = AsyncJob(
                job_id=uuid.uuid4().hex,
                signature=signature,
                status="running",
                started_at=time.time(),
                user=current_user,
                dataset_id=dataset_id,
                detector_id=detector_id,
            )
            self._jobs[job.job_id] = job
            self._current_id = job.job_id
            self._prune_locked()
            spawn_job = job

        self._spawn_thread(spawn_job, target)
        return job

    def _spawn_thread(self, job: AsyncJob, target: Callable[[AsyncJob], Any]) -> None:
        threading.Thread(
            target=self._run,
            args=(job, target),
            name=f"{self._name}-{job.job_id[:8]}",
            daemon=True,
        ).start()

    def _run(self, job: AsyncJob, target: Callable[[AsyncJob], Any]) -> None:
        from contextlib import ExitStack

        from vtsearch.auth import thread_user
        from vtscore.state.core import (
            get_context,
            get_detector_context,
            thread_dataset_context,
            thread_detector_context,
        )

        ds_ctx = get_context(job.dataset_id) if job.dataset_id else None
        det_ctx = get_detector_context(job.detector_id) if job.detector_id else None
        with ExitStack() as stack:
            # Bind the job so ``check_job_cancelled`` in the deep training /
            # eval loops can honour a cancel of this *running* job.
            stack.enter_context(bind_job_cancellation(job))
            if job.user is not None:
                stack.enter_context(thread_user(job.user))
            if ds_ctx is not None:
                stack.enter_context(thread_dataset_context(ds_ctx))
            if det_ctx is not None:
                stack.enter_context(thread_detector_context(det_ctx))
            self._run_inner(job, target)

    def _run_inner(self, job: AsyncJob, target: Callable[[AsyncJob], Any]) -> None:
        try:
            target(job)
        except CancelledError:
            # A poll of ``check_job_cancelled`` inside the training / eval
            # loop unwound the stack: this is a user cancel of a running
            # job, not a failure.  Mark it terminal-cancelled (never cache
            # its half-built result) and hand off to any parked pending.
            with self._lock:
                if job.status == "running":
                    job.status = "cancelled"
                job.done_event.set()
            self._promote_pending_if_any()
            return
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).exception("%s job failed", self._name)
            with self._lock:
                job.status = "error"
                job.error = str(exc) or exc.__class__.__name__
                job.done_event.set()
            self._promote_pending_if_any()
            return

        next_job: AsyncJob | None = None
        next_target: Callable[[AsyncJob], Any] | None = None
        with self._lock:
            if job.is_cancelled and job.status == "running":
                job.status = "cancelled"
            elif job.status == "running":
                job.status = "done"
                self._last_done = job
            job.done_event.set()

            promoted = self._take_pending_locked()
            if promoted is not None:
                next_job, next_target = promoted

        if next_job is not None and next_target is not None:
            self._spawn_thread(next_job, next_target)

    def _take_pending_locked(self) -> tuple[AsyncJob, Callable[[AsyncJob], Any]] | None:
        """Pop the pending slot for promotion, honouring cancellation.

        A cancel that arrived while the job was parked in the pending slot
        must terminate it, not let it run to completion anyway; such a job
        is marked ``cancelled`` and dropped.  Returns the ``(job, target)``
        to spawn, or ``None``.  Caller must hold ``self._lock``.
        """
        job = self._pending
        target = self._pending_target
        self._pending = None
        self._pending_target = None
        if job is None or target is None:
            return None
        if job.is_cancelled:
            job.status = "cancelled"
            job.done_event.set()
            return None
        job.status = "running"
        job.started_at = time.time()
        self._current_id = job.job_id
        return job, target

    def _promote_pending_if_any(self) -> None:
        """Promote the pending slot to running and spawn it, if present.

        Called from the error path so a failed job still hands off to the
        coalesced next request.
        """
        next_job: AsyncJob | None = None
        next_target: Callable[[AsyncJob], Any] | None = None
        with self._lock:
            promoted = self._take_pending_locked()
            if promoted is not None:
                next_job, next_target = promoted
        if next_job is not None and next_target is not None:
            self._spawn_thread(next_job, next_target)

    def _prune_locked(self) -> None:
        if len(self._jobs) <= self._max_history:
            return
        keep: set[str] = set()
        if self._current_id:
            keep.add(self._current_id)
        if self._pending is not None:
            keep.add(self._pending.job_id)
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

    def active_jobs(self) -> list[AsyncJob]:
        """Return the currently running and pending jobs (zero, one, or two)."""
        with self._lock:
            out: list[AsyncJob] = []
            if self._current_id is not None:
                j = self._jobs.get(self._current_id)
                if j is not None and j.status in ("running", "pending"):
                    out.append(j)
            if self._pending is not None and self._pending not in out:
                out.append(self._pending)
            return out

    def reset_for_tests(self) -> None:
        """Cancel any running job and clear all stored state."""
        with self._lock:
            for j in self._jobs.values():
                if j.status in ("running", "pending"):
                    j.cancel()
            self._jobs.clear()
            self._current_id = None
            self._pending = None
            self._pending_target = None
            self._last_done = None


# ---------------------------------------------------------------------- #
# Application-wide singletons
# ---------------------------------------------------------------------- #

#: Background runner for ``/api/learned-sort``.
learned_sort_jobs = JobManager("learned-sort")

#: Background runner for ``/api/eval/train-and-score``.
eval_jobs = JobManager("eval-train-score")

#: Background runner for ``/api/projection/build`` (VTSBrowse UMAP + pyramid).
projection_jobs = JobManager("projection")

#: Logical name → :class:`JobManager` lookup used by ``/api/jobs/active`` to
#: enumerate which (dataset_id, detector_id) pairs currently have background
#: work in flight. The string keys are the public job-type names exposed in
#: the response (consumed by the frontend pulldown for spinner tooltips), so
#: keep them stable across releases.
JOB_MANAGERS: dict[str, JobManager] = {
    "learned-sort": learned_sort_jobs,
    "eval": eval_jobs,
    "projection": projection_jobs,
}


def list_active_pairs() -> list[dict[str, Any]]:
    """Return ``[{dataset_id, detector_id, job_types}, ...]`` for every pair
    with at least one running or pending job across every registered
    :class:`JobManager`.

    Jobs with no ``(dataset_id, detector_id)`` recorded (legacy callers that
    skipped the kwargs, e.g. test fixtures) are dropped - there is no row
    in the pulldown to attach a spinner to without both ids.
    """
    pair_jobs: dict[tuple[str, str], list[str]] = {}
    for job_type, mgr in JOB_MANAGERS.items():
        for job in mgr.active_jobs():
            ds, det = job.dataset_id, job.detector_id
            if not ds or not det:
                continue
            pair_jobs.setdefault((ds, det), []).append(job_type)
    return [
        {"dataset_id": ds, "detector_id": det, "job_types": sorted(set(job_types))}
        for (ds, det), job_types in pair_jobs.items()
    ]


def reset_all_async_jobs_for_tests() -> None:
    """Reset every singleton job manager.  Called from the autouse fixture."""
    for mgr in JOB_MANAGERS.values():
        mgr.reset_for_tests()
