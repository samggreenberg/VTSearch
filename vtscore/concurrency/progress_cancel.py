"""Cancelling dataset loads, and reporting honestly whether anything stopped.

:func:`cancel_dataset_progress` and :func:`cancel_dataset_task` set the
cooperative-cancellation flag on the ``loading_tasks`` rows, then wait a grace
period and classify each row by whether a living worker acted on it.

Import these names from :mod:`vtscore.concurrency.progress`, which re-exports
every public one.
"""

import time
from typing import Any

from vtscore.concurrency.loading_tracker import loading_tasks

#: How long :func:`cancel_dataset_progress` waits for a worker to act on the
#: flag before reporting what it saw. Cooperative cancellation normally lands
#: within one progress tick (well under a second); a worker that has not
#: finished by then is classified by whether its thread is still alive, not by
#: the clock, so this only has to be long enough to avoid calling a prompt
#: cancel "pending".
CANCEL_ACK_GRACE_SECONDS = 2.0

#: Poll interval while waiting for acknowledgement.
_CANCEL_POLL_SECONDS = 0.02


def _cancel_acknowledged(target: str) -> bool:
    """Return ``True`` once *target* has reached a terminal state."""
    if loading_tasks.is_finished(target):
        return True
    tracker = loading_tasks.get_tracker(target)
    # A task that has been pruned out of the tracker is as stopped as it gets.
    return tracker is None or tracker.get()["status"] == "idle"


def _park_unresponsive(target: str) -> None:
    """Clear a progress entry that claims work no living thread is doing.

    Refusing to lie about the cancel is only half the fix: the phantom that
    made the refusal necessary is still on screen, and it will still be there
    tomorrow. Nothing is going to clear it — that is what "no worker" means —
    so the request that proved it stale clears it.
    """
    tracker = loading_tasks.get_tracker(target)
    if tracker is not None:
        tracker.update(
            "idle",
            "",
            0,
            0,
            error="Abandoned: no worker was running this operation.",
        )
    loading_tasks.mark_finished(target)


def _cancel_report(targets: list[str], grace_seconds: float) -> dict[str, Any]:
    """Wait out the grace period and classify every target in *targets*."""
    report: dict[str, Any] = {
        "ok": False,
        "targets": list(targets),
        "acknowledged": [],
        "pending": [],
        "unresponsive": [],
        "message": "",
    }
    if not targets:
        report["message"] = "Nothing to cancel: no dataset operation is running."
        return report

    deadline = time.monotonic() + max(0.0, grace_seconds)
    outstanding = list(targets)
    while True:
        outstanding = [t for t in outstanding if not _cancel_acknowledged(t)]
        if not outstanding or time.monotonic() >= deadline:
            break
        time.sleep(_CANCEL_POLL_SECONDS)

    still = set(outstanding)
    report["acknowledged"] = [t for t in targets if t not in still]
    for target in outstanding:
        alive = loading_tasks.worker_alive(target)
        report["pending" if alive else "unresponsive"].append(target)
    for target in report["unresponsive"]:
        _park_unresponsive(target)

    report["ok"] = bool(report["acknowledged"] or report["pending"])
    stopped = len(report["acknowledged"])
    stopping = len(report["pending"])
    stale = len(report["unresponsive"])
    if report["ok"]:
        parts = []
        if stopped:
            parts.append(f"{stopped} stopped")
        if stopping:
            parts.append(f"{stopping} still stopping")
        if stale:
            parts.append(f"{stale} stale entr{'y' if stale == 1 else 'ies'} cleared")
        report["message"] = "Cancelled: " + ", ".join(parts) + "."
    else:
        report["message"] = (
            f"Nothing acknowledged the cancel: {stale} progress "
            f"entr{'y' if stale == 1 else 'ies'} claimed work with no worker running it. "
            "The operation had already finished or died; the stale progress has been cleared."
        )
    return report


def cancel_dataset_progress(grace_seconds: float = CANCEL_ACK_GRACE_SECONDS) -> dict[str, Any]:
    """Signal the current dataset operation(s) to cancel, and report the outcome.

    Cancels every active per-task loading tracker, then waits up to
    *grace_seconds* for someone to act on the flag.

    Cancellation is cooperative — :meth:`ProgressTracker.cancel` sets an event
    and :meth:`ProgressTracker.check_cancelled` has to be *reached* by a
    running thread — so setting the flag says nothing about whether anything
    will happen.  Reporting success unconditionally therefore answered the one
    question a stuck-looking import raises ("is anything actually running?")
    with the same word either way (#3167).

    Returns a report with:

    ``targets``
        every task id that claimed to be working when the cancel arrived.
    ``acknowledged``
        targets that reached a terminal state within the grace period.
    ``pending``
        targets still running, with a live worker thread that will observe the
        flag; the cancel was delivered, it just has not landed yet.
    ``unresponsive``
        targets whose progress claimed work no live thread was doing.  These
        are stale trackers, not running jobs; they are parked at ``idle`` so
        the phantom does not outlive the request that exposed it.
    ``ok``
        ``True`` when at least one target acknowledged or is pending — i.e.
        when the cancel actually reached something.
    """
    targets = loading_tasks.active_task_ids()
    loading_tasks.cancel_all()
    return _cancel_report(targets, grace_seconds)


def cancel_dataset_task(task_id: str, grace_seconds: float = CANCEL_ACK_GRACE_SECONDS) -> dict[str, Any] | None:
    """Cancel one loading task and report the outcome, or ``None`` if unknown.

    Same contract as :func:`cancel_dataset_progress`, narrowed to a single
    task: a task whose tracker still claims work but whose worker thread has
    exited is reported as ``unresponsive`` rather than cancelled.
    """
    tracker = loading_tasks.get_tracker(task_id)
    if tracker is None:
        return None
    targets = [task_id] if tracker.get()["status"] != "idle" else []
    tracker.cancel()
    return _cancel_report(targets, grace_seconds)
