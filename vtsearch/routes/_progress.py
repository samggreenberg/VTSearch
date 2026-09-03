"""Parking the shared progress trackers at idle.

The sort and find progress bars are two process-wide singletons broadcast to
every SSE client, so whichever route ran last owns leaving its tracker parked.
The two idle helpers live together because they are the same concern applied to
the two trackers -- not because either can be expressed in terms of the other.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from werkzeug.exceptions import HTTPException

from vtscore.concurrency.progress import update_find_progress, update_sort_progress


def sort_idle() -> None:
    """Reset the sort progress bar to idle, clearing the whole-job step frame."""
    update_sort_progress("idle", "", step=None, total_steps=None)


def find_idle() -> None:
    """Park the shared ``find_progress`` tracker at idle, clearing the step frame.

    The find-side sibling of :func:`sort_idle`. Every
    scoring route (``/api/find``, ``/api/find-label``, ``/api/auto-detect``)
    reports through the one process-wide ``find_progress`` singleton and pushes
    it to every SSE client, so whichever route ran last owns leaving it parked
    at ``"idle"``.
    """
    update_find_progress("idle", "", step=None, total_steps=None)


@contextmanager
def find_idle_on_crash(recorder: Any = None) -> Iterator[None]:
    """Park ``find_progress`` at idle when the wrapped body dies unexpectedly.

    Every *anticipated* exit from a scoring route resets the tracker itself:
    ``_abort_find``, ``_abort_if_find_cancelled``, and the success path all end
    with an idle update. An unhandled exception (say an embedding-dimension
    mismatch raising ``RuntimeError`` mid-scoring) takes none of those paths, so
    without this guard the request 500s through the global error handler while
    the shared singleton stays at ``"running"`` on whatever step it died on —
    broadcast to every SSE client, and only cleared by the next Find.

    *recorder* (a :func:`vtscore.timing.record_task` handle, when the route runs
    one) is closed first, as a failed run: the idle update would otherwise trip
    its ``auto_finish`` hook and bank a crashed run's partial phase timings as a
    good cost sample.  ``abort()`` is left alone — it already parked the tracker
    and closed the recorder, and flask-smorest renders its envelope unchanged.
    """
    try:
        yield
    except HTTPException:
        raise
    except Exception:
        if recorder is not None:
            recorder.finish(ok=False)
        find_idle()
        raise
