"""Click-time tracking for vote ordering.

Operates on the active :class:`DetectorContext` (resolved per call) so the
library has no implicit dependency on the app-side proxy view.  See Phase 3
of ``docs/plans/extract-library.md``.
"""

from __future__ import annotations

from vtsearch.state.core import _state_lock, get_active_detector_context


def assign_click_time(media_id: int) -> int:
    """Assign the next click-time ordinal to a media and return it.

    Each call increments the active detector's counter so click-times are
    unique and monotonically increasing within that detector's session.
    """
    with _state_lock:
        ctx = get_active_detector_context()
        ctx.click_counter += 1
        ctx.vote_click_times[media_id] = ctx.click_counter
        return ctx.click_counter


def remove_click_time(media_id: int) -> None:
    """Remove the click-time entry for a media (e.g. when unlabelling)."""
    with _state_lock:
        get_active_detector_context().vote_click_times.pop(media_id, None)


def get_vote_click_times() -> dict[int, int]:
    """Return a copy of the click-time mapping."""
    with _state_lock:
        return get_active_detector_context().vote_click_times.copy()
