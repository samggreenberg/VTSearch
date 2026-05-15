"""Click-time tracking for vote ordering."""

from __future__ import annotations

from vtsearch.state.core import _state_lock, vote_click_times

# Import context-aware accessors for scalar state.
from vtsearch.state.core import _get_click_counter, _set_click_counter


def assign_click_time(media_id: int) -> int:
    """Assign the next click-time ordinal to a media and return it.

    Each call increments the global counter so click-times are unique and
    monotonically increasing.
    """
    with _state_lock:
        new_val = _get_click_counter() + 1
        _set_click_counter(new_val)
        vote_click_times[media_id] = new_val
        return new_val


def remove_click_time(media_id: int) -> None:
    """Remove the click-time entry for a media (e.g. when unlabelling)."""
    with _state_lock:
        vote_click_times.pop(media_id, None)


def get_vote_click_times() -> dict[int, int]:
    """Return a copy of the click-time mapping."""
    with _state_lock:
        return vote_click_times.copy()
