"""Click-time tracking for vote ordering."""

from __future__ import annotations

from vtsearch.utils.state_core import _state_lock, vote_click_times

# Import _click_counter via the module so we can mutate the global.
import vtsearch.utils.state_core as _core


def assign_click_time(media_id: int) -> int:
    """Assign the next click-time ordinal to a media and return it.

    Each call increments the global counter so click-times are unique and
    monotonically increasing.
    """
    with _state_lock:
        _core._click_counter += 1
        vote_click_times[media_id] = _core._click_counter
        return _core._click_counter


def remove_click_time(media_id: int) -> None:
    """Remove the click-time entry for a media (e.g. when unlabelling)."""
    with _state_lock:
        vote_click_times.pop(media_id, None)


def get_vote_click_times() -> dict[int, int]:
    """Return a copy of the click-time mapping."""
    with _state_lock:
        return vote_click_times.copy()
