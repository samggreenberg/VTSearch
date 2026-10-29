"""Achievement-recording hook seam (app-side wiring; library default = no-op).

The achievement system itself is app-tier: it lives in
:mod:`vtsearch.achievements` and persists counters into the current
user's ``vtsearch.settings`` file.  A handful of library-tier events are
worth crediting though - a vote landing, a dataset finishing its load, a
detector import, a find run - and those events are raised from inside
:mod:`vtscore`.

Rather than have the library import the app (an inverted dependency that
also drags Flask in at call time), each event is dispatched through a
registry of **recorders**.  ``vtsearch/shim/register_app_achievement_recorders()``
installs one per event at app startup; library-only callers leave the
registry empty and :func:`record_achievement` is a no-op.

This mirrors :func:`vtscore.state.register_setting_persister`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

#: Events the app may install a recorder for.  Registering an unknown
#: event is a programming error (typo in the shim wiring), so it raises
#: rather than silently never firing.
KNOWN_EVENTS: frozenset[str] = frozenset({"vote", "dataset_load", "detector_import", "find"})

_recorders: dict[str, Callable[..., Any]] = {}


def register_achievement_recorder(event: str, fn: Callable[..., Any]) -> None:
    """Install the app-side recorder for *event*.

    Called by ``vtsearch/shim`` at app startup.  *event* must be one of
    :data:`KNOWN_EVENTS`.
    """
    if event not in KNOWN_EVENTS:
        raise ValueError(f"Unknown achievement event {event!r}; known events: {sorted(KNOWN_EVENTS)}")
    _recorders[event] = fn


def record_achievement(event: str, *args: Any, **kwargs: Any) -> None:
    """Credit *event* via the registered recorder, or do nothing.

    Arguments are forwarded verbatim to the recorder, so each event's
    signature is whatever the matching ``vtsearch.achievements.record_*``
    function takes.
    """
    fn = _recorders.get(event)
    if fn is not None:
        fn(*args, **kwargs)
