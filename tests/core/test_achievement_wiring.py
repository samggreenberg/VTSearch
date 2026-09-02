"""The app must actually wire itself into the library's achievement seam.

`vtscore` raises achievement events through
:func:`vtscore.achievements_hooks.record_achievement`, and the recorders
that persist them are installed by
``vtsearch.shim.register_app_achievement_recorders()`` at startup.  That
indirection buys the library its Flask-free layering, but it trades a
*loud* failure for a *silent* one: before the seam, a broken import
raised; now a missing registration just means the event quietly credits
nothing, and every unit test that calls ``achievements.record_vote()``
directly would still pass.

These tests are the thing that notices.  The first pins the wiring
itself; the second drives a real vote through `vtscore.state` and checks
the counter moved, so the whole chain - library event → hook → app
recorder → per-user settings - is exercised end to end rather than
assumed.
"""

from __future__ import annotations

import pytest

from vtsearch import achievements
from vtscore import achievements_hooks
from vtscore.achievements_hooks import KNOWN_EVENTS


class TestStartupWiring:
    def test_app_registers_a_recorder_for_every_known_event(self):
        """Importing ``app`` must leave no library event unhandled.

        A new event added to :data:`KNOWN_EVENTS` without a matching
        ``register_achievement_recorder`` call in the shim fails here
        rather than silently never crediting.
        """
        assert set(achievements_hooks._recorders) == set(KNOWN_EVENTS)

    @pytest.mark.parametrize("event", sorted(KNOWN_EVENTS))
    def test_each_recorder_targets_the_app_implementation(self, event):
        """Each hook must reach ``vtsearch.achievements``, not a stub."""
        recorder = achievements_hooks._recorders[event]
        assert callable(recorder)
        assert recorder.__module__ == "vtsearch.shim"


class TestVoteCreditsEndToEnd:
    def test_a_real_vote_moves_the_counter_through_the_hook(self, isolated_settings):
        """Drive the library's vote path and assert the app counted it.

        This is the test that would have caught a dropped
        ``register_app_achievement_recorders()`` call: it never touches
        ``achievements.record_vote`` directly, so the credit can only
        arrive via the hook the shim installed.
        """
        from vtscore.state import set_vote
        from vtsearch.state import medias

        media_id = next(iter(medias))
        before = achievements.get_full_state()

        set_vote(media_id, "good")
        try:
            after = achievements.get_full_state()
        finally:
            set_vote(media_id, "none")

        def _counter(state: dict) -> int:
            return next(a["counter"] for a in state["achievements"] if a["id"] == "votes_cast")

        assert _counter(after) == _counter(before) + 1
