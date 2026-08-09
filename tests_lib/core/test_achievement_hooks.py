"""The library-tier achievement seam (:mod:`vtscore.achievements_hooks`).

Library code raises achievement events; the app installs the recorders
that persist them.  With no app wired up, every event must be a silent
no-op rather than an import of :mod:`vtsearch.achievements`.
"""

from __future__ import annotations

import pytest

from vtscore import achievements_hooks
from vtscore.achievements_hooks import KNOWN_EVENTS, record_achievement, register_achievement_recorder


@pytest.fixture(autouse=True)
def _clean_registry():
    saved = dict(achievements_hooks._recorders)
    achievements_hooks._recorders.clear()
    try:
        yield
    finally:
        achievements_hooks._recorders.clear()
        achievements_hooks._recorders.update(saved)


class TestRecordAchievement:
    def test_unregistered_event_is_a_no_op(self):
        for event in KNOWN_EVENTS:
            record_achievement(event, "detector-1", media_type="audio")

    def test_registered_recorder_receives_the_arguments_verbatim(self):
        calls: list[tuple] = []
        register_achievement_recorder("vote", lambda *a, **k: calls.append((a, k)))

        record_achievement("vote", "detector-1", media_type="audio", count_streak=False)

        assert calls == [(("detector-1",), {"media_type": "audio", "count_streak": False})]

    def test_only_the_named_event_fires(self):
        fired: list[str] = []
        register_achievement_recorder("find", lambda *a, **k: fired.append("find"))

        record_achievement("dataset_load", "server_folder")

        assert fired == []

    def test_unknown_event_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown achievement event"):
            register_achievement_recorder("nonsense", lambda: None)


class TestVoteCredit:
    def test_casting_a_vote_dispatches_the_vote_event(self):
        from vtscore.state.votes import _record_vote_locked

        calls: list[tuple] = []
        register_achievement_recorder("vote", lambda *a, **k: calls.append((a, k)))

        _record_vote_locked(count_streak=False)

        assert len(calls) == 1
        assert calls[0][1]["count_streak"] is False
