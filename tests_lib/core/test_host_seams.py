"""The host-seam snapshot (:mod:`vtscore.host_seams`).

Every seam is a process global shared by thousands of tests in a handful of
long-lived xdist workers, so a test that installs one leaks it into every test
that follows.  ``tests_shared.state_reset`` captures the seams each conftest
wired at startup and restores that snapshot before each test; these tests hold
that machinery to its contract.

The contract is *snapshot and restore*, never *reset to default*: the app tier
boots with the real ``vtsearch`` wiring installed by ``app.py``, so clearing
the slots would strip the very seams those tests exercise.
"""

from __future__ import annotations

import dataclasses

import pytest

from vtscore.host_seams import HostSeams, capture_host_seams, restore_host_seams

#: The seams the snapshot is responsible for.  Named here rather than derived
#: from ``HostSeams`` so that adding a field without teaching capture/restore
#: about it - or the reverse - fails instead of silently agreeing with itself.
_EXPECTED_SEAMS = {
    "request_context_predicate",
    "dataset_context_resolver",
    "detector_context_resolver",
    "request_user_resolver",
    "core_config_builder",
    "last_embedder_persistence_hook",
    "setting_persisters",
    "achievement_recorders",
}


@pytest.fixture
def restore_afterwards():
    """Put the seams back however this test found them.

    The autouse per-test reset already does this, but these tests mutate the
    seams deliberately and assert on the result, so they restore their own
    starting point rather than depending on the fixture under test.
    """
    saved = capture_host_seams()
    yield saved
    restore_host_seams(saved)


class TestSnapshotCoverage:
    def test_every_seam_has_a_field(self):
        assert {f.name for f in dataclasses.fields(HostSeams)} == _EXPECTED_SEAMS

    def test_capture_reads_the_live_globals(self, restore_afterwards):
        import vtscore.state.core as core

        def _sentinel() -> bool:
            return True

        core.register_request_context_predicate(_sentinel)

        assert capture_host_seams().request_context_predicate is _sentinel

    def test_plugin_families_are_deliberately_excluded(self, restore_afterwards):
        """``register_plugin_family`` looks like a seam but is not one.

        The library registers its own families at import time, so it is a
        plugin extension point with the app as one registrant among several.
        Restoring it would drop the built-ins.
        """
        from vtscore.plugins.inventory import _FAMILIES_REGISTRY

        assert "plugin_families" not in _EXPECTED_SEAMS
        before = dict(_FAMILIES_REGISTRY)
        restore_host_seams(capture_host_seams())
        assert _FAMILIES_REGISTRY == before


class TestRestore:
    def test_a_leaked_single_slot_seam_is_put_back(self, restore_afterwards):
        import vtscore.state.core as core

        original = restore_afterwards.request_context_predicate
        core.register_request_context_predicate(lambda: True)

        restore_host_seams(restore_afterwards)

        assert core._request_context_predicate is original

    def test_a_leaked_keyed_seam_is_put_back(self, restore_afterwards):
        import vtscore.state as state

        state.register_setting_persister("inclusion", lambda v: None)

        restore_host_seams(restore_afterwards)

        assert state._setting_persisters == restore_afterwards.setting_persisters

    def test_keyed_registries_are_restored_in_place(self, restore_afterwards):
        """Callers read the module global directly, so it must not be rebound."""
        import vtscore.achievements_hooks as achievements_hooks
        import vtscore.state as state

        live_persisters = state._setting_persisters
        live_recorders = achievements_hooks._recorders

        restore_host_seams(restore_afterwards)

        assert state._setting_persisters is live_persisters
        assert achievements_hooks._recorders is live_recorders

    def test_a_snapshot_is_not_a_view_of_the_live_registry(self, restore_afterwards):
        """Capturing copies, so a later mutation cannot reach back into it.

        Asserted by identity rather than by the key's absence: when both trees
        collect in one process the app tier's ``app.py`` has already installed
        a persister under every known key, so the snapshot legitimately starts
        non-empty.  What must hold in either tier is that the snapshot still
        holds whatever was registered *at capture time*.
        """
        import vtscore.state as state

        def _sentinel(value):
            return None

        snapshot = capture_host_seams()
        state.register_setting_persister("calibrate_count", _sentinel)

        assert snapshot.setting_persisters.get("calibrate_count") is not _sentinel
        assert state._setting_persisters["calibrate_count"] is _sentinel


class TestKnownSettingKeys:
    def test_an_unknown_key_is_rejected(self):
        from vtscore.state import register_setting_persister

        with pytest.raises(ValueError, match="Unknown setting key"):
            register_setting_persister("nonsense", lambda v: None)

    def test_every_known_key_is_accepted(self, restore_afterwards):
        from vtscore.state import KNOWN_SETTING_KEYS, register_setting_persister

        for key in KNOWN_SETTING_KEYS:
            register_setting_persister(key, lambda v: None)
