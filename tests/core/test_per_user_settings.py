"""Tests for the two-tier (server + per-user) settings layout.

Covers:

- Server-tier keys persist to ``SETTINGS_PATH`` (the shared file).
- Per-user keys persist to ``<user_data_dir>/user_settings.json``.
- Two users see independent per-user values but the same server-tier
  values.
- :func:`settings.get_user_settings` returns only per-user keys.
- Legacy ``data/settings.json`` files that contain a mix of tier keys
  are migrated on first load.
- Background threads pick up the per-user file matching their
  thread-local user.
"""

from __future__ import annotations

import json

from vtsearch import settings as settings_mod
from vtsearch import settings_store as settings_store_mod
from vtsearch.auth import set_thread_user


class TestTierRouting:
    def test_server_key_written_to_server_file(self, isolated_settings):
        settings_mod.set_saved_datasets_dir("/tmp/abc")
        # Server-tier file gets the value; per-user file does not.
        raw_server = json.loads(isolated_settings._server.read_text())
        assert raw_server["saved_datasets_dir"] == "/tmp/abc"
        # Per-user file is missing or empty of this key.
        if isolated_settings._user.exists():
            raw_user = json.loads(isolated_settings._user.read_text())
            assert "saved_datasets_dir" not in raw_user

    def test_user_key_written_to_user_file(self, isolated_settings):
        settings_mod.set_volume(0.42)
        # Per-user file gets the value; server file does not.
        raw_user = json.loads(isolated_settings._user.read_text())
        assert raw_user["volume"] == 0.42
        if isolated_settings._server.exists():
            raw_server = json.loads(isolated_settings._server.read_text())
            assert "volume" not in raw_server


class TestPerUserIsolation:
    def test_two_users_have_separate_volumes(self, isolated_settings):
        set_thread_user("alice")
        try:
            settings_mod.set_volume(0.25)
            assert settings_mod.get_volume() == 0.25
        finally:
            set_thread_user(None)

        set_thread_user("bob")
        try:
            settings_mod.set_volume(0.75)
            assert settings_mod.get_volume() == 0.75
        finally:
            set_thread_user(None)

        # Alice's volume is untouched by Bob.
        set_thread_user("alice")
        try:
            assert settings_mod.get_volume() == 0.25
        finally:
            set_thread_user(None)

    def test_two_users_share_server_tier(self, isolated_settings):
        set_thread_user("alice")
        try:
            settings_mod.set_saved_datasets_dir("/tmp/shared")
        finally:
            set_thread_user(None)

        set_thread_user("bob")
        try:
            assert str(settings_mod.get_saved_datasets_dir()) == "/tmp/shared"
        finally:
            set_thread_user(None)

    def test_get_user_settings_excludes_server_keys(self, isolated_settings):
        settings_mod.set_volume(0.5)
        settings_mod.set_saved_datasets_dir("/tmp/zzz")
        out = settings_mod.get_user_settings()
        assert "volume" in out
        # Server-tier keys are filtered out.
        assert "saved_datasets_dir" not in out
        # autofind_detectors is now a per-user key, so it IS surfaced here.
        assert "autofind_detectors" in out


class TestUnmigratedFiles:
    """Old on-disk shapes are *ignored*, not migrated.

    VTSearch used to split a pre-refactor flat ``settings.json`` across the
    two tiers on first load, rewriting both files. ``CLAUDE.md``'s
    backwards-compatibility policy forbids migration shims for saved data, so
    that one-shot rewrite is gone (issue #3413): a value the tier's model
    can't accept is dropped from the in-memory cache on load and the pydantic
    default applies, while the file on disk is left exactly as the user wrote
    it.
    """

    def test_legacy_mixed_file_ignores_user_keys(self, tmp_path, monkeypatch):
        """Per-user keys in the server file are inert - no migration, no rewrite."""
        from vtsearch import settings as settings_mod

        legacy = {
            "saved_datasets_dir": "/tmp/legacy",  # server tier
            "volume": 0.33,  # user tier - ignored where it sits
            "theme": "light",  # user tier - ignored where it sits
        }
        server_path = tmp_path / "settings.json"
        server_path.write_text(json.dumps(legacy))
        monkeypatch.setattr(settings_mod, "SETTINGS_PATH", server_path)
        settings_mod.set_user_data_dir_override(tmp_path / "users")
        settings_mod.reset()
        try:
            # The server-tier key still applies.
            assert str(settings_mod.get_saved_datasets_dir()) == "/tmp/legacy"
            # The stranded per-user keys do not: defaults win.
            assert settings_mod.get_volume() == 1.0
            assert settings_mod.get_theme() == "system"

            # Nothing on disk was rewritten, and no user file was conjured.
            assert json.loads(server_path.read_text()) == legacy
            assert not (tmp_path / "users" / "default" / "user_settings.json").exists()
        finally:
            settings_mod.set_user_data_dir_override(None)
            settings_mod.reset()

    def test_autofind_keys_in_server_file_still_read_through(self, tmp_path, monkeypatch):
        """The one deliberate tier exception survives: the Auto-Find trio.

        The CLI's documented ``--settings`` flat-file workflow puts
        ``autofind_detectors`` in the server file and expects the ``default``
        user to see it (see ``_DEFAULT_USER_FALLBACK_KEYS``). That read-through
        is a live feature, not a migration, so it is *not* sanitized away.
        """
        from vtsearch import settings as settings_mod

        server_path = tmp_path / "settings.json"
        server_path.write_text(json.dumps({"autofind_detectors": ["Dog Barks"]}))
        monkeypatch.setattr(settings_mod, "SETTINGS_PATH", server_path)
        settings_mod.set_user_data_dir_override(tmp_path / "users")
        settings_mod.reset()
        try:
            assert settings_mod.get_autofind_detectors() == ["Dog Barks"]
            assert settings_mod.get_all()["autofind_detectors"] == ["Dog Barks"]
        finally:
            settings_mod.set_user_data_dir_override(None)
            settings_mod.reset()

    def test_bad_autofind_value_in_server_file_is_dropped(self, tmp_path, monkeypatch):
        """The read-through is an exception to the *tier* rule, not to the
        value check: the trio is validated against ``UserSettings``, the model
        that actually reads it, so a hand-edit of the wrong shape still falls
        back to the default rather than reaching a caller."""
        from vtsearch import settings as settings_mod

        server_path = tmp_path / "settings.json"
        server_path.write_text(json.dumps({"autofind_detectors": "Dog Barks", "autofind_exporter": "json"}))
        monkeypatch.setattr(settings_mod, "SETTINGS_PATH", server_path)
        settings_mod.set_user_data_dir_override(tmp_path / "users")
        settings_mod.reset()
        try:
            assert settings_mod.get_autofind_detectors() == []  # bare string, not a list
            assert settings_mod.get_autofind_exporter() == "json"  # the valid sibling survives
        finally:
            settings_mod.set_user_data_dir_override(None)
            settings_mod.reset()

    def test_stranded_server_key_in_user_file_does_not_shadow(self, tmp_path, monkeypatch):
        """A server-tier key left in a user file must not shadow the real value.

        ``browse_signpost_vocab`` was per-user before it became a server-tier
        operator setting, so user files written before the move still carry a
        copy. ``get_all`` layers the user cache over the server one, so the
        stale copy used to be reported by GET while projection builds (which
        read the accessor) used the operator's. It is dropped on load now.
        """
        from vtsearch import settings as settings_mod

        server_path = tmp_path / "settings.json"
        server_path.write_text(json.dumps({"browse_signpost_vocab": {"image": ["operator"]}}))
        user_path = tmp_path / "users" / "default" / "user_settings.json"
        user_path.parent.mkdir(parents=True, exist_ok=True)
        user_path.write_text(json.dumps({"browse_signpost_vocab": {"image": ["stale"]}}))
        monkeypatch.setattr(settings_mod, "SETTINGS_PATH", server_path)
        settings_mod.set_user_data_dir_override(tmp_path / "users")
        settings_mod.reset()
        try:
            assert settings_mod.get_all()["browse_signpost_vocab"] == {"image": ["operator"]}
            assert settings_mod.get_browse_signpost_vocab() == {"image": ["operator"]}
            # The user's file itself is left alone - the drop is read-side only.
            assert json.loads(user_path.read_text()) == {"browse_signpost_vocab": {"image": ["stale"]}}
        finally:
            settings_mod.set_user_data_dir_override(None)
            settings_mod.reset()

    def test_unrelated_write_does_not_delete_dropped_keys(self, tmp_path, monkeypatch):
        """Sanitizing is a read-side view: a later setter must not eat the file.

        Dropping a value the model rejects has to stay non-destructive, or the
        migration we just deleted would come back in another form - silently,
        on whatever unrelated setting the user happened to change next.
        """
        from vtsearch import settings as settings_mod

        server_path = tmp_path / "settings.json"
        server_path.write_text(json.dumps({}))
        user_path = tmp_path / "users" / "default" / "user_settings.json"
        user_path.parent.mkdir(parents=True, exist_ok=True)
        user_path.write_text(json.dumps({"show_animations": "True", "hand_added": 7}))
        monkeypatch.setattr(settings_mod, "SETTINGS_PATH", server_path)
        settings_mod.set_user_data_dir_override(tmp_path / "users")
        settings_mod.reset()
        try:
            assert settings_mod.get_show_animations() == "show"  # default applies
            settings_mod.set_volume(0.42)
            on_disk = json.loads(user_path.read_text())
            assert on_disk["show_animations"] == "True"  # untouched
            assert on_disk["hand_added"] == 7  # extra keys still round-trip
            assert on_disk["volume"] == 0.42
        finally:
            settings_mod.set_user_data_dir_override(None)
            settings_mod.reset()


class TestServerLoadCrossProcessLock:
    """The first server-tier load runs under the cross-process ``file_lock``
    (file-lock -> settings-lock order), and concurrent first loads must not
    deadlock or disagree."""

    def _seed(self, tmp_path, monkeypatch):
        server_path = tmp_path / "settings.json"
        server_path.write_text(json.dumps({"saved_datasets_dir": "/tmp/legacy"}))
        monkeypatch.setattr(settings_mod, "SETTINGS_PATH", server_path)
        settings_mod.set_user_data_dir_override(tmp_path / "users")
        settings_mod.reset()
        return server_path

    def test_server_load_takes_file_lock(self, tmp_path, monkeypatch):
        import contextlib
        from pathlib import Path

        self._seed(tmp_path, monkeypatch)

        locked: list[str] = []
        real_file_lock = settings_store_mod.file_lock

        @contextlib.contextmanager
        def recording_file_lock(path):
            locked.append(Path(path).name)
            with real_file_lock(path):
                yield

        monkeypatch.setattr(settings_store_mod, "file_lock", recording_file_lock)
        try:
            assert str(settings_mod.get_saved_datasets_dir()) == "/tmp/legacy"
            assert "settings.json" in locked
        finally:
            settings_mod.set_user_data_dir_override(None)
            settings_mod.reset()

    def test_concurrent_first_load_reads_file_once(self, tmp_path, monkeypatch):
        """Four threads racing the first load must not deadlock, and only the
        winner of the file lock actually reads the file."""
        import threading

        self._seed(tmp_path, monkeypatch)

        calls = {"n": 0}
        real_load = settings_store_mod._load_path
        server_name = "settings.json"

        def counting_load(path):
            if path.name == server_name:
                calls["n"] += 1
            return real_load(path)

        monkeypatch.setattr(settings_store_mod, "_load_path", counting_load)
        try:
            barrier = threading.Barrier(4)
            results: list[str] = []
            errors: list[Exception] = []

            def worker():
                try:
                    barrier.wait(timeout=5)
                    results.append(str(settings_mod.get_saved_datasets_dir()))
                except Exception as exc:  # pragma: no cover - surfaced below
                    errors.append(exc)

            threads = [threading.Thread(target=worker) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            assert not errors, f"workers raised: {errors}"
            for t in threads:
                assert not t.is_alive(), "thread hung (likely deadlock)"
            assert results == ["/tmp/legacy"] * 4
            # The first thread to win the server file lock loads; the rest find
            # the cache already published and skip the read entirely.
            assert calls["n"] == 1
        finally:
            settings_mod.set_user_data_dir_override(None)
            settings_mod.reset()


class TestBackgroundThreadPropagation:
    def test_thread_local_user_routes_writes(self, isolated_settings):
        """A background thread's settings writes resolve to its thread-local user."""
        import threading

        results: dict = {}

        def worker(name: str, value: float) -> None:
            set_thread_user(name)
            try:
                settings_mod.set_volume(value)
                results[name] = settings_mod.get_volume()
            finally:
                set_thread_user(None)

        t1 = threading.Thread(target=worker, args=("carol", 0.11))
        t2 = threading.Thread(target=worker, args=("dave", 0.99))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["carol"] == 0.11
        assert results["dave"] == 0.99

        # Each user's file holds only their value.
        carol_path = isolated_settings._user.parent.parent / "carol" / "user_settings.json"
        dave_path = isolated_settings._user.parent.parent / "dave" / "user_settings.json"
        assert json.loads(carol_path.read_text())["volume"] == 0.11
        assert json.loads(dave_path.read_text())["volume"] == 0.99


class TestExportFilter:
    def test_get_user_settings_yields_user_view(self, isolated_settings):
        """get_user_settings() returns this user's data; never another user's."""
        set_thread_user("alice")
        try:
            settings_mod.set_volume(0.3)
        finally:
            set_thread_user(None)

        set_thread_user("bob")
        try:
            settings_mod.set_volume(0.7)
            out = settings_mod.get_user_settings()
            assert out["volume"] == 0.7
        finally:
            set_thread_user(None)
