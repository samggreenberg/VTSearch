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

import app as app_module  # noqa: F401  (triggers conftest media init)
from vtsearch import settings as settings_mod
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


class TestLegacyMigration:
    def test_legacy_mixed_file_migrates_user_keys(self, tmp_path, monkeypatch):
        """A pre-refactor settings.json with both tiers must be split on load."""
        from vtsearch import settings as settings_mod

        legacy = {
            "saved_datasets_dir": "/tmp/legacy",  # server tier
            "volume": 0.33,  # user tier
            "theme": "light",  # user tier
        }
        server_path = tmp_path / "settings.json"
        server_path.write_text(json.dumps(legacy))
        monkeypatch.setattr(settings_mod, "SETTINGS_PATH", server_path)
        settings_mod.set_user_data_dir_override(tmp_path / "users")
        settings_mod.reset()
        try:
            # Triggers _ensure_server_loaded + migration.
            assert settings_mod.get_volume() == 0.33
            assert settings_mod.get_theme() == "light"
            assert str(settings_mod.get_saved_datasets_dir()) == "/tmp/legacy"

            # Server file now contains only server-tier keys.
            remaining = json.loads(server_path.read_text())
            assert "volume" not in remaining
            assert "theme" not in remaining
            assert remaining["saved_datasets_dir"] == "/tmp/legacy"

            # Default user's file contains only per-user keys.
            user_path = tmp_path / "users" / "default" / "user_settings.json"
            assert user_path.exists()
            user_data = json.loads(user_path.read_text())
            assert user_data["volume"] == 0.33
            assert user_data["theme"] == "light"
            assert "saved_datasets_dir" not in user_data
        finally:
            settings_mod.set_user_data_dir_override(None)
            settings_mod.reset()

    def test_user_write_failure_keeps_cache_and_disk_consistent(self, tmp_path, monkeypatch):
        """If the per-user write fails, the in-memory cache and on-disk
        server file must stay aligned (no half-migrated state)."""
        from vtsearch import settings as settings_mod

        legacy = {
            "saved_datasets_dir": "/tmp/legacy",
            "volume": 0.33,
            "theme": "light",
        }
        server_path = tmp_path / "settings.json"
        server_path.write_text(json.dumps(legacy))
        monkeypatch.setattr(settings_mod, "SETTINGS_PATH", server_path)
        settings_mod.set_user_data_dir_override(tmp_path / "users")
        settings_mod.reset()

        real_atomic_write = settings_mod._atomic_write
        user_settings_path = tmp_path / "users" / "default" / "user_settings.json"

        def failing_atomic_write(path, data):
            if path == user_settings_path:
                raise OSError("simulated user-file write failure")
            return real_atomic_write(path, data)

        monkeypatch.setattr(settings_mod, "_atomic_write", failing_atomic_write)
        try:
            # Triggers migration; user-file write raises and is swallowed.
            settings_mod.get_saved_datasets_dir()

            # Server file on disk is untouched (legacy keys still present).
            on_disk = json.loads(server_path.read_text())
            assert on_disk == legacy

            # In-memory server cache matches disk (legacy keys still there).
            assert settings_mod._server_cache is not None
            assert settings_mod._server_cache.get("volume") == 0.33
            assert settings_mod._server_cache.get("theme") == "light"

            # No phantom user file was created.
            assert not user_settings_path.exists()
        finally:
            settings_mod.set_user_data_dir_override(None)
            settings_mod.reset()

    def test_server_rewrite_failure_keeps_cache_and_disk_consistent(self, tmp_path, monkeypatch):
        """If the server-file rewrite fails after the user write succeeds,
        the in-memory ``_server_cache`` must not have legacy keys popped;
        otherwise it would silently disagree with the on-disk server file."""
        from vtsearch import settings as settings_mod

        legacy = {
            "saved_datasets_dir": "/tmp/legacy",
            "volume": 0.33,
            "theme": "light",
        }
        server_path = tmp_path / "settings.json"
        server_path.write_text(json.dumps(legacy))
        monkeypatch.setattr(settings_mod, "SETTINGS_PATH", server_path)
        settings_mod.set_user_data_dir_override(tmp_path / "users")
        settings_mod.reset()

        real_atomic_write = settings_mod._atomic_write
        user_settings_path = tmp_path / "users" / "default" / "user_settings.json"

        def failing_atomic_write(path, data):
            if path == server_path:
                raise OSError("simulated server-file write failure")
            return real_atomic_write(path, data)

        monkeypatch.setattr(settings_mod, "_atomic_write", failing_atomic_write)
        try:
            # Triggers migration; user write succeeds, server rewrite raises.
            settings_mod.get_saved_datasets_dir()

            # User file was written successfully.
            assert user_settings_path.exists()
            user_data = json.loads(user_settings_path.read_text())
            assert user_data["volume"] == 0.33
            assert user_data["theme"] == "light"

            # Server file on disk still has legacy keys (rewrite failed).
            on_disk = json.loads(server_path.read_text())
            assert on_disk == legacy

            # Crucially, in-memory cache matches disk; legacy keys NOT popped.
            assert settings_mod._server_cache is not None
            assert settings_mod._server_cache.get("volume") == 0.33
            assert settings_mod._server_cache.get("theme") == "light"
            assert settings_mod._server_cache.get("saved_datasets_dir") == "/tmp/legacy"
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
