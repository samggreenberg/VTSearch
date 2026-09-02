"""Direct unit tests for :mod:`vtsearch.settings_store` primitives.

The settings routes exercise the store end-to-end, but its low-level file
I/O and cross-process sync-marker helpers are worth pinning directly: they are
pure, path-keyed functions with several error-swallowing branches (unreadable
file, malformed JSON, non-serialisable token) that route-level tests don't
reliably hit.
"""

from __future__ import annotations

import json

from vtsearch import settings_store as store


# ---------------------------------------------------------------------------
# _load_path / _atomic_write
# ---------------------------------------------------------------------------


class TestLoadPath:
    def test_missing_file_returns_empty(self, tmp_path):
        assert store._load_path(tmp_path / "nope.json") == {}

    def test_reads_valid_dict(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text(json.dumps({"a": 1, "b": "two"}), encoding="utf-8")
        assert store._load_path(p) == {"a": 1, "b": "two"}

    def test_malformed_json_returns_empty(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text("{not valid json", encoding="utf-8")
        assert store._load_path(p) == {}

    def test_non_dict_json_returns_empty(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        assert store._load_path(p) == {}


class TestAtomicWrite:
    def test_round_trips_via_load_path(self, tmp_path):
        p = tmp_path / "s.json"
        store._atomic_write(p, {"x": 42})
        assert store._load_path(p) == {"x": 42}

    def test_creates_parent_directories(self, tmp_path):
        p = tmp_path / "deep" / "nested" / "s.json"
        store._atomic_write(p, {"ok": True})
        assert p.exists()
        assert store._load_path(p) == {"ok": True}

    def test_leaves_no_temp_file_behind(self, tmp_path):
        p = tmp_path / "s.json"
        store._atomic_write(p, {"x": 1})
        # The per-writer "<name>.<pid>.<uuid>.tmp" scratch file is renamed into
        # place, so no ``.tmp`` sibling should survive. (Other files in tmp_path
        # belong to the autouse isolated_settings fixture, so scope to .tmp.)
        assert list(tmp_path.glob("s.json.*.tmp")) == []

    def test_overwrite_replaces_content(self, tmp_path):
        p = tmp_path / "s.json"
        store._atomic_write(p, {"v": 1})
        store._atomic_write(p, {"v": 2})
        assert store._load_path(p) == {"v": 2}


# ---------------------------------------------------------------------------
# sync marker helpers
# ---------------------------------------------------------------------------


class TestSyncMarker:
    def test_marker_path_is_sibling(self, tmp_path):
        user = tmp_path / "user.json"
        assert store._sync_marker_path(user) == tmp_path / "user.json.syncmark"

    def test_read_absent_marker_returns_none(self, tmp_path):
        assert store._read_sync_marker(tmp_path / "user.json") is None

    def test_write_then_read_round_trips(self, tmp_path):
        user = tmp_path / "user.json"
        store._write_sync_marker(user, 12345)
        assert store._read_sync_marker(user) == 12345

    def test_write_string_token(self, tmp_path):
        user = tmp_path / "user.json"
        store._write_sync_marker(user, "etag-abc")
        assert store._read_sync_marker(user) == "etag-abc"

    def test_write_none_writes_nothing(self, tmp_path):
        user = tmp_path / "user.json"
        store._write_sync_marker(user, None)
        assert not store._sync_marker_path(user).exists()
        assert store._read_sync_marker(user) is None

    def test_write_non_serialisable_is_swallowed(self, tmp_path):
        user = tmp_path / "user.json"
        # An object json can't encode must not raise — the marker is a hint.
        store._write_sync_marker(user, object())
        assert store._read_sync_marker(user) is None

    def test_read_malformed_marker_returns_none(self, tmp_path):
        user = tmp_path / "user.json"
        store._sync_marker_path(user).write_text("{bad json", encoding="utf-8")
        assert store._read_sync_marker(user) is None

    def test_read_non_dict_marker_returns_none(self, tmp_path):
        user = tmp_path / "user.json"
        store._sync_marker_path(user).write_text(json.dumps([1, 2]), encoding="utf-8")
        assert store._read_sync_marker(user) is None

    def test_read_marker_without_version_key_returns_none(self, tmp_path):
        user = tmp_path / "user.json"
        store._sync_marker_path(user).write_text(json.dumps({"other": 1}), encoding="utf-8")
        assert store._read_sync_marker(user) is None


# ---------------------------------------------------------------------------
# locking primitives
# ---------------------------------------------------------------------------


class TestLocking:
    def test_path_lock_is_stable_per_path(self, tmp_path):
        p = tmp_path / "s.json"
        assert store._path_lock_for(p) is store._path_lock_for(p)

    def test_distinct_paths_get_distinct_locks(self, tmp_path):
        a = store._path_lock_for(tmp_path / "a.json")
        b = store._path_lock_for(tmp_path / "b.json")
        assert a is not b

    def test_file_lock_acquires_and_releases(self, tmp_path):
        p = tmp_path / "s.json"
        with store.file_lock(p):
            store._atomic_write(p, {"held": True})
        # Re-acquiring after release must not deadlock.
        with store.file_lock(p):
            assert store._load_path(p) == {"held": True}
