"""Tests for the server file-path validation utility.

Covers:
- validate_server_filepath: rejects path traversal, accepts safe paths
- get_file_access_base_dir: returns None for DefaultLoginProvider, user dir otherwise
- Multi-user file access restrictions via API endpoints
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vtscore.security.path_validation import get_file_access_base_dir, validate_server_filepath


class TestValidateServerFilepath:
    """Direct unit tests for validate_server_filepath (no Flask client)."""

    def test_relative_path_within_base(self, tmp_path):
        result = validate_server_filepath("sub/file.json", base_dir=tmp_path)
        assert result == (tmp_path / "sub" / "file.json").resolve()

    def test_absolute_path_within_base(self, tmp_path):
        target = tmp_path / "file.json"
        result = validate_server_filepath(str(target), base_dir=tmp_path)
        assert result == target.resolve()

    def test_rejects_absolute_path_outside_base(self, tmp_path):
        with pytest.raises(ValueError, match="must be within"):
            validate_server_filepath("/etc/passwd", base_dir=tmp_path)

    def test_rejects_relative_path_traversal(self, tmp_path):
        with pytest.raises(ValueError, match="must be within"):
            validate_server_filepath("../../etc/passwd", base_dir=tmp_path)

    def test_rejects_dotdot_in_middle(self, tmp_path):
        with pytest.raises(ValueError, match="must be within"):
            validate_server_filepath("sub/../../etc/shadow", base_dir=tmp_path)

    def test_accepts_simple_filename(self, tmp_path):
        result = validate_server_filepath("results.json", base_dir=tmp_path)
        assert result == (tmp_path / "results.json").resolve()

    def test_accepts_nested_relative(self, tmp_path):
        result = validate_server_filepath("data/output/file.csv", base_dir=tmp_path)
        assert result == (tmp_path / "data" / "output" / "file.csv").resolve()

    def test_defaults_to_cwd(self):
        cwd = Path.cwd()
        result = validate_server_filepath("some_file.json")
        assert result == (cwd / "some_file.json").resolve()

    def test_rejects_absolute_outside_cwd(self):
        with pytest.raises(ValueError, match="must be within"):
            validate_server_filepath("/etc/passwd")

    def test_rejects_symlink_escape(self, tmp_path):
        """A symlink that points outside base_dir should be rejected."""
        link = tmp_path / "escape_link"
        try:
            link.symlink_to("/etc")
        except OSError:
            pytest.skip("Cannot create symlinks in this environment")
        with pytest.raises(ValueError, match="must be within"):
            validate_server_filepath("escape_link/passwd", base_dir=tmp_path)


# ---------------------------------------------------------------------------
# get_file_access_base_dir
# ---------------------------------------------------------------------------


class TestGetFileAccessBaseDir:
    """Test that get_file_access_base_dir returns the correct base for each provider."""

    def test_default_provider_returns_none(self):
        """DefaultLoginProvider → None (unrestricted, falls back to CWD)."""
        from vtsearch.auth import DefaultLoginProvider, get_login_provider, set_login_provider

        original = get_login_provider()
        try:
            set_login_provider(DefaultLoginProvider())
            assert get_file_access_base_dir() is None
        finally:
            set_login_provider(original)

    def test_non_default_provider_returns_user_data_dir(self, tmp_path):
        """Non-default provider → user data directory."""
        from vtsearch.auth import LoginProvider, get_login_provider, set_login_provider

        class MultiUserProvider(LoginProvider):
            name = "multi"

            def get_user(self, request):
                return "alice"

            def is_authenticated(self, request):
                return True

            def get_user_data_dir(self, username, base_data_dir):
                return tmp_path / username

        original = get_login_provider()
        try:
            set_login_provider(MultiUserProvider())
            result = get_file_access_base_dir()
            # Outside Flask context, get_current_user falls back to "default"
            assert result == tmp_path / "default"
        finally:
            set_login_provider(original)

    def test_trivial_provider_returns_user_data_dir(self):
        """TrivialLoginProvider → user data directory (non-default provider)."""
        from vtsearch.auth import TrivialLoginProvider, get_login_provider, set_login_provider

        original = get_login_provider()
        try:
            set_login_provider(TrivialLoginProvider())
            result = get_file_access_base_dir()
            # Should return a Path, not None
            assert result is not None
            assert isinstance(result, Path)
        finally:
            set_login_provider(original)


# ---------------------------------------------------------------------------
# Multi-user file access restriction via API
# ---------------------------------------------------------------------------


class TestMultiUserFileRestriction:
    """Verify that non-default login providers restrict file access to user data dir."""

    def _setup_multi_user(self, client, user_data_dir):
        """Set up a multi-user provider and return a cleanup function."""
        from vtsearch.auth import LoginProvider, get_login_provider, set_login_provider

        class TestMultiProvider(LoginProvider):
            name = "test_multi"

            def get_user(self, request):
                return "testuser"

            def is_authenticated(self, request):
                return True

            def get_user_data_dir(self, username, base_data_dir):
                return user_data_dir

        original = get_login_provider()
        set_login_provider(TestMultiProvider())
        return original

    def test_exporter_rejects_path_outside_user_dir(self, client, tmp_path):
        """Exporter filepath outside user data dir is rejected in multi-user mode."""
        user_dir = tmp_path / "testuser"
        user_dir.mkdir()
        original = self._setup_multi_user(client, user_dir)
        try:
            resp = client.post(
                "/api/exporters/export",
                json={
                    "exporter_name": "server_json_file",
                    "field_values": {"filepath": "/etc/evil.json"},
                },
            )
            assert resp.status_code == 400
            # flask-smorest error envelope: handler-level rejects (path
            # traversal) live under ``message``.
            assert "must be within" in resp.get_json()["message"]
        finally:
            from vtsearch.auth import set_login_provider

            set_login_provider(original)

    def test_exporter_accepts_path_inside_user_dir(self, client, tmp_path):
        """Exporter filepath inside user data dir is accepted in multi-user mode."""
        user_dir = tmp_path / "testuser"
        user_dir.mkdir()
        original = self._setup_multi_user(client, user_dir)
        try:
            target = user_dir / "results.json"
            resp = client.post(
                "/api/exporters/export",
                json={
                    "exporter_name": "server_json_file",
                    "field_values": {"filepath": str(target)},
                    "results": {},
                },
            )
            # Should not get a path-validation 400 (may get other errors, but not path-related)
            if resp.status_code == 400:
                assert "must be within" not in resp.get_json().get("message", "")
        finally:
            from vtsearch.auth import set_login_provider

            set_login_provider(original)

    def test_label_importer_rejects_path_outside_user_dir(self, client, tmp_path):
        """Label importer filepath outside user data dir is rejected in multi-user mode."""
        user_dir = tmp_path / "testuser"
        user_dir.mkdir()
        original = self._setup_multi_user(client, user_dir)
        try:
            resp = client.post(
                "/api/label-importers/import/server_json_file",
                json={"filepath": "/etc/shadow"},
            )
            assert resp.status_code == 400
            # Phase B: path traversal raised from the framework's normalize
            # pass via flask-smorest abort() → message key, not error.
            assert "must be within" in resp.get_json()["message"]
        finally:
            from vtsearch.auth import set_login_provider

            set_login_provider(original)

    def test_load_folder_rejects_path_outside_user_dir(self, client, tmp_path):
        """Load folder path outside user data dir is rejected in multi-user mode."""
        user_dir = tmp_path / "testuser"
        user_dir.mkdir()
        original = self._setup_multi_user(client, user_dir)
        try:
            resp = client.post(
                "/api/dataset/load-folder",
                json={"path": "/etc", "media_type": "audio"},
            )
            assert resp.status_code == 400
            # New error envelope (flask-smorest): human-readable text lives
            # under ``message`` (the legacy ``error`` key is gone).
            assert "must be within" in resp.get_json()["message"]
        finally:
            from vtsearch.auth import set_login_provider

            set_login_provider(original)

    def test_combine_datasets_rejects_path_outside_user_dir(self, client, tmp_path):
        """Combine datasets with paths outside user data dir is rejected in multi-user mode."""
        user_dir = tmp_path / "testuser"
        user_dir.mkdir()
        original = self._setup_multi_user(client, user_dir)
        try:
            resp = client.post(
                "/api/dataset/combine",
                json={"datasets": ["/etc/a.pkl", "/etc/b.pkl"]},
            )
            assert resp.status_code == 400
            # flask-smorest error envelope.
            assert "must be within" in resp.get_json()["message"]
        finally:
            from vtsearch.auth import set_login_provider

            set_login_provider(original)

    def test_settings_dir_rejects_path_outside_user_dir(self, client, tmp_path):
        """Setting directory paths outside user data dir is rejected in multi-user mode."""
        user_dir = tmp_path / "testuser"
        user_dir.mkdir()
        original = self._setup_multi_user(client, user_dir)
        try:
            resp = client.put(
                "/api/settings",
                json={"saved_datasets_dir": "/tmp/evil"},
            )
            assert resp.status_code == 400
            # New error envelope: human-readable text lives under
            # ``message`` (the legacy ``error`` key is gone).
            assert "must be within" in resp.get_json()["message"]
        finally:
            from vtsearch.auth import set_login_provider

            set_login_provider(original)

    def test_settings_dir_accepts_path_inside_user_dir(self, client, tmp_path):
        """Setting directory paths inside user data dir is accepted in multi-user mode."""
        user_dir = tmp_path / "testuser"
        user_dir.mkdir()
        original = self._setup_multi_user(client, user_dir)
        try:
            target_dir = user_dir / "my_datasets"
            resp = client.put(
                "/api/settings",
                json={"saved_datasets_dir": str(target_dir)},
            )
            assert resp.status_code == 200
        finally:
            from vtsearch.auth import set_login_provider

            set_login_provider(original)

    def test_default_provider_allows_any_path(self, client):
        """DefaultLoginProvider allows any path within CWD (single-user mode)."""
        from vtsearch.auth import DefaultLoginProvider, get_login_provider, set_login_provider

        original = get_login_provider()
        try:
            set_login_provider(DefaultLoginProvider())
            # Paths within CWD should be accepted (even if the file doesn't exist,
            # the path validation itself should pass)
            cwd = Path.cwd()
            resp = client.post(
                "/api/dataset/load-folder",
                json={"path": str(cwd / "some_folder"), "media_type": "audio"},
            )
            # Should not get a "must be within" error — may get "Invalid folder path"
            # since the folder doesn't exist, but not a path-validation error.
            # ``load-folder`` is on flask-smorest, so the human-readable text
            # lives under ``message``.
            if resp.status_code == 400:
                assert "must be within" not in resp.get_json().get("message", "")
        finally:
            set_login_provider(original)
