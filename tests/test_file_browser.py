"""Tests for the /api/browse file-browser endpoint.

Covers:
- Listing directories and files at the root
- Navigating into subdirectories
- Extension filtering
- Path traversal prevention
- Non-existent directory handling
- Multi-user isolation (users cannot browse into other users' folders)
- No server filesystem path leakage in responses
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


from vtsearch.auth import (
    LoginProvider,
    get_login_provider,
    set_login_provider,
)


def _make_browse_root(tmp_path: Path) -> Path:
    """Create a clean subdirectory to use as the browse root.

    The ``tmp_path`` fixture is shared with autouse fixtures (e.g.
    ``isolated_settings``) which create their own files there.  Using a
    dedicated subdirectory avoids those files appearing in browse results.
    """
    root = tmp_path / "browse_root"
    root.mkdir()
    return root


class TestBrowseEndpoint:
    """Tests for GET /api/browse."""

    def test_browse_root(self, client, tmp_path):
        """Listing root returns directories and files."""
        root = _make_browse_root(tmp_path)
        (root / "subdir").mkdir()
        (root / "file.csv").write_text("a,b\n1,2")
        (root / "file.json").write_text("{}")

        with patch("vtsearch.routes.file_browser._get_browse_root", return_value=root):
            resp = client.get("/api/browse")

        assert resp.status_code == 200
        data = resp.get_json()
        dir_names = [d["name"] for d in data["directories"]]
        file_names = [f["name"] for f in data["files"]]
        assert "subdir" in dir_names
        assert "file.csv" in file_names
        assert "file.json" in file_names
        assert data["current_path"] == ""
        assert "root" not in data  # no server path leakage

    def test_browse_subdir(self, client, tmp_path):
        """Navigating into a subdirectory lists its contents."""
        root = _make_browse_root(tmp_path)
        sub = root / "data"
        sub.mkdir()
        (sub / "labels.csv").write_text("md5,label\nabc,good")
        (sub / "nested").mkdir()

        with patch("vtsearch.routes.file_browser._get_browse_root", return_value=root):
            resp = client.get("/api/browse?path=data")

        assert resp.status_code == 200
        data = resp.get_json()
        dir_names = [d["name"] for d in data["directories"]]
        file_names = [f["name"] for f in data["files"]]
        assert "nested" in dir_names
        assert "labels.csv" in file_names
        assert data["current_path"] == "data"

    def test_extension_filter(self, client, tmp_path):
        """Extension parameter filters files by suffix."""
        root = _make_browse_root(tmp_path)
        (root / "file.csv").write_text("data")
        (root / "file.json").write_text("{}")
        (root / "file.txt").write_text("hello")

        with patch("vtsearch.routes.file_browser._get_browse_root", return_value=root):
            resp = client.get("/api/browse?extensions=.csv,.json")

        assert resp.status_code == 200
        data = resp.get_json()
        file_names = [f["name"] for f in data["files"]]
        assert "file.csv" in file_names
        assert "file.json" in file_names
        assert "file.txt" not in file_names

    def test_extension_filter_without_dot(self, client, tmp_path):
        """Extensions without leading dot are accepted."""
        root = _make_browse_root(tmp_path)
        (root / "file.csv").write_text("data")
        (root / "file.txt").write_text("hello")

        with patch("vtsearch.routes.file_browser._get_browse_root", return_value=root):
            resp = client.get("/api/browse?extensions=csv")

        assert resp.status_code == 200
        data = resp.get_json()
        file_names = [f["name"] for f in data["files"]]
        assert "file.csv" in file_names
        assert "file.txt" not in file_names

    def test_path_traversal_blocked(self, client, tmp_path):
        """Attempting to escape the root via .. is rejected."""
        root = _make_browse_root(tmp_path)
        with patch("vtsearch.routes.file_browser._get_browse_root", return_value=root):
            resp = client.get("/api/browse?path=../../etc")

        assert resp.status_code == 400
        assert "Invalid path" in resp.get_json()["error"]

    def test_nonexistent_dir(self, client, tmp_path):
        """Browsing a path that doesn't exist returns 404."""
        root = _make_browse_root(tmp_path)
        with patch("vtsearch.routes.file_browser._get_browse_root", return_value=root):
            resp = client.get("/api/browse?path=no_such_dir")

        assert resp.status_code == 404

    def test_hidden_files_excluded(self, client, tmp_path):
        """Files and dirs starting with '.' are not listed."""
        root = _make_browse_root(tmp_path)
        (root / ".hidden_dir").mkdir()
        (root / ".hidden_file").write_text("secret")
        (root / "visible.csv").write_text("data")

        with patch("vtsearch.routes.file_browser._get_browse_root", return_value=root):
            resp = client.get("/api/browse")

        assert resp.status_code == 200
        data = resp.get_json()
        all_names = [d["name"] for d in data["directories"]] + [f["name"] for f in data["files"]]
        assert ".hidden_dir" not in all_names
        assert ".hidden_file" not in all_names
        assert "visible.csv" in all_names

    def test_file_size_reported(self, client, tmp_path):
        """Files include their size in bytes."""
        root = _make_browse_root(tmp_path)
        content = "hello world"
        (root / "file.txt").write_text(content)

        with patch("vtsearch.routes.file_browser._get_browse_root", return_value=root):
            resp = client.get("/api/browse")

        assert resp.status_code == 200
        data = resp.get_json()
        file_entry = next(f for f in data["files"] if f["name"] == "file.txt")
        assert file_entry["size_bytes"] == len(content)

    def test_no_extensions_shows_all_files(self, client, tmp_path):
        """Without extensions param, all files are shown."""
        root = _make_browse_root(tmp_path)
        (root / "a.csv").write_text("")
        (root / "b.json").write_text("")
        (root / "c.pkl").write_bytes(b"")
        (root / "d.txt").write_text("")

        with patch("vtsearch.routes.file_browser._get_browse_root", return_value=root):
            resp = client.get("/api/browse")

        assert resp.status_code == 200
        data = resp.get_json()
        file_names = sorted(f["name"] for f in data["files"])
        assert file_names == ["a.csv", "b.json", "c.pkl", "d.txt"]

    def test_empty_directory(self, client, tmp_path):
        """Browsing an empty directory returns empty lists."""
        root = _make_browse_root(tmp_path)
        with patch("vtsearch.routes.file_browser._get_browse_root", return_value=root):
            resp = client.get("/api/browse")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["directories"] == []
        assert data["files"] == []

    def test_no_root_in_response(self, client, tmp_path):
        """Response must not expose the server's absolute root path."""
        root = _make_browse_root(tmp_path)
        (root / "file.txt").write_text("data")

        with patch("vtsearch.routes.file_browser._get_browse_root", return_value=root):
            resp = client.get("/api/browse")

        assert resp.status_code == 200
        data = resp.get_json()
        assert "root" not in data
        # Ensure no absolute path appears anywhere in the JSON values
        raw = resp.get_data(as_text=True)
        assert str(root) not in raw


# ---------------------------------------------------------------------------
# Multi-user isolation
# ---------------------------------------------------------------------------


def _make_multi_user_provider(user_data_dir: Path, username: str = "alice"):
    """Create a LoginProvider that confines *username* to *user_data_dir*."""

    class _TestMultiProvider(LoginProvider):
        name = "test_file_browser"

        def get_user(self, request):
            return username

        def is_authenticated(self, request):
            return True

        def get_user_data_dir(self, uname, base_data_dir):
            return user_data_dir

    return _TestMultiProvider()


class TestMultiUserBrowseIsolation:
    """Verify that users in multi-user mode are confined to their own data dir."""

    def test_user_sees_own_files(self, client, tmp_path):
        """A user can browse files within their own data directory."""
        alice_dir = tmp_path / "alice"
        alice_dir.mkdir()
        (alice_dir / "my_labels.csv").write_text("data")
        (alice_dir / "subdir").mkdir()

        original = get_login_provider()
        try:
            set_login_provider(_make_multi_user_provider(alice_dir))
            resp = client.get("/api/browse")
        finally:
            set_login_provider(original)

        assert resp.status_code == 200
        data = resp.get_json()
        file_names = [f["name"] for f in data["files"]]
        dir_names = [d["name"] for d in data["directories"]]
        assert "my_labels.csv" in file_names
        assert "subdir" in dir_names

    def test_traversal_to_other_user_blocked(self, client, tmp_path):
        """A user cannot browse into another user's directory via traversal."""
        alice_dir = tmp_path / "alice"
        alice_dir.mkdir()
        bob_dir = tmp_path / "bob"
        bob_dir.mkdir()
        (bob_dir / "secret.csv").write_text("bob's secrets")

        original = get_login_provider()
        try:
            set_login_provider(_make_multi_user_provider(alice_dir))
            # Try to escape to bob's directory
            resp = client.get("/api/browse?path=../bob")
        finally:
            set_login_provider(original)

        assert resp.status_code == 400
        assert "Invalid path" in resp.get_json()["error"]

    def test_absolute_path_to_other_user_blocked(self, client, tmp_path):
        """A user cannot browse another user's directory via an absolute path."""
        alice_dir = tmp_path / "alice"
        alice_dir.mkdir()
        bob_dir = tmp_path / "bob"
        bob_dir.mkdir()
        (bob_dir / "secret.csv").write_text("bob's secrets")

        original = get_login_provider()
        try:
            set_login_provider(_make_multi_user_provider(alice_dir))
            # Try to browse bob's directory directly
            resp = client.get(f"/api/browse?path={bob_dir}")
        finally:
            set_login_provider(original)

        assert resp.status_code == 400
        assert "Invalid path" in resp.get_json()["error"]

    def test_two_users_see_different_files(self, client, tmp_path):
        """Two users with separate data dirs see only their own files."""
        alice_dir = tmp_path / "alice"
        alice_dir.mkdir()
        (alice_dir / "alice_file.csv").write_text("alice data")

        bob_dir = tmp_path / "bob"
        bob_dir.mkdir()
        (bob_dir / "bob_file.csv").write_text("bob data")

        original = get_login_provider()
        try:
            # Alice's view
            set_login_provider(_make_multi_user_provider(alice_dir, "alice"))
            alice_resp = client.get("/api/browse")

            # Bob's view
            set_login_provider(_make_multi_user_provider(bob_dir, "bob"))
            bob_resp = client.get("/api/browse")
        finally:
            set_login_provider(original)

        alice_files = [f["name"] for f in alice_resp.get_json()["files"]]
        bob_files = [f["name"] for f in bob_resp.get_json()["files"]]

        assert "alice_file.csv" in alice_files
        assert "bob_file.csv" not in alice_files
        assert "bob_file.csv" in bob_files
        assert "alice_file.csv" not in bob_files

    def test_traversal_to_parent_blocked(self, client, tmp_path):
        """Users cannot escape their root via .. to reach the parent directory."""
        user_dir = tmp_path / "user"
        user_dir.mkdir()
        (tmp_path / "sensitive.txt").write_text("secret")

        original = get_login_provider()
        try:
            set_login_provider(_make_multi_user_provider(user_dir))
            resp = client.get("/api/browse?path=..")
        finally:
            set_login_provider(original)

        assert resp.status_code == 400

    def test_response_contains_no_absolute_paths(self, client, tmp_path):
        """Multi-user browse responses must not leak the server filesystem layout."""
        user_dir = tmp_path / "user"
        user_dir.mkdir()
        (user_dir / "docs").mkdir()
        (user_dir / "file.txt").write_text("hi")

        original = get_login_provider()
        try:
            set_login_provider(_make_multi_user_provider(user_dir))
            resp = client.get("/api/browse")
        finally:
            set_login_provider(original)

        assert resp.status_code == 200
        raw = resp.get_data(as_text=True)
        # The server root path must not appear in the response
        assert str(tmp_path) not in raw

    def test_single_user_mode_allows_cwd_browsing(self, client, tmp_path):
        """In single-user mode (DefaultLoginProvider), browsing uses CWD as root."""
        root = _make_browse_root(tmp_path)
        (root / "file.csv").write_text("data")

        # DefaultLoginProvider → _get_browse_root returns CWD → we mock it
        with patch("vtsearch.routes.file_browser._get_browse_root", return_value=root):
            resp = client.get("/api/browse")

        assert resp.status_code == 200
        data = resp.get_json()
        file_names = [f["name"] for f in data["files"]]
        assert "file.csv" in file_names


class TestMultiUserServerMediaFiles:
    """Verify that server media file endpoints are per-user in multi-user mode."""

    def test_upload_and_list_isolated_per_user(self, client, tmp_path):
        """Files uploaded by one user are not visible to another user."""
        import io

        alice_dir = tmp_path / "alice"
        alice_dir.mkdir()
        bob_dir = tmp_path / "bob"
        bob_dir.mkdir()

        original = get_login_provider()
        try:
            # Alice uploads a file
            set_login_provider(_make_multi_user_provider(alice_dir, "alice"))
            resp = client.post(
                "/api/server-media-files/upload",
                data={"file": (io.BytesIO(b"\x00" * 100), "alice_sound.wav")},
                content_type="multipart/form-data",
            )
            assert resp.status_code == 201

            # Alice sees her file
            resp = client.get("/api/server-media-files")
            alice_files = [f["filename"] for f in resp.get_json()["files"]]
            assert len(alice_files) == 1

            # Bob sees nothing
            set_login_provider(_make_multi_user_provider(bob_dir, "bob"))
            resp = client.get("/api/server-media-files")
            bob_files = [f["filename"] for f in resp.get_json()["files"]]
            assert len(bob_files) == 0
        finally:
            set_login_provider(original)

    def test_sort_server_cannot_access_other_users_file(self, client, tmp_path):
        """A user cannot use example-sort-server with another user's filename."""
        alice_dir = tmp_path / "alice"
        alice_media = alice_dir / "example_media"
        alice_media.mkdir(parents=True)
        (alice_media / "test.wav").write_bytes(b"\x00" * 100)

        bob_dir = tmp_path / "bob"
        bob_dir.mkdir()

        original = get_login_provider()
        try:
            # Bob tries to access alice's file via traversal
            set_login_provider(_make_multi_user_provider(bob_dir, "bob"))
            resp = client.post(
                "/api/example-sort-server",
                json={"filename": "../../alice/example_media/test.wav"},
            )
            assert resp.status_code == 400
        finally:
            set_login_provider(original)


class TestMultiUserDetectorFiles:
    """Verify that detector server-file endpoints are per-user in multi-user mode."""

    def test_detector_files_isolated_per_user(self, client, tmp_path):
        """Detector files in one user's dir are not visible to another."""
        import json

        alice_dir = tmp_path / "alice"
        alice_det = alice_dir / "detectors"
        alice_det.mkdir(parents=True)
        det_data = {"weights": {"0.weight": [[1.0]], "0.bias": [0.0]}, "threshold": 0.5, "media_type": "audio"}
        (alice_det / "alice_model.json").write_text(json.dumps(det_data))

        bob_dir = tmp_path / "bob"
        bob_det = bob_dir / "detectors"
        bob_det.mkdir(parents=True)

        original = get_login_provider()
        try:
            # Alice sees her detector
            set_login_provider(_make_multi_user_provider(alice_dir, "alice"))
            resp = client.get("/api/detector/server-files")
            alice_files = [f["name"] for f in resp.get_json()["files"]]
            assert "alice_model" in alice_files

            # Bob sees nothing
            set_login_provider(_make_multi_user_provider(bob_dir, "bob"))
            resp = client.get("/api/detector/server-files")
            bob_files = [f["name"] for f in resp.get_json()["files"]]
            assert "alice_model" not in bob_files
            assert len(bob_files) == 0

            # Bob cannot fetch alice's detector by name
            resp = client.get("/api/detector/server-files/alice_model")
            assert resp.status_code == 404
        finally:
            set_login_provider(original)
