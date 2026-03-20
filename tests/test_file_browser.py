"""Tests for the /api/browse file-browser endpoint.

Covers:
- Listing directories and files at the root
- Navigating into subdirectories
- Extension filtering
- Path traversal prevention
- Non-existent directory handling
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest


class TestBrowseEndpoint:
    """Tests for GET /api/browse."""

    def test_browse_root(self, client, tmp_path):
        """Listing root returns directories and files."""
        (tmp_path / "subdir").mkdir()
        (tmp_path / "file.csv").write_text("a,b\n1,2")
        (tmp_path / "file.json").write_text("{}")

        with patch("vtsearch.routes.file_browser._get_browse_root", return_value=tmp_path):
            resp = client.get("/api/browse")

        assert resp.status_code == 200
        data = resp.get_json()
        dir_names = [d["name"] for d in data["directories"]]
        file_names = [f["name"] for f in data["files"]]
        assert "subdir" in dir_names
        assert "file.csv" in file_names
        assert "file.json" in file_names
        assert data["current_path"] == ""
        assert data["root"] == str(tmp_path)

    def test_browse_subdir(self, client, tmp_path):
        """Navigating into a subdirectory lists its contents."""
        sub = tmp_path / "data"
        sub.mkdir()
        (sub / "labels.csv").write_text("md5,label\nabc,good")
        (sub / "nested").mkdir()

        with patch("vtsearch.routes.file_browser._get_browse_root", return_value=tmp_path):
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
        (tmp_path / "file.csv").write_text("data")
        (tmp_path / "file.json").write_text("{}")
        (tmp_path / "file.txt").write_text("hello")

        with patch("vtsearch.routes.file_browser._get_browse_root", return_value=tmp_path):
            resp = client.get("/api/browse?extensions=.csv,.json")

        assert resp.status_code == 200
        data = resp.get_json()
        file_names = [f["name"] for f in data["files"]]
        assert "file.csv" in file_names
        assert "file.json" in file_names
        assert "file.txt" not in file_names

    def test_extension_filter_without_dot(self, client, tmp_path):
        """Extensions without leading dot are accepted."""
        (tmp_path / "file.csv").write_text("data")
        (tmp_path / "file.txt").write_text("hello")

        with patch("vtsearch.routes.file_browser._get_browse_root", return_value=tmp_path):
            resp = client.get("/api/browse?extensions=csv")

        assert resp.status_code == 200
        data = resp.get_json()
        file_names = [f["name"] for f in data["files"]]
        assert "file.csv" in file_names
        assert "file.txt" not in file_names

    def test_path_traversal_blocked(self, client, tmp_path):
        """Attempting to escape the root via .. is rejected."""
        with patch("vtsearch.routes.file_browser._get_browse_root", return_value=tmp_path):
            resp = client.get("/api/browse?path=../../etc")

        assert resp.status_code == 400
        assert "Invalid path" in resp.get_json()["error"]

    def test_nonexistent_dir(self, client, tmp_path):
        """Browsing a path that doesn't exist returns 404."""
        with patch("vtsearch.routes.file_browser._get_browse_root", return_value=tmp_path):
            resp = client.get("/api/browse?path=no_such_dir")

        assert resp.status_code == 404

    def test_hidden_files_excluded(self, client, tmp_path):
        """Files and dirs starting with '.' are not listed."""
        (tmp_path / ".hidden_dir").mkdir()
        (tmp_path / ".hidden_file").write_text("secret")
        (tmp_path / "visible.csv").write_text("data")

        with patch("vtsearch.routes.file_browser._get_browse_root", return_value=tmp_path):
            resp = client.get("/api/browse")

        assert resp.status_code == 200
        data = resp.get_json()
        all_names = [d["name"] for d in data["directories"]] + [f["name"] for f in data["files"]]
        assert ".hidden_dir" not in all_names
        assert ".hidden_file" not in all_names
        assert "visible.csv" in all_names

    def test_file_size_reported(self, client, tmp_path):
        """Files include their size in bytes."""
        content = "hello world"
        (tmp_path / "file.txt").write_text(content)

        with patch("vtsearch.routes.file_browser._get_browse_root", return_value=tmp_path):
            resp = client.get("/api/browse")

        assert resp.status_code == 200
        data = resp.get_json()
        file_entry = next(f for f in data["files"] if f["name"] == "file.txt")
        assert file_entry["size_bytes"] == len(content)

    def test_no_extensions_shows_all_files(self, client, tmp_path):
        """Without extensions param, all files are shown."""
        (tmp_path / "a.csv").write_text("")
        (tmp_path / "b.json").write_text("")
        (tmp_path / "c.pkl").write_bytes(b"")
        (tmp_path / "d.txt").write_text("")

        with patch("vtsearch.routes.file_browser._get_browse_root", return_value=tmp_path):
            resp = client.get("/api/browse")

        assert resp.status_code == 200
        data = resp.get_json()
        file_names = sorted(f["name"] for f in data["files"])
        assert file_names == ["a.csv", "b.json", "c.pkl", "d.txt"]

    def test_empty_directory(self, client, tmp_path):
        """Browsing an empty directory returns empty lists."""
        with patch("vtsearch.routes.file_browser._get_browse_root", return_value=tmp_path):
            resp = client.get("/api/browse")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["directories"] == []
        assert data["files"] == []
