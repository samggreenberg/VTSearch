"""Tests for the server file-path validation utility.

Covers:
- validate_server_filepath: rejects path traversal, accepts safe paths
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vtsearch.utils.paths import validate_server_filepath


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
