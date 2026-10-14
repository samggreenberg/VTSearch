"""How ``vtsearch.__version__`` is resolved: git, then a baked file, then a fallback.

Kept apart from ``test_frontend.py``, whose tests all wait on the Angular
bundle: these need only the package and, for the first, the git checkout the
tests run from.
"""

from __future__ import annotations

import logging
import subprocess


class TestVersionResolution:
    """`vtsearch.__init__` resolves __version__ from git, then a baked file, then a fallback."""

    def test_git_resolver_returns_iso_utc_for_real_repo(self, caplog):
        from vtsearch import _version_from_git

        with caplog.at_level(logging.DEBUG, logger="vtsearch"):
            version = _version_from_git()
        # The resolver swallows its error, so the message carries its log (#4852).
        assert version is not None, f"tests run from a git checkout; git resolution must succeed: {caplog.text}"
        assert version.endswith("Z")

    def test_git_resolver_logs_why_it_failed(self, tmp_path, monkeypatch, caplog):
        import vtsearch as pkg

        def hung_git(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

        (tmp_path / ".git").mkdir()
        monkeypatch.setattr(pkg, "__file__", str(tmp_path / "vtsearch" / "__init__.py"))
        monkeypatch.setattr(pkg.subprocess, "run", hung_git)
        with caplog.at_level(logging.DEBUG, logger="vtsearch"):
            assert pkg._version_from_git() is None
        assert "TimeoutExpired" in caplog.text

    def test_file_resolver_reads_baked_version(self, tmp_path, monkeypatch):
        import vtsearch as pkg

        baked = tmp_path / "_version.txt"
        baked.write_text("2030-01-02T03:04:05Z\n", encoding="utf-8")
        monkeypatch.setattr(pkg, "__file__", str(tmp_path / "__init__.py"))
        assert pkg._version_from_file() == "2030-01-02T03:04:05Z"

    def test_file_resolver_returns_none_when_missing(self, tmp_path, monkeypatch):
        import vtsearch as pkg

        monkeypatch.setattr(pkg, "__file__", str(tmp_path / "__init__.py"))
        assert pkg._version_from_file() is None

    def test_fallback_when_git_and_file_unavailable(self, monkeypatch):
        import vtsearch as pkg

        monkeypatch.setattr(pkg, "_version_from_git", lambda: None)
        monkeypatch.setattr(pkg, "_version_from_file", lambda: None)
        assert pkg._resolve_version() == "0.0.0-unknown"
