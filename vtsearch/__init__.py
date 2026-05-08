"""VTSearch - Interactive ML-powered media similarity explorer."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

# Version is the UTC timestamp of HEAD's commit (ISO 8601, Z-terminated).
# Derived from git at import time — never stored in a tracked file — so
# parallel branches can't collide on a hand-bumped version constant.
# At deploy time (Docker, sdist) where .git is absent, we fall back to a
# baked `_version.txt` written by the build step.

_FALLBACK_VERSION = "0.0.0-unknown"


def _format_iso_utc(unix_ts: int) -> str:
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _version_from_git() -> str | None:
    repo_root = Path(__file__).resolve().parent.parent
    if not (repo_root / ".git").exists():
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "log", "-1", "--format=%ct", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    ts = out.stdout.strip()
    if not ts.isdigit():
        return None
    return _format_iso_utc(int(ts))


def _version_from_file() -> str | None:
    baked = Path(__file__).resolve().parent / "_version.txt"
    if not baked.exists():
        return None
    text = baked.read_text(encoding="utf-8").strip()
    return text or None


def _resolve_version() -> str:
    return _version_from_git() or _version_from_file() or _FALLBACK_VERSION


__version__ = _resolve_version()

__all__ = ["__version__"]
