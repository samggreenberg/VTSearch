"""VTSearch - Interactive ML-powered media similarity explorer."""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# Version is the UTC timestamp of HEAD's commit (ISO 8601, Z-terminated).
# Derived from git at import time and never stored in a tracked file, so
# parallel branches can't collide on a hand-bumped version constant.
# At deploy time (Docker, sdist) where .git is absent, we fall back to a
# baked `_version.txt` written by the build step.

_FALLBACK_VERSION = "0.0.0-unknown"

# A healthy checkout answers in tens of milliseconds, even a linked worktree on
# the GRID's NFS. The bound only stops a hung git from holding up the import;
# 2 s was tight enough to trip on a loaded suite node (#4852).
_GIT_TIMEOUT_S = 10

_log = logging.getLogger(__name__)


def _format_iso_utc(unix_ts: int) -> str:
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _version_from_git() -> str | None:
    repo_root = Path(__file__).resolve().parent.parent
    if not (repo_root / ".git").exists():
        _log.debug("version: no .git in %s", repo_root)
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "log", "-1", "--format=%ct", "HEAD"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as exc:
        # Debug, not warning: falling back is the designed path when git is unusable.
        # The record is what says which failure it was (#4852).
        _log.debug("version: git failed in %s: %r, stderr %r", repo_root, exc, getattr(exc, "stderr", None))
        return None
    ts = out.stdout.strip()
    if not ts.isdigit():
        _log.debug("version: git printed %r, not a commit time", ts)
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
