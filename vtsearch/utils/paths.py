"""Server file-path validation utilities.

Prevents path-traversal attacks by ensuring user-supplied file paths
resolve within an allowed base directory.  Used by the web API routes
for server-file importers and exporters.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path


def get_file_access_base_dir() -> Path | None:
    """Return the base directory for file-access validation.

    In single-user mode (:class:`~vtsearch.auth.DefaultLoginProvider`) this
    returns ``None``, which causes :func:`validate_server_filepath` to fall
    back to ``Path.cwd()`` — giving the single user full access to any path
    under the working directory.

    In multi-user mode (any non-default provider) this returns the current
    user's data directory so that each user is confined to their own
    ``data/<username>/`` subtree.
    """
    from vtsearch.auth import DefaultLoginProvider, get_login_provider, get_user_data_dir

    provider = get_login_provider()
    if isinstance(provider, DefaultLoginProvider):
        return None  # single-user: unrestricted (CWD)
    return get_user_data_dir()


def validate_server_filepath(filepath_str: str, base_dir: Path | None = None) -> Path:
    """Validate that *filepath_str* resolves within *base_dir*.

    Parameters
    ----------
    filepath_str:
        User-supplied file path (absolute or relative).
    base_dir:
        The directory the resolved path must reside in.
        Defaults to ``Path.cwd()``.

    Returns
    -------
    Path
        The resolved, canonicalised path.

    Raises
    ------
    ValueError
        If the resolved path is outside *base_dir*.
    """
    if base_dir is None:
        base_dir = Path.cwd()

    path = Path(filepath_str)

    # Resolve relative paths against base_dir, absolute paths as-is
    if not path.is_absolute():
        path = base_dir / path

    resolved = path.resolve()
    base_resolved = base_dir.resolve()

    try:
        resolved.relative_to(base_resolved)
    except ValueError:
        raise ValueError(
            f"Path must be within '{base_resolved}'. The given path resolves outside the allowed directory."
        )

    return resolved


def rglob_follow_symlinks(root: Path, pattern: str) -> list[Path]:
    """Like ``Path.rglob(pattern)`` but follows symlinks into directories.

    ``Path.rglob()`` does not descend into symlinked directories, which means
    media files inside symlinked sub-folders are silently skipped during
    dataset import.  This helper uses :func:`os.walk` with
    ``followlinks=True`` to ensure symlinked directory trees are traversed.
    """
    results: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=True):
        for filename in filenames:
            if fnmatch.fnmatch(filename, pattern):
                results.append(Path(dirpath) / filename)
    return results
