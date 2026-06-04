"""Server file-path validation utilities.

Prevents path-traversal attacks by ensuring user-supplied file paths
resolve within an allowed base directory.  Used by the web API routes
for server-file importers and exporters.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Iterator


def get_file_access_base_dir() -> Path | None:
    """Return the base directory for file-access validation.

    In single-user / no-auth mode (:class:`~vtsearch.auth.DefaultLoginProvider`)
    this returns ``None``, which tells :func:`validate_server_filepath` to apply
    **no** confinement: the lone trusted user may read from and write to any
    server-readable path.  There is no per-user boundary to protect, so the app
    does not impose one.

    In multi-user mode (any non-default provider) this returns the current
    user's data directory so that each user is confined to their own
    ``data/<username>/`` subtree.
    """
    from vtsearch.auth import DefaultLoginProvider, get_login_provider, get_user_data_dir

    provider = get_login_provider()
    if isinstance(provider, DefaultLoginProvider):
        return None  # single-user / no-auth: unrestricted file access
    return get_user_data_dir()


def validate_server_filepath(filepath_str: str, base_dir: Path | None = None) -> Path:
    """Resolve *filepath_str*, optionally asserting it stays within *base_dir*.

    Parameters
    ----------
    filepath_str:
        User-supplied file path (absolute or relative).
    base_dir:
        The single directory the resolved path must reside in.  When ``None``
        (the single-user / no-auth case; see :func:`get_file_access_base_dir`)
        the path is **unrestricted**: it is resolved (relative paths against the
        process CWD) and returned without any containment check.  When a path is
        given (the per-user data dir in multi-user mode) it is the only allowed
        root and an escape raises.

    Returns
    -------
    Path
        The resolved, canonicalised path.

    Raises
    ------
    ValueError
        If *base_dir* is given and the resolved path escapes it.
    """
    path = Path(filepath_str)

    if base_dir is None:
        # Single-user / no-auth mode: every server-readable path is allowed.
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()

    # Multi-user: confine to base_dir.  Resolve relative paths against it.
    if not path.is_absolute():
        path = base_dir / path

    resolved = path.resolve()
    try:
        resolved.relative_to(base_dir.resolve())
    except ValueError:
        raise ValueError(
            f"The given path resolves outside the allowed directory. "
            f"Paths must be within '{base_dir.resolve()}'."
        ) from None
    return resolved


def sanitize_template_value(value: str) -> str:
    """Make *value* safe to splice into a filesystem-path template.

    Server-side sync sources accept admin-defined templates like
    ``data/labels/{detector_name}.json`` and substitute user-controlled
    fields (``{detector_name}``, ``{username}``, …) at runtime.  Without
    sanitization, a value containing ``../`` would let the substitution
    escape the intended directory.

    Returns a value with path separators and parent-directory tokens
    replaced by ``_``.  Empty / ``.`` / ``..`` collapse to ``_``.
    """
    sanitized = value.replace("/", "_").replace("\\", "_").replace("\0", "_")
    if sanitized in ("", ".", ".."):
        return "_"
    return sanitized


def iter_rglob_follow_symlinks(root: Path, pattern: str) -> Iterator[Path]:
    """Stream files under *root* matching *pattern*, following symlinks.

    Generator twin of :func:`rglob_follow_symlinks`.  Yields each match as
    :func:`os.walk` discovers it, so the caller never holds the full file
    list in memory — essential when scanning a directory tree with more
    files than fit in RAM (see ``docs/plans/cli-stream-massive-images.md``).
    """
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=True):
        for filename in filenames:
            if fnmatch.fnmatch(filename, pattern):
                yield Path(dirpath) / filename


def rglob_follow_symlinks(root: Path, pattern: str) -> list[Path]:
    """Like ``Path.rglob(pattern)`` but follows symlinks into directories.

    ``Path.rglob()`` does not descend into symlinked directories, which means
    media files inside symlinked sub-folders are silently skipped during
    dataset import.  This helper uses :func:`os.walk` with
    ``followlinks=True`` to ensure symlinked directory trees are traversed.

    Materialises the full list; callers that can stream should prefer
    :func:`iter_rglob_follow_symlinks`.
    """
    return list(iter_rglob_follow_symlinks(root, pattern))


def iter_glob_top_level(root: Path, pattern: str) -> Iterator[Path]:
    """Stream files directly in *root* matching *pattern* (no recursion).

    Generator twin of :func:`glob_top_level`.
    """
    if not root.is_dir():
        return
    for entry in os.scandir(root):
        if entry.is_file(follow_symlinks=True) and fnmatch.fnmatch(entry.name, pattern):
            yield Path(entry.path)


def glob_top_level(root: Path, pattern: str) -> list[Path]:
    """Match *pattern* against files directly in *root* (no recursion).

    Mirrors :func:`rglob_follow_symlinks` but limited to the immediate
    children of *root*.  Subdirectories are not descended into.

    Materialises the full list; callers that can stream should prefer
    :func:`iter_glob_top_level`.
    """
    return list(iter_glob_top_level(root, pattern))
