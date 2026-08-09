"""Server file-path validation utilities.

Prevents path-traversal attacks by ensuring user-supplied file paths
resolve within an allowed base directory.  Used by the web API routes
for server-file importers and exporters.
"""

from __future__ import annotations

import fnmatch
import logging
import os
from pathlib import Path
from typing import Iterator

from vtscore.config import DATA_DIR

logger = logging.getLogger(__name__)


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
        The resolved, canonicalised path.  Callers must **use** this result
        rather than the raw input: under confinement the two disagree for
        relative paths, and consuming the raw string reopens the escape this
        function exists to close.  Callers that need a string (to forward
        into ``field_values`` / an origin) should prefer
        :func:`confine_server_filepath`, which picks the right one.

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
            f"The given path resolves outside the allowed directory. Paths must be within '{base_dir.resolve()}'."
        ) from None
    return resolved


def confine_server_filepath(filepath_str: str, base_dir: Path | None) -> str:
    """Validate *filepath_str* and return the path string callers must consume.

    :func:`validate_server_filepath` resolves a **relative** path against
    *base_dir* before its containment check, but every consumer of a stored
    path string resolves it against the process **CWD**.  Discarding the
    validator's return value therefore validates one path and then reads a
    different one: a user confined to ``data/bob`` who submits
    ``"data/alice"`` passes the check (``data/bob/data/alice`` is inside the
    base dir) and is then handed ``CWD/data/alice`` — another user's subtree.

    This wrapper closes that gap by handing back the canonical path the check
    actually approved, so validation and use share a single anchor.  Callers
    must store / forward the returned string rather than the raw input.

    When *base_dir* is ``None`` (single-user / no-auth; see
    :func:`get_file_access_base_dir`) the two anchors are already the same —
    both CWD — so the input is returned verbatim.  That keeps relative paths
    relative in stored origins, which stay portable across checkouts.

    Raises
    ------
    ValueError
        If *base_dir* is given and the resolved path escapes it.
    """
    resolved = validate_server_filepath(filepath_str, base_dir=base_dir)
    return filepath_str if base_dir is None else str(resolved)


def media_file_read_roots() -> list[Path] | None:
    """Return the roots a media's own file reference may be read from.

    ``None`` means "no confinement" and is what single-user / no-auth mode
    returns, matching :func:`get_file_access_base_dir`.

    In multi-user mode the allowed roots are the current user's data dir
    **and** :data:`~vtscore.config.DATA_DIR`.  The shared data dir has to be in
    the list because demo datasets download into it as siblings of the per-user
    dirs (``data/ESC-50-master/audio``, ``data/caltech-101/…``), so a thin demo
    dataset's ``media_path`` legitimately points outside ``data/<username>/``.
    That does mean a crafted reference can still name another user's subtree;
    the boundary this draws is "inside the app's data tree", which is what
    stops ``/etc/shadow`` and every other server file from being served.
    """
    base = get_file_access_base_dir()
    if base is None:
        return None
    return [base, DATA_DIR]


def resolve_media_file_path(filepath_str: str) -> Path | None:
    """Resolve a file reference carried *by a media*, or ``None`` if it escapes.

    ``media_path`` and the archive-member archive path arrive from a dataset
    pickle, which in a multi-user deployment is attacker-supplied data: the
    unpickler passes plain strings straight through, so a media can name any
    server-readable file and the media-serving routes will hand its bytes back
    to the requester.  Every read of such a reference goes through here first.

    Returns the resolved path when it lands inside one of
    :func:`media_file_read_roots`, and ``None`` when it does not, so callers
    fall through to their next resolution step (or serve nothing) instead of
    raising at request time.  In single-user mode the path is returned
    unchanged — the lone trusted user may read any server-readable file.
    """
    roots = media_file_read_roots()
    if roots is None:
        return Path(filepath_str)
    for root in roots:
        try:
            return validate_server_filepath(filepath_str, base_dir=root)
        except ValueError:
            continue
    logger.warning("Refusing to read media file outside the allowed data dirs: %s", filepath_str)
    return None


def sanitize_template_value(value: str) -> str:
    """Make *value* safe to splice into a filesystem-path template.

    Server-side sync sources accept admin-defined templates like
    ``data/labels/{detector_name}.json`` and substitute user-controlled
    fields (``{detector_name}``, ``{username}``, …) at runtime.  Without
    sanitization, a value containing ``../`` would let the substitution
    escape the intended directory.

    Returns a value with path separators and parent-directory tokens
    replaced by ``_``.  Empty and any all-dots token (``.``, ``..``,
    ``...``, …) collapse to ``_``: ``..`` addresses the parent directory
    and ``.`` the current one, while longer dot runs are meaningless
    path components that some shells / filesystems still treat specially,
    so none of them belong in a single path segment.
    """
    sanitized = value.replace("/", "_").replace("\\", "_").replace("\0", "_")
    if not sanitized or set(sanitized) == {"."}:
        return "_"
    return sanitized


def iter_rglob_follow_symlinks(root: Path, pattern: str) -> Iterator[Path]:
    """Stream files under *root* matching *pattern*, following symlinks.

    Generator twin of :func:`rglob_follow_symlinks`.  Yields each match as
    :func:`os.walk` discovers it, so the caller never holds the full file
    list in memory — essential when scanning a directory tree with more
    files than fit in RAM (see ``docs/plans/cli-stream-massive-images.md``).

    Because ``followlinks=True`` makes :func:`os.walk` descend into
    symlinked directories, a circular layout (a directory symlinked back
    to one of its ancestors) would otherwise loop forever and exhaust
    CPU/RAM.  We guard against that by tracking the ``(st_dev, st_ino)``
    of every directory we descend into and pruning any subdirectory we've
    already visited — so each real directory is walked at most once and a
    cycle terminates.
    """
    seen_dirs: set[tuple[int, int]] = set()
    try:
        root_stat = os.stat(root)
        seen_dirs.add((root_stat.st_dev, root_stat.st_ino))
    except OSError:
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        # Prune already-visited directories in place so os.walk won't
        # descend into them again, breaking symlink cycles.
        kept: list[str] = []
        for d in dirnames:
            try:
                st = os.stat(os.path.join(dirpath, d))
            except OSError:
                continue
            key = (st.st_dev, st.st_ino)
            if key in seen_dirs:
                continue
            seen_dirs.add(key)
            kept.append(d)
        dirnames[:] = kept
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
