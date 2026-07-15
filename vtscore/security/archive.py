"""Centralised, hardened tar extraction.

A single audited path for pulling members out of a :class:`tarfile.TarFile`,
so that no call site has to re-implement path-traversal / absolute-path /
symlink protection by hand.

On Python interpreters that ship PEP 706's extraction filters
(``tarfile.data_filter``, available on CPython ≥ 3.9.17 / 3.10.12 / 3.11.4 /
3.12), every member is extracted through the strict ``data`` filter, which:

* strips a leading ``/`` from absolute member names so they land *inside* the
  destination rather than at the filesystem root,
* refuses ``..`` traversal that would escape the destination, and
* refuses symlink / hardlink members whose target is absolute or points
  outside the destination.

On older interpreters (no ``data_filter``) :func:`safe_tar_extract` falls back
to :func:`_reject_unsafe_member`, which reproduces those same guarantees
manually before writing anything to disk.
"""

from __future__ import annotations

import os
import tarfile
from pathlib import Path

__all__ = ["safe_tar_extract"]

#: True when the running interpreter provides PEP 706 extraction filters.
_HAS_DATA_FILTER = hasattr(tarfile, "data_filter")


def _strip_leading_root(name: str) -> str:
    """Return *name* with any leading path root removed (PEP 706 semantics).

    Drops leading ``/`` and ``\\`` separators, and a Windows drive prefix
    (``C:\\``), so an absolute member name becomes relative and extracts
    *inside* the destination instead of at the filesystem root.
    """
    # Windows drive-letter absolute path, e.g. "C:\\evil" or "C:/evil".
    if len(name) > 1 and name[1] == ":":
        name = name[2:]
    return name.lstrip("/\\")


def _reject_unsafe_member(member: tarfile.TarInfo, dest_resolved: Path) -> str:
    """Validate *member* against *dest_resolved*; return its sanitised name.

    Raises :class:`ValueError` if the member (or, for links, its target) would
    escape *dest_resolved*.  The returned name has any absolute-path root
    stripped, mirroring :data:`tarfile.data_filter`, so the caller can extract
    it safely inside the destination.  Used only on interpreters that lack
    PEP 706 filters.
    """
    sanitised = _strip_leading_root(member.name)
    target = Path(os.path.normpath(dest_resolved / sanitised))
    if target != dest_resolved and not target.is_relative_to(dest_resolved):
        raise ValueError(f"Path traversal detected in archive member: {member.name!r}")

    if member.issym() or member.islnk():
        linkname = member.linkname
        if os.path.isabs(linkname) or _strip_leading_root(linkname) != linkname:
            raise ValueError(f"Absolute link target in archive member: {member.name!r} -> {linkname!r}")
        link_target = Path(os.path.normpath(target.parent / linkname))
        if not link_target.is_relative_to(dest_resolved):
            raise ValueError(f"Link escapes destination in archive member: {member.name!r} -> {linkname!r}")

    return sanitised


def safe_tar_extract(tar: tarfile.TarFile, member: tarfile.TarInfo, dest: str | Path) -> None:
    """Extract a single *member* of *tar* into *dest* with traversal protection.

    On Python ≥ 3.11.4 (any interpreter exposing :data:`tarfile.data_filter`)
    the extraction runs through the strict PEP 706 ``data`` filter.  On older
    interpreters the member is validated by :func:`_reject_unsafe_member`
    before being written.  Either way, absolute paths are confined to *dest*
    and traversal / unsafe-link members are rejected.
    """
    if _HAS_DATA_FILTER:
        tar.extract(member, dest, filter=tarfile.data_filter)
        return

    dest_resolved = Path(dest).resolve()
    sanitised = _reject_unsafe_member(member, dest_resolved)
    # Extract under the sanitised (root-stripped) name so absolute members land
    # inside dest, matching the data-filter branch above.
    member.name = sanitised
    # Safe: member (and its link target) were validated by _reject_unsafe_member.
    tar.extract(member, dest)
