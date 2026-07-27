"""Content fingerprinting helpers.

Every hash VTSearch computes over media bytes is a **content-identity
fingerprint**: it answers "are these the same bytes?" for dedup, cache keys,
ETags, and origin tracking.  None of them is a security primitive - nothing
here authenticates, signs, or protects anything.

That distinction has a runtime consequence.  On a FIPS-enabled host, OpenSSL
refuses to hand out MD5 (and SHA-1) at all, and ``hashlib.md5(...)`` raises
``ValueError: [digital envelope routines] unsupported``.  CPython normally
falls back to its built-in ``_md5`` implementation, but the Python builds
shipped by FIPS-oriented distributions (RHEL, Fedora) deliberately strip that
fallback so the policy cannot be bypassed.  Passing ``usedforsecurity=False``
is the supported escape hatch: it declares the non-security use, and those
builds then allow the digest.

Routing every call site through this module means the declaration is made in
exactly one place, and the next environment quirk is a one-file change rather
than a hundred-site sweep.  Call these helpers instead of ``hashlib``
directly for anything that fingerprints content.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = [
    "content_md5",
    "content_sha1",
    "file_md5",
    "new_md5",
]

# Bytes read per chunk when hashing a file incrementally.
_CHUNK_SIZE = 8192


def _as_bytes(data: bytes | str) -> bytes:
    """Coerce *data* to bytes, encoding text as UTF-8."""
    return data.encode("utf-8") if isinstance(data, str) else data


def content_md5(data: bytes | str) -> str:
    """Return the hex MD5 fingerprint of *data*.

    Accepts bytes or text; text is encoded as UTF-8 first.
    """
    return hashlib.md5(_as_bytes(data), usedforsecurity=False).hexdigest()


def content_sha1(data: bytes | str) -> str:
    """Return the hex SHA-1 fingerprint of *data*.

    Accepts bytes or text; text is encoded as UTF-8 first.
    """
    return hashlib.sha1(_as_bytes(data), usedforsecurity=False).hexdigest()


def new_md5() -> hashlib._Hash:
    """Return a fresh incremental MD5 hasher for ``.update()`` streaming.

    Use :func:`file_md5` instead when the input is simply a file on disk.
    """
    return hashlib.md5(usedforsecurity=False)


def file_md5(file_path: Path | str) -> str:
    """Return the hex MD5 fingerprint of a file, read in constant memory."""
    h = new_md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()
