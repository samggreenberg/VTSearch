"""Tests for :func:`vtscore.security.archive.safe_tar_extract`.

Covers the three classes of malicious tar member the centralised extractor
must neutralise: absolute-path members, ``../`` traversal members, and
symlink members that point outside the destination.  Each case is exercised on
both extraction paths — the PEP 706 ``data_filter`` branch (native on this
interpreter) and the manual fallback used on interpreters that lack it — by
forcing :data:`~vtscore.security.archive._HAS_DATA_FILTER` off for the second.
"""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path

import pytest

import vtscore.security.archive as safe_archive
from vtscore.security.archive import safe_tar_extract

# Errors either branch may raise for an unsafe member: ValueError from the
# manual fallback, tarfile.TarError (FilterError subclasses) from data_filter.
_UNSAFE_ERRORS = (ValueError, tarfile.TarError)


@pytest.fixture(params=[True, False], ids=["data_filter", "fallback"])
def branch(request, monkeypatch):
    """Run each test on both the data_filter and the manual-fallback path."""
    if request.param and not safe_archive._HAS_DATA_FILTER:  # pragma: no cover
        pytest.skip("interpreter lacks tarfile.data_filter")
    monkeypatch.setattr(safe_archive, "_HAS_DATA_FILTER", request.param)
    return request.param


def _write_tar(path: Path, build) -> None:
    """Create a tar at *path*; *build* receives the open TarFile to add members."""
    with tarfile.open(path, "w") as tf:
        build(tf)


def _add_file(tf: tarfile.TarFile, name: str, data: bytes = b"payload") -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def _add_symlink(tf: tarfile.TarFile, name: str, target: str) -> None:
    info = tarfile.TarInfo(name)
    info.type = tarfile.SYMTYPE
    info.linkname = target
    tf.addfile(info)


def _extract_all(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            safe_tar_extract(tf, member, dest)


def test_benign_member_extracts_inside_dest(branch, tmp_path):
    archive = tmp_path / "ok.tar"
    _write_tar(archive, lambda tf: _add_file(tf, "sub/good.txt", b"hello"))
    dest = tmp_path / "out"
    _extract_all(archive, dest)
    assert (dest / "sub" / "good.txt").read_bytes() == b"hello"


def test_absolute_path_member_confined_to_dest(branch, tmp_path):
    """An absolute member name is stripped of its leading ``/`` (PEP 706) and
    lands *inside* dest — never at the real filesystem location."""
    archive = tmp_path / "abs.tar"
    # Point the absolute name at a sentinel outside dest we can assert stays
    # untouched, rather than a real system path.
    outside = tmp_path / "evil_abs.txt"
    abs_name = str(outside)  # e.g. "/tmp/.../evil_abs.txt"
    assert os.path.isabs(abs_name)
    _write_tar(archive, lambda tf: _add_file(tf, abs_name, b"nope"))

    dest = tmp_path / "out"
    _extract_all(archive, dest)

    # The absolute target must not have been written.
    assert not outside.exists()
    # The payload landed under dest with the root stripped.
    landed = dest / abs_name.lstrip("/")
    assert landed.read_bytes() == b"nope"


def test_traversal_member_rejected(branch, tmp_path):
    archive = tmp_path / "trav.tar"
    _write_tar(archive, lambda tf: _add_file(tf, "../escape.txt", b"nope"))
    dest = tmp_path / "out"
    with pytest.raises(_UNSAFE_ERRORS):
        _extract_all(archive, dest)
    assert not (tmp_path / "escape.txt").exists()


def test_absolute_symlink_member_rejected(branch, tmp_path):
    archive = tmp_path / "link.tar"
    _write_tar(archive, lambda tf: _add_symlink(tf, "link", "/etc/passwd"))
    dest = tmp_path / "out"
    with pytest.raises(_UNSAFE_ERRORS):
        _extract_all(archive, dest)
    assert not (dest / "link").exists()


def test_escaping_relative_symlink_member_rejected(branch, tmp_path):
    archive = tmp_path / "link2.tar"
    _write_tar(archive, lambda tf: _add_symlink(tf, "link", "../../secret"))
    dest = tmp_path / "out"
    with pytest.raises(_UNSAFE_ERRORS):
        _extract_all(archive, dest)
    assert not (dest / "link").exists()
