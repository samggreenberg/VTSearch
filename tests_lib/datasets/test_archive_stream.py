"""Tests for no-extraction archive-member streaming (``archive_stream``)."""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from vtscore.datasets.archive_stream import (
    LOCAL_ARCHIVE_MEMBER_IMPORTER,
    ArchiveMemberError,
    archive_member_ref,
    build_archive_member_origin,
    member_size,
    read_member,
    read_member_range,
)

MEMBER_A = b"the quick brown fox jumps over the lazy dog" * 4
MEMBER_B = b"second member payload \x00\x01\x02 binary-ish" * 8


def _make_tar(tmp_path: Path) -> Path:
    archive = tmp_path / "shard_000000.tar"
    with tarfile.open(archive, "w") as tf:
        for name, payload in (("chunk_a.mp4", MEMBER_A), ("sub/chunk_b.aac", MEMBER_B)):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    return archive


def _make_zip(tmp_path: Path) -> Path:
    archive = tmp_path / "shard.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("chunk_a.mp4", MEMBER_A)
        zf.writestr("sub/chunk_b.aac", MEMBER_B)
    return archive


class TestReadMember:
    def test_member_size_tar(self, tmp_path):
        archive = _make_tar(tmp_path)
        assert member_size(archive, "chunk_a.mp4") == len(MEMBER_A)
        assert member_size(archive, "sub/chunk_b.aac") == len(MEMBER_B)

    def test_read_whole_member_tar(self, tmp_path):
        archive = _make_tar(tmp_path)
        assert read_member(archive, "chunk_a.mp4") == MEMBER_A
        assert read_member(archive, "sub/chunk_b.aac") == MEMBER_B

    def test_read_whole_member_zip(self, tmp_path):
        archive = _make_zip(tmp_path)
        assert read_member(archive, "chunk_a.mp4") == MEMBER_A
        assert read_member(archive, "sub/chunk_b.aac") == MEMBER_B

    @pytest.mark.parametrize("make", [_make_tar, _make_zip])
    def test_partial_range_reads_only_slice(self, tmp_path, make):
        archive = make(tmp_path)
        # A middle slice should equal the corresponding bytes of the member.
        start, length = 10, 17
        got = read_member_range(archive, "chunk_a.mp4", start, length)
        assert got == MEMBER_A[start : start + length]

    @pytest.mark.parametrize("make", [_make_tar, _make_zip])
    def test_range_to_end_when_length_none(self, tmp_path, make):
        archive = make(tmp_path)
        start = 25
        got = read_member_range(archive, "chunk_a.mp4", start, None)
        assert got == MEMBER_A[start:]

    def test_missing_member_raises(self, tmp_path):
        archive = _make_tar(tmp_path)
        with pytest.raises(ArchiveMemberError):
            member_size(archive, "does_not_exist.mp4")

    def test_missing_archive_raises(self, tmp_path):
        with pytest.raises(ArchiveMemberError):
            read_member(tmp_path / "nope.tar", "x")

    def test_index_cache_busts_on_mtime(self, tmp_path):
        archive = _make_tar(tmp_path)
        assert member_size(archive, "chunk_a.mp4") == len(MEMBER_A)
        # Rewrite the archive with a differently-sized member; the cache key
        # folds in size+mtime, so the new index must be picked up.
        new_payload = MEMBER_A + b"EXTRA"
        with tarfile.open(archive, "w") as tf:
            info = tarfile.TarInfo("chunk_a.mp4")
            info.size = len(new_payload)
            tf.addfile(info, io.BytesIO(new_payload))
        assert read_member(archive, "chunk_a.mp4") == new_payload


class TestArchiveMemberRef:
    def test_ref_from_top_level_field(self):
        media = {"archive_member": {"path": "/a/shard.tar", "member": "m.mp4"}}
        assert archive_member_ref(media) == ("/a/shard.tar", "m.mp4")

    def test_ref_from_origin_params(self):
        origin = build_archive_member_origin("/a/shard.tar", "m.mp4", "video")
        ref = archive_member_ref({"origin": origin})
        assert ref is not None
        path, member = ref
        assert Path(path).name == "shard.tar"
        assert member == "m.mp4"

    def test_origin_importer_name_gates_fallback(self):
        # A non-archive-member origin must not be misread as one.
        media = {"origin": {"importer": "server_folder", "params": {"path": "/x", "member": "m"}}}
        assert archive_member_ref(media) is None

    def test_returns_none_when_absent(self):
        assert archive_member_ref({"media_path": "/x/y.wav"}) is None

    def test_build_origin_shape(self):
        origin = build_archive_member_origin("/a/shard.tar", "m.mp4", "video")
        assert origin["importer"] == LOCAL_ARCHIVE_MEMBER_IMPORTER
        assert origin["params"]["member"] == "m.mp4"
        assert origin["params"]["media_type"] == "video"
        assert Path(origin["params"]["archive_path"]).name == "shard.tar"
