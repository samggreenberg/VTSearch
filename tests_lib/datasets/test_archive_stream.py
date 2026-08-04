"""Tests for no-extraction archive-member streaming (``archive_stream``)."""

from __future__ import annotations

import io
import os
import tarfile
import threading
import zipfile
from pathlib import Path

import pytest

from vtscore.datasets import archive_stream
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


def _make_one_member_tar(path: Path, payload: bytes = MEMBER_A, name: str = "m.bin") -> Path:
    with tarfile.open(path, "w") as tf:
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    return path


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
        _make_one_member_tar(archive, new_payload, "chunk_a.mp4")
        # Stamp a distinct mtime rather than trusting the filesystem's timestamp
        # resolution: this asserts the cache-key contract, not the clock.
        bumped = archive.stat().st_mtime_ns + 10**9
        os.utime(archive, ns=(bumped, bumped))
        assert read_member(archive, "chunk_a.mp4") == new_payload

    def test_index_cache_busts_on_same_tick_rewrite(self, tmp_path):
        """A rewrite that leaves (size, mtime) untouched must still be noticed.

        tar pads members to 512-byte blocks, so a member that grows by a few
        bytes can leave the archive's size unchanged; on a filesystem with
        coarse mtime the rewrite can also land in the same tick as the write we
        indexed.  Both are forced here so the stat key is *identical* and only
        the header-block probe can catch the change.
        """
        archive = _make_one_member_tar(tmp_path / "shard.tar", MEMBER_A, "chunk_a.mp4")
        stamp = archive.stat().st_mtime_ns
        size_before = archive.stat().st_size
        assert member_size(archive, "chunk_a.mp4") == len(MEMBER_A)

        new_payload = MEMBER_A + b"EXTRA"
        _make_one_member_tar(archive, new_payload, "chunk_a.mp4")
        os.utime(archive, ns=(stamp, stamp))  # same mtime tick as the indexed write
        assert archive.stat().st_size == size_before  # same 512-block padding
        assert archive.stat().st_mtime_ns == stamp

        assert read_member(archive, "chunk_a.mp4") == new_payload

    def test_settled_index_stops_probing(self, tmp_path):
        """Past the settle window a hit is served without re-reading the head."""
        archive = _make_tar(tmp_path)
        assert member_size(archive, "chunk_a.mp4") == len(MEMBER_A)
        entry = archive_stream._index_cache[archive_stream._cache_key(archive)]
        assert not entry.settled
        # Age the entry past the window; the next lookup settles it for good.
        entry.first_seen_ns -= archive_stream._SETTLE_NS + 1
        assert member_size(archive, "chunk_a.mp4") == len(MEMBER_A)
        assert entry.settled


class TestHandlePool:
    def test_repeated_reads_reuse_one_handle(self, tmp_path):
        archive_stream._reset_pool()
        archive = _make_tar(tmp_path)
        for _ in range(50):
            assert read_member_range(archive, "chunk_a.mp4", 0, 5) == MEMBER_A[:5]
        # A single pooled handle serves every request -- no fd-per-read leak.
        assert len(archive_stream._handle_cache) == 1
        pooled = archive_stream._handle_cache[archive_stream._cache_key(archive)]
        assert pooled._handle is not None

    def test_zip_reads_reuse_one_handle(self, tmp_path):
        archive_stream._reset_pool()
        archive = _make_zip(tmp_path)
        for _ in range(20):
            assert read_member(archive, "chunk_a.mp4") == MEMBER_A
        assert len(archive_stream._handle_cache) == 1

    def test_pool_bounds_open_handles_and_closes_evicted(self, tmp_path):
        archive_stream._reset_pool()
        first = _make_one_member_tar(tmp_path / "shard_0.tar")
        assert read_member(first, "m.bin") == MEMBER_A
        pooled_first = archive_stream._handle_cache[archive_stream._cache_key(first)]
        assert pooled_first._handle is not None

        # Fill the pool past its bound; the LRU archive (the first) is evicted.
        for i in range(1, archive_stream._MAX_OPEN_ARCHIVES + 1):
            a = _make_one_member_tar(tmp_path / f"shard_{i}.tar")
            assert read_member(a, "m.bin") == MEMBER_A

        assert len(archive_stream._handle_cache) == archive_stream._MAX_OPEN_ARCHIVES
        assert archive_stream._cache_key(first) not in archive_stream._handle_cache
        assert pooled_first._handle is None  # closed on eviction, fd released
        # Re-reading an evicted archive transparently re-pools it.
        assert read_member(first, "m.bin") == MEMBER_A

    def test_read_through_closed_handle_falls_back(self, tmp_path):
        archive_stream._reset_pool()
        archive = _make_tar(tmp_path)
        read_member(archive, "chunk_a.mp4")
        pooled = archive_stream._handle_cache[archive_stream._cache_key(archive)]
        pooled.close()  # simulate eviction under a stale reference
        _path, info, _is_zip = archive_stream._resolve(archive, "chunk_a.mp4")
        # A stale reference still returns correct bytes via a one-shot open.
        assert pooled.read(info, 0, None) == MEMBER_A

    def test_concurrent_range_reads_same_archive(self, tmp_path):
        archive_stream._reset_pool()
        archive = _make_tar(tmp_path)
        results: dict[int, bytes] = {}
        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                start = i % 10
                results[i] = read_member_range(archive, "chunk_a.mp4", start, 12)
            except Exception as exc:  # noqa: BLE001 - surfaced via `errors`
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(40)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        for i, got in results.items():
            start = i % 10
            assert got == MEMBER_A[start : start + 12]
        # Serialised reads share the single pooled handle.
        assert len(archive_stream._handle_cache) == 1

    def test_reset_pool_closes_handles(self, tmp_path):
        archive_stream._reset_pool()
        archive = _make_tar(tmp_path)
        read_member(archive, "chunk_a.mp4")
        pooled = archive_stream._handle_cache[archive_stream._cache_key(archive)]
        archive_stream._reset_pool()
        assert len(archive_stream._handle_cache) == 0
        assert pooled._handle is None


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
