"""Stream a single member out of a local tar/zip archive without extracting.

This is the no-extraction counterpart to :mod:`vtscore.datasets.archive`,
which extracts a whole archive to :data:`~vtscore.config.DATA_DIR`.  Here we
serve **one member on demand**: a ``{member_name: info}`` index is built once
per archive (cached, keyed by path/inode/size/mtime) and a byte range of a
single member is read by seeking within that member's file object.  So playback
of a media stored inside a multi-GB tar shard never materialises the member on disk
and, paired with HTTP Range, downloads only the bytes the browser actually
plays.

This is the mechanism a WebDataset-style corpus needs: tens of thousands of
audio/video chunks live in a handful of multi-GB ``shard_*.tar`` files, far too
large to extract a second on-disk copy of (multivent-raw's ``videos/`` alone is
4.1 TB).  A ``local_archive_member`` media records only ``{archive path,
member}`` and re-derives its bytes through here.

Per the "no persisted vectors / bytes" rule nothing here writes member bytes
anywhere: the only cache is the in-memory member **index** (TarInfo / ZipInfo
metadata, not content), which is rebuilt from the archive on the next start.
"""

from __future__ import annotations

import tarfile
import threading
import time
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

__all__ = [
    "LOCAL_ARCHIVE_MEMBER_IMPORTER",
    "ArchiveMemberError",
    "archive_member_ref",
    "build_archive_member_origin",
    "member_size",
    "read_member",
    "read_member_range",
]

#: Origin importer name for media served straight out of an archive member
#: (no extraction).  Resolves through
#: :class:`~vtscore.datasets.sources.local_archive_member.LocalArchiveMemberSource`.
LOCAL_ARCHIVE_MEMBER_IMPORTER = "local_archive_member"


class ArchiveMemberError(Exception):
    """Raised when an archive or one of its members cannot be read."""


def _cache_key(path: Path) -> tuple[str, int, int, int]:
    st = path.stat()
    return (str(path), st.st_ino, st.st_size, st.st_mtime_ns)


#: How long a cached index stays *unsettled* after it is built, measured on a
#: monotonic clock (so wall-clock or NFS server skew cannot affect it).
#:
#: Filesystems advance ``st_mtime`` only once per tick -- coarse enough on some
#: (HLTCOE's ``/scratch``, ext3, FAT) that two back-to-back writes share one
#: ``st_mtime_ns``.  An archive rewritten within one tick of the write we
#: indexed, to the same 512-block-padded size, therefore keeps an identical
#: cache key, and we would serve its stale index forever.  So while an entry is
#: younger than this window we re-verify it on every lookup against the
#: archive's first header block (cheap: one 512-byte read).  Once the window has
#: passed, any later rewrite necessarily stamps an mtime the cached key cannot
#: reproduce, so the stat key alone is authoritative and the probe stops --
#: steady-state reads pay nothing.
_SETTLE_NS = 2_000_000_000

#: Bytes of the archive head folded into the freshness probe: exactly one tar
#: header block (member name, size, checksum) or a zip local file header.
_PROBE_BYTES = 512


class _IndexEntry:
    """A cached member index plus what we need to re-verify its freshness."""

    __slots__ = ("first_seen_ns", "index", "probe", "settled")

    def __init__(self, index: dict[str, Any], probe: bytes) -> None:
        self.index = index
        self.probe = probe
        self.first_seen_ns = time.monotonic_ns()
        self.settled = False


def _read_probe(path: Path) -> bytes:
    """Return the archive's first :data:`_PROBE_BYTES` bytes (``b""`` on error)."""
    try:
        with path.open("rb") as f:
            return f.read(_PROBE_BYTES)
    except OSError:
        return b""


# Process-scoped index cache.  Keyed by (resolved path, inode, size, mtime_ns)
# so a replaced archive busts the cache; the value is an _IndexEntry wrapping a
# {member_name: info} map of tarfile.TarInfo / zipfile.ZipInfo objects
# (metadata only -- never bytes).
_index_cache: dict[tuple[str, int, int, int], _IndexEntry] = {}
_index_lock = threading.Lock()


#: Upper bound on simultaneously-open archive handles.  Reads seek within a
#: reused handle instead of reopening the archive on every HTTP Range request,
#: so a windowed-audio corpus no longer exhausts the process fd limit.  The
#: least-recently-used archive is closed once the pool exceeds this size.
_MAX_OPEN_ARCHIVES = 32


class _PooledArchive:
    """A reusable open handle to one tar/zip archive.

    Holds a single ``tarfile.TarFile`` / ``zipfile.ZipFile`` open across many
    range reads.  Reads on the same archive are serialised by ``_lock`` because
    ``TarFile`` (and, pre-parallel, ``ZipFile``) share one underlying file
    position across member file objects, so concurrent seeks would corrupt each
    other.  Different archives use different instances and read in parallel.
    """

    def __init__(self, path: Path, is_zip: bool) -> None:
        self.path = path
        self.is_zip = is_zip
        self._lock = threading.Lock()
        self._handle: Any = _open_archive(path, is_zip)

    def read(self, info: Any, start: int, length: int | None) -> bytes:
        with self._lock:
            handle = self._handle
            if handle is not None:
                return _read_from(handle, info, start, length, self.is_zip)
            # Evicted/closed out from under a stale reference: fall back to a
            # one-shot open/close for just this read so no fd is retained.
            handle = _open_archive(self.path, self.is_zip)
            try:
                return _read_from(handle, info, start, length, self.is_zip)
            finally:
                handle.close()

    def close(self) -> None:
        """Close the underlying handle, waiting for any in-flight read first."""
        with self._lock:
            handle, self._handle = self._handle, None
        if handle is not None:
            try:
                handle.close()
            except Exception:  # noqa: BLE001 - best-effort fd release
                pass


# Process-scoped pool of open archive handles, keyed like ``_index_cache`` and
# ordered least-recently-used first for eviction.
_handle_cache: OrderedDict[tuple[str, int, int, int], _PooledArchive] = OrderedDict()
_handle_lock = threading.Lock()


def _open_archive(path: Path, is_zip: bool) -> Any:
    if is_zip:
        return zipfile.ZipFile(path, "r")
    try:
        return tarfile.open(path, "r:*")
    except tarfile.TarError as exc:
        raise ArchiveMemberError(f"Not a readable tar/zip archive: {path} ({exc})") from exc


def _read_from(handle: Any, info: Any, start: int, length: int | None, is_zip: bool) -> bytes:
    if is_zip:
        with handle.open(info, "r") as f:
            if start:
                f.seek(start)
            return f.read() if length is None else f.read(length)
    extracted = handle.extractfile(info)
    if extracted is None:
        raise ArchiveMemberError(f"Member {info.name!r} is not a regular file")
    with extracted as f:
        if start:
            f.seek(start)
        return f.read() if length is None else f.read(length)


def _pooled_archive(path: Path, is_zip: bool) -> _PooledArchive:
    """Return the pooled handle for *path*, opening and caching it on first use.

    Insertion may evict the least-recently-used archive; evictions are closed
    outside ``_handle_lock`` (each ``close`` waits on its own read lock) so the
    pool never blocks on a slow ``close`` and locks are never nested.
    """
    key = _cache_key(path)
    with _handle_lock:
        pooled = _handle_cache.get(key)
        if pooled is not None:
            _handle_cache.move_to_end(key)
            return pooled

    # Open outside the pool lock; another thread may race us to the same key.
    opened = _PooledArchive(path, is_zip)
    to_close: list[_PooledArchive] = []
    with _handle_lock:
        existing = _handle_cache.get(key)
        if existing is not None:
            _handle_cache.move_to_end(key)
            pooled = existing
            to_close.append(opened)
        else:
            _handle_cache[key] = opened
            pooled = opened
            while len(_handle_cache) > _MAX_OPEN_ARCHIVES:
                _, evicted = _handle_cache.popitem(last=False)
                to_close.append(evicted)
    for handle in to_close:
        handle.close()
    return pooled


def _reset_pool() -> None:
    """Close and drop all pooled archive handles (test/shutdown hook)."""
    with _handle_lock:
        handles = list(_handle_cache.values())
        _handle_cache.clear()
    for handle in handles:
        handle.close()


def _is_zip(path: Path) -> bool:
    """Return True when *path* is a zip archive (sniffed, then by suffix)."""
    try:
        if zipfile.is_zipfile(path):
            return True
    except OSError:
        pass
    return path.name.lower().endswith(".zip")


def _build_index(path: Path) -> dict[str, Any]:
    """Build a ``{member_name: info}`` index for *path* (tar or zip)."""
    if _is_zip(path):
        with zipfile.ZipFile(path, "r") as zf:
            return {info.filename: info for info in zf.infolist() if not info.is_dir()}
    try:
        with tarfile.open(path, "r:*") as tf:
            return {m.name: m for m in tf.getmembers() if m.isfile()}
    except tarfile.TarError as exc:
        raise ArchiveMemberError(f"Not a readable tar/zip archive: {path} ({exc})") from exc


def _invalidate(key: tuple[str, int, int, int]) -> None:
    """Drop the cached index *and* the pooled handle for *key* (archive changed).

    The two locks are taken one after the other, never nested, and the evicted
    handle is closed outside both (``close`` waits on its own read lock).
    """
    with _index_lock:
        _index_cache.pop(key, None)
    with _handle_lock:
        pooled = _handle_cache.pop(key, None)
    if pooled is not None:
        pooled.close()


def _index(path: Path) -> dict[str, Any]:
    """Return the cached member index for *path*, building it on first use.

    A hit is trusted outright once the entry has settled (see
    :data:`_SETTLE_NS`); before that it is re-verified against the archive's
    first header block, so an archive rewritten within one mtime tick to the
    same padded size -- which leaves the stat key untouched -- is still noticed.
    """
    key = _cache_key(path)
    with _index_lock:
        entry = _index_cache.get(key)
    if entry is not None:
        if entry.settled:
            return entry.index
        if _read_probe(path) == entry.probe:
            # Past the coarse-mtime window the stat key can no longer be
            # reproduced by a later rewrite, so stop probing this entry.
            entry.settled = time.monotonic_ns() - entry.first_seen_ns > _SETTLE_NS
            return entry.index
        _invalidate(key)
    # Probe *before* building so an archive rewritten mid-build leaves a probe
    # that no longer matches, and the next lookup rebuilds rather than trusting.
    probe = _read_probe(path)
    index = _build_index(path)
    with _index_lock:
        _index_cache[key] = _IndexEntry(index, probe)
    return index


def _resolve(archive_path: str | Path, member: str) -> tuple[Path, Any, bool]:
    """Resolve *(archive_path, member)* to ``(path, info, is_zip)``.

    Raises :class:`ArchiveMemberError` if the archive is missing or the member
    is not present in it.
    """
    path = Path(archive_path)
    if not path.is_file():
        raise ArchiveMemberError(f"Archive not found: {path}")
    index = _index(path)
    info = index.get(member)
    if info is None:
        raise ArchiveMemberError(f"Member {member!r} not found in archive {path}")
    return path, info, isinstance(info, zipfile.ZipInfo)


def member_size(archive_path: str | Path, member: str) -> int:
    """Return the uncompressed size in bytes of *member* inside *archive_path*."""
    _path, info, is_zip = _resolve(archive_path, member)
    return int(info.file_size if is_zip else info.size)


def read_member(archive_path: str | Path, member: str) -> bytes:
    """Read and return the **whole** *member* from *archive_path*.

    Used by callers that genuinely need every byte (computing a content hash,
    embedding, transcoding a non-streamable container).  Range-served playback
    should use :func:`read_member_range` so it never holds the whole member in
    memory.
    """
    return read_member_range(archive_path, member, 0, None)


def read_member_range(
    archive_path: str | Path,
    member: str,
    start: int,
    length: int | None,
) -> bytes:
    """Read *length* bytes of *member* starting at byte offset *start*.

    ``length=None`` reads to the end of the member.  The archive handle is
    drawn from a bounded process-scoped pool (see :func:`_pooled_archive`) and
    left open for reuse, so a burst of HTTP Range requests over a windowed-audio
    corpus reuses a handful of fds instead of opening one per request.  The
    member's file object is *seeked* to *start* so only the requested slice is
    read -- the whole member is never materialised for a partial request.
    """
    path, info, is_zip = _resolve(archive_path, member)
    if start < 0:
        start = 0
    return _pooled_archive(path, is_zip).read(info, start, length)


def archive_member_ref(media: dict[str, Any]) -> tuple[str, str] | None:
    """Return *(archive_path, member)* for an archive-member media, or ``None``.

    Reads the top-level ``archive_member`` convenience field first (set at
    import and on re-derivation) and falls back to ``origin.params`` (the
    channel that survives the pickle round-trip), so a freshly imported media
    and one reopened from disk resolve identically.
    """
    ref = media.get("archive_member")
    if isinstance(ref, dict):
        path = ref.get("path")
        member = ref.get("member")
        if path and member:
            return str(path), str(member)

    origin = media.get("origin")
    if not isinstance(origin, dict) or origin.get("importer") != LOCAL_ARCHIVE_MEMBER_IMPORTER:
        return None
    params = origin.get("params", {})
    path = params.get("archive_path") or params.get("path")
    member = params.get("member")
    if path and member:
        return str(path), str(member)
    return None


def build_archive_member_origin(archive_path: str | Path, member: str, media_type: str) -> dict[str, Any]:
    """Build the ``local_archive_member`` origin for a member of *archive_path*.

    Records only the archive path + member name + output media type -- never
    the member bytes -- so the media re-derives ``origin -> archive member ->
    bytes`` on demand through
    :class:`~vtscore.datasets.sources.local_archive_member.LocalArchiveMemberSource`.
    """
    return {
        "importer": LOCAL_ARCHIVE_MEMBER_IMPORTER,
        "params": {
            "archive_path": str(Path(archive_path).resolve()),
            "member": str(member),
            "media_type": media_type,
        },
    }
