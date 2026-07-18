"""Local-archive support – extract zip/tar/rar archives and load their media.

This module centralises everything related to reading media out of an
archive **file** (as opposed to a plain directory).  It is shared by:

* the ``server_folder`` importer – which can take a single archive path in
  its folder field, or (with ``dig_archives`` enabled) extract any archives
  found *inside* the scanned folder; and
* the ``http_archive`` importer – which uses :func:`extract_archive` for
  downloaded URLs and routes local server paths through here too.

Origin model
------------
Media loaded from an archive carry a ``local_archive`` origin::

    {"importer": "local_archive", "params": {"path": "<abs archive path>",
                                             "media_type": "<output type>"}}

The origin records only the archive's path – never the extracted bytes – so
the system re-derives ``origin → archive → extracted file → embedding`` on
demand via
:class:`~vtscore.datasets.sources.local_archive.LocalArchiveSource`, in line
with the "no persisted vectors" rule.  Converter-produced media re-derive
through the converter origin's ``parent_importer`` / ``parent_path`` (see
:func:`~vtscore.converters.runner._build_converter_origin`); PDF pages
re-derive through the resolver's generic ``params.path`` fallback against the
cached extraction directory.

Extraction is cached under :data:`~vtscore.config.DATA_DIR` keyed by the
archive's path, size, and mtime, so re-imports and later resolves reuse the
same directory (and a replaced archive busts the cache).  Extracted contents
are source files, not embeddings/MLPs, so caching them on disk is consistent
with the existing ``http_archive`` resolve cache and the dataset-pickle
exception to the no-persist rule.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import threading
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional
from uuid import uuid4

from vtscore.config import DATA_DIR
from vtscore.security.archive import safe_tar_extract
from vtscore.security.path_validation import glob_top_level, rglob_follow_symlinks

if TYPE_CHECKING:
    from vtscore.datasets.importers.base import SourceSpec

__all__ = [
    "ARCHIVE_SUFFIXES",
    "append_medias",
    "build_local_archive_origin",
    "extract_archive",
    "extract_archive_cached",
    "find_archives",
    "is_archive_path",
    "iter_archive_chunks",
    "load_archive_into",
]

ProgressCallback = Callable[[str, str, int, int], None]

#: File suffixes recognised as archives.  Compound tar suffixes are listed
#: explicitly so :func:`is_archive_path` matches e.g. ``foo.tar.gz``.
ARCHIVE_SUFFIXES: tuple[str, ...] = (
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz2",
    ".tar.xz",
    ".txz",
    ".rar",
)

#: Glob patterns matching :data:`ARCHIVE_SUFFIXES`, for folder scans.
_ARCHIVE_GLOBS: tuple[str, ...] = tuple(f"*{suffix}" for suffix in ARCHIVE_SUFFIXES)

#: Synthetic origin importer name for media loaded out of a local archive.
#: Resolves through
#: :class:`~vtscore.datasets.sources.local_archive.LocalArchiveSource`.
LOCAL_ARCHIVE_IMPORTER = "local_archive"

_cache_lock = threading.Lock()


def _default_progress() -> ProgressCallback:
    from vtscore.concurrency.progress import get_thread_progress, update_progress  # noqa: PLC0415

    cb = get_thread_progress()
    return cb if cb is not None else update_progress


def is_archive_path(path: str | Path) -> bool:
    """Return ``True`` when *path*'s name ends with a known archive suffix.

    Purely lexical – does not touch the filesystem – so it works for paths
    that don't (yet) exist.
    """
    return Path(path).name.lower().endswith(ARCHIVE_SUFFIXES)


def find_archives(folder: Path, recursive: bool) -> list[Path]:
    """Return the archive files inside *folder* (sorted, de-duplicated)."""
    found: set[Path] = set()
    for pattern in _ARCHIVE_GLOBS:
        if recursive:
            found.update(rglob_follow_symlinks(folder, pattern))
        else:
            found.update(glob_top_level(folder, pattern))
    return sorted(found)


def build_local_archive_origin(archive_path: Path, output_type: str) -> dict[str, Any]:
    """Build the ``local_archive`` origin dict for media from *archive_path*."""
    return {
        "importer": LOCAL_ARCHIVE_IMPORTER,
        "params": {"path": str(Path(archive_path).resolve()), "media_type": output_type},
    }


def _reject_traversal(extract_dir_resolved: Path, member_name: str) -> None:
    """Raise ValueError if *member_name* would extract outside extract_dir.

    Validates before extraction so a malicious member is never written to disk.
    Rejects absolute paths, ``..`` traversal, and any name that – once joined
    and normalised – escapes the extraction root.
    """
    # Reject absolute member names outright; on Windows they'd also drop the
    # root prefix when joined, but we want to fail loudly either way.
    if member_name.startswith(("/", "\\")) or (len(member_name) > 1 and member_name[1] == ":"):
        raise ValueError(f"Path traversal detected in archive: {member_name}")

    # Use os.path.normpath-style joining without resolving symlinks: the
    # extract_dir is freshly created and contains no symlinks yet, and we
    # don't want a symlink planted by an earlier member in the same archive
    # to mask a later traversal.
    target = Path(os.path.normpath(extract_dir_resolved / member_name))
    if target != extract_dir_resolved and not target.is_relative_to(extract_dir_resolved):
        raise ValueError(f"Path traversal detected in archive: {member_name}")


def extract_archive(
    archive_path: Path,
    extract_dir: Path,
    on_progress: Optional[ProgressCallback] = None,
) -> None:
    """Extract *archive_path* into *extract_dir*, supporting zip/tar/rar.

    Supported formats: ``.zip``, ``.tar``, ``.tar.gz`` / ``.tgz``,
    ``.tar.bz2``, ``.tar.xz``, and ``.rar`` (the last requires the optional
    ``rarfile`` package).  Every member is validated against path traversal
    before it is written to disk.
    """
    if on_progress is None:
        on_progress = _default_progress()

    name = archive_path.name.lower()
    extract_dir_resolved = extract_dir.resolve()

    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            members = zf.namelist()
            total = len(members)
            for i, member in enumerate(members, 1):
                on_progress(
                    "extracting",
                    f"Extracting {member.split('/')[-1]}...",
                    i,
                    total,
                )
                _reject_traversal(extract_dir_resolved, member)
                zf.extract(member, extract_dir)

    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tf:
            members = tf.getmembers()
            total = len(members)
            for i, member in enumerate(members, 1):
                on_progress(
                    "extracting",
                    f"Extracting {member.name.split('/')[-1]}...",
                    i,
                    total,
                )
                safe_tar_extract(tf, member, extract_dir)

    elif name.endswith(".rar"):
        try:
            import rarfile  # optional dependency  # pyright: ignore[reportMissingImports]  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "RAR extraction requires the 'rarfile' package. Install it with: pip install rarfile"
            ) from exc
        with rarfile.RarFile(archive_path, "r") as rf:
            members = rf.namelist()
            total = len(members)
            for i, member in enumerate(members, 1):
                on_progress(
                    "extracting",
                    f"Extracting {member.split('/')[-1]}...",
                    i,
                    total,
                )
                _reject_traversal(extract_dir_resolved, member)
                rf.extract(member, extract_dir)

    else:
        raise ValueError(
            f"Unsupported archive format: {archive_path.name}. "
            "Supported formats: .zip, .tar, .tar.gz, .tar.bz2, .tar.xz, .rar"
        )


def extract_archive_cached(
    archive_path: str | Path,
    on_progress: Optional[ProgressCallback] = None,
) -> Path:
    """Extract *archive_path* into a cached directory and return it.

    The cache key folds in the archive's absolute path, size, and mtime, so
    repeated imports/resolves of the same archive reuse one extraction while
    a replaced archive busts the cache.  Concurrent extractions of the same
    archive race safely: each writes to a unique temp dir and the first to
    finish publishes it; the losers discard their copy.
    """
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    if not is_archive_path(archive_path):
        raise ValueError(
            f"Not a recognised archive: {archive_path.name}. "
            "Supported formats: .zip, .tar, .tar.gz, .tar.bz2, .tar.xz, .rar"
        )

    resolved = archive_path.resolve()
    st = resolved.stat()
    signature = f"{resolved}|{st.st_size}|{st.st_mtime_ns}".encode()
    digest = hashlib.md5(signature).hexdigest()[:16]
    cached_dir = DATA_DIR / f"local_archive_{digest}"

    if cached_dir.is_dir():
        return cached_dir

    DATA_DIR.mkdir(exist_ok=True)
    tmp_dir = DATA_DIR / f"local_archive_tmp_{uuid4().hex[:12]}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        extract_archive(resolved, tmp_dir, on_progress)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    # Publish atomically.  Re-check under the lock: two concurrent extractions
    # of the same archive can both miss the earlier is_dir() check, and without
    # this guard they'd both try to rename onto the same destination.
    with _cache_lock:
        if cached_dir.is_dir():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            try:
                tmp_dir.rename(cached_dir)
            except OSError:
                # e.g. cross-device rename; fall back to a copy + cleanup.
                shutil.copytree(tmp_dir, cached_dir, dirs_exist_ok=True)
                shutil.rmtree(tmp_dir, ignore_errors=True)
    return cached_dir


def append_medias(dst: dict[int, dict[str, Any]], src: dict[int, dict[str, Any]]) -> int:
    """Merge *src* medias into *dst*, re-keying with fresh sequential IDs.

    Returns the number of medias added.  *dst* is never cleared.
    """
    next_id = max(dst.keys(), default=0) + 1
    added = 0
    for media in src.values():
        media["id"] = next_id
        dst[next_id] = media
        next_id += 1
        added += 1
    return added


def iter_archive_chunks(
    archive_path: str | Path,
    output_type: str,
    specs: list["SourceSpec"],
    chunk_size: int,
    *,
    thin: bool = False,
    content_vectors: dict[str, Any] | None = None,
    content_md5s: dict[str, str] | None = None,
    custom_metadata_map: dict[str, dict[str, Any]] | None = None,
    on_progress: Optional[ProgressCallback] = None,
) -> Iterator[dict[int, dict[str, Any]]]:
    """Extract *archive_path* and yield its media in chunks.

    Mirrors the folder import pipeline against the extracted directory:
    direct (no-converter) rows stream through
    :func:`~vtscore.datasets.loader.load_dataset_from_folder_chunked`, PDFs
    expand into images when *output_type* is ``"image"``, and converter rows
    run via :func:`~vtscore.converters.runner.run_converters_on_folder`.  All
    media carry a ``local_archive`` origin pointing at *archive_path* so they
    re-derive on demand.  Each yielded chunk is a self-contained medias dict
    with IDs starting at 1.
    """
    from vtscore.converters.runner import run_converters_on_folder  # noqa: PLC0415
    from vtscore.datasets.loader import load_dataset_from_folder_chunked  # noqa: PLC0415
    from vtscore.datasets.pdf import load_pdf_images_into  # noqa: PLC0415
    from vtscore.media import get  # noqa: PLC0415

    cached_dir = extract_archive_cached(archive_path, on_progress)
    origin = build_local_archive_origin(Path(archive_path), output_type)

    for spec in specs:
        if spec.converter is not None:
            continue
        mt = get(spec.source_type)
        try:
            yield from load_dataset_from_folder_chunked(
                cached_dir,
                mt.folder_import_name,
                chunk_size,
                thin=thin,
                origin=origin,
                content_vectors=content_vectors or None,
                content_md5s=content_md5s or None,
                custom_metadata_map=custom_metadata_map or None,
                recursive=True,
            )
        except ValueError:
            # No files of this source type inside the archive – keep going;
            # PDF / converter rows may still produce output.
            pass

    if output_type == "image":
        pdf_chunk: dict[int, dict[str, Any]] = {}
        load_pdf_images_into(cached_dir, pdf_chunk, thin=thin, recursive=True)
        if pdf_chunk:
            yield pdf_chunk

    converter_specs = [s for s in specs if s.converter is not None]
    if converter_specs:
        converter_chunk: dict[int, dict[str, Any]] = {}
        run_converters_on_folder(
            folder_path=cached_dir,
            converter_specs=converter_specs,
            target_media_type=output_type,
            medias=converter_chunk,
            thin=thin,
            base_origin=origin,
            recursive=True,
        )
        if converter_chunk:
            yield converter_chunk


# A chunk size large enough that the chunked loader yields everything in one
# pass – used by the non-chunked :func:`load_archive_into` accumulator.
_UNBOUNDED_CHUNK = 1_000_000_000


def load_archive_into(
    archive_path: str | Path,
    output_type: str,
    specs: list["SourceSpec"],
    medias: dict[int, dict[str, Any]],
    *,
    thin: bool = False,
    content_vectors: dict[str, Any] | None = None,
    content_md5s: dict[str, str] | None = None,
    custom_metadata_map: dict[str, dict[str, Any]] | None = None,
    on_progress: Optional[ProgressCallback] = None,
) -> int:
    """Extract *archive_path* and append its media into *medias*.

    Non-chunked accumulator over :func:`iter_archive_chunks`; *medias* is
    appended to (never cleared) so a caller can merge several archives (and a
    surrounding folder scan) into one dataset.  Returns the number of media
    added.
    """
    added = 0
    for chunk in iter_archive_chunks(
        archive_path,
        output_type,
        specs,
        _UNBOUNDED_CHUNK,
        thin=thin,
        content_vectors=content_vectors,
        content_md5s=content_md5s,
        custom_metadata_map=custom_metadata_map,
        on_progress=on_progress,
    ):
        added += append_medias(medias, chunk)
    return added
