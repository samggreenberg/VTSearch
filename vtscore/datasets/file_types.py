"""Best-effort file-type labelling for media dicts.

The dataset registry stamps a ``file_type_counts`` histogram at ingest so the
Stats window can show what a dataset is made of.  Deriving that histogram from
each item's ``filename`` alone works for folder-shaped imports but collapses
for service-style importers, which name items after an opaque content id
(``"a7f3c9e2"``) rather than a file on disk: every item lands in one useless
``(unknown)`` bucket, and the Stats window reports nothing.

:func:`media_file_type` therefore walks a chain of increasingly indirect
signals — the filename, the origin name, the backing path, the backing URL and
finally the media bytes' magic numbers — returning the first that yields a
plausible type.  Nothing here does I/O: only bytes that are *already* in the
media dict are sniffed, so counting a large dataset stays cheap and never
reaches out to the network or the filesystem.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

__all__ = ["UNKNOWN_FILE_TYPE", "count_file_types", "media_file_type", "sniff_file_type"]

#: Bucket for items whose type no signal could establish.  Parenthesised so
#: the frontend can tell a sentinel from a real extension and skip the leading
#: dot it renders for ``jpg`` / ``wav`` / ….
UNKNOWN_FILE_TYPE = "(unknown)"

#: Longest run of characters after the final dot still treated as an
#: extension.  Keeps ``photo.2024-06-01-final`` and ``id.9f3ca77e1b`` out of
#: the histogram as bogus one-item "extensions".
_MAX_EXT_LEN = 5


def _ext_from_name(name: str) -> str:
    """Return the lowercase extension of *name*, or ``""`` when it has none.

    Only the final path segment is considered (so ``v1.2/audio`` has no
    extension), and the candidate must look like a real extension: short, and
    alphanumeric ASCII with at least one letter.
    """
    if not name:
        return ""
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    if "." not in base:
        return ""
    ext = base.rsplit(".", 1)[-1].lower()
    if not ext or len(ext) > _MAX_EXT_LEN:
        return ""
    if not (ext.isascii() and ext.isalnum()) or ext.isdigit():
        return ""
    return ext


def _ext_from_url(url: str) -> str:
    """Return the extension of *url*'s path, ignoring its query and fragment."""
    if not url:
        return ""
    path = url.split("#", 1)[0].split("?", 1)[0]
    return _ext_from_name(path)


def _ftyp_file_type(brand: bytes) -> str:
    """Map an ISO-BMFF ``ftyp`` major brand to a conventional extension."""
    b = brand.lower()
    if b == b"qt  ":
        return "mov"
    if b.startswith(b"m4a"):
        return "m4a"
    if b.startswith(b"m4v"):
        return "m4v"
    if b in (b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1"):
        return "heic"
    if b in (b"avif", b"avis"):
        return "avif"
    return "mp4"


def sniff_file_type(data: bytes) -> str:  # noqa: C901,PLR0911,PLR0912
    """Return a conventional extension for *data*'s format, or ``""``.

    Recognises the container formats VTSearch actually ingests (images, audio,
    video, PDFs and the archive types importers unpack) from their leading
    magic numbers.  The returned label is the format's usual extension, not
    the item's real filename: a JPEG named ``a7f3c9e2`` counts as ``jpg``.
    """
    head = bytes(data[:64])
    if len(head) < 4:
        return ""
    if head[:3] == b"\xff\xd8\xff":
        return "jpg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "tif"
    if head[:4] == b"%PDF":
        return "pdf"
    if head[:4] == b"fLaC":
        return "flac"
    if head[:4] == b"OggS":
        return "ogg"
    if head[:4] == b"RIFF":
        # RIFF is a family: the type lives in the fourcc after the size field.
        return {b"WEBP": "webp", b"WAVE": "wav", b"AVI ": "avi"}.get(head[8:12], "")
    if head[:4] == b"FORM" and head[8:12] in (b"AIFF", b"AIFC"):
        return "aiff"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        # Matroska and WebM share the EBML magic; the DocType string sits in
        # the EBML header, well inside the bytes we already read.
        return "webm" if b"webm" in head else "mkv"
    if head[4:8] == b"ftyp":
        return _ftyp_file_type(head[8:12])
    if head[:3] == b"FLV":
        return "flv"
    if head[:3] == b"ID3":
        return "mp3"
    if head[:2] == b"\x1f\x8b":
        return "gz"
    if head[:4] == b"7z\xbc\xaf":
        return "7z"
    if head[:4] == b"Rar!":
        return "rar"
    if head[:2] == b"PK":
        # Also covers the zip-backed office formats; without reading the
        # central directory we can only honestly say "zip".
        return "zip"
    if head[:2] == b"BM":
        return "bmp"
    if head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:
        # MPEG audio frame sync, for MP3s with no ID3 tag.  Checked last
        # because it is the loosest test here.
        return "mp3"
    return ""


def media_file_type(media: Mapping[str, Any]) -> str:
    """Return a file-type label for one media dict.

    Tries the item's name-shaped fields first (they carry what the user would
    call the file), then its backing location, then the magic numbers of any
    bytes already in memory.  Inline text with no file behind it counts as
    ``txt``; anything left over lands in :data:`UNKNOWN_FILE_TYPE`.
    """
    for key in ("filename", "origin_name", "media_path"):
        ext = _ext_from_name(str(media.get(key) or ""))
        if ext:
            return ext
    ext = _ext_from_url(str(media.get("media_url") or ""))
    if ext:
        return ext
    data = media.get("media_bytes")
    if data:
        sniffed = sniff_file_type(data)
        if sniffed:
            return sniffed
    if media.get("media_string") is not None:
        return "txt"
    return UNKNOWN_FILE_TYPE


def count_file_types(medias: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Tally :func:`media_file_type` over *medias*, most common first."""
    counter: Counter[str] = Counter()
    for media in medias:
        counter[media_file_type(media)] += 1
    return dict(counter.most_common())


def counts_are_uninformative(counts: Mapping[str, int]) -> bool:
    """True when *counts* tells the user nothing about a dataset's contents.

    Either empty, or every item sits in an unknown bucket — including the
    ``"(no extension)"`` bucket that entries stamped before
    :func:`media_file_type` existed use.
    """
    return not counts or all(key.startswith("(") for key in counts)
