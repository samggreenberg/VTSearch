"""Sample a folder of files and guess which media type dominates.

Used by the import modal to pre-fill the "Output Media Type" dropdown and
auto-populate :class:`~vtsearch.datasets.importers.base.SourceSpec` rows
based on what's actually in the folder, instead of making the user pick
blindly.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Optional

from vtsearch.media import get_by_extension


def detect_media_types_in_folder(
    folder: Path,
    recursive: bool = True,
    limit: int = 50,
) -> dict:
    """Walk *folder* and tally up to *limit* files by media type.

    Args:
        folder: Directory to scan.  Must exist and be a directory.
        recursive: When ``True`` (default) descend into sub-directories,
            following symlinks.  When ``False`` only files directly inside
            *folder* are sampled.
        limit: Maximum number of files to examine.  Sampling stops as soon
            as this many files have been counted, so the call stays fast
            even on multi-million-file trees.

    Returns:
        A dict with these keys:

        * ``sample_size`` (int) – how many files were actually examined.
        * ``counts_by_type`` (dict[str, int]) – ``type_id`` → count for
          every media type that matched at least one extension in the
          sample.  Files with extensions that no registered media type
          claims are counted under ``"unknown"``.
        * ``extensions`` (dict[str, int]) – lowercase extension (with the
          leading dot) → count, also limited to the sample.  Useful for
          debugging "what is this folder full of?".
        * ``dominant`` (str | None) – ``type_id`` of the most common
          recognised media type, or ``None`` when the sample contained
          no recognised files (all unknown or empty folder).
    """
    if not folder.is_dir():
        return {
            "sample_size": 0,
            "counts_by_type": {},
            "extensions": {},
            "dominant": None,
        }

    counts_by_type: Counter[str] = Counter()
    extensions: Counter[str] = Counter()
    examined = 0

    def _count_file(p: Path) -> bool:
        nonlocal examined
        if p.name.startswith("."):
            return False
        ext = p.suffix.lower()
        extensions[ext] += 1
        mt = get_by_extension(ext) if ext else None
        counts_by_type[mt.type_id if mt is not None else "unknown"] += 1
        examined += 1
        return examined >= limit

    if recursive:
        for dirpath, _dirnames, filenames in os.walk(folder, followlinks=True):
            for name in filenames:
                if _count_file(Path(dirpath) / name):
                    break
            if examined >= limit:
                break
    else:
        with os.scandir(folder) as it:
            for entry in it:
                if entry.is_file(follow_symlinks=True) and _count_file(Path(entry.path)):
                    break

    dominant: Optional[str] = None
    for type_id, _count in counts_by_type.most_common():
        if type_id != "unknown":
            dominant = type_id
            break

    return {
        "sample_size": examined,
        "counts_by_type": dict(counts_by_type),
        "extensions": dict(extensions),
        "dominant": dominant,
    }
