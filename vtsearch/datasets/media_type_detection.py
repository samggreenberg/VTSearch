"""Sample a folder of files and guess which media type dominates.

Used by the import modal to pre-fill the "Output Media Type" dropdown and
auto-populate :class:`~vtsearch.datasets.importers.base.SourceSpec` rows
based on what's actually in the folder, instead of making the user pick
blindly.

The sampler is defensive about pathological folder shapes: a folder full
of empty sub-directories, or one whose root has been symlinked to a huge
tree, can otherwise make :func:`os.walk` spelunk for many seconds before
hitting the file-count limit.  Three independent bounds keep the call
fast for the UI hint:

* a per-call **file cap** (the caller's ``limit``, default 50);
* a per-call **directory cap** (``max_dirs``, default 500);
* a per-call **wall-clock budget** (``time_budget_seconds``, default
  ``0.75``).

Whichever fires first ends the walk; the response reflects whatever has
been counted so far.  Symlinks are **not** followed during detection —
the import itself still follows them, but a detection sample doesn't
need to walk through a symlinked tree to make a good guess.
"""

from __future__ import annotations

import os
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from vtsearch.media import get_by_extension


def detect_media_types_in_folder(  # noqa: C901
    folder: Path,
    recursive: bool = True,
    limit: int = 50,
    max_dirs: int = 500,
    time_budget_seconds: float = 0.75,
) -> dict:
    """Walk *folder* and tally up to *limit* files by media type.

    Args:
        folder: Directory to scan.  Must exist and be a directory.
        recursive: When ``True`` (default) descend into sub-directories.
            Symlinks are **not** followed regardless — see the module
            docstring for the rationale.  When ``False`` only files
            directly inside *folder* are sampled.
        limit: Maximum number of files to examine.
        max_dirs: Maximum number of directories to enter.  When the cap
            is reached the walk stops with whatever has been counted so
            far.  Bounds the worst case for sparse trees.
        time_budget_seconds: Soft wall-clock budget for the whole walk.
            Checked between directory transitions and between file
            counts; the walk stops as soon as the budget is exceeded.

    Returns:
        A dict with these keys:

        * ``sample_size`` (int) – how many files were actually examined.
        * ``counts_by_type`` (dict[str, int]) – ``type_id`` → count for
          every media type that matched at least one extension in the
          sample.  Files with extensions that no registered media type
          claims are counted under ``"unknown"``.
        * ``extensions`` (dict[str, int]) – lowercase extension (with
          the leading dot) → count, also limited to the sample.
        * ``dominant`` (str | None) – ``type_id`` of the most common
          recognised media type, or ``None`` when the sample contained
          no recognised files (all unknown or empty folder).
        * ``truncated`` (bool) – ``True`` when the walk stopped because
          ``max_dirs`` or ``time_budget_seconds`` fired (i.e. the
          sample may be a less complete view of the folder than usual).
    """
    if not folder.is_dir():
        return {
            "sample_size": 0,
            "counts_by_type": {},
            "extensions": {},
            "dominant": None,
            "truncated": False,
        }

    counts_by_type: Counter[str] = Counter()
    extensions: Counter[str] = Counter()
    examined = 0
    dirs_visited = 0
    truncated = False
    deadline = time.monotonic() + time_budget_seconds

    def _count_file(p: Path) -> bool:
        """Add *p* to the running tallies.  Returns ``True`` when the
        file cap or time budget have been reached (i.e. the caller
        should stop walking)."""
        nonlocal examined
        if p.name.startswith("."):
            return False
        ext = p.suffix.lower()
        extensions[ext] += 1
        mt = get_by_extension(ext) if ext else None
        counts_by_type[mt.type_id if mt is not None else "unknown"] += 1
        examined += 1
        if examined >= limit:
            return True
        if time.monotonic() >= deadline:
            return True
        return False

    if recursive:
        # ``followlinks`` stays at its default (``False``) so a folder
        # symlinked to a huge tree does not blow up the budget.
        walker = os.walk(folder)
        for _dirpath, _dirnames, filenames in walker:
            dirs_visited += 1
            stop = False
            for name in filenames:
                if _count_file(Path(_dirpath) / name):
                    stop = True
                    break
            if stop:
                # Distinguish "hit file cap exactly" (not truncated) from
                # "ran out of time mid-directory" (truncated).
                if examined < limit and time.monotonic() >= deadline:
                    truncated = True
                break
            if dirs_visited >= max_dirs:
                truncated = True
                break
            if time.monotonic() >= deadline:
                truncated = True
                break
    else:
        with os.scandir(folder) as it:
            for entry in it:
                if entry.is_file(follow_symlinks=False) and _count_file(Path(entry.path)):
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
        "truncated": truncated,
    }
