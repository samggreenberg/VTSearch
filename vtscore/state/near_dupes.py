"""Deprecated alias for :mod:`vtscore.media.near_dupes`.

Near-duplicate detection is a pure algorithm — perceptual hashing (image pHash,
text SimHash) plus connected components over the closeness graph — and never
touched :class:`~vtscore.state.core.DatasetContext` or ``_state_lock``.  It has
moved to :mod:`vtscore.media.near_dupes`, beside the media types whose bytes it
hashes.

This module re-exports the new location so existing imports keep working, and
warns on import.  Import from :mod:`vtscore.media.near_dupes` instead (or keep
using the unchanged ``vtscore.state`` / ``vtsearch.state`` re-exports of
``collapse_near_duplicates``, ``phash_image`` and ``simhash_text``).
"""

from __future__ import annotations

import warnings
from typing import Any

from vtscore.media import near_dupes as _near_dupes
from vtscore.media.near_dupes import (  # noqa: F401
    collapse_near_duplicates,
    phash_image,
    simhash_text,
)

warnings.warn(
    "vtscore.state.near_dupes has moved to vtscore.media.near_dupes; "
    "import from there instead. This alias will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "collapse_near_duplicates",
    "phash_image",
    "simhash_text",
]


def __getattr__(name: str) -> Any:
    """Forward anything not re-exported above (module privates, new symbols)."""
    return getattr(_near_dupes, name)
