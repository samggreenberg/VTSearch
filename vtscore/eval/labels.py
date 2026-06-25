"""Ground-truth membership test shared by the eval harness.

Single source of truth for "does this media belong to *category*?", so the
text-sort, learned-sort, and voting-iteration evaluators agree.

Two dataset shapes are supported:

- **Multi-label** (e.g. Visual Genome): the media carries a ``"categories"``
  list of the categories it positively belongs to.  Membership is set
  membership, and — under the closed-world assumption — any category *not* in
  that list is a negative for the image.
- **Single-label** (every other demo dataset): the media carries one
  ``"category"`` string and membership is an exact string compare.

A media is multi-label iff it has a ``"categories"`` key; otherwise the legacy
single-label path is used.  Existing datasets have no ``"categories"`` key, so
their behavior is unchanged.
"""

from __future__ import annotations

from typing import Any


def media_is_positive(media: dict[str, Any], category: str) -> bool:
    """Return ``True`` if *media* is a positive example of *category*.

    For multi-label media (those with a ``"categories"`` list) this is set
    membership; for single-label media it is an exact ``"category"`` match.
    Under the closed-world assumption used by the eval harness, "not positive"
    is taken to mean "negative", so callers test negativity as
    ``not media_is_positive(...)``.
    """
    cats = media.get("categories")
    if cats is not None:
        return category in cats
    return media.get("category") == category
