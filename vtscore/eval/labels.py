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

from typing import Any, Optional


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


def region_box_for_category(
    media: dict[str, Any], category: str
) -> Optional[tuple[float, float, float, float]]:
    """Return the ground-truth region box for *category* on *media*, or ``None``.

    Datasets like Visual Genome stamp store-only ground-truth boxes on each
    media as ``media["regions"] = [{"box": [x0, y0, x1, y1], "label": cat}, ...]``
    (normalised ``[0, 1]`` coordinates - see
    ``docs/plans/visual-genome-dataset.md``).  The eval harness uses these to
    simulate a user who, when voting Good, also drags a region around the
    object instead of voting on the whole image.

    When more than one annotated region carries *category* (e.g. an image with
    two apples), we return the **minimal axis-aligned box that covers them
    all** (``min`` of the corners, ``max`` of the far corners).  Covering all
    of them keeps every annotated instance inside the voted region; picking one
    box arbitrarily would discard real signal and depend on annotation order.

    Returns ``None`` when *media* has no ``regions`` (single-label datasets, or
    a positive image with no box annotation for this category), so callers fall
    back to the whole-image embedding - exactly the behaviour of an image-level
    Good vote.
    """
    regions = media.get("regions")
    if not regions:
        return None
    boxes = [r["box"] for r in regions if r.get("label") == category]
    if not boxes:
        return None
    x0 = min(float(b[0]) for b in boxes)
    y0 = min(float(b[1]) for b in boxes)
    x1 = max(float(b[2]) for b in boxes)
    y1 = max(float(b[3]) for b in boxes)
    return (x0, y0, x1, y1)
