"""Dataset splitting utilities for evaluation."""

import hashlib
import logging
import random
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


def _category_seed(seed: int, category: str) -> int:
    """Derive a per-category seed so each category is shuffled independently."""
    h = hashlib.sha256(f"{seed}:{category}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def _split_one_category(ids: list[int], test_fraction: float, seed: int, category: str) -> tuple[list[int], list[int]]:
    """Shuffle one category's ids and return (test_ids, simulate_ids).

    Single-item categories cannot be split; the lone item is returned as
    simulate and ``test_ids`` is empty (callers are expected to flag this
    so the user is not left wondering why a category is absent from test).
    """
    ids = sorted(ids)  # deterministic order before shuffle
    rng = random.Random(_category_seed(seed, category))
    rng.shuffle(ids)

    n_test = round(len(ids) * test_fraction)
    # Ensure at least 1 in each split when the category is large enough
    if n_test == 0 and len(ids) >= 2:
        n_test = 1
    if n_test == len(ids) and len(ids) >= 2:
        n_test = len(ids) - 1

    return ids[:n_test], ids[n_test:]


def split_dataset(
    medias: dict[int, dict[str, Any]],
    test_fraction: float,
    seed: int,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    """Split a medias dict into simulation and test sets, stratified by category.

    Each category is split independently using the same fraction, so if
    ``test_fraction=0.2`` then roughly 20% of each category ends up in the
    test set and 80% in the simulation set.  Each category uses its own
    deterministic RNG derived from ``seed`` and the category name, so
    adding or removing categories does not change the split of other
    categories.

    Clip IDs are preserved (not renumbered) in both output dicts.

    Single-item categories cannot be both trained on and tested on; the
    lone item is placed in the simulation set and the category contributes
    nothing to the test set.  When this happens, a single summary
    ``logger.warning`` lists the affected categories so callers do not
    silently end up with categories that are absent from the test set.

    Args:
        medias: Mapping of media ID to media data dict.  Every media must have a
            ``"category"`` key.
        test_fraction: Fraction of each category to allocate to the test set.
            Must be in ``(0, 1)``.
        seed: Integer seed used to derive per-category random states for
            reproducible shuffling.

    Returns:
        A 2-tuple ``(simulate_clips, test_clips)`` where each element is a
        dict with the same structure as ``medias``.

    Raises:
        ValueError: If ``test_fraction`` is not in ``(0, 1)`` or if ``medias``
            is empty.
    """
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    if not medias:
        raise ValueError("medias dict is empty")

    # Group media IDs by category
    by_category: dict[str, list[int]] = defaultdict(list)
    for media_id, media in medias.items():
        by_category[media["category"]].append(media_id)

    simulate_clips: dict[int, dict[str, Any]] = {}
    test_clips: dict[int, dict[str, Any]] = {}
    single_item_categories: list[str] = []

    for category in sorted(by_category):
        ids = by_category[category]
        if len(ids) == 1:
            single_item_categories.append(category)

        test_ids, simulate_ids = _split_one_category(ids, test_fraction, seed, category)

        for cid in test_ids:
            test_clips[cid] = medias[cid]
        for cid in simulate_ids:
            simulate_clips[cid] = medias[cid]

    if single_item_categories:
        noun = "category" if len(single_item_categories) == 1 else "categories"
        logger.warning(
            "split_dataset: %d %s had only 1 item and contribute nothing to the test set (item placed in simulate): %s",
            len(single_item_categories),
            noun,
            ", ".join(single_item_categories),
        )

    return simulate_clips, test_clips
