"""Which loaded dataset a detector's labels were cast on (#4961).

The training-domain overlap check compares the active dataset against the
coverage atlas of the dataset a detector was trained on.  Nothing records that
dataset.  ``DetectorContext.votes_dataset_id`` looks like it should, but it
names the dataset the in-memory vote ids are keyed in: the detector load stamps
it with whatever dataset is active, and the rehydrate in
:mod:`vtscore.detectors.dataset_sync` restamps it on every dataset switch.

The labelset carries the answer instead.  Each Good / Bad element names its
media by origin and MD5, so the loaded dataset holding most of those media is
the one the votes were cast on.  That holds for a detector loaded from disk as
much as for one trained in this session, and it persists nothing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vtscore.datasets.labelset import LabelSet
    from vtscore.state.core import DatasetContext


def count_labels_held(labelset: LabelSet, ctx: DatasetContext) -> int:
    """How many of *labelset*'s Good / Bad elements name a media in *ctx*.

    Matched by origin + origin name or by MD5, never by file name alone: two
    datasets sharing a basename say nothing about where a label was cast.
    """
    from vtscore.state.media_lookup import cached_media_lookups, resolve_media_ids

    origin_lookup, md5_lookup, _names = cached_media_lookups(ctx)
    if not origin_lookup and not md5_lookup:
        return 0
    return sum(
        1
        for elem in labelset.elements
        if elem.label in ("good", "bad") and resolve_media_ids(elem.to_dict(), origin_lookup, md5_lookup)
    )


def labels_held_by_dataset(labelset: LabelSet, contexts: Iterable[DatasetContext]) -> dict[str, int]:
    """:func:`count_labels_held` for each of *contexts*, keyed by dataset id."""
    return {ctx.dataset_id: count_labels_held(labelset, ctx) for ctx in contexts}


def infer_training_dataset(counts: Mapping[str, int], *, prefer: str = "") -> str:
    """The dataset in *counts* holding the most labels, ``""`` when none holds any.

    *prefer* wins a tie, so a caller passing the active dataset never hears
    that the detector was trained elsewhere when the active dataset holds as
    many of its labels as any other.  Other ties go to the smallest id, so the
    answer does not depend on load order.
    """
    best = max(counts.values(), default=0)
    if best <= 0:
        return ""
    if counts.get(prefer, 0) == best:
        return prefer
    return min(dataset_id for dataset_id, n in counts.items() if n == best)
