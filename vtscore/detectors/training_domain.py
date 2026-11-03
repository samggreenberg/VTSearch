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
much as for one trained in this session.

Only while that dataset is loaded, though, so the detector also records it
(#5501): :data:`TRAINING_DATASET_KEY` names the dataset and how many of the
labels it held, written with the Evidence baseline whenever one is drawn there
(:mod:`vtscore.detectors.evidence_baseline`).  A pointer and a count, never a
vector, so the no-persisted-vectors rule is untouched; the check that reads it
fetches the dataset's own vectors from its saved file.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vtscore.datasets.labelset import LabelSet
    from vtscore.state.core import DatasetContext

#: The detector JSON key its training dataset is recorded under (#5501).
TRAINING_DATASET_KEY = "training_dataset"


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


def labelled_media_ids(labelset: LabelSet, ctx: DatasetContext) -> set[int]:
    """The ids of *ctx*'s media that *labelset*'s Good / Bad elements name, matched as :func:`count_labels_held` does."""
    from vtscore.state.media_lookup import cached_media_lookups, resolve_media_ids

    origin_lookup, md5_lookup, _names = cached_media_lookups(ctx)
    if not origin_lookup and not md5_lookup:
        return set()
    return {
        int(cid)
        for elem in labelset.elements
        if elem.label in ("good", "bad")
        for cid in resolve_media_ids(elem.to_dict(), origin_lookup, md5_lookup)
    }


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


def training_dataset_record(dataset_id: str, dataset_name: str, labels_held: int) -> dict[str, Any]:
    """The :data:`TRAINING_DATASET_KEY` entry naming *dataset_id* as the dataset holding *labels_held* of the labels."""
    return {"dataset_id": str(dataset_id), "dataset_name": str(dataset_name or ""), "labels_held": int(labels_held)}


def recorded_training_dataset(detector_data: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The training dataset recorded on *detector_data* (:func:`training_dataset_record`), or ``None``.

    ``None`` too for an entry that names no dataset or holds no label, so a
    reader never has to second-guess what it gets back.
    """
    entry = (detector_data or {}).get(TRAINING_DATASET_KEY)
    if not isinstance(entry, Mapping) or not entry.get("dataset_id"):
        return None
    try:
        held = int(entry.get("labels_held", 0))
    except (TypeError, ValueError):
        return None
    if held <= 0:
        return None
    return training_dataset_record(str(entry["dataset_id"]), str(entry.get("dataset_name") or ""), held)
