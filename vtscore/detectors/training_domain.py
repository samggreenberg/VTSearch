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
labels it held, written with the Evidence baseline whenever one is drawn there,
and on its own by a learned sort there that draws none (#5514;
:mod:`vtscore.detectors.evidence_baseline`).  A pointer and a count, never a
vector, so the no-persisted-vectors rule is untouched; the check that reads it
fetches the dataset's own vectors from its saved file.

A label does not say where it was cast when an import put its media there
(#5572).  Importing labels with a dataset active copies the media they name
into that dataset, so it holds every one of them: on its face, the dataset the
detector was trained on.  The detector records which media each import copied
where, under :data:`INGESTED_LABELS_KEY` (:func:`record_ingested_labels`), and
:func:`count_labels_held` does not count a label on one of them.  Named the
way a labelset names its media, so again no vector; a label the user casts on
the dataset's own media still counts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from vtscore.datasets.labelset import LabelSet
    from vtscore.state.core import DatasetContext, DetectorContext

#: The detector JSON key its training dataset is recorded under (#5501).
TRAINING_DATASET_KEY = "training_dataset"

#: The detector JSON key naming the media its label imports copied into each dataset (#5572)::
#:
#:     {"<dataset id>": [{"md5": ..., "origin": {...}, "origin_name": ..., "filename": ...}, ...]}
INGESTED_LABELS_KEY = "labels_ingested"


def count_labels_held(
    labelset: LabelSet, ctx: DatasetContext, *, ingested: Sequence[Mapping[str, Any]] | None = None
) -> int:
    """How many of *labelset*'s Good / Bad elements name a media in *ctx*.

    Matched by origin + origin name or by MD5, never by file name alone: two
    datasets sharing a basename say nothing about where a label was cast.
    *ingested*, the entries a label import copied into *ctx*
    (:func:`recorded_ingests`), names media that are not *ctx*'s own: a label
    naming only those is not counted (#5572).
    """
    from vtscore.state.media_lookup import cached_media_lookups, resolve_media_ids

    origin_lookup, md5_lookup, _names = cached_media_lookups(ctx)
    if not origin_lookup and not md5_lookup:
        return 0
    copied = {cid for entry in ingested or () for cid in resolve_media_ids(dict(entry), origin_lookup, md5_lookup)}
    return sum(
        1
        for elem in labelset.elements
        if elem.label in ("good", "bad")
        and any(cid not in copied for cid in resolve_media_ids(elem.to_dict(), origin_lookup, md5_lookup))
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


def labels_held_by_dataset(
    labelset: LabelSet,
    contexts: Iterable[DatasetContext],
    *,
    ingested: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, int]:
    """:func:`count_labels_held` for each of *contexts*, keyed by dataset id.

    *ingested* is :func:`recorded_ingests`'s answer: each dataset's count
    leaves out the labels on media an import copied into it (#5572).
    """
    ingested = ingested or {}
    return {ctx.dataset_id: count_labels_held(labelset, ctx, ingested=ingested.get(ctx.dataset_id)) for ctx in contexts}


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


def ingested_label_entry(entry: Mapping[str, Any]) -> dict[str, Any] | None:
    """How :data:`INGESTED_LABELS_KEY` names the media of label entry *entry*: its identity, without the label.

    ``None`` when *entry* names its media neither by origin nor by MD5, since
    :func:`count_labels_held` could not match it to anything.
    """
    named: dict[str, Any] = {}
    if entry.get("md5"):
        named["md5"] = str(entry["md5"])
    origin = entry.get("origin")
    if isinstance(origin, dict) and entry.get("origin_name"):
        named["origin"] = origin
        named["origin_name"] = str(entry["origin_name"])
    if not named:
        return None
    if entry.get("filename"):
        named["filename"] = str(entry["filename"])
    return named


def record_ingested_labels(
    detector_data: dict[str, Any], dataset_id: str, entries: Iterable[Mapping[str, Any]]
) -> bool:
    """Record on *detector_data* that a label import copied the media of *entries* into *dataset_id*; whether it changed.

    Added to what earlier imports recorded there, each media once, so a
    second import of the same labels changes nothing.  Nothing is recorded
    without a dataset id or for entries naming no media.
    """
    if not dataset_id:
        return False
    recorded = recorded_ingests(detector_data)
    held = list(recorded.get(dataset_id, []))
    added = False
    for entry in entries:
        named = ingested_label_entry(entry)
        if named is not None and named not in held:
            held.append(named)
            added = True
    if not added:
        return False
    detector_data[INGESTED_LABELS_KEY] = {**recorded, str(dataset_id): held}
    return True


def recorded_ingests(detector_data: Mapping[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    """What :func:`record_ingested_labels` recorded on *detector_data*: each dataset's copied-in entries.

    Empty without a record.  Entries that are not dicts naming a media, and
    datasets with none left, are dropped, so a reader never has to
    second-guess what it gets back.
    """
    raw = (detector_data or {}).get(INGESTED_LABELS_KEY)
    if not isinstance(raw, Mapping):
        return {}
    recorded: dict[str, list[dict[str, Any]]] = {}
    for dataset_id, entries in raw.items():
        if not dataset_id or not isinstance(entries, list):
            continue
        named = [n for e in entries if isinstance(e, Mapping) and (n := ingested_label_entry(e)) is not None]
        if named:
            recorded[str(dataset_id)] = named
    return recorded


def record_ingested_labels_on_file(
    path: Path,
    dataset_id: str,
    entries: Iterable[Mapping[str, Any]],
    *,
    det_ctx: DetectorContext | None = None,
) -> bool:
    """:func:`record_ingested_labels` on the detector file at *path*; whether it wrote.

    Read, recorded and written under the detector-JSON write lock, so it
    cannot lose a label sync's write, and *det_ctx*'s cached labelset is
    re-pointed at the new file, as
    :func:`~vtscore.detectors.domain_summary.adopt_domain_summary_on_file`
    does.  Never call this holding :data:`vtscore.state.core._state_lock`:
    the label sync takes that lock while it holds the write lock.
    """
    from vtscore.detectors.label_sync import label_sync_write_lock
    from vtscore.detectors.store import _read_detector, _write_detector

    entries = list(entries)
    if not dataset_id or not entries:
        return False
    with label_sync_write_lock:
        data = _read_detector(path)
        if data is None or not record_ingested_labels(data, dataset_id, entries):
            return False
        _write_detector(path, data)
    if det_ctx is not None:
        from vtscore.detectors.dataset_sync import _repoint_labelset_cache

        _repoint_labelset_cache(det_ctx, path)
    return True
