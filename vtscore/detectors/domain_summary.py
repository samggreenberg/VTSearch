"""The training-domain summary a detector carries, so its overlap check survives a handoff (#5567).

The training-domain overlap check (:mod:`vtsearch.domain_overlap`) compares
the active dataset with the vectors of the dataset a detector was trained on:
loaded, or read from its saved file through the pointer the detector records
(:data:`~vtscore.detectors.training_domain.TRAINING_DATASET_KEY`, #5501).  A
detector handed to another instance arrives without that dataset, so the check
had nothing to compare against.  The detector now records a summary of the
dataset as well, under :data:`DOMAIN_SUMMARY_KEY`, written with the pointer
wherever the pointer is written::

    {"format": 1, "dataset_id": ..., "dataset_name": ..., "media_type": ...,
     "embedder": ..., "dim": ..., "n_reference": <vectors summarised>,
     "anchors": [{"md5": ..., "origin": {...}, "origin_name": ..., "filename": ...}, ...],
     "calibration": [<sorted floats>]}

The **anchors** are items of the training dataset, named the way a labelset
names its media (by origin and MD5, never by vector), so a reader re-embeds
them the way a detector's labels are re-embedded
(:func:`~vtscore.detectors.labelset_embeddings.populate_label_embeddings`):
from the active dataset when it holds the same media, else from the origin's
file.  An anchor resolves wherever the detector's own labels do, since both
name media of the same dataset.  The **calibration** scores are scalars in the
embedder's geometry, stamped as the Evidence baseline's cuts are
(:mod:`~vtscore.detectors.evidence_baseline`): the embedder and its width, and
refused on any mismatch.  :mod:`vtscore.coverage.domain_summary` has the
statistics, and why there is no per-anchor radius.

A detector shared through an exported labelset, rather than as its file,
carries the record too (#5571).  The export writes it into the labelset's
``detector_meta`` under the same key
(:func:`~vtscore.detectors.input_spec.build_detector_meta`), and the import
stores it on the receiving detector (:func:`adopt_domain_summary`).  The
sender's pointer stays behind, but the import records one of its own beside
the summary when it has none, naming the summarised dataset as holding the
labels that arrived with it (#5572): the receiver's later votes elsewhere then
outweigh it once they outnumber those, as they would a training dataset handed
over with its file.  A summary with no pointer at all (stored before #5572) is read as
holding every label.  An exported labelset is otherwise embedder-free (#5131),
and this block is let through because it says which embedder drew it, a
reader in any other refuses it, and nothing trains on it.

Two process-scoped caches, nothing persisted but the record itself.  Building
a summary clusters up to 20 000 vectors and every draw on the training dataset
records one, so the last one built for each dataset is kept, keyed by its media
ids and embedder.  Reading one fetches and embeds each anchor the active
dataset does not hold, so the vectors resolved from files are kept per summary
and embedder, and a fetch that failed is not retried for
:data:`_RETRY_FAILED_SECONDS`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from vtscore.state.core import DatasetContext, DetectorContext

logger = logging.getLogger(__name__)

#: The detector JSON key the summary is recorded under.
DOMAIN_SUMMARY_KEY = "training_domain_summary"

#: The record's shape version; a reader refuses any other.
SUMMARY_FORMAT = 1

#: How long a failed anchor fetch is remembered before it is tried again.
_RETRY_FAILED_SECONDS = 300.0

#: Summaries whose resolved anchors are kept at once.  A run checks every
#: detector it scored, so this covers a dashboard's worth of detectors.
_MAX_RESOLVED = 16

# Guards both caches.  A leaf lock: nothing is fetched or embedded under it.
_lock = threading.Lock()
# Dataset id -> (fingerprint of the medias and embedder summarised, the summary).
_built: dict[str, tuple[tuple[Any, ...], dict[str, Any] | None]] = {}
# (summary digest, embedder) -> anchor index -> its vector, or the time its fetch failed.
_resolved: OrderedDict[tuple[str, str], dict[int, np.ndarray | float]] = OrderedDict()


class SummaryUnavailable(Exception):
    """A recorded summary cannot be read here; the message says why."""


def _anchor_entry(media: Mapping[str, Any]) -> dict[str, Any]:
    """How an anchor is named: a labelset element's identity fields, without a label."""
    entry: dict[str, Any] = {"md5": str(media.get("md5") or "")}
    origin = media.get("origin")
    if isinstance(origin, dict):
        entry["origin"] = origin
    origin_name = media.get("origin_name") or media.get("filename") or ""
    if origin_name:
        entry["origin_name"] = str(origin_name)
    if media.get("filename"):
        entry["filename"] = str(media["filename"])
    return entry


def build_domain_summary(
    medias: Mapping[int, dict[str, Any]],
    *,
    dataset_id: str,
    dataset_name: str,
    embedder: str,
    media_type: str,
) -> dict[str, Any] | None:
    """The :data:`DOMAIN_SUMMARY_KEY` record summarising *medias*, the training dataset's, or ``None``.

    Summarises every media's primary vector, in id order, as the overlap check
    stacks them (:func:`~vtscore.coverage.domain_summary.summarize_reference`).
    *embedder* is the dataset's primary embedder, the name the reader checks
    its own dataset's against.  ``None`` with no embedder name, no vectors, or
    too few to calibrate.
    """
    from vtscore.coverage.domain_shift import ReferenceTooSmall  # noqa: PLC0415
    from vtscore.coverage.domain_summary import summarize_reference  # noqa: PLC0415
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    if not embedder:
        return None
    named: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for _cid, media in sorted(medias.items()):
        emb = media_embedding(media)
        if emb is None:
            continue
        named.append(media)
        vectors.append(np.asarray(emb, dtype=np.float32))
    if not vectors:
        return None
    matrix = np.stack(vectors)
    try:
        anchor_rows, calibration = summarize_reference(matrix)
    except ReferenceTooSmall:
        return None
    return {
        "format": SUMMARY_FORMAT,
        "dataset_id": str(dataset_id),
        "dataset_name": str(dataset_name or ""),
        "media_type": str(media_type or ""),
        "embedder": str(embedder),
        "dim": int(matrix.shape[1]),
        "n_reference": int(matrix.shape[0]),
        "anchors": [_anchor_entry(named[int(row)]) for row in anchor_rows],
        "calibration": [float(score) for score in calibration],
    }


def domain_summary_for(dataset_id: str, medias: Mapping[int, dict[str, Any]]) -> dict[str, Any] | None:
    """:func:`build_domain_summary` for registered dataset *dataset_id*, whose media are *medias*.

    Its name, media type and embedder are read from the registry.  The last
    summary built for each dataset is reused while its media ids and embedder
    are unchanged, so every draw on the training dataset does not cluster it
    again.  ``None`` when there is nothing to summarise.  Never raises: the
    summary rides beside the writes it joins, which must not be lost to it.
    """
    from vtscore.datasets.registry import get_dataset  # noqa: PLC0415

    try:
        entry = get_dataset(dataset_id) or {}
        embedder = str(entry.get("embedder") or "")
        ids = np.fromiter(sorted(medias), dtype=np.int64, count=len(medias))
        fingerprint = (embedder, len(ids), hashlib.blake2b(ids.tobytes(), digest_size=16).hexdigest())
        with _lock:
            cached = _built.get(dataset_id)
        if cached is not None and cached[0] == fingerprint:
            summary = cached[1]
        else:
            summary = build_domain_summary(
                medias,
                dataset_id=dataset_id,
                dataset_name=str(entry.get("name") or ""),
                embedder=embedder,
                media_type=str(entry.get("media_type") or ""),
            )
            with _lock:
                _built[dataset_id] = (fingerprint, summary)
        if summary is not None and summary["dataset_name"] != str(entry.get("name") or ""):
            # A rename since it was built: the summary is the same, its label is not.
            summary = {**summary, "dataset_name": str(entry.get("name") or "")}
        return None if summary is None else dict(summary)
    except Exception:  # noqa: BLE001 - see the docstring
        logger.exception("Could not summarise dataset %s for its detectors", dataset_id)
        return None


def stored_domain_summary(detector_data: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The summary recorded on *detector_data*, or ``None`` when it has none or it is malformed.

    Malformed means any field a reader relies on is missing or of the wrong
    type, or the format is another version: a reader never has to
    second-guess what it gets back.
    """
    from vtscore.coverage.domain_shift import DOMAIN_SHIFT_MIN_CAL  # noqa: PLC0415

    entry = (detector_data or {}).get(DOMAIN_SUMMARY_KEY)
    if not isinstance(entry, Mapping) or entry.get("format") != SUMMARY_FORMAT:
        return None
    anchors = entry.get("anchors")
    calibration = entry.get("calibration")
    if not (isinstance(anchors, list) and anchors and all(isinstance(a, dict) for a in anchors)):
        return None
    if not isinstance(calibration, list) or len(calibration) < DOMAIN_SHIFT_MIN_CAL:
        return None
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in calibration):
        return None
    try:
        dim = int(entry.get("dim", 0))
        n_reference = int(entry.get("n_reference", 0))
    except (TypeError, ValueError):
        return None
    if not entry.get("dataset_id") or not entry.get("embedder") or dim <= 0:
        return None
    return {
        "format": SUMMARY_FORMAT,
        "dataset_id": str(entry["dataset_id"]),
        "dataset_name": str(entry.get("dataset_name") or ""),
        "media_type": str(entry.get("media_type") or ""),
        "embedder": str(entry["embedder"]),
        "dim": dim,
        "n_reference": n_reference,
        "anchors": [dict(a) for a in anchors],
        "calibration": [float(x) for x in calibration],
    }


def adopt_domain_summary(
    detector_data: dict[str, Any], detector_meta: Mapping[str, Any] | None, *, labels_held: int = 0
) -> bool:
    """Record the summary an imported labelset's *detector_meta* carries on *detector_data*; whether it did (#5571).

    How a detector shared through an exported labelset keeps its overlap
    check: the block an export writes into ``detector_meta`` under
    :data:`DOMAIN_SUMMARY_KEY` is stored on the receiving detector under the
    same key, with the fields :func:`stored_domain_summary` reads and nothing
    recomputed.  Only when the detector has no training dataset of its own (no
    summary, and no pointer to one), or when its own names the same dataset,
    which is a later draw of the same summary.  Otherwise the detector keeps
    its own: two summaries are never merged, and a pointer to another dataset
    says where this detector's labels were drawn, which a summary from
    elsewhere does not overrule.  A block that is missing, malformed or of
    another format version is ignored.

    *labels_held*, the number of Good / Bad labels the import brought
    (:func:`count_imported_labels`), is recorded beside the summary as the
    training-dataset pointer to the summarised dataset when the detector has
    none (#5572): those labels were cast on it, and the overlap check weighs
    it at that many and no more.  A pointer already there is left as it is:
    it is this detector's own record of how many of its labels the dataset
    held, and a labelset source it syncs with sends back every label it has.
    Without *labels_held* the summary is stored alone and read as holding
    every label.
    """
    from vtscore.detectors.training_domain import (  # noqa: PLC0415
        TRAINING_DATASET_KEY,
        recorded_training_dataset,
        training_dataset_record,
    )

    incoming = _incoming_summary(detector_meta)
    if incoming is None:
        return False
    own = stored_domain_summary(detector_data)
    pointer = recorded_training_dataset(detector_data)
    own_ids = {rec["dataset_id"] for rec in (own, pointer) if rec is not None}
    if own_ids - {incoming["dataset_id"]}:
        return False
    changed = False
    if own != incoming:
        detector_data[DOMAIN_SUMMARY_KEY] = incoming
        changed = True
    if labels_held > 0 and pointer is None:
        detector_data[TRAINING_DATASET_KEY] = training_dataset_record(
            incoming["dataset_id"], incoming["dataset_name"], int(labels_held)
        )
        changed = True
    return changed


def count_imported_labels(label_entries: Iterable[Mapping[str, Any]]) -> int:
    """How many of an import's *label_entries* are Good / Bad labels: :func:`adopt_domain_summary`'s *labels_held*."""
    return sum(1 for entry in label_entries if entry.get("label") in ("good", "bad"))


def adopt_domain_summary_on_file(
    path: Path,
    detector_meta: Mapping[str, Any] | None,
    *,
    det_ctx: DetectorContext | None = None,
    labels_held: int = 0,
) -> bool:
    """:func:`adopt_domain_summary` on the detector file at *path*; whether it wrote.

    Read, adopted and written under the detector-JSON write lock, so it cannot
    lose a label sync's write, and *det_ctx*'s cached labelset is re-pointed at
    the new file the way :func:`~vtscore.detectors.evidence_baseline.record_baseline`
    re-points it.  Never call this holding :data:`vtscore.state.core._state_lock`:
    the label sync takes that lock while it holds the write lock.
    """
    if _incoming_summary(detector_meta) is None:
        return False
    from vtscore.detectors.label_sync import label_sync_write_lock  # noqa: PLC0415
    from vtscore.detectors.store import _read_detector, _write_detector  # noqa: PLC0415

    with label_sync_write_lock:
        data = _read_detector(path)
        if data is None or not adopt_domain_summary(data, detector_meta, labels_held=labels_held):
            return False
        _write_detector(path, data)
    if det_ctx is not None:
        from vtscore.detectors.dataset_sync import _repoint_labelset_cache  # noqa: PLC0415

        _repoint_labelset_cache(det_ctx, path)
    return True


def _incoming_summary(detector_meta: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The well-formed summary *detector_meta* carries, or ``None``."""
    if not isinstance(detector_meta, Mapping):
        return None
    return stored_domain_summary({DOMAIN_SUMMARY_KEY: detector_meta.get(DOMAIN_SUMMARY_KEY)})


def _summary_digest(summary: Mapping[str, Any]) -> str:
    """A key for *summary*'s anchors: the same anchors resolve to the same vectors."""
    text = json.dumps(summary["anchors"], sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def _embed_anchor(anchor: Mapping[str, Any], *, media_type: str, embedder: str) -> np.ndarray | None:
    """*anchor*'s vector, fetched from its origin's file and embedded in *embedder*; ``None`` on failure."""
    from vtscore.datasets.labelset import LabeledElement  # noqa: PLC0415
    from vtscore.detectors.labelset_embeddings import _embed_one  # noqa: PLC0415

    elem = LabeledElement(
        md5=str(anchor.get("md5") or ""),
        label="",
        origin=anchor.get("origin") if isinstance(anchor.get("origin"), dict) else None,
        origin_name=str(anchor.get("origin_name") or ""),
        filename=str(anchor.get("filename") or ""),
    )
    if elem.origin is None:
        return None
    try:
        resolved = _embed_one(elem, media_type=media_type, embedder_name=embedder)
    except Exception:  # noqa: BLE001 - an unreadable origin is a miss, reported by the caller
        logger.warning("Could not embed reference item %r", elem.origin_name or elem.filename, exc_info=True)
        return None
    return None if resolved is None else np.asarray(resolved.vec, dtype=np.float32)


def anchor_vectors(summary: Mapping[str, Any], ctx: DatasetContext, *, embedder: str) -> np.ndarray:
    """The vectors of *summary*'s anchors, in *embedder*, in the summary's order.

    Each anchor is read from *ctx*, the active dataset, when it holds the same
    media (matched by origin or MD5, as labels are), else fetched from its
    origin's file and embedded.  The caller has checked that *embedder* is the
    one the summary was drawn in and *ctx*'s primary.  Raises
    :class:`SummaryUnavailable` at the first anchor that resolves neither way:
    with one missing, the items near it would read as atypical, so the check
    refuses rather than report a shift that is not there.
    """
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415
    from vtscore.state.media_lookup import cached_media_lookups, resolve_media_ids  # noqa: PLC0415

    anchors = summary["anchors"]
    origin_lookup, md5_lookup, _names = cached_media_lookups(ctx)
    key = (_summary_digest(summary), embedder)
    with _lock:
        known = dict(_resolved.get(key, {}))
    rows: list[np.ndarray] = []
    try:
        for index, anchor in enumerate(anchors):
            vec = None
            for cid in resolve_media_ids(anchor, origin_lookup, md5_lookup):
                media = ctx.medias.get(cid)
                emb = None if media is None else media_embedding(media)
                if emb is not None:
                    vec = np.asarray(emb, dtype=np.float32)
                    break
            if vec is None:
                seen = known.get(index)
                if isinstance(seen, np.ndarray):
                    vec = seen
                elif seen is None or time.monotonic() - seen >= _RETRY_FAILED_SECONDS:
                    vec = _embed_anchor(anchor, media_type=str(summary.get("media_type") or ""), embedder=embedder)
                    known[index] = vec if vec is not None else time.monotonic()
            if vec is None:
                name = anchor.get("origin_name") or anchor.get("filename") or anchor.get("md5") or f"#{index + 1}"
                raise SummaryUnavailable(
                    f"Reference item {name!r} of the training dataset's summary could not be fetched "
                    f"({index} of {len(anchors)} resolved before it)"
                )
            rows.append(vec)
    finally:
        with _lock:
            _resolved[key] = known
            _resolved.move_to_end(key)
            while len(_resolved) > _MAX_RESOLVED:
                _resolved.popitem(last=False)
    return np.stack(rows)


def reset_domain_summary_caches_for_tests() -> None:
    """Forget every built summary and resolved anchor (for conftest), so one test's cannot serve the next."""
    with _lock:
        _built.clear()
        _resolved.clear()
