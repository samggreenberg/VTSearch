"""The training-domain overlap check: how much of a dataset looks unlike a reference dataset.

Three callers ask it.  ``GET /api/datasets/registry/<id>/domain-shift`` takes
the reference from the caller (the Tune result's **Compare against** picker).
``GET /api/find/training-domain`` works the reference out itself, from the
dataset holding most of the active detector's labelled examples
(:mod:`vtscore.detectors.training_domain`), so the SPA can warn the moment a
detector meets a dataset unlike the one it was trained on, before the user
runs anything (#4961).  And every Find and AutoFind run asks it once per
detector (:func:`detector_training_domain`), so the warning reaches the Find
Results dialog, the run's notice and its exported results (#5501).  All three
refuse on the same conditions, kept here.

The training dataset is the loaded one holding most of the labels, or the one
recorded on the detector (:data:`~vtscore.detectors.training_domain.TRAINING_DATASET_KEY`)
when that holds more and is not loaded: its vectors are then read from its
saved file (:func:`~vtscore.embedding.matrix.stored_primary_vectors`), so a
run after a restart, or an unattended AutoFind on a fresh import, still has
something to compare against.

Where the training dataset is not here at all (a detector handed over from
another instance), the check reads the summary of it the detector records
beside the pointer (#5567, :mod:`vtscore.detectors.domain_summary`), or that
came with its labels in an exported labelset (#5571): its anchor items,
re-embedded here, and calibration scores.  The report has the same
shape and the same null; it has less power, since 128 anchors stand in for
the dataset's vectors (:mod:`vtscore.coverage.domain_summary`).

A dataset holds a label as evidence only when the label's media is its own:
media a label import copied into it are left out of its count (#5572,
:data:`~vtscore.detectors.training_domain.INGESTED_LABELS_KEY`), or the
dataset an import landed in would read as the one the detector was trained on.

One rule picks the training dataset for every reader
(:func:`training_reference`): this check, and the Evidence chip's baseline,
which is drawn only there and records the pointer and summary as it is
(``vtsearch.routes.detectors.scoring._evidence_baseline_home``).  Were the two
to disagree, a draw on a dataset holding a few of the labels would move the
pointer there and replace the summary, and the check would go quiet (#5572).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class DomainShiftUnavailable(Exception):
    """The overlap check cannot be run against this reference; the message says why."""


@dataclass
class OverlapCache:
    """What one run's checks share: each dataset's vectors, and each pair's answer.

    A Find or AutoFind checks every detector it ran, and detectors trained on
    the same dataset compare the same two vector sets, so the vectors are
    stacked once (an unloaded reference is read from disk once) and the report
    is computed once.  Lives for one run; nothing is persisted.
    """

    vectors: dict[str, Any] = field(default_factory=dict)
    answers: dict[tuple[str, ...], dict[str, Any] | str] = field(default_factory=dict)


def domain_shift_against(
    reference_id: str, active_ctx: Any, *, allow_unloaded: bool = False, cache: OverlapCache | None = None
) -> dict[str, Any]:
    """Report domain shift of *active_ctx*'s vectors against *reference_id*'s own vectors.

    :func:`vtscore.coverage.domain_shift_report`: a split-conformal kNN over the
    reference dataset's whole-image vectors, so a p-value is uniform for an
    item drawn from the reference, on every embedder, patch spaces included
    (#3348).  Raises :class:`DomainShiftUnavailable` when the reference is not
    loaded (unless *allow_unloaded*: then its vectors are read from its saved
    file), is the active dataset itself, uses another embedder, has too few
    embeddings to calibrate, or the active dataset has no embeddings.  The
    caller has already checked that *reference_id* is registered and readable.
    *cache* shares each dataset's vectors between the checks of one run
    (:func:`cached_domain_shift` shares the answer too).

    Until #3348 this ranked against the reference's coverage atlas, needed the
    atlas built, and refused patch embedders, where it fired on 60-80 % of the
    reference's own held-out data.
    """
    from vtscore.coverage.domain_shift import ReferenceTooSmall, domain_shift_report
    from vtscore.datasets.registry import get_dataset
    from vtscore.state.core import get_context

    ref_ctx = get_context(reference_id)
    if ref_ctx is None and not allow_unloaded:
        raise DomainShiftUnavailable("Load the reference dataset before running a domain-shift check")
    if active_ctx.dataset_id == reference_id:
        raise DomainShiftUnavailable("Active dataset and reference dataset are the same")
    # Typicality p-values are meaningless across embedding spaces: refuse
    # instead of returning confident nonsense.
    ref_embedder = (get_dataset(reference_id) or {}).get("embedder", "")
    active_embedder = (get_dataset(active_ctx.dataset_id) or {}).get("embedder", "")
    if ref_embedder != active_embedder:
        raise DomainShiftUnavailable(
            f"Embedder mismatch: reference uses {ref_embedder or 'unknown'!r}, "
            f"active dataset uses {active_embedder or 'unknown'!r}"
        )

    vectors = _cached_vectors(cache, active_ctx.dataset_id, lambda: _media_vectors(active_ctx))
    if vectors is None:
        raise DomainShiftUnavailable("Active dataset has no embeddings to compare")
    if ref_ctx is not None:
        reference = _cached_vectors(cache, reference_id, lambda: _media_vectors(ref_ctx))
    else:
        reference = _cached_vectors(cache, reference_id, lambda: _stored_vectors(reference_id))
    if reference is None:
        raise DomainShiftUnavailable("Reference dataset has no embeddings to compare against")
    try:
        report = domain_shift_report(reference, vectors)
    except ReferenceTooSmall as e:
        raise DomainShiftUnavailable(f"Reference dataset is too small for a domain-shift check: {e}") from e
    return {"reference_dataset_id": reference_id, **report}


def cached_domain_shift(
    reference_id: str, active_ctx: Any, *, allow_unloaded: bool = False, cache: OverlapCache | None = None
) -> dict[str, Any]:
    """:func:`domain_shift_against`, answered once per pair in a run that shares *cache*.

    Its refusal is remembered as well as its report, so a second detector
    trained on the same unreadable dataset does not try it again.
    """
    return _cached_answer(
        cache,
        (reference_id, active_ctx.dataset_id),
        lambda: domain_shift_against(reference_id, active_ctx, allow_unloaded=allow_unloaded, cache=cache),
    )


def domain_shift_from_summary(
    summary: dict[str, Any], active_ctx: Any, *, cache: OverlapCache | None = None
) -> dict[str, Any]:
    """Report domain shift of *active_ctx*'s vectors against a detector's training-domain summary (#5567).

    :func:`vtscore.coverage.summary_shift_report` against *summary*
    (:func:`~vtscore.detectors.domain_summary.stored_domain_summary`): its
    anchors, read from the active dataset or fetched from their origins and
    embedded (:func:`~vtscore.detectors.domain_summary.anchor_vectors`), and
    its calibration scores.  The same dict as :func:`domain_shift_against`,
    naming the summarised dataset as the reference.  Raises
    :class:`DomainShiftUnavailable` when the summary is of the active dataset
    itself, was drawn in another embedder, media type or vector width, an
    anchor cannot be fetched, or the active dataset has no embeddings.
    """
    from vtscore.coverage.domain_summary import summary_shift_report
    from vtscore.datasets.registry import get_dataset
    from vtscore.detectors.domain_summary import SummaryUnavailable, anchor_vectors

    if summary["dataset_id"] == active_ctx.dataset_id:
        raise DomainShiftUnavailable("Active dataset and reference dataset are the same")
    active = get_dataset(active_ctx.dataset_id) or {}
    active_embedder = active.get("embedder", "") or ""
    # The calibration scores live in the summary's embedder: refuse instead of
    # returning confident nonsense, as the full check does.
    if summary["embedder"] != active_embedder:
        raise DomainShiftUnavailable(
            f"Embedder mismatch: the training dataset's summary was drawn in {summary['embedder']!r}, "
            f"active dataset uses {active_embedder or 'unknown'!r}"
        )
    if summary["media_type"] and active.get("media_type") and summary["media_type"] != active["media_type"]:
        raise DomainShiftUnavailable(
            f"Media type mismatch: the training dataset holds {summary['media_type']}, "
            f"the active dataset {active['media_type']}"
        )
    vectors = _cached_vectors(cache, active_ctx.dataset_id, lambda: _media_vectors(active_ctx))
    if vectors is None:
        raise DomainShiftUnavailable("Active dataset has no embeddings to compare")
    if vectors.shape[1] != summary["dim"]:
        raise DomainShiftUnavailable(
            f"Vector width mismatch: the training dataset's summary is {summary['dim']}-wide, "
            f"the active dataset's vectors {vectors.shape[1]}-wide"
        )
    try:
        anchors = anchor_vectors(summary, active_ctx, embedder=active_embedder)
        report = summary_shift_report(anchors, summary["calibration"], vectors)
    except SummaryUnavailable as e:
        raise DomainShiftUnavailable(str(e)) from e
    except ValueError as e:  # an anchor re-embedded at another width, or too few scores to calibrate
        raise DomainShiftUnavailable(f"The training dataset's summary cannot be read here: {e}") from e
    return {"reference_dataset_id": summary["dataset_id"], **report}


def cached_summary_shift(
    summary: dict[str, Any], active_ctx: Any, *, cache: OverlapCache | None = None
) -> dict[str, Any]:
    """:func:`domain_shift_from_summary`, answered once per summary and dataset in a run that shares *cache*."""
    return _cached_answer(
        cache,
        ("summary", summary["dataset_id"], active_ctx.dataset_id),
        lambda: domain_shift_from_summary(summary, active_ctx, cache=cache),
    )


def _cached_answer(cache: OverlapCache | None, key: tuple[str, ...], compute: Any) -> dict[str, Any]:
    """*compute()*'s report, or its :class:`DomainShiftUnavailable` refusal, remembered under *key* in *cache*."""
    if cache is not None and key in cache.answers:
        answer = cache.answers[key]
        if isinstance(answer, str):
            raise DomainShiftUnavailable(answer)
        return dict(answer)
    try:
        report = compute()
    except DomainShiftUnavailable as e:
        if cache is not None:
            cache.answers[key] = str(e)
        raise
    if cache is not None:
        cache.answers[key] = report
    return dict(report)


def _cached_vectors(cache: OverlapCache | None, dataset_id: str, build: Any) -> Any:
    """*build()*, or what it gave earlier in this run for *dataset_id*."""
    if cache is None:
        return build()
    if dataset_id not in cache.vectors:
        cache.vectors[dataset_id] = build()
    return cache.vectors[dataset_id]


def _media_vectors(ctx: Any) -> Any:
    """The whole-image vector of every media of *ctx* that has one, stacked in id order; ``None`` when none has."""
    import numpy as np

    from vtscore.embedding.media_vectors import media_embedding

    vectors = [
        np.asarray(emb, dtype=np.float32)
        for _cid, media in sorted(ctx.medias.items())
        if (emb := media_embedding(media)) is not None
    ]
    return np.stack(vectors) if vectors else None


def _stored_vectors(dataset_id: str) -> Any:
    """:func:`_media_vectors` of a registered dataset that is not loaded, read from its saved file."""
    from pathlib import Path

    from vtscore.datasets.registry import get_dataset
    from vtscore.embedding.matrix import stored_primary_vectors

    pkl_path = (get_dataset(dataset_id) or {}).get("pkl_path") or ""
    if not pkl_path or not Path(pkl_path).is_file():
        raise DomainShiftUnavailable("The reference dataset's saved file is missing")
    return stored_primary_vectors(pkl_path)


def infer_training_reference(
    labelset: Any, active_dataset_id: str, *, ingested: dict[str, list[dict[str, Any]]] | None = None
) -> tuple[str, dict[str, int]]:
    """The loaded dataset the caller can read that holds most of *labelset*'s labels, and every count.

    :func:`~vtscore.detectors.training_domain.infer_training_dataset` over the
    loaded, readable datasets, the active one winning a tie; ``""`` when none
    holds any.  The live half of :func:`training_reference`, which both trust
    checks read their reference from.  The active dataset is always counted:
    the caller has it open.  *ingested*
    (:func:`~vtscore.detectors.training_domain.recorded_ingests`) leaves out
    of each dataset's count the labels on media an import copied into it
    (#5572).
    """
    from vtscore.datasets.registry import can_user_access, get_dataset, get_loaded_ids
    from vtscore.detectors.training_domain import infer_training_dataset, labels_held_by_dataset
    from vtscore.state.core import get_active_context, get_context
    from vtsearch.auth import get_current_user

    user = get_current_user()
    contexts = [
        ctx
        for dataset_id in sorted(get_loaded_ids())
        if get_dataset(dataset_id) is not None
        and can_user_access(dataset_id, user)
        and (ctx := get_context(dataset_id)) is not None
    ]
    active_ctx = get_active_context()
    if (
        active_dataset_id
        and active_ctx.dataset_id == active_dataset_id
        and all(ctx.dataset_id != active_dataset_id for ctx in contexts)
    ):
        contexts.append(active_ctx)
    counts = labels_held_by_dataset(labelset, contexts, ingested=ingested)
    return infer_training_dataset(counts, prefer=active_dataset_id), counts


def _recorded_reference(recorded: dict[str, Any] | None, counts: dict[str, int]) -> dict[str, Any] | None:
    """The training dataset recorded on the detector, when it is a candidate the live counts lack.

    That is, when *recorded* names a registered dataset the caller can read
    that is not loaded: a loaded one is already in *counts*, counted live,
    which beats the record.
    """
    from vtscore.datasets.registry import can_user_access, get_dataset, get_loaded_ids
    from vtsearch.auth import get_current_user

    if recorded is None:
        return None
    dataset_id = recorded["dataset_id"]
    if dataset_id in counts or dataset_id in get_loaded_ids():
        return None
    if get_dataset(dataset_id) is None or not can_user_access(dataset_id, get_current_user()):
        return None
    return recorded


def _summary_reference(
    summary: dict[str, Any] | None,
    recorded: dict[str, Any] | None,
    stored: dict[str, Any] | None,
    counts: dict[str, int],
    labels_total: int,
) -> dict[str, Any] | None:
    """The training dataset the detector's *summary* stands in for, when nothing better can (#5567).

    That is, when the summarised dataset is neither counted live in *counts*
    (loaded, and readable by the caller) nor *stored*, the recorded dataset
    whose saved file the full check reads.  It counts as holding what the
    *recorded* pointer says it held, since the pointer is recorded with it;
    an import from an exported labelset records one naming the labels that
    came with the summary (#5572).  A summary with no pointer at all (one
    imported before #5572) counts as holding all *labels_total* labelled
    examples: it is a summary of the dataset the detector was trained on, and
    nothing here says how many of the labels that dataset held.  ``None`` when
    the pointer names another dataset: the pointer says where this detector's
    labels were drawn.
    """
    if summary is None:
        return None
    dataset_id = summary["dataset_id"]
    if dataset_id in counts or (stored is not None and stored["dataset_id"] == dataset_id):
        return None
    if recorded is None:
        return {"dataset_id": dataset_id, "dataset_name": summary["dataset_name"], "labels_held": labels_total}
    if recorded["dataset_id"] != dataset_id:
        return None
    return {
        "dataset_id": dataset_id,
        "dataset_name": summary["dataset_name"] or recorded["dataset_name"],
        "labels_held": recorded["labels_held"],
    }


@dataclass(frozen=True)
class TrainingReference:
    """The dataset a detector's labels say it was trained on (:func:`training_reference`), and what that rests on.

    *dataset_id* is ``""`` when no dataset holds any of the labels.  *counts*
    is how many of the labels each candidate holds; *stored* the recorded
    training dataset when it is registered here but not loaded, at its
    recorded count; *travelled* the dataset the detector's summary stands in
    for, when nothing better does.
    """

    dataset_id: str
    counts: dict[str, int]
    stored: dict[str, Any] | None = None
    travelled: dict[str, Any] | None = None


def training_reference(
    labelset: Any,
    active_id: str,
    *,
    recorded: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
    ingested: dict[str, list[dict[str, Any]]] | None = None,
) -> TrainingReference:
    """The dataset, among those the caller can read, holding the most of *labelset*'s Good / Bad examples.

    The candidates are the loaded datasets, counted live less the labels on
    media an import copied in (*ingested*, #5572); the one *recorded* on the
    detector, at its recorded count, when that one is not loaded (#5501); and,
    when the recorded one is not here at all, the detector's *summary* of it
    at the same count (#5567), or at every labelled example with no pointer
    beside it (#5571).  *active_id* wins a tie.  :func:`training_domain_for`
    compares against the answer, and the Evidence chip draws its baseline only
    on it (#5089), so neither moves the pointer while the other still points
    elsewhere.
    """
    from vtscore.detectors.training_domain import infer_training_dataset

    labels_total = 0 if labelset is None else sum(1 for e in labelset.elements if e.label in ("good", "bad"))
    if labels_total == 0:
        return TrainingReference("", {})
    reference_id, counts = infer_training_reference(labelset, active_id, ingested=ingested)
    stored = _recorded_reference(recorded, counts)
    if stored is not None:
        counts = {**counts, stored["dataset_id"]: min(stored["labels_held"], labels_total)}
        reference_id = infer_training_dataset(counts, prefer=active_id)
    travelled = _summary_reference(summary, recorded, stored, counts, labels_total)
    if travelled is not None:
        counts = {**counts, travelled["dataset_id"]: min(travelled["labels_held"], labels_total)}
        reference_id = infer_training_dataset(counts, prefer=active_id)
    return TrainingReference(reference_id, counts, stored, travelled)


def detector_training_reference(
    labelset: Any, active_id: str, detector_data: dict[str, Any] | None
) -> TrainingReference:
    """:func:`training_reference` with the pointer, summary and ingest record read off *detector_data*."""
    from vtscore.detectors.domain_summary import stored_domain_summary
    from vtscore.detectors.training_domain import recorded_ingests, recorded_training_dataset

    return training_reference(
        labelset,
        active_id,
        recorded=recorded_training_dataset(detector_data),
        summary=stored_domain_summary(detector_data),
        ingested=recorded_ingests(detector_data),
    )


def training_domain_for(
    labelset: Any,
    active_ctx: Any,
    *,
    recorded: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
    ingested: dict[str, list[dict[str, Any]]] | None = None,
    cache: OverlapCache | None = None,
) -> dict[str, Any]:
    """The overlap check for a detector with *labelset* on *active_ctx*, against the dataset it was trained on.

    The reference is :func:`training_reference`'s: the dataset, among those
    the caller can read, holding the most of the labelset's Good / Bad
    examples.  The loaded ones are counted live, less the labels on media an
    import copied into them (*ingested*,
    :func:`~vtscore.detectors.training_domain.recorded_ingests`, #5572), and
    the one *recorded* on the detector
    (:func:`~vtscore.detectors.training_domain.recorded_training_dataset`) at
    its recorded count, when that one is not loaded (#5501).  When the recorded
    one is not here at all, the detector's *summary* of it
    (:func:`~vtscore.detectors.domain_summary.stored_domain_summary`) stands in
    for it at the same count, and is what the check compares against (#5567);
    so it does, too, when the recorded one's saved file cannot be read.  A
    summary that came with the labels in an exported labelset stands in at the
    labels that came with it (#5571, #5572).
    *active_ctx*'s dataset wins a tie.  ``domain_shift`` is ``None`` (and
    ``reason`` says why) when there is nothing to compare: no labelled
    examples, none of them in a dataset to compare with, the examples being
    from the active dataset itself, or any of :func:`domain_shift_against`'s
    or :func:`domain_shift_from_summary`'s refusals.  The counts ride along
    either way, so the caller can say what the reference rests on.
    """
    from vtscore.datasets.registry import get_dataset

    result = _unavailable_check("")

    def unavailable(reason: str) -> dict[str, Any]:
        result["reason"] = reason
        return result

    labels_total = 0 if labelset is None else sum(1 for e in labelset.elements if e.label in ("good", "bad"))
    if labelset is None or labels_total == 0:
        return unavailable("The detector has no labelled examples")
    result["labels_total"] = labels_total

    active_id = active_ctx.dataset_id
    reference = training_reference(labelset, active_id, recorded=recorded, summary=summary, ingested=ingested)
    reference_id, counts, stored, travelled = (
        reference.dataset_id,
        reference.counts,
        reference.stored,
        reference.travelled,
    )
    result["labels_in_active"] = counts.get(active_id, 0)
    if not reference_id:
        return unavailable("None of the loaded datasets holds the detector's labelled examples")
    result["reference_dataset_id"] = reference_id
    result["reference_dataset_name"] = (get_dataset(reference_id) or {}).get("name", "") or (
        travelled["dataset_name"] if travelled is not None and reference_id == travelled["dataset_id"] else ""
    )
    result["labels_in_reference"] = counts[reference_id]
    if reference_id == active_id:
        return unavailable("The detector's labelled examples are from this dataset")
    try:
        result["domain_shift"] = _compare(
            reference_id,
            active_ctx,
            allow_unloaded=stored is not None and reference_id == stored["dataset_id"],
            summary=summary if summary is not None and summary["dataset_id"] == reference_id else None,
            summary_only=travelled is not None and reference_id == travelled["dataset_id"],
            cache=cache,
        )
    except DomainShiftUnavailable as e:
        return unavailable(str(e))
    return result


def _compare(
    reference_id: str,
    active_ctx: Any,
    *,
    allow_unloaded: bool,
    summary: dict[str, Any] | None,
    summary_only: bool,
    cache: OverlapCache | None,
) -> dict[str, Any]:
    """:func:`training_domain_for`'s report against *reference_id*: its vectors, else its *summary* (#5567).

    *summary_only* when the reference is not here at all, so the summary is
    all there is.  Otherwise the reference's own vectors are compared against,
    and the summary, when the detector carries one, steps in only if they
    cannot be read (a recorded dataset whose saved file is gone, say).  Raises
    :class:`DomainShiftUnavailable` with the full check's reason when neither
    can be read.
    """
    if summary is not None and summary_only:
        return cached_summary_shift(summary, active_ctx, cache=cache)
    try:
        return cached_domain_shift(reference_id, active_ctx, allow_unloaded=allow_unloaded, cache=cache)
    except DomainShiftUnavailable as e:
        if summary is None:
            raise
        unreadable = e
    try:
        return cached_summary_shift(summary, active_ctx, cache=cache)
    except DomainShiftUnavailable:
        raise unreadable from None


def training_domain_check() -> dict[str, Any]:
    """The overlap check for the active detector on the active dataset (:func:`training_domain_for`).

    The training dataset recorded on the detector, its summary and what its
    imports copied in are read from its file, so a detector paired with a
    dataset after a restart, or on another instance, is still compared
    against the one it was trained on.
    ``domain_shift`` is ``None`` with a ``reason`` when there is no detector or
    dataset to check, as well as on every refusal of :func:`training_domain_for`.
    """
    from vtscore.detectors.learned_sort import resolve_active_labelset
    from vtscore.state.core import (
        get_active_context,
        get_active_detector_context,
        is_request_missing_dataset_context,
        is_request_missing_detector_context,
    )

    det_ctx = get_active_detector_context()
    if not det_ctx.detector_id or is_request_missing_detector_context(det_ctx):
        return _unavailable_check("No detector is active")
    active_ctx = get_active_context()
    if not active_ctx.dataset_id or is_request_missing_dataset_context(active_ctx):
        return _unavailable_check("No dataset is active")
    from vtscore.detectors.domain_summary import stored_domain_summary
    from vtscore.detectors.training_domain import recorded_ingests, recorded_training_dataset

    labelset, _media_type = resolve_active_labelset(det_ctx)
    data = _detector_data_on_file(det_ctx.detector_id)
    return training_domain_for(
        labelset,
        active_ctx,
        recorded=recorded_training_dataset(data),
        summary=stored_domain_summary(data),
        ingested=recorded_ingests(data),
    )


def detector_training_domain(
    detector_data: dict[str, Any], active_ctx: Any, *, cache: OverlapCache | None = None
) -> dict[str, Any] | None:
    """The overlap check for the detector whose JSON is *detector_data*, on *active_ctx* (#5501).

    What a Find or AutoFind asks of each detector it ran: the labels and the
    recorded training dataset are the file's, since the run never touches the
    detector's loaded context, and so are the training dataset's summary
    (#5567) and what its imports copied in (#5572).  ``None``, logged, when
    the check itself fails:
    it rides alongside the run's results, which must not be lost to it.
    """
    from vtscore.datasets.labelset import LabelSet
    from vtscore.detectors.domain_summary import stored_domain_summary
    from vtscore.detectors.training_domain import recorded_ingests, recorded_training_dataset

    try:
        labelset = LabelSet.from_dict(detector_data.get("labelset") or {})
        return training_domain_for(
            labelset,
            active_ctx,
            recorded=recorded_training_dataset(detector_data),
            summary=stored_domain_summary(detector_data),
            ingested=recorded_ingests(detector_data),
            cache=cache,
        )
    except Exception:  # noqa: BLE001 - see the docstring
        logger.exception("Training-domain check failed for detector %s", detector_data.get("name", "?"))
        return None


def _detector_data_on_file(detector_id: str) -> dict[str, Any] | None:
    """*detector_id*'s detector JSON, or ``None`` with no file."""
    from vtscore.detectors.registry import get_detector
    from vtscore.detectors.store import _detector_path, _read_detector

    name = (get_detector(detector_id) or {}).get("name")
    if not name:
        return None
    return _read_detector(_detector_path(name))


def _unavailable_check(reason: str) -> dict[str, Any]:
    """An empty :func:`training_domain_for` answer, refused for *reason*."""
    return {
        "reference_dataset_id": "",
        "reference_dataset_name": "",
        "labels_total": 0,
        "labels_in_reference": 0,
        "labels_in_active": 0,
        "domain_shift": None,
        "reason": reason,
    }
