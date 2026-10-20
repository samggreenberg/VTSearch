"""The Evidence coverage chip's training baseline, stamped onto the detector it was drawn for (#5089).

The chip's verdict ranks each item's distance to its predicted class's votes
against the same distance over the detector's training haystack
(:func:`~vtscore.detectors.evidence_coverage.reference_cutoffs`): the votes'
own leave-one-out spread is no null, since active learning picks them.  The
haystack is there when the detector is tested on the dataset it was trained
on, and absent in the handoff the chip exists for, so the two numbers that
summarise it - one cut per class - are recorded on the detector there and read
back everywhere else.

They are scalars, not vectors or weights, but they live in the embedder's
geometry, so they are **stamped**: the embedder and its width, the labelset's
signature (:func:`~vtscore.detectors.model_loading.labelset_signature`, so any
vote, flip or redrawn box stales them) and the report's ``k`` / ``alpha``.  A
reader refuses a baseline whose stamp differs in any field, and the next test
on the training dataset draws it again.

**Where it is drawn** (#5120).  A Find on the training dataset draws it from
its scores (:func:`draw_baseline`).  The Train view's learned sort scores the
same dataset with the same head at every vote, so it has the same inputs, but
drawing there would add a distance pass over the whole dataset to every vote.
The sort leaves a :class:`PendingDraw` instead (its labels, scores and line),
which :func:`defer_baseline_draw` draws on a background timer once the sorts
have been quiet for :data:`DRAW_QUIET_SECONDS`.  A newer sort's replaces it,
so a burst of votes costs one draw.  A reader that cannot wait lands it at once
with :func:`flush_baseline_draw`, and a normal exit lands any still waiting
(a SIGKILL loses it, and the next test on the training dataset draws it).
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from vtscore.detectors.evidence_coverage import DEFAULT_ALPHA, DEFAULT_K

if TYPE_CHECKING:
    from vtscore.datasets.labelset import LabelSet
    from vtscore.state.core import DatasetContext, DetectorContext

logger = logging.getLogger(__name__)

#: The detector JSON key the baseline is recorded under.
EVIDENCE_BASELINE_KEY = "evidence_baseline"

#: How long a detector's learned sorts must have been quiet before the baseline
#: the last one left is drawn (#5120).  Long enough that a run of votes, each
#: re-running the sort, costs one draw when it ends; the readers that cannot
#: wait flush it (:func:`flush_baseline_draw`).
DRAW_QUIET_SECONDS = 15.0

# How long a normal exit waits on a draw in flight before giving up, so a slow
# draw cannot hang interpreter shutdown.
_EXIT_FLUSH_TIMEOUT = 30.0


def baseline_stamp(
    labelset: "LabelSet", embedder: str, dim: int, *, k: int = DEFAULT_K, alpha: float = DEFAULT_ALPHA
) -> dict[str, Any] | None:
    """The identity a baseline is valid for, or ``None`` when the embedder is unnamed.

    *embedder* is the embedder the labelset's vectors and the scored items'
    were read in, and *dim* their width.  Without a name there is nothing to
    refuse a mismatch on, so no baseline is drawn or read.
    """
    from vtscore.detectors.model_loading import labelset_signature  # noqa: PLC0415

    if not embedder:
        return None
    signature = json.dumps(labelset_signature(labelset), separators=(",", ":"))
    return {
        "embedder": str(embedder),
        "dim": int(dim),
        "labels": hashlib.blake2b(signature.encode("utf-8"), digest_size=16).hexdigest(),
        "k": int(k),
        "alpha": float(alpha),
    }


def stored_baseline(detector_data: Mapping[str, Any] | None, stamp: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The baseline recorded on *detector_data*, if its stamp is *stamp* in every field; else ``None``."""
    if stamp is None or not detector_data:
        return None
    baseline = detector_data.get(EVIDENCE_BASELINE_KEY)
    if not isinstance(baseline, dict):
        return None
    if any(baseline.get(key) != value for key, value in stamp.items()):
        return None
    return dict(baseline)


def record_baseline(path: Path, baseline: Mapping[str, Any], *, det_ctx: "DetectorContext | None" = None) -> bool:
    """Record *baseline* on the detector file at *path*, unless it already holds it; whether it wrote.

    The read-modify-write runs under ``label_sync_write_lock``, as every other
    detector-JSON writer's does, so a vote landing meanwhile is never lost.
    With *det_ctx*, the detector's loaded context, its cached labelset is
    re-pointed at the file afterwards, so the write does not read as an outside
    edit and rehydrate the session it holds (#5120).

    Never call this holding :data:`vtscore.state.core._state_lock`: the label
    sync takes that lock while it holds the write lock.
    """
    from vtscore.detectors.dataset_sync import _repoint_labelset_cache  # noqa: PLC0415
    from vtscore.detectors.labelset_ops import label_sync_write_lock  # noqa: PLC0415
    from vtscore.detectors.store import _read_detector, _write_detector  # noqa: PLC0415

    with label_sync_write_lock:
        data = _read_detector(path)
        if data is None or data.get(EVIDENCE_BASELINE_KEY) == dict(baseline):
            return False
        data[EVIDENCE_BASELINE_KEY] = dict(baseline)
        _write_detector(path, data)
    if det_ctx is not None:
        _repoint_labelset_cache(det_ctx, path)
    return True


# ---------------------------------------------------------------------------
# Drawing it
# ---------------------------------------------------------------------------


def scored_queries(
    scores: Iterable[tuple[int, float]],
    medias: Mapping[int, dict[str, Any]],
    embedder: str | None,
    threshold: float,
) -> tuple[list[int], np.ndarray | None, np.ndarray]:
    """The scored media as the Evidence report reads them: ``(ids, vectors, predicted positive)``.

    Every ``(id, score)`` of *scores* whose media in *medias* has a vector in
    *embedder*, the space the labels were resolved in, called positive where
    its score clears *threshold*.  The vectors are ``None`` when none has one.
    """
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    ids: list[int] = []
    vectors: list[np.ndarray] = []
    predicted: list[bool] = []
    for cid, score in scores:
        media = medias.get(cid)
        if media is None:
            continue
        emb = media_embedding(media, embedder_name=embedder)
        if emb is None:
            continue
        ids.append(cid)
        vectors.append(np.asarray(emb, dtype=np.float32))
        predicted.append(score >= threshold)
    if not vectors:
        return ids, None, np.zeros(0, dtype=bool)
    return ids, np.stack(vectors), np.asarray(predicted, dtype=bool)


def labelset_evidence(det_ctx: "DetectorContext", labelset: "LabelSet") -> tuple[np.ndarray, np.ndarray] | None:
    """The detector's labelled evidence as ``(Good rows, Bad rows)``, or ``None`` when no label has a vector.

    The rows its head trains on
    (:func:`~vtscore.detectors.labelset_training.build_xy_from_labelset`): a
    Good's one vector, a Bad's flooded image-level and patch negatives on a
    patch dataset, read from the label embeddings *det_ctx* holds in memory.
    """
    from vtscore.detectors.labelset_training import build_xy_from_labelset  # noqa: PLC0415

    x_list, y_list, _groups, _score_rows = build_xy_from_labelset(det_ctx, labelset)
    if not x_list:
        return None
    x = np.asarray(x_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    return x[y == 1.0], x[y == 0.0]


def draw_baseline(
    stamp: Mapping[str, Any],
    *,
    dataset_id: str,
    dataset_name: str,
    pos: np.ndarray,
    neg: np.ndarray,
    query_ids: Sequence[int],
    query_matrix: np.ndarray,
    predicted_positive: np.ndarray,
    labelled: Iterable[int],
) -> dict[str, Any]:
    """The baseline the training dataset *dataset_id*'s scored media give, stamped *stamp*.

    :func:`~vtscore.detectors.evidence_coverage.reference_cutoffs` over the
    scored media (:func:`scored_queries`) less the *labelled* ones, each called
    as *predicted_positive* says: at the line the Find draws.  The evidence
    route and the learned sort's :class:`PendingDraw` both draw it here, so the
    two agree on what the training baseline is.
    """
    from vtscore.detectors.evidence_coverage import reference_cutoffs  # noqa: PLC0415

    voted = set(labelled)
    keep = np.array([cid not in voted for cid in query_ids], dtype=bool)
    predicted = np.asarray(predicted_positive, dtype=bool)
    return {
        **stamp,
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        **reference_cutoffs(
            pos, neg, query_matrix[keep], predicted[keep], k=int(stamp["k"]), alpha=float(stamp["alpha"])
        ),
    }


@dataclass(frozen=True, eq=False)
class PendingDraw:
    """What a baseline draw reads, captured where the labels and the training dataset's scores are both current (#5120).

    The learned sort leaves one on the detector's training dataset: *results*
    are its ranked rows (``id`` / ``score``) over *medias*, the snapshot of
    *dataset* it scored, and *threshold* its line; *labelset* is the labels it
    trained on, whose vectors in *embedder* *det_ctx* holds.  *stamp*,
    *dataset_id* / *dataset_name* and *path* say what the baseline is recorded
    as and where.  Nothing is computed until :meth:`draw`.
    """

    detector_id: str
    path: Path
    stamp: Mapping[str, Any]
    dataset_id: str
    dataset_name: str
    det_ctx: "DetectorContext"
    dataset: "DatasetContext"
    labelset: "LabelSet"
    medias: Mapping[int, dict[str, Any]]
    results: Sequence[Mapping[str, Any]]
    threshold: float
    embedder: str | None

    def draw(self) -> dict[str, Any] | None:
        """The baseline, or ``None`` when the detector has moved on since: a vote, or its labels re-read in another space."""
        from vtscore.detectors.learned_sort import resolve_active_labelset  # noqa: PLC0415
        from vtscore.detectors.training_domain import labelled_media_ids  # noqa: PLC0415

        current, _media_type = resolve_active_labelset(self.det_ctx)
        if current is None or (self.det_ctx.embedder or None) != self.embedder:
            return None
        stamp = self.stamp
        if baseline_stamp(current, stamp["embedder"], stamp["dim"], k=stamp["k"], alpha=stamp["alpha"]) != dict(stamp):
            return None
        evidence = labelset_evidence(self.det_ctx, self.labelset)
        if evidence is None:
            return None
        query_ids, query_matrix, predicted = scored_queries(
            ((row["id"], row["score"]) for row in self.results), self.medias, self.embedder, self.threshold
        )
        if query_matrix is None or query_matrix.shape[1] != stamp["dim"]:
            return None
        pos, neg = evidence
        return draw_baseline(
            stamp,
            dataset_id=self.dataset_id,
            dataset_name=self.dataset_name,
            pos=pos,
            neg=neg,
            query_ids=query_ids,
            query_matrix=query_matrix,
            predicted_positive=predicted,
            labelled=labelled_media_ids(self.labelset, self.dataset),
        )


# Guards ``_waiting`` and ``_drawing``.  A leaf lock: nothing is called out to
# while it is held except ``Timer.start`` / ``Timer.cancel``.
_draws_cv = threading.Condition()
# Detector id -> the draw waiting out the quiet, and its timer.
_waiting: dict[str, tuple[PendingDraw, threading.Timer]] = {}
# Detector ids whose draw is running now.
_drawing: set[str] = set()


def defer_baseline_draw(pending: PendingDraw, *, quiet: float | None = None) -> None:
    """Draw *pending* once its detector has been quiet for *quiet* seconds (:data:`DRAW_QUIET_SECONDS` by default).

    Replaces any draw still waiting for the same detector: a newer sort's
    labels and scores supersede it.  Returns at once.
    """
    timer = threading.Timer(DRAW_QUIET_SECONDS if quiet is None else quiet, _fire, args=(pending,))
    timer.daemon = True
    with _draws_cv:
        replaced = _waiting.get(pending.detector_id)
        if replaced is not None:
            replaced[1].cancel()
        _waiting[pending.detector_id] = (pending, timer)
        timer.start()


def _take_locked(detector_id: str, pending: PendingDraw | None = None) -> PendingDraw | None:
    """Pop *detector_id*'s waiting draw (only if it is *pending*, when given).  Caller holds ``_draws_cv``."""
    waiting = _waiting.get(detector_id)
    if waiting is None or (pending is not None and waiting[0] is not pending):
        return None
    del _waiting[detector_id]
    waiting[1].cancel()
    return waiting[0]


def _land(pending: PendingDraw) -> bool:
    """Draw *pending* and record it; whether it wrote.  Its detector is marked in flight by the caller."""
    try:
        baseline = pending.draw()
        return baseline is not None and record_baseline(pending.path, baseline, det_ctx=pending.det_ctx)
    except Exception:  # noqa: BLE001 - a side channel: the next test on the training dataset draws it
        logger.exception("Failed to draw the Evidence baseline of detector %s", pending.detector_id)
        return False
    finally:
        with _draws_cv:
            _drawing.discard(pending.detector_id)
            _draws_cv.notify_all()


def _fire(pending: PendingDraw) -> None:
    """Timer callback: draw *pending* unless a newer sort replaced it or a reader landed it first."""
    with _draws_cv:
        taken = _take_locked(pending.detector_id, pending)
        if taken is None:
            return
        _drawing.add(taken.detector_id)
    _land(taken)


def flush_baseline_draw(detector_id: str, *, timeout: float | None = _EXIT_FLUSH_TIMEOUT) -> bool:
    """Land *detector_id*'s waiting draw now, after any already in flight; whether this call recorded one.

    For a reader of the recorded baseline that cannot wait out the quiet: the
    evidence route off the training dataset, and a detector unload.
    """
    with _draws_cv:
        if not _draws_cv.wait_for(lambda: detector_id not in _drawing, timeout):
            logger.warning("Gave up waiting for the Evidence baseline draw of detector %s", detector_id)
            return False
        taken = _take_locked(detector_id)
        if taken is None:
            return False
        _drawing.add(detector_id)
    return _land(taken)


def discard_baseline_draw(detector_id: str) -> None:
    """Drop *detector_id*'s waiting draw unrun, as when a fresher one has been drawn in its place."""
    with _draws_cv:
        if _take_locked(detector_id) is not None:
            _draws_cv.notify_all()


def wait_for_baseline_draws(timeout: float | None = None) -> bool:
    """Wait until no draw is waiting or in flight; whether that held before *timeout*.  For tests."""
    with _draws_cv:
        return _draws_cv.wait_for(lambda: not _waiting and not _drawing, timeout)


def flush_baseline_draws(timeout: float | None = None) -> None:
    """Land every waiting draw now and wait out those in flight.  A normal exit calls it."""
    with _draws_cv:
        detector_ids = list(_waiting)
    for detector_id in detector_ids:
        flush_baseline_draw(detector_id, timeout=timeout)
    with _draws_cv:
        _draws_cv.wait_for(lambda: not _drawing, timeout)


def reset_baseline_draws_for_tests() -> None:
    """Drop every waiting draw unrun, after any in flight (for conftest), so one test's cannot land in the next."""
    with _draws_cv:
        _draws_cv.wait_for(lambda: not _drawing, 10.0)
        for _pending, timer in _waiting.values():
            timer.cancel()
        _waiting.clear()


# Land a baseline still waiting out the quiet on a normal exit (Ctrl-C,
# gunicorn SIGQUIT, ``sys.exit``), so votes cast just before it carry one.
atexit.register(flush_baseline_draws, _EXIT_FLUSH_TIMEOUT)
