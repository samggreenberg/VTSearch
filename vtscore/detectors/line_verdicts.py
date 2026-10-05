"""A finished test's verdict, kept on the detector (#4526).

Test mode (``docs/plans/test-mode.md``) measures a detector's line on a corpus
it never trained on: uniform picks from rank bands on both sides of the line,
and from them the line's precision, recall and F-beta as likely ranges
(:class:`~vtscore.training.thresholds.LineTest`).  This module keeps the
result of a finished test on the detector itself, so the detector's Stats and
the Dashboard's AutoRun tab can say what it was measured to ship, and so a
later test of the same dataset can resume from the picks already taken.

**What persists.**  One verdict per tested dataset, under
:data:`TEST_VERDICTS_KEY` in the detector's JSON beside its labelset: the
dataset's id and name, when the test finished, the balance it ran at, the
line's count and the corpus's size, the picks' ids and labels with the band
each was drawn from, and the precision, recall and F-beta ranges at Done.
Ids, labels and numbers only: never a score vector and never a model (the
*No Persisted Vectors* rule).  A newer test of the same dataset replaces the
older one.

**Stale.**  A verdict is about the head that ranked the corpus, and the head
is a function of the labels it was trained from
(:func:`~vtscore.detectors.model_loading.labelset_signature`, the identity
:func:`~vtscore.detectors.model_loading.cached_head_is_current` already reuses
a head by).  Each verdict records a digest of that signature
(:func:`labels_digest`); once the detector's labels change, which is what a
retrain is (a Train vote, **Add Corrections**, an import), the digest no
longer matches and the verdict reads stale.  It stays, flagged, as a spot
check's range does.  A test that finishes after **Add Corrections** already
folded the test set into the labels is stale from the start: its
``labels_digest`` is ``None``.  Deriving staleness on read rather than
marking it at write time means no writer of the labelset can forget to.

**Resume.**  A verdict also records a digest of the ranking its picks were
drawn from (:func:`ranking_digest`: the ids in rank order and the line's
count).  A later test of the same dataset whose ranking and line match takes
the kept picks back (:meth:`LineTestVerdict.kept_labels`), so the bands, the
ranges and *Lean the Threshold*'s re-estimates are there again without a
single new vote.  :func:`forget_verdict` drops a kept verdict: a reset (the
app's screenshot harness uses it between shots), since a user never needs
one.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from vtscore.training.thresholds.line_test import Estimate, found_words

if TYPE_CHECKING:
    from vtscore.datasets.labelset import LabelSet
    from vtscore.state.core import DetectorContext
    from vtscore.training.thresholds.line_test import LineTest

log = logging.getLogger(__name__)

#: The detector JSON's key for its kept verdicts, a list newest first.
TEST_VERDICTS_KEY = "test_verdicts"


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def labels_digest(labelset: "LabelSet") -> str:
    """A digest of the training labels in *labelset*: what a verdict is stale against.

    Taken over :func:`~vtscore.detectors.model_loading.labelset_signature`, so
    it changes exactly when the head trained from the labels would (a label,
    an element, a Good label's region box) and not on metadata or provenance,
    and it is stable across a JSON round-trip.
    """
    from vtscore.detectors.model_loading import labelset_signature  # noqa: PLC0415

    return _digest(json.dumps(labelset_signature(labelset), separators=(",", ":")))


def ranking_digest(ranking_ids: Iterable[int], line_count: int) -> str:
    """A digest of a ranking (ids in rank order) and its line's count: the frame a test's bands are cut from."""
    return _digest(",".join(str(int(i)) for i in ranking_ids) + f"|{int(line_count)}")


def _estimate(raw: Any) -> Estimate:
    return Estimate(float(raw["point"]), float(raw["lo"]), float(raw["hi"]))


@dataclass(frozen=True)
class LineTestVerdict:
    """One finished test of a detector's line on one dataset, as the detector keeps it.

    *picks* are ``(media_id, right, band)``: the media id in the dataset's id
    space, whether the user voted it a match, and the index of the band it was
    drawn from.  *labels_digest* is ``None`` for a verdict stale from the start.
    """

    dataset_id: str
    dataset_name: str
    tested_at: float
    beta: float
    line_count: int
    size: int
    ranking_digest: str
    labels_digest: str | None
    picks: tuple[tuple[int, bool, int], ...]
    precision: Estimate
    recall: Estimate
    fbeta: Estimate

    @classmethod
    def from_test(
        cls,
        test: "LineTest",
        *,
        dataset_id: str,
        dataset_name: str,
        labels_digest: str | None,
        tested_at: float | None = None,
    ) -> "LineTestVerdict":
        """The verdict of finished *test* on *dataset_id*.  Refuses a test with nothing to estimate."""
        if test.nothing_to_test:
            raise ValueError("a test with nothing to test has no verdict to keep")
        est = test.estimates()
        picks = tuple(sorted((int(cid), bool(right), int(test.pick_band[cid])) for cid, right in test.labels.items()))
        return cls(
            dataset_id=str(dataset_id),
            dataset_name=str(dataset_name),
            tested_at=time.time() if tested_at is None else float(tested_at),
            beta=float(test.beta),
            line_count=int(test.line_count),
            size=int(test.size),
            ranking_digest=ranking_digest(test.ranking_ids, test.line_count),
            labels_digest=labels_digest,
            picks=picks,
            precision=est.precision,
            recall=est.recall,
            fbeta=est.fbeta,
        )

    @classmethod
    def from_dict(cls, raw: Any) -> "LineTestVerdict | None":
        """Parse one kept verdict; ``None`` for an entry that does not hold one (hand-edited, truncated)."""
        try:
            return cls(
                dataset_id=str(raw["dataset_id"]),
                dataset_name=str(raw.get("dataset_name", "") or ""),
                tested_at=float(raw["tested_at"]),
                beta=float(raw["beta"]),
                line_count=int(raw["line_count"]),
                size=int(raw["size"]),
                ranking_digest=str(raw["ranking_digest"]),
                labels_digest=None if raw.get("labels_digest") is None else str(raw["labels_digest"]),
                picks=tuple((int(p["id"]), p["label"] == "good", int(p["band"])) for p in raw["picks"]),
                precision=_estimate(raw["precision"]),
                recall=_estimate(raw["recall"]),
                fbeta=_estimate(raw["fbeta"]),
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            return None

    def to_dict(self) -> dict[str, Any]:
        """The verdict as the detector JSON keeps it."""
        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "tested_at": self.tested_at,
            "beta": self.beta,
            "line_count": self.line_count,
            "size": self.size,
            "ranking_digest": self.ranking_digest,
            "labels_digest": self.labels_digest,
            "picks": [
                {"id": cid, "label": "good" if right else "bad", "band": band} for cid, right, band in self.picks
            ],
            "precision": self.precision.as_dict(),
            "recall": self.recall.as_dict(),
            "fbeta": self.fbeta.as_dict(),
        }

    @property
    def labelled(self) -> int:
        """The picks the ranges rest on."""
        return len(self.picks)

    @property
    def found(self) -> str:
        """The recall range in the spot check's words."""
        return found_words(self.recall)

    def stale(self, current_labels_digest: str | None) -> bool:
        """Whether the detector has been retrained since: its labels are no longer the ones that ranked the picks."""
        return self.labels_digest is None or self.labels_digest != current_labels_digest

    def kept_labels(self, ranking_ids: Sequence[int], line_count: int) -> dict[int, bool] | None:
        """The kept picks as ``{media_id: right}``, when *ranking_ids* and *line_count* are the ranking they came from; else ``None``."""
        if ranking_digest(ranking_ids, line_count) != self.ranking_digest:
            return None
        return {cid: right for cid, right, _ in self.picks}

    def summary(self, *, stale: bool, dataset_name: str | None = None) -> dict[str, Any]:
        """The verdict as a reader shows it: the ranges, the *found* words, and the stale mark; the picks stay on disk."""
        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name if dataset_name is None else dataset_name,
            "tested_at": self.tested_at,
            "beta": self.beta,
            "line_count": self.line_count,
            "size": self.size,
            "labelled": self.labelled,
            "precision": self.precision.as_dict(),
            "recall": self.recall.as_dict(),
            "fbeta": self.fbeta.as_dict(),
            "found": self.found,
            "stale": bool(stale),
        }


# ------------------------------------------------------------------ the detector JSON


def read_verdicts(data: Mapping[str, Any] | None) -> list[LineTestVerdict]:
    """The verdicts kept in a detector's JSON, newest first; entries that do not parse are skipped."""
    raw = (data or {}).get(TEST_VERDICTS_KEY)
    if not isinstance(raw, list):
        return []
    verdicts = [v for v in (LineTestVerdict.from_dict(item) for item in raw) if v is not None]
    return sorted(verdicts, key=lambda v: v.tested_at, reverse=True)


def verdict_for(data: Mapping[str, Any] | None, dataset_id: str) -> LineTestVerdict | None:
    """The verdict kept for *dataset_id*, or ``None``."""
    return next((v for v in read_verdicts(data) if v.dataset_id == dataset_id), None)


def put_verdict(data: dict[str, Any], verdict: LineTestVerdict) -> None:
    """Keep *verdict* in a detector's JSON, replacing any earlier one for its dataset (one per tested dataset)."""
    kept = [v for v in read_verdicts(data) if v.dataset_id != verdict.dataset_id]
    data[TEST_VERDICTS_KEY] = [v.to_dict() for v in sorted([verdict, *kept], key=lambda v: v.tested_at, reverse=True)]


def drop_verdict(data: dict[str, Any], dataset_id: str) -> bool:
    """Drop the verdict a detector's JSON keeps for *dataset_id*; ``True`` when there was one."""
    kept = read_verdicts(data)
    rest = [v for v in kept if v.dataset_id != dataset_id]
    if len(rest) == len(kept):
        return False
    data[TEST_VERDICTS_KEY] = [v.to_dict() for v in rest]
    return True


def current_labels_digest(data: Mapping[str, Any] | None) -> str:
    """:func:`labels_digest` of the labelset in a detector's JSON."""
    from vtscore.datasets.labelset import LabelSet  # noqa: PLC0415

    return labels_digest(LabelSet.from_dict((data or {}).get("labelset") or {}))


def _dataset_name(verdict: LineTestVerdict) -> str:
    """The dataset's name now, when it is still registered; else the one kept with the verdict."""
    from vtscore.datasets.registry import get_dataset  # noqa: PLC0415

    entry = get_dataset(verdict.dataset_id)
    return (entry or {}).get("name", "") or verdict.dataset_name


def verdict_summaries(data: Mapping[str, Any] | None, *, latest_only: bool = False) -> list[dict[str, Any]]:
    """Every verdict a detector's JSON keeps (or only the newest), summarised for a reader with its stale mark.

    The stale mark is against the labelset in the same JSON; the dataset's
    name is its registered name now, falling back to the one kept.
    """
    verdicts = read_verdicts(data)
    if latest_only:
        verdicts = verdicts[:1]
    if not verdicts:
        return []
    digest = current_labels_digest(data)
    return [v.summary(stale=v.stale(digest), dataset_name=_dataset_name(v)) for v in verdicts]


# ------------------------------------------------------------------ the active detector


def _detector_file(det_ctx: "DetectorContext"):
    """The path of *det_ctx*'s detector JSON, or ``None`` with no registry entry to name it."""
    from vtscore.detectors.registry import get_detector  # noqa: PLC0415
    from vtscore.detectors.store import _detector_path  # noqa: PLC0415

    entry = get_detector(det_ctx.detector_id) if det_ctx.detector_id else None
    if not entry or not entry.get("name"):
        return None
    return _detector_path(entry["name"])


def kept_verdict(det_ctx: "DetectorContext", dataset_id: str) -> tuple[LineTestVerdict, bool] | None:
    """The verdict *det_ctx*'s detector keeps for *dataset_id* and whether it is stale, or ``None``."""
    from vtscore.detectors.store import _read_detector  # noqa: PLC0415

    path = _detector_file(det_ctx)
    if path is None or not dataset_id:
        return None
    data = _read_detector(path)
    verdict = verdict_for(data, dataset_id)
    if verdict is None:
        return None
    return verdict, verdict.stale(current_labels_digest(data))


def keep_verdict(
    det_ctx: "DetectorContext",
    test: "LineTest",
    *,
    dataset_id: str,
    dataset_name: str,
    tested_at: float | None = None,
) -> LineTestVerdict | None:
    """Write finished *test*'s verdict onto *det_ctx*'s detector JSON; returns it, or ``None`` when there was nowhere to.

    Read, merged and written under
    :data:`~vtscore.detectors.label_sync.label_sync_write_lock`, as every
    detector-JSON read-modify-write is, so a concurrent label sync can neither
    lose the verdict nor be lost to it.  The verdict is stale from the start
    when ``det_ctx.find_eval_stale`` says **Add Corrections** already folded
    the test set into the labels.  The cached labelset is re-pointed at the
    file afterwards, so the write does not read as an outside edit and
    rehydrate the Find session the test belongs to.
    """
    from vtscore.detectors.dataset_sync import _repoint_labelset_cache  # noqa: PLC0415
    from vtscore.detectors.label_sync import label_sync_write_lock  # noqa: PLC0415
    from vtscore.detectors.store import _read_detector, _write_detector  # noqa: PLC0415

    if test.nothing_to_test or not test.phase().done or not dataset_id:
        return None
    path = _detector_file(det_ctx)
    if path is None:
        return None
    with label_sync_write_lock:
        data = _read_detector(path)
        if data is None:
            return None
        digest = None if det_ctx.find_eval_stale else current_labels_digest(data)
        verdict = LineTestVerdict.from_test(
            test, dataset_id=dataset_id, dataset_name=dataset_name, labels_digest=digest, tested_at=tested_at
        )
        put_verdict(data, verdict)
        _write_detector(path, data)
    _repoint_labelset_cache(det_ctx, path)
    log.info("kept the test verdict of detector %s on dataset %s", det_ctx.detector_id, dataset_id)
    return verdict


def forget_verdict(det_ctx: "DetectorContext", dataset_id: str) -> bool:
    """Drop the verdict *det_ctx*'s detector keeps for *dataset_id*, so the next test there starts afresh (a reset).

    ``True`` when there was one.  The same read-modify-write discipline as
    :func:`keep_verdict`; the file is not rewritten when there is nothing to drop.
    """
    from vtscore.detectors.dataset_sync import _repoint_labelset_cache  # noqa: PLC0415
    from vtscore.detectors.label_sync import label_sync_write_lock  # noqa: PLC0415
    from vtscore.detectors.store import _read_detector, _write_detector  # noqa: PLC0415

    path = _detector_file(det_ctx)
    if path is None or not dataset_id:
        return False
    with label_sync_write_lock:
        data = _read_detector(path)
        if data is None or not drop_verdict(data, dataset_id):
            return False
        _write_detector(path, data)
    _repoint_labelset_cache(det_ctx, path)
    return True
