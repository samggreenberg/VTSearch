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
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vtscore.detectors.evidence_coverage import DEFAULT_ALPHA, DEFAULT_K

if TYPE_CHECKING:
    from vtscore.datasets.labelset import LabelSet

#: The detector JSON key the baseline is recorded under.
EVIDENCE_BASELINE_KEY = "evidence_baseline"


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


def record_baseline(path: Path, baseline: Mapping[str, Any]) -> bool:
    """Record *baseline* on the detector file at *path*, unless it already holds it; whether it wrote.

    The read-modify-write runs under ``label_sync_write_lock``, as every other
    detector-JSON writer's does, so a vote landing meanwhile is never lost.
    """
    from vtscore.detectors.labelset_ops import label_sync_write_lock  # noqa: PLC0415
    from vtscore.detectors.store import _read_detector, _write_detector  # noqa: PLC0415

    with label_sync_write_lock:
        data = _read_detector(path)
        if data is None or data.get(EVIDENCE_BASELINE_KEY) == dict(baseline):
            return False
        data[EVIDENCE_BASELINE_KEY] = dict(baseline)
        _write_detector(path, data)
    return True
