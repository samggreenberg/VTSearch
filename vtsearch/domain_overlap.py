"""The training-domain overlap check: how much of the active dataset looks unlike a reference dataset.

Two routes ask it.  ``GET /api/datasets/registry/<id>/domain-shift`` takes the
reference from the caller (the Tune result's **Compare against** picker).
``GET /api/find/training-domain`` works the reference out itself, from the
loaded dataset holding most of the active detector's labelled examples
(:mod:`vtscore.detectors.training_domain`), so the SPA can warn the moment a
detector meets a dataset unlike the one it was trained on, before the user
runs anything (#4961).  Both refuse on the same conditions, kept here.
"""

from __future__ import annotations

from typing import Any


class DomainShiftUnavailable(Exception):
    """The overlap check cannot be run against this reference; the message says why."""


def domain_shift_against(reference_id: str, active_ctx: Any) -> dict[str, Any]:
    """Report domain shift of *active_ctx*'s vectors against *reference_id*'s own vectors.

    :func:`vtscore.coverage.domain_shift_report`: a split-conformal kNN over the
    reference dataset's whole-image vectors, so a p-value is uniform for an
    item drawn from the reference, on every embedder, patch spaces included
    (#3348).  Raises :class:`DomainShiftUnavailable` when the reference is not
    loaded, is the active dataset itself, uses another embedder, has too few
    embeddings to calibrate, or the active dataset has no embeddings.  The
    caller has already checked that *reference_id* is registered and readable.

    Until #3348 this ranked against the reference's coverage atlas, needed the
    atlas built, and refused patch embedders, where it fired on 60-80 % of the
    reference's own held-out data.
    """
    import numpy as np

    from vtscore.coverage.domain_shift import ReferenceTooSmall, domain_shift_report
    from vtscore.datasets.registry import get_dataset
    from vtscore.state.core import get_context

    ref_ctx = get_context(reference_id)
    if ref_ctx is None:
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

    vectors = _media_vectors(active_ctx)
    if not vectors:
        raise DomainShiftUnavailable("Active dataset has no embeddings to compare")
    reference = _media_vectors(ref_ctx)
    if not reference:
        raise DomainShiftUnavailable("Reference dataset has no embeddings to compare against")
    try:
        report = domain_shift_report(np.stack(reference), np.stack(vectors))
    except ReferenceTooSmall as e:
        raise DomainShiftUnavailable(f"Reference dataset is too small for a domain-shift check: {e}") from e
    return {"reference_dataset_id": reference_id, **report}


def _media_vectors(ctx: Any) -> list[Any]:
    """The whole-image vector of every media of *ctx* that has one, in id order."""
    import numpy as np

    from vtscore.embedding.media_vectors import media_embedding

    return [
        np.asarray(emb, dtype=np.float32)
        for _cid, media in sorted(ctx.medias.items())
        if (emb := media_embedding(media)) is not None
    ]


def infer_training_reference(labelset: Any, active_dataset_id: str) -> tuple[str, dict[str, int]]:
    """The loaded dataset the caller can read that holds most of *labelset*'s labels, and every count.

    :func:`~vtscore.detectors.training_domain.infer_training_dataset` over the
    loaded, readable datasets, the active one winning a tie; ``""`` when none
    holds any.  Both trust checks read their reference from here: the overlap
    check compares against it, and the evidence check draws its baseline on it
    (#5089).  The active dataset is always counted: the caller has it open.
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
    counts = labels_held_by_dataset(labelset, contexts)
    return infer_training_dataset(counts, prefer=active_dataset_id), counts


def training_domain_check() -> dict[str, Any]:
    """The overlap check for the active detector on the active dataset, against the dataset it was trained on.

    The reference is the loaded dataset, among those the caller can read,
    holding the most of the detector's Good / Bad examples; the active dataset
    wins a tie.  ``domain_shift`` is ``None`` (and ``reason`` says why) when
    there is nothing to compare: no detector or dataset, no labelled examples,
    none of them in a loaded dataset, the examples being from the active
    dataset itself, or any of :func:`domain_shift_against`'s refusals.  The
    counts ride along either way, so the caller can say what the reference
    rests on.
    """
    from vtscore.datasets.registry import get_dataset
    from vtscore.detectors.learned_sort import resolve_active_labelset
    from vtscore.state.core import (
        get_active_context,
        get_active_detector_context,
        is_request_missing_dataset_context,
        is_request_missing_detector_context,
    )

    result: dict[str, Any] = {
        "reference_dataset_id": "",
        "reference_dataset_name": "",
        "labels_total": 0,
        "labels_in_reference": 0,
        "labels_in_active": 0,
        "domain_shift": None,
        "reason": "",
    }

    def unavailable(reason: str) -> dict[str, Any]:
        result["reason"] = reason
        return result

    det_ctx = get_active_detector_context()
    if not det_ctx.detector_id or is_request_missing_detector_context(det_ctx):
        return unavailable("No detector is active")
    active_ctx = get_active_context()
    if not active_ctx.dataset_id or is_request_missing_dataset_context(active_ctx):
        return unavailable("No dataset is active")
    labelset, _media_type = resolve_active_labelset(det_ctx)
    labels_total = 0 if labelset is None else sum(1 for e in labelset.elements if e.label in ("good", "bad"))
    if labelset is None or labels_total == 0:
        return unavailable("The detector has no labelled examples")
    result["labels_total"] = labels_total

    reference_id, counts = infer_training_reference(labelset, active_ctx.dataset_id)
    result["labels_in_active"] = counts.get(active_ctx.dataset_id, 0)
    if not reference_id:
        return unavailable("None of the loaded datasets holds the detector's labelled examples")
    result["reference_dataset_id"] = reference_id
    result["reference_dataset_name"] = (get_dataset(reference_id) or {}).get("name", "") or ""
    result["labels_in_reference"] = counts[reference_id]
    if reference_id == active_ctx.dataset_id:
        return unavailable("The detector's labelled examples are from this dataset")
    try:
        result["domain_shift"] = domain_shift_against(reference_id, active_ctx)
    except DomainShiftUnavailable as e:
        return unavailable(str(e))
    return result
