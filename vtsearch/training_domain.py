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
    """Report domain shift of *active_ctx*'s vectors under *reference_id*'s coverage atlas.

    Raises :class:`DomainShiftUnavailable` when the reference is not loaded or
    has no atlas, is the active dataset itself, uses another embedder or a
    patch embedder, or the active dataset has no embeddings.  The caller has
    already checked that *reference_id* is registered and readable.

    **Refused for patch embedders.**  The guard's separation between "my own
    data" and "a different corpus" collapses to 0.13 on ``dinov3_patch`` —
    close to a constant — because patch spaces are far less concentrated than
    single-vector ones, and every threshold repair was priced and failed.  See
    ``docs/experiments/2026-08-30-fit-quality-3329/REPORT.md`` (B4-B6).
    """
    import numpy as np

    from vtscore.coverage.atlas import domain_shift_report
    from vtscore.datasets.registry import get_dataset
    from vtscore.embedding.binding import embedder_supports_patch_regions
    from vtscore.embedding.media_vectors import media_embedding
    from vtscore.state.core import get_context

    ref_ctx = get_context(reference_id)
    if ref_ctx is None:
        raise DomainShiftUnavailable("Load the reference dataset before running a domain-shift check")
    atlas = ref_ctx.coverage_atlas
    if atlas is None:
        raise DomainShiftUnavailable("Reference dataset has no coverage atlas; build it first")
    if active_ctx.dataset_id == reference_id:
        raise DomainShiftUnavailable("Active dataset and reference dataset are the same")
    # Typicality p-values are meaningless across embedding spaces: refuse
    # instead of returning confident nonsense (the atlas geometry is stamped
    # by its dataset's embedder).
    ref_embedder = (get_dataset(reference_id) or {}).get("embedder", "")
    active_embedder = (get_dataset(active_ctx.dataset_id) or {}).get("embedder", "")
    if ref_embedder != active_embedder:
        raise DomainShiftUnavailable(
            f"Embedder mismatch: reference uses {ref_embedder or 'unknown'!r}, "
            f"active dataset uses {active_embedder or 'unknown'!r}"
        )
    # Patch spaces are too diffuse for the atlas's fixed-alpha guard to carry
    # information — measured, not assumed: on ``dinov3_patch`` it fires on 80%
    # of the reference dataset's *own* held-out data against 93% for a
    # different corpus, a separation of 0.13, i.e. close to a constant.  Both
    # obvious repairs (dropping the path averaging, refitting alpha per
    # embedder) were priced and made it worse.  Refusing is the honest answer;
    # returning a verdict here answers a direct question with noise.
    if embedder_supports_patch_regions(ref_embedder):
        raise DomainShiftUnavailable(
            f"Domain-shift checks are not supported for patch embedders "
            f"({ref_embedder!r}): the typicality guard does not separate "
            f"in-domain from out-of-domain data in a patch space"
        )

    vectors = [
        np.asarray(emb, dtype=np.float32)
        for media in active_ctx.medias.values()
        if (emb := media_embedding(media)) is not None
    ]
    if not vectors:
        raise DomainShiftUnavailable("Active dataset has no embeddings to compare")
    return {"reference_dataset_id": reference_id, **domain_shift_report(atlas, np.stack(vectors))}


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
    from vtscore.datasets.registry import can_user_access, get_dataset, get_loaded_ids
    from vtscore.detectors.learned_sort import resolve_active_labelset
    from vtscore.detectors.training_domain import infer_training_dataset, labels_held_by_dataset
    from vtscore.state.core import (
        get_active_context,
        get_active_detector_context,
        get_context,
        is_request_missing_dataset_context,
        is_request_missing_detector_context,
    )
    from vtsearch.auth import get_current_user

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

    user = get_current_user()
    contexts = [
        ctx
        for dataset_id in sorted(get_loaded_ids())
        if get_dataset(dataset_id) is not None
        and can_user_access(dataset_id, user)
        and (ctx := get_context(dataset_id)) is not None
    ]
    counts = labels_held_by_dataset(labelset, contexts)
    reference_id = infer_training_dataset(counts, prefer=active_ctx.dataset_id)
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
