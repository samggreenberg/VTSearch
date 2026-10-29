"""Read a labelset against its label-embedding caches: training rows and labelled media.

:func:`build_xy_from_labelset` assembles the ``(X, y, groups, score_rows)`` a
head trains on from the caches
:func:`~vtscore.detectors.labelset_embeddings.populate_label_embeddings` left on
the detector's context.  The rest answer questions about which of the
labelset's elements a snapshot holds: the labelled media a threshold's
haystack drops, the hard-view Goods a head weighs less, and why a labelset
produced no trainable head.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vtscore.datasets.labelset import LabeledElement, LabelSet
from vtscore.detectors.labelset_embeddings import _lookups_for_snap


def build_xy_from_labelset(
    det_ctx,
    labelset: LabelSet,
) -> tuple[list[np.ndarray], list[float], list, dict]:
    """Build ``(X_list, y_list, groups, score_rows)`` from *det_ctx*'s caches.

    A Good element contributes its single cached vector.  A Bad element on a
    patch dataset contributes its flooded image-level + raw-patch negatives (cached in
    ``label_negative_regions`` by
    :func:`~vtscore.detectors.labelset_embeddings.populate_label_embeddings`); on a
    legacy dataset it contributes its single image-level vector, as before.
    ``groups`` carries one bag id per row - ``("g"/"b", element_id)`` - so the
    trainer/calibrator balance and split by element (image), not by row.  When
    nothing floods, every bag holds one row and the downstream path is
    byte-for-byte the pre-flood behaviour.

    ``score_rows`` maps each bag id to the full score-row stack that element's
    media is *scored* over at inference (``label_score_regions``), so threshold
    calibration collapses a Good bag exactly the way it collapses a Bad bag and
    the way the scorer collapses any image - see
    :func:`vtscore.detectors.training._calibration_score_rows`.  Empty on a
    legacy dataset, where there is nothing to correct.
    """
    from vtscore.detectors.labelset_elements import stable_element_id

    cache: dict[str, np.ndarray] = det_ctx.label_embeddings
    neg_cache: dict[str, list[np.ndarray]] = det_ctx.label_negative_regions
    score_cache: dict[str, list[np.ndarray]] = det_ctx.label_score_regions
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    groups: list = []
    score_rows: dict = {}

    def _record(group: tuple, eid: str) -> None:
        rows = score_cache.get(eid)
        if rows:
            score_rows[group] = np.stack(rows)

    for elem in labelset.elements:
        if elem.label not in ("good", "bad"):
            continue
        eid = stable_element_id(elem)
        if elem.label == "bad":
            flooded = neg_cache.get(eid)
            if flooded:
                for vec in flooded:
                    X_list.append(vec)
                    y_list.append(0.0)
                    groups.append(("b", eid))
                _record(("b", eid), eid)
                continue
        emb = cache.get(eid)
        if emb is None:
            continue
        X_list.append(emb)
        y_list.append(1.0 if elem.label == "good" else 0.0)
        group = ("g" if elem.label == "good" else "b", eid)
        groups.append(group)
        _record(group, eid)
    return X_list, y_list, groups, score_rows


def labelset_calibrating_groups(labelset: LabelSet) -> set:
    """Deprecated (#4362): the bags of :func:`build_xy_from_labelset` whose vote the learned sort chose.

    Part of the retired #4245 calibration filter, which chose the held-out
    votes the #4220 precision estimate could use as evidence.  A retrain no
    longer builds that estimate, so nothing reads this answer.  It still
    answers by each element's recorded surfacing provenance
    (:func:`~vtscore.datasets.vote_provenance.calibrates_precision`'s rule),
    with a ``DeprecationWarning``, until the name is removed.
    """
    from vtscore.datasets.vote_provenance import _calibrates_precision, _retired_calibration_filter, read_provenance
    from vtscore.detectors.labelset_elements import stable_element_id

    _retired_calibration_filter("labelset_calibrating_groups()")
    return {
        ("g" if elem.label == "good" else "b", stable_element_id(elem))
        for elem in labelset.elements
        if elem.label in ("good", "bad") and _calibrates_precision(read_provenance(elem.metadata))
    }


def labeled_media_ids(labelset: LabelSet, snap: dict[int, dict[str, Any]] | None) -> set[int]:
    """The media ids in *snap* that carry a good/bad label in *labelset*: the keys of :func:`labeled_media_labels`."""
    return set(labeled_media_labels(labelset, snap))


def labeled_media_labels(labelset: LabelSet, snap: dict[int, dict[str, Any]] | None) -> dict[int, bool]:
    """The media in *snap* that carry a good/bad label in *labelset*, as ``{media id: is Good}``.

    These are the in-dataset media the detector trains on, and therefore the
    ids the fold-anchored threshold must drop from its haystacks (issue #3308;
    see :func:`vtscore.detectors.training._fused_threshold`).  Cross-dataset
    elements that resolve to nothing in *snap* are absent from the haystack
    already, so they contribute nothing here.
    """
    if not snap:
        return {}
    from vtscore.detectors.labelset_elements import resolve_current_dataset_cid  # noqa: PLC0415
    from vtscore.state import build_media_lookup  # noqa: PLC0415

    # ``build_media_lookup`` reads ``media["id"]``; every Flask-loaded snap
    # carries it equal to the dict key, but minimal snaps (tests, embedded
    # callers) may not - default it to the key rather than requiring it.
    lookups = build_media_lookup({cid: {**m, "id": m.get("id", cid)} for cid, m in snap.items()})

    labels: dict[int, bool] = {}
    for elem in labelset.elements:
        if elem.label not in ("good", "bad"):
            continue
        cid = resolve_current_dataset_cid(elem, lookups)
        if cid is not None and cid in snap:
            labels[cid] = elem.label == "good"
    return labels


def hard_view_good_vectors(det_ctx, labelset: LabelSet, snap: dict[int, dict[str, Any]] | None) -> list[np.ndarray]:
    """The cached vectors of *labelset*'s Goods whose face in *snap* is a hard view (#5413).

    These are the rows :func:`build_xy_from_labelset` fits on, so a training pass
    hands them to :func:`vtscore.training.svm_prior.hard_view_rows` and the head
    weighs those Goods less.  A Good that resolves to no media in *snap*, or whose
    media carries no Face Quality verdict (any media type but a face), is not named.
    """
    if not snap:
        return []
    from vtscore.detectors.labelset_elements import resolve_current_dataset_cid, stable_element_id  # noqa: PLC0415
    from vtscore.media.face.imagekit_bridge import media_hard_view  # noqa: PLC0415
    from vtscore.state import build_media_lookup  # noqa: PLC0415

    lookups = build_media_lookup({cid: {**m, "id": m.get("id", cid)} for cid, m in snap.items()})
    cache: dict[str, np.ndarray] = det_ctx.label_embeddings
    out: list[np.ndarray] = []
    for elem in labelset.elements:
        if elem.label != "good":
            continue
        emb = cache.get(stable_element_id(elem))
        cid = resolve_current_dataset_cid(elem, lookups)
        if emb is not None and cid is not None and cid in snap and media_hard_view(snap[cid]):
            out.append(emb)
    return out


def labelset_resolution_report(
    det_ctx,
    labelset: LabelSet,
    *,
    media_type: str,
    snap: dict[int, dict[str, Any]] | None,
) -> dict[str, Any]:
    """Why *labelset* produced no trainable head, as a UI-facing diagnostic.

    :func:`~vtscore.detectors.labelset_training.train_from_labelset` and
    :func:`~vtscore.detectors.labelset_training.labelset_train_and_score` simply
    decline to train when the labels don't resolve into at least one Good and
    one Bad vector - which is the right shape for a *training* entry point but
    loses the "why did this detector produce nothing?" answer the find-label
    and portable-export routes show the user.  This rebuilds that answer from
    what the training pass already left on *det_ctx*: the elements
    :func:`build_xy_from_labelset` can assemble a row for are exactly the ones
    that resolved, so the failures are the complement.

    Call it **after**
    :func:`~vtscore.detectors.labelset_embeddings.populate_label_embeddings`
    (i.e. after the training attempt), never before - on a cold context it would
    report every element as a failure.  It is a failure-path-only report, so the extra
    :func:`~vtscore.state.media_lookup.build_media_lookup` pass it costs is
    paid only by a detector that is already not going to score.

    ``dataset_matched`` counts elements resolving to a media in *snap* by the
    full origin ▸ md5 ▸ name ladder :func:`resolve_current_dataset_cid` walks -
    not md5 alone, which is all the hand-rolled predecessor of this path could
    match on.
    """
    from vtscore.detectors.labelset_elements import resolve_current_dataset_cid, stable_element_id

    # Ask the trainer's own assembly which elements contributed, rather than
    # re-deriving "did this resolve?" from the embedding cache: a Bad element on
    # a patch dataset trains from ``label_negative_regions`` and can contribute
    # rows with nothing in ``label_embeddings``, so the cache alone would report
    # a label that trained fine as a failure.
    _x, y_list, groups, _score_rows = build_xy_from_labelset(det_ctx, labelset)
    contributed = {eid for _kind, eid in groups}
    has_good = any(y == 1.0 for y in y_list)
    has_bad = any(y == 0.0 for y in y_list)
    lookups = _lookups_for_snap(snap)

    total = 0
    dataset_matched = 0
    resolved_from_origin = 0
    failures: list[LabeledElement] = []
    for elem in labelset.elements:
        if elem.label not in ("good", "bad"):
            continue
        total += 1
        in_dataset = False
        if snap:
            cid = resolve_current_dataset_cid(elem, lookups)
            in_dataset = cid is not None and cid in snap
        if in_dataset:
            dataset_matched += 1
        if stable_element_id(elem) not in contributed:
            failures.append(elem)
        elif not in_dataset:
            resolved_from_origin += 1

    diagnostic: dict[str, Any] = {
        "total_labels": total,
        "dataset_matched": dataset_matched,
        "needed_resolution": total - dataset_matched,
        "resolved_from_origin": resolved_from_origin,
        "failed_resolution": len(failures),
        "has_good": has_good,
        "has_bad": has_bad,
        "media_type": media_type,
    }
    if failures:
        diagnostic["sample_failures"] = [
            {
                "origin": elem.origin,
                "origin_name": elem.origin_name,
                "filename": elem.filename,
                "md5": elem.md5[:12],
                "label": elem.label,
            }
            for elem in failures[:3]
        ]
    elif not has_good:
        # Under the label quota (#4643) a Good is all a detector needs: below
        # the quota the Goods' centroid answers, so no Bad is required.
        diagnostic["hint"] = "Every label resolved, but none is a Good: a detector needs at least one Good label"
    return diagnostic
