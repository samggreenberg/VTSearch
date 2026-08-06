"""Learned-sort orchestration: resolve labelset → train → score → reconcile.

These helpers backed the ``/api/learned-sort`` route handler in
``vtsearch/routes/sorting.py`` until they accreted the whole
labelset-resolution / vote-reconciliation / det-ctx-update pipeline.  None
of it touches Flask or the request context, so it belongs in the library
tier where it can be exercised without a Flask client (see
docs/plans/code-structure-review.md, Theme A).  The route is now
request↔library glue: it gathers settings / votes, calls
:func:`run_learned_sort` inside its background-job closure, and renders the
result.

The ``good`` / ``bad`` arguments are accepted as plain iterables *or* the
app-tier vote proxies; everything here normalises with ``set()`` / ``dict()``
/ ``list()`` so resolution stays correct whether a route passes the live
proxies (resolved inside the thread context) or a test passes literal sets.
"""

from __future__ import annotations


def resolve_active_labelset(det_ctx):
    """Resolve the labelset for the active detector → ``(labelset, media_type)``.

    Returns ``(None, "")`` for the empty sentinel, a detector with no id, or a
    detector whose store entry / file is missing.  Prefers the context's
    cached labelset when present to avoid a disk read.
    """
    from vtscore.datasets.labelset import LabelSet
    from vtscore.detectors.registry import get_detector
    from vtscore.detectors.store import _detector_path, _read_detector
    from vtscore.state.core import _empty_detector_context

    if det_ctx is _empty_detector_context or not det_ctx.detector_id:
        return None, ""
    if det_ctx.cached_labelset is not None:
        return det_ctx.cached_labelset, det_ctx.cached_labelset_media_type
    entry = get_detector(det_ctx.detector_id)
    if not entry or not entry.get("name"):
        return None, ""
    det_data = _read_detector(_detector_path(entry["name"]))
    if not det_data:
        return None, ""
    return LabelSet.from_dict(det_data.get("labelset") or {}), det_data.get("media_type", "") or ""


def resolve_labelset_local_state(labelset, snap):
    """Resolve labelset elements to local cids using the dataset snapshot.

    Returns ``(local_good, local_bad, training_medias, has_cross_dataset)``.
    All are ``None`` / ``False`` when *labelset* is ``None`` (the non-labelset
    path).
    """
    if labelset is None:
        return None, None, None, False

    from vtscore.state import cached_media_lookups, resolve_media_ids

    # *snap* is the active dataset's medias snapshot (this runs inside the
    # caller's ``thread_dataset_context(ds_ctx)``), so the revision-keyed cache
    # is consistent with it - reused across clicks instead of rebuilt (S14).
    origin_lookup, md5_lookup, name_lookup = cached_media_lookups()
    local_good: set[int] = set()
    local_bad: set[int] = set()
    training_medias: dict[int, dict] = {}
    has_cross_dataset = False
    for el in labelset.elements:
        if el.label not in ("good", "bad"):
            continue
        cids = resolve_media_ids(el.to_dict(), origin_lookup, md5_lookup, name_lookup)
        if not cids:
            has_cross_dataset = True
            continue
        target = local_good if el.label == "good" else local_bad
        for cid in cids:
            target.add(cid)
            if cid in snap:
                training_medias[cid] = snap[cid]
    return local_good, local_bad, training_medias, has_cross_dataset


def model_matches_local_votes(labelset, has_cross_dataset, local_good, local_bad, good, bad) -> bool:
    """Decide whether the trained model maps cleanly onto current-dataset votes.

    The progress cache is keyed on local cids, so we only inject when the
    training set is fully representable there; otherwise we'd return a
    cross-dataset model on a local-only replay.
    """
    if labelset is None:
        return True
    return not has_cross_dataset and local_good == set(good) and local_bad == set(bad)


def update_det_ctx_with_trained_model(det_ctx, model, threshold, labelset, training_medias, snap, good, bad) -> None:
    """Persist the freshly trained model + training set onto *det_ctx*."""
    det_ctx.model = model
    det_ctx.threshold = threshold
    if labelset is not None:
        det_ctx.training_medias = training_medias or {}
    else:
        training = {}
        for cid in list(good) + list(bad):
            if cid in snap:
                training[cid] = snap[cid]
        det_ctx.training_medias = training
    if snap:
        first = next(iter(snap.values()), {})
        # Stamp the *cache-space* marker with the space the MLP was actually
        # trained in: the detector's chosen primary when this dataset supplies
        # it, else the dataset score precedence (keying_embedder_for_snap).  For
        # a legacy detector with no chosen primary this is the precedence,
        # byte-for-byte the pre-per-detector behaviour.
        from vtscore.embedding.binding import keying_embedder_for_snap

        det_ctx.embedder = keying_embedder_for_snap(det_ctx, snap)
        det_ctx.media_type = first.get("media_type", "")


def build_learned_sort_signature(
    *,
    det_ctx,
    ds_ctx,
    snap,
    labelset,
    good,
    bad,
    region_boxes_snapshot,
    inclusion_value,
    calibrate_count_value,
    calibration_fraction_value,
):
    """Build the no-op short-circuit key for a learned-sort run.

    Two runs with equal signatures produce identical results, so the route's
    job manager can return the cached result instead of retraining.
    """
    from vtscore.detectors.labelset_elements import stable_element_id

    if labelset is not None:
        labels_sig = tuple(sorted((el.label, stable_element_id(el)) for el in labelset.elements))
    else:
        labels_sig = (
            ("good", tuple(sorted(good))),
            ("bad", tuple(sorted(bad))),
            ("regions", tuple(sorted(region_boxes_snapshot.items()))),
        )
    return (
        det_ctx.detector_id,
        ds_ctx.dataset_id,
        tuple(sorted(snap.keys())),
        labels_sig,
        inclusion_value,
        calibrate_count_value,
        calibration_fraction_value,
    )


def run_learned_sort(
    *,
    det_ctx,
    ds_ctx,
    snap,
    labelset,
    det_media_type,
    good,
    bad,
    region_boxes_snapshot,
    inclusion_value,
    calibrate_count_value,
    calibration_fraction_value,
):
    """Train and score a learned sort, reconciling the result with local votes.

    Runs inside the dataset/detector thread contexts so the vote proxies and
    progress cache resolve against the originating request's contexts even on
    a background daemon thread.  Trains via the labelset pipeline when
    *labelset* is set, otherwise the raw-vote pipeline; injects the live model
    into the progress cache when it maps cleanly onto current-dataset votes;
    and stores the model + training set on *det_ctx*.  Returns
    ``(results, threshold)``.
    """
    from vtscore.detectors.labeling_progress import inject_live_model
    from vtscore.detectors.labelset_training import labelset_train_and_score
    from vtscore.detectors.training import train_and_score
    from vtscore.state import update_learned_scores
    from vtscore.state.core import (
        _empty_detector_context,
        thread_dataset_context,
        thread_detector_context,
    )

    with thread_dataset_context(ds_ctx), thread_detector_context(det_ctx):
        if labelset is not None:
            results, threshold, model = labelset_train_and_score(
                det_ctx,
                labelset,
                media_type=det_media_type,
                clips_dict=snap,
                inclusion_value=inclusion_value,
                calibrate_count=calibrate_count_value,
                calibration_fraction=calibration_fraction_value,
            )
        else:
            results, threshold, model = train_and_score(
                snap,
                dict(good),
                dict(bad),
                inclusion_value,
                calibrate_count=calibrate_count_value,
                calibration_fraction=calibration_fraction_value,
                vote_region_boxes=region_boxes_snapshot,
                det_ctx=det_ctx,
            )

        update_learned_scores({r["id"]: r["score"] for r in results})

        local_good, local_bad, training_medias, has_cross_dataset = resolve_labelset_local_state(labelset, snap)

        if model is not None and model_matches_local_votes(
            labelset, has_cross_dataset, local_good, local_bad, good, bad
        ):
            inject_live_model(good, bad, model, threshold)

        if det_ctx is not _empty_detector_context and model is not None:
            update_det_ctx_with_trained_model(det_ctx, model, threshold, labelset, training_medias, snap, good, bad)

    return results, threshold
