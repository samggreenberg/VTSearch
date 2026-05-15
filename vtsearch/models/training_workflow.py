"""Apply labels and retrain a detector's MLP.

Provides :func:`apply_and_retrain` which resolves new label entries against
the loaded dataset, applies them, and retrains the detector model with
cross-validated threshold.
"""

from __future__ import annotations


def apply_and_retrain(
    detector_id: str,
    det_ctx: object,
    new_entries: list[dict],
    detector_name: str,
) -> tuple[int, bool]:
    """Resolve new label entries into a loaded detector and retrain its MLP.

    Temporarily overrides the request-scoped detector context so vote proxies
    resolve to *det_ctx* for the duration of this call, then restores the
    previous context.

    Returns ``(resolved_count, trained_bool)``.
    """
    from flask import g

    from vtsearch.models.label_sync import sync_labels_to_loaded_detector
    from vtsearch.utils import (
        apply_label,
        build_media_lookup,
        resolve_media_ids,
        snapshot_medias,
    )

    # Override the request-scoped detector context so vote proxies resolve to
    # this detector's context for the duration of this call.
    prev_det_ctx = getattr(g, "_detector_context", None)

    try:
        g._detector_context = det_ctx

        snap = snapshot_medias()
        if not snap:
            return 0, False

        origin_lookup, md5_lookup, name_lookup = build_media_lookup(snap)

        resolved = 0
        for entry in new_entries:
            label = entry.get("label", "")
            if label not in ("good", "bad"):
                continue
            cids = resolve_media_ids(entry, origin_lookup, md5_lookup, name_lookup)
            for cid in cids:
                apply_label(cid, label)
            if cids:
                resolved += 1

        # Persist the updated votes back to the detector file so the
        # labelset reflects any newly-resolved medias.
        sync_labels_to_loaded_detector()

        # Retrain MLP if we have at least one good and one bad vote.
        from vtsearch.utils import bad_votes, good_votes, vote_region_boxes

        trained = False
        if good_votes and bad_votes:
            from vtsearch.models.training import train_and_score
            from vtsearch.utils import (
                get_calibrate_count,
                get_calibration_fraction,
                get_inclusion,
                get_safe_thresholds,
            )

            _, threshold, model = train_and_score(
                snap,
                dict(good_votes),
                dict(bad_votes),
                get_inclusion(),
                safe_thresholds=get_safe_thresholds(),
                calibrate_count=get_calibrate_count(),
                calibration_fraction=get_calibration_fraction(),
                vote_region_boxes=dict(vote_region_boxes),
                det_ctx=det_ctx,
            )
            if model is not None:
                det_ctx.model = model
                det_ctx.threshold = threshold
                # Cache voted media items with embeddings.
                training = {}
                for cid in list(good_votes) + list(bad_votes):
                    if cid in snap:
                        training[cid] = snap[cid]
                det_ctx.training_medias = training
                if snap:
                    first = next(iter(snap.values()), {})
                    det_ctx.embedder = first.get("embedder", "")
                    det_ctx.media_type = first.get("type", "")
                trained = True

        return resolved, trained

    finally:
        g._detector_context = prev_det_ctx
