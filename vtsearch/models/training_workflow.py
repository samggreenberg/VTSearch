"""Apply labels and retrain a detector's MLP.

Provides :func:`apply_and_retrain` which resolves new label entries against
the loaded dataset, applies them, and retrains the detector model with
cross-validated threshold.
"""

from __future__ import annotations


def apply_and_retrain(
    model_id: str,
    det_ctx: object,
    new_entries: list[dict],
    tm_name: str,
) -> tuple[int, bool]:
    """Resolve new label entries into a loaded detector and retrain its MLP.

    Temporarily switches the active detector context to *model_id*, resolves
    the entries against the loaded dataset's medias, applies matching labels,
    retrains the MLP with cross-validated threshold, then restores the
    previously active context.

    Returns ``(resolved_count, trained_bool)``.
    """
    from vtsearch.routes.trainable_models import sync_labels_to_loaded_model
    from vtsearch.utils import (
        apply_label,
        build_media_lookup,
        resolve_media_ids,
        snapshot_medias,
    )
    from vtsearch.utils.state_core import (
        get_active_detector_id,
        set_active_detector_id,
    )

    # Save the current active context so we can restore it afterwards.
    prev_active_id = get_active_detector_id()

    try:
        set_active_detector_id(model_id)

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

        # Persist the updated votes back to the trainable-model file so the
        # labelset reflects any newly-resolved medias.
        sync_labels_to_loaded_model()

        # Retrain MLP if we have at least one good and one bad vote.
        from vtsearch.utils import bad_votes, good_votes

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
        set_active_detector_id(prev_active_id)
