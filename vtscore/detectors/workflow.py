"""Apply labels and retrain a detector's MLP.

Provides :func:`apply_and_retrain` which resolves new label entries against
the loaded dataset, applies them, and retrains the detector model with
cross-validated threshold.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vtscore.state.core import DetectorContext


def apply_and_retrain(  # noqa: C901
    detector_id: str,
    det_ctx: "DetectorContext",
    new_entries: list[dict],
    detector_name: str,
) -> tuple[int, bool]:
    """Resolve new label entries into a loaded detector and retrain its MLP.

    Temporarily overrides the active detector context so vote proxies resolve
    to *det_ctx* for the duration of this call (via
    :func:`vtscore.state.core.override_detector_context`), regardless of
    whether the caller is inside a Flask request or a background thread.

    Train-first ordering: the candidate vote set (existing votes + newly
    resolved entries) is fed to :func:`train_and_score` *before* any vote is
    written to ``det_ctx`` or persisted to disk.  If training raises, the
    detector's in-memory votes, persisted labelset, and active model are all
    left untouched — so a failed retrain can never leave a vote live with a
    stale model behind it (audit finding H7).

    Returns ``(resolved_count, trained_bool)``.
    """
    from vtscore.detectors.label_sync import sync_labels_to_loaded_detector
    from vtsearch.state import (
        apply_label,
        bad_votes,
        build_media_lookup,
        good_votes,
        resolve_media_ids,
        snapshot_medias,
        vote_region_boxes,
    )
    from vtscore.state.core import override_detector_context

    with override_detector_context(det_ctx):
        snap = snapshot_medias()
        if not snap:
            return 0, False

        origin_lookup, md5_lookup, name_lookup = build_media_lookup(snap)

        # 1) Resolve every entry into concrete (cid, label) pairs without
        #    touching any state yet.
        resolved_pairs: list[tuple[int, str]] = []
        resolved = 0
        for entry in new_entries:
            label = entry.get("label", "")
            if label not in ("good", "bad"):
                continue
            cids = resolve_media_ids(entry, origin_lookup, md5_lookup, name_lookup)
            for cid in cids:
                resolved_pairs.append((cid, label))
            if cids:
                resolved += 1

        # 2) Build the *proposed* vote dicts (current + new resolutions)
        #    for a dry-run training pass.  Same-side dedup is automatic
        #    via dict; new opposite-side labels supersede the old ones.
        proposed_good = dict(good_votes)
        proposed_bad = dict(bad_votes)
        for cid, label in resolved_pairs:
            if label == "good":
                proposed_bad.pop(cid, None)
                proposed_good[cid] = None
            else:
                proposed_good.pop(cid, None)
                proposed_bad[cid] = None

        # 3) Try retraining on the proposed votes first.  If this raises,
        #    nothing has been mutated yet — the detector stays in its
        #    prior consistent state and the exception propagates.
        new_model = None
        new_threshold = 0.5
        if proposed_good and proposed_bad:
            from vtscore.detectors.training import train_and_score
            from vtsearch.state import (
                get_calibrate_count,
                get_calibration_fraction,
                get_inclusion,
                get_safe_thresholds,
            )

            _, new_threshold, new_model = train_and_score(
                snap,
                proposed_good,
                proposed_bad,
                get_inclusion(),
                safe_thresholds=get_safe_thresholds(),
                calibrate_count=get_calibrate_count(),
                calibration_fraction=get_calibration_fraction(),
                vote_region_boxes=dict(vote_region_boxes),
                det_ctx=det_ctx,
            )

        # 4) Training succeeded (or was skipped because we don't yet
        #    have both classes).  Commit the votes and persist.
        for cid, label in resolved_pairs:
            apply_label(cid, label)

        sync_labels_to_loaded_detector()

        trained = False
        if new_model is not None:
            det_ctx.model = new_model
            det_ctx.threshold = new_threshold
            training = {}
            for cid in list(good_votes) + list(bad_votes):
                if cid in snap:
                    training[cid] = snap[cid]
            det_ctx.training_medias = training
            first = next(iter(snap.values()), {})
            det_ctx.embedder = first.get("embedder", "")
            det_ctx.media_type = first.get("type", "")
            trained = True

        return resolved, trained
