"""Resolve a detector's scoring model, training it on demand if needed.

:func:`resolve_or_train_detector` is the cold-path counterpart to the
``DetectorContext.model`` fast path: given a detector id and (optionally) its
on-disk data plus a media snapshot, it returns the MLP + threshold to score
with, training from the detector's labelset when no live model exists.

This logic lived inline in ``vtsearch/routes/detectors/scoring.py`` until it
grew its own resolution/embedding/training branches; it has no Flask or
request-context dependency, so it belongs in the library tier where it can be
exercised directly (see docs/plans/code-structure-review.md, Theme A).
"""

from __future__ import annotations

from typing import Any

from vtscore.concurrency.progress import update_find_progress


def resolve_or_train_detector(  # noqa: C901
    detector_id: str,
    det_data: dict | None,
    media_type: str,
    snap: dict | None,
    *,
    progress_step: int = 2,
    progress_total_steps: int = 4,
) -> tuple[Any | None, float, dict | None]:
    """Return (mlp, threshold, diagnostic) for *detector_id*.

    Tries the loaded :class:`~vtscore.state.core.DetectorContext` first.  Falls
    back to training on demand from the detector's labelset, embedding label
    media via its origin importer.  Returns ``(None, _, diag)`` when training
    is not possible.

    Inclusion changes don't need special handling here: ``set_inclusion``
    invalidates every loaded context's cached MLP, so after a slider move the
    ``det_ctx.model`` short-circuit below is skipped and the cold branch
    retrains at the new (per-detector) inclusion.
    """
    from vtscore.detectors.dataset_sync import invalidate_detector_model_on_embedder_mismatch
    from vtscore.state.core import get_detector_context

    det_ctx = get_detector_context(detector_id)
    if det_ctx is not None:
        # Defense against H5: scoring Auto-Find detectors iterates contexts
        # that aren't the active one, so the before_request hook can't
        # have invalidated their stale MLPs.  Drop them here so the next
        # branch trains fresh against *snap*'s embedder.
        snap_embedder = next(iter(snap.values()), {}).get("embedder", "") or "" if snap else ""
        invalidate_detector_model_on_embedder_mismatch(det_ctx, snap_embedder)
    if det_ctx is not None and det_ctx.model is not None:
        return det_ctx.model, det_ctx.threshold, None

    if det_data is None:
        return None, 0.5, None

    label_entries = det_data.get("labelset", {}).get("labels", [])
    if not label_entries:
        return None, 0.5, None

    update_find_progress(
        "running",
        "Training detector from labels…",
        current=0,
        total=0,
        step=progress_step,
        total_steps=progress_total_steps,
    )

    from vtscore.detectors.resolver import resolve_label_embeddings
    from vtscore.detectors.training import train_and_threshold

    X_list: list = []
    y_list: list[float] = []
    md5_to_emb = {}
    if snap:
        md5_to_emb = {c["md5"]: c["embedding"] for c in snap.values()}

    # Match origin-resolved label vectors to the snap's embedder space so
    # the two paths don't produce a mixed-space training set (silently
    # garbage MLP).  Empty when the snap is empty or untyped, which falls
    # back to the media type's default embedder.
    dataset_embedder = ""
    if snap:
        dataset_embedder = next(iter(snap.values()), {}).get("embedder", "") or ""

    unresolved: list[dict] = []
    for entry in label_entries:
        label_val = entry.get("label", "")
        if label_val not in ("good", "bad"):
            continue
        md5 = entry.get("md5", "")
        if md5 and md5 in md5_to_emb:
            X_list.append(md5_to_emb[md5])
            y_list.append(1.0 if label_val == "good" else 0.0)
        else:
            unresolved.append(entry)

    md5_matched = len(X_list)
    resolved = None
    if unresolved:
        n_unresolved = len(unresolved)
        update_find_progress(
            "running",
            f"Resolving {n_unresolved} label origins…",
            current=0,
            total=n_unresolved,
            step=progress_step,
            total_steps=progress_total_steps,
        )

        def _origin_progress(current: int, total: int) -> None:
            update_find_progress(
                "running",
                f"Resolving {n_unresolved} label origins…",
                current=current,
                total=total,
                step=progress_step,
                total_steps=progress_total_steps,
            )

        resolved = resolve_label_embeddings(
            unresolved,
            media_type,
            progress_callback=_origin_progress,
            embedder_name=dataset_embedder,
        )
        X_list.extend(resolved.embeddings)
        y_list.extend(resolved.labels)

    has_good = any(v == 1.0 for v in y_list)
    has_bad = any(v == 0.0 for v in y_list)
    if has_good and has_bad:
        update_find_progress(
            "running",
            "Cross-calibrating threshold…",
            current=0,
            total=0,
            step=progress_step,
            total_steps=progress_total_steps,
        )
        trained_mlp, threshold = train_and_threshold(X_list, y_list, snap=snap)
        return trained_mlp, threshold, None

    diagnostic: dict = {
        "total_labels": md5_matched + len(unresolved),
        "md5_matched": md5_matched,
        "needed_resolution": len(unresolved),
        "resolved_from_origin": resolved.resolved_count if resolved else 0,
        "failed_resolution": len(resolved.missing_entries) if resolved else len(unresolved),
        "has_good": has_good,
        "has_bad": has_bad,
        "media_type": media_type,
    }
    if resolved and resolved.missing_entries:
        samples = resolved.missing_entries[:3]
        diagnostic["sample_failures"] = [
            {
                "origin": e.get("origin"),
                "origin_name": e.get("origin_name", ""),
                "filename": e.get("filename", ""),
                "md5": e.get("md5", "")[:12],
                "label": e.get("label", ""),
            }
            for e in samples
        ]
    elif not unresolved and (not has_good or not has_bad):
        diagnostic["hint"] = "All labels matched by MD5 but all are the same class (need both good and bad)"

    return None, 0.5, diagnostic
