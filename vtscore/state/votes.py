"""Vote management, label history, text-sort suggestions, and learned scores.

Also contains compound vote operations (toggle_vote, apply_label) that
coordinate across vote dicts, click tracking, and the diversity tree.

All functions operate on the *active* :class:`DetectorContext` (and the
active :class:`DatasetContext` for the diversity tree side-effects).  They
resolve the context themselves via :func:`get_active_detector_context` /
:func:`get_active_context` — no module-level proxy names are imported, so
the library has no implicit dependency on the app-side proxy view.  See
Phase 3 of ``../docs/architecture.md``.
"""

from __future__ import annotations

from vtscore.state.clicks import assign_click_time, remove_click_time
from vtscore.state.core import (
    _state_lock,
    get_active_context,
    get_active_detector_context,
)
from vtscore.state.diversity import diversity_tree_label, diversity_tree_unlabel


def clear_votes() -> None:
    """Clear all votes and the full label history.

    Removes all entries from ``good_votes``, ``bad_votes``, and
    ``label_history`` in place on the active detector context. Does not affect
    any dataset's ``medias`` dict.  Also clears the progress model cache and
    click-time / score tracking.
    """
    from vtscore.detectors.labeling_progress import clear_progress_cache

    with _state_lock:
        ctx = get_active_detector_context()
        ctx.good_votes.clear()
        ctx.bad_votes.clear()
        ctx.label_history.clear()
        ctx.textsort_suggestions.clear()
        ctx.vote_click_times.clear()
        ctx.vote_region_boxes.clear()
        ctx.click_counter = 0
        ctx.last_learned_scores.clear()
        ctx.find_initial_labels.clear()
        clear_progress_cache()


def add_label_to_history(media_id: int, label: str) -> None:
    """Append a labelling event to the active detector's label history.

    Args:
        media_id: Integer ID of the media that was labelled.
        label: The assigned label; should be ``"good"`` or ``"bad"``.
    """
    import time

    with _state_lock:
        get_active_detector_context().label_history.append((media_id, label, time.time()))


def add_textsort_suggestion(text: str) -> None:
    """Record a text-sort query as a suggested detector/labelset name.

    Duplicates are moved to the end so the most-recently-voted query is last.

    Args:
        text: The text-sort query string to store.
    """
    with _state_lock:
        suggestions = get_active_detector_context().textsort_suggestions
        try:
            suggestions.remove(text)
        except ValueError:
            pass
        suggestions.append(text)


def get_textsort_suggestions() -> list[str]:
    """Return stored text-sort suggestions, most recent last."""
    with _state_lock:
        return list(get_active_detector_context().textsort_suggestions)


def update_learned_scores(scores: dict[int, float]) -> None:
    """Replace the stored learned-sort scores with *scores*."""
    with _state_lock:
        last_learned_scores = get_active_detector_context().last_learned_scores
        last_learned_scores.clear()
        last_learned_scores.update(scores)


def get_learned_scores() -> dict[int, float]:
    """Return a copy of the last learned-sort scores."""
    with _state_lock:
        return get_active_detector_context().last_learned_scores.copy()


def set_find_initial_labels(labels: dict[int, str]) -> None:
    """Store the detector-assigned labels from a find-label run.

    This snapshot is used to identify "corrections" -- items where the
    user subsequently changed the label from what the detector assigned.
    """
    with _state_lock:
        ctx = get_active_detector_context()
        ctx.find_initial_labels.clear()
        ctx.find_initial_labels.update(labels)


def get_find_initial_labels() -> dict[int, str]:
    """Return a copy of the find-label initial labels."""
    with _state_lock:
        return get_active_detector_context().find_initial_labels.copy()


# ---------------------------------------------------------------------------
# Compound operations (atomic vote toggle / label apply)
# ---------------------------------------------------------------------------


def toggle_vote(
    media_id: int,
    vote: str,
    region_box: tuple[float, float, float, float] | None = None,
) -> None:
    """Atomically toggle a good/bad vote for a media item.

    Implements the same toggle semantics as the ``/api/medias/<id>/vote``
    endpoint: if the media already has the requested vote it is removed
    (unlabelled); otherwise the vote is applied (overriding any existing
    opposite vote).

    When a vote switches polarity (good->bad or bad->good), the progress
    cache is partially invalidated: only cached steps from the point where
    the media first appeared in the training data are discarded.  Earlier
    steps (whose models never included this media) are preserved.

    This function acquires ``_state_lock`` so that the entire check-then-modify
    sequence is atomic with respect to concurrent requests.

    Args:
        media_id: Integer ID of the media to vote on.
        vote: ``"good"`` or ``"bad"``.
        region_box: Optional normalised ``(x0, y0, x1, y1)`` box that the
            user drew as part of a yes-vote.  Only honoured when *vote* is
            ``"good"`` and the vote is being *added* (not toggled off);
            stored in :attr:`DetectorContext.vote_region_boxes` so label
            export / detector sync emit it on the resulting LabeledElement.
            Ignored for no-votes (which are always image-level — see the
            patch-embedder v2 design).  Patch-embedder v2.
    """
    from vtscore.detectors.labeling_progress import invalidate_progress_cache_from

    added = False
    with _state_lock:
        ctx = get_active_detector_context()
        good_votes = ctx.good_votes
        bad_votes = ctx.bad_votes
        vote_region_boxes = ctx.vote_region_boxes
        if vote == "good":
            if media_id in good_votes:
                good_votes.pop(media_id, None)
                vote_region_boxes.pop(media_id, None)
                remove_click_time(media_id)
                add_label_to_history(media_id, "unlabel")
                if media_id not in bad_votes:
                    diversity_tree_unlabel(media_id)
            else:
                was_opposite = media_id in bad_votes
                bad_votes.pop(media_id, None)
                good_votes[media_id] = None
                if region_box is not None:
                    vote_region_boxes[media_id] = region_box
                else:
                    vote_region_boxes.pop(media_id, None)
                assign_click_time(media_id)
                add_label_to_history(media_id, "good")
                diversity_tree_label(media_id)
                if was_opposite:
                    invalidate_progress_cache_from(media_id)
                added = True
        else:
            if media_id in bad_votes:
                bad_votes.pop(media_id, None)
                remove_click_time(media_id)
                add_label_to_history(media_id, "unlabel")
                if media_id not in good_votes:
                    diversity_tree_unlabel(media_id)
            else:
                was_opposite = media_id in good_votes
                good_votes.pop(media_id, None)
                vote_region_boxes.pop(media_id, None)
                bad_votes[media_id] = None
                assign_click_time(media_id)
                add_label_to_history(media_id, "bad")
                diversity_tree_label(media_id)
                if was_opposite:
                    invalidate_progress_cache_from(media_id)
                added = True

    if added:
        from vtsearch.achievements import record_vote

        det_ctx = get_active_detector_context()
        record_vote(det_ctx.detector_id, media_type=det_ctx.media_type)


def apply_label(
    media_id: int,
    label: str,
    *,
    silent: bool = False,
    region_box: tuple[float, float, float, float] | None = None,
) -> None:
    """Atomically apply a label to a media (for imports).

    Unlike :func:`toggle_vote`, this always sets the label without toggling.
    No click-time is assigned (imported labels have no click-time).

    When *silent* is True, the label is recorded in ``good_votes``/``bad_votes``
    only — ``label_history`` is not appended and the diversity tree is not
    marked.  This is used when restoring a detector's saved labels into a new
    dataset: those labels are seeded so autopilot's good/bad-count gates are
    satisfied, but they should not contaminate the per-session Smart/Stable
    trends or pre-fill diversity coverage in the new dataset.

    Args:
        media_id: Integer ID of the media to label.
        label: ``"good"`` or ``"bad"``.
        silent: If True, skip history append and diversity-tree marking.
        region_box: Optional normalised ``(x0, y0, x1, y1)`` box from an
            imported labelset entry.  Only stored when *label* is ``"good"``
            (no-votes are always image-level).  Patch-embedder v2.
    """
    with _state_lock:
        ctx = get_active_detector_context()
        if label == "good":
            ctx.bad_votes.pop(media_id, None)
            ctx.good_votes[media_id] = None
            if region_box is not None:
                ctx.vote_region_boxes[media_id] = region_box
            else:
                ctx.vote_region_boxes.pop(media_id, None)
            if not silent:
                add_label_to_history(media_id, "good")
        else:
            ctx.good_votes.pop(media_id, None)
            ctx.vote_region_boxes.pop(media_id, None)
            ctx.bad_votes[media_id] = None
            if not silent:
                add_label_to_history(media_id, "bad")
        if not silent:
            diversity_tree_label(media_id)


def apply_label_with_click_time(media_id: int, label: str) -> None:
    """Atomically apply a label with click-time assignment (for fill-from-sort).

    Same as :func:`apply_label` but also assigns a click-time ordinal so the
    label appears in the frontend's click-time timeline.

    Args:
        media_id: Integer ID of the media to label.
        label: ``"good"`` or ``"bad"``.
    """
    with _state_lock:
        ctx = get_active_detector_context()
        if label == "good":
            ctx.bad_votes.pop(media_id, None)
            ctx.good_votes[media_id] = None
            ctx.vote_region_boxes.pop(media_id, None)
            add_label_to_history(media_id, "good")
        else:
            ctx.good_votes.pop(media_id, None)
            ctx.vote_region_boxes.pop(media_id, None)
            ctx.bad_votes[media_id] = None
            add_label_to_history(media_id, "bad")
        assign_click_time(media_id)
        diversity_tree_label(media_id)


def apply_labels_bulk_with_click_time(labels: list[tuple[int, str]], replace_all: bool = False) -> None:
    """Apply many labels in a single lock acquisition (for find-label scoring).

    Each entry is ``(media_id, label)`` where *label* is ``"good"`` or
    ``"bad"``.  All labels are applied atomically with click-time ordinals
    assigned in order.

    When *replace_all* is True, any pre-existing votes/click-times for IDs
    outside *labels* are cleared first.  This is what ``/api/find-label``
    wants: a detector trained on Dataset A holds Dataset A's media IDs in
    its DetectorContext, and switching to Dataset B must not leak those
    stale IDs into Dataset B's right-scroll Goods/Bads.
    """
    import time as _time

    with _state_lock:
        ctx = get_active_detector_context()
        good_votes = ctx.good_votes
        bad_votes = ctx.bad_votes
        vote_click_times = ctx.vote_click_times
        vote_region_boxes = ctx.vote_region_boxes
        label_history = ctx.label_history
        tree = get_active_context().diversity_tree
        if replace_all:
            kept = {mid for mid, _ in labels}
            for cid in [c for c in good_votes if c not in kept]:
                good_votes.pop(cid, None)
            for cid in [c for c in bad_votes if c not in kept]:
                bad_votes.pop(cid, None)
            for cid in [c for c in vote_click_times if c not in kept]:
                vote_click_times.pop(cid, None)
            for cid in [c for c in vote_region_boxes if c not in kept]:
                vote_region_boxes.pop(cid, None)
            ctx.find_initial_labels.clear()
        for media_id, label in labels:
            if label == "good":
                bad_votes.pop(media_id, None)
                good_votes[media_id] = None
                vote_region_boxes.pop(media_id, None)
                label_history.append((media_id, "good", _time.time()))
            else:
                good_votes.pop(media_id, None)
                vote_region_boxes.pop(media_id, None)
                bad_votes[media_id] = None
                label_history.append((media_id, "bad", _time.time()))
            ctx.click_counter += 1
            vote_click_times[media_id] = ctx.click_counter
            if tree is not None and media_id in tree.vector_to_leaf:
                tree.label(media_id)
