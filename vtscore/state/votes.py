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


def _record_vote_locked() -> None:
    """Credit one vote to the active detector's achievements.

    Must be called while ``_state_lock`` is held so the detector context
    cannot change between the vote landing in state and the achievement
    being credited (otherwise a concurrent context switch would credit the
    wrong detector).  Establishes the lock order ``_state_lock → _settings_lock``;
    no code path takes the locks in the reverse order.
    """
    from vtsearch.achievements import record_vote  # noqa: PLC0415

    det_ctx = get_active_detector_context()
    record_vote(det_ctx.detector_id, media_type=det_ctx.media_type)


def _current_label_locked(ctx, media_id: int) -> str:
    """Return ``"good"`` / ``"bad"`` / ``"none"`` for *media_id* (lock held)."""
    if media_id in ctx.good_votes:
        return "good"
    if media_id in ctx.bad_votes:
        return "bad"
    return "none"


def _set_vote_locked(
    media_id: int,
    target: str,
    region_box: tuple[float, float, float, float] | None = None,
) -> tuple[str, str, int | None]:
    """Set *media_id*'s vote to *target* under an already-held ``_state_lock``.

    Returns ``(old_label, new_label, click_time)`` where ``click_time`` is the
    ordinal assigned to a newly-labeled media (or ``None`` when *target* is
    ``"none"`` or the call was idempotent).
    """
    from vtscore.detectors.labeling_progress import invalidate_progress_cache_from

    if target not in ("good", "bad", "none"):
        raise ValueError(f"target must be 'good', 'bad', or 'none' (got {target!r})")

    ctx = get_active_detector_context()
    old = _current_label_locked(ctx, media_id)
    if old == target:
        # Idempotent — preserve everything (no history append, no cache churn,
        # no achievement credit).  This is the key change that closes the H1
        # counter-inflation race: concurrent tabs sending the same target on a
        # media that's already in that state no longer increment counters.
        existing_click = ctx.vote_click_times.get(media_id) if target != "none" else None
        return (old, old, existing_click)

    if target == "good":
        ctx.bad_votes.pop(media_id, None)
        ctx.good_votes[media_id] = None
        if region_box is not None:
            ctx.vote_region_boxes[media_id] = region_box
        else:
            ctx.vote_region_boxes.pop(media_id, None)
        click_time = assign_click_time(media_id)
        add_label_to_history(media_id, "good")
        diversity_tree_label(media_id)
    elif target == "bad":
        ctx.good_votes.pop(media_id, None)
        ctx.vote_region_boxes.pop(media_id, None)
        ctx.bad_votes[media_id] = None
        click_time = assign_click_time(media_id)
        add_label_to_history(media_id, "bad")
        diversity_tree_label(media_id)
    else:  # target == "none"
        ctx.good_votes.pop(media_id, None)
        ctx.bad_votes.pop(media_id, None)
        ctx.vote_region_boxes.pop(media_id, None)
        remove_click_time(media_id)
        add_label_to_history(media_id, "unlabel")
        diversity_tree_unlabel(media_id)
        click_time = None

    # Invalidate the progress cache on ANY training-set membership change for
    # this media.  The old `toggle_vote` only invalidated on polarity flips,
    # leaving cached steps stale on good→none / bad→none.  Cached models that
    # included this media are now wrong regardless of whether the new state is
    # the opposite polarity or "none", so the rule is: invalidate iff the media
    # was previously labeled.
    if old != "none":
        invalidate_progress_cache_from(media_id)

    # Counter increments only on a transition that produces a labeled state.
    # Un-vote (X→none) and idempotent re-apply (handled above) credit nothing.
    if target != "none":
        _record_vote_locked()

    return (old, target, click_time)


def set_vote(
    media_id: int,
    target: str,
    region_box: tuple[float, float, float, float] | None = None,
) -> tuple[str, str, int | None]:
    """Atomically set a media's vote to an absolute target state.

    *target* is one of ``"good"``, ``"bad"``, or ``"none"``.  Behaviour is
    **idempotent**: setting the target equal to the current state is a no-op
    that does not append to ``label_history``, does not increment achievement
    counters, and does not assign a new click-time.  Achievement credit fires
    exactly when the media moves from one state to a *different* labeled
    state (none→good, none→bad, good→bad, bad→good); un-vote and idempotent
    re-apply credit nothing.  The progress cache is invalidated for *media_id*
    whenever its prior training-set membership changes (i.e. when ``old`` was
    ``"good"`` or ``"bad"``).

    Concurrent rapid-toggle from multiple tabs no longer inflates counters,
    because each client sends an absolute target rather than a "toggle"
    intent — stale-view duplicates collapse into idempotent no-ops on the
    server (logical-bug-audit H1).

    Args:
        media_id: Integer ID of the media to vote on.
        target: ``"good"``, ``"bad"``, or ``"none"``.
        region_box: Optional normalised ``(x0, y0, x1, y1)`` good-vote region.
            Only honoured when *target* is ``"good"`` (patch-embedder v2:
            no-votes are always image-level).

    Returns:
        ``(old_label, new_label, click_time)``.  ``click_time`` is the
        ordinal assigned to the new label, or ``None`` for ``"none"`` /
        idempotent calls.
    """
    with _state_lock:
        return _set_vote_locked(media_id, target, region_box=region_box)


def toggle_vote(
    media_id: int,
    vote: str,
    region_box: tuple[float, float, float, float] | None = None,
) -> None:
    """Toggle a good/bad vote, computing the target from the current state.

    Implemented as a thin wrapper over :func:`set_vote` so all the
    correctness rules (idempotent semantics, cache invalidation on
    membership change, achievement-credit gating) are shared.  Kept for
    in-process callers that want the old "same vote toggles off" affordance
    (the HTTP ``/api/medias/<id>/vote`` endpoint no longer uses it — see
    :func:`set_vote`).

    Args:
        media_id: Integer ID of the media to vote on.
        vote: ``"good"`` or ``"bad"`` — the clicked direction.  Toggles off
            when *media_id* is already in that polarity, otherwise sets to
            *vote* (overriding any opposite vote).
        region_box: Optional normalised good-vote region; honoured only when
            the resulting target is ``"good"``.
    """
    with _state_lock:
        ctx = get_active_detector_context()
        current = _current_label_locked(ctx, media_id)
        if vote == "good":
            target = "none" if current == "good" else "good"
        elif vote == "bad":
            target = "none" if current == "bad" else "bad"
        else:
            raise ValueError(f"vote must be 'good' or 'bad' (got {vote!r})")
        _set_vote_locked(
            media_id,
            target,
            region_box=region_box if target == "good" else None,
        )


def apply_label(
    media_id: int,
    label: str,
    *,
    silent: bool = False,
    region_box: tuple[float, float, float, float] | None = None,
    record_achievement: bool = True,
) -> None:
    """Atomically apply a label to a media (for imports).

    Unlike :func:`toggle_vote`, this always sets the label without toggling.
    No click-time is assigned (imported labels have no click-time).

    When *silent* is True, the label is recorded in ``good_votes``/``bad_votes``
    only — ``label_history`` is not appended, the diversity tree is not
    marked, and achievement counters are not credited.  This is used when
    restoring a detector's saved labels into a new dataset: those labels are
    seeded so autopilot's good/bad-count gates are satisfied, but they should
    not contaminate the per-session Smart/Stable trends or pre-fill diversity
    coverage in the new dataset.

    Args:
        media_id: Integer ID of the media to label.
        label: ``"good"`` or ``"bad"``.
        silent: If True, skip history append, diversity-tree marking, and
            achievement credit.
        region_box: Optional normalised ``(x0, y0, x1, y1)`` box from an
            imported labelset entry.  Only stored when *label* is ``"good"``
            (no-votes are always image-level).  Patch-embedder v2.
        record_achievement: When True (the default), credit one vote in the
            active detector's achievement counters.  Set to False for
            system-driven label application that isn't a user vote action
            (e.g. auto-import from a labelset source, example-media seeding
            on detector load).
    """
    with _state_lock:
        ctx = get_active_detector_context()
        if label == "good":
            already = media_id in ctx.good_votes
            ctx.bad_votes.pop(media_id, None)
            ctx.good_votes[media_id] = None
            if region_box is not None:
                ctx.vote_region_boxes[media_id] = region_box
            else:
                ctx.vote_region_boxes.pop(media_id, None)
            if not silent:
                add_label_to_history(media_id, "good")
        else:
            already = media_id in ctx.bad_votes
            ctx.good_votes.pop(media_id, None)
            ctx.vote_region_boxes.pop(media_id, None)
            ctx.bad_votes[media_id] = None
            if not silent:
                add_label_to_history(media_id, "bad")
        if not silent:
            diversity_tree_label(media_id)
            if record_achievement and not already:
                _record_vote_locked()


def apply_label_with_click_time(media_id: int, label: str) -> None:
    """Atomically apply a label with click-time assignment (for fill-from-sort).

    Same as :func:`apply_label` but also assigns a click-time ordinal so the
    label appears in the frontend's click-time timeline.  Credits one vote in
    the active detector's achievement counters — bulk fill-from-sort flows
    are user vote actions and must show up in ``votes_cast`` etc. (audit
    finding C8).

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
        _record_vote_locked()


def apply_labels_bulk_with_click_time(labels: list[tuple[int, str]], replace_all: bool = False) -> None:
    """Apply many labels in a single lock acquisition (for find-label scoring).

    Each entry is ``(media_id, label)`` where *label* is ``"good"`` or
    ``"bad"``.  All labels are applied atomically with click-time ordinals
    assigned in order.  Each entry credits one vote in the active detector's
    achievement counters — bulk find-label flows are user vote actions and
    must show up in ``votes_cast`` etc. (audit finding C8).

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
                already = media_id in good_votes
                bad_votes.pop(media_id, None)
                good_votes[media_id] = None
                vote_region_boxes.pop(media_id, None)
                label_history.append((media_id, "good", _time.time()))
            else:
                already = media_id in bad_votes
                good_votes.pop(media_id, None)
                vote_region_boxes.pop(media_id, None)
                bad_votes[media_id] = None
                label_history.append((media_id, "bad", _time.time()))
            ctx.click_counter += 1
            vote_click_times[media_id] = ctx.click_counter
            if tree is not None and media_id in tree.vector_to_leaf:
                tree.label(media_id)
            if not already:
                _record_vote_locked()
