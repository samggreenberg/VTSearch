"""Vote management, label history, text-sort suggestions, and learned scores.

Also contains compound vote operations (toggle_vote, apply_label) that
coordinate across vote dicts, click tracking, and the coverage atlas.

All functions operate on the *active* :class:`DetectorContext` (and the
active :class:`DatasetContext` for the coverage atlas side-effects).  They
resolve the context themselves via :func:`get_active_detector_context` /
:func:`get_active_context` - no module-level proxy names are imported, so
the library has no implicit dependency on the app-side proxy view.  See
Phase 3 of ``../docs/architecture.md``.
"""

from __future__ import annotations

from typing import Any

from vtscore.state.clicks import assign_click_time, remove_click_time
from vtscore.state.core import (
    _state_lock,
    get_active_context,
    get_active_detector_context,
)
from vtscore.state.coverage import coverage_atlas_label, coverage_atlas_unlabel


def clear_votes() -> None:
    """Clear all votes and the full label history.

    Removes all entries from ``good_votes``, ``bad_votes``, and
    ``label_history`` in place on the active detector context. Does not affect
    any dataset's ``medias`` dict.  Also clears the progress model cache,
    click-time / score tracking, and the active dataset's coverage atlas
    evidence state - otherwise ``coverage_atlas_next_sample`` would keep
    skipping nodes that the just-cleared votes had marked covered, and the
    UI's diversity-level chip would stay elevated despite zero labels.
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
        ctx.verified_ids.clear()
        ctx.find_scores.clear()
        ctx.find_eval_stale = False
        # Everything that *made* this a Find session is gone, so the flag that
        # says "these votes are scoring output" must go too.  Leaving it set
        # would keep suppressing labelset write-back
        # (``label_sync.is_find_mode``) and keep the vote rehydrate's
        # find-session guard engaged (``dataset_sync``) for votes that are now
        # ordinary training labels.
        ctx.find_mode = False
        ds_atlas = get_active_context().coverage_atlas
        if ds_atlas is not None:
            ds_atlas.reset_labeled()
    # ``_progress_lock`` is acquired strictly outside ``_state_lock`` so the
    # two locks never establish a cross-module ordering (audit M1).
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


def set_find_scores(scores: dict[int, float]) -> None:
    """Store the frozen per-item detector scores from a find-label run.

    These are the single-pass scores the cutoff (Inclusion) slides over
    without re-scoring, and the basis for the Stats FP/FN sweep.  In-memory
    only.  See docs/plans/find-verification-workflow.md.
    """
    with _state_lock:
        ctx = get_active_detector_context()
        ctx.find_scores.clear()
        ctx.find_scores.update(scores)


def get_find_scores() -> dict[int, float]:
    """Return a copy of the frozen find-label scores."""
    with _state_lock:
        return dict(get_active_detector_context().find_scores)


def rethreshold_unverified_find_items() -> None:
    """Re-split *unverified* Find items good/bad at the current cutoff.

    Inclusion is a pure cutoff knob: sliding it moves
    :attr:`DetectorContext.threshold` over the frozen ``find_scores`` with no
    re-scoring.  Every scored item the human has not verified is re-assigned
    good/bad purely by whether its frozen score clears the (already updated)
    threshold; verified items keep their human vote wherever their score
    lands, and ``find_initial_labels`` (the eval baseline at the default
    cutoff) is left untouched.  Existing click-times are preserved so the
    right-scroll ordering doesn't churn on a slide.

    No-op outside Find mode or before a scoring pass has frozen ``find_scores``
    (e.g. Train-mode inclusion changes).  Must run *after*
    :func:`recompute_detector_thresholds_for_inclusion` has updated the
    threshold.  See docs/plans/find-verification-workflow.md.
    """
    with _state_lock:
        ctx = get_active_detector_context()
        # The request-missing sentinel (no detector identified) exposes none of
        # these fields; guard with getattr so an Inclusion/settings write with
        # no detector context is a clean no-op rather than an AttributeError.
        if not getattr(ctx, "find_mode", False) or not getattr(ctx, "find_scores", None):
            return
        threshold = ctx.threshold
        good_votes = ctx.good_votes
        bad_votes = ctx.bad_votes
        verified = ctx.verified_ids
        for media_id, score in ctx.find_scores.items():
            if media_id in verified:
                continue
            if score >= threshold:
                if media_id in good_votes:
                    continue
                bad_votes.pop(media_id, None)
                good_votes[media_id] = None
            else:
                if media_id in bad_votes:
                    continue
                good_votes.pop(media_id, None)
                bad_votes[media_id] = None


def _find_ranked_scores(ctx) -> list[tuple[int, float]]:
    """``find_scores`` as ``(id, score)`` sorted descending by score, then id.

    ``find_scores`` is an insertion-ordered dict with no ranking meaning; the
    Find work queue and boundary walk both need the same descending-by-score
    order the frontend's ``sortOrder`` carries, so materialise it once.  The
    ``-score`` primary / ``id`` secondary key gives a stable order across ties.
    Caller must hold ``_state_lock``.
    """
    return sorted(ctx.find_scores.items(), key=lambda kv: (-kv[1], kv[0]))


def find_queue_ids(label_filter: str) -> list[int]:
    """Return the Find work-queue media ids for *label_filter*, ranking order.

    Replaces the frontend's client-side derivation (``find-view``'s
    ``unverifiedGoodIds`` / ``goodIds``) so a windowed client that no longer
    holds the whole ranking can still scope the bulk actions over the *entire*
    positive set:

    - ``"unverified_good"`` — above-cutoff items the human hasn't verified, in
      descending-score order.  The left work queue (Browse / To Dataset /
      Export).
    - ``"good"`` — the full positive set: verified-good items (pinned by the
      human, wherever their score lands) followed by the unverified positives.
      The right-panel actions.

    Derived from the frozen ``find_scores`` + current ``threshold`` +
    ``verified_ids``, so it tracks Inclusion slides exactly like the frontend
    derivation it replaces.  Returns ``[]`` outside Find mode, before a scoring
    pass has frozen ``find_scores``, or for an unknown *label_filter*.
    """
    with _state_lock:
        ctx = get_active_detector_context()
        if not getattr(ctx, "find_mode", False) or not getattr(ctx, "find_scores", None):
            return []
        threshold = ctx.threshold
        if threshold is None:
            return []
        verified = ctx.verified_ids
        good_votes = ctx.good_votes
        ranked = _find_ranked_scores(ctx)
        unverified_good = [mid for mid, score in ranked if score >= threshold and mid not in verified]
        if label_filter == "unverified_good":
            return unverified_good
        if label_filter == "good":
            verified_good = [mid for mid, _ in ranked if mid in verified and mid in good_votes]
            return verified_good + unverified_good
        return []


def find_boundary_next(side: str, exclude: int | None = None) -> dict:
    """Return the next unverified item to review on the Find boundary walk.

    Replaces the frontend's client-side ``advanceToBoundary``.  Walks the frozen
    ``find_scores`` (descending) for the unverified item closest to the cutoff on
    each side — the lowest-scored item still ``>= threshold`` (``"above"``) and
    the highest-scored item ``< threshold`` (``"below"``) — then prefers *side*,
    falling back to the other when the preferred side is exhausted.  "Just sit
    and vote" then alternates faces of the decision boundary until both are
    empty.

    *exclude* drops one id from consideration (the item just voted, in case its
    verified-state hasn't been observed yet).

    Returns ``{"id": <media_id>, "side": "above"|"below"}`` for the pick, or
    ``{"id": None, "side": None}`` when no unverified item remains on either side
    (the done state).
    """
    with _state_lock:
        ctx = get_active_detector_context()
        if not getattr(ctx, "find_mode", False) or not getattr(ctx, "find_scores", None):
            return {"id": None, "side": None}
        threshold = ctx.threshold
        if threshold is None:
            return {"id": None, "side": None}
        verified = ctx.verified_ids
        closest_above: int | None = None
        closest_below: int | None = None
        for mid, score in _find_ranked_scores(ctx):
            if mid in verified or mid == exclude:
                continue
            if score >= threshold:
                closest_above = mid  # descending → keep overwriting → lowest above
            elif closest_below is None:
                closest_below = mid  # first sub-threshold hit → highest below
        order = ("above", "below") if side == "above" else ("below", "above")
        picks = {"above": closest_above, "below": closest_below}
        for candidate_side in order:
            if picks[candidate_side] is not None:
                return {"id": picks[candidate_side], "side": candidate_side}
        return {"id": None, "side": None}


# ---------------------------------------------------------------------------
# Compound operations (atomic vote toggle / label apply)
# ---------------------------------------------------------------------------


def _record_vote_locked(count_streak: bool = True) -> None:
    """Credit one vote to the active detector's achievements.

    Must be called while ``_state_lock`` is held so the detector context
    cannot change between the vote landing in state and the achievement
    being credited (otherwise a concurrent context switch would credit the
    wrong detector).  Establishes the lock order ``_state_lock → _settings_lock``;
    no code path takes the locks in the reverse order.

    *count_streak* is forwarded to :func:`record_vote`: pass False for bulk /
    non-individual vote applications so they credit every other achievement but
    do not inflate the Marathoner ``vote_streak`` (which counts consecutive
    *individual* hand-clicks).
    """
    from vtsearch.achievements import record_vote  # noqa: PLC0415

    det_ctx = get_active_detector_context()
    record_vote(det_ctx.detector_id, media_type=det_ctx.media_type, count_streak=count_streak)


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
    *,
    count_streak: bool = True,
) -> tuple[str, str, int | None]:
    """Set *media_id*'s vote to *target* under an already-held ``_state_lock``.

    Returns ``(old_label, new_label, click_time)`` where ``click_time`` is the
    ordinal assigned to a newly-labeled media (or ``None`` when *target* is
    ``"none"`` or the call was idempotent).

    *count_streak* forwards to the achievement credit: pass False for bulk
    votes (a single batch must not build a Marathoner streak); the default True
    is the single hand-clicked vote.

    Does **not** acquire ``_progress_lock``.  The public wrappers
    (:func:`set_vote`, :func:`toggle_vote`) decide whether to invalidate the
    progress cache *after* releasing ``_state_lock`` based on the returned
    ``old_label`` - see the lock-ordering note on those wrappers (audit M1).
    """
    if target not in ("good", "bad", "none"):
        raise ValueError(f"target must be 'good', 'bad', or 'none' (got {target!r})")

    ctx = get_active_detector_context()
    old = _current_label_locked(ctx, media_id)
    if old == target:
        # Idempotent - no history append, no cache churn, no achievement
        # credit.  This is the key change that closes the H1 counter-inflation
        # race: concurrent tabs sending the same target on a media that's
        # already in that state no longer increment counters.  The one
        # exception is an explicit ``region_box`` on a ``"good"`` re-vote,
        # which lets the user replace the recorded annotation without first
        # un-voting (drawing a new box on an already-good media is an
        # intentional user action - the stale-tab race the idempotency rule
        # closes was about counters, not region updates).  An absent
        # ``region_box`` on an idempotent call leaves the existing one alone.
        if target == "good" and region_box is not None:
            ctx.vote_region_boxes[media_id] = region_box
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
        coverage_atlas_label(media_id, good=True)
    elif target == "bad":
        ctx.good_votes.pop(media_id, None)
        ctx.vote_region_boxes.pop(media_id, None)
        ctx.bad_votes[media_id] = None
        click_time = assign_click_time(media_id)
        add_label_to_history(media_id, "bad")
        coverage_atlas_label(media_id, good=False)
    else:  # target == "none"
        ctx.good_votes.pop(media_id, None)
        ctx.bad_votes.pop(media_id, None)
        ctx.vote_region_boxes.pop(media_id, None)
        remove_click_time(media_id)
        add_label_to_history(media_id, "unlabel")
        coverage_atlas_unlabel(media_id)
        click_time = None

    # Counter increments only on a transition that produces a labeled state.
    # Un-vote (X→none) and idempotent re-apply (handled above) credit nothing.
    if target != "none":
        _record_vote_locked(count_streak=count_streak)

    return (old, target, click_time)


def _needs_progress_invalidate(old_label: str, new_label: str) -> bool:
    """Whether an ``old_label → new_label`` transition requires a progress-cache invalidate.

    Cached models that included this media are stale on ANY training-set
    membership change for it.  The old ``toggle_vote`` only invalidated on
    polarity flips, which left the cache stale on good→none / bad→none.  The
    rule is now: invalidate iff the media was previously labeled *and* the
    new label is different (idempotent re-applies - including a re-vote with
    a new region box - do not invalidate, matching the pre-M1 behaviour
    where the idempotent path returned early before the invalidate site).
    """
    return old_label != "none" and old_label != new_label


def _mark_verified_if_find_mode(ctx: Any, media_id: int, new_label: str) -> None:
    """In Find mode, a manual single-item vote *verifies* the item.

    The big Good/Bad buttons and the hover-vote both land here (via the
    ``/api/medias/<id>/vote`` endpoint → :func:`set_vote`).  A good/bad vote
    moves the item out of the left work queue into the right verified pile;
    un-voting (``none``) returns it to unverified.  The bulk find-label scoring
    path does *not* go through here, so detector-assigned labels stay
    unverified by construction.  See docs/plans/find-verification-workflow.md.
    """
    if not getattr(ctx, "find_mode", False):
        return
    if new_label in ("good", "bad"):
        ctx.verified_ids[media_id] = None
    else:
        ctx.verified_ids.pop(media_id, None)


def set_vote(
    media_id: int,
    target: str,
    region_box: tuple[float, float, float, float] | None = None,
    *,
    count_streak: bool = True,
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
    intent - stale-view duplicates collapse into idempotent no-ops on the
    server (logical-bug-audit H1).

    Args:
        media_id: Integer ID of the media to vote on.
        target: ``"good"``, ``"bad"``, or ``"none"``.
        region_box: Optional normalised ``(x0, y0, x1, y1)`` good-vote region.
            Only honoured when *target* is ``"good"`` (patch-embedder v2:
            no-votes are always image-level).
        count_streak: Pass False for bulk applications (e.g. the
            ``/api/medias/vote-bulk`` "Verified Good/Bad" action over a
            hand-selected set) so the batch credits every other vote
            achievement but does not inflate the Marathoner streak, which only
            counts consecutive *individual* hand-clicks.  Defaults to True (the
            single-item vote path).

    Returns:
        ``(old_label, new_label, click_time)``.  ``click_time`` is the
        ordinal assigned to the new label, or ``None`` for ``"none"`` /
        idempotent calls.
    """
    with _state_lock:
        result = _set_vote_locked(media_id, target, region_box=region_box, count_streak=count_streak)
        _mark_verified_if_find_mode(get_active_detector_context(), media_id, result[1])
    # Progress-cache invalidation runs *after* ``_state_lock`` is released so
    # we never establish a ``_state_lock → _progress_lock`` ordering across
    # the two modules (audit M1).
    old_label, new_label, _click_time = result
    if _needs_progress_invalidate(old_label, new_label):
        from vtscore.detectors.labeling_progress import invalidate_progress_cache_from

        invalidate_progress_cache_from(media_id)
    return result


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
    (the HTTP ``/api/medias/<id>/vote`` endpoint no longer uses it - see
    :func:`set_vote`).

    Args:
        media_id: Integer ID of the media to vote on.
        vote: ``"good"`` or ``"bad"`` - the clicked direction.  Toggles off
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
        old, new, _click_time = _set_vote_locked(
            media_id,
            target,
            region_box=region_box if target == "good" else None,
        )
        _mark_verified_if_find_mode(ctx, media_id, new)
    # See the lock-ordering note on :func:`set_vote`.
    if _needs_progress_invalidate(old, new):
        from vtscore.detectors.labeling_progress import invalidate_progress_cache_from

        invalidate_progress_cache_from(media_id)


def apply_label(
    media_id: int,
    label: str,
    *,
    silent: bool = False,
    region_box: tuple[float, float, float, float] | None = None,
    record_achievement: bool = True,
    count_streak: bool = True,
) -> None:
    """Atomically apply a label to a media (for imports).

    Unlike :func:`toggle_vote`, this always sets the label without toggling.
    No click-time is assigned (imported labels have no click-time).

    When *silent* is True, the label is recorded in ``good_votes``/``bad_votes``
    only - ``label_history`` is not appended, the coverage atlas is not
    marked, and achievement counters are not credited.  This is used when
    restoring a detector's saved labels into a new dataset: those labels are
    seeded so autopilot's good/bad-count gates are satisfied, but they should
    not contaminate the per-session Smart/Stable trends or pre-fill diversity
    coverage in the new dataset.

    Args:
        media_id: Integer ID of the media to label.
        label: ``"good"`` or ``"bad"``.
        silent: If True, skip history append, coverage-atlas marking, and
            achievement credit.
        region_box: Optional normalised ``(x0, y0, x1, y1)`` box from an
            imported labelset entry.  Only stored when *label* is ``"good"``
            (no-votes are always image-level).  Patch-embedder v2.
        record_achievement: When True (the default), credit one vote in the
            active detector's achievement counters.  Set to False for
            system-driven label application that isn't a user vote action
            (e.g. auto-import from a labelset source, example-media seeding
            on detector load).
        count_streak: Forwarded to the achievement credit (only when
            *record_achievement* is True).  Pass False for bulk applications
            such as label import, which credit the other vote achievements but
            must not build a Marathoner streak out of one import.  Defaults to
            True (e.g. a single add-to-pile upload is an individual vote).
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
            coverage_atlas_label(media_id, good=label == "good")
            if record_achievement and not already:
                _record_vote_locked(count_streak=count_streak)


def apply_label_with_click_time(media_id: int, label: str) -> None:
    """Atomically apply a label with click-time assignment (for fill-from-sort).

    Same as :func:`apply_label` but also assigns a click-time ordinal so the
    label appears in the frontend's click-time timeline.  Credits one vote in
    the active detector's achievement counters - bulk fill-from-sort flows
    are user vote actions and must show up in ``votes_cast`` etc. (audit
    finding C8).  It does *not* count toward the Marathoner ``vote_streak``:
    fill-from-sort applies a whole sort-window at once, which is not the
    consecutive-individual-hand-click effort the streak measures.

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
        coverage_atlas_label(media_id, good=label == "good")
        # Bulk fill-from-sort: credit votes_cast/days/hours/types but never the
        # individual-effort Marathoner streak.
        _record_vote_locked(count_streak=False)


def apply_labels_bulk_with_click_time(
    labels: list[tuple[int, str]],
    replace_all: bool = False,
    *,
    record_achievement: bool = True,
    preserve_verified: bool = False,
) -> set[int]:
    """Apply many labels in a single lock acquisition (for find-label scoring).

    Each entry is ``(media_id, label)`` where *label* is ``"good"`` or
    ``"bad"``.  All labels are applied atomically with click-time ordinals
    assigned in order.

    Set *record_achievement* to ``False`` when the labels are system-generated
    (e.g. Find-mode auto-scoring) rather than explicit user votes, so those
    labels do not count toward ``votes_cast`` or ``vote_streak`` achievements.
    When *record_achievement* is True, the credited votes still never touch the
    Marathoner ``vote_streak``: this applies a whole batch at once, which is
    not the consecutive-individual-hand-click effort the streak measures.

    When *replace_all* is True, any pre-existing votes/click-times for IDs
    outside *labels* are cleared first.  This is what ``/api/find-label``
    wants: a detector trained on Dataset A holds Dataset A's media IDs in
    its DetectorContext, and switching to Dataset B must not leak those
    stale IDs into Dataset B's right-scroll Goods/Bads.  Any ``verified_ids``
    entry outside *labels* goes with them: a verified marker whose vote was
    just dropped would keep inflating ``verified_count`` and the verified
    partitions for an item this label set no longer covers.

    Set *preserve_verified* to ``True`` when the labels are a detector's
    machine calls over a live Find session (``/api/find-label``).  Ids in
    ``verified_ids`` are then left exactly as they are - vote, click-time,
    region box, history - and returned instead of being overwritten, because
    the human already ruled on them and a re-score must not silently invert a
    recorded decision while still presenting it as human-verified (issue
    #2928).  The fold-corrections -> retrain -> re-score loop re-runs this path
    on purpose, so without the guard every verified item the retrained detector
    now scores on the other side of the cutoff would flip to the machine's call
    and stay marked verified.

    Returns:
        The ids skipped because they were verified (empty unless
        *preserve_verified* is set).
    """
    import time as _time

    with _state_lock:
        ctx = get_active_detector_context()
        good_votes = ctx.good_votes
        bad_votes = ctx.bad_votes
        vote_click_times = ctx.vote_click_times
        vote_region_boxes = ctx.vote_region_boxes
        verified_ids = ctx.verified_ids
        label_history = ctx.label_history
        atlas = get_active_context().coverage_atlas
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
            for cid in [c for c in verified_ids if c not in kept]:
                verified_ids.pop(cid, None)
            ctx.find_initial_labels.clear()
        preserved: set[int] = set()
        for media_id, label in labels:
            if preserve_verified and media_id in verified_ids:
                # The human's call stands, click-time and all.  Its machine
                # label still reaches ``find_initial_labels`` via the caller,
                # so a disagreement shows up as a correction in Stats.
                preserved.add(media_id)
                continue
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
            if atlas is not None and media_id in atlas.vector_to_leaf:
                atlas.label(media_id, good=label == "good")
            if not already and record_achievement:
                # Bulk batch: never contributes to the individual-effort streak.
                _record_vote_locked(count_streak=False)
        return preserved
