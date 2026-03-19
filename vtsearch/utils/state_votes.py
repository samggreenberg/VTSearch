"""Vote management, label history, text-sort suggestions, and learned scores.

Also contains compound vote operations (toggle_vote, apply_label) that
coordinate across vote dicts, click tracking, and the diversity tree.
"""

from __future__ import annotations

from vtsearch.utils.state_core import (
    _state_lock,
    bad_votes,
    good_votes,
    label_history,
    last_learned_scores,
    textsort_suggestions,
    vote_click_times,
)
from vtsearch.utils.state_clicks import assign_click_time, remove_click_time
from vtsearch.utils.state_diversity import diversity_tree_label, diversity_tree_unlabel

import vtsearch.utils.state_core as _core


def clear_votes() -> None:
    """Clear all votes and the full label history.

    Removes all entries from ``good_votes``, ``bad_votes``, and
    ``label_history`` in place. Does not affect the ``medias`` dict.
    Also clears the progress model cache and click-time / score tracking.
    """
    from vtsearch.models.progress import clear_progress_cache

    with _state_lock:
        good_votes.clear()
        bad_votes.clear()
        label_history.clear()
        textsort_suggestions.clear()
        vote_click_times.clear()
        _core._click_counter = 0
        last_learned_scores.clear()
        _core._find_initial_labels.clear()
        clear_progress_cache()


def add_label_to_history(media_id: int, label: str) -> None:
    """Append a labelling event to the global label history with a timestamp.

    Args:
        media_id: Integer ID of the media that was labelled.
        label: The assigned label; should be ``"good"`` or ``"bad"``.
    """
    import time

    with _state_lock:
        label_history.append((media_id, label, time.time()))


def add_textsort_suggestion(text: str) -> None:
    """Record a text-sort query as a suggested detector/labelset name.

    Duplicates are moved to the end so the most-recently-voted query is last.

    Args:
        text: The text-sort query string to store.
    """
    with _state_lock:
        # Remove existing occurrence so it moves to the end
        try:
            textsort_suggestions.remove(text)
        except ValueError:
            pass
        textsort_suggestions.append(text)


def get_textsort_suggestions() -> list[str]:
    """Return stored text-sort suggestions, most recent last."""
    with _state_lock:
        return list(textsort_suggestions)


def update_learned_scores(scores: dict[int, float]) -> None:
    """Replace the stored learned-sort scores with *scores*."""
    with _state_lock:
        last_learned_scores.clear()
        last_learned_scores.update(scores)


def get_learned_scores() -> dict[int, float]:
    """Return a copy of the last learned-sort scores."""
    with _state_lock:
        return last_learned_scores.copy()


def set_find_initial_labels(labels: dict[int, str]) -> None:
    """Store the detector-assigned labels from a find-label run.

    This snapshot is used to identify "corrections" — items where the
    user subsequently changed the label from what the detector assigned.
    """
    with _state_lock:
        _core._find_initial_labels.clear()
        _core._find_initial_labels.update(labels)


def get_find_initial_labels() -> dict[int, str]:
    """Return a copy of the find-label initial labels."""
    with _state_lock:
        return _core._find_initial_labels.copy()


# ---------------------------------------------------------------------------
# Compound operations (atomic vote toggle / label apply)
# ---------------------------------------------------------------------------


def toggle_vote(media_id: int, vote: str) -> None:
    """Atomically toggle a good/bad vote for a media item.

    Implements the same toggle semantics as the ``/api/medias/<id>/vote``
    endpoint: if the media already has the requested vote it is removed
    (unlabelled); otherwise the vote is applied (overriding any existing
    opposite vote).

    When a vote switches polarity (good→bad or bad→good), the progress
    cache is partially invalidated: only cached steps from the point where
    the media first appeared in the training data are discarded.  Earlier
    steps (whose models never included this media) are preserved.

    This function acquires ``_state_lock`` so that the entire check-then-modify
    sequence is atomic with respect to concurrent requests.

    Args:
        media_id: Integer ID of the media to vote on.
        vote: ``"good"`` or ``"bad"``.
    """
    from vtsearch.models.progress import invalidate_progress_cache_from

    with _state_lock:
        if vote == "good":
            if media_id in good_votes:
                good_votes.pop(media_id, None)
                remove_click_time(media_id)
                add_label_to_history(media_id, "unlabel")
                if media_id not in bad_votes:
                    diversity_tree_unlabel(media_id)
            else:
                was_opposite = media_id in bad_votes
                bad_votes.pop(media_id, None)
                good_votes[media_id] = None
                assign_click_time(media_id)
                add_label_to_history(media_id, "good")
                diversity_tree_label(media_id)
                if was_opposite:
                    invalidate_progress_cache_from(media_id)
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
                bad_votes[media_id] = None
                assign_click_time(media_id)
                add_label_to_history(media_id, "bad")
                diversity_tree_label(media_id)
                if was_opposite:
                    invalidate_progress_cache_from(media_id)


def apply_label(media_id: int, label: str) -> None:
    """Atomically apply a label to a media (for imports).

    Unlike :func:`toggle_vote`, this always sets the label without toggling.
    No click-time is assigned (imported labels have no click-time).

    Args:
        media_id: Integer ID of the media to label.
        label: ``"good"`` or ``"bad"``.
    """
    with _state_lock:
        if label == "good":
            bad_votes.pop(media_id, None)
            good_votes[media_id] = None
            add_label_to_history(media_id, "good")
        else:
            good_votes.pop(media_id, None)
            bad_votes[media_id] = None
            add_label_to_history(media_id, "bad")
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
        if label == "good":
            bad_votes.pop(media_id, None)
            good_votes[media_id] = None
            add_label_to_history(media_id, "good")
        else:
            good_votes.pop(media_id, None)
            bad_votes[media_id] = None
            add_label_to_history(media_id, "bad")
        assign_click_time(media_id)
        diversity_tree_label(media_id)


def apply_labels_bulk_with_click_time(labels: list[tuple[int, str]]) -> None:
    """Apply many labels in a single lock acquisition (for find-label scoring).

    Each entry is ``(media_id, label)`` where *label* is ``"good"`` or
    ``"bad"``.  All labels are applied atomically with click-time ordinals
    assigned in order.
    """
    import time as _time

    with _state_lock:
        tree = _core._diversity_tree
        for media_id, label in labels:
            if label == "good":
                bad_votes.pop(media_id, None)
                good_votes[media_id] = None
                label_history.append((media_id, "good", _time.time()))
            else:
                good_votes.pop(media_id, None)
                bad_votes[media_id] = None
                label_history.append((media_id, "bad", _time.time()))
            _core._click_counter += 1
            vote_click_times[media_id] = _core._click_counter
            if tree is not None and media_id in tree.vector_to_leaf:
                tree.label(media_id)
