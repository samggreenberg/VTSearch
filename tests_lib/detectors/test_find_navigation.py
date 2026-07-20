"""Unit tests for the server-side Find work-queue / boundary-walk queries.

These back the ``/api/find/queue-ids`` and ``/api/find/boundary-next`` endpoints
(scalability.md S3/S17/S19): they move the Find bulk-action id derivation and the
"just sit and vote" boundary walk off the client so a windowed frontend — which
no longer holds the whole ranking — still acts on the *entire* positive set.

Library tier: the queries are pure ``vtscore`` reads over the active detector
context's frozen Find state.
"""

from __future__ import annotations

from vtscore.state import find_boundary_next, find_queue_ids, get_active_detector_context, set_find_scores


def _seed_find_state(scores: dict[int, float], threshold: float, verified: set[int], good: set[int]) -> None:
    """Put the active detector context into a scored Find state."""
    ctx = get_active_detector_context()
    ctx.find_mode = True
    ctx.threshold = threshold
    set_find_scores(scores)
    ctx.verified_ids.clear()
    ctx.verified_ids.update({mid: None for mid in verified})
    ctx.good_votes.clear()
    ctx.good_votes.update({mid: None for mid in good})


# A canonical scored state reused across cases:
#   scores 1..5 descending; cutoff 0.5; item 1 verified-good (above),
#   item 4 verified (below); auto-good above-cutoff unverified = {2, 3}.
_SCORES = {1: 0.9, 2: 0.8, 3: 0.6, 4: 0.3, 5: 0.1}


class TestFindQueueIds:
    def test_unverified_good_is_above_cutoff_unverified_in_rank_order(self):
        _seed_find_state(_SCORES, 0.5, verified={1, 4}, good={1, 2, 3})
        assert find_queue_ids("unverified_good") == [2, 3]

    def test_good_is_verified_good_then_unverified_positives(self):
        _seed_find_state(_SCORES, 0.5, verified={1, 4}, good={1, 2, 3})
        # verified-good = {1}; unverified positives = [2, 3].
        assert find_queue_ids("good") == [1, 2, 3]

    def test_tracks_a_cutoff_slide(self):
        _seed_find_state(_SCORES, 0.5, verified=set(), good={1, 2, 3})
        assert find_queue_ids("unverified_good") == [1, 2, 3]
        # Raise the cutoff: item 3 (0.6) stays, but drop it below by moving to 0.7.
        get_active_detector_context().threshold = 0.7
        assert find_queue_ids("unverified_good") == [1, 2]

    def test_unknown_filter_is_empty(self):
        _seed_find_state(_SCORES, 0.5, verified=set(), good=set())
        assert find_queue_ids("nonsense") == []

    def test_empty_outside_find_mode(self):
        _seed_find_state(_SCORES, 0.5, verified=set(), good={1, 2, 3})
        get_active_detector_context().find_mode = False
        assert find_queue_ids("unverified_good") == []

    def test_empty_before_scoring(self):
        ctx = get_active_detector_context()
        ctx.find_mode = True
        ctx.threshold = 0.5
        ctx.find_scores.clear()
        assert find_queue_ids("good") == []


class TestFindBoundaryNext:
    def test_above_returns_lowest_unverified_above(self):
        _seed_find_state(_SCORES, 0.5, verified={1, 4}, good={1, 2, 3})
        assert find_boundary_next("above") == {"id": 3, "side": "above"}

    def test_below_returns_highest_unverified_below(self):
        _seed_find_state(_SCORES, 0.5, verified={1, 4}, good={1, 2, 3})
        assert find_boundary_next("below") == {"id": 5, "side": "below"}

    def test_exclude_skips_the_just_voted_item(self):
        _seed_find_state(_SCORES, 0.5, verified={1, 4}, good={1, 2, 3})
        assert find_boundary_next("above", exclude=3) == {"id": 2, "side": "above"}

    def test_falls_back_to_other_side_when_preferred_exhausted(self):
        # Verify everything above the cutoff; 'above' must fall back to 'below'.
        _seed_find_state(_SCORES, 0.5, verified={1, 2, 3, 4}, good={1, 2, 3})
        assert find_boundary_next("above") == {"id": 5, "side": "below"}

    def test_done_state_when_all_verified(self):
        _seed_find_state(_SCORES, 0.5, verified={1, 2, 3, 4, 5}, good={1, 2, 3})
        assert find_boundary_next("above") == {"id": None, "side": None}

    def test_none_outside_find_mode(self):
        _seed_find_state(_SCORES, 0.5, verified=set(), good=set())
        get_active_detector_context().find_mode = False
        assert find_boundary_next("above") == {"id": None, "side": None}
