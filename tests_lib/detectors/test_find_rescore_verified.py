"""Unit tests for the verified-vote guard in the Find bulk label apply.

``/api/find-label`` re-applies the detector's machine call to every item on
every run, and the fold-corrections -> retrain -> re-score loop re-runs it on
purpose.  Items the human already verified must keep their vote through that,
or a recorded human decision gets silently inverted while still being counted
as human-verified (issue #2928).

Library tier: ``apply_labels_bulk_with_click_time`` is a pure ``vtscore``
mutation of the active detector context.
"""

from __future__ import annotations

from vtscore.state import apply_labels_bulk_with_click_time, get_active_detector_context


def _seed_find_session(good: set[int], bad: set[int], verified: set[int]) -> None:
    """Put the active detector context into a voted, partly-verified Find state."""
    ctx = get_active_detector_context()
    ctx.find_mode = True
    ctx.good_votes.clear()
    ctx.bad_votes.clear()
    ctx.good_votes.update({mid: None for mid in good})
    ctx.bad_votes.update({mid: None for mid in bad})
    ctx.vote_click_times.clear()
    ctx.vote_click_times.update({mid: 100 + mid for mid in (good | bad)})
    ctx.click_counter = 200
    ctx.verified_ids.clear()
    ctx.verified_ids.update({mid: None for mid in verified})


class TestPreserveVerified:
    """``preserve_verified=True`` leaves verified items exactly as they were."""

    def test_verified_vote_is_not_overwritten(self):
        # 1 verified-good, 2 unverified-good; the new pass calls both bad.
        _seed_find_session(good={1, 2}, bad=set(), verified={1})
        apply_labels_bulk_with_click_time([(1, "bad"), (2, "bad")], replace_all=True, preserve_verified=True)
        ctx = get_active_detector_context()
        assert 1 in ctx.good_votes and 1 not in ctx.bad_votes  # human's call held
        assert 2 in ctx.bad_votes and 2 not in ctx.good_votes  # machine's call adopted
        assert 1 in ctx.verified_ids

    def test_verified_click_time_is_preserved(self):
        _seed_find_session(good={1, 2}, bad=set(), verified={1})
        apply_labels_bulk_with_click_time([(1, "bad"), (2, "bad")], replace_all=True, preserve_verified=True)
        ctx = get_active_detector_context()
        assert ctx.vote_click_times[1] == 101  # untouched
        assert ctx.vote_click_times[2] != 102  # re-stamped by this pass

    def test_returns_the_preserved_ids(self):
        _seed_find_session(good={1, 2}, bad={3}, verified={1, 3})
        preserved = apply_labels_bulk_with_click_time(
            [(1, "bad"), (2, "bad"), (3, "good")], replace_all=True, preserve_verified=True
        )
        assert preserved == {1, 3}

    def test_verified_bad_holds_against_a_good_call(self):
        # The mirror case: a human-culled false positive the new pass promotes.
        _seed_find_session(good=set(), bad={1}, verified={1})
        apply_labels_bulk_with_click_time([(1, "good")], replace_all=True, preserve_verified=True)
        ctx = get_active_detector_context()
        assert 1 in ctx.bad_votes and 1 not in ctx.good_votes

    def test_off_by_default(self):
        """The guard is opt-in; the plain bulk apply still reassigns everything."""
        _seed_find_session(good={1}, bad=set(), verified={1})
        preserved = apply_labels_bulk_with_click_time([(1, "bad")], replace_all=True)
        assert preserved == set()
        assert 1 in get_active_detector_context().bad_votes


class TestReplaceAllPrunesVerified:
    """``replace_all`` drops verified markers whose vote it just cleared."""

    def test_verified_id_outside_the_new_label_set_is_dropped(self):
        _seed_find_session(good={1, 2}, bad=set(), verified={1, 2})
        apply_labels_bulk_with_click_time([(1, "good")], replace_all=True, preserve_verified=True)
        ctx = get_active_detector_context()
        assert 2 not in ctx.good_votes  # vote cleared (not in the new label set)
        assert 2 not in ctx.verified_ids  # ...so the verified marker goes too
        assert 1 in ctx.verified_ids
