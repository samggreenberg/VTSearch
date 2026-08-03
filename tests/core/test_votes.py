import app as app_module
import vtscore.detectors.labeling_progress as labeling_progress
from vtscore.detectors.labeling_progress import (
    _cache_good_ids,
    _cache_bad_ids,
    _cached_steps,
    _compute_stable_status,
    _ensure_cache,
    _live_models,
    calculate_diversity_level_over_time,
    inject_live_model,
)
from vtscore.embedding.media_vectors import media_embedding
from vtsearch.state import (
    build_coverage_atlas,
    get_coverage_atlas,
    medias,
    label_history,
)


class TestVoteClip:
    def test_vote_good(self, client):
        resp = client.post("/api/medias/1/vote", json={"target": "good"})
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True
        assert 1 in app_module.good_votes

    def test_vote_bad(self, client):
        resp = client.post("/api/medias/1/vote", json={"target": "bad"})
        assert resp.status_code == 200
        assert 1 in app_module.bad_votes

    def test_unvote_good(self, client):
        """target=none removes a good vote."""
        client.post("/api/medias/1/vote", json={"target": "good"})
        assert 1 in app_module.good_votes
        client.post("/api/medias/1/vote", json={"target": "none"})
        assert 1 not in app_module.good_votes

    def test_unvote_bad(self, client):
        """target=none removes a bad vote."""
        client.post("/api/medias/1/vote", json={"target": "bad"})
        assert 1 in app_module.bad_votes
        client.post("/api/medias/1/vote", json={"target": "none"})
        assert 1 not in app_module.bad_votes

    def test_idempotent_re_vote_good(self, client):
        """Re-sending target=good on a good media is a no-op (H1 fix).

        Two stale-view tabs that both POST target=good no longer alternate
        ADD/REMOVE on the server; the second call is idempotent.
        """
        r1 = client.post("/api/medias/1/vote", json={"target": "good"})
        assert r1.status_code == 200
        r2 = client.post("/api/medias/1/vote", json={"target": "good"})
        assert r2.status_code == 200
        assert 1 in app_module.good_votes
        # Idempotent: a single label_history entry, not two.
        from vtsearch.state import label_history

        assert sum(1 for entry in label_history if entry[0] == 1) == 1

    def test_idempotent_re_vote_bad(self, client):
        r1 = client.post("/api/medias/1/vote", json={"target": "bad"})
        assert r1.status_code == 200
        r2 = client.post("/api/medias/1/vote", json={"target": "bad"})
        assert r2.status_code == 200
        assert 1 in app_module.bad_votes
        from vtsearch.state import label_history

        assert sum(1 for entry in label_history if entry[0] == 1) == 1

    def test_switch_from_good_to_bad(self, client):
        client.post("/api/medias/1/vote", json={"target": "good"})
        client.post("/api/medias/1/vote", json={"target": "bad"})
        assert 1 not in app_module.good_votes
        assert 1 in app_module.bad_votes

    def test_switch_from_bad_to_good(self, client):
        client.post("/api/medias/1/vote", json={"target": "bad"})
        client.post("/api/medias/1/vote", json={"target": "good"})
        assert 1 not in app_module.bad_votes
        assert 1 in app_module.good_votes

    def test_invalid_target_value(self, client):
        # Marshmallow OneOf validator on ``target`` rejects non-{good,bad,none}
        # values with the flask-smorest 422 envelope.
        resp = client.post("/api/medias/1/vote", json={"target": "meh"})
        assert resp.status_code == 422
        assert "target" in resp.get_json()["errors"]["json"]

    def test_missing_target_field(self, client):
        # Required-field validation runs in MediaVoteRequestSchema → 422.
        resp = client.post("/api/medias/1/vote", json={"wrong": "field"})
        assert resp.status_code == 422

    def test_legacy_vote_field_rejected(self, client):
        # The pre-H1 ``vote`` field is no longer accepted; the schema requires
        # ``target`` with absolute-state semantics so the toggle race cannot
        # be re-introduced by a client that hasn't been updated.
        resp = client.post("/api/medias/1/vote", json={"vote": "good"})
        assert resp.status_code == 422
        assert "target" in resp.get_json()["errors"]["json"]

    def test_vote_nonexistent_clip(self, client):
        resp = client.post("/api/medias/9999/vote", json={"target": "good"})
        assert resp.status_code == 404

    def test_multiple_clips_independent_votes(self, client):
        client.post("/api/medias/1/vote", json={"target": "good"})
        client.post("/api/medias/2/vote", json={"target": "bad"})
        assert 1 in app_module.good_votes
        assert 2 in app_module.bad_votes
        assert 1 not in app_module.bad_votes
        assert 2 not in app_module.good_votes


class TestVoteBulk:
    """POST /api/medias/vote-bulk: apply one target to many ids at once.

    Powers the Browser's "Remove from Good" cull (mark a hand-selected set of
    false-positives Bad in a single request).
    """

    def test_bulk_mark_bad_moves_from_good(self, client):
        client.post("/api/medias/1/vote", json={"target": "good"})
        client.post("/api/medias/2/vote", json={"target": "good"})
        resp = client.post("/api/medias/vote-bulk", json={"ids": [1, 2], "target": "bad"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["changed"] == 2
        assert data["missing"] == []
        assert 1 in app_module.bad_votes
        assert 2 in app_module.bad_votes
        assert 1 not in app_module.good_votes
        assert 2 not in app_module.good_votes

    def test_bulk_idempotent_not_counted(self, client):
        client.post("/api/medias/1/vote", json={"target": "bad"})
        # 1 is already bad → no change; 2 transitions none→bad → one change.
        resp = client.post("/api/medias/vote-bulk", json={"ids": [1, 2], "target": "bad"})
        assert resp.get_json()["changed"] == 1
        assert 1 in app_module.bad_votes
        assert 2 in app_module.bad_votes

    def test_bulk_reports_missing_ids(self, client):
        resp = client.post("/api/medias/vote-bulk", json={"ids": [1, 9999], "target": "bad"})
        data = resp.get_json()
        assert data["missing"] == [9999]
        assert data["changed"] == 1
        assert 1 in app_module.bad_votes

    def test_bulk_empty_ids_rejected(self, client):
        resp = client.post("/api/medias/vote-bulk", json={"ids": [], "target": "bad"})
        assert resp.status_code == 400

    def test_bulk_invalid_target_rejected(self, client):
        resp = client.post("/api/medias/vote-bulk", json={"ids": [1], "target": "meh"})
        assert resp.status_code == 422
        assert "target" in resp.get_json()["errors"]["json"]


class TestGetVotes:
    def test_empty_votes(self, client):
        resp = client.get("/api/votes")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["good"] == []
        assert data["bad"] == []
        assert data["click_times"] == {}
        assert data["learned_scores"] == {}

    def test_returns_good_votes(self, client):
        app_module.good_votes.update({k: None for k in [5, 1, 3]})
        resp = client.get("/api/votes")
        data = resp.get_json()
        assert data["good"] == [1, 3, 5]  # sorted regardless of insertion order

    def test_returns_bad_votes(self, client):
        app_module.bad_votes.update({k: None for k in [4, 2]})
        resp = client.get("/api/votes")
        data = resp.get_json()
        assert data["bad"] == [2, 4]  # sorted regardless of insertion order

    def test_returns_both(self, client):
        app_module.good_votes[1] = None
        app_module.bad_votes[2] = None
        resp = client.get("/api/votes")
        data = resp.get_json()
        assert data["good"] == [1]
        assert data["bad"] == [2]

    def test_votes_after_voting_via_api(self, client):
        client.post("/api/medias/3/vote", json={"target": "good"})
        client.post("/api/medias/5/vote", json={"target": "bad"})
        resp = client.get("/api/votes")
        data = resp.get_json()
        assert 3 in data["good"]
        assert 5 in data["bad"]


class TestClearVotes:
    def test_clear_votes_empties_good_and_bad(self, client):
        """POST /api/votes/clear should remove all votes."""
        client.post("/api/medias/1/vote", json={"target": "good"})
        client.post("/api/medias/2/vote", json={"target": "bad"})
        assert 1 in app_module.good_votes
        assert 2 in app_module.bad_votes

        resp = client.post("/api/votes/clear")
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True
        assert len(app_module.good_votes) == 0
        assert len(app_module.bad_votes) == 0

    def test_clear_votes_preserves_medias(self, client):
        """Clearing votes should not affect loaded medias."""
        num_before = len(medias)
        client.post("/api/medias/1/vote", json={"target": "good"})
        resp = client.post("/api/votes/clear")
        assert resp.status_code == 200
        assert len(medias) == num_before

    def test_get_votes_empty_after_clear(self, client):
        """GET /api/votes should return empty lists after clear."""
        client.post("/api/medias/1/vote", json={"target": "good"})
        client.post("/api/medias/2/vote", json={"target": "bad"})
        client.post("/api/votes/clear")
        resp = client.get("/api/votes")
        data = resp.get_json()
        assert data["good"] == []
        assert data["bad"] == []


class TestLabelHistory:
    def test_vote_adds_history_entry(self, client):
        client.post("/api/medias/1/vote", json={"target": "good"})
        assert len(label_history) == 1
        assert label_history[0][0] == 1
        assert label_history[0][1] == "good"

    def test_unvote_good_removes_the_click(self, client):
        """target=none says the click should never have happened.

        The history records what we now believe the labels were, not a journal
        of actions, so withdrawing a label deletes its event rather than
        appending an ``unlabel`` that every replay then has to undo.
        """
        client.post("/api/medias/1/vote", json={"target": "good"})
        client.post("/api/medias/1/vote", json={"target": "none"})
        assert label_history == []

    def test_unvote_bad_removes_the_click(self, client):
        client.post("/api/medias/1/vote", json={"target": "bad"})
        client.post("/api/medias/1/vote", json={"target": "none"})
        assert label_history == []

    def test_unvote_leaves_other_clicks_alone(self, client):
        client.post("/api/medias/1/vote", json={"target": "good"})
        client.post("/api/medias/2/vote", json={"target": "bad"})
        client.post("/api/medias/1/vote", json={"target": "none"})
        assert [(entry[0], entry[1]) for entry in label_history] == [(2, "bad")]

    def test_idempotent_re_vote_does_not_grow_history(self, client):
        """Re-sending the current state appends nothing to label_history (H1)."""
        client.post("/api/medias/1/vote", json={"target": "good"})
        client.post("/api/medias/1/vote", json={"target": "good"})
        client.post("/api/medias/1/vote", json={"target": "good"})
        assert len(label_history) == 1
        assert label_history[0][1] == "good"

    def test_switch_vote_corrects_the_existing_entry(self, client):
        """Switching good->bad rewrites that click; it is not a second click."""
        client.post("/api/medias/1/vote", json={"target": "good"})
        client.post("/api/medias/1/vote", json={"target": "bad"})
        assert len(label_history) == 1
        assert label_history[0][0] == 1
        assert label_history[0][1] == "bad"

    def test_switch_keeps_the_click_s_position_in_the_session(self, client):
        """The correction stays where the original click was, not at the end."""
        client.post("/api/medias/1/vote", json={"target": "good"})
        client.post("/api/medias/2/vote", json={"target": "bad"})
        client.post("/api/medias/3/vote", json={"target": "good"})

        client.post("/api/medias/1/vote", json={"target": "bad"})

        assert [(entry[0], entry[1]) for entry in label_history] == [(1, "bad"), (2, "bad"), (3, "good")]

    def test_unvote_then_revote(self, client):
        """Un-vote drops the click; re-voting appends a fresh one at the end."""
        client.post("/api/medias/1/vote", json={"target": "good"})
        client.post("/api/medias/1/vote", json={"target": "none"})  # un-vote
        client.post("/api/medias/1/vote", json={"target": "bad"})  # revote bad
        assert len(label_history) == 1
        assert label_history[0][1] == "bad"
        assert 1 in app_module.bad_votes
        assert 1 not in app_module.good_votes


class TestProgressCacheWithLabelChanges:
    """Verify the progress cache stays consistent when labels are changed."""

    def test_cache_removes_clip_on_unlabel(self, client):
        """After un-voting, the progress cache should not include the media."""
        client.post("/api/medias/1/vote", json={"target": "good"})
        client.post("/api/medias/2/vote", json={"target": "bad"})
        _ensure_cache(medias, label_history, 0)
        assert 1 in _cache_good_ids
        assert 2 in _cache_bad_ids

        # Un-vote media 1
        client.post("/api/medias/1/vote", json={"target": "none"})
        _ensure_cache(medias, label_history, 0)
        assert 1 not in _cache_good_ids
        assert 1 not in _cache_bad_ids
        assert 2 in _cache_bad_ids

    def test_cache_handles_switch_vote(self, client):
        """Switching good->bad should update cache running sets correctly."""
        client.post("/api/medias/1/vote", json={"target": "good"})
        client.post("/api/medias/2/vote", json={"target": "bad"})
        _ensure_cache(medias, label_history, 0)
        assert 1 in _cache_good_ids

        # Switch media 1 from good to bad
        client.post("/api/medias/1/vote", json={"target": "bad"})
        _ensure_cache(medias, label_history, 0)
        assert 1 not in _cache_good_ids
        assert 1 in _cache_bad_ids

    def test_cache_unvote_then_revote(self, client):
        """Un-vote then re-vote should leave cache in correct state."""
        client.post("/api/medias/1/vote", json={"target": "good"})
        client.post("/api/medias/2/vote", json={"target": "bad"})
        # Un-vote media 1
        client.post("/api/medias/1/vote", json={"target": "none"})
        # Revote as bad
        client.post("/api/medias/1/vote", json={"target": "bad"})
        _ensure_cache(medias, label_history, 0)
        assert 1 in _cache_bad_ids
        assert 1 not in _cache_good_ids

    def test_learned_sort_after_unvote(self, client):
        """Learned sort should work after un-voting a media."""
        client.post("/api/medias/1/vote", json={"target": "good"})
        client.post("/api/medias/2/vote", json={"target": "bad"})
        resp = client.post("/api/learned-sort", json={"wait": True})
        assert resp.status_code == 200

        # Un-vote the existing good, add a different good vote
        client.post("/api/medias/1/vote", json={"target": "none"})
        client.post("/api/medias/3/vote", json={"target": "good"})
        resp = client.post("/api/learned-sort", json={"wait": True})
        assert resp.status_code == 200

    def test_learned_sort_returns_400_after_unvoting_all_good(self, client):
        """If all good votes are removed, learned sort should return 400."""
        client.post("/api/medias/1/vote", json={"target": "good"})
        client.post("/api/medias/2/vote", json={"target": "bad"})
        # Un-vote the only good vote
        client.post("/api/medias/1/vote", json={"target": "none"})
        resp = client.post("/api/learned-sort", json={"wait": True})
        assert resp.status_code == 400

    def test_labeling_status_after_label_change(self, client):
        """labeling-status endpoint should not crash after label changes."""
        # Vote enough medias to get past the minimum threshold
        for i in range(1, 6):
            client.post(f"/api/medias/{i}/vote", json={"target": "good"})
        for i in range(6, 11):
            client.post(f"/api/medias/{i}/vote", json={"target": "bad"})

        resp = client.get("/api/labeling-status")
        assert resp.status_code == 200

        # Now un-vote a good and switch a bad vote
        client.post("/api/medias/1/vote", json={"target": "none"})  # un-vote
        client.post("/api/medias/6/vote", json={"target": "good"})  # switch bad->good

        resp = client.get("/api/labeling-status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["good_count"] == 5  # lost 1, gained 1
        assert data["bad_count"] == 4  # lost 1


class TestCorrectingAVoteInvalidatesFromThatPoint:
    """Changing your mind rewrites the click; it does not append a new one.

    Voting on an unlabeled media appends an event.  Changing an existing label
    is a different act: it says the earlier click was *wrong*.  So
    ``correct_label_in_history`` rewrites that event where it stands and
    ``invalidate_progress_cache_from`` drops every cached step from it onward -
    those models were fit on data we no longer believe.  Steps before it never
    saw the media and survive.

    The two halves are load-bearing together.  With an append-only history the
    invalidation replayed the *unchanged* events and reproduced the very models
    it had deleted, so it cost a rebuild and changed nothing.
    """

    def _vote(self, client, votes):
        for cid, target in votes:
            assert client.post(f"/api/medias/{cid}/vote", json={"target": target}).status_code == 200

    def _four_steps(self, client, first_target):
        """Cache 4 steps; media 1 is click #2 (zero-based index 2)."""
        self._vote(client, [(3, "good"), (4, "bad"), (1, first_target), (2, "bad")])
        _ensure_cache(medias, label_history, 0)
        assert len(_cached_steps) == 4

    def test_good_to_bad_truncates_from_the_corrected_click(self, client):
        self._four_steps(client, "good")
        self._vote(client, [(1, "bad")])
        assert len(_cached_steps) == 2, "steps before media 1's click must survive"

    def test_bad_to_good_truncates_from_the_corrected_click(self, client):
        self._four_steps(client, "bad")
        self._vote(client, [(1, "good")])
        assert len(_cached_steps) == 2

    def test_correcting_the_first_click_clears_everything(self, client):
        self._vote(client, [(1, "good"), (2, "bad")])
        _ensure_cache(medias, label_history, 0)
        assert len(_cached_steps) == 2

        self._vote(client, [(1, "bad")])
        assert len(_cached_steps) == 0, "nothing precedes click #0"

    def test_the_correction_rewrites_the_click_rather_than_appending(self, client):
        """20 clicks corrected is still 20 clicks, with the 15th changed."""
        self._four_steps(client, "good")
        assert len(label_history) == 4
        assert label_history[2] == (1, "good", label_history[2][2])

        self._vote(client, [(1, "bad")])

        assert len(label_history) == 4, "a correction must not add a 5th click"
        assert label_history[2][0] == 1
        assert label_history[2][1] == "bad"

    def test_the_correction_propagates_to_every_later_step(self, client):
        """The whole point: later steps retrain against the corrected label."""
        self._four_steps(client, "good")
        assert 1 in _cached_steps[3]["good_ids"]

        self._vote(client, [(1, "bad")])
        _ensure_cache(medias, label_history, 0)

        assert len(_cached_steps) == 4
        assert 1 in _cached_steps[3]["bad_ids"]
        assert 1 not in _cached_steps[3]["good_ids"]

    def test_unvote_removes_the_click_entirely(self, client):
        """An un-vote says the click should never have happened."""
        self._vote(client, [(3, "good"), (4, "bad"), (1, "good")])
        _ensure_cache(medias, label_history, 0)
        assert len(_cached_steps) == 3

        self._vote(client, [(1, "none")])

        assert len(label_history) == 2, "the click is dropped, not marked 'unlabel'"
        assert all(entry[0] != 1 for entry in label_history)
        assert len(_cached_steps) == 2

    def test_new_vote_does_not_clear_cache(self, client):
        """A first label on an unlabeled media appends and invalidates nothing."""
        self._vote(client, [(1, "good"), (2, "bad")])
        _ensure_cache(medias, label_history, 0)
        assert len(_cached_steps) == 2

        self._vote(client, [(3, "good")])
        assert len(_cached_steps) == 2, "Cache should not be cleared when adding a new vote"

    def test_running_ids_restored_to_the_surviving_prefix(self, client):
        self._four_steps(client, "good")
        self._vote(client, [(1, "bad")])

        assert len(_cached_steps) == 2
        assert 3 in _cache_good_ids
        assert 4 in _cache_bad_ids
        assert 1 not in _cache_good_ids
        assert 1 not in _cache_bad_ids

    def test_surviving_steps_are_not_retrained(self, client):
        self._four_steps(client, "good")
        kept_models = [step["model"] for step in _cached_steps[:2]]

        self._vote(client, [(1, "bad")])
        _ensure_cache(medias, label_history, 0)

        # Identity: the prefix was reused, only the suffix refit.
        assert [s["model"] for s in _cached_steps[:2]] == kept_models

    def test_labeling_progress_works_after_switch(self, client):
        """The /api/labeling-progress endpoint should work after a vote switch."""
        for i in range(1, 6):
            client.post(f"/api/medias/{i}/vote", json={"target": "good"})
        for i in range(6, 11):
            client.post(f"/api/medias/{i}/vote", json={"target": "bad"})

        resp = client.post("/api/labeling-progress")
        assert resp.status_code == 200

        # Switch a vote
        client.post("/api/medias/1/vote", json={"target": "bad"})

        resp = client.post("/api/labeling-progress")
        assert resp.status_code == 200


class TestProgressAtlasClonesAndSurvivesVotes:
    """The diversity replay clones the dataset context's atlas, and a
    good↔bad correction leaves that replay entirely alone.

    Building the coverage structure is hierarchical k-means over every
    embedding, so it must never be re-fit on a vote.  ``_build_coverage_atlas``
    clones the context atlas (shared node table, fresh label overlay), and
    coverage evidence is polarity-agnostic (``CoverageAtlas._covered`` tests
    ``n_pos + n_neg > 0``), so switching a label between the two channels
    cannot change any step's ``coverage_level()``.  Only an un-vote - which
    drops the click outright - rewinds the replay, and even then the atlas
    structure is kept.

    The atlas is also no longer built by the model walk at all: asking for the
    Smart or Stable curve used to pay for it, which on a dataset past
    ``COVERAGE_ATLAS_AUTO_THRESHOLD`` (no load-time atlas to clone) was the
    single largest cost in the request.
    """

    def _vote(self, client, votes):
        for cid, target in votes:
            resp = client.post(f"/api/medias/{cid}/vote", json={"target": target})
            assert resp.status_code == 200

    def test_progress_atlas_clones_context_atlas(self, client):
        build_coverage_atlas()
        ctx_atlas = get_coverage_atlas()
        assert ctx_atlas is not None

        self._vote(client, [(1, "good"), (2, "bad")])
        calculate_diversity_level_over_time(medias, label_history)

        prog_atlas = labeling_progress._cache_coverage_atlas
        assert prog_atlas is not None
        assert prog_atlas is not ctx_atlas
        # Immutable structure is shared by reference (no re-fit)...
        assert prog_atlas.nodes is ctx_atlas.nodes
        assert prog_atlas.vector_to_leaf is ctx_atlas.vector_to_leaf
        assert prog_atlas.nodes_by_depth is ctx_atlas.nodes_by_depth
        # ...while the label overlay is an independent object.
        assert prog_atlas._n_pos is not ctx_atlas._n_pos
        assert prog_atlas._n_neg is not ctx_atlas._n_neg
        assert prog_atlas._labeled is not ctx_atlas._labeled

    def test_model_walk_does_not_build_an_atlas(self, client):
        build_coverage_atlas()
        self._vote(client, [(1, "good"), (2, "bad")])

        _ensure_cache(medias, label_history, 0)

        assert _cached_steps, "the model walk should still have run"
        assert labeling_progress._cache_coverage_atlas is None

    def test_a_polarity_correction_preserves_the_diversity_replay(self, client):
        build_coverage_atlas()
        # Steps: 0=(3,good), 1=(4,bad), 2=(1,good), 3=(2,bad); media 1 at step 2.
        self._vote(client, [(3, "good"), (4, "bad"), (1, "good"), (2, "bad")])
        _ensure_cache(medias, label_history, 0)
        calculate_diversity_level_over_time(medias, label_history)
        atlas_before = labeling_progress._cache_coverage_atlas
        assert atlas_before is not None
        assert atlas_before.labeled_ids == {1, 2, 3, 4}
        series_before = list(labeling_progress._cached_diversity)

        # Switch media 1 (good→bad). The model steps from its click are dropped,
        # but coverage evidence is polarity-agnostic, so the diversity replay -
        # and the atlas it was built from - are untouched.
        self._vote(client, [(1, "bad")])
        assert len(_cached_steps) == 2
        assert labeling_progress._cache_coverage_atlas is atlas_before
        assert atlas_before.labeled_ids == {1, 2, 3, 4}
        assert labeling_progress._cached_diversity == series_before

    def test_correcting_the_first_vote_preserves_the_atlas(self, client):
        build_coverage_atlas()
        self._vote(client, [(1, "good"), (2, "bad")])
        _ensure_cache(medias, label_history, 0)
        calculate_diversity_level_over_time(medias, label_history)
        atlas_before = labeling_progress._cache_coverage_atlas
        assert atlas_before is not None

        # Media 1 is click #0, so every model step goes - but the atlas, the
        # expensive artifact, must survive a vote no matter what.
        self._vote(client, [(1, "bad")])
        assert len(_cached_steps) == 0
        assert labeling_progress._cache_coverage_atlas is atlas_before
        assert atlas_before.labeled_ids == {1, 2}


class TestStableIndicatorThresholds:
    """The Stable indicator should not turn green prematurely."""

    def _inject_stability(self, entries):
        """Inject fake stability entries into the progress cache."""
        _cached_steps.clear()
        for entry in entries:
            _cached_steps.append({"model": None, "threshold": None, "good_ids": [], "bad_ids": [], "stability": entry})

    def test_no_model_steps_not_counted(self, client):
        """Steps without a prior model (stability=None) should be excluded.

        When there's no prior model to compare against (first model step
        or after a gap), stability is None.  These should not count toward
        the stability assessment.
        """
        self._inject_stability(
            [
                None,  # no model yet
                None,  # no prior model (first model step)
                {"time_index": 2, "num_labels": 12, "num_flips": 0, "num_unlabeled": 88},
                {"time_index": 3, "num_labels": 13, "num_flips": 0, "num_unlabeled": 87},
                {"time_index": 4, "num_labels": 14, "num_flips": 0, "num_unlabeled": 86},
                {"time_index": 5, "num_labels": 15, "num_flips": 0, "num_unlabeled": 85},
            ]
        )
        result = _compute_stable_status(good=5, bad=5, total=10)
        assert result["status"] == "yellow", "Should be yellow: only 4 real entries (None entries excluded)"

    def test_needs_five_real_entries_for_green(self, client):
        """Green requires at least 5 stability entries."""
        # None (no prior model) + 4 real = only 4 usable → yellow
        self._inject_stability(
            [
                None,  # no prior model
                {"time_index": 1, "num_labels": 11, "num_flips": 0, "num_unlabeled": 89},
                {"time_index": 2, "num_labels": 12, "num_flips": 0, "num_unlabeled": 88},
                {"time_index": 3, "num_labels": 13, "num_flips": 0, "num_unlabeled": 87},
                {"time_index": 4, "num_labels": 14, "num_flips": 0, "num_unlabeled": 86},
            ]
        )
        result = _compute_stable_status(good=5, bad=5, total=10)
        assert result["status"] == "yellow", "4 real entries is not enough"

        # Add one more → 5 usable → green
        self._inject_stability(
            [
                None,  # no prior model
                {"time_index": 1, "num_labels": 11, "num_flips": 0, "num_unlabeled": 89},
                {"time_index": 2, "num_labels": 12, "num_flips": 0, "num_unlabeled": 88},
                {"time_index": 3, "num_labels": 13, "num_flips": 0, "num_unlabeled": 87},
                {"time_index": 4, "num_labels": 14, "num_flips": 0, "num_unlabeled": 86},
                {"time_index": 5, "num_labels": 15, "num_flips": 0, "num_unlabeled": 85},
            ]
        )
        result = _compute_stable_status(good=5, bad=5, total=10)
        assert result["status"] == "green", "5 real low-flip entries should be enough"

    def test_single_spike_prevents_green(self, client):
        """One spike above the max threshold should keep the indicator yellow."""
        entries: list = [None]  # no prior model
        for i in range(1, 7):
            entries.append({"time_index": i, "num_labels": 10 + i, "num_flips": 0, "num_unlabeled": 100 - i})
        # Inject one spike: 6 flips out of 93 unlabeled → ~6.5%, above 1% max threshold
        entries[-1] = {"time_index": 6, "num_labels": 16, "num_flips": 6, "num_unlabeled": 93}

        self._inject_stability(entries)
        result = _compute_stable_status(good=8, bad=8, total=16)
        assert result["status"] == "yellow", "A single >1% spike should prevent green"

    def test_low_flip_rate_still_yellow(self, client):
        """Even a modest flip rate (~1.5%) should stay yellow under the tight thresholds."""
        entries: list = [None]  # no prior model
        for i in range(1, 7):
            # 3 flips out of ~194 unlabeled → ~1.5% per step
            entries.append({"time_index": i, "num_labels": 10 + i, "num_flips": 3, "num_unlabeled": 200 - i})
        self._inject_stability(entries)
        result = _compute_stable_status(good=8, bad=8, total=16)
        assert result["status"] == "yellow", "~1.5% flip rate should stay yellow (threshold is 0.5% avg / 1% max)"

    def test_near_zero_flips_green(self, client):
        """Green requires practically zero flips; only the rare single flip tolerated."""
        entries: list = [None]  # no prior model
        for i in range(1, 7):
            # Mostly 0 flips, one step with 1 flip out of ~496 → 0.2%
            flips = 1 if i == 3 else 0
            entries.append({"time_index": i, "num_labels": 10 + i, "num_flips": flips, "num_unlabeled": 500 - i})
        self._inject_stability(entries)
        result = _compute_stable_status(good=8, bad=8, total=16)
        assert result["status"] == "green", "Near-zero flips with rare single flip on large dataset should be green"

    def test_model_gap_entries_excluded(self, client):
        """After a model gap, the first step back should also be excluded (no prior model)."""
        self._inject_stability(
            [
                None,  # no prior model (first model step)
                {"time_index": 1, "num_labels": 11, "num_flips": 0, "num_unlabeled": 89},
                {"time_index": 2, "num_labels": 12, "num_flips": 0, "num_unlabeled": 88},
                None,  # model lost (gap; user unlabeled all bad)
                None,  # still no model
                None,  # first model step after gap; no prior model
                {"time_index": 6, "num_labels": 16, "num_flips": 0, "num_unlabeled": 84},
                {"time_index": 7, "num_labels": 17, "num_flips": 0, "num_unlabeled": 83},
            ]
        )
        result = _compute_stable_status(good=5, bad=5, total=10)
        # Only 4 real entries (indices 1, 2, 6, 7) → yellow
        assert result["status"] == "yellow", "Gap entries (None) should be excluded, leaving only 4 real entries"


class TestStabilitySkipsUnchangedModel:
    """Stability should not be recorded when the training data hasn't changed."""

    def test_unchanged_training_data_skips_stability(self, client):
        """When good/bad IDs don't change between steps, stability should be None."""
        from vtscore.detectors.labeling_progress import clear_progress_cache

        clear_progress_cache()

        # Build a minimal clips_dict with embeddings
        import numpy as np

        rng = np.random.RandomState(42)
        clips = {}
        for i in range(10):
            clips[i] = {"embeddings": {"clap": rng.randn(8).astype(np.float32)}, "embedder": "clap"}

        # Label history: vote good on 0, bad on 1, then vote good on 0 AGAIN
        # The third event doesn't change the training data (0 is already good).
        history = [
            (0, "good", 1.0),
            (1, "bad", 2.0),
            (0, "good", 3.0),  # no-op: 0 is already good
        ]

        _ensure_cache(clips, history, inclusion_value=0)

        assert len(_cached_steps) == 3
        # Steps 0 and 1 change the training data; they should attempt training.
        # Step 2 has the same good/bad IDs as step 1; stability should be None.
        assert _cached_steps[2]["stability"] is None, (
            "Stability should not be recorded when training data didn't change"
        )

    def test_changed_training_data_records_stability(self, client):
        """When good/bad IDs DO change, stability should be computed (once a prior model exists)."""
        from vtscore.detectors.labeling_progress import clear_progress_cache

        clear_progress_cache()

        import numpy as np

        rng = np.random.RandomState(43)
        clips = {}
        for i in range(20):
            clips[i] = {"embeddings": {"clap": rng.randn(8).astype(np.float32)}, "embedder": "clap"}

        # Each step changes the training data
        history = [
            (0, "good", 1.0),
            (1, "bad", 2.0),  # first model (both polarities)
            (2, "good", 3.0),  # new model
            (3, "bad", 4.0),  # new model
            (4, "good", 5.0),  # new model
        ]

        _ensure_cache(clips, history, inclusion_value=0)

        assert len(_cached_steps) == 5
        # Step 0: only good votes, no model → stability None
        assert _cached_steps[0]["stability"] is None
        # Step 1: first model, no prior predictions → stability None
        assert _cached_steps[1]["stability"] is None
        # Steps 2-4: new model each time, prior predictions exist → stability recorded
        for i in [2, 3, 4]:
            assert _cached_steps[i]["stability"] is not None, (
                f"Step {i} changed training data and should have stability"
            )


class TestLiveModelReuse:
    """Progress cache should reuse models injected by train_and_score."""

    def test_injected_model_is_used(self, client):
        """When a live model matches the label set, _ensure_cache should use it."""
        import numpy as np
        import torch

        from vtscore.detectors.labeling_progress import clear_progress_cache
        from vtscore.training.mlp import train_model

        clear_progress_cache()

        rng = np.random.RandomState(99)
        clips = {}
        for i in range(10):
            clips[i] = {"embeddings": {"clap": rng.randn(8).astype(np.float32)}, "embedder": "clap"}

        # Train a live model for good={0,2}, bad={1}
        good = {0: None, 2: None}
        bad = {1: None}
        X = torch.tensor(
            np.array([media_embedding(clips[0]), media_embedding(clips[2]), media_embedding(clips[1])]),
            dtype=torch.float32,
        )
        y = torch.tensor([1.0, 1.0, 0.0]).unsqueeze(1)
        live_model = train_model(X, y, 8)

        inject_live_model(good, bad, live_model)

        # Now run _ensure_cache with a history that leads to {0,2} good, {1} bad
        history = [
            (0, "good", 1.0),
            (1, "bad", 2.0),
            (2, "good", 3.0),
        ]

        _ensure_cache(clips, history, inclusion_value=0)

        # Step 2 should have used the injected model (same label set)
        step = _cached_steps[2]
        assert step["model"] is live_model, "Should reuse the injected live model"
        # The cutoff is always derived by the cache itself, never carried in
        # with the model: a live threshold is cross-calibrated while every
        # other step's is in-sample, and mixing the two put a step-change into
        # the plotted curve wherever a live model happened to land.
        assert step["threshold"] is not None

    def test_non_matching_label_set_retrains(self, client):
        """When no live model matches, _ensure_cache should train normally."""
        import numpy as np
        import torch

        from vtscore.detectors.labeling_progress import clear_progress_cache
        from vtscore.training.mlp import train_model

        clear_progress_cache()

        rng = np.random.RandomState(99)
        clips = {}
        for i in range(10):
            clips[i] = {"embeddings": {"clap": rng.randn(8).astype(np.float32)}, "embedder": "clap"}

        # Inject a live model for a DIFFERENT label set
        good = {5: None, 6: None}
        bad = {7: None}
        X = torch.tensor(
            np.array([media_embedding(clips[5]), media_embedding(clips[6]), media_embedding(clips[7])]),
            dtype=torch.float32,
        )
        y = torch.tensor([1.0, 1.0, 0.0]).unsqueeze(1)
        live_model = train_model(X, y, 8)
        inject_live_model(good, bad, live_model)

        # Ensure cache for a different label set
        history = [
            (0, "good", 1.0),
            (1, "bad", 2.0),
        ]

        _ensure_cache(clips, history, inclusion_value=0)

        # Step 1 should NOT use the injected model (different label set)
        step = _cached_steps[1]
        assert step["model"] is not live_model, "Should not use a live model with mismatched labels"

    def test_clear_cache_removes_live_models(self, client):
        """clear_progress_cache should also clear injected live models."""
        import numpy as np
        import torch

        from vtscore.detectors.labeling_progress import clear_progress_cache
        from vtscore.training.mlp import train_model

        rng = np.random.RandomState(44)
        X = torch.tensor(rng.randn(2, 8).astype(np.float32))
        y = torch.tensor([1.0, 0.0]).unsqueeze(1)
        model = train_model(X, y, 8)

        inject_live_model({0: None}, {1: None}, model)
        assert len(_live_models) == 1

        clear_progress_cache()
        assert len(_live_models) == 0

    def test_learned_sort_injects_model(self, client):
        """The /api/learned-sort endpoint should inject the trained model."""
        from vtscore.detectors.labeling_progress import clear_progress_cache

        clear_progress_cache()

        app_module.good_votes.update({k: None for k in [1, 2, 3]})
        app_module.bad_votes.update({k: None for k in [18, 19, 20]})

        resp = client.post("/api/learned-sort", json={"wait": True})
        assert resp.status_code == 200

        # A live model should have been injected for the current vote set
        key = (frozenset(app_module.good_votes), frozenset(app_module.bad_votes))
        assert key in _live_models, "learned-sort should inject the live model"
        # The model alone: its cross-calibrated threshold is deliberately not
        # carried over, so the cache derives every step's cutoff the same way.
        assert _live_models[key] is not None

    def test_live_model_stability_computed(self, client):
        """When a live model is reused, stability should still be computed."""
        import numpy as np
        import torch

        from vtscore.detectors.labeling_progress import clear_progress_cache
        from vtscore.training.mlp import train_model

        clear_progress_cache()

        rng = np.random.RandomState(42)
        clips = {}
        for i in range(20):
            clips[i] = {"embeddings": {"clap": rng.randn(8).astype(np.float32)}, "embedder": "clap"}

        # Build a history with enough steps to have prior predictions
        history = [
            (0, "good", 1.0),
            (1, "bad", 2.0),
            (2, "good", 3.0),
            (3, "bad", 4.0),
            (4, "good", 5.0),  # step 4: good={0,2,4}, bad={1,3}
        ]

        # Inject a live model for step 4's label set
        good = {0: None, 2: None, 4: None}
        bad = {1: None, 3: None}
        embs = [media_embedding(clips[i]) for i in [0, 2, 4, 1, 3]]
        X = torch.tensor(np.array(embs), dtype=torch.float32)
        y = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0]).unsqueeze(1)
        live_model = train_model(X, y, 8)
        inject_live_model(good, bad, live_model)

        _ensure_cache(clips, history, inclusion_value=0)

        # Step 4 should use the live model
        step4 = _cached_steps[4]
        assert step4["model"] is live_model
        # Stability should still be computed (not None) since prior predictions exist
        assert step4["stability"] is not None, "Stability should be computed even with live model"
