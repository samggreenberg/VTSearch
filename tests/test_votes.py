import app as app_module
from vtsearch.models.progress import (
    _cache_good_ids,
    _cache_bad_ids,
    _cached_steps,
    _compute_stable_status,
    _ensure_cache,
)
from vtsearch.utils import medias, label_history


class TestVoteClip:
    def test_vote_good(self, client):
        resp = client.post("/api/medias/1/vote", json={"vote": "good"})
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True
        assert 1 in app_module.good_votes

    def test_vote_bad(self, client):
        resp = client.post("/api/medias/1/vote", json={"vote": "bad"})
        assert resp.status_code == 200
        assert 1 in app_module.bad_votes

    def test_toggle_good_off(self, client):
        """Voting good twice should toggle it off."""
        client.post("/api/medias/1/vote", json={"vote": "good"})
        assert 1 in app_module.good_votes
        client.post("/api/medias/1/vote", json={"vote": "good"})
        assert 1 not in app_module.good_votes

    def test_toggle_bad_off(self, client):
        """Voting bad twice should toggle it off."""
        client.post("/api/medias/1/vote", json={"vote": "bad"})
        assert 1 in app_module.bad_votes
        client.post("/api/medias/1/vote", json={"vote": "bad"})
        assert 1 not in app_module.bad_votes

    def test_switch_from_good_to_bad(self, client):
        client.post("/api/medias/1/vote", json={"vote": "good"})
        client.post("/api/medias/1/vote", json={"vote": "bad"})
        assert 1 not in app_module.good_votes
        assert 1 in app_module.bad_votes

    def test_switch_from_bad_to_good(self, client):
        client.post("/api/medias/1/vote", json={"vote": "bad"})
        client.post("/api/medias/1/vote", json={"vote": "good"})
        assert 1 not in app_module.bad_votes
        assert 1 in app_module.good_votes

    def test_invalid_vote_value(self, client):
        resp = client.post("/api/medias/1/vote", json={"vote": "meh"})
        assert resp.status_code == 400
        assert "vote must be" in resp.get_json()["error"]

    def test_missing_vote_field(self, client):
        resp = client.post("/api/medias/1/vote", json={"wrong": "field"})
        assert resp.status_code == 400

    def test_vote_nonexistent_clip(self, client):
        resp = client.post("/api/medias/9999/vote", json={"vote": "good"})
        assert resp.status_code == 404

    def test_multiple_clips_independent_votes(self, client):
        client.post("/api/medias/1/vote", json={"vote": "good"})
        client.post("/api/medias/2/vote", json={"vote": "bad"})
        assert 1 in app_module.good_votes
        assert 2 in app_module.bad_votes
        assert 1 not in app_module.bad_votes
        assert 2 not in app_module.good_votes


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
        client.post("/api/medias/3/vote", json={"vote": "good"})
        client.post("/api/medias/5/vote", json={"vote": "bad"})
        resp = client.get("/api/votes")
        data = resp.get_json()
        assert 3 in data["good"]
        assert 5 in data["bad"]


class TestClearVotes:
    def test_clear_votes_empties_good_and_bad(self, client):
        """POST /api/votes/clear should remove all votes."""
        client.post("/api/medias/1/vote", json={"vote": "good"})
        client.post("/api/medias/2/vote", json={"vote": "bad"})
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
        client.post("/api/medias/1/vote", json={"vote": "good"})
        resp = client.post("/api/votes/clear")
        assert resp.status_code == 200
        assert len(medias) == num_before

    def test_get_votes_empty_after_clear(self, client):
        """GET /api/votes should return empty lists after clear."""
        client.post("/api/medias/1/vote", json={"vote": "good"})
        client.post("/api/medias/2/vote", json={"vote": "bad"})
        client.post("/api/votes/clear")
        resp = client.get("/api/votes")
        data = resp.get_json()
        assert data["good"] == []
        assert data["bad"] == []


class TestLabelHistory:
    def test_vote_adds_history_entry(self, client):
        client.post("/api/medias/1/vote", json={"vote": "good"})
        assert len(label_history) == 1
        assert label_history[0][0] == 1
        assert label_history[0][1] == "good"

    def test_toggle_off_adds_unlabel_history(self, client):
        """Toggling off a vote should record an 'unlabel' event."""
        client.post("/api/medias/1/vote", json={"vote": "good"})
        client.post("/api/medias/1/vote", json={"vote": "good"})
        assert len(label_history) == 2
        assert label_history[1][1] == "unlabel"

    def test_toggle_off_bad_adds_unlabel_history(self, client):
        client.post("/api/medias/1/vote", json={"vote": "bad"})
        client.post("/api/medias/1/vote", json={"vote": "bad"})
        assert len(label_history) == 2
        assert label_history[1][1] == "unlabel"

    def test_switch_vote_adds_new_label_history(self, client):
        """Switching good->bad should add a 'bad' entry, not 'unlabel'."""
        client.post("/api/medias/1/vote", json={"vote": "good"})
        client.post("/api/medias/1/vote", json={"vote": "bad"})
        assert len(label_history) == 2
        assert label_history[0][1] == "good"
        assert label_history[1][1] == "bad"

    def test_toggle_off_then_revote(self, client):
        """Toggle off then revote should produce 3 history entries."""
        client.post("/api/medias/1/vote", json={"vote": "good"})
        client.post("/api/medias/1/vote", json={"vote": "good"})  # toggle off
        client.post("/api/medias/1/vote", json={"vote": "bad"})  # revote bad
        assert len(label_history) == 3
        assert label_history[0][1] == "good"
        assert label_history[1][1] == "unlabel"
        assert label_history[2][1] == "bad"
        assert 1 in app_module.bad_votes
        assert 1 not in app_module.good_votes


class TestProgressCacheWithLabelChanges:
    """Verify the progress cache stays consistent when labels are changed."""

    def test_cache_removes_clip_on_unlabel(self, client):
        """After toggling off, the progress cache should not include the media."""
        client.post("/api/medias/1/vote", json={"vote": "good"})
        client.post("/api/medias/2/vote", json={"vote": "bad"})
        _ensure_cache(medias, label_history, 0)
        assert 1 in _cache_good_ids
        assert 2 in _cache_bad_ids

        # Toggle off media 1
        client.post("/api/medias/1/vote", json={"vote": "good"})
        _ensure_cache(medias, label_history, 0)
        assert 1 not in _cache_good_ids
        assert 1 not in _cache_bad_ids
        assert 2 in _cache_bad_ids

    def test_cache_handles_switch_vote(self, client):
        """Switching good->bad should update cache running sets correctly."""
        client.post("/api/medias/1/vote", json={"vote": "good"})
        client.post("/api/medias/2/vote", json={"vote": "bad"})
        _ensure_cache(medias, label_history, 0)
        assert 1 in _cache_good_ids

        # Switch media 1 from good to bad
        client.post("/api/medias/1/vote", json={"vote": "bad"})
        _ensure_cache(medias, label_history, 0)
        assert 1 not in _cache_good_ids
        assert 1 in _cache_bad_ids

    def test_cache_toggle_off_then_revote(self, client):
        """Toggle off then revote should leave cache in correct state."""
        client.post("/api/medias/1/vote", json={"vote": "good"})
        client.post("/api/medias/2/vote", json={"vote": "bad"})
        # Toggle off media 1
        client.post("/api/medias/1/vote", json={"vote": "good"})
        # Revote as bad
        client.post("/api/medias/1/vote", json={"vote": "bad"})
        _ensure_cache(medias, label_history, 0)
        assert 1 in _cache_bad_ids
        assert 1 not in _cache_good_ids

    def test_learned_sort_after_toggle_off(self, client):
        """Learned sort should work after toggling off a vote."""
        client.post("/api/medias/1/vote", json={"vote": "good"})
        client.post("/api/medias/2/vote", json={"vote": "bad"})
        resp = client.post("/api/learned-sort")
        assert resp.status_code == 200

        # Toggle off good vote, add a different good vote
        client.post("/api/medias/1/vote", json={"vote": "good"})
        client.post("/api/medias/3/vote", json={"vote": "good"})
        resp = client.post("/api/learned-sort")
        assert resp.status_code == 200

    def test_learned_sort_returns_400_after_toggling_all_good(self, client):
        """If all good votes are toggled off, learned sort should return 400."""
        client.post("/api/medias/1/vote", json={"vote": "good"})
        client.post("/api/medias/2/vote", json={"vote": "bad"})
        # Toggle off the only good vote
        client.post("/api/medias/1/vote", json={"vote": "good"})
        resp = client.post("/api/learned-sort")
        assert resp.status_code == 400

    def test_labeling_status_after_label_change(self, client):
        """labeling-status endpoint should not crash after label changes."""
        # Vote enough medias to get past the minimum threshold
        for i in range(1, 6):
            client.post(f"/api/medias/{i}/vote", json={"vote": "good"})
        for i in range(6, 11):
            client.post(f"/api/medias/{i}/vote", json={"vote": "bad"})

        resp = client.get("/api/labeling-status")
        assert resp.status_code == 200

        # Now toggle off a good vote and switch a bad vote
        client.post("/api/medias/1/vote", json={"vote": "good"})  # toggle off
        client.post("/api/medias/6/vote", json={"vote": "good"})  # switch bad->good

        resp = client.get("/api/labeling-status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["good_count"] == 5  # lost 1, gained 1
        assert data["bad_count"] == 4  # lost 1


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
        """Green requires practically zero flips — only the rare single flip tolerated."""
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
                None,  # model lost (gap — user unlabeled all bad)
                None,  # still no model
                None,  # first model step after gap — no prior model
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
        from vtsearch.models.progress import clear_progress_cache

        clear_progress_cache()

        # Build a minimal clips_dict with embeddings
        import numpy as np

        clips = {}
        for i in range(10):
            clips[i] = {"embedding": np.random.randn(8).astype(np.float32)}

        # Label history: vote good on 0, bad on 1, then vote good on 0 AGAIN
        # The third event doesn't change the training data (0 is already good).
        history = [
            (0, "good", 1.0),
            (1, "bad", 2.0),
            (0, "good", 3.0),  # no-op: 0 is already good
        ]

        _ensure_cache(clips, history, inclusion_value=0)

        assert len(_cached_steps) == 3
        # Steps 0 and 1 change the training data — they should attempt training.
        # Step 2 has the same good/bad IDs as step 1 — stability should be None.
        assert _cached_steps[2]["stability"] is None, (
            "Stability should not be recorded when training data didn't change"
        )

    def test_changed_training_data_records_stability(self, client):
        """When good/bad IDs DO change, stability should be computed (once a prior model exists)."""
        from vtsearch.models.progress import clear_progress_cache

        clear_progress_cache()

        import numpy as np

        clips = {}
        for i in range(20):
            clips[i] = {"embedding": np.random.randn(8).astype(np.float32)}

        # Each step changes the training data
        history = [
            (0, "good", 1.0),
            (1, "bad", 2.0),   # first model (both polarities)
            (2, "good", 3.0),  # new model
            (3, "bad", 4.0),   # new model
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
