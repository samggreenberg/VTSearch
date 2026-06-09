"""Tests for the Find verification workflow backend (Phase 1).

Covers:
- mark-verified on find-mode votes (and un-verify on un-vote)
- the ``verified`` array on ``GET /api/votes``
- ``label_filter=unverified`` / ``verified`` export partitioning
- ``GET /api/find/stats`` (2x2 confusion + FP/FN inclusion sweep)

See docs/plans/find-verification-workflow.md.
"""

from __future__ import annotations

import app as app_module
from vtscore.state.core import get_active_detector_context
from vtsearch.state import set_find_initial_labels, set_find_scores, set_vote


class TestMarkVerified:
    """A single-item vote in Find mode verifies the item; un-voting un-verifies."""

    def test_find_mode_vote_marks_verified(self):
        ctx = get_active_detector_context()
        ctx.find_mode = True
        set_vote(1, "good")
        assert 1 in ctx.verified_ids
        set_vote(1, "bad")
        assert 1 in ctx.verified_ids  # still verified (flipped, not un-voted)
        set_vote(1, "none")
        assert 1 not in ctx.verified_ids  # un-vote un-verifies

    def test_non_find_mode_vote_does_not_verify(self):
        ctx = get_active_detector_context()
        ctx.find_mode = False
        ctx.verified_ids.clear()
        set_vote(2, "good")
        assert 2 not in ctx.verified_ids

    def test_clear_votes_clears_verified(self, client):
        ctx = get_active_detector_context()
        ctx.find_mode = True
        set_vote(1, "good")
        assert 1 in ctx.verified_ids
        resp = client.post("/api/votes/clear")
        assert resp.status_code == 200
        assert dict(get_active_detector_context().verified_ids) == {}


class TestVotesVerifiedField:
    """``GET /api/votes`` exposes the verified ids."""

    def test_verified_array_present(self, client):
        ctx = get_active_detector_context()
        ctx.find_mode = True
        set_vote(1, "good")
        set_vote(2, "good")
        resp = client.get("/api/votes")
        data = resp.get_json()
        assert "verified" in data
        assert set(data["verified"]) == {1, 2}

    def test_verified_empty_outside_find_mode(self, client):
        ctx = get_active_detector_context()
        ctx.find_mode = False
        ctx.verified_ids.clear()
        app_module.good_votes[1] = None
        resp = client.get("/api/votes")
        assert resp.get_json()["verified"] == []


class TestUnverifiedExport:
    """``label_filter=unverified`` / ``verified`` partition by verified_ids."""

    def _setup(self):
        ctx = get_active_detector_context()
        # 1,2 unverified good; 3 verified good; 4 verified bad
        app_module.good_votes.update({1: None, 2: None, 3: None})
        app_module.bad_votes.update({4: None})
        ctx.verified_ids.clear()
        ctx.verified_ids.update({3: None, 4: None})

    def test_unverified_filter(self, client):
        self._setup()
        resp = client.get("/api/labels/export?label_filter=unverified")
        md5s = {e["md5"] for e in resp.get_json()["labels"]}
        assert md5s == {app_module.medias[1]["md5"], app_module.medias[2]["md5"]}

    def test_verified_filter(self, client):
        self._setup()
        resp = client.get("/api/labels/export?label_filter=verified")
        labels = resp.get_json()["labels"]
        by_md5 = {e["md5"]: e["label"] for e in labels}
        assert by_md5 == {
            app_module.medias[3]["md5"]: "good",
            app_module.medias[4]["md5"]: "bad",
        }


class TestFindStats:
    """``GET /api/find/stats`` 2x2 confusion + FP/FN inclusion sweep."""

    def _setup(self):
        ctx = get_active_detector_context()
        ctx.find_mode = True
        ctx.threshold = 0.5
        set_find_scores({1: 0.9, 2: 0.8, 3: 0.2, 4: 0.1})
        # Detector's call at the default cutoff.
        set_find_initial_labels({1: "good", 2: "good", 3: "bad", 4: "bad"})
        # Human: confirm 1 good; cull 2 (false positive) to bad; rescue 4 to good.
        app_module.good_votes.update({1: None, 4: None})
        app_module.bad_votes.update({2: None})
        ctx.verified_ids.clear()
        ctx.verified_ids.update({1: None, 2: None, 4: None})

    def test_confusion_counts(self, client):
        self._setup()
        data = client.get("/api/find/stats").get_json()
        assert data["verified_count"] == 3
        assert data["confirmed_good"] == 1  # id1
        assert data["confirmed_bad"] == 0
        assert data["culled_false_pos"] == 1  # id2
        assert data["rescued_false_neg"] == 1  # id4
        assert data["agreements"] == 1
        assert data["corrections"] == 2
        assert data["precision_on_reviewed"] == 0.5  # 1 / (1 + 1)

    def test_sweep_shape_and_values(self, client):
        self._setup()
        data = client.get("/api/find/stats").get_json()
        sweep = data["sweep"]
        assert len(sweep) == 21
        assert [p["inclusion"] for p in sweep] == list(range(-10, 11))
        # With no cached fold orderings, every point uses threshold 0.5:
        # verified-bad id2 (0.8) is a false positive; verified-good id4 (0.1)
        # is a false negative; id1 (0.9) is correctly above the line.
        for p in sweep:
            assert p["false_pos"] == 1
            assert p["false_neg"] == 1

    def test_empty_when_nothing_verified(self, client):
        ctx = get_active_detector_context()
        ctx.verified_ids.clear()
        data = client.get("/api/find/stats").get_json()
        assert data["verified_count"] == 0
        assert data["agreement_rate"] == 0.0
        assert data["precision_on_reviewed"] == 0.0
        assert len(data["sweep"]) == 21
