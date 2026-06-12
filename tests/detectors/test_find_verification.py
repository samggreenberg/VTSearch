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
from helpers import setup_trainable_model_in_registry
from tests import load_detector_and_wait
from vtscore.detectors.store import _detector_path, _read_detector
from vtscore.state.core import get_active_detector_context
from vtscore.state.votes import rethreshold_unverified_find_items
from vtsearch.state import set_find_initial_labels, set_find_scores, set_vote, snapshot_medias


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


class TestRethresholdUnverified:
    """Sliding the cutoff re-splits unverified items only; verified items hold."""

    def _setup(self):
        ctx = get_active_detector_context()
        ctx.find_mode = True
        ctx.threshold = 0.5
        set_find_scores({1: 0.9, 2: 0.8, 3: 0.2, 4: 0.1})
        # Initial split at 0.5: 1,2 good; 3,4 bad.  Human verified id1 (good).
        app_module.good_votes.clear()
        app_module.bad_votes.clear()
        app_module.good_votes.update({1: None, 2: None})
        app_module.bad_votes.update({3: None, 4: None})
        ctx.verified_ids.clear()
        ctx.verified_ids.update({1: None})
        return ctx

    def test_raise_cutoff_demotes_unverified(self):
        ctx = self._setup()
        ctx.threshold = 0.85  # now only id1 (0.9) clears it
        rethreshold_unverified_find_items()
        # id1 verified-good stays good even though... it still clears 0.85 anyway;
        # id2 (0.8) is unverified and now below the line -> bad.
        assert 1 in app_module.good_votes
        assert 2 in app_module.bad_votes
        assert 3 in app_module.bad_votes
        assert 4 in app_module.bad_votes

    def test_verified_item_holds_against_cutoff(self):
        ctx = self._setup()
        # Verify id2 as good, then raise the cutoff above its score.
        ctx.verified_ids.update({2: None})
        ctx.threshold = 0.85
        rethreshold_unverified_find_items()
        # id2 is verified-good; the cutoff must not demote it.
        assert 2 in app_module.good_votes
        assert 2 not in app_module.bad_votes

    def test_lower_cutoff_promotes_unverified(self):
        ctx = self._setup()
        ctx.threshold = 0.05  # everything clears it
        rethreshold_unverified_find_items()
        for cid in (1, 2, 3, 4):
            assert cid in app_module.good_votes

    def test_noop_outside_find_mode(self):
        ctx = self._setup()
        ctx.find_mode = False
        ctx.threshold = 0.85
        rethreshold_unverified_find_items()
        # No re-split: the initial 0.5 assignment stands.
        assert 2 in app_module.good_votes

    def test_inclusion_post_returns_threshold(self, client):
        resp = client.post("/api/inclusion", json={"inclusion": 0})
        assert resp.status_code == 200
        assert "threshold" in resp.get_json()


class TestFindStats:
    """``GET /api/find/stats`` over the ADOPTED label set (all items, with
    unverified flood-filled), plus the FP/FN inclusion sweep."""

    def _setup(self):
        ctx = get_active_detector_context()
        ctx.find_mode = True
        ctx.threshold = 0.5
        set_find_scores({1: 0.9, 2: 0.8, 3: 0.2, 4: 0.1})
        # Detector's call at the default cutoff (find-label labelled all four).
        set_find_initial_labels({1: "good", 2: "good", 3: "bad", 4: "bad"})
        # Human: confirm 1 good; cull 2 (false positive) to bad; rescue 4 to
        # good; leave 3 untouched (unverified bad).  Final adopted label set:
        app_module.good_votes.update({1: None, 4: None})
        app_module.bad_votes.update({2: None, 3: None})
        ctx.verified_ids.clear()
        ctx.verified_ids.update({1: None, 2: None, 4: None})

    def test_confusion_counts_over_all_items(self, client):
        self._setup()
        data = client.get("/api/find/stats").get_json()
        assert data["total_good"] == 2  # ids 1, 4
        assert data["total_bad"] == 2  # ids 2, 3
        assert data["verified_count"] == 3  # human checked 1, 2, 4
        assert data["confirmed_good"] == 1  # id1 (det good, adopted good)
        assert data["confirmed_bad"] == 1  # id3 (det bad, adopted bad - unverified)
        assert data["culled_false_pos"] == 1  # id2 (det good, adopted bad)
        assert data["rescued_false_neg"] == 1  # id4 (det bad, adopted good)
        assert data["agreements"] == 2
        assert data["corrections"] == 2
        assert data["agreement_rate"] == 0.5
        assert data["precision"] == 0.5  # confirmed_good 1 / (1 + culled_fp 1)

    def test_sweep_shape_and_values(self, client):
        self._setup()
        data = client.get("/api/find/stats").get_json()
        sweep = data["sweep"]
        assert len(sweep) == 21
        assert [p["inclusion"] for p in sweep] == list(range(-10, 11))
        # No cached fold orderings -> every point uses threshold 0.5.
        # Adopted-bad above the line: id2 (0.8) -> 1 FP. Adopted-good below it:
        # id4 (0.1) -> 1 FN. (id1=0.9 good above, id3=0.2 bad below: correct.)
        for p in sweep:
            assert p["false_pos"] == 1
            assert p["false_neg"] == 1

    def test_empty_when_no_votes(self, client):
        ctx = get_active_detector_context()
        ctx.verified_ids.clear()
        app_module.good_votes.clear()
        app_module.bad_votes.clear()
        data = client.get("/api/find/stats").get_json()
        assert data["total_good"] == 0
        assert data["total_bad"] == 0
        assert data["agreement_rate"] == 0.0
        assert data["precision"] == 0.0
        assert len(data["sweep"]) == 21


class TestCorrectionsToDetector:
    """``POST /api/find/corrections-to-detector`` folds the Find corrections
    into the active detector's labelset for future scoring while leaving the
    current Find session frozen (and flagged out of date)."""

    def _labelset_labels(self, name: str) -> list[dict]:
        """Return the on-disk labelset entries for detector *name*."""
        data = _read_detector(_detector_path(name))
        assert data is not None
        return data["labelset"]["labels"]

    def _setup_find(self, client):
        """Register + load a detector, then run find-label so it enters find
        mode with a full ``find_initial_labels`` baseline."""
        detector_id = setup_trainable_model_in_registry(
            "corrections-model",
            good_ids=[1, 2, 3],
            bad_ids=[18, 19, 20],
            snap=snapshot_medias(),
        )
        load_detector_and_wait(client, detector_id)
        resp = client.post("/api/find-label", json={"detector_id": detector_id})
        assert resp.status_code == 200, resp.get_json()
        return detector_id

    def _make_two_corrections(self, client):
        """Flip the detector's call on one good and one bad item; return the ids."""
        ctx = get_active_detector_context()
        initial = dict(ctx.find_initial_labels)
        good_item = next(cid for cid, lbl in initial.items() if lbl == "good")
        bad_item = next(cid for cid, lbl in initial.items() if lbl == "bad")
        assert client.post(f"/api/medias/{good_item}/vote", json={"target": "bad"}).status_code == 200
        assert client.post(f"/api/medias/{bad_item}/vote", json={"target": "good"}).status_code == 200
        return good_item, bad_item

    def test_no_find_run_returns_400(self, client):
        """Without a find-label baseline there are no corrections to take."""
        detector_id = setup_trainable_model_in_registry(
            "no-find-run",
            good_ids=[1, 2, 3],
            bad_ids=[18, 19, 20],
            snap=snapshot_medias(),
        )
        load_detector_and_wait(client, detector_id)
        resp = client.post("/api/find/corrections-to-detector")
        assert resp.status_code == 400

    def test_no_corrections_is_noop(self, client):
        """Right after find-label, every adopted label matches the detector's
        call, so there is nothing to add and the labelset is untouched."""
        self._setup_find(client)
        before = len(self._labelset_labels("corrections-model"))
        resp = client.post("/api/find/corrections-to-detector")
        assert resp.status_code == 200, resp.get_json()
        data = resp.get_json()
        assert data["corrections_added"] == 0
        after = len(self._labelset_labels("corrections-model"))
        assert after == before
        # Nothing changed, so the evaluation is not stale.
        assert get_active_detector_context().find_eval_stale is False

    def test_corrections_added_to_labelset(self, client):
        """Flipping the detector's call on two items folds those corrections
        into the labelset with the human label."""
        self._setup_find(client)
        good_item, bad_item = self._make_two_corrections(client)

        resp = client.post("/api/find/corrections-to-detector")
        assert resp.status_code == 200, resp.get_json()
        data = resp.get_json()
        assert data["corrections_added"] == 2

        labels = self._labelset_labels("corrections-model")
        by_md5 = {el["md5"]: el["label"] for el in labels}
        assert by_md5[app_module.medias[good_item]["md5"]] == "bad"
        assert by_md5[app_module.medias[bad_item]["md5"]] == "good"
        assert data["num_labels"] == len(labels)

    def test_session_frozen_and_marked_stale(self, client):
        """The Find session is NOT reset: votes / verified / find baseline hold,
        the cached MLP is invalidated for the next scoring pass, and the
        evaluation is flagged stale."""
        self._setup_find(client)
        good_item, bad_item = self._make_two_corrections(client)
        ctx = get_active_detector_context()
        initial_before = dict(ctx.find_initial_labels)
        scores_before = dict(ctx.find_scores)

        resp = client.post("/api/find/corrections-to-detector")
        assert resp.status_code == 200, resp.get_json()

        # Frozen: the find baseline, scores, and verifications are untouched.
        assert dict(ctx.find_initial_labels) == initial_before
        assert dict(ctx.find_scores) == scores_before
        assert good_item in ctx.verified_ids
        assert bad_item in ctx.verified_ids
        assert good_item in ctx.bad_votes  # the human vote held
        assert bad_item in ctx.good_votes
        # The cached MLP is dropped so the next scoring pass retrains.
        assert ctx.model is None
        assert ctx.find_eval_stale is True

    def test_stale_survives_rehydrate_and_shows_in_stats(self, client):
        """A follow-up request must not rehydrate the frozen votes away (the
        labelset write bumped the file mtime), and Stats reports ``stale``."""
        self._setup_find(client)
        self._make_two_corrections(client)
        assert client.post("/api/find/corrections-to-detector").status_code == 200

        # A later request runs before_request -> ensure_votes_match_active_dataset.
        # The cached-mtime re-point must keep it a no-op so the frozen eval holds.
        data = client.get("/api/find/stats").get_json()
        assert data["stale"] is True
        assert data["corrections"] == 2

    def test_fresh_find_label_clears_stale(self, client):
        """Re-scoring (a genuine new evaluation) clears the stale flag."""
        detector_id = self._setup_find(client)
        self._make_two_corrections(client)
        assert client.post("/api/find/corrections-to-detector").status_code == 200
        assert get_active_detector_context().find_eval_stale is True

        resp = client.post("/api/find-label", json={"detector_id": detector_id})
        assert resp.status_code == 200, resp.get_json()
        assert get_active_detector_context().find_eval_stale is False
        assert client.get("/api/find/stats").get_json()["stale"] is False
