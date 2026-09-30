"""Tests for the Find verification workflow backend (Phase 1).

Covers:
- mark-verified on find-mode votes (and un-verify on un-vote)
- the ``verified`` array on ``GET /api/votes``
- ``label_filter=unverified`` / ``verified`` export partitioning
- ``GET /api/find/stats`` (2x2 confusion, the floor's verdict, precision curve)
- verified votes surviving a re-score (issue #2928)
- a live Find session surviving a detector-file write (issue #2786)
"""

from __future__ import annotations

from tests.helpers import setup_trainable_model_in_registry
from tests import load_detector_and_wait
from vtscore.detectors.dataset_sync import reset_mtime_cache_for_tests
from vtscore.detectors.store import _detector_path, _read_detector, _write_detector
from vtscore.state.core import get_active_detector_context
from vtscore.state.votes import rethreshold_unverified_find_items
from vtsearch.state import (
    bad_votes,
    good_votes,
    medias,
    set_find_initial_labels,
    set_find_scores,
    set_vote,
    snapshot_medias,
)


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
        good_votes[1] = None
        resp = client.get("/api/votes")
        assert resp.get_json()["verified"] == []


class TestUnverifiedExport:
    """``label_filter=unverified`` / ``verified`` partition by verified_ids."""

    def _setup(self):
        ctx = get_active_detector_context()
        # 1,2 unverified good; 3 verified good; 4 verified bad
        good_votes.update({1: None, 2: None, 3: None})
        bad_votes.update({4: None})
        ctx.verified_ids.clear()
        ctx.verified_ids.update({3: None, 4: None})

    def test_unverified_filter(self, client):
        self._setup()
        resp = client.get("/api/labels/export?label_filter=unverified")
        md5s = {e["md5"] for e in resp.get_json()["labels"]}
        assert md5s == {medias[1]["md5"], medias[2]["md5"]}

    def test_verified_filter(self, client):
        self._setup()
        resp = client.get("/api/labels/export?label_filter=verified")
        labels = resp.get_json()["labels"]
        by_md5 = {e["md5"]: e["label"] for e in labels}
        assert by_md5 == {
            medias[3]["md5"]: "good",
            medias[4]["md5"]: "bad",
        }


class TestRethresholdUnverified:
    """Sliding the cutoff re-splits unverified items only; verified items hold."""

    def _setup(self):
        ctx = get_active_detector_context()
        ctx.find_mode = True
        ctx.threshold = 0.5
        set_find_scores({1: 0.9, 2: 0.8, 3: 0.2, 4: 0.1})
        # Initial split at 0.5: 1,2 good; 3,4 bad.  Human verified id1 (good).
        good_votes.clear()
        bad_votes.clear()
        good_votes.update({1: None, 2: None})
        bad_votes.update({3: None, 4: None})
        ctx.verified_ids.clear()
        ctx.verified_ids.update({1: None})
        return ctx

    def test_raise_cutoff_demotes_unverified(self):
        ctx = self._setup()
        ctx.threshold = 0.85  # now only id1 (0.9) clears it
        rethreshold_unverified_find_items()
        # id1 verified-good stays good even though... it still clears 0.85 anyway;
        # id2 (0.8) is unverified and now below the line -> bad.
        assert 1 in good_votes
        assert 2 in bad_votes
        assert 3 in bad_votes
        assert 4 in bad_votes

    def test_verified_item_holds_against_cutoff(self):
        ctx = self._setup()
        # Verify id2 as good, then raise the cutoff above its score.
        ctx.verified_ids.update({2: None})
        ctx.threshold = 0.85
        rethreshold_unverified_find_items()
        # id2 is verified-good; the cutoff must not demote it.
        assert 2 in good_votes
        assert 2 not in bad_votes

    def test_lower_cutoff_promotes_unverified(self):
        ctx = self._setup()
        ctx.threshold = 0.05  # everything clears it
        rethreshold_unverified_find_items()
        for cid in (1, 2, 3, 4):
            assert cid in good_votes

    def test_noop_outside_find_mode(self):
        ctx = self._setup()
        ctx.find_mode = False
        ctx.threshold = 0.85
        rethreshold_unverified_find_items()
        # No re-split: the initial 0.5 assignment stands.
        assert 2 in good_votes

    def test_floor_post_returns_threshold(self, client):
        resp = client.post("/api/min-precision", json={"min_precision": 0.5})
        assert resp.status_code == 200
        assert "threshold" in resp.get_json()


class TestFindStats:
    """``GET /api/find/stats`` over the ADOPTED label set (all items, with
    unverified flood-filled), the floor's verdict on the line (#4246), and the
    precision curve against the number returned (#4242)."""

    def _setup(self):
        ctx = get_active_detector_context()
        ctx.find_mode = True
        ctx.threshold = 0.5
        set_find_scores({1: 0.9, 2: 0.8, 3: 0.2, 4: 0.1})
        # Detector's call at the default cutoff (find-label labelled all four).
        set_find_initial_labels({1: "good", 2: "good", 3: "bad", 4: "bad"})
        # Human: confirm 1 good; cull 2 (false positive) to bad; rescue 4 to
        # good; leave 3 untouched (unverified bad).  Final adopted label set:
        good_votes.update({1: None, 4: None})
        bad_votes.update({2: None, 3: None})
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
        # Kept rate over the CHECKED items the detector called good (ids 1, 2):
        # id1 kept, id2 culled.
        assert data["verified_called_good"] == 2
        assert data["verified_kept_good"] == 1
        assert data["verified_precision"] == 0.5

    def test_kept_rate_ignores_unchecked_matches(self, client):
        """Unchecked items above the line are not counted as right (#4242)."""
        self._setup()
        # Ten more detector-good items nobody checked: the old rate read 11/12.
        extra = {cid: 0.7 for cid in range(5, 15) if cid in medias}
        set_find_scores({1: 0.9, 2: 0.8, 3: 0.2, 4: 0.1, **extra})
        set_find_initial_labels({1: "good", 2: "good", 3: "bad", 4: "bad", **{cid: "good" for cid in extra}})
        good_votes.update({cid: None for cid in extra})
        data = client.get("/api/find/stats").get_json()
        assert data["verified_called_good"] == 2
        assert data["verified_precision"] == 0.5

    def test_floor_rides_with_the_line(self, client):
        """The chart marks the floor and says whether the line keeps it (#4246, #4272)."""
        self._setup()
        client.post("/api/min-precision", json={"min_precision": 0.75})
        data = client.get("/api/find/stats").get_json()
        assert data["floor"] == {
            "min_precision": 0.75,
            "status": "unchecked",
            "count": 32,
            "range": None,
            "schedule": {"candidate": 32, "rounds": 1, "picks": 11},
        }
        # The sweep went with the Inclusion stepper.
        assert "sweep" not in data
        assert "inclusion" not in data

    def test_empty_when_no_votes(self, client):
        ctx = get_active_detector_context()
        ctx.verified_ids.clear()
        good_votes.clear()
        bad_votes.clear()
        data = client.get("/api/find/stats").get_json()
        assert data["total_good"] == 0
        assert data["total_bad"] == 0
        assert data["agreement_rate"] == 0.0
        assert data["verified_precision"] is None
        assert data["verified_called_good"] == 0
        assert data["verified_kept_good"] == 0

    def test_precision_curve_over_the_ranking(self, client):
        """Each point is the top k by score, with verified precision over what was checked in it."""
        self._setup()
        data = client.get("/api/find/stats").get_json()
        assert data["n_scored"] == 4
        assert data["n_returned"] == 2  # 0.9 and 0.8 clear 0.5
        curve = data["precision_curve"]
        assert [p["n_returned"] for p in curve] == [1, 2, 3, 4]
        assert [p["threshold"] for p in curve] == [0.9, 0.8, 0.2, 0.1]
        # Ranked 1, 2, 3, 4; checked 1 (good), 2 (bad), 4 (good); 3 unchecked.
        assert [(p["checked"], p["checked_good"]) for p in curve] == [(1, 1), (2, 1), (2, 1), (3, 2)]
        assert [p["verified_precision"] for p in curve] == [1.0, 0.5, 0.5, 0.6667]

    def test_the_curve_carries_no_estimate(self, client):
        """No model-based estimate reaches the chart (#4360).

        The #4220 estimator breaks most of its "at least" promises once its
        reference pool is consistent (#4256), so the curve is verified
        precision alone and the only range is the spot check's.
        """
        self._setup()
        data = client.get("/api/find/stats").get_json()
        assert not {"estimate_status", "calibration_positives", "min_calibration_positives"} & data.keys()
        assert all(
            set(p) == {"n_returned", "threshold", "checked", "checked_good", "verified_precision"}
            for p in data["precision_curve"]
        )


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
        assert by_md5[medias[good_item]["md5"]] == "bad"
        assert by_md5[medias[bad_item]["md5"]] == "good"
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

    def test_correction_supersedes_an_md5_matched_entry(self, client):
        """A correction replaces the media's prior entry even when that entry
        names the file by a different origin (issue #3174).

        Matching on ``element_key`` alone left the stale entry in place, so the
        labelset ended up holding the same media twice with contradicting
        labels - and training saw both bags.
        """
        self._setup_find(client)

        # Replace one media's entry with a stale, md5-only record of the same
        # bytes: same file, different provenance, so ``element_key`` differs.
        stale_id = next(iter(medias))
        stale_md5 = medias[stale_id]["md5"]
        path = _detector_path("corrections-model")
        data = _read_detector(path)
        assert data is not None
        data["labelset"]["labels"] = [el for el in data["labelset"]["labels"] if el.get("md5") != stale_md5] + [
            {"md5": stale_md5, "label": "good", "origin_name": "elsewhere.wav"}
        ]
        _write_detector(path, data)

        # Flip the detector's call on that media so it becomes a correction.
        ctx = get_active_detector_context()
        opposite = "bad" if ctx.find_initial_labels.get(stale_id) == "good" else "good"
        assert client.post(f"/api/medias/{stale_id}/vote", json={"target": opposite}).status_code == 200

        assert client.post("/api/find/corrections-to-detector").status_code == 200

        entries = [el for el in self._labelset_labels("corrections-model") if el.get("md5") == stale_md5]
        assert len(entries) == 1, f"labelset stored the same media twice: {entries}"
        assert entries[0]["label"] == opposite


class TestReScoreKeepsVerifiedVotes:
    """Re-scoring must not overwrite a human-verified vote (#2928).

    ``POST /api/find-label`` re-applies the detector's call to every item, and
    the fold-corrections -> retrain -> re-score loop re-runs it on purpose.  The
    bulk apply used to reassign *every* vote from the new threshold split while
    nothing cleared ``verified_ids``, so an item the human had ruled on came
    back carrying the machine's opposite label - excluded from the work queue,
    counted in ``verified_count``, and pinned there by the floor's
    re-threshold - i.e. the human's decision silently inverted while still
    presented as human-verified.
    """

    DETECTOR = "rescore-verified"

    def _setup_find(self, client):
        detector_id = setup_trainable_model_in_registry(
            self.DETECTOR,
            good_ids=[1, 2, 3],
            bad_ids=[18, 19, 20],
            snap=snapshot_medias(),
        )
        load_detector_and_wait(client, detector_id)
        resp = client.post("/api/find-label", json={"detector_id": detector_id})
        assert resp.status_code == 200, resp.get_json()
        return detector_id, resp.get_json()

    def _rescue_lowest_scored(self, client, data):
        """Verify the detector's most confident Bad as Good; return its id.

        The lowest-scored item stays below the cutoff on any re-score, so the
        machine's call on the next pass reliably disagrees with the human's.
        """
        lowest = min(data["results"], key=lambda r: r["score"])
        assert lowest["score"] < data["threshold"]
        cid = lowest["id"]
        assert client.post(f"/api/medias/{cid}/vote", json={"target": "good"}).status_code == 200
        ctx = get_active_detector_context()
        assert cid in ctx.verified_ids and cid in ctx.good_votes
        return cid

    def test_verified_vote_survives_a_re_score(self, client):
        detector_id, data = self._setup_find(client)
        cid = self._rescue_lowest_scored(client, data)
        click_before = get_active_detector_context().vote_click_times[cid]

        assert client.post("/api/find-label", json={"detector_id": detector_id}).status_code == 200

        ctx = get_active_detector_context()
        assert cid in ctx.good_votes, "the re-score overwrote the human's vote"
        assert cid not in ctx.bad_votes
        assert cid in ctx.verified_ids, "the item lost its verified marker"
        assert ctx.vote_click_times[cid] == click_before, "the human's click-time was re-stamped"

    def test_re_score_still_relabels_unverified_items(self, client):
        """The guard is narrow: everything unverified adopts the new pass."""
        detector_id, data = self._setup_find(client)
        self._rescue_lowest_scored(client, data)

        resp = client.post("/api/find-label", json={"detector_id": detector_id})
        assert resp.status_code == 200, resp.get_json()
        fresh = resp.get_json()
        ctx = get_active_detector_context()
        for entry in fresh["results"]:
            if entry["id"] in ctx.verified_ids:
                continue
            assert (entry["id"] in ctx.good_votes) == (entry["score"] >= fresh["threshold"])

    def test_re_score_reports_the_adopted_counts(self, client):
        """``good_count`` / ``bad_count`` describe the labels actually applied."""
        detector_id, data = self._setup_find(client)
        self._rescue_lowest_scored(client, data)

        body = client.post("/api/find-label", json={"detector_id": detector_id}).get_json()
        ctx = get_active_detector_context()
        assert body["good_count"] == len(ctx.good_votes)
        assert body["bad_count"] == len(ctx.bad_votes)

    def test_disagreement_reads_as_a_correction_in_stats(self, client):
        """The machine's call still seeds the eval baseline, so a held human
        vote the re-scored detector disagrees with shows up as a correction
        rather than quietly agreeing with itself."""
        detector_id, data = self._setup_find(client)
        cid = self._rescue_lowest_scored(client, data)

        assert client.post("/api/find-label", json={"detector_id": detector_id}).status_code == 200

        assert get_active_detector_context().find_initial_labels[cid] == "bad"
        stats = client.get("/api/find/stats").get_json()
        assert stats["rescued_false_neg"] >= 1
        assert stats["corrections"] >= 1


class TestFindSessionSurvivesDetectorFileWrite:
    """A detector-file write must not discard a live Find session (#2786).

    ``ensure_votes_match_active_dataset`` (the ``before_request`` rehydrate)
    re-derives the cid-keyed votes from the on-disk labelset whenever the
    detector file's mtime moves.  For a Find session that is wrong: the votes
    are the detector's per-item calls over the active dataset, not training
    labels, and ``find_scores`` / ``find_initial_labels`` / ``verified_ids``
    live only in memory.  Rehydrating replaced the scoring output with the
    detector's *training* labels and un-verified everything the human had
    confirmed, while the client's ranking and cutoff - which nothing refreshes -
    kept showing the run that produced them.  The user-visible symptom was the
    right panel reporting N presumed-good against a left panel showing a
    completely different number above the cutoff.
    """

    DETECTOR = "find-session-write"

    def _setup_find(self, client):
        detector_id = setup_trainable_model_in_registry(
            self.DETECTOR,
            good_ids=[1, 2, 3],
            bad_ids=[18, 19, 20],
            snap=snapshot_medias(),
        )
        load_detector_and_wait(client, detector_id)
        resp = client.post("/api/find-label", json={"detector_id": detector_id})
        assert resp.status_code == 200, resp.get_json()
        return detector_id, resp.get_json()

    def _touch_detector_file(self):
        """Rewrite the detector file byte-identically, so only its mtime moves.

        Stands in for every real writer that can fire mid-session: a
        labelset-source pull, a second tab, an external edit.
        """
        path = _detector_path(self.DETECTOR)
        data = _read_detector(path)
        assert data is not None
        _write_detector(path, data)
        reset_mtime_cache_for_tests()

    def test_write_does_not_wipe_find_session(self, client):
        _, data = self._setup_find(client)
        above = [r["id"] for r in data["results"] if r["score"] >= data["threshold"]]
        assert above, "detector labelled nothing good; the fixture can't exercise this"

        # The human verifies one item, as a vote in the Find view does.
        assert client.post(f"/api/medias/{above[0]}/vote", json={"target": "good"}).status_code == 200

        before = client.get("/api/votes").get_json()
        # Baseline: the right panel's pile and the left panel's above-cutoff
        # ranking are the same set.
        assert set(before["good"]) == set(above)
        assert before["verified"] == [above[0]]

        self._touch_detector_file()

        after = client.get("/api/votes").get_json()
        assert set(after["good"]) == set(above), (
            "the Find session was rehydrated away: the right panel now reports the "
            "detector's training labels while the left panel still draws the Find cutoff"
        )
        assert after["verified"] == [above[0]], "the human's verification was forgotten"

        ctx = get_active_detector_context()
        assert ctx.find_mode is True
        assert len(ctx.find_scores) == len(data["results"])
        assert len(ctx.find_initial_labels) == len(data["results"])

    def test_write_still_refreshes_the_cached_labelset(self, client):
        """Preserving the session must not mean going blind to the file.

        The labelset cache is re-pointed at the new bytes, so the next reader
        sees them and the rehydrate pre-check stops re-firing.
        """
        self._setup_find(client)
        path = _detector_path(self.DETECTOR)

        data = _read_detector(path)
        assert data is not None
        data["labelset"]["labels"] = data["labelset"]["labels"][:2]
        _write_detector(path, data)
        reset_mtime_cache_for_tests()

        assert client.get("/api/votes").status_code == 200
        ctx = get_active_detector_context()
        assert len(ctx.cached_labelset) == 2
        assert ctx.cached_labelset_mtime == path.stat().st_mtime
        assert ctx.labelset_good_count + ctx.labelset_bad_count == 2

    def test_dataset_switch_still_rehydrates(self, client):
        """The guard is narrow: only an *unchanged* dataset protects the session.

        Across a dataset switch the cids really are meaningless, so the
        rehydrate must still run and the Find session must still go.
        """
        _, data = self._setup_find(client)
        ctx = get_active_detector_context()
        assert ctx.find_mode is True

        ctx.votes_dataset_id = "a-different-dataset"
        self._touch_detector_file()

        assert client.get("/api/votes").status_code == 200
        ctx = get_active_detector_context()
        assert ctx.find_mode is False
        assert len(ctx.find_scores) == 0
        assert len(ctx.good_votes) < len(data["results"])

    def test_clear_votes_clears_find_mode(self, client):
        """``/api/votes/clear`` ends the session, so the flag must clear too.

        Otherwise a stale ``find_mode`` would keep the rehydrate guard engaged
        (and labelset write-back suppressed) for votes that are now ordinary
        training labels.
        """
        self._setup_find(client)
        assert get_active_detector_context().find_mode is True
        assert client.post("/api/votes/clear").status_code == 200
        assert get_active_detector_context().find_mode is False
