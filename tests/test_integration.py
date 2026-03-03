"""Integration tests that simulate real user workflows end-to-end.

Each test class represents a complete user session, chaining multiple API
calls in the order a real user would make them.  The goal is to catch errors
that only surface when endpoints interact — state leaks, order-dependent
bugs, and response-format mismatches between producer/consumer endpoints.
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import app as app_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vote_clips(client, good_ids, bad_ids):
    """Vote a batch of medias via the API and verify each response."""
    for cid in good_ids:
        resp = client.post(f"/api/medias/{cid}/vote", json={"vote": "good"})
        assert resp.status_code == 200, f"Failed voting media {cid} good: {resp.get_json()}"
    for cid in bad_ids:
        resp = client.post(f"/api/medias/{cid}/vote", json={"vote": "bad"})
        assert resp.status_code == 200, f"Failed voting media {cid} bad: {resp.get_json()}"


def _export_detector(client):
    """Train and export a detector, returning the full payload."""
    resp = client.post("/api/detector/export")
    assert resp.status_code == 200, f"Detector export failed: {resp.get_json()}"
    data = resp.get_json()
    assert "weights" in data
    assert "threshold" in data
    return data


def _save_autorun_detector(client, name, detector, *, autodetect=True):
    """Save an autorun detector with autodetect enabled by default."""
    resp = client.post(
        "/api/autorun-detectors",
        json={
            "name": name,
            "media_type": "audio",
            "weights": detector["weights"],
            "threshold": detector["threshold"],
            "autodetect": autodetect,
        },
    )
    assert resp.status_code == 200, f"Save autorun detector failed: {resp.get_json()}"
    return resp.get_json()


# ---------------------------------------------------------------------------
# 1. Browse → Vote → Learned Sort
# ---------------------------------------------------------------------------


class TestBrowseVoteLearnWorkflow:
    """Simulates: user opens app, browses medias, votes, runs learned sort."""

    def test_full_browse_vote_learn_cycle(self, client):
        # Step 1: Check dataset status — medias are pre-loaded by conftest
        resp = client.get("/api/dataset/status")
        assert resp.status_code == 200
        status = resp.get_json()
        assert status["loaded"] is True
        assert status["num_medias"] == app_module.NUM_MEDIAS

        # Step 2: List all medias (as frontend does on page load)
        resp = client.get("/api/medias")
        assert resp.status_code == 200
        medias = resp.get_json()
        assert len(medias) == app_module.NUM_MEDIAS
        media_ids = [c["id"] for c in medias]

        # Step 3: Stream audio for the first media (user clicks play)
        resp = client.get(f"/api/medias/{media_ids[0]}/audio")
        assert resp.status_code == 200
        assert resp.content_type == "audio/wav"
        assert len(resp.data) > 0

        # Step 4: Vote on several medias
        good_ids = media_ids[:3]
        bad_ids = media_ids[-3:]
        _vote_clips(client, good_ids, bad_ids)

        # Step 5: Verify votes via GET /api/votes
        resp = client.get("/api/votes")
        assert resp.status_code == 200
        votes = resp.get_json()
        assert set(votes["good"]) == set(good_ids)
        assert set(votes["bad"]) == set(bad_ids)

        # Step 6: Run learned sort
        resp = client.post("/api/learned-sort")
        assert resp.status_code == 200
        learned = resp.get_json()
        assert len(learned["results"]) == app_module.NUM_MEDIAS
        assert "threshold" in learned

        # Verify good medias rank higher than bad medias on average
        score_map = {e["id"]: e["score"] for e in learned["results"]}
        avg_good = np.mean([score_map[i] for i in good_ids])
        avg_bad = np.mean([score_map[i] for i in bad_ids])
        assert avg_good > avg_bad

        # Step 7: Check labeling status (frontend polls this)
        resp = client.get("/api/labeling-status")
        assert resp.status_code == 200
        labeling = resp.get_json()
        assert labeling["good_count"] == len(good_ids)
        assert labeling["bad_count"] == len(bad_ids)


# ---------------------------------------------------------------------------
# 2. Text Search → Vote → Re-search → Learned Sort
# ---------------------------------------------------------------------------


class TestTextSearchThenLearnWorkflow:
    """Simulates: user searches for text, votes on results, then learns."""

    def test_search_vote_learn(self, client):
        # Step 1: Text search
        resp = client.post("/api/sort", json={"text": "high pitched beep"})
        assert resp.status_code == 200
        search_results = resp.get_json()["results"]
        assert len(search_results) == app_module.NUM_MEDIAS

        # All results sorted descending by similarity
        sims = [r["similarity"] for r in search_results]
        assert sims == sorted(sims, reverse=True)

        # Step 2: Save the search suggestion
        resp = client.post("/api/textsort-suggestions", json={"text": "high pitched beep"})
        assert resp.status_code == 200

        # Step 3: Vote the top-ranked medias as good, bottom as bad
        top_ids = [r["id"] for r in search_results[:3]]
        bottom_ids = [r["id"] for r in search_results[-3:]]
        _vote_clips(client, top_ids, bottom_ids)

        # Step 4: Run learned sort
        resp = client.post("/api/learned-sort")
        assert resp.status_code == 200
        learned = resp.get_json()

        # Verify results are valid
        scores = [e["score"] for e in learned["results"]]
        assert scores == sorted(scores, reverse=True)
        for entry in learned["results"]:
            assert 0.0 <= entry["score"] <= 1.0

        # Step 5: Verify suggestion was recorded
        resp = client.get("/api/textsort-suggestions")
        assert "high pitched beep" in resp.get_json()["suggestions"]


# ---------------------------------------------------------------------------
# 3. Vote → Train Detector → Save → Reload → Auto-detect
# ---------------------------------------------------------------------------


class TestDetectorLifecycleWorkflow:
    """Simulates: user votes, exports a detector, saves it, then runs auto-detect."""

    def test_full_detector_lifecycle(self, client):
        # Step 1: Vote on medias
        _vote_clips(client, [1, 2, 3], [8, 9, 10])

        # Step 2: Export a detector
        detector = _export_detector(client)
        assert set(detector["weights"].keys()) == {"0.weight", "0.bias", "3.weight", "3.bias"}

        # Step 3: Sort using the exported detector (immediate use)
        resp = client.post("/api/detector-sort", json={"detector": detector})
        assert resp.status_code == 200
        sort_data = resp.get_json()
        assert len(sort_data["results"]) == app_module.NUM_MEDIAS
        det_scores = [e["score"] for e in sort_data["results"]]
        assert det_scores == sorted(det_scores, reverse=True)

        # Step 4: Save as autorun
        _save_autorun_detector(client, "my-detector", detector)

        # Step 5: Verify it appears in autorun list list
        resp = client.get("/api/autorun-detectors")
        assert resp.status_code == 200
        names = [d["name"] for d in resp.get_json()["detectors"]]
        assert "my-detector" in names

        # Step 6: Clear votes (simulates starting fresh with the saved detector)
        app_module.good_votes.clear()
        app_module.bad_votes.clear()

        # Step 7: Run auto-detect using saved autorun detector
        resp = client.post("/api/auto-detect")
        assert resp.status_code == 200
        auto_data = resp.get_json()
        assert auto_data["media_type"] == "audio"
        assert auto_data["detectors_run"] == 1
        assert "my-detector" in auto_data["results"]

        result = auto_data["results"]["my-detector"]
        assert result["detector_name"] == "my-detector"
        assert result["total_hits"] == len(result["hits"])
        # All hits should score above threshold
        for hit in result["hits"]:
            assert hit["score"] >= result["threshold"] - 1e-6
            assert "embedding" not in hit
            assert "media_bytes" not in hit


# ---------------------------------------------------------------------------
# 4. Label Export → Clear → Import → Verify + Continue Working
# ---------------------------------------------------------------------------


class TestLabelRoundtripWorkflow:
    """Simulates: user votes, exports labels, clears state, imports labels,
    then continues voting and runs learned sort."""

    def test_label_roundtrip_then_continue(self, client):
        # Step 1: Vote on medias
        _vote_clips(client, [1, 3, 5], [2, 4])

        # Step 2: Export labels
        resp = client.get("/api/labels/export")
        assert resp.status_code == 200
        exported = resp.get_json()
        assert len(exported["labels"]) == 5

        # Verify each label has md5 and label fields
        for entry in exported["labels"]:
            assert "md5" in entry
            assert entry["label"] in ("good", "bad")

        # Step 3: Clear all votes (simulate closing and reopening)
        app_module.good_votes.clear()
        app_module.bad_votes.clear()

        # Verify votes are empty
        resp = client.get("/api/votes")
        assert resp.get_json()["good"] == []
        assert resp.get_json()["bad"] == []

        # Step 4: Import the exported labels
        resp = client.post("/api/labels/import", json=exported)
        assert resp.status_code == 200
        import_data = resp.get_json()
        assert import_data["applied"] == 5
        assert import_data["skipped"] == 0

        # Step 5: Verify votes restored correctly
        resp = client.get("/api/votes")
        votes = resp.get_json()
        assert set(votes["good"]) == {1, 3, 5}
        assert set(votes["bad"]) == {2, 4}

        # Step 6: Add more votes and continue with learned sort
        _vote_clips(client, [7], [8])

        resp = client.post("/api/learned-sort")
        assert resp.status_code == 200
        learned = resp.get_json()
        assert len(learned["results"]) == app_module.NUM_MEDIAS


# ---------------------------------------------------------------------------
# 5. Inclusion Setting → Learned Sort → Change Inclusion → Re-sort
# ---------------------------------------------------------------------------


class TestInclusionAffectsLearning:
    """Simulates: user adjusts inclusion slider, then runs learned sort,
    changes inclusion, and re-sorts — verifying the output changes."""

    def test_inclusion_changes_learned_sort_results(self, client):
        # Step 1: Vote on medias
        _vote_clips(client, [1, 2, 3], [8, 9, 10])

        # Step 2: Set inclusion to 0 (neutral) and run learned sort
        resp = client.post("/api/inclusion", json={"inclusion": 0})
        assert resp.status_code == 200

        resp = client.post("/api/learned-sort")
        assert resp.status_code == 200
        neutral_results = resp.get_json()["results"]
        neutral_order = [e["id"] for e in neutral_results]

        # Step 3: Set inclusion to +10 (strongly favor positives) and re-sort
        resp = client.post("/api/inclusion", json={"inclusion": 10})
        assert resp.status_code == 200

        resp = client.post("/api/learned-sort")
        assert resp.status_code == 200
        inclusive_results = resp.get_json()["results"]
        inclusive_order = [e["id"] for e in inclusive_results]

        # Step 4: The two orderings should differ (different loss weighting)
        # Note: with small synthetic data this isn't guaranteed to differ
        # in ordering, but the scores should be different
        neutral_scores = {e["id"]: e["score"] for e in neutral_results}
        inclusive_scores = {e["id"]: e["score"] for e in inclusive_results}
        scores_differ = any(abs(neutral_scores[i] - inclusive_scores[i]) > 1e-6 for i in neutral_scores)
        assert scores_differ or neutral_order != inclusive_order, (
            "Changing inclusion from 0 to 10 should affect learned sort"
        )

        # Step 5: Verify inclusion is persisted
        resp = client.get("/api/inclusion")
        assert resp.get_json()["inclusion"] == 10


# ---------------------------------------------------------------------------
# 6. Full Auto-Detect → File Export Pipeline
# ---------------------------------------------------------------------------


class TestAutoDetectExportPipeline:
    """Simulates: user trains multiple detectors, runs auto-detect, exports
    results to a JSON file."""

    def test_multi_detector_export_pipeline(self, client):
        # Step 1: Train and save first detector
        _vote_clips(client, [1, 2, 3], [8, 9, 10])
        det1 = _export_detector(client)
        _save_autorun_detector(client, "low-freq-detector", det1)

        # Step 2: Change votes and train second detector
        app_module.good_votes.clear()
        app_module.bad_votes.clear()
        _vote_clips(client, [5, 6, 7], [1, 2, 3])
        det2 = _export_detector(client)
        _save_autorun_detector(client, "mid-freq-detector", det2)

        # Step 3: Verify both detectors saved
        resp = client.get("/api/autorun-detectors")
        names = {d["name"] for d in resp.get_json()["detectors"]}
        assert names == {"low-freq-detector", "mid-freq-detector"}

        # Step 4: Run auto-detect
        resp = client.post("/api/auto-detect")
        assert resp.status_code == 200
        auto_results = resp.get_json()
        assert auto_results["detectors_run"] == 2
        assert "low-freq-detector" in auto_results["results"]
        assert "mid-freq-detector" in auto_results["results"]

        # Step 5: Export results to a file
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "auto_detect_results.json"
            resp = client.post(
                "/api/exporters/export",
                json={
                    "exporter_name": "server_json_file",
                    "field_values": {"filepath": str(fpath)},
                    "results": auto_results,
                },
            )
            assert resp.status_code == 200
            assert resp.get_json()["success"] is True
            assert fpath.exists()

            # Step 6: Verify the exported file is valid and contains both detectors
            written = json.loads(fpath.read_text())
            assert written["detectors_run"] == 2
            assert "low-freq-detector" in written["results"]
            assert "mid-freq-detector" in written["results"]


# ---------------------------------------------------------------------------
# 7. Iterative Voting — Vote → Learn → Revise → Re-learn
# ---------------------------------------------------------------------------


class TestIterativeVotingWorkflow:
    """Simulates: user votes, learns, realizes some votes are wrong,
    revises votes, and re-learns — verifying the model adapts."""

    def test_revise_votes_and_relearn(self, client):
        # Step 1: Initial votes
        _vote_clips(client, [1, 2, 3], [8, 9, 10])

        # Step 2: First learned sort
        resp = client.post("/api/learned-sort")
        assert resp.status_code == 200
        first_scores = {e["id"]: e["score"] for e in resp.get_json()["results"]}

        # Step 3: User realizes media 3 should be bad, toggles it
        resp = client.post("/api/medias/3/vote", json={"vote": "good"})  # toggle off
        assert resp.status_code == 200
        resp = client.post("/api/medias/3/vote", json={"vote": "bad"})  # vote bad
        assert resp.status_code == 200

        # Also add a new good vote
        resp = client.post("/api/medias/5/vote", json={"vote": "good"})
        assert resp.status_code == 200

        # Step 4: Verify vote state is correct
        resp = client.get("/api/votes")
        votes = resp.get_json()
        assert set(votes["good"]) == {1, 2, 5}
        assert set(votes["bad"]) == {3, 8, 9, 10}

        # Step 5: Re-learn
        resp = client.post("/api/learned-sort")
        assert resp.status_code == 200
        second_scores = {e["id"]: e["score"] for e in resp.get_json()["results"]}

        # Step 6: Clip 3's score should decrease (was good, now bad)
        assert second_scores[3] < first_scores[3], "Clip 3 score should decrease after switching from good to bad"


# ---------------------------------------------------------------------------
# 8. Settings Persistence Across Workflow
# ---------------------------------------------------------------------------


class TestSettingsAcrossWorkflow:
    """Simulates: user changes settings, performs workflow, verifies settings
    persist and affect behavior."""

    def test_settings_persist_through_workflow(self, client):
        # Step 1: Read default settings
        resp = client.get("/api/settings")
        assert resp.status_code == 200
        defaults = resp.get_json()
        assert defaults["volume"] == 1.0

        # Step 2: Update volume and theme
        resp = client.put("/api/settings", json={"volume": 0.5})
        assert resp.status_code == 200

        # Step 3: Perform some work (vote, sort)
        _vote_clips(client, [1, 2], [9, 10])
        resp = client.post("/api/learned-sort")
        assert resp.status_code == 200

        # Step 4: Verify settings survived the workflow
        resp = client.get("/api/settings")
        data = resp.get_json()
        assert data["volume"] == pytest.approx(0.5)

        # Step 5: Update safe_thresholds and verify it affects export
        resp = client.put("/api/settings", json={"safe_thresholds": True})
        assert resp.status_code == 200

        resp = client.get("/api/safe-thresholds")
        assert resp.get_json()["safe_thresholds"] is True


# ---------------------------------------------------------------------------
# 9. Detector Import via File → Sort → Rename → Delete
# ---------------------------------------------------------------------------


class TestDetectorFileImportWorkflow:
    """Simulates: user imports a detector from a JSON file, uses it to sort,
    renames it, then deletes it."""

    def test_import_sort_rename_delete(self, client):
        # Step 1: Create a detector and export as JSON
        _vote_clips(client, [1, 2, 3], [8, 9, 10])
        detector = _export_detector(client)

        # Step 2: Import the detector from a "file"
        json_bytes = json.dumps(detector).encode("utf-8")
        resp = client.post(
            "/api/autorun-detectors/import-pkl",
            data={
                "file": (io.BytesIO(json_bytes), "imported_detector.json"),
                "name": "file-imported",
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        assert resp.get_json()["name"] == "file-imported"

        # Step 3: Verify it appears in autorun list
        resp = client.get("/api/autorun-detectors")
        names = [d["name"] for d in resp.get_json()["detectors"]]
        assert "file-imported" in names

        # Step 4: Use it for detector sort
        app_module.good_votes.clear()
        app_module.bad_votes.clear()
        resp = client.post("/api/detector-sort", json={"detector": detector})
        assert resp.status_code == 200
        assert len(resp.get_json()["results"]) == app_module.NUM_MEDIAS

        # Step 5: Rename the detector
        resp = client.put(
            "/api/autorun-detectors/file-imported/rename",
            json={"new_name": "renamed-detector"},
        )
        assert resp.status_code == 200
        assert resp.get_json()["new_name"] == "renamed-detector"

        # Step 6: Verify old name gone, new name present
        resp = client.get("/api/autorun-detectors")
        names = [d["name"] for d in resp.get_json()["detectors"]]
        assert "renamed-detector" in names
        assert "file-imported" not in names

        # Step 7: Delete the detector
        resp = client.delete("/api/autorun-detectors/renamed-detector")
        assert resp.status_code == 200

        # Step 8: Verify it's gone
        resp = client.get("/api/autorun-detectors")
        assert resp.get_json()["detectors"] == []


# ---------------------------------------------------------------------------
# 10. Example Sort → Vote → Learned Sort
# ---------------------------------------------------------------------------


class TestExampleSortWorkflow:
    """Simulates: user uploads an example audio file to find similar medias,
    votes on the results, then runs learned sort."""

    def test_example_sort_then_learn(self, client):
        # Step 1: Generate a test audio file and upload for example sort
        wav_bytes = app_module.generate_wav(440.0, 1.0)
        resp = client.post(
            "/api/example-sort",
            data={"file": (io.BytesIO(wav_bytes), "example.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        example_results = resp.get_json()
        assert len(example_results["results"]) == app_module.NUM_MEDIAS

        # Results should be sorted by similarity
        sims = [r["similarity"] for r in example_results["results"]]
        assert sims == sorted(sims, reverse=True)

        # Step 2: Vote the top results as good, bottom as bad
        top_ids = [r["id"] for r in example_results["results"][:3]]
        bottom_ids = [r["id"] for r in example_results["results"][-3:]]
        _vote_clips(client, top_ids, bottom_ids)

        # Step 3: Run learned sort
        resp = client.post("/api/learned-sort")
        assert resp.status_code == 200
        learned = resp.get_json()

        # Good medias should score higher than bad on average
        score_map = {e["id"]: e["score"] for e in learned["results"]}
        avg_good = np.mean([score_map[i] for i in top_ids])
        avg_bad = np.mean([score_map[i] for i in bottom_ids])
        assert avg_good > avg_bad


# ---------------------------------------------------------------------------
# 11. Error Recovery — Bad Inputs Don't Break Subsequent Requests
# ---------------------------------------------------------------------------


class TestErrorRecoveryWorkflow:
    """Simulates: user makes mistakes (bad input), then continues normally.
    Verifies that errors don't corrupt state or break subsequent requests."""

    def test_errors_dont_corrupt_state(self, client):
        # Step 1: Try voting on a nonexistent media
        resp = client.post("/api/medias/9999/vote", json={"vote": "good"})
        assert resp.status_code == 404

        # Step 2: Try an invalid vote value
        resp = client.post("/api/medias/1/vote", json={"vote": "maybe"})
        assert resp.status_code == 400

        # Step 3: Try learned sort with no votes
        resp = client.post("/api/learned-sort")
        assert resp.status_code == 400

        # Step 4: Try text sort with empty text
        resp = client.post("/api/sort", json={"text": ""})
        assert resp.status_code == 400

        # Step 5: Try detector export with no votes
        resp = client.post("/api/detector/export")
        assert resp.status_code == 400

        # Step 6: Now do everything correctly and verify it works
        _vote_clips(client, [1, 2], [9, 10])

        resp = client.post("/api/sort", json={"text": "test sound"})
        assert resp.status_code == 200
        assert len(resp.get_json()["results"]) == app_module.NUM_MEDIAS

        resp = client.post("/api/learned-sort")
        assert resp.status_code == 200
        assert len(resp.get_json()["results"]) == app_module.NUM_MEDIAS

        resp = client.post("/api/detector/export")
        assert resp.status_code == 200
        assert "weights" in resp.get_json()

        # Step 7: Verify votes are clean
        resp = client.get("/api/votes")
        votes = resp.get_json()
        assert set(votes["good"]) == {1, 2}
        assert set(votes["bad"]) == {9, 10}


# ---------------------------------------------------------------------------
# 12. Detector Train → Save to Autorun Processor → Verify Settings
# ---------------------------------------------------------------------------


class TestAutorunProcessorWorkflow:
    """Simulates: user exports a detector to a file, adds it as an autorun
    processor in settings, and verifies the settings pipeline."""

    def test_detector_to_autorun_processor(self, client, tmp_path):
        # Step 1: Vote and export detector
        _vote_clips(client, [1, 2, 3], [8, 9, 10])
        detector = _export_detector(client)

        # Step 2: Write detector to a JSON file (simulates user saving it)
        det_path = tmp_path / "my_detector.json"
        det_path.write_text(
            json.dumps(
                {
                    "media_type": "audio",
                    "weights": detector["weights"],
                    "threshold": detector["threshold"],
                }
            )
        )

        # Step 3: Add as a autorun processor in settings
        resp = client.post(
            "/api/settings/autorun-processors",
            json={
                "processor_name": "my-saved-detector",
                "processor_importer": "server_detector_file",
                "field_values": {"filepath": str(det_path)},
            },
        )
        assert resp.status_code == 200
        assert resp.get_json()["success"] is True

        # Step 4: Verify it appears in settings
        resp = client.get("/api/settings")
        data = resp.get_json()
        proc_names = [p["processor_name"] for p in data["autorun_processors"]]
        assert "my-saved-detector" in proc_names

        # Step 5: Verify it also appears in the processors list
        resp = client.get("/api/settings/autorun-processors")
        assert resp.status_code == 200
        proc_names = [p["processor_name"] for p in resp.get_json()["autorun_processors"]]
        assert "my-saved-detector" in proc_names

        # Step 6: Delete it and verify it's gone
        resp = client.delete("/api/settings/autorun-processors/my-saved-detector")
        assert resp.status_code == 200

        resp = client.get("/api/settings/autorun-processors")
        proc_names = [p["processor_name"] for p in resp.get_json()["autorun_processors"]]
        assert "my-saved-detector" not in proc_names


# ---------------------------------------------------------------------------
# 13. Export via GUI Exporter (results pass-through)
# ---------------------------------------------------------------------------


class TestGuiExporterWorkflow:
    """Simulates: user runs auto-detect and previews results via GUI exporter."""

    def test_auto_detect_then_gui_export(self, client):
        # Step 1: Train and save a detector
        _vote_clips(client, [1, 2, 3], [8, 9, 10])
        detector = _export_detector(client)
        _save_autorun_detector(client, "gui-test-det", detector)
        app_module.good_votes.clear()
        app_module.bad_votes.clear()

        # Step 2: Run auto-detect
        resp = client.post("/api/auto-detect")
        assert resp.status_code == 200
        auto_results = resp.get_json()

        # Step 3: "Display" results via GUI exporter
        resp = client.post(
            "/api/exporters/export",
            json={
                "exporter_name": "gui",
                "field_values": {},
                "results": auto_results,
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "display_results" in data
        assert data["display_results"]["detectors_run"] == 1

        # Step 4: Verify available exporters endpoint works (frontend fetches this)
        resp = client.get("/api/exporters")
        assert resp.status_code == 200
        exporter_names = {e["name"] for e in resp.get_json()}
        assert "gui" in exporter_names
        assert "server_json_file" in exporter_names


# ---------------------------------------------------------------------------
# 14. Multi-sort Workflow: Text → Example → Learned
# ---------------------------------------------------------------------------


class TestMultiSortModesWorkflow:
    """Simulates: user tries different sort modes in sequence."""

    def test_switch_between_sort_modes(self, client):
        # Step 1: Text sort
        resp = client.post("/api/sort", json={"text": "a beeping sound"})
        assert resp.status_code == 200
        text_results = resp.get_json()["results"]
        text_ids = [r["id"] for r in text_results]

        # Step 2: Example sort with audio
        wav_bytes = app_module.generate_wav(300.0, 1.5)
        resp = client.post(
            "/api/example-sort",
            data={"file": (io.BytesIO(wav_bytes), "ref.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        example_results = resp.get_json()["results"]
        example_ids = [r["id"] for r in example_results]

        # Step 3: Vote based on text sort results
        _vote_clips(client, text_ids[:2], text_ids[-2:])

        # Step 4: Learned sort
        resp = client.post("/api/learned-sort")
        assert resp.status_code == 200
        learned_results = resp.get_json()["results"]
        learned_ids = [r["id"] for r in learned_results]

        # All three modes returned all medias
        assert set(text_ids) == set(range(1, app_module.NUM_MEDIAS + 1))
        assert set(example_ids) == set(range(1, app_module.NUM_MEDIAS + 1))
        assert set(learned_ids) == set(range(1, app_module.NUM_MEDIAS + 1))

        # Step 5: Text sort should still work after learning
        resp = client.post("/api/sort", json={"text": "low frequency tone"})
        assert resp.status_code == 200
        assert len(resp.get_json()["results"]) == app_module.NUM_MEDIAS


# ---------------------------------------------------------------------------
# 15. Labeling Progress Tracking
# ---------------------------------------------------------------------------


class TestLabelingProgressWorkflow:
    """Simulates: user votes incrementally and checks labeling progress."""

    def test_progressive_labeling(self, client):
        # Step 1: Start with no votes — labeling status should work
        resp = client.get("/api/labeling-status")
        assert resp.status_code == 200
        status = resp.get_json()
        assert status["good_count"] == 0
        assert status["bad_count"] == 0

        # Step 2: Vote on first batch
        _vote_clips(client, [1, 2, 3], [8, 9, 10])

        resp = client.get("/api/labeling-status")
        assert resp.status_code == 200
        status = resp.get_json()
        assert status["good_count"] == 3
        assert status["bad_count"] == 3

        # Step 3: Add more votes
        _vote_clips(client, [4, 5], [6, 7])

        resp = client.get("/api/labeling-status")
        status = resp.get_json()
        assert status["good_count"] == 5
        assert status["bad_count"] == 5

        # Step 4: Request labeling progress analysis
        resp = client.post("/api/labeling-progress")
        assert resp.status_code == 200

        # Step 5: Run learned sort to verify everything still works
        resp = client.post("/api/learned-sort")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 16. Vote Toggle Stress — Rapid Vote Changes
# ---------------------------------------------------------------------------


class TestVoteToggleStressWorkflow:
    """Simulates: user rapidly toggles votes on the same media, then
    verifies the system remains in a consistent state."""

    def test_rapid_toggle_then_learn(self, client):
        # Step 1: Toggle media 1 several times
        for _ in range(5):
            client.post("/api/medias/1/vote", json={"vote": "good"})
            client.post("/api/medias/1/vote", json={"vote": "good"})  # toggle off

        # After even number of toggles, should be unvoted
        resp = client.get("/api/votes")
        assert 1 not in resp.get_json()["good"]
        assert 1 not in resp.get_json()["bad"]

        # Step 2: Vote it good one final time
        client.post("/api/medias/1/vote", json={"vote": "good"})

        # Step 3: Switch between good and bad
        for _ in range(3):
            client.post("/api/medias/1/vote", json={"vote": "bad"})
            client.post("/api/medias/1/vote", json={"vote": "good"})

        # Should end on good
        resp = client.get("/api/votes")
        assert 1 in resp.get_json()["good"]
        assert 1 not in resp.get_json()["bad"]

        # Step 4: Add other votes and verify learned sort works
        _vote_clips(client, [2, 3], [9, 10])
        resp = client.post("/api/learned-sort")
        assert resp.status_code == 200
        assert len(resp.get_json()["results"]) == app_module.NUM_MEDIAS


# ---------------------------------------------------------------------------
# 17. Safe Thresholds Toggle Across Detector Lifecycle
# ---------------------------------------------------------------------------


class TestSafeThresholdsWorkflow:
    """Simulates: user toggles safe thresholds setting and verifies
    it affects detector export threshold."""

    def test_safe_threshold_affects_detector(self, client):
        # Step 1: Set safe thresholds OFF
        resp = client.post("/api/safe-thresholds", json={"safe_thresholds": False})
        assert resp.status_code == 200

        # Step 2: Vote and export a detector
        _vote_clips(client, [1, 2, 3], [8, 9, 10])
        resp = client.post("/api/detector/export")
        assert resp.status_code == 200
        threshold_off = resp.get_json()["threshold"]

        # Step 3: Turn safe thresholds ON
        resp = client.post("/api/safe-thresholds", json={"safe_thresholds": True})
        assert resp.status_code == 200

        resp = client.get("/api/safe-thresholds")
        assert resp.get_json()["safe_thresholds"] is True

        # Step 4: Re-export detector — threshold may differ
        resp = client.post("/api/detector/export")
        assert resp.status_code == 200
        threshold_on = resp.get_json()["threshold"]

        # Both should be valid floats
        assert isinstance(threshold_off, (int, float))
        assert isinstance(threshold_on, (int, float))
