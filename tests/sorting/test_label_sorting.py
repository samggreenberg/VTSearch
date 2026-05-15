"""Tests for label sorting features: click-time tracking, learned-sort scores,
the enriched /api/votes response, and label-file-sort model selection."""

import io
import json
from unittest.mock import patch

import numpy as np

import app as app_module
import vtsearch.state as _state
import vtsearch.state.core as _core
from vtsearch.state import (
    medias,
    vote_click_times,
    last_learned_scores,
)


class TestClickTimeTracking:
    """Verify that voting via the API assigns monotonically-increasing click times."""

    def test_vote_assigns_click_time(self, client):
        resp = client.post("/api/medias/1/vote", json={"vote": "good"})
        assert resp.status_code == 200
        assert 1 in vote_click_times
        assert vote_click_times[1] == 1

    def test_sequential_votes_increment(self, client):
        client.post("/api/medias/1/vote", json={"vote": "good"})
        client.post("/api/medias/2/vote", json={"vote": "bad"})
        client.post("/api/medias/3/vote", json={"vote": "good"})
        assert vote_click_times[1] == 1
        assert vote_click_times[2] == 2
        assert vote_click_times[3] == 3

    def test_unvote_removes_click_time(self, client):
        client.post("/api/medias/1/vote", json={"vote": "good"})
        assert 1 in vote_click_times
        # Toggle off
        client.post("/api/medias/1/vote", json={"vote": "good"})
        assert 1 not in vote_click_times

    def test_revote_gets_new_click_time(self, client):
        client.post("/api/medias/1/vote", json={"vote": "good"})
        assert vote_click_times[1] == 1
        # Toggle off (does not increment counter)
        client.post("/api/medias/1/vote", json={"vote": "good"})
        # Vote again — should get a new, higher click time
        client.post("/api/medias/1/vote", json={"vote": "good"})
        assert vote_click_times[1] == 2

    def test_switch_vote_gets_new_click_time(self, client):
        client.post("/api/medias/1/vote", json={"vote": "good"})
        assert vote_click_times[1] == 1
        # Switch from good to bad
        client.post("/api/medias/1/vote", json={"vote": "bad"})
        assert vote_click_times[1] == 2
        assert 1 in app_module.bad_votes
        assert 1 not in app_module.good_votes

    def test_imported_labels_have_no_click_time(self, client):
        """Labels added via import should not receive a click time."""
        labels = [{"md5": app_module.medias[1]["md5"], "label": "good"}]
        client.post("/api/labels/import", json={"labels": labels})
        assert 1 in app_module.good_votes
        assert 1 not in vote_click_times


class TestVotesEndpointEnriched:
    """The /api/votes endpoint should include click_times and learned_scores."""

    def test_votes_response_includes_click_times(self, client):
        client.post("/api/medias/1/vote", json={"vote": "good"})
        client.post("/api/medias/2/vote", json={"vote": "bad"})
        resp = client.get("/api/votes")
        data = resp.get_json()
        assert "click_times" in data
        assert data["click_times"]["1"] == 1
        assert data["click_times"]["2"] == 2

    def test_votes_response_includes_learned_scores(self, client):
        # Manually set some learned scores
        _state.last_learned_scores[1] = 0.95
        _state.last_learned_scores[2] = 0.1
        resp = client.get("/api/votes")
        data = resp.get_json()
        assert "learned_scores" in data
        assert data["learned_scores"]["1"] == 0.95
        assert data["learned_scores"]["2"] == 0.1

    def test_votes_response_empty_initially(self, client):
        resp = client.get("/api/votes")
        data = resp.get_json()
        assert data["click_times"] == {}
        assert data["learned_scores"] == {}

    def test_learned_scores_populated_after_learned_sort(self, client):
        """After a learned-sort, /api/votes should include scores for all medias."""
        # Set up votes (need at least one good and one bad)
        app_module.good_votes.update({1: None, 2: None})
        app_module.bad_votes.update({3: None, 4: None})
        # Trigger learned sort
        resp = client.post("/api/learned-sort", json={"wait": True})
        assert resp.status_code == 200
        sort_data = resp.get_json()
        assert "results" in sort_data

        # Now check /api/votes
        resp = client.get("/api/votes")
        data = resp.get_json()
        assert len(data["learned_scores"]) > 0
        # Every media should have a score
        for cid in app_module.medias:
            assert str(cid) in data["learned_scores"]


class TestClearVotesResetsState:
    """Clearing votes should also clear click times and learned scores."""

    def test_clear_votes_clears_click_times(self, client):
        client.post("/api/medias/1/vote", json={"vote": "good"})
        assert len(vote_click_times) == 1
        _state.clear_votes()
        assert len(vote_click_times) == 0
        assert _core._get_click_counter() == 0

    def test_clear_votes_clears_learned_scores(self, client):
        _state.last_learned_scores[1] = 0.9
        _state.clear_votes()
        assert len(last_learned_scores) == 0


class TestLabelFileSortModelSelection:
    """Verify that /api/label-file-sort uses the correct model for the current media type."""

    def _make_label_file(self, paths_and_labels):
        """Create a JSON label file in memory."""
        labels = [{"path": str(p), "label": lbl} for p, lbl in paths_and_labels]
        content = json.dumps({"labels": labels})
        buf = io.BytesIO(content.encode("utf-8"))
        buf.name = "labels.json"
        return buf

    def test_no_file_returns_400(self, client):
        resp = client.post("/api/label-file-sort")
        assert resp.status_code == 400

    def test_no_medias_returns_400(self, client):
        saved = dict(medias)
        medias.clear()
        try:
            buf = self._make_label_file([])
            resp = client.post(
                "/api/label-file-sort",
                data={"file": (buf, "labels.json")},
                content_type="multipart/form-data",
            )
            assert resp.status_code == 400
        finally:
            medias.update(saved)

    def test_uses_current_media_type_embedder(self, client, tmp_path):
        """The endpoint should call embed_media on the embedder matching the loaded dataset,
        not hardcode embed_audio_file / CLAP."""
        # Determine the current media type from loaded test medias
        media_type = next(iter(medias.values())).get("type", "audio")
        embedding_dim = next(iter(medias.values()))["embedding"].shape[0]

        # Create fake media files on disk
        good_file = tmp_path / "good.bin"
        bad_file = tmp_path / "bad.bin"
        good_file.write_bytes(b"\x00" * 100)
        bad_file.write_bytes(b"\x00" * 100)

        fake_emb = np.random.default_rng(42).standard_normal(embedding_dim).astype(np.float32)

        from vtsearch.media import embedders_for_type

        emb = embedders_for_type(media_type)[0]

        with patch.object(emb, "embed_media", return_value=fake_emb) as mock_embed:
            buf = self._make_label_file([(good_file, "good"), (bad_file, "bad")])
            resp = client.post(
                "/api/label-file-sort",
                data={"file": (buf, "labels.json")},
                content_type="multipart/form-data",
            )
            assert resp.status_code == 200
            data = resp.get_json()
            assert "results" in data
            assert data["loaded"] == 2
            # Verify embed_media was called on the embedder
            assert mock_embed.call_count == 2

    def test_invalid_label_file_returns_400(self, client):
        buf = io.BytesIO(b"not json")
        resp = client.post(
            "/api/label-file-sort",
            data={"file": (buf, "labels.json")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert "invalid" in resp.get_json()["error"].lower()

    def test_empty_labels_returns_400(self, client):
        buf = io.BytesIO(json.dumps({"labels": []}).encode())
        resp = client.post(
            "/api/label-file-sort",
            data={"file": (buf, "labels.json")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
