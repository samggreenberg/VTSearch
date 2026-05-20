"""Integration tests for diversity tree wiring: build, label, unlabel, API."""

import pytest

from vtsearch.state import (
    add_label_to_history,
    bad_votes,
    build_diversity_tree,
    medias,
    diversity_tree_label,
    diversity_tree_next_sample,
    diversity_tree_unlabel,
    get_diversity_tree,
    good_votes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_tree():
    """Build the diversity tree from the global test medias."""
    build_diversity_tree()
    tree = get_diversity_tree()
    assert tree is not None
    return tree


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


class TestBuildDiversityTree:
    def test_builds_from_clips(self):
        tree = _build_tree()
        # k=3 diversity tree
        assert tree.k == 3
        # All medias with embeddings should be in the tree
        for cid, media in medias.items():
            if media.get("embedding") is not None:
                assert cid in tree.vector_to_leaf

    def test_empty_clips_yields_none(self):
        saved = dict(medias)
        medias.clear()
        try:
            build_diversity_tree()
            assert get_diversity_tree() is None
        finally:
            medias.update(saved)

    def test_replays_existing_votes(self):
        """If votes exist before tree is built, they are replayed."""
        good_votes[1] = None
        bad_votes[2] = None
        tree = _build_tree()
        assert 1 in tree.labeled_ids
        assert 2 in tree.labeled_ids

    def test_rebuild_clears_old_state(self):
        tree1 = _build_tree()
        diversity_tree_label(1)
        assert 1 in tree1.labeled_ids
        # Rebuild from scratch (no votes now since fixture cleared them
        # and we only labeled via the tree helper, not good_votes)
        build_diversity_tree()
        tree2 = get_diversity_tree()
        assert 1 not in tree2.labeled_ids


class TestDiversityTreeLabel:
    def test_label_marks_seen(self):
        tree = _build_tree()
        diversity_tree_label(1)
        leaf = tree.lookup(1)
        assert leaf in tree.seen

    def test_label_no_tree_is_noop(self):
        """Labeling when tree is None should not raise."""
        diversity_tree_label(1)  # no tree built yet


class TestDiversityTreeUnlabel:
    def test_unlabel_clears_seen(self):
        tree = _build_tree()
        diversity_tree_label(1)
        assert len(tree.seen) > 0
        diversity_tree_unlabel(1)
        assert len(tree.seen) == 0

    def test_unlabel_no_tree_is_noop(self):
        diversity_tree_unlabel(1)  # no tree built yet


class TestDiversityTreeNextSample:
    def test_returns_media_id(self):
        _build_tree()
        next_id = diversity_tree_next_sample()
        assert next_id is not None
        assert next_id in medias

    def test_returns_none_when_no_tree(self):
        assert diversity_tree_next_sample() is None

    def test_advances_after_labeling(self):
        _build_tree()
        first = diversity_tree_next_sample()
        assert first is not None
        diversity_tree_label(first)
        second = diversity_tree_next_sample()
        # After labeling the first sample, the tree should suggest
        # a different media (or None if fully seen).
        if second is not None:
            assert second != first or len(medias) == 1


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------


class TestDiversityTreeNextEndpoint:
    def test_returns_id_when_tree_exists(self, client):
        _build_tree()
        resp = client.post("/api/diversity-tree/next", json={})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "id" in data
        assert "diversity_level" in data
        assert "exhausted" in data
        assert data["id"] is not None
        assert data["id"] in medias
        assert data["diversity_level"] == 0  # nothing labeled yet
        assert data["exhausted"] is False

    def test_returns_null_when_no_tree(self, client):
        resp = client.post("/api/diversity-tree/next", json={})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["id"] is None
        assert data["diversity_level"] == 0
        assert data["exhausted"] is False  # no tree != exhausted

    def test_get_still_works(self, client):
        """GET without scores still returns a valid result."""
        _build_tree()
        resp = client.get("/api/diversity-tree/next")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["id"] is not None
        assert data["id"] in medias

    def test_diversity_level_after_labeling(self, client):
        tree = _build_tree()
        first_id = tree.nodes["0"]["ids"][0]
        diversity_tree_label(first_id)
        resp = client.post("/api/diversity-tree/next", json={})
        data = resp.get_json()
        assert data["diversity_level"] >= 1

    def test_exhausted_when_all_nodes_seen(self, client):
        """When every node in the tree has been seen, exhausted should be True."""
        tree = _build_tree()
        # Label every vector so all leaves (and ancestors) become seen
        for vid in tree.vector_to_leaf:
            diversity_tree_label(vid)
        resp = client.post("/api/diversity-tree/next", json={})
        data = resp.get_json()
        assert data["id"] is None
        assert data["exhausted"] is True

    def test_scores_influence_selection(self, client):
        """When scores are posted, the highest-scored element in the node is returned."""
        tree = _build_tree()
        # Build scores: give the highest score to a specific media
        all_ids = list(tree.vector_to_leaf.keys())
        target_id = all_ids[-1]  # pick the last one
        scores = {str(vid): 0.1 for vid in all_ids}
        scores[str(target_id)] = 1.0

        resp = client.post("/api/diversity-tree/next", json={"scores": scores})
        data = resp.get_json()
        assert data["id"] == target_id

    def test_threshold_flips_to_lowest(self, client):
        """When threshold is below the node median, the lowest-scored element is returned."""
        tree = _build_tree()
        all_ids = list(tree.vector_to_leaf.keys())
        # Give all media high scores (above threshold) so median > threshold
        scores = {str(vid): 0.9 for vid in all_ids}
        lowest_id = all_ids[0]
        scores[str(lowest_id)] = 0.1  # one outlier low score

        resp = client.post(
            "/api/diversity-tree/next",
            json={"scores": scores, "threshold": 0.5},
        )
        data = resp.get_json()
        # Median of the node is 0.9 (most elements) which is >= 0.5,
        # so the lowest-scored element should be returned
        assert data["id"] == lowest_id

    def test_threshold_keeps_highest_when_below(self, client):
        """When threshold is above the node median, the highest-scored element is returned."""
        tree = _build_tree()
        all_ids = list(tree.vector_to_leaf.keys())
        # Give all media low scores (below threshold)
        scores = {str(vid): 0.1 for vid in all_ids}
        highest_id = all_ids[-1]
        scores[str(highest_id)] = 0.3  # still below threshold

        resp = client.post(
            "/api/diversity-tree/next",
            json={"scores": scores, "threshold": 0.5},
        )
        data = resp.get_json()
        # Median is ~0.1 which is < 0.5, so highest-scored element returned
        assert data["id"] == highest_id


# ---------------------------------------------------------------------------
# Vote integration: voting via the API should update the tree
# ---------------------------------------------------------------------------


class TestVoteUpdatesTree:
    def test_good_vote_labels_tree(self, client):
        tree = _build_tree()
        cid = 1
        resp = client.post(f"/api/medias/{cid}/vote", json={"target": "good"})
        assert resp.status_code == 200
        assert cid in tree.labeled_ids

    def test_bad_vote_labels_tree(self, client):
        tree = _build_tree()
        cid = 2
        resp = client.post(f"/api/medias/{cid}/vote", json={"target": "bad"})
        assert resp.status_code == 200
        assert cid in tree.labeled_ids

    def test_toggle_unlabels_tree(self, client):
        tree = _build_tree()
        cid = 1
        # Vote good
        client.post(f"/api/medias/{cid}/vote", json={"target": "good"})
        assert cid in tree.labeled_ids
        # Toggle good off (unlabel)
        client.post(f"/api/medias/{cid}/vote", json={"target": "good"})
        assert cid not in tree.labeled_ids

    def test_switch_vote_keeps_labeled(self, client):
        """Switching from good to bad should keep the media labeled."""
        tree = _build_tree()
        cid = 3
        client.post(f"/api/medias/{cid}/vote", json={"target": "good"})
        assert cid in tree.labeled_ids
        # Switch to bad
        client.post(f"/api/medias/{cid}/vote", json={"target": "bad"})
        assert cid in tree.labeled_ids


# ---------------------------------------------------------------------------
# Label import integration
# ---------------------------------------------------------------------------


class TestLabelImportUpdatesTree:
    def test_import_labels_tree(self, client):
        tree = _build_tree()
        cid = 1
        md5 = medias[cid]["md5"]
        resp = client.post(
            "/api/labels/import",
            json={"labels": [{"md5": md5, "label": "good"}]},
        )
        assert resp.status_code == 200
        assert cid in tree.labeled_ids


# ---------------------------------------------------------------------------
# Span level progression
# ---------------------------------------------------------------------------


class TestSpanLevelProgression:
    def test_span_advances_with_next_sample_labeling(self):
        """Following next_sample suggestions should advance the diversity level."""
        tree = _build_tree()
        assert tree.diversity_level() == 0

        # Label medias guided by next_sample until the tree is exhausted or
        # we've done enough iterations to cover the full tree.
        max_iters = len(medias) + 1
        for _ in range(max_iters):
            sample = diversity_tree_next_sample()
            if sample is None:
                break
            diversity_tree_label(sample)

        # After labeling all suggested medias the tree must be fully covered
        final_level = tree.diversity_level()
        assert final_level == tree.total_nodes, (
            f"Expected diversity_level {tree.total_nodes} (fully covered) but got {final_level}"
        )

    def test_span_advances_beyond_zero_via_votes(self, client):
        """Voting on medias from different subtrees advances the span level past 0."""
        tree = _build_tree()
        depth = tree.depth()
        if depth < 1:
            pytest.skip("Tree too shallow to test progression beyond 0")

        # Use next_sample to find medias from each depth-1 subtree and vote on them
        for _ in range(len(medias)):
            sample = diversity_tree_next_sample()
            if sample is None:
                break
            resp = client.post(f"/api/medias/{sample}/vote", json={"target": "good"})
            assert resp.status_code == 200

        # After voting on diverse medias, the level should have advanced past 1
        assert tree.diversity_level() >= 2, f"Expected diversity_level >= 2 but got {tree.diversity_level()}"

    def test_labeling_status_reflects_span_progression(self, client):
        """The /api/labeling-status endpoint should reflect advancing span level."""
        tree = _build_tree()

        # Before any labels, span should be red with level 0
        resp = client.get("/api/labeling-status")
        data = resp.get_json()
        assert data["span"]["level"] == 0
        assert data["span"]["status"] == "red"

        # Label all medias to fully cover the tree
        for vid in tree.vector_to_leaf:
            diversity_tree_label(vid)
            good_votes[vid] = None

        resp = client.get("/api/labeling-status")
        data = resp.get_json()
        assert data["span"]["level"] == tree.total_nodes
        assert data["span"]["status"] == "green"


# ---------------------------------------------------------------------------
# Label importer updates diversity tree
# ---------------------------------------------------------------------------


class TestLabelImporterUpdatesTree:
    def test_label_importer_updates_tree(self, client, tmp_path):
        """Labels imported via /api/label-importers/import should update the tree."""
        import json

        tree = _build_tree()
        cid = 1
        md5 = medias[cid]["md5"]

        # Import a label via the server_json_file label importer
        labels_data = {"labels": [{"md5": md5, "label": "good"}]}
        p = tmp_path / "labels.json"
        p.write_text(json.dumps(labels_data))

        resp = client.post(
            "/api/label-importers/import/server_json_file",
            json={"filepath": str(p)},
        )
        assert resp.status_code == 200
        assert cid in tree.labeled_ids


# ---------------------------------------------------------------------------
# Diversity level over time (progress API)
# ---------------------------------------------------------------------------


class TestDiversityLevelOverTime:
    def test_progress_includes_diversity_level_over_time(self, client):
        """The /api/labeling-progress response should include diversity_level_over_time."""
        _build_tree()

        # Need at least one good and one bad vote + label history
        good_votes[1] = None
        bad_votes[2] = None
        add_label_to_history(1, "good")
        add_label_to_history(2, "bad")
        diversity_tree_label(1)
        diversity_tree_label(2)

        resp = client.post("/api/labeling-progress")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "diversity_level_over_time" in data
        assert isinstance(data["diversity_level_over_time"], list)
        assert len(data["diversity_level_over_time"]) == 2

    def test_diversity_level_entries_have_expected_fields(self, client):
        """Each entry should have num_labels, diversity_level, and depth."""
        _build_tree()

        good_votes[1] = None
        bad_votes[2] = None
        add_label_to_history(1, "good")
        add_label_to_history(2, "bad")
        diversity_tree_label(1)
        diversity_tree_label(2)

        resp = client.post("/api/labeling-progress")
        data = resp.get_json()
        for entry in data["diversity_level_over_time"]:
            assert "num_labels" in entry
            assert "diversity_level" in entry
            assert "depth" in entry

    def test_diversity_level_monotonically_increases(self, client):
        """Diversity level should not decrease when only adding labels."""
        tree = _build_tree()

        # Label several medias from different parts of the tree
        cids = sorted(tree.vector_to_leaf.keys())[:6]
        for i, cid in enumerate(cids):
            if i % 2 == 0:
                good_votes[cid] = None
                add_label_to_history(cid, "good")
            else:
                bad_votes[cid] = None
                add_label_to_history(cid, "bad")
            diversity_tree_label(cid)

        resp = client.post("/api/labeling-progress")
        data = resp.get_json()
        levels = [e["diversity_level"] for e in data["diversity_level_over_time"]]
        for i in range(1, len(levels)):
            assert levels[i] >= levels[i - 1], f"Diversity level decreased at step {i}: {levels[i - 1]} -> {levels[i]}"

    def test_diversity_level_no_drop_after_vote_polarity_switch(self, client):
        """Switching a vote from bad→good must not cause a diversity dip in the progress cache."""
        tree = _build_tree()

        # Label several medias to build up diversity — need both good and bad
        cids = sorted(tree.vector_to_leaf.keys())[:6]
        for i, cid in enumerate(cids):
            if i % 2 == 0:
                client.post(f"/api/medias/{cid}/vote", json={"target": "good"})
            else:
                client.post(f"/api/medias/{cid}/vote", json={"target": "bad"})

        # Record the diversity level before the switch
        resp = client.post("/api/labeling-progress")
        data = resp.get_json()
        levels_before = [e["diversity_level"] for e in data["diversity_level_over_time"]]
        assert len(levels_before) > 0, "Expected diversity history entries"
        peak_before = max(levels_before)

        # Switch the last bad vote to good (polarity switch triggers cache invalidation)
        switch_cid = cids[1]  # was voted bad
        client.post(f"/api/medias/{switch_cid}/vote", json={"target": "good"})

        # Fetch diversity history again — no entry should be below the pre-switch peak
        resp = client.post("/api/labeling-progress")
        data = resp.get_json()
        levels_after = [e["diversity_level"] for e in data["diversity_level_over_time"]]
        for i, level in enumerate(levels_after):
            if i < len(levels_before):
                # Kept entries should be unchanged
                continue
            # New entries should not dip below the peak
            assert level >= peak_before, (
                f"Diversity level dropped to {level} at step {i} after vote polarity switch "
                f"(peak before switch was {peak_before}). Full history: {levels_after}"
            )

    def test_labeling_status_includes_diversity_level(self, client):
        """The /api/labeling-status span info should include diversity_level."""
        _build_tree()
        diversity_tree_label(1)
        good_votes[1] = None

        resp = client.get("/api/labeling-status")
        data = resp.get_json()
        assert "diversity_level" in data["span"]
        assert isinstance(data["span"]["diversity_level"], (int, float))
