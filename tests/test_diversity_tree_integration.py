"""Integration tests for diversity tree wiring: build, label, unlabel, API."""

from vtsearch.utils import (
    add_label_to_history,
    bad_votes,
    build_diversity_tree,
    clips,
    diversity_tree_label,
    diversity_tree_next_sample,
    diversity_tree_unlabel,
    get_diversity_tree,
    good_votes,
    label_history,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_tree():
    """Build the diversity tree from the global test clips."""
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
        # All clips with embeddings should be in the tree
        for cid, clip in clips.items():
            if clip.get("embedding") is not None:
                assert cid in tree.vector_to_leaf

    def test_empty_clips_yields_none(self):
        saved = dict(clips)
        clips.clear()
        try:
            build_diversity_tree()
            assert get_diversity_tree() is None
        finally:
            clips.update(saved)

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
    def test_returns_clip_id(self):
        _build_tree()
        next_id = diversity_tree_next_sample()
        assert next_id is not None
        assert next_id in clips

    def test_returns_none_when_no_tree(self):
        assert diversity_tree_next_sample() is None

    def test_advances_after_labeling(self):
        _build_tree()
        first = diversity_tree_next_sample()
        diversity_tree_label(first)
        second = diversity_tree_next_sample()
        # After labeling the root representative, the tree should suggest
        # a different clip (or None if fully seen).
        if second is not None:
            assert second != first or len(clips) == 1


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------


class TestDiversityTreeNextEndpoint:
    def test_returns_id_when_tree_exists(self, client):
        _build_tree()
        resp = client.get("/api/diversity-tree/next")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "id" in data
        assert "diversity_level" in data
        assert "exhausted" in data
        assert data["id"] is not None
        assert data["id"] in clips
        assert data["diversity_level"] == -1  # nothing labeled yet
        assert data["exhausted"] is False

    def test_returns_null_when_no_tree(self, client):
        resp = client.get("/api/diversity-tree/next")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["id"] is None
        assert data["diversity_level"] == -1
        assert data["exhausted"] is False  # no tree != exhausted

    def test_diversity_level_after_labeling(self, client):
        tree = _build_tree()
        root_rep = tree.nodes["0"]["representative"]
        diversity_tree_label(root_rep)
        resp = client.get("/api/diversity-tree/next")
        data = resp.get_json()
        assert data["diversity_level"] >= 0

    def test_exhausted_when_all_nodes_seen(self, client):
        """When every node in the tree has been seen, exhausted should be True."""
        tree = _build_tree()
        # Label every vector so all leaves (and ancestors) become seen
        for vid in tree.vector_to_leaf:
            diversity_tree_label(vid)
        resp = client.get("/api/diversity-tree/next")
        data = resp.get_json()
        assert data["id"] is None
        assert data["exhausted"] is True


# ---------------------------------------------------------------------------
# Vote integration: voting via the API should update the tree
# ---------------------------------------------------------------------------


class TestVoteUpdatesTree:
    def test_good_vote_labels_tree(self, client):
        tree = _build_tree()
        cid = 1
        resp = client.post(f"/api/clips/{cid}/vote", json={"vote": "good"})
        assert resp.status_code == 200
        assert cid in tree.labeled_ids

    def test_bad_vote_labels_tree(self, client):
        tree = _build_tree()
        cid = 2
        resp = client.post(f"/api/clips/{cid}/vote", json={"vote": "bad"})
        assert resp.status_code == 200
        assert cid in tree.labeled_ids

    def test_toggle_unlabels_tree(self, client):
        tree = _build_tree()
        cid = 1
        # Vote good
        client.post(f"/api/clips/{cid}/vote", json={"vote": "good"})
        assert cid in tree.labeled_ids
        # Toggle good off (unlabel)
        client.post(f"/api/clips/{cid}/vote", json={"vote": "good"})
        assert cid not in tree.labeled_ids

    def test_switch_vote_keeps_labeled(self, client):
        """Switching from good to bad should keep the clip labeled."""
        tree = _build_tree()
        cid = 3
        client.post(f"/api/clips/{cid}/vote", json={"vote": "good"})
        assert cid in tree.labeled_ids
        # Switch to bad
        client.post(f"/api/clips/{cid}/vote", json={"vote": "bad"})
        assert cid in tree.labeled_ids


# ---------------------------------------------------------------------------
# Label import integration
# ---------------------------------------------------------------------------


class TestLabelImportUpdatesTree:
    def test_import_labels_tree(self, client):
        tree = _build_tree()
        cid = 1
        md5 = clips[cid]["md5"]
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
        assert tree.diversity_level() == -1

        # Label clips guided by next_sample until the tree is exhausted or
        # we've done enough iterations to cover the full tree.
        max_iters = len(clips) + 1
        for _ in range(max_iters):
            sample = diversity_tree_next_sample()
            if sample is None:
                break
            diversity_tree_label(sample)

        # After labeling all suggested clips the tree must be fully covered
        final_level = tree.diversity_level()
        assert final_level == tree.depth(), (
            f"Expected diversity_level {tree.depth()} (fully covered) but got {final_level}"
        )

    def test_span_advances_beyond_zero_via_votes(self, client):
        """Voting on clips from different subtrees advances the span level past 0."""
        tree = _build_tree()
        depth = tree.depth()
        if depth < 1:
            # Tree too shallow to test progression beyond 0
            return

        # Use next_sample to find clips from each depth-1 subtree and vote on them
        for _ in range(len(clips)):
            sample = diversity_tree_next_sample()
            if sample is None:
                break
            resp = client.post(f"/api/clips/{sample}/vote", json={"vote": "good"})
            assert resp.status_code == 200

        # After voting on diverse clips, the level should have advanced past 0
        assert tree.diversity_level() >= 1, (
            f"Expected diversity_level >= 1 but got {tree.diversity_level()}"
        )

    def test_labeling_status_reflects_span_progression(self, client):
        """The /api/labeling-status endpoint should reflect advancing span level."""
        tree = _build_tree()

        # Before any labels, span should be red with level -1
        resp = client.get("/api/labeling-status")
        data = resp.get_json()
        assert data["span"]["level"] == -1
        assert data["span"]["status"] == "red"

        # Label all clips to fully cover the tree
        for vid in tree.vector_to_leaf:
            diversity_tree_label(vid)
            good_votes[vid] = None

        resp = client.get("/api/labeling-status")
        data = resp.get_json()
        assert data["span"]["level"] == tree.depth()
        assert data["span"]["status"] == "green"


# ---------------------------------------------------------------------------
# Label importer updates diversity tree
# ---------------------------------------------------------------------------


class TestLabelImporterUpdatesTree:
    def test_label_importer_updates_tree(self, client):
        """Labels imported via /api/label-importers/import should update the tree."""
        import io
        import json

        tree = _build_tree()
        cid = 1
        md5 = clips[cid]["md5"]

        # Import a label via the json_file label importer
        labels_data = {"labels": [{"md5": md5, "label": "good"}]}
        file_content = json.dumps(labels_data).encode()

        resp = client.post(
            "/api/label-importers/import/json_file",
            data={"file": (io.BytesIO(file_content), "labels.json")},
            content_type="multipart/form-data",
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

        # Label several clips from different parts of the tree
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
            assert levels[i] >= levels[i - 1], (
                f"Diversity level decreased at step {i}: {levels[i-1]} -> {levels[i]}"
            )

    def test_labeling_status_includes_fractional_level(self, client):
        """The /api/labeling-status span info should include fractional_level."""
        _build_tree()
        diversity_tree_label(1)
        good_votes[1] = None

        resp = client.get("/api/labeling-status")
        data = resp.get_json()
        assert "fractional_level" in data["span"]
        assert isinstance(data["span"]["fractional_level"], (int, float))
