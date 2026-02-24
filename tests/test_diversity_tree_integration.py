"""Integration tests for diversity tree wiring: build, label, unlabel, API."""

import numpy as np
import pytest

from vtsearch.utils import (
    bad_votes,
    build_diversity_tree,
    clips,
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
        assert data["id"] is not None
        assert data["id"] in clips
        assert data["diversity_level"] == -1  # nothing labeled yet

    def test_returns_null_when_no_tree(self, client):
        resp = client.get("/api/diversity-tree/next")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["id"] is None
        assert data["diversity_level"] == -1

    def test_diversity_level_after_labeling(self, client):
        tree = _build_tree()
        root_rep = tree.nodes["0"]["representative"]
        diversity_tree_label(root_rep)
        resp = client.get("/api/diversity-tree/next")
        data = resp.get_json()
        assert data["diversity_level"] >= 0


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
