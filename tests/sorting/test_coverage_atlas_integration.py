"""Integration tests for coverage atlas wiring: build, label, unlabel, API."""

import pytest

from vtsearch.state import (
    add_label_to_history,
    bad_votes,
    build_coverage_atlas,
    medias,
    coverage_atlas_label,
    coverage_atlas_next_sample,
    coverage_atlas_unlabel,
    get_coverage_atlas,
    good_votes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_atlas():
    """Build the coverage atlas from the global test medias."""
    build_coverage_atlas()
    atlas = get_coverage_atlas()
    assert atlas is not None
    return atlas


def _covered(atlas, name):
    node = atlas.nodes[name]
    return node["n_pos"] + node["n_neg"] > 0


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


class TestBuildCoverageAtlas:
    def test_builds_from_clips(self):
        atlas = _build_atlas()
        # k=3 coverage atlas
        assert atlas.k == 3
        # All medias with embeddings should be in the atlas
        for cid, media in medias.items():
            if media.get("embedding") is not None:
                assert cid in atlas.vector_to_leaf

    def test_empty_clips_yields_none(self):
        saved = dict(medias)
        medias.clear()
        try:
            build_coverage_atlas()
            assert get_coverage_atlas() is None
        finally:
            medias.update(saved)

    def test_replays_existing_votes(self):
        """If votes exist before the atlas is built, they are replayed per class."""
        good_votes[1] = None
        bad_votes[2] = None
        atlas = _build_atlas()
        assert 1 in atlas.labeled_ids
        assert 2 in atlas.labeled_ids
        assert atlas.nodes["0"]["n_pos"] == 1
        assert atlas.nodes["0"]["n_neg"] == 1

    def test_rebuild_clears_old_state(self):
        atlas1 = _build_atlas()
        coverage_atlas_label(1, good=True)
        assert 1 in atlas1.labeled_ids
        # Rebuild from scratch (no votes now since fixture cleared them
        # and we only labeled via the atlas helper, not good_votes)
        build_coverage_atlas()
        atlas2 = get_coverage_atlas()
        assert 1 not in atlas2.labeled_ids


class TestCoverageAtlasLabel:
    def test_label_marks_covered(self):
        atlas = _build_atlas()
        coverage_atlas_label(1, good=True)
        leaf = atlas.lookup(1)
        assert _covered(atlas, leaf)

    def test_label_no_atlas_is_noop(self):
        """Labeling when the atlas is None should not raise."""
        coverage_atlas_label(1, good=True)  # no atlas built yet


class TestCoverageAtlasUnlabel:
    def test_unlabel_clears_evidence(self):
        atlas = _build_atlas()
        coverage_atlas_label(1, good=True)
        assert len(atlas.labeled_ids) > 0
        coverage_atlas_unlabel(1)
        assert len(atlas.labeled_ids) == 0
        assert not any(_covered(atlas, name) for name in atlas.nodes)

    def test_unlabel_no_atlas_is_noop(self):
        coverage_atlas_unlabel(1)  # no atlas built yet


class TestCoverageAtlasNextSample:
    def test_returns_media_id(self):
        _build_atlas()
        next_id = coverage_atlas_next_sample()
        assert next_id is not None
        assert next_id in medias

    def test_returns_none_when_no_atlas(self):
        assert coverage_atlas_next_sample() is None

    def test_advances_after_labeling(self):
        _build_atlas()
        first = coverage_atlas_next_sample()
        assert first is not None
        coverage_atlas_label(first, good=True)
        second = coverage_atlas_next_sample()
        # After labeling the first sample, the atlas should suggest
        # a different media (or None if fully covered).
        if second is not None:
            assert second != first or len(medias) == 1


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------


class TestCoverageAtlasNextEndpoint:
    def test_returns_id_when_atlas_exists(self, client):
        _build_atlas()
        resp = client.post("/api/coverage-atlas/next", json={})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "id" in data
        assert "coverage_level" in data
        assert "exhausted" in data
        assert data["id"] is not None
        assert data["id"] in medias
        assert data["coverage_level"] == 0  # nothing labeled yet
        assert data["exhausted"] is False

    def test_returns_null_when_no_atlas(self, client):
        resp = client.post("/api/coverage-atlas/next", json={})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["id"] is None
        assert data["coverage_level"] == 0
        assert data["exhausted"] is False  # no atlas != exhausted

    def test_get_still_works(self, client):
        """GET without scores still returns a valid result."""
        _build_atlas()
        resp = client.get("/api/coverage-atlas/next")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["id"] is not None
        assert data["id"] in medias

    def test_coverage_level_after_labeling(self, client):
        atlas = _build_atlas()
        first_id = atlas.nodes["0"]["ids"][0]
        coverage_atlas_label(first_id, good=True)
        resp = client.post("/api/coverage-atlas/next", json={})
        data = resp.get_json()
        assert data["coverage_level"] >= 1

    def test_exhausted_when_all_nodes_covered(self, client):
        """When every node in the atlas carries evidence, exhausted should be True."""
        atlas = _build_atlas()
        # Label every vector so all leaves (and ancestors) become covered
        for vid in atlas.vector_to_leaf:
            coverage_atlas_label(vid, good=True)
        resp = client.post("/api/coverage-atlas/next", json={})
        data = resp.get_json()
        assert data["id"] is None
        assert data["exhausted"] is True

    def test_scores_influence_selection(self, client):
        """When scores are posted, the highest-scored element in the node is returned."""
        atlas = _build_atlas()
        # Build scores: give the highest score to the root's most typical
        # media — always inside the surprise pool regardless of whether the
        # node tempers by typicality (concentrated direction) or not.
        all_ids = list(atlas.vector_to_leaf.keys())
        target_id = atlas.nodes["0"]["ids"][0]
        scores = {str(vid): 0.1 for vid in all_ids}
        scores[str(target_id)] = 1.0

        resp = client.post("/api/coverage-atlas/next", json={"scores": scores})
        data = resp.get_json()
        assert data["id"] == target_id

    def test_threshold_flips_to_lowest(self, client):
        """When threshold is below the node median, the lowest-scored element is returned."""
        atlas = _build_atlas()
        all_ids = list(atlas.vector_to_leaf.keys())
        # Give all media high scores (above threshold) so median > threshold.
        # The low outlier sits on the root's most typical media so it is
        # inside the surprise pool under either pooling branch.
        scores = {str(vid): 0.9 for vid in all_ids}
        lowest_id = atlas.nodes["0"]["ids"][0]
        scores[str(lowest_id)] = 0.1  # one outlier low score

        resp = client.post(
            "/api/coverage-atlas/next",
            json={"scores": scores, "threshold": 0.5},
        )
        data = resp.get_json()
        # Median of the node is 0.9 (most elements) which is >= 0.5,
        # so the lowest-scored element should be returned
        assert data["id"] == lowest_id

    def test_threshold_keeps_highest_when_below(self, client):
        """When threshold is above the node median, the highest-scored element is returned."""
        atlas = _build_atlas()
        all_ids = list(atlas.vector_to_leaf.keys())
        # Give all media low scores (below threshold).  The high outlier sits
        # on the root's most typical media so it is inside the surprise pool
        # under either pooling branch.
        scores = {str(vid): 0.1 for vid in all_ids}
        highest_id = atlas.nodes["0"]["ids"][0]
        scores[str(highest_id)] = 0.3  # still below threshold

        resp = client.post(
            "/api/coverage-atlas/next",
            json={"scores": scores, "threshold": 0.5},
        )
        data = resp.get_json()
        # Median is ~0.1 which is < 0.5, so highest-scored element returned
        assert data["id"] == highest_id


# ---------------------------------------------------------------------------
# Vote integration: voting via the API should update the atlas
# ---------------------------------------------------------------------------


class TestVoteUpdatesAtlas:
    def test_good_vote_labels_atlas(self, client):
        atlas = _build_atlas()
        cid = 1
        resp = client.post(f"/api/medias/{cid}/vote", json={"target": "good"})
        assert resp.status_code == 200
        assert cid in atlas.labeled_ids
        assert atlas.nodes["0"]["n_pos"] == 1

    def test_bad_vote_labels_atlas(self, client):
        atlas = _build_atlas()
        cid = 2
        resp = client.post(f"/api/medias/{cid}/vote", json={"target": "bad"})
        assert resp.status_code == 200
        assert cid in atlas.labeled_ids
        assert atlas.nodes["0"]["n_neg"] == 1

    def test_unvote_unlabels_atlas(self, client):
        atlas = _build_atlas()
        cid = 1
        # Vote good
        client.post(f"/api/medias/{cid}/vote", json={"target": "good"})
        assert cid in atlas.labeled_ids
        # Un-vote (absolute target=none)
        client.post(f"/api/medias/{cid}/vote", json={"target": "none"})
        assert cid not in atlas.labeled_ids

    def test_switch_vote_moves_evidence(self, client):
        """Switching from good to bad keeps the media labeled and flips the channel."""
        atlas = _build_atlas()
        cid = 3
        client.post(f"/api/medias/{cid}/vote", json={"target": "good"})
        assert cid in atlas.labeled_ids
        assert atlas.nodes["0"]["n_pos"] == 1
        # Switch to bad
        client.post(f"/api/medias/{cid}/vote", json={"target": "bad"})
        assert cid in atlas.labeled_ids
        assert atlas.nodes["0"]["n_pos"] == 0
        assert atlas.nodes["0"]["n_neg"] == 1


# ---------------------------------------------------------------------------
# Label import integration
# ---------------------------------------------------------------------------


class TestLabelImportUpdatesAtlas:
    def test_import_labels_atlas(self, client):
        atlas = _build_atlas()
        cid = 1
        md5 = medias[cid]["md5"]
        resp = client.post(
            "/api/labels/import",
            json={"labels": [{"md5": md5, "label": "good"}]},
        )
        assert resp.status_code == 200
        assert cid in atlas.labeled_ids


# ---------------------------------------------------------------------------
# Span level progression
# ---------------------------------------------------------------------------


class TestSpanLevelProgression:
    def test_span_advances_with_next_sample_labeling(self):
        """Following next_sample suggestions should advance the coverage level."""
        atlas = _build_atlas()
        assert atlas.coverage_level() == 0

        # Label medias guided by next_sample until the atlas is exhausted or
        # we've done enough iterations to cover the full atlas.
        max_iters = len(medias) + 1
        for _ in range(max_iters):
            sample = coverage_atlas_next_sample()
            if sample is None:
                break
            coverage_atlas_label(sample, good=True)

        # After labeling all suggested medias the atlas must be fully covered
        final_level = atlas.coverage_level()
        assert final_level == atlas.total_nodes, (
            f"Expected coverage_level {atlas.total_nodes} (fully covered) but got {final_level}"
        )

    def test_span_advances_beyond_zero_via_votes(self, client):
        """Voting on medias from different subtrees advances the span level past 0."""
        atlas = _build_atlas()
        depth = atlas.depth()
        if depth < 1:
            pytest.skip("Atlas too shallow to test progression beyond 0")

        # Use next_sample to find medias from each depth-1 subtree and vote on them
        for _ in range(len(medias)):
            sample = coverage_atlas_next_sample()
            if sample is None:
                break
            resp = client.post(f"/api/medias/{sample}/vote", json={"target": "good"})
            assert resp.status_code == 200

        # After voting on diverse medias, the level should have advanced past 1
        assert atlas.coverage_level() >= 2, f"Expected coverage_level >= 2 but got {atlas.coverage_level()}"

    def test_labeling_status_reflects_span_progression(self, client):
        """The /api/labeling-status endpoint should reflect advancing span level."""
        atlas = _build_atlas()

        # Before any labels, span should be red with level 0
        resp = client.get("/api/labeling-status")
        data = resp.get_json()
        assert data["span"]["level"] == 0
        assert data["span"]["status"] == "red"

        # Label all medias to fully cover the atlas
        for vid in atlas.vector_to_leaf:
            coverage_atlas_label(vid, good=True)
            good_votes[vid] = None

        resp = client.get("/api/labeling-status")
        data = resp.get_json()
        assert data["span"]["level"] == atlas.total_nodes
        assert data["span"]["status"] == "green"


# ---------------------------------------------------------------------------
# Label importer updates coverage atlas
# ---------------------------------------------------------------------------


class TestLabelImporterUpdatesAtlas:
    def test_label_importer_updates_atlas(self, client, tmp_path):
        """Labels imported via /api/label-importers/import should update the atlas."""
        import json

        atlas = _build_atlas()
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
        assert cid in atlas.labeled_ids


# ---------------------------------------------------------------------------
# Diversity level over time (progress API)
# ---------------------------------------------------------------------------


class TestDiversityLevelOverTime:
    def test_progress_includes_diversity_level_over_time(self, client):
        """The /api/labeling-progress response should include diversity_level_over_time."""
        _build_atlas()

        # Need at least one good and one bad vote + label history
        good_votes[1] = None
        bad_votes[2] = None
        add_label_to_history(1, "good")
        add_label_to_history(2, "bad")
        coverage_atlas_label(1, good=True)
        coverage_atlas_label(2, good=False)

        resp = client.post("/api/labeling-progress")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "diversity_level_over_time" in data
        assert isinstance(data["diversity_level_over_time"], list)
        assert len(data["diversity_level_over_time"]) == 2

    def test_diversity_level_entries_have_expected_fields(self, client):
        """Each entry should have num_labels, diversity_level, and depth."""
        _build_atlas()

        good_votes[1] = None
        bad_votes[2] = None
        add_label_to_history(1, "good")
        add_label_to_history(2, "bad")
        coverage_atlas_label(1, good=True)
        coverage_atlas_label(2, good=False)

        resp = client.post("/api/labeling-progress")
        data = resp.get_json()
        for entry in data["diversity_level_over_time"]:
            assert "num_labels" in entry
            assert "diversity_level" in entry
            assert "depth" in entry

    def test_diversity_level_monotonically_increases(self, client):
        """Diversity level should not decrease when only adding labels."""
        atlas = _build_atlas()

        # Label several medias from different parts of the atlas
        cids = sorted(atlas.vector_to_leaf.keys())[:6]
        for i, cid in enumerate(cids):
            if i % 2 == 0:
                good_votes[cid] = None
                add_label_to_history(cid, "good")
                coverage_atlas_label(cid, good=True)
            else:
                bad_votes[cid] = None
                add_label_to_history(cid, "bad")
                coverage_atlas_label(cid, good=False)

        resp = client.post("/api/labeling-progress")
        data = resp.get_json()
        levels = [e["diversity_level"] for e in data["diversity_level_over_time"]]
        for i in range(1, len(levels)):
            assert levels[i] >= levels[i - 1], f"Diversity level decreased at step {i}: {levels[i - 1]} -> {levels[i]}"

    def test_diversity_level_no_drop_after_vote_polarity_switch(self, client):
        """Switching a vote from bad→good must not cause a diversity dip in the progress cache."""
        atlas = _build_atlas()

        # Label several medias to build up diversity; need both good and bad
        cids = sorted(atlas.vector_to_leaf.keys())[:6]
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

        # Fetch diversity history again; no entry should be below the pre-switch peak
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
        _build_atlas()
        coverage_atlas_label(1, good=True)
        good_votes[1] = None

        resp = client.get("/api/labeling-status")
        data = resp.get_json()
        assert "diversity_level" in data["span"]
        assert isinstance(data["span"]["diversity_level"], (int, float))


# ---------------------------------------------------------------------------
# Audit M14; `/api/dataset/clear` must not leave a stale coverage atlas
# ---------------------------------------------------------------------------


class TestClearDatasetClearsAtlas:
    """Regression for audit finding M14.

    After ``/api/dataset/clear`` the active dataset's coverage atlas must
    not return media IDs that no longer exist in ``medias``.  In current
    code ``clear_medias`` nulls ``ctx.coverage_atlas`` and the unregister
    path makes the cleared context unreachable; the tests here pin that
    behavior so a future refactor cannot reintroduce the stale-ID bug.
    """

    def test_next_sample_returns_none_after_clear(self, client):
        """POST /api/dataset/clear → GET /api/coverage-atlas/next returns id=None."""
        _build_atlas()
        # Sanity: atlas exists and returns an id.
        pre = client.post("/api/coverage-atlas/next", json={})
        assert pre.status_code == 200
        assert pre.get_json()["id"] is not None

        saved = dict(medias)
        try:
            resp = client.post("/api/dataset/clear")
            assert resp.status_code == 200

            resp = client.post("/api/coverage-atlas/next", json={})
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["id"] is None
            assert data["coverage_level"] == 0
            # No atlas != exhausted (matches `test_returns_null_when_no_atlas`).
            assert data["exhausted"] is False
        finally:
            medias.update(saved)

    def test_clear_medias_nulls_coverage_atlas(self):
        """``clear_medias`` releases the atlas so a stale reference can't survive."""
        from vtsearch.state import clear_medias

        _build_atlas()
        assert get_coverage_atlas() is not None
        saved = dict(medias)
        try:
            clear_medias()
            assert get_coverage_atlas() is None
        finally:
            medias.update(saved)


# ---------------------------------------------------------------------------
# R1: `/api/votes/clear` must reset the coverage atlas's evidence state
# ---------------------------------------------------------------------------


class TestClearVotesResetsAtlas:
    """``clear_votes`` must reset the atlas's evidence channels.

    Without the reset, ``next_sample`` keeps skipping nodes that the
    just-cleared votes had marked covered and the diversity-level UI stays
    elevated despite zero labels.
    """

    def test_clear_votes_resets_evidence(self, client):
        atlas = _build_atlas()
        # Vote on multiple medias so several nodes become covered.
        for cid in list(atlas.vector_to_leaf)[:3]:
            resp = client.post(f"/api/medias/{cid}/vote", json={"target": "good"})
            assert resp.status_code == 200
        assert atlas.coverage_level() > 0
        assert len(atlas.labeled_ids) > 0

        resp = client.post("/api/votes/clear")
        assert resp.status_code == 200

        assert atlas.coverage_level() == 0
        assert atlas.labeled_ids == set()
        assert all(n["n_pos"] == 0 and n["n_neg"] == 0 for n in atlas.nodes.values())

    def test_next_sample_starts_over_after_clear_votes(self, client):
        """After clearing votes, the very first uncovered node is the root again."""
        atlas = _build_atlas()
        # Pick `next_sample`'s first suggestion and label it via the API so
        # the atlas picks it up.  Then clear and verify next_sample once
        # more returns a valid (now uncovered) media id.
        first = coverage_atlas_next_sample()
        assert first is not None
        resp = client.post(f"/api/medias/{first}/vote", json={"target": "good"})
        assert resp.status_code == 200
        assert first in atlas.labeled_ids

        resp = client.post("/api/votes/clear")
        assert resp.status_code == 200

        # With zero covered nodes the BFS walk starts from the root again and
        # may pick the same id; the important part is that we get a real
        # in-medias id rather than None.
        nxt = coverage_atlas_next_sample()
        assert nxt is not None
        assert nxt in medias


# ---------------------------------------------------------------------------
# R2: Detector swap on the same dataset must re-derive the atlas's evidence
# ---------------------------------------------------------------------------


class TestDetectorSwapResyncsAtlas:
    """``ensure_votes_match_active_dataset`` must replay the new detector's
    votes onto the dataset's coverage atlas.

    Restored labels are applied with ``silent=True`` (see
    ``label_restoration.py``), which skips the per-vote atlas update.
    Without the bulk re-sync, the atlas would keep reflecting the previous
    detector's evidence state.
    """

    def test_resync_coverage_atlas_to_detector_resets_then_replays(self):
        from vtscore.state import (
            DetectorContext,
            get_active_context,
            resync_coverage_atlas_to_detector,
        )

        atlas = _build_atlas()
        ids = list(atlas.vector_to_leaf)
        # Mark a couple of ids as covered via the per-vote helper.
        for cid in ids[:2]:
            coverage_atlas_label(cid, good=True)
        assert len(atlas.labeled_ids) > 0

        # Build a detector context that "votes" only on a different id so
        # the replay should produce a different evidence state.
        det = DetectorContext("test-det")
        if len(ids) > 2:
            det.good_votes[ids[2]] = None

        resync_coverage_atlas_to_detector(get_active_context(), det)

        # The previous evidence must be gone, replaced by the leaves of the
        # newly-labeled id (and its ancestors).
        assert atlas.labeled_ids == set(det.good_votes) | set(det.bad_votes)
        if len(ids) > 2:
            assert ids[2] in atlas.labeled_ids
            assert _covered(atlas, atlas.lookup(ids[2]))
            for old_cid in ids[:2]:
                assert old_cid not in atlas.labeled_ids
        else:
            # Tiny test corpus; at minimum, the reset half must have run.
            assert atlas.labeled_ids == set()

    def test_resync_is_noop_when_no_atlas(self):
        """``resync_coverage_atlas_to_detector`` tolerates a missing atlas."""
        from vtscore.state import (
            DatasetContext,
            DetectorContext,
            resync_coverage_atlas_to_detector,
        )

        ds = DatasetContext("no-atlas")
        det = DetectorContext("d")
        det.good_votes[1] = None
        # Must not raise.
        resync_coverage_atlas_to_detector(ds, det)
