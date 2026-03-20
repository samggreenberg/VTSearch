"""Tests for the corrections tracking and export category.

Covers:
- _find_initial_labels state tracking
- is_correction annotation on label export entries
- label_filter=corrections filtering
- Corrections cleared when votes are cleared
"""

from __future__ import annotations

import app as app_module
from vtsearch.utils import (
    get_find_initial_labels,
    set_find_initial_labels,
)
import vtsearch.utils.state_core as _core


class TestFindInitialLabelsState:
    """Tests for the _find_initial_labels state management."""

    def test_set_and_get_initial_labels(self):
        """set_find_initial_labels stores labels, get returns a copy."""
        labels = {1: "good", 2: "bad", 3: "good"}
        set_find_initial_labels(labels)
        result = get_find_initial_labels()
        assert result == labels
        # Verify it's a copy, not the same object
        result[99] = "good"
        assert 99 not in get_find_initial_labels()

    def test_initial_labels_cleared_on_clear_votes(self, client):
        """clear_votes should also clear _find_initial_labels."""
        set_find_initial_labels({1: "good", 2: "bad"})
        assert len(get_find_initial_labels()) == 2

        resp = client.post("/api/votes/clear")
        assert resp.status_code == 200
        assert get_find_initial_labels() == {}

    def test_initial_labels_empty_by_default(self):
        """Before any find-label run, initial labels should be empty."""
        assert get_find_initial_labels() == {}

    def test_set_replaces_previous(self):
        """Setting initial labels replaces the previous snapshot."""
        set_find_initial_labels({1: "good"})
        set_find_initial_labels({2: "bad"})
        result = get_find_initial_labels()
        assert result == {2: "bad"}


class TestIsCorrectionAnnotation:
    """Tests for the is_correction field in label export."""

    def test_no_corrections_without_find_labels(self, client):
        """When no find-label has run, is_correction should not be present."""
        app_module.good_votes[1] = None
        resp = client.get("/api/labels/export")
        data = resp.get_json()
        assert len(data["labels"]) == 1
        # No find_initial_labels => no is_correction field
        assert "is_correction" not in data["labels"][0]

    def test_is_correction_false_when_unchanged(self, client):
        """Items that match the detector's original label are not corrections."""
        app_module.good_votes.update({1: None, 2: None})
        app_module.bad_votes.update({3: None})
        set_find_initial_labels({1: "good", 2: "good", 3: "bad"})

        resp = client.get("/api/labels/export")
        data = resp.get_json()
        for entry in data["labels"]:
            assert entry["is_correction"] is False

    def test_is_correction_true_when_changed(self, client):
        """Items where the user changed the label are marked as corrections."""
        # Detector initially said: 1=good, 2=bad
        set_find_initial_labels({1: "good", 2: "bad"})
        # User flipped both
        app_module.bad_votes[1] = None  # was good, now bad
        app_module.good_votes[2] = None  # was bad, now good

        resp = client.get("/api/labels/export")
        data = resp.get_json()
        assert len(data["labels"]) == 2
        for entry in data["labels"]:
            assert entry["is_correction"] is True

    def test_mixed_corrections_and_unchanged(self, client):
        """Mix of corrected and unchanged labels reports correctly."""
        set_find_initial_labels({1: "good", 2: "bad", 3: "good"})
        # 1 stays good (no correction), 2 flipped to good (correction),
        # 3 flipped to bad (correction)
        app_module.good_votes.update({1: None, 2: None})
        app_module.bad_votes[3] = None

        resp = client.get("/api/labels/export")
        data = resp.get_json()
        corrections = {e["md5"]: e["is_correction"] for e in data["labels"]}
        md5_1 = app_module.medias[1]["md5"]
        md5_2 = app_module.medias[2]["md5"]
        md5_3 = app_module.medias[3]["md5"]
        assert corrections[md5_1] is False
        assert corrections[md5_2] is True
        assert corrections[md5_3] is True


class TestCorrectionsFilter:
    """Tests for label_filter=corrections."""

    def test_corrections_filter_returns_only_changed(self, client):
        """label_filter=corrections returns only items the user changed."""
        set_find_initial_labels({1: "good", 2: "bad", 3: "good"})
        app_module.good_votes.update({1: None, 2: None})  # 2 was bad -> correction
        app_module.bad_votes[3] = None  # 3 was good -> correction

        resp = client.get("/api/labels/export?label_filter=corrections")
        data = resp.get_json()
        assert len(data["labels"]) == 2
        md5s = {e["md5"] for e in data["labels"]}
        assert app_module.medias[2]["md5"] in md5s
        assert app_module.medias[3]["md5"] in md5s
        assert all(e["is_correction"] is True for e in data["labels"])

    def test_corrections_filter_empty_when_no_changes(self, client):
        """label_filter=corrections returns nothing if user didn't change any labels."""
        set_find_initial_labels({1: "good", 2: "bad"})
        app_module.good_votes[1] = None
        app_module.bad_votes[2] = None

        resp = client.get("/api/labels/export?label_filter=corrections")
        data = resp.get_json()
        assert data["labels"] == []

    def test_corrections_filter_empty_without_find_labels(self, client):
        """label_filter=corrections returns all labels when no find-label has run."""
        app_module.good_votes[1] = None
        app_module.bad_votes[2] = None

        resp = client.get("/api/labels/export?label_filter=corrections")
        data = resp.get_json()
        # Without find_initial_labels, corrections filter still works
        # (no items have is_correction=True, so empty result)
        assert data["labels"] == []

    def test_other_filters_still_work(self, client):
        """Existing good/bad/both filters are unaffected by corrections feature."""
        set_find_initial_labels({1: "good", 2: "bad"})
        app_module.good_votes[1] = None
        app_module.bad_votes[2] = None

        resp = client.get("/api/labels/export?label_filter=good")
        data = resp.get_json()
        assert len(data["labels"]) == 1
        assert data["labels"][0]["label"] == "good"

        resp = client.get("/api/labels/export?label_filter=bad")
        data = resp.get_json()
        assert len(data["labels"]) == 1
        assert data["labels"][0]["label"] == "bad"

        resp = client.get("/api/labels/export?label_filter=both")
        data = resp.get_json()
        assert len(data["labels"]) == 2
