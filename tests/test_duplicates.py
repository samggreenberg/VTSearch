"""Tests for duplicate-content collapsing.

Covers:
- ``collapse_duplicates`` function (pure state manipulation)
- ``get_dupe_count`` helper
- ``/api/dataset/status`` reporting ``num_dupes``
- Label export expanding dupe sets to all members
- Label import collapsing by MD5 back to the representative
"""

import copy

import numpy as np

import app as app_module
from vtsearch.datasets.labelset import LabelSet
from vtsearch.utils.state import collapse_duplicates, get_dupe_count


def _make_media(cid, md5="unique", origin_importer="test", filename=None, category="cat"):
    """Helper: build a minimal media dict."""
    fname = filename or f"media_{cid}.wav"
    return {
        "id": cid,
        "type": "audio",
        "duration": 1.0,
        "file_size": 100,
        "md5": md5,
        "embedding": np.zeros(3, dtype=np.float32),
        "media_bytes": b"\x00",
        "filename": fname,
        "category": category,
        "origin": {"importer": origin_importer, "params": {"path": f"/data/{cid}"}},
        "origin_name": fname,
    }


class TestCollapseDuplicates:
    def test_no_dupes_returns_zero(self):
        media_dict = {
            1: _make_media(1, md5="aaa"),
            2: _make_media(2, md5="bbb"),
            3: _make_media(3, md5="ccc"),
        }
        count = collapse_duplicates(media_dict)
        assert count == 0
        assert len(media_dict) == 3

    def test_single_dupe_group(self):
        media_dict = {
            1: _make_media(1, md5="same"),
            2: _make_media(2, md5="same"),
            3: _make_media(3, md5="unique"),
        }
        count = collapse_duplicates(media_dict)
        assert count == 1
        assert len(media_dict) == 2
        # Representative (first) is kept
        assert 1 in media_dict
        assert 3 in media_dict
        # Duplicate removed
        assert 2 not in media_dict

    def test_multiple_dupe_groups(self):
        media_dict = {
            1: _make_media(1, md5="aaa"),
            2: _make_media(2, md5="aaa"),
            3: _make_media(3, md5="bbb"),
            4: _make_media(4, md5="bbb"),
            5: _make_media(5, md5="ccc"),
        }
        count = collapse_duplicates(media_dict)
        assert count == 2
        assert len(media_dict) == 3
        assert 1 in media_dict
        assert 3 in media_dict
        assert 5 in media_dict

    def test_three_way_dupe(self):
        media_dict = {
            1: _make_media(1, md5="same"),
            2: _make_media(2, md5="same"),
            3: _make_media(3, md5="same"),
        }
        count = collapse_duplicates(media_dict)
        assert count == 1
        assert len(media_dict) == 1
        assert 1 in media_dict

    def test_representative_gets_dupe_set_origin(self):
        media_dict = {
            1: _make_media(1, md5="same", filename="first.wav"),
            2: _make_media(2, md5="same", filename="second.wav"),
        }
        collapse_duplicates(media_dict)
        rep = media_dict[1]
        assert rep["origin"]["importer"] == "dupe_set"
        assert rep["origin"]["params"]["name"] == "first.wav"

    def test_members_list_contains_all_duplicates(self):
        media_dict = {
            1: _make_media(1, md5="same", filename="a.wav", origin_importer="folder"),
            2: _make_media(2, md5="same", filename="b.wav", origin_importer="http_archive"),
            3: _make_media(3, md5="same", filename="c.wav", origin_importer="demo"),
        }
        collapse_duplicates(media_dict)
        members = media_dict[1]["origin"]["members"]
        assert len(members) == 3
        assert members[0]["filename"] == "a.wav"
        assert members[1]["filename"] == "b.wav"
        assert members[2]["filename"] == "c.wav"
        # Each member retains its original origin
        assert members[0]["origin"]["importer"] == "folder"
        assert members[1]["origin"]["importer"] == "http_archive"
        assert members[2]["origin"]["importer"] == "demo"

    def test_md5_preserved_on_representative(self):
        media_dict = {
            1: _make_media(1, md5="abc123"),
            2: _make_media(2, md5="abc123"),
        }
        collapse_duplicates(media_dict)
        assert media_dict[1]["md5"] == "abc123"

    def test_empty_dict(self):
        media_dict = {}
        count = collapse_duplicates(media_dict)
        assert count == 0


class TestGetDupeCount:
    def test_no_dupes(self):
        media_dict = {
            1: _make_media(1, md5="aaa"),
            2: _make_media(2, md5="bbb"),
        }
        assert get_dupe_count(media_dict) == 0

    def test_with_dupes(self):
        media_dict = {
            1: _make_media(1, md5="same"),
            2: _make_media(2, md5="same"),
            3: _make_media(3, md5="unique"),
        }
        collapse_duplicates(media_dict)
        assert get_dupe_count(media_dict) == 1

    def test_uses_global_medias_by_default(self):
        saved = copy.deepcopy(app_module.medias)
        app_module.medias.clear()
        try:
            app_module.medias[1] = _make_media(1, md5="same")
            app_module.medias[2] = _make_media(2, md5="same")
            collapse_duplicates(app_module.medias)
            assert get_dupe_count() == 1
        finally:
            app_module.medias.clear()
            app_module.medias.update(saved)


class TestDatasetStatusDupes:
    def test_status_includes_num_dupes(self, client):
        resp = client.get("/api/dataset/status")
        data = resp.get_json()
        assert "num_dupes" in data
        # Default test medias have no dupes
        assert data["num_dupes"] == 0

    def test_status_reports_dupes_after_collapse(self, client):
        # Temporarily create a dupe (deepcopy because collapse mutates in place)
        saved = copy.deepcopy(app_module.medias)
        app_module.medias[999] = copy.deepcopy(app_module.medias[1])
        app_module.medias[999]["id"] = 999
        app_module.medias[999]["filename"] = "dupe.wav"
        # Same MD5 => duplicate
        collapse_duplicates(app_module.medias)
        try:
            resp = client.get("/api/dataset/status")
            data = resp.get_json()
            assert data["num_dupes"] == 1
        finally:
            app_module.medias.clear()
            app_module.medias.update(saved)


class TestLabelExportExpandsDupes:
    def test_export_expands_dupe_set(self, client):
        """Voting on a dupe-set representative exports labels for all members."""
        # Set up a dupe-set representative
        saved = dict(app_module.medias)
        rep = copy.deepcopy(app_module.medias[1])
        rep["origin"] = {
            "importer": "dupe_set",
            "params": {"name": "a.wav"},
            "members": [
                {
                    "origin": {"importer": "folder", "params": {"path": "/data"}},
                    "origin_name": "a.wav",
                    "filename": "a.wav",
                    "category": "cat1",
                },
                {
                    "origin": {"importer": "folder", "params": {"path": "/data"}},
                    "origin_name": "b.wav",
                    "filename": "b.wav",
                    "category": "cat2",
                },
            ],
        }
        app_module.medias[1] = rep
        try:
            # Vote on the representative
            app_module.good_votes[1] = None
            resp = client.get("/api/labels/export")
            data = resp.get_json()
            labels = data["labels"]
            # Should have 2 entries (one per member), not 1
            dupe_labels = [l for l in labels if l["md5"] == rep["md5"]]
            assert len(dupe_labels) == 2
            filenames = {l["filename"] for l in dupe_labels}
            assert filenames == {"a.wav", "b.wav"}
            # Both have the same label
            assert all(l["label"] == "good" for l in dupe_labels)
        finally:
            app_module.medias.clear()
            app_module.medias.update(saved)

    def test_export_non_dupe_media_unchanged(self, client):
        """Normal (non-dupe) medias still export a single label entry."""
        app_module.good_votes[1] = None
        resp = client.get("/api/labels/export")
        data = resp.get_json()
        assert len(data["labels"]) == 1
        assert data["labels"][0]["label"] == "good"

    def test_roundtrip_through_dupe_collapse(self, client):
        """Export from dupes, then import back into a dataset with dupes uncollapsed."""
        saved = dict(app_module.medias)
        md5 = app_module.medias[1]["md5"]

        # Create dupe-set representative
        rep = copy.deepcopy(app_module.medias[1])
        rep["origin"] = {
            "importer": "dupe_set",
            "params": {"name": "a.wav"},
            "members": [
                {"origin": {"importer": "test", "params": {}}, "origin_name": "a.wav", "filename": "a.wav", "category": "c"},
                {"origin": {"importer": "test", "params": {}}, "origin_name": "b.wav", "filename": "b.wav", "category": "c"},
            ],
        }
        app_module.medias[1] = rep
        app_module.good_votes[1] = None

        resp = client.get("/api/labels/export")
        exported = resp.get_json()
        assert len(exported["labels"]) == 2

        # Clear votes and import back — both entries share the same MD5,
        # so they should match the single representative.
        app_module.good_votes.clear()
        resp = client.post("/api/labels/import", json=exported)
        data = resp.get_json()
        # Both entries resolve to the same media (by MD5), so applied == 2
        assert data["applied"] == 2
        assert 1 in app_module.good_votes

        app_module.medias.clear()
        app_module.medias.update(saved)


class TestLabelImportWithDupes:
    def test_import_matches_representative_by_md5(self, client):
        """Importing a label for any original dupe MD5 matches the representative."""
        saved = dict(app_module.medias)
        md5 = app_module.medias[1]["md5"]

        rep = copy.deepcopy(app_module.medias[1])
        rep["origin"] = {
            "importer": "dupe_set",
            "params": {"name": "a.wav"},
            "members": [
                {"origin": {"importer": "test", "params": {}}, "origin_name": "a.wav", "filename": "a.wav", "category": "c"},
                {"origin": {"importer": "test", "params": {}}, "origin_name": "b.wav", "filename": "b.wav", "category": "c"},
            ],
        }
        app_module.medias[1] = rep
        try:
            # Import using the shared MD5
            labels = [{"md5": md5, "label": "good"}]
            resp = client.post("/api/labels/import", json={"labels": labels})
            data = resp.get_json()
            assert data["applied"] == 1
            assert 1 in app_module.good_votes
        finally:
            app_module.medias.clear()
            app_module.medias.update(saved)


class TestCollapseDuplicatesIntegration:
    def test_collapse_then_vote_then_export(self):
        """Full workflow: collapse dupes, vote, export, verify expansion."""
        media_dict = {
            1: _make_media(1, md5="same", filename="x.wav"),
            2: _make_media(2, md5="same", filename="y.wav"),
            3: _make_media(3, md5="other", filename="z.wav"),
        }
        collapse_duplicates(media_dict)

        # Vote on the representative
        good = {1: None}
        bad = {3: None}
        ls = LabelSet.from_clips_and_votes(media_dict, good, bad)
        exported = ls.to_dict()

        # The dupe group (md5="same") should produce 2 good labels
        good_labels = [e for e in exported["labels"] if e["label"] == "good"]
        bad_labels = [e for e in exported["labels"] if e["label"] == "bad"]
        assert len(good_labels) == 2
        assert len(bad_labels) == 1
        assert {l["filename"] for l in good_labels} == {"x.wav", "y.wav"}
