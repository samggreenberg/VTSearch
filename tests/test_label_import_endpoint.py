"""Tests for label import API endpoints and resolution helpers.

Covers:
- POST /api/label-importers/import/<name> endpoint
- resolve_media_ids: union matching (origin+name AND md5)
- find_missing_entries
- next_media_id
- Missing element handling in label import responses
"""

from __future__ import annotations

import json


import app as app_module


# ---------------------------------------------------------------------------
# API – POST /api/label-importers/import/<name>
# ---------------------------------------------------------------------------


class TestLabelImportEndpoint:
    def test_unknown_importer_returns_404(self, client):
        res = client.post("/api/label-importers/import/no_such_importer")
        assert res.status_code == 404
        assert "no_such_importer" in res.get_json()["error"]

    def test_json_importer_applies_good_label(self, client, tmp_path):
        md5 = app_module.medias[1]["md5"]
        payload = json.dumps({"labels": [{"md5": md5, "label": "good"}]})
        p = tmp_path / "labels.json"
        p.write_text(payload)
        res = client.post(
            "/api/label-importers/import/server_json_file",
            json={"filepath": str(p)},
        )
        assert res.status_code == 200
        result = res.get_json()
        assert result["applied"] == 1
        assert result["skipped"] == 0
        assert result["missing_count"] == 0
        assert 1 in app_module.good_votes

    def test_json_importer_applies_bad_label(self, client, tmp_path):
        md5 = app_module.medias[2]["md5"]
        payload = json.dumps({"labels": [{"md5": md5, "label": "bad"}]})
        p = tmp_path / "labels.json"
        p.write_text(payload)
        res = client.post(
            "/api/label-importers/import/server_json_file",
            json={"filepath": str(p)},
        )
        assert res.status_code == 200
        assert 2 in app_module.bad_votes

    def test_json_importer_reports_unknown_md5_as_missing(self, client, tmp_path):
        payload = json.dumps({"labels": [{"md5": "no_such_md5", "label": "good"}]})
        p = tmp_path / "labels.json"
        p.write_text(payload)
        res = client.post(
            "/api/label-importers/import/server_json_file",
            json={"filepath": str(p)},
        )
        assert res.status_code == 200
        result = res.get_json()
        assert result["applied"] == 0
        assert result["missing_count"] == 1
        assert len(result["missing"]) == 1

    def test_json_importer_skips_invalid_label_value(self, client, tmp_path):
        md5 = app_module.medias[1]["md5"]
        payload = json.dumps({"labels": [{"md5": md5, "label": "meh"}]})
        p = tmp_path / "labels.json"
        p.write_text(payload)
        res = client.post(
            "/api/label-importers/import/server_json_file",
            json={"filepath": str(p)},
        )
        assert res.status_code == 200
        result = res.get_json()
        assert result["applied"] == 0
        assert result["skipped"] == 1

    def test_csv_importer_applies_labels(self, client, tmp_path):
        md5_1 = app_module.medias[1]["md5"]
        md5_2 = app_module.medias[2]["md5"]
        csv_text = f"md5,label\n{md5_1},good\n{md5_2},bad\n"
        p = tmp_path / "labels.csv"
        p.write_text(csv_text)
        res = client.post(
            "/api/label-importers/import/server_csv_file",
            json={"filepath": str(p)},
        )
        assert res.status_code == 200
        result = res.get_json()
        assert result["applied"] == 2
        assert 1 in app_module.good_votes
        assert 2 in app_module.bad_votes

    def test_csv_importer_reports_unknown_md5_as_missing(self, client, tmp_path):
        csv_text = "md5,label\nunknown_hash,good\n"
        p = tmp_path / "labels.csv"
        p.write_text(csv_text)
        res = client.post(
            "/api/label-importers/import/server_csv_file",
            json={"filepath": str(p)},
        )
        assert res.status_code == 200
        result = res.get_json()
        assert result["applied"] == 0
        assert result["missing_count"] == 1

    def test_import_overrides_existing_label(self, client, tmp_path):
        app_module.good_votes[1] = None
        md5 = app_module.medias[1]["md5"]
        payload = json.dumps({"labels": [{"md5": md5, "label": "bad"}]})
        p = tmp_path / "labels.json"
        p.write_text(payload)
        client.post(
            "/api/label-importers/import/server_json_file",
            json={"filepath": str(p)},
        )
        assert 1 not in app_module.good_votes
        assert 1 in app_module.bad_votes

    def test_import_response_has_message(self, client, tmp_path):
        payload = json.dumps({"labels": []})
        p = tmp_path / "labels.json"
        p.write_text(payload)
        res = client.post(
            "/api/label-importers/import/server_json_file",
            json={"filepath": str(p)},
        )
        assert res.status_code == 200
        assert "message" in res.get_json()

    def test_json_roundtrip_via_importer(self, client, tmp_path):
        """Export labels via the old route and re-import via label importer endpoint."""
        app_module.good_votes.update({k: None for k in [1, 3, 5]})
        app_module.bad_votes.update({k: None for k in [2, 4]})

        export_res = client.get("/api/labels/export")
        exported = export_res.get_json()

        app_module.good_votes.clear()
        app_module.bad_votes.clear()

        p = tmp_path / "labels.json"
        p.write_text(json.dumps(exported))
        res = client.post(
            "/api/label-importers/import/server_json_file",
            json={"filepath": str(p)},
        )
        result = res.get_json()
        assert result["applied"] == 5
        assert set(app_module.good_votes) == {1, 3, 5}
        assert set(app_module.bad_votes) == {2, 4}

    def test_multiple_clips_via_csv(self, client, tmp_path):
        lines = ["md5,label"]
        good_ids = [1, 2, 3]
        bad_ids = [4, 5]
        for cid in good_ids:
            lines.append(f"{app_module.medias[cid]['md5']},good")
        for cid in bad_ids:
            lines.append(f"{app_module.medias[cid]['md5']},bad")
        csv_text = "\n".join(lines)
        p = tmp_path / "labels.csv"
        p.write_text(csv_text)
        res = client.post(
            "/api/label-importers/import/server_csv_file",
            json={"filepath": str(p)},
        )
        result = res.get_json()
        assert result["applied"] == 5
        assert set(app_module.good_votes) == {1, 2, 3}
        assert set(app_module.bad_votes) == {4, 5}

    def test_import_syncs_model_registry_num_training(self, client, tmp_path):
        """Importing labels should update the loaded model's num_training in the registry."""
        from vtsearch.models.registry import add_loaded_model_id, get_model, register_model, reset_for_tests
        from vtsearch.settings import get_trainable_models_dir, set_trainable_models_dir
        from vtsearch.models.trainable_model_store import _write_model

        reset_for_tests()
        set_trainable_models_dir(str(tmp_path / "models"))

        # Create a trainable model file on disk
        tm_name = "test_import_sync"
        from pathlib import Path

        model_dir = Path(get_trainable_models_dir())
        model_dir.mkdir(parents=True, exist_ok=True)
        _write_model(model_dir / f"{tm_name}.json", {
            "name": tm_name,
            "media_type": "audio",
            "examples": [],
            "labelset": {"labels": []},
        })

        entry = register_model(
            name="Import Sync Test",
            media_type="audio",
            trainable=True,
            trainable_model_name=tm_name,
            num_training=0,
        )
        add_loaded_model_id(entry["id"])

        # Set active detector so sync_labels_to_loaded_model() can find it
        from vtsearch.utils.state_core import DetectorContext, register_detector_context, set_thread_detector_context

        det_ctx = DetectorContext(entry["id"])
        register_detector_context(det_ctx)
        set_thread_detector_context(det_ctx)

        # Import a label
        md5 = app_module.medias[1]["md5"]
        payload = json.dumps({"labels": [{"md5": md5, "label": "good"}]})
        p = tmp_path / "labels.json"
        p.write_text(payload)
        res = client.post(
            "/api/label-importers/import/server_json_file",
            json={"filepath": str(p)},
        )
        assert res.status_code == 200
        assert res.get_json()["applied"] == 1

        # The registry entry should now reflect the updated label count
        updated = get_model(entry["id"])
        assert updated["num_training"] == 1

    def test_path_traversal_absolute_rejected(self, client):
        """Absolute paths outside the allowed directory must be rejected."""
        res = client.post(
            "/api/label-importers/import/server_json_file",
            json={"filepath": "/etc/passwd"},
        )
        assert res.status_code == 400

    def test_path_traversal_relative_rejected(self, client):
        """Relative paths that escape the base directory must be rejected."""
        res = client.post(
            "/api/label-importers/import/server_csv_file",
            json={"filepath": "../../../etc/shadow"},
        )
        assert res.status_code == 400


# ---------------------------------------------------------------------------
# resolve_media_ids: union matching (origin+name AND md5)
# ---------------------------------------------------------------------------


class TestResolveClipIdsUnion:
    """Test that resolve_media_ids returns the union of origin+name and md5 matches."""

    def test_md5_match_only(self):
        from vtsearch.utils import build_media_lookup, resolve_media_ids

        # Entry with only md5, no origin — should match by md5
        md5 = app_module.medias[1]["md5"]
        origin_lookup, md5_lookup, _ = build_media_lookup(app_module.medias)
        entry = {"md5": md5, "label": "good"}
        cids = resolve_media_ids(entry, origin_lookup, md5_lookup)
        assert 1 in cids

    def test_origin_match_only(self):
        from vtsearch.utils import build_media_lookup, resolve_media_ids

        origin_lookup, md5_lookup, _ = build_media_lookup(app_module.medias)
        media = app_module.medias[1]
        entry = {"origin": media["origin"], "origin_name": media["origin_name"], "label": "good"}
        cids = resolve_media_ids(entry, origin_lookup, md5_lookup)
        assert 1 in cids

    def test_union_of_origin_and_md5(self):
        """When origin matches media A and md5 matches media B, both are returned."""
        from vtsearch.utils import build_media_lookup, resolve_media_ids

        # Give media 2 the same md5 as media 1 (simulate content-duplicate under different origin)
        orig_md5 = app_module.medias[2]["md5"]
        app_module.medias[2]["md5"] = app_module.medias[1]["md5"]
        try:
            origin_lookup, md5_lookup, _ = build_media_lookup(app_module.medias)
            clip1 = app_module.medias[1]
            # Entry: origin matches media 1, md5 also matches media 1 AND media 2
            entry = {
                "md5": clip1["md5"],
                "origin": clip1["origin"],
                "origin_name": clip1["origin_name"],
                "label": "good",
            }
            cids = resolve_media_ids(entry, origin_lookup, md5_lookup)
            assert 1 in cids
            assert 2 in cids
        finally:
            app_module.medias[2]["md5"] = orig_md5

    def test_no_match_returns_empty(self):
        from vtsearch.utils import build_media_lookup, resolve_media_ids

        origin_lookup, md5_lookup, _ = build_media_lookup(app_module.medias)
        entry = {
            "md5": "nonexistent",
            "origin": {"importer": "nope", "params": {}},
            "origin_name": "x",
            "label": "good",
        }
        cids = resolve_media_ids(entry, origin_lookup, md5_lookup)
        assert cids == []

    def test_name_fallback_matches_by_origin_name(self):
        """When md5 and origin don't match, name_lookup matches by origin_name alone."""
        from vtsearch.utils import build_media_lookup, resolve_media_ids

        origin_lookup, md5_lookup, name_lookup = build_media_lookup(app_module.medias)
        origin_name = app_module.medias[1].get("origin_name", "")
        assert origin_name, "Test media 1 must have origin_name"
        entry = {"md5": "wrong_md5", "origin_name": origin_name, "label": "good"}
        # Without name_lookup — no match
        cids = resolve_media_ids(entry, origin_lookup, md5_lookup)
        assert cids == []
        # With name_lookup — matches by origin_name
        cids = resolve_media_ids(entry, origin_lookup, md5_lookup, name_lookup)
        assert 1 in cids

    def test_name_fallback_uses_filename_when_no_origin_name(self):
        """name_lookup falls back to 'filename' key when origin_name is absent."""
        from vtsearch.utils import build_media_lookup, resolve_media_ids

        origin_lookup, md5_lookup, name_lookup = build_media_lookup(app_module.medias)
        origin_name = app_module.medias[1].get("origin_name", "")
        assert origin_name
        entry = {"md5": "wrong_md5", "filename": origin_name, "label": "good"}
        cids = resolve_media_ids(entry, origin_lookup, md5_lookup, name_lookup)
        assert 1 in cids

    def test_name_fallback_skipped_when_md5_matches(self):
        """Name fallback is NOT used when md5 already matches (avoids false positives)."""
        from vtsearch.utils import build_media_lookup, resolve_media_ids

        origin_lookup, md5_lookup, name_lookup = build_media_lookup(app_module.medias)
        md5 = app_module.medias[1]["md5"]
        entry = {"md5": md5, "origin_name": "some_other_name", "label": "good"}
        cids = resolve_media_ids(entry, origin_lookup, md5_lookup, name_lookup)
        assert 1 in cids


# ---------------------------------------------------------------------------
# find_missing_entries
# ---------------------------------------------------------------------------


class TestFindMissingEntries:
    def test_all_present_returns_empty(self):
        from vtsearch.utils import build_media_lookup, find_missing_entries

        origin_lookup, md5_lookup, _ = build_media_lookup(app_module.medias)
        entries = [{"md5": app_module.medias[i]["md5"], "label": "good"} for i in [1, 2, 3]]
        missing = find_missing_entries(entries, origin_lookup, md5_lookup)
        assert missing == []

    def test_unknown_entries_returned(self):
        from vtsearch.utils import build_media_lookup, find_missing_entries

        origin_lookup, md5_lookup, _ = build_media_lookup(app_module.medias)
        entries = [
            {"md5": app_module.medias[1]["md5"], "label": "good"},
            {"md5": "totally_unknown", "label": "bad"},
        ]
        missing = find_missing_entries(entries, origin_lookup, md5_lookup)
        assert len(missing) == 1
        assert missing[0]["md5"] == "totally_unknown"

    def test_invalid_labels_excluded(self):
        from vtsearch.utils import build_media_lookup, find_missing_entries

        origin_lookup, md5_lookup, _ = build_media_lookup(app_module.medias)
        entries = [
            {"md5": "unknown1", "label": "good"},
            {"md5": "unknown2", "label": "meh"},  # invalid label — excluded
        ]
        missing = find_missing_entries(entries, origin_lookup, md5_lookup)
        assert len(missing) == 1


# ---------------------------------------------------------------------------
# next_media_id
# ---------------------------------------------------------------------------


class TestNextClipId:
    def test_empty_dict_returns_1(self):
        from vtsearch.utils.state import next_media_id

        assert next_media_id({}) == 1

    def test_returns_max_plus_one(self):
        from vtsearch.utils.state import next_media_id

        assert next_media_id(app_module.medias) == max(app_module.medias) + 1


# ---------------------------------------------------------------------------
# API – missing elements in label import response
# ---------------------------------------------------------------------------


class TestLabelImportMissingElements:
    def test_response_includes_missing_entries(self, client, tmp_path):
        """Labels referencing unknown elements should appear in 'missing'."""
        known_md5 = app_module.medias[1]["md5"]
        payload = json.dumps(
            {
                "labels": [
                    {"md5": known_md5, "label": "good"},
                    {
                        "md5": "unknown_abc123",
                        "label": "bad",
                        "origin": {"importer": "folder", "params": {"path": "/data"}},
                        "origin_name": "mystery.wav",
                    },
                ]
            }
        )
        p = tmp_path / "labels.json"
        p.write_text(payload)
        res = client.post(
            "/api/label-importers/import/server_json_file",
            json={"filepath": str(p)},
        )
        assert res.status_code == 200
        result = res.get_json()
        assert result["applied"] == 1
        assert result["missing_count"] == 1
        assert result["missing"][0]["md5"] == "unknown_abc123"
        assert result["missing"][0]["origin"]["importer"] == "folder"

    def test_no_missing_when_all_match(self, client, tmp_path):
        md5 = app_module.medias[1]["md5"]
        payload = json.dumps({"labels": [{"md5": md5, "label": "good"}]})
        p = tmp_path / "labels.json"
        p.write_text(payload)
        res = client.post(
            "/api/label-importers/import/server_json_file",
            json={"filepath": str(p)},
        )
        result = res.get_json()
        assert result["missing_count"] == 0
        assert result["missing"] == []

    def test_auto_resolve_ingests_and_applies(self, client, tmp_path):
        """Missing elements are auto-resolved from their origin during import."""
        import hashlib

        # Save/restore medias since auto-resolve adds new entries
        saved = dict(app_module.medias)
        try:
            # Create a text file that can be ingested
            text_dir = tmp_path / "texts"
            text_dir.mkdir()
            content = "Hello world, this is a test paragraph for auto-resolve embedding."
            (text_dir / "hello.txt").write_text(content)
            md5 = hashlib.md5(content.encode()).hexdigest()

            origin = {"importer": "folder", "params": {"path": str(text_dir), "media_type": "text"}}

            known_md5 = app_module.medias[1]["md5"]
            payload = json.dumps(
                {
                    "labels": [
                        {"md5": known_md5, "label": "good"},
                        {
                            "md5": md5,
                            "label": "bad",
                            "origin": origin,
                            "origin_name": "hello.txt",
                            "filename": "hello.txt",
                        },
                    ]
                }
            )
            p = tmp_path / "labels.json"
            p.write_text(payload)
            res = client.post(
                "/api/label-importers/import/server_json_file",
                json={"filepath": str(p)},
            )
            assert res.status_code == 200
            result = res.get_json()
            # Both labels should be applied (1 existing + 1 auto-resolved)
            assert result["applied"] == 2
            assert result["ingested"] == 1
            assert result["missing_count"] == 0
            assert result["missing"] == []
            # The new media should be in the dataset and labeled
            new_ids = [cid for cid, m in app_module.medias.items() if m.get("md5") == md5]
            assert len(new_ids) == 1
            assert new_ids[0] in app_module.bad_votes
        finally:
            app_module.medias.clear()
            app_module.medias.update(saved)

    def test_auto_resolve_reports_unresolvable(self, client, tmp_path):
        """Elements that can't be resolved are reported in the response."""
        known_md5 = app_module.medias[1]["md5"]
        payload = json.dumps(
            {
                "labels": [
                    {"md5": known_md5, "label": "good"},
                    {
                        "md5": "totally_unknown_md5",
                        "label": "bad",
                        "origin": {"importer": "folder", "params": {"path": "/nonexistent/path"}},
                        "origin_name": "ghost.wav",
                    },
                ]
            }
        )
        p = tmp_path / "labels.json"
        p.write_text(payload)
        res = client.post(
            "/api/label-importers/import/server_json_file",
            json={"filepath": str(p)},
        )
        assert res.status_code == 200
        result = res.get_json()
        assert result["applied"] == 1
        assert result["missing_count"] == 1
        assert result["missing"][0]["md5"] == "totally_unknown_md5"
        assert "could not be resolved" in result["message"]
