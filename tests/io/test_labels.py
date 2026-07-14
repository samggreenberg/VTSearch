import json

import app as app_module


def _read_ndjson(resp):
    """Parse a streamed NDJSON export response into a list of label dicts."""
    text = resp.get_data(as_text=True)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class TestExportLabels:
    def test_empty_export(self, client):
        resp = client.get("/api/labels/export")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data == {"labels": []}

    def test_export_good_labels(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        resp = client.get("/api/labels/export")
        data = resp.get_json()
        assert len(data["labels"]) == 2
        assert all(e["label"] == "good" for e in data["labels"])

    def test_export_bad_labels(self, client):
        app_module.bad_votes.update({k: None for k in [3, 4]})
        resp = client.get("/api/labels/export")
        data = resp.get_json()
        assert len(data["labels"]) == 2
        assert all(e["label"] == "bad" for e in data["labels"])

    def test_export_mixed_labels(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        resp = client.get("/api/labels/export")
        data = resp.get_json()
        assert len(data["labels"]) == 4

    def test_export_contains_md5_and_label(self, client):
        app_module.good_votes[1] = None
        resp = client.get("/api/labels/export")
        data = resp.get_json()
        entry = data["labels"][0]
        assert "md5" in entry
        assert "label" in entry
        assert entry["md5"] == app_module.medias[1]["md5"]
        assert entry["label"] == "good"

    def test_export_does_not_include_creation_info(self, client):
        app_module.good_votes[1] = None
        resp = client.get("/api/labels/export")
        data = resp.get_json()
        assert "dataset_creation_info" not in data


class TestExportLabelsNdjson:
    """The streaming ``?format=ndjson`` variant of ``GET /api/labels/export`` (S13)."""

    def test_empty_export_streams_nothing(self, client):
        resp = client.get("/api/labels/export?format=ndjson")
        assert resp.status_code == 200
        assert resp.mimetype == "application/x-ndjson"
        assert _read_ndjson(resp) == []

    def test_streams_one_line_per_label(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        resp = client.get("/api/labels/export?format=ndjson")
        assert resp.mimetype == "application/x-ndjson"
        rows = _read_ndjson(resp)
        assert len(rows) == 4
        assert all("md5" in r and "label" in r for r in rows)

    def test_ndjson_matches_buffered_labels(self, client):
        """The streamed rows equal the buffered ``labels`` list, entry for entry."""
        app_module.good_votes.update({k: None for k in [1, 3, 5]})
        app_module.bad_votes.update({k: None for k in [2, 4]})

        buffered = client.get("/api/labels/export").get_json()["labels"]
        streamed = _read_ndjson(client.get("/api/labels/export?format=ndjson"))
        assert streamed == buffered

    def test_goods_only_filter(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        resp = client.get("/api/labels/export?format=ndjson&goods_only=true")
        rows = _read_ndjson(resp)
        assert len(rows) == 2
        assert all(r["label"] == "good" for r in rows)

    def test_corrections_filter_streams_only_changed(self, client):
        from vtsearch.state import set_find_initial_labels

        set_find_initial_labels({1: "good", 2: "bad", 3: "good"})
        app_module.good_votes.update({1: None, 2: None})  # 2 was bad -> correction
        app_module.bad_votes[3] = None  # 3 was good -> correction

        resp = client.get("/api/labels/export?format=ndjson&label_filter=corrections")
        rows = _read_ndjson(resp)
        assert len(rows) == 2
        assert all(r["is_correction"] is True for r in rows)
        md5s = {r["md5"] for r in rows}
        assert app_module.medias[2]["md5"] in md5s
        assert app_module.medias[3]["md5"] in md5s

    def test_corrections_filter_empty_without_find_labels(self, client):
        app_module.good_votes[1] = None
        app_module.bad_votes[2] = None
        resp = client.get("/api/labels/export?format=ndjson&label_filter=corrections")
        assert _read_ndjson(resp) == []

    def test_enrich_attaches_custom_metadata_but_no_available_columns(self, client):
        app_module.good_votes[1] = None
        resp = client.get("/api/labels/export?format=ndjson&enrich=true")
        rows = _read_ndjson(resp)
        assert len(rows) == 1
        # ``available_columns`` is a whole-set aggregate and is never a row.
        assert all("labels" not in r and "available_columns" not in r for r in rows)
        assert "custom_metadata" in rows[0]

    def test_invalid_format_rejected(self, client):
        resp = client.get("/api/labels/export?format=csv")
        assert resp.status_code == 422


class TestImportLabels:
    def test_import_good_label(self, client):
        labels = [{"md5": app_module.medias[1]["md5"], "label": "good"}]
        resp = client.post("/api/labels/import", json={"labels": labels})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["applied"] == 1
        assert data["skipped"] == 0
        assert 1 in app_module.good_votes

    def test_import_bad_label(self, client):
        labels = [{"md5": app_module.medias[1]["md5"], "label": "bad"}]
        resp = client.post("/api/labels/import", json={"labels": labels})
        assert resp.status_code == 200
        assert 1 in app_module.bad_votes

    def test_import_skips_unknown_md5(self, client):
        labels = [{"md5": "nonexistent_md5", "label": "good"}]
        resp = client.post("/api/labels/import", json={"labels": labels})
        data = resp.get_json()
        assert data["applied"] == 0
        assert data["skipped"] == 1

    def test_import_overrides_existing_label(self, client):
        app_module.good_votes[1] = None
        labels = [{"md5": app_module.medias[1]["md5"], "label": "bad"}]
        client.post("/api/labels/import", json={"labels": labels})
        assert 1 not in app_module.good_votes
        assert 1 in app_module.bad_votes

    def test_import_mixed_known_and_unknown(self, client):
        labels = [
            {"md5": app_module.medias[1]["md5"], "label": "good"},
            {"md5": "unknown_md5", "label": "good"},
        ]
        resp = client.post("/api/labels/import", json={"labels": labels})
        data = resp.get_json()
        assert data["applied"] == 1
        assert data["skipped"] == 1

    def test_import_invalid_label_value(self, client):
        labels = [{"md5": app_module.medias[1]["md5"], "label": "meh"}]
        resp = client.post("/api/labels/import", json={"labels": labels})
        data = resp.get_json()
        assert data["applied"] == 0
        assert data["skipped"] == 1

    def test_import_not_a_list(self, client):
        resp = client.post(
            "/api/labels/import",
            json={"labels": "not a list"},
        )
        # Marshmallow validates ``labels`` as a list → 422 with the
        # standard flask-smorest envelope.
        assert resp.status_code == 422

    def test_import_multiple_labels(self, client):
        labels = []
        for cid in [1, 2, 3]:
            labels.append({"md5": app_module.medias[cid]["md5"], "label": "good"})
        for cid in [4, 5]:
            labels.append({"md5": app_module.medias[cid]["md5"], "label": "bad"})
        resp = client.post("/api/labels/import", json={"labels": labels})
        data = resp.get_json()
        assert data["applied"] == 5
        assert data["skipped"] == 0
        assert set(app_module.good_votes) == {1, 2, 3}
        assert set(app_module.bad_votes) == {4, 5}

    def test_roundtrip_export_import(self, client):
        """Export labels, clear votes, import, and verify same state."""
        app_module.good_votes.update({k: None for k in [1, 3, 5]})
        app_module.bad_votes.update({k: None for k in [2, 4]})
        resp = client.get("/api/labels/export")
        exported = resp.get_json()

        app_module.good_votes.clear()
        app_module.bad_votes.clear()

        resp = client.post("/api/labels/import", json=exported)
        data = resp.get_json()
        assert data["applied"] == 5
        assert set(app_module.good_votes) == {1, 3, 5}
        assert set(app_module.bad_votes) == {2, 4}

    def test_import_matches_by_origin(self, client):
        """Labels with origin+origin_name match the correct media."""
        media = app_module.medias[1]
        labels = [
            {
                "md5": "wrong_md5_on_purpose",
                "label": "good",
                "origin": media["origin"],
                "origin_name": media["origin_name"],
            }
        ]
        resp = client.post("/api/labels/import", json={"labels": labels})
        data = resp.get_json()
        assert data["applied"] == 1
        assert 1 in app_module.good_votes

    def test_import_duplicate_md5_labels_both_clips(self, client):
        """Two medias sharing the same MD5 should both receive the label."""
        # Temporarily give media 2 the same MD5 as media 1
        original_md5 = app_module.medias[2]["md5"]
        app_module.medias[2]["md5"] = app_module.medias[1]["md5"]
        try:
            shared_md5 = app_module.medias[1]["md5"]
            labels = [{"md5": shared_md5, "label": "good"}]
            resp = client.post("/api/labels/import", json={"labels": labels})
            data = resp.get_json()
            assert data["applied"] == 1
            # Both medias with the same MD5 should receive the label
            assert 1 in app_module.good_votes
            assert 2 in app_module.good_votes
        finally:
            app_module.medias[2]["md5"] = original_md5
