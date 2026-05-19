"""Tests for export boolean options and fill-from-sort.

Covers:
- Auto-detect negative_hits in CLI scoring
- POST /api/labels/fill-from-sort (dry run and confirm)
- Export filtering with Good/Bad/Both sides
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import app as app_module

SAMPLE_RESULTS = {
    "media_type": "audio",
    "detectors_run": 1,
    "results": {
        "test_det": {
            "detector_name": "test_det",
            "threshold": 0.5,
            "total_hits": 2,
            "hits": [
                {"id": 1, "filename": "a.wav", "score": 0.9, "label": "good"},
                {"id": 2, "filename": "b.wav", "score": 0.7, "label": "good"},
            ],
            "negative_hits": [
                {"id": 3, "filename": "c.wav", "score": 0.3, "label": "bad"},
            ],
        },
    },
}


# ---------------------------------------------------------------------------
# Fill from Sort – dry run
# ---------------------------------------------------------------------------


class TestFillFromSortDryRun:
    def _sort_results(self):
        """Build a simple sort_results list with all media ids."""
        from vtsearch.state import medias

        results = []
        for i, cid in enumerate(sorted(medias.keys())):
            # Alternate scores: first half above 0.5, second half below
            score = 0.9 - (i * 0.04)
            results.append({"id": cid, "score": round(score, 4)})
        return results

    def test_dry_run_returns_counts(self, client):
        results = self._sort_results()
        resp = client.post(
            "/api/labels/fill-from-sort",
            json={
                "sort_results": results,
                "threshold": 0.5,
                "sides": "good",
                "confirm": False,
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "good_count" in data
        assert "bad_count" in data
        assert data["good_count"] >= 0
        assert data["bad_count"] == 0  # sides="good" ignores bad

    def test_dry_run_bad_side(self, client):
        results = self._sort_results()
        resp = client.post(
            "/api/labels/fill-from-sort",
            json={
                "sort_results": results,
                "threshold": 0.5,
                "sides": "bad",
                "confirm": False,
            },
        )
        data = resp.get_json()
        assert data["good_count"] == 0  # sides="bad" ignores good
        assert data["bad_count"] >= 0

    def test_dry_run_both_sides(self, client):
        results = self._sort_results()
        resp = client.post(
            "/api/labels/fill-from-sort",
            json={
                "sort_results": results,
                "threshold": 0.5,
                "sides": "both",
                "confirm": False,
            },
        )
        data = resp.get_json()
        assert data["good_count"] + data["bad_count"] > 0

    def test_dry_run_excludes_already_voted(self, client):
        """Clips that already have votes should not be counted."""
        app_module.good_votes.update({1: None, 2: None})
        app_module.bad_votes.update({3: None})
        results = self._sort_results()
        resp = client.post(
            "/api/labels/fill-from-sort",
            json={
                "sort_results": results,
                "threshold": 0.5,
                "sides": "both",
                "confirm": False,
            },
        )
        data = resp.get_json()
        total = data["good_count"] + data["bad_count"]
        from vtsearch.state import medias

        assert total == len(medias) - 3

    def test_missing_sort_results_returns_400(self, client):
        resp = client.post(
            "/api/labels/fill-from-sort",
            json={"threshold": 0.5, "sides": "good"},
        )
        # Required-field validation runs in the FillFromSort schema → 422.
        # Keeping the test name for grep continuity.
        assert resp.status_code == 422

    def test_missing_threshold_returns_400(self, client):
        resp = client.post(
            "/api/labels/fill-from-sort",
            json={"sort_results": [], "sides": "good"},
        )
        # Required-field validation runs in the FillFromSort schema → 422.
        assert resp.status_code == 422

    def test_invalid_sides_returns_400(self, client):
        resp = client.post(
            "/api/labels/fill-from-sort",
            json={"sort_results": [], "threshold": 0.5, "sides": "invalid"},
        )
        # OneOf validation runs in the schema → 422.
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Fill from Sort – confirm
# ---------------------------------------------------------------------------


class TestFillFromSortConfirm:
    def _sort_results(self):
        from vtsearch.state import medias

        results = []
        for i, cid in enumerate(sorted(medias.keys())):
            score = 0.9 - (i * 0.04)
            results.append({"id": cid, "score": round(score, 4)})
        return results

    def test_confirm_applies_labels(self, client):
        results = self._sort_results()
        resp = client.post(
            "/api/labels/fill-from-sort",
            json={
                "sort_results": results,
                "threshold": 0.5,
                "sides": "both",
                "confirm": True,
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "good_applied" in data
        assert "bad_applied" in data
        assert "results" in data

        # Verify labels were actually applied
        total_applied = data["good_applied"] + data["bad_applied"]
        from vtsearch.state import medias

        assert total_applied == len(medias)  # all were unlabeled
        assert len(app_module.good_votes) == data["good_applied"]
        assert len(app_module.bad_votes) == data["bad_applied"]

    def test_confirm_returns_exporter_compatible_results(self, client):
        results = self._sort_results()
        resp = client.post(
            "/api/labels/fill-from-sort",
            json={
                "sort_results": results,
                "threshold": 0.5,
                "sides": "good",
                "confirm": True,
            },
        )
        data = resp.get_json()
        r = data["results"]
        assert "media_type" in r
        assert "detectors_run" in r
        assert "results" in r
        det = r["results"]["fill_from_sort"]
        assert "detector_name" in det
        assert "threshold" in det
        assert "hits" in det

    def test_confirm_does_not_relabel_already_voted(self, client):
        # Pre-label some medias
        app_module.good_votes.update({1: None})
        app_module.bad_votes.update({2: None})

        results = self._sort_results()
        resp = client.post(
            "/api/labels/fill-from-sort",
            json={
                "sort_results": results,
                "threshold": 0.5,
                "sides": "both",
                "confirm": True,
            },
        )
        data = resp.get_json()
        from vtsearch.state import medias

        total_applied = data["good_applied"] + data["bad_applied"]
        assert total_applied == len(medias) - 2  # 2 already voted

    def test_confirm_good_only(self, client):
        results = self._sort_results()
        resp = client.post(
            "/api/labels/fill-from-sort",
            json={
                "sort_results": results,
                "threshold": 0.5,
                "sides": "good",
                "confirm": True,
            },
        )
        data = resp.get_json()
        assert data["bad_applied"] == 0
        assert len(app_module.bad_votes) == 0

    def test_confirm_bad_only(self, client):
        results = self._sort_results()
        resp = client.post(
            "/api/labels/fill-from-sort",
            json={
                "sort_results": results,
                "threshold": 0.5,
                "sides": "bad",
                "confirm": True,
            },
        )
        data = resp.get_json()
        assert data["good_applied"] == 0
        assert len(app_module.good_votes) == 0


# ---------------------------------------------------------------------------
# CLI _score_medias_with_detectors negative_hits
# ---------------------------------------------------------------------------


class TestCliScoringNegativeHits:
    def test_trainable_model_scoring_returns_negative_hits(self, client):
        """The detector CLI scorer should include negative_hits."""
        import torch

        from vtscore.cli import _score_medias_with_detectors
        from vtscore.training import build_model_from_weights
        from vtscore.detectors.training import serialize_weights, train_and_threshold
        from vtsearch.state import (
            medias,
            snapshot_medias,
        )

        snap = snapshot_medias()
        good_ids = [1, 2, 3]
        bad_ids = [18, 19, 20]
        X = [snap[i]["embedding"] for i in good_ids + bad_ids]
        y = [1.0] * len(good_ids) + [0.0] * len(bad_ids)
        mlp, threshold = train_and_threshold(X, y, snap=snap)
        # Round-trip through serialize_weights to mirror the production path.
        weights = serialize_weights(mlp)
        rebuilt = build_model_from_weights(weights)
        assert rebuilt is not None
        # Sanity check: torch tensor coverage matches
        with torch.no_grad():
            out = rebuilt(torch.zeros((1, weights["0.weight"][0].__len__()), dtype=torch.float32))
        assert out is not None

        detector_mlps = {"test": {"mlp": mlp, "threshold": threshold}}
        det_results = _score_medias_with_detectors(medias, detector_mlps)
        for det_result in det_results.values():
            assert "negative_hits" in det_result
            assert isinstance(det_result["negative_hits"], list)
            total = len(det_result["hits"]) + len(det_result["negative_hits"])
            assert total == len(medias)


# ---------------------------------------------------------------------------
# Export with Good/Bad/Both filtered results via API
# ---------------------------------------------------------------------------


class TestExportWithFilteredResults:
    def test_export_good_only(self, client):
        """Exporting with only Good hits sends only positive hits to the exporter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "good_only.json"
            resp = client.post(
                "/api/exporters/export",
                json={
                    "exporter_name": "server_json_file",
                    "field_values": {"filepath": str(fpath)},
                    "results": {
                        "media_type": "audio",
                        "detectors_run": 1,
                        "results": {
                            "det": {
                                "detector_name": "det",
                                "threshold": 0.5,
                                "total_hits": 2,
                                "hits": [
                                    {"id": 1, "filename": "a.wav", "score": 0.9},
                                    {"id": 2, "filename": "b.wav", "score": 0.7},
                                ],
                            },
                        },
                    },
                },
            )
            assert resp.status_code == 200
            written = json.loads(fpath.read_text())
            assert len(written["results"]["det"]["hits"]) == 2

    def test_export_filtered_bad_only(self, client):
        """When frontend sends only negative hits as 'hits', they export correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "bad_only.json"
            resp = client.post(
                "/api/exporters/export",
                json={
                    "exporter_name": "server_json_file",
                    "field_values": {"filepath": str(fpath)},
                    "results": {
                        "media_type": "audio",
                        "detectors_run": 1,
                        "results": {
                            "det": {
                                "detector_name": "det",
                                "threshold": 0.5,
                                "total_hits": 3,
                                "hits": [
                                    {"id": 3, "filename": "c.wav", "score": 0.3, "label": "bad"},
                                    {"id": 4, "filename": "d.wav", "score": 0.2, "label": "bad"},
                                    {"id": 5, "filename": "e.wav", "score": 0.1, "label": "bad"},
                                ],
                            },
                        },
                    },
                },
            )
            assert resp.status_code == 200
            written = json.loads(fpath.read_text())
            assert len(written["results"]["det"]["hits"]) == 3
            for hit in written["results"]["det"]["hits"]:
                assert hit["label"] == "bad"

    def test_export_both_sides(self, client):
        """When frontend sends combined good+bad hits, they export correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "both.json"
            resp = client.post(
                "/api/exporters/export",
                json={
                    "exporter_name": "server_json_file",
                    "field_values": {"filepath": str(fpath)},
                    "results": {
                        "media_type": "audio",
                        "detectors_run": 1,
                        "results": {
                            "det": {
                                "detector_name": "det",
                                "threshold": 0.5,
                                "total_hits": 3,
                                "hits": [
                                    {"id": 1, "filename": "a.wav", "score": 0.9, "label": "good"},
                                    {"id": 2, "filename": "b.wav", "score": 0.7, "label": "good"},
                                    {"id": 3, "filename": "c.wav", "score": 0.3, "label": "bad"},
                                ],
                            },
                        },
                    },
                },
            )
            assert resp.status_code == 200
            written = json.loads(fpath.read_text())
            assert len(written["results"]["det"]["hits"]) == 3

    def test_fill_from_sort_results_exportable(self, client):
        """Results from fill-from-sort should be exportable via the file exporter."""
        from vtsearch.state import medias

        sort_results = [{"id": cid, "score": round(0.9 - (i * 0.04), 4)} for i, cid in enumerate(sorted(medias.keys()))]

        fill_resp = client.post(
            "/api/labels/fill-from-sort",
            json={
                "sort_results": sort_results,
                "threshold": 0.5,
                "sides": "good",
                "confirm": True,
            },
        )
        assert fill_resp.status_code == 200
        fill_data = fill_resp.get_json()

        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "fill_export.json"
            export_resp = client.post(
                "/api/exporters/export",
                json={
                    "exporter_name": "server_json_file",
                    "field_values": {"filepath": str(fpath)},
                    "results": fill_data["results"],
                },
            )
            assert export_resp.status_code == 200
            assert fpath.exists()
            written = json.loads(fpath.read_text())
            assert "fill_from_sort" in written["results"]


class TestExportWithSelectedColumns:
    """Exporting labels via the API respects selected_columns."""

    def test_csv_export_with_selected_columns(self, client):
        """POST /api/exporters/export with labels + selected_columns uses those columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "cols.csv"
            resp = client.post(
                "/api/exporters/export",
                json={
                    "exporter_name": "server_csv_file",
                    "field_values": {"filepath": str(fpath)},
                    "results": {
                        "labels": [
                            {"label": "good", "md5": "aaa", "filename": "a.wav", "category": "x"},
                            {"label": "bad", "md5": "bbb", "filename": "b.wav", "category": "y"},
                        ],
                        "selected_columns": ["label", "filename"],
                    },
                },
            )
            assert resp.status_code == 200
            import csv

            with open(fpath, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            # origin is always appended as last column for re-import
            assert rows[0] == ["label", "filename", "origin"]
            assert rows[1] == ["good", "a.wav", ""]
            assert rows[2] == ["bad", "b.wav", ""]

    def test_json_export_with_selected_columns(self, client):
        """POST /api/exporters/export with labels + selected_columns filters JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "cols.json"
            resp = client.post(
                "/api/exporters/export",
                json={
                    "exporter_name": "server_json_file",
                    "field_values": {"filepath": str(fpath)},
                    "results": {
                        "labels": [
                            {"label": "good", "md5": "aaa", "filename": "a.wav", "category": "x"},
                        ],
                        "selected_columns": ["filename", "category"],
                    },
                },
            )
            assert resp.status_code == 200
            written = json.loads(Path(fpath).read_text())
            assert written["labels"] == [{"filename": "a.wav", "category": "x"}]

    def test_changed_columns_between_exports(self, client):
        """Exporting twice with different selected_columns produces different output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            labels = [{"label": "good", "md5": "aaa", "filename": "a.wav", "category": "x"}]

            fpath1 = Path(tmpdir) / "v1.csv"
            client.post(
                "/api/exporters/export",
                json={
                    "exporter_name": "server_csv_file",
                    "field_values": {"filepath": str(fpath1)},
                    "results": {"labels": labels, "selected_columns": ["label", "md5", "filename"]},
                },
            )

            fpath2 = Path(tmpdir) / "v2.csv"
            client.post(
                "/api/exporters/export",
                json={
                    "exporter_name": "server_csv_file",
                    "field_values": {"filepath": str(fpath2)},
                    "results": {"labels": labels, "selected_columns": ["category"]},
                },
            )

            import csv

            with open(fpath1, newline="", encoding="utf-8") as f:
                header1 = next(csv.reader(f))
            with open(fpath2, newline="", encoding="utf-8") as f:
                header2 = next(csv.reader(f))

            # origin is always appended as last column
            assert header1 == ["label", "md5", "filename", "origin"]
            assert header2 == ["category", "origin"]


class TestAvailableColumnsNoDuplicates:
    """Enriched export should not return duplicate columns differing only by case."""

    def test_no_duplicate_category_column(self, client):
        """Category appears once in available_columns, not twice (lowercase + capitalized)."""
        app_module.good_votes[1] = None
        resp = client.get("/api/labels/export?enrich=true")
        assert resp.status_code == 200
        data = resp.get_json()
        cols = data["available_columns"]
        category_cols = [c for c in cols if c.lower() == "category"]
        assert len(category_cols) == 1, f"Expected 1 category column, got: {category_cols}"
