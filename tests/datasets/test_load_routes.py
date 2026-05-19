"""Tests for the uncovered branches of ``vtsearch.routes.datasets.load``.

Covers endpoints that had zero direct tests:

* ``GET  /api/dataset/export`` — round-trips the current medias to a
  pickle download.
* ``POST /api/dataset/load-file`` — accepts a multipart pickle upload
  (validates the no-file / empty-filename branches).
* ``POST /api/dataset/load-source`` — the ``_load_from_origin``
  pseudo-origin and unknown-importer branches.

The tests do not exercise the background import thread itself; that
lives in ``tests/datasets/test_parallel_loading.py`` and friends.
"""

from __future__ import annotations

import io


# ---------------------------------------------------------------------------
# GET /api/dataset/export
# ---------------------------------------------------------------------------


class TestExportDataset:
    def test_returns_pickle_for_loaded_dataset(self, client, tmp_path):
        from vtsearch.datasets.loader import load_dataset_from_pickle
        from vtsearch.state import medias

        # Default fixture medias are loaded; export should succeed.
        resp = client.get("/api/dataset/export")
        assert resp.status_code == 200
        assert resp.content_type == "application/octet-stream"
        assert resp.headers["Content-Disposition"].endswith(".pkl")

        # The exported pickle round-trips back into a loadable dataset.
        pkl_path = tmp_path / "roundtrip.pkl"
        pkl_path.write_bytes(resp.data)
        roundtrip: dict = {}
        load_dataset_from_pickle(pkl_path, roundtrip)
        assert len(roundtrip) == len(medias)

    def test_returns_400_when_no_dataset(self, client):
        from vtsearch.state import medias

        saved = dict(medias)
        medias.clear()
        try:
            resp = client.get("/api/dataset/export")
            assert resp.status_code == 400
            assert "No dataset loaded" in resp.get_json()["message"]
        finally:
            medias.update(saved)

    def test_returns_500_when_export_raises(self, client, monkeypatch):
        """Exporter exceptions are caught and surfaced as 500 + detail."""

        def _boom(_snap):
            raise RuntimeError("disk on fire")

        import vtsearch.routes.datasets.load as load_mod

        monkeypatch.setattr(load_mod, "export_dataset_to_file", _boom)
        resp = client.get("/api/dataset/export")
        assert resp.status_code == 500
        body = resp.get_json()
        assert "disk on fire" in body["message"]


# ---------------------------------------------------------------------------
# POST /api/dataset/load-file
# ---------------------------------------------------------------------------


class TestLoadFile:
    def test_missing_file_returns_400(self, client):
        resp = client.post("/api/dataset/load-file")
        assert resp.status_code == 400
        assert "No file provided" in resp.get_json()["message"]

    def test_empty_filename_returns_400(self, client):
        """An uploaded file with no filename is rejected with 400."""
        data = {"file": (io.BytesIO(b"junk"), "")}
        resp = client.post("/api/dataset/load-file", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400
        assert "No file selected" in resp.get_json()["message"]


# ---------------------------------------------------------------------------
# POST /api/dataset/load-source
# ---------------------------------------------------------------------------


class TestLoadSource:
    def test_unknown_importer_returns_400(self, client):
        resp = client.post(
            "/api/dataset/load-source",
            json={"source": {"importer": "not_a_real_importer", "params": {}}},
        )
        assert resp.status_code == 400
        assert "Unknown importer" in resp.get_json()["message"]

    def test_dupe_set_without_members_returns_400(self, client):
        """The ``dupe_set`` pseudo-origin needs ``members`` — empty rejects."""
        resp = client.post(
            "/api/dataset/load-source",
            json={"source": {"importer": "dupe_set", "members": []}},
        )
        assert resp.status_code == 400
        assert "dupe_set" in resp.get_json()["message"]

    def test_dupe_set_unwraps_to_inner_origin(self, client):
        """``dupe_set`` recurses into the first member's origin — an inner
        unknown-importer surface as the inner's 400, proving the unwrap ran."""
        resp = client.post(
            "/api/dataset/load-source",
            json={
                "source": {
                    "importer": "dupe_set",
                    "members": [{"origin": {"importer": "still_not_real", "params": {}}}],
                }
            },
        )
        assert resp.status_code == 400
        assert "still_not_real" in resp.get_json()["message"]

    def test_missing_source_field_returns_422(self, client):
        """The flask-smorest schema gates the ``source`` field as required."""
        resp = client.post("/api/dataset/load-source", json={})
        assert resp.status_code == 422
