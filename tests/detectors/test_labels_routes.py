"""Tests for the lower-covered branches of ``vtsearch.routes.detectors.labels``.

The existing ``test_labelset_elements_api.py`` covers ``labels-detail``,
``vote``, and ``thumbnail`` error paths.  This file fills in the
remaining gaps:

* ``POST /api/detectors/<name>/import-labels/<importer>`` - the entire
  90-line method was at 0% coverage.
* ``GET /api/detectors/<name>/labels/<element_id>/preview`` - file path
  resolution, mimetype selection, text-mode JSON branch.

Tests use a stub :class:`~vtscore.labels.importers.LabelImporter` to
avoid touching the real filesystem importer chain.
"""

from __future__ import annotations

import shutil

import pytest

from vtscore.detectors.store import _detector_path, _read_detector, _write_detector
from vtsearch.settings import get_detectors_dir


@pytest.fixture(autouse=True)
def clean_detectors_dir():
    tm_dir = get_detectors_dir()
    if tm_dir.is_dir():
        shutil.rmtree(tm_dir)
    yield
    tm_dir = get_detectors_dir()
    if tm_dir.is_dir():
        shutil.rmtree(tm_dir)


def _write_seed_detector(name: str = "labels-target", media_type: str = "audio") -> str:
    """Write a detector with an empty labelset and register it."""
    from vtscore.detectors.registry import register_detector, reset_for_tests

    reset_for_tests()
    _write_detector(
        _detector_path(name),
        {
            "name": name,
            "text_query": "",
            "media_type": media_type,
            "examples": [],
            "labelset": {"labels": []},
        },
    )
    entry = register_detector(name=name, media_type=media_type)
    return entry["id"]


# ---------------------------------------------------------------------------
# POST /api/detectors/<name>/import-labels/<importer>
# ---------------------------------------------------------------------------


class TestImportLabelsRoute:
    """Cover the previously-untested ``import_labels_into_detector`` route."""

    def test_unknown_detector_returns_404(self, client):
        res = client.post(
            "/api/detectors/missing/import-labels/server_json_file",
            json={"filepath": "/tmp/nope.json"},
        )
        assert res.status_code == 404

    def test_unknown_importer_returns_404(self, client):
        _write_seed_detector()
        res = client.post(
            "/api/detectors/labels-target/import-labels/not_a_real_importer",
            json={"filepath": "/tmp/foo.json"},
        )
        assert res.status_code == 404
        body = res.get_json()
        assert "not_a_real_importer" in body["error"]

    def test_missing_filepath_returns_422(self, client):
        _write_seed_detector()
        # server_json_file declares ``filepath`` as required.
        res = client.post(
            "/api/detectors/labels-target/import-labels/server_json_file",
            json={},
        )
        assert res.status_code == 422

    def test_invalid_filepath_returns_400(self, client):
        _write_seed_detector()
        # Path-traversal attempt - caught by the framework's field-driven
        # server_path validator (see vtscore/plugins/normalize.py).
        res = client.post(
            "/api/detectors/labels-target/import-labels/server_json_file",
            json={"filepath": "/etc/passwd"},
        )
        assert res.status_code == 400

    def test_importer_run_error_returns_400(self, client, tmp_path, monkeypatch):
        """A ValueError from the importer surfaces as 400 with the message."""
        _write_seed_detector()
        # A file that doesn't exist → server_json_file raises ValueError.
        from vtscore.config import DATA_DIR

        # Place the path under DATA_DIR so the framework's server_path validator accepts it.
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        target = DATA_DIR / "does-not-exist.json"
        if target.exists():
            target.unlink()

        res = client.post(
            "/api/detectors/labels-target/import-labels/server_json_file",
            json={"filepath": str(target)},
        )
        assert res.status_code == 400
        assert "not found" in res.get_json()["error"].lower()

    def test_happy_path_merges_labels(self, client, tmp_path, monkeypatch):
        """Successful import writes the entries into the on-disk labelset."""
        from vtscore.config import DATA_DIR

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        labels_file = DATA_DIR / "import_happy.json"
        labels_file.write_text(
            '{"labels": ['
            '{"md5": "a1b2", "label": "good", "origin_name": "x.wav"},'
            '{"md5": "c3d4", "label": "bad", "origin_name": "y.wav"}'
            "]}"
        )
        try:
            _write_seed_detector()
            res = client.post(
                "/api/detectors/labels-target/import-labels/server_json_file",
                json={"filepath": str(labels_file)},
            )
            assert res.status_code == 200, res.get_data(as_text=True)
            body = res.get_json()
            assert body["applied"] == 2
            assert body["skipped"] == 0
            assert body["num_labels"] == 2
            assert "Added 2 label" in body["message"]

            # On-disk labelset reflects the new entries.
            data = _read_detector(_detector_path("labels-target"))
            assert data is not None
            labels = data["labelset"]["labels"]
            assert len(labels) == 2
            md5s = {lbl["md5"] for lbl in labels}
            assert md5s == {"a1b2", "c3d4"}
        finally:
            if labels_file.exists():
                labels_file.unlink()

    def test_skips_invalid_label_values(self, client):
        """Entries whose ``label`` isn't 'good'/'bad' are counted as skipped."""
        from vtscore.config import DATA_DIR

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        labels_file = DATA_DIR / "import_invalid.json"
        labels_file.write_text(
            '{"labels": ['
            '{"md5": "aa", "label": "good"},'
            '{"md5": "bb", "label": "maybe"},'  # invalid value
            '{"md5": "cc"}'  # no label at all
            "]}"
        )
        try:
            _write_seed_detector()
            res = client.post(
                "/api/detectors/labels-target/import-labels/server_json_file",
                json={"filepath": str(labels_file)},
            )
            assert res.status_code == 200
            body = res.get_json()
            assert body["applied"] == 1
            assert body["skipped"] == 2
        finally:
            if labels_file.exists():
                labels_file.unlink()

    def test_dedups_existing_md5_label_pair(self, client):
        """An entry whose (md5, label) pair is already present is skipped."""
        from vtscore.config import DATA_DIR

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        labels_file = DATA_DIR / "import_dupe.json"
        labels_file.write_text('{"labels": [{"md5": "shared", "label": "good"},{"md5": "fresh", "label": "good"}]}')
        try:
            # Pre-seed the detector with a matching (md5, label) pair.
            _write_seed_detector()
            data = _read_detector(_detector_path("labels-target"))
            assert data is not None
            data["labelset"] = {"labels": [{"md5": "shared", "label": "good", "origin_name": "old.wav"}]}
            _write_detector(_detector_path("labels-target"), data)

            res = client.post(
                "/api/detectors/labels-target/import-labels/server_json_file",
                json={"filepath": str(labels_file)},
            )
            assert res.status_code == 200
            body = res.get_json()
            assert body["applied"] == 1, "only the fresh entry should land"
            assert body["skipped"] == 1
            assert body["num_labels"] == 2
        finally:
            if labels_file.exists():
                labels_file.unlink()


# ---------------------------------------------------------------------------
# GET /api/detectors/<name>/labels/<element_id>/preview
# ---------------------------------------------------------------------------


class TestPreviewLabelRoute:
    """Cover the file-streaming preview route, which had 0% coverage."""

    def test_unknown_detector_returns_404(self, client):
        res = client.get("/api/detectors/missing/labels/abc/preview")
        assert res.status_code == 404

    def test_unknown_element_returns_404(self, client):
        _write_seed_detector()
        res = client.get("/api/detectors/labels-target/labels/missing/preview")
        assert res.status_code == 404

    def test_unresolvable_file_returns_404(self, client):
        """An element whose origin can't resolve to a file should 404 cleanly."""
        _write_seed_detector()
        # Inject a labelset with one element whose origin doesn't exist.
        data = _read_detector(_detector_path("labels-target"))
        assert data is not None
        data["labelset"] = {
            "labels": [
                {
                    "md5": "deadbeef",
                    "label": "good",
                    "origin": {"importer": "ghost", "params": {}},
                    "origin_name": "ghost.wav",
                    "filename": "ghost.wav",
                }
            ]
        }
        _write_detector(_detector_path("labels-target"), data)

        detail = client.get("/api/detectors/labels-target/labels-detail").get_json()
        elem_id = detail["good"][0]["id"]
        res = client.get(f"/api/detectors/labels-target/labels/{elem_id}/preview")
        assert res.status_code == 404

    def test_text_media_returns_content_json(self, client, tmp_path, monkeypatch):
        """For text media the route returns a JSON body, not raw bytes."""
        from contextlib import contextmanager

        text_path = tmp_path / "doc.txt"
        text_path.write_text("hello world how are you")

        @contextmanager
        def _fake_resolve(origin, origin_name="", filename=""):
            yield text_path

        import vtscore.detectors.labelset_elements as le_mod

        monkeypatch.setattr(le_mod, "resolve_element_to_path", _fake_resolve)

        _write_seed_detector(media_type="text")
        data = _read_detector(_detector_path("labels-target"))
        assert data is not None
        data["labelset"] = {
            "labels": [
                {
                    "md5": "deadbeef",
                    "label": "good",
                    "origin": {"importer": "ghost", "params": {}},
                    "origin_name": "doc.txt",
                    "filename": "doc.txt",
                }
            ]
        }
        _write_detector(_detector_path("labels-target"), data)

        detail = client.get("/api/detectors/labels-target/labels-detail").get_json()
        elem_id = detail["good"][0]["id"]
        res = client.get(f"/api/detectors/labels-target/labels/{elem_id}/preview")
        assert res.status_code == 200
        body = res.get_json()
        assert body["content"] == "hello world how are you"
        assert body["word_count"] == 5
        assert body["character_count"] == 23

    def test_binary_media_returns_file_bytes(self, client, tmp_path, monkeypatch):
        """For non-text media the route streams the file bytes."""
        from contextlib import contextmanager

        img_path = tmp_path / "tiny.png"
        # 1×1 PNG (valid header).
        img_path.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff"
            b"\xff?\x00\x05\xfe\x02\xfe\xa3wW\x1d\x00\x00\x00\x00IEND\xaeB`\x82"
        )

        @contextmanager
        def _fake_resolve(origin, origin_name="", filename=""):
            yield img_path

        import vtscore.detectors.labelset_elements as le_mod

        monkeypatch.setattr(le_mod, "resolve_element_to_path", _fake_resolve)

        _write_seed_detector(media_type="image")
        data = _read_detector(_detector_path("labels-target"))
        assert data is not None
        data["labelset"] = {
            "labels": [
                {
                    "md5": "deadbeef",
                    "label": "good",
                    "origin": {"importer": "ghost", "params": {}},
                    "origin_name": "tiny.png",
                    "filename": "tiny.png",
                }
            ]
        }
        _write_detector(_detector_path("labels-target"), data)

        detail = client.get("/api/detectors/labels-target/labels-detail").get_json()
        elem_id = detail["good"][0]["id"]
        res = client.get(f"/api/detectors/labels-target/labels/{elem_id}/preview")
        assert res.status_code == 200
        assert res.content_type == "image/png"
        # PNG magic header survives the round-trip.
        assert res.data[:8] == b"\x89PNG\r\n\x1a\n"
