"""Tests for the browser-side ``Import from Local Folder`` flow.

These tests cover the new ``/api/dataset/import-local-folder`` endpoint that
streams files uploaded from the user's *browser* machine to a server-side
temporary directory and then runs the regular folder importer over that
temp dir.  They also pin down the path-traversal sanitiser used by the
endpoint.
"""

import io
from pathlib import PurePosixPath
from unittest.mock import patch

from vtsearch.routes.datasets._helpers import _safe_relative_upload_path


class TestSafeRelativeUploadPath:
    """The browser sends ``webkitRelativePath`` as each part's filename.

    We must accept those (relative POSIX paths) but reject anything that
    would let a malicious caller escape the upload tempdir.
    """

    def test_simple_filename(self):
        assert _safe_relative_upload_path("foo.txt") == PurePosixPath("foo.txt")

    def test_nested_subdirectory(self):
        assert _safe_relative_upload_path("a/b/c.wav") == PurePosixPath("a/b/c.wav")

    def test_backslashes_normalised_to_slashes(self):
        assert _safe_relative_upload_path("a\\b\\c.wav") == PurePosixPath("a/b/c.wav")

    def test_empty_string_rejected(self):
        assert _safe_relative_upload_path("") is None

    def test_absolute_path_rejected(self):
        assert _safe_relative_upload_path("/etc/passwd") is None

    def test_dotdot_segment_rejected(self):
        assert _safe_relative_upload_path("a/../etc/passwd") is None

    def test_leading_dotdot_rejected(self):
        assert _safe_relative_upload_path("../foo.txt") is None

    def test_dot_segments_skipped(self):
        # "." segments are stripped, not rejected.
        assert _safe_relative_upload_path("./foo/./bar.txt") == PurePosixPath("foo/bar.txt")

    def test_only_dot_segments_rejected(self):
        # Strips down to nothing → invalid.
        assert _safe_relative_upload_path("./.") is None

    def test_null_byte_rejected(self):
        assert _safe_relative_upload_path("foo\x00bar.txt") is None


class TestImportLocalFolderEndpoint:
    """End-to-end shape of the new endpoint.

    The actual importer execution runs in a background thread, which the
    tests intentionally don't trigger; we patch ``_run_origin_load_in_background``
    to verify that the endpoint sets up the temp dir and field values
    correctly without blocking on real embedding.
    """

    def test_no_files_returns_400(self, client):
        resp = client.post(
            "/api/dataset/import-local-folder",
            data={"media_type": "audio"},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        # flask-smorest error envelope: ``message`` (not ``error``).
        assert "No files uploaded" in resp.get_json()["message"]

    def test_missing_media_type_returns_400(self, client):
        resp = client.post(
            "/api/dataset/import-local-folder",
            data={"files": (io.BytesIO(b"hello"), "myfolder/a.txt")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert "media_type" in resp.get_json()["message"]

    def test_invalid_clipper_params_returns_400(self, client):
        resp = client.post(
            "/api/dataset/import-local-folder",
            data={
                "media_type": "audio",
                "clipper_params": "not-json",
                "files": (io.BytesIO(b"hello"), "myfolder/a.wav"),
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert "clipper_params" in resp.get_json()["message"]

    def test_uploads_files_to_tempdir_and_starts_load(self, client, tmp_path, monkeypatch):
        # Redirect the upload temp root into the test's tmp_path so we can
        # inspect the resulting layout without polluting the repo's data dir.
        monkeypatch.setattr(
            "vtsearch.routes.datasets.load.LOCAL_UPLOADS_DIR",
            tmp_path / "uploads",
        )

        captured: dict = {}

        def _fake_run(load_fn, origin, **kwargs):
            # Trigger the load_fn synchronously against a throwaway dict so
            # the cleanup path runs deterministically and we can observe
            # what the importer would have seen.
            captured["origin"] = origin
            captured["kwargs"] = kwargs

            # Don't actually invoke the real folder importer (that would try
            # to embed audio).  Just record that load_fn is callable.
            captured["load_fn"] = load_fn
            return "task-fake-1"

        with patch(
            "vtsearch.routes.datasets.load._run_origin_load_in_background",
            side_effect=_fake_run,
        ):
            resp = client.post(
                "/api/dataset/import-local-folder",
                data={
                    "media_type": "audio",
                    "files": [
                        (io.BytesIO(b"AAA"), "myfolder/one.wav"),
                        (io.BytesIO(b"BBB"), "myfolder/sub/two.wav"),
                    ],
                },
                content_type="multipart/form-data",
            )

        assert resp.status_code == 200, resp.get_data(as_text=True)
        body = resp.get_json()
        assert body["ok"] is True
        assert body["task_id"] == "task-fake-1"

        # Origin params should NOT leak the tempdir path; instead they mark
        # the dataset as a browser upload so reload-from-origin is naturally
        # disabled.
        assert captured["origin"]["importer"] == "server_folder"
        assert captured["origin"]["params"]["path"] == "<browser_upload>"
        assert captured["origin"]["params"]["media_type"] == "audio"

    def test_path_traversal_filenames_dropped(self, client, tmp_path, monkeypatch):
        """Files whose multipart filename tries to escape the tempdir are skipped."""
        monkeypatch.setattr(
            "vtsearch.routes.datasets.load.LOCAL_UPLOADS_DIR",
            tmp_path / "uploads",
        )

        # Track which paths actually got written into the temp dir.
        written: list[str] = []

        def _fake_run(load_fn, origin, **kwargs):
            # Walk the temp dir and snapshot relative paths.
            from pathlib import Path as _P

            field_path = None
            # The temp dir lives directly under LOCAL_UPLOADS_DIR; pick
            # the first (and only) child.
            uploads_root = tmp_path / "uploads"
            for child in uploads_root.iterdir():
                if child.is_dir():
                    field_path = child
                    break
            assert field_path is not None
            for f in field_path.rglob("*"):
                if f.is_file():
                    written.append(str(_P(f).relative_to(field_path).as_posix()))
            return "task-fake-2"

        with patch(
            "vtsearch.routes.datasets.load._run_origin_load_in_background",
            side_effect=_fake_run,
        ):
            resp = client.post(
                "/api/dataset/import-local-folder",
                data={
                    "media_type": "audio",
                    "files": [
                        (io.BytesIO(b"OK"), "good/file.wav"),
                        (io.BytesIO(b"BAD"), "../escape.wav"),
                        (io.BytesIO(b"BAD"), "/etc/passwd"),
                    ],
                },
                content_type="multipart/form-data",
            )

        assert resp.status_code == 200, resp.get_data(as_text=True)
        # Only the safe file should land on disk.
        assert written == ["good/file.wav"]


class TestFolderImporterPickerVisibility:
    """The renamed ``server_folder`` importer powers the dedicated
    "Server Folder" picker card via ``picker_view``.

    The browser-side upload flow has its own ``local_folder`` importer
    that delegates to ``/api/dataset/import-local-folder`` (and from there
    re-enters the ``server_folder`` importer on a server-side temp directory)."""

    def test_server_folder_importer_uses_server_folder_picker_view(self, client):
        resp = client.get("/api/dataset/all-importers")
        importers = {imp["name"]: imp for imp in resp.get_json()["importers"]}
        assert importers["server_folder"]["picker_view"] == "server_folder"
        # The Server Folder card is part of the picker (not hidden).
        assert importers["server_folder"]["hidden_from_picker"] is False

    def test_local_folder_importer_hidden_from_picker(self, client):
        """The Local Folder card is still registered as its own importer but
        is hidden from the picker (the browser-upload Local tab is retired)."""
        resp = client.get("/api/dataset/all-importers")
        importers = {imp["name"]: imp for imp in resp.get_json()["importers"]}
        assert "local_folder" in importers
        local = importers["local_folder"]
        assert local["picker_view"] == "local_folder"
        assert local["hidden_from_picker"] is True
        assert "browser" in local["description"].lower()

    def test_local_files_importer_hidden_from_picker(self, client):
        """The Local Files card mirrors Local Folder and is likewise hidden."""
        resp = client.get("/api/dataset/all-importers")
        importers = {imp["name"]: imp for imp in resp.get_json()["importers"]}
        assert "local_files" in importers
        local = importers["local_files"]
        assert local["picker_view"] == "local_files"
        assert local["hidden_from_picker"] is True
        assert "browser" in local["description"].lower()

    def test_server_files_importer_card_exists(self, client):
        """The Server Files importer takes a paths file on the server."""
        resp = client.get("/api/dataset/all-importers")
        importers = {imp["name"]: imp for imp in resp.get_json()["importers"]}
        assert "server_files" in importers
        server = importers["server_files"]
        assert server["picker_view"] == "form"
        assert server["hidden_from_picker"] is False
        keys = {f["key"] for f in server["fields"]}
        assert "paths_file" in keys
        assert "media_type" in keys

    def test_server_folder_importer_description_mentions_server(self, client):
        resp = client.get("/api/dataset/importers")
        folder_imp = next(i for i in resp.get_json()["importers"] if i["name"] == "server_folder")
        # The user reading the picker should not be misled into thinking
        # this scans their browser machine.
        assert "server" in folder_imp["description"].lower()
