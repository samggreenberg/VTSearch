"""The ``outputs`` request shape on the three import routes (#4703).

``POST /api/dataset/import/<importer>``, ``import-local-folder`` and
``import-local-files`` take an ``outputs`` list and answer with one task id
per dataset in ``task_ids``; without it they behave as before and list their
single task.  The background runner is patched out: what these pin down is
the request parsing and the hand-off, not the load itself.

``GET /api/datasets/registry`` then says which datasets came from the same
import (#4747): ``import_group`` and ``output_category`` on every entry.
"""

from __future__ import annotations

import io
import json
from unittest.mock import patch

from vtscore.datasets.importers.base import OutputSpec

_OUTPUTS = [
    {"media_type": "audio", "embedder": "clap"},
    {
        "media_type": "image",
        "category": "document",
        "source_specs": [{"source_type": "document", "converter": "document2image", "params": {}}],
    },
]


class TestGenericImportRoute:
    def test_outputs_start_a_multi_dataset_import(self, client, tmp_path):
        captured: dict = {}

        def fake_multi(importer, field_values, outputs, **kwargs):
            captured.update(importer=importer, field_values=field_values, outputs=outputs, kwargs=kwargs)
            return ["t-1", "t-2"]

        with patch("vtsearch.routes.datasets.staging._run_multi_output_load_in_background", side_effect=fake_multi):
            resp = client.post(
                "/api/dataset/import/server_folder",
                json={"path": str(tmp_path), "outputs": _OUTPUTS, "dataset_name": "holiday", "autofind": "false"},
            )

        assert resp.status_code == 200, resp.get_data(as_text=True)
        body = resp.get_json()
        assert body["task_id"] == "t-1"
        assert body["task_ids"] == ["t-1", "t-2"]

        assert captured["importer"].name == "server_folder"
        assert [o.media_type for o in captured["outputs"]] == ["audio", "image"]
        assert captured["outputs"][1].category == "document"
        assert isinstance(captured["outputs"][0], OutputSpec)
        assert "outputs" not in captured["field_values"], "the list is consumed, not forwarded as a form value"
        assert captured["field_values"]["dataset_name"] == "holiday"
        assert captured["kwargs"]["on_finished"] is not None

    def test_single_dataset_request_lists_its_one_task(self, client, tmp_path):
        with patch("vtsearch.routes.datasets.staging._run_importer_in_background", return_value="t-9"):
            resp = client.post("/api/dataset/import/server_folder", json={"path": str(tmp_path), "media_type": "audio"})

        assert resp.status_code == 200, resp.get_data(as_text=True)
        assert resp.get_json()["task_id"] == "t-9"
        assert resp.get_json()["task_ids"] == ["t-9"]

    def test_invalid_outputs_are_a_400(self, client, tmp_path):
        resp = client.post(
            "/api/dataset/import/server_folder",
            json={"path": str(tmp_path), "outputs": [{"media_type": "hologram"}]},
        )
        assert resp.status_code == 400
        assert "Unknown media type" in resp.get_json()["message"]

    def test_converter_not_producing_the_output_is_a_400(self, client, tmp_path):
        resp = client.post(
            "/api/dataset/import/server_folder",
            json={
                "path": str(tmp_path),
                "outputs": [
                    {
                        "media_type": "face",
                        "source_specs": [{"source_type": "document", "converter": "document2image", "params": {}}],
                    }
                ],
            },
        )
        assert resp.status_code == 400
        assert "produces" in resp.get_json()["message"]

    def test_an_importer_fixed_to_one_dataset_refuses_outputs(self, client):
        from vtscore.datasets import DEMO_DATASETS

        resp = client.post(
            "/api/dataset/import/demo",
            json={"name": next(iter(DEMO_DATASETS)), "outputs": [{"media_type": "audio"}]},
        )
        assert resp.status_code == 400
        assert "one dataset per run" in resp.get_json()["message"]


class TestLocalFolderRoute:
    def test_outputs_start_a_multi_dataset_upload(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("vtsearch.routes.datasets.load.LOCAL_UPLOADS_DIR", tmp_path / "uploads")
        captured: dict = {}

        def fake_multi(importer, field_values, outputs, **kwargs):
            captured.update(importer=importer, field_values=field_values, outputs=outputs, kwargs=kwargs)
            return ["t-a", "t-b"]

        with patch("vtsearch.routes.datasets.load._run_multi_output_load_in_background", side_effect=fake_multi):
            resp = client.post(
                "/api/dataset/import-local-folder",
                data={
                    "outputs": json.dumps(_OUTPUTS),
                    "dataset_name": "snaps",
                    "build_projection": "true",
                    "files": [(io.BytesIO(b"AAA"), "myfolder/one.wav"), (io.BytesIO(b"BBB"), "myfolder/two.pdf")],
                },
                content_type="multipart/form-data",
            )

        assert resp.status_code == 200, resp.get_data(as_text=True)
        assert resp.get_json()["task_ids"] == ["t-a", "t-b"]
        assert resp.get_json()["task_id"] == "t-a"

        assert captured["importer"].name == "server_folder"
        assert [o.media_type for o in captured["outputs"]] == ["audio", "image"]
        fv = captured["field_values"]
        assert fv["dataset_name"] == "snaps"
        assert fv["build_projection"] == "true"
        upload_dir = tmp_path / "uploads"
        assert str(fv["path"]).startswith(str(upload_dir)), "the importer walks the staged upload"
        assert (upload_dir / fv["path"].split("/")[-1] / "myfolder" / "one.wav").exists()

        # The synthetic origin names the browser upload, per output type.
        origin = captured["kwargs"]["origin_for"](captured["outputs"][1], {"media_type": "image"})
        assert origin == {"importer": "server_folder", "params": {"path": "<browser_upload>", "media_type": "image"}}
        # The cleanup hook removes the staged upload.
        captured["kwargs"]["cleanup"]()
        assert not (upload_dir / fv["path"].split("/")[-1]).exists()

    def test_media_type_is_optional_with_outputs_only(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("vtsearch.routes.datasets.load.LOCAL_UPLOADS_DIR", tmp_path / "uploads")
        resp = client.post(
            "/api/dataset/import-local-folder",
            data={"files": (io.BytesIO(b"AAA"), "myfolder/one.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert "media_type" in resp.get_json()["message"]

    def test_invalid_outputs_are_a_400(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("vtsearch.routes.datasets.load.LOCAL_UPLOADS_DIR", tmp_path / "uploads")
        resp = client.post(
            "/api/dataset/import-local-folder",
            data={"outputs": "[not json", "files": (io.BytesIO(b"AAA"), "myfolder/one.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert "Invalid outputs" in resp.get_json()["message"]

    def test_single_dataset_upload_lists_its_one_task(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("vtsearch.routes.datasets.load.LOCAL_UPLOADS_DIR", tmp_path / "uploads")
        with patch("vtsearch.routes.datasets.load._run_origin_load_in_background", return_value="t-7"):
            resp = client.post(
                "/api/dataset/import-local-folder",
                data={"media_type": "audio", "files": (io.BytesIO(b"AAA"), "myfolder/one.wav")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200, resp.get_data(as_text=True)
        assert resp.get_json()["task_ids"] == ["t-7"]


class TestLocalFilesRoute:
    def test_outputs_start_a_multi_dataset_upload(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("vtsearch.routes.datasets.load.LOCAL_UPLOADS_DIR", tmp_path / "uploads")
        captured: dict = {}

        def fake_multi(importer, field_values, outputs, **kwargs):
            captured.update(importer=importer, field_values=field_values, outputs=outputs, kwargs=kwargs)
            return ["t-x"]

        with patch("vtsearch.routes.datasets.load._run_multi_output_load_in_background", side_effect=fake_multi):
            resp = client.post(
                "/api/dataset/import-local-files",
                data={
                    "outputs": json.dumps([{"media_type": "audio"}]),
                    "paths_file": (io.BytesIO(b"/data/a.wav\n"), "paths.txt"),
                },
                content_type="multipart/form-data",
            )

        assert resp.status_code == 200, resp.get_data(as_text=True)
        assert resp.get_json()["task_ids"] == ["t-x"]
        assert captured["importer"].name == "server_files"
        assert captured["field_values"]["dataset_name"] == "Local files upload"
        assert captured["field_values"]["paths_file"].endswith("paths_file.txt")
        origin = captured["kwargs"]["origin_for"](captured["outputs"][0], {"media_type": "audio"})
        assert origin["params"] == {"paths_file": "<browser_upload>", "media_type": "audio"}


class TestRegistryListsTheImportGroup:
    def test_siblings_carry_their_group_and_category(self, client, tmp_path):
        from vtscore.datasets.registry import register_dataset

        for category, media_type in (("image", "image"), ("document", "image")):
            register_dataset(
                name=f"holiday – {category}",
                media_type=media_type,
                num_items=1,
                pkl_path=str(tmp_path / f"{category}.pkl"),
                import_group="g-holiday",
                output_category=category,
            )
        register_dataset(name="solo", media_type="audio", num_items=1, pkl_path=str(tmp_path / "solo.pkl"))

        resp = client.get("/api/datasets/registry")

        assert resp.status_code == 200, resp.get_data(as_text=True)
        by_name = {d["name"]: d for d in resp.get_json()["datasets"]}
        assert by_name["holiday – image"]["import_group"] == "g-holiday"
        assert by_name["holiday – document"]["import_group"] == "g-holiday"
        assert by_name["holiday – document"]["output_category"] == "document"
        assert by_name["holiday – document"]["media_type"] == "image"
        assert by_name["solo"]["import_group"] is None
        assert by_name["solo"]["output_category"] == ""

    def test_an_entry_saved_before_the_fields_reads_as_ungrouped(self, client, tmp_path):
        """The manifest is saved data with no migration: an older entry just lacks the keys."""
        from vtscore.datasets import registry

        entry = registry.register_dataset(name="old", media_type="audio", num_items=1, pkl_path=str(tmp_path / "o.pkl"))
        entries = registry._load()
        for e in entries:
            if e["id"] == entry["id"]:
                del e["import_group"], e["output_category"]
        registry._save(entries)

        resp = client.get("/api/datasets/registry")

        (listed,) = resp.get_json()["datasets"]
        assert listed["import_group"] is None
        assert listed["output_category"] == ""
