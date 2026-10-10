"""``GET /api/medias/<id>/source`` and ``POST /api/medias/source-batch`` (#4749).

A face crop resolves to its photo in the Image dataset the same multi-dataset
import produced; the routes read that sibling through its own context, ask for
it to be loaded when it is not, and never resolve into a dataset the current
user cannot read.
"""

from __future__ import annotations

from typing import Any

from vtscore.datasets.registry import register_dataset
from vtscore.state.core import DatasetContext, register_context


def _photo(mid: int, name: str) -> dict[str, Any]:
    return {
        "id": mid,
        "media_type": "image",
        "md5": f"md5-{mid}",
        "filename": name,
        "origin": {"importer": "server_folder", "params": {"path": "/data/photos"}},
        "origin_name": name,
        "media_path": f"/data/photos/{name}",
    }


def _face(mid: int, source_file: str, box: list[float] | None = None) -> dict[str, Any]:
    media: dict[str, Any] = {
        "id": mid,
        "media_type": "face",
        "md5": f"face-md5-{mid}",
        "filename": f"{source_file}→face_0.jpg",
        "origin": {
            "importer": "converter",
            "params": {
                "converter": "image2face",
                "source_file": source_file,
                "source_path": f"/data/photos/{source_file}",
            },
        },
        "origin_name": f"{source_file}→face_0.jpg",
        "media_path": f"/data/photos/{source_file}",
    }
    if box is not None:
        media["source_box"] = box
    return media


def _register(name: str, media_type: str, category: str, group: str | None, **kw: Any) -> dict[str, Any]:
    return register_dataset(
        name=name,
        media_type=media_type,
        num_items=2,
        pkl_path=f"/tmp/{name}.pkl",
        import_group=group,
        output_category=category,
        **kw,
    )


def _load(entry: dict[str, Any], *medias: dict[str, Any]) -> DatasetContext:
    ctx = DatasetContext(entry["id"])
    ctx.medias.update({m["id"]: m for m in medias})
    register_context(ctx)
    return ctx


def _setup(*, load_images: bool = True, image_kw: dict[str, Any] | None = None, group: str | None = "g1"):
    """A Face dataset (loaded) and its Image sibling; returns ``(face_id, image_id)``."""
    faces = _register("t – Face", "face", "face", group)
    images = _register("t – Image", "image", "image", group, **(image_kw or {}))
    _load(faces, _face(1, "b.jpg", box=[0.1, 0.2, 0.3, 0.4]), _face(2, "gone.jpg"), _photo(3, "stray.jpg"))
    if load_images:
        _load(images, _photo(1, "a.jpg"), _photo(2, "b.jpg"))
    return faces["id"], images["id"]


def _get(client, face_id: str, media_id: int):
    return client.get(f"/api/medias/{media_id}/source", headers={"X-Dataset-Id": face_id})


def _batch(client, face_id: str, ids: list[int]):
    return client.post("/api/medias/source-batch", json={"ids": ids}, headers={"X-Dataset-Id": face_id})


class TestMediaSource:
    def test_resolves_the_photo_in_the_loaded_sibling(self, client):
        face_id, image_id = _setup()
        resp = _get(client, face_id, 1)
        assert resp.status_code == 200, resp.get_json()
        assert resp.get_json() == {"dataset_id": image_id, "media_id": 2, "box": [0.1, 0.2, 0.3, 0.4]}

    def test_an_unloaded_sibling_is_a_409_naming_it(self, client):
        face_id, image_id = _setup(load_images=False)
        resp = _get(client, face_id, 1)
        assert resp.status_code == 409
        body = resp.get_json()
        assert body["error_code"] == "source_not_loaded"
        assert body["dataset_id"] == image_id

    def test_no_import_group_is_a_404(self, client):
        face_id, _image_id = _setup(group=None)
        resp = _get(client, face_id, 1)
        assert resp.status_code == 404
        assert resp.get_json()["error_code"] == "no_source_dataset"

    def test_an_unreadable_sibling_reads_as_no_sibling(self, client):
        """The Image sibling belongs to alice and is not shared: the default user gets no answer."""
        face_id, _image_id = _setup(image_kw={"created_by": "alice"})
        resp = _get(client, face_id, 1)
        assert resp.status_code == 404
        body = resp.get_json()
        assert body["error_code"] == "no_source_dataset"
        assert "dataset_id" not in body

    def test_a_shared_sibling_resolves(self, client):
        face_id, image_id = _setup(image_kw={"created_by": "alice", "readers": ["*"]})
        resp = _get(client, face_id, 1)
        assert resp.status_code == 200
        assert resp.get_json()["dataset_id"] == image_id

    def test_no_matching_photo_is_a_404(self, client):
        face_id, _image_id = _setup()
        resp = _get(client, face_id, 2)
        assert resp.status_code == 404
        assert resp.get_json()["error_code"] == "source_not_found"

    def test_a_plain_media_is_not_derived(self, client):
        face_id, _image_id = _setup()
        resp = _get(client, face_id, 3)
        assert resp.status_code == 404
        assert resp.get_json()["error_code"] == "not_derived"

    def test_an_unknown_media_is_a_404(self, client):
        face_id, _image_id = _setup()
        assert _get(client, face_id, 99).status_code == 404

    def test_the_dataset_header_is_required(self, client):
        _setup()
        resp = client.get("/api/medias/1/source", headers={"X-Dataset-Id": ""})
        assert resp.status_code == 400


class TestMediaSourceBatch:
    def test_returns_resolved_ids_in_request_order_and_omits_the_rest(self, client):
        face_id, image_id = _setup()
        resp = _batch(client, face_id, [3, 99, 1, 2])
        assert resp.status_code == 200, resp.get_json()
        assert resp.get_json() == [{"id": 1, "dataset_id": image_id, "media_id": 2, "box": [0.1, 0.2, 0.3, 0.4]}]

    def test_an_unloaded_sibling_is_a_409_naming_it(self, client):
        face_id, image_id = _setup(load_images=False)
        resp = _batch(client, face_id, [1, 2])
        assert resp.status_code == 409
        body = resp.get_json()
        assert (body["error_code"], body["dataset_id"]) == ("source_not_loaded", image_id)

    def test_ids_with_no_source_never_ask_for_a_load(self, client):
        """A plain media needs no sibling, so an unloaded one does not block it."""
        face_id, _image_id = _setup(load_images=False)
        resp = _batch(client, face_id, [3, 99])
        assert resp.status_code == 200
        assert resp.get_json() == []

    def test_reader_confinement_applies(self, client):
        face_id, _image_id = _setup(image_kw={"created_by": "alice"})
        resp = _batch(client, face_id, [1])
        assert resp.status_code == 200
        assert resp.get_json() == []
