"""``GET /api/sort/collapsed``: a ranking of faces, grouped by photo (#4751).

The grouping runs over the whole stored ranking, once per sort token, and is
windowed after; votes are read per request and stay on the faces; the photo
dataset must be loaded, and only one the user can read is grouped into.
"""

from __future__ import annotations

from typing import Any

from tests import load_detector_and_wait
from tests.fixtures.medias import NUM_MEDIAS
from tests.helpers import setup_trainable_model_in_registry
from vtscore.datasets.registry import register_dataset
from vtscore.state.core import (
    DatasetContext,
    DetectorContext,
    register_context,
    register_detector_context,
    unregister_context,
)
from vtscore.state.sort_results_cache import sort_results_cache
from vtsearch.state import snapshot_medias


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


def _face(mid: int, source_file: str, box: list[float]) -> dict[str, Any]:
    return {
        "id": mid,
        "media_type": "face",
        "md5": f"face-md5-{mid}",
        "filename": f"{source_file}→face_{mid}.jpg",
        "origin": {
            "importer": "converter",
            "params": {
                "converter": "image2face",
                "source_file": source_file,
                "source_path": f"/data/photos/{source_file}",
            },
        },
        "origin_name": f"{source_file}→face_{mid}.jpg",
        "media_path": f"/data/photos/{source_file}",
        "source_box": box,
    }


_BOX = {1: [0.1, 0.1, 0.2, 0.2], 2: [0.5, 0.5, 0.6, 0.6], 3: [0.3, 0.3, 0.4, 0.4], 4: [0.0, 0.0, 0.1, 0.1]}
#: Faces 1 and 3 are in a.jpg, face 2 in b.jpg, face 4's photo was left out of the import.
_FACES = {1: "a.jpg", 2: "b.jpg", 3: "a.jpg", 4: "gone.jpg"}
#: Ranked 1, 2, 4, 3: photo a's best and worst faces are the ranking's ends.
_RANKING = [{"id": 1, "score": 0.9}, {"id": 2, "score": 0.7}, {"id": 4, "score": 0.5}, {"id": 3, "score": 0.1}]


def _register(name: str, media_type: str, group: str | None, **kw: Any) -> dict[str, Any]:
    return register_dataset(
        name=name,
        media_type=media_type,
        num_items=2,
        pkl_path=f"/tmp/{name}.pkl",
        import_group=group,
        output_category=media_type,
        **kw,
    )


def _load_images(image_id: str) -> None:
    ctx = DatasetContext(image_id)
    ctx.medias.update({10: _photo(10, "a.jpg"), 11: _photo(11, "b.jpg")})
    register_context(ctx)


def _setup(*, load_images: bool = True, image_kw: dict[str, Any] | None = None) -> tuple[str, str, str]:
    """Face and Image datasets of one import, a ranking of the faces; returns ``(face_id, image_id, token)``."""
    faces = _register("t – Face", "face", "g1")
    images = _register("t – Image", "image", "g1", **(image_kw or {}))
    face_ctx = DatasetContext(faces["id"])
    face_ctx.medias.update({mid: _face(mid, name, _BOX[mid]) for mid, name in _FACES.items()})
    register_context(face_ctx)
    if load_images:
        _load_images(images["id"])
    token = sort_results_cache.store(list(_RANKING), 0.6, dataset_id=faces["id"])
    return faces["id"], images["id"], token


def _detector(good: tuple[int, ...] = (), bad: tuple[int, ...] = ()) -> str:
    ctx = DetectorContext("det-collapse", name="collapse")
    ctx.good_votes.update({mid: None for mid in good})
    ctx.bad_votes.update({mid: None for mid in bad})
    register_detector_context(ctx)
    return ctx.detector_id


def _get(client, face_id: str, token: str, *, offset: int = 0, limit: int = 50, detector_id: str = ""):
    headers = {"X-Dataset-Id": face_id}
    if detector_id:
        headers["X-Detector-Id"] = detector_id
    return client.get(f"/api/sort/collapsed?token={token}&offset={offset}&limit={limit}", headers=headers)


class TestCollapsed:
    def test_groups_the_ranking_by_photo(self, client):
        face_id, image_id, token = _setup()
        resp = _get(client, face_id, token, detector_id=_detector())
        assert resp.status_code == 200, resp.get_json()
        body = resp.get_json()
        assert body["results"] == [
            {
                "source_dataset_id": image_id,
                "source_media_id": 10,
                "score": 0.9,
                "faces": [
                    {"media_id": 1, "score": 0.9, "box": _BOX[1], "vote": None},
                    {"media_id": 3, "score": 0.1, "box": _BOX[3], "vote": None},
                ],
            },
            {
                "source_dataset_id": image_id,
                "source_media_id": 11,
                "score": 0.7,
                "faces": [{"media_id": 2, "score": 0.7, "box": _BOX[2], "vote": None}],
            },
            {
                "source_dataset_id": None,
                "source_media_id": None,
                "score": 0.5,
                "faces": [{"media_id": 4, "score": 0.5, "box": _BOX[4], "vote": None}],
            },
        ]
        assert (body["total"], body["face_total"], body["has_more"]) == (3, 4, False)

    def test_windows_the_grouped_list_not_the_ranking(self, client):
        """Face 3 is the ranking's last row, yet it rides in the first window, with its photo."""
        face_id, _image_id, token = _setup()
        first = _get(client, face_id, token, limit=1).get_json()
        assert [f["media_id"] for f in first["results"][0]["faces"]] == [1, 3]
        assert (first["offset"], first["limit"], first["total"], first["has_more"]) == (0, 1, 3, True)
        rest = _get(client, face_id, token, offset=1, limit=5).get_json()
        assert [r["source_media_id"] for r in rest["results"]] == [11, None]
        assert rest["has_more"] is False

    def test_votes_are_read_per_request(self, client):
        face_id, _image_id, token = _setup()
        detector_id = _detector(good=(3,), bad=(2,))
        body = _get(client, face_id, token, detector_id=detector_id).get_json()
        votes = {f["media_id"]: f["vote"] for row in body["results"] for f in row["faces"]}
        assert votes == {1: None, 2: "bad", 3: "good", 4: None}

        client.post(
            "/api/medias/1/vote",
            json={"target": "good"},
            headers={"X-Dataset-Id": face_id, "X-Detector-Id": detector_id},
        )
        again = _get(client, face_id, token, detector_id=detector_id).get_json()
        assert again["results"][0]["faces"][0]["vote"] == "good"

    def test_an_unloaded_photo_dataset_is_a_409_naming_it_then_groups_once_loaded(self, client):
        face_id, image_id, token = _setup(load_images=False)
        resp = _get(client, face_id, token)
        assert resp.status_code == 409
        body = resp.get_json()
        assert (body["error_code"], body["dataset_id"]) == ("source_not_loaded", image_id)

        _load_images(image_id)
        assert _get(client, face_id, token).get_json()["total"] == 3

    def test_a_photo_dataset_unloaded_since_grouping_asks_again(self, client):
        face_id, image_id, token = _setup()
        assert _get(client, face_id, token).status_code == 200
        unregister_context(image_id)
        resp = _get(client, face_id, token)
        assert resp.status_code == 409
        assert resp.get_json()["dataset_id"] == image_id

    def test_an_unreadable_photo_dataset_leaves_every_face_a_row_of_its_own(self, client):
        face_id, _image_id, token = _setup(image_kw={"created_by": "alice"})
        body = _get(client, face_id, token).get_json()
        assert [r["source_dataset_id"] for r in body["results"]] == [None, None, None, None]
        assert [r["faces"][0]["media_id"] for r in body["results"]] == [1, 2, 4, 3]

    def test_an_unknown_token_is_a_404(self, client):
        face_id, _image_id, _token = _setup()
        assert _get(client, face_id, "no-such-token").status_code == 404

    def test_another_datasets_token_is_a_404(self, client):
        face_id, image_id, _token = _setup()
        other = sort_results_cache.store(list(_RANKING), 0.6, dataset_id=image_id)
        assert _get(client, face_id, other).status_code == 404

    def test_the_dataset_header_is_required(self, client):
        _face_id, _image_id, token = _setup()
        resp = client.get(f"/api/sort/collapsed?token={token}", headers={"X-Dataset-Id": ""})
        assert resp.status_code == 400


class TestCollapsedOverRealSorts:
    def test_a_text_sort_over_plain_media_is_one_row_per_item(self, client):
        """No item is a converter output: collapsing changes nothing but the shape."""
        sort = client.post("/api/sort", json={"text": "a beeping sound"}).get_json()
        body = client.get(f"/api/sort/collapsed?token={sort['sort_token']}&limit=100").get_json()
        assert (body["total"], body["face_total"]) == (NUM_MEDIAS, NUM_MEDIAS)
        assert [r["faces"][0]["media_id"] for r in body["results"]] == [r["id"] for r in sort["results"]]
        assert all(r["source_media_id"] is None for r in body["results"])

    def test_find_label_hands_back_a_token_for_its_ranking(self, client):
        detector_id = setup_trainable_model_in_registry(
            "collapse-find", good_ids=[], bad_ids=[], snap=snapshot_medias()
        )
        load_detector_and_wait(client, detector_id)
        for media_id in (1, 2, 3):
            client.post(f"/api/medias/{media_id}/vote", json={"target": "good"})
        for media_id in (17, 18, 19, 20):
            client.post(f"/api/medias/{media_id}/vote", json={"target": "bad"})
        find = client.post("/api/find-label", json={"detector_id": detector_id}).get_json()
        assert find["results"] and isinstance(find["sort_token"], str)

        body = client.get(f"/api/sort/collapsed?token={find['sort_token']}&limit=100").get_json()
        assert [r["faces"][0]["media_id"] for r in body["results"]] == [r["id"] for r in find["results"]]
        # Find adopted a label for every item; the rows carry them.
        assert {r["faces"][0]["vote"] for r in body["results"]} <= {"good", "bad"}
