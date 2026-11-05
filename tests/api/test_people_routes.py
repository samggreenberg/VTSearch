"""The People routes: a Face dataset's people, one person, the photos two people share, a build (#5589).

A small Face dataset of one import with its Image sibling: three people and
one unplaced hard view over four photos.

* a.jpg holds faces 1 (person A), 4 (B), 7 (C) and 8 (a hard view, unplaced);
* b.jpg holds 2 (A) and 5 (B);
* c.jpg holds 3 (A), d.jpg holds 6 (B).

So A is rank 1 and B rank 2 (three faces each, ties to the lowest face id),
C rank 3; A appears with B in two photos and with C in one.  The fit is the
real one: what the routes serve is checked against people the fit has to find.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from vtscore.concurrency.async_jobs import projection_jobs
from vtscore.datasets import people as pp
from vtscore.datasets.registry import register_dataset
from vtscore.detectors.centroid_head import FACE_BALL_COMPAT, FACE_BALL_FLOORS
from vtscore.media.face.imagekit_bridge import COMPAT_METADATA_KEY, HARD_VIEW_QUALITY, QUALITY_METADATA_KEY
from vtscore.state.core import DatasetContext, register_context, unregister_context

SPACE = FACE_BALL_COMPAT
GOOD_Q = HARD_VIEW_QUALITY[SPACE] + 10.0
HARD_Q = HARD_VIEW_QUALITY[SPACE] - 5.0
D = 32

#: Face id -> (photo, axis its vector sits near, Face Quality).  Faces 1 and 4
#: have the best quality of their people, so they are the representatives.
FACES: dict[int, tuple[str, int, float]] = {
    1: ("a.jpg", 0, GOOD_Q + 5.0),
    2: ("b.jpg", 0, GOOD_Q),
    3: ("c.jpg", 0, GOOD_Q),
    4: ("a.jpg", 1, GOOD_Q + 5.0),
    5: ("b.jpg", 1, GOOD_Q),
    6: ("d.jpg", 1, GOOD_Q),
    7: ("a.jpg", 2, GOOD_Q),
    8: ("a.jpg", 3, HARD_Q),
}
PHOTOS = {"a.jpg": 10, "b.jpg": 11, "c.jpg": 12, "d.jpg": 13}


def _box(face_id: int) -> list[float]:
    x = face_id / 10
    return [x, x, x + 0.05, x + 0.05]


def _vector(face_id: int, axis: int) -> np.ndarray:
    rng = np.random.default_rng(face_id)
    vec = np.zeros(D)
    vec[axis] = 1.0
    if axis < 3:  # the people's faces sit close around their axis; the hard view sits on its own
        vec = vec + 0.1 * rng.standard_normal(D) / np.sqrt(D)
    return (vec / np.linalg.norm(vec)).astype(np.float32)


def _face(face_id: int, photo: str, axis: int, quality: float, compat: str) -> dict[str, Any]:
    return {
        "id": face_id,
        "media_type": "face",
        "md5": f"face-md5-{face_id}",
        "filename": f"{photo}→face_{face_id}.jpg",
        "origin": {
            "importer": "converter",
            "params": {"converter": "image2face", "source_file": photo, "source_path": f"/data/photos/{photo}"},
        },
        "origin_name": f"{photo}→face_{face_id}.jpg",
        "media_path": f"/data/photos/{photo}",
        "source_box": _box(face_id),
        "embedder": "face",
        "embeddings": {"face": _vector(face_id, axis)},
        "custom_metadata": {COMPAT_METADATA_KEY: compat, QUALITY_METADATA_KEY: quality},
    }


def _photo(media_id: int, name: str) -> dict[str, Any]:
    return {
        "id": media_id,
        "media_type": "image",
        "md5": f"md5-{media_id}",
        "filename": name,
        "origin": {"importer": "server_folder", "params": {"path": "/data/photos"}},
        "origin_name": name,
        "media_path": f"/data/photos/{name}",
    }


def _load_images(image_id: str) -> None:
    ctx = DatasetContext(image_id)
    ctx.medias.update({mid: _photo(mid, name) for name, mid in PHOTOS.items()})
    register_context(ctx)


def _await_jobs() -> None:
    """Block until the projection runner has nothing running or parked."""
    for _ in range(600):
        active = projection_jobs.active_jobs()
        if not active:
            return
        active[0].done_event.wait(timeout=0.05)
    raise TimeoutError("projection job did not finish")


class _Faces:
    """The Face dataset (loaded) and its Image sibling (registered; loaded unless asked not to)."""

    def __init__(
        self,
        tmp_path,
        *,
        compat: str = SPACE,
        fit: bool = True,
        load_images: bool = True,
        image_kw: dict[str, Any] | None = None,
    ) -> None:
        faces = register_dataset(
            name="t – Face",
            media_type="face",
            num_items=len(FACES),
            pkl_path=str(tmp_path / "faces.pkl"),
            import_group="g1",
            output_category="face",
        )
        images = register_dataset(
            name="t – Image",
            media_type="image",
            num_items=len(PHOTOS),
            pkl_path=str(tmp_path / "images.pkl"),
            import_group="g1",
            output_category="image",
            **(image_kw or {}),
        )
        self.face_id: str = faces["id"]
        self.image_id: str = images["id"]
        self.ctx = DatasetContext(self.face_id)
        self.ctx.medias.update({fid: _face(fid, *spec, compat) for fid, spec in FACES.items()})
        register_context(self.ctx)
        if load_images:
            _load_images(self.image_id)
        if fit:
            assert pp.people_for(self.ctx) is not None

    def get(self, client, path: str):
        return client.get(path, headers={"X-Dataset-Id": self.face_id})

    def post(self, client, path: str, body: dict | None = None):
        return client.post(path, json=body or {}, headers={"X-Dataset-Id": self.face_id})

    def listing(self, client) -> dict[str, Any]:
        resp = self.get(client, "/api/people")
        assert resp.status_code == 200, resp.get_json()
        return resp.get_json()

    def move(self, face_id: int, axis: int) -> None:
        """Re-embed one face near another person's axis (a fit after this moves it)."""
        photo, _axis, quality = FACES[face_id]
        self.ctx.medias[face_id] = _face(face_id, photo, axis, quality, SPACE)


@pytest.fixture
def faces(tmp_path) -> _Faces:
    return _Faces(tmp_path)


class TestList:
    def test_lists_each_person_with_counts_and_representative(self, client, faces):
        body = faces.listing(client)
        assert body["status"] == "ready"
        assert body["people"] == [
            {"id": 1, "face_count": 3, "photo_count": 3, "representative": 1},
            {"id": 2, "face_count": 3, "photo_count": 3, "representative": 4},
            {"id": 3, "face_count": 1, "photo_count": 1, "representative": 7},
        ]
        assert (body["unplaced"], body["unplaced_ids"]) == (1, [8])
        assert body["cutoff"] == pytest.approx(FACE_BALL_FLOORS[1.0])
        assert (body["compat"], body["method"], body["linkage"]) == (SPACE, pp.PEOPLE_METHOD, pp.PEOPLE_LINKAGE)
        assert isinstance(body["partition"], str) and body["partition"]

    def test_the_identity_is_stable_across_asks(self, client, faces):
        assert faces.listing(client)["partition"] == faces.listing(client)["partition"]

    def test_a_first_ask_builds_then_answers_ready(self, client, tmp_path):
        faces = _Faces(tmp_path, fit=False)
        first = faces.listing(client)
        assert first["status"] == "building"
        assert first["job_id"] == faces.ctx._people_job_id
        _await_jobs()
        ready = faces.listing(client)
        assert ready["status"] == "ready"
        assert [p["face_count"] for p in ready["people"]] == [3, 3, 1]

    def test_an_uncalibrated_face_space_is_told_so(self, client, tmp_path):
        faces = _Faces(tmp_path, compat="ext-acme:3", fit=False)
        body = faces.listing(client)
        assert body == {
            "status": "not_calibrated",
            "compat": "ext-acme:3",
            "message": "People is not calibrated for this face embedder (ext-acme:3).",
        }
        assert projection_jobs.active_jobs() == []

    def test_a_dataset_of_another_type_is_a_409(self, client):
        ctx = DatasetContext("photos-only")
        ctx.medias.update({mid: _photo(mid, name) for name, mid in PHOTOS.items()})
        register_context(ctx)
        resp = client.get("/api/people", headers={"X-Dataset-Id": "photos-only"})
        assert resp.status_code == 409
        assert resp.get_json()["error_code"] == "no_people"
        person = client.get("/api/people/1?partition=x", headers={"X-Dataset-Id": "photos-only"})
        assert (person.status_code, person.get_json()["error_code"]) == (409, "no_people")

    def test_the_dataset_header_is_required(self, client, faces):
        assert client.get("/api/people", headers={"X-Dataset-Id": ""}).status_code == 400
        assert client.post("/api/people/build", json={}, headers={"X-Dataset-Id": ""}).status_code == 400


class TestBuild:
    def test_without_force_it_answers_as_the_list_does(self, client, faces):
        resp = faces.post(client, "/api/people/build")
        assert resp.status_code == 200
        assert resp.get_json() == faces.listing(client)

    def test_an_empty_body_is_no_force(self, client, faces):
        resp = client.post("/api/people/build", headers={"X-Dataset-Id": faces.face_id})
        assert resp.status_code == 200 and resp.get_json()["status"] == "ready"

    def test_a_forced_fit_that_moves_people_turns_a_held_id_away(self, client, faces):
        """Person 1 is A until a fit makes B the largest; a client holding A's id gets a 409, not B."""
        old = faces.listing(client)
        assert faces.get(client, f"/api/people/1?partition={old['partition']}").get_json()["representative"] == 1

        faces.move(3, axis=1)
        # The stored partition still covers every face, so it serves until it is fit again.
        assert faces.listing(client)["partition"] == old["partition"]
        building = faces.post(client, "/api/people/build", {"force": True}).get_json()
        assert building["status"] == "building" and building["job_id"]
        _await_jobs()

        new = faces.listing(client)
        assert new["partition"] != old["partition"]
        assert new["people"][0] == {"id": 1, "face_count": 4, "photo_count": 4, "representative": 4}
        stale = faces.get(client, f"/api/people/1?partition={old['partition']}")
        assert stale.status_code == 409
        assert stale.get_json()["error_code"] == "people_changed"
        fresh = faces.get(client, f"/api/people/1?partition={new['partition']}")
        assert fresh.status_code == 200
        assert set(fresh.get_json()["face_ids"]) == {3, 4, 5, 6}

    def test_a_forced_fit_that_finds_the_same_people_keeps_their_ids(self, client, faces):
        old = faces.listing(client)
        assert faces.post(client, "/api/people/build", {"force": True}).get_json()["status"] == "building"
        _await_jobs()
        assert faces.listing(client)["partition"] == old["partition"]
        assert faces.get(client, f"/api/people/2?partition={old['partition']}").status_code == 200


class TestPerson:
    def test_faces_photos_and_who_appears_with_them(self, client, faces):
        partition = faces.listing(client)["partition"]
        resp = faces.get(client, f"/api/people/1?partition={partition}")
        assert resp.status_code == 200, resp.get_json()
        body = resp.get_json()
        assert (body["partition"], body["id"], body["representative"]) == (partition, 1, 1)
        assert body["face_ids"][0] == 1 and sorted(body["face_ids"]) == [1, 2, 3]
        # Photos are numbered by their lowest face: a.jpg 0, b.jpg 1, c.jpg 2, d.jpg 3.
        assert body["photos"] == [
            {"photo": 0, "face_ids": [1]},
            {"photo": 1, "face_ids": [2]},
            {"photo": 2, "face_ids": [3]},
        ]
        assert body["appears_with"] == [
            {"id": 2, "shared": 2, "representative": 4},
            {"id": 3, "shared": 1, "representative": 7},
        ]

    def test_needs_no_image_sibling(self, client, tmp_path):
        faces = _Faces(tmp_path, load_images=False)
        partition = faces.listing(client)["partition"]
        resp = faces.get(client, f"/api/people/3?partition={partition}")
        assert resp.status_code == 200
        assert resp.get_json()["appears_with"] == [
            {"id": 1, "shared": 1, "representative": 1},
            {"id": 2, "shared": 1, "representative": 4},
        ]

    def test_an_id_with_no_person_is_a_404(self, client, faces):
        partition = faces.listing(client)["partition"]
        assert faces.get(client, f"/api/people/4?partition={partition}").status_code == 404
        assert faces.get(client, f"/api/people/0?partition={partition}").status_code == 404

    def test_the_partition_is_required(self, client, faces):
        assert faces.get(client, "/api/people/1").status_code == 422

    def test_people_not_found_yet_are_a_409_and_start_no_fit(self, client, tmp_path):
        faces = _Faces(tmp_path, fit=False)
        resp = faces.get(client, "/api/people/1?partition=anything")
        assert resp.status_code == 409
        assert resp.get_json()["error_code"] == "people_not_ready"
        assert projection_jobs.active_jobs() == [] and faces.ctx._people_job_id is None

    def test_a_removed_face_changes_the_identity(self, client, faces):
        """The stored partition still serves, without the face; ids read before the removal are turned away."""
        old = faces.listing(client)
        del faces.ctx.medias[2]
        new = faces.listing(client)
        assert new["partition"] != old["partition"]
        assert new["people"][0] == {"id": 1, "face_count": 2, "photo_count": 2, "representative": 1}
        assert faces.get(client, f"/api/people/1?partition={old['partition']}").get_json()["error_code"] == (
            "people_changed"
        )


class TestSharedPhotos:
    def _shared(self, client, faces: _Faces, a: int = 1, b: int = 2):
        partition = faces.listing(client)["partition"]
        return faces.get(client, f"/api/people/{a}/with/{b}?partition={partition}")

    def test_the_photos_two_people_share_with_everyone_in_them(self, client, faces):
        resp = self._shared(client, faces)
        assert resp.status_code == 200, resp.get_json()
        body = resp.get_json()
        assert (body["id"], body["other"]) == (1, 2)
        assert body["photos"] == [
            {
                "photo": 0,
                "source_dataset_id": faces.image_id,
                "source_media_id": PHOTOS["a.jpg"],
                "faces": [
                    {"media_id": 1, "box": _box(1), "person": 1},
                    {"media_id": 4, "box": _box(4), "person": 2},
                    {"media_id": 7, "box": _box(7), "person": 3},
                    {"media_id": 8, "box": _box(8), "person": None},
                ],
            },
            {
                "photo": 1,
                "source_dataset_id": faces.image_id,
                "source_media_id": PHOTOS["b.jpg"],
                "faces": [
                    {"media_id": 2, "box": _box(2), "person": 1},
                    {"media_id": 5, "box": _box(5), "person": 2},
                ],
            },
        ]

    def test_two_people_share_only_the_photos_both_are_in(self, client, faces):
        """B and C share a.jpg alone: b.jpg and d.jpg hold no face of C."""
        body = self._shared(client, faces, 2, 3).get_json()
        assert [p["photo"] for p in body["photos"]] == [0]

    def test_an_unloaded_image_sibling_is_the_shared_409(self, client, tmp_path):
        faces = _Faces(tmp_path, load_images=False)
        resp = self._shared(client, faces)
        assert resp.status_code == 409
        body = resp.get_json()
        assert (body["error_code"], body["dataset_id"]) == ("source_not_loaded", faces.image_id)

        _load_images(faces.image_id)
        assert self._shared(client, faces).get_json()["photos"][0]["source_media_id"] == PHOTOS["a.jpg"]
        unregister_context(faces.image_id)
        assert self._shared(client, faces).status_code == 409

    def test_an_image_sibling_the_user_cannot_read_names_no_photo(self, client, tmp_path):
        faces = _Faces(tmp_path, image_kw={"created_by": "alice"})
        body = self._shared(client, faces).get_json()
        assert [(p["source_dataset_id"], p["source_media_id"]) for p in body["photos"]] == [(None, None)] * 2
        assert [f["media_id"] for f in body["photos"][0]["faces"]] == [1, 4, 7, 8]

    def test_an_unknown_person_or_a_stale_partition_is_refused(self, client, faces):
        partition = faces.listing(client)["partition"]
        assert faces.get(client, f"/api/people/1/with/9?partition={partition}").status_code == 404
        stale = faces.get(client, "/api/people/1/with/2?partition=0000000000000000")
        assert (stale.status_code, stale.get_json()["error_code"]) == (409, "people_changed")
