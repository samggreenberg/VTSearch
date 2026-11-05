"""The photo index: which faces share a photo, and which people appear together (#5587).

Plain medias dicts, built the way the converter runner and duplicate collapse
leave them, and once against :func:`resolve_source_item` to show that a photo
here is one item of the Image sibling.
"""

from __future__ import annotations

from typing import Any

import pytest

from vtscore.datasets.co_occurrence import CoAppearance, PhotoIndex, SharedPhoto, faces_by_photo, photo_keys
from vtscore.datasets.source_item import resolve_source_item

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _photo(mid: int, name: str, path: str = "") -> dict[str, Any]:
    """A photo of the Image sibling, as the folder loader builds it."""
    return {
        "id": mid,
        "media_type": "image",
        "md5": f"md5-{name}",
        "filename": name,
        "origin": {"importer": "server_folder", "params": {"path": "/data/photos"}},
        "origin_name": name,
        "media_path": path or f"/data/photos/{name}",
    }


def _face(mid: int, source_file: str, source_path: str | None = None, index: int = 0) -> dict[str, Any]:
    """Face *index* of *source_file*, as the converter runner emits it.

    *source_path* defaults to the file under ``/data/photos``; ``""`` records
    none, as a ref written before the runner kept the resolved path.
    """
    path = f"/data/photos/{source_file}" if source_path is None else source_path
    params: dict[str, Any] = {"converter": "image2face", "source_file": source_file, "converter_out_index": str(index)}
    if path:
        params["source_path"] = path
    name = f"{source_file}→face_{index}.jpg"
    return {
        "id": mid,
        "media_type": "face",
        "md5": f"face-md5-{mid}",
        "filename": name,
        "origin": {"importer": "converter", "params": params},
        "origin_name": name,
        "media_path": path,
    }


def _dupe_set(rep: dict[str, Any], *members: dict[str, Any]) -> dict[str, Any]:
    """*rep* collapsed over *members* (itself first), as ``collapse_duplicates`` leaves it."""
    out = dict(rep)
    out["origin"] = {
        "importer": "dupe_set",
        "params": {"name": rep["origin_name"]},
        "members": [
            {
                "md5": m["md5"],
                "origin": m["origin"],
                "origin_name": m["origin_name"],
                "filename": m["filename"],
                "category": "custom",
            }
            for m in (rep, *members)
        ],
    }
    return out


def _plain(mid: int) -> dict[str, Any]:
    """A directly-imported face crop: no converter ref, so no photo."""
    return {
        "id": mid,
        "media_type": "face",
        "md5": f"plain-md5-{mid}",
        "filename": f"crop_{mid}.jpg",
        "origin": {"importer": "server_folder", "params": {"path": "/data/crops"}},
        "origin_name": f"crop_{mid}.jpg",
    }


def _medias(*items: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {m["id"]: m for m in items}


# ---------------------------------------------------------------------------
# Photo keys and the index
# ---------------------------------------------------------------------------


class TestPhotoKeys:
    def test_source_path_is_the_key(self):
        assert photo_keys(_face(1, "trip/a.jpg")) == [("path", "/data/photos/trip/a.jpg")]

    def test_source_file_stands_in_when_no_path_is_recorded(self):
        assert photo_keys(_face(1, "trip/a.jpg", source_path="")) == [("file", "trip/a.jpg")]

    def test_a_dupe_set_carries_a_key_per_converter_member(self):
        face = _dupe_set(_face(1, "a.jpg"), _face(2, "copy-of-a.jpg"), _plain(3))
        assert photo_keys(face) == [("path", "/data/photos/a.jpg"), ("path", "/data/photos/copy-of-a.jpg")]

    def test_a_media_that_is_not_derived_has_none(self):
        assert photo_keys(_plain(1)) == []
        assert photo_keys({"id": 2, "origin": None}) == []
        assert photo_keys({"id": 3, "origin": {"importer": "converter", "params": {"converter": "image2face"}}}) == []


class TestPhotoIndex:
    def test_one_photo_with_three_faces(self):
        index = PhotoIndex(_medias(_face(1, "a.jpg", index=0), _face(2, "a.jpg", index=1), _face(3, "a.jpg", index=2)))
        assert len(index) == 1
        assert index.faces_by_photo == {0: (1, 2, 3)}
        assert [index.photo_of(i) for i in (1, 2, 3)] == [0, 0, 0]

    def test_photos_are_numbered_by_their_lowest_face_id(self):
        medias = _medias(_face(7, "b.jpg"), _face(3, "a.jpg"), _face(5, "b.jpg", index=1), _face(9, "c.jpg"))
        assert faces_by_photo(medias) == {0: (3,), 1: (5, 7), 2: (9,)}

    def test_a_face_with_no_converter_ref_is_in_no_photo(self):
        index = PhotoIndex(_medias(_face(1, "a.jpg"), _plain(2)))
        assert index.faces_by_photo == {0: (1,)}
        assert index.photo_of(2) is None
        assert index.photos_of([2]) == []
        assert index.photo_of(99) is None

    def test_names_that_collide_across_folders_are_two_photos(self):
        """``b.jpg`` in two folders (a Manifest staging dir, say) is two photos, told apart by path."""
        medias = _medias(
            _face(1, "b.jpg", source_path="/staging/one/b.jpg"),
            _face(2, "b.jpg", source_path="/staging/two/b.jpg"),
            _face(3, "b.jpg", source_path="/staging/two/b.jpg", index=1),
        )
        assert faces_by_photo(medias) == {0: (1,), 1: (2, 3)}

    def test_a_collapsed_face_joins_the_copies_of_its_photo(self):
        """Face 3 names only the copy; the collapsed face 1 carries both, so all three are one photo."""
        medias = _medias(
            _dupe_set(_face(1, "a.jpg"), _face(10, "copy-of-a.jpg")),
            _dupe_set(_face(2, "a.jpg", index=1), _face(11, "copy-of-a.jpg", index=1)),
            _face(3, "copy-of-a.jpg", index=2),
            _face(4, "b.jpg"),
        )
        assert faces_by_photo(medias) == {0: (1, 2, 3), 1: (4,)}

    def test_faces_in_names_every_face_and_refuses_an_unknown_photo(self):
        index = PhotoIndex(_medias(_face(1, "a.jpg"), _face(2, "a.jpg", index=1)))
        assert index.faces_in(0) == (1, 2)
        with pytest.raises(IndexError):
            index.faces_in(1)
        with pytest.raises(IndexError):
            index.faces_in(-1)

    def test_faces_by_photo_is_a_copy(self):
        index = PhotoIndex(_medias(_face(1, "a.jpg")))
        index.faces_by_photo.clear()
        assert index.faces_by_photo == {0: (1,)}

    def test_an_empty_dataset_has_no_photos(self):
        index = PhotoIndex({})
        assert len(index) == 0
        assert index.faces_by_photo == {}


# ---------------------------------------------------------------------------
# Agreement with SourceResolver
# ---------------------------------------------------------------------------


class TestAgreesWithSourceResolver:
    """Every face of one photo resolves to one sibling item, and two photos to two items."""

    @staticmethod
    def _items_per_photo(faces: dict[int, dict[str, Any]], sibling: dict[int, dict[str, Any]]) -> dict[int, set]:
        index = PhotoIndex(faces)
        return {
            photo: {resolve_source_item(faces[face_id], sibling) for face_id in face_ids}
            for photo, face_ids in index.faces_by_photo.items()
        }

    def test_a_duplicate_collapsed_photo(self):
        # The folder held a.jpg and a byte-identical copy-of-a.jpg, and b.jpg.  The
        # Image sibling collapsed the copies into one item named a.jpg ...
        sibling = _medias(_dupe_set(_photo(1, "a.jpg"), _photo(2, "copy-of-a.jpg")), _photo(3, "b.jpg"))
        # ... and the Face dataset collapsed their identical faces.  Face 13 is a
        # face of the copy whose twin from a.jpg was since removed.
        faces = _medias(
            _dupe_set(_face(11, "a.jpg"), _face(21, "copy-of-a.jpg")),
            _dupe_set(_face(12, "a.jpg", index=1), _face(22, "copy-of-a.jpg", index=1)),
            _face(13, "copy-of-a.jpg", index=2),
            _face(14, "b.jpg"),
        )
        assert self._items_per_photo(faces, sibling) == {0: {1}, 1: {3}}

    def test_names_that_collide_across_folders(self):
        sibling = _medias(_photo(1, "b.jpg", path="/staging/one/b.jpg"), _photo(2, "b.jpg", path="/staging/two/b.jpg"))
        faces = _medias(
            _face(11, "b.jpg", source_path="/staging/one/b.jpg"),
            _face(12, "b.jpg", source_path="/staging/two/b.jpg"),
            _face(13, "b.jpg", source_path="/staging/one/b.jpg", index=1),
        )
        assert self._items_per_photo(faces, sibling) == {0: {1}, 1: {2}}


# ---------------------------------------------------------------------------
# People: photos_of, appears_with, shared_photos
# ---------------------------------------------------------------------------


def _three_people_one_photo() -> PhotoIndex:
    """Photo 0 holds faces 1, 2 and 3; photo 1 holds face 4 alone."""
    return PhotoIndex(
        _medias(_face(1, "a.jpg"), _face(2, "a.jpg", index=1), _face(3, "a.jpg", index=2), _face(4, "b.jpg"))
    )


class TestPhotosOf:
    def test_each_photo_once_ascending(self):
        index = _three_people_one_photo()
        assert index.photos_of([4, 1, 2]) == [0, 1]
        assert index.photos_of([]) == []

    def test_ids_with_no_photo_are_ignored(self):
        index = PhotoIndex(_medias(_face(1, "a.jpg"), _plain(2)))
        assert index.photos_of([1, 2, 99]) == [0]


class TestAppearsWith:
    def test_three_people_in_one_photo(self):
        index = _three_people_one_photo()
        groups = [[1], [2], [3, 4]]
        assert index.appears_with(groups, 0) == [CoAppearance(1, (0,)), CoAppearance(2, (0,))]
        assert index.appears_with(groups, 2) == [CoAppearance(0, (0,)), CoAppearance(1, (0,))]

    def test_most_shared_first_ties_in_group_order(self):
        # Person 0 is in photos 0-3.  Group 1 shares one, group 2 two, group 3
        # two, group 4 none.
        medias = _medias(
            _face(1, "p0.jpg"),
            _face(2, "p1.jpg"),
            _face(3, "p2.jpg"),
            _face(4, "p3.jpg"),
            _face(5, "p0.jpg", index=1),  # group 1
            _face(6, "p1.jpg", index=1),  # group 2
            _face(7, "p2.jpg", index=1),  # group 2
            _face(8, "p2.jpg", index=2),  # group 3
            _face(9, "p3.jpg", index=1),  # group 3
            _face(10, "elsewhere.jpg"),  # group 4
        )
        index = PhotoIndex(medias)
        groups = [[1, 2, 3, 4], [5], [6, 7], [8, 9], [10]]
        found = index.appears_with(groups, 0)
        assert [(co.group, co.shared) for co in found] == [(2, 2), (3, 2), (1, 1)]
        assert found[0].photos == (1, 2)

    def test_two_faces_in_one_shared_photo_count_it_once(self):
        index = _three_people_one_photo()
        assert index.appears_with([[1], [2, 3]], 0) == [CoAppearance(1, (0,))]

    def test_a_person_in_no_photo_appears_with_nobody(self):
        index = PhotoIndex(_medias(_face(1, "a.jpg"), _plain(2)))
        assert index.appears_with([[2], [1]], 0) == []

    def test_a_person_out_of_range_is_refused(self):
        index = _three_people_one_photo()
        with pytest.raises(IndexError):
            index.appears_with([[1]], 1)
        with pytest.raises(IndexError):
            index.appears_with([[1]], -1)


class TestSharedPhotos:
    def test_every_face_in_the_shared_photo_is_listed(self):
        index = _three_people_one_photo()
        assert index.shared_photos([1], [2, 4]) == [SharedPhoto(0, (1, 2, 3))]

    def test_people_who_share_nothing(self):
        index = _three_people_one_photo()
        assert index.shared_photos([1], [4]) == []
        assert index.shared_photos([], [1]) == []

    def test_several_shared_photos_ascending(self):
        medias = _medias(
            _face(1, "b.jpg"),
            _face(2, "b.jpg", index=1),
            _face(3, "a.jpg"),
            _face(4, "a.jpg", index=1),
            _face(5, "a.jpg", index=2),
        )
        index = PhotoIndex(medias)
        assert index.shared_photos([1, 3], [2, 4]) == [SharedPhoto(0, (1, 2)), SharedPhoto(1, (3, 4, 5))]
