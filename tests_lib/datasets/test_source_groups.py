"""Grouping a ranking of converter outputs by source item: Collapse to photos (#4751).

The pure seam (:func:`group_by_source`) with a stub locator, and once against
live contexts through :class:`~vtscore.datasets.source_item.SourceLocator`.
"""

from __future__ import annotations

from typing import Any

import pytest

from vtscore.datasets.registry import register_dataset
from vtscore.datasets.source_groups import SourceGroup, SourceNotLoadedError, group_by_source
from vtscore.datasets.source_item import (
    NO_SIBLING,
    NOT_DERIVED,
    NOT_FOUND,
    RESOLVED,
    SIBLING_NOT_LOADED,
    SourceLocation,
    SourceLocator,
)
from vtscore.state.core import DatasetContext, register_context


def _stub_locator(table: dict[int, SourceLocation]):
    """A locator answering from *table* by media id; unknown ids are not derived."""

    def locate(media: dict[str, Any]) -> SourceLocation:
        return table.get(media["id"], SourceLocation(NOT_DERIVED))

    return locate


def _at(photo: int, box: list[float] | None = None) -> SourceLocation:
    return SourceLocation(RESOLVED, dataset_id="img", media_id=photo, box=box)


def _medias(*ids: int) -> dict[int, dict[str, Any]]:
    return {i: {"id": i} for i in ids}


def _shape(groups: list[SourceGroup]) -> list[tuple[int | None, float, list[int]]]:
    return [(g.source_media_id, g.score, [m.media_id for m in g.members]) for g in groups]


class TestGroupBySource:
    def test_faces_of_one_photo_become_one_row_scored_as_the_best(self):
        """``{1A, 1B, 2C, 2D}`` ranked A, C, B, D → photos 1, 2."""
        ranking = [
            {"id": 10, "score": 0.9},
            {"id": 20, "score": 0.8},
            {"id": 11, "score": 0.7},
            {"id": 21, "score": 0.1},
        ]
        locate = _stub_locator({10: _at(1), 11: _at(1), 20: _at(2), 21: _at(2)})
        groups = group_by_source(ranking, _medias(10, 11, 20, 21), locate)
        assert _shape(groups) == [(1, 0.9, [10, 11]), (2, 0.8, [20, 21])]
        assert all(g.source_dataset_id == "img" and g.resolved for g in groups)

    def test_a_photo_whose_faces_straddle_any_window_edge_is_still_one_row(self):
        """The grouping sees the whole ranking: the best and worst face of photo 1 share a row."""
        ranking = [{"id": 1, "score": 0.9}] + [{"id": 100 + i, "score": 0.5 - i / 100} for i in range(10)]
        ranking.append({"id": 2, "score": 0.0})
        table = {1: _at(7), 2: _at(7)} | {100 + i: _at(1000 + i) for i in range(10)}
        groups = group_by_source(ranking, _medias(*(r["id"] for r in ranking)), _stub_locator(table))
        assert len(groups) == 11
        assert [m.media_id for m in groups[0].members] == [1, 2]

    def test_every_output_lands_in_exactly_one_row(self):
        ranking = [{"id": i, "score": 1 - i / 10} for i in range(8)]
        table = {0: _at(1), 2: _at(1), 3: SourceLocation(NOT_FOUND, dataset_id="img"), 5: _at(2), 6: _at(1)}
        groups = group_by_source(ranking, _medias(*range(8)), _stub_locator(table))
        members = [m.media_id for g in groups for m in g.members]
        assert sorted(members) == list(range(8))

    def test_an_output_with_no_nameable_source_is_a_row_of_its_own(self):
        ranking = [{"id": 1, "score": 0.9}, {"id": 2, "score": 0.8}, {"id": 3, "score": 0.7}, {"id": 4, "score": 0.6}]
        table = {
            1: SourceLocation(NOT_FOUND, dataset_id="img", box=[0.1, 0.1, 0.2, 0.2]),
            2: SourceLocation(NO_SIBLING),
            4: _at(5),
        }
        groups = group_by_source(ranking, _medias(1, 2, 3, 4), _stub_locator(table))
        assert _shape(groups) == [(None, 0.9, [1]), (None, 0.8, [2]), (None, 0.7, [3]), (5, 0.6, [4])]
        assert groups[0].source_dataset_id is None and not groups[0].resolved
        assert groups[0].members[0].box == [0.1, 0.1, 0.2, 0.2]

    def test_a_row_missing_from_the_medias_is_its_own_row(self):
        groups = group_by_source([{"id": 9, "score": 0.5}], {}, _stub_locator({9: _at(1)}))
        assert _shape(groups) == [(None, 0.5, [9])]

    def test_boxes_ride_on_the_members(self):
        ranking = [{"id": 1, "score": 0.9}, {"id": 2, "score": 0.4}]
        table = {1: _at(1, [0.0, 0.0, 0.5, 0.5]), 2: _at(1, [0.5, 0.5, 1.0, 1.0])}
        (group,) = group_by_source(ranking, _medias(1, 2), _stub_locator(table))
        assert [m.box for m in group.members] == [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]]

    def test_similarity_rows_group_too(self):
        """Text and example sorts rank by ``similarity``, not ``score``."""
        ranking = [{"id": 1, "similarity": 0.3}, {"id": 2, "similarity": 0.2}]
        (group,) = group_by_source(ranking, _medias(1, 2), _stub_locator({1: _at(4), 2: _at(4)}))
        assert (group.score, [m.score for m in group.members]) == (0.3, [0.3, 0.2])

    def test_an_unloaded_sibling_stops_the_grouping_naming_it(self):
        ranking = [{"id": 1, "score": 0.9}, {"id": 2, "score": 0.5}]
        table = {1: _at(1), 2: SourceLocation(SIBLING_NOT_LOADED, dataset_id="img")}
        with pytest.raises(SourceNotLoadedError) as exc:
            group_by_source(ranking, _medias(1, 2), _stub_locator(table))
        assert exc.value.dataset_id == "img"

    def test_an_empty_ranking_has_no_rows(self):
        assert group_by_source([], {}, _stub_locator({})) == []


# ---------------------------------------------------------------------------
# Against live contexts
# ---------------------------------------------------------------------------


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


def _register(name: str, media_type: str) -> dict[str, Any]:
    return register_dataset(
        name=name,
        media_type=media_type,
        num_items=2,
        pkl_path=f"/tmp/{name}.pkl",
        import_group="g1",
        output_category=media_type,
    )


def test_groups_live_faces_by_their_photos_in_the_image_sibling():
    faces_entry = _register("t – Face", "face")
    images_entry = _register("t – Image", "image")
    faces = {
        1: _face(1, "a.jpg", [0.1, 0.1, 0.2, 0.2]),
        2: _face(2, "b.jpg", [0.3, 0.3, 0.4, 0.4]),
        3: _face(3, "a.jpg", [0.6, 0.6, 0.8, 0.8]),
        4: _face(4, "gone.jpg", [0.0, 0.0, 0.1, 0.1]),
    }
    face_ctx = DatasetContext(faces_entry["id"])
    face_ctx.medias.update(faces)
    register_context(face_ctx)
    image_ctx = DatasetContext(images_entry["id"])
    image_ctx.medias.update({10: _photo(10, "a.jpg"), 11: _photo(11, "b.jpg")})
    register_context(image_ctx)

    ranking = [{"id": 2, "score": 0.9}, {"id": 1, "score": 0.8}, {"id": 4, "score": 0.5}, {"id": 3, "score": 0.2}]
    groups = group_by_source(ranking, faces, SourceLocator(faces_entry["id"]).locate)

    assert [(g.source_dataset_id, g.source_media_id) for g in groups] == [
        (images_entry["id"], 11),
        (images_entry["id"], 10),
        (None, None),
    ]
    assert [m.media_id for m in groups[1].members] == [1, 3]
    assert groups[1].members[1].box == [0.6, 0.6, 0.8, 0.8]
