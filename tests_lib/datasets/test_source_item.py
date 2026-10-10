"""Resolving a converter output to its source item in a sibling dataset (#4749).

The pure seam (:func:`resolve_source_item`) against plain medias dicts, the
sibling choice (:func:`find_source_sibling`) against registry rows, and the
live locator (:func:`locate_source`) against registered contexts.
"""

from __future__ import annotations

from typing import Any

from vtscore.datasets import source_item
from vtscore.datasets.registry import register_dataset
from vtscore.datasets.source_item import (
    SourceResolver,
    converter_refs,
    find_source_sibling,
    locate_source,
    resolve_source_item,
    source_type_of,
)
from vtscore.state.core import DatasetContext, register_context

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _photo(mid: int, name: str, path: str = "") -> dict[str, Any]:
    """A directly-imported photo, as the folder loader builds it."""
    return {
        "id": mid,
        "media_type": "image",
        "md5": f"md5-{mid}",
        "filename": name,
        "origin": {"importer": "server_folder", "params": {"path": "/data/photos"}},
        "origin_name": name,
        "media_path": path or f"/data/photos/{name}",
    }


def _converter_origin(source_file: str, source_path: str = "", converter: str = "image2face") -> dict[str, Any]:
    return {
        "importer": "converter",
        "params": {
            "converter": converter,
            "source_file": source_file,
            "source_path": source_path or f"/data/photos/{source_file}",
            "converter_out_index": "0",
        },
    }


def _face(mid: int, source_file: str, source_path: str = "", box: list[float] | None = None) -> dict[str, Any]:
    """A face crop, as the converter runner emits it."""
    media: dict[str, Any] = {
        "id": mid,
        "media_type": "face",
        "md5": f"face-md5-{mid}",
        "filename": f"{source_file}→face_0.jpg",
        "origin": _converter_origin(source_file, source_path),
        "origin_name": f"{source_file}→face_0.jpg",
        "media_path": source_path or f"/data/photos/{source_file}",
    }
    if box is not None:
        media["source_box"] = box
    return media


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


def _medias(*items: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {m["id"]: m for m in items}


# ---------------------------------------------------------------------------
# resolve_source_item
# ---------------------------------------------------------------------------


class TestResolveSourceItem:
    def test_source_file_joins_the_siblings_origin_name(self):
        photos = _medias(_photo(1, "a.jpg"), _photo(2, "trip/b.jpg"))
        assert resolve_source_item(_face(9, "trip/b.jpg"), photos) == 2

    def test_name_collision_is_settled_by_source_path(self):
        photos = _medias(
            _photo(1, "b.jpg", path="/staging/one/b.jpg"),
            _photo(2, "b.jpg", path="/staging/two/b.jpg"),
        )
        assert resolve_source_item(_face(9, "b.jpg", source_path="/staging/two/b.jpg"), photos) == 2
        assert resolve_source_item(_face(9, "b.jpg", source_path="/staging/one/b.jpg"), photos) == 1

    def test_collision_the_path_cannot_settle_is_no_answer(self):
        """Two photos share the name and neither is on the face's path: refuse, never guess."""
        photos = _medias(
            _photo(1, "b.jpg", path="/staging/one/b.jpg"),
            _photo(2, "b.jpg", path="/staging/two/b.jpg"),
        )
        assert resolve_source_item(_face(9, "b.jpg", source_path="/elsewhere/b.jpg"), photos) is None

    def test_a_media_that_is_not_derived_has_no_source(self):
        photos = _medias(_photo(1, "a.jpg"))
        assert resolve_source_item(_photo(5, "a.jpg"), photos) is None
        assert resolve_source_item({"id": 6, "origin": None}, photos) is None

    def test_no_matching_name_is_no_answer(self):
        assert resolve_source_item(_face(9, "gone.jpg"), _medias(_photo(1, "a.jpg"))) is None

    def test_a_collapsed_face_resolves_through_its_members(self):
        """Two identical photos make two identical faces; collapse leaves a dupe_set face."""
        face = _dupe_set(_face(9, "a.jpg"), _face(10, "copy-of-a.jpg"))
        assert converter_refs(face)[0]["source_file"] == "a.jpg"
        photos = _medias(_photo(1, "a.jpg"))
        assert resolve_source_item(face, photos) == 1

    def test_a_photo_collapsed_under_another_name_is_found_by_member_name(self):
        """The face names ``b.jpg``; the sibling folded ``b.jpg`` into the ``a.jpg`` representative."""
        photos = _medias(_dupe_set(_photo(1, "a.jpg"), _photo(2, "b.jpg")), _photo(3, "c.jpg"))
        assert resolve_source_item(_face(9, "b.jpg"), photos) == 1

    def test_a_supplied_name_lookup_is_used(self):
        photos = _medias(_photo(1, "a.jpg"), _photo(2, "b.jpg"))
        # A lookup that points "a.jpg" at 2 proves the resolver read it rather than rebuilding.
        assert resolve_source_item(_face(9, "a.jpg"), photos, name_lookup={"a.jpg": [2]}) == 2

    def test_a_lookup_id_missing_from_the_medias_is_ignored(self):
        photos = _medias(_photo(1, "a.jpg"))
        assert resolve_source_item(_face(9, "a.jpg"), photos, name_lookup={"a.jpg": [1, 77]}) == 1

    def test_one_resolver_serves_many_outputs(self):
        resolver = SourceResolver(_medias(_photo(1, "a.jpg"), _photo(2, "b.jpg")))
        assert [resolver.resolve(_face(i, n)) for i, n in ((9, "b.jpg"), (10, "a.jpg"), (11, "x.jpg"))] == [2, 1, None]


class TestSourceTypeOf:
    def test_face_crops_come_from_images(self):
        assert source_type_of(_face(9, "a.jpg")) == "image"

    def test_a_video_frame_comes_from_a_video(self):
        frame = {"id": 1, "origin": _converter_origin("clip.mp4", converter="video2image")}
        assert source_type_of(frame) == "video"

    def test_unknown_converter_and_plain_media_have_none(self):
        assert source_type_of({"id": 1, "origin": _converter_origin("a.jpg", converter="no_such_converter")}) is None
        assert source_type_of(_photo(1, "a.jpg")) is None


# ---------------------------------------------------------------------------
# find_source_sibling
# ---------------------------------------------------------------------------


def _row(rid: str, media_type: str, category: str, group: str | None = "g1", **kw: Any) -> dict[str, Any]:
    return {"id": rid, "media_type": media_type, "output_category": category, "import_group": group, **kw}


def _sibling_id(dataset_id: str, source_type: str, **kw: Any) -> str:
    sibling = find_source_sibling(dataset_id, source_type, **kw)
    assert sibling is not None
    return sibling["id"]


class TestFindSourceSibling:
    def test_the_category_picks_the_image_output_not_the_document_output(self):
        """Both are image datasets; only the Image output holds the photos."""
        rows = [
            _row("doc", "image", "document"),
            _row("img", "image", "image"),
            _row("face", "face", "face"),
        ]
        assert _sibling_id("face", "image", entries=rows) == "img"

    def test_another_groups_image_dataset_is_not_a_sibling(self):
        rows = [_row("img", "image", "image", group="other"), _row("face", "face", "face")]
        assert find_source_sibling("face", "image", entries=rows) is None

    def test_no_group_means_no_sibling(self):
        rows = [_row("img", "image", "", group=None), _row("face", "face", "", group=None)]
        assert find_source_sibling("face", "image", entries=rows) is None

    def test_a_dataset_is_never_its_own_sibling(self):
        """A rendered page's source category is its own dataset's: it has no source item."""
        rows = [_row("doc", "image", "document"), _row("img", "image", "image")]
        assert find_source_sibling("doc", "document", entries=rows) is None

    def test_unknown_dataset_has_no_sibling(self):
        assert find_source_sibling("nope", "image", entries=[_row("img", "image", "image")]) is None

    def test_readable_by_confines_to_readable_siblings(self):
        face = register_dataset(
            name="t – Face",
            media_type="face",
            num_items=1,
            pkl_path="/tmp/f.pkl",
            created_by="alice",
            import_group="g",
            output_category="face",
        )
        img = register_dataset(
            name="t – Image",
            media_type="image",
            num_items=1,
            pkl_path="/tmp/i.pkl",
            created_by="alice",
            import_group="g",
            output_category="image",
        )
        assert _sibling_id(face["id"], "image", readable_by="alice") == img["id"]
        assert find_source_sibling(face["id"], "image", readable_by="bob") is None
        assert _sibling_id(face["id"], "image") == img["id"]


# ---------------------------------------------------------------------------
# locate_source: live contexts
# ---------------------------------------------------------------------------


def _register_group(*, load_images: bool = True) -> tuple[dict[str, Any], dict[str, Any], DatasetContext]:
    """Register a Face + Image pair of one import; load the faces, and the photos if asked."""
    face_entry = register_dataset(
        name="t – Face",
        media_type="face",
        num_items=1,
        pkl_path="/tmp/f.pkl",
        import_group="g",
        output_category="face",
    )
    img_entry = register_dataset(
        name="t – Image",
        media_type="image",
        num_items=2,
        pkl_path="/tmp/i.pkl",
        import_group="g",
        output_category="image",
    )
    face_ctx = DatasetContext(face_entry["id"])
    face_ctx.medias.update(_medias(_face(1, "b.jpg", box=[0.1, 0.2, 0.3, 0.4]), _face(2, "gone.jpg")))
    register_context(face_ctx)
    if load_images:
        img_ctx = DatasetContext(img_entry["id"])
        img_ctx.medias.update(_medias(_photo(1, "a.jpg"), _photo(2, "b.jpg")))
        register_context(img_ctx)
    return face_entry, img_entry, face_ctx


class TestLocateSource:
    def test_resolves_into_the_loaded_sibling_with_the_box(self):
        face_entry, img_entry, face_ctx = _register_group()
        found = locate_source(face_ctx.medias[1], face_entry["id"])
        assert found == source_item.SourceLocation(
            source_item.RESOLVED, dataset_id=img_entry["id"], media_id=2, box=[0.1, 0.2, 0.3, 0.4]
        )

    def test_an_unloaded_sibling_is_named(self):
        face_entry, img_entry, face_ctx = _register_group(load_images=False)
        found = locate_source(face_ctx.medias[1], face_entry["id"])
        assert found.status == source_item.SIBLING_NOT_LOADED
        assert found.dataset_id == img_entry["id"]
        assert found.media_id is None

    def test_no_match_in_a_loaded_sibling(self):
        face_entry, img_entry, face_ctx = _register_group()
        found = locate_source(face_ctx.medias[2], face_entry["id"])
        assert (found.status, found.dataset_id, found.box) == (source_item.NOT_FOUND, img_entry["id"], None)

    def test_an_unreadable_sibling_reads_as_none(self):
        face_entry, _img_entry, face_ctx = _register_group()
        found = locate_source(face_ctx.medias[1], face_entry["id"], user="mallory")
        assert found.status == source_item.NO_SIBLING
        assert found.dataset_id is None

    def test_a_plain_media_is_not_derived(self):
        face_entry, _img_entry, _face_ctx = _register_group()
        assert locate_source(_photo(5, "a.jpg"), face_entry["id"]).status == source_item.NOT_DERIVED

    def test_one_locator_resolves_many(self):
        face_entry, img_entry, face_ctx = _register_group()
        locator = source_item.SourceLocator(face_entry["id"])
        found = [locator.locate(m) for m in face_ctx.medias.values()]
        assert [(f.status, f.media_id) for f in found] == [(source_item.RESOLVED, 2), (source_item.NOT_FOUND, None)]


# ---------------------------------------------------------------------------
# End to end: the real folder loader and converter runner agree on the join
# ---------------------------------------------------------------------------


class TestJoinThroughTheRealLoaders:
    """One folder, scanned once for photos and once for faces, as a multi-dataset import does.

    Pins the premise the seam rests on: the runner's ``source_file`` is the
    folder loader's ``origin_name`` (both scan-relative), and its
    ``source_path`` is the loader's ``media_path`` (both resolved).
    """

    def test_every_face_resolves_to_its_own_photo(self, tmp_path):
        import io
        from unittest.mock import MagicMock, patch

        from PIL import Image

        from vtscore.converters.image2face import CONVERTER
        from vtscore.converters.runner import run_converters_on_folder
        from vtscore.datasets.loader_folder import load_dataset_from_folder

        names = ["a.png", "trip/b.png", "trip/c.png"]
        for shade, name in enumerate(names):
            buf = io.BytesIO()
            Image.new("RGB", (100, 100), (40 * shade, 90, 90)).save(buf, format="PNG")
            (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / name).write_bytes(buf.getvalue())

        photos: dict[int, dict[str, Any]] = {}
        load_dataset_from_folder(tmp_path, "image", photos)
        faces: dict[int, dict[str, Any]] = {}
        detector = MagicMock()
        detector.detect.return_value = ([[10, 10, 60, 60]], [0.9])
        with patch.object(CONVERTER, "_make_detector", return_value=detector):
            run_converters_on_folder(
                tmp_path,
                target_media_type="face",
                medias=faces,
                converter_specs=[{"converter": "image2face", "params": {"padding": "0", "min_size": "1"}}],
            )

        assert len(faces) == len(names)
        photo_by_name = {m["origin_name"]: mid for mid, m in photos.items()}
        resolver = SourceResolver(photos)
        for face in faces.values():
            params = face["origin"]["params"]
            photo_id = photo_by_name[params["source_file"]]
            assert params["source_path"] == photos[photo_id]["media_path"], "the path tiebreak compares like with like"
            assert resolver.resolve(face) == photo_id
