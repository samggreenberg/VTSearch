"""The ``fhibe`` loader's decisions, on a synthetic release small enough to reason about (#4699).

What needs pinning is what the loader decides rather than what it reuses:

* **Two-person photos are left out**, as Sony recommends for face verification:
  one person can appear in them under a second subject id.
* **Ids come from FHIBE's uid**, so a consent-revocation release that drops a
  subject leaves every other photo's id where it was.
* **Crop labels are assigned by IoU, not inherited.** The subject's crop takes
  the photo's id; a face elsewhere is kept, labelled for nobody; a box that
  half-overlaps the subject is dropped rather than scored as a negative.
* **A face embedder pairs only with face datasets**, and every other embedder
  only with image datasets.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import pytest

_PILE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"


@pytest.fixture(scope="module")
def pc():
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    import pile_config

    return pile_config


@pytest.fixture
def mod(pc):
    pytest.importorskip("PIL")
    from pilebuild.loaders import fhibe

    fhibe._PHOTOS.clear()
    yield fhibe
    fhibe._PHOTOS.clear()


#: (uid, subject, humans, face [x, y, w, h]) on a 200 x 150 frame.
_ROWS = [
    ("aaaaaaaa-0000-0000-0000-000000000001", "subj-a", 1, [20, 30, 40, 40]),
    ("bbbbbbbb-0000-0000-0000-000000000002", "subj-a", 1, [100, 50, 40, 40]),
    ("cccccccc-0000-0000-0000-000000000003", "subj-b", 1, [60, 60, 40, 40]),
    ("dddddddd-0000-0000-0000-000000000004", "subj-b", 2, [10, 10, 40, 40]),  # two-person
]


def _png(w: int = 200, h: int = 150) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), (90, 110, 130)).save(buf, format="PNG")
    return buf.getvalue()


def _release(tmp_path: Path, rows=_ROWS) -> Path:
    """A release laid out the way the MDC archive nests it."""
    data = tmp_path / "fhibe-test" / "fhibe.2026.u.X_public_fullres" / "data" / "raw" / "fhibe_fullres"
    data.mkdir(parents=True)
    with (data / "filepaths.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["uid", "img", "json"])
        for uid, subj, humans, box in rows:
            d = data / subj / uid
            d.mkdir(parents=True)
            (d / f"main_{uid}.png").write_bytes(_png())
            subs = [{"subject_id": subj, "is_primary": True, "face_bbox": box}]
            if humans == 2:
                subs.append({"subject_id": "subj-z", "is_primary": False, "face_bbox": [150, 20, 30, 30]})
            ann = {"image_annotation": {"humans_per_image": humans, "image_width": 200, "image_height": 150}}
            (d / f"main_annos_{uid}.json").write_text(json.dumps({**ann, "subject_annotation": subs}))
            wr.writerow([uid, f"{subj}/{uid}/main_{uid}.png", f"{subj}/{uid}/main_annos_{uid}.json"])
    return data


@pytest.fixture
def release(tmp_path, pc, monkeypatch):
    data = _release(tmp_path)
    monkeypatch.setattr(pc, "FHIBE_ROOT", tmp_path)
    monkeypatch.setattr(pc, "FHIBE_RELEASE", "fhibe-test")
    return data


class TestPhotos:
    def test_two_person_photos_are_left_out(self, mod, release):
        photos = mod.read_photos(release)
        assert [p.uid[:8] for p in photos] == ["aaaaaaaa", "bbbbbbbb", "cccccccc"]

    def test_the_face_box_is_converted_from_xywh(self, mod, release):
        p = mod.read_photos(release)[0]
        assert p.face == (20.0, 30.0, 60.0, 70.0)

    def test_ids_survive_a_release_that_drops_a_subject(self, mod, tmp_path):
        before = {p.uid: p.pid for p in mod.read_photos(_release(tmp_path / "v1"))}
        mod._PHOTOS.clear()
        after = {p.uid: p.pid for p in mod.read_photos(_release(tmp_path / "v2", rows=_ROWS[1:]))}
        assert after == {uid: pid for uid, pid in before.items() if uid in after}

    def test_the_release_is_found_however_deep_it_nests(self, mod, release):
        assert mod.data_dir() == release


class TestPhotoArm:
    def test_one_media_per_one_person_photo(self, mod, pc, release, monkeypatch):
        monkeypatch.setitem(pc.DATASETS, "fhibe_t", {"boxed": True, "kind": "fhibe", "long_side": 100})
        medias: dict[int, dict] = {}
        mod.load("fhibe_t", medias, "siglip")
        assert len(medias) == 3
        m = medias[mod.photo_id(_ROWS[0][0])]
        assert m["media_type"] == "image"
        assert (m["category"], m["categories"]) == ("subj-a", ["subj-a"])
        assert m["regions"] == [{"box": [0.1, 0.2, 0.3, pytest.approx(70 / 150)], "label": "subj-a"}]
        assert max(m["width"], m["height"]) == 100

    def test_the_build_label_check_passes(self, mod, pc, release, monkeypatch):
        from pilebuild.audit import label_problems

        monkeypatch.setitem(pc.DATASETS, "fhibe_t", {"boxed": True, "kind": "fhibe", "long_side": 100})
        medias: dict[int, dict] = {}
        mod.load("fhibe_t", medias, "siglip")
        assert label_problems("fhibe_t", medias) == []

    def test_staged_copies_are_reused_and_owner_only(self, mod, pc, release):
        photos = mod.read_photos(release)
        out = mod.stage(release, photos, 100, workers=1)
        assert (out.stat().st_mode & 0o077) == 0
        first = {f.name: f.stat().st_mtime_ns for f in out.iterdir()}
        mod.stage(release, photos, 100, workers=1)
        assert {f.name: f.stat().st_mtime_ns for f in out.iterdir()} == first


class _Detector:
    """Stands in for MTCNN: fixed raw boxes and confidences, in the stored copy's pixels."""

    def __init__(self, boxes, probs):
        self.boxes, self.probs = boxes, probs

    def detect(self, img):
        return self.boxes, self.probs


class TestFaceLabels:
    @pytest.fixture
    def converter(self):
        pytest.importorskip("vtscore")
        from vtscore.converters.image2face import CONVERTER

        return CONVERTER

    def _photo(self, mod):
        return mod.Photo(7, "u", "x.png", "subj-a", 200, 150, (20.0, 30.0, 60.0, 70.0))

    def test_subject_elsewhere_and_ambiguous(self, mod, converter, monkeypatch):
        from PIL import Image

        boxes = [
            [21, 31, 61, 71],  # the subject: IoU ~0.9
            [40, 30, 80, 70],  # half on the subject's face: IoU ~0.33 -> dropped
            [130, 60, 170, 100],  # nowhere near: kept, labelled for nobody
        ]
        monkeypatch.setattr(converter, "_make_detector", lambda: _Detector(boxes, [0.99, 0.9, 0.8]))
        subject, extras, dropped = mod.face_crops(converter, Image.new("RGB", (200, 150)), self._photo(mod), 1.0)
        assert subject is not None
        assert len(extras) == 1 and dropped == 1

    def test_a_missed_subject_yields_no_subject_crop(self, mod, converter, monkeypatch):
        from PIL import Image

        monkeypatch.setattr(converter, "_make_detector", lambda: _Detector([[130, 60, 170, 100]], [0.95]))
        subject, extras, dropped = mod.face_crops(converter, Image.new("RGB", (200, 150)), self._photo(mod), 1.0)
        assert subject is None and len(extras) == 1 and dropped == 0

    def test_below_the_app_threshold_is_not_a_detection(self, mod, converter, monkeypatch):
        from PIL import Image

        monkeypatch.setattr(converter, "_make_detector", lambda: _Detector([[21, 31, 61, 71]], [0.4]))
        subject, extras, _ = mod.face_crops(converter, Image.new("RGB", (200, 150)), self._photo(mod), 1.0)
        assert subject is None and extras == []

    def test_the_annotation_is_scaled_to_the_stored_copy(self, mod, converter, monkeypatch):
        from PIL import Image

        # The stored copy is half size, so the subject's raw box is half the annotation.
        # (80 px face -> 40 px stored: padded to 60, clear of image2face's 32 px floor.)
        p = mod.Photo(7, "u", "x.png", "subj-a", 200, 150, (20.0, 30.0, 100.0, 110.0))
        monkeypatch.setattr(converter, "_make_detector", lambda: _Detector([[10, 15, 50, 55]], [0.99]))
        subject, _, _ = mod.face_crops(converter, Image.new("RGB", (100, 75)), p, 0.5)
        assert subject is not None

    def test_a_face_under_the_app_floor_is_no_crop(self, mod, converter, monkeypatch):
        from PIL import Image

        # 20 px stored, 30 px padded: image2face drops it, so the census counts it a miss and so must this.
        monkeypatch.setattr(converter, "_make_detector", lambda: _Detector([[10, 15, 30, 35]], [0.99]))
        subject, extras, _ = mod.face_crops(converter, Image.new("RGB", (100, 75)), self._photo(mod), 0.5)
        assert subject is None and extras == []

    def test_crop_ids_pair_with_photo_ids(self, mod, pc, release, converter, monkeypatch):
        monkeypatch.setitem(
            pc.DATASETS, "fhibe_faces_t", {"boxed": False, "kind": "fhibe", "long_side": 200, "media_type": "face"}
        )
        photos = {p.uid: p for p in mod.read_photos(release)}
        order = sorted(photos.values(), key=lambda p: p.pid)

        class _EachPhoto:
            """Finds the annotated face of the photo being read, plus one stray face at the edge."""

            def detect(self, img):
                p = order.pop(0)  # the loader reads photos in id order
                return [list(p.face), [150, 100, 190, 140]], [0.99, 0.9]

        det = _EachPhoto()
        monkeypatch.setattr(converter, "_make_detector", lambda: det)
        medias: dict[int, dict] = {}
        mod.load("fhibe_faces_t", medias, "face")
        subjects = {k: m for k, m in medias.items() if not k & mod.EXTRA_BIT}
        extras = {k: m for k, m in medias.items() if k & mod.EXTRA_BIT}
        assert set(subjects) == {p.pid for p in photos.values()}
        assert all(m["media_type"] == "face" and m["categories"] == [m["category"]] for m in subjects.values())
        assert len(extras) == 3 and all(m["categories"] == [] for m in extras.values())
        assert all(m["source_photo_id"] in subjects for m in extras.values())


class TestPairing:
    def test_face_embedder_embeds_only_face_datasets(self, pc):
        pairs = pc.cells()
        assert ("fhibe_faces_1024", "face") in pairs and ("fhibe_1024", "siglip") in pairs
        assert ("fhibe_faces_1024", "siglip") not in pairs
        assert ("fhibe_1024", "face") not in pairs and ("coco_better", "face") not in pairs

    def test_every_non_face_dataset_keeps_every_image_embedder(self, pc):
        image_embedders = [e for e in pc.EMBEDDERS if e != "face"]
        for ds in pc.DATASETS:
            if pc.media_type_of(ds) == "image":
                assert all((ds, e) in pc.cells() for e in image_embedders), ds


class TestPrivateCells:
    """FHIBE's cells live owner-only beside the release; the world-readable pile holds links."""

    def test_the_pile_name_links_into_the_private_dir(self, pc, tmp_path, monkeypatch):
        import build_pile

        monkeypatch.setattr(pc, "EMBEDDINGS", tmp_path / "pile")
        monkeypatch.setattr(pc, "FHIBE_ROOT", tmp_path / "fhibe")
        (tmp_path / "pile").mkdir()
        build_pile._link_private("fhibe_1024", "siglip")
        link = pc.cell_path("fhibe_1024", "siglip")
        assert link.is_symlink() and not link.exists()  # dangling until written
        link.write_bytes(b"cell")
        home = pc.private_cell_dir("fhibe_1024")
        assert (home / link.name).read_bytes() == b"cell"
        assert (home.stat().st_mode & 0o077) == 0
        build_pile._link_private("fhibe_1024", "siglip")  # idempotent

    def test_a_real_file_in_the_way_is_refused(self, pc, tmp_path, monkeypatch):
        import build_pile

        monkeypatch.setattr(pc, "EMBEDDINGS", tmp_path / "pile")
        monkeypatch.setattr(pc, "FHIBE_ROOT", tmp_path / "fhibe")
        (tmp_path / "pile").mkdir()
        pc.cell_path("fhibe_640", "siglip").write_bytes(b"old")
        with pytest.raises(SystemExit):
            build_pile._link_private("fhibe_640", "siglip")

    def test_other_datasets_stay_in_the_pile(self, pc):
        assert pc.private_cell_dir("coco_better") is None
