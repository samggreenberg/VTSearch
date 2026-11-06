"""The user guide's face example agrees with itself.

The guide's *Faces* shots are taken on the face example
(`scripts/screenshots/face-example.mjs`, #5605): 80 Celebrities in Context
photos that `scripts/screenshots/face_fixture.py` fetches and pins in
`face-slice.json`, imported as Image + Face on InsightFace's buffalo_s models,
with a face detector trained on pinned votes. The example lives in three
files, in two languages, and the harness only finds a disagreement at the next
refresh, when it fails a shot:

- The slice is what `face_fixture.py` says it is (the first `PHOTOS_PER_PERSON`
  of each of `PEOPLE`'s demo photos), pinned by a well-formed hash, and the
  photos keep distinct names in the one flat folder they are copied into.
- Every face the JavaScript names (the votes, the face the Train shot serves)
  and the shared photo the People shot opens is cut from a photo in the slice.
- The pipeline config names exactly the models whose hashes are pinned.
- The pack carries the face ball the app reads beside its config (#5621): the
  shipped buffalo_s entry, the same floors the test suite pins.

So they are checked here, without a browser, an app or a download.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "scripts" / "screenshots" / "face-example.mjs"
FIXTURE = REPO / "scripts" / "screenshots" / "face_fixture.py"
SLICE = REPO / "scripts" / "screenshots" / "face-slice.json"
DEMO_MANIFEST = REPO / "vtscore" / "media" / "assets" / "celebrity_together.json"

#: A face's file name as the image2face converter writes it.
FACE_NAME = re.compile(r"([0-9a-f]{32})_face_\d+\.png")


def _fixture() -> ModuleType:
    spec = importlib.util.spec_from_file_location("face_fixture", FIXTURE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _slice() -> dict:
    return json.loads(SLICE.read_text(encoding="utf-8"))


def _named_faces() -> list[str]:
    """Every face file name the example's JavaScript names."""
    return FACE_NAME.findall(EXAMPLE.read_text(encoding="utf-8"))


def _const(name: str) -> str:
    match = re.search(rf"export const {name} = '([^']*)';", EXAMPLE.read_text(encoding="utf-8"))
    assert match, f"face-example.mjs no longer declares {name} as a string"
    return match.group(1)


class TestTheSlice:
    def test_is_the_first_photos_of_each_person(self):
        fixture = _fixture()
        demo = json.loads(DEMO_MANIFEST.read_text(encoding="utf-8"))["photos"]
        pinned = [(p["person"], p["path"]) for p in _slice()["photos"]]
        wanted = [(person, path) for person in fixture.PEOPLE for path in demo[person][: fixture.PHOTOS_PER_PERSON]]
        assert pinned == wanted, (
            "face-slice.json is not the first PHOTOS_PER_PERSON demo photos of each of PEOPLE; "
            "re-pin with `python scripts/screenshots/face_fixture.py pin` and re-take the face shots"
        )

    def test_every_photo_is_pinned_by_a_sha256(self):
        for photo in _slice()["photos"]:
            assert re.fullmatch(r"[0-9a-f]{64}", photo["sha256"]), photo

    def test_names_stay_distinct_in_one_folder(self):
        names = [Path(p["path"]).name for p in _slice()["photos"]]
        assert len(names) == len(set(names)), "two pinned photos share a file name, and the photo folder is flat"


class TestTheShots:
    def test_every_named_face_is_cut_from_a_pinned_photo(self):
        photos = {Path(p["path"]).stem for p in _slice()["photos"]}
        faces = _named_faces()
        assert len(faces) >= 17, "face-example.mjs names fewer faces than its votes and its shown face"
        missing = sorted({stem for stem in faces if stem not in photos})
        assert not missing, f"faces cut from photos outside face-slice.json: {missing}"

    def test_the_shared_photo_is_pinned(self):
        names = {Path(p["path"]).name for p in _slice()["photos"]}
        assert _const("SHARED_PHOTO") in names


class TestTheModels:
    def test_config_names_exactly_the_pinned_models(self):
        fixture = _fixture()
        config = fixture.FACE_CONFIG
        assert {config["detector"]["model_bundle"], config["embedder"]["model"]} == set(fixture.MODEL_SHA256)
        assert config["embedder"]["config"] == fixture.EMBEDDER_CONFIG.name
        assert fixture.EMBEDDER_CONFIG.is_file()

    def test_the_space_is_the_calibrated_one(self):
        """The People cut is fitted in buffalo_s's space (#5497's practice entry), so the shots need these models."""
        from tests_shared.face_ball import COMPAT

        fixture = _fixture()
        compat = json.loads(fixture.EMBEDDER_CONFIG.read_text(encoding="utf-8"))["model"]["compatibility"]
        assert compat == COMPAT == fixture.FACE_COMPAT

    def test_the_pack_carries_the_shipped_ball(self, tmp_path, monkeypatch):
        """An app pointed at the pack finds buffalo_s's ball beside its config, so People has people (#5621).

        The floors are the ones the suite pins (`tests_shared.face_ball`), #5497's shipped entry, so
        the shots show the deployed cut and neither copy can drift without the other.
        """
        from tests_shared.face_ball import CENTRE, COMPAT, FLOORS, SLOPE
        from vtscore.media.face import face_ball
        from vtscore.media.face.imagekit_bridge import CONFIG_ENV_VAR

        config = _fixture().write_pack_configs(tmp_path / "imagekit-buffalo_s")
        assert config.is_file() and (config.parent / "arcface_mbf.json").is_file()
        monkeypatch.setenv(CONFIG_ENV_VAR, str(config))
        face_ball.reset_for_tests()
        assert face_ball.face_ball_path() == config.parent / face_ball.FACE_BALL_NAME
        ball = face_ball.face_ball_for(COMPAT)
        assert ball is not None and ball.use
        assert ball.betas == {beta: (floor, SLOPE) for beta, floor in FLOORS.items()}
        assert ball.centre == CENTRE
