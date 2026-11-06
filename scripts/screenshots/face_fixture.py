#!/usr/bin/env python
"""The face path's worked example: a slice of Celebrities in Context, and the face models it needs.

    python scripts/screenshots/face_fixture.py models    # -> prints the face pipeline config's path
    python scripts/screenshots/face_fixture.py photos    # -> prints the photo folder's path
    python scripts/screenshots/face_fixture.py pin       # rewrite face-slice.json (maintainers only)

The user guide's *Faces* shots (People, Show in photo, Collapse to photos) are
taken on these photos, beside the Smiley example every other shot uses
(`face-example.mjs`). Cartoon drawings have no faces a face detector finds, so
this example is real photographs, and it needs real face models.

**The photos** are a slice of the Celebrities in Context demo (Celebrity
Together, Zhong, Arandjelovic & Zisserman, ACCV 2018, from the Hugging Face
mirror in `vtscore/datasets/downloader/core.py`): the first 20 pinned photos of
each of `PEOPLE`, 80 in all. Those four were chosen because their photos share
people (two films' casts): Andrew Garfield and Emma Stone share about a dozen,
and Jamie Foxx shares some with each, so the People view has a person who
appears with others. They are fetched with the demo's own downloader into its
own cache (`data/celebrity_together/images/`), so loading the demo later costs
nothing for them, then copied into `data/doc-fixtures/celebrities/`, the
folder the harness imports. It is flat, the photos under their mirror names:
the app captions a face with its photo's path, and a folder per person would
put that person's name under everyone else in their photos. `face-slice.json` pins each photo by its SHA-256,
and a photo that no longer matches fails the run, naming the photo.

**The models** are InsightFace's buffalo_s pack (the v0.7 model-zoo release,
`BUFFALO_S_URL`): `det_500m.onnx` (SCRFD-500M) and `w600k_mbf.onnx`
(MobileFaceNet ArcFace), the space VTSearch's face ball and People cut are
calibrated in (`oss-arcface-mbf:1`). The app never downloads face models
(`docs/DEPLOYMENT.md`, "Face models"), so this script does, once, into
`<models>/imagekit-buffalo_s/` (the models folder is `VTSEARCH_MODELS_DIR`,
`data/models/` by default) with a `face.json` beside them; `refresh.sh` points
the app it starts at that file (`IMAGEKIT_FACE_CONFIG`). The zip and both
models are checked against `BUFFALO_S_SHA256` / `MODEL_SHA256`. The weights are
for **non-commercial research only** (InsightFace's licence), which is why they
are fetched here and never committed.

**The face ball** goes beside them too, as `face_ball.json`: the app reads the
face search line's floors, and the People cut, beside the pipeline config in
use (`vtscore/media/face/face_ball.py`), and a space with no entry has no
people (#5621). The file is a calculated asset, fitted on the GRID from FHIBE
and never in the repo, so buffalo_s's entry is pinned here (`FACE_BALL_ENTRY`)
the way the models are: the floors #5497 fitted for `oss-arcface-mbf:1`, as its
report records them and the GRID's live file holds them. That is the shipped
cut, so the face shots show what a deployment shows; the test suite pins the
same numbers (`tests_shared/face_ball.py`), and `tests_lib/meta/test_face_example.py`
holds the two to each other.

Every failure is loud and names what is missing: there is no quiet fallback in
which the face shots are skipped, since a fixture that skips when its models are
absent would let the shots rot unnoticed.

Idempotent: a pack whose models match their hashes and a folder whose photos
match theirs are left alone. Nothing is written into the repo except by `pin`.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

HERE = Path(__file__).resolve().parent
SLICE_FILE = HERE / "face-slice.json"
PHOTOS = _REPO_ROOT / "data" / "doc-fixtures" / "celebrities"
#: The photo folder's fingerprint, kept beside it so the import never sees it.
PHOTOS_STAMP = PHOTOS.with_suffix(".fingerprint")

#: The people whose pinned demo photos make up the slice, and how many of each.
PEOPLE = ["Andrew Garfield", "Emma Stone", "Jamie Foxx", "Kerry Washington"]
PHOTOS_PER_PERSON = 20

#: InsightFace's buffalo_s pack: the model-zoo release VTSearch's face calibration was made on.
BUFFALO_S_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"
BUFFALO_S_SHA256 = "d85a87f503f691807cd8bb97128bdf7a0660326cd9cd02657127fa978bab8b5e"
#: The two files of the pack the face path runs, by SHA-256 (the pack's three others are not used).
MODEL_SHA256 = {
    "det_500m.onnx": "5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a",
    "w600k_mbf.onnx": "9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f",
}
#: The embedder config `w600k_mbf.onnx` is described by, shipped in the imagekit package.
EMBEDDER_CONFIG = _REPO_ROOT / "imagekit" / "face" / "extraction" / "arcface_mbf.json"
#: The pipeline config written beside the models (paths resolve against its directory).
FACE_CONFIG = {
    "provider": "oss",
    "detector": {"model_bundle": "det_500m.onnx"},
    "embedder": {"model": "w600k_mbf.onnx", "config": "arcface_mbf.json"},
}
#: The face space `w600k_mbf.onnx` embeds into (the `compatibility` in `arcface_mbf.json`).
FACE_COMPAT = "oss-arcface-mbf:1"
#: buffalo_s's face ball, as `face_ball.json` stores an entry: the floors #5497 fitted on FHIBE
#: (`docs/experiments/2026-11-06-face-ball-per-embedder-5497`, its report's "The fit reproduces what
#: shipped"), which are the numbers #5478 shipped. `check` carries the report's go/no-go readings;
#: `provenance` says where this copy came from. Pinned like the models: the real file is a
#: calculated asset on the GRID, and the harness must give the same cut on every machine.
FACE_BALL_ENTRY = {
    "use": True,
    "betas": {"1": {"floor": 0.4425, "slope": 0.25}, "4": {"floor": 0.385, "slope": 0.25}},
    "centre": 0.6371,
    "check": {"use": True, "gain": {"1": -0.003, "4": 0.055}, "gain_se": {"1": 0.001, "4": 0.001}, "clean_gap": 0.84},
    "provenance": {
        "corpus": "FHIBE faces",
        "commit": "c534adabf",
        "tool": "scripts/data/fit_face_ball.py, pinned by scripts/screenshots/face_fixture.py",
        "report": "docs/experiments/2026-11-06-face-ball-per-embedder-5497",
    },
}


class FixtureError(RuntimeError):
    """Something the face fixture needs is missing or wrong; the message says what."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _note(message: str) -> None:
    print(f"[face fixture] {message}", file=sys.stderr)


# ----------------------------------------------------------------------
# The models
# ----------------------------------------------------------------------


def pack_dir() -> Path:
    """Where the buffalo_s pack goes: `imagekit-buffalo_s/` in the app's models folder."""
    from vtscore.config import paths  # noqa: PLC0415

    return Path(paths.MODELS_CACHE_DIR).expanduser() / "imagekit-buffalo_s"


def _models_current(folder: Path) -> list[str]:
    """What is missing or wrong in *folder*'s pack, one line each (empty when it is current)."""
    problems = []
    for name, want in MODEL_SHA256.items():
        path = folder / name
        if not path.is_file():
            problems.append(f"{path} is missing")
        elif _sha256(path) != want:
            problems.append(f"{path} is not the pinned file (SHA-256 {want})")
    return problems


def _fetch_pack(folder: Path) -> None:
    """Download buffalo_s.zip, check it, and extract the two models the face path runs."""
    from vtscore.datasets.downloader.core import download_file_atomic  # noqa: PLC0415

    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "buffalo_s.zip"
    _note(f"downloading the buffalo_s face models (~128 MB) from {BUFFALO_S_URL} ...")
    try:
        download_file_atomic(BUFFALO_S_URL, archive, 0, lambda *_args: None)
    except Exception as exc:
        raise FixtureError(
            f"could not download the buffalo_s face models from {BUFFALO_S_URL}: {str(exc).rstrip('.')}. "
            f"Fetch the zip by hand, check its SHA-256 is {BUFFALO_S_SHA256}, and put "
            f"{', '.join(MODEL_SHA256)} in {folder}."
        ) from exc
    try:
        got = _sha256(archive)
        if got != BUFFALO_S_SHA256:
            raise FixtureError(
                f"{BUFFALO_S_URL} is not the pinned buffalo_s.zip: SHA-256 {got}, expected {BUFFALO_S_SHA256}. "
                "The release may have changed; the face shots are calibrated on the pinned models only."
            )
        with zipfile.ZipFile(archive) as zf:
            for name in MODEL_SHA256:
                with zf.open(name) as src, open(folder / name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    finally:
        archive.unlink(missing_ok=True)


def write_pack_configs(folder: Path) -> Path:
    """Write the pack's configs into *folder*: the embedder's, the pipeline's and the face ball's.

    Returns the pipeline config's path, the one `refresh.sh` names in `IMAGEKIT_FACE_CONFIG`.
    The ball's file goes beside it because that is where the app reads it
    (:func:`vtscore.media.face.face_ball.face_ball_path`), so the pack is complete on its own,
    whatever the machine's `imagekit/` directory holds. The entry is read back through the app's
    own parser first, so a malformed pin fails the run instead of leaving People empty.
    """
    from vtscore.media.face.face_ball import FACE_BALL_NAME, FILE_VERSION, FaceBall  # noqa: PLC0415

    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(EMBEDDER_CONFIG, folder / EMBEDDER_CONFIG.name)
    config = folder / "face.json"
    config.write_text(json.dumps(FACE_CONFIG, indent=2) + "\n")
    try:
        ball = FaceBall.from_json(FACE_COMPAT, FACE_BALL_ENTRY)
    except ValueError as exc:
        raise FixtureError(f"the pinned buffalo_s face ball entry is malformed: {exc}") from exc
    if not ball.use:
        raise FixtureError("the pinned buffalo_s face ball entry is no-go, and the face shots need the ball")
    file = {"version": FILE_VERSION, "spaces": {FACE_COMPAT: ball.to_json()}}
    (folder / FACE_BALL_NAME).write_text(json.dumps(file, indent=1) + "\n")
    return config


def ensure_models() -> Path:
    """The buffalo_s pack with its `face.json` and `face_ball.json`, fetched if missing; returns the config's path."""
    folder = pack_dir()
    if _models_current(folder):
        _fetch_pack(folder)
    problems = _models_current(folder)
    if problems:
        raise FixtureError("the buffalo_s face models are not in place: " + "; ".join(problems))
    return write_pack_configs(folder)


# ----------------------------------------------------------------------
# The photos
# ----------------------------------------------------------------------


def _pinned() -> dict:
    return json.loads(SLICE_FILE.read_text(encoding="utf-8"))


def fingerprint() -> str:
    """What decides the photo folder: every pinned photo and its hash, in order, and the folder's layout."""
    digest = hashlib.sha256(b"layout: flat\n")
    for photo in _pinned()["photos"]:
        digest.update(f"{photo['person']}/{photo['path']}:{photo['sha256']}\n".encode())
    return digest.hexdigest()[:16]


def ensure_photos() -> Path:
    """Fetch the pinned slice (through the demo's downloader and cache) into `PHOTOS`; return it."""
    from vtscore.datasets.downloader import download_celebrity_together  # noqa: PLC0415

    photos = _pinned()["photos"]
    stamp = fingerprint()
    if PHOTOS_STAMP.is_file() and PHOTOS_STAMP.read_text().strip() == stamp:
        return PHOTOS
    _note(f"fetching {len(photos)} Celebrities in Context photos ...")
    try:
        images = download_celebrity_together([p["path"] for p in photos])
    except Exception as exc:
        raise FixtureError(f"could not download the Celebrities in Context photos: {str(exc).rstrip('.')}.") from exc
    if PHOTOS.exists():
        shutil.rmtree(PHOTOS)
    for photo in photos:
        source = images / photo["path"]
        got = _sha256(source)
        if got != photo["sha256"]:
            raise FixtureError(
                f"{source} ({photo['person']}) is not the pinned photo: SHA-256 {got}, expected {photo['sha256']}. "
                "Delete it and run again to fetch it afresh; if the mirror changed, re-pin with "
                "`face_fixture.py pin` and re-take the face shots."
            )
        # One flat folder: the app labels an item by its path, so a folder per
        # person would caption every face with the name of whoever its photo
        # was pinned for, the people beside them included.
        PHOTOS.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, PHOTOS / Path(photo["path"]).name)
    PHOTOS_STAMP.write_text(stamp)
    return PHOTOS


def pin() -> Path:
    """Rewrite `face-slice.json` from the demo's manifest and the photos on disk (maintainers only)."""
    from vtscore.datasets.downloader import download_celebrity_together  # noqa: PLC0415
    from vtscore.media.image.demo._demo_sources import celebrity_together_photos  # noqa: PLC0415

    manifest = celebrity_together_photos()
    chosen = [(person, path) for person in PEOPLE for path in manifest[person][:PHOTOS_PER_PERSON]]
    images = download_celebrity_together([path for _person, path in chosen])
    account = {
        "source": "Celebrity Together (Zhong, Arandjelovic & Zisserman, ACCV 2018), the Celebrities in Context demo",
        "mirror": "https://huggingface.co/datasets/Juoguessr/celebritytogether",
        "people": PEOPLE,
        "photos": [{"person": person, "path": path, "sha256": _sha256(images / path)} for person, path in chosen],
    }
    SLICE_FILE.write_text(json.dumps(account, indent=1) + "\n", encoding="utf-8")
    return SLICE_FILE


def main(argv: list[str]) -> int:
    steps = {"models": ensure_models, "photos": ensure_photos, "pin": pin}
    if len(argv) != 2 or argv[1] not in steps:
        print(f"usage: {argv[0]} {{{','.join(steps)}}}", file=sys.stderr)
        return 2
    try:
        print(steps[argv[1]]())
    except FixtureError as exc:
        print(f"face fixture: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
