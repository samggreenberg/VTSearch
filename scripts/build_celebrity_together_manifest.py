"""Build the photo manifest for the Celebrities in Context demo.

The demo (``celebrity_together_s`` / ``_m``) downloads a few thousand loose
photos from the Celebrity Together HuggingFace mirror rather than the 11 GB
archive, so it has to know *which* photos in advance.  This script pins them:
for every person in
:data:`~vtscore.media.image._demo_categories.CELEBRITY_TOGETHER_CATEGORIES` it
picks ``--per-person`` photos where

* that person's face is labelled exactly once (twice is a collage of them),
  at least ``--min-face`` pixels on its short side;
* the face is at most ``--max-face-frac`` of the photo's height, so the photo
  shows the scene around them.  Without this about a third of the picks were
  side-by-side headshot collages, the zoomed-in framing the demo exists to
  avoid.  The label file records no photo sizes, so each candidate's height is
  read from the first few KB of its JPEG;
* the photo is on the mirror as a loose file (about 58% of the dataset's photos
  are; the rest exist only inside the archive).

Picks are deterministic: each person's candidates are taken in filename order
(the filenames are MD5 hex, so that order is a fixed shuffle), and people take
turns, one photo per turn, so a photo holding two of the demo's people goes to
whichever reaches it first and never appears twice.  A photo is its filename:
the mirror files some photos under two folders (the dataset found them by two
people's searches), and such a photo is still picked once.  The order within a
person is the order the demo slices in, so (S)'s photos are the first of (M)'s.

Usage::

    python scripts/build_celebrity_together_manifest.py            # rewrite the shipped manifest
    python scripts/build_celebrity_together_manifest.py --check    # fail if it would change

Output: ``vtscore/media/assets/celebrity_together.json``.  Needs network access
to huggingface.co: the 33 MB label file, ~100 paged listing calls, and one
small ranged read per candidate photo (~6,500; ``--heights-cache`` keeps them
between runs).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from vtscore.datasets.downloader.core import (  # noqa: E402
    CELEBRITY_TOGETHER_DOWNLOAD_WORKERS,
    CELEBRITY_TOGETHER_FILE_URL,
    CELEBRITY_TOGETHER_REPO_ID,
)
from vtscore.media.image._demo_categories import CELEBRITY_TOGETHER_CATEGORIES  # noqa: E402

_ASSET_PATH = _REPO / "vtscore" / "media" / "assets" / "celebrity_together.json"
#: The dataset's own label file: every image's name, and every face's box and identity.
_LABELS_FILE = "imdb_v1.1.mat"
#: Bytes read to find a JPEG's size.  The frame header sits after any EXIF block,
#: which is usually far smaller; a photo whose header lies past it is read whole.
_HEADER_BYTES = 16384


class Face(NamedTuple):
    image_id: int  # 1-based index into the label file's image names
    label: int  # 1-based index into its class names; 0 is an unnamed face
    short_side: float  # the box's smaller edge, pixels
    height: float  # the box's height, pixels


def _labelled_faces(mat_path: Path) -> tuple[list[str], list[str], list[Face]]:
    """Return the image names, the class names, and every named face."""
    import scipy.io as sio  # noqa: PLC0415

    ct = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)["imdb"].ct
    names = [str(n) for n in ct.images.name]
    classes = [str(c) for c in ct.meta.classes]
    det = ct.faces.det  # (4, n_faces): x1, y1, x2, y2
    faces = []
    for i, (image_id, label) in enumerate(zip(ct.faces.imageID, ct.faces.label)):
        if int(label) > 0:
            w, h = float(det[2, i] - det[0, i]), float(det[3, i] - det[1, i])
            faces.append(Face(int(image_id), int(label), min(w, h), h))
    return names, classes, faces


def _mirror_files() -> set[str]:
    """Every photo the mirror serves as a loose file, as ``n<id>/<md5>.jpg``."""
    from huggingface_hub import HfApi  # noqa: PLC0415

    tree = HfApi().list_repo_tree(
        CELEBRITY_TOGETHER_REPO_ID, repo_type="dataset", path_in_repo="images", recursive=True
    )
    return {e.path.removeprefix("images/") for e in tree if e.path.endswith(".jpg") and hasattr(e, "size")}


class PhotoHeights:
    """Photo heights in pixels, read from the mirror on demand and memoised (optionally on disk)."""

    def __init__(self, cache_path: Path | None) -> None:
        self._cache_path = cache_path
        self._heights: dict[str, int] = {}
        if cache_path is not None and cache_path.is_file():
            self._heights = json.loads(cache_path.read_text())

    def __getitem__(self, photo: str) -> int:
        if photo not in self._heights:
            self.prefetch([photo])
        return self._heights[photo]

    def prefetch(self, photos: list[str]) -> None:
        """Read every unknown photo's height, several at a time."""
        todo = [p for p in dict.fromkeys(photos) if p not in self._heights]
        if not todo:
            return
        with ThreadPoolExecutor(max_workers=CELEBRITY_TOGETHER_DOWNLOAD_WORKERS) as pool:
            for photo, height in zip(todo, pool.map(self._read_height, todo)):
                self._heights[photo] = height
        if self._cache_path is not None:
            self._cache_path.write_text(json.dumps(self._heights))

    @staticmethod
    def _read_height(photo: str) -> int:
        from PIL import Image  # noqa: PLC0415

        head = _get(CELEBRITY_TOGETHER_FILE_URL.format(photo), ranged=True)
        try:
            return Image.open(io.BytesIO(head)).size[1]
        except Exception:
            return Image.open(io.BytesIO(_get(CELEBRITY_TOGETHER_FILE_URL.format(photo), ranged=False))).size[1]


def _get(url: str, *, ranged: bool) -> bytes:
    """GET *url* (its first :data:`_HEADER_BYTES` when *ranged*), waiting out rate limits."""
    import requests  # noqa: PLC0415

    headers = {"Range": f"bytes=0-{_HEADER_BYTES - 1}"} if ranged else {}
    attempts = 8
    for attempt in range(1, attempts + 1):
        response = requests.get(url, headers=headers, timeout=60)
        if response.status_code not in (429, 500, 502, 503, 504):
            response.raise_for_status()
            return response.content
        if attempt < attempts:
            time.sleep(float(response.headers.get("Retry-After") or 15 * attempt))
    raise RuntimeError(f"{url} kept answering with a retryable status after {attempts} attempts")


def _filename(photo: str) -> str:
    """A mirror path's ``<md5>.jpg``: the photo's identity, whichever folder holds it."""
    return photo.rsplit("/", 1)[-1]


def build(per_person: int, min_face: float, max_face_frac: float, heights: PhotoHeights) -> dict[str, list[str]]:
    """Pick *per_person* photos for every demo person (module docstring)."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    mat_path = Path(hf_hub_download(CELEBRITY_TOGETHER_REPO_ID, _LABELS_FILE, repo_type="dataset"))
    names, classes, faces = _labelled_faces(mat_path)
    on_mirror = _mirror_files()

    wanted = {classes.index(person) + 1: person for person in CELEBRITY_TOGETHER_CATEGORIES}
    boxes: dict[tuple[str, str], list[Face]] = defaultdict(list)
    for face in faces:
        if face.label in wanted:
            boxes[(wanted[face.label], names[face.image_id - 1])].append(face)
    candidates: dict[str, dict[str, float]] = defaultdict(dict)  # person -> {photo: face height}
    for (person, photo), found in boxes.items():
        if len(found) == 1 and found[0].short_side >= min_face and photo in on_mirror:
            candidates[person][photo] = found[0].height
    queues = {person: sorted(candidates[person], key=_filename) for person in CELEBRITY_TOGETHER_CATEGORIES}

    # Read heights ahead in bulk: most people need about twice their quota of
    # candidates, and one photo at a time would be one round trip each.
    heights.prefetch([photo for queue in queues.values() for photo in queue[: 2 * per_person]])

    taken: set[str] = set()
    picks: dict[str, list[str]] = {person: [] for person in CELEBRITY_TOGETHER_CATEGORIES}
    cursor = dict.fromkeys(CELEBRITY_TOGETHER_CATEGORIES, 0)
    for _turn in range(per_person):
        for person in CELEBRITY_TOGETHER_CATEGORIES:
            queue = queues[person]
            while True:
                if cursor[person] >= len(queue):
                    raise SystemExit(f"{person} has only {len(picks[person])} usable photos; need {per_person}")
                photo = queue[cursor[person]]
                cursor[person] += 1
                if _filename(photo) in taken:
                    continue
                heights.prefetch(queue[cursor[person] - 1 : cursor[person] + 31])
                if candidates[person][photo] <= max_face_frac * heights[photo]:
                    break
            taken.add(_filename(photo))
            picks[person].append(photo)
    return picks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--per-person", type=int, default=60, help="photos per person (the largest variant's count)")
    ap.add_argument("--min-face", type=float, default=48.0, help="smallest face box edge, in pixels")
    ap.add_argument(
        "--max-face-frac", type=float, default=0.22, help="largest face height, as a fraction of the photo's"
    )
    ap.add_argument("--heights-cache", type=Path, help="JSON file to keep photo heights in between runs")
    ap.add_argument("--check", action="store_true", help="exit 1 if the shipped manifest differs, writing nothing")
    args = ap.parse_args()

    heights = PhotoHeights(args.heights_cache)
    manifest = {
        "repo_id": CELEBRITY_TOGETHER_REPO_ID,
        "photos": build(args.per_person, args.min_face, args.max_face_frac, heights),
    }
    text = json.dumps(manifest, indent=1) + "\n"
    if args.check:
        same = _ASSET_PATH.is_file() and _ASSET_PATH.read_text() == text
        print("manifest is current" if same else "manifest differs from a fresh build")
        return 0 if same else 1
    _ASSET_PATH.write_text(text)
    n = sum(len(v) for v in manifest["photos"].values())
    print(f"wrote {n} photos for {len(manifest['photos'])} people to {_ASSET_PATH.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
