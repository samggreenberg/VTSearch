"""Build the photo manifest for the Celebrities in Context demo.

The demo (``celebrity_together_s`` / ``_m``) downloads a few thousand loose
photos from the Celebrity Together HuggingFace mirror rather than the 11 GB
archive, so it has to know *which* photos in advance.  This script pins them:
for every person in
:data:`~vtscore.media.image._demo_categories.CELEBRITY_TOGETHER_CATEGORIES` it
picks ``--per-person`` photos where that person's face is labelled, at least
``--min-face`` pixels on its short side, and present on the mirror (about 58% of
the dataset's photos are; the rest exist only inside the archive).

Picks are deterministic: each person's candidates are taken in filename order
(the filenames are MD5 hex, so that order is a fixed shuffle), and people take
turns, one photo per turn, so a photo holding two of the demo's people goes to
whichever reaches it first and never appears twice.  The order within a person
is the order the demo slices in, so (S)'s photos are the first of (M)'s.

Usage::

    python scripts/build_celebrity_together_manifest.py            # rewrite the shipped manifest
    python scripts/build_celebrity_together_manifest.py --check    # fail if it would change

Output: ``vtscore/media/assets/celebrity_together.json``.  Needs network access
to huggingface.co (the 33 MB label file plus ~100 paged listing calls).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from vtscore.datasets.downloader.core import CELEBRITY_TOGETHER_REPO_ID  # noqa: E402
from vtscore.media.image._demo_categories import CELEBRITY_TOGETHER_CATEGORIES  # noqa: E402

_ASSET_PATH = _REPO / "vtscore" / "media" / "assets" / "celebrity_together.json"
#: The dataset's own label file: every image's name, and every face's box and identity.
_LABELS_FILE = "imdb_v1.1.mat"


def _labelled_faces(mat_path: Path) -> tuple[list[str], list[str], list[tuple[int, int, float]]]:
    """Return the image names, the class names, and one ``(image_id, label, short_side)`` per labelled face.

    ``image_id`` and ``label`` are the file's own 1-based indices into the two
    name lists (label 0 is an unnamed face, and is dropped); ``short_side`` is
    the face box's smaller edge in pixels.
    """
    import scipy.io as sio  # noqa: PLC0415

    ct = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)["imdb"].ct
    names = [str(n) for n in ct.images.name]
    classes = [str(c) for c in ct.meta.classes]
    det = ct.faces.det  # (4, n_faces): x1, y1, x2, y2
    faces = []
    for i, (image_id, label) in enumerate(zip(ct.faces.imageID, ct.faces.label)):
        if int(label) > 0:
            short = float(min(det[2, i] - det[0, i], det[3, i] - det[1, i]))
            faces.append((int(image_id), int(label), short))
    return names, classes, faces


def _mirror_files() -> set[str]:
    """Every photo the mirror serves as a loose file, as ``n<id>/<md5>.jpg``."""
    from huggingface_hub import HfApi  # noqa: PLC0415

    tree = HfApi().list_repo_tree(CELEBRITY_TOGETHER_REPO_ID, repo_type="dataset", path_in_repo="images", recursive=True)
    return {e.path.removeprefix("images/") for e in tree if e.path.endswith(".jpg") and hasattr(e, "size")}


def build(per_person: int, min_face: float) -> dict[str, list[str]]:
    """Pick *per_person* photos for every demo person (module docstring)."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    mat_path = Path(
        hf_hub_download(CELEBRITY_TOGETHER_REPO_ID, _LABELS_FILE, repo_type="dataset")
    )
    names, classes, faces = _labelled_faces(mat_path)
    on_mirror = _mirror_files()

    wanted = {classes.index(person) + 1: person for person in CELEBRITY_TOGETHER_CATEGORIES}
    candidates: dict[str, set[str]] = defaultdict(set)
    for image_id, label, short in faces:
        name = names[image_id - 1]
        if label in wanted and short >= min_face and name in on_mirror:
            candidates[wanted[label]].add(name)
    queues = {person: sorted(candidates[person]) for person in CELEBRITY_TOGETHER_CATEGORIES}

    taken: set[str] = set()
    picks: dict[str, list[str]] = {person: [] for person in CELEBRITY_TOGETHER_CATEGORIES}
    for _turn in range(per_person):
        for person in CELEBRITY_TOGETHER_CATEGORIES:
            queue = queues[person]
            while queue and queue[0] in taken:
                queue.pop(0)
            if not queue:
                raise SystemExit(f"{person} has only {len(picks[person])} usable photos; need {per_person}")
            photo = queue.pop(0)
            taken.add(photo)
            picks[person].append(photo)
    return picks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--per-person", type=int, default=60, help="photos per person (the largest variant's count)")
    ap.add_argument("--min-face", type=float, default=48.0, help="smallest face box edge, in pixels")
    ap.add_argument("--check", action="store_true", help="exit 1 if the shipped manifest differs, writing nothing")
    args = ap.parse_args()

    manifest = {
        "repo_id": CELEBRITY_TOGETHER_REPO_ID,
        "photos": build(args.per_person, args.min_face),
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
