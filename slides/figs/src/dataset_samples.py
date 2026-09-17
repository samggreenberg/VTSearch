#!/usr/bin/env python
"""Real media for the deck's dataset cards, fetched on demand.

    python slides/figs/src/dataset_samples.py visual-genome   # -> prints the paths
    python slides/figs/src/dataset_samples.py caltech101
    python slides/figs/src/dataset_samples.py coco-val

A dataset card says *here is the dataset, here is where to get it, here is what
it looks like*, and the third of those is the one a drawing cannot fake. A
rectangle labelled "photograph" tells a room nothing it could not have guessed;
eight real frames tell them in one glance that Visual Genome is messy web
photography and Caltech-101 is one centred object on a plain background, which
is the whole reason the two behave differently in a study.

So every card's strip is real pixels, and this module is the one place that
knows how to get them. Three rules it works to:

* **Nothing lands in the repo.** Downloads go to `data/`, which is gitignored;
  only the composed figure is committed. The sources run from 116 KB of
  hand-picked Visual Genome frames to a 1 GB COCO zip, and a figure generator
  is not a reason to grow the tree by either.
* **Every step is idempotent.** A re-run after a layout change re-uses what is
  on disk, so iterating on a card costs no network at all.
* **The selection is pinned, not sampled at render time.** The frames are
  named here as ids, so re-running the generator redraws the same card. A card
  that reshuffled its own photographs on every run would make an innocent
  figure edit look like a data change in review.

`coco_fixture.py` already owns the COCO download for the screenshot harness,
and this module borrows it rather than fetching val2017 a second time.
"""

from __future__ import annotations

import json
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
DATA = _REPO_ROOT / "data"

# --------------------------------------------------------------------------
# Visual Genome
# --------------------------------------------------------------------------

VG_SAMPLES = DATA / "vg-samples"
VG_HOME = "https://homes.cs.washington.edu/~ranjay/visualgenome/"
#: The two historical image folders. An id lives in exactly one of them and
#: nothing published says which, so a fetch tries both — which is also what the
#: app's own downloader does when it reads a VG id off `objects.json`.
VG_IMAGE_URL = "https://cs.stanford.edu/people/rak248/{folder}/{image_id}.jpg"
VG_FOLDERS = ("VG_100K", "VG_100K_2")

#: Twelve real Visual Genome frames, by image id.
#:
#: Drawn from a uniform sample of the id space and then filtered to *scenes* —
#: a street, a lecture theatre, a lake — rather than close-up portraits. That
#: filter is not cherry-picking the dataset's quality: VG's character is that
#: one photograph carries a dozen nameable objects, and a head-and-shoulders
#: shot is the one frame in the pile that does not show it. Nothing here is
#: chosen for being clean; several are badly lit, off-centre or motion-blurred,
#: which is the other half of the point.
#:
#: Ids rather than a live sample so the card is reproducible. Any id in
#: `objects.json` is fetchable the same way; these twelve are simply the ones
#: this figure draws.
VG_SAMPLE_IDS: tuple[int, ...] = (
    2317455,  # a cement truck on a street of shopfronts
    2328140,  # a living room, staircase behind it
    2328977,  # a rowing boat on a misty lake at dawn
    2331544,  # a church spire against cloud
    2351750,  # a laptop on a cluttered desk
    2354937,  # two street signs on one post
    2365588,  # a tennis match from the baseline
    2370239,  # a zebra grazing under a tree
    2376748,  # a night street, a crowd behind a banner
    2382238,  # people playing in falling snow
    2412521,  # a red-winged blackbird on a post
    2342445,  # a plate of pasta, shot from above
)


def visual_genome_samples(ids: tuple[int, ...] = VG_SAMPLE_IDS) -> list[Path]:
    """Fetch (once) and return the sample frames, in the order given."""
    VG_SAMPLES.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for image_id in ids:
        path = VG_SAMPLES / f"{image_id}.jpg"
        if not path.exists():
            _fetch_vg(image_id, path)
        out.append(path)
    return out


def _fetch_vg(image_id: int, path: Path) -> None:
    for folder in VG_FOLDERS:
        url = VG_IMAGE_URL.format(folder=folder, image_id=image_id)
        try:
            # S310: the URL is the constant above with an integer substituted;
            # there is no scheme for a caller to choose. Same call, same
            # reason, as `coco_fixture.py`'s download.
            with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
                data = response.read()
        except Exception:  # noqa: BLE001 — a 404 in the first folder is the normal path
            continue
        # The server answers a missing id with an HTML 404 body under a 404
        # status; urllib raises on that, but a proxy that rewrites the status
        # would not, so the bytes are checked rather than the status.
        if data[:2] == b"\xff\xd8":
            path.write_bytes(data)
            return
    raise SystemExit(f"visual genome: image {image_id} is in neither {' nor '.join(VG_FOLDERS)}")


# --------------------------------------------------------------------------
# Caltech-101
# --------------------------------------------------------------------------

CALTECH_ZIP = DATA / "caltech-101.zip"
CALTECH_DIR = DATA / "caltech-101"
CALTECH_URL = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip"

#: One frame from each of twelve categories, `(category, index within it)`.
#:
#: One per category on purpose. Caltech-101's defining property is that its
#: categories are disjoint and its frames are *centred single objects*, so a
#: strip that drew several of one category would show the within-class
#: similarity and hide the thing that matters — that the whole dataset looks
#: like this, whatever the label.
#:
#: The `Faces` and `Faces_easy` categories are deliberately not among them.
#: They are ordinary members of the dataset and the slide's notes say so, but
#: they are photographs of identifiable people who sat for a computer-vision
#: paper in 2003, and a talk does not need to put one of them on a wall.
CALTECH_SAMPLES: tuple[tuple[str, int], ...] = (
    ("airplanes", 0),
    ("Leopards", 0),
    ("Motorbikes", 0),
    ("chandelier", 0),
    ("grand_piano", 0),
    ("sunflower", 0),
    ("laptop", 0),
    ("umbrella", 0),
    ("watch", 0),
    ("starfish", 0),
    ("ketch", 0),
    ("brain", 0),
)


def _checked(archive: Path, name: str, root: Path) -> None:
    """Refuse a member whose path would land outside `root`."""
    if not (root / name).resolve().is_relative_to(root):
        raise SystemExit(f"{archive.name}: member {name!r} would write outside {root}")


def _extract(archive: Path, into: Path) -> None:
    """Unpack a zip or a tar under `into`, refusing any member that escapes it.

    Member by member rather than `extractall`, and not as ceremony: an archive
    can name `../` or ship a symlink and write anywhere this process can, and
    these arrive over the network from a third party. The linter's bandit rules
    refuse the bulk call for that reason.
    """
    root = into.resolve()
    into.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                _checked(archive, info.filename, root)
                # The high bits of a zip's external attributes carry the unix
                # mode; 0xA000 there is a symlink, which extracts as a file
                # whose *contents* are a path and is the other way out of the
                # tree.
                if (info.external_attr >> 16) & 0xF000 == 0xA000:
                    raise SystemExit(f"{archive.name}: refusing symlink member {info.filename!r}")
                zf.extract(info, into)
        return
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if member.issym() or member.islnk():
                raise SystemExit(f"{archive.name}: refusing link member {member.name!r}")
            _checked(archive, member.name, root)
            tar.extract(member, into)


def caltech101_dir() -> Path:
    """The extracted `101_ObjectCategories` tree, downloading it if absent."""
    root = CALTECH_DIR / "caltech-101" / "101_ObjectCategories"
    if root.is_dir():
        return root
    if not CALTECH_ZIP.exists():
        CALTECH_ZIP.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CALTECH_URL, CALTECH_ZIP)  # noqa: S310 — constant https URL
    _extract(CALTECH_ZIP, CALTECH_DIR)
    # The distribution nests a second archive: the outer zip holds
    # `101_ObjectCategories.tar.gz` rather than the folders themselves.
    inner = next(CALTECH_DIR.rglob("101_ObjectCategories.tar.gz"), None)
    if inner is not None and not root.is_dir():
        _extract(inner, inner.parent)
    root = next(iter(CALTECH_DIR.rglob("101_ObjectCategories")), None)
    if root is None:
        raise SystemExit(f"caltech-101: no 101_ObjectCategories under {CALTECH_DIR}")
    return root


def caltech101_samples(picks: tuple[tuple[str, int], ...] = CALTECH_SAMPLES) -> list[tuple[str, Path]]:
    """`(category, path)` for each pick, downloading the source if absent."""
    root = caltech101_dir()
    out: list[tuple[str, Path]] = []
    for category, index in picks:
        frames = sorted((root / category).glob("*.jpg"))
        if not frames:
            raise SystemExit(f"caltech-101: no frames under {root / category}")
        out.append((category, frames[index % len(frames)]))
    return out


def caltech101_categories() -> list[str]:
    """Every category directory in the extracted tree, sorted."""
    return sorted(p.name for p in caltech101_dir().iterdir() if p.is_dir())


# --------------------------------------------------------------------------
# COCO val2017
# --------------------------------------------------------------------------

#: Eight val2017 frames, by COCO image id, chosen for *scenes with several
#: annotated things in them* — which is what makes COCO's exhaustive annotation
#: visible when the boxes are drawn over it. A card showing one object per
#: frame would look like Caltech-101 and say nothing about the difference.
COCO_SAMPLE_IDS: tuple[int, ...] = (
    139,  # a lounge: tv, person, book, clock, potted plant — 10 classes
    441247,  # the densest frame in val2017: 14 classes over 24 boxes
    190236,  # a desk: laptop, keyboard, mouse, book, backpack
    67616,  # a street: bicycle, car, truck, fire hydrant, parking meter
    76547,  # outdoors: bench, bicycle, traffic light, tie
    139099,  # a pavement: motorcycle, dog, bicycle, car, fruit
    494913,  # a cat on a couch, with the room annotated around it
    350148,  # a cafe table: cake, carrot, clock, car through the window
    246968,  # a kitchen: microwave, oven, remote, banana
    172595,  # a workstation: keyboard, laptop, mouse, cell phone, book
    785,  # a skier, and almost nothing else — the sparse end
    724,  # a stop sign on an empty street
)


def coco_val() -> tuple[Path, dict]:
    """`(images dir, parsed instances_val2017.json)`, downloading if absent."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import coco_fixture

    coco_fixture.ensure_corpus()
    return coco_fixture.IMAGES, json.loads(coco_fixture.ANNOTATIONS.read_text())


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which == "visual-genome":
        for path in visual_genome_samples():
            print(path)
    elif which == "caltech101":
        for category, path in caltech101_samples():
            print(f"{category}\t{path}")
    elif which == "coco-val":
        images, _ = coco_val()
        print(images)
    else:
        print("usage: dataset_samples.py {visual-genome|caltech101|coco-val}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
