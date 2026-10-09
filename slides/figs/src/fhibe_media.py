"""FHIBE's photographs and face boxes, for the deck's face card, read where they live.

FHIBE, Sony AI's Fair Human-Centric Image Benchmark (Xiang et al., *Nature*
648, 2025), is the set the pile's identity cells are built from (#4699). Unlike
the other cards' sources it is never fetched here: each user registers with
Sony and downloads their own copy, which may not be passed on, so there is no
url to download from and nothing lands under `data/`. This module reads the
release the pile reads — `pile_config.FHIBE_ROOT / FHIBE_RELEASE`, the
owner-only copy on the GRID — and the card is rendered there
(`make-data-cards.py --only fhibe`), with `VTS_FHIBE_ROOT` / `VTS_FHIBE_RELEASE`
overriding the path exactly as they do for the build.

What it reads is what the supply census read and no more: `filepaths.csv`, and
from each photo's JSON the frame size, how many people are in it, and the
consenting subject's id and face box. The demographic annotations are never
opened. Non-consenting bystanders are anonymised in the pixels and carry no
annotation, so they never enter a row.

Showing FHIBE's photos on a slide is allowed: its Terms of Use (2.1(d), read
2026-10-09) let a publication about work done on it, and the talks about that
publication, carry up to twenty of them, and its subjects consented to their
identifiable images being published (the paper's Methods). The card keeps a
count against that limit (`make-data-cards.FHIBE_PHOTO_LIMIT`).
"""

from __future__ import annotations

import csv
import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Where the pile keeps the release, and which one: the same names and defaults
#: as `scripts/experiments/pile/pile_config.py`, so one export serves both.
DEFAULT_ROOT = Path(f"/expscratch/{os.environ.get('USER', '')}/fhibe")
DEFAULT_RELEASE = "fhibe-full-resolution-674a7dcf"
#: The long side the originals (12–32 MP PNGs) are brought down to before a
#: card is drawn from them: more than a slide can show, far less than a frame
#: holds. The face box is scaled with it.
WORKING_LONG_SIDE = 1600


@dataclass(frozen=True)
class Photo:
    """One one-person photo: where it is, whose it is, and where the face is."""

    uid: str
    path: Path
    subject: str
    width: int
    height: int
    face: tuple[float, float, float, float]  # x, y, w, h, in the original's pixels

    @property
    def fraction(self) -> float:
        """The face box's share of the frame, which is what the pile bands by."""
        return self.face[2] * self.face[3] / (self.width * self.height)


@dataclass(frozen=True)
class Release:
    """What one FHIBE release holds, as far as the card needs to know."""

    root: Path
    #: Every photo in `filepaths.csv`, two-person photos included.
    images: int
    #: Every consenting subject in any photo.
    subjects: frozenset[str]
    #: The one-person photos, by uid. Two-person photos are left out entirely,
    #: as the pile leaves them out: a person can appear in them under a second id.
    photos: tuple[Photo, ...]

    def by_subject(self) -> dict[str, list[Photo]]:
        out: dict[str, list[Photo]] = {}
        for photo in self.photos:
            out.setdefault(photo.subject, []).append(photo)
        return out

    def median_photos(self) -> int:
        """One-person photos per subject, at the median over subjects who have one."""
        return round(statistics.median(len(v) for v in self.by_subject().values()))

    def others(self, photo: Photo) -> list[Photo]:
        """The same subject's other one-person photos: a query's positives."""
        return [p for p in self.photos if p.subject == photo.subject and p.uid != photo.uid]


def release_dir(root: Path | None = None, release: str | None = None) -> Path:
    """The directory holding the release's `filepaths.csv`.

    The archive nests it five levels down
    (`fhibe.<date>.u.<id>_public_fullres/data/raw/fhibe_fullres`), found by
    search rather than spelled so the next release's inner name need not be
    known — the same search the pile's loader makes.
    """
    base = Path(os.environ.get("VTS_FHIBE_ROOT") or root or DEFAULT_ROOT)
    base = base / (os.environ.get("VTS_FHIBE_RELEASE") or release or DEFAULT_RELEASE)
    hits = sorted(base.glob("*/data/raw/*/filepaths.csv")) or sorted(base.rglob("filepaths.csv"))
    if not hits:
        raise SystemExit(
            f"fhibe: no filepaths.csv under {base}. FHIBE is never downloaded by a figure script: the card "
            f"is drawn where the release is (the GRID), or VTS_FHIBE_ROOT / VTS_FHIBE_RELEASE point at a copy."
        )
    return hits[0].parent


def read_release(root: Path) -> Release:
    """Every row of `filepaths.csv`, keeping the one-person photos with their face box."""
    photos: list[Photo] = []
    subjects: set[str] = set()
    images = 0
    with (root / "filepaths.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            images += 1
            annotation = json.loads((root / row["json"]).read_text())
            frame, subs = annotation["image_annotation"], annotation["subject_annotation"]
            subjects.update(s["subject_id"] for s in subs)
            if int(frame["humans_per_image"]) != 1 or len(subs) != 1:
                continue
            x, y, w, h = (float(v) for v in subs[0]["face_bbox"])  # FHIBE stores [x, y, w, h]
            photos.append(
                Photo(
                    row["uid"],
                    root / row["img"],
                    subs[0]["subject_id"],
                    int(frame["image_width"]),
                    int(frame["image_height"]),
                    (x, y, w, h),
                )
            )
    photos.sort(key=lambda p: p.uid)
    return Release(root, images, frozenset(subjects), tuple(photos))


def face_tile(
    photo: Photo, aspect: float = 4 / 3, long_side: int = WORKING_LONG_SIDE
) -> tuple[Any, tuple[float, float, float, float]]:
    """The photo cropped to `aspect` around its face, and the face box in the crop's pixels.

    `data_card.fit_tile` crops about the centre, which is right for a scene
    and wrong for a portrait-format phone photo whose face may sit in the top
    third: a centre crop would cut it off. The window here is the largest the
    frame allows at `aspect`, centred on the face as far as the frame lets it
    be and slid back inside where it runs off an edge.
    """
    from PIL import Image

    with Image.open(photo.path) as original:
        image = original.convert("RGB")
    scale = min(1.0, long_side / max(image.size))
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.BICUBIC)
    x, y, w, h = (v * scale for v in photo.face)
    width, height = image.size
    if width / height > aspect:
        crop_w, crop_h = height * aspect, float(height)
    else:
        crop_w, crop_h = float(width), width / aspect
    left = min(max(0.0, x + w / 2 - crop_w / 2), width - crop_w)
    top = min(max(0.0, y + h / 2 - crop_h / 2), height - crop_h)
    crop = image.crop((round(left), round(top), round(left + crop_w), round(top + crop_h)))
    return crop, (x - left, y - top, w, h)
