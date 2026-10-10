"""``fhibe``: Sony's FHIBE as an identity benchmark, as whole photos or as face crops (#4699).

FHIBE is 10,901 photos of 1-2 consenting, paid subjects. This builds one media
per **one-person** photo (Sony recommends leaving two-person photos out of face
verification: a person can appear there under a second subject id), with
``category`` = FHIBE's subject id. A query is one identity; its positives are
that subject's other photos, ~5 of them, and everyone else is a negative.

**The stored size is a dataset parameter, not a detail.** The supply census
(``docs/experiments/2026-10-09-fhibe-supply-4699``) found MTCNN's recall is flat
at ~95% for any face of 24 px or more and falls off a cliff below, so how small
the build stores a photo decides how hard *finding* the face is. ``long_side``
in :data:`pile_config.DATASETS` sets it; the owner chose to build 1024 and 640.
The originals (14 MB PNGs, 140 GB) are read once per size and staged as JPEG q90
beside the release (:func:`stage`), so every cell of a size reads one copy.

**Two arms per size.** ``fhibe_<size>`` is the photo, ``boxed`` with the annotated
face box as its region, for the image embedders. ``fhibe_faces_<size>`` is the
app's own ``image2face`` output on that copy, embedded with FaceNet. Crop labels
are **assigned, not inherited**: ``apply_converter_to_demo`` would stamp the
subject on every face it found, bystanders and texture false positives
included. Here a crop is matched to the annotated face box by IoU on the
detector's raw box (padding is applied after, exactly as ``image2face`` does):

* the best kept detection at IoU >= :data:`SUBJECT_IOU` is the subject's crop
  and takes the photo's id, so the photo and crop arms pair;
* a kept detection at IoU < :data:`ELSEWHERE_IOU` is someone or something else.
  It stays in the pool, labelled for nobody -- it is what the app would show;
* anything between is dropped: it may be part of the subject's own face, and
  scoring it as a negative would be wrong.

A photo whose subject MTCNN missed yields no subject crop, so the crop arm's
identity scores are conditional on finding the face. The census's per-band hit
rate is the localization stage; the two multiply to end-to-end.

**Ids survive a re-release.** A photo's id is the first 48 bits of FHIBE's
``uid``, not its position, so a consent-revocation release that drops one
subject leaves every other id unchanged. Extra crops set bit 60, above any
photo id.
"""

from __future__ import annotations

import csv
import io
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import NamedTuple

import pile_config as pc

from pilebuild.env import log

#: IoU against the annotated face box at which a crop IS the subject, and below
#: which it is unambiguously not -- the census's hit rule and its "elsewhere".
SUBJECT_IOU = 0.5
ELSEWHERE_IOU = 0.1
JPEG_QUALITY = 90
EXTRA_BIT = 1 << 60


class Photo(NamedTuple):
    """One one-person FHIBE photo, in the original's pixel space."""

    pid: int
    uid: str
    img: str  # path relative to filepaths.csv
    subject: str
    width: int
    height: int
    face: tuple[float, float, float, float]  # x0, y0, x1, y1


def data_dir(root: Path | None = None, release: str | None = None) -> Path:
    """The directory holding the release's ``filepaths.csv``.

    The archive nests it five levels down
    (``fhibe.<date>.u.<id>_public_fullres/data/raw/fhibe_fullres``); found by
    search rather than spelled, so the next release's inner name need not be known.
    """
    base = (root or pc.FHIBE_ROOT) / (release or pc.FHIBE_RELEASE)
    hits = sorted(base.glob("*/data/raw/*/filepaths.csv")) or sorted(base.rglob("filepaths.csv"))
    if not hits:
        raise SystemExit(
            f"fhibe: no filepaths.csv under {base}; download the release (scripts/experiments/pile/fhibe/)"
        )
    return hits[0].parent


def photo_id(uid: str) -> int:
    return int(uid.replace("-", "")[:12], 16)


_PHOTOS: dict[str, list[Photo]] = {}


def read_photos(root: Path) -> list[Photo]:
    """Every one-person photo, sorted by id. Two-person photos are left out entirely."""
    key = str(root)
    if key in _PHOTOS:
        return _PHOTOS[key]
    photos, two = [], 0
    for row in csv.DictReader((root / "filepaths.csv").open()):
        d = json.loads((root / row["json"]).read_text())
        ia, subs = d["image_annotation"], d["subject_annotation"]
        if int(ia["humans_per_image"]) != 1 or len(subs) != 1:
            two += 1
            continue
        x, y, w, h = (float(v) for v in subs[0]["face_bbox"])  # FHIBE stores [x, y, w, h]
        photos.append(
            Photo(
                photo_id(row["uid"]),
                row["uid"],
                row["img"],
                subs[0]["subject_id"],
                int(ia["image_width"]),
                int(ia["image_height"]),
                (x, y, x + w, y + h),
            )
        )
    photos.sort(key=lambda p: p.pid)
    ids = [p.pid for p in photos]
    if len(set(ids)) != len(ids):
        raise SystemExit("fhibe: two photos share a 48-bit uid prefix; widen photo_id")
    log(f"  fhibe: {len(photos):,} one-person photos, {two:,} two-person photos left out")
    _PHOTOS[key] = photos
    return photos


# ---------------------------------------------------------------------------
# staging the downscaled copies
# ---------------------------------------------------------------------------


def _stage_one(args: tuple[str, list[tuple[int, str]]]) -> str:
    """Decode one original ONCE and write each ``(long_side, dest)`` as JPEG q90."""
    from PIL import Image  # noqa: PLC0415

    src, outs = args
    with Image.open(src) as im:
        im = im.convert("RGB")
        for long_side, dest in outs:
            s = min(1.0, long_side / max(im.size))
            small = im.resize((round(im.width * s), round(im.height * s)), Image.BICUBIC) if s < 1.0 else im
            tmp = dest + ".tmp"
            small.save(tmp, "JPEG", quality=JPEG_QUALITY)
            os.replace(tmp, dest)
    return src


def fhibe_sizes() -> list[int]:
    """Every stored size a ``fhibe`` dataset declares, so one decode serves them all."""
    return sorted({int(d["long_side"]) for d in pc.DATASETS.values() if d.get("kind") == "fhibe"})


def stage(root: Path, photos: list[Photo], long_side: int, workers: int | None = None) -> Path:
    """Write ``<uid>.jpg`` for every photo that lacks one; return *long_side*'s directory.

    The originals are 14 MB PNGs and decoding them is most of the staging cost,
    so every size in :func:`fhibe_sizes` is written from one decode rather than
    one decode per size. Owner-only, beside the release
    (:func:`pile_config.fhibe_derived_dir`), so it is deleted with the release
    under FHIBE's revocation terms.
    """
    sizes = sorted({long_side, *fhibe_sizes()})
    dirs = {L: pc.fhibe_derived_dir(L) for L in sizes}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        d.chmod(0o700)
    todo = []
    for p in photos:
        outs = [(L, str(dirs[L] / f"{p.uid}.jpg")) for L in sizes if not (dirs[L] / f"{p.uid}.jpg").exists()]
        if outs:
            todo.append((str(root / p.img), outs))
    if todo:
        # os.cpu_count() reports the node, not the allocation. Spawned, not
        # forked: the build process already holds torch's threads.
        n = workers or max(1, len(os.sched_getaffinity(0)))
        log(f"  fhibe: staging {len(todo):,} photos at long sides {sizes} on {n} workers")
        old = os.umask(0o077)
        try:
            with ProcessPoolExecutor(n, mp_context=multiprocessing.get_context("spawn")) as pool:
                for k, _ in enumerate(pool.map(_stage_one, todo, chunksize=8), 1):
                    if k % 1000 == 0:
                        log(f"  fhibe: staged {k:,}/{len(todo):,}")
        finally:
            os.umask(old)
    return dirs[long_side]


# ---------------------------------------------------------------------------
# media
# ---------------------------------------------------------------------------


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _base(mid: int, media_type: str, data: bytes, filename: str, embedder_name: str, origin_name: str) -> dict:
    return {
        "id": mid,
        "media_type": media_type,
        "embedder": embedder_name,
        "duration": 0,
        "file_size": len(data),
        "md5": "",
        "embeddings": {},
        "media_bytes": data,
        "media_string": None,
        "filename": filename,
        "origin": {"importer": "fhibe", "params": {"embedder": embedder_name, "release": pc.FHIBE_RELEASE}},
        "origin_name": origin_name,
    }


def photo_media(p: Photo, data: bytes, size: tuple[int, int], embedder_name: str) -> dict:
    """The whole-photo media: one identity, its face box as the region."""
    x0, y0, x1, y1 = p.face
    box = [max(0.0, x0 / p.width), max(0.0, y0 / p.height), min(1.0, x1 / p.width), min(1.0, y1 / p.height)]
    return {
        **_base(p.pid, "image", data, f"{p.uid}.jpg", embedder_name, f"fhibe:{p.uid}"),
        "width": size[0],
        "height": size[1],
        "category": p.subject,
        "categories": [p.subject],
        "regions": [{"box": box, "label": p.subject}],
    }


def face_crops(converter, img, p: Photo, scale: float) -> tuple[dict | None, list[tuple[bytes, int, int]], int]:
    """``(subject_crop, extra_crops, n_dropped)`` from the app's ``image2face`` on *img*.

    Each crop is ``(png_bytes, width, height)``. Detection, threshold, padding
    and minimum size are the converter's own (its private helpers, so a change to
    the shipped rule is a change here); only the labelling is this loader's.
    """
    threshold, padding, min_size = converter._resolve_params(None)
    detector = converter._make_detector()
    if detector is None:
        raise SystemExit("fhibe: image2face has no MTCNN (facenet-pytorch missing); refusing to build an empty arm")
    w, h = img.size
    boxes, probs = detector.detect(img)
    if boxes is None:
        return None, [], 0
    face = [c * scale for c in p.face]
    scored = []
    for raw, prob in zip(boxes, probs):
        kept = converter._box_from_detection(raw, prob, w, h, threshold, padding, min_size)
        if kept is not None:
            scored.append((iou(face, [float(v) for v in raw]), kept))
    scored.sort(key=lambda t: (-t[0], -t[1][4]))

    def crop(k) -> tuple[bytes, int, int]:
        x0, y0, x1, y1, _conf = k
        buf = io.BytesIO()
        img.crop((x0, y0, x1, y1)).save(buf, format="PNG")  # image2face saves a decoded image as PNG
        return buf.getvalue(), x1 - x0, y1 - y0

    subject = None
    if scored and scored[0][0] >= SUBJECT_IOU:
        subject = crop(scored[0][1])
        scored = scored[1:]
    extras = [crop(k) for v, k in scored if v < ELSEWHERE_IOU]
    dropped = sum(1 for v, _ in scored if v >= ELSEWHERE_IOU)
    return subject, extras, dropped


def face_media(mid: int, crop: tuple[bytes, int, int], subject: str, source: Photo, embedder_name: str) -> dict:
    data, w, h = crop
    return {
        **_base(mid, "face", data, f"{source.uid}_face.png", embedder_name, f"fhibe:{source.uid}"),
        "width": w,
        "height": h,
        "category": subject,
        "categories": [subject] if subject else [],
        "source_photo_id": source.pid,
    }


def load(dataset: str, medias: dict[int, dict], embedder_name: str) -> None:
    """Populate *medias* with the photos, or the face crops, of every one-person photo."""
    from PIL import Image  # noqa: PLC0415

    spec = pc.DATASETS[dataset]
    long_side = int(spec["long_side"])
    root = data_dir()
    photos = read_photos(root)
    copies = stage(root, photos, long_side)
    faces = spec.get("media_type") == "face"
    converter = None
    if faces:
        from vtscore.converters.image2face import CONVERTER as converter  # noqa: PLC0415, N811

    found = missed = extra = dropped = 0
    for p in photos:
        data = (copies / f"{p.uid}.jpg").read_bytes()
        with Image.open(io.BytesIO(data)) as im:
            im = im.convert("RGB")
            size = im.size
            if not faces:
                medias[p.pid] = photo_media(p, data, size, embedder_name)
                continue
            subject, extras, n_drop = face_crops(converter, im, p, size[0] / p.width)
        if subject is not None:
            medias[p.pid] = face_media(p.pid, subject, p.subject, p, embedder_name)
            found += 1
        else:
            missed += 1
        for k, c in enumerate(extras[:16]):
            medias[EXTRA_BIT | (p.pid << 4) | k] = face_media(EXTRA_BIT | (p.pid << 4) | k, c, "", p, embedder_name)
        extra += min(len(extras), 16)
        dropped += n_drop
    if faces:
        log(
            f"  fhibe: {dataset}: subject found in {found:,} of {len(photos):,} photos "
            f"({missed:,} missed), {extra:,} other faces kept as negatives, {dropped:,} ambiguous dropped"
        )
    else:
        log(f"  fhibe: {dataset}: {len(medias):,} photos at long side {long_side}")


def check(dataset: str) -> str:
    """What a rebuild reads: the release's ``filepaths.csv`` and every one-person photo it lists."""
    root = data_dir()
    photos = read_photos(root)
    missing = [p.img for p in photos if not (root / p.img).exists()]
    if missing:
        raise SystemExit(f"{dataset}: {len(missing)} FHIBE images missing under {root} (e.g. {missing[0]})")
    return f"FHIBE {pc.FHIBE_RELEASE}: {len(photos):,} one-person photos present"
