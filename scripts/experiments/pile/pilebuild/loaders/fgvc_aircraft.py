"""``fgvc_aircraft``: telling sibling aircraft variants apart, the unknown-class bench (#5490).

The question it exists for: someone wants a 2028 Ford Bar and not a 2027 Ford Foo, two models no
model was trained on. Nobody can name either to a text tower or a detector, so the user types the
general class and votes. FGVC-Aircraft (Maji et al. 2013; :data:`pile_config.FGVC_ROOT`) is that
problem on labels a human can check: 10,000 photos, 100 variants under 70 families, so a 737-400
has seven 737 siblings that differ in a fuselage plug or a winglet.

**The cells are the variants that have siblings.** :data:`FAMILIES` holds the 16 families with
two or more variants (46 variants). A cell's positives are its variant's 100 photos, boxed by
FGVC's aircraft box; its negatives are everything else in the pool, the siblings first among
them. Every image is evaluable in every cell: FGVC names exactly one variant per photo, so a
sibling is a provable negative, not a doubtful one -- which is what `scale_core`'s designated
membership (a positive scorable only in its own cell) would have thrown away.

**The other 54 variants are in the pool as negatives**, 5,400 aircraft that are not any cell's
target, and :data:`DISTRACTORS` COCO images with no airplane stand for everything else in a
library. They come from `coco_better`'s own pool, every one evaluable for all three airplane
bands and holding none -- a negative its review proved, not an unannotated one -- drawn by hash
rank so a rebuild draws the same ones.

**The banner is cropped.** Every FGVC photo carries a 20-pixel copyright strip along its bottom
edge (the dataset's README says to remove it); a text tower reads text, and a photographer's
credit is not part of an aircraft. The crops are staged once as JPEG q95 under
:data:`pile_config.FGVC_STAGED`, and a media's ``origin_name`` names the staged copy, so the
zero-shot scorers read the same pixels the embedders did.

**What a user would type is part of the dataset**, in :data:`FAMILIES` and :func:`prompt_levels`:
the variant's real name (an upper bound -- the models have likely seen these names), its family,
its manufacturer, the bare class, and two names no model can know -- the family with a variant
number that was never built (``Boeing 737-1100``, a "2027 Ford Foo" refresh) and the
manufacturer with an invented model name (``Boeing Veltra``, a new nameplate).

Licence: non-commercial research only (Oxford VGG). The cells are vectors of these images and
stay on the GRID, like COCO's.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import NamedTuple

import pile_config as pc
from pilebuild.env import log

#: The copyright strip along the bottom of every FGVC photo, in pixels.
BANNER_PX = 20

#: COCO images with no airplane, drawn from `coco_better`'s pool.
DISTRACTORS = 10_000

#: FGVC ids are 7-digit strings; COCO's run to 581,929. Offset so the two never collide.
FGVC_ID_OFFSET = 10_000_000


class Family(NamedTuple):
    manufacturer: str  # what a user types for the maker ("Boeing")
    name: str  # the family as a user types it
    variants: dict[str, str]  # FGVC variant label -> its real name as typed
    fake_number: str  # the family with a variant number that was never built
    fake_name: str  # the maker with an invented model name


#: FGVC family label -> how each level is typed. Names carry the maker the way a user would type
#: them; no articles (SAM3 takes noun phrases, and #5395 found an article alone moved
#: GroundingDINO's AP from .64 to .21). The never-built numbers were checked against each
#: family's production list; the invented names are no aircraft's.
FAMILIES: dict[str, Family] = {
    "A320": Family(
        "Airbus",
        "Airbus A320 family",
        {"A318": "Airbus A318", "A319": "Airbus A319", "A320": "Airbus A320", "A321": "Airbus A321"},
        "Airbus A324",
        "Airbus Corvane",
    ),
    "A330": Family(
        "Airbus",
        "Airbus A330",
        {"A330-200": "Airbus A330-200", "A330-300": "Airbus A330-300"},
        "Airbus A330-450",
        "Airbus Meridel",
    ),
    "A340": Family(
        "Airbus",
        "Airbus A340",
        {
            "A340-200": "Airbus A340-200",
            "A340-300": "Airbus A340-300",
            "A340-500": "Airbus A340-500",
            "A340-600": "Airbus A340-600",
        },
        "Airbus A340-700",
        "Airbus Solvanta",
    ),
    "Boeing 737": Family(
        "Boeing",
        "Boeing 737",
        {f"737-{n}00": f"Boeing 737-{n}00" for n in range(2, 10)},
        "Boeing 737-1100",
        "Boeing Veltra",
    ),
    "Boeing 747": Family(
        "Boeing",
        "Boeing 747",
        {f"747-{n}00": f"Boeing 747-{n}00" for n in range(1, 5)},
        "Boeing 747-650",
        "Boeing Aurelan",
    ),
    "Boeing 757": Family(
        "Boeing",
        "Boeing 757",
        {"757-200": "Boeing 757-200", "757-300": "Boeing 757-300"},
        "Boeing 757-400",
        "Boeing Castrel",
    ),
    "Boeing 767": Family(
        "Boeing",
        "Boeing 767",
        {"767-200": "Boeing 767-200", "767-300": "Boeing 767-300", "767-400": "Boeing 767-400"},
        "Boeing 767-500",
        "Boeing Norvane",
    ),
    "Boeing 777": Family(
        "Boeing",
        "Boeing 777",
        {"777-200": "Boeing 777-200", "777-300": "Boeing 777-300"},
        "Boeing 777-500",
        "Boeing Talvero",
    ),
    "BAE 146": Family(
        "British Aerospace",
        "BAE 146",
        {"BAE 146-200": "BAE 146-200", "BAE 146-300": "BAE 146-300"},
        "BAE 146-400",
        "British Aerospace Kestrane",
    ),
    "CRJ-700": Family(
        "Canadair",
        "Canadair CRJ",
        {"CRJ-700": "Canadair CRJ-700", "CRJ-900": "Canadair CRJ-900"},
        "Canadair CRJ-1100",
        "Canadair Ostrava",
    ),
    "Cessna Citation": Family(
        "Cessna",
        "Cessna Citation",
        {"Cessna 525": "Cessna Citation 525", "Cessna 560": "Cessna Citation 560"},
        "Cessna Citation 590",
        "Cessna Pellira",
    ),
    "Dash 8": Family(
        "de Havilland",
        "de Havilland Dash 8",
        {"DHC-8-100": "de Havilland Dash 8-100", "DHC-8-300": "de Havilland Dash 8-300"},
        "de Havilland Dash 8-500",
        "de Havilland Corrigan",
    ),
    "Embraer E-Jet": Family(
        "Embraer",
        "Embraer E-Jet",
        {"E-170": "Embraer E170", "E-190": "Embraer E190", "E-195": "Embraer E195"},
        "Embraer E185",
        "Embraer Taluma",
    ),
    "Embraer ERJ 145": Family(
        "Embraer",
        "Embraer ERJ",
        {"ERJ 135": "Embraer ERJ 135", "ERJ 145": "Embraer ERJ 145"},
        "Embraer ERJ 155",
        "Embraer Varesso",
    ),
    "Gulfstream": Family(
        "Gulfstream",
        "Gulfstream",
        {"Gulfstream IV": "Gulfstream IV", "Gulfstream V": "Gulfstream V"},
        "Gulfstream IX",
        "Gulfstream Altheon",
    ),
    "MD-80": Family(
        "McDonnell Douglas",
        "McDonnell Douglas MD-80 series",
        {"MD-80": "McDonnell Douglas MD-80", "MD-87": "McDonnell Douglas MD-87"},
        "McDonnell Douglas MD-84",
        "McDonnell Douglas Brevant",
    ),
}

#: The prompt levels, most to least specific; the two ``fake_*`` levels are names no model knows.
LEVELS = ("variant", "family", "manufacturer", "generic", "fake_number", "fake_name")

#: The bare class, as typed.
GENERIC = "airplane"


def cells() -> list[str]:
    """Every cell: the FGVC variant labels of :data:`FAMILIES`."""
    return sorted(v for fam in FAMILIES.values() for v in fam.variants)


def family_of(variant: str) -> str:
    for fam_label, fam in FAMILIES.items():
        if variant in fam.variants:
            return fam_label
    raise KeyError(variant)


def prompt_levels(variant: str) -> dict[str, str]:
    """``{level: text}`` a user hunting *variant* might type, for every level of :data:`LEVELS`."""
    fam = FAMILIES[family_of(variant)]
    return {
        "variant": fam.variants[variant],
        "family": fam.name,
        "manufacturer": f"{fam.manufacturer} airplane",
        "generic": GENERIC,
        "fake_number": fam.fake_number,
        "fake_name": fam.fake_name,
    }


class FgvcImage(NamedTuple):
    fgvc_id: str
    variant: str
    family: str
    manufacturer: str
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in the original's pixels


def _read_pairs(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            k, v = line.split(" ", 1)
            out[k] = v.strip()
    return out


def read_images(data: Path | None = None) -> list[FgvcImage]:
    """Every FGVC photo (trainval and test: nothing here trains) with its three labels and box."""
    data = data or pc.FGVC_DATA
    labels = {}
    for level in ("variant", "family", "manufacturer"):
        merged: dict[str, str] = {}
        for split in ("trainval", "test"):
            merged.update(_read_pairs(data / f"images_{level}_{split}.txt"))
        labels[level] = merged
    boxes = {}
    for line in (data / "images_box.txt").read_text().splitlines():
        if line.strip():
            k, *xs = line.split()
            boxes[k] = tuple(int(x) for x in xs)
    ids = sorted(labels["variant"])
    if len(ids) != 10_000:
        raise SystemExit(f"fgvc_aircraft: {len(ids)} labelled photos under {data}, expected 10,000")
    return [FgvcImage(i, labels["variant"][i], labels["family"][i], labels["manufacturer"][i], boxes[i]) for i in ids]


def _hash_rank(key: str) -> str:
    return hashlib.sha256(f"fgvc_aircraft:{key}".encode()).hexdigest()


def distractor_ids() -> list[tuple[int, str]]:
    """``(coco id, origin_name)`` of :data:`DISTRACTORS` `coco_better` images that provably hold no airplane."""
    import sys  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "calibration"))
    from _cells_io import load_medias  # noqa: PLC0415

    bands = {f"airplane@{b}" for b in ("small", "medium", "large")}
    medias = load_medias(pc.cell_path("coco_better", "siglip"))
    keep = [
        (iid, m["origin_name"])
        for iid, m in medias.items()
        if bands <= set(m.get("evaluable_categories") or [])
        and not any(c.partition("@")[0] == "airplane" for c in m.get("categories") or [])
    ]
    keep.sort(key=lambda t: _hash_rank(str(t[0])))
    if len(keep) < DISTRACTORS:
        raise SystemExit(f"fgvc_aircraft: only {len(keep)} airplane-free coco_better images")
    return sorted(keep[:DISTRACTORS])


def stage(images: list[FgvcImage]) -> None:
    """Write each photo without its banner to :data:`pile_config.FGVC_STAGED`, once."""
    from PIL import Image  # noqa: PLC0415

    pc.FGVC_STAGED.mkdir(parents=True, exist_ok=True)
    todo = [im for im in images if not (pc.FGVC_STAGED / f"{im.fgvc_id}.jpg").exists()]
    if todo:
        log(f"  fgvc_aircraft: staging {len(todo):,} banner-cropped copies in {pc.FGVC_STAGED}")
    for im in todo:
        with Image.open(pc.FGVC_DATA / "images" / f"{im.fgvc_id}.jpg") as src:
            w, h = src.size
            crop = src.convert("RGB").crop((0, 0, w, h - BANNER_PX))
        dest = pc.FGVC_STAGED / f"{im.fgvc_id}.jpg"
        tmp = dest.with_suffix(".tmp")
        crop.save(tmp, format="JPEG", quality=95)
        tmp.rename(dest)


def _media(iid: int, data: bytes, filename: str, origin_name: str, embedder_name: str, **labels) -> dict:
    return {
        "id": iid,
        "media_type": "image",
        "embedder": embedder_name,
        "duration": 0,
        "file_size": 0,
        "md5": "",
        "embeddings": {},
        "media_bytes": data,
        "media_string": None,
        "filename": filename,
        **labels,
        "origin": {"importer": "fgvc_aircraft", "params": {"embedder": embedder_name, "labels": "fgvc"}},
        "origin_name": origin_name,
    }


def load(dataset: str, medias: dict[int, dict], embedder_name: str) -> None:
    """The 10,000 FGVC photos (banner cropped) and :data:`DISTRACTORS` airplane-free COCO images."""
    import zipfile  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    images = read_images()
    stage(images)
    all_cells = cells()
    targets = set(all_cells)
    for im in images:
        path = pc.FGVC_STAGED / f"{im.fgvc_id}.jpg"
        data = path.read_bytes()
        with Image.open(io.BytesIO(data)) as pic:
            w, h = pic.size
        x0, y0, x1, y1 = im.box
        box = [x0 / w, y0 / h, min(x1, w - 1) / w, min(y1, h - 1) / h]
        cats = [im.variant] if im.variant in targets else []
        medias[FGVC_ID_OFFSET + int(im.fgvc_id)] = _media(
            FGVC_ID_OFFSET + int(im.fgvc_id),
            data,
            path.name,
            str(path),
            embedder_name,
            category=cats[0] if cats else "",
            categories=cats,
            evaluable_categories=all_cells,
            labels_exhaustive=True,
            coco_scored=False,
            regions=[{"box": box, "label": c} for c in cats],
            fgvc={"variant": im.variant, "family": im.family, "manufacturer": im.manufacturer},
        )
    zips: dict[str, zipfile.ZipFile] = {}
    for iid, origin in distractor_ids():
        arc, member = origin.split("::", 1)
        if arc not in zips:
            zips[arc] = zipfile.ZipFile(arc)
        medias[iid] = _media(
            iid,
            zips[arc].read(member),
            Path(member).name,
            origin,
            embedder_name,
            category="",
            categories=[],
            evaluable_categories=all_cells,
            labels_exhaustive=True,
            coco_scored=True,
            regions=[],
            fgvc=None,
        )
    log(f"  fgvc_aircraft: {len(images):,} FGVC photos + {DISTRACTORS:,} COCO distractors, {len(all_cells)} cells")


def check(dataset: str) -> str:
    """What a rebuild reads: FGVC's label files and images, and `coco_better`'s SigLIP cell for the distractors."""
    for path in (pc.FGVC_DATA / "images_box.txt", pc.FGVC_DATA / "images", pc.cell_path("coco_better", "siglip")):
        if not path.exists():
            raise SystemExit(f"{dataset}: missing {path}")
    labelled = {im.variant for im in read_images()}
    missing = sorted(set(cells()) - labelled)
    if missing:
        raise SystemExit(f"{dataset}: FAMILIES names variants FGVC does not: {missing}")
    return f"FGVC-Aircraft 2013b (10,000 photos, {len(cells())} cells) + coco_better's SigLIP cell present"
