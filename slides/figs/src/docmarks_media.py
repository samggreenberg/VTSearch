"""Real pages and real marks from DocMarks' four sources, for the deck's cards.

The DocMarks corpus itself lives on the GRID, so a checkout cannot open it —
but its four *sources* are public, and every roster class's id names the page
and the mark it was anchored on (`spods/stamp_00293_1` is SPODS page 293, mark
1; see `cluster_marks.assign_class_ids`). So the anchor crop of every SPODS,
Tobacco800 and StaVer class can be rebuilt here, with the corpus's own source
readers (`scripts/experiments/docmarks/sources/`) and its own mask settings,
and nothing is drawn from a guess.

The UCSF classes were cut from letterhead bands on the GRID and have no such
id, so their two marks here are cut by hand from pinned public letters in the
Industry Documents Library: the same artwork, not the corpus's own copy.

Downloads go under the gitignored `data/docmarks-sources/`, idempotently:

* SPODS — a 2.9 GB RAR from IIT Kharagpur, unpacked with `unar`;
* Tobacco800 and StaVer — the Kaggle mirrors the corpus builder uses, through
  the public dataset-download endpoint;
* UCSF — single-page PDFs by document id, rendered with PyMuPDF.
"""

from __future__ import annotations

import functools
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = _REPO_ROOT / "data" / "docmarks-sources"
_DOCMARKS = _REPO_ROOT / "scripts" / "experiments" / "docmarks"

SPODS_URL = "https://facweb.iitkgp.ac.in/~jay/spods/spods.rar"
KAGGLE = "https://www.kaggle.com/api/v1/datasets/download/"
TOBACCO800_SLUG = "kaiquanmah/tobacco800-with-ground-truth"
STAVER_SLUG = "rtatman/stamp-verification-staver-dataset"
UCSF_PDF = "https://download.industrydocuments.ucsf.edu/{a}/{b}/{c}/{d}/{id}/{id}.pdf"

#: UCSF letters, by document id: a spread of the four companies whose marks
#: the roster holds. Pinned, so the card redraws the same pages.
UCSF_PAGES = (
    "ffbb0108",
    "ffbk0077",
    "ffcm0123",
    "ffmb0040",
    "fjhw0048",
    "ffbb0213",
    "ffbg0207",
    "ffbj0201",
)
#: Hand-cut crops of two UCSF roster marks, as `(doc id, page-fraction box)`.
UCSF_MARKS = {
    "ucsf/logo_p_lorillard_crest": ("ffbk0077", (0.03, 0.06, 0.28, 0.20)),
    "ucsf/logo_rjr_script": ("ffmb0040", (0.02, 0.045, 0.39, 0.13)),
}


def _modules() -> tuple[Any, Any, Any, Any, Any]:
    if str(_DOCMARKS) not in sys.path:
        sys.path.insert(0, str(_DOCMARKS))
    import docmarks_config
    from sources import _common, spods, staver, tobacco800

    return docmarks_config, _common, spods, staver, tobacco800


def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (vtsearch slide figures)"})  # noqa: S310 — constant https URLs
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(request) as resp, tmp.open("wb") as out:  # noqa: S310 — constant https URLs
            shutil.copyfileobj(resp, out)
        tmp.rename(dest)
    return dest


def _unzip(archive: Path, into: Path) -> None:
    root = into.resolve()
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if not (root / info.filename).resolve().is_relative_to(root):
                raise SystemExit(f"{archive.name}: member {info.filename!r} would write outside {root}")
            zf.extract(info, into)


def spods_root() -> Path:
    into = ROOT / "spods"
    if not into.is_dir():
        archive = _download(SPODS_URL, ROOT / "spods.rar")
        subprocess.run(["unar", "-q", "-o", str(into), str(archive)], check=True)  # noqa: S603, S607
        archive.unlink()
    return into


def _kaggle(slug: str, name: str) -> Path:
    into = ROOT / name
    if not into.is_dir():
        archive = _download(KAGGLE + slug, ROOT / f"{name}.zip")
        _unzip(archive, into)
        archive.unlink()
    return into


def tobacco800_root() -> Path:
    return _kaggle(TOBACCO800_SLUG, "t800")


def staver_root() -> Path:
    return _kaggle(STAVER_SLUG, "staver")


def ucsf_page(doc_id: str, dpi: int = 150) -> Any:
    """First page of a UCSF document, rendered."""
    import pymupdf
    from PIL import Image

    pdf = _download(
        UCSF_PDF.format(a=doc_id[0], b=doc_id[1], c=doc_id[2], d=doc_id[3], id=doc_id),
        ROOT / "ucsf" / f"{doc_id}.pdf",
    )
    pix = pymupdf.open(pdf)[0].get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


@functools.cache
def _tobacco800_marks() -> tuple[dict[str, list], dict[str, Path]]:
    _, _, _, _, tobacco800 = _modules()
    root = tobacco800_root()
    marks: dict[str, list] = {}
    for xml in sorted(root.rglob("*.xml")):
        for stem, found in tobacco800.parse_gedi(xml.read_text(encoding="utf-8", errors="replace")).items():
            marks.setdefault(stem, []).extend(found)
    images = {p.stem.lower(): p for p in root.rglob("*.tif")}
    return marks, images


def page(source: str, stem: str) -> tuple[Any, list[tuple[str, tuple[int, int, int, int]]]]:
    """`(page image, [(kind, (x, y, w, h))])` for one source page, marks in the corpus's order."""
    from PIL import Image

    cfg, common, spods, staver, _ = _modules()
    if source == "spods":
        pages_dir, gt = spods.find_tree(spods_root())
        number = int(stem)
        image = Image.open(pages_dir / f"image ({number}).png").convert("RGB")
        found = spods.marks_for_page(
            gt,
            number,
            width=image.width,
            height=image.height,
            min_area_frac=cfg.MIN_MARK_AREA_FRAC,
            max_area_frac=cfg.MAX_MARK_AREA_FRAC,
        ).marks
        return image, [(m.kind, m.box) for m in found]
    if source == "tobacco800":
        marks, images = _tobacco800_marks()
        image = Image.open(images[stem.lower()]).convert("RGB")
        return image, [(m.kind, m.box) for m in marks.get(stem.lower(), [])]
    if source == "staver":
        scans, masks, _ = staver.find_tree(staver_root())
        scan = next(p for p in scans.rglob("*") if p.stem.lower() == stem.lower())
        mask = next(p for p in masks.rglob("*") if staver.gt_stem_key(p.stem) == stem.lower())
        image = Image.open(scan).convert("RGB")
        with Image.open(mask) as m:
            boxes = common.mask_to_boxes(
                m.convert("L"),
                min_area_frac=cfg.MIN_MARK_AREA_FRAC,
                merge_gap=common.merge_gap_for_page(image.width, image.height),
            )
        boxes, _ = common.reject_oversize(boxes, image.width, image.height, cfg.MAX_MARK_AREA_FRAC)
        return image, [("stamp", b) for b in boxes]
    raise SystemExit(f"docmarks media: no reader for source {source!r}")


def staver_stems() -> list[str]:
    _, _, _, staver, _ = _modules()
    scans, _, _ = staver.find_tree(staver_root())
    return sorted(p.stem.lower() for p in scans.rglob("*.png"))


def anchor(class_id: str) -> tuple[Any, tuple[int, int, int, int]]:
    """The page a roster class was anchored on, and the anchor mark's box."""
    if class_id in UCSF_MARKS:
        doc_id, (a, b, c, d) = UCSF_MARKS[class_id]
        image = ucsf_page(doc_id, dpi=200)
        w, h = image.size
        return image, (int(a * w), int(b * h), int((c - a) * w), int((d - b) * h))
    source, rest = class_id.split("/", 1)
    kind, _, tail = rest.partition("_")
    stem, index = tail.rsplit("_", 1)
    image, marks = page(source, stem)
    mark_kind, box = marks[int(index)]
    if mark_kind != kind:
        raise SystemExit(f"{class_id}: mark {index} on {stem} is a {mark_kind}, not a {kind}")
    return image, box


def mark_crop(class_id: str, pad: float = 0.08) -> Any:
    image, (x, y, w, h) = anchor(class_id)
    return image.crop(
        (
            max(0, int(x - pad * w)),
            max(0, int(y - pad * h)),
            min(image.width, int(x + w * (1 + pad))),
            min(image.height, int(y + h * (1 + pad))),
        )
    )


def datasheet_roster() -> list[str]:
    import re

    text = (_DOCMARKS / "DATASHEET.md").read_text()
    return re.findall(r"^\| `([a-z0-9]+/[^`]+)` \|", text, flags=re.M)


if __name__ == "__main__":
    print(json.dumps({"spods": str(spods_root()), "t800": str(tobacco800_root()), "staver": str(staver_root())}))
