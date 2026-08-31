"""StaVer — scanned dummy bills carrying rubber stamps, with pixel-accurate GT.

Originally published by DFKI (``madm.dfki.de/downloads-ds-staver``); that host
was unreachable when this adapter was written, so the Kaggle mirror is the
default route and DFKI is kept as a documented fallback.

The published dataset is **400 document images** with pixel-accurate ground
truth plus per-file ``info`` text describing how many stamps are present,
whether they overlap printed text, whether a signature is present, and whether
the stamp is black or coloured.  A previous VTSearch study used 259 pages and 36
multi-instance classes, i.e. a filtered subset; this adapter takes the whole
published set and lets ``build_corpus.py`` do the filtering, so the threshold is
a visible parameter rather than a number baked into a corpus.

Like SPODS, StaVer ships *where* the stamps are, not *which* stamp each one is.
Identity comes from :mod:`cluster_marks` and is flagged ``clustered``.

The ``info`` files are worth parsing for one reason beyond metadata: the
recorded stamp count is an independent check on the mask decomposition.  If the
connected-component pass yields four marks on a page the dataset says has one
stamp, the merge gap is wrong — and that is a bug that would otherwise show up
only as a mysteriously bad class inventory.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from . import _common
from ._common import Mark, Page

KAGGLE_SLUG = "rtatman/stamp-verification-staver-dataset"
DFKI_PAGE = "https://madm.dfki.de/downloads-ds-staver"

#: Directory names the mirror is known to use, lowercased for matching.
_SCAN_DIR_HINTS = ("scans", "images", "documents", "scan")
_MASK_DIR_HINTS = ("ground-truth-pixel", "ground_truth_pixel", "groundtruth", "ground-truth")
_INFO_DIR_HINTS = ("info", "infos")

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

#: Stem suffixes the ground-truth files carry that the scan does not.  On the
#: Kaggle mirror a page is ``scans/stampDS-00001.png``, its pixel mask is
#: ``ground-truth-pixel/stampDS-00001-px.png`` and its binary map is
#: ``ground-truth-maps/stampDS-00001-gt.png``.  Indexing masks by raw stem
#: therefore matches nothing at all: measured on the real archive (#3343) it
#: produced 427 "no ground-truth mask" warnings and ZERO usable StaVer pages,
#: while looking for all the world like a source that had simply been skipped.
#: The fixtures could not catch this because they were built from the
#: documented layout, which does not mention the suffixes.
_GT_STEM_SUFFIXES = ("-px", "-gt", "_px", "_gt")


def gt_stem_key(stem: str) -> str:
    """The scan stem a ground-truth filename belongs to, lowercased."""
    low = stem.lower()
    for suffix in _GT_STEM_SUFFIXES:
        if low.endswith(suffix):
            return low[: -len(suffix)]
    return low


def fetch(raw_root: Path) -> Path:
    """Download StaVer from Kaggle.  Returns the unpacked directory."""
    dest = raw_root / "staver"
    _common.kaggle_download(KAGGLE_SLUG, dest)
    return dest


def _find_dir(root: Path, hints: tuple[str, ...]) -> Optional[Path]:
    """First directory under *root* whose name matches one of *hints*.

    Mirrors reshuffle nesting depth, so this searches rather than assuming.
    """
    for path in sorted(root.rglob("*")):
        if path.is_dir() and path.name.lower().replace(" ", "-") in hints:
            return path
    return None


def find_tree(unpacked: Path) -> tuple[Path, Path, Optional[Path]]:
    """Locate ``(scans, masks, info)``.  *info* is optional; the others are not."""
    masks = _find_dir(unpacked, _MASK_DIR_HINTS)
    scans = _find_dir(unpacked, _SCAN_DIR_HINTS)
    info = _find_dir(unpacked, _INFO_DIR_HINTS)

    # Some mirrors drop the scans at the archive root beside the GT directory.
    if scans is None and masks is not None:
        siblings = [p for p in masks.parent.iterdir() if p.is_dir() and p != masks and p != info]
        scans = siblings[0] if len(siblings) == 1 else None

    if scans is None or masks is None:
        found = sorted({p.name for p in unpacked.rglob("*") if p.is_dir()})[:12]
        raise _common.FetchError(
            f"StaVer layout not recognised under {unpacked}: need a scans dir and a "
            f"ground-truth-pixel dir (found dirs: {found}). "
            f"If the Kaggle mirror changed, the original is at {DFKI_PAGE}."
        )
    return scans, masks, info


def parse_info(text: str) -> dict[str, Any]:
    """Parse one StaVer ``info`` file into a dict.

    The files are loose ``key: value`` lines with inconsistent casing and
    spacing between releases, so this normalises keys and pulls out the one
    field used as a check (``stamps``) as an int when it looks like one.
    """
    out: dict[str, Any] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")
        value = value.strip()
        if not key:
            continue
        if re.fullmatch(r"\d+", value):
            out[key] = int(value)
        elif value.lower() in ("yes", "true"):
            out[key] = True
        elif value.lower() in ("no", "false"):
            out[key] = False
        else:
            out[key] = value
    return out


def expected_stamp_count(info: dict[str, Any]) -> Optional[int]:
    """The stamp count StaVer records for a page, if it records one."""
    for key in ("number_of_stamps", "stamps", "num_stamps", "no_of_stamps", "stamp_count"):
        value = info.get(key)
        if isinstance(value, int):
            return value
    return None


def build_pages(
    unpacked: Path,
    *,
    min_area_frac: float,
    merge_gap: int = 10,
    limit: int | None = None,
) -> tuple[list[Page], list[str]]:
    """Every StaVer page as a :class:`Page`.

    Returns ``(pages, warnings)``; *warnings* names pages where the number of
    merged components disagrees with the count the dataset records, which is the
    signal that the merge gap needs tuning rather than that the data is wrong.
    """
    from PIL import Image

    scans_dir, masks_dir, info_dir = find_tree(unpacked)
    masks_by_stem = {gt_stem_key(p.stem): p for p in masks_dir.rglob("*") if p.suffix.lower() in _IMAGE_SUFFIXES}
    info_by_stem = {gt_stem_key(p.stem): p for p in (info_dir.rglob("*") if info_dir else []) if p.is_file()}

    scans = sorted(p for p in scans_dir.rglob("*") if p.suffix.lower() in _IMAGE_SUFFIXES)
    if limit is not None:
        scans = scans[:limit]

    pages: list[Page] = []
    warnings: list[str] = []
    for scan in scans:
        stem = scan.stem.lower()
        mask_path = masks_by_stem.get(stem)
        if mask_path is None:
            warnings.append(f"staver/{stem}: no ground-truth mask")
            continue

        with Image.open(scan) as im:
            width, height = im.size
        with Image.open(mask_path) as im:
            boxes = _common.mask_to_boxes(im.convert("L"), min_area_frac=min_area_frac)
        boxes = _common.merge_overlapping(boxes, gap=merge_gap)

        meta: dict[str, Any] = {}
        info_path = info_by_stem.get(stem)
        if info_path is not None:
            meta = parse_info(info_path.read_text(encoding="utf-8", errors="replace"))
            expected = expected_stamp_count(meta)
            if expected is not None and expected != len(boxes):
                warnings.append(f"staver/{stem}: info says {expected} stamp(s), mask decomposed to {len(boxes)}")

        pages.append(
            Page(
                page_id=f"staver/{stem}",
                source="staver",
                path=str(scan),
                width=width,
                height=height,
                marks=[Mark(kind="stamp", box=b, class_id=None, provenance="gt") for b in boxes],
                meta=meta,
            )
        )
    return pages, warnings
