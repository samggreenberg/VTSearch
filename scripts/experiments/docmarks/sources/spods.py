"""SPODS — 1,088 scanned pseudo-official document pages with per-category masks.

Nandedkar, Mukhopadhyay & Sural, ICVGIP 2016.  The authors authored realistic
official documents, printed them, applied logos/stamps/signatures and scanned
the result — so the marks are real ink on real paper, but the documents are not
anyone's real correspondence, which is why they could publish them at all.

**What SPODS actually ships** (confirmed by walking the RAR's headers, since the
paper does not say):

    SPODS_Dataset/image (1..1088).png                        <- the pages
    Ground truth (GT1)/logo/image (1..1088).png              <- binary masks
    Ground truth (GT1)/stamp/image (1..1088).png
    Ground truth (GT1)/signature/image (1..1088).png
    Ground truth (GT1)/text/image (1..1088).png

Note what is **not** there: any notion of *which* logo.  The ground truth is
four binary masks per page, one per category.  A previous VTSearch study
(2026-07-13) reported "64 logo/stamp classes" for SPODS with class names like
``logo_14`` — those class identities were derived by that study, not read off
the dataset, and nothing verified them.  This adapter therefore emits marks with
``class_id=None`` and leaves identity to :mod:`cluster_marks`, whose output is
explicitly flagged ``provenance="clustered"`` and routed through a human audit.

Signature and text masks are parsed and recorded but never promoted to query
classes: a handwritten signature is a different mark every time it is made, so
it is not an *instance* in the sense structural search means, and the text mask
is the page body.  Keeping them costs nothing and gives the study a documented
negative control.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from . import _common
from ._common import Mark, Page

#: Direct download, no registration.  The dataset's own page still advertises
#: the ``www.facweb.iitkgp.ernet.in`` host, which has been decommissioned and
#: 503s; ``facweb.iitkgp.ac.in`` serves the same file.  A previous study
#: recorded SPODS as "offline" on the strength of the dead hostname.
SPODS_URL = "https://facweb.iitkgp.ac.in/~jay/spods/spods.rar"
SPODS_SIZE_BYTES = 2_937_772_915

PAGES_DIR = "SPODS_Dataset"
GT_DIR = "Ground truth (GT1)"

#: Categories that become query classes, and those kept only as context.
MARK_CATEGORIES = ("logo", "stamp")
CONTEXT_CATEGORIES = ("signature", "text")

_IMAGE_RE = re.compile(r"image\s*\((\d+)\)\.png$", re.IGNORECASE)


def fetch(raw_root: Path) -> Path:
    """Download and unpack SPODS.  Returns the directory holding both trees."""
    archive = raw_root / "spods" / "spods.rar"
    unpacked = raw_root / "spods" / "unpacked"
    _common.http_download(SPODS_URL, archive)
    _common.extract_rar(archive, unpacked)
    return unpacked


def page_number(path: Path) -> Optional[int]:
    """``image (417).png`` -> ``417``; ``None`` for anything else."""
    m = _IMAGE_RE.search(path.name)
    return int(m.group(1)) if m else None


def find_tree(unpacked: Path) -> tuple[Path, Path]:
    """Locate the pages dir and the GT dir under *unpacked*.

    Extractors disagree about whether to create a top-level wrapper directory,
    so this searches rather than assuming a depth.
    """
    pages = next((p for p in unpacked.rglob(PAGES_DIR) if p.is_dir()), None)
    gt = next((p for p in unpacked.rglob(GT_DIR) if p.is_dir()), None)
    if pages is None or gt is None:
        raise _common.FetchError(
            f"SPODS layout not recognised under {unpacked}: "
            f"expected '{PAGES_DIR}/' and '{GT_DIR}/' directories "
            f"(found: {sorted(p.name for p in unpacked.iterdir())[:8]})"
        )
    return pages, gt


def marks_for_page(gt_dir: Path, page_no: int, *, min_area_frac: float, merge_gap: int = 6) -> list[Mark]:
    """Every mark on one page, from that page's four category masks.

    Each mask is reduced to connected components, and components within
    *merge_gap* pixels are merged: a stamp's mask is typically a ring plus the
    text inside it plus a broken arc where the ink did not take, and treating
    those as three marks would poison the class inventory.
    """
    from PIL import Image

    marks: list[Mark] = []
    for kind in (*MARK_CATEGORIES, *CONTEXT_CATEGORIES):
        mask_path = gt_dir / kind / f"image ({page_no}).png"
        if not mask_path.exists():
            continue
        with Image.open(mask_path) as im:
            boxes = _common.mask_to_boxes(im.convert("L"), min_area_frac=min_area_frac)
        # Text masks are per-word and must not be merged into one page-sized blob.
        if kind != "text":
            boxes = _common.merge_overlapping(boxes, gap=merge_gap)
        marks.extend(Mark(kind=kind, box=b, class_id=None, provenance="gt") for b in boxes)
    return marks


def build_pages(unpacked: Path, *, min_area_frac: float, limit: int | None = None) -> list[Page]:
    """Every SPODS page as a :class:`Page`, marks attached, identities unset."""
    from PIL import Image

    pages_dir, gt_dir = find_tree(unpacked)
    numbered = sorted(
        ((n, p) for p in pages_dir.glob("*.png") if (n := page_number(p)) is not None),
        key=lambda t: t[0],
    )
    if limit is not None:
        numbered = numbered[:limit]

    out: list[Page] = []
    for page_no, path in numbered:
        with Image.open(path) as im:
            width, height = im.size
        out.append(
            Page(
                page_id=f"spods/{page_no:05d}",
                source="spods",
                path=str(path),
                width=width,
                height=height,
                marks=marks_for_page(gt_dir, page_no, min_area_frac=min_area_frac),
                meta={"page_no": page_no},
            )
        )
    return out
