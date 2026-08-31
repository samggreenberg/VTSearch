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

Signature masks are parsed and recorded but never promoted to query classes:
a handwritten signature is a different mark every time it is made, so it is not
an *instance* in the sense structural search means.  It costs nothing and gives
the study a documented negative control.

The text mask is **not** a set of marks.  It is the page body — a property of
the page rather than a thing on it — so it is recorded as ``meta["text_frac"]``
and ``meta["text_components"]`` and emits no :class:`Mark`.  See
:func:`marks_for_page` for what emitting it used to cost.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, NamedTuple, Optional

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

#: Categories that become query classes.
MARK_CATEGORIES = ("logo", "stamp")

#: Recorded as marks but never promoted to a query class.  ``text`` is
#: deliberately absent: it is the page body, and is kept as page metadata
#: instead — see :func:`marks_for_page`.
LOCALISED_CONTEXT_CATEGORIES = ("signature",)

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


class PageMarks(NamedTuple):
    """What one page's four masks yield: its marks, page-level facts, complaints."""

    marks: list[Mark]
    meta: dict[str, Any]
    warnings: list[str]


def marks_for_page(
    gt_dir: Path,
    page_no: int,
    *,
    width: int,
    height: int,
    min_area_frac: float,
    merge_gap: int | None = None,
    max_area_frac: float = 1.0,
) -> PageMarks:
    """Every mark on one page, from that page's four category masks.

    Each mask is decomposed into connected components, components within
    *merge_gap* pixels are merged, and only then is the area floor applied to
    the merged group's ink.  That order is the whole point: a stamp's mask is a
    ring plus the text inside it plus a broken arc where the ink did not take,
    and filtering first deletes the fragments before the merge can reassemble
    them (issue #3361).  ``merge_gap`` defaults to
    :func:`_common.merge_gap_for_page`.

    **The ``text`` mask yields no marks.**  It is the page body — not a thing
    *on* the page but a property *of* it — and its components are words, which
    are not marks in any sense this corpus uses.  Filtered, what survives is not
    even words: it is whichever headings and ruled tables happened to have an
    underline welding their glyphs into one component.  Those were never query
    classes (``MARK_CATEGORIES``), but they were real entries in ``page.marks``
    and leaked into every consumer that iterated it without a kind filter.  The
    mask's information is kept where it belongs, as ``meta["text_frac"]`` and
    ``meta["text_components"]``; the non-queryable negative control the study
    wanted is ``signature``, which is a localised mark and stays one.
    """
    from PIL import Image

    if merge_gap is None:
        merge_gap = _common.merge_gap_for_page(width, height)

    marks: list[Mark] = []
    meta: dict[str, Any] = {}
    warnings: list[str] = []

    for kind in (*MARK_CATEGORIES, *LOCALISED_CONTEXT_CATEGORIES):
        mask_path = gt_dir / kind / f"image ({page_no}).png"
        if not mask_path.exists():
            continue
        with Image.open(mask_path) as im:
            boxes = _common.mask_to_boxes(im.convert("L"), min_area_frac=min_area_frac, merge_gap=merge_gap)
        boxes, oversize = _common.reject_oversize(boxes, width, height, max_area_frac)
        for box in oversize:
            frac = box[2] * box[3] / float(width * height)
            warnings.append(f"spods/{page_no:05d}: dropped a {kind} box covering {frac:.1%} of the page")
        marks.extend(Mark(kind=kind, box=b, class_id=None, provenance="gt") for b in boxes)

    text_mask = gt_dir / "text" / f"image ({page_no}).png"
    if text_mask.exists():
        with Image.open(text_mask) as im:
            comps = _common.mask_components(im.convert("L"))
        meta["text_components"] = len(comps)
        meta["text_frac"] = round(sum(c.ink for c in comps) / float(width * height), 5)

    return PageMarks(marks, meta, warnings)


def build_pages(
    unpacked: Path,
    *,
    min_area_frac: float,
    max_area_frac: float = 1.0,
    limit: int | None = None,
) -> tuple[list[Page], list[str]]:
    """Every SPODS page as a :class:`Page`, marks attached, identities unset.

    Returns ``(pages, warnings)``; *warnings* names every box rejected for
    covering more than *max_area_frac* of its page, so a mask artefact is
    reported rather than silently vanishing.
    """
    from PIL import Image

    pages_dir, gt_dir = find_tree(unpacked)
    numbered = sorted(
        ((n, p) for p in pages_dir.glob("*.png") if (n := page_number(p)) is not None),
        key=lambda t: t[0],
    )
    if limit is not None:
        numbered = numbered[:limit]

    out: list[Page] = []
    warnings: list[str] = []
    for page_no, path in numbered:
        with Image.open(path) as im:
            width, height = im.size
        found = marks_for_page(
            gt_dir,
            page_no,
            width=width,
            height=height,
            min_area_frac=min_area_frac,
            max_area_frac=max_area_frac,
        )
        warnings.extend(found.warnings)
        out.append(
            Page(
                page_id=f"spods/{page_no:05d}",
                source="spods",
                path=str(path),
                width=width,
                height=height,
                marks=found.marks,
                meta={"page_no": page_no, **found.meta},
            )
        )
    return out, warnings
