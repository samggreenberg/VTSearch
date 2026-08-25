"""Tobacco800 — 1,290 real scanned business documents with logo and signature GT.

A public subset of the IIT-CDIP tobacco-litigation collection, ground-truthed by
the University of Maryland's Language and Media Processing lab in GEDI XML.  Of
the 1,290 pages, 412 carry a logo.  The published logo-matching protocol keeps
the 21 logo categories with **two or more** occurrences, which is a fine bar for
a matching paper and a useless one for a train-and-search eval: at two
instances, using one as the query leaves exactly one thing to retrieve.
``build_corpus.py`` applies its own ``--min-instances`` and prints the survival
curve so the bar is chosen from the data.

Two things to know about the ground truth:

* The v2.0 XML carries **the true identity of each signature instance**, which
  is what makes Tobacco800 a signature-authorship benchmark.  It is much less
  clear that logo zones carry an identity attribute — the literature describes
  the 21 logo categories as something researchers derived.  This adapter
  therefore reads any identity-looking attribute it finds and falls back to
  clustering when there is none, rather than assuming either way.
* Tobacco800 is drawn from IIT-CDIP, the same archive behind RVL-CDIP and UCSF's
  Tobacco industry.  Any of those used as a "distractor" pool is certain to
  contain more instances of these same letterheads.  ``docmarks_config`` encodes
  that as a contamination rule; do not work around it.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET  # noqa: S405 - local trusted GT files, see parse_gedi
from pathlib import Path
from typing import Any, Optional

from . import _common
from ._common import Mark, Page

#: The official TC-11 host (``tc11.cvc.uab.es/datasets/Tobacco800_1``) did not
#: resolve when this was written; the Kaggle mirror bundles images and GT.
KAGGLE_SLUG = "kaiquanmah/tobacco800-with-ground-truth"
TC11_PAGE = "https://tc11.cvc.uab.es/datasets/Tobacco800_1"

#: GEDI zone types that are marks we care about.
_ZONE_KINDS = {
    "dllogo": "logo",
    "dlsignature": "signature",
}

#: Attributes that have carried an instance identity in some GEDI releases,
#: most-specific first.  ``id`` is deliberately absent: it is a per-zone serial
#: number, not an identity, and treating it as one would give every mark its own
#: singleton class.
_IDENTITY_ATTRS = ("logoid", "logo_id", "authorid", "author_id", "name", "label", "companyname")

_IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def fetch(raw_root: Path) -> Path:
    dest = raw_root / "tobacco800"
    _common.kaggle_download(KAGGLE_SLUG, dest)
    return dest


def _lower_attrs(elem: ET.Element) -> dict[str, str]:
    return {k.lower(): v for k, v in elem.attrib.items()}


def _int_attr(attrs: dict[str, str], *names: str) -> Optional[int]:
    for n in names:
        v = attrs.get(n)
        if v is None:
            continue
        try:
            return int(round(float(v)))
        except ValueError:
            continue
    return None


def parse_gedi(xml_text: str) -> dict[str, list[Mark]]:
    """Parse one GEDI XML file into ``{page image stem: [Mark, ...]}``.

    These are local ground-truth files shipped with the dataset, not remote
    input, so stdlib XML parsing is appropriate; there is no entity expansion in
    GEDI and nothing here is fetched at parse time.
    """
    root = ET.fromstring(xml_text)  # noqa: S314 - local trusted GT file
    out: dict[str, list[Mark]] = {}

    for page in root.iter():
        if not page.tag.lower().endswith("dl_page"):
            continue
        pattrs = _lower_attrs(page)
        src = pattrs.get("src") or pattrs.get("filename") or ""
        stem = Path(src).stem.lower()
        if not stem:
            continue
        marks = out.setdefault(stem, [])

        for zone in page.iter():
            if not zone.tag.lower().endswith("dl_zone"):
                continue
            zattrs = _lower_attrs(zone)
            kind = _ZONE_KINDS.get(str(zattrs.get("gedi_type", "")).lower())
            if kind is None:
                continue
            x = _int_attr(zattrs, "col", "x", "left")
            y = _int_attr(zattrs, "row", "y", "top")
            w = _int_attr(zattrs, "width", "w")
            h = _int_attr(zattrs, "height", "h")
            if None in (x, y, w, h) or w <= 0 or h <= 0:  # type: ignore[operator]
                continue

            identity = next((zattrs[a] for a in _IDENTITY_ATTRS if zattrs.get(a)), None)
            class_id = None
            if identity:
                slug = re.sub(r"[^a-z0-9]+", "_", identity.strip().lower()).strip("_")
                if slug:
                    class_id = f"tobacco800/{kind}_{slug}"

            marks.append(
                Mark(
                    kind=kind,
                    box=(x, y, w, h),  # type: ignore[arg-type]
                    class_id=class_id,
                    provenance="gt" if class_id else "gt",
                )
            )
    return out


def build_pages(unpacked: Path, *, limit: int | None = None) -> tuple[list[Page], list[str]]:
    """Every Tobacco800 page as a :class:`Page`.

    Pages with no logo and no signature are kept: they are the dataset's own
    in-domain negatives, and they are the only distractors guaranteed *not* to
    be contaminated by the same collection's unlabelled letterheads.
    """
    from PIL import Image

    xml_marks: dict[str, list[Mark]] = {}
    for xml_path in sorted(unpacked.rglob("*.xml")):
        try:
            for stem, marks in parse_gedi(xml_path.read_text(encoding="utf-8", errors="replace")).items():
                xml_marks.setdefault(stem, []).extend(marks)
        except ET.ParseError:
            continue

    images = sorted(p for p in unpacked.rglob("*") if p.suffix.lower() in _IMAGE_SUFFIXES)
    if limit is not None:
        images = images[:limit]

    warnings: list[str] = []
    if not xml_marks:
        warnings.append(f"tobacco800: no GEDI XML parsed under {unpacked} — is this the ground-truth mirror?")

    pages: list[Page] = []
    matched_stems: set[str] = set()
    for img in images:
        stem = img.stem.lower()
        with Image.open(img) as im:
            width, height = im.size
        marks = xml_marks.get(stem, [])
        if marks:
            matched_stems.add(stem)
        meta: dict[str, Any] = {"stem": stem}
        pages.append(
            Page(
                page_id=f"tobacco800/{stem}",
                source="tobacco800",
                path=str(img),
                width=width,
                height=height,
                marks=marks,
                meta=meta,
            )
        )

    unmatched = set(xml_marks) - matched_stems
    if unmatched:
        warnings.append(f"tobacco800: {len(unmatched)} GT page(s) had no matching image, e.g. {sorted(unmatched)[:3]}")
    return pages, warnings
