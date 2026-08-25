"""UCSF Industry Documents Library — the haystack, and weak letterhead classes.

Two jobs, from one open Solr index (no registration, no token):

**Distractors.** Real scanned corporate pages in the millions.  This is what
makes a retrieval number mean something: StaVer's own 259 pages cannot separate
a good ranker from a lucky one, and the shipped SPODS/StaVer/Tobacco800 sets
together are ~2,700 pages.

**Weakly-labelled letterhead classes.** Single-page documents of ``type:letter``
written by a given company are, in the overwhelming majority, that company's
letterhead — so the ``author`` field is an instance label for the letterhead
logo, at a scale no hand annotation reaches.  Measured 2026-08-25 against the
live index, single-page ``type:letter`` counts per author:

    RJR                          162,197
    PHILIP MORRIS                 73,320
    LOR, LORILLARD                21,120

with 1,802,100 single-page tobacco letters in total and 13,216,456 short
tobacco documents carrying a ``collection``.

**The label is weak and this module never pretends otherwise.**  Marks it emits
carry ``provenance="weak"`` and no box: the metadata says the page is *from*
Philip Morris, not *where* on the page the logo is, nor even that a logo was
printed at all.  Two things must happen before this stratum is quoted:

1. a human spot-check of sampled pages per author, to estimate what fraction
   really carry the letterhead (``make_audit_slate.py --task letterhead``), and
2. one hand-drawn query crop per class, since there is no box to crop from.

Prefer ``author`` over ``collection``.  ``collection`` is provenance — whose
filing cabinet the page sat in — so a letter *in* the Philip Morris collection
is about as likely to be incoming mail on a law firm's letterhead.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Iterator, Optional

from . import _common
from ._common import Mark, Page

API_URL = "https://metadata.idl.ucsf.edu/solr/ltdl3/query"
DOWNLOAD_URL = "https://download.industrydocuments.ucsf.edu"

#: Fields worth carrying into the manifest.  ``documentdate`` earns its place:
#: a corporate logo is redesigned every decade or so, and a class that silently
#: spans two designs will look like a method failure rather than what it is.
FIELDS = "id,collection,collectioncode,author,industry,type,documentdate,pages,title,brand"

#: The endpoint ignores ``rows`` and returns up to this many docs per request.
PAGE_SIZE = 1000


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def build_query(
    *,
    industry: Optional[str] = None,
    author: Optional[str] = None,
    doc_type: Optional[str] = "letter",
    max_pages: int = 1,
) -> str:
    """Assemble a Solr ``q``.  Multi-word values are quoted."""

    def q(value: str) -> str:
        return f'"{value}"' if " " in value or "," in value else value

    clauses = [f"pages:[1 TO {max_pages}]" if max_pages > 1 else "pages:1"]
    if industry:
        clauses.append(f"industry:{q(industry)}")
    if author:
        clauses.append(f"author:{q(author)}")
    if doc_type:
        clauses.append(f"type:{q(doc_type)}")
    return " AND ".join(clauses)


def solr_docs(query: str, *, limit: int, session: Any = None, pause_s: float = 0.0) -> Iterator[dict[str, Any]]:
    """Yield up to *limit* Solr documents for *query*, deep-paged by cursorMark.

    ``start``-based paging breaks down past a few thousand rows on this index;
    ``cursorMark`` with a deterministic ``sort`` is the supported deep-paging
    route and is stable across requests, so a resumed job sees the same order.
    """
    import requests

    session = session or requests.Session()
    cursor = "*"
    seen = 0
    while seen < limit:
        params = {
            "q": query,
            "rows": str(min(PAGE_SIZE, limit - seen)),
            "wt": "json",
            "fl": FIELDS,
            "sort": "id asc",
            "cursorMark": cursor,
        }
        resp = session.get(API_URL, params=params, timeout=(20, 120))
        resp.raise_for_status()
        payload = resp.json()
        docs = payload.get("response", {}).get("docs", [])
        if not docs:
            return
        for doc in docs:
            yield doc
            seen += 1
            if seen >= limit:
                return
        next_cursor = payload.get("nextCursorMark")
        if not next_cursor or next_cursor == cursor:
            return
        cursor = next_cursor
        if pause_s:
            time.sleep(pause_s)


def count(query: str, session: Any = None) -> int:
    """``numFound`` for *query* — used to size a pull before making it."""
    import requests

    session = session or requests.Session()
    resp = session.get(
        API_URL,
        params={"q": query, "rows": "0", "wt": "json"},
        timeout=(20, 60),
    )
    resp.raise_for_status()
    return int(resp.json()["response"]["numFound"])


def pdf_url(doc_id: str) -> str:
    """UCSF's split-character path scheme: ``ffbb0019`` -> ``f/f/b/b/ffbb0019``."""
    if len(doc_id) < 4:
        raise ValueError(f"document id too short for the UCSF URL scheme: {doc_id!r}")
    a, b, c, d = doc_id[0], doc_id[1], doc_id[2], doc_id[3]
    return f"{DOWNLOAD_URL}/{a}/{b}/{c}/{d}/{doc_id}/{doc_id}.pdf"


def first_value(doc: dict[str, Any], key: str) -> Optional[str]:
    """Solr multi-valued fields come back as lists; take the first."""
    value = doc.get(key)
    if isinstance(value, list):
        return str(value[0]) if value else None
    return str(value) if value is not None else None


def decade(documentdate: Optional[str]) -> Optional[str]:
    """``"1996 January 24"`` -> ``"1990s"``.

    Used to optionally split a letterhead class by era.  A 1965 Philip Morris
    mark and a 1995 one may be different artwork; whether structural search
    should treat them as one class is a *finding*, not a nuisance, so the corpus
    records the decade and lets the study decide.
    """
    if not documentdate:
        return None
    m = re.search(r"\b(1[89]\d{2}|20\d{2})\b", documentdate)
    if not m:
        return None
    return f"{int(m.group(1)) // 10 * 10}s"


def doc_to_page(
    doc: dict[str, Any],
    image_path: str,
    width: int,
    height: int,
    *,
    page_index: int = 0,
    letterhead_author: Optional[str] = None,
    split_by_decade: bool = False,
) -> Page:
    """One rendered page image plus its Solr metadata as a :class:`Page`.

    When *letterhead_author* is given the page gets a boxless ``weak`` mark for
    that author's letterhead class.  No box is invented: a fabricated box would
    be indistinguishable downstream from a real one, and the whole point of the
    ``provenance`` field is that the difference stays visible.
    """
    doc_id = str(doc["id"])
    industry = first_value(doc, "industry")
    era = decade(first_value(doc, "documentdate"))

    marks: list[Mark] = []
    if letterhead_author:
        class_id = f"ucsf/letterhead_{_slug(letterhead_author)}"
        if split_by_decade and era:
            class_id = f"{class_id}_{era}"
        marks.append(Mark(kind="logo", box=(0, 0, 0, 0), class_id=class_id, provenance="weak"))

    return Page(
        page_id=f"ucsf/{doc_id}#{page_index}",
        source="ucsf",
        path=image_path,
        width=width,
        height=height,
        marks=marks,
        meta={
            "doc_id": doc_id,
            "industry": industry,
            "collection": first_value(doc, "collection"),
            "author": first_value(doc, "author"),
            "type": first_value(doc, "type"),
            "documentdate": first_value(doc, "documentdate"),
            "decade": era,
            "title": first_value(doc, "title"),
        },
    )


def fetch_and_render(
    docs: list[dict[str, Any]],
    raw_root: Path,
    out_images: Path,
    *,
    dpi: int = 150,
    letterhead_author: Optional[str] = None,
    split_by_decade: bool = False,
    max_pages_per_doc: int = 1,
    on_error: Optional[Any] = None,
) -> list[Page]:
    """Download each doc's PDF, render its pages to PNG, return :class:`Page`\\ s.

    Failures are skipped rather than fatal: at 200k documents a handful of dead
    ids or malformed PDFs is expected, and an aborted pull is far more expensive
    than a slightly short one.  Every skip is reported through *on_error* so the
    count is visible instead of silent.
    """
    from vtscore.datasets.pdf import render_pdf_pages

    pdf_dir = raw_root / "ucsf" / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)

    pages: list[Page] = []
    for doc in docs:
        doc_id = str(doc.get("id", ""))
        if len(doc_id) < 4:
            continue
        pdf_path = pdf_dir / f"{doc_id}.pdf"
        try:
            _common.http_download(pdf_url(doc_id), pdf_path)
            rendered = render_pdf_pages(pdf_path, dpi=dpi)
        except Exception as exc:  # noqa: BLE001 - a dead id must not kill a 200k pull
            if on_error:
                on_error(doc_id, exc)
            pdf_path.unlink(missing_ok=True)
            continue

        for idx, (_name, image) in enumerate(rendered[:max_pages_per_doc]):
            image_path = out_images / f"{doc_id}_{idx}.png"
            if not image_path.exists():
                image.save(image_path)
            pages.append(
                doc_to_page(
                    doc,
                    str(image_path),
                    image.width,
                    image.height,
                    page_index=idx,
                    letterhead_author=letterhead_author,
                    split_by_decade=split_by_decade,
                )
            )
    return pages
