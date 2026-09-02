"""UCSF Industry Documents Library — the haystack, and letterhead candidates.

Two jobs, from one open Solr index (no registration, no token):

**Distractors.** Real scanned corporate pages in the millions.  This is what
makes a retrieval number mean something: StaVer's own 259 pages cannot separate
a good ranker from a lucky one, and the shipped SPODS/StaVer/Tobacco800 sets
together are ~2,700 pages.

**Letterhead candidates.** Single-page ``type:letter`` documents written by a
given company are, in the overwhelming majority, printed on that company's
letterhead.  Measured 2026-08-25 against the live index:

    RJR                          162,197
    PHILIP MORRIS                 73,320
    LOR, LORILLARD                21,120

with 1,802,100 single-page tobacco letters in total and 13,216,456 short
tobacco documents carrying a ``collection``.

**``author`` is a candidate pool, not a class.**  This is the one thing to get
right here.  The field asserts a page is *from* a company; it has never looked
at the mark.  Making it a class id bakes two failures straight into the ground
truth:

* a company that redesigned its letterhead yields **one class holding two
  different artworks**, so a detector is punished for telling them apart;
* two subsidiaries sharing artwork yield **two classes holding the same mark**,
  so a detector is punished for recognising it.

Both are exactly the errors a mark-retrieval eval exists to measure, silently
written into the labels.  A class means *this artwork* and nothing else, so the
author narrows millions of pages to a high-yield pool and identity is then
settled by looking: cluster the letterhead bands, adjudicate the clusters, and
record same/different explicitly.  Marks emitted here carry
``provenance="candidate"`` and are never admitted as query classes until that
has happened.

For the same reason ``documentdate`` is recorded but never enters a class id.
Era is a fact about the calendar, not about the mark.

Prefer ``author`` over ``collection``.  ``collection`` is provenance — whose
filing cabinet the page sat in — so a letter *in* the Philip Morris collection
is about as likely to be incoming mail on a law firm's letterhead.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Iterator, Optional

import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from . import _common
from ._common import Mark, Page

API_URL = "https://metadata.idl.ucsf.edu/solr/ltdl3/query"
DOWNLOAD_URL = "https://download.industrydocuments.ucsf.edu"

#: Fields worth carrying into the manifest, as provenance for a page.  None of
#: them decides a class: identity comes from adjudicating the mark itself.
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


def year(documentdate: Optional[str]) -> Optional[int]:
    """``"1996 January 24"`` -> ``1996``.

    Recorded as provenance only.  It is deliberately **not** part of any class
    id: a class means "this artwork", and splitting one artwork across eras — or
    fusing two different artworks because they share an era — would make the
    label a statement about the calendar rather than about the mark.
    """
    if not documentdate:
        return None
    m = re.search(r"\b(1[89]\d{2}|20\d{2})\b", documentdate)
    return int(m.group(1)) if m else None


def letterhead_band(width: int, height: int, band_frac: float) -> tuple[int, int, int, int]:
    """The top-of-page strip a letterhead occupies, as ``(x, y, w, h)``.

    UCSF ships no boxes, and a class cannot be adjudicated from a mark nobody
    can see.  A letterhead sits at the top of the page by definition, so the top
    strip is a coarse but *honest* locator: wide enough to contain the mark,
    tight enough that clustering the strips separates one company's artwork from
    another's.  It is a candidate region, never a ground-truth box — the tight
    box comes from the hand-drawn query crop after adjudication.
    """
    return (0, 0, width, max(1, int(round(height * band_frac))))


def doc_to_page(
    doc: dict[str, Any],
    image_path: str,
    width: int,
    height: int,
    *,
    page_index: int = 0,
    letterhead_author: Optional[str] = None,
    band_frac: float = 0.22,
    industry: Optional[str] = None,
) -> Page:
    """One rendered page image plus its Solr metadata as a :class:`Page`.

    When *letterhead_author* is given, the page gets a **candidate** mark over
    the top-of-page band — deliberately *not* a class.  The author field says
    the page is *from* a company; it has not looked at the mark, so it cannot
    say which artwork is on it, or whether one is there at all.  Turning that
    into a class id would bake two failures into the ground truth: one company
    that redesigned its letterhead becomes a single class holding two different
    marks, and two subsidiaries sharing artwork become two classes holding the
    same mark.

    So the author narrows millions of pages to a high-yield pool, and identity
    is settled downstream by clustering the bands and adjudicating them — the
    same path SPODS and StaVer take.  ``provenance="candidate"`` marks are never
    admitted as query classes until that has happened.
    """
    doc_id = str(doc["id"])

    marks: list[Mark] = []
    if letterhead_author:
        marks.append(
            Mark(
                kind="logo",
                box=letterhead_band(width, height, band_frac),
                class_id=None,
                provenance="candidate",
            )
        )

    return Page(
        page_id=f"ucsf/{doc_id}#{page_index}",
        source="ucsf",
        path=image_path,
        width=width,
        height=height,
        marks=marks,
        meta={
            "doc_id": doc_id,
            # The industry we QUERIED, not one read back from the response.
            # `industry` is indexed but NOT stored in UCSF's Solr: `industry:
            # Tobacco` filters correctly while `fl=industry` returns nothing, so
            # every page in the 2026-08-31 build recorded `industry: null`.
            # That silently disarmed the one contamination rule that bites --
            # Tobacco800 and UCSF's Tobacco industry are both IIT-CDIP, so
            # scoring one against the other counts a CORRECT retrieval as a
            # false positive.  The caller knows the industry because it is the
            # thing it searched for; taking it from there needs no Solr change.
            "industry": industry or first_value(doc, "industry"),
            "collection": first_value(doc, "collection"),
            "author": first_value(doc, "author"),
            "letterhead_author": letterhead_author,
            "type": first_value(doc, "type"),
            "documentdate": first_value(doc, "documentdate"),
            "year": year(first_value(doc, "documentdate")),
            "title": first_value(doc, "title"),
        },
    )


def _fetch_workers() -> int:
    """Concurrent PDF fetches.  Small on purpose.

    README and GRID-RUNBOOK both warn that widening this is rude to a shared
    public archive.  That warning was written as a caution rather than from a
    measurement, and the measurement now exists: ~120,000 requests at ~3/s drew
    zero 429/503/509, and the 4,003 failures in the 2026-08-31 build were
    per-document **403s** -- PDFs indexed but not public, permanent and
    unrelated to pacing.

    Absence of a limit at 3 req/s is not proof that 9 req/s is welcome, so the
    pool does not assume: :class:`_Throttle` shrinks it and inserts delay the
    moment UCSF says 429, which turns a guess about their tolerance into a
    measurement of it.
    """
    import os

    return max(1, int(os.environ.get("VTS_DOCMARKS_FETCH_WORKERS", "3")))


class _Throttle:
    """Concurrency gate that gives ground when the server pushes back.

    Backs off multiplicatively on a rate-limit response and gives up a worker
    each time, so sustained pushback converges on serial fetching rather than
    hammering through it.  Recovers slowly on success, so one 429 does not cost
    the rest of the run.
    """

    def __init__(self, workers: int) -> None:
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(workers)
        self._delay = 0.0
        self.workers = workers
        self.penalties = 0

    def __enter__(self) -> "_Throttle":
        self._sem.acquire()
        with self._lock:
            delay = self._delay
        if delay:
            time.sleep(delay)
        return self

    def __exit__(self, *exc: Any) -> None:
        self._sem.release()

    def penalise(self) -> None:
        with self._lock:
            self.penalties += 1
            self._delay = min(60.0, max(1.0, self._delay * 2))
            if self.workers > 1:
                self.workers -= 1
                self._sem.acquire(blocking=False)

    def reward(self) -> None:
        if self._delay:
            with self._lock:
                self._delay = max(0.0, self._delay * 0.9)


def _render_workers() -> int:
    """How many render processes to run.

    ``os.cpu_count()`` reports the NODE (40 on these boxes), not this job's
    cgroup, so trusting it would oversubscribe an 8-core allocation eightfold
    and steal time from whoever else is on the node.  SLURM's own variable is
    the only honest source.  Capped at 4 because the measured speedup plateaus
    there (3.99x at 4 workers, 3.95x at 8).
    """
    import os

    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE") or 2)
    return max(1, min(4, allocated - 1))


def _render_to_disk(
    pdf_path: str, out_images: str, doc_id: str, dpi: int, max_pages_per_doc: int
) -> list[tuple[int, str, int, int]]:
    """Render one already-downloaded PDF and save its pages.  Runs in a worker.

    Returns ``(page_index, image_path, width, height)`` per page -- deliberately
    small, so no PIL image ever crosses the process boundary.

    The existing-file fast path matters more than it looks: rendering is 0.188 s
    a page, so re-rendering 200k already-saved pages would cost ~10 h on every
    resume of a job that is explicitly designed to be resumable.
    """
    from pathlib import Path as _Path

    out_dir = _Path(out_images)
    targets = [out_dir / f"{doc_id}_{i}.png" for i in range(max_pages_per_doc)]
    if targets and all(t.exists() for t in targets):
        from PIL import Image as _Image

        done = []
        for idx, t in enumerate(targets):
            with _Image.open(t) as im:
                done.append((idx, str(t), im.width, im.height))
        return done

    from vtscore.datasets.pdf import render_pdf_pages

    rendered = render_pdf_pages(_Path(pdf_path), dpi=dpi)
    out: list[tuple[int, str, int, int]] = []
    for idx, (_name, image) in enumerate(rendered[:max_pages_per_doc]):
        image_path = out_dir / f"{doc_id}_{idx}.png"
        if not image_path.exists():
            image.save(image_path)
        out.append((idx, str(image_path), image.width, image.height))
    return out


def fetch_and_render(
    docs: list[dict[str, Any]],
    raw_root: Path,
    out_images: Path,
    *,
    dpi: int = 150,
    letterhead_author: Optional[str] = None,
    band_frac: float = 0.22,
    max_pages_per_doc: int = 1,
    on_error: Optional[Any] = None,
    industry: Optional[str] = None,
) -> list[Page]:
    """Download each doc's PDF, render its pages to PNG, return :class:`Page`\\ s.

    Failures are skipped rather than fatal: at 200k documents a handful of dead
    ids or malformed PDFs is expected, and an aborted pull is far more expensive
    than a slightly short one.  Every skip is reported through *on_error* so the
    count is visible instead of silent.
    """
    import requests

    pdf_dir = raw_root / "ucsf" / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)

    # THE SHAPE OF THIS LOOP IS THE WHOLE POINT.  Measured in steady state on
    # job 602799: 0.327 s/page waiting on UCSF and 0.188 s/page rendering, on
    # one core of eight.  Rendering is local CPU that involves UCSF not at all,
    # so it parallelises with no ethical question attached; fetching does not,
    # and README/GRID-RUNBOOK are explicit that widening it would be rude to a
    # shared public archive and probably get us rate-limited.
    #
    # So: ONE serial fetcher -- UCSF sees exactly the request pattern it saw
    # before this change -- feeding a pool that renders.  Throughput goes from
    # 1.94 to ~3.2 pages/s and the bottleneck becomes the archive's own latency,
    # which is where it should be.  Do not "improve" this by widening the fetch.
    #
    # Processes rather than threads: measured 3.99x vs 3.04x on 48 real PDFs
    # (PyMuPDF and PIL release the GIL, but not enough of the time).  Both
    # plateau at 4 workers, so more is waste.
    workers = _render_workers()
    inflight_cap = max(2 * workers, 4)

    indexed: list[tuple[int, Page]] = []
    order = 0

    def _drain(futures: dict, block_until: int) -> None:
        while len(futures) > block_until:
            done = next(as_completed(futures))
            doc, seq = futures.pop(done)
            doc_id = str(doc.get("id", ""))
            try:
                rendered = done.result()
            except Exception as exc:  # noqa: BLE001 - a dead id must not kill a 200k pull
                if on_error:
                    on_error(doc_id, exc)
                (pdf_dir / f"{doc_id}.pdf").unlink(missing_ok=True)
                continue
            for idx, image_path, width, height in rendered:
                indexed.append(
                    (
                        seq * 100 + idx,
                        doc_to_page(
                            doc,
                            image_path,
                            width,
                            height,
                            page_index=idx,
                            letterhead_author=letterhead_author,
                            band_frac=band_frac,
                            industry=industry,
                        ),
                    )
                )

    throttle = _Throttle(_fetch_workers())
    local = threading.local()

    def _session() -> Any:
        # requests.Session is not documented thread-safe; one per thread costs
        # nothing and keeps keep-alive working within each.
        if not hasattr(local, "s"):
            local.s = requests.Session()
        return local.s

    def _download(doc: dict[str, Any]) -> tuple[dict[str, Any], Optional[str], Optional[Exception]]:
        doc_id = str(doc.get("id", ""))
        if len(doc_id) < 4:
            return doc, None, None
        pdf_path = pdf_dir / f"{doc_id}.pdf"
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            return doc, doc_id, None  # cached: no request, no throttle
        for attempt in range(4):
            try:
                with throttle:
                    _common.http_download(pdf_url(doc_id), pdf_path, session=_session())
                throttle.reward()
                return doc, doc_id, None
            except _common.RateLimited as exc:
                # Not the document's fault -- back off and ask again.  A 403 is
                # permanent and drops through to the branch below; conflating
                # the two would discard documents we merely asked for too fast.
                throttle.penalise()
                if attempt == 3:
                    return doc, None, exc
            except Exception as exc:  # noqa: BLE001 - a dead id must not kill a 200k pull
                pdf_path.unlink(missing_ok=True)
                return doc, None, exc
        return doc, None, None

    with (
        ProcessPoolExecutor(max_workers=workers) as pool,
        ThreadPoolExecutor(max_workers=max(1, _fetch_workers())) as fetchers,
    ):
        futures: dict = {}
        for doc, doc_id, exc in fetchers.map(_download, docs):
            if exc is not None:
                if on_error:
                    on_error(str(doc.get("id", "")), exc)
                continue
            if doc_id is None:
                continue
            pdf_path = pdf_dir / f"{doc_id}.pdf"
            futures[pool.submit(_render_to_disk, str(pdf_path), str(out_images), doc_id, dpi, max_pages_per_doc)] = (
                doc,
                order,
            )
            order += 1
            # Bounded in-flight work: an unbounded submit queue would hold every
            # rendered page's metadata and starve nothing but memory.
            _drain(futures, inflight_cap)
        _drain(futures, 0)

    if throttle.penalties:
        print(f"  ucsf: backed off {throttle.penalties} time(s); fetch workers now {throttle.workers}")

    # Restore document order.  Tier assignment hashes the page id so it does not
    # care, but a manifest that reshuffles between runs is a needless diff and
    # makes two builds hard to compare by eye.
    indexed.sort(key=lambda kv: kv[0])
    return [p for _, p in indexed]
