"""Document dataset downloaders: UCSF Industry Documents."""

from pathlib import Path
from typing import Optional

import requests

from vtscore.datasets.downloader import core as _core
from vtscore.concurrency.progress import resolve_progress_callback
from vtscore.datasets.downloader.core import ProgressCallback


def _describe(exc: Exception) -> str:
    """A download failure in a few words: the HTTP status when there is one, else the exception."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is not None:
        return f"HTTP {status}"
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}"[:200] if text else type(exc).__name__


def download_ucsf_documents(  # noqa: C901
    categories: list[str],
    docs_per_category: int = 25,
    on_progress: Optional[ProgressCallback] = None,
) -> Path:
    """Download UCSF Industry Documents Library PDFs by industry category.

    Queries the UCSF Industry Documents Library Solr API for short documents
    (1-3 pages) within each *category* (industry name), downloads individual
    PDFs, and organises them into category subdirectories under
    ``DATA_DIR / "ucsf_documents"``.

    The PDFs are the ``ucsf_documents`` demo's source media: documents, which
    the demo load converts to page images (``document2image``) or extracted
    text (``document2text``) before embedding.

    Args:
        categories: Industry names recognised by the UCSF IDL Solr index
            (e.g. ``["Tobacco", "Food", "Drug"]``).
        docs_per_category: Maximum number of PDFs to download per category.
        on_progress: Optional progress callback.  Falls back to the
            the calling thread's progress sink when ``None``.

    Returns:
        Path to the ``ucsf_documents/`` directory containing category
        subdirectories with ``.pdf`` files (e.g. ``data/ucsf_documents``).

    Raises:
        RuntimeError: When not one PDF is on disk afterwards, naming what the
            library answered.  A category that comes up short is tolerated
            (the demo is then smaller); nothing at all used to come back as an
            empty folder, and the load ended as "Import produced no medias"
            with no word on why (#5680).
    """
    if on_progress is None:
        on_progress = resolve_progress_callback()

    extract_dir = _core.DATA_DIR / "ucsf_documents"
    _core.DATA_DIR.mkdir(exist_ok=True)

    # Fast-path: if every category already has PDFs, skip the download.
    all_present = True
    for cat in categories:
        cat_dir = extract_dir / cat
        if not cat_dir.exists() or not any(cat_dir.glob("*.pdf")):
            all_present = False
            break

    if all_present:
        return extract_dir

    extract_dir.mkdir(parents=True, exist_ok=True)

    total_docs = len(categories) * docs_per_category
    downloaded = 0
    # The first failure of each kind, for the error raised when nothing lands.
    failures: dict[str, str] = {}

    # Every PDF is a separate transfer, but they all run back-to-back under one
    # continuous "downloading" phase. If each PDF reported its own byte fraction
    # (downloaded/size), the first completed PDF would drive that fraction to
    # 1.0 — filling the load bar's download slice — and the pacer's monotonic
    # clamp would then freeze the bar for every remaining PDF, since each new
    # PDF's byte fraction restarts from 0 on its own scale (issue #2616).
    # Reporting on the cumulative document count instead keeps the reported
    # fraction monotone across the whole set, so the bar advances one PDF at a
    # time and never freezes or rewinds. ``downloaded`` is read at call time, so
    # this closure always reflects how many PDFs have finished so far.
    def _cumulative_progress(status: str, message: str, _current: int, _total: int) -> None:
        on_progress(status, message, downloaded, total_docs)

    for cat in categories:
        cat_dir = extract_dir / cat
        cat_dir.mkdir(exist_ok=True)

        # Skip if this category already has enough PDFs.
        existing_pdfs = list(cat_dir.glob("*.pdf"))
        if len(existing_pdfs) >= docs_per_category:
            downloaded += docs_per_category
            continue

        # Query the Solr API for short document IDs in this industry.
        on_progress("downloading", f"Querying UCSF API for {cat} documents...", downloaded, total_docs)

        # Quote multi-word industry names for the Solr query.
        solr_cat = f'"{cat}"' if " " in cat else cat
        params = {
            "q": f"industry:{solr_cat} AND pages:[1 TO 3]",
            "rows": str(docs_per_category),
            "wt": "json",
            "fl": "id",
            "sort": "id asc",
        }

        try:
            resp = requests.get(_core.UCSF_IDL_API_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            docs = data.get("response", {}).get("docs", [])
        except Exception as exc:
            failures.setdefault("query", f"the index query for {cat!r} failed ({_describe(exc)})")
            docs = []

        # The UCSF IDL Solr endpoint ignores the ``rows`` limit and returns up
        # to 1000 ids per query, so cap client-side; otherwise every category
        # would download ~1000 PDFs instead of ``docs_per_category``.
        docs = docs[:docs_per_category]

        # Download each PDF using the UCSF split-character URL scheme.
        for doc in docs:
            doc_id = doc.get("id", "")
            if not doc_id or len(doc_id) < 4:
                continue

            pdf_path = cat_dir / f"{doc_id}.pdf"
            if pdf_path.exists():
                downloaded += 1
                on_progress("downloading", f"Cached {doc_id}.pdf", downloaded, total_docs)
                continue

            url = f"{_core.UCSF_IDL_DOWNLOAD_URL}/{doc_id[0]}/{doc_id[1]}/{doc_id[2]}/{doc_id[3]}/{doc_id}/{doc_id}.pdf"

            try:
                # Atomic (temp + rename): a partial PDF left at pdf_path by a
                # hard kill would pass the exists() cache gate forever. Progress
                # is reported on the cumulative document count (see
                # _cumulative_progress) so the unified load bar stays monotone
                # across the back-to-back PDF transfers.
                _core.download_file_atomic(url, pdf_path, 0, _cumulative_progress)
                downloaded += 1
                on_progress("downloading", f"Downloaded {doc_id}.pdf", downloaded, total_docs)
            except Exception as exc:
                # One PDF that will not download costs the demo one document.
                failures.setdefault("download", f"downloading {doc_id}.pdf failed ({_describe(exc)})")
                pdf_path.unlink(missing_ok=True)

    if not any(extract_dir.glob("*/*.pdf")):
        why = "; ".join(failures.values()) or "the index listed no documents"
        raise RuntimeError(
            f"No UCSF Industry Documents could be downloaded: {why}. "
            f"The library ({_core.UCSF_IDL_DOWNLOAD_URL}) may be unreachable from this machine or refusing "
            f"automated downloads; PDFs placed under {extract_dir}/<category>/ are used as they are."
        )

    return extract_dir
