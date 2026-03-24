"""Document dataset downloaders: UCSF Industry Documents."""

from pathlib import Path
from typing import Optional

import requests

from vtsearch.datasets.downloader import core as _core
from vtsearch.datasets.downloader.core import ProgressCallback


def download_ucsf_documents(
    categories: list[str],
    docs_per_category: int = 25,
    on_progress: Optional[ProgressCallback] = None,
) -> Path:
    """Download UCSF Industry Documents Library PDFs by industry category.

    Queries the UCSF Industry Documents Library Solr API for short documents
    (1-3 pages) within each *category* (industry name), downloads individual
    PDFs, and organises them into category subdirectories under
    ``DATA_DIR / "ucsf_documents"``.

    Each PDF can later be rendered to page images with
    :func:`~vtsearch.datasets.pdf.render_pdf_pages` for use as an image demo
    dataset.

    Args:
        categories: Industry names recognised by the UCSF IDL Solr index
            (e.g. ``["Tobacco", "Food", "Drug"]``).
        docs_per_category: Maximum number of PDFs to download per category.
        on_progress: Optional progress callback.  Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        Path to the ``ucsf_documents/`` directory containing category
        subdirectories with ``.pdf`` files (e.g. ``data/ucsf_documents``).
    """
    if on_progress is None:
        on_progress = _core._default_progress()

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
        except Exception:
            docs = []

        # Download each PDF using the UCSF split-character URL scheme.
        for doc in docs:
            doc_id = doc.get("id", "")
            if not doc_id or len(doc_id) < 4:
                continue

            pdf_path = cat_dir / f"{doc_id}.pdf"
            if pdf_path.exists():
                downloaded += 1
                on_progress("downloading", f"Cached {doc_id}.pdf ({downloaded}/{total_docs})", downloaded, total_docs)
                continue

            url = (
                f"{_core.UCSF_IDL_DOWNLOAD_URL}/{doc_id[0]}/{doc_id[1]}/{doc_id[2]}/{doc_id[3]}"
                f"/{doc_id}/{doc_id}.pdf"
            )

            try:
                _core.download_file_with_progress(url, pdf_path, 0, on_progress)
                downloaded += 1
                on_progress(
                    "downloading", f"Downloaded {doc_id}.pdf ({downloaded}/{total_docs})", downloaded, total_docs
                )
            except Exception:
                # Skip failed downloads silently.
                pdf_path.unlink(missing_ok=True)

    return extract_dir
