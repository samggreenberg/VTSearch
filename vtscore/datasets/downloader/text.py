"""Text dataset downloaders: 20 Newsgroups, BBC News, AG News, IMDB,
DBpedia-14 (Wikipedia ontology), arXiv abstracts, Reuters-21578."""

import json
import os
import re
import shutil
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Optional, cast
from urllib.parse import urlencode

import requests

from vtscore.datasets.downloader import core as _core
from vtscore.datasets.downloader.core import ProgressCallback


def download_20newsgroups(
    categories: list[str],
    on_progress: Optional[ProgressCallback] = None,
) -> tuple[list[str], list[int], list[str]]:
    """Download and prepare a subset of the 20 Newsgroups text dataset.

    Uses scikit-learn's :func:`sklearn.datasets.fetch_20newsgroups` (which
    handles caching automatically) to fetch training articles for the requested
    category names. Category names are mapped from simplified labels (e.g.
    ``"science"``) to the full newsgroup names (e.g. ``"sci.space"``) before
    downloading, then mapped back for the returned ``target_names``.

    Args:
        categories: List of simplified category names to include. Recognised
            values and their newsgroup mappings are:

            - ``"world"``    -> ``"talk.politics.misc"``
            - ``"sports"``   -> ``"rec.sport.baseball"``
            - ``"business"`` -> ``"misc.forsale"``
            - ``"science"``  -> ``"sci.space"``

            Any category not in the mapping is passed through unchanged as the
            full newsgroup name.
        on_progress: Optional progress callback. Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        A 3-tuple ``(texts, labels, category_names)`` where:

        - ``texts`` is a list of article strings (headers, footers, and quoted
          text removed).
        - ``labels`` is a list of integer category indices, aligned with
          ``texts``, referencing ``category_names``.
        - ``category_names`` is a list of simplified category name strings,
          ordered to correspond with label index values.
    """
    if on_progress is None:
        on_progress = _core._default_progress()

    from sklearn.datasets import fetch_20newsgroups

    on_progress("downloading", "Downloading 20 Newsgroups dataset...", 0, 0)

    # Map our category names to 20 newsgroups categories.
    # Covers all 20 newsgroups under shorter, friendlier aliases.
    category_mapping = {
        "world": "talk.politics.misc",
        "sports": "rec.sport.baseball",
        "business": "misc.forsale",
        "science": "sci.space",
        "technology": "comp.graphics",
        "medicine": "sci.med",
        "cars": "rec.autos",
        "hockey": "rec.sport.hockey",
        "electronics": "sci.electronics",
        "crypto": "sci.crypt",
        "religion": "soc.religion.christian",
        "guns": "talk.politics.guns",
        "atheism": "alt.atheism",
        "mac": "comp.sys.mac.hardware",
        "pc_hardware": "comp.sys.ibm.pc.hardware",
        "windows": "comp.os.ms-windows.misc",
        "x_windows": "comp.windows.x",
        "motorcycles": "rec.motorcycles",
        "mideast": "talk.politics.mideast",
        "religion_misc": "talk.religion.misc",
    }

    # Get the actual newsgroup categories to download
    newsgroup_categories = [category_mapping.get(cat, cat) for cat in categories]

    # Download the dataset (sklearn handles caching automatically).
    # cast(Any) because the stubs widen the return type to a
    # `Bunch | tuple` union driven by the `return_X_y` overload; with the
    # default `return_X_y=False` we always get a Bunch with `.data`,
    # `.target`, `.target_names`.
    newsgroups = cast(
        Any,
        fetch_20newsgroups(
            subset="train",
            categories=newsgroup_categories,
            remove=("headers", "footers", "quotes"),
            shuffle=True,
            random_state=42,
        ),
    )

    # Map back to our category names
    texts = newsgroups.data
    labels = newsgroups.target
    target_names = [
        list(category_mapping.keys())[list(category_mapping.values()).index(newsgroups.target_names[i])]
        if newsgroups.target_names[i] in category_mapping.values()
        else newsgroups.target_names[i]
        for i in range(len(newsgroups.target_names))
    ]

    return texts, labels, target_names


def download_bbc_news(  # noqa: C901
    on_progress: Optional[ProgressCallback] = None,
) -> dict[str, list[str]]:
    """Download and prepare the BBC News full-text dataset.

    Downloads ``bbc-fulltext.zip`` from the configured ``BBC_NEWS_URL`` into
    ``DATA_DIR`` if it is not already present, then extracts it.  The zip is
    deleted after extraction to reclaim disk space.

    The dataset contains ~2225 articles across five topic categories:
    ``business``, ``entertainment``, ``politics``, ``sport``, and ``tech``.

    Args:
        on_progress: Optional progress callback. Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        A dict mapping category name to a list of article text strings, e.g.
        ``{"business": ["Article text...", ...], "sport": [...], ...}``.
    """
    if on_progress is None:
        on_progress = _core._default_progress()

    extract_dir = _core.DATA_DIR / "bbc-fulltext"
    _core.DATA_DIR.mkdir(exist_ok=True)

    if not extract_dir.exists():
        unique_id = uuid.uuid4().hex[:8]
        temp_archive = _core.DATA_DIR / f".dl_{unique_id}_bbc-fulltext.zip"
        temp_extract = _core.DATA_DIR / f".extract_{unique_id}_bbc-fulltext"

        try:
            on_progress("downloading", "Starting BBC News download...", 0, 0)
            _core.download_file_with_progress(
                _core.BBC_NEWS_URL,
                temp_archive,
                _core.BBC_NEWS_DOWNLOAD_SIZE_MB * 1024 * 1024,
                on_progress,
            )

            if not extract_dir.exists():
                on_progress("downloading", "Extracting BBC News dataset...", 0, 0)
                raw_dir = temp_extract / "raw"
                raw_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(temp_archive, "r") as zip_ref:
                    # The zip may contain a top-level folder (e.g. "bbc/"); extract
                    # all members and then locate category directories below.
                    members = zip_ref.namelist()
                    total = len(members)
                    for i, member in enumerate(members):
                        if i % 100 == 0 or i == total - 1:
                            on_progress(
                                "downloading",
                                "Extracting BBC News dataset...",
                                i + 1,
                                total,
                            )
                        zip_ref.extract(member, raw_dir)

                # Find the directory that contains the category subfolders.
                _bbc_root = _find_bbc_root(raw_dir)
                if _bbc_root is None:
                    raise RuntimeError(f"Could not locate BBC News category directories inside {raw_dir}")

                if not extract_dir.exists():
                    try:
                        shutil.copytree(_bbc_root, extract_dir)
                    except FileExistsError:
                        pass  # Another download finished first
        finally:
            temp_archive.unlink(missing_ok=True)
            if temp_extract.exists():
                shutil.rmtree(temp_extract, ignore_errors=True)

    # Read articles grouped by category directory name.
    categories_articles: dict[str, list[str]] = {}
    for category_dir in sorted(extract_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        articles: list[str] = []
        for txt_file in sorted(category_dir.glob("*.txt")):
            try:
                text = txt_file.read_text(encoding="utf-8", errors="replace").strip()
            except Exception:
                continue
            if text:
                articles.append(text)
        if articles:
            categories_articles[category_dir.name] = articles

    return categories_articles


def download_ag_news(
    on_progress: Optional[ProgressCallback] = None,
) -> dict[str, list[str]]:
    """Download and prepare the AG News text classification dataset.

    Downloads the AG News training CSV into ``DATA_DIR`` if it is not already
    present.  The CSV has no header row; each line is
    ``"class_index","title","description"`` where class_index is 1-4:

    1 = World, 2 = Sports, 3 = Business, 4 = Sci/Tech.

    Title and description are concatenated into a single article string.

    Args:
        on_progress: Optional progress callback. Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        A dict mapping category name to a list of article text strings, e.g.
        ``{"World": ["Article text...", ...], "Sports": [...], ...}``.
    """
    import csv  # noqa: PLC0415

    if on_progress is None:
        on_progress = _core._default_progress()

    csv_path = _core.DATA_DIR / "ag_news_train.csv"
    _core.DATA_DIR.mkdir(exist_ok=True)

    if not csv_path.exists():
        unique_id = uuid.uuid4().hex[:8]
        temp_path = _core.DATA_DIR / f".dl_{unique_id}_ag_news_train.csv"
        try:
            on_progress("downloading", "Starting AG News download...", 0, 0)
            _core.download_file_with_progress(
                _core.AG_NEWS_URL,
                temp_path,
                _core.AG_NEWS_DOWNLOAD_SIZE_MB * 1024 * 1024,
                on_progress,
            )
            if not csv_path.exists():
                try:
                    os.rename(temp_path, csv_path)
                except OSError:
                    pass  # Another download finished first
        finally:
            temp_path.unlink(missing_ok=True)

    label_to_category = {
        "1": "World",
        "2": "Sports",
        "3": "Business",
        "4": "Sci/Tech",
    }

    categories_articles: dict[str, list[str]] = {}
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            class_idx, title, description = row[0], row[1], row[2]
            category = label_to_category.get(class_idx)
            if category is None:
                continue
            # Combine title and description into one article string.
            text = f"{title.strip()} {description.strip()}".strip()
            if text:
                categories_articles.setdefault(category, []).append(text)

    return categories_articles


def download_imdb(
    on_progress: Optional[ProgressCallback] = None,
) -> dict[str, list[str]]:
    """Download and prepare the Stanford IMDB Large Movie Review dataset.

    Downloads ``aclImdb_v1.tar.gz`` from the configured ``IMDB_URL`` into
    ``DATA_DIR`` if it is not already present, then extracts it.  The archive
    is deleted after extraction to reclaim disk space.

    The dataset contains 50 000 movie reviews split evenly into positive
    (``pos``) and negative (``neg``) sentiment categories, with 25 000
    reviews in each of the ``train`` and ``test`` splits.  Both splits are
    merged so the caller can slice freely.

    Args:
        on_progress: Optional progress callback. Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        A dict mapping category name to a list of review text strings, e.g.
        ``{"pos": ["Great film...", ...], "neg": ["Terrible...", ...]}``.
    """
    if on_progress is None:
        on_progress = _core._default_progress()

    extract_dir = _core.DATA_DIR / "aclImdb"
    _core._download_and_extract(
        url=_core.IMDB_URL,
        archive_name="aclImdb_v1.tar.gz",
        extract_to=_core.DATA_DIR,
        check_path=extract_dir,
        download_size_mb=_core.IMDB_DOWNLOAD_SIZE_MB,
        dataset_name="IMDB",
        on_progress=on_progress,
    )

    # Read reviews grouped by sentiment category, merging train + test splits.
    categories_reviews: dict[str, list[str]] = {}
    for sentiment in ("pos", "neg"):
        reviews: list[str] = []
        for split in ("train", "test"):
            split_dir = extract_dir / split / sentiment
            if not split_dir.is_dir():
                continue
            for txt_file in sorted(split_dir.glob("*.txt")):
                try:
                    text = txt_file.read_text(encoding="utf-8", errors="replace").strip()
                except Exception:
                    continue
                if text:
                    reviews.append(text)
        if reviews:
            categories_reviews[sentiment] = reviews

    return categories_reviews


# ---------------------------------------------------------------------------
# DBpedia-14 (Wikipedia ontology topics)
# ---------------------------------------------------------------------------

# Fast.ai's mirror of the DBpedia-14 ontology classification dataset
# (Zhang, Zhao & LeCun, 2015). Each row is a Wikipedia abstract labelled
# with one of 14 ontology classes derived from Wikipedia's infobox data.
DBPEDIA14_CLASSES = [
    "Company",
    "EducationalInstitution",
    "Artist",
    "Athlete",
    "OfficeHolder",
    "MeanOfTransportation",
    "Building",
    "NaturalPlace",
    "Village",
    "Animal",
    "Plant",
    "Album",
    "Film",
    "WrittenWork",
]


def download_dbpedia(
    on_progress: Optional[ProgressCallback] = None,
) -> dict[str, list[str]]:
    """Download and prepare the DBpedia-14 Wikipedia-ontology dataset.

    The tarball at :data:`~vtscore.datasets.downloader.core.DBPEDIA_URL`
    is downloaded into ``DATA_DIR`` if it isn't already present, then
    extracted into ``DATA_DIR / "dbpedia_csv"``.  The extracted directory
    contains ``train.csv`` and ``test.csv`` with rows of the form
    ``class_index,"title","abstract"`` where ``class_index`` is 1-14.

    Both splits are concatenated so the caller can slice freely.  The
    abstract often begins with a leading space and an escaped ``\\n``
    sequence from the upstream CSV — both are cleaned up before returning.

    Args:
        on_progress: Optional progress callback.  Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        A dict mapping ontology class name (e.g. ``"Company"``,
        ``"Animal"``) to a list of ``title + abstract`` strings.
    """
    import csv  # noqa: PLC0415

    if on_progress is None:
        on_progress = _core._default_progress()

    extract_dir = _core.DATA_DIR / "dbpedia_csv"
    _core._download_and_extract(
        url=_core.DBPEDIA_URL,
        archive_name="dbpedia_csv.tgz",
        extract_to=_core.DATA_DIR,
        check_path=extract_dir,
        download_size_mb=_core.DBPEDIA_DOWNLOAD_SIZE_MB,
        dataset_name="DBpedia-14",
        on_progress=on_progress,
    )

    classes_path = extract_dir / "classes.txt"
    class_names: list[str]
    if classes_path.exists():
        class_names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        class_names = list(DBPEDIA14_CLASSES)

    categories_articles: dict[str, list[str]] = {}
    for split in ("train.csv", "test.csv"):
        csv_path = extract_dir / split
        if not csv_path.exists():
            continue
        with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:
                    continue
                try:
                    class_idx = int(row[0])
                except ValueError:
                    continue
                if class_idx < 1 or class_idx > len(class_names):
                    continue
                category = class_names[class_idx - 1]
                title = row[1].strip()
                abstract = row[2].replace("\\n", " ").strip()
                text = f"{title} {abstract}".strip()
                if text:
                    categories_articles.setdefault(category, []).append(text)

    return categories_articles


# ---------------------------------------------------------------------------
# arXiv abstracts (export API)
# ---------------------------------------------------------------------------

# Default arXiv subject categories spanning CS, math, physics, biology,
# astrophysics, and statistics — enough variety to be a useful
# "multilingual scientific search" demo without an overwhelming download.
ARXIV_DEFAULT_CATEGORIES = [
    "cs.AI",
    "cs.CV",
    "cs.LG",
    "cs.CL",
    "cs.CR",
    "math.AG",
    "math.CO",
    "math.PR",
    "physics.gen-ph",
    "q-bio.GN",
    "astro-ph.CO",
    "stat.ML",
]

# arXiv asks API clients to be polite: keep batches small and sleep between
# requests so we don't hammer their export endpoint.
_ARXIV_BATCH_SIZE = 200
_ARXIV_REQUEST_SLEEP_S = 3.0
_ARXIV_TIMEOUT_S = 30
_ARXIV_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _parse_arxiv_feed(xml_bytes: bytes) -> list[str]:
    """Extract ``title + summary`` text from an arXiv Atom feed."""
    import xml.etree.ElementTree as ET  # noqa: PLC0415

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []

    abstracts: list[str] = []
    for entry in root.findall("atom:entry", _ARXIV_ATOM_NS):
        title_el = entry.find("atom:title", _ARXIV_ATOM_NS)
        summary_el = entry.find("atom:summary", _ARXIV_ATOM_NS)
        title = (title_el.text or "").strip() if title_el is not None else ""
        summary = (summary_el.text or "").strip() if summary_el is not None else ""
        # arXiv wraps abstracts to ~80 cols — collapse whitespace.
        title = re.sub(r"\s+", " ", title)
        summary = re.sub(r"\s+", " ", summary)
        text = f"{title} {summary}".strip()
        if text:
            abstracts.append(text)
    return abstracts


def download_arxiv_abstracts(  # noqa: C901
    categories: Optional[list[str]] = None,
    max_per_category: int = 2000,
    on_progress: Optional[ProgressCallback] = None,
) -> dict[str, list[str]]:
    """Download arXiv abstracts via the export API, grouped by primary category.

    Fetches up to ``max_per_category`` abstracts for each category by paging
    the public arXiv export API at
    :data:`~vtscore.datasets.downloader.core.ARXIV_API_URL`.  Results are
    cached to a JSON file in ``DATA_DIR`` so subsequent loads avoid re-hitting
    the API.

    A 3-second sleep separates requests, per the arXiv API access policy.

    Args:
        categories: List of arXiv subject categories to fetch (e.g.
            ``["cs.AI", "math.AG"]``).  Defaults to
            :data:`ARXIV_DEFAULT_CATEGORIES`.
        max_per_category: Soft cap on the number of abstracts retrieved
            per category.  The API may return fewer if a category has less
            content; this just bounds the work.
        on_progress: Optional progress callback.  Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        A dict mapping category name to a list of ``title + abstract``
        strings, e.g. ``{"cs.AI": ["Deep Learning...", ...], ...}``.
    """
    if on_progress is None:
        on_progress = _core._default_progress()

    cats = list(categories) if categories else list(ARXIV_DEFAULT_CATEGORIES)

    _core.DATA_DIR.mkdir(exist_ok=True)
    # Cache key includes the chosen categories + cap so different presets
    # don't clobber each other.
    cache_path = _core.DATA_DIR / "arxiv_abstracts.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cached = None
        if isinstance(cached, dict) and set(cats).issubset(cached.keys()):
            return {cat: list(cached.get(cat, [])) for cat in cats}

    total_target = max_per_category * len(cats)
    fetched_total = 0
    out: dict[str, list[str]] = {}

    for cat in cats:
        cat_abstracts: list[str] = []
        offset = 0
        while offset < max_per_category:
            limit = min(_ARXIV_BATCH_SIZE, max_per_category - offset)
            params = {
                "search_query": f"cat:{cat}",
                "start": offset,
                "max_results": limit,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            url = f"{_core.ARXIV_API_URL}?{urlencode(params)}"
            on_progress(
                "downloading",
                f"Fetching arXiv {cat} ({offset + limit}/{max_per_category})...",
                fetched_total,
                total_target,
            )
            try:
                resp = requests.get(url, timeout=_ARXIV_TIMEOUT_S)
                resp.raise_for_status()
            except requests.RequestException:
                break

            batch = _parse_arxiv_feed(resp.content)
            if not batch:
                break
            cat_abstracts.extend(batch)
            fetched_total += len(batch)
            offset += limit
            if len(batch) < limit:
                break  # No more results for this category.
            time.sleep(_ARXIV_REQUEST_SLEEP_S)

        out[cat] = cat_abstracts

    try:
        cache_path.write_text(json.dumps(out), encoding="utf-8")
    except OSError:
        pass

    return out


# ---------------------------------------------------------------------------
# Reuters-21578
# ---------------------------------------------------------------------------

# Reuters-21578 ships with ~135 TOPICS categories but most are sparsely
# populated.  These ten cover the vast majority of labelled documents
# and are the canonical "ModApté" subset used in classification papers.
REUTERS21578_TOP_TOPICS = [
    "earn",
    "acq",
    "money-fx",
    "grain",
    "crude",
    "trade",
    "interest",
    "ship",
    "wheat",
    "corn",
]

_REUTERS_REUTERS_RE = re.compile(rb"<REUTERS\b[^>]*>(.*?)</REUTERS>", re.DOTALL)
_REUTERS_TOPICS_RE = re.compile(rb"<TOPICS>(.*?)</TOPICS>", re.DOTALL)
_REUTERS_BODY_RE = re.compile(rb"<BODY>(.*?)</BODY>", re.DOTALL)
_REUTERS_TITLE_RE = re.compile(rb"<TITLE>(.*?)</TITLE>", re.DOTALL)
_REUTERS_D_RE = re.compile(rb"<D>(.*?)</D>", re.DOTALL)


def download_reuters21578(
    on_progress: Optional[ProgressCallback] = None,
) -> dict[str, list[str]]:
    """Download and parse the Reuters-21578 ModApté text classification corpus.

    Fetches the original UCI tarball at
    :data:`~vtscore.datasets.downloader.core.REUTERS21578_URL` into
    ``DATA_DIR`` and extracts it into ``DATA_DIR / "reuters21578"``.  The
    archive contains a series of ``reut2-*.sgm`` SGML files; each
    ``<REUTERS>`` block carries one news story plus its ``<TOPICS>`` labels.

    A document is included once per TOPIC it carries.  Documents with no
    ``<BODY>`` or no TOPICS labels are skipped — these are the original
    "ModApté" filtering rules.

    Args:
        on_progress: Optional progress callback.  Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        A dict mapping TOPIC name (e.g. ``"earn"``, ``"acq"``) to a list of
        ``title + body`` strings.
    """
    if on_progress is None:
        on_progress = _core._default_progress()

    extract_dir = _core.DATA_DIR / "reuters21578"
    _core._download_and_extract(
        url=_core.REUTERS21578_URL,
        archive_name="reuters21578.tar.gz",
        extract_to=extract_dir,
        check_path=extract_dir,
        download_size_mb=_core.REUTERS21578_DOWNLOAD_SIZE_MB,
        dataset_name="Reuters-21578",
        on_progress=on_progress,
    )

    sgm_files = sorted(extract_dir.glob("reut2-*.sgm"))
    categories_articles: dict[str, list[str]] = {}

    for sgm_path in sgm_files:
        try:
            raw = sgm_path.read_bytes()
        except OSError:
            continue
        for block in _REUTERS_REUTERS_RE.findall(raw):
            topics_match = _REUTERS_TOPICS_RE.search(block)
            if not topics_match:
                continue
            topics = [
                t.decode("latin-1", errors="replace").strip() for t in _REUTERS_D_RE.findall(topics_match.group(1))
            ]
            topics = [t for t in topics if t]
            if not topics:
                continue
            body_match = _REUTERS_BODY_RE.search(block)
            if not body_match:
                continue
            body = body_match.group(1).decode("latin-1", errors="replace").strip()
            # Strip the conventional " Reuter\x03" trailer.
            body = re.sub(r"\s*Reuter\s*\x03?\s*$", "", body).strip()
            title_match = _REUTERS_TITLE_RE.search(block)
            title = title_match.group(1).decode("latin-1", errors="replace").strip() if title_match else ""
            text = f"{title} {body}".strip()
            if not text:
                continue
            for topic in topics:
                categories_articles.setdefault(topic, []).append(text)

    return categories_articles


def _find_bbc_root(directory: Path) -> Optional[Path]:
    """Return the first directory under *directory* that contains BBC category subfolders."""
    _BBC_CATEGORIES = {"business", "entertainment", "politics", "sport", "tech"}
    # Check the directory itself first.
    subdirs = {p.name for p in directory.iterdir() if p.is_dir()}
    if subdirs & _BBC_CATEGORIES:
        return directory
    # One level of nesting (common when the zip has a top-level folder).
    for child in directory.iterdir():
        if child.is_dir():
            grandchildren = {p.name for p in child.iterdir() if p.is_dir()}
            if grandchildren & _BBC_CATEGORIES:
                return child
    return None
