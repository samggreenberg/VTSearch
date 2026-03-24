"""Text dataset downloaders: 20 Newsgroups, BBC News, AG News, IMDB."""

import zipfile
from pathlib import Path
from typing import Optional

from vtsearch.config import DATA_DIR
from vtsearch.datasets.downloader.core import (
    AG_NEWS_DOWNLOAD_SIZE_MB,
    AG_NEWS_URL,
    BBC_NEWS_DOWNLOAD_SIZE_MB,
    BBC_NEWS_URL,
    IMDB_DOWNLOAD_SIZE_MB,
    IMDB_URL,
    ProgressCallback,
    _default_progress,
    _download_and_extract,
    download_file_with_progress,
)


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
        on_progress = _default_progress()

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

    # Download the dataset (sklearn handles caching automatically)
    newsgroups = fetch_20newsgroups(
        subset="train",
        categories=newsgroup_categories,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
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


def download_bbc_news(
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
        on_progress = _default_progress()

    zip_path = DATA_DIR / "bbc-fulltext.zip"
    extract_dir = DATA_DIR / "bbc-fulltext"
    DATA_DIR.mkdir(exist_ok=True)

    if not extract_dir.exists():
        if not zip_path.exists():
            on_progress("downloading", "Starting BBC News download...", 0, 0)
            download_file_with_progress(
                BBC_NEWS_URL,
                zip_path,
                BBC_NEWS_DOWNLOAD_SIZE_MB * 1024 * 1024,
                on_progress,
            )

        on_progress("downloading", "Extracting BBC News dataset...", 0, 0)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # The zip may contain a top-level folder (e.g. "bbc/"); extract
            # all members and then locate category directories below.
            members = zip_ref.namelist()
            total = len(members)
            for i, member in enumerate(members):
                if i % 100 == 0 or i == total - 1:
                    on_progress(
                        "downloading",
                        f"Extracting BBC News dataset ({i + 1}/{total})...",
                        i + 1,
                        total,
                    )
                zip_ref.extract(member, DATA_DIR / "bbc-fulltext-raw")

        # Find the directory that contains the category subfolders.
        raw_root = DATA_DIR / "bbc-fulltext-raw"
        _bbc_root = _find_bbc_root(raw_root)
        if _bbc_root is None:
            raise RuntimeError(f"Could not locate BBC News category directories inside {raw_root}")

        import shutil

        shutil.copytree(_bbc_root, extract_dir)
        shutil.rmtree(raw_root, ignore_errors=True)
        zip_path.unlink(missing_ok=True)

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
        on_progress = _default_progress()

    csv_path = DATA_DIR / "ag_news_train.csv"
    DATA_DIR.mkdir(exist_ok=True)

    if not csv_path.exists():
        on_progress("downloading", "Starting AG News download...", 0, 0)
        download_file_with_progress(
            AG_NEWS_URL,
            csv_path,
            AG_NEWS_DOWNLOAD_SIZE_MB * 1024 * 1024,
            on_progress,
        )

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
        on_progress = _default_progress()

    extract_dir = DATA_DIR / "aclImdb"
    _download_and_extract(
        url=IMDB_URL,
        archive_name="aclImdb_v1.tar.gz",
        extract_to=DATA_DIR,
        check_path=extract_dir,
        download_size_mb=IMDB_DOWNLOAD_SIZE_MB,
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
