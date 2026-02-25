"""Dataset downloading utilities.

All public functions accept an optional ``on_progress`` callback with the
signature ``(status: str, message: str, current: int, total: int) -> None``.
When omitted the functions fall back to the application-wide
:func:`~vtsearch.utils.update_progress` reporter; pass an explicit callback
to use these functions outside the Flask app (scripts, notebooks, tests).
"""

import tarfile
import zipfile
from pathlib import Path
from typing import Callable, Optional

import requests

from vtsearch.config import (
    CALTECH101_DOWNLOAD_SIZE_MB,
    CALTECH101_URL,
    CALTECH256_DOWNLOAD_SIZE_MB,
    CALTECH256_URL,
    CIFAR10_DOWNLOAD_SIZE_MB,
    CIFAR10_URL,
    DATA_DIR,
    ESC50_DOWNLOAD_SIZE_MB,
    ESC50_URL,
    IMAGE_DIR,
    UCF101_SUBSET_DOWNLOAD_SIZE_MB,
    UCF101_SUBSET_URL,
    VIDEO_DIR,
)

ProgressCallback = Callable[[str, str, int, int], None]


def _default_progress() -> ProgressCallback:
    """Lazily resolve the application-wide progress callback."""
    from vtsearch.utils import update_progress

    return update_progress


def download_file_with_progress(
    url: str,
    dest_path: Path,
    expected_size: int = 0,
    on_progress: Optional[ProgressCallback] = None,
) -> None:
    """Download a file from a URL to a local path, reporting byte-level progress.

    Streams the HTTP response in 8 KB chunks and calls *on_progress*
    after each chunk so that a polling client can track download progress.

    Args:
        url: The HTTP/HTTPS URL to download from.
        dest_path: Local filesystem path where the downloaded file will be written.
        expected_size: Expected file size in bytes, used as a fallback when the
            server does not supply a ``Content-Length`` header. Pass 0 (default)
            if the size is unknown.
        on_progress: Optional progress callback. Falls back to the
            application-wide ``update_progress`` when ``None``.

    Raises:
        requests.HTTPError: If the server returns a non-2xx status code.
    """
    if on_progress is None:
        on_progress = _default_progress()

    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    if total_size == 0:
        total_size = expected_size

    downloaded = 0
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            downloaded += size
            on_progress("downloading", f"Downloading {dest_path.name}...", downloaded, total_size)


def download_esc50(on_progress: Optional[ProgressCallback] = None) -> Path:
    """Download and extract the ESC-50 environmental sounds dataset.

    Downloads ``esc50.zip`` from the configured ``ESC50_URL`` into ``DATA_DIR``
    if it is not already present, then extracts it and deletes the zip to
    reclaim disk space. Both steps report progress via *on_progress*.

    Args:
        on_progress: Optional progress callback. Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        Path to the ``audio/`` subdirectory inside the extracted ``ESC-50-master``
        directory (e.g. ``data/ESC-50-master/audio``).
    """
    if on_progress is None:
        on_progress = _default_progress()

    zip_path = DATA_DIR / "esc50.zip"
    extract_dir = DATA_DIR / "ESC-50-master"
    DATA_DIR.mkdir(exist_ok=True)

    if not extract_dir.exists():
        if not zip_path.exists():
            on_progress("downloading", "Starting download...", 0, 0)
            download_file_with_progress(ESC50_URL, zip_path, ESC50_DOWNLOAD_SIZE_MB * 1024 * 1024, on_progress)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            members = zip_ref.namelist()
            total = len(members)
            for i, member in enumerate(members, 1):
                on_progress(
                    "downloading",
                    f"Extracting {member.split('/')[-1]}...",
                    i,
                    total,
                )
                zip_ref.extract(member, DATA_DIR)

        zip_path.unlink(missing_ok=True)

    return extract_dir / "audio"


def download_cifar10(on_progress: Optional[ProgressCallback] = None) -> Path:
    """Download and extract the CIFAR-10 image classification dataset.

    Downloads ``cifar-10-python.tar.gz`` from the configured ``CIFAR10_URL``
    into ``DATA_DIR`` if it is not already present, then extracts it and deletes
    the archive to reclaim disk space. Both steps report progress via
    *on_progress*.

    Args:
        on_progress: Optional progress callback. Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        Path to the ``cifar-10-batches-py/`` directory containing the raw pickle
        batch files (e.g. ``data/cifar-10-batches-py``).
    """
    if on_progress is None:
        on_progress = _default_progress()

    tar_path = DATA_DIR / "cifar-10-python.tar.gz"
    extract_dir = DATA_DIR / "cifar-10-batches-py"
    DATA_DIR.mkdir(exist_ok=True)

    if not extract_dir.exists():
        if not tar_path.exists():
            on_progress("downloading", "Starting CIFAR-10 download...", 0, 0)
            download_file_with_progress(CIFAR10_URL, tar_path, CIFAR10_DOWNLOAD_SIZE_MB * 1024 * 1024, on_progress)

        on_progress("downloading", "Extracting CIFAR-10...", 0, 0)
        with tarfile.open(tar_path, "r:gz") as tar_ref:
            tar_ref.extractall(DATA_DIR, filter="data")

        tar_path.unlink(missing_ok=True)

    return extract_dir


def download_caltech101(on_progress: Optional[ProgressCallback] = None) -> Path:
    """Download and extract the Caltech-101 image classification dataset.

    Downloads ``caltech-101.zip`` from the configured ``CALTECH101_URL``
    into ``DATA_DIR`` if it is not already present, then extracts it.
    The zip contains a nested ``101_ObjectCategories.tar.gz`` archive
    which is extracted in a second pass to produce the final category
    directories.  Both archives are deleted after extraction to reclaim
    disk space.

    Args:
        on_progress: Optional progress callback. Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        Path to the ``101_ObjectCategories/`` directory containing category
        subfolders of JPEG images (e.g.
        ``data/caltech-101/101_ObjectCategories``).
    """
    if on_progress is None:
        on_progress = _default_progress()

    zip_path = DATA_DIR / "caltech-101.zip"
    extract_dir = DATA_DIR / "caltech-101"
    DATA_DIR.mkdir(exist_ok=True)
    IMAGE_DIR.mkdir(exist_ok=True, parents=True)

    categories_dir = extract_dir / "101_ObjectCategories"
    if not categories_dir.exists():
        if not zip_path.exists():
            on_progress("downloading", "Starting Caltech-101 download...", 0, 0)
            download_file_with_progress(
                CALTECH101_URL, zip_path, CALTECH101_DOWNLOAD_SIZE_MB * 1024 * 1024, on_progress
            )

        on_progress("downloading", "Extracting Caltech-101 zip...", 0, 0)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            members = zip_ref.namelist()
            total = len(members)
            for i, member in enumerate(members, 1):
                if i % 100 == 0 or i == total:
                    on_progress(
                        "downloading",
                        f"Extracting Caltech-101 zip ({i}/{total})...",
                        i,
                        total,
                    )
                zip_ref.extract(member, DATA_DIR)

        zip_path.unlink(missing_ok=True)

        # The zip contains 101_ObjectCategories.tar.gz (a nested archive).
        # Extract it to produce the actual category directories.
        inner_tar = extract_dir / "101_ObjectCategories.tar.gz"
        if inner_tar.exists() and not categories_dir.exists():
            on_progress("downloading", "Extracting 101_ObjectCategories...", 0, 0)
            with tarfile.open(inner_tar, "r:gz") as tar_ref:
                tar_ref.extractall(extract_dir, filter="data")
            inner_tar.unlink(missing_ok=True)

    return categories_dir


def download_caltech256(on_progress: Optional[ProgressCallback] = None) -> Path:
    """Download and extract the Caltech-256 image classification dataset.

    Downloads ``256_ObjectCategories.tar`` from the configured
    ``CALTECH256_URL`` into ``DATA_DIR`` if it is not already present, then
    extracts it.  The archive is deleted after extraction to reclaim disk
    space.

    Args:
        on_progress: Optional progress callback. Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        Path to the ``256_ObjectCategories/`` directory containing category
        subfolders of JPEG images (e.g.
        ``data/caltech-256/256_ObjectCategories``).
    """
    if on_progress is None:
        on_progress = _default_progress()

    tar_path = DATA_DIR / "256_ObjectCategories.tar"
    extract_dir = DATA_DIR / "caltech-256"
    DATA_DIR.mkdir(exist_ok=True)
    IMAGE_DIR.mkdir(exist_ok=True, parents=True)

    categories_dir = extract_dir / "256_ObjectCategories"
    if not categories_dir.exists():
        if not tar_path.exists():
            on_progress("downloading", "Starting Caltech-256 download...", 0, 0)
            download_file_with_progress(
                CALTECH256_URL, tar_path, CALTECH256_DOWNLOAD_SIZE_MB * 1024 * 1024, on_progress
            )

        on_progress("downloading", "Extracting Caltech-256...", 0, 0)
        with tarfile.open(tar_path, "r:") as tar_ref:
            tar_ref.extractall(extract_dir, filter="data")

        tar_path.unlink(missing_ok=True)

    return categories_dir


def download_ucf101_subset(on_progress: Optional[ProgressCallback] = None) -> Path:
    """Download and extract a UCF-101 subset for video demo datasets.

    Downloads a 171 MB subset of UCF-101 (10 action classes, 405 medias) from
    HuggingFace and extracts it into ``VIDEO_DIR / "ucf101"`` with one
    subdirectory per action class.  Videos from all splits (train/val/test) are
    merged into a single flat category structure so that
    :func:`~vtsearch.datasets.loader.load_video_metadata_from_folders` can
    scan them directly.

    If the dataset is already present (at least one ``*.avi`` in a
    subdirectory), the download is skipped and the existing path is returned.

    Args:
        on_progress: Optional progress callback. Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        Path to the ``ucf101/`` directory inside ``VIDEO_DIR`` (e.g.
        ``data/video/ucf101``), containing category subdirectories with
        ``.avi`` files.
    """
    if on_progress is None:
        on_progress = _default_progress()

    video_dir = VIDEO_DIR / "ucf101"
    VIDEO_DIR.mkdir(exist_ok=True, parents=True)

    # Already downloaded and extracted — nothing to do.
    if video_dir.exists() and any(video_dir.glob("*/*.avi")):
        return video_dir

    tar_path = DATA_DIR / "UCF101_subset.tar.gz"
    extract_dir = DATA_DIR / "UCF101_subset"
    DATA_DIR.mkdir(exist_ok=True)

    # Download the tar.gz if we don't already have the extracted tree.
    if not extract_dir.exists():
        if not tar_path.exists():
            on_progress("downloading", "Starting UCF-101 subset download...", 0, 0)
            download_file_with_progress(
                UCF101_SUBSET_URL,
                tar_path,
                UCF101_SUBSET_DOWNLOAD_SIZE_MB * 1024 * 1024,
                on_progress,
            )

        on_progress("downloading", "Extracting UCF-101 subset...", 0, 0)
        with tarfile.open(tar_path, "r:gz") as tar_ref:
            tar_ref.extractall(DATA_DIR, filter="data")

        tar_path.unlink(missing_ok=True)

    # Flatten the train/val/test splits into VIDEO_DIR/ucf101/<Category>/.
    import shutil

    video_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        split_dir = extract_dir / split
        if not split_dir.is_dir():
            continue
        for category_dir in split_dir.iterdir():
            if not category_dir.is_dir():
                continue
            dest_cat = video_dir / category_dir.name
            dest_cat.mkdir(exist_ok=True)
            for video_file in category_dir.iterdir():
                if video_file.is_file():
                    dest = dest_cat / video_file.name
                    if not dest.exists():
                        shutil.move(str(video_file), str(dest))

    # Clean up the extracted staging directory.
    shutil.rmtree(extract_dir, ignore_errors=True)

    return video_dir


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

            - ``"world"``    → ``"talk.politics.misc"``
            - ``"sports"``   → ``"rec.sport.baseball"``
            - ``"business"`` → ``"misc.forsale"``
            - ``"science"``  → ``"sci.space"``

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
