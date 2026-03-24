"""Dataset downloading utilities.

All public functions accept an optional ``on_progress`` callback with the
signature ``(status: str, message: str, current: int, total: int) -> None``.
When omitted the functions fall back to the application-wide
:func:`~vtsearch.utils.update_progress` reporter; pass an explicit callback
to use these functions outside the Flask app (scripts, notebooks, tests).
"""

import os
import shutil
import tarfile
import uuid
import zipfile
from pathlib import Path
from typing import Callable, Optional

import requests

from vtsearch.config import DATA_DIR

# Demo dataset directory paths (derived from DATA_DIR)
IMAGE_DIR = DATA_DIR / "images"
VIDEO_DIR = DATA_DIR / "video"

# Demo dataset URLs
ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
SAMPLE_VIDEOS_URL = "https://github.com/sample-datasets/video-clips/archive/refs/heads/main.zip"
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CALTECH101_URL = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip"
CALTECH256_URL = "https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar?download=1"
UCF101_SUBSET_URL = "https://huggingface.co/datasets/sayakpaul/ucf101-subset/resolve/main/UCF101_subset.tar.gz"
BBC_NEWS_URL = "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip"
AG_NEWS_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
GTZAN_URL = "https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz"
SPEECH_COMMANDS_V2_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
URBANSOUND8K_URL = "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz"
OXFORD_FLOWERS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
OXFORD_FLOWERS_LABELS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
FOOD101_URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
EUROSAT_URL = "https://huggingface.co/datasets/blanchon/EuroSAT_RGB/resolve/main/EuroSAT_RGB.zip"
STANFORD_DOGS_URL = "http://vision.stanford.edu/aditya86/ImageNetDogDataset/images.tar"
UCSF_IDL_API_URL = "https://metadata.idl.ucsf.edu/solr/ltdl3/query"
UCSF_IDL_DOWNLOAD_URL = "https://download.industrydocuments.ucsf.edu"

# Demo dataset download size estimates (MB)
ESC50_DOWNLOAD_SIZE_MB = 600
SAMPLE_VIDEOS_DOWNLOAD_SIZE_MB = 150
CIFAR10_DOWNLOAD_SIZE_MB = 170
CALTECH101_DOWNLOAD_SIZE_MB = 131
CALTECH256_DOWNLOAD_SIZE_MB = 1200
UCF101_SUBSET_DOWNLOAD_SIZE_MB = 171
BBC_NEWS_DOWNLOAD_SIZE_MB = 2
AG_NEWS_DOWNLOAD_SIZE_MB = 30
IMDB_DOWNLOAD_SIZE_MB = 84
GTZAN_DOWNLOAD_SIZE_MB = 1200
SPEECH_COMMANDS_V2_DOWNLOAD_SIZE_MB = 2300
URBANSOUND8K_DOWNLOAD_SIZE_MB = 6000
OXFORD_FLOWERS_DOWNLOAD_SIZE_MB = 330
FOOD101_DOWNLOAD_SIZE_MB = 5000
EUROSAT_DOWNLOAD_SIZE_MB = 90
STANFORD_DOGS_DOWNLOAD_SIZE_MB = 750
UCSF_IDL_DOWNLOAD_SIZE_MB = 50

ProgressCallback = Callable[[str, str, int, int], None]


def _default_progress() -> ProgressCallback:
    """Lazily resolve the progress callback for the current thread."""
    from vtsearch.utils.progress import get_thread_progress

    cb = get_thread_progress()
    if cb is not None:
        return cb
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


_GZIP_MAGIC = b"\x1f\x8b"
_ZIP_MAGIC = b"PK"
# Uncompressed tar: first 257 bytes contain "ustar" at offset 257,
# but a simpler heuristic is that the file does NOT start with common
# non-archive signatures (HTML, JSON, plain text error pages).
_HTML_SIGNATURES = (b"<", b"<!",  b"{")


def _validate_archive(archive_path: Path, archive_name: str, dataset_name: str) -> None:
    """Check that a downloaded file looks like a genuine archive.

    Deletes the file and raises ``RuntimeError`` with a user-friendly
    message when the content does not match the expected format.
    """
    suffix = archive_name.lower()
    try:
        header = archive_path.read_bytes()[:4]
    except OSError:
        return  # file vanished – let the caller deal with it

    ok = True
    if suffix.endswith((".tar.gz", ".tgz")):
        # Accept genuine gzip OR a raw tar that the CDN decompressed on the
        # fly (HuggingFace Xet storage does this).
        ok = header[:2] == _GZIP_MAGIC or tarfile.is_tarfile(archive_path)
    elif suffix.endswith(".zip"):
        ok = header[:2] == _ZIP_MAGIC
    elif suffix.endswith(".tar"):
        # For plain tar we just check it doesn't look like HTML/text
        ok = not any(header.startswith(sig) for sig in _HTML_SIGNATURES)

    if not ok:
        archive_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download {dataset_name}: the server returned an invalid file "
            f"instead of the expected archive. This usually means the download URL is "
            f"temporarily unavailable or has changed. Please try again later."
        )


def _move_tree_contents(src: Path, dst: Path) -> None:
    """Move all children of *src* into *dst*, skipping already-existing targets."""
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if not target.exists():
            try:
                child.rename(target)
            except OSError:
                if child.is_dir():
                    shutil.copytree(child, target)
                else:
                    shutil.copy2(child, target)


def _download_and_extract(
    *,
    url: str,
    archive_name: str,
    extract_to: Path,
    check_path: Path,
    download_size_mb: int,
    dataset_name: str,
    on_progress: ProgressCallback,
) -> None:
    """Download an archive and extract it if *check_path* does not already exist.

    Supports ``.tar.gz`` / ``.tgz`` (gzip tar), ``.tar`` (uncompressed tar),
    and ``.zip`` archives.  The archive file is deleted after successful
    extraction to reclaim disk space.

    Each invocation downloads and extracts into unique temporary paths so that
    concurrent calls targeting the same archive do not interfere with each
    other.  After extraction the content is moved to the final location; if
    another call finished first the duplicate is simply cleaned up.

    Args:
        url: Download URL for the archive.
        archive_name: Filename to save the downloaded archive as inside
            ``DATA_DIR`` (e.g. ``"genres.tar.gz"``).
        extract_to: Directory into which the archive contents are extracted.
        check_path: Path whose existence signals that extraction is already
            complete (often the same as *extract_to* or a subdirectory of it).
        download_size_mb: Expected download size in megabytes (for progress).
        dataset_name: Human-readable dataset name used in progress messages.
        on_progress: Progress callback.
    """
    if check_path.exists():
        return

    unique_id = uuid.uuid4().hex[:8]
    temp_archive = DATA_DIR / f".dl_{unique_id}_{archive_name}"
    temp_extract = extract_to.parent / f".extract_{unique_id}_{extract_to.name}"
    DATA_DIR.mkdir(exist_ok=True)

    try:
        on_progress("downloading", f"Starting {dataset_name} download...", 0, 0)
        download_file_with_progress(url, temp_archive, download_size_mb * 1024 * 1024, on_progress)

        # Another download may have finished while we were downloading.
        if check_path.exists():
            return

        # Validate the downloaded file looks like a real archive before trying
        # to extract it.  A common failure mode is the server returning an HTML
        # error page (e.g. 404/503) which gets saved with a .tar.gz extension.
        _validate_archive(temp_archive, archive_name, dataset_name)

        on_progress("downloading", f"Extracting {dataset_name}...", 0, 0)
        temp_extract.mkdir(parents=True, exist_ok=True)

        suffix = archive_name.lower()
        if suffix.endswith((".tar.gz", ".tgz")):
            # Use "r:*" to auto-detect compression — some CDNs (e.g. HuggingFace
            # Xet) transparently decompress .tar.gz files during transfer.
            with tarfile.open(temp_archive, "r:*") as tar_ref:
                members = tar_ref.getmembers()
                total = len(members)
                for i, member in enumerate(members):
                    if i % 100 == 0 or i == total - 1:
                        on_progress(
                            "downloading", f"Extracting {dataset_name} ({i + 1}/{total})...", i + 1, total
                        )
                    tar_ref.extract(member, temp_extract, filter="data")
        elif suffix.endswith(".tar"):
            with tarfile.open(temp_archive, "r:") as tar_ref:
                members = tar_ref.getmembers()
                total = len(members)
                for i, member in enumerate(members):
                    if i % 100 == 0 or i == total - 1:
                        on_progress(
                            "downloading", f"Extracting {dataset_name} ({i + 1}/{total})...", i + 1, total
                        )
                    tar_ref.extract(member, temp_extract, filter="data")
        elif suffix.endswith(".zip"):
            with zipfile.ZipFile(temp_archive, "r") as zip_ref:
                members = zip_ref.namelist()
                total = len(members)
                for i, member in enumerate(members):
                    if i % 100 == 0 or i == total - 1:
                        on_progress(
                            "downloading", f"Extracting {dataset_name} ({i + 1}/{total})...", i + 1, total
                        )
                    # Guard against path traversal in zip entries
                    member_path = Path(temp_extract) / member
                    if not str(member_path.resolve()).startswith(str(Path(temp_extract).resolve())):
                        raise ValueError(f"Path traversal detected in archive: {member}")
                    zip_ref.extract(member, temp_extract)
        else:
            raise ValueError(f"Unsupported archive format: {archive_name}")

        # Another download may have finished while we were extracting.
        if check_path.exists():
            return

        # Move extracted content to final location.
        if not extract_to.exists():
            try:
                os.rename(temp_extract, extract_to)
                return
            except OSError:
                pass  # extract_to appeared between check and rename (race)

        # extract_to already existed (e.g. it is DATA_DIR) — move children.
        _move_tree_contents(temp_extract, extract_to)
    finally:
        temp_archive.unlink(missing_ok=True)
        if temp_extract.exists():
            shutil.rmtree(temp_extract, ignore_errors=True)


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

    extract_dir = DATA_DIR / "ESC-50-master"
    _download_and_extract(
        url=ESC50_URL,
        archive_name="esc50.zip",
        extract_to=DATA_DIR,
        check_path=extract_dir,
        download_size_mb=ESC50_DOWNLOAD_SIZE_MB,
        dataset_name="ESC-50",
        on_progress=on_progress,
    )
    return extract_dir / "audio"


def download_gtzan(on_progress: Optional[ProgressCallback] = None) -> Path:
    """Download and extract the GTZAN Music Genre Classification dataset.

    Downloads ``genres.tar.gz`` from HuggingFace into ``DATA_DIR`` if it is
    not already present, then extracts it.  The archive is deleted after
    extraction to reclaim disk space.

    The dataset contains 1000 30-second audio tracks across 10 music genres
    (100 tracks per genre), stored as ``.wav`` files in genre subdirectories.

    Args:
        on_progress: Optional progress callback.  Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        Path to the ``genres/`` directory containing genre subdirectories
        with ``.wav`` files (e.g. ``data/gtzan/genres``).
    """
    if on_progress is None:
        on_progress = _default_progress()

    genres_dir = DATA_DIR / "gtzan" / "genres"
    _download_and_extract(
        url=GTZAN_URL,
        archive_name="genres.tar.gz",
        extract_to=DATA_DIR / "gtzan",
        check_path=genres_dir,
        download_size_mb=GTZAN_DOWNLOAD_SIZE_MB,
        dataset_name="GTZAN",
        on_progress=on_progress,
    )
    return genres_dir


def download_speech_commands_v2(on_progress: Optional[ProgressCallback] = None) -> Path:
    """Download and extract the Google Speech Commands v2 dataset.

    Downloads ``speech_commands_v0.02.tar.gz`` from TensorFlow into
    ``DATA_DIR`` if it is not already present, then extracts it.  The archive
    is deleted after extraction to reclaim disk space.

    The dataset contains ~105 000 one-second ``.wav`` utterances of 35
    keywords, each stored in a keyword subdirectory.

    Args:
        on_progress: Optional progress callback.  Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        Path to the ``speech_commands_v2/`` directory containing keyword
        subdirectories with ``.wav`` files (e.g.
        ``data/speech_commands_v2``).
    """
    if on_progress is None:
        on_progress = _default_progress()

    extract_dir = DATA_DIR / "speech_commands_v2"
    _download_and_extract(
        url=SPEECH_COMMANDS_V2_URL,
        archive_name="speech_commands_v0.02.tar.gz",
        extract_to=extract_dir,
        check_path=extract_dir,
        download_size_mb=SPEECH_COMMANDS_V2_DOWNLOAD_SIZE_MB,
        dataset_name="Speech Commands v2",
        on_progress=on_progress,
    )
    return extract_dir


def download_urbansound8k(on_progress: Optional[ProgressCallback] = None) -> Path:
    """Download and extract the UrbanSound8K dataset.

    Downloads ``UrbanSound8K.tar.gz`` from Zenodo into ``DATA_DIR`` if it is
    not already present, then extracts it.  The archive is deleted after
    extraction to reclaim disk space.

    The dataset contains 8732 labeled sound excerpts across 10 urban sound
    classes, organized in numbered fold directories with a metadata CSV.

    Args:
        on_progress: Optional progress callback.  Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        Path to the ``UrbanSound8K/`` directory containing ``audio/`` and
        ``metadata/`` subdirectories (e.g. ``data/UrbanSound8K``).
    """
    if on_progress is None:
        on_progress = _default_progress()

    extract_dir = DATA_DIR / "UrbanSound8K"
    _download_and_extract(
        url=URBANSOUND8K_URL,
        archive_name="UrbanSound8K.tar.gz",
        extract_to=DATA_DIR,
        check_path=extract_dir,
        download_size_mb=URBANSOUND8K_DOWNLOAD_SIZE_MB,
        dataset_name="UrbanSound8K",
        on_progress=on_progress,
    )
    return extract_dir


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

    extract_dir = DATA_DIR / "cifar-10-batches-py"
    _download_and_extract(
        url=CIFAR10_URL,
        archive_name="cifar-10-python.tar.gz",
        extract_to=DATA_DIR,
        check_path=extract_dir,
        download_size_mb=CIFAR10_DOWNLOAD_SIZE_MB,
        dataset_name="CIFAR-10",
        on_progress=on_progress,
    )
    return extract_dir


def download_caltech101(on_progress: Optional[ProgressCallback] = None) -> Path:
    """Download and extract the Caltech-101 image classification dataset.

    Downloads ``caltech-101.zip`` from the configured ``CALTECH101_URL``
    into ``DATA_DIR`` if it is not already present, then extracts it.
    The zip contains a nested ``101_ObjectCategories.tar.gz`` archive
    which is extracted in a second pass to produce the final category
    directories.  Both archives are deleted after extraction to reclaim
    disk space.

    Each invocation uses unique temporary paths so that concurrent calls
    do not interfere with each other.

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

    extract_dir = DATA_DIR / "caltech-101"
    categories_dir = extract_dir / "101_ObjectCategories"

    if categories_dir.exists():
        return categories_dir

    unique_id = uuid.uuid4().hex[:8]
    temp_archive = DATA_DIR / f".dl_{unique_id}_caltech-101.zip"
    temp_extract = DATA_DIR / f".extract_{unique_id}_caltech-101"
    DATA_DIR.mkdir(exist_ok=True)
    IMAGE_DIR.mkdir(exist_ok=True, parents=True)

    try:
        on_progress("downloading", "Starting Caltech-101 download...", 0, 0)
        download_file_with_progress(
            CALTECH101_URL, temp_archive, CALTECH101_DOWNLOAD_SIZE_MB * 1024 * 1024, on_progress
        )

        if categories_dir.exists():
            return categories_dir

        on_progress("downloading", "Extracting Caltech-101 zip...", 0, 0)
        temp_extract.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(temp_archive, "r") as zip_ref:
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
                zip_ref.extract(member, temp_extract)

        # The zip contains 101_ObjectCategories.tar.gz (a nested archive).
        # Extract it to produce the actual category directories.
        inner_tar = temp_extract / "caltech-101" / "101_ObjectCategories.tar.gz"
        if inner_tar.exists():
            on_progress("downloading", "Extracting 101_ObjectCategories...", 0, 0)
            inner_dest = temp_extract / "caltech-101"
            with tarfile.open(inner_tar, "r:gz") as tar_ref:
                members = tar_ref.getmembers()
                total = len(members)
                for i, member in enumerate(members):
                    if i % 100 == 0 or i == total - 1:
                        on_progress(
                            "downloading",
                            f"Extracting 101_ObjectCategories ({i + 1}/{total})...",
                            i + 1,
                            total,
                        )
                    tar_ref.extract(member, inner_dest, filter="data")
            inner_tar.unlink(missing_ok=True)

        if categories_dir.exists():
            return categories_dir

        # Move extracted caltech-101 dir to final location.
        temp_caltech = temp_extract / "caltech-101"
        if temp_caltech.exists():
            if not extract_dir.exists():
                try:
                    os.rename(temp_caltech, extract_dir)
                except OSError:
                    _move_tree_contents(temp_caltech, extract_dir)
            else:
                _move_tree_contents(temp_caltech, extract_dir)

        return categories_dir
    finally:
        temp_archive.unlink(missing_ok=True)
        if temp_extract.exists():
            shutil.rmtree(temp_extract, ignore_errors=True)


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

    IMAGE_DIR.mkdir(exist_ok=True, parents=True)

    categories_dir = DATA_DIR / "caltech-256" / "256_ObjectCategories"
    _download_and_extract(
        url=CALTECH256_URL,
        archive_name="256_ObjectCategories.tar",
        extract_to=DATA_DIR / "caltech-256",
        check_path=categories_dir,
        download_size_mb=CALTECH256_DOWNLOAD_SIZE_MB,
        dataset_name="Caltech-256",
        on_progress=on_progress,
    )
    return categories_dir


def download_oxford_flowers(on_progress: Optional[ProgressCallback] = None) -> Path:
    """Download and extract the Oxford Flowers 102 dataset.

    Downloads the image tarball and the labels MAT file from the Oxford VGG
    website into ``DATA_DIR`` if they are not already present, then extracts
    the images.  Archives are deleted after extraction to reclaim disk space.

    The dataset contains 8189 images of 102 flower species.  Images are stored
    in a single flat directory (``jpg/``) with numeric filenames; the class
    label for each image is provided in a separate MATLAB ``.mat`` file.

    Args:
        on_progress: Optional progress callback.  Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        Path to the ``oxford_flowers/`` directory containing ``jpg/`` and
        ``imagelabels.mat`` (e.g. ``data/oxford_flowers``).
    """
    if on_progress is None:
        on_progress = _default_progress()

    IMAGE_DIR.mkdir(exist_ok=True, parents=True)

    extract_dir = DATA_DIR / "oxford_flowers"
    _download_and_extract(
        url=OXFORD_FLOWERS_URL,
        archive_name="102flowers.tgz",
        extract_to=extract_dir,
        check_path=extract_dir / "jpg",
        download_size_mb=OXFORD_FLOWERS_DOWNLOAD_SIZE_MB,
        dataset_name="Oxford Flowers",
        on_progress=on_progress,
    )

    # Download labels file if not present.
    labels_path = extract_dir / "imagelabels.mat"
    if not labels_path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        on_progress("downloading", "Downloading Oxford Flowers labels...", 0, 0)
        download_file_with_progress(OXFORD_FLOWERS_LABELS_URL, labels_path, 1024 * 1024, on_progress)

    return extract_dir


def download_food101(on_progress: Optional[ProgressCallback] = None) -> Path:
    """Download and extract the Food-101 dataset.

    Downloads ``food-101.tar.gz`` from ETH Zurich into ``DATA_DIR`` if it is
    not already present, then extracts it.  The archive is deleted after
    extraction to reclaim disk space.

    The dataset contains 101 000 images across 101 food categories (1000
    images per category), stored in category subdirectories.

    Args:
        on_progress: Optional progress callback.  Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        Path to the ``food-101/images/`` directory containing category
        subdirectories with JPEG images (e.g. ``data/food-101/images``).
    """
    if on_progress is None:
        on_progress = _default_progress()

    IMAGE_DIR.mkdir(exist_ok=True, parents=True)

    images_dir = DATA_DIR / "food-101" / "images"
    _download_and_extract(
        url=FOOD101_URL,
        archive_name="food-101.tar.gz",
        extract_to=DATA_DIR,
        check_path=images_dir,
        download_size_mb=FOOD101_DOWNLOAD_SIZE_MB,
        dataset_name="Food-101",
        on_progress=on_progress,
    )
    return images_dir


def download_eurosat(on_progress: Optional[ProgressCallback] = None) -> Path:
    """Download and extract the EuroSAT RGB dataset.

    Downloads ``EuroSAT_RGB.zip`` from HuggingFace into ``DATA_DIR`` if it
    is not already present, then extracts it.  The archive is deleted after
    extraction to reclaim disk space.

    The dataset contains 27 000 Sentinel-2 satellite image patches across
    10 land-use classes, stored in class subdirectories.

    Args:
        on_progress: Optional progress callback.  Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        Path to the ``EuroSAT_RGB/`` directory containing class
        subdirectories with JPEG images (e.g. ``data/EuroSAT_RGB``).
    """
    if on_progress is None:
        on_progress = _default_progress()

    IMAGE_DIR.mkdir(exist_ok=True, parents=True)

    extract_dir = DATA_DIR / "EuroSAT_RGB"
    _download_and_extract(
        url=EUROSAT_URL,
        archive_name="EuroSAT_RGB.zip",
        extract_to=DATA_DIR,
        check_path=extract_dir,
        download_size_mb=EUROSAT_DOWNLOAD_SIZE_MB,
        dataset_name="EuroSAT",
        on_progress=on_progress,
    )
    return extract_dir


def download_stanford_dogs(on_progress: Optional[ProgressCallback] = None) -> Path:
    """Download and extract the Stanford Dogs dataset.

    Downloads ``images.tar`` from the Stanford Vision Lab into ``DATA_DIR``
    if it is not already present, then extracts it.  The archive is deleted
    after extraction to reclaim disk space.

    The dataset contains ~20 580 images across 120 dog breed classes.  The
    archive extracts to an ``Images/`` directory with breed subdirectories
    named like ``n02085620-Chihuahua``.

    Args:
        on_progress: Optional progress callback.  Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        Path to the ``stanford_dogs/Images/`` directory containing breed
        subdirectories with JPEG images (e.g.
        ``data/stanford_dogs/Images``).
    """
    if on_progress is None:
        on_progress = _default_progress()

    IMAGE_DIR.mkdir(exist_ok=True, parents=True)

    images_dir = DATA_DIR / "stanford_dogs" / "Images"
    _download_and_extract(
        url=STANFORD_DOGS_URL,
        archive_name="stanford_dogs_images.tar",
        extract_to=DATA_DIR / "stanford_dogs",
        check_path=images_dir,
        download_size_mb=STANFORD_DOGS_DOWNLOAD_SIZE_MB,
        dataset_name="Stanford Dogs",
        on_progress=on_progress,
    )
    return images_dir


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

    extract_dir = DATA_DIR / "UCF101_subset"
    _download_and_extract(
        url=UCF101_SUBSET_URL,
        archive_name="UCF101_subset.tar.gz",
        extract_to=DATA_DIR,
        check_path=extract_dir,
        download_size_mb=UCF101_SUBSET_DOWNLOAD_SIZE_MB,
        dataset_name="UCF-101 subset",
        on_progress=on_progress,
    )

    # Flatten the train/val/test splits into VIDEO_DIR/ucf101/<Category>/.

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


def download_bbc_news(
    on_progress: Optional[ProgressCallback] = None,
) -> dict[str, list[str]]:
    """Download and prepare the BBC News full-text dataset.

    Downloads ``bbc-fulltext.zip`` from the configured ``BBC_NEWS_URL`` into
    ``DATA_DIR`` if it is not already present, then extracts it.  The zip is
    deleted after extraction to reclaim disk space.

    The dataset contains ~2225 articles across five topic categories:
    ``business``, ``entertainment``, ``politics``, ``sport``, and ``tech``.

    Each invocation uses unique temporary paths so that concurrent calls
    do not interfere with each other.

    Args:
        on_progress: Optional progress callback. Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        A dict mapping category name to a list of article text strings, e.g.
        ``{"business": ["Article text…", …], "sport": […], …}``.
    """
    if on_progress is None:
        on_progress = _default_progress()

    extract_dir = DATA_DIR / "bbc-fulltext"
    DATA_DIR.mkdir(exist_ok=True)

    if not extract_dir.exists():
        unique_id = uuid.uuid4().hex[:8]
        temp_archive = DATA_DIR / f".dl_{unique_id}_bbc-fulltext.zip"
        temp_extract = DATA_DIR / f".extract_{unique_id}_bbc-fulltext"

        try:
            on_progress("downloading", "Starting BBC News download...", 0, 0)
            download_file_with_progress(
                BBC_NEWS_URL,
                temp_archive,
                BBC_NEWS_DOWNLOAD_SIZE_MB * 1024 * 1024,
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
                                f"Extracting BBC News dataset ({i + 1}/{total})...",
                                i + 1,
                                total,
                            )
                        zip_ref.extract(member, raw_dir)

                # Find the directory that contains the category subfolders.
                _bbc_root = _find_bbc_root(raw_dir)
                if _bbc_root is None:
                    raise RuntimeError(
                        f"Could not locate BBC News category directories inside {raw_dir}"
                    )

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
        ``{"World": ["Article text…", …], "Sports": […], …}``.
    """
    import csv  # noqa: PLC0415

    if on_progress is None:
        on_progress = _default_progress()

    csv_path = DATA_DIR / "ag_news_train.csv"
    DATA_DIR.mkdir(exist_ok=True)

    if not csv_path.exists():
        unique_id = uuid.uuid4().hex[:8]
        temp_path = DATA_DIR / f".dl_{unique_id}_ag_news_train.csv"
        try:
            on_progress("downloading", "Starting AG News download...", 0, 0)
            download_file_with_progress(
                AG_NEWS_URL,
                temp_path,
                AG_NEWS_DOWNLOAD_SIZE_MB * 1024 * 1024,
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
        ``{"pos": ["Great film…", …], "neg": ["Terrible…", …]}``.
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


def download_ucsf_documents(
    categories: list[str],
    docs_per_category: int = 25,
    on_progress: Optional[ProgressCallback] = None,
) -> Path:
    """Download UCSF Industry Documents Library PDFs by industry category.

    Queries the UCSF Industry Documents Library Solr API for short documents
    (1–3 pages) within each *category* (industry name), downloads individual
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
        on_progress = _default_progress()

    extract_dir = DATA_DIR / "ucsf_documents"
    DATA_DIR.mkdir(exist_ok=True)

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
            resp = requests.get(UCSF_IDL_API_URL, params=params, timeout=30)
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

            url = f"{UCSF_IDL_DOWNLOAD_URL}/{doc_id[0]}/{doc_id[1]}/{doc_id[2]}/{doc_id[3]}/{doc_id}/{doc_id}.pdf"

            try:
                download_file_with_progress(url, pdf_path, 0, on_progress)
                downloaded += 1
                on_progress(
                    "downloading", f"Downloaded {doc_id}.pdf ({downloaded}/{total_docs})", downloaded, total_docs
                )
            except Exception:
                # Skip failed downloads silently.
                pdf_path.unlink(missing_ok=True)

    return extract_dir


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
