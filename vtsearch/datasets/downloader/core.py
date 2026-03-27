"""Core downloading utilities: progress helpers, archive validation, and extraction.

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
STANFORD_DOGS_URL = "https://huggingface.co/datasets/Alanox/stanford-dogs/resolve/main/images.tar.gz"
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
