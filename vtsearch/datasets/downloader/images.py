"""Image dataset downloaders: CIFAR-10, Caltech-101/256, Oxford Flowers, Food-101, EuroSAT, Stanford Dogs."""

import os
import shutil
import tarfile
import uuid
import zipfile
from pathlib import Path
from typing import Optional

from vtsearch.datasets.downloader import core as _core
from vtsearch.datasets.downloader.core import ProgressCallback


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
        on_progress = _core._default_progress()

    extract_dir = _core.DATA_DIR / "cifar-10-batches-py"
    _core._download_and_extract(
        url=_core.CIFAR10_URL,
        archive_name="cifar-10-python.tar.gz",
        extract_to=_core.DATA_DIR,
        check_path=extract_dir,
        download_size_mb=_core.CIFAR10_DOWNLOAD_SIZE_MB,
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
        on_progress = _core._default_progress()

    extract_dir = _core.DATA_DIR / "caltech-101"
    categories_dir = extract_dir / "101_ObjectCategories"

    if categories_dir.exists():
        return categories_dir

    unique_id = uuid.uuid4().hex[:8]
    temp_archive = _core.DATA_DIR / f".dl_{unique_id}_caltech-101.zip"
    temp_extract = _core.DATA_DIR / f".extract_{unique_id}_caltech-101"
    _core.DATA_DIR.mkdir(exist_ok=True)
    _core.IMAGE_DIR.mkdir(exist_ok=True, parents=True)

    try:
        on_progress("downloading", "Starting Caltech-101 download...", 0, 0)
        _core.download_file_with_progress(
            _core.CALTECH101_URL, temp_archive, _core.CALTECH101_DOWNLOAD_SIZE_MB * 1024 * 1024, on_progress
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
                    _core._move_tree_contents(temp_caltech, extract_dir)
            else:
                _core._move_tree_contents(temp_caltech, extract_dir)

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
        on_progress = _core._default_progress()

    _core.IMAGE_DIR.mkdir(exist_ok=True, parents=True)

    categories_dir = _core.DATA_DIR / "caltech-256" / "256_ObjectCategories"
    _core._download_and_extract(
        url=_core.CALTECH256_URL,
        archive_name="256_ObjectCategories.tar",
        extract_to=_core.DATA_DIR / "caltech-256",
        check_path=categories_dir,
        download_size_mb=_core.CALTECH256_DOWNLOAD_SIZE_MB,
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
        on_progress = _core._default_progress()

    _core.IMAGE_DIR.mkdir(exist_ok=True, parents=True)

    extract_dir = _core.DATA_DIR / "oxford_flowers"
    _core._download_and_extract(
        url=_core.OXFORD_FLOWERS_URL,
        archive_name="102flowers.tgz",
        extract_to=extract_dir,
        check_path=extract_dir / "jpg",
        download_size_mb=_core.OXFORD_FLOWERS_DOWNLOAD_SIZE_MB,
        dataset_name="Oxford Flowers",
        on_progress=on_progress,
    )

    # Download labels file if not present.
    labels_path = extract_dir / "imagelabels.mat"
    if not labels_path.exists():
        _core.DATA_DIR.mkdir(exist_ok=True)
        on_progress("downloading", "Downloading Oxford Flowers labels...", 0, 0)
        _core.download_file_with_progress(_core.OXFORD_FLOWERS_LABELS_URL, labels_path, 1024 * 1024, on_progress)

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
        on_progress = _core._default_progress()

    _core.IMAGE_DIR.mkdir(exist_ok=True, parents=True)

    images_dir = _core.DATA_DIR / "food-101" / "images"
    _core._download_and_extract(
        url=_core.FOOD101_URL,
        archive_name="food-101.tar.gz",
        extract_to=_core.DATA_DIR,
        check_path=images_dir,
        download_size_mb=_core.FOOD101_DOWNLOAD_SIZE_MB,
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
        on_progress = _core._default_progress()

    _core.IMAGE_DIR.mkdir(exist_ok=True, parents=True)

    extract_dir = _core.DATA_DIR / "EuroSAT_RGB"
    _core._download_and_extract(
        url=_core.EUROSAT_URL,
        archive_name="EuroSAT_RGB.zip",
        extract_to=_core.DATA_DIR,
        check_path=extract_dir,
        download_size_mb=_core.EUROSAT_DOWNLOAD_SIZE_MB,
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
        on_progress = _core._default_progress()

    _core.IMAGE_DIR.mkdir(exist_ok=True, parents=True)

    images_dir = _core.DATA_DIR / "stanford_dogs" / "Images"
    _core._download_and_extract(
        url=_core.STANFORD_DOGS_URL,
        archive_name="stanford_dogs_images.tar",
        extract_to=_core.DATA_DIR / "stanford_dogs",
        check_path=images_dir,
        download_size_mb=_core.STANFORD_DOGS_DOWNLOAD_SIZE_MB,
        dataset_name="Stanford Dogs",
        on_progress=on_progress,
    )
    return images_dir
