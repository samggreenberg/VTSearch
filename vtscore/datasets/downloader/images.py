"""Image dataset downloaders: CIFAR-10, Caltech-101/256, Oxford Flowers, Food-101, EuroSAT, Stanford Dogs, Places365."""

import os
import shutil
import tarfile
import uuid
import zipfile
from pathlib import Path
from typing import Optional

from vtscore.datasets.downloader import core as _core
from vtscore.datasets.downloader.core import ProgressCallback


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


def _extract_members(members, extract_one, on_progress, message: str, *, start: int) -> None:
    """Extract *members* one at a time, reporting progress every 100 items.

    *extract_one* is called as ``extract_one(member)`` for each member (it
    closes over the archive's own ``extract`` call and destination).  Indices
    run from *start* (1 for the outer zip pass, 0 for the nested tar pass to
    match the cadence each used inline); progress is pushed on every 100th
    member and on the final one.
    """
    total = len(members)
    for offset, member in enumerate(members):
        i = offset + start
        if i % 100 == 0 or i == total - (1 - start):
            on_progress("downloading", message, offset + 1, total)
        extract_one(member)


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
            _extract_members(
                zip_ref.namelist(),
                lambda member, zip_ref=zip_ref: zip_ref.extract(member, temp_extract),
                on_progress,
                "Extracting Caltech-101 zip...",
                start=1,
            )

        # The zip contains 101_ObjectCategories.tar.gz (a nested archive).
        # Extract it to produce the actual category directories.
        inner_tar = temp_extract / "caltech-101" / "101_ObjectCategories.tar.gz"
        if inner_tar.exists():
            on_progress("downloading", "Extracting 101_ObjectCategories...", 0, 0)
            inner_dest = temp_extract / "caltech-101"
            with tarfile.open(inner_tar, "r:gz") as tar_ref:
                _extract_members(
                    tar_ref.getmembers(),
                    lambda member, tar_ref=tar_ref: tar_ref.extract(member, inner_dest, filter="data"),
                    on_progress,
                    "Extracting 101_ObjectCategories...",
                    start=0,
                )
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

    # Download labels file if not present.  Atomic (temp + rename): a
    # partial file left at labels_path would pass the exists() gate forever.
    labels_path = extract_dir / "imagelabels.mat"
    if not labels_path.exists():
        _core.DATA_DIR.mkdir(exist_ok=True)
        on_progress("downloading", "Downloading Oxford Flowers labels...", 0, 0)
        _core.download_file_atomic(_core.OXFORD_FLOWERS_LABELS_URL, labels_path, 1024 * 1024, on_progress)

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
        archive_name="stanford_dogs_images.tar.gz",
        extract_to=_core.DATA_DIR / "stanford_dogs",
        check_path=images_dir,
        download_size_mb=_core.STANFORD_DOGS_DOWNLOAD_SIZE_MB,
        dataset_name="Stanford Dogs",
        on_progress=on_progress,
    )
    return images_dir


def download_roxford5k(on_progress: Optional[ProgressCallback] = None) -> Path:
    """Download and extract the Revisited Oxford Buildings (ROxford5k) dataset.

    Downloads the 5063-image Oxford Buildings tarball from the Oxford VGG mirror
    into a flat ``jpg/`` directory, plus the small "revisited" ground-truth
    pickle (``gnd_roxford5k.pkl``) that carries the cleaned-up query bounding
    boxes and easy/hard/junk relevance lists.  Both are skipped if already
    present.  The image archive is deleted after extraction to reclaim disk
    space.

    This is the structural embedder's instance-retrieval demo: the same building
    photographed from different viewpoints is a single "instance", which a
    structural (SIFT/VLAD) search can match but a semantic embedder cannot.

    Args:
        on_progress: Optional progress callback. Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        Path to the ``roxford5k/`` directory containing ``jpg/`` (the flat image
        folder) and ``gnd_roxford5k.pkl`` (the ground-truth file).
    """
    if on_progress is None:
        on_progress = _core._default_progress()

    _core.IMAGE_DIR.mkdir(exist_ok=True, parents=True)

    extract_dir = _core.DATA_DIR / "roxford5k"
    jpg_dir = extract_dir / "jpg"
    _core._download_and_extract(
        url=_core.ROXFORD_IMAGES_URL,
        archive_name="oxbuild_images-v1.tgz",
        extract_to=jpg_dir,
        check_path=jpg_dir,
        download_size_mb=_core.ROXFORD_IMAGES_DOWNLOAD_SIZE_MB,
        dataset_name="ROxford5k",
        on_progress=on_progress,
    )

    # Pull the (small) revisited ground-truth pickle alongside the images.
    # Atomic (temp + rename): a partial file left at gnd_path would pass the
    # exists() gate forever.
    gnd_path = extract_dir / "gnd_roxford5k.pkl"
    if not gnd_path.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        on_progress("downloading", "Downloading ROxford5k ground truth...", 0, 0)
        _core.download_file_atomic(_core.ROXFORD_GND_URL, gnd_path, 1024 * 1024, on_progress)

    return extract_dir


def download_openlogo(on_progress: Optional[ProgressCallback] = None) -> Path:
    """Download the OpenLogo (QMUL-OpenLogo) logo dataset from HuggingFace.

    OpenLogo is distributed as a `FiftyOne <https://voxel51.com>`_ dataset: a flat
    ``data/`` folder of ~27k JPEGs plus a ``samples.json`` describing each image's
    ``ground_truth`` detections (brand label + normalized ``[x, y, w, h]`` box).
    Because the media is thousands of loose files with no single archive, it is
    fetched with :func:`huggingface_hub.snapshot_download` rather than the
    single-URL :func:`download_file_with_progress` helper.  The large preview GIF
    is skipped.  The stored HuggingFace token (if the user signed in) is passed
    through for rate-limit headroom, though the dataset is public.

    This is the structural embedder's instance-matching *logo* demo: one boxed
    example of a brand's logo should rank that brand's other in-the-wild photos
    above the other brands, which a semantic embedder cannot do.  Labels are
    parsed from ``samples.json`` with the stdlib, so ``fiftyone`` is not required.

    Args:
        on_progress: Optional progress callback. Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        Path to the ``openlogo/`` directory containing ``data/`` (the flat image
        folder) and ``samples.json`` (the per-image detection annotations).
    """
    if on_progress is None:
        on_progress = _core._default_progress()

    _core.IMAGE_DIR.mkdir(exist_ok=True, parents=True)

    extract_dir = _core.DATA_DIR / "openlogo"
    samples_json = extract_dir / "samples.json"
    data_dir = extract_dir / "data"
    if samples_json.exists() and data_dir.is_dir():
        return extract_dir

    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from huggingface_hub.utils.tqdm import tqdm as _hf_tqdm  # noqa: PLC0415

    from vtscore.security.hf_auth import get_token  # noqa: PLC0415

    _core.DATA_DIR.mkdir(exist_ok=True)
    # Brief placeholder while snapshot_download resolves the file list (a network
    # round-trip before any file is fetched). Kept short and size-first so it
    # stays legible in the frontend's narrow, ellipsized detail slot; real
    # per-file progress (below) supersedes it within a second or two.
    on_progress(
        "downloading",
        f"Downloading OpenLogo (~{_core.OPENLOGO_DOWNLOAD_SIZE_MB // 1024} GB)...",
        0,
        0,
    )

    # OpenLogo is thousands of loose files, so there is no single byte stream to
    # tap like download_file_with_progress does. snapshot_download drives its
    # over-the-files progress bar through ``tqdm_class``; subclass it to forward
    # the file count to ``on_progress`` so the UI shows a live, measurable bar
    # ("1,234/27,000") instead of a static, ellipsized sentence.
    class _OpenlogoProgress(_hf_tqdm):  # type: ignore[misc, valid-type]
        def update(self, n: int = 1) -> Optional[bool]:
            displayed = super().update(n)
            total = int(self.total or 0)
            if total > 0:
                on_progress("downloading", "Downloading OpenLogo logos", int(self.n), total)
            return displayed

    snapshot_download(
        repo_id=_core.OPENLOGO_REPO_ID,
        repo_type="dataset",
        local_dir=str(extract_dir),
        ignore_patterns=["*.gif"],
        token=get_token(),
        tqdm_class=_OpenlogoProgress,
    )
    return extract_dir


def download_places365(on_progress: Optional[ProgressCallback] = None) -> Path:
    """Download and extract the Places365 validation set.

    Downloads ``val_256.tar`` (the 256x256 validation split, ~501 MB) from the
    MIT CSAIL Places2 server into ``DATA_DIR`` if it is not already present,
    then extracts it.  The archive is deleted after extraction to reclaim
    disk space.  The per-image label file ``places365_val.txt`` is also
    fetched from the CSAILVision GitHub repository so that downstream code
    can map each image filename to a scene category index.

    The validation set contains 36 500 images across 365 scene categories
    (100 images per category).  Images are stored in a single flat
    ``val_256/`` directory with names like ``Places365_val_00000001.jpg``.

    Args:
        on_progress: Optional progress callback.  Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        Path to the ``places365/`` directory containing ``val_256/`` (the
        flat image folder) and ``places365_val.txt`` (the label mapping
        file).
    """
    if on_progress is None:
        on_progress = _core._default_progress()

    _core.IMAGE_DIR.mkdir(exist_ok=True, parents=True)

    extract_dir = _core.DATA_DIR / "places365"
    _core._download_and_extract(
        url=_core.PLACES365_URL,
        archive_name="val_256.tar",
        extract_to=extract_dir,
        check_path=extract_dir / "val_256",
        download_size_mb=_core.PLACES365_DOWNLOAD_SIZE_MB,
        dataset_name="Places365",
        on_progress=on_progress,
    )

    # Pull the per-image label file out of the canonical "filelist"
    # tarball (~65 MB) hosted on the MIT mirror.  The historical raw-
    # GitHub mirror went 404, so we now fetch the bundle that ships the
    # whole train/val/test/category file set and extract only the val
    # member we need.
    labels_path = extract_dir / "places365_val.txt"
    if not labels_path.exists():
        _core.DATA_DIR.mkdir(exist_ok=True)
        extract_dir.mkdir(exist_ok=True, parents=True)
        unique_id = uuid.uuid4().hex[:8]
        filelist_archive = _core.DATA_DIR / f".dl_{unique_id}_places365_filelist.tar"
        try:
            on_progress("downloading", "Downloading Places365 labels...", 0, 0)
            _core.download_file_with_progress(
                _core.PLACES365_LABELS_FILELIST_URL,
                filelist_archive,
                _core.PLACES365_LABELS_FILELIST_SIZE_MB * 1024 * 1024,
                on_progress,
            )
            with tarfile.open(filelist_archive, "r") as tar:
                member = tar.getmember("places365_val.txt")
                src = tar.extractfile(member)
                if src is None:
                    raise RuntimeError("places365_val.txt not extractable from filelist tar")
                with open(labels_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        finally:
            if filelist_archive.exists():
                filelist_archive.unlink()

    return extract_dir


def download_visual_genome(on_progress: Optional[ProgressCallback] = None) -> Path:
    """Download and extract the Visual Genome dataset (images + object annotations).

    Visual Genome ships its ~108k images as two zips (the historical
    ``VG_100K`` / ``VG_100K_2`` splits) and its per-object annotations
    (object name + pixel bounding box per region) as a separate
    ``objects.json.zip``.  All three are extracted into ``data/visual_genome/``;
    each archive is deleted after extraction to reclaim disk space.

    Unlike the folder-per-class image demos, Visual Genome is **multi-label**:
    one image is typically a positive example of several object categories at
    once.  The category assignment and bounding-box normalization happen at
    load time in :mod:`vtscore.media.image._demo_sources`, which reads the
    extracted ``objects.json`` — this function only fetches the raw files.

    Args:
        on_progress: Optional progress callback.  Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        Path to the ``visual_genome/`` directory containing ``VG_100K/`` and
        ``VG_100K_2/`` (the flat image folders) plus ``objects.json``.
    """
    if on_progress is None:
        on_progress = _core._default_progress()

    _core.IMAGE_DIR.mkdir(exist_ok=True, parents=True)
    vg_dir = _core.DATA_DIR / "visual_genome"

    _core._download_and_extract(
        url=_core.VISUAL_GENOME_IMAGES_URL,
        archive_name="vg_images.zip",
        extract_to=vg_dir,
        check_path=vg_dir / "VG_100K",
        download_size_mb=_core.VISUAL_GENOME_IMAGES_DOWNLOAD_SIZE_MB,
        dataset_name="Visual Genome images (1/2)",
        on_progress=on_progress,
    )
    _core._download_and_extract(
        url=_core.VISUAL_GENOME_IMAGES2_URL,
        archive_name="vg_images2.zip",
        extract_to=vg_dir,
        check_path=vg_dir / "VG_100K_2",
        download_size_mb=_core.VISUAL_GENOME_IMAGES2_DOWNLOAD_SIZE_MB,
        dataset_name="Visual Genome images (2/2)",
        on_progress=on_progress,
    )
    _core._download_and_extract(
        url=_core.VISUAL_GENOME_OBJECTS_URL,
        archive_name="vg_objects.json.zip",
        extract_to=vg_dir,
        check_path=vg_dir / "objects.json",
        download_size_mb=_core.VISUAL_GENOME_OBJECTS_DOWNLOAD_SIZE_MB,
        dataset_name="Visual Genome annotations",
        on_progress=on_progress,
    )
    return vg_dir
