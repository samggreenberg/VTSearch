"""Video dataset downloaders: UCF-101 subset, UCF-101 full, HMDB51, KTH Actions."""

import shutil
from pathlib import Path
from typing import Optional

from vtscore.datasets.downloader import core as _core
from vtscore.datasets.downloader.core import ProgressCallback


def download_ucf101_subset(on_progress: Optional[ProgressCallback] = None) -> Path:
    """Download and extract a UCF-101 subset for video demo datasets.

    Downloads a 171 MB subset of UCF-101 (10 action classes, 405 medias) from
    HuggingFace and extracts it into ``VIDEO_DIR / "ucf101"`` with one
    subdirectory per action class.  Videos from all splits (train/val/test) are
    merged into a single flat category structure so that
    :func:`~vtscore.datasets.loader.load_video_metadata_from_folders` can
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
        on_progress = _core._default_progress()

    video_dir = _core.VIDEO_DIR / "ucf101"
    _core.VIDEO_DIR.mkdir(exist_ok=True, parents=True)

    # Already downloaded and extracted - nothing to do.
    if video_dir.exists() and any(video_dir.glob("*/*.avi")):
        return video_dir

    extract_dir = _core.DATA_DIR / "UCF101_subset"
    _core._download_and_extract(
        url=_core.UCF101_SUBSET_URL,
        archive_name="UCF101_subset.tar.gz",
        extract_to=_core.DATA_DIR,
        check_path=extract_dir,
        download_size_mb=_core.UCF101_SUBSET_DOWNLOAD_SIZE_MB,
        dataset_name="UCF-101 subset",
        on_progress=on_progress,
    )

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


def download_hmdb51(on_progress: Optional[ProgressCallback] = None) -> Path:
    """Download and extract the HMDB51 dataset.

    Downloads the ~2 GB ``hmdb51.zip`` from a public HuggingFace mirror and
    extracts it into ``VIDEO_DIR / "hmdb51"`` with one subdirectory per action
    class of ``.avi`` files.  The archive already carries the canonical
    ``hmdb51/<category>/*.avi`` tree, so extraction needs no ``unrar`` step.

    (The original Serre-Lab ``hmdb51_org.rar`` URL is dead, and the lab's page
    now offers only Google Drive links that 404, so the loader relies on the
    HuggingFace mirror — see ``HMDB51_URL``.)

    Args:
        on_progress: Optional progress callback.

    Returns:
        Path to the ``hmdb51/`` directory inside ``VIDEO_DIR``, containing
        one subdirectory per action category with ``.avi`` files.
    """
    if on_progress is None:
        on_progress = _core._default_progress()

    video_dir = _core.VIDEO_DIR / "hmdb51"
    _core.VIDEO_DIR.mkdir(exist_ok=True, parents=True)

    # Already downloaded and extracted.
    if video_dir.exists() and any(video_dir.glob("*/*.avi")):
        return video_dir

    # The zip's single top-level folder is ``hmdb51/``; extract it under
    # DATA_DIR then move its category subdirectories into VIDEO_DIR/hmdb51.
    extract_dir = _core.DATA_DIR / "hmdb51"
    _core._download_and_extract(
        url=_core.HMDB51_URL,
        archive_name="hmdb51.zip",
        extract_to=_core.DATA_DIR,
        check_path=extract_dir,
        download_size_mb=_core.HMDB51_DOWNLOAD_SIZE_MB,
        dataset_name="HMDB51",
        on_progress=on_progress,
    )

    video_dir.mkdir(parents=True, exist_ok=True)
    _core._move_tree_contents(extract_dir, video_dir)
    shutil.rmtree(extract_dir, ignore_errors=True)

    return video_dir


def download_ucf101_full(on_progress: Optional[ProgressCallback] = None) -> Path:
    """Download and extract the full UCF-101 dataset (101 action classes).

    Downloads the ~7 GB ``UCF-101.zip`` from a HuggingFace mirror and
    extracts it into ``VIDEO_DIR / "ucf101_full"`` with one subdirectory
    per action class.

    Args:
        on_progress: Optional progress callback.

    Returns:
        Path to the ``ucf101_full/`` directory inside ``VIDEO_DIR``,
        containing 101 category subdirectories with ``.avi`` files.
    """
    if on_progress is None:
        on_progress = _core._default_progress()

    video_dir = _core.VIDEO_DIR / "ucf101_full"
    _core.VIDEO_DIR.mkdir(exist_ok=True, parents=True)

    # Already downloaded and extracted.
    if video_dir.exists() and any(video_dir.glob("*/*.avi")):
        return video_dir

    extract_dir = _core.DATA_DIR / "UCF-101"
    _core._download_and_extract(
        url=_core.UCF101_FULL_URL,
        archive_name="UCF-101.zip",
        extract_to=_core.DATA_DIR,
        check_path=extract_dir,
        download_size_mb=_core.UCF101_FULL_DOWNLOAD_SIZE_MB,
        dataset_name="UCF-101 (full)",
        on_progress=on_progress,
    )

    # Move extracted UCF-101/ to VIDEO_DIR/ucf101_full/.
    video_dir.mkdir(parents=True, exist_ok=True)
    _core._move_tree_contents(extract_dir, video_dir)
    shutil.rmtree(extract_dir, ignore_errors=True)

    return video_dir


def download_kth(on_progress: Optional[ProgressCallback] = None) -> Path:
    """Download and extract the KTH Actions dataset (6 action classes).

    Downloads six individual ZIP files (one per action, ~1.1 GB total)
    from the KTH website and extracts each into its own category
    subdirectory under ``VIDEO_DIR / "kth"``.

    Args:
        on_progress: Optional progress callback.

    Returns:
        Path to the ``kth/`` directory inside ``VIDEO_DIR``, containing
        six category subdirectories (boxing, handclapping, handwaving,
        jogging, running, walking) with ``.avi`` files.
    """
    if on_progress is None:
        on_progress = _core._default_progress()

    video_dir = _core.VIDEO_DIR / "kth"
    _core.VIDEO_DIR.mkdir(exist_ok=True, parents=True)

    # Already downloaded and extracted (all actions present).
    actions = _core.KTH_ACTIONS
    if video_dir.exists() and all(
        (video_dir / action).exists() and any((video_dir / action).glob("*.avi")) for action in actions
    ):
        return video_dir

    video_dir.mkdir(parents=True, exist_ok=True)
    total = len(actions)

    for i, action in enumerate(actions):
        cat_dir = video_dir / action
        if cat_dir.exists() and any(cat_dir.glob("*.avi")):
            continue  # This action already extracted.

        cat_dir.mkdir(parents=True, exist_ok=True)
        _core._download_and_extract(
            url=f"{_core.KTH_BASE_URL}{action}.zip",
            archive_name=f"{action}.zip",
            extract_to=cat_dir,
            check_path=cat_dir,
            is_complete=lambda cat_dir=cat_dir: any(cat_dir.glob("*.avi")),
            download_size_mb=0,
            dataset_name=f"KTH {action}",
            on_progress=on_progress,
            member_filter=lambda m: m.lower().endswith(".avi"),
            flatten=True,
        )
        on_progress("extracting", f"Extracted KTH {action}...", i + 1, total)

    return video_dir
