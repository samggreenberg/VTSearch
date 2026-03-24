"""Video dataset downloaders: UCF-101 subset."""

from pathlib import Path
from typing import Optional

from vtsearch.config import DATA_DIR
from vtsearch.datasets.downloader.core import (
    UCF101_SUBSET_DOWNLOAD_SIZE_MB,
    UCF101_SUBSET_URL,
    VIDEO_DIR,
    ProgressCallback,
    _default_progress,
    _download_and_extract,
)


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
