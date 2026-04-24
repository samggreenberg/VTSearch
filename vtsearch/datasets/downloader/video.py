"""Video dataset downloaders: UCF-101 subset, UCF-101 full, HMDB51, KTH Actions."""

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from vtsearch.datasets.downloader import core as _core
from vtsearch.datasets.downloader.core import ProgressCallback


def _extract_rar(rar_path: Path, extract_to: Path) -> None:
    """Extract a RAR archive using the system ``unrar`` command.

    Raises ``RuntimeError`` with installation instructions when ``unrar``
    is not found on the system PATH, or if extraction hangs / fails.
    """
    extract_to.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["unrar", "x", "-o+", "-y", str(rar_path), str(extract_to) + "/"],
            check=True,
            capture_output=True,
            # Cap at 30 minutes. A malformed RAR could otherwise hang the
            # loader thread indefinitely.
            timeout=1800,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "The 'unrar' command is required to extract HMDB51 but was not found. "
            "Install it with: apt-get install unrar (Debian/Ubuntu) or "
            "brew install unrar (macOS)."
        ) from None
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"unrar timed out after {e.timeout}s while extracting {rar_path.name}. The archive may be corrupt."
        ) from None


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
        on_progress = _core._default_progress()

    video_dir = _core.VIDEO_DIR / "ucf101"
    _core.VIDEO_DIR.mkdir(exist_ok=True, parents=True)

    # Already downloaded and extracted — nothing to do.
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

    Downloads the ~2 GB ``hmdb51_org.rar`` archive from the Serre Lab,
    extracts the outer RAR (which contains 51 per-category ``.rar`` files),
    then extracts each inner RAR into its own category subdirectory under
    ``VIDEO_DIR / "hmdb51"``.

    Requires the ``unrar`` command to be installed on the system.

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

    import uuid

    unique_id = uuid.uuid4().hex[:8]
    archive_path = _core.DATA_DIR / f".dl_{unique_id}_hmdb51_org.rar"
    staging_dir = _core.DATA_DIR / f".extract_{unique_id}_hmdb51"
    _core.DATA_DIR.mkdir(exist_ok=True, parents=True)

    try:
        on_progress("downloading", "Starting HMDB51 download...", 0, 0)
        _core.download_file_with_progress(
            _core.HMDB51_URL,
            archive_path,
            _core.HMDB51_DOWNLOAD_SIZE_MB * 1024 * 1024,
            on_progress,
        )

        if video_dir.exists() and any(video_dir.glob("*/*.avi")):
            return video_dir

        # Extract outer RAR → 51 per-category .rar files.
        on_progress("downloading", "Extracting HMDB51 (outer archive)...", 0, 0)
        _extract_rar(archive_path, staging_dir)

        # Extract each inner .rar into a category subdirectory.
        inner_rars = sorted(staging_dir.glob("*.rar"))
        total_rars = len(inner_rars)
        video_dir.mkdir(parents=True, exist_ok=True)

        for i, rar_file in enumerate(inner_rars):
            cat_name = rar_file.stem  # e.g. "brush_hair"
            on_progress("downloading", f"Extracting HMDB51 ({cat_name})...", i + 1, total_rars)
            cat_dir = video_dir / cat_name
            _extract_rar(rar_file, cat_dir)
    finally:
        archive_path.unlink(missing_ok=True)
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

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

        import uuid
        import zipfile

        unique_id = uuid.uuid4().hex[:8]
        zip_name = f"{action}.zip"
        zip_path = _core.DATA_DIR / f".dl_{unique_id}_{zip_name}"

        try:
            url = f"{_core.KTH_BASE_URL}{zip_name}"
            on_progress("downloading", f"Downloading KTH {action} ({i + 1}/{total})...", i, total)
            _core.download_file_with_progress(url, zip_path, 0, on_progress)

            on_progress("downloading", f"Extracting KTH {action}...", i + 1, total)
            cat_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.namelist():
                    # Extract only .avi files, into the category directory.
                    if member.lower().endswith(".avi"):
                        basename = Path(member).name
                        dest = cat_dir / basename
                        if not dest.exists():
                            with zf.open(member) as src, open(dest, "wb") as dst:
                                shutil.copyfileobj(src, dst)
        finally:
            zip_path.unlink(missing_ok=True)

    return video_dir
