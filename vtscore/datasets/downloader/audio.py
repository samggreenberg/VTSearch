"""Audio dataset downloaders: ESC-50, GTZAN, Speech Commands v2, UrbanSound8K,
TUT Sound Events 2017."""

import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Optional

from vtscore.datasets.downloader import core as _core
from vtscore.datasets.downloader.core import ProgressCallback


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
        on_progress = _core._default_progress()

    extract_dir = _core.DATA_DIR / "ESC-50-master"
    _core._download_and_extract(
        url=_core.ESC50_URL,
        archive_name="esc50.zip",
        extract_to=_core.DATA_DIR,
        check_path=extract_dir,
        download_size_mb=_core.ESC50_DOWNLOAD_SIZE_MB,
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
        on_progress = _core._default_progress()

    genres_dir = _core.DATA_DIR / "gtzan" / "genres"
    _core._download_and_extract(
        url=_core.GTZAN_URL,
        archive_name="genres.tar.gz",
        extract_to=_core.DATA_DIR / "gtzan",
        check_path=genres_dir,
        download_size_mb=_core.GTZAN_DOWNLOAD_SIZE_MB,
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
        on_progress = _core._default_progress()

    extract_dir = _core.DATA_DIR / "speech_commands_v2"
    _core._download_and_extract(
        url=_core.SPEECH_COMMANDS_V2_URL,
        archive_name="speech_commands_v0.02.tar.gz",
        extract_to=extract_dir,
        check_path=extract_dir,
        download_size_mb=_core.SPEECH_COMMANDS_V2_DOWNLOAD_SIZE_MB,
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
        on_progress = _core._default_progress()

    extract_dir = _core.DATA_DIR / "UrbanSound8K"
    _core._download_and_extract(
        url=_core.URBANSOUND8K_URL,
        archive_name="UrbanSound8K.tar.gz",
        extract_to=_core.DATA_DIR,
        check_path=extract_dir,
        download_size_mb=_core.URBANSOUND8K_DOWNLOAD_SIZE_MB,
        dataset_name="UrbanSound8K",
        on_progress=on_progress,
    )
    return extract_dir


def download_tut_sound_events_2017(on_progress: Optional[ProgressCallback] = None) -> Path:
    """Download and extract the TUT Sound Events 2017 street recordings.

    Pulls the development (two audio zips) and evaluation (one audio zip)
    sets from Zenodo into ``DATA_DIR / "tut_sound_events_2017"``, extracting
    only the ``.wav`` recordings (the metadata/annotation zips are skipped on
    purpose).  Each archive's recordings land in their own subdirectory so the
    two development zips don't collide and per-archive caching works.

    The 32 recordings (24 development + 8 evaluation) are full ~4-minute
    binaural street soundscapes, each containing many sound events scattered
    over time - long-form material intended for hands-on clipping.

    Args:
        on_progress: Optional progress callback.  Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        Path to the ``tut_sound_events_2017/`` directory whose subdirectories
        hold the extracted ``.wav`` files.
    """
    if on_progress is None:
        on_progress = _core._default_progress()

    base = _core.DATA_DIR / "tut_sound_events_2017"
    archives = _core.TUT_SOUND_EVENTS_2017_ARCHIVES

    # Cached when every archive's subdirectory already holds at least one wav.
    if base.exists() and all((base / slug).exists() and any((base / slug).glob("*.wav")) for _url, slug in archives):
        return base

    base.mkdir(parents=True, exist_ok=True)
    total = len(archives)

    for i, (url, slug) in enumerate(archives):
        dest_dir = base / slug
        if dest_dir.exists() and any(dest_dir.glob("*.wav")):
            continue  # This archive already extracted.

        unique_id = uuid.uuid4().hex[:8]
        zip_path = _core.DATA_DIR / f".dl_{unique_id}_tut_{slug}.zip"

        try:
            on_progress("downloading", f"Downloading TUT Sound Events 2017 ({slug})...", i, total)
            _core.download_file_with_progress(url, zip_path, 0, on_progress)

            on_progress("downloading", f"Extracting TUT Sound Events 2017 ({slug})...", i + 1, total)
            dest_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.namelist():
                    # Extract only the audio, flattened into the archive's dir.
                    if member.lower().endswith(".wav"):
                        dest = dest_dir / Path(member).name
                        if not dest.exists():
                            with zf.open(member) as src, open(dest, "wb") as dst:
                                shutil.copyfileobj(src, dst)
        finally:
            zip_path.unlink(missing_ok=True)

    return base
