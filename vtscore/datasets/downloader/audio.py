"""Audio dataset downloaders: ESC-50, GTZAN, Speech Commands v2, UrbanSound8K,
TUT Sound Events 2017, Clotho, Apollo 11, BirdVox-full-night, Nixon tapes."""

import shutil
from pathlib import Path
from typing import Optional

import requests

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


def download_clotho(on_progress: Optional[ProgressCallback] = None) -> Path:
    """Download and extract the Clotho (v1) evaluation split.

    Downloads ``clotho_audio_evaluation.7z`` from Zenodo into
    ``DATA_DIR / "clotho"`` if it is not already present, then extracts it
    (via ``py7zr``).  The archive is deleted after extraction to reclaim disk
    space.

    The evaluation split is 1045 real-world Freesound recordings (15-30 s
    each).  Clotho is an audio *captioning* dataset with no class labels, so
    the demo treats every recording as one undifferentiated bucket - it exists
    to exercise natural-language text->audio retrieval, not classification.

    Args:
        on_progress: Optional progress callback.  Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        Path to the ``clotho/`` directory holding the extracted ``.wav`` files
        (e.g. ``data/clotho``).
    """
    if on_progress is None:
        on_progress = _core._default_progress()

    extract_dir = _core.DATA_DIR / "clotho"
    _core._download_and_extract(
        url=_core.CLOTHO_EVAL_URL,
        archive_name="clotho_audio_evaluation.7z",
        extract_to=extract_dir,
        check_path=extract_dir,
        download_size_mb=_core.CLOTHO_EVAL_DOWNLOAD_SIZE_MB,
        dataset_name="Clotho",
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

        dest_dir.mkdir(parents=True, exist_ok=True)
        _core._download_and_extract(
            url=url,
            archive_name=f"tut_{slug}.zip",
            extract_to=dest_dir,
            check_path=dest_dir,
            is_complete=lambda dest_dir=dest_dir: any(dest_dir.glob("*.wav")),
            download_size_mb=0,
            dataset_name=f"TUT Sound Events 2017 ({slug})",
            on_progress=on_progress,
            member_filter=lambda m: m.lower().endswith(".wav"),
            flatten=True,
        )
        on_progress("extracting", f"Extracted TUT Sound Events 2017 ({slug})...", i + 1, total)

    return base


# ---------------------------------------------------------------------------
# Long-form recordings: Apollo 11, BirdVox-full-night, Nixon White House Tapes
#
# These three differ from the demos above in that a size variant downloads only
# *its own slice* of the source.  ESC-50 and friends pull one archive and slice
# afterwards, which is fine at 600 MB; these run to 5-10 GB, so each exposes a
# ``*_manifest()`` returning the full ordered item list.  The caller slices that
# manifest and passes the selection to the downloader, which fetches exactly
# those items.  Sorting every manifest deterministically is what keeps the
# S/M/L/A slices stable from one load to the next.
# ---------------------------------------------------------------------------


def _fetch_text(url: str, label: str = "") -> str:
    """GET *url* through the SSRF-guarded session and return its body as text.

    Routed through :func:`~vtscore.datasets.downloader.core.fetch_text_with_retry`
    so a manifest/index fetch gets the same retry budget as the transfer it
    precedes: these are the Internet Archive and NARA, and a single unlucky
    handshake here used to sink the whole load before a byte was downloaded.
    """
    return _core.fetch_text_with_retry(url, label)


# A per-file demo set is dozens to hundreds of separate transfers from one
# third-party host, so the odds that *some* file fails are far higher than for
# the single-archive demos.  The Internet Archive in particular serves each file
# from a data node that can answer HTTP 500 for minutes on end while its
# siblings stay healthy, which is what sank a whole Apollo 11 load over one
# track of thirty (issue #3227).  Losing a handful of hours-long recordings out
# of a hundred costs the user nothing they would notice - losing the load does -
# so a failed file is set aside, retried once at the end of the set (by which
# point a transient node error has usually cleared), and only fails the load if
# too many of them pile up.  A skipped file is simply absent from *dest_dir*, so
# the next load of the same dataset fetches it rather than treating it as done.
_MAX_FAILED_FRACTION = 0.25

# Per-file failures worth setting aside rather than sinking the set: the remote
# gave up (connection or a retryable status exhausted its budget), or answered
# with a hard status such as 404 for this one file.  Everything else - a
# cancelled load, a gated dataset, a full disk - is about the *run*, not the
# file, and must still stop it.
_TOLERATED_FILE_FAILURES = (_core.RemoteUnreachableError, requests.HTTPError)


def _download_one_file_into(
    dest_dir: Path,
    item: tuple[str, str, int],
    dataset_name: str,
    on_progress: ProgressCallback,
    index: int,
    total: int,
) -> None:
    """Fetch one ``(url, filename, size)`` *item*, or report it already cached.

    The file lands on a ``.part`` sibling first and is renamed into place only
    once the transfer completes, so the presence of the final name is an honest
    "this file is whole" marker and an interrupted run resumes per file rather
    than restarting the set.  ``download_file_with_progress`` itself resumes a
    partial ``.part`` via an HTTP ``Range`` request.
    """
    url, filename, size = item
    final = dest_dir / filename
    if final.exists():
        on_progress("downloading", f"{dataset_name}: {filename} (cached)", index + 1, total)
        return
    partial = dest_dir / f"{filename}.part"
    _core.download_file_with_progress(url, partial, expected_size=size, on_progress=on_progress)
    partial.replace(final)
    on_progress("downloading", f"{dataset_name}: downloaded {filename}", index + 1, total)


def _download_files_into(
    dest_dir: Path,
    items: list[tuple[str, str, int]],
    dataset_name: str,
    on_progress: ProgressCallback,
) -> None:
    """Download ``(url, filename, size)`` *items* into *dest_dir*, skipping cached ones.

    Files the remote refuses to serve are set aside and retried once after the
    rest of the set, then skipped; the download only fails if more than
    :data:`_MAX_FAILED_FRACTION` of the set ends up missing.  See that constant
    for why one bad file must not sink the whole download.

    Raises:
        RemoteUnreachableError: If too much of the set could not be fetched.
            The last per-file failure is its ``__cause__``.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    total = len(items)
    failed: list[tuple[int, tuple[str, str, int]]] = []
    last_exc: BaseException | None = None

    for i, item in enumerate(items):
        try:
            _download_one_file_into(dest_dir, item, dataset_name, on_progress, i, total)
        except _TOLERATED_FILE_FAILURES as exc:
            last_exc = exc
            failed.append((i, item))
            on_progress("downloading", f"{dataset_name}: {item[1]} unavailable - will retry at the end", i + 1, total)

    # Second pass: downloading the rest of the set has bought each failure
    # minutes of backoff, which is usually all a wobbling data node needed.
    still_failed: list[tuple[int, tuple[str, str, int]]] = []
    for i, item in failed:
        on_progress("downloading", f"{dataset_name}: retrying {item[1]}...", i + 1, total)
        try:
            _download_one_file_into(dest_dir, item, dataset_name, on_progress, i, total)
        except _TOLERATED_FILE_FAILURES as exc:
            last_exc = exc
            still_failed.append((i, item))

    if not still_failed:
        return
    if len(still_failed) > total * _MAX_FAILED_FRACTION:
        raise _core.RemoteUnreachableError(
            f"{dataset_name}: {len(still_failed)} of {total} files could not be downloaded, "
            f"which is too much of the dataset to load without. {last_exc}",
            url=still_failed[0][1][0],
        ) from last_exc
    on_progress(
        "downloading",
        f"{dataset_name}: skipped {len(still_failed)} of {total} files the server wouldn't serve",
        total,
        total,
    )


def apollo11_audio_manifest() -> list[tuple[str, int]]:
    """Return the Apollo 11 MP3 track list as ``(filename, size_bytes)``, name-sorted.

    Read from the Internet Archive metadata API for the ``Apollo11Audio`` item
    (CC PD Mark 1.0).  Track names are irregular enough that they can't be
    generated (``11-03301.mp3``, ``155-AAA.mp3``, ``11-03703_1_OF_6.mp3``), and
    only the API carries the byte sizes the download progress needs.
    """
    import json  # noqa: PLC0415

    url = f"{_core.ARCHIVE_ORG_METADATA_URL}/{_core.APOLLO11_AUDIO_ITEM}"
    payload = json.loads(_fetch_text(url, "the Apollo 11 track list"))
    tracks = [
        (f["name"], int(f.get("size") or 0))
        for f in payload.get("files", [])
        if f.get("format") == _core.APOLLO11_AUDIO_FORMAT and f.get("name")
    ]
    return sorted(tracks)


def download_apollo11_audio(
    tracks: Optional[list[tuple[str, int]]] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> Path:
    """Download the selected Apollo 11 MP3 *tracks* into ``data/apollo11_audio``.

    Args:
        tracks: ``(filename, size_bytes)`` pairs to fetch, normally a slice of
            :func:`apollo11_audio_manifest`.  ``None`` fetches every track
            (~10.1 GB).
        on_progress: Optional progress callback.  Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        Path to the ``apollo11_audio/`` directory holding the ``.mp3`` files.
    """
    if on_progress is None:
        on_progress = _core._default_progress()
    if tracks is None:
        tracks = apollo11_audio_manifest()

    dest_dir = _core.DATA_DIR / "apollo11_audio"
    base = f"{_core.ARCHIVE_ORG_DOWNLOAD_URL}/{_core.APOLLO11_AUDIO_ITEM}"
    items = [(f"{base}/{name}", name, size) for name, size in tracks]
    _download_files_into(dest_dir, items, "Apollo 11 audio", on_progress)
    return dest_dir


def _segment_audio_file(src: Path, dest_dir: Path, stem: str, seconds: float) -> int:
    """Split *src* into ``seconds``-long FLAC chunks named ``<stem>_NNNN.flac``.

    Streams the source frame-block by frame-block so a ten-hour recording never
    has to be resident in memory, and keeps the native sample rate and 16-bit
    depth (BirdVox ships 24 kHz mono), so a chunk is bit-identical to its span
    of the source.  Returns the number of chunks written.
    """
    import soundfile as sf  # noqa: PLC0415

    dest_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with sf.SoundFile(str(src)) as handle:
        frames_per_chunk = max(1, int(seconds * handle.samplerate))
        while True:
            block = handle.read(frames_per_chunk, dtype="int16")
            if len(block) == 0:
                break
            out = dest_dir / f"{stem}_{written:04d}.flac"
            sf.write(str(out), block, handle.samplerate, format="FLAC")
            written += 1
    return written


def birdvox_full_night_manifest() -> list[str]:
    """Return the BirdVox-full-night recording-unit ids, in a stable order."""
    return list(_core.BIRDVOX_FULL_NIGHT_UNITS)


def download_birdvox_full_night(
    units: Optional[list[str]] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> Path:
    """Download the selected BirdVox-full-night *units* and segment them.

    Each unit is a single ~10-hour FLAC on Zenodo.  Handing one of those to the
    clipper as a single media would mean decoding hours of audio into memory at
    once, so every unit is split into
    ``core.BIRDVOX_SEGMENT_SECONDS``-long FLAC chunks and the source file is
    deleted once its chunks are on disk.  The chunks are still long-form
    (10 minutes apiece), which is the point: the flight calls inside them are
    sub-second events the user has to go and find.

    Segmentation runs into a ``.partial`` directory that is renamed into place
    only on success, so an interrupted run re-does the unit rather than leaving
    a half-segmented directory that looks cached.

    Args:
        units: Unit ids to fetch (e.g. ``["unit01"]``), normally a slice of
            :func:`birdvox_full_night_manifest`.  ``None`` fetches all six
            (~5.65 GB).
        on_progress: Optional progress callback.  Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        Path to the ``birdvox_full_night/`` directory whose per-unit
        subdirectories hold the segmented ``.flac`` chunks.
    """
    if on_progress is None:
        on_progress = _core._default_progress()
    if units is None:
        units = birdvox_full_night_manifest()

    base = _core.DATA_DIR / "birdvox_full_night"
    base.mkdir(parents=True, exist_ok=True)
    total = len(units)

    for i, unit in enumerate(units):
        unit_dir = base / unit
        if unit_dir.exists() and any(unit_dir.glob("*.flac")):
            on_progress("downloading", f"BirdVox-full-night: {unit} (cached)", i + 1, total)
            continue

        flac_path = base / f"{unit}.flac"
        if not flac_path.exists():
            partial = base / f"{unit}.flac.part"
            _core.download_file_with_progress(
                _core.BIRDVOX_FULL_NIGHT_URL_TEMPLATE.format(unit=unit),
                partial,
                on_progress=on_progress,
            )
            partial.replace(flac_path)

        on_progress("extracting", f"Segmenting BirdVox-full-night {unit}...", i, total)
        staging = base / f"{unit}.partial"
        shutil.rmtree(staging, ignore_errors=True)
        _segment_audio_file(flac_path, staging, unit, _core.BIRDVOX_SEGMENT_SECONDS)
        shutil.rmtree(unit_dir, ignore_errors=True)
        staging.replace(unit_dir)
        flac_path.unlink(missing_ok=True)
        on_progress("extracting", f"Segmented BirdVox-full-night {unit}", i + 1, total)

    return base


def nixon_tape_manifest() -> list[str]:
    """Return the Nixon White House Tape numbers this demo offers, in order."""
    return list(_core.NIXON_TAPE_NUMBERS)


def nixon_tape_conversation_urls(tape: str) -> list[str]:
    """Return the conversation-MP3 URLs linked from one tape's index page.

    The MP3s live on ``catalog.archives.gov`` under per-tape NARA catalog ids
    that can't be derived from the tape number, so the page is the only place
    the mapping exists.  Duplicates are dropped and the result is sorted, which
    puts the conversations in their recorded order (the filenames end in
    ``-<tape>-<conversation>-pa.mp3``).
    """
    import re  # noqa: PLC0415

    html = _fetch_text(f"{_core.NIXON_TAPES_PAGE_URL}/{tape}", f"the tape {tape} index")
    return sorted(set(re.findall(_core.NIXON_TAPE_MP3_PATTERN, html)))


def download_nixon_tapes(
    tapes: Optional[list[str]] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> Path:
    """Download the selected Nixon White House *tapes* into ``data/nixon_tapes``.

    One MP3 per recorded conversation, filed under a per-tape subdirectory.
    Conversations run from well under a minute to over half an hour, so a
    single tape is a few hundred recordings and roughly 850 MB.

    Args:
        tapes: Tape numbers to fetch (e.g. ``["001"]``), normally a slice of
            :func:`nixon_tape_manifest`.  ``None`` fetches all twelve
            (~10.2 GB).
        on_progress: Optional progress callback.  Falls back to the
            application-wide ``update_progress`` when ``None``.

    Returns:
        Path to the ``nixon_tapes/`` directory whose per-tape subdirectories
        hold the conversation ``.mp3`` files.
    """
    if on_progress is None:
        on_progress = _core._default_progress()
    if tapes is None:
        tapes = nixon_tape_manifest()

    base = _core.DATA_DIR / "nixon_tapes"
    base.mkdir(parents=True, exist_ok=True)
    total = len(tapes)

    for i, tape in enumerate(tapes):
        tape_dir = base / tape
        if tape_dir.exists() and any(tape_dir.glob("*.mp3")):
            on_progress("downloading", f"Nixon tape {tape} (cached)", i + 1, total)
            continue

        urls = nixon_tape_conversation_urls(tape)
        if not urls:
            # NARA is still releasing tapes; a number that has gone quiet just
            # contributes nothing rather than failing the whole load.
            on_progress("downloading", f"Nixon tape {tape}: no audio online", i + 1, total)
            continue

        staging = base / f"{tape}.partial"
        shutil.rmtree(staging, ignore_errors=True)
        items = [(url, url.rsplit("/", 1)[-1], 0) for url in urls]
        _download_files_into(staging, items, f"Nixon tape {tape}", on_progress)
        shutil.rmtree(tape_dir, ignore_errors=True)
        staging.replace(tape_dir)
        on_progress("downloading", f"Downloaded Nixon tape {tape}", i + 1, total)

    return base
