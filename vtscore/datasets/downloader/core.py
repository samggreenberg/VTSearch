"""Core downloading utilities: progress helpers, archive validation, and extraction.

All public functions accept an optional ``on_progress`` callback with the
signature ``(status: str, message: str, current: int, total: int) -> None``.
When omitted the functions fall back to the application-wide
:func:`~vtscore.concurrency.progress.update_progress` reporter; pass an explicit callback
to use these functions outside the Flask app (scripts, notebooks, tests).
"""

import os
import shutil
import tarfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urljoin

import requests

from vtscore.config import DATA_DIR
from vtscore.security.hf_auth import GatedResourceError, auth_header_for_url
from vtscore.security.url_validation import validate_url

_MAX_REDIRECTS = 10
# Statuses that mean "this resource is gated / needs credentials we don't have".
# These are surfaced as a short, actionable GatedResourceError rather than a
# raw HTTPError, and are never retried (retrying without auth can't succeed).
_AUTH_REQUIRED_STATUS = frozenset({401, 403})

# Large dataset archives are pulled from flaky third-party CDNs / object stores
# (Caltech's OSN bucket, Zenodo, HuggingFace, university mirrors) that routinely
# drop a connection mid-stream on a multi-GB transfer.  When that happens we
# retry with an HTTP Range request and *resume* from the bytes already on disk
# instead of restarting from zero, with exponential backoff between attempts.
_MAX_DOWNLOAD_ATTEMPTS = 6
_RETRY_BACKOFF_BASE_S = 1.0
_RETRY_BACKOFF_MAX_S = 30.0
# Transient HTTP statuses worth retrying (rate-limit + gateway/CDN hiccups).
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
# Connection-level failures (dropped/incomplete read, reset, read timeout) that
# a resume can recover from.  An IncompleteRead surfaces as ChunkedEncodingError.
_RETRYABLE_EXCEPTIONS = (
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)

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
# 20 Newsgroups is fetched via scikit-learn, but we pre-download its archive
# ourselves (through download_file_with_progress) so the transfer gets a
# fail-fast timeout, retries, and byte-level progress instead of sklearn's
# no-timeout urlretrieve. The URL/checksum/filename themselves come from
# sklearn's own ARCHIVE metadata at runtime to stay in lock-step; this size is
# only the progress-bar fallback when the server omits Content-Length.
TWENTY_NEWSGROUPS_DOWNLOAD_SIZE_MB = 14
BBC_NEWS_URL = "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip"
AG_NEWS_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DBPEDIA_URL = "https://s3.amazonaws.com/fast-ai-nlp/dbpedia_csv.tgz"
ARXIV_API_URL = "http://export.arxiv.org/api/query"
REUTERS21578_URL = "http://kdd.ics.uci.edu/databases/reuters21578/reuters21578.tar.gz"
GTZAN_URL = "https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz"
SPEECH_COMMANDS_V2_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
URBANSOUND8K_URL = "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz"
# TUT Sound Events 2017 (DCASE): long-form (~4 min) binaural street recordings.
# Unlike the other audio demos these are full uncut soundscapes, so a single
# file contains many sound events scattered over time - ideal for hands-on
# clipping practice.  The development set ships as two audio zips and the
# evaluation set as one; each entry is ``(url, slug)`` where *slug* names the
# per-archive extraction subdirectory.  Annotations (the ``meta`` zips) are
# intentionally not downloaded: the demo treats every recording as one
# undifferentiated "street" bucket so the user clips it themselves.
TUT_SOUND_EVENTS_2017_ARCHIVES = (
    (
        "https://zenodo.org/records/400516/files/TUT-sound-events-2017-development.audio.1.zip",
        "development_1",
    ),
    (
        "https://zenodo.org/records/400516/files/TUT-sound-events-2017-development.audio.2.zip",
        "development_2",
    ),
    (
        "https://zenodo.org/records/1040179/files/TUT-sound-events-2017-evaluation.audio.zip",
        "evaluation",
    ),
)
OXFORD_FLOWERS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
OXFORD_FLOWERS_LABELS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
FOOD101_URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
EUROSAT_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip"
STANFORD_DOGS_URL = "https://huggingface.co/datasets/Alanox/stanford-dogs/resolve/main/images.tar.gz"
PLACES365_URL = "http://data.csail.mit.edu/places/places365/val_256.tar"
# The official labels file ships inside a ~67 MB tarball alongside
# train/test/category lists.  The historical raw.githubusercontent.com
# mirror (CSAILVision/places365 master branch) started returning 404,
# so we now pull the canonical bundle from the MIT mirror and extract
# only the `places365_val.txt` member we need.
PLACES365_LABELS_FILELIST_URL = "http://data.csail.mit.edu/places/places365/filelist_places365-standard.tar"
PLACES365_LABELS_FILELIST_SIZE_MB = 65
UCSF_IDL_API_URL = "https://metadata.idl.ucsf.edu/solr/ltdl3/query"
UCSF_IDL_DOWNLOAD_URL = "https://download.industrydocuments.ucsf.edu"

# Revisited Oxford Buildings (ROxford5k): the canonical instance-retrieval
# benchmark and the structural embedder's demo dataset.  The 5063-image set is
# the original Oxford Buildings tarball (the "revisited" protocol reuses the same
# images with cleaned-up, ground-truth annotations shipped in a separate pickle).
ROXFORD_IMAGES_URL = "https://thor.robots.ox.ac.uk/oxbuildings/oxbuild_images-v1.tgz"
ROXFORD_GND_URL = "https://cmp.felk.cvut.cz/revisitop/data/datasets/roxford5k/gnd_roxford5k.pkl"

# OpenLogo (QMUL-OpenLogo): the structural embedder's instance-matching *logo*
# demo — 352 brands / 27k in-the-wild photos with ground-truth boxes, aggregated
# from 7 logo datasets (FlickrLogos-27/32, Logo32plus, BelgaLogos, WebLogo-2M
# test, Logo-in-the-Wild, SportsLogo).  It is distributed as a FiftyOne dataset
# on HuggingFace: a flat ``data/`` media folder plus a ``samples.json`` carrying
# per-image ``ground_truth`` detections (normalized ``[x, y, w, h]`` boxes).
# Because the media is thousands of loose files (no single archive), it is pulled
# with ``huggingface_hub.snapshot_download`` rather than the single-URL helper,
# and parsed from ``samples.json`` with the stdlib (no ``fiftyone`` dependency).
OPENLOGO_REPO_ID = "Voxel51/OpenLogo"

# Visual Genome (v1.4): a multi-label scene dataset of ~108k dense-annotated
# photos.  Images ship as two zips (the historical VG_100K / VG_100K_2 splits);
# object annotations (per-object name + pixel bounding box) ship as a separate
# JSON zip.  Unlike the other image demos this is multi-label ground truth — see
# docs/plans/visual-genome-dataset.md.
VISUAL_GENOME_IMAGES_URL = "https://cs.stanford.edu/people/rak248/VG_100K/images.zip"
VISUAL_GENOME_IMAGES2_URL = "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip"
VISUAL_GENOME_OBJECTS_URL = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip"

# HMDB51
HMDB51_URL = "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar"

# UCF101 full (ZIP mirror on HuggingFace - no auth required)
UCF101_FULL_URL = "https://huggingface.co/datasets/quchenyuan/UCF101-ZIP/resolve/main/UCF-101.zip"

# KTH Actions
KTH_BASE_URL = "https://www.csc.kth.se/cvap/actions/"
KTH_ACTIONS = ("walking", "jogging", "running", "boxing", "handwaving", "handclapping")

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
DBPEDIA_DOWNLOAD_SIZE_MB = 70
ARXIV_DOWNLOAD_SIZE_MB = 30
REUTERS21578_DOWNLOAD_SIZE_MB = 8
GTZAN_DOWNLOAD_SIZE_MB = 1200
SPEECH_COMMANDS_V2_DOWNLOAD_SIZE_MB = 2300
URBANSOUND8K_DOWNLOAD_SIZE_MB = 6000
# Dev audio.1 (~1.1 GB) + audio.2 (~213 MB) + eval audio (~388 MB).
TUT_SOUND_EVENTS_2017_DOWNLOAD_SIZE_MB = 1730
OXFORD_FLOWERS_DOWNLOAD_SIZE_MB = 330
FOOD101_DOWNLOAD_SIZE_MB = 5000
EUROSAT_DOWNLOAD_SIZE_MB = 90
STANFORD_DOGS_DOWNLOAD_SIZE_MB = 750
PLACES365_DOWNLOAD_SIZE_MB = 501
UCSF_IDL_DOWNLOAD_SIZE_MB = 50
ROXFORD_IMAGES_DOWNLOAD_SIZE_MB = 1850
OPENLOGO_DOWNLOAD_SIZE_MB = 4640
VISUAL_GENOME_IMAGES_DOWNLOAD_SIZE_MB = 9700
VISUAL_GENOME_IMAGES2_DOWNLOAD_SIZE_MB = 5300
VISUAL_GENOME_OBJECTS_DOWNLOAD_SIZE_MB = 110
HMDB51_DOWNLOAD_SIZE_MB = 2000
UCF101_FULL_DOWNLOAD_SIZE_MB = 6960
KTH_DOWNLOAD_SIZE_MB = 1150

ProgressCallback = Callable[[str, str, int, int], None]


def _default_progress() -> ProgressCallback:
    """Lazily resolve the progress callback for the current thread."""
    from vtscore.concurrency.progress import get_thread_progress

    cb = get_thread_progress()
    if cb is not None:
        return cb
    from vtscore.concurrency.progress import update_progress

    return update_progress


def _request_headers(url: str, headers: Optional[dict]) -> dict:
    """Merge caller *headers* with a HuggingFace bearer token when *url* is a Hub
    host.  The auth header is recomputed per hop so it follows redirects only to
    other HuggingFace hosts, never to a presigned CDN / Xet target."""
    merged = dict(headers or {})
    merged.update(auth_header_for_url(url))
    return merged


def _open_validated_stream(session: requests.Session, url: str, headers: Optional[dict] = None) -> requests.Response:
    """GET *url* as a stream, following redirects manually so every hop is
    re-checked by :func:`validate_url`.

    We follow redirects by hand (``allow_redirects=False``) so a public URL
    cannot redirect to an internal host (SSRF), bypassing the up-front check
    callers performed. The ``(connect, read)`` timeout fails fast on an
    unresponsive host and aborts if the server stalls for 60s mid-stream.

    Returns the final, non-redirect response; the caller owns closing it.
    """
    current_url = url
    response = session.get(
        current_url,
        stream=True,
        timeout=(10, 60),
        allow_redirects=False,
        headers=_request_headers(current_url, headers),
    )
    redirects = 0
    while response.is_redirect or response.is_permanent_redirect:
        if redirects >= _MAX_REDIRECTS:
            response.close()
            raise requests.TooManyRedirects(f"Exceeded {_MAX_REDIRECTS} redirects following {url}")
        location = response.headers.get("Location")
        if not location:
            break
        next_url = urljoin(current_url, location)
        validate_url(next_url)
        response.close()
        current_url = next_url
        response = session.get(
            current_url,
            stream=True,
            timeout=(10, 60),
            allow_redirects=False,
            headers=_request_headers(current_url, headers),
        )
        redirects += 1
    return response


def _total_size_from_headers(response: requests.Response, downloaded: int, expected_size: int) -> int:
    """Determine the file's full size from response headers.

    Prefers the ``Content-Range`` total of a 206 partial response
    (``bytes 8104304-1183006719/1183006720``); otherwise adds the remaining
    ``Content-Length`` to what is already on disk, and finally falls back to
    the caller-supplied *expected_size*.
    """
    content_range = response.headers.get("Content-Range", "")
    if "/" in content_range:
        try:
            return int(content_range.rsplit("/", 1)[1])
        except ValueError:
            pass
    content_length = int(response.headers.get("content-length", 0))
    if content_length:
        return downloaded + content_length
    return expected_size


def _backoff_and_notify(
    on_progress: ProgressCallback, dest_path: Path, attempt: int, downloaded: int, total_size: int
) -> None:
    """Report a recoverable interruption and sleep for an exponential backoff."""
    backoff = min(_RETRY_BACKOFF_BASE_S * (2 ** (attempt - 1)), _RETRY_BACKOFF_MAX_S)
    on_progress(
        "downloading",
        f"Connection interrupted at {downloaded:,} bytes - resuming {dest_path.name} "
        f"(attempt {attempt + 1}/{_MAX_DOWNLOAD_ATTEMPTS})...",
        downloaded,
        total_size,
    )
    time.sleep(backoff)


def _gated_error(url: str, status: int) -> GatedResourceError:
    """Build a short, user-facing :class:`GatedResourceError` for a 401/403.

    The wording is tailored to whether the request hit the HuggingFace Hub and
    whether a token is already stored, so the message tells the user the right
    next step (sign in, or request access) without dumping the URL or an HTTP
    body that would overflow the UI.
    """
    from vtscore.security.hf_auth import is_authenticated  # noqa: PLC0415

    is_hf = bool(auth_header_for_url(url) or "huggingface.co" in url or "hf.co" in url)
    if is_hf:
        if is_authenticated():
            msg = (
                "This dataset is gated on HuggingFace and your signed-in account "
                "hasn't been granted access. Open the dataset page on huggingface.co, "
                "accept its terms, then retry."
            )
        else:
            msg = (
                "This dataset is gated on HuggingFace. Sign in with HuggingFace "
                "(in Settings) using an account that has access, then retry."
            )
    else:
        msg = "Access denied: this dataset requires authentication that VTSearch doesn't have."
    return GatedResourceError(msg, url=url, status=status)


def download_file_with_progress(  # noqa: C901
    url: str,
    dest_path: Path,
    expected_size: int = 0,
    on_progress: Optional[ProgressCallback] = None,
) -> None:
    """Download a file from a URL to a local path, reporting byte-level progress.

    Streams the HTTP response in 8 KB chunks and calls *on_progress* after each
    chunk so that a polling client can track download progress.

    The transfer is resilient to mid-stream connection drops: if the server
    closes the connection early (surfacing as ``ChunkedEncodingError`` /
    ``IncompleteRead``), times out, or returns a transient 5xx/429, the download
    retries up to ``_MAX_DOWNLOAD_ATTEMPTS`` times with exponential backoff,
    resuming from the bytes already on disk via an HTTP ``Range`` request rather
    than restarting from zero. Servers that ignore ``Range`` (resending the full
    body) or transport-compress the response are handled by restarting the file.

    Args:
        url: The HTTP/HTTPS URL to download from.
        dest_path: Local filesystem path where the downloaded file will be written.
        expected_size: Expected file size in bytes, used as a fallback when the
            server does not supply a ``Content-Length`` header. Pass 0 (default)
            if the size is unknown.
        on_progress: Optional progress callback. Falls back to the
            application-wide ``update_progress`` when ``None``.

    Raises:
        requests.HTTPError: If the server returns a non-retryable error status.
        requests.exceptions.ChunkedEncodingError / ConnectionError / Timeout: If
            the connection keeps failing after the final retry attempt.
    """
    if on_progress is None:
        on_progress = _default_progress()

    session = requests.Session()
    downloaded = 0  # bytes already on disk at dest_path
    total_size = 0  # full size of the file once known
    # Resume relies on the server honoring Range requests on the *raw* body.
    # If a hop transport-compresses the response (Content-Encoding), byte
    # offsets no longer line up with iter_content's decoded output, so we fall
    # back to restarting the file rather than risk a corrupt resume.
    resume_ok = True

    for attempt in range(1, _MAX_DOWNLOAD_ATTEMPTS + 1):
        last_attempt = attempt == _MAX_DOWNLOAD_ATTEMPTS
        use_range = downloaded > 0 and resume_ok
        if not use_range:
            downloaded = 0  # first attempt, or a resume we can't safely do -> start over
        headers = {"Range": f"bytes={downloaded}-"} if use_range else None

        response = None
        try:
            response = _open_validated_stream(session, url, headers)

            # Transient server-side error: back off and retry (resuming if we can).
            if response.status_code in _RETRYABLE_STATUS and not last_attempt:
                _backoff_and_notify(on_progress, dest_path, attempt, downloaded, total_size)
                continue
            # Auth-required: this is gated content we can't fetch with the
            # credentials we have.  Surface a short, actionable message instead
            # of a raw HTTPError, and don't retry (it can't succeed unchanged).
            if response.status_code in _AUTH_REQUIRED_STATUS:
                raise _gated_error(url, response.status_code)
            response.raise_for_status()

            if response.headers.get("Content-Encoding"):
                resume_ok = False
            # We asked to resume but the server resent the whole body (200, not
            # 206 Partial Content): it ignored Range, so restart from scratch.
            if use_range and response.status_code != 206:
                downloaded = 0

            if total_size == 0:
                total_size = _total_size_from_headers(response, downloaded, expected_size)

            mode = "ab" if downloaded else "wb"
            with open(dest_path, mode) as f:
                for chunk in response.iter_content(chunk_size=8192):
                    downloaded += f.write(chunk)
                    on_progress("downloading", f"Downloading {dest_path.name}...", downloaded, total_size)
            return  # stream consumed cleanly
        except _RETRYABLE_EXCEPTIONS:
            if last_attempt:
                raise
            _backoff_and_notify(on_progress, dest_path, attempt, downloaded, total_size)
        finally:
            if response is not None:
                response.close()


def download_file_atomic(
    url: str,
    final_path: Path,
    expected_size: int,
    on_progress: ProgressCallback,
) -> None:
    """Download *url* to *final_path* via a unique temp file + atomic rename.

    :func:`download_file_with_progress` deliberately leaves partial bytes at
    its destination when every retry fails (its resume feature depends on
    that), so callers that gate the download on ``final_path.exists()`` must
    never point it at the final path directly - a failed run would leave a
    truncated file that every subsequent run treats as a complete cached
    copy.  This wrapper downloads to a sibling temp file and only publishes
    the final path once the stream completed cleanly.  A concurrent
    completed download wins; the temp file is always removed.
    """
    final_path.parent.mkdir(parents=True, exist_ok=True)
    unique_id = uuid.uuid4().hex[:8]
    temp_path = final_path.parent / f".dl_{unique_id}_{final_path.name}"
    try:
        download_file_with_progress(url, temp_path, expected_size, on_progress)
        if not final_path.exists():
            try:
                os.rename(temp_path, final_path)
            except OSError:
                pass  # Another download finished first
    finally:
        temp_path.unlink(missing_ok=True)


_GZIP_MAGIC = b"\x1f\x8b"
_ZIP_MAGIC = b"PK"
# Uncompressed tar: first 257 bytes contain "ustar" at offset 257,
# but a simpler heuristic is that the file does NOT start with common
# non-archive signatures (HTML, JSON, plain text error pages).
_HTML_SIGNATURES = (b"<", b"<!", b"{")


def _validate_archive(archive_path: Path, archive_name: str, dataset_name: str) -> None:
    """Check that a downloaded file looks like a genuine archive.

    Deletes the file and raises ``RuntimeError`` with a user-friendly
    message when the content does not match the expected format.
    """
    suffix = archive_name.lower()
    try:
        # Read only the magic bytes - read_bytes() would materialise the
        # whole (potentially multi-GB) archive in memory to inspect 4 bytes.
        with open(archive_path, "rb") as f:
            header = f.read(4)
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


def _extract_archive(
    archive_path: Path, archive_name: str, dest_dir: Path, dataset_name: str, on_progress: ProgressCallback
) -> None:
    """Extract *archive_path* into *dest_dir*, dispatching by filename suffix.

    Supports ``.tar.gz`` / ``.tgz`` (gzip tar), ``.tar`` (uncompressed tar), and
    ``.zip`` archives.  Tar members are extracted with ``filter="data"`` (which
    rejects unsafe absolute/traversal paths); zip members are validated against
    *dest_dir* to guard against path traversal (zip-slip).  Raises ``ValueError``
    for an unsupported archive format.
    """
    suffix = archive_name.lower()
    if suffix.endswith((".tar.gz", ".tgz", ".tar")):
        # Iterate lazily instead of calling getmembers() - the latter must
        # decompress the entire gzip stream just to read tar headers, then
        # extraction decompresses it *again*.  Lazy iteration decompresses
        # once and avoids a minutes-long stall on multi-GB archives.
        # Use "r:*" for gzip tars to auto-detect compression - some CDNs (e.g.
        # HuggingFace Xet) transparently decompress .tar.gz files during transfer.
        mode = "r:" if suffix.endswith(".tar") else "r:*"
        total_bytes = archive_path.stat().st_size
        with open(archive_path, "rb") as raw_f:
            with tarfile.open(fileobj=raw_f, mode=mode) as tar_ref:
                for i, member in enumerate(tar_ref):
                    if i % 100 == 0:
                        on_progress("downloading", f"Extracting {dataset_name}...", raw_f.tell(), total_bytes)
                    tar_ref.extract(member, dest_dir, filter="data")
        on_progress("downloading", f"Extracting {dataset_name}...", total_bytes, total_bytes)
    elif suffix.endswith(".zip"):
        from vtscore.datasets.archive import _reject_traversal

        dest_resolved = Path(dest_dir).resolve()
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            members = zip_ref.namelist()
            total = len(members)
            for i, member in enumerate(members):
                if i % 100 == 0 or i == total - 1:
                    on_progress("downloading", f"Extracting {dataset_name}...", i + 1, total)
                # Guard against path traversal in zip entries.  Shares the
                # strict check with archive.py: the previous inline
                # startswith() prefix test lacked a trailing separator, so
                # a sibling dir with the dest as a name prefix passed.
                _reject_traversal(dest_resolved, member)
                zip_ref.extract(member, dest_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_name}")


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

        _extract_archive(temp_archive, archive_name, temp_extract, dataset_name, on_progress)

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

        # extract_to already existed (e.g. it is DATA_DIR) - move children.
        _move_tree_contents(temp_extract, extract_to)
    finally:
        temp_archive.unlink(missing_ok=True)
        if temp_extract.exists():
            shutil.rmtree(temp_extract, ignore_errors=True)
