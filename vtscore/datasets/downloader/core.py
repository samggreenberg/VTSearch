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
from typing import Callable, Literal, Optional
from urllib.parse import urljoin

import requests

from vtscore.config import DATA_DIR
from vtscore.security.archive import safe_tar_extract
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
# VGGFace2 test split (500 identities, folder-per-identity ``test/n######/*.jpg``),
# served as a single gzip tarball from the HuggingFace mirror of the (now
# account-gated) Oxford VGG release.  We use only the test split - it is the
# smaller half (~2 GB vs the 37 GB train tar) yet still carries hundreds of
# in-the-wild, pose/age-varied photos per person, which is exactly what the
# Faces demo needs.
VGGFACE2_TEST_URL = "https://huggingface.co/datasets/ProgramComputer/VGGFace2/resolve/main/data/vggface2_test.tar.gz"
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
# Clotho (v1) evaluation split: 1045 real-world Freesound clips, 15-30s each,
# hosted on Zenodo as a single ``.7z`` archive (extraction needs py7zr).  Clotho
# is an audio *captioning* set; VTSearch imports only the audio (no class
# labels) into one bucket, so it serves as the compositional-real-world-sound
# playground for text->audio (CLAP) retrieval.  The much larger 3.4 GB
# development split is intentionally skipped - the eval split alone is a
# GTZAN-sized demo.
CLOTHO_EVAL_URL = "https://zenodo.org/records/3490684/files/clotho_audio_evaluation.7z"
OXFORD_FLOWERS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
OXFORD_FLOWERS_LABELS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
FOOD101_URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
EUROSAT_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip"
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
# Parallel workers for the OpenLogo snapshot. With ~27k tiny files the download is
# latency-bound on per-file round trips, so we widen snapshot_download's default of
# 8 to fetch many concurrently. Kept modest to stay under HF's public rate limits.
OPENLOGO_DOWNLOAD_WORKERS = 16

# Visual Genome (v1.4): a multi-label scene dataset of ~108k dense-annotated
# photos.  Images ship as two zips (the historical VG_100K / VG_100K_2 splits);
# object annotations (per-object name + pixel bounding box) ship as a separate
# JSON zip.  Unlike the other image demos this is multi-label ground truth — see
# docs/plans/visual-genome-dataset.md.
# Enrico (Enhanced Rico): 1,460 Android mobile-UI screenshots (a curated,
# de-duplicated Rico subset), each labeled with one of 20 "design topic"
# categories (screen function: Login, Chat, Maps, Settings, Gallery, …).  MIT
# licensed.  Ships as a small ``screenshots.zip`` (JPEGs named
# ``<rico_screen_id>-screenshot.jpg``) plus a separate ``design_topics.csv``
# (``screen_id,topic``) that carries the labels — the two are fetched together
# by ``download_enrico``.  This is VTSearch's born-digital *screenshot* demo:
# unlike the natural-photo image demos, the content is rendered UI, so it
# stresses the embedder on digitally-native imagery.
ENRICO_SCREENSHOTS_URL = "https://userinterfaces.aalto.fi/enrico/resources/screenshots.zip"
# The Aalto-hosted ``design_topics.csv`` now 404s; the upstream Enrico GitHub
# repo still serves the identical ``screen_id,topic`` file, so we pull it there.
ENRICO_TOPICS_URL = "https://raw.githubusercontent.com/luileito/enrico/master/design_topics.csv"

# RICO-Screen2Words: 22,417 Android mobile-UI screenshots (built on Rico), each
# carrying its app's Google Play *category* plus human caption summaries.
# CC-BY-4.0.  Distributed on the Hub as parquet shards whose ``image`` column is
# an Image feature (embedded JPEG bytes); ``download_rico_screen2words`` pulls
# the train split (8 shards ≈ 1.7 GB), decodes each screenshot to a
# ``<category>/<screenId>.jpg`` file, then deletes the parquet.  Born-digital
# mobile-UI screenshots — a second, harder-labelled screenshot demo alongside
# Enrico (app genre vs. screen function).
RICO_SCREEN2WORDS_REPO_ID = "bevaya/RICO-Screen2Words"
RICO_SCREEN2WORDS_SHARDS = [f"data/train-{i:05d}-of-00008.parquet" for i in range(8)]

# RVL-CDIP: 16-class document-image classification.  The canonical
# ``aharley/rvl_cdip`` is a 38 GB tarball (impractical as a demo); instead we
# pull a demo-sized, class-balanced 100-images-per-class parquet mirror whose
# ``image``/``label`` columns decode straight into a folder-per-class tree.
# All three splits (train 50 + test 25 + validation 25 per class = ~1,600
# images across 16 classes, ~180 MB) are pulled so every class is represented.
# The shard filenames carry a content hash, so they are resolved at download
# time rather than hardcoded.  (The former ``umair894`` 300-per-class mirror
# was abandoned: its single shard held only the ``invoice`` class.)
RVL_CDIP_REPO_ID = "jordyvl/rvl_cdip_100_examples_per_class"

VISUAL_GENOME_IMAGES_URL = "https://cs.stanford.edu/people/rak248/VG_100K/images.zip"
VISUAL_GENOME_IMAGES2_URL = "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip"
VISUAL_GENOME_OBJECTS_URL = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip"

# HMDB51
# The original Serre-Lab RAR (hmdb51_org.rar) is dead — its host now redirects
# to a homepage that serves HTML — and the lab's current page offers the data
# only via Google Drive links that 404.  Use a public HuggingFace zip mirror
# that carries the same ``hmdb51/<category>/*.avi`` tree (no unrar needed).
HMDB51_URL = "https://huggingface.co/datasets/jili5044/hmdb51/resolve/main/hmdb51.zip"

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
VGGFACE2_TEST_DOWNLOAD_SIZE_MB = 1935
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
CLOTHO_EVAL_DOWNLOAD_SIZE_MB = 1200
OXFORD_FLOWERS_DOWNLOAD_SIZE_MB = 330
FOOD101_DOWNLOAD_SIZE_MB = 5000
EUROSAT_DOWNLOAD_SIZE_MB = 90
PLACES365_DOWNLOAD_SIZE_MB = 501
UCSF_IDL_DOWNLOAD_SIZE_MB = 50
ROXFORD_IMAGES_DOWNLOAD_SIZE_MB = 1850
ENRICO_DOWNLOAD_SIZE_MB = 110
RICO_SCREEN2WORDS_DOWNLOAD_SIZE_MB = 1720
RVL_CDIP_DOWNLOAD_SIZE_MB = 180
OPENLOGO_DOWNLOAD_SIZE_MB = 4640
# Verified against the servers' Content-Length (2026-07-14): images.zip is
# 9,730,308,001 B (9280 MiB), images2.zip 5,471,658,058 B (5218 MiB), and
# objects.json.zip 55,323,929 B (53 MiB).  These only seed the "Downloading
# ~X GB" message / progress fallback — the real total comes from Content-Length.
VISUAL_GENOME_IMAGES_DOWNLOAD_SIZE_MB = 9280
VISUAL_GENOME_IMAGES2_DOWNLOAD_SIZE_MB = 5218
VISUAL_GENOME_OBJECTS_DOWNLOAD_SIZE_MB = 53
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


def fetch_url_bytes(url: str) -> bytes:
    """GET *url* into memory under the same SSRF policy as a file download.

    The in-memory twin of :func:`download_file_with_progress`, for callers that
    want a small payload (a URL-backed media's content) rather than a file on
    disk.  It runs :func:`~vtscore.security.url_validation.validate_url` up
    front — which is what confines the fetch to ``http(s)`` and refuses
    private/internal hosts — and then goes through
    :func:`_open_validated_stream`, so every redirect hop is re-checked and no
    public URL can bounce the request onto an internal one.

    Use this instead of :func:`urllib.request.urlopen` for anything whose URL
    came from outside the server: urllib's default opener also services
    ``file://`` and ``ftp://``, which turns a fetch into an arbitrary local
    file read.

    Raises:
        ValueError: If *url* fails the SSRF guard (bad scheme, no hostname,
            private/internal address).
        requests.HTTPError: If the server returns an error status.
    """
    validate_url(url)
    session = requests.Session()
    try:
        response = _open_validated_stream(session, url)
        try:
            response.raise_for_status()
            return response.content
        finally:
            response.close()
    finally:
        session.close()


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
            # Emit progress at most every ~200 ms rather than once per chunk: a
            # multi-GB download makes hundreds of thousands of chunk writes, and
            # a per-chunk SSE push is real CPU that can throttle the transfer.
            # Throttle on wall-clock (time.monotonic), not chunk count, so slow
            # links still update; always emit a final 100% call after the loop.
            last_emit = time.monotonic()
            with open(dest_path, mode) as f:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    downloaded += f.write(chunk)
                    now = time.monotonic()
                    if now - last_emit >= 0.2:
                        last_emit = now
                        on_progress("downloading", f"Downloading {dest_path.name}...", downloaded, total_size)
            on_progress("downloading", f"Downloading {dest_path.name}...", downloaded, total_size)
            return  # stream consumed cleanly
        except _RETRYABLE_EXCEPTIONS:
            if last_attempt:
                raise
            _backoff_and_notify(on_progress, dest_path, attempt, downloaded, total_size)
        finally:
            if response is not None:
                response.close()


def fetch_remote_signature(url: str) -> Optional[str]:
    """Fetch a lightweight signature (ETag / Last-Modified / size) for *url*.

    Used to detect when a remote archive has changed so a caller's extraction
    cache can invalidate instead of serving stale bytes forever. Issues a
    ranged GET for a single byte (more universally honoured than HEAD by the
    flaky third-party CDNs these archives live on) and reads only the
    response headers.

    Returns ``None`` if the probe fails outright or the server's response
    carries none of the three signals - callers should fail open (trust an
    existing cache) rather than block on a CDN that doesn't answer.
    """
    try:
        validate_url(url)
        session = requests.Session()
        response = _open_validated_stream(session, url, headers={"Range": "bytes=0-0"})
    except Exception:
        return None
    try:
        if response.status_code >= 400:
            return None
        etag = response.headers.get("ETag", "")
        last_modified = response.headers.get("Last-Modified", "")
        content_range = response.headers.get("Content-Range", "")
        total = content_range.rsplit("/", 1)[-1] if "/" in content_range else response.headers.get("Content-Length", "")
        if not etag and not last_modified and not total:
            return None
        return f"{etag}|{last_modified}|{total}"
    except Exception:
        return None
    finally:
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
# 7-Zip archives begin with the 6-byte signature "7z\xbc\xaf\x27\x1c"; the
# first 4 bytes are enough to tell a real archive from an HTML error page.
_SEVENZIP_MAGIC = b"7z\xbc\xaf"
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
    elif suffix.endswith(".7z"):
        ok = header == _SEVENZIP_MAGIC
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


def _extract_7z(archive_path: Path, dest_dir: Path, dataset_name: str, on_progress: ProgressCallback) -> None:
    """Extract a ``.7z`` archive into *dest_dir* using ``py7zr``.

    ``py7zr`` is a declared dependency (installed by ``install.sh``), but only
    the Clotho audio demo ships as ``.7z``, so it is still imported lazily and a
    missing install surfaces as a short, actionable ``RuntimeError`` rather than
    an ``ImportError``.  Every member name is validated against *dest_dir* before
    extraction to guard against path traversal (the same protection the zip
    branch applies).
    """
    try:
        import py7zr  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError(
            f"Extracting {dataset_name} needs the 'py7zr' package to read .7z archives. "
            f"Install it with 'pip install py7zr' and try again."
        ) from exc

    from vtscore.datasets.archive import _reject_traversal  # noqa: PLC0415

    dest_resolved = Path(dest_dir).resolve()
    on_progress("extracting", f"Extracting {dataset_name}...", 0, 0)
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        for name in archive.getnames():
            _reject_traversal(dest_resolved, name)
        # Safe: every member name was traversal-checked against dest_dir above.
        archive.extractall(path=dest_dir)  # noqa: S202
    on_progress("extracting", f"Extracting {dataset_name}...", 1, 1)


def _extract_tar(
    archive_path: Path,
    mode: Literal["r:", "r:*"],
    dest_dir: Path,
    dataset_name: str,
    on_progress: ProgressCallback,
    member_filter: Optional[Callable[[str], bool]],
    flatten: bool,
) -> None:
    # Iterate lazily instead of calling getmembers() - the latter must
    # decompress the entire gzip stream just to read tar headers, then
    # extraction decompresses it *again*.  Lazy iteration decompresses
    # once and avoids a minutes-long stall on multi-GB archives.
    total_bytes = archive_path.stat().st_size
    with open(archive_path, "rb") as raw_f, tarfile.open(fileobj=raw_f, mode=mode) as tar_ref:
        for i, member in enumerate(tar_ref):
            if i % 100 == 0:
                on_progress("extracting", f"Extracting {dataset_name}...", raw_f.tell(), total_bytes)
            if member_filter is not None and not member_filter(member.name):
                continue
            if not flatten:
                safe_tar_extract(tar_ref, member, dest_dir)
                continue
            if member.isdir():
                continue
            src = tar_ref.extractfile(member)
            if src is None:
                continue
            dest = Path(dest_dir) / Path(member.name).name
            with src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
    on_progress("extracting", f"Extracting {dataset_name}...", total_bytes, total_bytes)


def _extract_zip(
    archive_path: Path,
    dest_dir: Path,
    dataset_name: str,
    on_progress: ProgressCallback,
    member_filter: Optional[Callable[[str], bool]],
    flatten: bool,
) -> None:
    from vtscore.datasets.archive import _reject_traversal

    dest_resolved = Path(dest_dir).resolve()
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        members = zip_ref.namelist()
        total = len(members)
        for i, member in enumerate(members):
            if i % 100 == 0 or i == total - 1:
                on_progress("extracting", f"Extracting {dataset_name}...", i + 1, total)
            if member_filter is not None and not member_filter(member):
                continue
            if not flatten:
                # Guard against path traversal in zip entries.  Shares the
                # strict check with archive.py: the previous inline
                # startswith() prefix test lacked a trailing separator, so
                # a sibling dir with the dest as a name prefix passed.
                _reject_traversal(dest_resolved, member)
                zip_ref.extract(member, dest_dir)
                continue
            if member.endswith("/"):
                continue
            dest = Path(dest_dir) / Path(member).name
            with zip_ref.open(member) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)


def _extract_archive(
    archive_path: Path,
    archive_name: str,
    dest_dir: Path,
    dataset_name: str,
    on_progress: ProgressCallback,
    *,
    member_filter: Optional[Callable[[str], bool]] = None,
    flatten: bool = False,
) -> None:
    """Extract *archive_path* into *dest_dir*, dispatching by filename suffix.

    Supports ``.tar.gz`` / ``.tgz`` (gzip tar), ``.tar`` (uncompressed tar),
    ``.zip``, and ``.7z`` archives.  Tar members go through
    :func:`~vtscore.security.archive.safe_tar_extract` (which rejects unsafe
    absolute/traversal/link paths); zip members are validated against
    *dest_dir* to guard against path traversal (zip-slip).  Raises
    ``ValueError`` for an unsupported archive format.

    Args:
        member_filter: Optional predicate on the member's archive path (e.g.
            ``lambda m: m.lower().endswith(".wav")``).  Members for which it
            returns ``False`` are skipped entirely.  Ignored for ``.7z``
            (only the common zip/tar downloaders that need filtering use it).
        flatten: When ``True``, every extracted member is written directly
            into *dest_dir* under its basename (``Path(member).name``),
            discarding any directory structure inside the archive.  Because
            only the basename is used, traversal is not a concern for
            flattened members.  Directory entries are skipped.  Ignored for
            ``.7z``.
    """
    suffix = archive_name.lower()
    if suffix.endswith((".tar.gz", ".tgz", ".tar")):
        # Use "r:*" for gzip tars to auto-detect compression - some CDNs (e.g.
        # HuggingFace Xet) transparently decompress .tar.gz files during transfer.
        mode = "r:" if suffix.endswith(".tar") else "r:*"
        _extract_tar(archive_path, mode, dest_dir, dataset_name, on_progress, member_filter, flatten)
    elif suffix.endswith(".zip"):
        _extract_zip(archive_path, dest_dir, dataset_name, on_progress, member_filter, flatten)
    elif suffix.endswith(".7z"):
        _extract_7z(archive_path, dest_dir, dataset_name, on_progress)
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
    is_complete: Optional[Callable[[], bool]] = None,
    member_filter: Optional[Callable[[str], bool]] = None,
    flatten: bool = False,
) -> None:
    """Download an archive and extract it unless it is already complete.

    Supports ``.tar.gz`` / ``.tgz`` (gzip tar), ``.tar`` (uncompressed tar),
    ``.zip``, and ``.7z`` archives.  The archive file is deleted after
    successful extraction to reclaim disk space.

    Each invocation downloads and extracts into unique temporary paths so that
    concurrent calls targeting the same archive do not interfere with each
    other.  After extraction the content is moved to the final location; if
    another call finished first the duplicate is simply cleaned up.

    The temp archive and temp extraction are spooled onto the *destination*
    filesystem (next to, or inside, the **resolved** *extract_to*) rather than
    under ``DATA_DIR``.  When *extract_to* (or a parent) is a symlink onto a
    different, bigger volume - the shared demo-cache setup, or any relocated
    dataset dir - this keeps the multi-GB temp bytes off ``DATA_DIR``'s
    (possibly small) volume and makes the final publish a true same-filesystem
    rename instead of a cross-device copy.  When nothing is symlinked the
    resolved target sits under ``DATA_DIR``, so this reduces to the previous
    behavior.

    Args:
        url: Download URL for the archive.
        archive_name: Filename to save the downloaded archive as (e.g.
            ``"genres.tar.gz"``).
        extract_to: Directory into which the archive contents are extracted.
        check_path: Path whose existence signals that extraction is already
            complete (often the same as *extract_to* or a subdirectory of it).
            Ignored when *is_complete* is given.
        download_size_mb: Expected download size in megabytes (for progress).
        dataset_name: Human-readable dataset name used in progress messages.
        on_progress: Progress callback.
        is_complete: Optional predicate that returns ``True`` when the dataset
            is already fully extracted.  Prefer this over *check_path* when a
            bare directory (e.g. an empty, partially-extracted folder) would
            otherwise be a false positive that blocks re-download: the predicate
            can require the expected content (labeled images, etc.) to be
            present rather than merely that a path exists.
        member_filter: Optional per-member predicate forwarded to
            :func:`_extract_archive`; see its docstring.
        flatten: Forwarded to :func:`_extract_archive`; see its docstring.
    """

    def _complete() -> bool:
        return is_complete() if is_complete is not None else check_path.exists()

    if _complete():
        return

    # Spool both temp files onto the filesystem where the extracted content will
    # ultimately live.  If extract_to already resolves to a directory (e.g.
    # DATA_DIR itself, or a symlink onto a relocated cache), the publish moves
    # children *into* it, so spool inside it; otherwise the publish renames the
    # temp tree *to* extract_to, so spool next to it (in its resolved parent).
    # Following symlinks here means the temp bytes and the final rename stay on
    # one filesystem instead of crossing a device boundary.
    resolved = extract_to.resolve()
    spool_dir = resolved if resolved.is_dir() else resolved.parent
    unique_id = uuid.uuid4().hex[:8]
    temp_archive = spool_dir / f".dl_{unique_id}_{archive_name}"
    temp_extract = spool_dir / f".extract_{unique_id}_{extract_to.name}"
    spool_dir.mkdir(parents=True, exist_ok=True)

    try:
        on_progress("downloading", f"Starting {dataset_name} download...", 0, 0)
        download_file_with_progress(url, temp_archive, download_size_mb * 1024 * 1024, on_progress)

        # Another download may have finished while we were downloading.
        if _complete():
            return

        # Validate the downloaded file looks like a real archive before trying
        # to extract it.  A common failure mode is the server returning an HTML
        # error page (e.g. 404/503) which gets saved with a .tar.gz extension.
        _validate_archive(temp_archive, archive_name, dataset_name)

        on_progress("extracting", f"Extracting {dataset_name}...", 0, 0)
        temp_extract.mkdir(parents=True, exist_ok=True)

        _extract_archive(
            temp_archive,
            archive_name,
            temp_extract,
            dataset_name,
            on_progress,
            member_filter=member_filter,
            flatten=flatten,
        )

        # Another download may have finished while we were extracting.
        if _complete():
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
