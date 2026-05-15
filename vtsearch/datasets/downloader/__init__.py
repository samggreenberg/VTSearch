"""Dataset downloading utilities.

This package is split into sub-modules by media type:

- :mod:`~vtsearch.datasets.downloader.core` — shared constants, progress helpers,
  ``download_file_with_progress``, archive validation & extraction
- :mod:`~vtsearch.datasets.downloader.audio` — ESC-50, GTZAN, Speech Commands v2,
  UrbanSound8K
- :mod:`~vtsearch.datasets.downloader.images` — CIFAR-10, Caltech-101/256, Oxford
  Flowers, Food-101, EuroSAT, Stanford Dogs, Places365
- :mod:`~vtsearch.datasets.downloader.video` — UCF-101 subset
- :mod:`~vtsearch.datasets.downloader.text` — 20 Newsgroups, BBC News, AG News, IMDB
- :mod:`~vtsearch.datasets.downloader.documents` — UCSF Industry Documents

All public symbols are re-exported here for backward compatibility so that
``from vtsearch.datasets.downloader import download_esc50`` continues to work.
"""

# Core utilities & constants
from vtsearch.datasets.downloader.core import (
    AG_NEWS_DOWNLOAD_SIZE_MB,
    AG_NEWS_URL,
    ARXIV_API_URL,
    ARXIV_DOWNLOAD_SIZE_MB,
    BBC_NEWS_DOWNLOAD_SIZE_MB,
    BBC_NEWS_URL,
    CALTECH101_DOWNLOAD_SIZE_MB,
    CALTECH101_URL,
    CALTECH256_DOWNLOAD_SIZE_MB,
    CALTECH256_URL,
    CIFAR10_DOWNLOAD_SIZE_MB,
    CIFAR10_URL,
    DBPEDIA_DOWNLOAD_SIZE_MB,
    DBPEDIA_URL,
    ESC50_DOWNLOAD_SIZE_MB,
    ESC50_URL,
    EUROSAT_DOWNLOAD_SIZE_MB,
    EUROSAT_URL,
    FOOD101_DOWNLOAD_SIZE_MB,
    FOOD101_URL,
    GTZAN_DOWNLOAD_SIZE_MB,
    GTZAN_URL,
    HMDB51_DOWNLOAD_SIZE_MB,
    HMDB51_URL,
    IMAGE_DIR,
    IMDB_DOWNLOAD_SIZE_MB,
    IMDB_URL,
    REUTERS21578_DOWNLOAD_SIZE_MB,
    REUTERS21578_URL,
    KTH_ACTIONS,
    KTH_BASE_URL,
    KTH_DOWNLOAD_SIZE_MB,
    OXFORD_FLOWERS_DOWNLOAD_SIZE_MB,
    OXFORD_FLOWERS_LABELS_URL,
    OXFORD_FLOWERS_URL,
    PLACES365_DOWNLOAD_SIZE_MB,
    PLACES365_LABELS_FILELIST_SIZE_MB,
    PLACES365_LABELS_FILELIST_URL,
    PLACES365_URL,
    SAMPLE_VIDEOS_URL,
    SPEECH_COMMANDS_V2_DOWNLOAD_SIZE_MB,
    SPEECH_COMMANDS_V2_URL,
    STANFORD_DOGS_DOWNLOAD_SIZE_MB,
    STANFORD_DOGS_URL,
    UCF101_FULL_DOWNLOAD_SIZE_MB,
    UCF101_FULL_URL,
    UCF101_SUBSET_DOWNLOAD_SIZE_MB,
    UCF101_SUBSET_URL,
    UCSF_IDL_API_URL,
    UCSF_IDL_DOWNLOAD_URL,
    UCSF_IDL_DOWNLOAD_SIZE_MB,
    URBANSOUND8K_DOWNLOAD_SIZE_MB,
    URBANSOUND8K_URL,
    VIDEO_DIR,
    ProgressCallback,
    _default_progress,
    _download_and_extract,
    _move_tree_contents,
    _validate_archive,
    download_file_with_progress,
)

# Audio downloaders
from vtsearch.datasets.downloader.audio import (
    download_esc50,
    download_gtzan,
    download_speech_commands_v2,
    download_urbansound8k,
)

# Image downloaders
from vtsearch.datasets.downloader.images import (
    download_caltech101,
    download_caltech256,
    download_cifar10,
    download_eurosat,
    download_food101,
    download_oxford_flowers,
    download_places365,
    download_stanford_dogs,
)

# Video downloaders
from vtsearch.datasets.downloader.video import (
    download_hmdb51,
    download_kth,
    download_ucf101_full,
    download_ucf101_subset,
)

# Text downloaders
from vtsearch.datasets.downloader.text import (
    ARXIV_DEFAULT_CATEGORIES,
    DBPEDIA14_CLASSES,
    REUTERS21578_TOP_TOPICS,
    _find_bbc_root,
    download_20newsgroups,
    download_ag_news,
    download_arxiv_abstracts,
    download_bbc_news,
    download_dbpedia,
    download_imdb,
    download_reuters21578,
)

# Document downloaders
from vtsearch.datasets.downloader.documents import download_ucsf_documents

__all__ = [
    # Core (private helpers re-exported for tests)
    "_default_progress",
    "_download_and_extract",
    "_move_tree_contents",
    "_validate_archive",
    # Core
    "AG_NEWS_DOWNLOAD_SIZE_MB",
    "AG_NEWS_URL",
    "ARXIV_API_URL",
    "ARXIV_DEFAULT_CATEGORIES",
    "ARXIV_DOWNLOAD_SIZE_MB",
    "BBC_NEWS_DOWNLOAD_SIZE_MB",
    "BBC_NEWS_URL",
    "CALTECH101_DOWNLOAD_SIZE_MB",
    "CALTECH101_URL",
    "CALTECH256_DOWNLOAD_SIZE_MB",
    "CALTECH256_URL",
    "CIFAR10_DOWNLOAD_SIZE_MB",
    "CIFAR10_URL",
    "DBPEDIA14_CLASSES",
    "DBPEDIA_DOWNLOAD_SIZE_MB",
    "DBPEDIA_URL",
    "ESC50_DOWNLOAD_SIZE_MB",
    "ESC50_URL",
    "EUROSAT_DOWNLOAD_SIZE_MB",
    "EUROSAT_URL",
    "FOOD101_DOWNLOAD_SIZE_MB",
    "FOOD101_URL",
    "GTZAN_DOWNLOAD_SIZE_MB",
    "GTZAN_URL",
    "HMDB51_DOWNLOAD_SIZE_MB",
    "HMDB51_URL",
    "IMAGE_DIR",
    "IMDB_DOWNLOAD_SIZE_MB",
    "IMDB_URL",
    "REUTERS21578_DOWNLOAD_SIZE_MB",
    "REUTERS21578_TOP_TOPICS",
    "REUTERS21578_URL",
    "KTH_ACTIONS",
    "KTH_BASE_URL",
    "KTH_DOWNLOAD_SIZE_MB",
    "OXFORD_FLOWERS_DOWNLOAD_SIZE_MB",
    "OXFORD_FLOWERS_LABELS_URL",
    "OXFORD_FLOWERS_URL",
    "PLACES365_DOWNLOAD_SIZE_MB",
    "PLACES365_LABELS_FILELIST_SIZE_MB",
    "PLACES365_LABELS_FILELIST_URL",
    "PLACES365_URL",
    "ProgressCallback",
    "SAMPLE_VIDEOS_URL",
    "SPEECH_COMMANDS_V2_DOWNLOAD_SIZE_MB",
    "SPEECH_COMMANDS_V2_URL",
    "STANFORD_DOGS_DOWNLOAD_SIZE_MB",
    "STANFORD_DOGS_URL",
    "UCF101_FULL_DOWNLOAD_SIZE_MB",
    "UCF101_FULL_URL",
    "UCF101_SUBSET_DOWNLOAD_SIZE_MB",
    "UCF101_SUBSET_URL",
    "UCSF_IDL_API_URL",
    "UCSF_IDL_DOWNLOAD_SIZE_MB",
    "UCSF_IDL_DOWNLOAD_URL",
    "URBANSOUND8K_DOWNLOAD_SIZE_MB",
    "URBANSOUND8K_URL",
    "VIDEO_DIR",
    "download_file_with_progress",
    # Audio
    "download_esc50",
    "download_gtzan",
    "download_speech_commands_v2",
    "download_urbansound8k",
    # Images
    "download_caltech101",
    "download_caltech256",
    "download_cifar10",
    "download_eurosat",
    "download_food101",
    "download_oxford_flowers",
    "download_places365",
    "download_stanford_dogs",
    # Video
    "download_hmdb51",
    "download_kth",
    "download_ucf101_full",
    "download_ucf101_subset",
    # Text
    "download_20newsgroups",
    "download_ag_news",
    "download_arxiv_abstracts",
    "download_bbc_news",
    "download_dbpedia",
    "download_imdb",
    "download_reuters21578",
    "_find_bbc_root",
    # Documents
    "download_ucsf_documents",
]
