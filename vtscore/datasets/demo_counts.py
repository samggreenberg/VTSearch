"""Exact media counts for demo datasets, measured once and written down.

The pre-download ``# Media`` figure shown in the dataset picker used to be
*estimated* at request time by :func:`vtsearch.routes.datasets.ui._estimate_demo_num_files`,
which multiplies a single per-category average (``items_per_category``) by the
slice fraction.  That estimate is exact only when every category has the same
number of items.  For sources with **uneven** category sizes (Caltech-101's
``airplanes`` holds 800 images while most classes hold ~60-130) the average
diverges from the true per-category sum, so the advertised count came out
~37-40% low (e.g. Caltech-101 (S) advertised 300 but the loader embedded 412).

To make the advertised number accurate, the true totals are computed ahead of
time and recorded here, keyed by demo-dataset id.  The dataset picker prefers
this table and only falls back to the estimate for ids that have not yet been
measured.

Regenerating
------------
Counts are derived by the real loader's *collection* phase (download → list →
per-category slice), which is what determines how many media end up in the
dataset.  To (re)compute a dataset's count, download its source and run::

    python scripts/compute_demo_counts.py <dataset_id> [<dataset_id> ...]

The script prints ready-to-paste ``"id": count,`` lines; add them below.  Each
source's four S/M/L/A variants share one category-size profile, so measuring a
source once yields all four counts.

Measured entries
----------------
All entries below were produced by ``scripts/compute_demo_counts.py`` against
the downloaded source (the collection phase with embedding stubbed), except
Caltech-101 S/M which were confirmed against the cached ``.pkl`` files.  Each
source's S/M/L slices partition its categories, so ``S + M + L == A`` holds as
a consistency check.

* ``caltech101_*`` (image) — 412 / 838 / 1704 / 2954
* ``caltech256_*`` (image) — 534 / 1087 / 2179 / 3800
* ``eurosat_*`` (image) — 3853 / 7713 / 15434 / 27000
* ``oxford_flowers_102_a`` (image) — 8189
* ``places365_*`` (image) — 5110 / 10220 / 21170 / 36500
* ``ucsf_documents_a`` (image) — 150 (6 categories × 25 PDFs each)
* ``reuters21578_*`` (text) — 1361 / 2731 / 5463 / 9555
* ``20newsgroups_*`` (text) — 1194 / 2389 / 4775 / 8358
* ``arxiv_abstracts_*`` (text) — 28 / 57 / 115 / 200
* ``bbc_news_a`` (text) — 2225
* ``wikipedia_topics_*`` (dbpedia text) — 89992 / 179998 / 360010 / 630000
* ``gtzan_a`` (audio) — 1000 (one media per track; AppleDouble sidecars skipped)
* ``urbansound8k_*`` (audio) — 1240 / 2497 / 4995 / 8732
* ``speech_commands_v2_*`` (audio) — 15104 / 30238 / 60487 / 105829
* ``ucf101_*`` (video) — 55 / 114 / 236 / 405 (one media per video file)
* ``ucf101_full_*`` (video) — 1857 / 3808 / 7655 / 13320
* ``kth_*`` (video) — 84 / 168 / 347 / 599

Datasets whose categories are uniform (e.g. ESC-50 at 40 clips/category,
Food-101 at 1000/category, AG News / IMDB which are class-balanced) are already
exact under the estimate and do not need an entry here, though recording a
measured one does no harm (``bbc_news_a`` above happened to match its estimate).
"""

from __future__ import annotations

# Exact post-load media count per demo-dataset id.  Absent ids fall back to the
# per-category-average estimate.  Keep sorted by id for easy diffing.
DEMO_MEDIA_COUNTS: dict[str, int] = {
    "20newsgroups_a": 8358,
    "20newsgroups_l": 4775,
    "20newsgroups_m": 2389,
    "20newsgroups_s": 1194,
    "arxiv_abstracts_a": 200,
    "arxiv_abstracts_l": 115,
    "arxiv_abstracts_m": 57,
    "arxiv_abstracts_s": 28,
    "bbc_news_a": 2225,
    "caltech101_a": 2954,
    "caltech101_l": 1704,
    "caltech101_m": 838,
    "caltech101_s": 412,
    "caltech256_a": 3800,
    "caltech256_l": 2179,
    "caltech256_m": 1087,
    "caltech256_s": 534,
    "eurosat_a": 27000,
    "eurosat_l": 15434,
    "eurosat_m": 7713,
    "eurosat_s": 3853,
    "gtzan_a": 1000,
    "kth_a": 599,
    "kth_l": 347,
    "kth_m": 168,
    "kth_s": 84,
    "oxford_flowers_102_a": 8189,
    "places365_a": 36500,
    "places365_l": 21170,
    "places365_m": 10220,
    "places365_s": 5110,
    "reuters21578_a": 9555,
    "reuters21578_l": 5463,
    "reuters21578_m": 2731,
    "reuters21578_s": 1361,
    "roxford5k_a": 5063,
    "roxford5k_s": 501,
    "speech_commands_v2_a": 105829,
    "speech_commands_v2_l": 60487,
    "speech_commands_v2_m": 30238,
    "speech_commands_v2_s": 15104,
    "ucf101_a": 405,
    "ucf101_full_a": 13320,
    "ucf101_full_l": 7655,
    "ucf101_full_m": 3808,
    "ucf101_full_s": 1857,
    "ucf101_l": 236,
    "ucf101_m": 114,
    "ucf101_s": 55,
    "ucsf_documents_a": 150,
    "urbansound8k_a": 8732,
    "urbansound8k_l": 4995,
    "urbansound8k_m": 2497,
    "urbansound8k_s": 1240,
    "wikipedia_topics_a": 630000,
    "wikipedia_topics_l": 360010,
    "wikipedia_topics_m": 179998,
    "wikipedia_topics_s": 89992,
}


def exact_demo_count(dataset_id: str) -> int | None:
    """Return the measured exact media count for *dataset_id*, or ``None``.

    ``None`` means the dataset has not been measured yet and callers should
    fall back to the per-category-average estimate.
    """
    return DEMO_MEDIA_COUNTS.get(dataset_id)
