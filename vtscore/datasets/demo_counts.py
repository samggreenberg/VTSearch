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

Verified entries
----------------
* ``caltech101_*`` — S/M confirmed against the cached ``.pkl`` files
  (412 / 838 media); L/A computed from the same real per-category folder sizes
  using the loader's exact slice formula (1704 / 2954).

Datasets whose categories are uniform (e.g. ESC-50 at 40 clips/category,
Food-101 at 1000/category) are already exact under the estimate and do not need
an entry here, though recording a measured one does no harm.
"""

from __future__ import annotations

# Exact post-load media count per demo-dataset id.  Absent ids fall back to the
# per-category-average estimate.  Keep sorted by id for easy diffing.
DEMO_MEDIA_COUNTS: dict[str, int] = {
    "caltech101_a": 2954,
    "caltech101_l": 1704,
    "caltech101_m": 838,
    "caltech101_s": 412,
}


def exact_demo_count(dataset_id: str) -> int | None:
    """Return the measured exact media count for *dataset_id*, or ``None``.

    ``None`` means the dataset has not been measured yet and callers should
    fall back to the per-category-average estimate.
    """
    return DEMO_MEDIA_COUNTS.get(dataset_id)
