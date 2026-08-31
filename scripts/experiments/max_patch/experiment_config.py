"""The pre-registered experiment grid for the Max-Patch study.

Kept in one place so prepare, the SLURM array indexer, and the report generator
all agree on exactly which cells exist.  Sizing knobs are env-overridable so the
grid can be trimmed if the cluster is busy without editing code.

Arms
----
Every arm is an ``(embedder, style)`` pair:

* ``dinov2_patch`` x {``max_patch``, ``max_patch_hac``, ``max_patch_pca_hac``,
  ``whole_image``}
* ``dinov3_patch`` x {same}
* ``siglip``       x {``whole_image``}

``max_patch`` is the tree-free geometry the study picked and #2886 shipped
(nearest-patch Good votes, all-patch Bad flood, raw-patch max-pool scoring).
The ``max_patch_hac`` pair are the raw-patch-leaf tree hybrids.  ``whole_image``
on the DINO embedders is a CLS-only control that isolates "does *any* patch
machinery help over the plain global vector?"; on SigLIP it is the standard
single-vector baseline.

**The ``max_hac`` arm the study ran against is gone.**  It was a delegation to
the production HAC region tree, which #2886 deleted when it adopted MaxPatch, so
the arm cannot be re-run from this tree; its published numbers live in
``docs/experiments/2026-07-29-max-patch/REPORT.md`` and ``analyze.py`` still labels
``max_hac`` rows found in archived result CSVs.
"""

from __future__ import annotations

import os
import zlib

# --- Datasets (image demo ids) ---
# visual_genome_m and openlogo_a carry ground-truth region boxes (region votes
# are real); caltech101_m is the boxless centered-object control.
DATASETS = os.environ.get("MAXPATCH_DATASETS", "visual_genome_m,openlogo_a,caltech101_m").split(",")

# --- Embedders ---
EMBEDDERS = os.environ.get("MAXPATCH_EMBEDDERS", "dinov2_patch,dinov3_patch,siglip").split(",")

# --- Styles per embedder kind ---
PATCH_STYLES = os.environ.get("MAXPATCH_PATCH_STYLES", "max_patch,max_patch_hac,max_patch_pca_hac,whole_image").split(
    ","
)
SINGLE_STYLES = ["whole_image"]

# --- Sizing knobs (env-overridable) ---
N_CATEGORIES = int(os.environ.get("MAXPATCH_N_CATEGORIES", "6"))
SEEDS = list(range(int(os.environ.get("MAXPATCH_N_SEEDS", "4"))))
MAX_STEPS = int(os.environ.get("MAXPATCH_MAX_STEPS", "150"))

#: Whether the simulated user's opening query is embedded through the
#: embedder's ``description_wrappers`` ensemble ("Enrich Sort Descriptions") or
#: plainly.  Tracks the app's shipped default for ``enrich_descriptions``
#: (``vtsearch/settings_models.py``), which is ``False``; passed explicitly at
#: every ``embed_text_query`` call so the opening this harness measures is the
#: one a real user sees rather than whatever the library happens to default to
#: (#3341).  The sibling knob is ``calibration/experiment_config.SEED_ENRICH``.
SEED_ENRICH = os.environ.get("MAXPATCH_SEED_ENRICH", "0") == "1"

# --- Startup-sort exemplars ---
# Per category, this many candidate exemplars are cropped + embedded at prepare
# time; seed *s* uses candidate ``s % len(candidates)``, so every arm at a given
# (category, seed) starts from the *same* exemplar image.
EXEMPLAR_CANDIDATES = int(os.environ.get("MAXPATCH_EXEMPLAR_CANDIDATES", "8"))

# --- Production-faithful fixed choices (pre-registered) ---
INCLUSION = 0
SIM_FRACTION = 0.5
CALIBRATE_COUNT = 2
CALIBRATION_FRACTION = 0.5
SAFE_THRESHOLDS = False
REGION_VOTING = True
MEDIA_TYPE = "image"

# --- Minimum positives a category needs to be usable ---
_MIN_CATEGORY_COUNT = int(os.environ.get("MAXPATCH_MIN_CAT_COUNT", "20"))

# --- Scale bands (the axis the study's hypothesis is actually about) ---
# Reference scales as fractions of image area: one DINOv3 patch is 1/196, the
# smallest pooled candidate the HAC tree can propose (a leaf) is ~1/12.
PATCH_AREA = 1 / 196  # ~0.51 %
LEAF_AREA = 1 / 12  # ~8.3 %

#: Band edges over the **voted** (union) box area.  The pre-registered
#: hypothesis says MaxPatch should win below leaf scale and MaxHAC above it, so
#: the bands straddle ``LEAF_AREA`` with a wide band on each side.
SCALE_BANDS: list[tuple[str, float, float]] = [
    ("sub_patch", 0.0, PATCH_AREA),
    ("patch_to_leaf", PATCH_AREA, LEAF_AREA),
    ("leaf_to_4x", LEAF_AREA, 4 * LEAF_AREA),
    ("above_4x", 4 * LEAF_AREA, 1.01),
]

#: Categories per scale band.  Total grid size is
#: ``len(SCALE_BANDS) * N_PER_BAND`` categories per dataset, so raising this
#: multiplies SLURM cells - see the runner's README before bumping it.
N_PER_BAND = int(os.environ.get("MAXPATCH_N_PER_BAND", "6"))

#: Drop categories whose median voted box covers more than this fraction of the
#: image.  Above it a "region vote" is indistinguishable from an image-level
#: vote, which is the exact confound that made the boxless Caltech-101 arm
#: uninformative about large targets: we would be measuring "what happens when
#: the user ignores region voting", not "what happens when the target is large".
MAX_VOTED_AREA = float(os.environ.get("MAXPATCH_MAX_VOTED_AREA", "0.80"))


def is_patch_embedder(embedder: str) -> bool:
    """True for embedders that produce a patch grid + HAC tree."""
    return embedder.endswith("_patch")


def styles_for_embedder(embedder: str) -> list[str]:
    """The style arms an embedder participates in."""
    return PATCH_STYLES if is_patch_embedder(embedder) else SINGLE_STYLES


def pickle_name(dataset: str, embedder: str) -> str:
    """Basename of the per-(dataset, embedder) pickle prepare writes."""
    return f"{dataset}__{embedder}.pkl"


def crops_basename(dataset: str, embedder: str) -> str:
    """Basename (no extension) of the exemplar-crop artifacts prepare writes."""
    return f"{dataset}__{embedder}__crops"


def category_rng_seed(category: str) -> int:
    """Deterministic RNG seed for a category's exemplar-candidate draw.

    CRC32 rather than ``hash()`` so the draw is stable across processes
    (PYTHONHASHSEED) and repo checkouts.
    """
    return zlib.crc32(category.encode("utf-8")) & 0x7FFFFFFF


def select_categories_by_prevalence(category_counts: dict[str, int], n: int = N_CATEGORIES) -> list[str]:
    """Pick *n* categories spanning common->rare, deterministically.

    Categories with fewer than ``_MIN_CATEGORY_COUNT`` positives are dropped
    (their held-out test sets would be too small to estimate FNR).  The rest are
    sorted by count and sampled at even rank intervals, so the chosen set spans
    the prevalence range present in the dataset rather than clustering at one end.

    Used for **boxless** datasets, where no scale axis exists.  Boxed datasets
    take :func:`select_categories_by_scale` instead - see :func:`select_categories`.
    """
    usable = sorted(
        ((c, n_) for c, n_ in category_counts.items() if n_ >= _MIN_CATEGORY_COUNT),
        key=lambda kv: kv[1],
        reverse=True,
    )
    if len(usable) <= n:
        return [c for c, _ in usable]
    idx = [round(i * (len(usable) - 1) / (n - 1)) for i in range(n)]
    return [usable[i][0] for i in sorted(set(idx))]


def band_for_area(area: float) -> str | None:
    """Name the :data:`SCALE_BANDS` entry *area* falls in, or ``None`` if outside."""
    for name, lo, hi in SCALE_BANDS:
        if lo <= area < hi:
            return name
    return None


def select_categories_by_scale(
    medias: dict,
    category_counts: dict[str, int],
    n_per_band: int = N_PER_BAND,
) -> tuple[list[str], dict]:
    """Pick categories **stratified by voted-box scale**, deterministically.

    The study's hypothesis is about object *scale* (does the tree's smallest
    pooled candidate still match the object?), so the sample has to span scale
    on purpose.  Selecting by prevalence - the old behaviour - left scale
    coverage to chance, which is how the first run ended up with 7 of 12
    categories below leaf scale and only 5 above, mixed in sign, with the
    crossover the study exists to locate resting on those 5 points.

    Selection, per band:

    1. Drop categories with fewer than ``_MIN_CATEGORY_COUNT`` positives (their
       held-out test sets are too small to estimate FNR).
    2. Drop categories with no boxes at all, and those whose median voted box
       exceeds :data:`MAX_VOTED_AREA` - at that size a "region vote" *is* an
       image-level vote and the cell measures nothing about scale.
    3. Bucket the survivors by their **voted** (union) box area, never by
       per-instance area - see :func:`vtscore.eval.labels.voted_box_area`.
    4. Within each band keep the ``n_per_band`` categories with the lowest
       ``union_inflation``, i.e. those whose vote is typically one clean object
       rather than a union over scattered instances.  Ties break on category
       name so the pick is reproducible.

    Returns ``(selected, report)``.  *report* records every band's candidates,
    picks, and the categories dropped by the whole-image cap, so the run can
    log what it left out instead of silently truncating.
    """
    from vtscore.eval.labels import category_scale_stats  # noqa: PLC0415

    stats: dict[str, dict] = {}
    dropped_large: list[tuple[str, float]] = []
    for cat, count in category_counts.items():
        if count < _MIN_CATEGORY_COUNT:
            continue
        s = category_scale_stats(medias, cat)
        if s is None:
            continue
        if s["voted_area"] > MAX_VOTED_AREA:
            dropped_large.append((cat, s["voted_area"]))
            continue
        stats[cat] = s

    selected: list[str] = []
    report: dict = {
        "dropped_above_max_voted_area": sorted(dropped_large),
        "max_voted_area": MAX_VOTED_AREA,
        "bands": {},
    }
    for name, lo, hi in SCALE_BANDS:
        in_band = sorted(
            (c for c, s in stats.items() if lo <= s["voted_area"] < hi),
            key=lambda c: (stats[c]["union_inflation"], c),
        )
        picks = in_band[:n_per_band]
        selected.extend(picks)
        report["bands"][name] = {
            "range": [lo, hi],
            "target": n_per_band,
            "n_candidates": len(in_band),
            "under_populated": len(picks) < n_per_band,
            "selected": picks,
            "not_selected": in_band[n_per_band:],
            "scales": {c: stats[c] for c in picks},
        }
    return sorted(selected), report


def select_categories(medias: dict, category_counts: dict[str, int]) -> tuple[list[str], dict]:
    """Choose this dataset's categories: scale-stratified when boxed, else prevalence.

    A dataset with no ground-truth boxes has no scale axis to stratify - every
    Good vote on it is image-level regardless of how big the object is - so it
    falls back to the prevalence spread and its ``report`` says so.
    """
    selected, report = select_categories_by_scale(medias, category_counts)
    if selected:
        report["mode"] = "scale_bands"
        return selected, report
    return select_categories_by_prevalence(category_counts), {
        "mode": "prevalence",
        "reason": "dataset carries no ground-truth region boxes; no scale axis to stratify",
    }


def array_cells(categories_by_dataset: dict[str, dict[str, list[str]]]) -> list[dict]:
    """Enumerate the (dataset, embedder, category, seed) cells for the SLURM array.

    *categories_by_dataset* is keyed ``{dataset: {embedder: [category, ...]}}``
    (category counts can differ per embedder only if a load partially failed; in
    the normal case they agree).  Each cell runs **all styles** for its embedder
    inside one task (they share the loaded pickle), so MaxHAC and MaxPatch
    trajectories are paired on identical data, splits, and exemplars.
    Deterministic order so a task index maps to a stable cell across submissions.
    """
    cells: list[dict] = []
    for ds in DATASETS:
        per_emb = categories_by_dataset.get(ds, {})
        for emb in EMBEDDERS:
            for cat in per_emb.get(emb, []):
                for seed in SEEDS:
                    cells.append({"dataset": ds, "embedder": emb, "category": cat, "seed": seed})
    return cells
