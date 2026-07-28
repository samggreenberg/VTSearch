"""The pre-registered experiment grid for the Max-Patch study.

Kept in one place so prepare, the SLURM array indexer, and the report generator
all agree on exactly which cells exist.  Sizing knobs are env-overridable so the
grid can be trimmed if the cluster is busy without editing code.

Arms
----
Every arm is an ``(embedder, style)`` pair:

* ``dinov2_patch`` x {``max_hac``, ``max_patch``, ``whole_image``}
* ``dinov3_patch`` x {``max_hac``, ``max_patch``, ``whole_image``}
* ``siglip``       x {``whole_image``}

``max_hac`` is today's production patch pipeline (HAC region tree, snap-to-node
Good votes, leaf-flood Bad votes, region max-pool scoring).  ``max_patch`` is
the tree-free alternative under test (nearest-patch Good votes, all-patch Bad
flood, raw-patch max-pool scoring).  ``whole_image`` on the DINO embedders is a
CLS-only control that isolates "does *any* patch machinery help over the plain
global vector?"; on SigLIP it is the standard single-vector baseline.
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
PATCH_STYLES = os.environ.get("MAXPATCH_PATCH_STYLES", "max_hac,max_patch,whole_image").split(",")
SINGLE_STYLES = ["whole_image"]

# --- Sizing knobs (env-overridable) ---
N_CATEGORIES = int(os.environ.get("MAXPATCH_N_CATEGORIES", "6"))
SEEDS = list(range(int(os.environ.get("MAXPATCH_N_SEEDS", "4"))))
MAX_STEPS = int(os.environ.get("MAXPATCH_MAX_STEPS", "150"))

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


def select_categories(category_counts: dict[str, int], n: int = N_CATEGORIES) -> list[str]:
    """Pick *n* categories spanning common->rare, deterministically.

    Categories with fewer than ``_MIN_CATEGORY_COUNT`` positives are dropped
    (their held-out test sets would be too small to estimate FNR).  The rest are
    sorted by count and sampled at even rank intervals, so the chosen set spans
    the prevalence range present in the dataset rather than clustering at one end.
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
