"""The pre-registered grid for the Gaussian-process head pilot (issue #3954).

One place for what the stages, the cell enumerator and the summariser have to
agree on.  Sizing knobs are env-overridable so the grid can be trimmed on a
slow box without editing code; the report records the values that ran.
"""

from __future__ import annotations

import os

# --- Datasets (image, SigLIP).  ``caltech101_m`` is the slice the MLP-vs-SVM
# study ran on; ``_s`` is a seventh of the source and leaves most query
# categories under ten positives, too few to split into a pool and a test half.
DATASETS = os.environ.get("GPHEAD_DATASETS", "caltech101_m").split(",")
EMBEDDER = os.environ.get("GPHEAD_EMBEDDER", "siglip")
MEDIA_TYPE = "image"

# --- Sizing knobs ---
N_CATEGORIES = int(os.environ.get("GPHEAD_N_CATEGORIES", "6"))
SEEDS = list(range(int(os.environ.get("GPHEAD_N_SEEDS", "5"))))
MAX_STEPS = int(os.environ.get("GPHEAD_MAX_STEPS", "150"))
WORKERS = int(os.environ.get("GPHEAD_WORKERS", "4"))

# --- Stage A: the label-curve sweep (bare estimators, given N labels) ---
STAGE_A_TRAINERS = os.environ.get("GPHEAD_STAGE_A_TRAINERS", "svm_linear,gp_rbf,gp_dot,gp_rbf@fixed").split(",")
LABEL_COUNTS = [int(x) for x in os.environ.get("GPHEAD_LABEL_COUNTS", "8,16,32,64,128").split(",")]

# --- Stage B: the Autopilot voting simulation.  An arm is a (trainer,
# strategy, safe_thresholds, standalone_cut) tuple; the directory under RESULTS
# is the arm name.  ``app`` is the shipped detector: linear-SVM head, fused
# threshold, the rank-nearest-cut Hard pick.  ``app_xcal`` is the same head on
# the plain cross-calibration cut, which is the only threshold rule the
# standalone arms can share (their fold models are not the app's head, so the
# fold-anchored fusion has nothing to anchor; see ``_safe_threshold_for_step``)
# - the parity control that separates "the GP head" from "the threshold rule".
# ``gp_rbf`` takes that cut as a raw score, as every standalone arm did before
# this study; ``gp_rbf_rank`` carries it by rank (``standalone_cut="rank"``),
# the transfer production makes, because a GP's probability scale moves with
# every refit and the raw cut collapses on it; the ``*_blend`` arms take the
# app's own fold-less fallback (``safe_thresholds=True`` lands a standalone
# trainer on the schedule blend of the cut with a GMM fitted on the final
# model's haystack scores), the one shipped rule that is scale-consistent by
# construction (see the report).
ARMS: dict[str, tuple[str, str, bool, str]] = {
    "app": ("app", "autopilot", True, "raw"),
    "app_xcal": ("app", "autopilot", False, "raw"),
    "gp_rbf": ("gp_rbf", "autopilot", False, "raw"),
    "gp_rbf_rank": ("gp_rbf", "autopilot", False, "rank"),
    "gp_rbf_blend": ("gp_rbf", "autopilot", True, "raw"),
    "gp_dot_blend": ("gp_dot", "autopilot", True, "raw"),
    "gp_rbf_blend_straddle": ("gp_rbf", "autopilot_uncertainty", True, "raw"),
    "gp_rbf_blend_maxvar": ("gp_rbf", "autopilot_maxvar", True, "raw"),
}
STAGE_B_ARMS = os.environ.get("GPHEAD_ARMS", ",".join(ARMS)).split(",")

# --- Production-faithful fixed choices ---
INCLUSION = 0
SIM_FRACTION = 0.5
CALIBRATE_COUNT = 2
#: ``None`` = the app's per-embedder default (0.3 for a single-vector space).
CALIBRATION_FRACTION: float | None = None
#: Natural prevalence only: the ``_m`` slice has too few positives per
#: category to thin to 1% and keep a measurable test half.
PREVALENCE_ARMS: list[float | None] = [None]

# --- Minimum positives a category needs to be usable ---
MIN_CATEGORY_COUNT = int(os.environ.get("GPHEAD_MIN_CAT_COUNT", "16"))


def select_categories(category_counts: dict[str, int], n: int = N_CATEGORIES) -> list[str]:
    """Pick *n* query categories spanning common -> rare, deterministically.

    Only categories that have an eval query (``EVAL_DATASETS``) are candidates,
    because the Autopilot opening is the text sort for that query.  Those with
    fewer than :data:`MIN_CATEGORY_COUNT` positives are dropped; the rest are
    sorted by count and sampled at even rank intervals.
    """
    usable = sorted(
        ((c, k) for c, k in category_counts.items() if k >= MIN_CATEGORY_COUNT),
        key=lambda kv: (-kv[1], kv[0]),
    )
    if len(usable) <= n:
        return [c for c, _ in usable]
    idx = sorted({round(i * (len(usable) - 1) / (n - 1)) for i in range(n)})
    return [usable[i][0] for i in idx]


def cells(categories_by_dataset: dict[str, list[str]]) -> list[dict]:
    """Enumerate the ``(dataset, category, seed)`` cells, in a stable order."""
    out: list[dict] = []
    for ds in DATASETS:
        for cat in categories_by_dataset.get(ds, []):
            for seed in SEEDS:
                out.append({"dataset": ds, "category": cat, "seed": seed})
    return out


def cell_stem(cell: dict) -> str:
    return f"{cell['dataset']}__{cell['category'].replace(' ', '_')}__s{cell['seed']}"
