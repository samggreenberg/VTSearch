"""Pre-registered grid for the calibration study (issue #2781).

One place the prepare stage, the SLURM array indexer, and the analyzer all agree
on.  See ``docs/plans/calibration-experiment.md`` for the design.

Arms (each an ``(embedder, style)`` pair):

* ``visual_genome_m`` (boxed; ground-truth regions):
  ``siglip`` / ``siglip_l`` × ``whole_image`` (row-wise conformal), and
  ``dinov3_patch`` × {``max_patch``, ``max_patch_pca_hac``} (grouped bag
  calibration).

  **Only the ``dinov3_patch`` arms actually region-vote.** Region voting needs a
  stored ``patch_grid`` to pool the dragged box and ``patch_regions`` to
  max-pool at scoring time; the single-vector embedders have neither, so
  :data:`REGION_VOTING_BY_DATASET` degrades to whole-image training *and*
  whole-image scoring for them, and they blend under the **binary** schedule.
  This docstring previously called the whole set "region voting", which is how
  #2877 came to report a binary-voting environment as a region-voting one.  The raw-patch tree arm additionally re-pools its own per-node
  scores under ``topk`` and ``pnorm`` (remedial variants, emitted as extra rows).
* ``caltech101_m`` (binary voting; boxless): ``siglip`` / ``siglip_l`` ×
  ``whole_image`` only — the ordinary row-wise conformal path most users hit.

Category selection (scale-band on the boxed VG set, prevalence-spread on the
boxless Caltech set) is copied from the Max-Patch runner so the two studies
select the *same* categories and their pickles are interchangeable.
"""

from __future__ import annotations

import os
import zlib

# --- Datasets and their embedders (arms differ per dataset) ---
DATASETS = os.environ.get("CALIB_DATASETS", "visual_genome_m,caltech101_m").split(",")

DATASET_EMBEDDERS: dict[str, list[str]] = {
    "visual_genome_m": os.environ.get("CALIB_VG_EMBEDDERS", "siglip,siglip2_l,dinov3_patch").split(","),
    "caltech101_m": os.environ.get("CALIB_CALTECH_EMBEDDERS", "siglip,siglip2_l,dinov3_patch").split(","),
    # COCO-2017-val, assembled from the #2790 sweep cache by
    # ``build_coco_pickle.py`` (issue #2841).  Whole-image embedders only: that
    # cache holds each image's whole vector and its HAC region vectors but not
    # the raw patch grid, so no region-voting style can be built from it.
    "coco_val": os.environ.get("CALIB_COCO_EMBEDDERS", "siglip,siglip2_l,dinov3_patch").split(","),
    # Box-size-banded VG (PR #3123, still unmerged).  Only the embedder table is
    # needed: the pile already holds every cell, and prepare_data reads a cell
    # pickle in place when it exists.  Registering them here rather than basing
    # the run on that branch keeps the code at dev HEAD.
    "vg_box_small": os.environ.get("CALIB_VGBOX_EMBEDDERS", "siglip,siglip2_l,dinov3_patch").split(","),
    "vg_box_medium": os.environ.get("CALIB_VGBOX_EMBEDDERS", "siglip,siglip2_l,dinov3_patch").split(","),
    "vg_box_large": os.environ.get("CALIB_VGBOX_EMBEDDERS", "siglip,siglip2_l,dinov3_patch").split(","),
    # #3156's verified scale dataset with the box-size band collapsed away
    # (#3115): 12 hand-checked classes, 300 positives each against one shared
    # 3900-image negative pool, so **prevalence is identical in every cell**.
    # That is what a calibration study wants and what `visual_genome_m` is not -
    # its selected categories run 25 to 1645 positives, and the thin ones
    # produce cells with no trainable step at all.  Both embedders are in the
    # pile, so one dataset supplies BOTH voting modes and the mode contrast
    # stops being confounded with the dataset.
    "vg_scale_any": os.environ.get("CALIB_VGSCALE_EMBEDDERS", "siglip,dinov3_patch").split(","),
    # The same-class-across-bands set (#3156): one class list at three box
    # scales, so a small-vs-large difference is about size rather than about
    # which words happen to live at which size. Its categories are
    # band-suffixed (`bus@small`), and every one of them is the experiment --
    # see CATEGORY_MODE "all".
    "vg_scale": os.environ.get("CALIB_VGSCALE_EMBEDDERS", "siglip,dinov3_patch").split(","),
}

#: Region voting (drag the ground-truth box) only makes sense on a boxed dataset.
#: COCO *is* boxed, but its cached vectors cannot feed a patch style (see above),
#: so it runs as a second binary-voting dataset - which is exactly the axis
#: #2841 asks about separately from region voting.
#:
#: NOTE: this flag is necessary but **not sufficient**.  It is per-*dataset*,
#: while whether region voting actually happens is per-*embedder*: a boxed
#: dataset paired with a single-vector embedder silently runs as binary voting
#: (no ``patch_grid`` to pool, no ``patch_regions`` to max-pool).
#: ``simulate_voting_iterations`` now warns when that combination is requested.
REGION_VOTING_BY_DATASET: dict[str, bool] = {
    "visual_genome_m": True,
    "caltech101_m": False,
    "coco_val": False,
}

#: Whether a dataset carries ground-truth region boxes.  This is the *dataset*
#: half of region voting; the other half is the embedder (it must emit a patch
#: grid).  Kept separate from :data:`REGION_VOTING_BY_DATASET` - which older
#: analyzers read as a per-dataset label - so that marking a dataset boxed here
#: cannot silently change another study's control selection.
#:
#: ``coco_val`` is boxed and, since the pile gained ``coco_val__dinov3_patch``,
#: is now genuinely region-capable - the second region-voting environment the
#: #2905 confound needed.  The older map still says False because it predates
#: that cell.
BOXED_BY_DATASET: dict[str, bool] = {
    "visual_genome_m": True,
    "caltech101_m": False,
    "coco_val": True,
    "vg_box_small": True,
    "vg_box_medium": True,
    "vg_box_large": True,
    "vg_scale": True,
    "vg_scale_any": True,
}


def region_voting_for(dataset: str, embedder: str) -> bool:
    """Region voting needs **both** halves: boxes (dataset) and a patch grid (embedder).

    Asserting the premise per *cell* rather than per dataset is the control for
    the mis-specification behind #2877, #2897 and #2905, where a boxed dataset
    paired with a single-vector embedder was reported as a region-voting arm
    while it silently ran as binary voting.
    """
    return BOXED_BY_DATASET.get(dataset, False) and is_patch_embedder(embedder)


# --- Styles per embedder kind ---
PATCH_STYLES = os.environ.get("CALIB_PATCH_STYLES", "max_patch,max_patch_pca_hac").split(",")
SINGLE_STYLES = ["whole_image"]

#: The style whose per-node scores get re-pooled into the remedial arms.
REPOOL_STYLE = "max_patch_pca_hac"
REPOOL_VARIANTS = [v for v in os.environ.get("CALIB_REPOOL_VARIANTS", "topk,pnorm").split(",") if v]
REPOOL_TOPK = int(os.environ.get("CALIB_REPOOL_TOPK", "4"))

#: Inclusion values the fold orderings are re-thresholded at for the budget sweep.
INCLUSION_SWEEP_KS = [int(k) for k in os.environ.get("CALIB_SWEEP_KS", "-4,-2,-1,0,1,2,4").split(",")]

# --- Sizing knobs ---
SEEDS = list(range(int(os.environ.get("CALIB_N_SEEDS", "4"))))
MAX_STEPS = int(os.environ.get("CALIB_MAX_STEPS", "150"))
EXEMPLAR_CANDIDATES = int(os.environ.get("CALIB_EXEMPLAR_CANDIDATES", "8"))

# --- Production-faithful fixed choices (pre-registered) ---
INCLUSION = 0
SIM_FRACTION = 0.5
#: Number of cross-calibration folds.  Production is 2, which is why it was a
#: constant - but 2 folds make the fold-anchored ``qmean``/``qmedian`` combine
#: arms byte-identical, so the combine question cannot be asked without moving
#: it.  Changing this changes the *trajectory* (different splits, different
#: per-fold models), so a folds contrast is a run-level A/B, not a paired arm.
CALIBRATE_COUNT = int(os.environ.get("CALIB_CALIBRATE_COUNT", "2"))
CALIBRATION_FRACTION = 0.5
#: The #2781 study pre-registered safe_thresholds OFF (conformal path only);
#: the #2799 safe-threshold GMM study flips this on via CALIB_SAFE_THRESHOLDS=1.
SAFE_THRESHOLDS = os.environ.get("CALIB_SAFE_THRESHOLDS", "0") == "1"
MEDIA_TYPE = "image"

#: The #2852 anchored-mixture study (design + pre-registered decision rules:
#: ``docs/plans/population-anchored-calibration.md``) flips this on via
#: ``CALIB_ANCHORED=1``; every step then additionally emits the label-anchored,
#: fold-anchored ("cross-LabeledGMM"), and rank-transfer arm rows.  Requires
#: ``CALIB_SAFE_THRESHOLDS=1`` (the anchored arms ride the variant-row path).
ANCHORED = os.environ.get("CALIB_ANCHORED", "0") == "1"
#: Anchor-weight grid: each labelled score counts as this many haystack scores
#: in the anchored EM.  Log-spaced from "one label = one haystack point" to
#: "labels dominate the fit" - the fusion knob the sweep exists to place.
ANCHORED_WEIGHTS = [float(w) for w in os.environ.get("CALIB_ANCHORED_WEIGHTS", "1,3,10,30,100").split(",") if w]
#: Cut rules re-cutting each anchored fit: production midpoint, and the
#: rate-optimal crossing (well-founded on an anchored fit, where the
#: components *are* the classes - the #2836 identification term is gone).
ANCHORED_RULES = [r for r in os.environ.get("CALIB_ANCHORED_RULES", "mid,rate").split(",") if r]
#: Fold-anchored + rank-transfer arms cost one sim-set scoring pass per
#: calibration fold per step; disable to keep only the cheap final-model arms.
ANCHORED_FOLD_ARMS = os.environ.get("CALIB_ANCHORED_FOLD_ARMS", "1") == "1"
#: How the fold arms combine per-fold cuts in quantile space.
ANCHORED_FOLD_COMBINES = [c for c in os.environ.get("CALIB_ANCHORED_FOLD_COMBINES", "qmean,qmedian").split(",") if c]
#: Vote-count checkpoints the anchored analyzer windows on (the plan's deep
#: regime; each window is (previous checkpoint, checkpoint]).
ANCHORED_CHECKPOINTS = [
    int(c) for c in os.environ.get("CALIB_ANCHORED_CHECKPOINTS", "20,50,100,200,300").split(",") if c
]

#: Inclusion values the **fold-anchored cut rules** are swept over (issue
#: #2865), into the ``__cutincl.csv`` side frame.  Empty (the default) = off,
#: and every other study runs exactly as before.
#:
#: Not to be confused with :data:`INCLUSION_SWEEP_KS`, which sweeps the
#: *conformal* rule's ``alpha(k)`` budget.  This one asks a different question -
#: which cut rule should answer the Inclusion knob at all - so its rows are
#: scored at their own ``k``, not at :data:`INCLUSION`.
#:
#: The arms are :data:`ANCHORED_WEIGHTS` x :data:`ANCHORED_RULES` x
#: :data:`ANCHORED_FOLD_COMBINES`, so a run that wants the #2865 candidate set
#: sets ``CALIB_ANCHORED_RULES=mid,mid_tilt,rate,cross_tilt,q_tilt``.
CUT_INCLUSION_KS = [int(k) for k in os.environ.get("CALIB_CUT_INCL_KS", "").split(",") if k.strip()]

#: Step sizes the eval-only ``q_tilt`` rule expands over - its free parameter,
#: in combined-fold-quantile units per inclusion step.  Every other rule ignores
#: this.  Empty = the single placeholder default in
#: ``vtscore.training.thresholds.FOLD_ANCHOR_QTILT_STEP``, which is a
#: placeholder and not a measurement; a run that means to *place* the parameter
#: has to pass a grid here.
CUT_INCLUSION_QTILT_STEPS = [float(s) for s in os.environ.get("CALIB_CUT_INCL_QTILT_STEPS", "").split(",") if s.strip()]

#: Calibration fold counts to score **counterfactually** at every step (issue
#: #2897), on top of whatever :data:`CALIBRATE_COUNT` the run lives under.
#: Empty (the default) = off, and every other study runs exactly as before.
#:
#: This is nearly free per *K* relative to what it buys, and exact rather than
#: approximate, because the folds are nested: each is an independent stratified
#: draw off one ``RandomState(42)`` at a size that does not depend on the count,
#: so the K folds a live ``calibrate_count=K`` run would train are the first K of
#: the Kmax folds this run trains.  One run therefore measures every K's regret
#: *and* every K's wall clock, paired within the step - which is why the fold
#: count, alone among the knobs here, does not need one full run per value to be
#: screened.  It still needs the A/B runs to close: K also steers acquisition.
#:
#: Cost: ``max(FOLD_COUNTS) - CALIBRATE_COUNT`` extra fold fits per step, so the
#: grid's *maximum* sets the price, not its length.  Size it from a real cell.
FOLD_COUNTS = [int(k) for k in os.environ.get("CALIB_FOLD_COUNTS", "").split(",") if k.strip()]

#: Which head each step trains (``vtscore.eval.voting_iterations.HEADS``).
#: Unset (the default) hands ``head=None`` to the harness, which resolves it to
#: the head the live detector actually trains — ``linear_svm``.  That is the
#: only setting a study's headline numbers can be read off, because questions
#: like #2799's ("should safe_thresholds be forced on for every VTSearch
#: user?") are answerable only on the shipped head.  Set ``CALIB_HEAD=linear``
#: for the logistic head the SVM replaced (#2790/#2809), or ``CALIB_HEAD=mlp``
#: for the historical auto-sized-MLP arm (#2781).
HEAD = os.environ.get("CALIB_HEAD") or None

#: Which safe-threshold mix-in schedule the run *lives* under (issue #2841).
#: This steers the trajectory - the blended threshold feeds Autopilot's Hard
#: pick - so an A/B between schedules needs one full run per value here.
BLEND_SCHEDULE = os.environ.get("CALIB_BLEND_SCHEDULE") or None


#: Extra schedules to score *counterfactually* on this run's trajectory, one
#: metric row each (tagged ``schedule``).  Free relative to the simulation, but
#: blind to acquisition feedback - the screen, not the verdict.  ``"all"``
#: expands to the whole registry.
def _schedule_variants() -> list[str]:
    raw = os.environ.get("CALIB_SCHEDULE_VARIANTS", "").strip()
    if not raw:
        return []
    if raw == "all":
        from vtscore.training.blend_schedules import schedule_names  # noqa: PLC0415

        return schedule_names()
    return [s.strip() for s in raw.split(",") if s.strip()]


SCHEDULE_VARIANTS = _schedule_variants()


def _opt_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else None


def _opt_float(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else None


#: Acquisition-side cut, as an **offset** from :data:`INCLUSION`.  The threshold
#: does two unrelated jobs - it is the reported decision line *and* the rank
#: position Autopilot's ``hard`` pick samples around.  This knob moves only the
#: second; reporting and every metric stay at :data:`INCLUSION`, so the arms
#: remain comparable.
#:
#: Direction is the opposite of the intuition from the cost weights, because the
#: pick reads the threshold as a **rank position**: a *negative* offset raises
#: the cut, moves it *up* the ranking, and returns *more* positives.
#:
#: Unset = whatever ``vtscore.training.thresholds`` currently ships, so an
#: unconfigured run measures what users actually get.  Do not restate the value
#: here: PR #2876 shipped -3, and it is **-1** today.  A number written into
#: this comment goes stale silently and a study then mis-states its own
#: baseline.  ``0`` is the pre-#2876 control, one threshold doing both jobs.
ACQ_INCLUSION_OFFSET = _opt_int("CALIB_ACQ_INCLUSION_OFFSET")
if ACQ_INCLUSION_OFFSET is None:
    from vtscore.training.thresholds import ACQUISITION_INCLUSION_OFFSET

    ACQ_INCLUSION_OFFSET = ACQUISITION_INCLUSION_OFFSET

#: The ``rank_pin`` arm: place the acquisition cut at this quantile of the
#: simulation-set scores directly, rather than by naming an inclusion.  Requires
#: ``CALIB_ACQ_INCLUSION_OFFSET=0``; the two name the same cut.
ACQ_RANK_PERCENTILE = _opt_float("CALIB_ACQ_RANK_PERCENTILE")

#: Minimum positives a category must have **in the simulation half** to be kept.
#: A long-horizon run (#2841 follow-up: does pure x-cal ever overtake the blend?)
#: is bounded by positives, not pool size: once autopilot has exhausted them,
#: every further vote is a negative and the conformal positive-quantile stops
#: improving, so the tail of the curve would measure nothing.  0 disables the
#: filter, which is the behaviour of every run before the follow-up.
MIN_SIM_POSITIVES = int(os.environ.get("CALIB_MIN_SIM_POSITIVES", "0"))

# --- Category-selection parameters (copied from the Max-Patch runner) ---
_MIN_CATEGORY_COUNT = int(os.environ.get("CALIB_MIN_CAT_COUNT", "20"))
N_CATEGORIES = int(os.environ.get("CALIB_N_CATEGORIES", "6"))  # prevalence-spread count (Caltech)
N_PER_BAND = int(os.environ.get("CALIB_N_PER_BAND", "6"))  # scale-band count (VG)
MAX_VOTED_AREA = float(os.environ.get("CALIB_MAX_VOTED_AREA", "0.80"))

PATCH_AREA = 1 / 196  # one DINOv3 patch, ~0.51 % of the image
LEAF_AREA = 1 / 12  # smallest HAC leaf, ~8.3 %
SCALE_BANDS: list[tuple[str, float, float]] = [
    ("sub_patch", 0.0, PATCH_AREA),
    ("patch_to_leaf", PATCH_AREA, LEAF_AREA),
    ("leaf_to_4x", LEAF_AREA, 4 * LEAF_AREA),
    ("above_4x", 4 * LEAF_AREA, 1.01),
]


def is_patch_embedder(embedder: str) -> bool:
    """True for embedders that produce a patch grid + HAC tree."""
    return embedder.endswith("_patch")


def styles_for_embedder(embedder: str) -> list[str]:
    """The style arms an embedder participates in."""
    return PATCH_STYLES if is_patch_embedder(embedder) else SINGLE_STYLES


def styles_for(dataset: str, embedder: str) -> list[str]:
    """The style arms one ``(dataset, embedder)`` cell runs.

    A patch embedder only earns its patch styles where the dataset can supply
    box supervision.  On a **boxless** dataset a Good vote has no box to pool,
    so it falls back to the image-level vector, while every Bad vote floods the
    full-image row **plus ~197 raw patches** as negatives.  No patch row is ever
    positive, so the patch geometry teaches only "patch-like => negative", and
    max-pooling it at inference buys nothing while re-opening the asymmetry
    behind the boxless-``max_patch`` failure (perfect ranking, zero FPR,
    catastrophic FNR -- see :mod:`vtscore.eval.patch_styles`).

    If the user can only answer in booleans about whole images, then the Bad
    pile and the haystack should be whole images too.
    """
    if is_patch_embedder(embedder) and not BOXED_BY_DATASET.get(dataset, False):
        return SINGLE_STYLES
    return styles_for_embedder(embedder)


def embedders_for_dataset(dataset: str) -> list[str]:
    return DATASET_EMBEDDERS.get(dataset, [])


def pickle_name(dataset: str, embedder: str) -> str:
    return f"{dataset}__{embedder}.pkl"


def crops_basename(dataset: str, embedder: str) -> str:
    return f"{dataset}__{embedder}__crops"


def category_rng_seed(category: str) -> int:
    """Deterministic (process-stable) RNG seed for a category's exemplar draw."""
    return zlib.crc32(category.encode("utf-8")) & 0x7FFFFFFF


def select_categories_by_prevalence(category_counts: dict[str, int], n: int = N_CATEGORIES) -> list[str]:
    """Pick *n* categories spanning common->rare (boxless datasets)."""
    usable = sorted(
        ((c, n_) for c, n_ in category_counts.items() if n_ >= _MIN_CATEGORY_COUNT),
        key=lambda kv: kv[1],
        reverse=True,
    )
    if len(usable) <= n:
        return [c for c, _ in usable]
    idx = [round(i * (len(usable) - 1) / (n - 1)) for i in range(n)]
    return [usable[i][0] for i in sorted(set(idx))]


def select_categories_by_scale(
    medias: dict,
    category_counts: dict[str, int],
    n_per_band: int = N_PER_BAND,
) -> tuple[list[str], dict]:
    """Pick categories stratified by voted-box scale (boxed datasets)."""
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


#: Force a category-selection mode instead of inferring it from the medias.
#: ``"prevalence"`` is what the already-box-banded ``vg_box_*`` sets want: they
#: are a box-size axis by construction, so re-banding *within* one leaves most
#: bands empty (wave 2 of #3129 collapsed to 5/4/2 categories out of 40).
#: ``"scale"`` forces banding; unset infers as before.
CATEGORY_MODE = os.environ.get("CALIB_CATEGORY_MODE", "").strip().lower()


def select_categories(medias: dict, category_counts: dict[str, int]) -> tuple[list[str], dict]:
    """Scale-stratified when boxed, else prevalence-spread.

    ``CALIB_CATEGORY_MODE`` overrides the inference in either direction.
    """
    if CATEGORY_MODE == "all":
        # A designated dataset carries its design in its category list: the
        # cells were built to be run, prevalence is identical across them by
        # construction, and re-banding an already-banded set would discard the
        # experiment. Take every category the pickle holds.
        return sorted(category_counts), {
            "mode": "all",
            "reason": "dataset's categories are the experimental design (designated cells)",
        }
    if CATEGORY_MODE == "prevalence":
        return select_categories_by_prevalence(category_counts), {
            "mode": "prevalence",
            "reason": "forced by CALIB_CATEGORY_MODE (the dataset has no scale axis to stratify on)",
        }
    selected, report = select_categories_by_scale(medias, category_counts)
    if selected:
        report["mode"] = "scale_bands"
        return selected, report
    return select_categories_by_prevalence(category_counts), {
        "mode": "prevalence",
        "reason": "dataset carries no ground-truth region boxes; no scale axis to stratify",
    }


def array_cells(categories_by_dataset: dict[str, dict[str, list[str]]]) -> list[dict]:
    """Enumerate ``(dataset, embedder, category, seed)`` cells for the SLURM array.

    Each cell runs **all styles** for its embedder inside one task (they share
    the loaded pickle), so an embedder's arms are paired on identical data,
    splits, and exemplar.  Deterministic order -> a task index maps to a stable
    cell across submissions.
    """
    cells: list[dict] = []
    for ds in DATASETS:
        per_emb = categories_by_dataset.get(ds, {})
        for emb in embedders_for_dataset(ds):
            for cat in per_emb.get(emb, []):
                for seed in SEEDS:
                    cells.append({"dataset": ds, "embedder": emb, "category": cat, "seed": seed})
    return cells
