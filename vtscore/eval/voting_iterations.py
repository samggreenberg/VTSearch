"""Evaluate learned-sort cost over simulated voting iterations.

For each combination of seed *s*, dataset *d*, and target category *c*:

1. Load the dataset and split medias into **D_sim** (simulation) and
   **D_test** (held-out) using *s* to control the random split.
2. Assign ground-truth labels based on *c*: medias whose ``"category"``
   matches *c* are positive (``good``), others are negative (``bad``).
3. Vote on D_sim one item at a time, choosing *which* item to vote on next by
   reproducing the app's **Autopilot** flow (order seeded by *s*).
4. At each step *t* (once at least one good **and** one bad vote exist),
   train a model on votes so far, find a threshold, score D_test, and record
   the inclusion-weighted cost (``fpr_weight * FPR + fnr_weight * FNR``).

Which item the simulated user votes on at each step is chosen by the
``autopilot`` vote-order strategy (see :mod:`vtscore.eval.al_strategies`): seed
from text sort (or a few random known-good examples), then the standard
Good / Bad / Hard / New phases.  This is the only strategy the eval runs — the
point is to measure how the tool itself would function, not to compare
acquisition heuristics.

The result is a :class:`pandas.DataFrame` with columns
``seed, dataset, category, strategy, t, n_good, n_bad, cost, fpr, fnr``.

``n_good``/``n_bad`` are the number of good/bad votes the model was trained
on for that row. The very first scored step has only one of each, so its
``cost``/``fpr``/``fnr`` are extremely noisy; these counts let downstream
analysis filter or weight rows by how many votes actually informed them
rather than treating a 1-vs-1 model as if it were as reliable as a 50-vs-50
one.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

from vtscore.embedding.media_vectors import media_embedding
from vtscore.eval.al_strategies import ALContext, select_next
from vtscore.eval.autopilot_flow import AutopilotFlow, app_has_detector
from vtscore.eval.labels import media_is_positive, region_box_for_category
from vtscore.eval.trainers import _cross_calibrated_threshold, _parse_trainer_spec
from vtscore.training.blend_schedules import BlendContext
from vtscore.training.mlp import LINEAR_HEAD, _auto_hidden_dim, train_model
from vtscore.training.thresholds import (
    calculate_cross_calibration_threshold,
    calculate_safe_threshold,
    classify_threshold_provenance,
    compute_fold_orderings,
    compute_grouped_fold_node_scores,
    threshold_from_fold_orderings,
)


@dataclass
class _StepModel:
    """A trained per-step ranker plus the metadata the eval loop records.

    ``predict`` maps an ``(N, D)`` numpy embedding matrix to per-row
    ``P(positive)`` scores in ``[0, 1]`` — the trainer-agnostic scoring contract
    (identical to :data:`vtscore.eval.trainers.PredictFn`).  ``torch_model`` is
    set only for the MLP path, where region-aware datasets need the raw module
    to max-pool over patch regions; it is ``None`` for the SVM path (which the
    experiment only ever runs on single-vector, region-free datasets).
    ``backend``/``device`` are recorded on every result row so the report can
    say which engine produced each number.
    """

    predict: Callable[[Any], "np.ndarray"]
    torch_model: Optional[Any]
    backend: str
    device: str


#: Torch head choices for the harness's per-step ranker.  ``"mlp"`` is the
#: historical harness candidate — a hidden layer auto-sized from the vote count
#: (:func:`~vtscore.training.mlp._auto_hidden_dim`).  ``"linear"`` is the head
#: the live detector actually trains since #2790/#2809
#: (:data:`~vtscore.training.mlp.LINEAR_HEAD`, a single ``Linear(d, 1)`` =
#: logistic regression), so a ``"linear"`` run measures the shipped detector.
#: The choice is threaded into the calibration folds too, exactly as production
#: threads one width through ``_train_and_score_xy``.
HEADS: tuple[str, ...] = ("mlp", "linear")


def _resolve_hidden_dim(head: str, n_votes: int) -> int:
    """Hidden width for *head* at *n_votes* votes (0 = the linear head)."""
    if head == "linear":
        return LINEAR_HEAD
    if head == "mlp":
        return _auto_hidden_dim(n_votes)
    raise ValueError(f"unknown head {head!r}; expected one of {HEADS}")


#: Identifying columns every emitted row (main or sweep) leads with.  ``phase``
#: and ``app_trained`` ride along so any downstream analysis - including the
#: calibration study's threshold rows - can filter to the steps at which the app
#: would actually have had a trained detector on screen.
_IDENT_COLUMNS: tuple[str, ...] = (
    "seed",
    "dataset",
    "category",
    "strategy",
    "trainer",
    "head",
    "style",
    "prevalence_arm",
    "realized_prevalence",
    "t",
    "n_good",
    "n_bad",
    "phase",
    "app_trained",
)

#: Canonical column order for the voting-iterations result frame.  Kept in one
#: place so :func:`run_voting_iterations_eval` and downstream tooling agree.
_VOTING_COLUMNS: tuple[str, ...] = (
    *_IDENT_COLUMNS,
    "cost",
    "fpr",
    "fnr",
    "auroc",
    "average_precision",
    "train_seconds",
    "xcal_seconds",
    "pool_score_seconds",
    "test_score_seconds",
    "backend",
    "device",
    "elapsed_seconds",
)

#: Column order for the calibration study's main per-step frame (issue #2781),
#: emitted only when ``emit_calibration_metrics``.  One row per ``pool_variant``;
#: under ``safe_thresholds`` additionally one row per safe-threshold GMM variant
#: (issue #2799), tagged in ``gmm_variant`` (``""`` on every other row), and one
#: row per counterfactual mix-in schedule (issue #2841), tagged in ``schedule``.
#: The two variant families are disjoint: a ``gmm_variant`` row varies the GMM
#: *cut* at the production schedule, a ``schedule`` row varies the *schedule* at
#: the production cut, and the base rows carry ``""`` in both.
_CALIBRATION_COLUMNS: tuple[str, ...] = (
    *_IDENT_COLUMNS,
    "pool_variant",
    "gmm_variant",
    "schedule",
    "threshold",
    "threshold_provenance",
    "degenerate",
    "threshold_percentile",
    "xcal_threshold",
    "gmm_cut",
    "blend_weight",
    "cost",
    "fpr",
    "fnr",
    "auroc",
    "average_precision",
    "oracle_threshold",
    "oracle_cost",
    "oracle_fpr",
    "oracle_fnr",
    "regret",
    "cal_oracle_threshold",
    "cal_oracle_cost",
    "rule_inefficiency",
    "calibration_shift",
    "n_pool_rows",
    "train_seconds",
    "xcal_seconds",
    "pool_score_seconds",
    "test_score_seconds",
    "backend",
    "device",
    "elapsed_seconds",
)

#: Column order for the inclusion-budget sweep side frame (long format, one row
#: per (step, inclusion k)); written to a separate CSV by the runner.
_INCLUSION_SWEEP_COLUMNS: tuple[str, ...] = (
    *_IDENT_COLUMNS,
    "inclusion_k",
    "alpha",
    "sweep_threshold",
    "sweep_fpr",
    "sweep_fnr",
    "excess_fnr",
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


#: Minimum positives an arm must retain after prevalence downsampling.  Below
#: this the held-out test set has too few positives for a stable FNR estimate,
#: so the arm is skipped rather than reported with a noisy denominator.
_MIN_PREVALENCE_POSITIVES = 15


def _inclusion_weights(inclusion: int) -> tuple[float, float]:
    """Return ``(fpr_weight, fnr_weight)`` for a given inclusion value."""
    if inclusion >= 0:
        return 1.0, 2.0**inclusion
    return 2.0 ** (-inclusion), 1.0


def _prevalence(clips_dict: dict[int, dict[str, Any]], target_category: str) -> float:
    """Fraction of *clips_dict* that is positive for *target_category*."""
    if not clips_dict:
        return 0.0
    n_pos = sum(1 for m in clips_dict.values() if media_is_positive(m, target_category))
    return n_pos / len(clips_dict)


def _downsample_to_prevalence(
    clips_dict: dict[int, dict[str, Any]],
    target_category: str,
    target_prevalence: float,
    rng: np.random.RandomState,
) -> Optional[dict[int, dict[str, Any]]]:
    """Return a copy of *clips_dict* with positives thinned to ~*target_prevalence*.

    All negatives are kept; positives are randomly downsampled (via *rng*, so the
    arm is deterministic in the eval seed) to the largest count ``k`` with
    ``k / (k + n_neg) <= target_prevalence``.  Returns ``None`` when that leaves
    fewer than :data:`_MIN_PREVALENCE_POSITIVES` positives (the arm is then
    skipped).  Multi-label datasets are handled through ``media_is_positive``.
    """
    import numpy as np  # noqa: PLC0415

    pos_ids = [cid for cid in clips_dict if media_is_positive(clips_dict[cid], target_category)]
    neg_ids = [cid for cid in clips_dict if not media_is_positive(clips_dict[cid], target_category)]
    n_neg = len(neg_ids)
    if n_neg == 0:
        return None
    keep_k = int(target_prevalence * n_neg / (1.0 - target_prevalence))
    keep_k = min(keep_k, len(pos_ids))
    if keep_k < _MIN_PREVALENCE_POSITIVES:
        return None
    chosen = rng.choice(np.array(pos_ids, dtype=np.int64), size=keep_k, replace=False)
    keep = set(int(c) for c in chosen) | set(neg_ids)
    return {cid: clips_dict[cid] for cid in clips_dict if cid in keep}


def _split_media_ids(
    clips_dict: dict[int, dict[str, Any]],
    sim_fraction: float,
    rng: np.random.RandomState,
) -> tuple[list[int], list[int]]:
    """Randomly partition media IDs into simulation and test sets."""
    all_ids = sorted(clips_dict.keys())
    shuffled = rng.permutation(all_ids).tolist()
    n_sim = max(1, int(len(shuffled) * sim_fraction))
    return shuffled[:n_sim], shuffled[n_sim:]


def _good_training_vec(
    media: dict[str, Any],
    target_category: str,
    region_voting: bool,
) -> np.ndarray:
    """Return the training vector for one Good vote on *media*.

    With *region_voting* the simulated user drags the ground-truth box around
    the object: when *media* carries a stored ``patch_grid`` and an annotated
    region for *target_category*, the box is pooled on-the-fly via
    :func:`vtscore.detectors.training.pool_box_from_media` (the same path the
    live region-vote flow uses).  Falls back to the whole-image embedding when
    region voting is off, the media has no patch grid (single-vector
    embedders), or no box is annotated for this category - exactly an
    image-level Good vote.
    """
    if region_voting:
        from vtscore.detectors.training import pool_box_from_media  # noqa: PLC0415

        pooled = pool_box_from_media(media, region_box_for_category(media, target_category))
        if pooled is not None:
            return pooled
    return media_embedding(media)


def _blend_safe_threshold(
    threshold: float,
    step: _StepModel,
    region_aware: bool,
    sim_clips: dict[int, dict[str, Any]] | None,
    X_all_clips: Any,
    ctx: "BlendContext",
    style_obj: Any = None,
    schedule: str | None = None,
) -> tuple[float, list[float]]:
    """Blend *threshold* with a GMM threshold over the simulation set's scores.

    Region-aware datasets (MLP + patch embedder only) score the sim set via
    region max-pool (matching the test-set scoring); single-vector datasets use
    the pre-computed whole-image matrix *X_all_clips* through the step's
    trainer-agnostic ``predict``.  Kept separate so the per-step loop stays flat.

    With an explicit detection *style_obj* (see :mod:`vtscore.eval.patch_styles`)
    the sim set is scored through that style - the same scorer the test set
    uses - so the GMM sees the distribution the threshold will actually cut.

    Returns ``(blended_threshold, sim_scores)`` - the scores the GMM was fitted
    on ride along so the #2799 safe-threshold variant rows can re-cut the same
    distribution without a second scoring pass.
    """
    import numpy as np  # noqa: PLC0415

    if style_obj is not None:
        assert sim_clips is not None and step.torch_model is not None
        all_scores = list(style_obj.score_media(step.torch_model, sim_clips).values())
    elif region_aware:
        from vtscore.detectors.training import score_media_with_model  # noqa: PLC0415

        assert sim_clips is not None and step.torch_model is not None
        all_scores = [r["score"] for r in score_media_with_model(step.torch_model, sim_clips)]
    else:
        all_scores = np.asarray(step.predict(np.asarray(X_all_clips))).ravel().tolist()
    return calculate_safe_threshold(threshold, all_scores, ctx, schedule=schedule), all_scores


def _evaluate_on_test(
    step: _StepModel,
    threshold: float,
    clips_dict: dict[int, dict[str, Any]],
    test_ids: list[int],
    target_category: str,
    inclusion: int,
    region_aware: bool = False,
    style_obj: Any = None,
) -> dict[str, float]:
    """Score *test_ids* with *step* and return the per-step metrics.

    Returns the operating-point metrics the user cares about — inclusion-weighted
    ``cost``, ``fpr``, ``fnr`` (all computed at *threshold*) — plus the
    threshold-independent ranking metrics ``auroc`` and ``average_precision``,
    which isolate "how good is the ranking" from "how good is the threshold".

    When *region_aware* the test media carry ``patch_regions`` (a patch
    embedder), so scoring max-pools the MLP over every region of each image -
    exactly the live detector's inference for patch datasets (an image scores
    by its best-matching region).  Otherwise each media is scored by its single
    whole-image vector through the step's trainer-agnostic ``predict``.
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.eval.label_curve import _auroc, _average_precision  # noqa: PLC0415

    nan = float("nan")
    if not test_ids:
        return {"cost": nan, "fpr": nan, "fnr": nan, "auroc": nan, "average_precision": nan}

    if style_obj is not None:
        # Explicit detection style (see vtscore.eval.patch_styles): the style
        # owns the whole image-scoring rule (whole-image / region max-pool /
        # raw-patch max-pool), replacing both branches below.
        assert step.torch_model is not None
        test_clips = {cid: clips_dict[cid] for cid in test_ids}
        score_map = style_obj.score_media(step.torch_model, test_clips)
        scores = [score_map[cid] for cid in test_ids]
    elif region_aware:
        from vtscore.detectors.training import score_media_with_model  # noqa: PLC0415

        assert step.torch_model is not None
        test_clips = {cid: clips_dict[cid] for cid in test_ids}
        score_map = {r["id"]: r["score"] for r in score_media_with_model(step.torch_model, test_clips)}
        scores = [score_map[cid] for cid in test_ids]
    else:
        embs = np.array([media_embedding(clips_dict[cid]) for cid in test_ids])
        scores = np.asarray(step.predict(embs)).ravel().tolist()

    true_labels = [1.0 if media_is_positive(clips_dict[cid], target_category) else 0.0 for cid in test_ids]

    total_pos = sum(1 for lbl in true_labels if lbl == 1.0)
    total_neg = len(true_labels) - total_pos

    fp = fn = 0
    for score, label in zip(scores, true_labels, strict=True):
        predicted = 1 if score >= threshold else 0
        if predicted == 1 and label == 0.0:
            fp += 1
        elif predicted == 0 and label == 1.0:
            fn += 1

    fpr = fp / total_neg if total_neg > 0 else 0.0
    fnr = fn / total_pos if total_pos > 0 else 0.0

    fpr_weight, fnr_weight = _inclusion_weights(inclusion)
    cost = fpr_weight * fpr + fnr_weight * fnr

    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(true_labels, dtype=np.float64)
    return {
        "cost": round(cost, 6),
        "fpr": round(fpr, 6),
        "fnr": round(fnr, 6),
        "auroc": round(_auroc(scores_arr, labels_arr), 6),
        "average_precision": round(_average_precision(scores_arr, labels_arr), 6),
    }


def _r(x: float) -> float:
    """Round to 6 dp when finite, else pass NaN/inf through unchanged."""
    import math  # noqa: PLC0415

    return round(x, 6) if math.isfinite(x) else x


def _operating_metrics(
    scores: "np.ndarray",
    labels: "np.ndarray",
    threshold: float,
    inclusion: int,
    cal_scores: "np.ndarray | None",
    cal_labels: "np.ndarray | None",
    *,
    pool_variant: str,
    provenance: str,
    n_pool_rows: float,
) -> dict[str, Any]:
    """Full per-step calibration metrics for one pooling (issue #2781).

    ``scores``/``labels`` are the held-out test scores+labels under *pool_variant*
    at the trained *threshold*.  Computes the trained cost, the oracle cost (best
    cut on the test scores) and the resulting **regret**, plus the
    calibration-set oracle that splits regret into *rule inefficiency*
    (trained-vs-best-use-of-calibration) and *calibration→test shift* (best cut
    on calibration vs. best cut on test).  ``cal_scores``/``cal_labels`` are the
    pooled calibration fold orderings under the same pooling; ``None`` skips the
    decomposition (leaves those columns NaN).
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.eval.calibration_metrics import (  # noqa: PLC0415
        inclusion_weights,
        is_degenerate,
        operating_cost,
        oracle_cut,
        threshold_percentile,
    )
    from vtscore.eval.label_curve import _auroc, _average_precision  # noqa: PLC0415

    wf, wn = inclusion_weights(inclusion)
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    cost, fpr, fnr = operating_cost(scores, labels, threshold, wf, wn)
    o_thr, o_cost, o_fpr, o_fnr = oracle_cut(scores, labels, wf, wn)
    regret = cost - o_cost

    nan = float("nan")
    if cal_scores is not None and np.asarray(cal_scores).size > 0:
        cal_scores = np.asarray(cal_scores, dtype=np.float64)
        cal_labels = np.asarray(cal_labels, dtype=np.float64)
        c_thr, _, _, _ = oracle_cut(cal_scores, cal_labels, wf, wn)
        cal_oracle_cost, _, _ = operating_cost(scores, labels, c_thr, wf, wn)
        rule_inefficiency = cost - cal_oracle_cost
        calibration_shift = cal_oracle_cost - o_cost
    else:
        c_thr = nan
        cal_oracle_cost = nan
        rule_inefficiency = nan
        calibration_shift = nan

    return {
        "pool_variant": pool_variant,
        # Safe-threshold study columns (issues #2799, #2841): defaults here; the
        # base row and the per-variant rows overwrite them where they apply.
        "gmm_variant": "",
        "schedule": "",
        "xcal_threshold": _r(float(threshold)),
        "gmm_cut": nan,
        "blend_weight": nan,
        "threshold": _r(float(threshold)),
        "threshold_provenance": provenance,
        "degenerate": 1 if is_degenerate(scores, threshold) else 0,
        "threshold_percentile": _r(threshold_percentile(scores, threshold)),
        "cost": _r(cost),
        "fpr": _r(fpr),
        "fnr": _r(fnr),
        "auroc": _r(float(_auroc(scores, labels))),
        "average_precision": _r(float(_average_precision(scores, labels))),
        "oracle_threshold": _r(float(o_thr)),
        "oracle_cost": _r(o_cost),
        "oracle_fpr": _r(o_fpr),
        "oracle_fnr": _r(o_fnr),
        "regret": _r(regret),
        "cal_oracle_threshold": _r(float(c_thr)),
        "cal_oracle_cost": _r(cal_oracle_cost),
        "rule_inefficiency": _r(rule_inefficiency),
        "calibration_shift": _r(calibration_shift),
        "n_pool_rows": _r(float(n_pool_rows)),
    }


#: Safe-threshold GMM variants (issue #2799): ``(name, fit_scores, cut, space)``.
#: ``fit_scores`` picks which sim-set score distribution the GMM is fitted on
#: ("pooled" = the style's inference max-pool, what production fits post-#2797;
#: "image" = the whole-image vector scores, the historical pre-#2797 geometry).
#: ``cut`` picks the rule ("mid" = midpoint-of-means, what production ships:
#: #2798 swapped it for the crossing, #2799 measured that as a net loss and
#: #2833 reverted it; "cross" = the equal-density crossing, kept as a variant
#: for the #2836 follow-up).  ``space`` fits the GMM on sigmoid scores ("sig")
#: or their logits ("logit", the #2798 open idea).  ``xcal_only`` is the
#: no-blend control: the raw conformal threshold at the same step.
#: ``pooled_mid`` must reproduce the production blend exactly.
_SAFE_GMM_VARIANTS: tuple[tuple[str, str, str, str], ...] = (
    ("xcal_only", "", "", ""),
    ("image_mid", "image", "mid", "sig"),
    ("image_cross", "image", "cross", "sig"),
    ("pooled_mid", "pooled", "mid", "sig"),
    ("pooled_cross", "pooled", "cross", "sig"),
    ("pooled_mid_logit", "pooled", "mid", "logit"),
    ("pooled_cross_logit", "pooled", "cross", "logit"),
)

#: Sigmoid scores are clipped into ``[eps, 1-eps]`` before the logit transform
#: so saturated scores stay finite.
_LOGIT_EPS = 1e-6


def _gmm_cuts(scores: list[float], space: str) -> dict[str, float]:
    """Both GMM cut rules from one fit of *scores*, in sigmoid score space.

    Mirrors :func:`vtscore.training.thresholds.calculate_gmm_threshold`'s
    semantics per fallback (0.5 below 2 scores, median on fit failure) so the
    ``("mid", "sig")`` cut blended on the pooled scores reproduces the
    production safe threshold bit-for-bit.  With ``space="logit"`` the GMM is
    fitted on the logit-transformed sample and the cut mapped back through the
    sigmoid; the median fallback stays in the original space (the sigmoid is
    monotone, so the two medians agree).
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.training.thresholds import fit_score_gmm, gmm_fit_array  # noqa: PLC0415

    if len(scores) < 2:
        return {"mid": 0.5, "cross": 0.5}
    arr = gmm_fit_array(scores)
    fit_arr = arr
    if space == "logit":
        p = np.clip(arr, _LOGIT_EPS, 1.0 - _LOGIT_EPS)
        fit_arr = np.log(p) - np.log1p(-p)
    fit = fit_score_gmm(fit_arr)
    if fit is None:
        med = float(np.median(arr))
        return {"mid": med, "cross": med}
    cuts = {"mid": fit.midpoint(), "cross": fit.crossing_or_midpoint()}
    if space == "logit":
        cuts = {name: float(1.0 / (1.0 + np.exp(-v))) for name, v in cuts.items()}
    return cuts


def _safe_gmm_variant_rows(
    details: dict[str, Any],
    base_scores: "np.ndarray",
    base_labels: "np.ndarray",
    sim_pooled_scores: list[float],
    sim_image_scores: list[float],
    inclusion: int,
    n_pool_rows: float,
) -> list[dict[str, Any]]:
    """One metric row per safe-threshold GMM variant (issue #2799).

    Every variant re-cuts the *same* per-step model: the two candidate sim-set
    score distributions are fitted once per (fit, space) pair, each fit yields
    both cut rules, and each cut is blended with the step's conformal threshold on
    the production label ramp.  All variants are evaluated against the same
    held-out test scores (*base_scores*, the inference max-pool), so the rows
    are paired within a step by construction.
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.training.thresholds import blend_gmm_threshold, safe_blend_weight  # noqa: PLC0415

    xcal = float(details["xcal_threshold"])
    n_votes = int(details["n_votes"])
    pre_blend_provenance = str(details.get("pre_blend_provenance", "conformal"))
    fold_orderings = details.get("fold_orderings") or []
    cal_scores = np.array([s for scores, _ in fold_orderings for s in scores]) if fold_orderings else None
    cal_labels = np.array([lb for _, labels_ in fold_orderings for lb in labels_]) if fold_orderings else None

    fit_scores = {"pooled": sim_pooled_scores, "image": sim_image_scores}
    cuts: dict[tuple[str, str], dict[str, float]] = {}
    for name, _fit, _cut, _space in _SAFE_GMM_VARIANTS:
        if _fit and (_fit, _space) not in cuts:
            cuts[(_fit, _space)] = _gmm_cuts(fit_scores[_fit], _space)

    ctx = BlendContext(
        n_labels=n_votes,
        n_good=int(details.get("n_good", 0)),
        n_bad=int(details.get("n_bad", n_votes)),
    )
    weight = safe_blend_weight(ctx)
    rows: list[dict[str, Any]] = []
    for name, _fit, _cut, _space in _SAFE_GMM_VARIANTS:
        if name == "xcal_only":
            threshold = xcal
            gmm_cut = float("nan")
            provenance = pre_blend_provenance
        else:
            gmm_cut = cuts[(_fit, _space)][_cut]
            threshold = blend_gmm_threshold(xcal, gmm_cut, ctx)
            provenance = "gmm_blend"
        row = _operating_metrics(
            base_scores,
            base_labels,
            threshold,
            inclusion,
            cal_scores,
            cal_labels,
            pool_variant="max",
            provenance=provenance,
            n_pool_rows=n_pool_rows,
        )
        row["gmm_variant"] = name
        row["schedule"] = ""
        row["xcal_threshold"] = _r(xcal)
        row["gmm_cut"] = _r(gmm_cut)
        row["blend_weight"] = _r(weight)
        rows.append(row)
    return rows


def _schedule_variant_rows(
    details: dict[str, Any],
    base_scores: "np.ndarray",
    base_labels: "np.ndarray",
    sim_pooled_scores: list[float],
    inclusion: int,
    n_pool_rows: float,
    schedules: list[str],
) -> list[dict[str, Any]]:
    """One metric row per mix-in **schedule** (issue #2841).

    The dual of :func:`_safe_gmm_variant_rows`, which holds the schedule fixed
    and varies the GMM *cut*: here the cut is production's (midpoint of the
    component means, fitted on the style's inference pool) and the *schedule*
    varies.  Every schedule re-combines the same two candidate cuts from the
    same per-step model against the same held-out test scores, so the rows are
    paired within a step by construction and differ only in the mix-in rule.

    This is the study's **screen**, not its verdict.  Holding the trajectory
    fixed is precisely what makes it cheap - one simulation scores every
    schedule - but the blended threshold also feeds acquisition (Autopilot's
    Hard phase picks the item nearest the decision boundary), so schedules that
    would have labelled *different items* cannot show that here.  The A/B runs
    exist to measure what this screen structurally cannot see.
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.training.thresholds import (  # noqa: PLC0415
        blend_gmm_threshold,
        fit_gmm_threshold,
        safe_blend_weight,
    )

    xcal = float(details["xcal_threshold"])
    n_votes = int(details["n_votes"])
    pre_blend_provenance = str(details.get("pre_blend_provenance", "conformal"))
    fold_orderings = details.get("fold_orderings") or []
    cal_scores = np.array([s for scores, _ in fold_orderings for s in scores]) if fold_orderings else None
    cal_labels = np.array([lb for _, labels_ in fold_orderings for lb in labels_]) if fold_orderings else None

    # Required, not defaulted: the ``rare``/``pos`` families ramp on the class
    # split, so a guessed split would silently mis-score two whole families
    # rather than fail.  The caller sets both keys alongside ``n_votes``.
    ctx = BlendContext(n_labels=n_votes, n_good=int(details["n_good"]), n_bad=int(details["n_bad"]))
    # One fit, re-combined under every schedule.  The corridor schedules need
    # the component means, so the fit object rides along with the cut.
    gmm_cut, gmm_fit = fit_gmm_threshold(sim_pooled_scores)

    rows: list[dict[str, Any]] = []
    for name in schedules:
        threshold = blend_gmm_threshold(xcal, gmm_cut, ctx, schedule=name, fit=gmm_fit)
        weight = safe_blend_weight(ctx, name)
        row = _operating_metrics(
            base_scores,
            base_labels,
            threshold,
            inclusion,
            cal_scores,
            cal_labels,
            pool_variant="max",
            provenance="gmm_blend" if weight < 1.0 else pre_blend_provenance,
            n_pool_rows=n_pool_rows,
        )
        row["gmm_variant"] = ""
        row["schedule"] = name
        row["xcal_threshold"] = _r(xcal)
        row["gmm_cut"] = _r(gmm_cut)
        row["blend_weight"] = _r(weight)
        rows.append(row)
    return rows


def _calibration_metric_rows(
    step: _StepModel,
    threshold: float,
    details: dict[str, Any],
    clips_dict: dict[int, dict[str, Any]],
    test_ids: list[int],
    target_category: str,
    inclusion: int,
    style_obj: Any,
    repool_variants: list[str],
    topk: int,
) -> tuple[list[dict[str, Any]], "np.ndarray", "np.ndarray"]:
    """Per-step metric rows for the base pooling plus each remedial re-pool.

    Scores the test set's per-node sigmoids once through *style_obj*, then pools
    them ``max`` (base) and — for the raw-patch tree arm, which carries
    ``fold_node_data`` — ``topk`` / ``pnorm``.  Each remedial variant recalibrates
    its own threshold by re-pooling the same fold models' held-out node scores,
    so every arm has a genuine *trained* cost and an *oracle* cost.  Returns one
    row dict per pooling, each tagged with ``pool_variant``.
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.eval import calibration_metrics as cm  # noqa: PLC0415
    from vtscore.eval.patch_styles import _forward_sigmoid_chunked  # noqa: PLC0415

    assert step.torch_model is not None  # the calibration study only runs the MLP style path
    model = step.torch_model
    provenance = details.get("provenance", "conformal")
    fold_orderings = details.get("fold_orderings") or []
    fold_node_data = details.get("fold_node_data")

    test_clips = {cid: clips_dict[cid] for cid in test_ids}
    ids, flat, seg = style_obj.node_scores(model, test_clips)
    labels = np.array([1.0 if media_is_positive(clips_dict[cid], target_category) else 0.0 for cid in ids])
    n_pool_rows = float(cm.segment_counts(seg, flat.shape[0]).mean()) if len(ids) else float("nan")

    rows: list[dict[str, Any]] = []

    # --- Base pooling (max): the arm's real operating point. ---
    base_scores = cm.segment_max_pool(flat, seg)
    base_cal_scores = np.array([s for scores, _ in fold_orderings for s in scores]) if fold_orderings else None
    base_cal_labels = np.array([lb for _, labels_ in fold_orderings for lb in labels_]) if fold_orderings else None
    base = _operating_metrics(
        base_scores,
        labels,
        threshold,
        inclusion,
        base_cal_scores,
        base_cal_labels,
        pool_variant="max",
        provenance=provenance,
        n_pool_rows=n_pool_rows,
    )
    if "xcal_threshold" in details:
        # Under safe_thresholds the base row's threshold is the blended one;
        # record the pre-blend conformal cut alongside it (issue #2799).
        base["xcal_threshold"] = _r(float(details["xcal_threshold"]))
    rows.append(base)

    # --- Remedial re-pools: only where the same fold models exposed node data
    # (the raw-patch tree arm) and the base threshold was a real conformal cut. ---
    if fold_node_data and repool_variants:
        # Final-model node scores over the bad-voted bags -> the pnorm test null.
        neg_rows = details.get("neg_score_rows") or []
        if neg_rows:
            null_concat = np.concatenate([np.asarray(r, dtype=np.float32) for r in neg_rows], axis=0)
            test_null = np.sort(np.asarray(_forward_sigmoid_chunked(model, null_concat), dtype=np.float64))
        else:
            test_null = np.empty(0, dtype=np.float64)

        for variant in repool_variants:
            # Recalibrate the threshold: re-pool each fold's held-out calibration
            # groups under this variant, then run the conformal rule on the pool.
            v_orderings: list[tuple[list[float], list[float]]] = []
            for blocks, blk_labels in fold_node_data:
                if variant == "pnorm":
                    fold_null = cm.negative_block_null(blocks, blk_labels)
                    pooled = cm.pool_blocks(blocks, "pnorm", null_sorted=fold_null)
                else:
                    pooled = cm.pool_blocks(blocks, variant, topk=topk)
                v_orderings.append((pooled, list(blk_labels)))
            v_threshold = threshold_from_fold_orderings(v_orderings, inclusion)

            # Re-pool the test node scores under this variant.
            if variant == "pnorm":
                v_scores = cm.segment_pnorm_pool(flat, seg, test_null)
            else:
                v_scores = cm.segment_topk_mean_pool(flat, seg, topk)
            v_cal_scores = np.array([s for scores, _ in v_orderings for s in scores])
            v_cal_labels = np.array([lb for _, labels_ in v_orderings for lb in labels_])
            rows.append(
                _operating_metrics(
                    v_scores,
                    labels,
                    v_threshold,
                    inclusion,
                    v_cal_scores,
                    v_cal_labels,
                    pool_variant=variant,
                    provenance="conformal",
                    n_pool_rows=n_pool_rows,
                )
            )

    return rows, base_scores, labels


def _inclusion_sweep_rows(
    details: dict[str, Any],
    base_scores: "np.ndarray",
    base_labels: "np.ndarray",
    inclusion_sweep_ks: list[int],
) -> list[dict[str, Any]]:
    """Re-threshold the base fold orderings at each inclusion *k* and measure test FNR.

    Near-free (no refits): pools the cached fold orderings once and applies the
    conformal rule at each ``k``, then measures the realised test FPR/FNR at that
    cut.  Checks the Inclusion budget ``alpha(k) = 0.25 * 2^-k`` against the
    measured FNR under the **grouped** calibration path (issue #2781 / the
    grouped-arm follow-up in inclusion-calibration-bias.md).  Returns ``[]`` when
    the base threshold was a fallback (no real orderings to sweep).
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.eval.calibration_metrics import inclusion_weights, operating_cost  # noqa: PLC0415

    fold_orderings = details.get("fold_orderings") or []
    if not fold_orderings:
        return []
    base_scores = np.asarray(base_scores, dtype=np.float64)
    base_labels = np.asarray(base_labels, dtype=np.float64)
    out: list[dict[str, Any]] = []
    for k in inclusion_sweep_ks:
        thr_k = threshold_from_fold_orderings(fold_orderings, k)
        wf, wn = inclusion_weights(k)
        cost_k, fpr_k, fnr_k = operating_cost(base_scores, base_labels, thr_k, wf, wn)
        alpha_k = 0.25 * 2.0 ** (-k)
        out.append(
            {
                "inclusion_k": k,
                "alpha": _r(alpha_k),
                "sweep_threshold": _r(float(thr_k)),
                "sweep_fpr": _r(fpr_k),
                "sweep_fnr": _r(fnr_k),
                "excess_fnr": _r(fnr_k - alpha_k),
            }
        )
    return out


# ------------------------------------------------------------------
# Active-learning acquisition helpers
# ------------------------------------------------------------------


def _score_pool(
    step: _StepModel,
    pool_ids: list[int],
    clips_dict: dict[int, dict[str, Any]],
) -> dict[int, float]:
    """Return ``{pool_id: score}`` for the current model over the pool.

    Scores each pool item by its single whole-image vector - the fast path the
    acquisition strategies rank uncertainty over.  This intentionally uses the
    whole-image embedding even for patch datasets (where *test* scoring
    max-pools over regions): acquisition only needs a monotone uncertainty
    signal to order the pool, and the whole-image score is a cheap, adequate
    proxy for that ordering.
    """
    import numpy as np  # noqa: PLC0415

    if not pool_ids:
        return {}
    embs = np.array([media_embedding(clips_dict[cid]) for cid in pool_ids])
    scores = np.asarray(step.predict(embs)).ravel().tolist()
    return dict(zip(pool_ids, scores, strict=True))


def _labelset_error_cost(
    model_steps: list[tuple[Any, float]],
    good_votes: dict[int, None],
    bad_votes: dict[int, None],
    clips_dict: dict[int, dict[str, Any]],
    inclusion: int,
) -> Optional[float]:
    """Weighted FPR/FNR of the most recent model on the **labelled** set.

    Feeds the Smart indicator.  Mirrors ``labeling_progress._score_step``: each
    cached step's model is scored against the current labelset — the only
    ground truth the app has — so the trend the simulated user reacts to is the
    one a real user's status panel would show.  Deliberately *not* the held-out
    test split: those labels must never reach the vote order.
    """
    if not model_steps:
        return None
    step, threshold = model_steps[-1]
    fpr_weight, fnr_weight = _inclusion_weights(inclusion)

    ids = list(good_votes) + list(bad_votes)
    labels = [1.0] * len(good_votes) + [0.0] * len(bad_votes)
    total_pos = len(good_votes)
    total_neg = len(bad_votes)
    if not ids or total_pos == 0 or total_neg == 0:
        return None

    scores = _score_pool(step, ids, clips_dict)
    fp = fn = 0
    for cid, true_label in zip(ids, labels, strict=True):
        predicted = 1 if scores.get(cid, 0.0) >= threshold else 0
        if predicted == 1 and true_label == 0.0:
            fp += 1
        elif predicted == 0 and true_label == 1.0:
            fn += 1
    fpr = fp / total_neg if total_neg > 0 else 0.0
    fnr = fn / total_pos if total_pos > 0 else 0.0
    return fpr_weight * fpr + fnr_weight * fnr


def _build_eval_atlas(embeddings: dict[int, np.ndarray], min_node_size: int) -> Any:
    """Build a coverage atlas over *embeddings* for the autopilot New phase.

    Returns ``None`` when there are no vectors.  Uses the same hierarchical
    k-means partition the live dataset builds (see
    :class:`~vtscore.state.coverage_atlas.CoverageAtlas`); *min_node_size* is
    exposed so a caller with a small simulation set can drive the partition
    deeper than the production floor (20) and actually resolve density cells.
    """
    from vtscore.state.coverage_atlas import CoverageAtlas, auto_max_depth  # noqa: PLC0415

    if not embeddings:
        return None
    return CoverageAtlas(
        embeddings,
        k=3,
        max_depth=auto_max_depth(len(embeddings), k=3, min_node_size=min_node_size),
        min_node_size=min_node_size,
    )


def _train_and_calibrate(
    trainer: str,
    good_votes: dict[int, None],
    bad_votes: dict[int, None],
    clips_dict: dict[int, dict[str, Any]],
    target_category: str,
    *,
    region_voting: bool,
    input_dim: int,
    inclusion: int,
    calibrate_count: int,
    calibration_fraction: float,
    head: str = "mlp",
    style_obj: Any = None,
    emit_calibration_metrics: bool = False,
) -> tuple[_StepModel, float, int, dict[str, float], dict[str, Any]]:
    """Train the step's ranker and calibrate its threshold from the current votes.

    *head* selects the torch head on both torch paths (see :data:`HEADS`):
    ``"mlp"`` keeps the auto-sized hidden layer, ``"linear"`` trains the
    production linear head.  It is ignored by the SVM path, which has no torch
    head at all.

    Dispatches on *trainer*: ``"mlp"`` runs the production MLP path unchanged
    (see :func:`_mlp_train_and_calibrate`); any ``svm_*`` name runs the SVM path
    (see :func:`_svm_train_and_calibrate`).  Returns ``(step, threshold,
    n_labels, timings, details)`` where *timings* has ``train_seconds`` and
    ``xcal_seconds`` for the fit and threshold-calibration wall clocks, and
    *details* is empty unless *emit_calibration_metrics* (the #2781 study),
    carrying the fold orderings, node scores, and threshold provenance the
    calibration metrics need.

    With an explicit *style_obj* (MLP only) the vote-to-vector assembly is
    delegated to the style (see :func:`_style_train_and_calibrate`).
    """
    if style_obj is not None:
        return _style_train_and_calibrate(
            style_obj,
            good_votes,
            bad_votes,
            clips_dict,
            target_category,
            region_voting=region_voting,
            input_dim=input_dim,
            inclusion=inclusion,
            calibrate_count=calibrate_count,
            calibration_fraction=calibration_fraction,
            head=head,
            emit_calibration_metrics=emit_calibration_metrics,
        )
    if trainer == "mlp":
        return _mlp_train_and_calibrate(
            good_votes,
            bad_votes,
            clips_dict,
            target_category,
            region_voting=region_voting,
            input_dim=input_dim,
            inclusion=inclusion,
            calibrate_count=calibrate_count,
            calibration_fraction=calibration_fraction,
            head=head,
        )
    return _svm_train_and_calibrate(
        trainer,
        good_votes,
        bad_votes,
        clips_dict,
        target_category,
        inclusion=inclusion,
        calibrate_count=calibrate_count,
        calibration_fraction=calibration_fraction,
    )


def _mlp_train_and_calibrate(
    good_votes: dict[int, None],
    bad_votes: dict[int, None],
    clips_dict: dict[int, dict[str, Any]],
    target_category: str,
    *,
    region_voting: bool,
    input_dim: int,
    inclusion: int,
    calibrate_count: int,
    calibration_fraction: float,
    head: str = "mlp",
) -> tuple[_StepModel, float, int, dict[str, float], dict[str, Any]]:
    """The torch arm — numerically identical to the pre-trainer harness at ``head="mlp"``.

    At ``head="mlp"`` (the default) this is the harness's small-MLP candidate,
    not the live detector's head: production trains the linear (logistic) head
    instead (the #2790 finding, see ``vtscore.training.mlp.LINEAR_HEAD``).
    ``head="linear"`` selects that production head, so the reported cost is the
    shipped detector's.  Everything *around* the head mirrors the production
    ``_train_and_score_xy`` / ``train_and_threshold`` pipeline either way:

    Good votes region-pool their ground-truth box when *region_voting* is on
    (and the media supports it); Bad votes always train on the whole-image
    vector - matching the live detector, where only Yes-votes carry a region.

    * ``hidden_dim`` comes from the head (sized from the *full* label count on
      the MLP head, 0 on the linear one) and is forced onto the
      calibration folds, so the fold models share the final model's architecture
      (production likewise threads one width into
      ``cross_calibration_threshold_cached``).  Letting each fold auto-size to
      its own smaller train split would train narrower fold nets and report a
      threshold no single-architecture pipeline ever produces.
    * the fold splits use a fresh ``RandomState(42)`` - the fixed seed
      ``cross_calibration_threshold_cached`` always calibrates with - rather than
      the shared per-seed simulation RNG, so the calibration is byte-for-byte
      what production runs for this vote set.  The eval seed still varies the
      data (which media are voted, in what order, and the held-out test split);
      only the calibration folds are pinned, as they are in production.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    for vid in good_votes:
        X_list.append(_good_training_vec(clips_dict[vid], target_category, region_voting))
        y_list.append(1.0)
    for vid in bad_votes:
        X_list.append(media_embedding(clips_dict[vid]))
        y_list.append(0.0)

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    n_labels = len(good_votes) + len(bad_votes)

    hidden_dim = _resolve_hidden_dim(head, n_labels)
    t_xcal = time.monotonic()
    threshold = calculate_cross_calibration_threshold(
        X_list,
        y_list,
        input_dim,
        inclusion,
        rng=np.random.RandomState(42),
        calibrate_count=calibrate_count,
        calibration_fraction=calibration_fraction,
        hidden_dim=hidden_dim,
    )
    xcal_seconds = time.monotonic() - t_xcal
    t_train = time.monotonic()
    model = train_model(X, y, input_dim, hidden_dim=hidden_dim)
    train_seconds = time.monotonic() - t_train

    device = str(next(model.parameters()).device)

    def predict(X_test: Any) -> np.ndarray:
        with torch.no_grad():
            t = torch.tensor(np.asarray(X_test), dtype=torch.float32).to(next(model.parameters()).device)
            return torch.sigmoid(model(t)).squeeze(1).cpu().numpy()

    step = _StepModel(
        predict=predict,
        torch_model=model,
        backend="torch-cuda" if device.startswith("cuda") else "torch-cpu",
        device=device,
    )
    return step, threshold, n_labels, {"train_seconds": train_seconds, "xcal_seconds": xcal_seconds}, {}


def _style_train_and_calibrate(
    style_obj: Any,
    good_votes: dict[int, None],
    bad_votes: dict[int, None],
    clips_dict: dict[int, dict[str, Any]],
    target_category: str,
    *,
    region_voting: bool,
    input_dim: int,
    inclusion: int,
    calibrate_count: int,
    calibration_fraction: float,
    head: str = "mlp",
    emit_calibration_metrics: bool = False,
) -> tuple[_StepModel, float, int, dict[str, float], dict[str, Any]]:
    """Style-driven torch path (the Max-Patch experiment arms).

    The detection style (see :mod:`vtscore.eval.patch_styles`) supplies the
    vote-to-vector rules: each Good vote contributes ``style.good_vec`` (given
    the ground-truth box when *region_voting* and the media has one), each Bad
    vote floods ``style.bad_vecs`` - one row on a whole-image style, the CLS +
    HAC leaves on ``max_hac``, every raw patch on ``max_patch``.

    Training and calibration are **bag-aware**, exactly like the production
    vote path (:func:`vtscore.detectors.training._train_and_score_xy`): the
    head (see :data:`HEADS`) and the safe-threshold ramp size on distinct *votes* rather
    than flooded rows, the calibration folds split by bag, and the final fit
    weights each bag equally.  On a whole-image style every bag is one row, so
    this collapses to the historical single-vector behaviour.

    Calibration additionally runs in **inference geometry**: each bag is handed
    its ``style.score_rows`` stack so a Good bag collapses the same way a Bad
    bag (and every held-out image) does.  Without this a Good bag is a max over
    its 1 training row while a Bad bag is a max over the ~197 rows it flooded,
    and the calibrated cut lands above the score range production actually
    produces - see :func:`vtscore.training.thresholds.compute_fold_orderings`.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from vtscore.detectors.training import _flood_context  # noqa: PLC0415

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    groups: list = []
    score_rows_by_group: dict = {}
    for vid in good_votes:
        box = region_box_for_category(clips_dict[vid], target_category) if region_voting else None
        X_list.append(np.asarray(style_obj.good_vec(clips_dict[vid], box), dtype=np.float32))
        y_list.append(1.0)
        groups.append(("g", vid))
        score_rows_by_group[("g", vid)] = style_obj.score_rows(clips_dict[vid])
    for vid in bad_votes:
        for vec in style_obj.bad_vecs(clips_dict[vid]):
            X_list.append(np.asarray(vec, dtype=np.float32))
            y_list.append(0.0)
            groups.append(("b", vid))
        score_rows_by_group[("b", vid)] = style_obj.score_rows(clips_dict[vid])

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    n_votes, cal_groups, sample_weights = _flood_context(X_list, y_list, groups)

    hidden_dim = _resolve_hidden_dim(head, n_votes)
    t_xcal = time.monotonic()
    details: dict[str, Any] = {}
    if emit_calibration_metrics:
        threshold, details = _calibrate_with_details(
            X_list,
            y_list,
            input_dim,
            inclusion,
            calibrate_count=calibrate_count,
            calibration_fraction=calibration_fraction,
            hidden_dim=hidden_dim,
            cal_groups=cal_groups,
            score_rows_by_group=score_rows_by_group if cal_groups is not None else None,
        )
        # Bad-voted bags' inference row stacks: the final model scores these to
        # form the pnorm null (F_neg) at test time (see _calibration_metric_rows).
        details["neg_score_rows"] = [score_rows_by_group[("b", vid)] for vid in bad_votes]
    else:
        threshold = calculate_cross_calibration_threshold(
            X_list,
            y_list,
            input_dim,
            inclusion,
            rng=np.random.RandomState(42),
            calibrate_count=calibrate_count,
            calibration_fraction=calibration_fraction,
            hidden_dim=hidden_dim,
            groups=cal_groups,
            score_rows_by_group=score_rows_by_group if cal_groups is not None else None,
        )
    xcal_seconds = time.monotonic() - t_xcal
    t_train = time.monotonic()
    if sample_weights is not None:
        model = train_model(X, y, input_dim, hidden_dim=hidden_dim, sample_weights=sample_weights)
    else:
        model = train_model(X, y, input_dim, hidden_dim=hidden_dim)
    train_seconds = time.monotonic() - t_train

    device = str(next(model.parameters()).device)

    def predict(X_test: Any) -> np.ndarray:
        with torch.no_grad():
            t = torch.tensor(np.asarray(X_test), dtype=torch.float32).to(next(model.parameters()).device)
            return torch.sigmoid(model(t)).squeeze(1).cpu().numpy()

    step = _StepModel(
        predict=predict,
        torch_model=model,
        backend="torch-cuda" if device.startswith("cuda") else "torch-cpu",
        device=device,
    )
    return step, threshold, n_votes, {"train_seconds": train_seconds, "xcal_seconds": xcal_seconds}, details


def _calibrate_with_details(
    X_list: list[np.ndarray],
    y_list: list[float],
    input_dim: int,
    inclusion: int,
    *,
    calibrate_count: int,
    calibration_fraction: float,
    hidden_dim: int | None,
    cal_groups: list | None,
    score_rows_by_group: dict | None,
) -> tuple[float, dict[str, Any]]:
    """Compute the trained threshold **and** the calibration study's provenance.

    Replaces the plain :func:`calculate_cross_calibration_threshold` call on the
    style path when the #2781 metrics are requested.  Trains the calibration
    folds exactly once and returns ``(threshold, details)`` where *details* holds:

    * ``provenance`` — which code path set the threshold (``conformal`` /
      ``no_good_sentinel`` / ``too_few_default``), via
      :func:`~vtscore.training.thresholds.classify_threshold_provenance`.
    * ``fold_orderings`` — the pooled ``(scores, labels)`` per fold under the
      base (max) pooling, for the calibration-set oracle and the inclusion sweep.
    * ``fold_node_data`` — per-fold, per-group **node** scores (grouped path
      only), so a remedial pooling variant can recalibrate off the same fold
      models without retraining; ``None`` on the row-wise (whole-image) path.

    On the grouped path the fold models are trained once via
    :func:`~vtscore.training.thresholds.compute_grouped_fold_node_scores` and the
    base orderings are the max-pool of the node data, so the threshold is
    identical to what production's grouped calibration produces for this arm.
    """
    import numpy as np  # noqa: PLC0415

    if cal_groups is not None:
        fold_node_data, fallback = compute_grouped_fold_node_scores(
            X_list,
            y_list,
            input_dim,
            groups=cal_groups,
            rng=np.random.RandomState(42),
            calibrate_count=calibrate_count,
            calibration_fraction=calibration_fraction,
            hidden_dim=hidden_dim,
            score_rows_by_group=score_rows_by_group,
        )
        if fallback is not None:
            return fallback, {
                "provenance": classify_threshold_provenance(fallback),
                "fold_orderings": [],
                "fold_node_data": None,
            }
        # Base (max) orderings from the same fold node data -> identical to
        # production's grouped calibration for this arm.
        fold_orderings = [([float(np.max(b)) for b in blocks], labels) for blocks, labels in fold_node_data]
        threshold = threshold_from_fold_orderings(fold_orderings, inclusion)
        return threshold, {
            "provenance": classify_threshold_provenance(None),
            "fold_orderings": fold_orderings,
            "fold_node_data": fold_node_data,
        }

    # Row-wise path (whole-image styles): no bag flooding, no node re-pooling.
    fold_orderings, fallback = compute_fold_orderings(
        X_list,
        y_list,
        input_dim,
        rng=np.random.RandomState(42),
        calibrate_count=calibrate_count,
        calibration_fraction=calibration_fraction,
        hidden_dim=hidden_dim,
    )
    if fallback is not None:
        return fallback, {
            "provenance": classify_threshold_provenance(fallback),
            "fold_orderings": [],
            "fold_node_data": None,
        }
    threshold = threshold_from_fold_orderings(fold_orderings, inclusion)
    return threshold, {
        "provenance": classify_threshold_provenance(None),
        "fold_orderings": fold_orderings,
        "fold_node_data": None,
    }


def _svm_train_and_calibrate(
    trainer: str,
    good_votes: dict[int, None],
    bad_votes: dict[int, None],
    clips_dict: dict[int, dict[str, Any]],
    target_category: str,
    *,
    inclusion: int,
    calibrate_count: int,
    calibration_fraction: float,
) -> tuple[_StepModel, float, int, dict[str, float], dict[str, Any]]:
    """SVM path — single-vector only (the experiment never region-votes an SVM).

    Threshold uses the trainer-agnostic cross-calibration port
    (:func:`vtscore.eval.trainers._cross_calibrated_threshold`) — the natural
    analogue of the MLP's production calibration — with the fold models pinned
    to the sklearn CPU backend (they are tiny and only feed the threshold, so
    paying GPU launch overhead per fold would be wasteful).  The *final* fit
    honours the ambient backend (cuML on a GPU unless ``VTSEARCH_DISABLE_CUML``
    forces sklearn), and that backend is what the row records and what produces
    the scores.  The SVM fit seed is pinned to 42, mirroring the MLP's fixed
    calibration seed; the eval seed still varies which items are voted.
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.eval.trainers import _train_svm_factory  # noqa: PLC0415
    from vtscore.training.svm import train_svm  # noqa: PLC0415

    X = np.array(
        [media_embedding(clips_dict[vid]) for vid in good_votes]
        + [media_embedding(clips_dict[vid]) for vid in bad_votes],
        dtype=np.float32,
    )
    y = np.array([1] * len(good_votes) + [0] * len(bad_votes), dtype=np.int32)
    n_labels = len(good_votes) + len(bad_votes)

    kernel, kwargs = _parse_trainer_spec(trainer)

    # Fold models for the threshold are pinned to sklearn CPU (tiny fits).
    fold_trainer = _train_svm_factory(kernel, backend="sklearn", **kwargs)
    t_xcal = time.monotonic()
    threshold = _cross_calibrated_threshold(
        X,
        y,
        fold_trainer,
        42,
        inclusion_value=inclusion,
        calibrate_count=calibrate_count,
        cal_fraction=calibration_fraction,
    )
    xcal_seconds = time.monotonic() - t_xcal

    t_train = time.monotonic()
    clf = train_svm(X, y, kernel=kernel, inclusion_value=inclusion, seed=42, **kwargs)  # type: ignore[arg-type]
    train_seconds = time.monotonic() - t_train

    step = _StepModel(
        predict=clf.predict_proba,
        torch_model=None,
        backend=clf.backend,
        device="cuda" if clf.backend == "cuml" else "cpu",
    )
    return step, threshold, n_labels, {"train_seconds": train_seconds, "xcal_seconds": xcal_seconds}, {}


# ------------------------------------------------------------------
# Single (seed, dataset, category) evaluation
# ------------------------------------------------------------------


def simulate_voting_iterations(  # noqa: C901
    clips_dict: dict[int, dict[str, Any]],
    target_category: str,
    seed: int,
    dataset_name: str = "",
    inclusion: int = 0,
    sim_fraction: float = 0.5,
    safe_thresholds: bool = False,
    calibrate_count: int = 2,
    calibration_fraction: float = 0.5,
    region_voting: bool = False,
    strategy: str = "autopilot",
    max_steps: Optional[int] = None,
    atlas_min_node_size: int = 20,
    seed_scores: Optional[dict[int, float]] = None,
    trainer: str = "mlp",
    head: str = "mlp",
    target_prevalence: Optional[float] = None,
    style: Optional[str] = None,
    emit_calibration_metrics: bool = False,
    repool_variants: Optional[list[str]] = None,
    repool_topk: int = 4,
    inclusion_sweep_ks: Optional[list[int]] = None,
    sweep_sink: Optional[list[dict[str, Any]]] = None,
    autopilot_fidelity: bool = True,
    blend_schedule: Optional[str] = None,
    schedule_variants: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Simulate voting on *clips_dict* and evaluate at every step.

    Args:
        clips_dict: Pre-loaded media dict (``{id: clip_data}``).
        target_category: Category treated as the positive class.
        seed: Random seed for splitting and vote ordering.
        dataset_name: Label included in result rows.
        inclusion: Inclusion setting in ``[-10, 10]``.
        trainer: Which ranker to train at each step — ``"mlp"`` (default, the
            production path, numerically unchanged) or an SVM name
            (``"svm_linear"``, ``"svm_rbf"``, or a parameterised spec such as
            ``"svm_rbf@C=3,gamma=scale"``).  The autopilot vote order adapts to
            the chosen model, so MLP and SVM trajectories diverge after the
            first retrain even at the same seed — by design (the question is
            which model makes *VTSearch* better, and VTSearch's vote order
            depends on the model).
        head: Which torch head the ``"mlp"`` trainer fits at each step (see
            :data:`HEADS`): ``"mlp"`` (default) keeps the harness's historical
            auto-sized hidden layer, ``"linear"`` trains the production linear
            (logistic) head shipped in #2790/#2809 — the head a live VTSearch
            detector actually has, so a ``"linear"`` run's thresholds and costs
            are the ones users see.  The head is threaded into the calibration
            folds as well, mirroring how production threads one width through
            ``_train_and_score_xy``.  Ignored on the SVM trainers (no torch
            head); recorded in the ``head`` result column.
        style: Optional detection-style name (see
            :mod:`vtscore.eval.patch_styles`): ``"whole_image"``, ``"max_hac"``,
            or ``"max_patch"``.  When set (MLP trainer only), the style owns the
            vote-to-vector assembly, the test/sim scoring rule, and the
            bag-aware flooding of Bad votes - the Max-Patch experiment arms.
            ``None`` (default) keeps the historical behaviour byte-for-byte
            (including its whole-image Bad votes on patch datasets).  Recorded
            in the ``style`` result column.
        target_prevalence: When set (e.g. ``0.01`` for the 1%-prevalence rare
            arm), positives across the whole dataset are deterministically
            downsampled — using ``seed`` — to that fraction *before* the
            sim/test split, so every FPR/FNR is measured at the target
            prevalence.  ``None`` (default) uses the category's natural
            prevalence and is numerically identical to the pre-prevalence
            harness.  The arm is skipped (returns ``[]``) if it would leave
            fewer than :data:`_MIN_PREVALENCE_POSITIVES` positives, to keep the
            test-set FNR estimable.
        sim_fraction: Fraction of medias used for simulated voting.
        safe_thresholds: When ``True``, blend the cross-calibration threshold
            with a GMM-based threshold for robustness with small label counts.
            Under ``emit_calibration_metrics`` with a *style*, each step
            additionally emits one metric row per safe-threshold GMM variant
            (:data:`_SAFE_GMM_VARIANTS`, tagged in the ``gmm_variant`` column) -
            the #2799 measurement arms.
        calibrate_count: Number of random Train/Calibrate splits for threshold
            calibration (default 2).
        calibration_fraction: Fraction of labelled data reserved for
            calibration in each split (default 0.5).
        region_voting: When ``True``, each Good vote trains on the region-pooled
            vector of the media's ground-truth box for *target_category* (the
            minimal box covering every annotated instance), instead of the
            whole-image vector - simulating a user who drags a region around the
            object.  Requires a patch embedder: media without a ``patch_grid``
            or without an annotated box fall back to the whole-image vector.
            Scoring is unaffected by this flag - a patch dataset always scores
            region-aware (max-pool over regions), so the only thing this toggles
            is the Good-vote training vector, isolating region voting's effect.
        strategy: Vote-order strategy naming *which* pool item the simulated
            user labels next (see :data:`vtscore.eval.al_strategies.STRATEGIES`).
            Only ``"autopilot"`` (the default) exists: it reproduces the app's
            real user flow — seed from text sort (or random known-good examples),
            then the standard Good / Bad / Hard / New phases.
        max_steps: Cap on the number of voting steps (pool items labelled).
            ``None`` (default) votes on the entire simulation set.
        atlas_min_node_size: Minimum leaf population for the coverage atlas the
            autopilot New phase reads (default 20, the production floor).  Lower
            it for small simulation sets so diversity cells actually resolve.
        seed_scores: Optional ``{media_id: similarity}`` text-sort ranking (each
            item's cosine to the typed query).  When provided the autopilot seed
            follows the text sort (top items for the initial goods, the sort's
            cutoff for the initial bads); ``None`` (default) means the dataset
            has no text sort, so autopilot seeds from random known-good examples.
        autopilot_fidelity: When ``True`` (default) the simulated user follows
            the app's own phase machine
            (:class:`vtscore.eval.autopilot_flow.AutopilotFlow`): no detector is
            consulted before the Good/Bad quorum, Bad votes come from the text
            sort's cutoff, Hard picks are nearest-by-rank, and Hard → New → Done
            are driven by the smart/stable/span indicators rather than step
            parity.  ``False`` restores the older approximation so previously
            published studies reproduce byte-for-byte; see ``docs/EVAL.md``.
            Metrics are recorded at every trainable step in both modes — only
            the *vote order* and the ``app_trained`` flag differ.
        blend_schedule: Which safe-threshold mix-in schedule the run *lives*
            under (see :mod:`vtscore.training.blend_schedules`, issue #2841).
            ``None`` selects production.  This one steers the trajectory: the
            blended threshold feeds Autopilot's Hard pick, so two schedules
            label different items and diverge — which is why comparing
            schedules properly needs one run each, not the counterfactual rows
            below.
        schedule_variants: Extra schedule names to *score counterfactually* at
            every step, emitted as one metric row each tagged in the
            ``schedule`` column (requires ``emit_calibration_metrics``).  These
            re-cut the same trajectory, so they screen many schedules for the
            price of one run but are blind to acquisition feedback.

    Returns:
        List of row dicts.  Keys: ``seed, dataset, category, strategy, trainer,
        head, style, prevalence_arm, realized_prevalence, t, n_good, n_bad, phase,
        app_trained, cost, fpr, fnr, auroc, average_precision, train_seconds,
        xcal_seconds, pool_score_seconds, test_score_seconds, backend, device,
        elapsed_seconds``.  ``n_good``/``n_bad`` report the vote counts behind
        each row so callers can tell apart metrics learned from a 1-vs-1 model
        and a many-vs-many one.  ``app_trained`` is 1 exactly when the app would
        have had a trained detector on screen at that step: a threshold recorded
        where it is 0 is one no user would ever see, which is what issue #2788's
        cold-start degenerates turned out to be.
    """
    import numpy as np  # noqa: PLC0415

    rng = np.random.RandomState(seed)
    # Note: no torch.manual_seed() here - train_model handles its own
    # RNG seeding via fork_rng, keeping it thread-safe.
    start_time = time.monotonic()

    if head not in HEADS:
        raise ValueError(f"unknown head {head!r}; expected one of {HEADS}")
    if head != "mlp" and trainer != "mlp":
        raise ValueError(f"head={head!r} only applies to the torch trainer; got trainer={trainer!r}")

    style_obj: Any = None
    if style is not None:
        if trainer != "mlp":
            raise ValueError(f"detection styles only support the MLP trainer; got trainer={trainer!r}")
        from vtscore.eval.patch_styles import resolve_style  # noqa: PLC0415

        style_obj = resolve_style(style)

    prevalence_arm = "natural" if target_prevalence is None else f"rare_{target_prevalence:g}"
    if target_prevalence is not None:
        # Thin positives to the target prevalence *before* splitting, so both the
        # votable sim pool and the held-out test pool sit at that prevalence.
        downsampled = _downsample_to_prevalence(clips_dict, target_category, target_prevalence, rng)
        if downsampled is None:
            return []  # too few positives survive - skip this arm
        clips_dict = downsampled
    realized_prevalence = round(_prevalence(clips_dict, target_category), 6)

    sim_ids, test_ids = _split_media_ids(clips_dict, sim_fraction, rng)

    # Ensure the test set has both positive and negative medias.  Routes through
    # ``media_is_positive`` so multi-label (Visual Genome) images - where the
    # target may be a non-primary category - are counted correctly.
    test_pos = [cid for cid in test_ids if media_is_positive(clips_dict[cid], target_category)]
    test_neg = [cid for cid in test_ids if not media_is_positive(clips_dict[cid], target_category)]
    if not test_pos or not test_neg:
        return []

    # A patch dataset exposes ``patch_regions`` per media; such datasets are
    # scored region-aware (max-pool over regions) the same way the live
    # detector scores them, regardless of how the Good votes were assembled.
    region_aware = any(clips_dict[cid].get("patch_regions") for cid in clips_dict)

    import torch  # noqa: PLC0415

    # Whole-image embeddings of the simulation pool.  These feed the autopilot
    # selector (the example-sort good centroid and the coverage atlas); the
    # Good-vote *training* vector can still be region-pooled below when
    # ``region_voting`` is on.
    sim_embeddings: dict[int, np.ndarray] = {
        cid: np.asarray(media_embedding(clips_dict[cid]), dtype=np.float32) for cid in sim_ids
    }
    input_dim = int(next(iter(sim_embeddings.values())).shape[0])

    # The autopilot New phase reads a coverage atlas built over the pool; it is
    # labelled in lock-step with the votes below so its coverage advances.
    atlas = _build_eval_atlas(sim_embeddings, atlas_min_node_size) if strategy == "autopilot" else None

    # Pre-compute embeddings for safe-threshold GMM scoring.  Restrict to the
    # simulation set so the held-out ``test_ids`` never feed into the GMM that
    # picks the threshold - otherwise the test scores leak into calibration
    # and the reported metrics are biased upward.  Region-aware datasets keep a
    # sim-set snapshot and score it per-step via region max-pool (to match how
    # the test set is scored); single-vector datasets pre-stack whole-image
    # embeddings once.
    sim_clips: dict[int, dict[str, Any]] | None = None
    X_all_clips: Any = None
    if safe_thresholds and (region_aware or style_obj is not None):
        sim_clips = {cid: clips_dict[cid] for cid in sim_ids}
    elif safe_thresholds:
        gmm_clip_embs = np.array([media_embedding(clips_dict[cid]) for cid in sorted(sim_ids)])
        X_all_clips = torch.tensor(gmm_clip_embs, dtype=torch.float32)

    # The #2799 safe-threshold variant rows additionally fit a GMM on the sim
    # set's *whole-image* scores (the historical pre-#2797 fit geometry), so
    # the whole-image matrix is pre-stacked once here.
    X_sim_image: "np.ndarray | None" = None
    if safe_thresholds and emit_calibration_metrics and style_obj is not None:
        X_sim_image = np.stack([sim_embeddings[cid] for cid in sorted(sim_ids)])

    good_votes: dict[int, None] = {}
    bad_votes: dict[int, None] = {}
    labeled: dict[int, float] = {}
    rows: list[dict[str, Any]] = []

    # Voting proceeds one item at a time: the autopilot selector picks the next
    # pool item using the *current* detector (trained at the previous step), the
    # item's ground-truth label is revealed, a fresh model is trained on all
    # votes so far, and the coverage atlas is labelled so its New-phase coverage
    # advances.  Before a trainable model exists the selector runs its seed/bad
    # phases (text sort or example sort), so a cold start still makes real picks.
    pool = sorted(sim_ids)
    # Ground-truth pool labels: autopilot draws its random known-good seed
    # examples from the positives here when no text sort is available.  Cheap to
    # build once up front.
    pool_labels = {cid: (1.0 if media_is_positive(clips_dict[cid], target_category) else 0.0) for cid in sim_ids}
    step: _StepModel | None = None
    threshold = 0.5
    pool_scores: dict[int, float] = {}
    n_steps = len(pool) if max_steps is None else min(max_steps, len(pool))

    # The app's phase machine, driving the vote order the way Autopilot does.
    # Disabled (``None``) under ``autopilot_fidelity=False``, which leaves the
    # selector on its legacy parity interleave.
    flow: Any = None
    if autopilot_fidelity and strategy == "autopilot":
        flow = AutopilotFlow()
    # Recent per-step models, kept so the Smart indicator can re-score the last
    # window of them against the *current* labelset - exactly what the app's
    # ``_eval_cached_models`` does over its per-step cache.
    recent_steps: list[tuple[Any, float]] = []

    for t in range(1, n_steps + 1):
        if not pool:
            break
        phase = flow.phase if flow is not None else None
        ctx = ALContext(
            pool_ids=pool,
            embeddings=sim_embeddings,
            labeled=labeled,
            scores=pool_scores,
            model=step,
            threshold=threshold,
            atlas=atlas,
            rng=rng,
            pool_labels=pool_labels,
            seed_scores=seed_scores,
            phase=phase,
        )
        cid = select_next(strategy, ctx)
        pool.remove(cid)
        is_positive = media_is_positive(clips_dict[cid], target_category)
        if is_positive:
            good_votes[cid] = None
            labeled[cid] = 1.0
        else:
            bad_votes[cid] = None
            labeled[cid] = 0.0
        # Mirror the vote onto the coverage atlas so the New phase's next_sample
        # advances past covered regions (the app labels the atlas the same way).
        if atlas is not None and cid in atlas.vector_to_leaf:
            atlas.label(cid, good=is_positive)

        # Need at least 1 good and 1 bad to train
        if not good_votes or not bad_votes:
            step = None
            # The phase still advances - the app's Good phase ends on its third
            # positive whether or not a detector could be trained, and without
            # this the flow would never leave ``good`` and the run would vote
            # positives forever.
            if flow is not None:
                flow.update(
                    len(good_votes),
                    len(bad_votes),
                    remaining_unlabeled=len(pool),
                    span=atlas.span_info() if atlas is not None else None,
                )
            continue

        step, threshold, n_labels, timings, details = _train_and_calibrate(
            trainer,
            good_votes,
            bad_votes,
            clips_dict,
            target_category,
            region_voting=region_voting,
            input_dim=input_dim,
            inclusion=inclusion,
            calibrate_count=calibrate_count,
            calibration_fraction=calibration_fraction,
            head=head,
            style_obj=style_obj,
            emit_calibration_metrics=emit_calibration_metrics,
        )

        # Apply safe threshold blending if enabled
        sim_pooled_scores: list[float] | None = None
        if safe_thresholds:
            xcal_threshold = threshold
            # Vote-level class counts, so a schedule can ramp on the rarer class
            # (#2841).  The harness votes one media at a time, so bags and votes
            # coincide here and the counts are simply the two vote dicts' sizes.
            blend_ctx = BlendContext(n_labels=n_labels, n_good=len(good_votes), n_bad=len(bad_votes))
            threshold, sim_pooled_scores = _blend_safe_threshold(
                threshold,
                step,
                region_aware,
                sim_clips,
                X_all_clips,
                blend_ctx,
                style_obj=style_obj,
                schedule=blend_schedule,
            )
            if emit_calibration_metrics:
                details["pre_blend_provenance"] = details.get("provenance", "conformal")
                details["provenance"] = "gmm_blend"
                details["xcal_threshold"] = xcal_threshold
                details["n_votes"] = n_labels
                details["n_good"] = len(good_votes)
                details["n_bad"] = len(bad_votes)

        # Evaluate on the held-out test set.  The calibration study (#2781)
        # emits one row per pooling (base + remedial) instead of the single
        # metrics row, but both paths score the same test set here.
        calibration: tuple[list[dict[str, Any]], np.ndarray, np.ndarray] | None = None
        metrics: dict[str, float] = {}
        t_test = time.monotonic()
        if emit_calibration_metrics and style_obj is not None:
            calibration = _calibration_metric_rows(
                step,
                threshold,
                details,
                clips_dict,
                test_ids,
                target_category,
                inclusion,
                style_obj,
                repool_variants or [],
                repool_topk,
            )
        else:
            metrics = _evaluate_on_test(
                step,
                threshold,
                clips_dict,
                test_ids,
                target_category,
                inclusion,
                region_aware=region_aware,
                style_obj=style_obj,
            )
        test_score_seconds = time.monotonic() - t_test

        # Score the remaining pool with the fresh model so the next step's
        # autopilot Hard pick can rank it.
        t_pool = time.monotonic()
        pool_scores = _score_pool(step, pool, clips_dict)
        pool_score_seconds = time.monotonic() - t_pool

        # Advance the app's phase machine on this step's model: the Smart
        # indicator needs the labelset error cost, Stable the prediction flips
        # over the still-unlabeled pool, Span the atlas's coverage.
        if flow is not None:
            recent_steps.append((step, threshold))
            del recent_steps[:-10]  # the app regresses over the last 10 steps
            flow.record_step(
                _labelset_error_cost(recent_steps, good_votes, bad_votes, clips_dict, inclusion),
                {cid: (1 if s >= threshold else 0) for cid, s in pool_scores.items()},
            )
            flow.update(
                len(good_votes),
                len(bad_votes),
                remaining_unlabeled=len(pool),
                span=atlas.span_info() if atlas is not None else None,
            )

        # Identifying columns shared by every row this step emits.
        base_row = {
            "seed": seed,
            "dataset": dataset_name,
            "category": target_category,
            "strategy": strategy,
            "trainer": trainer,
            "head": head,
            "style": style or "",
            "prevalence_arm": prevalence_arm,
            "realized_prevalence": realized_prevalence,
            "t": t,
            "n_good": len(good_votes),
            "n_bad": len(bad_votes),
            "phase": flow.phase if flow is not None else "",
            "app_trained": 1 if (flow is None or app_has_detector(flow.phase)) else 0,
        }
        timing_cols = {
            "train_seconds": round(timings["train_seconds"], 6),
            "xcal_seconds": round(timings["xcal_seconds"], 6),
            "pool_score_seconds": round(pool_score_seconds, 6),
            "test_score_seconds": round(test_score_seconds, 6),
            "backend": step.backend,
            "device": step.device,
            "elapsed_seconds": round(time.monotonic() - start_time, 3),
        }

        if calibration is not None:
            metric_rows, base_scores, base_labels = calibration
            # One extra row per safe-threshold GMM variant (issue #2799), all
            # evaluated against the same held-out max-pooled test scores.
            if X_sim_image is not None and sim_pooled_scores is not None:
                sim_image_scores = np.asarray(step.predict(X_sim_image)).ravel().tolist()
                metric_rows.extend(
                    _safe_gmm_variant_rows(
                        details,
                        base_scores,
                        base_labels,
                        sim_pooled_scores,
                        sim_image_scores,
                        inclusion,
                        n_pool_rows=metric_rows[0]["n_pool_rows"],
                    )
                )
            # One extra row per mix-in schedule (issue #2841), on the production
            # cut.  Independent of the image-geometry rows above: the schedule
            # screen only needs the pooled sim scores the blend actually fits.
            if schedule_variants and sim_pooled_scores is not None:
                metric_rows.extend(
                    _schedule_variant_rows(
                        details,
                        base_scores,
                        base_labels,
                        sim_pooled_scores,
                        inclusion,
                        n_pool_rows=metric_rows[0]["n_pool_rows"],
                        schedules=schedule_variants,
                    )
                )
            for mr in metric_rows:
                rows.append({**base_row, **mr, **timing_cols})
            # The near-free inclusion-budget sweep, into the side sink.
            if inclusion_sweep_ks and sweep_sink is not None:
                for sr in _inclusion_sweep_rows(details, base_scores, base_labels, inclusion_sweep_ks):
                    sweep_sink.append({**base_row, **sr})
        else:
            rows.append({**base_row, **metrics, **timing_cols})

    return rows


# ------------------------------------------------------------------
# Full evaluation across seeds x datasets x categories
# ------------------------------------------------------------------


def run_voting_iterations_eval(
    dataset_clips: dict[str, dict[int, dict[str, Any]]],
    seeds: list[int],
    categories: Optional[dict[str, list[str]]] = None,
    inclusion: int = 0,
    sim_fraction: float = 0.5,
    safe_thresholds: bool = False,
    calibrate_count: int = 2,
    calibration_fraction: float = 0.5,
    region_voting: bool = False,
    strategies: Optional[list[str]] = None,
    max_steps: Optional[int] = None,
    atlas_min_node_size: int = 20,
    seed_scores: Optional[dict[str, dict[str, dict[int, float]]]] = None,
    trainers: Optional[list[str]] = None,
    prevalence_arms: Optional[list[Optional[float]]] = None,
    styles: Optional[list[Optional[str]]] = None,
    autopilot_fidelity: bool = True,
) -> pd.DataFrame:
    """Run the voting-iterations evaluation over multiple seeds/datasets/categories.

    Args:
        dataset_clips: Mapping of dataset name to a pre-loaded medias dict.
            Each medias dict maps ``int`` media IDs to media data dicts
            (must carry a resolvable embedding in the per-embedder
            ``"embeddings"`` store and a ``"category"`` key).
        seeds: List of random seeds to iterate over.
        categories: Optional mapping of dataset name to list of target
            categories.  If ``None`` or a dataset is missing from the dict,
            all unique categories in that dataset are used.
        inclusion: Inclusion setting in ``[-10, 10]``.
        sim_fraction: Fraction of medias reserved for simulated voting.
        safe_thresholds: When ``True``, blend thresholds with GMM
            (see :func:`simulate_voting_iterations`).
        calibrate_count: Number of random Train/Calibrate splits for threshold
            calibration (default 2).
        calibration_fraction: Fraction of labelled data reserved for
            calibration in each split (default 0.5).
        region_voting: When ``True``, Good votes train on the ground-truth
            region-pooled vector for patch datasets (see
            :func:`simulate_voting_iterations`).
        strategies: Vote-order strategies to run (see
            :data:`vtscore.eval.al_strategies.STRATEGIES`).  ``None`` (default)
            runs ``["autopilot"]``, the only strategy; the name is recorded in
            the ``strategy`` result column.
        max_steps: Cap on the number of voting steps per run (see
            :func:`simulate_voting_iterations`).
        atlas_min_node_size: Minimum coverage-atlas leaf population for the
            autopilot New phase (see :func:`simulate_voting_iterations`).
        seed_scores: Optional text-sort rankings keyed
            ``{dataset: {category: {media_id: similarity}}}``.  When a
            (dataset, category) has an entry, the autopilot seed follows that
            text ranking; otherwise it seeds from random known-good examples.
        trainers: Which rankers to run at each cell (see
            :func:`simulate_voting_iterations`).  ``None`` (default) runs
            ``["mlp"]``; pass e.g. ``["mlp", "svm_linear", "svm_rbf"]`` for the
            head-to-head comparison.  Recorded in the ``trainer`` column.
        prevalence_arms: Which prevalence arms to run per (dataset, category).
            ``None`` (default) runs ``[None]`` (natural prevalence only); pass
            e.g. ``[None, 0.01]`` to add the 1%-prevalence rare arm.  Recorded
            in the ``prevalence_arm`` / ``realized_prevalence`` columns.
        styles: Which detection styles to run per cell (see
            :func:`simulate_voting_iterations`).  ``None`` (default) runs
            ``[None]`` - the historical style-less behaviour; pass e.g.
            ``["max_hac", "max_patch"]`` for the Max-Patch experiment arms.
            Recorded in the ``style`` column (``""`` for the style-less run).
        autopilot_fidelity: Follow the app's own Autopilot phase machine
            (default ``True``); see :func:`simulate_voting_iterations`.  Pass
            ``False`` to reproduce studies published before the flow was
            aligned.

    Returns:
        A :class:`~pandas.DataFrame` with the columns listed in
        :data:`_VOTING_COLUMNS`.
    """
    import pandas as pd  # noqa: PLC0415

    strategy_list = strategies if strategies is not None else ["autopilot"]
    trainer_list = trainers if trainers is not None else ["mlp"]
    arm_list = prevalence_arms if prevalence_arms is not None else [None]
    style_list = styles if styles is not None else [None]
    all_rows: list[dict[str, Any]] = []

    for ds_name, clips_dict in dataset_clips.items():
        # Determine target categories.  For multi-label datasets each image's
        # ``category`` is only its primary, so fall back to the union of every
        # image's ``categories`` list when present.
        if categories and ds_name in categories:
            target_cats = categories[ds_name]
        else:
            cat_set: set[str] = set()
            for cid in clips_dict:
                media = clips_dict[cid]
                cat_set.update(media.get("categories") or [media["category"]])
            target_cats = sorted(cat_set)

        for seed in seeds:
            for cat in target_cats:
                cat_seed_scores = (seed_scores or {}).get(ds_name, {}).get(cat)
                for arm in arm_list:
                    for strategy in strategy_list:
                        for trainer in trainer_list:
                            for style in style_list:
                                rows = simulate_voting_iterations(
                                    clips_dict,
                                    target_category=cat,
                                    seed=seed,
                                    dataset_name=ds_name,
                                    inclusion=inclusion,
                                    sim_fraction=sim_fraction,
                                    safe_thresholds=safe_thresholds,
                                    calibrate_count=calibrate_count,
                                    calibration_fraction=calibration_fraction,
                                    region_voting=region_voting,
                                    strategy=strategy,
                                    max_steps=max_steps,
                                    atlas_min_node_size=atlas_min_node_size,
                                    seed_scores=cat_seed_scores,
                                    trainer=trainer,
                                    target_prevalence=arm,
                                    style=style,
                                    autopilot_fidelity=autopilot_fidelity,
                                )
                                all_rows.extend(rows)

    return pd.DataFrame(all_rows, columns=pd.Index(list(_VOTING_COLUMNS)))


def run_voting_iterations_eval_from_pickles(
    dataset_paths: dict[str, str],
    seeds: list[int],
    categories: Optional[dict[str, list[str]]] = None,
    inclusion: int = 0,
    sim_fraction: float = 0.5,
    safe_thresholds: bool = False,
    calibrate_count: int = 2,
    calibration_fraction: float = 0.5,
    region_voting: bool = False,
    strategies: Optional[list[str]] = None,
    max_steps: Optional[int] = None,
    atlas_min_node_size: int = 20,
    seed_scores: Optional[dict[str, dict[str, dict[int, float]]]] = None,
    trainers: Optional[list[str]] = None,
    prevalence_arms: Optional[list[Optional[float]]] = None,
    styles: Optional[list[Optional[str]]] = None,
    autopilot_fidelity: bool = True,
) -> pd.DataFrame:
    """Convenience wrapper that loads datasets from pickle files.

    Args:
        dataset_paths: Mapping of dataset name to pickle file path.
        seeds: List of random seeds.
        categories: Optional category filter (see :func:`run_voting_iterations_eval`).
        inclusion: Inclusion setting in ``[-10, 10]``.
        sim_fraction: Fraction of medias for simulation.
        safe_thresholds: When ``True``, blend thresholds with GMM.
        calibrate_count: Number of random Train/Calibrate splits for threshold
            calibration (default 2).
        calibration_fraction: Fraction of labelled data reserved for
            calibration in each split (default 0.5).
        region_voting: When ``True``, Good votes train on the ground-truth
            region-pooled vector for patch datasets (see
            :func:`simulate_voting_iterations`).
        strategies: Vote-order strategies to run (see
            :func:`run_voting_iterations_eval`).
        max_steps: Cap on the number of voting steps per run.
        atlas_min_node_size: Minimum coverage-atlas leaf population for the
            autopilot New phase.
        seed_scores: Optional text-sort rankings keyed
            ``{dataset: {category: {media_id: similarity}}}`` (see
            :func:`run_voting_iterations_eval`).

    Returns:
        A :class:`~pandas.DataFrame` identical to :func:`run_voting_iterations_eval`
        (columns: ``seed, dataset, category, strategy, t, n_good, n_bad, cost,
        fpr, fnr, elapsed_seconds``).
    """
    from vtscore.datasets.loader import load_dataset_from_pickle

    dataset_clips: dict[str, dict[int, dict[str, Any]]] = {}
    for name, path in dataset_paths.items():
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(Path(path), medias)
        dataset_clips[name] = medias

    return run_voting_iterations_eval(
        dataset_clips,
        seeds=seeds,
        categories=categories,
        inclusion=inclusion,
        sim_fraction=sim_fraction,
        safe_thresholds=safe_thresholds,
        calibrate_count=calibrate_count,
        calibration_fraction=calibration_fraction,
        region_voting=region_voting,
        strategies=strategies,
        max_steps=max_steps,
        atlas_min_node_size=atlas_min_node_size,
        seed_scores=seed_scores,
        trainers=trainers,
        prevalence_arms=prevalence_arms,
        styles=styles,
        autopilot_fidelity=autopilot_fidelity,
    )
