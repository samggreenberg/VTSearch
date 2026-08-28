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

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from vtscore.training.thresholds import FoldAnchoredCut

from vtscore.embedding.media_vectors import media_embedding
from vtscore.eval.al_strategies import ALContext, select_next
from vtscore.eval.autopilot_flow import SMART_WINDOW, AutopilotFlow, app_has_detector
from vtscore.eval.startup_schedule import StartupState, parse_startup_schedule, round_cut
from vtscore.eval.labels import evaluable_pool, media_is_positive, region_box_for_category
from vtscore.eval.score_dumps import maybe_dump_predictions
from vtscore.eval.trainers import _cross_calibrated_threshold, _parse_trainer_spec
from vtscore.training.blend_schedules import BlendContext
from vtscore.training.mlp import LINEAR_HEAD, LINEAR_SVM_HEAD, _auto_hidden_dim, train_model
from vtscore.training.thresholds import (
    ACQUISITION_INCLUSION_OFFSET,
    CUT_KIND_INTERIOR,
    FOLD_ANCHOR_QTILT_STEP,
    NO_GOOD_THRESHOLD,
    acquisition_inclusion,
    anchored_gmm_fit,
    calculate_safe_threshold,
    calibration_folds,
    classify_threshold_provenance,
    compute_fold_orderings,
    compute_grouped_fold_node_scores,
    fold_anchored_gmm_threshold,
    gmm_cut_from_fit,
    rank_transfer,
    threshold_from_fold_orderings,
    threshold_from_folds,
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


#: Head choices for the harness's per-step ranker, all three reached through
#: the same ``hidden_dim`` sentinel production threads.  ``"linear_svm"`` is the
#: head the live detector trains (:data:`~vtscore.training.mlp.LINEAR_SVM_HEAD`,
#: a single ``Linear(d, 1)`` fitted to the maximum-margin boundary), so a
#: ``"linear_svm"`` run measures the shipped detector.  ``"linear"`` is the same
#: architecture fitted with balanced BCE — logistic regression, the head shipped
#: between #2790/#2809 and the SVM switch — and ``"mlp"`` is the older harness
#: candidate, a hidden layer auto-sized from the vote count
#: (:func:`~vtscore.training.mlp._auto_hidden_dim`).  The choice is threaded into
#: the calibration folds too, exactly as production threads one sentinel through
#: ``_train_and_score_xy``.
HEADS: tuple[str, ...] = ("mlp", "linear", "linear_svm")

#: The head the **app** trains, and therefore the harness's default arm:
#: ``vtscore.detectors.training.train_and_threshold`` pins ``hidden_dim =
#: LINEAR_SVM_HEAD`` on every production fit.  ``head=None`` resolves to this,
#: the way ``style=None`` and ``blend_schedule=None`` resolve to the app's
#: geometry and blend schedule — an eval default that isn't the app default
#: measures a detector nobody ships (see the "Eval Default Arm IS the App"
#: rule).  If the shipped head ever changes, move this with it:
#: ``test_harness_linear_head`` pins the two against each other by training the
#: real app pipeline, so the suite fails rather than letting the default arm
#: drift silently.
PRODUCTION_HEAD: str = "linear_svm"


#: The detection geometry a live detector uses on a patch dataset - the style an
#: unspecified ``style=`` resolves to below.  Named rather than inlined so that
#: "is this run measuring the shipped geometry?" is a question something can
#: *ask*: `scripts/experiments/preflight.sh` compares a study's configured
#: styles against it, the way it compares a study's head against
#: :data:`PRODUCTION_HEAD`.  The HAC hybrids in
#: :mod:`vtscore.eval.patch_styles` are experiment-only arms; #2886 removed the
#: region tree from ingest.
PRODUCTION_PATCH_STYLE: str = "max_patch"


def _resolve_hidden_dim(head: str, n_votes: int) -> int:
    """``hidden_dim`` sentinel for *head* at *n_votes* votes.

    The two linear heads return their sentinels (they have no width to size);
    only ``"mlp"`` consults the vote count.
    """
    if head == "linear_svm":
        return LINEAR_SVM_HEAD
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
    #: The parameterised opening this run took (issue #3267), verbatim - so a
    #: pooled frame says which arm each row came from without depending on the
    #: directory it was read out of.  Empty on every run that took the app's
    #: own opening, which is every study before #3267.
    "startup_schedule",
    # --- Acquisition/reporting decoupling (docs/ML.md, threshold calibration).
    #: The threshold handed to the *selector* this step - cut
    #: ``acq_inclusion_offset`` inclusion steps below ``threshold``.  Equal to it
    #: on steps with no fold-anchored fit to re-cut, and at offset 0.
    "acq_threshold",
    #: Where each threshold sits in the **pool** score distribution the selector
    #: actually ranks - the two are emitted together on purpose.  Autopilot's
    #: ``hard`` pick works in rank space (:func:`~vtscore.eval.al_strategies.
    #: _hard_pick_by_index`), so "did the sampling position move, and how far"
    #: is a question about these two numbers, not about the thresholds.  Without
    #: them a sign error in the acquisition cut is invisible.
    "acq_pool_percentile",
    "report_pool_percentile",
)

#: Canonical column order for the voting-iterations result frame.  Kept in one
#: place so :func:`run_voting_iterations_eval` and downstream tooling agree.
_VOTING_COLUMNS: tuple[str, ...] = (
    *_IDENT_COLUMNS,
    "cost",
    "fpr",
    "fnr",
    #: The operating point in the words a reader picks off a menu (#3281).
    #: ``recall`` is exactly ``1 - fnr`` and is emitted anyway: asking someone to
    #: invert an FNR in their head is where the reading errors come from.  One
    #: definition for all three, in ``calibration_metrics.detection_metrics``.
    "precision",
    "recall",
    "f1",
    #: The counts behind them, so a rate can be re-derived, weighted or pooled
    #: without going back to the cells.
    "n_test_pos",
    "n_test_neg",
    "n_flagged",
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

#: Column order for the per-click **pick log** (issue #3267): one row for every
#: vote the simulated user casts, emitted only when the caller passes a
#: ``pick_sink``.
#:
#: The main frame cannot answer the questions this study asks.  It starts at the
#: first *trainable* step - before one Good and one Bad vote coexist there is no
#: model, no threshold and no metrics row - so the opening, which is the whole
#: subject here, is exactly the part it does not record.  This frame records
#: every click instead: what was picked, whether it turned out to be a positive,
#: and **where on the seed sort it came from**, which is what makes "why was this
#: arm better" answerable rather than merely visible in the totals.
_PICK_COLUMNS: tuple[str, ...] = (
    "seed",
    "dataset",
    "category",
    "startup_schedule",
    "style",
    "t",
    "phase",
    #: Index of the schedule round this click was spent in, or -1 outside one.
    "startup_round",
    #: The round's cut on the seed sort, and where that lands in the sort's own
    #: score distribution - the sampling *position*, which is what the arms
    #: actually differ by.  NaN / -1 outside a round.
    "startup_cut",
    "startup_cut_percentile",
    "picked_id",
    #: Ground truth for the click: 1 if the item was a positive.
    "picked_label",
    #: Where the picked item sat in the seed sort - as a 0-based rank over the
    #: whole sort and as a percentile (0 = top).  Together with ``picked_label``
    #: this is the mining record: how deep the arm had to reach for each
    #: positive it found.
    "picked_seed_rank",
    "picked_seed_percentile",
    #: The seed-sort similarity of the picked item.
    "picked_seed_score",
    #: The detector score the *previous* step's model gave this item, and the
    #: acquisition cut it was picked against.  NaN before a model exists.
    "picked_detector_score",
    "acq_threshold",
    #: Whether this click was spent PAST the written schedule, held on its last
    #: round because one vote class was still empty.  An arm's opening is only
    #: as long as it was written where this is False, so it is what makes a
    #: length-matched control actually length-matched - and a cell whose whole
    #: horizon is held is total Good-starvation, the phenomenon #3267 is about.
    "startup_held",
    #: How many such clicks have been spent so far in this trajectory.
    "startup_extended_clicks",
    #: Running vote totals **after** this click.
    "n_good",
    "n_bad",
    #: Pool items still unlabelled after this click.
    "n_pool",
)

#: Column order for the calibration study's main per-step frame (issue #2781),
#: emitted only when ``emit_calibration_metrics``.  One row per ``pool_variant``;
#: under ``safe_thresholds`` additionally one row per safe-threshold GMM variant
#: (issue #2799), tagged in ``gmm_variant`` (``""`` on every other row).  The
#: fold-count arms (issues #2897, #3116, #3115) ride the same tag as
#: ``folds_k{K}_{xcal,blend,anchored,anchored_qmedian,tmean,tmedian,qmean,qmedian}``
#: and additionally fill ``fold_count`` / ``fold_seconds`` / ``n_cal_scores`` /
#: ``n_folds_used``.
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
    "cut_fallback",
    "cut_fallback_kind",
    "cut_fail_reason",
    "raw_cut_cost",
    "raw_cut_fpr",
    "raw_cut_fnr",
    "cost",
    "fpr",
    "fnr",
    "precision",
    "recall",
    "f1",
    "n_test_pos",
    "n_test_neg",
    "n_flagged",
    "auroc",
    "average_precision",
    "oracle_threshold",
    "oracle_cost",
    "oracle_fpr",
    "oracle_fnr",
    "regret",
    "oracle_threshold_honest",
    "oracle_cost_honest",
    "regret_honest",
    "cal_oracle_threshold",
    "cal_oracle_cost",
    "rule_inefficiency",
    "calibration_shift",
    "calibration_shift_honest",
    "n_pool_rows",
    "fold_count",
    "fold_seconds",
    "n_cal_scores",
    "n_folds_used",
    "train_seconds",
    "xcal_seconds",
    "pool_score_seconds",
    "test_score_seconds",
    "backend",
    "device",
    "elapsed_seconds",
)

#: Column order for the cut-decomposition side frame (issue #2836): one row per
#: (step, geometry), written to a separate CSV by the runner.  Carries the fitted
#: mixture parameters, the two families' fit quality, the label-supervised class
#: moments, and every cut in the decomposition chain, so the analyzer can test
#: the derivation offline without re-running the simulation.
#:
#: The chain telescopes ``tau_cross -> tau_priorfree -> tau_supervised ->
#: tau_sim_oracle -> tau_test_oracle``; each consecutive pair differs in exactly
#: one assumption (prior/loss, component identification, Gaussian shape, finite
#: sim set), so the terms sum to the total error of today's rule.
_CUT_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    *_IDENT_COLUMNS,
    "geometry",
    "sim_n",
    "sim_prevalence",
    "fallback_median",
    # Fitted Gaussian mixture.
    "gmm_ok",
    "w_lo",
    "mu_lo",
    "var_lo",
    "w_hi",
    "mu_hi",
    "var_hi",
    "gmm_loglik",
    "pred_offset_equal_var",
    "gmm_logit_loglik",
    # Fitted Gumbel + Normal mixture.  Its component parameters are in LOGIT
    # units (that is where the extreme-value limit lives and where it is fitted);
    # its log likelihood is converted back to score space so the two families are
    # directly comparable.  Reported per component, with ``evt_gumbel_is_low``
    # saying which mode the Gumbel landed on - #2836 assumed that was always the
    # low one and threw away every fit that said otherwise, which #2846 measured
    # at 14 % of production-like fits.
    "evt_ok",
    "evt_fit_fail",
    "evt_gumbel_is_low",
    "evt_w_gumbel",
    "evt_loc",
    "evt_scale",
    "evt_mu",
    "evt_var",
    "evt_loglik",
    "evt_loglik_gain",
    # Label-supervised class moments (diagnostic only).
    "s_mu_neg",
    "s_var_neg",
    "s_mu_pos",
    "s_var_pos",
    "s_prevalence",
    # The cut chain.
    "tau_mid",
    "tau_cross",
    "tau_priorfree",
    "tau_rate",
    "tau_gumbel_cross",
    "tau_gumbel_priorfree",
    "tau_gumbel_rate",
    "tau_gumbel_any_cross",
    "tau_gumbel_any_priorfree",
    "tau_gumbel_any_rate",
    # #2881's tail-quantile sweep, one column per swept alpha (in milli-alpha).
    "tau_tail_a040",
    "tau_tail_a080",
    "tau_tail_a110",
    "tau_tail_a158",
    "tau_tail_a220",
    "tau_tail_a300",
    "tau_tail_a400",
    "tau_bagfit_mid",
    "tau_bagfit_priorfree",
    "tau_supervised",
    "tau_sim_oracle",
    "tau_sim_oracle_f050",
    "tau_sim_oracle_f100",
    "tau_sim_oracle_f250",
    "tau_sim_oracle_f500",
    "tau_sim_oracle_bag",
    "tau_sim_oracle_smooth",
    "tau_test_oracle",
    # #2883: the reference point, honestly.  `tau_test_oracle` above is the
    # argmin of the empirical cost on the test sample itself, so it is a sample
    # minimum and the gap measured against it is biased high.  The cross-fitted
    # pair chooses the cut and pays for it on disjoint folds; the two costs
    # bracket the population optimum the chain actually wants.
    "tau_test_oracle_honest",
    "cost_test_oracle_naive",
    "cost_test_oracle_honest",
    # Sample sizes the last link's variance should scale with, recorded so the
    # scaling claim is read off the run rather than off the dataset's nominal
    # size (thinning, the prevalence arm and per-category positives all move it).
    "sim_n_pos",
    "test_n",
    "test_n_pos",
    # Where the true optimum sits in each fitted Bad component's upper tail.
    "oracle_lo_sf_gauss",
    "oracle_lo_sf_evt",
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

#: Column order for the **cut-rule x inclusion** side frame (issue #2865): one
#: row per (step, fold-anchored arm, inclusion ``k``), written to its own CSV.
#:
#: Distinct from :data:`_INCLUSION_SWEEP_COLUMNS`, which sweeps the *conformal*
#: rule's budget and asks whether its ``alpha(k)`` guarantee holds.  This frame
#: sweeps the **fold-anchored** estimator's cut *rules* and asks which one
#: should answer the knob at all - so every row is scored under the cost weights
#: **of its own k** (not the run's reporting inclusion), against the oracle at
#: that same k, which is what makes an arm's regret comparable across the knob.
#:
#: ``admitted_frac`` is the second decision number and the one with no analogue
#: anywhere else in the harness: a rule that moves the *threshold* without
#: moving the *admitted set* has not restored the knob.  Because the cut is
#: carried to the final model as a quantile, a whole band of the slider can
#: realize to one admitted set on a cleanly separated haystack - so the
#: analyzer's headline is how many distinct admitted sets survive across the
#: nominal range, per arm.
_CUT_INCLUSION_COLUMNS: tuple[str, ...] = (
    *_IDENT_COLUMNS,
    "arm",
    "cut_rule",
    "anchor_weight",
    "combine",
    "qtilt_step",
    "inclusion_k",
    "fold_quantile",
    "cut_threshold",
    "cut_cost",
    "cut_fpr",
    "cut_fnr",
    "k_oracle_threshold",
    "k_oracle_cost",
    "cut_regret",
    "admitted_frac",
    "n_admitted",
    "n_test",
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


#: Minimum positives an arm must retain after prevalence downsampling.  Below
#: this the held-out test set has too few positives for a stable FNR estimate,
#: so the arm is skipped rather than reported with a noisy denominator.
_MIN_PREVALENCE_POSITIVES = 15


def _inclusion_weights(inclusion: int) -> tuple[float, float]:
    """``(fpr_weight, fnr_weight)`` for an inclusion value.

    Delegates to the production definition so a measured cost and the shipped
    threshold rule can never disagree about what an inclusion value prices.
    """
    from vtscore.training.thresholds import inclusion_cost_weights  # noqa: PLC0415

    return inclusion_cost_weights(inclusion)


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


def _score_sim_set_with_model(
    model: Any,
    region_aware: bool,
    sim_clips: dict[int, dict[str, Any]] | None,
    X_all_clips: Any,
    sim_ids: list[int],
    style_obj: Any = None,
) -> tuple[list[int], list[float]]:
    """``(ids, scores)`` for the simulation set under an arbitrary torch *model*.

    The same scorer the test set uses, so the population estimator sees the
    distribution the threshold will actually cut: through the detection style
    when one is given, else region max-pool on a patch dataset, else the
    pre-computed whole-image matrix *X_all_clips* (stacked over
    ``sorted(sim_ids)`` - that ordering is preserved).
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    if style_obj is not None:
        assert sim_clips is not None
        score_map = style_obj.score_media(model, sim_clips)
        ids = list(score_map.keys())
        return ids, [float(score_map[cid]) for cid in ids]
    if region_aware:
        from vtscore.detectors.training import score_media_with_model  # noqa: PLC0415

        assert sim_clips is not None
        scored = score_media_with_model(model, sim_clips)
        return [int(r["id"]) for r in scored], [float(r["score"]) for r in scored]
    with torch.no_grad():
        t = torch.tensor(np.asarray(X_all_clips), dtype=torch.float32).to(next(model.parameters()).device)
        scores = torch.sigmoid(model(t)).squeeze(1).cpu().numpy()
    return sorted(sim_ids), [float(s) for s in scores]


def _pool_percentile(pool_scores: dict[int, float], threshold: float) -> float:
    """Fraction of the *unlabelled pool* scoring below *threshold*.

    The selector's ``hard`` pick works in rank space over the pool, so this - not
    the threshold's value, and not its percentile in the held-out test scores -
    is the number that says where the next item comes from.  Returns NaN on an
    empty pool rather than a misleading 0.0.
    """
    import numpy as np  # noqa: PLC0415

    if not pool_scores:
        return float("nan")
    arr = np.asarray(list(pool_scores.values()), dtype=np.float64)
    return round(float((arr < threshold).mean()), 6)


def _sorted_percentile(descending: list[float], value: float) -> float:
    """Where *value* cuts a **descending** score list, as a fraction from the top.

    ``0`` = above every score, ``1`` = below every score.  Used for the pick
    log's ``startup_cut_percentile``, so a round's cut is reported as the
    sampling *position* it actually is rather than as a bare similarity whose
    scale differs per category.  Infinite cuts (the ``top`` round) read 0.
    """
    import numpy as np  # noqa: PLC0415

    if not descending:
        return float("nan")
    idx = int(np.searchsorted(-np.asarray(descending, dtype=np.float64), -value, side="left"))
    return _r(idx / len(descending))


def _blend_xcal_input(threshold: float, details: dict[str, Any]) -> float:
    """The x-cal side of the schedule blend, with the app's sentinel substitution.

    When the fold computation could not calibrate at all it returns a *sentinel*
    rather than a cut - ``0.5`` on the too-few-labels / fewer-than-two-per-class
    paths, :data:`~vtscore.training.thresholds.NO_GOOD_THRESHOLD` when the split
    itself is degenerate (see
    :func:`~vtscore.training.thresholds.compute_fold_orderings`).  Production
    does **not** blend whichever sentinel came back: ``_fused_threshold`` feeds
    the blend ``NO_GOOD_THRESHOLD`` ("we never computed a cut, so admit
    nothing") whenever ``folds.fallback is not None``, regardless of the
    sentinel's value.  This applies that same substitution, so the harness's
    shipped-threshold arm blends what the app blends.

    It matters exactly in the cold start: the production schedules ramp from
    ``lo=6`` labels, so a step past that with one class still under two votes -
    the rare-class starvation the autopilot flow reaches whenever the Bad phase
    keeps surfacing positives - carries real weight on the x-cal side, and
    ``2.0`` vs ``0.5`` moves both the recorded operating point and the
    acquisition cut.

    *details* carries ``fold_fallback`` on every torch path (``None`` when the
    folds are real).  The SVM arms carry no fold fallback at all - their
    threshold comes from the trainer-agnostic port, which has no production
    counterpart to mirror - so they blend their own returned value unchanged.
    """
    return NO_GOOD_THRESHOLD if details.get("fold_fallback") is not None else threshold


def _safe_threshold_for_step(
    threshold: float,
    step: _StepModel,
    details: dict[str, Any],
    region_aware: bool,
    sim_clips: dict[int, dict[str, Any]] | None,
    X_all_clips: Any,
    ctx: "BlendContext",
    sim_ids: list[int],
    inclusion: int,
    style_obj: Any = None,
    schedule: str | None = None,
    voted_ids: "set[int] | None" = None,
) -> tuple[float, list[float], list[int], list[Any], str, "FoldAnchoredCut | None"]:
    """The harness's **shipped** safe threshold - the same rule the app applies.

    Scores the simulation set (the harness's haystack) with the final model and
    with each calibration fold model, then cuts via
    :func:`~vtscore.training.thresholds.fold_anchored_gmm_threshold` at the
    production defaults (the ``FOLD_ANCHOR_*`` constants).  This is the
    estimator :func:`vtscore.detectors.training._safe_threshold` ships, called
    with the same arguments, so the harness's baseline arm cannot drift from
    the app's behaviour - the paired ``*_variant`` rows are where deliberate
    deviations live.

    Falls back to the schedule blend
    (:func:`~vtscore.training.thresholds.calculate_safe_threshold`) exactly
    where production does: no usable calibration folds.  The blend's x-cal side
    carries production's sentinel substitution (see :func:`_blend_xcal_input`) -
    a step whose folds fell back blends ``NO_GOOD_THRESHOLD``, not whichever
    sentinel the fold rule returned.  The SVM arms always land on the blend -
    their fold models are standalone sklearn estimators rather than the head
    the app trains, so there is no production path for them to match.

    Returns ``(threshold, sim_scores, sim_ids, fold_haystacks, provenance, cut)``.
    The fitted :class:`~vtscore.training.thresholds.FoldAnchoredCut` rides along
    (``None`` on the blend fallback) so a caller can re-cut the *same* fit at
    another inclusion without refitting - which is what the acquisition cut does
    (``acq_inclusion_offset``; see ``docs/ML.md``, threshold calibration).
    The sim scores ride along so the #2799 / #2836 / #2852 variant rows can
    re-cut the same distribution without a second scoring pass, their media ids
    with them so a variant can attach each score's true label without assuming
    the scorer preserved any ordering, and the per-fold haystack score arrays
    so the fold-anchored variant grid re-fits without re-scoring.

    *voted_ids* mirrors the app's #3308 exclusion
    (:func:`vtscore.detectors.training._fused_threshold`): the voted items are
    dropped from every haystack the fold-anchored estimator fits on - the
    per-fold arrays (which are also what the returned ``fold_haystacks`` carry,
    so the variant grids inherit the same population convention) and the final
    model's realization sample.  The *returned* sim scores stay complete: they
    feed evaluation and acquisition, not the fit.
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.training.thresholds import fit_fold_anchored_cut  # noqa: PLC0415

    exclude = voted_ids or set()

    def _drop_voted(scores: list[float], score_ids: list[int]) -> "np.ndarray":
        arr = np.asarray(scores, dtype=np.float64)
        if not exclude:
            return arr
        keep = np.fromiter((i not in exclude for i in score_ids), dtype=bool, count=len(score_ids))
        return arr[keep]

    final_model = step.torch_model
    if style_obj is not None or region_aware:
        assert final_model is not None
        ids, all_scores = _score_sim_set_with_model(
            final_model, region_aware, sim_clips, X_all_clips, sim_ids, style_obj
        )
    else:
        # Trainer-agnostic: the SVM arms have no torch model to forward.
        ids = sorted(sim_ids)
        all_scores = np.asarray(step.predict(np.asarray(X_all_clips))).ravel().tolist()

    fold_models = details.get("fold_models") or []
    fold_orderings = details.get("fold_orderings") or []
    n_folds = min(len(fold_models), len(fold_orderings))
    fold_haystacks: list[Any] = []
    for model in fold_models[:n_folds]:
        fids, fscores = _score_sim_set_with_model(model, region_aware, sim_clips, X_all_clips, sim_ids, style_obj)
        fold_haystacks.append(_drop_voted(fscores, fids))

    # #3116: the #2897 fold-count arms need a haystack per fold to re-fit the
    # *shipped* rule at each K, and `details["fold_models"]` is trimmed to the
    # live `calibrate_count`.  Score the extra Kmax-run folds here, where the
    # sim set and the scoring machinery are already in hand, and stash them
    # beside the orderings for :func:`_fold_count_variant_rows`.  Only the
    # models past the live prefix are scored - the folds are nested, so the
    # first `n_folds` haystacks are the ones just computed above, and this adds
    # `Kmax - calibrate_count` scoring passes rather than Kmax of them.
    fold_data = details.get("fold_count_data")
    if fold_data is not None and fold_data.get("models"):
        extended = list(fold_haystacks)
        for model in fold_data["models"][len(extended) :]:
            fids, fscores = _score_sim_set_with_model(model, region_aware, sim_clips, X_all_clips, sim_ids, style_obj)
            extended.append(_drop_voted(fscores, fids))
        fold_data["haystacks"] = extended

    fit_final = _drop_voted(all_scores, ids).tolist()
    cut = fit_fold_anchored_cut(fold_haystacks, fold_orderings[:n_folds], fit_final) if fold_haystacks else None
    if cut is not None:
        anchored = cut.threshold_at(inclusion)
        if np.isfinite(anchored):
            return anchored, all_scores, ids, fold_haystacks, cut.provenance, cut
    blended = calculate_safe_threshold(_blend_xcal_input(threshold, details), all_scores, ctx, schedule=schedule)
    return blended, all_scores, ids, fold_haystacks, "gmm_blend", None


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
    ``cost``, ``fpr``, ``fnr``, ``precision``, ``recall`` and ``f1`` (all
    computed at *threshold*, the last three via
    :func:`~vtscore.eval.calibration_metrics.detection_metrics`) — plus the
    threshold-independent ranking metrics ``auroc`` and ``average_precision``,
    which isolate "how good is the ranking" from "how good is the threshold".

    When *region_aware* the test media carry a ``patch_grid`` (a patch
    embedder), so scoring max-pools the MLP over every score row of each image -
    exactly the live detector's inference for patch datasets (an image scores
    by its best-matching row).  Otherwise each media is scored by its single
    whole-image vector through the step's trainer-agnostic ``predict``.
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.eval.label_curve import _auroc, _average_precision  # noqa: PLC0415

    nan = float("nan")
    if not test_ids:
        return {
            "cost": nan,
            "fpr": nan,
            "fnr": nan,
            "precision": nan,
            "recall": nan,
            "f1": nan,
            "n_test_pos": nan,
            "n_test_neg": nan,
            "n_flagged": nan,
            "auroc": nan,
            "average_precision": nan,
        }

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

    maybe_dump_predictions(clips_dict, test_ids, scores, true_labels, threshold, target_category, suffix="__eval")

    fpr_weight, fnr_weight = _inclusion_weights(inclusion)
    cost = fpr_weight * fpr + fnr_weight * fnr

    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(true_labels, dtype=np.float64)
    from vtscore.eval.calibration_metrics import detection_metrics  # noqa: PLC0415

    det = detection_metrics(scores_arr, labels_arr, threshold)
    return {
        "cost": round(cost, 6),
        "fpr": round(fpr, 6),
        "fnr": round(fnr, 6),
        **{k: _r(v) for k, v in det.items()},
        "auroc": round(_auroc(scores_arr, labels_arr), 6),
        "average_precision": round(_average_precision(scores_arr, labels_arr), 6),
    }


#: ``fold_anchored[2/4]`` / ``fold_conformal_qmean[3/4]`` - *a* of *k*.
_PROVENANCE_USED_RE = re.compile(r"\[(\d+)/(\d+)\]$")


def _folds_used(provenance: str, k: int) -> float:
    """How many of the *k* folds contributed a cut, read off *provenance*.

    NaN for the arms where the question has no answer: the pooled conformal cut
    and the blend take one quantile over every fold's scores at once, so no fold
    ever "contributes a cut" that could be counted or dropped.  Reporting *k*
    there would look like agreement with the combining arms and hide exactly the
    asymmetry #3115 is about - a single-class fold is silently *in* the pool and
    explicitly *out* of a mean.
    """
    m = _PROVENANCE_USED_RE.search(provenance)
    return float(m.group(1)) if m else float("nan")


def _r(x: float) -> float:
    """Round to 6 dp when finite, else pass NaN/inf through unchanged."""
    import math  # noqa: PLC0415

    return round(x, 6) if math.isfinite(x) else x


#: Memo for :func:`_honest_oracle`, keyed on a digest of its exact inputs.
#: Bounded because the only reuse that matters is *within* a step, where every
#: variant row measures the same ``(base_scores, base_labels)`` at the same
#: inclusion: a step emits dozens of rows and the cross-fitted oracle is five
#: sorts, so recomputing it per row would be the dominant cost of the
#: decomposition.  Across steps the key changes and the old entry is dead, so a
#: handful of slots is all this ever needs.
_HONEST_ORACLE_MEMO: "dict[bytes, tuple[float, float]]" = {}
_HONEST_ORACLE_MEMO_MAX = 8


def _honest_oracle(scores: "np.ndarray", labels: "np.ndarray", wf: float, wn: float) -> tuple[float, float]:
    """Memoized :func:`~vtscore.eval.transfer_rules.honest_test_oracle`.

    Safe to memoize because the estimator is a pure function of its arguments -
    its own resampling is seeded from a digest of the score array rather than
    from global RNG state (see :func:`vtscore.eval.transfer_rules._rng`), so two
    calls on equal inputs return bit-identical results with or without the cache.
    The key is a digest of both arrays *and* the cost weights, because the same
    test scores are decomposed at more than one inclusion in a single run.
    """
    import hashlib  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    from vtscore.eval.transfer_rules import honest_test_oracle  # noqa: PLC0415

    s = np.ascontiguousarray(scores, dtype=np.float64)
    lb = np.ascontiguousarray(labels, dtype=np.float64)
    h = hashlib.blake2b(s.tobytes(), digest_size=16)
    h.update(lb.tobytes())
    h.update(np.asarray([wf, wn], dtype=np.float64).tobytes())
    key = h.digest()
    hit = _HONEST_ORACLE_MEMO.get(key)
    if hit is not None:
        return hit
    out = honest_test_oracle(s, lb, wf, wn)
    if len(_HONEST_ORACLE_MEMO) >= _HONEST_ORACLE_MEMO_MAX:
        _HONEST_ORACLE_MEMO.clear()
    _HONEST_ORACLE_MEMO[key] = out
    return out


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

    **Two reference points, because the naive one is optimistic** (#3116, #3248).
    ``oracle_cost`` is the minimum of the empirical cost over the very test
    sample it is then scored on - :func:`~vtscore.eval.calibration_metrics.oracle_cut`'s
    own docstring calls it a lower bound on achievable cost rather than a rule -
    so every gap measured against it is inflated by however much that minimum
    overfits.  #2883 measured that optimism directly and found it was the *whole*
    of the sibling ``transfer`` term (+0.041 naive, −0.001 cross-fitted).
    ``oracle_cost_honest`` is the same quantity cross-fitted
    (:func:`~vtscore.eval.transfer_rules.honest_test_oracle`: cut chosen on K−1
    folds, paid on the held-out one), and the pair **brackets** the population
    optimum rather than pinning it - naive from below, honest from above, since
    the honest cut sees only ``(K−1)/K`` of the sample.  ``regret`` and
    ``calibration_shift`` therefore ship beside ``regret_honest`` and
    ``calibration_shift_honest``; read the two as an interval, and prefer the
    honest one whenever a *level* rather than a paired contrast is being quoted.
    ``rule_inefficiency`` is untouched by the choice - it never references the
    test oracle - so both decompositions telescope exactly:

    * ``rule_inefficiency + calibration_shift        == regret``
    * ``rule_inefficiency + calibration_shift_honest == regret_honest``

    **The split's reference moves with anything that feeds ``cal_scores``**
    (#3116).  ``c_thr`` is estimated *from the calibration set*, so a study that
    sweeps a knob changing that set's size or content - ``calibrate_count`` is
    the case on record - is moving the yardstick it measures against.  As the
    calibration set grows, ``c_thr`` converges on the test-oracle cut, which
    shrinks ``calibration_shift`` and widens ``rule_inefficiency`` **from one
    cause, in opposite directions**, with their sum pinned to ``regret`` by
    construction.  #2897 read exactly that anti-correlation as a finding; it is
    algebra.  Do not report the two terms as independent effects of such a knob
    without a reference held fixed across the arms - and note that
    ``rule_inefficiency`` is a signed cost gap between two cuts, **not** a
    variance: it is routinely negative (the trained cut beating a
    calibration-set "oracle" that overfits a handful of scores), and a study
    asking whether the *threshold* got less variable wants ``sd(threshold)``
    across seeds, which ``analyze_folds_2897.py`` reports.
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.eval.calibration_metrics import (  # noqa: PLC0415
        detection_metrics,
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
    # The honest half of the bracket; NaN on a test sample too small or too
    # one-sided to cross-fit, which leaves the honest columns empty rather than
    # silently falling back to the optimistic reference they exist to correct.
    o_cost_honest, o_thr_honest = _honest_oracle(scores, labels, wf, wn)
    regret_honest = cost - o_cost_honest

    nan = float("nan")
    if cal_scores is not None and np.asarray(cal_scores).size > 0:
        cal_scores = np.asarray(cal_scores, dtype=np.float64)
        cal_labels = np.asarray(cal_labels, dtype=np.float64)
        c_thr, _, _, _ = oracle_cut(cal_scores, cal_labels, wf, wn)
        cal_oracle_cost, _, _ = operating_cost(scores, labels, c_thr, wf, wn)
        rule_inefficiency = cost - cal_oracle_cost
        calibration_shift = cal_oracle_cost - o_cost
        calibration_shift_honest = cal_oracle_cost - o_cost_honest
    else:
        c_thr = nan
        cal_oracle_cost = nan
        rule_inefficiency = nan
        calibration_shift = nan
        calibration_shift_honest = nan

    return {
        "pool_variant": pool_variant,
        # Safe-threshold study columns (issue #2799): defaults here; the base
        # row and the per-variant rows overwrite them where they apply.
        "gmm_variant": "",
        "schedule": "",
        "xcal_threshold": _r(float(threshold)),
        "gmm_cut": nan,
        "blend_weight": nan,
        # Fold-count study columns (issue #2897); only the fold-count arms set
        # them.  ``n_cal_scores`` is the pooled calibration-set size the
        # conformal quantile is taken over, which is what K actually buys.
        "fold_count": nan,
        "fold_seconds": nan,
        "n_cal_scores": nan,
        # #3115: how many of the K folds actually contributed a cut, parsed off
        # the arm's own provenance.  A combine rule and a pooled quantile weight
        # a degenerate (single-class) fold completely differently, so a contrast
        # between them is only readable next to the count of folds that were
        # dropped rather than averaged.
        "n_folds_used": nan,
        # Cut-rule study columns (issue #2836); only the variant rows set them.
        # ``cut_fallback_kind`` says *what was substituted* where ``cut_fallback``
        # only says *that* something was, which the two emitting families answer
        # differently on the same fits (issue #2900).
        "cut_fallback": 0,
        "cut_fallback_kind": CUT_KIND_INTERIOR,
        "cut_fail_reason": "",
        "raw_cut_cost": nan,
        "raw_cut_fpr": nan,
        "raw_cut_fnr": nan,
        "threshold": _r(float(threshold)),
        "threshold_provenance": provenance,
        "degenerate": 1 if is_degenerate(scores, threshold) else 0,
        "threshold_percentile": _r(threshold_percentile(scores, threshold)),
        "cost": _r(cost),
        "fpr": _r(fpr),
        "fnr": _r(fnr),
        **{k: _r(v) for k, v in detection_metrics(scores, labels, threshold).items()},
        "auroc": _r(float(_auroc(scores, labels))),
        "average_precision": _r(float(_average_precision(scores, labels))),
        "oracle_threshold": _r(float(o_thr)),
        "oracle_cost": _r(o_cost),
        "oracle_fpr": _r(o_fpr),
        "oracle_fnr": _r(o_fnr),
        "regret": _r(regret),
        # The cross-fitted reference and the two terms it re-bases (#3116).
        # Bracket, not replacement: `oracle_cost` bounds the population optimum
        # from below and `oracle_cost_honest` from above, so a level quoted from
        # either alone is one end of an interval.
        "oracle_threshold_honest": _r(float(o_thr_honest)),
        "oracle_cost_honest": _r(o_cost_honest),
        "regret_honest": _r(regret_honest),
        "cal_oracle_threshold": _r(float(c_thr)),
        "cal_oracle_cost": _r(cal_oracle_cost),
        "rule_inefficiency": _r(rule_inefficiency),
        "calibration_shift": _r(calibration_shift),
        "calibration_shift_honest": _r(calibration_shift_honest),
        "n_pool_rows": _r(float(n_pool_rows)),
    }


#: Safe-threshold cut variants (issues #2799, #2836): ``(name, fit_scores, rule)``.
#: ``fit_scores`` picks which sim-set score distribution the mixture is fitted on
#: ("pooled" = the style's inference max-pool, what production fits post-#2797;
#: "image" = the whole-image vector scores, the historical pre-#2797 geometry).
#: ``rule`` names a cut in :mod:`vtscore.eval.cut_rules` - the ``lam``-tilt
#: family over the Gaussian mixture ("mid" is production; "cross" is #2798's
#: count-optimal crossing, reverted by #2833; "priorfree"/"rate" are #2836's
#: rate-optimal tilts), the same tilts over a Gumbel-low mixture ("gumbel_*") and
#: over one whose Gumbel may land on either mode ("gumbel_any_*", #2846's repair
#: to the first family's fallback rate), and the two label-reading diagnostics
#: ("supervised", "sim_oracle") that locate the error rather than compete to
#: ship.  ``xcal_only`` is the no-blend control: the conformal threshold at the
#: same step, or - on a step whose folds fell back - the same sentinel the
#: shipped blend feeds its x-cal side (:func:`_blend_xcal_input`), so the
#: control is the blend's own input with the mix-in removed rather than a cut
#: nobody computed.  ``pooled_mid`` must reproduce the production blend exactly.
#: The ``tail_a*`` sweep is #2881's one-constant rule at seven tail levels - the
#: fitted Bad component's own quantile rather than a crossing of any kind.
#:
#: **The rule names here are the ones in :mod:`vtscore.eval.cut_rules`**, spelled
#: out rather than derived, because this module is deliberately import-light (no
#: numpy at import time) and importing the rule tables to build the list would
#: undo that.  ``test_cut_rules`` asserts the two agree, so the duplication
#: cannot drift silently - which matters more than usual here, since a rule that
#: is defined but never emitted produces a table with a missing row rather than
#: an error.
#:
#: The #2798 logit-space variants are gone: #2799 measured them at +0.0006 cost
#: (dead) and each extra fit costs a step's CPU that the #2836 arms need.
_SAFE_GMM_VARIANTS: tuple[tuple[str, str, str], ...] = (
    ("xcal_only", "", ""),
    ("image_mid", "image", "mid"),
    ("image_cross", "image", "cross"),
    ("image_priorfree", "image", "priorfree"),
    ("image_rate", "image", "rate"),
    ("pooled_mid", "pooled", "mid"),
    ("pooled_cross", "pooled", "cross"),
    ("pooled_priorfree", "pooled", "priorfree"),
    ("pooled_rate", "pooled", "rate"),
    ("pooled_gumbel_cross", "pooled", "gumbel_cross"),
    ("pooled_gumbel_priorfree", "pooled", "gumbel_priorfree"),
    ("pooled_gumbel_rate", "pooled", "gumbel_rate"),
    ("pooled_gumbel_any_cross", "pooled", "gumbel_any_cross"),
    ("pooled_gumbel_any_priorfree", "pooled", "gumbel_any_priorfree"),
    ("pooled_gumbel_any_rate", "pooled", "gumbel_any_rate"),
    ("pooled_tail_a040", "pooled", "tail_a040"),
    ("pooled_tail_a080", "pooled", "tail_a080"),
    ("pooled_tail_a110", "pooled", "tail_a110"),
    ("pooled_tail_a158", "pooled", "tail_a158"),
    ("pooled_tail_a220", "pooled", "tail_a220"),
    ("pooled_tail_a300", "pooled", "tail_a300"),
    ("pooled_tail_a400", "pooled", "tail_a400"),
    ("pooled_supervised", "pooled", "supervised"),
    ("pooled_sim_oracle", "pooled", "sim_oracle"),
    # #2883: the last link's shape.  Four subsample levels give the learning
    # curve in sim-set size (the test set and the trajectory are identical
    # across them - only the number of labelled sim scores moves), and two
    # variance-reduced estimators of the same target test whether the empirical
    # minimiser is the bound `family_headroom_exhausted` treats it as.
    ("pooled_sim_oracle_f050", "pooled", "sim_oracle_f050"),
    ("pooled_sim_oracle_f100", "pooled", "sim_oracle_f100"),
    ("pooled_sim_oracle_f250", "pooled", "sim_oracle_f250"),
    ("pooled_sim_oracle_f500", "pooled", "sim_oracle_f500"),
    ("pooled_sim_oracle_bag", "pooled", "sim_oracle_bag"),
    ("pooled_sim_oracle_smooth", "pooled", "sim_oracle_smooth"),
    # The label-free counterpart: bag the mixture fit rather than the labelled
    # cost curve.  Exploratory, not ship-gated - see PREREG.
    ("pooled_bagfit_mid", "pooled", "bagfit_mid"),
    ("pooled_bagfit_priorfree", "pooled", "bagfit_priorfree"),
)

#: Variants that read the sim set's true labels.  Reported for the decomposition,
#: never eligible to ship - a rule cannot see these labels in the app.
_ORACLE_VARIANTS: frozenset[str] = frozenset(
    {
        "pooled_supervised",
        "pooled_sim_oracle",
        # #2883's readings of the same sim set - label-reading for the same
        # reason and, like the two above, emitting NaN rather than falling back
        # to a midpoint under another rule's name.
        "pooled_sim_oracle_f050",
        "pooled_sim_oracle_f100",
        "pooled_sim_oracle_f250",
        "pooled_sim_oracle_f500",
        "pooled_sim_oracle_bag",
        "pooled_sim_oracle_smooth",
    }
)


def _safe_gmm_variant_rows(
    details: dict[str, Any],
    base_scores: "np.ndarray",
    base_labels: "np.ndarray",
    sim_scores_by_geometry: dict[str, list[float]],
    sim_labels_by_geometry: dict[str, "np.ndarray"],
    inclusion: int,
    n_pool_rows: float,
    schedule: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Metric rows per cut variant, plus the per-geometry decomposition rows.

    Every variant re-cuts the *same* per-step model: each candidate sim-set score
    distribution is fitted once (both mixture families), every cut rule reads that
    one fit, and each cut is blended with the step's conformal threshold on the
    production label ramp.  All variants are evaluated against the same held-out
    test scores (*base_scores*, the inference max-pool), so the rows are paired
    within a step by construction.

    Two costs are recorded per variant: ``cost`` at the *blended* threshold (what
    a user would get, and what the ship decision reads) and ``raw_cut_cost`` at
    the unblended cut (what the *rule* is worth, undamped by the conformal
    threshold it is averaged with).  On the ramp the blend can shrink a large cut
    difference to a small cost difference, so the rule comparison belongs on the
    raw column and the ship comparison on the blended one.

    A rule whose root does not exist on a given fit falls back to that fit's
    midpoint and is flagged in ``cut_fallback`` so the analyzer can exclude
    fallen-back steps from a rule's own contrast rather than silently scoring
    the midpoint under another name.  **The midpoint is this family's fallback,
    not production's** - the shipped ``rate`` rule
    (:func:`~vtscore.training.thresholds.gmm_cut_from_fit`) continues past the
    inter-mean interval at its own first-order slope instead, so on the fits
    where this flag fires these arms are measuring a different rule than the
    app runs.  That divergence is **kept on purpose** (issue #2900): this family
    compares tilts against each other on one fit, and a rule-independent
    stand-in is what keeps ``rate`` commensurable with the ``cross`` and
    ``priorfree`` siblings it is differenced against - at inclusion 0 it is what
    keeps ``rate`` bit-identical to ``priorfree``, which is how every report in
    ``docs/experiments/gmm-cut/`` reads those rows.  It is no longer *invisible*
    though: ``cut_fallback_kind`` carries
    :data:`~vtscore.eval.cut_rules.CUT_KIND_MIDPOINT` on exactly these steps,
    against the production family's ``continued`` / ``degenerate_midpoint``, so
    an analysis that wants the shipped path can filter for it instead of reading
    a substituted midpoint as "what the app would have done".  The fold-anchored
    family below calls the production function directly and so does not have
    that gap at all.  For the EVT rules
    ``cut_fail_reason`` additionally names *which* guard declined, because the
    repairs those guards want are different and the counts alone cannot tell them
    apart (issue #2846).  The oracle variants do not fall back; they emit NaN cuts
    and are dropped by the analyzer's joins.

    Returns ``(variant_rows, diagnostic_rows)``; the diagnostic rows carry the
    fitted mixture parameters and every cut in the decomposition chain, one row
    per (step, geometry), and still need the caller's identifying columns.
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.eval.calibration_metrics import inclusion_weights, operating_cost, oracle_cut  # noqa: PLC0415
    from vtscore.eval.cut_rules import CUT_KIND_MIDPOINT, decomposition_cuts  # noqa: PLC0415
    from vtscore.eval.transfer_rules import honest_test_oracle  # noqa: PLC0415
    from vtscore.training.thresholds import blend_gmm_threshold, safe_blend_weight  # noqa: PLC0415

    xcal = float(details["xcal_threshold"])
    n_votes = int(details["n_votes"])
    pre_blend_provenance = str(details.get("pre_blend_provenance", "conformal"))
    fold_orderings = details.get("fold_orderings") or []
    cal_scores = np.array([s for scores, _ in fold_orderings for s in scores]) if fold_orderings else None
    cal_labels = np.array([lb for _, labels_ in fold_orderings for lb in labels_]) if fold_orderings else None
    wf, wn = inclusion_weights(inclusion)
    nan = float("nan")

    # One fit pass per geometry; every rule below reads these.
    cuts_by_geometry: dict[str, dict[str, float]] = {}
    reasons_by_geometry: dict[str, dict[str, str]] = {}
    diag_rows: list[dict[str, Any]] = []
    geometries = sorted({fit for _n, fit, _r in _SAFE_GMM_VARIANTS if fit})
    for geometry in geometries:
        scores = sim_scores_by_geometry[geometry]
        labels = sim_labels_by_geometry[geometry]
        if len(scores) < 2:
            # Mirrors calculate_gmm_threshold's "too few scores" default so the
            # production-blend sanity check holds at every step.
            cuts_by_geometry[geometry] = dict.fromkeys((r for _n, f, r in _SAFE_GMM_VARIANTS if f == geometry), 0.5)
            reasons_by_geometry[geometry] = {}
            continue
        cuts, params, reasons = decomposition_cuts(scores, labels, wf, wn)
        cuts_by_geometry[geometry] = cuts
        reasons_by_geometry[geometry] = reasons
        diag = {
            "geometry": geometry,
            "sim_prevalence": _r(float(np.mean(labels))) if len(labels) else nan,
            # The count, not just the rate: a threshold estimated from labelled
            # scores is limited by the *rarer* class, so #2883's scaling claim is
            # about positives and prevalence alone cannot express it.
            "sim_n_pos": _r(float(np.sum(labels == 1.0))) if len(labels) else nan,
        }
        # ``evt_fit_fail`` is a reason string; everything else in params is numeric.
        diag.update({k: v if isinstance(v, str) else _r(float(v)) for k, v in params.items()})
        diag.update({f"tau_{name}": _r(float(value)) for name, value in cuts.items()})
        diag["tau_test_oracle"] = nan  # filled below, once the test oracle is known
        diag_rows.append(diag)

    ctx = BlendContext(
        n_labels=n_votes,
        n_good=int(details.get("n_good", 0)),
        n_bad=int(details.get("n_bad", n_votes)),
    )
    weight = safe_blend_weight(ctx, schedule)
    rows: list[dict[str, Any]] = []
    for name, geometry, rule in _SAFE_GMM_VARIANTS:
        fallback = 0
        fallback_kind = CUT_KIND_INTERIOR
        fail_reason = ""
        if name == "xcal_only":
            threshold = xcal
            gmm_cut = nan
            provenance = pre_blend_provenance
        else:
            gmm_cut = cuts_by_geometry[geometry][rule]
            if not np.isfinite(gmm_cut) and name not in _ORACLE_VARIANTS:
                gmm_cut = cuts_by_geometry[geometry].get("mid", nan)
                fallback = 1
                fallback_kind = CUT_KIND_MIDPOINT
                # Empty for the Gaussian rules, which have no reason vocabulary;
                # the EVT rules name the guard that declined so a fallback can be
                # attributed rather than merely counted (issue #2846).
                fail_reason = reasons_by_geometry[geometry].get(rule, "")
            threshold = blend_gmm_threshold(xcal, gmm_cut, ctx, schedule=schedule) if np.isfinite(gmm_cut) else nan
            provenance = "gmm_blend"
        if not np.isfinite(threshold):
            continue
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
        row["cut_fallback"] = fallback
        row["cut_fallback_kind"] = fallback_kind
        row["cut_fail_reason"] = fail_reason
        if np.isfinite(gmm_cut):
            raw_cost, raw_fpr, raw_fnr = operating_cost(base_scores, base_labels, gmm_cut, wf, wn)
            row["raw_cut_cost"] = _r(raw_cost)
            row["raw_cut_fpr"] = _r(raw_fpr)
            row["raw_cut_fnr"] = _r(raw_fnr)
        rows.append(row)

    # The last link in the chain: the best cut on the held-out test set.  Read off
    # any emitted row (all share the same base_scores/base_labels oracle).
    if rows and diag_rows:
        # #2883: that cut is the argmin of the empirical cost on the test sample
        # *itself*, so its cost is a sample minimum - biased low, which biases
        # `transfer` high by however much the reference overfits.  Record the
        # cross-fitted version beside it (cut and cost on disjoint folds) so the
        # last link can be reported as a bracket instead of a point.
        honest_cost, honest_tau = honest_test_oracle(base_scores, base_labels, wf, wn)
        # From `oracle_cut` itself, NOT by re-scoring at `rows[0]["oracle_threshold"]`:
        # that column is rounded on the way out, and re-evaluating a cost at a
        # rounded threshold moves items across the boundary.  With ~55 test
        # positives one FNR step is 1/55 = 0.018 - half the size of the term this
        # study is measuring - so the rounding is not a rounding error here.
        _naive_tau, naive_cost, _nfpr, _nfnr = oracle_cut(base_scores, base_labels, wf, wn)
        n_test = float(np.asarray(base_labels).size)
        n_test_pos = float(np.sum(np.asarray(base_labels) == 1.0))
        for diag in diag_rows:
            diag["tau_test_oracle"] = rows[0]["oracle_threshold"]
            diag["tau_test_oracle_honest"] = _r(honest_tau)
            diag["cost_test_oracle_naive"] = _r(naive_cost)
            diag["cost_test_oracle_honest"] = _r(honest_cost)
            diag["test_n"] = _r(n_test)
            diag["test_n_pos"] = _r(n_test_pos)
    return rows, diag_rows


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


def _fold_count_arms(
    prefix: list[tuple[list[float], list[float]]],
    xcal: float,
    inclusion: int,
    haystacks: "list[np.ndarray] | None",
    final_fit_scores: list[float] | None,
    ctx: BlendContext | None,
    gmm_cut: float | None,
    gmm_fit: Any,
    schedule: str | None,
) -> list[tuple[str, float, str, float]]:
    """``(arm, threshold, provenance, blend weight)`` for one fold prefix.

    Split out of :func:`_fold_count_variant_rows` so the arm table is one
    readable list rather than a branch pile inside the K loop; every arm here is
    a different *rule* applied to the **same** already-trained folds, so adding
    one costs arithmetic and no fits.

    ``haystacks`` is ``None`` when the step does not carry one sim-set score
    array per fold in the prefix, which gates every arm that has to read a cut
    in the fold's own distribution (:func:`~vtscore.training.thresholds.rank_transfer`).
    ``final_fit_scores`` is the final model's haystack the quantile-realizing
    arms cut on - under the #3308 convention the caller passes it with the
    voted items already dropped, matching the fold ``haystacks``; the ``blend``
    arm's GMM inputs (*gmm_cut*, *gmm_fit*) are fitted upstream on the full
    distribution, as production's fallback is.
    """
    import dataclasses  # noqa: PLC0415

    from vtscore.training.thresholds import (  # noqa: PLC0415
        FOLD_CONFORMAL_COMBINES,
        blend_gmm_threshold,
        combined_fold_conformal_threshold,
        fit_fold_anchored_cut,
        fold_anchored_gmm_threshold,
        safe_blend_weight,
    )

    nan = float("nan")
    arms: list[tuple[str, float, str, float]] = [("xcal", xcal, "conformal", nan)]
    if ctx is not None and gmm_cut is not None:
        weight = safe_blend_weight(ctx, schedule)
        blended = blend_gmm_threshold(xcal, gmm_cut, ctx, schedule=schedule, fit=gmm_fit)
        arms.append(("blend", blended, "gmm_blend" if weight < 1.0 else "conformal", weight))

    # #3115, the combine rule.  ``xcal`` above IS the pooled control - it calls
    # `threshold_from_fold_orderings` verbatim - so these are challengers rather
    # than a re-emission of it.  The score-space pair needs nothing the pooled
    # arm does not already have; the quantile-space pair needs a haystack per
    # fold, the same condition the anchored arms carry.
    #
    # `transferable` binds that condition once - "can this arm read a cut in a
    # fold's own scale?" - so it is asked in one place instead of re-derived at
    # each use site, and so both `None` cases narrow for the type checker.
    final_scores = final_fit_scores if final_fit_scores else None
    transferable = haystacks if (haystacks is not None and final_scores is not None) else None
    for combine in FOLD_CONFORMAL_COMBINES:
        quantile_space = combine.startswith("q")
        if quantile_space and transferable is None:
            continue
        value, prov = combined_fold_conformal_threshold(
            prefix,
            inclusion,
            combine=combine,
            fold_haystacks=transferable if quantile_space else None,
            final_scores=final_scores if quantile_space else None,
        )
        arms.append((combine, value, prov, nan))

    # The arms for the rule users actually get.  ``anchored`` is production
    # (`FOLD_ANCHOR_COMBINE`); ``anchored_qmedian`` re-cuts the *same* fit under
    # the robust combine, which puts #3115's contamination question on the
    # shipped path rather than only on the retired blend.
    #
    # Fitted **once** and re-cut, not fitted twice.  `FoldAnchoredCut` exists to
    # separate the fit from the cut, and the combine rule is read at cut time, so
    # a second `fold_anchored_gmm_threshold` call would re-run one anchored EM
    # per fold to reach the same mixtures - doubling the study's dominant cost
    # (sum over the K grid, so 52 EM fits per step at K<=16, not 16).  It is also
    # the stronger contrast: the two rows differ in the combine and in *nothing
    # else*, including the fits' own numerical noise.
    if transferable is not None and final_scores is not None:
        cut = fit_fold_anchored_cut(transferable, prefix, final_scores)
        if cut is None:
            # Both arms land on the same terminal fallback; take it from the
            # shipped helper rather than duplicating its ladder here.
            value, prov = fold_anchored_gmm_threshold(transferable, prefix, final_scores, inclusion)
            arms.extend([("anchored", value, prov, nan), ("anchored_qmedian", value, prov, nan)])
        else:
            arms.append(("anchored", cut.threshold_at(inclusion), cut.provenance, nan))
            robust = dataclasses.replace(cut, combine="qmedian")
            arms.append(("anchored_qmedian", robust.threshold_at(inclusion), robust.provenance, nan))
    return arms


def _fold_count_variant_rows(
    details: dict[str, Any],
    base_scores: "np.ndarray",
    base_labels: "np.ndarray",
    inclusion: int,
    n_pool_rows: float,
    counts: list[int],
    sim_pooled_scores: list[float] | None,
    schedule: str | None,
    sim_fit_scores: list[float] | None = None,
) -> list[dict[str, Any]]:
    """One metric row per calibration **fold count** K (issue #2897).

    *sim_fit_scores* is the sim haystack under the #3308 population convention
    (voted items dropped); the ``anchored`` / quantile-space arms realize their
    cuts on it, mirroring the shipped rule, while the retired ``blend`` arm
    keeps fitting *sim_pooled_scores* - production's fallback blend also keeps
    the full distribution.  ``None`` falls back to *sim_pooled_scores*.

    The study's screen for "does more cross-calibration buy anything, and what
    does it cost".  It is exact rather than approximate, because the folds are
    *nested*: :func:`~vtscore.training.thresholds.compute_fold_orderings` draws
    each fold as an independent stratified split off one ``RandomState(42)``
    stream, at a per-fold size that does not depend on the count, so the K folds
    a live ``calibrate_count=K`` run would train are byte-for-byte the first K
    of the Kmax folds trained here.  Slicing the prefix therefore reproduces
    each K's threshold exactly, and the arm at ``K == calibrate_count``
    reproduces this step's own pre-blend conformal cut - the control that
    licenses the rest of the table.

    Arms per K, because the fold count and the shipped threshold are different
    questions - and, since #3115, because *how* the folds are combined is a
    third one.  Every arm below re-reads the same already-trained fold prefix,
    so the whole table costs arithmetic:

    * ``folds_k{K}_xcal`` - the raw cross-calibration cut, the thing K is
      actually a knob on.
    * ``folds_k{K}_blend`` - that cut after the ``cap50`` safe-threshold mix-in.
      The blend weight depends only on the vote counts, so it is identical
      across K and this arm isolates how much of K's benefit survives being
      averaged with the GMM cut.  Emitted only when the step has the pooled sim
      scores the blend fits.
    * ``folds_k{K}_anchored`` - **production's rule** (#3116).  The blend above
      was retired by the 2026-08-05 population-anchored run; the shipped path is
      :func:`~vtscore.training.thresholds.fold_anchored_gmm_threshold`, which
      fits one anchored mixture *per fold* and combines them in quantile space.
      K therefore moves the shipped threshold through a path the other two arms
      do not exercise at all - the blend's GMM half is a single unanchored fit
      on the sim haystack and is K-independent by construction, so ``blend``
      varies only in its x-cal half.  Without this arm every conclusion about
      what ``calibrate_count`` does to the threshold users actually get is
      partial.  Emitted when the step carries at least K fold haystacks, which
      :func:`_safe_threshold_for_step` supplies (so: under safe thresholds).
    * ``folds_k{K}_{tmean,tmedian,qmean,qmedian}`` - **the combine rule**
      (#3115).  ``xcal`` above *is* the pooled control: it calls
      :func:`~vtscore.training.thresholds.threshold_from_fold_orderings`
      verbatim, which pools every fold's held-out scores into one bag and takes
      a single conformal quantile.  These four take one cut per fold and combine
      them instead, in score space (``t*``) or in fold-quantile space (``q*``);
      see :data:`~vtscore.training.thresholds.FOLD_CONFORMAL_COMBINES` for what
      each leg of the contrast isolates.  ``q*`` needs a haystack per fold and
      so shares the ``anchored`` arm's condition.
    * ``folds_k{K}_anchored_qmedian`` - production's rule re-cut under the
      robust combine, which puts the contamination question on the **shipped**
      path and not only on the retired blend.

    Note what is *not* a defect here: the unanchored ``fit_gmm_threshold`` is
    hoisted out of the K loop, and that is correct rather than a frozen
    reference - it reads only ``sim_pooled_scores``, which no fold count
    touches, so re-fitting it per K would return the same cut at K times the
    price.  The gap #3116 identified is the missing arm above, not the hoist.

    ``fold_seconds`` is the calibration wall clock this K would have cost: the
    measured fit time of its own folds plus the count-independent overhead of
    the threshold rule.  It is measured inside the Kmax run, so every K's timing
    shares one machine, one process and one cache state - the *ratios* are the
    load-bearing part, not the absolute seconds.

    This is the study's screen, not its verdict, for the usual reason (see
    :func:`_schedule_variant_rows`): K also steers acquisition through the
    threshold Autopilot's Hard pick ranks around, and a screen that holds the
    trajectory fixed cannot see the votes a different K would have collected.
    The live A/B runs exist to measure exactly that.
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.training.thresholds import (  # noqa: PLC0415
        fit_gmm_threshold,
        threshold_from_fold_orderings,
    )

    fold_data = details.get("fold_count_data")
    if not fold_data:
        return []
    orderings = fold_data["orderings"]
    seconds = fold_data["seconds"]
    overhead = float(fold_data.get("overhead_seconds") or 0.0)
    haystacks = fold_data.get("haystacks") or []
    if not orderings:
        return []

    ctx = None
    gmm_cut = gmm_fit = None
    if sim_pooled_scores:
        ctx = BlendContext(
            n_labels=int(details["n_votes"]),
            n_good=int(details["n_good"]),
            n_bad=int(details["n_bad"]),
        )
        gmm_cut, gmm_fit = fit_gmm_threshold(sim_pooled_scores)

    rows: list[dict[str, Any]] = []
    for k in counts:
        if k < 1 or k > len(orderings):
            continue
        prefix = orderings[:k]
        cal_scores = np.array([s for scores, _ in prefix for s in scores])
        cal_labels = np.array([lb for _, labels_ in prefix for lb in labels_])
        xcal = threshold_from_fold_orderings(prefix, inclusion)
        fold_seconds = _r(float(sum(seconds[:k])) + overhead)

        arms = _fold_count_arms(
            prefix,
            xcal,
            inclusion,
            haystacks[:k] if len(haystacks) >= k else None,
            sim_fit_scores if sim_fit_scores is not None else sim_pooled_scores,
            ctx,
            gmm_cut,
            gmm_fit,
            schedule,
        )

        for arm, threshold, provenance, weight in arms:
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
            row["gmm_variant"] = f"folds_k{k}_{arm}"
            row["schedule"] = schedule or ""
            row["xcal_threshold"] = _r(xcal)
            row["gmm_cut"] = _r(gmm_cut) if gmm_cut is not None else float("nan")
            row["blend_weight"] = _r(weight)
            row["fold_count"] = k
            row["fold_seconds"] = fold_seconds
            row["n_cal_scores"] = int(cal_scores.size)
            row["n_folds_used"] = _folds_used(provenance, k)
            rows.append(row)
    return rows


#: Default sweep grid for the anchored-mixture eval arms (issue #2852).  Each
#: (anchor_weight, cut rule[, fold combine]) combination is one paired
#: within-step variant; the GRID run overrides these via
#: ``simulate_voting_iterations``'s ``anchored_*`` parameters to exhaust the
#: grid registered in ``docs/plans/population-anchored-calibration.md``.
_ANCHORED_WEIGHTS: tuple[float, ...] = (1.0, 10.0, 100.0)
_ANCHORED_RULES: tuple[str, ...] = ("mid", "rate")
_ANCHORED_FOLD_COMBINES: tuple[str, ...] = ("qmean",)


def _anchored_variant_rows(
    details: dict[str, Any],
    base_scores: "np.ndarray",
    base_labels: "np.ndarray",
    sim_scores: list[float],
    sim_ids: list[int],
    good_ids: list[int],
    bad_ids: list[int],
    fold_haystacks: list,
    inclusion: int,
    n_pool_rows: float,
    weights: list[float],
    rules: list[str],
    fold_combines: list[str],
    fold_anchored: bool,
) -> list[dict[str, Any]]:
    """Metric rows for the anchored-mixture threshold arms (issue #2852).

    Three families, all evaluated against the same held-out test scores as the
    #2799 variants so every row is step-paired with the shipped blend
    (``pooled_mid``) and pure x-cal (``xcal_only``):

    * ``anchored_w{W}_{rule}`` - the **label-anchored** mixture: anchored EM on
      the final model's sim-set (haystack) scores with the voted items' own
      final-model scores clamped to their labelled component.  One EM per
      anchor weight; each cut rule re-cuts the same fit.
    * ``fold_anchored_w{W}_{rule}_{combine}`` - the **fold-anchored**
      ("cross-LabeledGMM") repair: per calibration fold, anchored EM on that
      fold model's haystack scores with that fold's held-out labelled scores
      as anchors (honest anchors, one shared scale), each fold's cut carried
      to the final model by rank transfer and combined in quantile space.
    * ``rank_transfer`` - the conformal x-cal cut carried from the pooled fold
      haystack distribution to the final model's as a quantile: the
      scale-transfer-only arm that attributes H1 (see the plan).

    Anchored thresholds are used **raw** - the estimator replaces the blend
    rather than feeding it - so ``blend_weight`` is NaN and ``raw_cut_*``
    equals the headline cost columns.  The estimator path actually taken
    (anchored / unanchored fallback / fold tally) is recorded in
    ``threshold_provenance``.

    *fold_haystacks* are the per-fold sim-set score arrays the shipped
    threshold already computed (:func:`_safe_threshold_for_step`); the fold
    family re-cuts them rather than paying the scoring passes twice.  The grid
    point at the production ``(κ, rule, combine)`` therefore reproduces the
    step's own shipped cut exactly - the grid's *other* points are the
    deviation under test.
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.eval.calibration_metrics import inclusion_weights, operating_cost  # noqa: PLC0415

    xcal = float(details["xcal_threshold"])
    fold_orderings = details.get("fold_orderings") or []
    cal_scores = np.array([s for scores, _ in fold_orderings for s in scores]) if fold_orderings else None
    cal_labels = np.array([lb for _, labels_ in fold_orderings for lb in labels_]) if fold_orderings else None
    wf, wn = inclusion_weights(inclusion)
    nan = float("nan")

    score_by_id = dict(zip(sim_ids, sim_scores, strict=True))
    anchor_scores = [score_by_id[cid] for cid in (*good_ids, *bad_ids) if cid in score_by_id]
    anchor_labels = [1.0] * sum(1 for cid in good_ids if cid in score_by_id) + [0.0] * sum(
        1 for cid in bad_ids if cid in score_by_id
    )
    # The #3308 population convention: the voted items anchor the fits, so they
    # are dropped from the free haystack sample instead of sitting in it twice
    # (once free, once clamped) - matching the shipped cut and the (already
    # filtered) *fold_haystacks* the fold family re-cuts.
    voted = set(good_ids) | set(bad_ids)
    final_scores = np.asarray(
        [s for i, s in zip(sim_ids, sim_scores, strict=True) if i not in voted], dtype=np.float64
    )

    rows: list[dict[str, Any]] = []

    def emit(name: str, threshold: float, provenance: str, cut_kind: str) -> None:
        """Record one anchored arm.  *cut_kind* is the production rule's own
        ``cut_fallback_kind`` (:func:`~vtscore.training.thresholds.gmm_cut_from_fit`),
        so these rows say ``continued`` / ``degenerate_midpoint`` where the
        decomposition family says ``midpoint`` - the two substitute different
        values on the same fits (issue #2900).
        """
        if not np.isfinite(threshold):
            return
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
        row["xcal_threshold"] = _r(xcal)
        row["gmm_cut"] = _r(threshold)
        row["blend_weight"] = nan
        row["cut_fallback"] = int(bool(cut_kind))
        row["cut_fallback_kind"] = cut_kind
        raw_cost, raw_fpr, raw_fnr = operating_cost(base_scores, base_labels, threshold, wf, wn)
        row["raw_cut_cost"] = _r(raw_cost)
        row["raw_cut_fpr"] = _r(raw_fpr)
        row["raw_cut_fnr"] = _r(raw_fnr)
        rows.append(row)

    # --- Label-anchored family: one anchored EM per weight, re-cut per rule. ---
    for weight in weights:
        fit, provenance = anchored_gmm_fit(final_scores, anchor_scores, anchor_labels, anchor_weight=weight)
        if fit is None:
            continue
        for rule in rules:
            if rule in ("mid_tilt", "q_tilt"):
                # Fold-level rules, defined in fold-quantile space: a single
                # label-anchored fit has no folds to tilt across.  The fold
                # family below sweeps them; here they are skipped rather than
                # fed to gmm_cut_from_fit, which (correctly) rejects them.
                continue
            cut, cut_kind = gmm_cut_from_fit(fit, rule, wf, wn)
            emit(f"anchored_w{weight:g}_{rule}", cut, provenance, cut_kind)

    # --- Fold-anchored family + the rank-transfer attribution arm. ---
    if fold_anchored and fold_haystacks and fold_orderings:
        _emit_fold_anchored_rows(
            emit,
            xcal,
            fold_haystacks,
            fold_orderings,
            final_scores,
            inclusion,
            weights,
            rules,
            fold_combines,
        )

    return rows


def _emit_fold_anchored_rows(
    emit: Callable[[str, float, str, str], None],
    xcal: float,
    fold_haystacks: list,
    fold_orderings: list[tuple[list[float], list[float]]],
    final_scores: "np.ndarray",
    inclusion: int,
    weights: list[float],
    rules: list[str],
    fold_combines: list[str],
) -> None:
    """Emit the fold-family arm rows over pre-computed per-fold sim scores.

    The scoring pass per fold model is the fold arms' whole marginal cost, and
    the shipped threshold already paid it (see :func:`_safe_threshold_for_step`),
    so the grid re-cuts those arrays rather than re-scoring per grid point.
    ``rank_transfer`` reuses them too: the conformal cut carried from the pooled
    fold haystack distribution to the final model's - the scale-transfer-only
    attribution arm of the plan.

    Every grid point goes through the same
    :func:`~vtscore.training.thresholds.fold_anchored_gmm_threshold` the app
    ships; the grid *is* the deviation under test, so the arm at the production
    (κ, rule, combine) reproduces the shipped cut exactly.

    These rows carry no ``cut_fallback_kind``: a fold-anchored threshold is
    composed from one per-fit cut *per fold* in quantile space, so there is no
    single fit whose fallback branch the row could name.  A per-fold breakdown
    would be a different column with a different unit of observation, and the
    ``mid_tilt`` rule the app ships already degrades a rate-less fold to plain
    ``mid`` rather than to a substituted value (see
    :meth:`~vtscore.training.thresholds.FoldAnchoredCut._quantile_at`).
    """
    import numpy as np  # noqa: PLC0415

    n_folds = min(len(fold_haystacks), len(fold_orderings))
    fold_hay, orderings = fold_haystacks[:n_folds], fold_orderings[:n_folds]

    emit(
        "rank_transfer",
        rank_transfer(xcal, np.concatenate(fold_hay), final_scores),
        "rank_transfer",
        CUT_KIND_INTERIOR,
    )
    for weight in weights:
        for rule in rules:
            for combine in fold_combines:
                threshold, provenance = fold_anchored_gmm_threshold(
                    fold_hay,
                    orderings,
                    final_scores,
                    inclusion,
                    anchor_weight=weight,
                    cut_rule=rule,
                    combine=combine,
                )
                emit(f"fold_anchored_w{weight:g}_{rule}_{combine}", threshold, provenance, CUT_KIND_INTERIOR)


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
    # dump: calibration path -- `ids` is aligned with base_scores and labels.
    maybe_dump_predictions(clips_dict, list(ids), base_scores, list(labels), threshold, target_category)
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
    # How many held-out scores the conformal quantile was actually taken over,
    # on the SHIPPED row rather than only on the fold-count variant rows (issue
    # #3287).  It was declared in `_CALIBRATION_COLUMNS` and filled only by the
    # #2897 arms, so the one quantity `calibration_fraction` directly controls -
    # the resolution of the quantile the threshold is read from - was NaN on
    # every production row.  A knob whose mechanism is invisible in the output
    # can only be argued about; this makes it a column.
    if base_cal_scores is not None:
        base["n_cal_scores"] = int(np.asarray(base_cal_scores).size)
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


def _cut_inclusion_rows(
    details: dict[str, Any],
    base_scores: "np.ndarray",
    base_labels: "np.ndarray",
    fold_haystacks: list,
    sim_scores: list[float],
    ks: list[int],
    weights: list[float],
    rules: list[str],
    fold_combines: list[str],
    qtilt_steps: list[float],
) -> list[dict[str, Any]]:
    """Sweep the fold-anchored cut *rules* across the whole Inclusion knob (#2865).

    The shipped ``mid`` cut was picked by two calibration runs that scored every
    arm at inclusion 0, and a bare midpoint ignores the cost weights inclusion
    arrives as - so it made the knob a no-op for every detector with usable
    folds.  ``mid_tilt`` was shipped to restore the tilt while reproducing the
    measured arm exactly at 0, but the *tilt* has never been priced against its
    alternatives.  This frame is that measurement.

    **Nearly free, and faithful for the same reason.**  The expensive part of a
    fold-anchored threshold is the per-fold anchored EM, and it does not depend
    on the cut rule, the combine, or the inclusion - so the fit is taken **once
    per anchor weight** via the app's own :func:`fit_fold_anchored_cut`, and
    every (rule, combine, k) point re-cuts it through the app's own
    :meth:`~vtscore.training.thresholds.FoldAnchoredCut.threshold_at`.  That is
    not merely cheaper than calling
    :func:`~vtscore.training.thresholds.fold_anchored_gmm_threshold` per point,
    it is exactly what production does when the user drags the slider
    (:func:`vtscore.state.core.recompute_detector_thresholds_for_inclusion`
    re-cuts a cached estimator with no refit), so the sweep measures the object
    the app actually re-cuts rather than a chain of independent retrains.

    A weight whose fit fails outright contributes no rows.  The terminal
    fallbacks :func:`fold_anchored_gmm_threshold` applies in that case
    (final-model unanchored midpoint, then its median) are deliberately *not*
    reproduced here: both are inclusion-blind by construction, so they would
    enter this frame as arms that trivially lose the knob-liveness comparison
    while saying nothing about the rule under test.

    Every row is scored under the cost weights of **its own** ``k`` and against
    the oracle cut at that same ``k`` - the run's reporting inclusion does not
    enter - so regret is comparable along the knob as well as across arms.

    *sim_scores* arrives under the #3308 population convention (the caller
    passes the voted-items-dropped haystack when it has one), matching the
    already-filtered *fold_haystacks*, so every re-cut here realizes its
    quantile on the same population the shipped cut does.
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.eval.calibration_metrics import inclusion_weights, operating_cost, oracle_cut  # noqa: PLC0415

    fold_orderings = details.get("fold_orderings") or []
    if not fold_orderings or not fold_haystacks or not ks:
        return []

    n_folds = min(len(fold_haystacks), len(fold_orderings))
    fold_hay, orderings = fold_haystacks[:n_folds], fold_orderings[:n_folds]
    final_scores = np.asarray(sim_scores, dtype=np.float64)
    base_scores = np.asarray(base_scores, dtype=np.float64)
    base_labels = np.asarray(base_labels, dtype=np.float64)
    n_test = int(base_scores.size)

    # The oracle depends only on k, not on the arm - hoist it out of the arm
    # loop so a 5-rule x 8-weight grid pays for it once per k rather than 40x.
    oracle_by_k = {}
    for k in ks:
        wf, wn = inclusion_weights(k)
        o_thr, o_cost, _o_fpr, _o_fnr = oracle_cut(base_scores, base_labels, wf, wn)
        oracle_by_k[k] = (wf, wn, o_thr, o_cost)

    arms = _cut_inclusion_arms(fold_hay, orderings, final_scores, weights, rules, fold_combines, qtilt_steps)

    out: list[dict[str, Any]] = []
    for arm, arm_cut in arms:
        for k in ks:
            wf, wn, o_thr, o_cost = oracle_by_k[k]
            thr = arm_cut.threshold_at(k)
            if not np.isfinite(thr):
                continue
            cost, fpr, fnr = operating_cost(base_scores, base_labels, thr, wf, wn)
            n_admitted = int(np.count_nonzero(base_scores >= thr))
            out.append(
                {
                    **arm,
                    "inclusion_k": k,
                    "fold_quantile": _r(arm_cut.quantile_at(k)),
                    "cut_threshold": _r(float(thr)),
                    "cut_cost": _r(cost),
                    "cut_fpr": _r(fpr),
                    "cut_fnr": _r(fnr),
                    "k_oracle_threshold": _r(float(o_thr)),
                    "k_oracle_cost": _r(o_cost),
                    "cut_regret": _r(cost - o_cost),
                    "admitted_frac": _r(n_admitted / n_test if n_test else 0.0),
                    "n_admitted": n_admitted,
                    "n_test": n_test,
                }
            )
    return out


def _cut_inclusion_arms(
    fold_hay: list,
    orderings: list[tuple[list[float], list[float]]],
    final_scores: "np.ndarray",
    weights: list[float],
    rules: list[str],
    fold_combines: list[str],
    qtilt_steps: list[float],
) -> "list[tuple[dict[str, Any], Any]]":
    """The ``(identity columns, re-cut estimator)`` pairs the #2865 sweep scores.

    One anchored fit per *weight* - the only expensive step, and the only one
    that depends on none of the swept axes - re-cut into every (rule, combine,
    step) arm by :func:`dataclasses.replace`.  A weight whose fit fails
    contributes no arms.
    """
    from vtscore.training.thresholds import fit_fold_anchored_cut  # noqa: PLC0415

    arms: list[tuple[dict[str, Any], Any]] = []
    for weight in weights:
        cut = fit_fold_anchored_cut(fold_hay, orderings, final_scores, anchor_weight=weight)
        if cut is None:
            continue
        for rule in rules:
            # ``q_tilt`` alone carries a free step size, so it is the only rule
            # that expands over ``qtilt_steps``; the others take a NaN step so
            # the whole frame keeps one row shape.
            steps = list(qtilt_steps) if rule == "q_tilt" else [float("nan")]
            for combine in fold_combines:
                for step in steps:
                    arm_cut = replace(cut, cut_rule=rule, combine=combine)
                    name = f"fold_anchored_w{weight:g}_{rule}_{combine}"
                    if rule == "q_tilt":
                        arm_cut = replace(arm_cut, qtilt_step=step)
                        name += f"_s{step:g}"
                    ident = {
                        "arm": name,
                        "cut_rule": rule,
                        "anchor_weight": weight,
                        "combine": combine,
                        "qtilt_step": _r(step),
                    }
                    arms.append((ident, arm_cut))
    return arms


# ------------------------------------------------------------------
# Active-learning acquisition helpers
# ------------------------------------------------------------------


def _score_pool(
    step: _StepModel,
    pool_ids: list[int],
    clips_dict: dict[int, dict[str, Any]],
    *,
    region_aware: bool = False,
    style_obj: Any = None,
    sim_clips: dict[int, dict[str, Any]] | None = None,
    sim_scored: tuple[list[int], list[float]] | None = None,
) -> dict[int, float]:
    """Return ``{pool_id: score}`` for the current model over the pool.

    **In the same score space the thresholds are cut in** (issue #2943).  That
    is not a refinement, it is a correctness requirement: the Hard pick locates
    its cutoff with the *absolute* comparison ``ranking[cid] <= threshold``
    (:func:`~vtscore.eval.al_strategies._hard_pick_by_index`), so a ranking and
    a cut that live in different spaces put the cutoff index in the wrong place.
    On a patch dataset the reporting/acquisition cuts are fitted on the style's
    region max-pooled scores, and a max over ~197 patch rows stochastically
    dominates the single whole-image row - so scoring the pool whole-image would
    depress every pool score relative to the cut and drag the cutoff index
    systematically toward the top of the ranking.  The app has no such gap: its
    learned sort ranks the very same pooled scores its threshold cuts.

    Three paths, mirroring :func:`_evaluate_on_test` / :func:`_score_sim_set_with_model`:

    * *sim_scored* - the ``(ids, scores)`` the safe-threshold step already
      computed over the whole simulation set, in exactly this geometry.  The
      pool is a subset of that set, so restricting it is free and removes the
      scoring pass entirely.
    * a *style_obj* / *region_aware* dataset with no such scores (the
      ``safe_thresholds=False`` control arm) - score through the style.  The
      **full** sim set is scored rather than just the pool: the style memoises
      its flattened patch matrix per media-id set, and the pool loses an item
      every step, so scoring the shrinking pool would re-flatten from scratch
      each step *and* leak a cache entry per step.
    * everything else (single-vector datasets, the SVM arms) - the trainer-
      agnostic whole-image ``predict``, which is already the threshold's space.
    """
    import numpy as np  # noqa: PLC0415

    if not pool_ids:
        return {}
    if sim_scored is not None:
        ids, scores = sim_scored
        pool_set = set(pool_ids)
        return {cid: float(s) for cid, s in zip(ids, scores, strict=True) if cid in pool_set}
    if (style_obj is not None or region_aware) and sim_clips:
        assert step.torch_model is not None
        pool_set = set(pool_ids)
        ids, scores = _score_sim_set_with_model(
            step.torch_model, region_aware, sim_clips, None, sorted(sim_clips), style_obj
        )
        return {cid: float(s) for cid, s in zip(ids, scores, strict=True) if cid in pool_set}
    embs = np.array([media_embedding(clips_dict[cid]) for cid in pool_ids])
    scores = np.asarray(step.predict(embs)).ravel().tolist()
    return dict(zip(pool_ids, scores, strict=True))


def _labelset_error_costs(
    model_steps: list[tuple[Any, float]],
    good_votes: dict[int, None],
    bad_votes: dict[int, None],
    clips_dict: dict[int, dict[str, Any]],
    inclusion: int,
) -> list[float]:
    """Weighted FPR/FNR of **every** recent model on the current labelled set.

    Feeds the Smart indicator.  Mirrors ``labeling_progress._eval_cached_models``
    /``_score_step``: every model in the window is re-scored against the
    *current* labelset — the only ground truth the app has — with its own cached
    threshold, so all points of the slope regression share one eval set and the
    trend isolates model improvement.  Scoring each model against the labelset
    it was trained on instead would confound model change with labelset growth:
    autopilot deliberately votes boundary items, which are mispredicted at first
    and inflate the later costs of a frozen-cost history.

    Deliberately *not* the held-out test split: those labels must never reach
    the vote order.  Returns ``[]`` when the labelset has no usable eval set
    (either class empty), matching ``_eval_cached_models``, which leaves the
    Smart indicator on its "not enough points" branch.
    """
    import numpy as np  # noqa: PLC0415

    ids = list(good_votes) + list(bad_votes)
    labels = [1.0] * len(good_votes) + [0.0] * len(bad_votes)
    total_pos = len(good_votes)
    total_neg = len(bad_votes)
    if not model_steps or not ids or total_pos == 0 or total_neg == 0:
        return []

    fpr_weight, fnr_weight = _inclusion_weights(inclusion)
    # One eval matrix, reused by every model in the window - the app's
    # ``_build_eval_set`` builds its tensor once for the same reason.
    embs = np.array([media_embedding(clips_dict[cid]) for cid in ids])

    costs: list[float] = []
    for step, threshold in model_steps:
        scores = np.asarray(step.predict(embs)).ravel()
        fp = fn = 0
        for score, true_label in zip(scores.tolist(), labels, strict=True):
            predicted = 1 if score >= threshold else 0
            if predicted == 1 and true_label == 0.0:
                fp += 1
            elif predicted == 0 and true_label == 1.0:
                fn += 1
        fpr = fp / total_neg
        fnr = fn / total_pos
        costs.append(fpr_weight * fpr + fnr_weight * fnr)
    return costs


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
    head: str = PRODUCTION_HEAD,
    style_obj: Any = None,
    emit_calibration_metrics: bool = False,
    fold_count_variants: list[int] | None = None,
) -> tuple[_StepModel, float, int, dict[str, float], dict[str, Any]]:
    """Train the step's ranker and calibrate its threshold from the current votes.

    *head* selects the head on both production paths (see :data:`HEADS`):
    ``"linear_svm"`` (the default, :data:`PRODUCTION_HEAD`) trains the head the
    live detector has, ``"linear"`` the logistic head it replaced, ``"mlp"`` the
    legacy auto-sized hidden layer.  It is ignored by the standalone SVM path,
    which fits its own estimator rather than a head.

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
            fold_count_variants=fold_count_variants,
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
    head: str = PRODUCTION_HEAD,
) -> tuple[_StepModel, float, int, dict[str, float], dict[str, Any]]:
    """The production arm — numerically identical to the pre-trainer harness at ``head="mlp"``.

    At ``head="linear_svm"`` (the default, :data:`PRODUCTION_HEAD`) this trains
    the live detector's head: production pins the linear SVM on every fit (see
    ``vtscore.training.mlp.LINEAR_SVM_HEAD``), so the reported thresholds and
    costs are the shipped detector's.  ``head="linear"`` (the logistic head the
    SVM replaced) and ``head="mlp"`` (the small-MLP candidate #2781 measured)
    are the named legacy arms.  Everything *around* the head mirrors the
    production ``_train_and_score_xy`` / ``train_and_threshold`` pipeline
    whichever is chosen:

    Good votes region-pool their ground-truth box when *region_voting* is on
    (and the media supports it); Bad votes always train on the whole-image
    vector.

    **This is the single-vector path.**  Bad votes here are one row because a
    single-vector media *has* one row - not because the live detector works that
    way.  On a patch dataset the live detector floods a Bad vote over the
    image's whole score-row stack, and
    :func:`simulate_voting_iterations` routes such datasets to the
    ``max_patch`` style (:func:`_style_train_and_calibrate`) rather than here,
    so the default arm matches the app.  Do not "restore" whole-image Bad votes
    on patch data: that trains ~196 rows per rejected image down never while
    inference max-pools them.

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
    # The folds' orderings *and* models ride out in ``details`` unconditionally:
    # the shipped safe threshold anchors on the fold models' held-out scores, so
    # they are an input to the baseline arm, not study-only extras.
    folds = calibration_folds(
        X_list,
        y_list,
        input_dim,
        calibrate_count=calibrate_count,
        calibration_fraction=calibration_fraction,
        hidden_dim=hidden_dim,
        rng=np.random.RandomState(42),
    )
    threshold = threshold_from_folds(folds, inclusion)
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
    details = {
        "fold_orderings": folds.orderings,
        "fold_models": folds.models,
        # Which sentinel (if any) the fold rule returned: the blend's x-cal side
        # is NO_GOOD_THRESHOLD whenever this is set, as production's does
        # (see :func:`_blend_xcal_input`).
        "fold_fallback": folds.fallback,
    }
    return step, threshold, n_labels, {"train_seconds": train_seconds, "xcal_seconds": xcal_seconds}, details


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
    head: str = PRODUCTION_HEAD,
    emit_calibration_metrics: bool = False,
    fold_count_variants: list[int] | None = None,
) -> tuple[_StepModel, float, int, dict[str, float], dict[str, Any]]:
    """Style-driven torch path (the Max-Patch experiment arms).

    The detection style (see :mod:`vtscore.eval.patch_styles`) supplies the
    vote-to-vector rules: each Good vote contributes ``style.good_vec`` (given
    the ground-truth box when *region_voting* and the media has one), each Bad
    vote floods ``style.bad_vecs`` - one row on a whole-image style, the
    image-level vector + every raw patch on ``max_patch``, every tree node on
    the HAC hybrids.

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
            fold_count_variants=fold_count_variants,
        )
        # Bad-voted bags' inference row stacks: the final model scores these to
        # form the pnorm null (F_neg) at test time (see _calibration_metric_rows).
        details["neg_score_rows"] = [score_rows_by_group[("b", vid)] for vid in bad_votes]
    else:
        # Same fold work as the metrics branch, minus the study extras: the
        # shipped safe threshold anchors on the fold models, so they ride out
        # in ``details`` on every path (see :func:`_safe_threshold_for_step`).
        folds = calibration_folds(
            X_list,
            y_list,
            input_dim,
            calibrate_count=calibrate_count,
            calibration_fraction=calibration_fraction,
            hidden_dim=hidden_dim,
            rng=np.random.RandomState(42),
            groups=cal_groups,
            score_rows_by_group=score_rows_by_group if cal_groups is not None else None,
        )
        threshold = threshold_from_folds(folds, inclusion)
        details = {
            "fold_orderings": folds.orderings,
            "fold_models": folds.models,
            "fold_fallback": folds.fallback,
        }
    xcal_seconds = time.monotonic() - t_xcal
    # Under the #2897 screen this step trained Kmax folds, not ``calibrate_count``
    # of them.  Bill the reported wall clock for the live count only, so the
    # baseline row's timing stays the one an uninstrumented run would report; the
    # per-K costs live in each fold-count arm's own ``fold_seconds``.
    extra = (details.get("fold_count_data") or {}).get("seconds")
    if extra:
        xcal_seconds -= sum(extra[calibrate_count:])
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
    fold_count_variants: list[int] | None = None,
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
    * ``fold_fallback`` — the sentinel the fold rule returned, or ``None`` when
      the folds are real.  The shipped blend substitutes ``NO_GOOD_THRESHOLD``
      for the x-cal side whenever this is set, as production does (see
      :func:`_blend_xcal_input`).
    * ``fold_count_data`` — only under *fold_count_variants* (issue #2897): the
      **full** Kmax fold orderings, their per-fold seconds, and the
      count-independent overhead, for :func:`_fold_count_variant_rows`.

    On the grouped path the fold models are trained once via
    :func:`~vtscore.training.thresholds.compute_grouped_fold_node_scores` and the
    base orderings are the max-pool of the node data, so the threshold is
    identical to what production's grouped calibration produces for this arm.

    *fold_count_variants* raises the number of folds actually trained to
    ``max(calibrate_count, *variants)`` while leaving everything the step
    returns computed off the first ``calibrate_count`` of them.  That is exact,
    not an approximation: the folds are nested (see
    :func:`~vtscore.training.thresholds.compute_fold_orderings`) and
    ``train_model`` is seeded per call, so the extra folds cannot perturb the
    live threshold, the fold models, or the trajectory - they only cost time.
    """
    import numpy as np  # noqa: PLC0415

    k_max = max(calibrate_count, *(fold_count_variants or [calibrate_count]))
    t_folds = time.monotonic()
    fold_seconds: list[float] = []

    def _with_fold_data(details: dict[str, Any], orderings: list) -> dict[str, Any]:
        """Attach the fold-count screen's inputs and trim *details* to K live folds."""
        if fold_count_variants:
            details["fold_count_data"] = {
                "orderings": orderings,
                # The **untrimmed** fold models, so the #3116 anchored arm can
                # re-fit production's rule at every K.  `details["fold_models"]`
                # is deliberately cut to the live count so nothing downstream
                # can accidentally widen the shipped threshold's own fit.
                "models": list(fold_models),
                "seconds": fold_seconds,
                # Everything in the calibration wall clock that is *not* a fold
                # fit (the pooled conformal rule, the node max-pool): paid once
                # at every K, so it belongs in each arm's cost.
                "overhead_seconds": max(0.0, (time.monotonic() - t_folds) - sum(fold_seconds)),
            }
        return details

    # The trained fold models ride along in details["fold_models"] so the
    # #2852 fold-anchored arm can score the haystack on each fold's own scale
    # without retraining; production callers never see them.
    fold_models: list = []
    if cal_groups is not None:
        fold_node_data, fallback = compute_grouped_fold_node_scores(
            X_list,
            y_list,
            input_dim,
            groups=cal_groups,
            rng=np.random.RandomState(42),
            calibrate_count=k_max,
            calibration_fraction=calibration_fraction,
            hidden_dim=hidden_dim,
            score_rows_by_group=score_rows_by_group,
            model_sink=fold_models,
            seconds_sink=fold_seconds,
        )
        if fallback is not None:
            return fallback, {
                "provenance": classify_threshold_provenance(fallback),
                "fold_orderings": [],
                "fold_node_data": None,
                "fold_models": [],
                "fold_fallback": fallback,
            }
        # Base (max) orderings from the same fold node data -> identical to
        # production's grouped calibration for this arm.
        all_orderings = [([float(np.max(b)) for b in blocks], labels) for blocks, labels in fold_node_data]
        fold_orderings = all_orderings[:calibrate_count]
        threshold = threshold_from_fold_orderings(fold_orderings, inclusion)
        return threshold, _with_fold_data(
            {
                "provenance": classify_threshold_provenance(None),
                "fold_orderings": fold_orderings,
                "fold_node_data": fold_node_data[:calibrate_count],
                "fold_models": fold_models[:calibrate_count],
                "fold_fallback": None,
            },
            all_orderings,
        )

    # Row-wise path (whole-image styles): no bag flooding, no node re-pooling.
    all_orderings, fallback = compute_fold_orderings(
        X_list,
        y_list,
        input_dim,
        rng=np.random.RandomState(42),
        calibrate_count=k_max,
        calibration_fraction=calibration_fraction,
        hidden_dim=hidden_dim,
        model_sink=fold_models,
        seconds_sink=fold_seconds,
    )
    if fallback is not None:
        return fallback, {
            "provenance": classify_threshold_provenance(fallback),
            "fold_orderings": [],
            "fold_node_data": None,
            "fold_models": [],
            "fold_fallback": fallback,
        }
    fold_orderings = all_orderings[:calibrate_count]
    threshold = threshold_from_fold_orderings(fold_orderings, inclusion)
    return threshold, _with_fold_data(
        {
            "provenance": classify_threshold_provenance(None),
            "fold_orderings": fold_orderings,
            "fold_node_data": None,
            "fold_models": fold_models[:calibrate_count],
            "fold_fallback": None,
        },
        all_orderings,
    )


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
    safe_thresholds: bool = True,
    calibrate_count: int = 2,
    calibration_fraction: Optional[float] = None,
    region_voting: bool = False,
    strategy: str = "autopilot",
    max_steps: Optional[int] = None,
    atlas_min_node_size: int = 20,
    seed_scores: Optional[dict[int, float]] = None,
    trainer: str = "mlp",
    head: Optional[str] = None,
    target_prevalence: Optional[float] = None,
    style: Optional[str] = None,
    emit_calibration_metrics: bool = False,
    repool_variants: Optional[list[str]] = None,
    repool_topk: int = 4,
    inclusion_sweep_ks: Optional[list[int]] = None,
    sweep_sink: Optional[list[dict[str, Any]]] = None,
    blend_schedule: Optional[str] = None,
    schedule_variants: Optional[list[str]] = None,
    cut_diag_sink: Optional[list[dict[str, Any]]] = None,
    autopilot_fidelity: bool = True,
    anchored_thresholds: bool = False,
    anchored_weights: Optional[list[float]] = None,
    anchored_rules: Optional[list[str]] = None,
    anchored_fold_arms: bool = True,
    anchored_fold_combines: Optional[list[str]] = None,
    fold_count_variants: Optional[list[int]] = None,
    cut_inclusion_ks: Optional[list[int]] = None,
    cut_inclusion_sink: Optional[list[dict[str, Any]]] = None,
    cut_inclusion_qtilt_steps: Optional[list[float]] = None,
    acq_inclusion_offset: int = ACQUISITION_INCLUSION_OFFSET,
    acq_rank_percentile: Optional[float] = None,
    startup_schedule: Optional[str] = None,
    pick_sink: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Simulate voting on *clips_dict* and evaluate at every step.

    Args:
        clips_dict: Pre-loaded media dict (``{id: clip_data}``).
        target_category: Category treated as the positive class.
        seed: Random seed for splitting and vote ordering.
        dataset_name: Label included in result rows.
        inclusion: Inclusion setting in ``[-10, 10]``.
        trainer: Which ranker to train at each step — ``"mlp"`` (default, the
            production path, whose head is chosen by *head*) or a standalone
            SVM name
            (``"svm_linear"``, ``"svm_rbf"``, or a parameterised spec such as
            ``"svm_rbf@C=3,gamma=scale"``).  The autopilot vote order adapts to
            the chosen model, so MLP and SVM trajectories diverge after the
            first retrain even at the same seed — by design (the question is
            which model makes *VTSearch* better, and VTSearch's vote order
            depends on the model).
        head: Which head the ``"mlp"`` trainer fits at each step (see
            :data:`HEADS`).  ``None`` (default) resolves to the **app's** head,
            :data:`PRODUCTION_HEAD` — the linear SVM a live VTSearch detector
            actually has, so a default run's thresholds and costs are the ones
            users see.  ``"linear"`` (the logistic head the SVM replaced) and
            ``"mlp"`` (the harness's auto-sized hidden layer, #2781) are the
            explicitly-named legacy arms.  The head is threaded into the
            calibration folds as well, mirroring how production threads one
            sentinel through ``_train_and_score_xy``.  Rejected on the
            standalone SVM trainers, which fit their own estimator rather than
            a head; the *resolved* name is recorded in the ``head`` result
            column (blank on those trainers).
        style: Optional detection-style name (see
            :mod:`vtscore.eval.patch_styles`): ``"whole_image"``,
            ``"max_patch"`` (the production geometry), or one of the
            ``"max_patch_hac"`` hybrids.  When set (MLP trainer only), the style owns the
            vote-to-vector assembly, the test/sim scoring rule, and the
            bag-aware flooding of Bad votes - the Max-Patch experiment arms.
            ``None`` (default) resolves to the **app's** geometry: a patch
            dataset (any media with a ``patch_grid``) on the MLP trainer gets
            ``"max_patch"``, everything else keeps the historical single-vector
            path byte-for-byte.  The *resolved* name is what lands in the
            ``style`` result column, so a row always says which geometry
            produced it.
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
        safe_thresholds: The shipped threshold path - fuse the haystack score
            distribution into the trained cut (the fold-anchored estimator, see
            :func:`vtscore.training.thresholds.fold_anchored_gmm_threshold`).
            **On by default, matching the app**, which has no switch for it.
            Set ``False`` only to run the no-fusion control arm: pure
            cross-calibration, which the app can no longer produce.
            Under ``emit_calibration_metrics`` with a *style*, each step
            additionally emits one metric row per safe-threshold cut variant
            (:data:`_SAFE_GMM_VARIANTS`, tagged in the ``gmm_variant`` column) -
            the #2799/#2836 measurement arms - and, when *cut_diag_sink* is
            given, one :data:`_CUT_DIAGNOSTIC_COLUMNS` row per (step, geometry)
            carrying the fitted mixture parameters and the #2836 decomposition
            chain.
        calibrate_count: Number of random Train/Calibrate splits for threshold
            calibration (default 2).
        calibration_fraction: Fraction of labelled data reserved for
            calibration in each split.  ``None`` (default) resolves to the
            **app's** per-space split
            (:func:`vtscore.training.thresholds.production_split_for`, issue
            #3287): 0.5 when the dataset carries a ``patch_grid`` (built by a
            patch embedder), 0.3 otherwise - so a default run's folds are
            split the way a live detector's are.
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
        cut_inclusion_ks: Inclusion values the **fold-anchored cut rules** are
            swept over for issue #2865, into *cut_inclusion_sink* (columns
            :data:`_CUT_INCLUSION_COLUMNS`).  Orthogonal to
            *inclusion_sweep_ks*, which sweeps the conformal rule's budget: this
            one asks which cut rule should answer the Inclusion knob, so its
            rows are scored at their own ``k`` rather than at *inclusion*.  The
            arms come from *anchored_weights* x *anchored_rules* x
            *anchored_fold_combines*, so ``anchored_rules=["mid", "mid_tilt",
            "rate", "cross_tilt", "q_tilt"]`` is the candidate set the issue
            names.  ``None`` (default) = off, and every other study is unchanged.
        cut_inclusion_sink: List the #2865 rows are appended to.  Required for
            *cut_inclusion_ks* to do anything.
        cut_inclusion_qtilt_steps: Step sizes the eval-only ``q_tilt`` rule is
            expanded over (its free parameter; every other rule ignores this).
            Defaults to the single placeholder
            :data:`~vtscore.training.thresholds.FOLD_ANCHOR_QTILT_STEP`.
        acq_inclusion_offset: Cut the threshold handed to the **selector** at
            ``inclusion + acq_inclusion_offset``, leaving reporting and every
            metric at *inclusion* so arms stay comparable.  Defaults to
            :data:`~vtscore.training.thresholds.ACQUISITION_INCLUSION_OFFSET`
            (-3), **the shipped app behaviour** - the harness matches production
            here as it does everywhere else, so a baseline arm measures what
            users get.  Pass ``0`` for the pre-#2876 control, where one threshold
            did both jobs.

            The direction is the opposite of the intuition from the cost
            weights, because Autopilot's ``hard`` pick reads the threshold as a
            **rank position**, not a decision boundary: a *negative* offset
            prices false alarms higher, *raises* the cut, moves it *up* the
            ranking, and so returns *more* positives.  Requires a fold-anchored
            cut for the step; steps that fall back to the schedule blend keep
            the reporting threshold (the blend has no inclusion-aware form).
        startup_schedule: A parameterised Autopilot **opening** (issue #3267),
            e.g. ``"n6@k-6,n6@k-2,n6@k0"``; see
            :mod:`vtscore.eval.startup_schedule` for the grammar.  ``None``
            (default) is the **app's own** opening - three positives off the top
            of the seed sort, four negatives at its cutoff - and leaves the
            trajectory byte-for-byte what it was before the knob existed.  A
            schedule replaces only the pre-detector phases; the learned Hard
            sort that follows is unchanged and still samples at
            *acq_inclusion_offset*.  Requires *seed_scores*: a schedule names
            positions on the seed sort, so there has to be one.
        pick_sink: List the per-click :data:`_PICK_COLUMNS` rows are appended
            to - one per vote, including the opening's, which emit no main row
            because no model exists yet.  ``None`` (default) = off.
        acq_rank_percentile: Alternative acquisition cut - place it at this
            quantile of the simulation-set score distribution directly, rather
            than by naming an inclusion.  This is the ``rank_pin`` arm: same
            intent, one fewer indirection.  Requires
            ``acq_inclusion_offset=0``, since the two name the same cut.
        anchored_thresholds: When ``True`` (requires ``safe_thresholds``,
            ``emit_calibration_metrics``, and a *style*), each step additionally
            emits one metric row per anchored-mixture arm (issue #2852): the
            label-anchored family (``anchored_w{W}_{rule}``), the fold-anchored
            "cross-LabeledGMM" family (``fold_anchored_w{W}_{rule}_{combine}``),
            and the ``rank_transfer`` attribution arm - see
            :func:`_anchored_variant_rows`.  The fold arms score the sim set
            once per calibration fold model per step, so they cost roughly one
            extra scoring pass per fold.
        anchored_weights: Anchor-weight grid for the anchored arms (default
            :data:`_ANCHORED_WEIGHTS`).  Each labelled score counts as this
            many haystack scores in the anchored EM's M-step.
        anchored_rules: Cut rules applied to each anchored fit (default
            :data:`_ANCHORED_RULES`): ``"mid"`` (plain midpoint), ``"rate"``
            (rate-optimal crossing at the live inclusion weights), and/or
            ``"mid_tilt"`` (the shipped rule: midpoint anchored at inclusion 0,
            rate tilt away from it).  ``"mid_tilt"`` is defined in
            fold-quantile space, so it applies to the fold-anchored family
            only; the label-anchored family skips it.
        anchored_fold_arms: Include the fold-anchored + rank-transfer arms
            (default ``True``); ``False`` keeps only the cheap label-anchored
            family (no per-fold scoring passes).
        anchored_fold_combines: How the fold arms combine per-fold cuts in
            quantile space (default :data:`_ANCHORED_FOLD_COMBINES`):
            ``"qmean"`` and/or ``"qmedian"``.
        fold_count_variants: Calibration fold counts to score counterfactually
            (issue #2897; requires ``emit_calibration_metrics`` and a *style*).
            Each step trains ``max(calibrate_count, *variants)`` folds instead of
            ``calibrate_count`` and emits one ``folds_k{K}_xcal`` row - plus a
            ``folds_k{K}_blend`` row where the step has a safe-threshold fit -
            per K, carrying that K's regret and its measured ``fold_seconds``.
            The folds are nested, so the live threshold and the trajectory are
            byte-identical to a plain run at ``calibrate_count`` and the arm at
            ``K == calibrate_count`` reproduces this step's own conformal cut;
            see :func:`_fold_count_variant_rows`.  Costs ``Kmax - calibrate_count``
            extra fold fits per step and nothing else.
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

    # One filter for the whole cell, before anything reads a label: on a
    # scale-banded dataset an image can hold the category at the wrong size,
    # and every "not positive means negative" test below would score it as a
    # negative.  A no-op for every dataset that does not designate its cells.
    clips_dict = evaluable_pool(clips_dict, target_category)

    rng = np.random.RandomState(seed)
    # Note: no torch.manual_seed() here - train_model handles its own
    # RNG seeding via fork_rng, keeping it thread-safe.
    start_time = time.monotonic()

    # These are pre-registered experiment knobs, so they are validated beside
    # the other argument checks rather than deep in the loop: a run that dies
    # forty minutes in on a typo has held a cluster slot for nothing.
    startup_state: StartupState | None = None
    if startup_schedule:
        if seed_scores is None:
            raise ValueError(
                "startup_schedule needs seed_scores: a schedule names positions on the "
                "seed sort, and there is no sort to name them on without one"
            )
        if not autopilot_fidelity or strategy != "autopilot":
            raise ValueError("startup_schedule requires autopilot_fidelity and the autopilot strategy")
        startup_state = StartupState(parse_startup_schedule(startup_schedule))

    if acq_rank_percentile is not None:
        if acq_inclusion_offset != 0:
            raise ValueError(
                "acq_inclusion_offset and acq_rank_percentile are mutually exclusive; "
                "pass acq_inclusion_offset=0 to run the rank-pinned arm "
                f"(the default is {ACQUISITION_INCLUSION_OFFSET}, the shipped acquisition cut)"
            )
        if not 0.0 <= acq_rank_percentile <= 1.0:
            raise ValueError(f"acq_rank_percentile must be in [0, 1], got {acq_rank_percentile}")

    if head is not None:
        if head not in HEADS:
            raise ValueError(f"unknown head {head!r}; expected one of {HEADS}")
        if trainer != "mlp":
            raise ValueError(f"head={head!r} only applies to the production trainer; got trainer={trainer!r}")
    # **The default arm must be the app's default.**  Production pins the linear
    # SVM head on every fit (``hidden_dim = LINEAR_SVM_HEAD`` in
    # ``vtscore.detectors.training.train_and_threshold``), so an unspecified head
    # resolves to it — the same way *style* and *blend_schedule* resolve to the
    # app's geometry and schedule below.  The head fits the final model *and*
    # the calibration folds, so it moves the thresholds and, through the vote
    # order, the whole trajectory: defaulting to a retired head would make every
    # unqualified run measure a detector nobody ships.  ``head="linear"`` (the
    # logistic head) and ``head="mlp"`` stay available as named legacy arms.
    head = head or PRODUCTION_HEAD

    if style is not None and trainer != "mlp":
        raise ValueError(f"detection styles only support the MLP trainer; got trainer={trainer!r}")

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

    # A patch dataset exposes a ``patch_grid`` per media; such datasets are
    # scored region-aware (max-pool over the image's score rows) the same way
    # the live detector scores them, regardless of how the Good votes were
    # assembled.
    region_aware = any(clips_dict[cid].get("patch_grid") is not None for cid in clips_dict)

    # `region_voting` is a request, not a guarantee: `_good_training_vec` pools
    # the ground-truth box only when the media carries a stored `patch_grid`,
    # and falls back to the whole-image embedding otherwise - which is the same
    # condition `region_aware` above tests.  On a single-vector embedder that
    # fallback fires for EVERY vote, so the run is plain binary voting under a
    # flag that says otherwise, and it scores whole-image and blends under the
    # binary schedule too.  None of that shows up in the output: #2877 shipped a
    # report calling `visual_genome_m x siglip` a region-voting environment
    # before anyone checked, because the dataset is boxed and the harness config
    # said "region voting" next to its name.  Say so loudly.
    if region_voting and not region_aware:
        import warnings  # noqa: PLC0415

        warnings.warn(
            "region_voting=True but no media carries a patch_grid, so every Good "
            "vote falls back to its whole-image embedding: this run is BINARY "
            "voting. Region voting needs a patch embedder (e.g. dinov3_patch). "
            "See docs/experiments/acquisition-inclusion/REPORT_SECOND_ENVIRONMENT.md.",
            RuntimeWarning,
            stacklevel=2,
        )

    # **The default arm must be the app's default.**  On a patch dataset the
    # live detector floods a Bad vote over the image's whole score-row stack
    # (``bad_negative_vecs``) and trains/calibrates bag-aware; the style-less
    # path here trains a Bad vote on one image-level row.  That gap predates
    # #2886 but MaxPatch widened it from 1-vs-24 to 1-vs-197 rows: the default
    # arm would train ~196 patch rows per rejected image down never, while
    # scoring max-pools all of them, so it would systematically under-suppress
    # and its numbers would not describe the shipped tool.  An eval default that
    # doesn't match the app default can't be trusted, so a patch dataset
    # defaults to the ``max_patch`` style - which *is* the production geometry
    # (its methods delegate to ``pool_box_from_media`` / ``bad_negative_vecs`` /
    # ``media_score_rows``).  The resolved name is recorded in the ``style``
    # column, so a result row always says which geometry produced it.
    #
    # Single-vector datasets are untouched: no patch grid, no style, and the
    # historical ``_mlp_train_and_calibrate`` path runs byte-for-byte.  Non-MLP
    # trainers are untouched too - they have no head for a style to drive.
    if style is None and region_aware and trainer == "mlp":
        style = PRODUCTION_PATCH_STYLE

    style_obj: Any = None
    if style is not None:
        from vtscore.eval.patch_styles import resolve_style  # noqa: PLC0415

        style_obj = resolve_style(style)
    # Mirror the app's per-mode schedule default (#2841): with no explicit arm, a
    # patch dataset blends under the region schedule and a single-vector one
    # under the binary schedule, exactly as `_blend_schedule_for_snap` decides in
    # `vtscore.detectors.training`.  Without this the harness would measure a
    # schedule no detector actually uses.
    if blend_schedule is None:
        from vtscore.training.blend_schedules import production_schedule_for  # noqa: PLC0415

        blend_schedule = production_schedule_for(region_voting=region_aware)

    # Mirror the app's per-space split default (#3287/#3290): with no explicit
    # arm, the Train/Calibrate fraction of each fold is the one a live
    # detector would resolve for this dataset's embedder.  ``region_aware``
    # (any media carrying a ``patch_grid``) is the harness's spelling of "the
    # pickle was built by a patch embedder" - the same capability the app
    # reads off ``supports_patch_regions`` in
    # ``vtscore.detectors.training.resolve_calibration_fraction``, which the
    # ``training.split_fraction_default`` mirror in
    # ``scripts/check-eval-app-sync.py`` pins against this block.  Note it is
    # deliberately NOT the voting mode: ``dinov3_patch`` datasets take 0.5 in
    # both their styles, including boxless ``whole_image``.
    if calibration_fraction is None:
        from vtscore.training.thresholds import production_split_for  # noqa: PLC0415

        calibration_fraction = production_split_for(patch_space=region_aware)

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
    # The snapshot is built for every region-aware / styled run, not only the
    # safe-threshold ones: the pool scorer needs it too, and it is a dict of
    # references to media already in memory.
    sim_clips: dict[int, dict[str, Any]] | None = None
    X_all_clips: Any = None
    if region_aware or style_obj is not None:
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
    #: The selector's threshold - cut ``acq_inclusion_offset`` steps below the
    #: reporting one.  Kept as its own name so the two jobs cannot silently
    #: re-merge (they were one variable, and that is how the #2847 positives
    #: regression got in).
    acq_threshold = 0.5
    pool_scores: dict[int, float] = {}
    n_steps = len(pool) if max_steps is None else min(max_steps, len(pool))

    # The app's phase machine, driving the vote order the way Autopilot does.
    # Disabled (``None``) under ``autopilot_fidelity=False``, which leaves the
    # selector on its legacy parity interleave.
    flow: Any = None
    if autopilot_fidelity and strategy == "autopilot":
        flow = AutopilotFlow(startup=startup_state)
    # Each schedule round's cut on the seed sort, resolved once: the app fits a
    # cosine sort's GMM over the whole sort and never refits it as votes come
    # in, so these are constants of the run rather than per-step state.
    startup_cuts: list[float] = []
    if startup_state is not None and seed_scores is not None:
        sort_values = list(seed_scores.values())
        startup_cuts = [round_cut(sort_values, rnd) for rnd in startup_state.rounds]
    # The seed sort as a ranking, for the pick log: where in the sort each click
    # landed is the mining record the study reads.
    seed_rank: dict[int, int] = {}
    if pick_sink is not None and seed_scores is not None:
        seed_rank = {cid: i for i, cid in enumerate(sorted(seed_scores, key=lambda c: seed_scores[c], reverse=True))}
    seed_sorted_scores: list[float] = sorted(seed_scores.values(), reverse=True) if seed_scores else []
    # Recent per-step models (each with the threshold it was calibrated at),
    # re-scored every step against the *current* labelset so the Smart
    # indicator's slope regresses over one shared eval set - exactly what the
    # app's ``_eval_cached_models`` does over its per-step cache.
    recent_steps: list[tuple[Any, float]] = []

    for t in range(1, n_steps + 1):
        if not pool:
            break
        phase = flow.phase if flow is not None else None
        startup_round = startup_state.index if (startup_state is not None and not startup_state.done) else -1
        startup_cut = startup_cuts[startup_round] if startup_round >= 0 else None
        ctx = ALContext(
            pool_ids=pool,
            embeddings=sim_embeddings,
            labeled=labeled,
            scores=pool_scores,
            model=step,
            # The ONLY consumer that moves.  Reporting, the metric rows and the
            # phase machine all stay on ``threshold``.
            threshold=acq_threshold,
            atlas=atlas,
            rng=rng,
            pool_labels=pool_labels,
            seed_scores=seed_scores,
            phase=phase,
            startup_cut=startup_cut,
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

        if pick_sink is not None:
            rank = seed_rank.get(cid, -1)
            n_sorted = len(seed_rank)
            pick_sink.append(
                {
                    "seed": seed,
                    "dataset": dataset_name,
                    "category": target_category,
                    "startup_schedule": startup_schedule or "",
                    "style": style or "",
                    "t": t,
                    "phase": phase or "",
                    "startup_round": startup_round,
                    "startup_held": bool(startup_state.held_for_quorum) if startup_state is not None else False,
                    "startup_extended_clicks": int(startup_state.extended_clicks) if startup_state is not None else 0,
                    "startup_cut": _r(startup_cut) if startup_cut is not None else float("nan"),
                    "startup_cut_percentile": (
                        _sorted_percentile(seed_sorted_scores, startup_cut) if startup_cut is not None else float("nan")
                    ),
                    "picked_id": cid,
                    "picked_label": 1 if is_positive else 0,
                    "picked_seed_rank": rank,
                    "picked_seed_percentile": (
                        _r(rank / (n_sorted - 1)) if n_sorted > 1 and rank >= 0 else float("nan")
                    ),
                    "picked_seed_score": _r(seed_scores[cid]) if seed_scores and cid in seed_scores else float("nan"),
                    "picked_detector_score": _r(pool_scores[cid]) if cid in pool_scores else float("nan"),
                    "acq_threshold": _r(acq_threshold),
                    "n_good": len(good_votes),
                    "n_bad": len(bad_votes),
                    "n_pool": len(pool),
                }
            )

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
            fold_count_variants=fold_count_variants,
        )

        # Apply the shipped safe threshold if enabled
        sim_pooled_scores: list[float] | None = None
        sim_pooled_ids: list[int] = []
        sim_fold_haystacks: list[Any] = []
        if safe_thresholds:
            # The x-cal side of the blend, not the raw fold return: a step whose
            # folds fell back blends NO_GOOD_THRESHOLD (see _blend_xcal_input),
            # and the variant families below re-blend this same input, so their
            # rows stay paired with the shipped one.
            xcal_threshold = _blend_xcal_input(threshold, details)
            # Vote-level class counts, so the fallback blend's schedule can ramp
            # on the rarer class (#2841).  The harness votes one media at a
            # time, so bags and votes coincide here and the counts are the two
            # vote dicts' sizes.
            blend_ctx = BlendContext(n_labels=n_labels, n_good=len(good_votes), n_bad=len(bad_votes))
            threshold, sim_pooled_scores, sim_pooled_ids, sim_fold_haystacks, safe_provenance, safe_cut = (
                _safe_threshold_for_step(
                    threshold,
                    step,
                    details,
                    region_aware,
                    sim_clips,
                    X_all_clips,
                    blend_ctx,
                    sim_ids,
                    inclusion,
                    style_obj=style_obj,
                    schedule=blend_schedule,
                    voted_ids=set(good_votes) | set(bad_votes),
                )
            )
            if emit_calibration_metrics:
                details["pre_blend_provenance"] = details.get("provenance", "conformal")
                details["provenance"] = safe_provenance
                details["xcal_threshold"] = xcal_threshold
                details["n_votes"] = n_labels
                details["n_good"] = len(good_votes)
                details["n_bad"] = len(bad_votes)

        # The selector's cut.  Recomputed from scratch every step - never
        # carried over - so a step with nothing to re-cut falls back to the
        # reporting threshold rather than sampling this step's scores against
        # the last step's cut.
        acq_threshold = threshold
        if safe_thresholds:
            if acq_rank_percentile is not None:
                if sim_pooled_scores:
                    acq_threshold = float(
                        np.quantile(np.asarray(sim_pooled_scores, dtype=np.float64), acq_rank_percentile)
                    )
            elif acq_inclusion_offset != 0 and safe_cut is not None:
                # Re-cut the *same* fold-anchored fit.  O(1) - the mixture was
                # fitted above; ``threshold_at`` is monotone by construction, so
                # the arms are nested and offset 0 reproduces the reporting cut
                # exactly.  ``safe_cut is None`` is the schedule-blend fallback
                # (~5% of steps, concentrated in the cold start): the blend has
                # no inclusion-aware form, so there is nothing honest to re-cut.
                cand = safe_cut.threshold_at(acquisition_inclusion(inclusion, acq_inclusion_offset))
                if np.isfinite(cand):
                    acq_threshold = float(cand)

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
        # autopilot Hard pick can rank it - in the geometry the cut it will be
        # compared against was fitted in (#2943).  The safe-threshold path has
        # already scored the whole sim set that way, so hand those scores over
        # rather than paying for a second pass.
        t_pool = time.monotonic()
        pool_scores = _score_pool(
            step,
            pool,
            clips_dict,
            region_aware=region_aware,
            style_obj=style_obj,
            sim_clips=sim_clips,
            sim_scored=(sim_pooled_ids, sim_pooled_scores) if sim_pooled_scores else None,
        )
        pool_score_seconds = time.monotonic() - t_pool

        # Advance the app's phase machine on this step's model: the Smart
        # indicator needs the labelset error cost, Stable the prediction flips
        # over the still-unlabeled pool, Span the atlas's coverage.
        if flow is not None:
            recent_steps.append((step, threshold))
            del recent_steps[:-SMART_WINDOW]  # the app regresses over the last 10 steps
            flow.record_step(
                _labelset_error_costs(recent_steps, good_votes, bad_votes, clips_dict, inclusion),
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
            # Blank on the standalone SVM trainers: they fit no head, so
            # naming one here would attribute the row to a head never trained.
            "head": head if trainer == "mlp" else "",
            "style": style or "",
            "prevalence_arm": prevalence_arm,
            "realized_prevalence": realized_prevalence,
            "t": t,
            "n_good": len(good_votes),
            "n_bad": len(bad_votes),
            "phase": flow.phase if flow is not None else "",
            "app_trained": 1 if (flow is None or app_has_detector(flow.phase)) else 0,
            "startup_schedule": startup_schedule or "",
            "acq_threshold": round(float(acq_threshold), 6),
            # Measured against the pool the selector ranks, not the test set, so
            # the pair answers "how much did the sampling position move".
            "acq_pool_percentile": _pool_percentile(pool_scores, acq_threshold),
            "report_pool_percentile": _pool_percentile(pool_scores, threshold),
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
            # The final model's haystack under the #3308 population convention:
            # the voted items dropped, exactly as `_safe_threshold_for_step`
            # dropped them from the fold haystacks - so every fold-anchored
            # variant fit below stays paired with the shipped cut's population.
            _voted_step_ids = set(good_votes) | set(bad_votes)
            sim_fit_scores: list[float] | None = None
            if sim_pooled_scores is not None:
                sim_fit_scores = [
                    s for i, s in zip(sim_pooled_ids, sim_pooled_scores, strict=True) if i not in _voted_step_ids
                ]
            # One extra row per safe-threshold GMM variant (issue #2799), all
            # evaluated against the same held-out max-pooled test scores.
            if X_sim_image is not None and sim_pooled_scores is not None:
                sim_image_ids = sorted(sim_ids)
                sim_image_scores = np.asarray(step.predict(X_sim_image)).ravel().tolist()
                variant_rows, diag_rows = _safe_gmm_variant_rows(
                    details,
                    base_scores,
                    base_labels,
                    {"pooled": sim_pooled_scores, "image": sim_image_scores},
                    {
                        "pooled": np.array([pool_labels[cid] for cid in sim_pooled_ids], dtype=np.float64),
                        "image": np.array([pool_labels[cid] for cid in sim_image_ids], dtype=np.float64),
                    },
                    inclusion,
                    n_pool_rows=metric_rows[0]["n_pool_rows"],
                    schedule=blend_schedule,
                )
                metric_rows.extend(variant_rows)
                if cut_diag_sink is not None:
                    for dr in diag_rows:
                        cut_diag_sink.append({**base_row, **dr})
            # One extra row per mix-in schedule (issue #2841), on the production
            # cut.  Independent of the cut-variant rows above: the schedule
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
            # The #2897 fold-count arms.  Unlike the arms above these need no
            # sim scores of their own - they re-cut fold orderings the step
            # already trained - so they run whether or not safe_thresholds is on;
            # the pooled sim scores, when present, only add the blended arm.
            if fold_count_variants:
                metric_rows.extend(
                    _fold_count_variant_rows(
                        details,
                        base_scores,
                        base_labels,
                        inclusion,
                        n_pool_rows=metric_rows[0]["n_pool_rows"],
                        counts=fold_count_variants,
                        sim_pooled_scores=sim_pooled_scores,
                        schedule=blend_schedule,
                        sim_fit_scores=sim_fit_scores,
                    )
                )
            # The #2852 anchored-mixture arms, paired against the same test
            # scores (and against pooled_mid / xcal_only above).
            if anchored_thresholds and sim_pooled_scores is not None:
                metric_rows.extend(
                    _anchored_variant_rows(
                        details,
                        base_scores,
                        base_labels,
                        sim_pooled_scores,
                        sim_pooled_ids,
                        list(good_votes),
                        list(bad_votes),
                        sim_fold_haystacks,
                        inclusion,
                        n_pool_rows=metric_rows[0]["n_pool_rows"],
                        weights=anchored_weights if anchored_weights is not None else list(_ANCHORED_WEIGHTS),
                        rules=anchored_rules if anchored_rules is not None else list(_ANCHORED_RULES),
                        fold_combines=(
                            anchored_fold_combines
                            if anchored_fold_combines is not None
                            else list(_ANCHORED_FOLD_COMBINES)
                        ),
                        fold_anchored=anchored_fold_arms,
                    )
                )
            for mr in metric_rows:
                rows.append({**base_row, **mr, **timing_cols})
            # The near-free inclusion-budget sweep, into the side sink.
            if inclusion_sweep_ks and sweep_sink is not None:
                for sr in _inclusion_sweep_rows(details, base_scores, base_labels, inclusion_sweep_ks):
                    sweep_sink.append({**base_row, **sr})
            # The #2865 cut-rule x inclusion sweep, into its own side sink.
            # Needs the per-fold sim scores the fold-anchored arms use, so it
            # rides the same `sim_pooled_scores is not None` gate they do.
            if cut_inclusion_ks and cut_inclusion_sink is not None and sim_pooled_scores is not None:
                for cr in _cut_inclusion_rows(
                    details,
                    base_scores,
                    base_labels,
                    sim_fold_haystacks,
                    sim_fit_scores if sim_fit_scores is not None else sim_pooled_scores,
                    cut_inclusion_ks,
                    weights=anchored_weights if anchored_weights is not None else list(_ANCHORED_WEIGHTS),
                    rules=anchored_rules if anchored_rules is not None else list(_ANCHORED_RULES),
                    fold_combines=(
                        anchored_fold_combines if anchored_fold_combines is not None else list(_ANCHORED_FOLD_COMBINES)
                    ),
                    qtilt_steps=(
                        cut_inclusion_qtilt_steps if cut_inclusion_qtilt_steps is not None else [FOLD_ANCHOR_QTILT_STEP]
                    ),
                ):
                    cut_inclusion_sink.append({**base_row, **cr})
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
    safe_thresholds: bool = True,
    calibrate_count: int = 2,
    calibration_fraction: Optional[float] = None,
    region_voting: bool = False,
    strategies: Optional[list[str]] = None,
    max_steps: Optional[int] = None,
    atlas_min_node_size: int = 20,
    seed_scores: Optional[dict[str, dict[str, dict[int, float]]]] = None,
    trainers: Optional[list[str]] = None,
    prevalence_arms: Optional[list[Optional[float]]] = None,
    styles: Optional[list[Optional[str]]] = None,
    autopilot_fidelity: bool = True,
    startup_schedule: Optional[str] = None,
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
        safe_thresholds: The shipped fused threshold path; on by default,
            matching the app.  ``False`` is the no-fusion control arm.
            (see :func:`simulate_voting_iterations`).
        calibrate_count: Number of random Train/Calibrate splits for threshold
            calibration (default 2).
        calibration_fraction: Fraction of labelled data reserved for
            calibration in each split.  ``None`` (default) resolves per
            dataset to the app's per-space split (see
            :func:`simulate_voting_iterations`).
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
            ``[None]``, which resolves per dataset to whatever the **app** does
            - ``max_patch`` on a patch dataset, the single-vector path
            otherwise; pass e.g. ``["whole_image", "max_patch"]`` to pin the
            Max-Patch experiment arms explicitly.  The *resolved* name is
            recorded in the ``style`` column (``""`` only when no style ran).
        autopilot_fidelity: Follow the app's own Autopilot phase machine
            (default ``True``); see :func:`simulate_voting_iterations`.  Pass
            ``False`` to reproduce studies published before the flow was
            aligned.
        startup_schedule: A parameterised Autopilot opening (issue #3267); see
            :func:`simulate_voting_iterations`.  ``None`` (default) is the app's
            own opening.  Requires a *seed_scores* entry for every cell run.

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
                                    startup_schedule=startup_schedule,
                                )
                                all_rows.extend(rows)

    return pd.DataFrame(all_rows, columns=pd.Index(list(_VOTING_COLUMNS)))


def run_voting_iterations_eval_from_pickles(
    dataset_paths: dict[str, str],
    seeds: list[int],
    categories: Optional[dict[str, list[str]]] = None,
    inclusion: int = 0,
    sim_fraction: float = 0.5,
    safe_thresholds: bool = True,
    calibrate_count: int = 2,
    calibration_fraction: Optional[float] = None,
    region_voting: bool = False,
    strategies: Optional[list[str]] = None,
    max_steps: Optional[int] = None,
    atlas_min_node_size: int = 20,
    seed_scores: Optional[dict[str, dict[str, dict[int, float]]]] = None,
    trainers: Optional[list[str]] = None,
    prevalence_arms: Optional[list[Optional[float]]] = None,
    styles: Optional[list[Optional[str]]] = None,
    autopilot_fidelity: bool = True,
    startup_schedule: Optional[str] = None,
) -> pd.DataFrame:
    """Convenience wrapper that loads datasets from pickle files.

    Args:
        dataset_paths: Mapping of dataset name to pickle file path.
        seeds: List of random seeds.
        categories: Optional category filter (see :func:`run_voting_iterations_eval`).
        inclusion: Inclusion setting in ``[-10, 10]``.
        sim_fraction: Fraction of medias for simulation.
        safe_thresholds: The shipped fused threshold path; on by default,
            matching the app.  ``False`` is the no-fusion control arm.
        calibrate_count: Number of random Train/Calibrate splits for threshold
            calibration (default 2).
        calibration_fraction: Fraction of labelled data reserved for
            calibration in each split.  ``None`` (default) resolves per
            dataset to the app's per-space split (see
            :func:`simulate_voting_iterations`).
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
        startup_schedule=startup_schedule,
    )
