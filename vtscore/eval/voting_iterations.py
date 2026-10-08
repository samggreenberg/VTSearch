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
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from vtscore.training.thresholds import FoldAnchoredCut, LineBudgets

from vtscore.detectors.cost_trend import SMART_INCLUSION, smart_cut
from vtscore.embedding.media_vectors import media_embedding
from vtscore.eval.al_strategies import ALContext, band_pick, is_autopilot_strategy, select_next
from vtscore.eval.autopilot_flow import SMART_WINDOW, AutopilotFlow, app_has_detector
from vtscore.eval.startup_schedule import StartupState, parse_startup_schedule, round_cut
from vtscore.eval.arms_anchored import (
    _ANCHORED_FOLD_COMBINES,
    _ANCHORED_RULES,
    _ANCHORED_WEIGHTS,
    _anchored_variant_rows,
)
from vtscore.eval.arms_fit_quality import _fit_quality_rows
from vtscore.eval.live_threshold_rules import check_live_threshold
from vtscore.eval.live_threshold_rules import live_threshold as retired_live_threshold
from vtscore.eval.arms_fold_count import _fold_count_variant_rows, parse_fold_count_schedule
from vtscore.eval.arms_inclusion import _cut_inclusion_rows, _inclusion_sweep_rows
from vtscore.eval.arms_safe_gmm import _safe_gmm_variant_rows
from vtscore.eval.arms_schedule import _schedule_variant_rows
from vtscore.detectors.centroid_head import CENTROID_THRESHOLD
from vtscore.detectors.label_quota import TIER_CENTROID
from vtscore.detectors.label_quota import label_quota as label_quota_tier
from vtscore.eval.row_metrics import operating_metrics, round6
from vtscore.eval.step_model import (
    APP_TRAINER,
    HEADS,
    PRODUCTION_HEAD,
    StepModel,
    resolve_trainer_name,
    score_sim_set_with_model,
)
from vtscore.eval.step_trainers import (
    _build_eval_atlas,
    _centroid_step,
    _labelset_error_costs,
    _score_pool,
    _train_and_calibrate,
)
from vtscore.eval import scale_bands
from vtscore.eval.labels import evaluable_pool, media_is_positive
from vtscore.eval.score_dumps import maybe_dump_predictions
from vtscore.eval.voting_columns import (
    FBETA_COLUMNS,
    FIT_QUALITY_STRIDE_DEFAULT,
    STOPPING_MARGIN_COLUMNS,
    VOTING_COLUMNS,
)
from vtscore.training.blend_schedules import BlendContext
from vtscore.training.thresholds import (
    WEAK_CHECK_COOLDOWN,
    WEAK_CHECK_MIN_VOTES,
    WEAK_SEPARATION_D,
    LineRanking,
    SpotCheck,
    balance_line,
    balance_schedule,
    balance_state,
    check_shape,
    fbeta_count,
    fit_labels_line,
    weak_check_due,
    ACQUISITION_INCLUSION_OFFSET,
    CALIBRATION_SPLIT_SEED,
    apply_vote_exclusion,
    FOLD_ANCHOR_QTILT_STEP,
    NO_GOOD_THRESHOLD,
    NO_BALANCE,
    ACQUISITION_ARGMAX_FACTOR,
    ACQUISITION_TARGET_PRECISION,
    CHECK_SHAPES,
    CHECK_TRIM,
    acquisition_inclusion,
    acquisition_threshold,
    calculate_safe_threshold,
    line_inclusion,
    resolve_line_knobs,
    target_precision_threshold,
    threshold_from_fold_orderings,
    walk_positives,
)


#: The detection geometry a live detector uses on a patch dataset - the style an
#: unspecified ``style=`` resolves to below.  Named rather than inlined so that
#: "is this run measuring the shipped geometry?" is a question something can
#: *ask*: `scripts/experiments/preflight.sh` compares a study's configured
#: styles against it, the way it compares a study's head against
#: :data:`PRODUCTION_HEAD`.  The HAC hybrids in
#: :mod:`vtscore.eval.patch_styles` are experiment-only arms; #2886 removed the
#: region tree from ingest.
PRODUCTION_PATCH_STYLE: str = "max_patch"

#: The one detection style the #3322 skyline is defined for in v1 - spelled here
#: rather than imported for the same reason :data:`_SAFE_GMM_VARIANTS` spells its
#: rule names: this module is deliberately import-light at module scope.
#: ``test_skyline_arm`` pins it against :class:`~vtscore.eval.patch_styles.WholeImageStyle`.
_WHOLE_IMAGE_STYLE: str = "whole_image"


#: The **supervised-skyline** arms (issue #3322), tagged in the ``gmm_variant``
#: column exactly like every other variant family, and emitted **once per run**
#: rather than once per step: a skyline is vote-independent, so its row is a
#: constant of the ``(cell, seed)`` and repeating it 150 times would only invite
#: a reader to average it against itself.
#:
#: * ``skyline_train_full`` - the **primary** arm and the one the decomposition
#:   is defined against.  The same head, trained through the same trainer, on the
#:   **entire simulation split with full ground-truth labels**, evaluated on the
#:   untouched test split.  The standard supervised-skyline arm from the
#:   active-learning literature: same hypothesis class, same features, full
#:   supervision, disjoint eval, so its cost is "how learnable is this class in
#:   this embedding with this head" and nothing else.
#: * ``skyline_test_xfit`` - the optional **bracket** partner, cross-fitted over
#:   the test split (train on K-1 folds, score the held-out one).  Never a naive
#:   train-on-test fit: a ~769-parameter linear head on a test set of comparable
#:   size can shatter near-arbitrary labelings, so a naive fit would report
#:   ``d / n_test`` under the name "learnability" and hand back near-zero cost on
#:   a class nothing can learn.  Cross-fitting is the SVM analogue of
#:   :func:`~vtscore.eval.transfer_rules.honest_test_oracle`, which does the same
#:   thing one level down for the *cut*.
SKYLINE_TRAIN_FULL: str = "skyline_train_full"
SKYLINE_TEST_XFIT: str = "skyline_test_xfit"
SKYLINE_ARMS: tuple[str, ...] = (SKYLINE_TRAIN_FULL, SKYLINE_TEST_XFIT)

#: ``threshold_provenance`` on a skyline row.  A skyline is a statement about a
#: **ranking**, so it is deliberately *not* routed through a calibrated cut - that
#: would re-mix in exactly the term ``regret`` already isolates.  Its threshold is
#: the test oracle's, which makes ``cost == oracle_cost`` and ``regret == 0`` on
#: the row **by construction**: read a skyline row's ``oracle_cost``, never its
#: ``regret``.
SKYLINE_PROVENANCE: str = "skyline_test_oracle"

#: RNG salt for the cross-fitted arm's test-set partition.  Combined with the
#: run's own ``seed`` so the folds are reproducible per cell, and kept off the
#: trajectory's ``RandomState`` so turning the arm on cannot move a single vote.
_SKYLINE_SEED: int = 3322


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


#: Minimum positives an arm must retain after prevalence downsampling.  Below
#: this the held-out test set has too few positives for a stable FNR estimate,
#: so the arm is skipped rather than reported with a noisy denominator.
_MIN_PREVALENCE_POSITIVES = 15


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


def thin_haystack(
    clips_dict: dict[int, dict[str, Any]],
    sim_ids: list[int],
    target_category: str,
    prevalence: float,
    seed: int,
) -> list[int]:
    """*sim_ids* with its negatives thinned so positives are ~*prevalence* of it (#4184).

    The **haystack** arm: what the cut rules see changes, what they are graded
    on does not.  Only the simulation half is touched - after the split, from
    its own RNG - so the held-out test set, every band's cohort and the run's
    own ``RandomState(seed)`` stream are exactly the natural run's, and a cell
    pairs with its natural twin.  Cost is FPR + FNR, which does not depend on
    prevalence, so the pairing is a like-for-like comparison of cut rules.

    All positives are kept.  A pool already at or above *prevalence* comes back
    unchanged.  ``text_baseline.py`` calls this too, so the click-0 notch is cut
    over the same thinned pool the rungs vote in.
    """
    import numpy as np  # noqa: PLC0415

    if not 0.0 < prevalence < 1.0:
        raise ValueError(f"haystack_prevalence must be in (0, 1), got {prevalence!r}")
    pos = [cid for cid in sim_ids if media_is_positive(clips_dict[cid], target_category)]
    neg = [cid for cid in sim_ids if not media_is_positive(clips_dict[cid], target_category)]
    keep_neg = int(round(len(pos) * (1.0 - prevalence) / prevalence))
    if not pos or keep_neg >= len(neg):
        return list(sim_ids)
    # A stream of its own, so thinning draws nothing from the run's RNG.
    rng = np.random.RandomState([seed, 4184])
    kept = set(int(c) for c in rng.choice(np.array(sorted(neg), dtype=np.int64), size=keep_neg, replace=False))
    return [cid for cid in sim_ids if cid in kept or media_is_positive(clips_dict[cid], target_category)]


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


def _pool_uncertainty(
    step: StepModel, pool_ids: list[int], clips_dict: dict[int, dict[str, Any]]
) -> dict[int, float] | None:
    """``{pool_id: std}`` from ``step.predict_std``, or ``None`` when it has none.

    Whole-image vectors only: the ``gp_*`` trainers that set ``predict_std``
    fit single vectors, exactly as the SVM path does, so there is no region
    geometry to pool here.
    """
    import numpy as np  # noqa: PLC0415

    if step.predict_std is None or not pool_ids:
        return None
    embs = np.array([media_embedding(clips_dict[cid]) for cid in pool_ids])
    spread = np.asarray(step.predict_std(embs), dtype=np.float64).ravel()
    return {cid: float(s) for cid, s in zip(pool_ids, spread)}


def _no_recut(_inclusion: float) -> None:
    """The re-cut of a step with no fold-anchored fit: there is none to re-derive.

    Handed to :func:`~vtscore.detectors.cost_trend.smart_cut`, which then keeps
    the step's reporting line: the balance's kept set, the schedule blend, a
    retired rung's cut, or the conformal cut of an arm with safe thresholds off.
    Any line no inclusion drew asks, and on the default arm that is every line
    the balance keeps (#4272, #4413).  Where the step's folds split but none yielded a
    fold-anchored fit, the app's seam (``recut_detector_threshold``) would
    re-cut the fold orderings instead - a difference ``progress.smart_status``
    declares, which reaches the default arm only on such a step.
    """
    return None


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
    return round6(idx / len(descending))


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


def _line_columns(details: dict[str, Any]) -> dict[str, Any]:
    """The balance columns of a step's row: the beta, the set its line keeps, and the check's ranges.

    The state is the app's own (:func:`~vtscore.training.thresholds.balance_state`,
    #4413), built by :func:`_safe_threshold_for_step`.  ``floor_status`` is
    empty and every count -1 / range NaN on a step with no balance line - the
    Inclusion arm, safe thresholds off, or nothing scored yet.  (The
    ``floor_*`` names predate the balance; they report the balance's state.)
    """
    nan = float("nan")
    state = details.get("line_state")
    beta = details.get("beta")
    if beta is None or state is None:
        return {
            "beta": beta if beta is not None else nan,
            "floor_status": "",
            "floor_count": -1,
            "range_lo": nan,
            "range_hi": nan,
            "check_labelled": -1,
            "check_right": -1,
            "check_stale": -1,
            "check_audited": -1,
        }
    # The kept set's precision range: what the range columns have always read.
    rng = state.precision
    return {
        "beta": beta,
        "floor_status": state.status,
        "floor_count": state.count,
        "range_lo": round6(rng.lo) if rng is not None else nan,
        "range_hi": round6(rng.hi) if rng is not None else nan,
        "check_labelled": rng.labelled if rng is not None else -1,
        "check_right": rng.right if rng is not None else -1,
        "check_stale": (1 if state.stale else 0) if rng is not None else -1,
        # The set the walk audited (#4427): under the advisory shape not the
        # set the line keeps, which ``floor_count`` reports.
        "check_audited": state.audited if rng is not None else -1,
    }


#: Autopilot's opening (#4496): the phases on the text or example sort, where the
#: app trains no detector, so a weak-separation check under ``weak_phase="learned"``
#: waits for the flow to leave them.
_OPENING_PHASES = frozenset({"good", "bad", "more"})

#: Where Autopilot's ``more`` walk can draw (#4637): ``"seed"`` is the app, the
#: top of the text sort; ``"detector"`` the top of the step's detector ranking.
MORE_WALKS = ("seed", "detector")

#: The pick-log phase of a band pick (#4482): an ordinary click, past the opening,
#: drawn the way the spot check draws (:func:`~vtscore.eval.al_strategies.band_pick`).
BAND_PHASE = "band"


def _preference_line_for_step(
    ranking: LineRanking,
    details: dict[str, Any],
    beta: float | None,
    check: "SpotCheck | None",
    labels: "Mapping[int, bool] | None",
    line_shape: "str | None" = None,
) -> tuple[float | None, str]:
    """The line the arm's balance draws over the step's ranking, and its provenance; ``(None, "")`` with none.

    The app's own rule: the labels' line (#4452), the class model the step's
    fold orderings imply cut at the prevalence it estimates on this ranking,
    where the arm forces no check shape.  A forced shape (``walk_shape``)
    keeps the count line the app drew before #4452 (#4413): the finished
    F-beta walk's set where the shape lets it move the line, else the
    mixture's F-beta argmax under the balance's cap.  Leaves the state the
    row's line columns read in ``details["line_state"]`` and the beta in
    ``details["beta"]``.
    """
    if beta is None:
        return None, ""
    details["beta"] = beta
    if line_shape is None:
        # The app's line (#4452): the labels' class model (the step's fold
        # orderings) cut at the prevalence it estimates on this ranking - the
        # Train side.  ``details["find_line"]`` carries it to the test side,
        # which re-estimates the prevalence on the withheld half the way a
        # Find on a new corpus does.
        labels_line = fit_labels_line(
            details.get("fold_orderings") or None, ranking.scores, ranking.ids.tolist(), labels or {}
        )
        details["find_line"] = labels_line
        if labels_line is not None:
            kept = labels_line.threshold(beta)
            details["line_state"] = balance_state(beta, check, ranking, threshold=kept)
            details["train_prevalence"] = labels_line.prevalence
            return kept, "balance"
        details["line_state"] = balance_state(beta, check, ranking)
        return None, ""
    # A forced check shape (``walk_shape``): the count line the app drew
    # before #4452, kept as a measurement arm.
    proposal = fbeta_count(ranking, beta, labels or {})
    details["line_state"] = balance_state(beta, check, ranking, proposal=proposal, shape=line_shape)
    return balance_line(ranking, beta, check, proposal=proposal, shape=line_shape), "balance"


def _safe_threshold_for_step(
    threshold: float,
    step: StepModel,
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
    exclusion_min_remainder: float | None = None,
    cut_rule: str | None = None,
    check: "SpotCheck | None" = None,
    labels: "Mapping[int, bool] | None" = None,
    beta: float | None = None,
    line_shape: "str | None" = None,
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

    *voted_ids* drives the app's #3308 exclusion, and drives it through the app's
    own :func:`~vtscore.training.thresholds.apply_vote_exclusion` /
    :func:`~vtscore.training.thresholds.drop_voted` rather than a copy of them:
    the voted items are dropped from every haystack the fold-anchored estimator
    fits on - the per-fold arrays (which are also what the returned
    ``fold_haystacks`` carry, so the variant grids inherit the same population
    convention) and the final model's realization sample - unless the remainder
    falls under the floor, in which case the whole step keeps its full
    haystacks.  The *returned* sim scores stay complete: they feed evaluation
    and acquisition, not the fit.

    *exclusion_min_remainder* is the **#3312 arm knob** and the one thing here
    that is allowed to differ from the app: ``None`` (every production caller,
    and the harness's own default arm) resolves to the shipped floor, ``0``
    excludes unconditionally, and ``math.inf`` switches the exclusion off
    entirely - the pre-#3308 baseline.  Because the resolution happens inside
    :func:`~vtscore.training.thresholds.resolve_exclusion_floor`, the default
    arm cannot drift from the app even though the knob exists.

    *cut_rule* is the #3557 run-level arm knob, on the same terms: ``None``
    resolves to the app's :data:`~vtscore.training.thresholds.FOLD_ANCHOR_CUT_RULE`
    inside this function, so the default arm is production by construction.

    **The line is drawn where the app draws it.**  *beta* is the resolved
    balance (``None`` for the Inclusion arm).  Under a balance the line keeps
    a set (#4272, #4413): the top *count* unvoted items of the sim set this
    final model scored, where *count* is the set *check* - the run's spot
    check - ended on where the check's shape lets it move the line, or else
    the mixture's F-beta argmax under the balance's cap;
    :func:`~vtscore.training.thresholds.balance_line` is the app's own rule,
    called, not copied, and its ranking rides out in
    ``details["line_ranking"]`` for the check to draw its bands from, with the
    balance's state in ``details["line_state"]`` for the row.  With no
    balance the line is the fold-anchored cut at *inclusion* through the shared
    :func:`~vtscore.training.thresholds.reporting_line`.  Either way the line
    rides out in ``details["reporting_line"]`` so the acquisition cut can take
    its origin from it.
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.training.thresholds import (  # noqa: PLC0415
        FOLD_ANCHOR_CUT_RULE,
        ReportingLine,
        drop_voted,
        fit_fold_anchored_cut,
        reporting_line,
    )

    final_model = step.torch_model
    # The final model's pass over the haystack (#3314).  Real app work - it is
    # what the shipped cut is realized on and what the browse view ranks - and
    # it is K-INDEPENDENT, so it belongs in the denominator of a cost ratio and
    # not in `cal_seconds`.  Untimed it would simply be missing from the step,
    # which makes every fold count look more expensive than it is.
    t_final = time.monotonic()
    if style_obj is not None or region_aware:
        assert final_model is not None
        ids, all_scores = score_sim_set_with_model(
            final_model, region_aware, sim_clips, X_all_clips, sim_ids, style_obj
        )
    else:
        # Trainer-agnostic: the SVM arms have no torch model to forward.
        ids = sorted(sim_ids)
        all_scores = np.asarray(step.predict(np.asarray(X_all_clips))).ravel().tolist()
    details["final_score_seconds"] = time.monotonic() - t_final

    # The floor decision is taken once, on the final model's remainder, and
    # applies to every haystack in the step - all-or-nothing, so the fold and
    # final populations can never diverge.  ``apply_vote_exclusion`` is the
    # app's own decision function, so the default arm reproduces production by
    # construction rather than by a comment promising it does.
    fit_final, excluding = apply_vote_exclusion(all_scores, ids, voted_ids, min_remainder=exclusion_min_remainder)

    def _hay(scores: list[float], score_ids: list[int]) -> "np.ndarray":
        """One fold's haystack under this step's exclusion decision."""
        return drop_voted(scores, score_ids, voted_ids or ()) if excluding else np.asarray(scores, dtype=np.float64)

    fold_models = details.get("fold_models") or []
    fold_orderings = details.get("fold_orderings") or []
    n_folds = min(len(fold_models), len(fold_orderings))
    fold_haystacks: list[Any] = []
    # Per-fold haystack scoring seconds (#3314).  Scoring the sim set with each
    # fold model is a real, K-proportional part of the calibration a live run at
    # K pays for - the shipped rule anchors every fold's mixture on that fold's
    # own haystack - and it is paid *here*, outside `train_seconds`,
    # `xcal_seconds`, `pool_score_seconds` and `test_score_seconds`.  Left
    # unmeasured it is invisible to every cost model built out of those four,
    # which would make a fold-count affordability ceiling read a fraction of
    # what K actually costs.
    haystack_seconds: list[float] = []
    for model in fold_models[:n_folds]:
        t_hay = time.monotonic()
        if callable(model) and not hasattr(model, "parameters"):
            # A standalone trainer's fold model (the #3959 GP folds): a bare
            # ``predict`` callable on the whole-image matrix, in the same id
            # order the final model's trainer-agnostic pass above uses.
            fids = sorted(sim_ids)
            fscores = np.asarray(model(np.asarray(X_all_clips))).ravel().tolist()
        else:
            fids, fscores = score_sim_set_with_model(model, region_aware, sim_clips, X_all_clips, sim_ids, style_obj)
        hay = _hay(fscores, fids)
        haystack_seconds.append(time.monotonic() - t_hay)
        fold_haystacks.append(hay)

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
            t_hay = time.monotonic()
            fids, fscores = score_sim_set_with_model(model, region_aware, sim_clips, X_all_clips, sim_ids, style_obj)
            hay = _hay(fscores, fids)
            haystack_seconds.append(time.monotonic() - t_hay)
            extended.append(hay)
        fold_data["haystacks"] = extended
        # Aligned with `haystacks`, so a K-prefix of one is a K-prefix of the
        # other.  Every entry is one scoring pass over the same sim set, so the
        # seconds are comparable across folds and across K by construction.
        fold_data["haystack_seconds"] = haystack_seconds

    cut = (
        fit_fold_anchored_cut(
            fold_haystacks,
            fold_orderings[:n_folds],
            fit_final.tolist(),
            cut_rule=cut_rule if cut_rule is not None else FOLD_ANCHOR_CUT_RULE,
        )
        if fold_haystacks
        else None
    )
    # The ranking the line keeps a set of: every scored sim item, the voted
    # ones marked, as ``_fused_threshold`` parks it on the detector context.
    ranking = LineRanking.from_scores(ids, all_scores, voted_ids or ())
    details["line_ranking"] = ranking
    kept, provenance = _preference_line_for_step(ranking, details, beta, check, labels, line_shape)
    if kept is not None:
        # No inclusion drew this line: acquisition derives its origin from it.
        details["reporting_line"] = ReportingLine(kept, None, None)
        return kept, all_scores, ids, fold_haystacks, provenance, cut
    line = reporting_line(cut, None, inclusion_value=inclusion, min_precision=None)
    details["reporting_line"] = line
    if cut is not None and line.threshold is not None:
        return line.threshold, all_scores, ids, fold_haystacks, cut.provenance, cut
    blended = calculate_safe_threshold(_blend_xcal_input(threshold, details), all_scores, ctx, schedule=schedule)
    return blended, all_scores, ids, fold_haystacks, "gmm_blend", None


def _check_test_bands(
    test_bands: Optional[list[str] | str],
    target_category: str,
    target_prevalence: Optional[float],
) -> None:
    """Refuse a ``test_bands`` request the run cannot honour, at the door.

    Two of them.  A target with no band suffix has no size axis, so there is
    nothing to break down and an empty breakdown would read as three bands that
    happened to be empty.  And ``target_prevalence`` draws from the split RNG
    before the split and changes which images are in the pool at all, so a
    sibling band's replayed cohort would no longer be the one that band's own
    arm holds out -- a 3x3 table whose cells are not comparable, which is worse
    than no table.
    """
    if not test_bands:
        return
    if scale_bands.parse_cell(target_category)[1] is None:
        raise ValueError(
            f"test_bands was asked for on {target_category!r}, which carries no band suffix; "
            "cross-band testing needs a scale-banded cell such as 'car@small'"
        )
    if target_prevalence is not None:
        raise ValueError(
            "test_bands and target_prevalence cannot both be set: prevalence thinning draws "
            "from the split RNG and changes the pool, so a sibling band's cohort would no "
            "longer be the one that band's own arm holds out, and the 3x3 table would not pair"
        )


def _check_train_mix(
    train_mix: "Optional[str | dict[str, float]]",
    target_category: str,
    target_prevalence: Optional[float],
) -> None:
    """Refuse a ``train_mix`` request the run cannot honour, at the door (#4160).

    The target has to name the mixed cell the retag writes (``car@mix-equal``),
    so a row's ``category`` says which arm it is. A pure band there would put a
    mixed arm's rows under a pure arm's name. Prevalence thinning is refused for
    the reason :func:`_check_test_bands` gives: the mix's split is a replay of
    the pure arms' splits and thinning moves the pool the replay reads.
    """
    if train_mix is None:
        return
    band = scale_bands.parse_cell(target_category)[1]
    if not scale_bands.is_mix_band(band):
        raise ValueError(
            f"train_mix needs a mixed target such as 'car@mix-equal', got {target_category!r}; "
            "a pure band there would report a mixed arm under a pure arm's name"
        )
    if target_prevalence is not None:
        raise ValueError(
            "train_mix and target_prevalence cannot both be set: the mix replays the pure "
            "arms' splits, and prevalence thinning changes the pool that replay reads"
        )


def _resolve_band_cohorts(
    unfiltered: dict[int, dict[str, Any]],
    target_category: str,
    *,
    test_bands: Optional[list[str] | str],
    sim_fraction: float,
    seed: int,
    own_test_ids: list[int],
    target_prevalence: Optional[float],
) -> dict[str, list[int]]:
    """The per-band held-out positive cohorts, or ``{}`` when the run asks for none.

    ``test_bands="auto"`` takes every band the class has; a list names them.

    **Refused rather than approximated under prevalence thinning.**  A cohort is
    the set the arm trained on *that* band would have held out, which is only
    true while the split is a function of ``(pool, seed)`` alone.
    ``target_prevalence`` draws from the same RNG before the split and changes
    which images are in the pool at all, so the replay would silently pick a
    different cohort -- and an off-diagonal reading against a cohort that is not
    the diagonal's is a 3x3 table whose cells cannot be compared.  Say so
    instead: an unavailable arm is visible, a mis-paired one is not.

    The own band's cohort is asserted against the harness's real split, so the
    replay in :func:`~vtscore.eval.scale_bands.holdout_ids` cannot drift from
    :func:`_split_media_ids` without a run failing here.
    """
    if not test_bands:
        return {}
    cls, band = scale_bands.parse_cell(target_category)
    assert band is not None  # `_check_test_bands` refused this before any work ran
    wanted = None if test_bands == "auto" else list(test_bands)
    cohorts = scale_bands.band_cohorts(unfiltered, target_category, sim_fraction=sim_fraction, seed=seed, bands=wanted)
    stray = scale_bands.unreportable_bands(unfiltered, cls)
    if stray:
        import warnings  # noqa: PLC0415

        warnings.warn(
            f"{cls!r} carries bands {stray} that have no reported column "
            f"(reported: {list(scale_bands.REPORTED_BANDS)}); they are measured into nothing",
            RuntimeWarning,
            stacklevel=2,
        )
    own = cohorts.get(band)
    if own is not None:
        expected = sorted(cid for cid in own_test_ids if media_is_positive(unfiltered[cid], target_category))
        if sorted(own) != expected:
            raise AssertionError(
                "the replayed own-band cohort is not the harness's own held-out positives; "
                "scale_bands.holdout_ids has drifted from _split_media_ids"
            )
    return cohorts


def _score_media_ids(
    step: StepModel,
    clips_dict: dict[int, dict[str, Any]],
    ids: list[int],
    *,
    region_aware: bool = False,
    style_obj: Any = None,
) -> list[float]:
    """Score *ids* with *step*, in whichever geometry the run is using.

    Extracted from :func:`_evaluate_on_test` so the cross-band cohorts (#4044)
    are scored by the **same** rule as the headline test set rather than by a
    second copy of the three branches.  A per-band FNR read off a different
    scoring geometry than the FNR it sits beside would be the one number in the
    row that is not comparable with its neighbours.
    """
    import numpy as np  # noqa: PLC0415

    if not ids:
        return []
    if style_obj is not None:
        # Explicit detection style (see vtscore.eval.patch_styles): the style
        # owns the whole image-scoring rule (whole-image / region max-pool /
        # raw-patch max-pool), replacing both branches below.
        assert step.torch_model is not None
        clips = {cid: clips_dict[cid] for cid in ids}
        score_map = style_obj.score_media(step.torch_model, clips)
        return [score_map[cid] for cid in ids]
    if region_aware:
        from vtscore.detectors.training import score_media_with_model  # noqa: PLC0415

        assert step.torch_model is not None
        clips = {cid: clips_dict[cid] for cid in ids}
        score_map = {r["id"]: r["score"] for r in score_media_with_model(step.torch_model, clips)}
        return [score_map[cid] for cid in ids]
    embs = np.array([media_embedding(clips_dict[cid]) for cid in ids])
    return np.asarray(step.predict(embs)).ravel().tolist()


def _band_metrics(
    step: StepModel,
    threshold: float,
    clips_dict: dict[int, dict[str, Any]],
    cohorts: Optional[dict[str, list[int]]],
    *,
    region_aware: bool = False,
    style_obj: Any = None,
    neg_ids: Optional[list[int]] = None,
    target_category: str = "",
) -> dict[str, float]:
    """FNR per size band at the shipped cut, plus the count behind each (#4044).

    **One threshold, one FPR, three FNRs.**  The bands of a class share their
    negative pool by construction, so a negative has no size *for the class* and
    there is exactly one false-positive rate per arm -- the row's own ``fpr``.
    What decomposes is the miss rate, because a positive does have a size.

    Measured at *threshold*, the cut the app would actually be running, so these
    columns do **not** move with ``pool_variant``: every row a step emits
    carries the same band breakdown, read at the shipped operating point.

    ``fnr_<band>`` for the arm's **own** band is the same quantity as the row's
    ``fnr`` restricted to positives, which is what
    ``test_the_own_band_column_agrees_with_the_headline_fnr`` checks -- the
    cheapest available guard against a cohort built off the wrong pool.

    *cohorts* is ``None`` when the run did not ask for the breakdown, and the
    columns are emitted anyway, all NaN -- the same shape
    :data:`~vtscore.eval.voting_columns.SKYLINE_COLUMNS` takes, and for the same
    reason: the result frame's schema is fixed, so a frame from a run with the
    breakdown and one from a run without it have to concatenate.  ``None`` and
    an empty dict are **not** the same answer: nobody looked, versus the band
    has no held-out positives, which is why the counts go NaN in the first case
    and 0 in the second.

    *neg_ids*, when given, are the run's held-out negatives, and each band also
    gets ``auroc_<band>``: its cohort ranked against those negatives (#4160).
    That is the threshold-free half. An arm can lose a band on its ranking or
    on its cut, and the FNR alone cannot tell the two apart. It costs one more
    scoring pass over the negatives, so it is opt-in (NaN otherwise).
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.eval.label_curve import _auroc  # noqa: PLC0415

    nan = float("nan")
    out: dict[str, float] = {}
    neg_scores = None
    if cohorts is not None and neg_ids:
        neg_scores = np.asarray(
            _score_media_ids(step, clips_dict, neg_ids, region_aware=region_aware, style_obj=style_obj),
            dtype=np.float64,
        )
    for band in scale_bands.REPORTED_BANDS:
        out[f"auroc_{band}"] = nan
        if cohorts is None:
            out[f"n_test_pos_{band}"] = nan
            out[f"fnr_{band}"] = nan
            out[f"recall_{band}"] = nan
            continue
        ids = cohorts.get(band) or []
        n = len(ids)
        out[f"n_test_pos_{band}"] = float(n)
        if not n:
            out[f"fnr_{band}"] = nan
            out[f"recall_{band}"] = nan
            continue
        scores = np.asarray(
            _score_media_ids(step, clips_dict, ids, region_aware=region_aware, style_obj=style_obj),
            dtype=np.float64,
        )
        # The per-image evidence behind a cross-band miss rate, so a report can
        # show which images an arm missed. Off unless VTS_DUMP_TEST_SCORES is set.
        maybe_dump_predictions(clips_dict, ids, scores, [1] * n, threshold, target_category, suffix=f"__band_{band}")
        # Every id in a cohort is a positive of its own band's cell, so the miss
        # rate is just the share scoring under the cut.
        fnr = float(np.mean(scores < threshold))
        out[f"fnr_{band}"] = round(fnr, 6)
        out[f"recall_{band}"] = round(1.0 - fnr, 6)
        if neg_scores is not None and len(neg_scores):
            labels = np.concatenate([np.ones(n), np.zeros(len(neg_scores))])
            out[f"auroc_{band}"] = round(_auroc(np.concatenate([scores, neg_scores]), labels), 6)
    return out


def _precision_frame(
    t: int,
    threshold: float,
    test_scores: Any,
    test_labels: Any,
    pool_scores: "list[float] | None",
    pool_ids: "list[int] | None",
    voted: "dict[int, float]",
    fold_orderings: list[Any],
    fold_haystacks: list[Any],
    cal_phase: "list[str] | None" = None,
    cal_vote: "list[int] | None" = None,
) -> dict[str, Any]:
    """Everything a live precision estimate could read at step *t*, plus the truth (#4220).

    The truth is the test half: its final-model scores and labels.  The
    evidence is what the app has: its own pool scores, each voted item's
    in-sample final-model score and label, and every calibration fold's
    held-out vote scores with that fold model's own haystack - the fold scores
    live on their model's scale, so an estimator that transfers them to the
    final model needs the haystack to rank them against.  Folds are stored flat
    with an index array because they differ in length.
    """
    import numpy as np  # noqa: PLC0415

    f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
    frame: dict[str, Any] = {
        "t": np.int32(t),
        "threshold": np.float32(threshold),
        "test_scores": f32(test_scores),
        "test_labels": np.asarray(test_labels, dtype=np.uint8),
        "pool_scores": f32(pool_scores if pool_scores is not None else []),
    }
    by_id = dict(zip(pool_ids or [], pool_scores or [], strict=False))
    vids = [v for v in voted if v in by_id]
    frame["vote_scores"] = f32([by_id[v] for v in vids])
    frame["vote_labels"] = np.asarray([voted[v] for v in vids], dtype=np.uint8)
    cal_s, cal_y, cal_f, hay_s, hay_f = [], [], [], [], []
    for k, (sc, lb) in enumerate(fold_orderings):
        cal_s.extend(sc)
        cal_y.extend(lb)
        cal_f.extend([k] * len(sc))
    for k, hay in enumerate(fold_haystacks):
        hay = np.asarray(hay).ravel()
        hay_s.extend(hay.tolist())
        hay_f.extend([k] * len(hay))
    frame["fold_cal_scores"] = f32(cal_s)
    frame["fold_cal_labels"] = np.asarray(cal_y, dtype=np.uint8)
    frame["fold_cal_fold"] = np.asarray(cal_f, dtype=np.uint8)
    # The phase that surfaced each calibration vote, aligned with the scores
    # (#4224); empty when the trainer could not say (e.g. the grouped path).
    frame["fold_cal_phase"] = np.asarray(cal_phase if cal_phase and len(cal_phase) == len(cal_s) else [], dtype="U12")
    frame["fold_cal_vote"] = np.asarray(cal_vote if cal_vote and len(cal_vote) == len(cal_s) else [], dtype=np.int64)
    frame["fold_hay_scores"] = f32(hay_s)
    frame["fold_hay_fold"] = np.asarray(hay_f, dtype=np.uint8)
    return frame


def _fresh_corpus_line(test: "LineRanking", pool: "LineRanking", vote_labels: "dict[int, bool]") -> dict[float, int]:
    """What the shipped unchecked line keeps on the test half at each preset balance (#4389, #4413).

    A cold Find over a corpus holding the session's votes: the test half plus
    the voted items at the final model's scores (read off *pool*, the
    session's ranking), the votes marked voted and anchoring the mixture, and
    the line at :func:`~vtscore.training.thresholds.balance_count` with the
    mixture's F-beta argmax as its proposal - the smaller of the cap and the
    argmax.  Capped at the test half's size.  Pure read.
    """
    from vtscore.eval.voting_columns import RANK_FRAME_BETAS  # noqa: PLC0415
    from vtscore.training.thresholds import balance_count, fbeta_count  # noqa: PLC0415

    in_test = set(test.ids.tolist())
    keep, scores = [], []
    for v in vote_labels:
        if int(v) in in_test:
            continue
        try:
            scores.append(pool.score_of(int(v)))
        except KeyError:  # an unscorable vote is out of the ranking, as it is out of the app's
            continue
        keep.append(int(v))
    corpus = LineRanking.from_scores([*test.ids.tolist(), *keep], [*test.scores.tolist(), *scores], keep)
    labels = {v: bool(vote_labels[v]) for v in keep}
    return {b: int(min(balance_count(b, None, fbeta_count(corpus, b, labels)), test.size)) for b in RANK_FRAME_BETAS}


def _labels_line_counts(test_scores: Any, find_on_test: Any, fallback_threshold: float | None) -> dict[float, int]:
    """What the app's labels line keeps on the test half at each preset beta (#4452, #4471).

    *find_on_test* is the labels line with its corpus side fitted on the test
    half - Find's own fit, the one the headline row cuts at the run's beta -
    so the count at the run's beta is the headline's returned set exactly.
    With no class model this step the retrain's *fallback_threshold* draws
    the line, the same set at every beta.  A count can be 0: the line may
    keep nothing.  -1 with neither.  ``score >= threshold``, as the headline
    reads it.
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.eval.voting_columns import RANK_FRAME_BETAS  # noqa: PLC0415

    s = np.asarray(test_scores, dtype=np.float64)
    out: dict[float, int] = {}
    for b in RANK_FRAME_BETAS:
        thr = float(find_on_test.threshold(b)) if find_on_test is not None else fallback_threshold
        out[b] = int(np.count_nonzero(s >= thr)) if thr is not None and np.isfinite(thr) else -1
    return out


def _rank_frame(
    kind: str,
    t: int,
    test_ids: Sequence[int],
    test_scores: Any,
    test_labels: Any,
    pool_ranking: "LineRanking | None" = None,
    voted: "Iterable[int]" = (),
    pool_labels: "dict[int, float] | None" = None,
    vote_labels: "dict[int, bool] | None" = None,
    find_on_test: Any = None,
    fallback_threshold: float | None = None,
) -> dict[str, Any]:
    """Where the positives sit in the test half's ranking and in the session's unvoted pool (#4357).

    Both rankings are :class:`~vtscore.training.thresholds.LineRanking` orders
    (score descending, ties by id, unscorable media left out), so the test
    half's top *K* is the set a balance's line keeps on a fresh corpus, and the
    pool's top *K* unvoted is the candidate the spot check samples.  With the
    session's votes (*vote_labels*, ``True`` = Good) it also records how many
    the shipped unchecked line keeps on the test half at each preset balance
    (:func:`_fresh_corpus_line`, #4389); -1 without them.  At each preset
    beta it records the arm's balance line: under the app's labels line
    (*find_on_test* or *fallback_threshold*, #4452) what that line keeps
    (:func:`_labels_line_counts`), else the count line a forced check shape
    draws.  A full-label skyline passes its own labels line as
    *find_on_test* (#4486).  Pure read.
    See :data:`~vtscore.eval.voting_columns.RANK_FRAME_COLUMNS`.
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.eval.voting_columns import RANK_FRAME_BETAS, beta_tag  # noqa: PLC0415

    def _ranks(ids: Any, label_of: Any) -> tuple[int, int, str]:
        pos = np.flatnonzero(np.fromiter((label_of(int(i)) >= 0.5 for i in ids), dtype=bool, count=len(ids)))
        return len(ids), int(pos.size), " ".join(str(int(r)) for r in pos)

    labels = dict(zip((int(i) for i in test_ids), (float(v) for v in test_labels), strict=True))
    test = LineRanking.from_scores(list(test_ids), test_scores)
    n_test, n_test_pos, test_ranks = _ranks(test.ids, labels.__getitem__)
    n_pool, n_pool_pos, pool_ranks = -1, -1, ""
    if pool_ranking is not None and pool_labels is not None:
        n_pool, n_pool_pos, pool_ranks = _ranks(pool_ranking.unvoted_ids(voted), pool_labels.__getitem__)
    line_k: dict[float, int] = dict.fromkeys(RANK_FRAME_BETAS, -1)
    if pool_ranking is not None and vote_labels:
        line_k = _fresh_corpus_line(test, pool_ranking, vote_labels)
    if find_on_test is not None or fallback_threshold is not None:
        line_k.update(_labels_line_counts(test_scores, find_on_test, fallback_threshold))
    return {
        "kind": kind,
        "t": int(t),
        "n_test": n_test,
        "n_test_pos": n_test_pos,
        "test_pos_ranks": test_ranks,
        "n_pool": n_pool,
        "n_pool_pos": n_pool_pos,
        "pool_pos_ranks": pool_ranks,
        **{f"test_line_k_{beta_tag(b)}": line_k[b] for b in RANK_FRAME_BETAS},
    }


def _evaluate_on_test(
    step: StepModel,
    threshold: float,
    clips_dict: dict[int, dict[str, Any]],
    test_ids: list[int],
    target_category: str,
    inclusion: int,
    region_aware: bool = False,
    style_obj: Any = None,
    scored_sink: "list[Any] | None" = None,
    find_line: Any = None,
    beta: float | None = None,
    out: "dict[str, Any] | None" = None,
) -> dict[str, float]:
    """Score *test_ids* with *step* and return the per-step metrics.

    With *find_line* (the app's labels line, #4452) the threshold is the one a
    Find on the withheld half would draw: the same class model, the prevalence
    re-estimated on these scores.

    Returns the operating-point metrics the user cares about — the objective
    ``fbeta`` at *beta* with its preset columns
    (:func:`~vtscore.eval.calibration_metrics.fbeta_metrics`, #4584),
    inclusion-weighted ``cost``, ``fpr``, ``fnr``, ``precision``, ``recall`` and
    ``f1`` (all computed at *threshold*, the last three via
    :func:`~vtscore.eval.calibration_metrics.detection_metrics`) — plus the
    threshold-independent ranking metrics ``auroc`` and ``average_precision``,
    which isolate "how good is the ranking" from "how good is the threshold".

    When *region_aware* the test media carry a ``patch_grid`` (a patch
    embedder), so scoring max-pools the MLP over every score row of each image -
    exactly the live detector's inference for patch datasets (an image scores
    by its best-matching row).  Otherwise each media is scored by its single
    whole-image vector through the step's trainer-agnostic ``predict``.

    *scored_sink*, when given, receives the ``(scores, labels)`` arrays so a
    caller can derive more metrics from the same pass (issue #3959).
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
            **dict.fromkeys(FBETA_COLUMNS, nan),
            "n_test_pos": nan,
            "n_test_neg": nan,
            "n_flagged": nan,
            "auroc": nan,
            "average_precision": nan,
        }

    scores = _score_media_ids(step, clips_dict, test_ids, region_aware=region_aware, style_obj=style_obj)

    true_labels = [1.0 if media_is_positive(clips_dict[cid], target_category) else 0.0 for cid in test_ids]

    maybe_dump_predictions(clips_dict, test_ids, scores, true_labels, threshold, target_category, suffix="__eval")

    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(true_labels, dtype=np.float64)
    if find_line is not None and beta is not None:
        threshold = float(find_line.on_corpus(scores_arr).threshold(float(beta)))
        if out is not None:
            out["find_threshold"] = threshold
    from vtscore.eval.calibration_metrics import (  # noqa: PLC0415
        detection_metrics,
        fbeta_metrics,
        inclusion_weights,
        operating_cost,
    )

    cost, fpr, fnr = operating_cost(scores_arr, labels_arr, threshold, *inclusion_weights(inclusion))
    if scored_sink is not None:
        scored_sink.extend((scores_arr, labels_arr))

    det = detection_metrics(scores_arr, labels_arr, threshold)
    return {
        "cost": round(cost, 6),
        "fpr": round(fpr, 6),
        "fnr": round(fnr, 6),
        **{k: round6(v) for k, v in det.items()},
        **{k: round6(v) for k, v in fbeta_metrics(scores_arr, labels_arr, threshold, beta).items()},
        "auroc": round(_auroc(scores_arr, labels_arr), 6),
        "average_precision": round(_average_precision(scores_arr, labels_arr), 6),
    }


def _standalone_calibration_row(
    threshold: float,
    details: dict[str, Any],
    scored: list[Any],
    inclusion: int,
) -> dict[str, Any]:
    """The calibration study's base row for a standalone (style-less) trainer.

    :func:`~vtscore.eval.row_metrics.operating_metrics` on the test scores
    :func:`_evaluate_on_test` already computed, with the step's fold orderings
    (when its trainer kept any) as the calibration set, so ``oracle_cost``,
    ``regret`` and ``threshold_provenance`` mean on a ``gp_*`` row exactly what
    they mean on the app's.  The provenance is the shipped estimator's when the
    safe-threshold path set one, else the trainer's own ``threshold_rule``.
    """
    import numpy as np  # noqa: PLC0415

    scores, labels = scored
    fold_orderings = details.get("fold_orderings") or []
    cal_scores = np.array([s for sc, _ in fold_orderings for s in sc]) if fold_orderings else None
    cal_labels = np.array([lb for _, lbs in fold_orderings for lb in lbs]) if fold_orderings else None
    row = operating_metrics(
        scores,
        labels,
        threshold,
        inclusion,
        cal_scores,
        cal_labels,
        pool_variant="max",
        provenance=str(details.get("provenance") or details.get("threshold_rule") or "standalone"),
        n_pool_rows=1.0,
        beta=details.get("beta"),
    )
    if "xcal_threshold" in details:
        row["xcal_threshold"] = round6(float(details["xcal_threshold"]))
    if cal_scores is not None:
        row["n_cal_scores"] = int(cal_scores.size)
    return row


def _calibration_metric_rows(
    step: StepModel,
    threshold: float,
    details: dict[str, Any],
    clips_dict: dict[int, dict[str, Any]],
    test_ids: list[int],
    target_category: str,
    inclusion: int,
    style_obj: Any,
    repool_variants: list[str],
    topk: int,
) -> tuple[list[dict[str, Any]], "np.ndarray", "np.ndarray", list[int]]:
    """Per-step metric rows for the base pooling plus each remedial re-pool.

    Scores the test set's per-node sigmoids once through *style_obj*, then pools
    them ``max`` (base) and — for the raw-patch tree arm, which carries
    ``fold_node_data`` — ``topk`` / ``pnorm``.  Each remedial variant recalibrates
    its own threshold by re-pooling the same fold models' held-out node scores,
    so every arm has a genuine *trained* cost and an *oracle* cost.  Returns one
    row dict per pooling, each tagged with ``pool_variant``, then the base
    pooling's test scores, their labels and the media ids they belong to.
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
    # The test side models Find (#4452): a Find on a new corpus applies the
    # labels' class model with the prevalence re-estimated on that corpus, so
    # the withheld half is cut at the threshold Find would draw there, not at
    # the Train side's.
    train_threshold = threshold
    find_line = details.get("find_line")
    find_prevalence = float("nan")
    if find_line is not None and details.get("beta") is not None:
        on_test = find_line.on_corpus(base_scores)
        find_prevalence = on_test.prevalence
        threshold = float(on_test.threshold(float(details["beta"])))
        details["find_threshold"] = threshold
        # The same fit, for the rank frames' count at every preset beta (#4471).
        details["find_on_test"] = on_test
    # dump: calibration path -- `ids` is aligned with base_scores and labels.
    maybe_dump_predictions(clips_dict, list(ids), base_scores, list(labels), threshold, target_category)
    base_cal_scores = np.array([s for scores, _ in fold_orderings for s in scores]) if fold_orderings else None
    base_cal_labels = np.array([lb for _, labels_ in fold_orderings for lb in labels_]) if fold_orderings else None
    base = operating_metrics(
        base_scores,
        labels,
        threshold,
        inclusion,
        base_cal_scores,
        base_cal_labels,
        pool_variant="max",
        provenance=provenance,
        n_pool_rows=n_pool_rows,
        beta=details.get("beta"),
    )
    if "xcal_threshold" in details:
        # Under safe_thresholds the base row's threshold is the blended one;
        # record the pre-blend conformal cut alongside it (issue #2799).
        base["xcal_threshold"] = round6(float(details["xcal_threshold"]))
    # Train's threshold and the two prevalence estimates behind the Find one (#4452).
    base["train_threshold"] = round6(float(train_threshold))
    base["train_prevalence"] = round6(float(details.get("train_prevalence", float("nan"))))
    base["find_prevalence"] = round6(find_prevalence)
    # How many held-out scores the conformal quantile was actually taken over,
    # on the SHIPPED row rather than only on the fold-count variant rows (issue
    # #3287).  It was declared in `CALIBRATION_COLUMNS` and filled only by the
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
                operating_metrics(
                    v_scores,
                    labels,
                    v_threshold,
                    inclusion,
                    v_cal_scores,
                    v_cal_labels,
                    pool_variant=variant,
                    provenance="conformal",
                    n_pool_rows=n_pool_rows,
                    beta=details.get("beta"),
                )
            )

    return rows, base_scores, labels, [int(i) for i in ids]


def _centroid_test(
    good_votes: dict[int, None],
    clips_dict: dict[int, dict[str, Any]],
    test_ids: list[int],
    target_category: str,
    inclusion: int,
    *,
    region_voting: bool,
    region_aware: bool,
    style_obj: Any,
    beta: float | None,
    calibration_rows: bool,
) -> tuple[StepModel, list[dict[str, Any]], "tuple[list[dict[str, Any]], np.ndarray, np.ndarray, list[int]] | None"]:
    """What a Test on the withheld half gives under the label quota (#4643): the Goods' centroid.

    The centroid is cut where Test cuts it, on the corpus it searches - here
    the withheld half - so the step is built (:func:`_centroid_step`) with that
    half's scorer, in the run's own geometry, and scored on it at
    :data:`~vtscore.detectors.centroid_head.CENTROID_THRESHOLD`.  Its line
    does not take the balance, so there is no Find line to re-draw: *beta*
    only weights the F-beta columns.

    Returns the step, its metric rows (one, from :func:`_calibration_metric_rows`
    when *calibration_rows*, else :func:`_evaluate_on_test`'s), and the
    calibration path's ``(rows, scores, labels, ids)`` or ``None``.
    """
    from vtscore.detectors.centroid_head import CENTROID_THRESHOLD  # noqa: PLC0415

    def _score(step: StepModel) -> list[float]:
        return _score_media_ids(step, clips_dict, list(test_ids), region_aware=region_aware, style_obj=style_obj)

    step = _centroid_step(
        good_votes, clips_dict, target_category, _score, region_voting=region_voting, style_obj=style_obj
    )
    if calibration_rows:
        calibration = _calibration_metric_rows(
            step,
            CENTROID_THRESHOLD,
            {"provenance": "centroid", "beta": beta},
            clips_dict,
            test_ids,
            target_category,
            inclusion,
            style_obj,
            [],
            0,
        )
        return step, calibration[0], calibration
    metrics = _evaluate_on_test(
        step,
        CENTROID_THRESHOLD,
        clips_dict,
        test_ids,
        target_category,
        inclusion,
        region_aware=region_aware,
        style_obj=style_obj,
        beta=beta,
    )
    return step, [metrics], None


# ------------------------------------------------------------------
# Supervised skyline (issue #3322)
# ------------------------------------------------------------------


def _skyline_fit_and_score(
    good_ids: list[int],
    bad_ids: list[int],
    score_ids: list[int],
    clips_dict: dict[int, dict[str, Any]],
    target_category: str,
    *,
    trainer: str,
    head: str,
    style_obj: Any,
    region_voting: bool,
    input_dim: int,
    inclusion: int,
    calibrate_count: int,
    calibration_fraction: float,
    details_sink: dict[str, Any] | None = None,
) -> tuple[dict[int, float], StepModel, dict[str, float], float]:
    """Train one fully-supervised head and score *score_ids* with it.

    The whole point of the skyline is that it differs from a mortal step in
    **the labels and nothing else**, so this goes through
    :func:`_train_and_calibrate` rather than fitting an estimator of its own:
    same head, same inclusion class-weights, same pinned fit seed, same backend,
    same bag-aware flooding.  A skyline that trained through a private
    ``LinearSVC`` would report "fewer labels" and "different trainer" as one
    number, and would need its own ``Mirror(...)`` entry in
    ``scripts/check-eval-app-sync.py`` to stay honest; delegation needs neither.

    The threshold the trainer calibrates is computed and **discarded** - the
    caller takes the test oracle's cut instead (see :data:`SKYLINE_PROVENANCE`).
    It is paid for rather than skipped because the alternative is a second entry
    point into the trainer that only the skyline uses, which is exactly the kind
    of near-copy this module keeps out.

    *details_sink*, when given, receives the trainer's details: its calibration
    folds' held-out orderings, from which the full-label model's labels line is
    drawn (#4486).

    Returns ``({media_id: score}, step, timings, test_score_seconds)``.
    """
    from vtscore.eval import calibration_metrics as cm  # noqa: PLC0415

    step, _threshold, _n_labels, timings, details = _train_and_calibrate(
        trainer,
        dict.fromkeys(good_ids),
        dict.fromkeys(bad_ids),
        clips_dict,
        target_category,
        region_voting=region_voting,
        input_dim=input_dim,
        inclusion=inclusion,
        calibrate_count=calibrate_count,
        calibration_fraction=calibration_fraction,
        head=head,
        style_obj=style_obj,
        emit_calibration_metrics=False,
    )
    if details_sink is not None:
        details_sink.update(details)
    assert step.torch_model is not None  # v1 runs the styled torch path only
    t_score = time.monotonic()
    ids, flat, seg = style_obj.node_scores(step.torch_model, {cid: clips_dict[cid] for cid in score_ids})
    pooled = cm.segment_max_pool(flat, seg)
    score_seconds = time.monotonic() - t_score
    return ({cid: float(v) for cid, v in zip(ids, pooled, strict=True)}, step, timings, score_seconds)


def _model_meta(model: Any) -> dict[str, Any]:
    """The class model's scalars at full precision, for a snapshot's JSON; its Bads' scores travel in the folds."""
    return {k: v for k, v in asdict(model).items() if k != "neg_logits"}


def _fold_arrays(orderings: Any) -> dict[str, Any]:
    """The calibration folds' held-out scores and labels as flat arrays, for a test-score snapshot (#4490).

    ``fold_index`` says which fold each score was held out in, so a replay can rebuild
    :func:`~vtscore.training.thresholds.labels_line.class_score_model`'s input exactly.
    """
    import numpy as np  # noqa: PLC0415

    parts = [
        (np.asarray(s, dtype=np.float64).ravel(), np.asarray(y, dtype=np.float64).ravel()) for s, y in orderings or []
    ]
    parts = [(s[: min(s.size, y.size)], y[: min(s.size, y.size)]) for s, y in parts]
    return {
        "fold_scores": np.concatenate([s for s, _ in parts]) if parts else np.empty(0, dtype=np.float64),
        "fold_labels": (np.concatenate([y for _, y in parts]) if parts else np.empty(0)).astype(np.int8),
        "fold_index": np.concatenate([np.full(s.size, i, dtype=np.int16) for i, (s, _) in enumerate(parts)])
        if parts
        else np.empty(0, dtype=np.int16),
    }


def _ceiling_find_line(details: dict[str, Any], score_map: dict[int, float], ordered_test: list[int]) -> Any:
    """The ceiling's Find line (#4486): what a Find on the withheld half returns with every training label known.

    The labels line from the full-label model's own calibration folds (*details*, the trainer's), its corpus
    side fitted on the withheld half, which holds no votes: the same rule the session rows are read under.
    The skyline row itself stays at the oracle's cut on the test labels, which other studies read; only the
    rank frame carries this line.  ``None`` when the folds support no class model.
    """
    import numpy as np  # noqa: PLC0415

    test_scores = np.array([score_map[cid] for cid in ordered_test], dtype=np.float64)
    return fit_labels_line(details.get("fold_orderings") or None, test_scores, ordered_test, {})


def _ceiling_line(
    details: dict[str, Any],
    score_map: dict[int, float],
    ordered_test: list[int],
    test_labels: Any,
    want_frame: bool,
    test_score_sink: Optional[list[dict[str, Any]]],
) -> Any:
    """The ceiling's Find line when a frame or a snapshot wants it; leaves the snapshot in *test_score_sink* (#4490)."""
    import numpy as np  # noqa: PLC0415

    if not want_frame and test_score_sink is None:
        return None
    find_line = _ceiling_find_line(details, score_map, ordered_test)
    if test_score_sink is not None:
        test_score_sink.append(
            {
                "t": 0,
                "phase": "ceiling",
                "scores": np.array([score_map[cid] for cid in ordered_test], dtype=np.float64),
                "labels": np.asarray(test_labels).astype(np.int8),
                "ids": np.asarray(ordered_test, dtype=np.int64),
                "train_threshold": float("nan"),
                "beta": float("nan"),
                "model": None if find_line is None else _model_meta(find_line.model),
                **_fold_arrays(details.get("fold_orderings")),
            }
        )
    return find_line


def _skyline_arm_rows(
    arms: list[str],
    clips_dict: dict[int, dict[str, Any]],
    target_category: str,
    sim_ids: list[int],
    test_ids: list[int],
    inclusion: int,
    *,
    trainer: str,
    head: str,
    style_obj: Any,
    region_voting: bool,
    input_dim: int,
    calibrate_count: int,
    calibration_fraction: float,
    seed: int,
    rank_frame_sink: Optional[list[dict[str, Any]]] = None,
    rank_ident: Optional[dict[str, Any]] = None,
    test_score_sink: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """One metric row per requested skyline arm (issue #3322), or ``[]``.

    Both arms are scored on the **untouched test split** - the same ``test_ids``
    every mortal row is scored on, pooled through the same
    ``style_obj.node_scores`` + segment max, so the two sides of
    ``training_regret`` differ in the model and in nothing else.  The #3308
    population convention is satisfied trivially rather than by re-applying
    :func:`~vtscore.training.thresholds.apply_vote_exclusion`: that convention is
    about the *haystack a threshold's population estimate is fitted on*, and a
    skyline fits no such estimate - it takes the test oracle's cut - so there is
    no haystack here to exclude votes from.

    ``skyline_test_xfit`` pools **cross-fitted** scores: the test split is
    partitioned into :data:`~vtscore.eval.transfer_rules.HONEST_ORACLE_FOLDS`
    folds and each item is scored by a head that never saw it.  Two caveats ride
    on that row and neither is a defect:

    * The pooled scores come from K different heads, so they share a *ranking*
      but not a calibrated scale.  That is fine for ``oracle_cost`` / ``auroc`` /
      ``average_precision``, which is all this arm is read for.
    * Its ``oracle_cost`` is still a sample minimum over those pooled scores, and
      is optimistic for exactly the reason ``oracle_cost`` always is - which is
      why ``oracle_cost_honest`` ships beside it, cross-fitting the *cut* on top
      of the cross-fitted *model*.

    Returns each row already carrying its ``gmm_variant`` tag and its own timing
    / backend columns; the caller supplies the identifying columns.  With a
    *rank_frame_sink*, each arm also appends its test ranking there as a rank
    frame of that arm's kind (#4357), under *rank_ident*.  With a
    *test_score_sink*, the full-label arm also leaves a ``phase="ceiling"``
    snapshot there (#4490): the withheld half's scores and labels, the class
    model its Find line used, and the calibration folds it was drawn from.
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.eval import calibration_metrics as cm  # noqa: PLC0415

    nan = float("nan")
    started = time.monotonic()
    ordered_test = sorted(test_ids)
    test_labels = np.array(
        [1.0 if media_is_positive(clips_dict[cid], target_category) else 0.0 for cid in ordered_test],
        dtype=np.float64,
    )
    wf, wn = cm.inclusion_weights(inclusion)

    def _row(
        name: str,
        score_map: dict[int, float],
        step: StepModel,
        timings: dict[str, float],
        secs: float,
        find_on_test: Any = None,
    ):
        scores = np.array([score_map[cid] for cid in ordered_test], dtype=np.float64)
        o_thr, _o_cost, _o_fpr, _o_fnr = cm.oracle_cut(scores, test_labels, wf, wn)
        if not np.isfinite(o_thr):
            return None
        if rank_frame_sink is not None:
            rank_frame_sink.append(
                {
                    **(rank_ident or {}),
                    **_rank_frame(name, 0, ordered_test, scores, test_labels, find_on_test=find_on_test),
                }
            )
        row = operating_metrics(
            scores,
            test_labels,
            float(o_thr),
            inclusion,
            None,
            None,
            pool_variant="max",
            provenance=SKYLINE_PROVENANCE,
            n_pool_rows=1.0,
            # A skyline belongs to no step, so no balance drew a line on it.
            beta=None,
        )
        row["gmm_variant"] = name
        row["schedule"] = ""
        row.update(
            {
                "calibrate_count": calibrate_count,
                "train_seconds": round(timings["train_seconds"], 6),
                "final_score_seconds": nan,
                "xcal_seconds": round(timings["xcal_seconds"], 6),
                "pool_score_seconds": nan,
                "test_score_seconds": round(secs, 6),
                "backend": step.backend,
                "device": step.device,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        )
        return row

    rows: list[dict[str, Any]] = []

    if SKYLINE_TRAIN_FULL in arms:
        sim_pos = [cid for cid in sorted(sim_ids) if media_is_positive(clips_dict[cid], target_category)]
        sim_neg = [cid for cid in sorted(sim_ids) if not media_is_positive(clips_dict[cid], target_category)]
        if sim_pos and sim_neg:
            sky_details: dict[str, Any] = {}
            score_map, step, timings, secs = _skyline_fit_and_score(
                sim_pos,
                sim_neg,
                ordered_test,
                clips_dict,
                target_category,
                trainer=trainer,
                head=head,
                style_obj=style_obj,
                region_voting=region_voting,
                input_dim=input_dim,
                inclusion=inclusion,
                calibrate_count=calibrate_count,
                calibration_fraction=calibration_fraction,
                details_sink=sky_details,
            )
            find_line = _ceiling_line(
                sky_details, score_map, ordered_test, test_labels, rank_frame_sink is not None, test_score_sink
            )
            row = _row(SKYLINE_TRAIN_FULL, score_map, step, timings, secs, find_on_test=find_line)
            if row is not None:
                rows.append(row)

    if SKYLINE_TEST_XFIT in arms:
        xfit = _skyline_xfit_scores(
            ordered_test,
            clips_dict,
            target_category,
            trainer=trainer,
            head=head,
            style_obj=style_obj,
            region_voting=region_voting,
            input_dim=input_dim,
            inclusion=inclusion,
            calibrate_count=calibrate_count,
            calibration_fraction=calibration_fraction,
            seed=seed,
        )
        if xfit is not None:
            row = _row(SKYLINE_TEST_XFIT, *xfit)
            if row is not None:
                rows.append(row)

    return rows


def _skyline_xfit_scores(
    ordered_test: list[int],
    clips_dict: dict[int, dict[str, Any]],
    target_category: str,
    *,
    trainer: str,
    head: str,
    style_obj: Any,
    region_voting: bool,
    input_dim: int,
    inclusion: int,
    calibrate_count: int,
    calibration_fraction: float,
    seed: int,
) -> tuple[dict[int, float], StepModel, dict[str, float], float] | None:
    """Cross-fitted test-side skyline scores: every item scored by a head that never saw it.

    Partitions *ordered_test* into :data:`~vtscore.eval.transfer_rules.HONEST_ORACLE_FOLDS`
    folds, trains the fully-supervised head on the complement of each, and scores
    the held-out fold with it.  ``None`` when the split cannot be made honestly -
    fewer items than folds, or a fold whose complement carries only one class -
    which leaves the bracket arm out of the frame rather than quietly falling back
    to the train-on-test fit it exists to avoid.

    Returns the same tuple shape as :func:`_skyline_fit_and_score`, with the
    per-fold wall clocks summed and the *last* fold's step standing in for the
    backend/device columns (every fold trains on the same backend).
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.eval.transfer_rules import HONEST_ORACLE_FOLDS  # noqa: PLC0415

    if len(ordered_test) < HONEST_ORACLE_FOLDS:
        return None
    # Seeded off the run's own seed, never off the trajectory's RandomState:
    # drawing from `rng` would move every subsequent vote, so turning the arm on
    # would silently change the run it is meant to describe.
    order = np.random.default_rng([_SKYLINE_SEED, seed]).permutation(len(ordered_test))
    scores: dict[int, float] = {}
    timings = {"train_seconds": 0.0, "xcal_seconds": 0.0}
    total_secs = 0.0
    step: StepModel | None = None
    for part in np.array_split(order, HONEST_ORACLE_FOLDS):
        held = [ordered_test[i] for i in part]
        if not held:
            continue
        held_set = set(held)
        fit_pos: list[int] = []
        fit_neg: list[int] = []
        for cid in ordered_test:
            if cid in held_set:
                continue
            (fit_pos if media_is_positive(clips_dict[cid], target_category) else fit_neg).append(cid)
        if not fit_pos or not fit_neg:
            return None
        score_map, step, fold_timings, secs = _skyline_fit_and_score(
            fit_pos,
            fit_neg,
            held,
            clips_dict,
            target_category,
            trainer=trainer,
            head=head,
            style_obj=style_obj,
            region_voting=region_voting,
            input_dim=input_dim,
            inclusion=inclusion,
            calibrate_count=calibrate_count,
            calibration_fraction=calibration_fraction,
        )
        scores.update({cid: score_map[cid] for cid in held})
        timings["train_seconds"] += fold_timings["train_seconds"]
        timings["xcal_seconds"] += fold_timings["xcal_seconds"]
        total_secs += secs
    if step is None or len(scores) != len(ordered_test):
        return None
    return scores, step, timings, total_secs


def _apply_skyline_decomposition(rows: list[dict[str, Any]], skyline_rows: list[dict[str, Any]]) -> None:
    """Fill the :data:`SKYLINE_COLUMNS` on *rows* from the primary skyline arm.

    ``training_regret = oracle_cost(row) - oracle_cost(skyline)`` on the naive
    reference and the cross-fitted one alike, so both decompositions telescope
    exactly against the ``regret`` / ``regret_honest`` the row already carries.
    Applied to the skyline rows too, which makes ``skyline_train_full``'s own
    ``training_regret`` exactly ``0`` and ``skyline_test_xfit``'s the bracket
    between the two references - both of which are the right readings.

    A no-op when :data:`SKYLINE_TRAIN_FULL` did not run: the columns stay NaN
    rather than being re-based on the bracket partner, which measures capacity on
    the test sample and not learnability.
    """
    ref = next((r for r in skyline_rows if r["gmm_variant"] == SKYLINE_TRAIN_FULL), None)
    if ref is None:
        return
    floor = ref["oracle_cost"]
    floor_honest = ref["oracle_cost_honest"]
    for row in [*rows, *skyline_rows]:
        row["skyline_oracle_cost"] = floor
        row["skyline_oracle_cost_honest"] = floor_honest
        row["training_regret"] = round6(row["oracle_cost"] - floor)
        row["training_regret_honest"] = round6(row["oracle_cost_honest"] - floor_honest)


# ------------------------------------------------------------------
# Single (seed, dataset, category) evaluation
# ------------------------------------------------------------------


@dataclass(frozen=True)
class _RunKnobs:
    """The pre-registered knobs of one :func:`simulate_voting_iterations` cell.

    A record rather than a tuple, so adding a knob cannot silently rebind the
    existing ones at the unpacking call site - the same hazard the keyword-only
    marker on :func:`simulate_voting_iterations` closes for its callers.
    """

    fold_schedule: "Callable[[int], int] | None"
    startup_state: StartupState | None
    skyline_arms: list[str]
    head: str
    trainer: str
    calibration_seed: int


def _resolve_head(head: Optional[str], trainer: str) -> str:
    """Validate an explicit head name, or resolve ``None`` to the app's.

    *trainer* must already be normalised by
    :func:`~vtscore.eval.step_model.resolve_trainer_name`.

    Raises:
        ValueError: If *head* is not a known head, or is given for a trainer
            that fits its own estimator rather than a head.
    """

    if head is not None:
        if head not in HEADS:
            raise ValueError(f"unknown head {head!r}; expected one of {HEADS}")
        if trainer != APP_TRAINER:
            raise ValueError(
                f"head={head!r} only applies to trainer={APP_TRAINER!r} (the app's own "
                f"pipeline, whose head this selects); got trainer={trainer!r}, which fits "
                f"its own standalone estimator and has no head to choose"
            )
    # **The default arm must be the app's default.**  Production pins the linear
    # SVM head on every fit (``hidden_dim = LINEAR_SVM_HEAD`` in
    # ``vtscore.detectors.training.train_and_threshold``), so an unspecified head
    # resolves to it — the same way *style* and *blend_schedule* resolve to the
    # app's geometry and schedule below.  The head fits the final model *and*
    # the calibration folds, so it moves the thresholds and, through the vote
    # order, the whole trajectory: defaulting to a retired head would make every
    # unqualified run measure a detector nobody ships.  ``head="linear"`` (the
    # logistic head) and ``head="mlp"`` stay available as named legacy arms.
    # (This is the *head* knob, not the ``trainer`` knob beside it: the trainer
    # picks the pipeline, the head picks what that pipeline fits — issue #3764.)
    head = head or PRODUCTION_HEAD
    return head


def _resolve_startup_state(
    startup_schedule: Optional[str],
    seed_scores: Optional[dict[int, float]],
    autopilot_fidelity: bool,
    strategy: str,
) -> StartupState | None:
    """Parse an explicit startup schedule, checking the run can actually honour it.

    Raises:
        ValueError: If a schedule is named without the seed sort it addresses
            positions on, or outside the faithful autopilot strategy it steers.
    """

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
    return startup_state


def _parse_opening_diversity(spec: Optional[str]) -> Optional[tuple[float, int]]:
    """``"<tau>/<k>"`` -> ``(tau, k)`` for the #4197 knob; ``None`` stays ``None``.

    Strict, like every arm-defining knob: a misread spec is an arm measuring
    something its launcher does not say.
    """
    if spec is None or not str(spec).strip():
        return None
    try:
        tau_s, k_s = str(spec).split("/")
        tau, k = float(tau_s), int(k_s)
    except ValueError as exc:
        raise ValueError(f"opening_diversity must be '<tau>/<k>', e.g. '0.85/1'; got {spec!r}") from exc
    if not (0.0 < tau <= 1.0) or k < 1:
        raise ValueError(f"opening_diversity needs 0 < tau <= 1 and k >= 1; got {spec!r}")
    return tau, k


def _check_acquisition_3546(
    acq_inclusion_offset: float,
    acq_rank_percentile: Optional[float],
    acq_p_crossing: "float | str | None",
    acq_origin: str,
    acq_target_p: "float | str | None",
) -> None:
    """#3546's knobs: the offset's origin, and a target pick precision (the offset is its fallback)."""
    if acq_origin not in ACQ_ORIGINS:
        raise ValueError(f"acq_origin must be one of {ACQ_ORIGINS}, got {acq_origin!r}")
    if acq_origin != "line" and acq_inclusion_offset == 0:
        raise ValueError("acq_origin only moves the offset cut; it needs a nonzero acq_inclusion_offset")
    if acq_target_p is None or acq_target_p == ACQ_TARGET_OFF:
        return
    if isinstance(acq_target_p, str) or not 0.0 < float(acq_target_p) < 1.0:
        raise ValueError(f"acq_target_p must lie in (0, 1) or be {ACQ_TARGET_OFF!r}, got {acq_target_p!r}")
    if acq_rank_percentile is not None or acq_p_crossing not in (None, ACQ_P_CROSSING_OFF):
        raise ValueError(
            "acq_target_p names the acquisition cut: pass no acq_rank_percentile and no acq_p_crossing "
            "to run the target-precision arm"
        )


def _check_acquisition_arm(
    acq_inclusion_offset: float,
    acq_rank_percentile: Optional[float],
    acq_p_crossing: "float | str | None",
    acq_origin: str = "line",
    acq_target_p: "float | str | None" = None,
) -> None:
    """The acquisition cut's knobs name one cut: the shipped offset, a rank pin, the P-aware crossing (#4409),
    or a target pick precision (#3546); *acq_origin* says where the offset counts from."""
    _check_acquisition_3546(acq_inclusion_offset, acq_rank_percentile, acq_p_crossing, acq_origin, acq_target_p)
    if acq_rank_percentile is not None:
        if acq_inclusion_offset != 0:
            raise ValueError(
                "acq_inclusion_offset and acq_rank_percentile are mutually exclusive; "
                "pass acq_inclusion_offset=0 to run the rank-pinned arm "
                f"(the default is {ACQUISITION_INCLUSION_OFFSET}, the shipped acquisition cut)"
            )
        if not 0.0 <= acq_rank_percentile <= 1.0:
            raise ValueError(f"acq_rank_percentile must be in [0, 1], got {acq_rank_percentile}")
    if acq_p_crossing is not None and acq_p_crossing != ACQ_P_CROSSING_OFF:
        if acq_inclusion_offset != 0 or acq_rank_percentile is not None:
            raise ValueError(
                "acq_p_crossing replaces the acquisition cut: pass acq_inclusion_offset=0 and no "
                "acq_rank_percentile to run the P-aware arm"
            )
        if isinstance(acq_p_crossing, str) or not acq_p_crossing > 0:
            raise ValueError(
                f"acq_p_crossing must be > 0 (a multiple of the argmax's depth) or {ACQ_P_CROSSING_OFF!r}, "
                f"got {acq_p_crossing!r}"
            )


#: ``acq_p_crossing="off"``: the line - 4 offset even under a balance, whose
#: default is otherwise the shipped argmax factor (#4409).
ACQ_P_CROSSING_OFF = "off"

#: Where the acquisition offset counts its Inclusion steps from (#3546):
#: ``"line"`` is the app (the inclusion the reporting line sits at, recovered
#: through the fold-anchored scale); ``"inclusion"`` is the old origin, the
#: run's Inclusion knob (0 by default) whatever line the balance drew - the
#: floor-off arm #4333 proposed.
ACQ_ORIGINS = ("line", "inclusion")

#: ``acq_target_p="off"``: the offset cut even under a balance (the #3546
#: pricing's control, and every offset arm).
ACQ_TARGET_OFF = "off"


def resolve_acquisition_target(acq_target_p: "float | str | None", beta: Optional[float]) -> Optional[float]:
    """The target pick precision an arm samples at, or ``None`` for the offset cut (#3546).

    ``None`` is the app's rule: under a balance the shipped
    :data:`~vtscore.training.thresholds.ACQUISITION_TARGET_PRECISION`, on the
    Inclusion arm the offset cut.  ``"off"`` forces the offset cut; a number
    pins the target.
    """
    if acq_target_p == ACQ_TARGET_OFF:
        return None
    if acq_target_p is None:
        return ACQUISITION_TARGET_PRECISION if beta is not None else None
    return float(acq_target_p)


#: The harness's name for the full balance walk whose end moves the line (what the app shipped before #4427's pricing).
WALK_SHAPE_FULL = "walk"


def resolve_walk_shape(walk_shape: Optional[str], beta: Optional[float]) -> Optional[str]:
    """The end-of-run check's shape for an arm: the app's (:func:`check_shape`) when none is named; ``None`` with no balance.

    ``"walk"`` is the full walk, ``"advisory"`` and ``"trim"`` force a shape (#4427).  Validated here, so a malformed
    arm fails before anything expensive runs.
    """
    if walk_shape is not None and walk_shape not in (WALK_SHAPE_FULL, *CHECK_SHAPES):
        raise ValueError(f"walk_shape must be one of {(WALK_SHAPE_FULL, *CHECK_SHAPES)}, got {walk_shape!r}")
    if beta is None:
        return None
    return check_shape(beta) if walk_shape is None else walk_shape


def resolve_acquisition_factor(acq_p_crossing: "float | str | None", beta: Optional[float]) -> Optional[float]:
    """The share of the F-beta argmax's depth the acquisition cut sits at, or ``None`` for the offset cut.

    The harness's counterpart of what :func:`vtscore.state.core.detector_acquisition_threshold`
    does with its *beta*: ``None`` is the app's default - under a balance the
    shipped :data:`~vtscore.training.thresholds.ACQUISITION_ARGMAX_FACTOR`
    (``None`` since the #4427 revert: the line - 4 offset), on the Inclusion
    arm the offset (``acq_inclusion_offset``);
    :data:`ACQ_P_CROSSING_OFF` forces the offset under a balance whatever the
    constant says; a number is the arm (#4409).
    """
    if acq_p_crossing == ACQ_P_CROSSING_OFF:
        return None
    if acq_p_crossing is None:
        return ACQUISITION_ARGMAX_FACTOR if beta is not None else None
    return float(acq_p_crossing)


def _resolve_run_knobs(
    *,
    fold_count_schedule: str | None,
    calibrate_count: int,
    startup_schedule: Optional[str],
    seed_scores: Optional[dict[int, float]],
    autopilot_fidelity: bool,
    strategy: str,
    skyline_arms: Optional[list[str]],
    emit_calibration_metrics: bool,
    acq_inclusion_offset: float,
    acq_rank_percentile: Optional[float],
    head: Optional[str],
    acq_p_crossing: "float | str | None" = None,
    acq_origin: str = "line",
    acq_target_p: "float | str | None" = None,
    trainer: str,
    style: Optional[str],
    calibration_seed: Optional[int],
) -> _RunKnobs:
    """Validate a cell's pre-registered knobs and resolve the defaults among them.

    Called from the top of :func:`simulate_voting_iterations`, before anything
    expensive runs, and the only place these checks belong: a run that dies
    forty minutes in on a typo has held a cluster slot for nothing, so a
    malformed knob has to kill the cell at second zero.  Gathering them under
    one name is what keeps that contract visible - a ``raise`` added deep in the
    stepping loop now reads as an obvious departure rather than as one more line
    among seven hundred.

    Note this resolves only the defaults that can be decided from the arguments
    alone.  The two that need the loaded pool - the detection *style* and the
    pair in :func:`_resolve_production_defaults` - are keyed on ``region_aware``
    and so are resolved later, once the medias have been read.

    Raises:
        ValueError: If a knob is malformed, or two knobs are mutually exclusive.
    """

    # These are pre-registered experiment knobs, so they are validated beside
    # the other argument checks rather than deep in the loop: a run that dies
    # forty minutes in on a typo has held a cluster slot for nothing.
    # Parsed here, with the other pre-registered knobs, and not in the loop: a
    # malformed schedule must kill the cell at second zero rather than forty
    # minutes in, and the parse is what validates the spec at all.
    _fold_schedule = parse_fold_count_schedule(fold_count_schedule, calibrate_count)

    startup_state = _resolve_startup_state(startup_schedule, seed_scores, autopilot_fidelity, strategy)

    skyline_arms = list(skyline_arms or [])
    if skyline_arms:
        unknown = [a for a in skyline_arms if a not in SKYLINE_ARMS]
        if unknown:
            raise ValueError(f"unknown skyline arm(s) {unknown}; expected a subset of {list(SKYLINE_ARMS)}")
        if not emit_calibration_metrics:
            raise ValueError(
                "skyline_arms requires emit_calibration_metrics: the decomposition is defined "
                "against the calibration frame's oracle_cost/regret columns, which the plain "
                "frame does not carry"
            )

    _check_acquisition_arm(acq_inclusion_offset, acq_rank_percentile, acq_p_crossing, acq_origin, acq_target_p)

    trainer = resolve_trainer_name(trainer)
    head = _resolve_head(head, trainer)

    if style is not None and trainer != APP_TRAINER:
        raise ValueError(
            f"detection styles only apply to trainer={APP_TRAINER!r} (the app's own pipeline); got trainer={trainer!r}"
        )
    # **The default arm must be the app's default.**  Production always splits
    # Train/Calibrate off a fresh ``RandomState(CALIBRATION_SPLIT_SEED)`` - the
    # pin is deliberate (issue #2934 fixed the unseeded global draw that made
    # thresholds move run to run), so an unspecified *calibration_seed* resolves
    # to that same 42 and a default run is byte-for-byte what it was.  An
    # explicit value is the #3794 measurement arm: held at one cell seed, it
    # redraws *only* the calibration split, which is the run-to-run noise a
    # single-seed number hides.  It is not a way to "add randomness" to the
    # default arm - the app has none here to simulate.
    if calibration_seed is not None and not isinstance(calibration_seed, int):
        raise ValueError(f"calibration_seed must be an int or None; got {calibration_seed!r}")
    return _RunKnobs(
        fold_schedule=_fold_schedule,
        startup_state=startup_state,
        skyline_arms=skyline_arms,
        head=head,
        trainer=trainer,
        calibration_seed=CALIBRATION_SPLIT_SEED if calibration_seed is None else calibration_seed,
    )


def _resolve_production_defaults(
    *,
    blend_schedule: Optional[str],
    calibration_fraction: Optional[float],
    region_aware: bool,
) -> tuple[str, float]:
    """Resolve the unnamed blend schedule and split fraction to the app's own.

    **The eval default arm must be the app's default.**  Both ``default``
    mirrors in ``scripts/check-eval-app-sync.py`` -
    ``training.blend_schedule_default`` and ``training.split_fraction_default`` -
    name *this* function as their harness side, so a tripped gate points the
    reconciler at twenty lines instead of at the thousand that make up
    :func:`simulate_voting_iterations`.  It also gives that gate something real
    to check: its harness-side test is an existence check on the named symbol,
    which the enclosing function would satisfy even if both resolutions below
    were deleted outright.

    Both resolutions are keyed on *region_aware* - whether any media in the pool
    carries a ``patch_grid`` - which is why they run here rather than in
    :func:`_resolve_run_knobs`, alongside the knobs that need no pool.

    Args:
        blend_schedule: The named schedule arm, or ``None`` for the default.
        calibration_fraction: The explicit Train/Calibrate split, or ``None``.
        region_aware: Whether the pool carries patch grids.

    Returns:
        The resolved ``(blend_schedule, calibration_fraction)`` pair.
    """

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
    return blend_schedule, calibration_fraction


def _check_inclusion_arm(inclusion: float, beta: float | None) -> None:
    """Refuse a non-zero *inclusion* under a balance (#4361, #4413).

    A set preference wins over the knob: the line is the set the balance
    keeps, so *inclusion* would only re-weight the ``cost`` column and never
    move the line - an arm that looks swept and measures one line.  Only the
    Inclusion arm (``beta="off"``) draws its line at *inclusion*.
    """
    if beta is None or inclusion == 0:
        return
    raise ValueError(
        f"inclusion={inclusion!r} draws the line only on the Inclusion arm (beta={NO_BALANCE!r}); under a "
        f"balance (beta {beta:g}) it would only re-weight cost. Pass beta={NO_BALANCE!r} to sweep it."
    )


def simulate_voting_iterations(  # noqa: C901
    clips_dict: dict[int, dict[str, Any]],
    target_category: str,
    seed: int,
    *,
    dataset_name: str = "",
    inclusion: int = 0,
    sim_fraction: float = 0.5,
    safe_thresholds: bool = True,
    calibrate_count: int = 2,
    fold_count_schedule: str | None = None,
    calibration_fraction: Optional[float] = None,
    region_voting: bool = False,
    strategy: str = "autopilot",
    max_steps: Optional[int] = None,
    atlas_min_node_size: int = 20,
    seed_scores: Optional[dict[int, float]] = None,
    trainer: str = APP_TRAINER,
    head: Optional[str] = None,
    target_prevalence: Optional[float] = None,
    haystack_prevalence: Optional[float] = None,
    style: Optional[str] = None,
    emit_calibration_metrics: bool = False,
    repool_variants: Optional[list[str]] = None,
    repool_topk: int = 4,
    inclusion_sweep_ks: Optional[list[int]] = None,
    sweep_sink: Optional[list[dict[str, Any]]] = None,
    blend_schedule: Optional[str] = None,
    schedule_variants: Optional[list[str]] = None,
    cut_diag_sink: Optional[list[dict[str, Any]]] = None,
    fit_quality_sink: Optional[list[dict[str, Any]]] = None,
    fit_quality_stride: int = FIT_QUALITY_STRIDE_DEFAULT,
    autopilot_fidelity: bool = True,
    anchored_thresholds: bool = False,
    anchored_weights: Optional[list[float]] = None,
    anchored_rules: Optional[list[str]] = None,
    anchored_fold_arms: bool = True,
    anchored_fold_combines: Optional[list[str]] = None,
    fold_count_variants: Optional[list[int]] = None,
    cut_inclusion_ks: Optional[list[int] | list[float]] = None,
    cut_inclusion_sink: Optional[list[dict[str, Any]]] = None,
    cut_inclusion_qtilt_steps: Optional[list[float]] = None,
    acq_inclusion_offset: float = ACQUISITION_INCLUSION_OFFSET,
    acq_rank_percentile: Optional[float] = None,
    acq_p_crossing: "float | str | None" = None,
    acq_origin: str = "line",
    acq_target_p: "float | str | None" = None,
    smart_gate: str = "app",
    startup_schedule: Optional[str] = None,
    opening_diversity: Optional[str] = None,
    more_walk: str = "seed",
    band_share: Optional[int] = None,
    pick_sink: Optional[list[dict[str, Any]]] = None,
    precision_frame_sink: Optional[list[dict[str, Any]]] = None,
    precision_frame_steps: Optional[Sequence[int]] = None,
    rank_frame_sink: Optional[list[dict[str, Any]]] = None,
    rank_frame_steps: Optional[Sequence[int]] = None,
    test_score_sink: Optional[list[dict[str, Any]]] = None,
    line_test_sink: Optional[list[dict[str, Any]]] = None,
    line_test_budgets: Optional["LineBudgets"] = None,
    sim_size: Optional[int] = None,
    exclusion_min_remainder: Optional[float] = None,
    live_cut_rule: Optional[str] = None,
    live_threshold: Optional[str] = None,
    skyline_arms: Optional[list[str]] = None,
    calibration_seed: Optional[int] = None,
    standalone_cut: str = "raw",
    test_bands: Optional[list[str] | str] = None,
    train_mix: "Optional[str | dict[str, float]]" = None,
    test_band_auroc: bool = False,
    beta: "Optional[float | str]" = None,
    walk_picks: Optional[int] = None,
    walk_tol: float = 0.0,
    walk_fine: bool = False,
    walk_guard: Optional[float] = None,
    walk_shape: Optional[str] = None,
    spot_check: str = "weak",
    weak_separation: float = WEAK_SEPARATION_D,
    weak_min_t: int = WEAK_CHECK_MIN_VOTES,
    weak_repeat: int = WEAK_CHECK_COOLDOWN,
    weak_phase: str = "learned",
    label_quota: Optional[bool] = None,
) -> list[dict[str, Any]]:
    """Simulate voting on *clips_dict* and evaluate at every step.

    Args:
        clips_dict: Pre-loaded media dict (``{id: clip_data}``).
        target_category: Category treated as the positive class.
        seed: Random seed for splitting and vote ordering.
        dataset_name: Label included in result rows.
        inclusion: The Inclusion arm's line, in ``[-10, 10]``.  A set balance
            wins over it, so a non-zero value needs ``beta="off"`` and is
            refused under a balance (#4361).
        trainer: Which **pipeline** runs at each step.  ``"app"``
            (:data:`APP_TRAINER`, the default) is VTSearch's own — the app's
            ``train_model`` fit plus production fold calibration — and *which
            model it fits is the separate* ``head`` *argument*.  Anything else
            is a standalone estimator that fits and thresholds itself:
            ``"svm_linear"``, ``"svm_rbf"``, or a parameterised spec such as
            ``"svm_rbf@C=3,gamma=scale"`` (see
            :mod:`vtscore.eval.sweep_trainers`).  The pre-#3764 spelling
            ``"mlp"`` is still accepted for ``"app"`` and normalised on the way
            in; it named no MLP, since the arm's default head is the linear SVM.
            The autopilot vote order adapts to the chosen model, so the app and
            SVM trajectories diverge after the first retrain even at the same
            seed — by design (the question is which model makes *VTSearch*
            better, and VTSearch's vote order depends on the model).
        head: Which classifier head the ``"app"`` pipeline fits at each step
            (see :data:`HEADS`).  ``None`` (default) resolves to the **app's**
            head, :data:`PRODUCTION_HEAD` — the linear SVM a live VTSearch
            detector actually has, so a default run's thresholds and costs are
            the ones users see.  ``"linear"`` (the logistic head the SVM
            replaced) and ``"mlp"`` (the harness's auto-sized hidden layer,
            #2781) are the explicitly-named legacy arms.  The head is threaded
            into the calibration folds as well, mirroring how production threads
            one sentinel through ``_train_and_score_xy``.  Rejected on the
            standalone SVM trainers, which fit their own estimator rather than
            a head; the *resolved* name is recorded in the ``head`` result
            column (blank on those trainers).
        style: Optional detection-style name (see
            :mod:`vtscore.eval.patch_styles`): ``"whole_image"``,
            ``"max_patch"`` (the production geometry), or one of the
            ``"max_patch_hac"`` hybrids.  When set (``"app"`` trainer only), the style owns the
            vote-to-vector assembly, the test/sim scoring rule, and the
            bag-aware flooding of Bad votes - the Max-Patch experiment arms.
            ``None`` (default) resolves to the **app's** geometry: a patch
            dataset (any media with a ``patch_grid``) on the ``"app"`` trainer gets
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
        haystack_prevalence: When set (e.g. ``0.05``), the *simulation* half's
            negatives are thinned after the split so its positives are that
            fraction of it - see :func:`thin_haystack`.  Unlike
            ``target_prevalence`` the test set is untouched, so it combines with
            ``test_bands`` and each cell pairs with its natural twin (#4184).
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
            given, one :data:`CUT_DIAGNOSTIC_COLUMNS` row per (step, geometry)
            carrying the fitted mixture parameters and the #2836 decomposition
            chain.
        calibrate_count: Number of random Train/Calibrate splits for threshold
            calibration (default 2).
        fold_count_schedule: **Eval-only** (#3314).  ``"K@N"`` resolves
            *calibrate_count* per step from the vote count -
            ``K(n_votes) = K while n_votes < N, else calibrate_count`` - so a
            run can spend more folds where they are cheapest and decay to
            production's count as the labelset grows.  ``None`` (every other
            caller) keeps *calibrate_count* constant, which is byte-identical
            to the behaviour before the knob existed.  See
            :func:`parse_fold_count_schedule` for why this is a harness knob
            and not a shipped setting.
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
            :data:`CUT_INCLUSION_COLUMNS`).  Orthogonal to
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
            (-4), **the shipped app behaviour** - the harness matches production
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
        pick_sink: List the per-click :data:`PICK_COLUMNS` rows are appended
            to - one per vote, including the opening's, which emit no main row
            because no model exists yet.  ``None`` (default) = off.
        precision_frame_sink: List one :func:`_precision_frame` dict is
            appended to at each step in *precision_frame_steps* - the per-image
            evidence and truth a precision-floor estimator is priced on (#4220).
            Only the calibration-metrics path fills it.  ``None`` (default) = off.
        precision_frame_steps: The steps (``t``) to record; ignored without a sink.
        rank_frame_sink: List the
            :data:`~vtscore.eval.voting_columns.RANK_FRAME_COLUMNS` rows are appended
            to (#4357): where the positives sit in the test half's ranking and
            in the session's unvoted pool.  One ``step`` row at each step in
            *rank_frame_steps*, one ``last`` row for the last ordinary step (the
            ranking the end-of-run spot check draws from, before its votes),
            and one row per skyline arm.  Only the calibration-metrics path
            fills it.  ``None`` (default) = off.
        rank_frame_steps: The ordinary steps (``t``) to record ``step`` rows
            at; ignored without a sink.
        line_test_sink: List the
            :data:`~vtscore.eval.voting_columns.LINE_TEST_COLUMNS` row is
            appended to (#4523): Test mode's autopilot run on the withheld half
            as it stood at the last ordinary click, every pick answered from
            the truth (:func:`vtscore.eval.line_test_arm.line_test_row`).
            Runs after the loop, so it cannot perturb the trajectory, and
            only under a balance (*beta*): the Test is the balance line's.
            Only the calibration-metrics path fills it.  ``None`` (default) = off.
        line_test_budgets: The Test's targets and budgets; ``None`` (default)
            is the app's :data:`~vtscore.training.thresholds.DEFAULT_BUDGETS`.
        opening_diversity: ``"<tau>/<k>"`` - an experiment knob (issue #4197),
            not app behaviour.  While the opening walks the top of the seed sort
            (``good`` / ``more``), pass over candidates with cosine >= *tau* to at
            least *k* Bads voted so far (the text query's sibling cluster).
            ``None`` - the default - is the app.  See
            :func:`vtscore.eval.al_strategies._diverse_top`.
        more_walk: Where Autopilot's ``more`` walk draws (issue #4637), an
            experiment knob.  ``"seed"`` - the default - is the app: the top of
            the text sort.  ``"detector"`` takes the top of the step's detector
            ranking instead, and records the walk's steps as shown
            (``app_trained``): the app showing the opening's detector from the
            end of the Bad phase (#4604).  Needs the app's own opening (no
            *startup_schedule*, no *opening_diversity*) and the phase machine.
        band_share: An experiment knob (issue #4482): one in *band_share* of
            Autopilot's picks past the opening (the phases where the app shows
            a detector) is a **band pick** instead of the phase's own, a draw
            uniform within one band of the unvoted ranking, cycling through the
            bands a spot check starts from
            (:func:`~vtscore.eval.al_strategies.band_pick`).  It is an ordinary
            click, logged with phase ``"band"``.  ``None`` - the default - is the
            app.  Needs the balance (*beta*) and the phase machine.
        acq_rank_percentile: Alternative acquisition cut - place it at this
            quantile of the simulation-set score distribution directly, rather
            than by naming an inclusion.  This is the ``rank_pin`` arm: same
            intent, one fewer indirection.  Requires
            ``acq_inclusion_offset=0``, since the two name the same cut.
        acq_p_crossing: The **balance-aware** acquisition cut (#4409, #4413):
            place it at this multiple of the depth of the mixture's F-beta
            argmax over the step's unvoted line ranking (no cap), read as a
            rank: ``1.0`` samples at the argmax, ``0.5`` halfway up to the top.
            So a precision-leaning balance samples high, a recall-leaning one
            deep.  A number requires ``acq_inclusion_offset=0`` and a balance.
            ``None`` (the default) is the app's rule: under a balance the
            shipped :data:`~vtscore.training.thresholds.ACQUISITION_ARGMAX_FACTOR`
            (the ``acq_inclusion_offset`` cut is then the fallback for a step
            with no mixture estimate), on the Inclusion arm the offset cut;
            ``"off"`` forces the offset cut under a balance (the pricing's
            control, ``docs/experiments/2026-10-01-acquisition-fbeta-4409``).
        acq_origin: Where the offset counts from (#3546): ``"line"`` (the
            default, the app) or ``"inclusion"``, the run's Inclusion knob
            whatever line the balance drew - the old origin.
        acq_target_p: The **target pick precision** (#3546): sample at the
            score where the labels line's corpus posterior falls below this
            share (:func:`~vtscore.training.thresholds.target_precision_threshold`),
            falling back to the offset cut on a step with no labels line.
            ``None`` (the default) is the app's rule: under a balance the
            shipped ``ACQUISITION_TARGET_PRECISION``, on the Inclusion arm the
            offset cut.  ``"off"`` forces the offset cut under a balance.
        smart_gate: #4359's bound on the Smart light: ``"app"`` (the default)
            reads it; ``"never"`` holds it yellow for the phase decision, so
            Autopilot stays in ``hard`` to the end of the run.
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
        skyline_arms: **Supervised-skyline** arms to measure once per run
            (issue #3322; see :data:`SKYLINE_ARMS`).  ``None``/``[]`` (default)
            = off, and every other study runs exactly as before.  Requires
            ``emit_calibration_metrics``, because the arm exists to split that
            frame's ``oracle_cost`` into a learnability floor plus a
            ``training_regret``; skipped with a warning on a patch column, whose
            skyline needs a supervision decision this does not improvise (see
            :data:`SKYLINE_COLUMNS` and issue #3321).  Costs one extra fit per
            arm per run - the skyline is vote-independent - not one per step.
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
        calibration_seed: Seed of the Train/Calibrate fold splits (issue #3794).
            ``None`` (default) resolves to the app's own
            :data:`~vtscore.training.thresholds.CALIBRATION_SPLIT_SEED`, so the
            default arm's calibration is byte-for-byte production's.  An
            explicit value is a **measurement** arm, not extra realism: hold
            *seed* fixed so the data — which media are voted, in what order, the
            held-out split — cannot move, sweep this instead, and the spread
            across the sweep is the noise the pinned draw hides.  Reseeding it
            on a *default* run would measure a detector nobody ships, because
            production pins the split too (issue #2934 pinned it on purpose).
            Recorded verbatim in the ``calibration_seed`` column, so a pooled
            frame says which draw each row came from.
        live_cut_rule: The fold-anchored cut rule the **live** threshold uses
            (issue #3557) - the one reporting, acquisition and every downstream
            vote read.  ``None`` (default) is the app's own
            :data:`~vtscore.training.thresholds.FOLD_ANCHOR_CUT_RULE`, so the
            default arm cannot drift from production.  A named rule is a
            **run-level** arm, never a paired one: the acquisition cut re-cuts
            the same estimator at ``inclusion + acq_inclusion_offset``, so a
            rule that moves the cut below ``k = 0`` moves which media get voted
            from the first fitted step on.  The ``__cutincl`` frame's re-cuts
            are the paired, reporting-only view of the same rules.
        live_threshold: A **retired** live threshold rule (issue #4184), one of
            :data:`~vtscore.eval.live_threshold_rules.LIVE_THRESHOLD_RULES`, or
            ``None`` (default) for the shipped fold-anchored cut.  The named rule
            replaces the step's live cut after the fused path has fitted it -
            reporting, acquisition and every later vote read it - so, like
            *live_cut_rule*, it is a run-level arm.  The acquisition offset does
            not apply under it: the retired rules predate the offset and have no
            inclusion-aware form, so acquisition aims at the reporting cut.
            Needs *safe_thresholds*; exclusive with *live_cut_rule*.

        standalone_cut: How a ``gp_*`` trainer's cross-calibration cut reaches
            its final model (issue #3954).  ``"raw"`` (the default, and what
            the ``svm_*`` arms always do) applies the pooled held-out cut as a
            raw score; ``"rank"`` reads each fold's cut as a quantile of that
            fold model's simulation-set scores and realizes the averaged
            quantile on the final model's - the transfer production's
            fold-anchored estimator makes, for a probability whose scale moves
            with every refit.  See
            :func:`vtscore.eval.step_trainers._rank_transferred_threshold`.
            Only the ``gp_*`` trainers honour it; ``"rank"`` with any other
            trainer is an error, as is a region-aware dataset.
            ``"anchored"`` (issue #3959) is the GP-native rule: the GP fits its
            own calibration folds on the app's splits and the **shipped**
            fold-anchored estimator runs on them - each fold's mixture on that
            fold GP's haystack scores, anchored by its held-out votes, carried
            to the final GP by quantile.  Needs *safe_thresholds* (that is where
            the estimator runs); see
            :func:`vtscore.eval.step_trainers._gp_train_and_calibrate`.
        test_bands: Report the miss rate **per size band** beside the headline
            one (issue #4044).  ``"auto"`` takes every band the target's class
            has; a list names them; ``None`` (the default) turns the whole thing
            off and changes nothing.  Only meaningful on a scale-banded cell
            (``car@small``), where an image holding the class at another size is
            excluded from the cell by
            :func:`~vtscore.eval.labels.media_is_evaluable` and so was
            previously unreachable at any size but the trained one.
            Each band's cohort is the held-out set *that band's own arm* would
            have tested against, so the three FNRs of one arm and the same
            band's FNR under a different arm are paired -- see
            :mod:`vtscore.eval.scale_bands`.  There is deliberately no per-band
            FPR: the bands share their negatives, so the row's single ``fpr`` is
            the whole of that half.  Incompatible with ``target_prevalence``,
            which moves the pool the replay depends on.
        train_mix: Train on the class at a **mix of sizes** (issue #4160):
            ``"natural"`` or a per-band weight mapping. *target_category* is
            then a mixed cell (``car@mix-equal``) whose positives are drawn by
            :func:`~vtscore.eval.scale_bands.paired_mix` from the pure bands'
            sim halves at *seed*, and whose test positives are the pure bands'
            held-out cohorts.  The split is that function's rather than
            :func:`_split_media_ids`'s, because re-splitting the mixed pool
            would train on images a pure arm tests on.  With ``test_bands`` the
            per-band columns are the same images the pure arms report.  ``None``
            (the default) changes nothing.
        test_band_auroc: Under ``test_bands``, also rank each band's cohort
            against the held-out negatives (``auroc_<band>``, #4160).  Off by
            default because it scores the negatives a second time each step.
        beta: The balance the reporting line is drawn at (#4413), resolved by
            :func:`~vtscore.training.thresholds.resolve_line_knobs`.  A number
            pins the **balance arm**: the set the balance keeps - the top
            *count* unvoted items of the sim set, the mixture's F-beta argmax
            under the balance's cap until the run's spot check ends, then
            what the check's shape makes of the walk's end.  ``"off"``
            (:data:`~vtscore.training.thresholds.NO_BALANCE`) is the
            **Inclusion arm** - the line at *inclusion* - which is what every
            study before #4245 measured; an arm that sweeps *inclusion* has
            to pass it, because a set preference wins over the knob.  ``None``
            (the default) is the app's own default balance, ``DEFAULT_BETA``,
            so the default arm reports the line a live detector draws.
        walk_picks: The balance walk's picks a band (#4427's arms; the
            schedule's 5 when ``None``), *walk_tol* its tolerance (a deeper
            step within it of the best is flat and the walk looks one band
            further; 0, the app's, is the strict rise), *walk_fine* whether
            every band past the start is split in two, *walk_guard* the
            precision guard (a deeper band whose audited share right is below
            it times the start set's ends the walk at the best set so far)
            and *walk_shape* how the check treats the line: ``None``, the
            app's, follows :func:`~vtscore.training.thresholds.check_shape`
            (``advisory`` at beta <= 1: the votes train, the line keeps the
            unchecked rule's count; ``trim`` above: the walk may only step
            shallower and the line takes its end); ``"walk"`` is the full
            walk whose end moves the line, what the app shipped before
            #4427's pricing; ``"advisory"`` and ``"trim"`` force a shape.
            All apply to the end-of-run balance walk only.
        spot_check: When the simulated user runs the balance's **spot check**
            (#4272, the band walk of #4388 with #4413's F-beta stop).
            ``"end"``: once the voting steps are spent - *max_steps* reached,
            or the pool exhausted - the user checks the line the way the app's
            check step does: the unvoted ranking is fixed off the current one
            and cut into bands (the top 8, the next 8, 16, 32, ...), the walk
            starts at the bands holding the balance's cap (the top 32 at
            beta <= 1, 128 above it), each band's picks are answered from
            ground truth **and cast as votes** (provenance ``check``), the
            model retrains on them and one row is emitted per band with
            ``phase == "check"`` and ``t`` still counting every vote cast, so
            the check's rows sit past *max_steps*.  The walk stops at the
            F-beta peak, and the last row's line is what the check's shape
            makes of it, its ranges flagged ``check_stale`` where the retrain
            moved them.  Until then every row reports the ``unchecked`` line,
            which is exactly what a headless run exports.  ``"off"`` never
            checks: the whole run is the unchecked line.  ``"weak"`` (#4496,
            the default since the owner's ruling of 2026-10-05: the app's
            Autopilot checks where
            :func:`~vtscore.training.thresholds.weak_check_due` says) is
            ``"end"`` plus the check the app runs mid-session: at the first
            ordinary step from *weak_min_t* votes on whose labels line
            separates weakly
            (:attr:`~vtscore.training.thresholds.LabelsLine.separation` below
            *weak_separation*), the user checks then and there.  Its picks are
            clicks: they count in ``t`` and the voting budget, and their rows
            carry ``phase == "prompt"``, so they read as ordinary clicks (frames
            record on them, at each requested click a round reaches).  It starts
            only when its opening bands fit the voting budget left, and a deeper
            band that would overrun the budget ends it there, as a user closing
            the step does, so a run's clicks still end at *max_steps*.  With
            *weak_repeat* > 0 the prompt returns once that many votes have
            been cast since the last prompted check ended, while the labels
            still separate weakly; 0 prompts once.  Ignored on the Inclusion
            arm, and when nothing is left unvoted to check.
        weak_separation: The d' below which ``spot_check="weak"`` prompts (the app's ``WEAK_SEPARATION_D``).
        weak_min_t: The fewest votes before ``spot_check="weak"`` prompts (``WEAK_CHECK_MIN_VOTES``).
        weak_repeat: Votes after a prompted check before it may prompt again
            (``WEAK_CHECK_COOLDOWN``); 0 prompts once, the priced alternative.
        weak_phase: Where in Autopilot's flow ``spot_check="weak"`` may prompt.
            ``"learned"`` (the default, the app's) or ``"any"`` (the arm #4496
            priced first): ``"learned"`` prompts only once the flow has left
            its opening (``good``/``bad``/``more``, on the text or
            example sort), where the app trains no detector and so has neither
            a separation to read nor a ranking to check.  Without a flow
            (a non-Autopilot strategy) both prompt anywhere.
        label_quota: Whether the withheld half is scored as Test scores it
            under the app's label quota (#4643,
            :mod:`vtscore.detectors.label_quota`).  ``None`` (the default) is
            the app: on the ``"app"`` trainer, from the first Good vote until
            the votes meet the quota, a row's test metrics are the Goods'
            centroid's (:func:`_centroid_test`), cut on the withheld half, and
            ``detector_tier`` says ``centroid``; from the quota on they are the
            trained head's, as before.  The Train side - acquisition, the
            lights, the spot check - still runs on the trained head wherever
            there is a Good and a Bad, because the Train view's learned sort
            does.  ``False`` is the pre-#4643 arm: no row before the first Good
            and Bad, the head from there.  The standalone trainers are not the
            app and default to ``False``.

    Returns:
        List of row dicts.  Keys: ``seed, dataset, category, strategy, trainer,
        head, style, prevalence_arm, realized_prevalence, t, n_good, n_bad, phase,
        app_trained, cost, fpr, fnr, auroc, average_precision, train_seconds,
        xcal_seconds, pool_score_seconds, test_score_seconds, backend, device,
        elapsed_seconds``.  ``n_good``/``n_bad`` report the vote counts behind
        each row so callers can tell apart metrics learned from a 1-vs-1 model
        and a many-vs-many one.  Under the label quota the rows start at the
        first Good vote, and ``detector_tier`` (``centroid`` / ``trained``)
        says which detector a Test there gives.  ``app_trained`` is 1 exactly when the app would
        have had a trained detector on screen at that step: a threshold recorded
        where it is 0 is one no user would ever see, which is what issue #2788's
        cold-start degenerates turned out to be.  Under ``test_bands`` each row
        also carries :data:`~vtscore.eval.voting_columns.BAND_COLUMNS`.
    """
    import numpy as np  # noqa: PLC0415

    # Refused before any work, not on the way past the split: a run that dies
    # three minutes in on an argument combination readable at the door is a
    # SLURM array slot spent to learn nothing (#4044).
    _check_test_bands(test_bands, target_category, target_prevalence)
    _check_train_mix(train_mix, target_category, target_prevalence)
    if target_prevalence is not None and haystack_prevalence is not None:
        raise ValueError("target_prevalence and haystack_prevalence are two different arms; set one")
    check_live_threshold(live_threshold, safe_thresholds=safe_thresholds, live_cut_rule=live_cut_rule)

    # Cross-band cohorts are built from the images the filter below REMOVES, so
    # they have to be taken off the unfiltered pool (#4044).
    unfiltered = clips_dict

    mix_split: Optional[tuple[list[int], list[int], dict[str, list[int]]]] = None
    if train_mix is not None:
        cls, mix_label = scale_bands.parse_cell(target_category)
        assert mix_label is not None  # `_check_train_mix` refused this at the door
        clips_dict, mix_sim, mix_test, mix_cohorts, _ = scale_bands.paired_mix(
            clips_dict, cls, train_mix, sim_fraction=sim_fraction, seed=seed, label=mix_label
        )
        mix_split = (mix_sim, mix_test, mix_cohorts)

    # One filter for the whole cell, before anything reads a label: on a
    # scale-banded dataset an image can hold the category at the wrong size,
    # and every "not positive means negative" test below would score it as a
    # negative.  A no-op for every dataset that does not designate its cells.
    clips_dict = evaluable_pool(clips_dict, target_category)

    rng = np.random.RandomState(seed)
    # Note: no torch.manual_seed() here - train_model handles its own
    # RNG seeding via fork_rng, keeping it thread-safe.
    start_time = time.monotonic()

    diversity = _parse_opening_diversity(opening_diversity)
    if band_share is not None and (isinstance(band_share, bool) or not isinstance(band_share, int) or band_share < 1):
        raise ValueError(f"band_share must be an integer >= 1 (one pick in band_share) or None; got {band_share!r}")
    if more_walk not in MORE_WALKS:
        raise ValueError(f"more_walk must be one of {MORE_WALKS}; got {more_walk!r}")
    if more_walk != "seed" and (startup_schedule is not None or diversity is not None):
        raise ValueError(
            "more_walk='detector' walks the app's own opening; drop startup_schedule and opening_diversity"
        )
    knobs = _resolve_run_knobs(
        fold_count_schedule=fold_count_schedule,
        calibrate_count=calibrate_count,
        startup_schedule=startup_schedule,
        seed_scores=seed_scores,
        autopilot_fidelity=autopilot_fidelity,
        strategy=strategy,
        skyline_arms=skyline_arms,
        emit_calibration_metrics=emit_calibration_metrics,
        acq_inclusion_offset=acq_inclusion_offset,
        acq_rank_percentile=acq_rank_percentile,
        head=head,
        acq_p_crossing=acq_p_crossing,
        acq_origin=acq_origin,
        acq_target_p=acq_target_p,
        trainer=trainer,
        style=style,
        calibration_seed=calibration_seed,
    )
    _fold_schedule = knobs.fold_schedule
    startup_state = knobs.startup_state
    skyline_arms = knobs.skyline_arms
    head = knobs.head
    # Resolved to the app's pinned split seed unless this cell is a #3794
    # calibration-noise arm; from here on it is an int, never ``None``.
    calibration_seed = knobs.calibration_seed
    # Normalised once, at the top: the retired ``"mlp"`` spelling never reaches
    # the dispatch, the guards, or the result rows (issue #3764).
    trainer = knobs.trainer
    # The balance the reporting line is drawn at (#4413): unpinned is the app's
    # default (DEFAULT_BETA), so the default arm cuts where a live detector
    # does; ``"off"`` is the Inclusion arm and a number the balance arm.
    # Resolved - and so validated - before anything expensive runs.
    beta = resolve_line_knobs(beta)
    walk_shape_resolved = resolve_walk_shape(walk_shape, beta)
    # What the line reads a finished walk as: the app's shape when the arm names none; the full walk's end moves the line.
    line_shape = None if walk_shape is None else (CHECK_TRIM if walk_shape == WALK_SHAPE_FULL else walk_shape)
    # The arm draws the app's labels line (#4452): a balance with no forced check shape.
    labels_arm = beta is not None and line_shape is None
    _check_inclusion_arm(inclusion, beta)
    # Test's label quota (#4643): the app's rule on the app's trainer unless the
    # arm says otherwise.  A standalone trainer is not the app.
    quota_on = (trainer == APP_TRAINER) if label_quota is None else bool(label_quota)
    # The acquisition cut's rule (#4409): the shipped argmax factor under a
    # balance unless the arm says otherwise, the offset everywhere else.
    acq_factor = resolve_acquisition_factor(acq_p_crossing, beta)
    if acq_factor is not None and beta is None:
        raise ValueError("acq_p_crossing needs a balance: it places the acquisition cut at its F-beta argmax (#4413)")

    prevalence_arm = "natural" if target_prevalence is None else f"rare_{target_prevalence:g}"
    if target_prevalence is not None:
        # Thin positives to the target prevalence *before* splitting, so both the
        # votable sim pool and the held-out test pool sit at that prevalence.
        downsampled = _downsample_to_prevalence(clips_dict, target_category, target_prevalence, rng)
        if downsampled is None:
            return []  # too few positives survive - skip this arm
        clips_dict = downsampled
    realized_prevalence = round(_prevalence(clips_dict, target_category), 6)

    if mix_split is None:
        sim_ids, test_ids = _split_media_ids(clips_dict, sim_fraction, rng)
        band_cohorts = _resolve_band_cohorts(
            unfiltered,
            target_category,
            test_bands=test_bands,
            sim_fraction=sim_fraction,
            seed=seed,
            own_test_ids=test_ids,
            target_prevalence=target_prevalence,
        )
    else:
        # The mixed arm's split is `paired_mix`'s: re-splitting the mixed pool
        # would train on images a pure arm holds out.
        sim_ids, test_ids, all_cohorts = mix_split
        wanted = None if test_bands in (None, "auto") else set(test_bands)
        band_cohorts = {b: ids for b, ids in all_cohorts.items() if wanted is None or b in wanted}

    # A smaller Train pool (#4452's wider world): a seeded subsample of the
    # simulation half, after the split so the withheld half - the Find side -
    # is the full one whatever the pool's size.
    if sim_size is not None and len(sim_ids) > int(sim_size):
        sub_rng = np.random.RandomState(int(seed) + 7919)
        keep = sub_rng.choice(len(sim_ids), size=int(sim_size), replace=False)
        sim_ids = [sim_ids[int(i)] for i in sorted(keep)]
        prevalence_arm = f"sim_{int(sim_size)}"
        realized_prevalence = round(
            sum(1 for cid in sim_ids if media_is_positive(clips_dict[cid], target_category)) / len(sim_ids), 6
        )

    # After the split and the cohorts, so neither moves (#4184).
    if haystack_prevalence is not None:
        sim_ids = thin_haystack(clips_dict, sim_ids, target_category, haystack_prevalence, seed)
        prevalence_arm = f"haystack_{haystack_prevalence:g}"
        realized_prevalence = round(
            sum(1 for cid in sim_ids if media_is_positive(clips_dict[cid], target_category)) / len(sim_ids), 6
        )

    # Ensure the test set has both positive and negative medias.  Routes through
    # ``media_is_positive`` so multi-label (Visual Genome) images - where the
    # target may be a non-primary category - are counted correctly.
    test_pos = [cid for cid in test_ids if media_is_positive(clips_dict[cid], target_category)]
    test_neg = [cid for cid in test_ids if not media_is_positive(clips_dict[cid], target_category)]
    if not test_pos or not test_neg:
        return []
    band_neg_ids = test_neg if (test_bands and test_band_auroc) else None

    # A patch dataset exposes a ``patch_grid`` per media; such datasets are
    # scored region-aware (max-pool over the image's score rows) the same way
    # the live detector scores them, regardless of how the Good votes were
    # assembled.
    region_aware = any(clips_dict[cid].get("patch_grid") is not None for cid in clips_dict)

    # `region_voting` is a request, not a guarantee: `good_training_vec` pools
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
            "See docs/experiments/2026-08-07-acquisition-inclusion/REPORT_SECOND_ENVIRONMENT.md.",
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
    # historical ``_app_train_and_calibrate`` path runs byte-for-byte.  The
    # standalone-SVM trainers are untouched too - they have no head for a style
    # to drive.
    if style is None and region_aware and trainer == APP_TRAINER:
        style = PRODUCTION_PATCH_STYLE

    style_obj: Any = None
    if style is not None:
        from vtscore.eval.patch_styles import resolve_style  # noqa: PLC0415

        style_obj = resolve_style(style)

    # **v1 is the whole-image columns.**  Full supervision hands out *image*
    # labels, but the mortal max_patch flow trains on GT-box-pooled vectors (the
    # sim user drags a box), so a patch column's skyline is a design decision -
    # "oracle boxes + all images", or the multiple-instance problem of which
    # patch of a positive image is the positive - and not something to improvise
    # inside a metric row.  That decision is the named open item on #3321.  Skip
    # rather than raise: a sweep over `styles=["whole_image", "max_patch"]` should
    # get the decomposition on the column that has one instead of dying on the
    # column that doesn't, and the skip is loud here and visible in the frame
    # (no `skyline_*` rows, NaN decomposition columns) rather than silent.
    #
    # **v2 (#4159): a patch column under REGION voting gets `skyline_train_full`
    # with oracle boxes.** Owner ruling on #3321's open item: supervise with each
    # positive's ground-truth box and every image, which is exactly what
    # `_train_and_calibrate(region_voting=True)` already does for a mortal Good
    # vote -- the sim user drags the GT box -- so the skyline still differs from a
    # mortal step in the labels and nothing else. It needs datasets whose
    # positives carry the box (`coco_better` carries exactly one per positive,
    # #4096). The cross-fitted bracket stays whole-image only: cross-fitting a
    # box-supervised head over the TEST split is a second design nobody has made.
    if skyline_arms and (style_obj is None or style_obj.name != _WHOLE_IMAGE_STYLE):
        import warnings  # noqa: PLC0415

        keep = [a for a in skyline_arms if a == SKYLINE_TRAIN_FULL] if (region_voting and style_obj is not None) else []
        dropped = [a for a in skyline_arms if a not in keep]
        if dropped:
            why = (
                "the cross-fitted bracket is whole-image only"
                if region_voting and style_obj is not None
                else "a patch column's skyline is defined only under region voting (oracle boxes, #4159)"
            )
            warnings.warn(
                f"skyline_arms={dropped} skipped for style={style or 'none'!r}: {why}; see issues #3321, #4159.",
                RuntimeWarning,
                stacklevel=2,
            )
        skyline_arms = keep
    blend_schedule, calibration_fraction = _resolve_production_defaults(
        blend_schedule=blend_schedule,
        calibration_fraction=calibration_fraction,
        region_aware=region_aware,
    )

    import torch  # noqa: PLC0415

    # Whole-image embeddings of the simulation pool.  These feed the autopilot
    # selector (the example-sort good centroid and the coverage atlas); the
    # Good-vote *training* vector can still be region-pooled below when
    # ``region_voting`` is on.
    sim_embeddings: dict[int, np.ndarray] = {
        cid: np.asarray(media_embedding(clips_dict[cid]), dtype=np.float32) for cid in sim_ids
    }
    input_dim = int(next(iter(sim_embeddings.values())).shape[0])

    # The rank-transfer haystack for a ``gp_*`` arm (issue #3954): the
    # simulation set's whole-image vectors, in id order.  ``None`` keeps every
    # trainer on its raw-score cut.
    if standalone_cut not in ("raw", "rank", "anchored"):
        raise ValueError(f"standalone_cut must be 'raw', 'rank' or 'anchored', got {standalone_cut!r}")
    fold_anchored = standalone_cut == "anchored"
    if fold_anchored:
        if not trainer.startswith("gp_"):
            raise ValueError(f"standalone_cut='anchored' applies to the gp_* trainers only; got trainer={trainer!r}")
        if region_aware:
            raise ValueError(
                "standalone_cut='anchored' needs a single-vector dataset (the gp_* arms score whole images)"
            )
        if not safe_thresholds:
            raise ValueError(
                "standalone_cut='anchored' needs safe_thresholds=True: the fold-anchored estimator runs there"
            )
    haystack_X: "np.ndarray | None" = None
    if standalone_cut == "rank":
        if not trainer.startswith("gp_"):
            raise ValueError(f"standalone_cut='rank' applies to the gp_* trainers only; got trainer={trainer!r}")
        if region_aware:
            raise ValueError("standalone_cut='rank' needs a single-vector dataset (the gp_* arms score whole images)")
        haystack_X = np.stack([sim_embeddings[cid] for cid in sorted(sim_ids)])

    # The autopilot New phase reads a coverage atlas built over the pool; it is
    # labelled in lock-step with the votes below so its coverage advances.
    atlas = _build_eval_atlas(sim_embeddings, atlas_min_node_size) if is_autopilot_strategy(strategy) else None

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
    #: The phase that surfaced each vote (#4224: which votes calibrated the cut).
    vote_phase: dict[int, str] = {}
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
    step: StepModel | None = None
    threshold = 0.5
    #: The selector's threshold - cut ``acq_inclusion_offset`` steps below the
    #: reporting one.  Kept as its own name so the two jobs cannot silently
    #: re-merge (they were one variable, and that is how the #2847 positives
    #: regression got in).
    acq_threshold = 0.5
    pool_scores: dict[int, float] = {}
    # The detector's per-item spread over the pool, for the uncertainty-driven
    # strategies (issue #3954).  ``None`` until a trainer that reports one has
    # scored the pool; the app trainer never does.
    pool_uncertainty: dict[int, float] | None = None
    n_steps = len(pool) if max_steps is None else min(max_steps, len(pool))

    # The app's phase machine, driving the vote order the way Autopilot does.
    # Disabled (``None``) under ``autopilot_fidelity=False``, which leaves the
    # selector on its legacy parity interleave.
    flow: Any = None
    if autopilot_fidelity and is_autopilot_strategy(strategy):
        flow = AutopilotFlow(startup=startup_state, smart_gate=smart_gate)
    elif more_walk != "seed":
        raise ValueError(
            "more_walk='detector' needs Autopilot's phase machine (an autopilot strategy, autopilot_fidelity)"
        )
    if band_share is not None and (flow is None or beta is None):
        raise ValueError("band_share needs Autopilot's phase machine and a balance (beta)")
    # #4482's band picks: how many picks past the opening, and how many of them were band picks.
    learned_picks = 0
    band_picks = 0
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

    def _cast(cid: int, phase_name: str | None) -> bool:
        """Reveal *cid*'s ground truth as a vote; whether it was positive."""
        pool.remove(cid)
        vote_phase[cid] = phase_name or ""
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
        return is_positive

    def _log_pick(cid: int, is_positive: bool, phase_name: str | None, startup_round: int, startup_cut) -> None:
        """One pick-log row, after the vote landed (the counts are post-vote)."""
        if pick_sink is None:
            return
        rank = seed_rank.get(cid, -1)
        n_sorted = len(seed_rank)
        pick_sink.append(
            {
                "seed": seed,
                "dataset": dataset_name,
                "category": target_category,
                "startup_schedule": startup_schedule or "",
                "calibration_seed": calibration_seed,
                "style": style or "",
                "t": len(good_votes) + len(bad_votes),
                "phase": phase_name or "",
                "startup_round": startup_round,
                "startup_held": bool(startup_state.held_for_quorum) if startup_state is not None else False,
                "startup_extended_clicks": int(startup_state.extended_clicks) if startup_state is not None else 0,
                "startup_cut": round6(startup_cut) if startup_cut is not None else float("nan"),
                "startup_cut_percentile": (
                    _sorted_percentile(seed_sorted_scores, startup_cut) if startup_cut is not None else float("nan")
                ),
                "picked_id": cid,
                "picked_label": 1 if is_positive else 0,
                "picked_seed_rank": rank,
                "picked_seed_percentile": (
                    round6(rank / (n_sorted - 1)) if n_sorted > 1 and rank >= 0 else float("nan")
                ),
                "picked_seed_score": round6(seed_scores[cid]) if seed_scores and cid in seed_scores else float("nan"),
                "picked_detector_score": round6(pool_scores[cid]) if cid in pool_scores else float("nan"),
                "acq_threshold": round6(acq_threshold),
                "n_good": len(good_votes),
                "n_bad": len(bad_votes),
                "n_pool": len(pool),
            }
        )

    if spot_check not in ("end", "off", "weak"):
        raise ValueError(f"spot_check must be 'end', 'off' or 'weak', got {spot_check!r}")
    if weak_phase not in ("any", "learned"):
        raise ValueError(f"weak_phase must be 'any' or 'learned', got {weak_phase!r}")
    # The balance's spot check (#4272, #4413), run once the voting steps are
    # spent: the simulated user checks the line as the app's check step does,
    # its picks answered from ground truth and cast as votes.  ``line_ranking``
    # is the ranking the last step's line was drawn over, which the walk's
    # bands are cut from and a finished result is fingerprinted against.
    check: SpotCheck | None = None
    line_ranking: LineRanking | None = None
    # The rank frame's identity, and the inputs of the last ordinary step's
    # frame (#4357), which is emitted once the loop ends.
    rank_ident = {
        "seed": seed,
        "dataset": dataset_name,
        "category": target_category,
        "calibration_seed": calibration_seed,
        "style": style or "",
    }
    last_ordinary: dict[str, Any] | None = None
    # ``t`` counts every vote cast, the check's included: one per ordinary
    # step, a round's worth per check round.
    # #4496: under ``spot_check="weak"`` the app prompts a check when the labels
    # separate weakly.  ``check_phase`` stamps the running check's rows and
    # picks: ``"prompt"`` for a prompted check (clicks), ``"check"`` for the
    # end-of-run one.
    check_phase = "check"
    end_checked = False
    weak_due = False
    last_weak_check: int | None = None

    def _start_check() -> "SpotCheck | None":
        """The app's check over the line's unvoted ranking; ``None`` with nothing unvoted in it."""
        assert line_ranking is not None and beta is not None
        # The whole unvoted ranking, in rank order: the walk's bands are cut
        # from it (#4388), and it starts at the balance's cap.
        candidate = tuple(int(i) for i in line_ranking.unvoted_ids(set(good_votes) | set(bad_votes)))
        if not candidate:
            return None
        # Seeded off the run's own RNG, after every trajectory draw, so a run
        # without the check is byte-identical up to here.  Recall is read
        # against the mixture's count of the unvoted ranking's positives, or
        # the balance's cap when the mixture has no estimate (#4419), as the
        # app does.
        n_pos = walk_positives(
            line_ranking,
            beta,
            {**dict.fromkeys(good_votes, True), **dict.fromkeys(bad_votes, False)},
            set(good_votes) | set(bad_votes),
        )
        return SpotCheck.start_balance(
            candidate,
            beta,
            n_pos,
            seed=int(rng.randint(2**31 - 1)),
            picks=walk_picks,
            tol=walk_tol,
            fine=walk_fine,
            guard=walk_guard,
            shallow_only=walk_shape_resolved == CHECK_TRIM,
        )

    # The last finished check: what a prompted check the budget cut short leaves in place.
    finished_check: SpotCheck | None = None

    def _check_start_cost() -> int:
        """The most a check's opening bands cost: what a prompted one needs left in the voting budget."""
        if beta is None:
            return 0
        schedule = balance_schedule(beta)
        return schedule.rounds * (walk_picks or schedule.picks)

    t = 0

    def _ident_row(
        t: int,
        row_phase: str,
        acq_threshold: float,
        threshold: float,
        pool_scores: dict[int, float],
        details: dict[str, Any],
        tier: str,
    ) -> dict[str, Any]:
        """The identifying columns shared by every row a step emits.

        Read off the run's live state (the votes, the pool, the phase machine)
        as it stands when called, plus what the step computed.  A step with no
        trained head yet (a Good and no Bad, under the label quota) passes NaN
        lines, no pool scores and no details beyond the balance.
        """
        return {
            "seed": seed,
            "dataset": dataset_name,
            "category": target_category,
            "strategy": strategy,
            "trainer": trainer,
            # Blank on the standalone SVM trainers: they fit no head, so
            # naming one here would attribute the row to a head never trained.
            "head": head if trainer == APP_TRAINER else "",
            "style": style or "",
            "prevalence_arm": prevalence_arm,
            "realized_prevalence": realized_prevalence,
            "t": t,
            "n_good": len(good_votes),
            "n_bad": len(bad_votes),
            # The haystack the threshold was fitted on, and what is left of it
            # after this step's votes.  `n_remainder` is *exactly* the quantity
            # the #3308 exclusion floor is compared against
            # (`apply_vote_exclusion` counts the unvoted scores), and `pool` has
            # already had this step's vote removed by the time this row is
            # built - so an analyzer can reconstruct, per step, whether the
            # exclusion fired, without the harness having to report it (#3312).
            # Their ratio is the axis the mechanism runs on: the effect is
            # bounded by the votes' share of the haystack.
            "n_haystack": len(sim_ids),
            "n_remainder": len(pool),
            "phase": row_phase,
            # The three lights behind that phase (#3560).  Already computed by
            # `flow.update` above and previously discarded; the phase alone
            # cannot say whether Smart or Stable is what holds a run in `hard`.
            "smart": flow.smart if flow is not None else "",
            "stable": flow.stable if flow is not None else "",
            "span": flow.span if flow is not None else "",
            "span_level": flow.span_level if flow is not None else -1,
            "span_depth": flow.span_depth if flow is not None else -1,
            "span_target": flow.span_target if flow is not None else -1,
            # How close each rule came to firing, beside whether it did (#3560).
            # Read off the dicts the phase machine already built this step, so a
            # margin cannot disagree with the light above it; NaN where no phase
            # machine ran, or where the rule itself declined to fit one.
            **{col: (getattr(flow, col) if flow is not None else float("nan")) for col in STOPPING_MARGIN_COLUMNS},
            "app_trained": 1
            if (flow is None or app_has_detector(flow.phase, more_shown=more_walk == "detector"))
            else 0,
            "startup_schedule": startup_schedule or "",
            "calibration_seed": calibration_seed,
            "acq_threshold": round(float(acq_threshold), 6),
            # Measured against the pool the selector ranks, not the test set, so
            # the pair answers "how much did the sampling position move".
            "acq_pool_percentile": _pool_percentile(pool_scores, acq_threshold),
            "report_pool_percentile": _pool_percentile(pool_scores, threshold),
            **_line_columns(details),
            # Which detector a Test at this click gives (#4643): the Goods'
            # centroid under the label quota, else the trained head.
            "detector_tier": tier,
        }

    while True:
        picks: list[int] | None = None
        # The vote count before this step's votes: a frame is due at every
        # requested click this step reaches, which a check round can jump past.
        t_before = t
        if check is not None and check.running:
            if check_phase == "prompt" and t + len(check.pending) > n_steps:
                # The budget is spent mid-check: the user stops there, as closing
                # the step does, and the end-of-run check takes over (#4496).
                check.cancel()
                check = finished_check
                last_weak_check = t
            else:
                picks = list(check.pending)
        if picks is None and not (check is not None and check.running):
            if weak_due and t + _check_start_cost() <= n_steps and pool and line_ranking is not None:
                # The labels separate weakly (#4496): the user checks now, and
                # the check's picks are clicks, so its opening bands must fit
                # the budget left.
                weak_due = False
                last_weak_check = t
                started = _start_check()
                if started is not None and started.pending:
                    check, check_phase = started, "prompt"
                    picks = list(check.pending)
            if picks is None and (t >= n_steps or not pool):
                # The voting steps are spent.  Check the line once, if the run
                # checks at all and there is a ranking with something unvoted in it.
                if spot_check == "off" or end_checked or beta is None or line_ranking is None:
                    break
                end_checked = True
                started = _start_check()
                if started is None:
                    break
                check, check_phase = started, "check"
                picks = list(check.pending)
                if not picks:
                    break

        if picks is not None:
            # A check round: every pick is answered at once.  The candidate was
            # fixed at the start, so the retrain each round triggers cannot
            # move what the next round samples.
            phase = check_phase
            startup_round, startup_cut = -1, None
            round_votes = {cid: _cast(cid, phase) for cid in picks}
            t = len(good_votes) + len(bad_votes)
            for cid in picks:
                _log_pick(cid, round_votes[cid], phase, startup_round, startup_cut)
            is_positive = round_votes[picks[-1]]
            assert check is not None and line_ranking is not None
            check.record(round_votes)
            if check.finished and check_phase == "prompt":
                last_weak_check = t
            if check.finished:
                finished_check = check
                # The set the line keeps from here on, as it stands with the
                # check's own votes cast: what the retrains below are
                # compared against for ``check_stale``.
                check.fingerprint = line_ranking.fingerprint(check.k, set(good_votes) | set(bad_votes))
        else:
            phase = flow.phase if flow is not None else None
            startup_round = startup_state.index if (startup_state is not None and not startup_state.done) else -1
            startup_cut = startup_cuts[startup_round] if startup_round >= 0 else None
            cid = None
            pick_phase = phase
            if band_share is not None and phase is not None and app_has_detector(phase) and pool_scores:
                # #4482: every band_share-th pick past the opening is a band pick, on its
                # own generator so the rest of the run's draws are the arm-free run's.
                learned_picks += 1
                if learned_picks % band_share == 0:
                    assert beta is not None
                    cid = band_pick(
                        pool,
                        pool_scores,
                        balance_schedule(beta).candidate,
                        band_picks,
                        np.random.default_rng([seed, 4482, len(good_votes) + len(bad_votes)]),
                    )
                    if cid is not None:
                        band_picks += 1
                        pick_phase = BAND_PHASE
            if cid is None:
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
                    uncertainty=pool_uncertainty,
                    opening_diversity=diversity,
                    more_walk=more_walk,
                )
                cid = select_next(strategy, ctx)
            is_positive = _cast(cid, pick_phase)
            t = len(good_votes) + len(bad_votes)
            _log_pick(cid, is_positive, pick_phase, startup_round, startup_cut)

        n_votes_now = len(good_votes) + len(bad_votes)
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
                    last_vote_good=is_positive,
                )
            if quota_on and good_votes:
                # No head without a Bad, but Test gives the Goods' centroid
                # from the first Good (#4643), so this click has a row.  The
                # Train side has nothing to record: no line, no pool scores.
                # The row's balance is the one a trained step records, which
                # only the safe-threshold path draws a line at.
                row_beta = beta if safe_thresholds else None
                t_test = time.monotonic()
                c_step, c_rows, _c_cal = _centroid_test(
                    good_votes,
                    clips_dict,
                    test_ids,
                    target_category,
                    inclusion,
                    region_voting=region_voting,
                    region_aware=region_aware,
                    style_obj=style_obj,
                    beta=row_beta,
                    calibration_rows=emit_calibration_metrics and style_obj is not None,
                )
                c_seconds = time.monotonic() - t_test
                c_bands = _band_metrics(
                    c_step,
                    CENTROID_THRESHOLD,
                    unfiltered,
                    band_cohorts if test_bands else None,
                    region_aware=region_aware,
                    style_obj=style_obj,
                    neg_ids=band_neg_ids,
                    target_category=target_category,
                )
                c_ident = _ident_row(
                    t,
                    flow.phase if flow is not None else "",
                    float("nan"),
                    float("nan"),
                    {},
                    {"beta": row_beta},
                    "centroid",
                )
                c_timing = {
                    "calibrate_count": 0,
                    "train_seconds": float("nan"),
                    "final_score_seconds": float("nan"),
                    "xcal_seconds": float("nan"),
                    "pool_score_seconds": float("nan"),
                    "test_score_seconds": round(c_seconds, 6),
                    "backend": c_step.backend,
                    "device": c_step.device,
                    "elapsed_seconds": round(time.monotonic() - start_time, 3),
                }
                for mr in c_rows:
                    rows.append({**c_ident, **mr, **c_bands, **c_timing})
            continue

        # The live fold count for THIS step.  Constant unless #3314's schedule
        # knob is set, and then a function of the vote count only - never of
        # anything the step has already computed, so the count is decided
        # before the fit and cannot depend on it.
        step_calibrate_count = calibrate_count if _fold_schedule is None else _fold_schedule(n_votes_now)
        step, threshold, n_labels, timings, details = _train_and_calibrate(
            trainer,
            good_votes,
            bad_votes,
            clips_dict,
            target_category,
            region_voting=region_voting,
            input_dim=input_dim,
            inclusion=inclusion,
            calibrate_count=step_calibrate_count,
            calibration_fraction=calibration_fraction,
            head=head,
            style_obj=style_obj,
            emit_calibration_metrics=emit_calibration_metrics,
            fold_count_variants=fold_count_variants,
            calibration_seed=calibration_seed,
            haystack_X=haystack_X,
            fold_anchored=fold_anchored,
        )

        # Apply the shipped safe threshold if enabled
        sim_pooled_scores: list[float] | None = None
        sim_pooled_ids: list[int] = []
        sim_fold_haystacks: list[Any] = []
        safe_cut = None
        if safe_thresholds:
            # What the fold computation returned before any fusion: a retired
            # live rule (#4184) falls back to it where it has nothing to cut on.
            fold_threshold = threshold
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
                    exclusion_min_remainder=exclusion_min_remainder,
                    cut_rule=live_cut_rule,
                    check=check,
                    labels={**dict.fromkeys(good_votes, True), **dict.fromkeys(bad_votes, False)},
                    beta=beta,
                    line_shape=line_shape,
                )
            )
            line_ranking = details.get("line_ranking")
            if (
                spot_check == "weak"
                and picks is None
                and (weak_phase == "any" or flow is None or flow.phase not in _OPENING_PHASES)
            ):
                weak_line = details.get("find_line")
                # The app's rule (#4496): Autopilot checks where this says.
                if weak_line is not None and weak_check_due(
                    weak_line.separation,
                    t,
                    last_weak_check,
                    threshold=weak_separation,
                    min_votes=weak_min_t,
                    cooldown=weak_repeat if weak_repeat > 0 else None,
                ):
                    weak_due = True
            if live_threshold is not None:
                # A retired rung replaces the shipped cut, and the fit it
                # replaced is dropped with it so acquisition cannot re-cut an
                # estimator this arm does not run.
                threshold, safe_provenance = retired_live_threshold(
                    live_threshold,
                    fold_threshold=fold_threshold,
                    details=details,
                    haystack_scores=sim_pooled_scores,
                    ctx=blend_ctx,
                    schedule=blend_schedule,
                    fitted_cut=safe_cut,
                    shipped_threshold=threshold,
                    shipped_provenance=safe_provenance,
                    inclusion=inclusion,
                )
                safe_cut = None
                # The retired rung drew this line, not the operating point.
                details.pop("reporting_line", None)
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
            # The offset cut: re-cut the *same* fold-anchored fit.  O(1) - the
            # mixture was fitted above; ``threshold_at`` is monotone by
            # construction, so the arms are nested and offset 0 reproduces the
            # reporting cut exactly.  ``safe_cut is None`` is the schedule-blend
            # fallback (~5% of steps, concentrated in the cold start): the blend
            # has no inclusion-aware form, so there is nothing honest to re-cut.
            # The offset's origin is the inclusion the line sits at: the knob's
            # under the Inclusion arm, and under a preference the fallback's or
            # the one a promised line derives to (#4245).  It is the cut of the
            # offset arm, and what the argmax arm falls back to with no mixture
            # estimate, as the app does (#4409).
            offset_cut: float | None = None
            # The target precision names the cut unless an explicit factor or rank pin does.
            acq_target = (
                resolve_acquisition_target(acq_target_p, beta)
                if acq_factor is None and acq_rank_percentile is None
                else None
            )
            if acq_inclusion_offset != 0 and safe_cut is not None:
                line = details.get("reporting_line") if acq_origin == "line" else None
                origin = line_inclusion(line, safe_cut) if line is not None else inclusion
                cand = safe_cut.threshold_at(
                    acquisition_inclusion(origin if origin is not None else inclusion, acq_inclusion_offset)
                )
                if np.isfinite(cand):
                    offset_cut = float(cand)
            if acq_factor is not None:
                # #4409: sample at a share of the depth of the mixture's F-beta
                # argmax over the unvoted ranking (no cap), read as a rank,
                # through the library's acquisition_threshold, as the app does
                # (#4413).  The mixture is the one the line's proposal already
                # fitted on this ranking (memoised on it), so this costs a
                # posterior read.
                ranking_now = details.get("line_ranking")
                cand = None
                if ranking_now is not None and beta is not None:
                    voted_now = set(good_votes) | set(bad_votes)
                    labels_now = {**dict.fromkeys(good_votes, True), **dict.fromkeys(bad_votes, False)}
                    cand = acquisition_threshold(ranking_now, beta, labels_now, voted_now, factor=acq_factor)
                if cand is not None and np.isfinite(cand):
                    acq_threshold = float(cand)
                elif offset_cut is not None:
                    acq_threshold = offset_cut
            elif acq_target is not None:
                # #3546: sample where the labels line's corpus posterior falls below the
                # target share, as the app does; the offset cut is the fallback.
                cand = target_precision_threshold(details.get("find_line"), acq_target)
                if cand is not None and np.isfinite(cand):
                    acq_threshold = float(cand)
                elif offset_cut is not None:
                    acq_threshold = offset_cut
            elif acq_rank_percentile is not None:
                if sim_pooled_scores:
                    acq_threshold = float(
                        np.quantile(np.asarray(sim_pooled_scores, dtype=np.float64), acq_rank_percentile)
                    )
            elif offset_cut is not None:
                acq_threshold = offset_cut

        details.pop("find_threshold", None)
        details.pop("find_on_test", None)
        # Under the label quota a Test here gives the Goods' centroid, not this
        # head (#4643): the withheld half is scored as that.
        centroid = quota_on and label_quota_tier(len(good_votes), len(bad_votes)).tier == TIER_CENTROID
        centroid_step: StepModel | None = None
        # Evaluate on the held-out test set.  The calibration study (#2781)
        # emits one row per pooling (base + remedial) instead of the single
        # metrics row, but both paths score the same test set here.
        calibration: tuple[list[dict[str, Any]], np.ndarray, np.ndarray, list[int]] | None = None
        metrics: dict[str, float] = {}
        t_test = time.monotonic()
        if centroid:
            centroid_step, c_rows, calibration = _centroid_test(
                good_votes,
                clips_dict,
                test_ids,
                target_category,
                inclusion,
                region_voting=region_voting,
                region_aware=region_aware,
                style_obj=style_obj,
                beta=details.get("beta"),
                calibration_rows=emit_calibration_metrics and style_obj is not None,
            )
            if calibration is None:
                metrics = c_rows[0]
        elif emit_calibration_metrics and style_obj is not None:
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
            if test_score_sink is not None:
                # The withheld half as Find sees it (#4452's Find scenarios):
                # its scores and labels, Train's threshold and the labels' class
                # model, so any Find corpus drawn from it is priced post hoc.
                find_line = details.get("find_line")
                test_score_sink.append(
                    {
                        "t": int(t),
                        "phase": check_phase if picks is not None else (flow.phase if flow is not None else ""),
                        # Full precision (#4523): the Test arm replays this snapshot, and a
                        # float32 score or a rounded model moves its line and its tail.
                        "scores": np.asarray(calibration[1], dtype=np.float64),
                        "labels": np.asarray(calibration[2], dtype=np.int8),
                        # Which image each score is (#4490): a report can name the wrong images a line keeps.
                        "ids": np.asarray(calibration[3], dtype=np.int64),
                        "train_threshold": float(threshold),
                        "beta": float(details["beta"]) if details.get("beta") is not None else float("nan"),
                        "model": None if find_line is None else _model_meta(find_line.model),
                        **_fold_arrays(details.get("fold_orderings")),
                    }
                )
        else:
            scored: list[Any] = []
            metrics = _evaluate_on_test(
                step,
                threshold,
                clips_dict,
                test_ids,
                target_category,
                inclusion,
                region_aware=region_aware,
                style_obj=style_obj,
                scored_sink=scored,
                find_line=details.get("find_line"),
                beta=details.get("beta"),
                out=details,
            )
            if emit_calibration_metrics and trainer != APP_TRAINER and scored:
                # A standalone trainer has no style, so it never reaches the
                # calibration rows above - which is where the oracle cut, the
                # regret split and the provenance live.  Without them a head
                # comparison cannot say whether an arm lost on its ranking or
                # on its cut (the #3954 pilot's open gap), so the same base
                # row is built here from the same test pass (issue #3959).
                metrics = {**metrics, **_standalone_calibration_row(threshold, details, scored, inclusion)}
        test_score_seconds = time.monotonic() - t_test

        # The per-size breakdown (#4044), at the shipped cut and therefore the
        # same on every row this step emits -- including the calibration study's
        # per-pooling rows, whose own thresholds re-cut the headline columns and
        # not this one.
        band_metrics = _band_metrics(
            centroid_step if centroid_step is not None else step,
            # The size bands are cohorts of the withheld half, so they are cut
            # where Find cuts it (#4452); the Train side's threshold otherwise.
            CENTROID_THRESHOLD if centroid_step is not None else details.get("find_threshold", threshold),
            unfiltered,
            band_cohorts if test_bands else None,
            region_aware=region_aware,
            style_obj=style_obj,
            neg_ids=band_neg_ids,
            target_category=target_category,
        )

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
        pool_uncertainty = _pool_uncertainty(step, pool, clips_dict)

        # Advance the app's phase machine on this step's model: the Smart
        # indicator needs the labelset error cost, Stable the prediction flips
        # over the still-unlabeled pool, Span the atlas's coverage.
        if flow is not None:
            # Scored at the model's own Inclusion 0 cut and priced at that
            # inclusion, as the app's Smart is, whatever this arm reports at
            # (issue #4243).  A step with no fitted cut to re-derive keeps its
            # reporting line, which is then inclusion-blind.
            # The line was served at the operating point's inclusion - none at
            # all when a balance kept a set (#4272, #4413).
            _line = details.get("reporting_line")
            _served = _line.inclusion if _line is not None else inclusion
            recent_steps.append(
                (step, smart_cut(threshold, _served, safe_cut.threshold_at if safe_cut is not None else _no_recut))
            )
            # The app regresses over its last SMART_WINDOW *models*; here every
            # step trains one, so the last SMART_WINDOW steps are the same set.
            del recent_steps[:-SMART_WINDOW]
            flow.record_step(
                _labelset_error_costs(
                    recent_steps,
                    good_votes,
                    bad_votes,
                    clips_dict,
                    SMART_INCLUSION,
                    region_aware=region_aware,
                    style_obj=style_obj,
                ),
                pool_scores,
                threshold,
                # Rates are over the haystack the votes came out of, labeled
                # items included, as the app divides by its whole scored pool.
                num_pool=len(sim_ids),
            )
            flow.update(
                len(good_votes),
                len(bad_votes),
                remaining_unlabeled=len(pool),
                span=atlas.span_info() if atlas is not None else None,
                last_vote_good=is_positive,
            )

        # Identifying columns shared by every row this step emits.
        base_row = _ident_row(
            t,
            check_phase if picks is not None else (flow.phase if flow is not None else ""),
            acq_threshold,
            threshold,
            pool_scores,
            details,
            "centroid" if centroid else "trained",
        )
        timing_cols = {
            # The fold count this step actually LIVED at.  Constant on every run
            # but #3314's scheduled arm - and recorded regardless, because a
            # run-level knob whose only other record is the directory the cells
            # were read out of is unreadable in a frame concatenated across arms
            # (#3287's `calibration_fraction` lesson, one knob over).
            "calibrate_count": step_calibrate_count,
            "train_seconds": round(timings["train_seconds"], 6),
            #: The final model's own pass over the haystack, on the shipped
            #: safe-threshold path (#3314).  NaN when safe thresholds are off,
            #: where there is no such pass.  K-independent, so a per-step cost
            #: ratio wants it in the denominator.
            "final_score_seconds": round(float(details.get("final_score_seconds", float("nan"))), 6),
            "xcal_seconds": round(timings["xcal_seconds"], 6),
            "pool_score_seconds": round(pool_score_seconds, 6),
            "test_score_seconds": round(test_score_seconds, 6),
            "backend": step.backend,
            "device": step.device,
            "elapsed_seconds": round(time.monotonic() - start_time, 3),
        }

        if calibration is not None and centroid:
            # The Goods' centroid's row (#4643).  The study extras below all
            # vary the trained head's cut, which no Test gives at this click.
            # The Test arm still starts from here if this is the last click:
            # the centroid's line keeps the same set at every balance.
            metric_rows, base_scores, base_labels, base_ids = calibration
            if (rank_frame_sink is not None or line_test_sink is not None) and (
                picks is None or check_phase == "prompt"
            ):
                last_ordinary = {
                    "t": t,
                    "test_ids": base_ids,
                    "test_scores": base_scores,
                    "test_labels": base_labels,
                    "pool_ranking": line_ranking,
                    "voted": frozenset(good_votes) | frozenset(bad_votes),
                    "pool_labels": pool_labels,
                    "vote_labels": {**dict.fromkeys(good_votes, True), **dict.fromkeys(bad_votes, False)},
                    "find_on_test": None,
                    "fallback_threshold": CENTROID_THRESHOLD,
                }
            for mr in metric_rows:
                rows.append({**base_row, **mr, **band_metrics, **timing_cols})
        elif calibration is not None:
            metric_rows, base_scores, base_labels, base_ids = calibration
            # A prompted check's rounds are clicks (#4496); the end-of-run check's are not.
            is_click = picks is None or check_phase == "prompt"
            if (rank_frame_sink is not None or line_test_sink is not None) and is_click:
                # Kept by reference and turned into the ``last`` frame after the
                # loop: nothing here is mutated later, and the voted set is a
                # snapshot, so it is this step's ranking whatever runs after it.
                last_ordinary = {
                    "t": t,
                    "test_ids": base_ids,
                    "test_scores": base_scores,
                    "test_labels": base_labels,
                    "pool_ranking": line_ranking,
                    "voted": frozenset(good_votes) | frozenset(bad_votes),
                    "pool_labels": pool_labels,
                    "vote_labels": {**dict.fromkeys(good_votes, True), **dict.fromkeys(bad_votes, False)},
                    # The labels line's arm (#4452): Find's fit on the withheld
                    # half, or with no class model this step the retrain's
                    # fallback cut, which keeps the same set at every beta.
                    "find_on_test": details.get("find_on_test") if labels_arm else None,
                    "fallback_threshold": (
                        float(threshold)
                        if labels_arm and details.get("find_on_test") is None and threshold is not None
                        else None
                    ),
                }
                if (
                    rank_frame_sink is not None
                    and rank_frame_steps
                    and any(t_before < s <= t for s in rank_frame_steps)
                ):
                    rank_frame_sink.append({**rank_ident, **_rank_frame("step", **last_ordinary)})
            if (
                precision_frame_sink is not None
                and precision_frame_steps
                and is_click
                and any(t_before < s <= t for s in precision_frame_steps)
            ):
                # The trainer builds its rows Goods first, then Bads, in vote order.
                vote_order = list(good_votes) + list(bad_votes)
                cal_votes = [
                    vote_order[i]
                    for fold in (details.get("fold_holdout_rows") or [])
                    for i in fold
                    if i < len(vote_order)
                ]
                precision_frame_sink.append(
                    _precision_frame(
                        t,
                        threshold,
                        base_scores,
                        base_labels,
                        sim_pooled_scores,
                        sim_pooled_ids,
                        {**{g: 1.0 for g in good_votes}, **{b: 0.0 for b in bad_votes}},
                        details.get("fold_orderings") or [],
                        sim_fold_haystacks,
                        cal_phase=[vote_phase.get(v, "") for v in cal_votes],
                        cal_vote=cal_votes,
                    )
                )
            # The final model's haystack under the #3308 population convention:
            # the voted items dropped, exactly as `_safe_threshold_for_step`
            # dropped them from the fold haystacks - so every fold-anchored
            # variant fit below stays paired with the shipped cut's population.
            # The same floor applies, via the same decision function, so a
            # variant grid can never sit on a different population than the
            # shipped cut it is measured against.
            _voted_step_ids = set(good_votes) | set(bad_votes)
            sim_fit_scores: list[float] | None = None
            if sim_pooled_scores is not None:
                _fit_arr, _ = apply_vote_exclusion(
                    sim_pooled_scores,
                    sim_pooled_ids,
                    _voted_step_ids,
                    min_remainder=exclusion_min_remainder,
                )
                sim_fit_scores = _fit_arr.tolist()
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
            # The #3329 goodness-of-fit frame.  Deliberately NOT nested in the
            # block above: that one needs `X_sim_image`, which only a region run
            # has, and the binary control arm is half of this study's design -
            # gating on it would have emitted nothing for exactly the arm the
            # max-pooling hypothesis is contrasted against.  The `image`
            # geometry rides along only where it exists.
            if (
                fit_quality_sink is not None
                and sim_pooled_scores is not None
                and (t <= 3 or t % max(1, fit_quality_stride) == 0)
            ):
                fq_scores: dict[str, Any] = {"pooled": sim_pooled_scores}
                fq_labels: dict[str, Any] = {
                    "pooled": np.array([pool_labels[cid] for cid in sim_pooled_ids], dtype=np.float64)
                }
                if X_sim_image is not None:
                    fq_image_ids = sorted(sim_ids)
                    fq_scores["image"] = np.asarray(step.predict(X_sim_image)).ravel().tolist()
                    fq_labels["image"] = np.array([pool_labels[cid] for cid in fq_image_ids], dtype=np.float64)
                fit_quality_sink.extend(_fit_quality_rows(base_row, safe_cut, fq_scores, fq_labels, threshold))
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
                        sim_fit_scores=sim_fit_scores,
                    )
                )
            for mr in metric_rows:
                rows.append({**base_row, **mr, **band_metrics, **timing_cols})
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
            rows.append({**base_row, **metrics, **band_metrics, **timing_cols})

    if rank_frame_sink is not None and last_ordinary is not None:
        rank_frame_sink.append({**rank_ident, **_rank_frame("last", **last_ordinary)})

    # --- The Test arm (#4523), once per run. ---
    #
    # Test mode's autopilot on the withheld half as the last ordinary click
    # left it: Find's labels line drawn there at the run's balance, uniform
    # picks within rank bands answered from the truth, to Done.  Test votes
    # never train, and the arm runs after the loop, so the trajectory above is
    # exactly what it is without the arm.  Seeded off the run's seed, so a
    # replay of the saved snapshot (`line_test_arm.row_from_snapshot`) can
    # reproduce this row.
    if line_test_sink is not None and last_ordinary is not None and beta is not None:
        from vtscore.eval.line_test_arm import line_test_row  # noqa: PLC0415
        from vtscore.training.thresholds import DEFAULT_BUDGETS  # noqa: PLC0415

        line_test_sink.append(
            {
                **rank_ident,
                **line_test_row(
                    last_ordinary["t"],
                    last_ordinary["test_scores"],
                    last_ordinary["test_labels"],
                    float(beta),
                    find_on_test=last_ordinary["find_on_test"],
                    fallback_threshold=last_ordinary["fallback_threshold"],
                    budgets=line_test_budgets if line_test_budgets is not None else DEFAULT_BUDGETS,
                    seed=seed,
                ),
            }
        )

    # --- The supervised skyline (issue #3322), once per run. ---
    #
    # Deliberately **after** the loop rather than before it: every fit here draws
    # from some RNG somewhere, and a skyline computed up front would have to be
    # proved not to perturb the trajectory it is meant to describe.  Computed
    # here it cannot, whatever the trainer does internally - the votes are
    # already cast.  `test_skyline_does_not_perturb_the_trajectory` pins that.
    if skyline_arms:
        skyline_rows = _skyline_arm_rows(
            skyline_arms,
            clips_dict,
            target_category,
            sim_ids,
            test_ids,
            inclusion,
            trainer=trainer,
            head=head,
            style_obj=style_obj,
            region_voting=region_voting,
            input_dim=input_dim,
            calibrate_count=calibrate_count,
            calibration_fraction=calibration_fraction,
            seed=seed,
            rank_frame_sink=rank_frame_sink,
            rank_ident=rank_ident,
            test_score_sink=test_score_sink,
        )
        _apply_skyline_decomposition(rows, skyline_rows)
        # `t=0` and `app_trained=0`: the skyline belongs to no step, so it is
        # given a step index no trajectory can occupy (the first *trainable*
        # step is t>=2) rather than being duplicated onto all of them.  The vote
        # counts are the full supervision it was handed, and `n_remainder=0`
        # says exactly what that means - nothing in the haystack was left
        # unlabelled.
        n_sky_good = sum(1 for cid in sim_ids if pool_labels[cid] == 1.0)
        skyline_ident = {
            "seed": seed,
            "dataset": dataset_name,
            "category": target_category,
            "strategy": strategy,
            "trainer": trainer,
            "head": head if trainer == APP_TRAINER else "",
            "style": style or "",
            "prevalence_arm": prevalence_arm,
            "realized_prevalence": realized_prevalence,
            "t": 0,
            "n_good": n_sky_good,
            "n_bad": len(sim_ids) - n_sky_good,
            "n_haystack": len(sim_ids),
            "n_remainder": 0,
            "phase": "",
            # A skyline belongs to no step, so no phase ran and no indicator
            # was ever read for it (#3560).  Blank / -1 is the "not measured"
            # spelling every other unphased row uses; a `red` here would say
            # the rules had looked and refused, which they never did.
            "smart": "",
            "stable": "",
            "span": "",
            "span_level": -1,
            "span_depth": -1,
            "span_target": -1,
            **{col: float("nan") for col in STOPPING_MARGIN_COLUMNS},
            "app_trained": 0,
            "startup_schedule": startup_schedule or "",
            "calibration_seed": calibration_seed,
            "acq_threshold": float("nan"),
            "acq_pool_percentile": float("nan"),
            "report_pool_percentile": float("nan"),
            # No line state: a skyline belongs to no step, so no line was cut on it.
            **_line_columns({}),
            # The skyline is fitted to every label; no Test gives it (#4643).
            "detector_tier": "",
        }
        rows.extend({**skyline_ident, **sr} for sr in skyline_rows)

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
    calibration_seed: Optional[int] = None,
    standalone_cut: str = "raw",
    beta: "Optional[float | str]" = None,
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
        inclusion: The Inclusion arm's line, in ``[-10, 10]``.  It draws the
            line only on the Inclusion arm (``beta="off"``), and a non-zero
            value under a balance is refused (#4361): a set balance wins, so it
            would only re-weight ``cost``.  The app has no such setting
            (#4269); 0 is the cut it draws with no balance.
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
        trainers: Which pipelines to run at each cell (see
            :func:`simulate_voting_iterations`).  ``None`` (default) runs
            ``["app"]``; pass e.g. ``["app", "svm_linear", "svm_rbf"]`` for the
            head-to-head comparison.  Recorded in the ``trainer`` column.
        prevalence_arms: Which prevalence arms to run per (dataset, category).
            ``None`` (default) runs ``[None]`` (the dataset's own prevalence only); pass
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
        calibration_seed: Seed of the Train/Calibrate fold splits (issue #3794);
            see :func:`simulate_voting_iterations`.  ``None`` (default) is the
            app's own pinned split, which is what every ordinary study wants.
            A calibration-noise arm sweeps it the other way round from the usual
            grid - one *seeds* entry, many calls, one *calibration_seed* each -
            so that the data is held while the split is redrawn; the frames
            concatenate and the ``calibration_seed`` column tells them apart.

    Returns:
        A :class:`~pandas.DataFrame` with the columns listed in
        :data:`VOTING_COLUMNS`.
    """
    import pandas as pd  # noqa: PLC0415

    # Refused here as well as per cell, so a misconfigured grid fails before its first cell runs.
    _check_inclusion_arm(inclusion, resolve_line_knobs(beta))
    strategy_list = strategies if strategies is not None else ["autopilot"]
    trainer_list = trainers if trainers is not None else [APP_TRAINER]
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
                                    calibration_seed=calibration_seed,
                                    standalone_cut=standalone_cut,
                                    beta=beta,
                                )
                                all_rows.extend(rows)

    return pd.DataFrame(all_rows, columns=pd.Index(list(VOTING_COLUMNS)))


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
    beta: "Optional[float | str]" = None,
) -> pd.DataFrame:
    """Convenience wrapper that loads datasets from pickle files.

    Args:
        dataset_paths: Mapping of dataset name to pickle file path.
        seeds: List of random seeds.
        categories: Optional category filter (see :func:`run_voting_iterations_eval`).
        inclusion: The Inclusion arm's line, in ``[-10, 10]``; a non-zero value
            needs ``beta="off"`` (see :func:`run_voting_iterations_eval`).
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
        beta: The balance the line is drawn at (see
            :func:`simulate_voting_iterations`): ``None`` is the app's default
            balance, ``"off"`` the Inclusion arm.

    Returns:
        A :class:`~pandas.DataFrame` identical to :func:`run_voting_iterations_eval`
        (columns: ``seed, dataset, category, strategy, t, n_good, n_bad, cost,
        fpr, fnr, elapsed_seconds``).
    """
    from vtscore.datasets.loader import load_dataset_from_pickle

    _check_inclusion_arm(inclusion, resolve_line_knobs(beta))
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
        beta=beta,
    )
