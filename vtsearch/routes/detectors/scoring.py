"""Detector scoring routes.

Implements the active-dataset scoring endpoints (find-label and auto-detect)
on top of the detector concept.  Detectors are loaded into ``DetectorContext``
instances on demand; weights live exclusively in RAM.

Migrated to ``flask_smorest`` so the routes are described in
``/api/openapi.json``.
"""

from __future__ import annotations

import logging
from typing import Any

from flask_smorest import Blueprint, abort

from vtscore.concurrency.progress import CancelledError, find_progress, update_find_progress
from vtscore.detectors.model_loading import resolve_or_train_detector
from vtsearch.hooks import state_sync_exempt
from vtsearch.routes._context import require_dataset_header, require_detector_header
from vtsearch.routes._progress import find_idle, find_idle_on_crash
from vtsearch.schemas.detectors import (
    AutoDetectRequestSchema,
    AutoDetectResponseSchema,
    AutoFindBrowsePrepResponseSchema,
    AutoFindRunResponseSchema,
    FindCorrectionsToDetectorResponseSchema,
    FindEvidenceCoverageResponseSchema,
    FindLabelRequestSchema,
    FindLabelResponseSchema,
    FindStatsResponseSchema,
)
from vtsearch.state import snapshot_medias

logger = logging.getLogger(__name__)

detector_scoring_bp = Blueprint(
    "detector_scoring",
    __name__,
    description="Run a detector against the active dataset (find-label) "
    "or run every AutoFind detector at once (auto-detect).",
)


def _detector_type(det_data: dict | None) -> str:
    """The locked embedder type of a detector JSON (legacy-migrated)."""
    from vtsearch.autofind import detector_type  # noqa: PLC0415

    return detector_type(det_data)


def _dataset_supplies_detector_type(det_data: dict | None, snap: dict) -> bool:
    """Whether the active snap binds an embedder of the detector's locked type.

    See :func:`vtsearch.autofind.dataset_supplies_detector_type`.
    """
    from vtsearch.autofind import dataset_supplies_detector_type  # noqa: PLC0415

    return dataset_supplies_detector_type(det_data, snap)


def _type_incompatible_message(det_data: dict | None) -> str:
    """Human-facing 409 message for a type-incompatible detector/dataset pair."""
    from vtscore.embedding.binding import EMBEDDER_TYPE_LABELS  # noqa: PLC0415

    det_type = _detector_type(det_data)
    label = EMBEDDER_TYPE_LABELS.get(det_type, det_type or "compatible")
    return (
        f"This detector scores in a {label} embedder space, which the active "
        "dataset doesn't provide. Load a dataset that binds a matching embedder type."
    )


def _abort_if_find_cancelled() -> None:
    """Abort 409 (resetting find progress) if ``/api/find/cancel`` was called.

    Polled between the expensive stages of the synchronous scoring routes so
    a cancel takes effect at the next stage boundary.
    """
    if find_progress.is_cancelled:
        find_idle()
        abort(409, message="Test cancelled")


def _keep_line_ranking(results: list[dict], threshold: float) -> float:
    """Give a Find pass that reused a cached head the ranking its line is read against (#4272, #4273).

    Training stores the ranking the line is drawn over (``line_ranking``).  A
    head reused as it was (:func:`~vtscore.detectors.model_loading.cached_head_is_current`)
    brings none when the last one was dropped - by a dataset switch, or by
    ending a Find session - and then the balance's state had nothing to count
    and a spot check had nothing to draw from.  So a pass with no ranking
    builds one from the scores it just computed, with the votes a person had
    cast before it marked voted, as training marks the labelset's items.  A
    ranking training has just stored is left alone.  Returns the threshold to
    use: *threshold* unchanged when there was a ranking already, no balance,
    or no class model behind the head.

    Under the balance (#4452) nothing is counted on this corpus: the line is
    the labels' class model with the prevalence re-estimated on these scores,
    which is what a cold Find over the same corpus computes; the ranking only
    reports how many the line keeps here.
    """
    from vtscore.state.core import (  # noqa: PLC0415
        get_active_detector_context,
        human_voted_ids,
    )
    from vtscore.training.thresholds import LineRanking  # noqa: PLC0415
    from vtsearch.state import line_knobs  # noqa: PLC0415

    det_ctx = get_active_detector_context()
    if det_ctx.line_ranking is not None or not results:
        return threshold
    voted = human_voted_ids(det_ctx)
    det_ctx.line_ranking = LineRanking.from_scores([r["id"] for r in results], [r["score"] for r in results], voted)
    det_ctx.gate_passed = None  # a structural re-rank after this pass sets it afresh (#4505)
    beta = line_knobs()["beta"]
    if beta is not None and det_ctx.labels_line is not None:
        # The labels' line (#4452): the class model the labels gave the head,
        # with the prevalence re-estimated on this corpus as a cold Find would
        # (the same EM over its scores) - never a count drawn on it.  The
        # ranking stays for the line's state: how many it keeps here.
        det_ctx.labels_line = det_ctx.labels_line.on_corpus(
            [r["score"] for r in results],
            [r["id"] for r in results],
            {cid: True for cid in det_ctx.good_votes if cid in voted}
            | {cid: False for cid in det_ctx.bad_votes if cid in voted},
        )
        return float(det_ctx.labels_line.threshold(beta))
    # No balance, or no class model behind this head (too few votes, one
    # class): the threshold the retrain stored stands - no count on this
    # corpus (#4452).
    return threshold


@detector_scoring_bp.route("/api/find-label", methods=["POST"])
@detector_scoring_bp.arguments(FindLabelRequestSchema)
@detector_scoring_bp.response(200, FindLabelResponseSchema)
@detector_scoring_bp.alt_response(400, description="No medias loaded, or the detector has no labels for scoring.")
@detector_scoring_bp.alt_response(404, description="Detector not found.")
@detector_scoring_bp.alt_response(
    409,
    description="Dataset lacks the detector's embedder type, or Find was cancelled via /api/find/cancel.",
)
@require_dataset_header
@require_detector_header
def find_label(body: dict):
    """Score all loaded medias with a detector and apply labels based on threshold.

    Resolves the detector from the registry, scores every loaded media, and
    applies Good/Bad labels for ALL elements based on the threshold.  Returns
    the sort results so the frontend can display the stripe and scroll order.
    """
    from vtscore.detectors.registry import get_detector as reg_get_detector
    from vtscore.detectors.store import _detector_path, _read_detector
    from vtsearch.state import (
        apply_labels_bulk_with_click_time,
        set_find_initial_labels,
        set_find_scores,
    )

    # Total high-level steps: resolve(1) + optional train(2) + score(3) + apply(4)
    _FIND_LABEL_STEPS = 4
    #: Timing task name; its step names and shipped weights live in
    #: :data:`vtscore.timing.tasks.TASKS`.
    _TRAIN_SCORE_TASK = "train_and_score"

    detector_id = body["detector_id"]

    # Clear a leftover cancel flag from a previously-cancelled run so
    # the new operation doesn't trip on it immediately.
    find_progress.reset_cancel()

    update_find_progress(
        "running",
        "Resolving detector…",
        current=0,
        total=0,
        step=1,
        total_steps=_FIND_LABEL_STEPS,
    )

    d = reg_get_detector(detector_id)
    if d is None:
        find_idle()
        abort(404, message=f"Detector '{detector_id}' not found")

    snap = snapshot_medias()
    if not snap:
        find_idle()
        abort(400, message="No medias loaded")

    media_type = d.get("media_type", "") or next(iter(snap.values())).get("media_type", "image")

    # Every exit below — success, abort, and unexpected crash alike — parks the
    # tracker at "idle". The crash case is the guard's job (see
    # :func:`find_idle_on_crash`).
    from vtscore import timing  # noqa: PLC0415

    find_progress.set_step_weights(timing.step_weights(_TRAIN_SCORE_TASK, media_type=media_type))
    with find_idle_on_crash():
        det_path = _detector_path(d["name"])
        det_data = _read_detector(det_path)

        # Type gate: the active dataset must bind an embedder of the detector's
        # locked type before we spend a train/score in the wrong space.
        if not _dataset_supplies_detector_type(det_data, snap):
            find_idle()
            abort(409, message=_type_incompatible_message(det_data))

        _abort_if_find_cancelled()
        mlp, threshold, diagnostic = resolve_or_train_detector(
            detector_id,
            det_data,
            media_type,
            snap,
            progress_step=2,
            progress_total_steps=_FIND_LABEL_STEPS,
        )
        if mlp is None:
            find_idle()
            if diagnostic is not None and not diagnostic["has_good"] and not diagnostic["failed_resolution"]:
                # Every label resolved and none is a Good: there is nothing to
                # sort toward (#4643).  One Good is enough - below the label
                # quota the Goods' centroid answers - so name that, not a
                # resolution failure.
                abort(
                    400,
                    message=f"Detector '{d['name']}' needs at least one Good label before it can be tested.",
                    resolution_diagnostic=diagnostic,
                )
            if diagnostic is not None:
                error_msg = (
                    f"Detector '{d['name']}' could not be trained: "
                    f"{diagnostic['total_labels']} training labels found, "
                    f"{diagnostic['dataset_matched']} matched the current dataset, "
                    f"{diagnostic['needed_resolution']} needed origin resolution, "
                    f"{diagnostic['resolved_from_origin']} resolved successfully, "
                    f"{diagnostic['failed_resolution']} failed to resolve. "
                    f"Has good={diagnostic['has_good']}, has bad={diagnostic['has_bad']}."
                )
                if diagnostic.get("sample_failures"):
                    first = diagnostic["sample_failures"][0]
                    error_msg += (
                        f" First failure: importer={first['origin'].get('importer', '?') if first['origin'] else 'None'}, "
                        f"origin_name={first['origin_name']!r}, "
                        f"params={first['origin'].get('params', {}) if first['origin'] else '{}'}"
                    )
                if diagnostic.get("hint"):
                    error_msg += f" Hint: {diagnostic['hint']}"
                failed = diagnostic["failed_resolution"]
                total = diagnostic["total_labels"]
                mt = diagnostic.get("media_type", "items")
                mt_plural = mt + "s" if mt and not mt.endswith("s") else mt
                abort(
                    400,
                    message=error_msg,
                    resolution_diagnostic=diagnostic,
                    warning=(f"{failed} of your {total} {mt_plural} could not be resolved from their original files."),
                )
            abort(400, message=f"Detector '{d['name']}' has no labels for scoring")

        n_total = len(snap)
        update_find_progress(
            "running",
            f"Scoring {n_total} items…",
            current=0,
            total=n_total,
            step=3,
            total_steps=_FIND_LABEL_STEPS,
        )

        # Region-aware scoring: for patch datasets (DINOv2/v3, EUPE) this
        # max-pools over each media's region vectors and surfaces the winning
        # region's box as ``best_region`` so the gallery thumbnails and focus-view
        # Highlight overlay can outline it - matching the learned-sort path.  Plain
        # single-vector datasets take the cached embedding-matrix path inside and
        # gain no ``best_region`` field.
        from vtscore.detectors.training import score_media_with_model
        from vtscore.embedding.binding import keying_embedder_for_type

        # Score in the same space the MLP was trained in: the concrete embedder of
        # the detector's locked type this dataset supplies, else the dataset score
        # precedence (keying_embedder_for_snap) - matching resolve_or_train_detector's
        # cold path and the learned-sort cache-space marker.
        _abort_if_find_cancelled()
        score_emb = keying_embedder_for_type(_detector_type(det_data), snap)
        results = score_media_with_model(mlp, snap, embedder_name=score_emb or None)
        update_find_progress(
            "running",
            f"Scoring {n_total} items…",
            current=n_total,
            total=n_total,
            step=3,
            total_steps=_FIND_LABEL_STEPS,
        )

        # Stage-2 structural re-rank for a saved structural (SIFT/VLAD) detector:
        # geometrically verify the VLAD shortlist against the detector's RegionYes
        # templates and re-rank by the match-statistic verification classifier.
        # This is the Find counterpart to the re-rank already wired into the
        # vote-driven (``train_and_score``) and learned-sort (``labelset_train_and_score``)
        # paths, so a pre-trained structural detector verifies in Find too instead of
        # stopping at the coarse VLAD retrieval.  A no-op for every non-structural
        # detector (gated on the active snapshot carrying ``local_features``).  See
        # docs/plans/structural-embedder.md.
        from vtscore.datasets.labelset import LabelSet
        from vtscore.detectors.labelset_training import maybe_labelset_structural_rerank
        from vtscore.state.core import get_active_detector_context

        labelset = LabelSet.from_dict((det_data or {}).get("labelset") or {})
        # The balance's set is drawn on the Stage-1 scores.  A structural re-rank
        # then replaces both the ranking and the cut with its classifier's
        # boundary, as it does on every other path: it has no balance line.
        threshold = _keep_line_ranking(results, threshold)
        results, threshold = maybe_labelset_structural_rerank(
            get_active_detector_context(), labelset, results, threshold, snap
        )
        # Store the final (post-rerank) cutoff on the context so server-side reads of
        # the Find cutoff — the work-queue / boundary-walk endpoints, balance
        # re-thresholding — agree with the labels this pass just applied. A no-op for
        # the non-structural path (threshold unchanged), authoritative for the
        # structural one.
        get_active_detector_context().threshold = threshold

        _abort_if_find_cancelled()
        update_find_progress(
            "running",
            f"Applying labels to {n_total} items…",
            current=0,
            total=n_total,
            step=4,
            total_steps=_FIND_LABEL_STEPS,
        )
        label_pairs = []
        for entry in results:
            label_pairs.append((entry["id"], "good" if entry["score"] >= threshold else "bad"))
        # Items the human already verified keep their vote: a re-score (the normal
        # fold-corrections -> retrain -> re-score loop) must not silently invert a
        # recorded human decision while still counting it as verified (issue #2928).
        # Everything else adopts this pass's call.
        apply_labels_bulk_with_click_time(
            label_pairs, replace_all=True, record_achievement=False, preserve_verified=True
        )

        # The detector's own call for *every* item, verified ones included: this is
        # the evaluation baseline Stats crosses the adopted labels against, so a
        # verified item the retrained detector now disagrees with reads as a
        # correction rather than vanishing.
        set_find_initial_labels({mid: lbl for mid, lbl in label_pairs})
        # Freeze the single-pass scores so the line (the balance's)
        # re-thresholds without re-scoring, and the Stats precision curve can
        # read them.
        set_find_scores({entry["id"]: entry["score"] for entry in results})

        from vtscore.detectors.registry import set_find_mode

        set_find_mode(True)

        det_ctx = get_active_detector_context()
        # A fresh scoring pass IS the current evaluation, so any "stale" flag left by
        # a prior corrections-to-detector fold no longer applies.
        det_ctx.find_eval_stale = False
        # A test of the line was over the previous pass's scores (#4524).
        det_ctx.line_test = None
        # The counts the client shows are the labels this pass *adopted*, which is
        # the threshold split everywhere except the verified items that held their
        # human vote.  ``replace_all`` left exactly this label set behind, so the
        # vote dicts are the count.
        good_count = len(det_ctx.good_votes)
        bad_count = len(det_ctx.bad_votes)

        from vtscore.labels.sync import sync_to_labelset_source

        sync_to_labelset_source()

        update_find_progress(
            "idle",
            "Done",
            current=n_total,
            total=n_total,
            step=_FIND_LABEL_STEPS,
            total_steps=_FIND_LABEL_STEPS,
        )

        from vtsearch.achievements import record_find

        record_find(n_total)

        # Find is not yet windowed: its "just sit and vote" boundary walk and the
        # Browse / To Dataset / Export bulk actions still read the full client-side
        # ranking, so find-label returns the whole ``results`` list. The server-side
        # replacements those flows will switch to already exist
        # (``/api/find/queue-ids``, ``/api/find/boundary-next``); wiring the Find
        # frontend onto them + windowing this response is the remaining slice (see
        # docs/plans/scalability.md S3/S17/S19).
        from vtscore.state.core import detector_balance_state  # noqa: PLC0415
        from vtsearch.state import get_beta  # noqa: PLC0415

        from vtscore.detectors.label_quota import served_quota  # noqa: PLC0415

        return {
            "ok": True,
            "results": results,
            "threshold": round(threshold, 4),
            # What the balance says about the line (#4272, #4413).
            "balance": detector_balance_state(det_ctx, get_beta()),
            "good_count": good_count,
            "bad_count": bad_count,
            "detector_name": d.get("name", ""),
            # Which detector the labels gave, and what is still owed (#4643).
            "label_quota": served_quota(mlp, labelset),
        }


@detector_scoring_bp.route("/api/find/stats", methods=["GET"])
@detector_scoring_bp.response(200, FindStatsResponseSchema)
def find_stats():
    """Detector-evaluation stats over the **adopted** Find label set.

    Like Export / Browse / To-Dataset, this treats unverified items as if
    verified at their current cutoff: truth is the full ``good_votes`` /
    ``bad_votes`` set (human votes flood-filled with the detector's call on
    everything untouched).  This can give false confidence in the detector -
    that's the price of not verifying every item - but it reports the real
    counts.  Crosses each item's adopted label against the detector's original
    call (``find_initial_labels``) for a 2x2 confusion, and reports what the
    balance says about the current line (``balance``): the beta it was cut at
    and what a spot check found.

    The **Kept rate** (``verified_precision``) is the exception to "treat
    unverified as verified": it counts only the items the user checked, since
    counting every unchecked match as right would read 99% whatever the checks
    found.  The **precision curve** charts how right the returned set is against
    how much is returned - verified precision on the checked items - at
    log-spaced return counts (see :mod:`vtsearch.routes.detectors._find_precision`).
    It carries no model-based estimate (#4360).
    Pure read; no new state.
    """
    from vtscore.state.core import detector_balance_state, get_active_detector_context
    from vtsearch.routes.detectors._find_precision import curve_counts, verified_precision_at
    from vtsearch.state import get_beta

    det_ctx = get_active_detector_context()
    good = det_ctx.good_votes
    bad = det_ctx.bad_votes
    initial = det_ctx.find_initial_labels
    scores = det_ctx.find_scores

    # Confusion of adopted label (truth, over ALL items) vs. the detector's
    # original call.  Unverified items adopted the detector's call, so they
    # land in the confirmed cells; corrections come from human overrides.
    confirmed_good = sum(1 for cid in good if initial.get(cid) != "bad")
    rescued_fn = sum(1 for cid in good if initial.get(cid) == "bad")
    confirmed_bad = sum(1 for cid in bad if initial.get(cid) != "good")
    culled_fp = sum(1 for cid in bad if initial.get(cid) == "good")

    total_good = len(good)
    total_bad = len(bad)
    total_items = total_good + total_bad
    agreements = confirmed_good + confirmed_bad
    corrections = culled_fp + rescued_fn
    agreement_rate = agreements / total_items if total_items else 0.0
    # Kept rate over the checked items the detector called Good.  A checked
    # item's adopted label is the user's vote (verified items hold it).
    verified = det_ctx.verified_ids
    verified_called_good = [cid for cid in verified if initial.get(cid) == "good"]
    verified_kept = sum(1 for cid in verified_called_good if cid in good)
    verified_precision = verified_kept / len(verified_called_good) if verified_called_good else None

    # Precision against the number returned.  Ranked as the Find list ranks
    # (score descending, id to break ties), so point k is the top k the user sees.
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    ranked_ids = [cid for cid, _s in ranked]
    n_returned = sum(1 for _cid, s in ranked if s >= det_ctx.threshold)
    counts = curve_counts(len(ranked), extra=n_returned or None)
    checked_good = {cid: cid in good for cid in verified if cid in scores}
    verified_points = verified_precision_at(ranked_ids, checked_good, counts)
    precision_curve = [
        {
            "n_returned": k,
            "threshold": round(ranked[k - 1][1], 4),
            "checked": n_checked,
            "checked_good": n_good,
            "verified_precision": None if v_prec is None else round(v_prec, 4),
        }
        for k, (n_checked, n_good, v_prec) in zip(counts, verified_points, strict=True)
    ]

    return {
        "total_good": total_good,
        "total_bad": total_bad,
        "verified_count": len(det_ctx.verified_ids),
        "confirmed_good": confirmed_good,
        "confirmed_bad": confirmed_bad,
        "culled_false_pos": culled_fp,
        "rescued_false_neg": rescued_fn,
        "agreements": agreements,
        "corrections": corrections,
        "agreement_rate": round(agreement_rate, 4),
        "verified_precision": None if verified_precision is None else round(verified_precision, 4),
        "verified_called_good": len(verified_called_good),
        "verified_kept_good": verified_kept,
        "threshold": round(det_ctx.threshold, 4),
        # The balance the line was cut at, and what its check found (#4246, #4413).
        "balance": detector_balance_state(det_ctx, get_beta()),
        "n_scored": len(ranked),
        "n_returned": n_returned,
        "stale": getattr(det_ctx, "find_eval_stale", False),
        "precision_curve": precision_curve,
    }


def _empty_evidence_coverage() -> dict:
    """The ``available: False`` report the evidence-coverage endpoint returns
    when there is nothing to measure (no scored Find run, no resolvable
    labelset, or no cached label embeddings)."""
    from vtscore.detectors.evidence_coverage import DEFAULT_ALPHA, DEFAULT_K

    return {
        "available": False,
        "n_items": 0,
        "n_pos_labels": 0,
        "n_neg_labels": 0,
        "k": DEFAULT_K,
        "alpha": DEFAULT_ALPHA,
        "frac_unsupported": 0.0,
        "expected_unsupported": DEFAULT_ALPHA,
        "z_score": 0.0,
        "median_support": 1.0,
        "frac_low_trust": 0.0,
        "median_trust": 1.0,
        "unsupported": False,
    }


@detector_scoring_bp.route("/api/find/evidence-coverage", methods=["GET"])
@detector_scoring_bp.response(200, FindEvidenceCoverageResponseSchema)
def find_evidence_coverage():
    """Evidence-coverage report for the active detector on the active dataset.

    The cross-user complement to the atlas domain-shift report
    (``GET /api/datasets/registry/<id>/domain-shift``): that one needs the
    *training* dataset loaded with a built atlas, which is absent in a true
    detector handoff (userB has userA's detector JSON, not userA's haystack).
    This asks only what the detector itself carries — its labelset, re-embedded
    in memory against the active dataset's embedder at load — so it fires even
    when the training haystack was never handed over.

    For every scored item it measures how far the detector's *predicted* class
    sits from the labeled evidence of that class (a conformal support p-value
    ``D``, calibrated against the labelset's own leave-one-out distances) and
    the trust-score ratio ``TS``, then summarises the share of the dataset the
    detector is calling without supervision behind it.  Returns
    ``available: False`` (rather than 4xx) when there is no scored Find run or
    no resolvable labelset, so the Stats modal can simply hide the section.
    Pure read; no new state.  See docs/plans/coverage-atlas.md §6.1 (phase v0).
    """
    import numpy as np

    from vtscore.detectors.evidence_coverage import evidence_coverage_report
    from vtscore.detectors.labelset_training import build_xy_from_labelset
    from vtscore.detectors.learned_sort import resolve_active_labelset
    from vtscore.embedding.media_vectors import media_embedding
    from vtscore.state.core import get_active_detector_context

    det_ctx = get_active_detector_context()
    scores = det_ctx.find_scores
    if not scores:
        return _empty_evidence_coverage()

    labelset, _ = resolve_active_labelset(det_ctx)
    if labelset is None:
        return _empty_evidence_coverage()

    # (X, y) re-derived from the in-memory label embeddings the detector load
    # already populated from its saved labelset — cross-user by construction,
    # nothing persisted.  Good rows are y == 1, Bad (incl. patch-flood
    # negatives) are y == 0.
    x_list, y_list, _groups, _score_rows = build_xy_from_labelset(det_ctx, labelset)
    if not x_list:
        return _empty_evidence_coverage()
    x = np.asarray(x_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    pos = x[y == 1.0]
    neg = x[y == 0.0]

    # Query side: every scored item's embedding in the *same* space the labelset
    # was resolved in (``det_ctx.embedder``), with its current predicted class
    # from the frozen score vs the live threshold.
    embedder = det_ctx.embedder or None
    threshold = det_ctx.threshold
    snap = snapshot_medias()
    q_vecs: list[Any] = []
    q_pred: list[bool] = []
    for cid, score in scores.items():
        media = snap.get(cid)
        if media is None:
            continue
        emb = media_embedding(media, embedder_name=embedder)
        if emb is None:
            continue
        q_vecs.append(np.asarray(emb, dtype=np.float32))
        q_pred.append(score >= threshold)
    if not q_vecs:
        return _empty_evidence_coverage()

    report = evidence_coverage_report(pos, neg, np.stack(q_vecs), np.asarray(q_pred, dtype=bool))
    return {"available": True, **report}


@detector_scoring_bp.route("/api/find/corrections-to-detector", methods=["POST"])
@detector_scoring_bp.response(200, FindCorrectionsToDetectorResponseSchema)
@detector_scoring_bp.alt_response(400, description="No Find run to take corrections from.")
@detector_scoring_bp.alt_response(404, description="No active detector to update.")
@detector_scoring_bp.alt_response(409, description="Detector vote state is not aligned with the active dataset.")
@require_dataset_header
@require_detector_header
def find_corrections_to_detector():
    """Fold the Find corrections into the active detector's labelset for *future*
    scoring, leaving the current Find session frozen.

    A *correction* is an item whose adopted Find label differs from the
    detector's original call (``find_initial_labels``): a rescued
    false-negative (now good, the detector said bad) or a culled false-positive
    (now bad, the detector said good).  This matches the ``corrections`` export
    filter and the Stats "corrections" count.

    The correction items are written into the detector's on-disk labelset
    (superseding any prior entry for the same source media) and the registry's
    training counters are refreshed.  The cached MLP is invalidated so the next
    scoring pass retrains from the merged labelset.

    The current Find session is deliberately *not* re-scored or reset: its
    scores, queue, votes, and verification stay pinned to the detector version
    that produced them, so the displayed evaluation (and ``GET /api/find/stats``)
    keeps showing the previous detector's results.  That evaluation is now out of
    date relative to the retrained detector, which ``find_eval_stale`` records so
    the Stats note can say so; the retrained detector takes effect the next time
    the dataset is scored.
    """
    from vtscore.datasets.labelset import LabelSet, element_identity_keys
    from vtscore.detectors.dataset_sync import _detector_file_mtime, validated_vote_snapshot
    from vtscore.detectors.input_spec import extract_input_spec_from_medias
    from vtscore.detectors.labelset_ops import label_sync_write_lock
    from vtscore.detectors.registry import get_detector as reg_get_detector
    from vtscore.detectors.registry import update_detector
    from vtscore.detectors.store import _detector_path, _read_detector, _write_detector
    from vtscore.state.core import _state_lock, get_active_detector_context

    det_ctx = get_active_detector_context()
    detector_id = det_ctx.detector_id or ""
    reg = reg_get_detector(detector_id) if detector_id else None
    if reg is None or not reg.get("name"):
        abort(404, message="No active detector to update")
    name = reg["name"]

    # read→merge→write races the loaded-detector label sync (and the other
    # detector-JSON writers), so it runs under the same lock — acquired
    # before ``_state_lock``, matching label_sync's ordering.
    with label_sync_write_lock:
        path = _detector_path(name)
        data = _read_detector(path)
        if data is None:
            abort(404, message=f"Detector '{name}' not found")

        # Atomic (medias, good_votes, bad_votes, region boxes) snapshot keyed in the
        # active dataset's cid space, so the votes we compose with the medias can't
        # straddle a concurrent dataset switch on this detector.
        snap = validated_vote_snapshot()
        if not snap.safe:
            abort(409, message="Cannot add corrections: detector vote state is not aligned with the active dataset")

        initial = det_ctx.find_initial_labels
        if not initial:
            abort(400, message="No Test run to take corrections from. Score the dataset first.")

        existing_ls = LabelSet.from_dict(data.get("labelset") or {})

        # A correction's adopted label differs from the detector's original call.
        corr_good = {cid: None for cid in snap.good_votes if initial.get(cid) == "bad"}
        corr_bad = {cid: None for cid in snap.bad_votes if initial.get(cid) == "good"}
        num_corrections = len(corr_good) + len(corr_bad)

        if num_corrections == 0:
            return {
                "ok": True,
                "name": name,
                "corrections_added": 0,
                "num_labels": len(existing_ls),
            }

        corrections_ls = LabelSet.from_clips_and_votes(
            snap.medias,
            corr_good,
            corr_bad,
            expand_dupes=False,
            vote_region_boxes=snap.vote_region_boxes,
            vote_provenance=snap.vote_provenance,
        )

        # Merge: a correction supersedes any prior entry for the same source media
        # (so a culled false-positive flips its old "good" entry to "bad").
        # "Same source media" is the union of identities the element could carry
        # - origin *and* md5 - because a prior entry may name the file by a
        # different origin than the active dataset does (issue #3174); matching
        # on the preferred key alone would leave the stale entry in place and
        # store the media twice, with contradicting labels.
        corr_keys: set = set()
        for el in corrections_ls.elements:
            corr_keys.update(element_identity_keys(el))
        merged_elements = [
            el for el in existing_ls.elements if not any(k in corr_keys for k in element_identity_keys(el))
        ]
        merged_elements.extend(corrections_ls.elements)
        merged = LabelSet(merged_elements)

        data["labelset"] = merged.to_dict()

        # Keep the stored input_spec in sync with the active dataset's clipper, as
        # ``save_detector_labels`` does.
        captured_spec = extract_input_spec_from_medias(snap.medias)
        if captured_spec is not None:
            data["input_spec"] = captured_spec
        elif "input_spec" in data:
            data.pop("input_spec", None)
        _write_detector(path, data)

        # Freeze the current Find session.  Writing the file bumped its mtime, which
        # would make the before_request rehydrate wipe the in-memory votes /
        # find_scores / find_initial_labels and re-derive them from the new labelset.
        # Re-point the cached labelset + mtime at the file we just wrote so that
        # rehydrate is a no-op, invalidate the cached MLP so the next scoring pass
        # retrains from the merged labelset, and flag the displayed evaluation stale.
        new_mtime = _detector_file_mtime(path)
        media_type = data.get("media_type", "") or ""
        with _state_lock:
            det_ctx.cached_labelset = merged
            det_ctx.cached_labelset_mtime = new_mtime
            det_ctx.cached_labelset_media_type = media_type or det_ctx.cached_labelset_media_type
            det_ctx.labelset_good_count = sum(1 for el in merged.elements if el.label == "good")
            det_ctx.labelset_bad_count = sum(1 for el in merged.elements if el.label == "bad")
            det_ctx.model = None
            det_ctx.find_eval_stale = True

        import time as _time

        update_detector(reg["id"], num_training=len(merged), last_trained_at=_time.time())

        return {
            "ok": True,
            "name": name,
            "corrections_added": num_corrections,
            "num_labels": len(merged),
        }


@detector_scoring_bp.route("/api/auto-detect", methods=["POST"])
@detector_scoring_bp.arguments(AutoDetectRequestSchema)
@detector_scoring_bp.response(200, AutoDetectResponseSchema)
@detector_scoring_bp.alt_response(
    400,
    description="No medias loaded, or no AutoFind detectors match the active media type.",
)
@detector_scoring_bp.alt_response(404, description="Named detector is not on the caller's AutoFind list.")
@detector_scoring_bp.alt_response(409, description="Find was cancelled via /api/find/cancel.")
def auto_detect(body: dict):
    """Score the active dataset with every detector on the caller's AutoFind list.

    Iterates :func:`~vtsearch.settings.get_autofind_detectors` and trains each
    one's MLP on demand from its on-disk labelset.  Returns one result column
    per detector. Pass ``detector_name`` to run a single AutoFind detector.

    The synchronous, scripted sibling of the Dashboard's background AutoFind
    (``POST /api/datasets/registry/<dataset_id>/autofind``); both run through
    :mod:`vtsearch.autofind`.  This one reports on the shared Find
    tracker and is cancelled by ``/api/find/cancel``.
    """
    from vtsearch.autofind import (  # noqa: PLC0415
        AutoFindUnavailable,
        plan_autofind,
        run_autofind_export,
        score_autofind,
    )

    snap = snapshot_medias()

    # Clear a leftover cancel flag from a previously-cancelled run.
    find_progress.reset_cancel()

    try:
        plan = plan_autofind(snap, detector_name=body.get("detector_name") or "")
    except AutoFindUnavailable as exc:
        abort(exc.status, message=exc.message)

    # A cold detector's train writes "running" to the shared tracker from inside
    # the workers (``resolve_or_train_detector``), so this route owns parking it
    # again — on the way out of a crash via the guard, and on the two ordinary
    # exits below.
    with find_idle_on_crash():
        try:
            response = score_autofind(plan, snap)
        except CancelledError:
            find_idle()
            abort(409, message="Find cancelled")
        find_idle()

    auto_export = run_autofind_export(response)
    if auto_export is not None:
        response["auto_export"] = auto_export
    return response


@detector_scoring_bp.route("/api/autofind/runs/<run_id>", methods=["GET"])
@detector_scoring_bp.response(200, AutoFindRunResponseSchema)
@detector_scoring_bp.alt_response(
    404,
    description="No such run for the caller: unknown, another user's, or aged out of the kept window.",
)
def get_autofind_run(run_id: str):
    """Results of a finished background AutoFind, for the user who started it.

    ``run_id`` is the ``task_id`` of the AutoFind task (``autofind.run_id`` on
    its ``loading-tasks`` row).  Runs are kept in memory only, and only the
    most recent few, so an old or pre-restart run answers 404 like one that
    never existed.
    """
    from vtsearch.auth import get_current_user  # noqa: PLC0415
    from vtsearch.autofind import get_autofind_run as _get_run  # noqa: PLC0415

    record = _get_run(run_id, get_current_user())
    if record is None:
        abort(404, message="AutoFind results not found")
    return record


@detector_scoring_bp.route("/api/autofind/runs/<run_id>/browse-prep", methods=["POST"])
@state_sync_exempt
@detector_scoring_bp.response(200, AutoFindBrowsePrepResponseSchema)
@detector_scoring_bp.alt_response(
    404,
    description="No such run for the caller: unknown, another user's, or aged out of the kept window.",
)
def prep_autofind_run_browse(run_id: str):
    """Start laying out a finished run's Good results for Browse, if the server is idle.

    The Find Results dialog asks this while it is open (#4683), so its Browse
    button finds the map built, or part-way there, rather than starting a fit
    when pressed.  The map is the one that button builds: the run's Good
    results on the run's own dataset, which need not be the caller's active one
    (no ``X-Dataset-Id`` is read).  Nothing is started while other work is in
    flight: ``busy`` says to ask again later.

    ``@state_sync_exempt``: the dialog repeats the request while the server is
    busy, which is exactly when ``_state_lock`` is most likely held, and the
    handler reads no request proxy - it resolves the run's dataset by id.
    """
    from vtsearch.auth import get_current_user  # noqa: PLC0415
    from vtsearch.autofind import prep_run_browse  # noqa: PLC0415

    answer = prep_run_browse(run_id, get_current_user())
    if answer is None:
        abort(404, message="AutoFind results not found")
    return answer
