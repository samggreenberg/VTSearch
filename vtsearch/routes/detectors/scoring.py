"""Detector scoring routes.

Implements the active-dataset scoring endpoints (find-label and auto-detect)
on top of the detector concept.  Detectors are loaded into ``DetectorContext``
instances on demand; weights live exclusively in RAM.

Migrated to ``flask_smorest`` so the routes are described in
``/api/openapi.json``. See ``docs/plans/openapi-schema.md``.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from flask_smorest import Blueprint, abort

from vtscore.concurrency.memory_budget import cap_workers_by_memory
from vtscore.concurrency.progress import CancelledError, find_progress, update_find_progress
from vtscore.detectors.model_loading import resolve_or_train_detector
from vtscore.embedding.media_vectors import EMBEDDINGS_KEY
from vtsearch.routes._shared import require_dataset_header, require_detector_header
from vtsearch.schemas.detectors import (
    AutoDetectRequestSchema,
    AutoDetectResponseSchema,
    FindCorrectionsToDetectorResponseSchema,
    FindEvidenceCoverageResponseSchema,
    FindLabelRequestSchema,
    FindLabelResponseSchema,
    FindStatsResponseSchema,
)
from vtscore.utils.scores import sigmoid_to_finite_scores
from vtsearch.state import snapshot_medias

logger = logging.getLogger(__name__)

detector_scoring_bp = Blueprint(
    "detector_scoring",
    __name__,
    description="Run a detector against the active dataset (find-label) "
    "or run every Auto-Find detector at once (auto-detect).",
)

# Keys excluded from API responses (large binary/vector data).  ``embeddings``
# is the v3 dict-keyed vector store (``vtscore/embedding/media_vectors.py``);
# ``embedding`` is its dropped legacy singular form, kept here so a media dict
# rehydrated from an old pickle can't leak one either.
_HEAVYWEIGHT_KEYS = (
    EMBEDDINGS_KEY,
    "embedding",
    "media_bytes",
    "media_string",
    "thumbnail_bytes",
)


def _detector_type(det_data: dict | None) -> str:
    """The locked embedder type of a detector JSON (legacy-migrated)."""
    from vtscore.detectors.embedder_type import detector_embedder_type_from_data  # noqa: PLC0415

    return detector_embedder_type_from_data(det_data or {})


def _dataset_bound_embedders(snap: dict) -> list[str]:
    """The embedder names the active snap binds (keys of the first media's vectors)."""
    from vtscore.embedding.media_vectors import media_embedder_names  # noqa: PLC0415

    return media_embedder_names(next(iter(snap.values()), {})) if snap else []


def _dataset_supplies_detector_type(det_data: dict | None, snap: dict) -> bool:
    """Whether the active snap binds an embedder of the detector's locked type.

    A detector scores its labels in the concrete embedder of its locked type the
    active dataset supplies; when the dataset binds no embedder of that type the
    labels would re-embed in a foreign space (garbage scores), so the pair is
    incompatible.  A legacy/typeless detector is always compatible (resolved at
    first train via the score precedence).
    """
    from vtscore.embedding.binding import detector_dataset_compatible  # noqa: PLC0415

    return detector_dataset_compatible(_detector_type(det_data), _dataset_bound_embedders(snap))


def _compatible_detectors(detectors_to_run: list, snap: dict) -> list:
    """Drop Auto-Find detectors whose locked type the active dataset can't supply.

    Scoring an incompatible detector's labels in a foreign space would be
    garbage, so they are skipped; a legacy/typeless detector stays (resolved via
    the score precedence).  Each item is a ``(name, det_data, entry)`` triple.
    Aborts 400 if the gate leaves nothing to run.
    """
    kept = [trip for trip in detectors_to_run if _dataset_supplies_detector_type(trip[1], snap)]
    if not kept:
        abort(
            400,
            message="No Auto-Find detectors are compatible with the active dataset's embedder types.",
        )
    return kept


def _type_incompatible_message(det_data: dict | None) -> str:
    """Human-facing 409 message for a type-incompatible detector/dataset pair."""
    from vtscore.embedding.binding import EMBEDDER_TYPE_LABELS  # noqa: PLC0415

    det_type = _detector_type(det_data)
    label = EMBEDDER_TYPE_LABELS.get(det_type, det_type or "compatible")
    return (
        f"This detector scores in a {label} embedder space, which the active "
        "dataset doesn't provide. Load a dataset that binds a matching embedder type."
    )


def _media_info_for_response(media: dict) -> dict:
    """Return a copy of *media* without heavyweight fields."""
    return {k: v for k, v in media.items() if k not in _HEAVYWEIGHT_KEYS}


def _abort_if_find_cancelled() -> None:
    """Abort 409 (resetting find progress) if ``/api/find/cancel`` was called.

    Polled between the expensive stages of the synchronous scoring routes so
    a cancel takes effect at the next stage boundary.
    """
    if find_progress.is_cancelled:
        update_find_progress("idle", "", step=None, total_steps=None)
        abort(409, message="Find cancelled")


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
    #: Timing-profile task name; its step names and shipped fallback weights
    #: live in :data:`vtscore.timing.tasks.TASKS`. An admin
    #: ``VTSEARCH_TIMING_PROFILE`` replaces them with the seconds this
    #: deployment's GPU actually spends training and scoring.
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
        update_find_progress("idle", "")
        abort(404, message=f"Detector '{detector_id}' not found")

    snap = snapshot_medias()
    if not snap:
        update_find_progress("idle", "")
        abort(400, message="No medias loaded")

    media_type = d.get("media_type", "") or next(iter(snap.values())).get("media_type", "image")

    # Both the train and score steps scale with the active dataset, so the bar
    # can be paced against its real size instead of a fixed guess. Every exit
    # below — success and abort alike — parks the tracker at "idle", which is
    # what closes the recorder.
    from vtscore import timing  # noqa: PLC0415

    score_embedder = next(iter(snap.values())).get("embedder", "")
    find_progress.set_step_weights(
        timing.step_weights(_TRAIN_SCORE_TASK, media_type=media_type, embedder=score_embedder, n=len(snap))
    )
    recorder = timing.record_task(
        find_progress,
        _TRAIN_SCORE_TASK,
        media_type=media_type,
        embedder=score_embedder,
        auto_finish=True,
    )
    recorder.start()
    recorder.set_scale(n=len(snap))
    det_path = _detector_path(d["name"])
    det_data = _read_detector(det_path)

    # Type gate: the active dataset must bind an embedder of the detector's
    # locked type before we spend a train/score in the wrong space.
    if not _dataset_supplies_detector_type(det_data, snap):
        update_find_progress("idle", "")
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
        update_find_progress("idle", "")
        if diagnostic is not None:
            error_msg = (
                f"Detector '{d['name']}' could not be trained: "
                f"{diagnostic['total_labels']} training labels found, "
                f"{diagnostic['md5_matched']} matched current dataset by MD5, "
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
    from types import SimpleNamespace

    from vtscore.detectors.training import score_media_with_model
    from vtscore.embedding.binding import keying_embedder_for_snap

    # Score in the same space the MLP was trained in: the concrete embedder of
    # the detector's locked type this dataset supplies, else the dataset score
    # precedence (keying_embedder_for_snap) - matching resolve_or_train_detector's
    # cold path and the learned-sort cache-space marker.
    _abort_if_find_cancelled()
    score_emb = keying_embedder_for_snap(SimpleNamespace(embedder_type=_detector_type(det_data)), snap)
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
    results, threshold = maybe_labelset_structural_rerank(
        get_active_detector_context(), labelset, results, threshold, snap
    )
    # Store the final (post-rerank) cutoff on the context so server-side reads of
    # the Find cutoff — the work-queue / boundary-walk endpoints, Inclusion
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
    good_count = 0
    bad_count = 0
    for entry in results:
        if entry["score"] >= threshold:
            label_pairs.append((entry["id"], "good"))
            good_count += 1
        else:
            label_pairs.append((entry["id"], "bad"))
            bad_count += 1
    apply_labels_bulk_with_click_time(label_pairs, replace_all=True, record_achievement=False)

    set_find_initial_labels({mid: lbl for mid, lbl in label_pairs})
    # Freeze the single-pass scores so the cutoff (Inclusion) re-thresholds
    # without re-scoring, and the Stats FP/FN sweep can read them.
    set_find_scores({entry["id"]: entry["score"] for entry in results})

    from vtscore.detectors.registry import set_find_mode

    set_find_mode(True)

    # A fresh scoring pass IS the current evaluation, so any "stale" flag left by
    # a prior corrections-to-detector fold no longer applies.
    from vtscore.state.core import get_active_detector_context

    get_active_detector_context().find_eval_stale = False

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
    return {
        "ok": True,
        "results": results,
        "threshold": round(threshold, 4),
        "good_count": good_count,
        "bad_count": bad_count,
        "detector_name": d.get("name", ""),
    }


def _resolve_autofind_names(body: dict, media_type: str) -> list[str]:
    """Return the Auto-Find detector names to consider for this request.

    Aborts with 404 when ``detector_name`` is given but not in the Auto-Find
    list, and with 400 when no Auto-Find detectors are configured at all.
    """
    from vtsearch.settings import get_autofind_detectors  # noqa: PLC0415

    autofind_names = get_autofind_detectors()
    single_name = body.get("detector_name") or ""
    if single_name:
        if single_name not in autofind_names:
            abort(404, message=f"Detector '{single_name}' not flagged for Auto-Find")
        return [single_name]
    if not autofind_names:
        abort(400, message=f"No Auto-Find detectors found for media type: {media_type}")
    return autofind_names


def _collect_detectors_for_media_type(
    autofind_names: list[str], media_type: str
) -> tuple[list[tuple[str, dict, dict | None]], list[str]]:
    """Load detector data + registry entry for each Auto-Find name matching *media_type*.

    Returns ``(detectors, missing)``: *missing* holds names whose detector
    file no longer exists on disk (a stale Auto-Find reference). Names whose
    media type simply doesn't match the active dataset are skipped without
    being reported - those are legitimately inapplicable, not broken.
    """
    from vtscore.detectors.registry import find_by_name, list_detectors  # noqa: PLC0415
    from vtscore.detectors.store import _detector_path, _read_detector  # noqa: PLC0415

    detectors: list[tuple[str, dict, dict | None]] = []
    missing: list[str] = []
    for name in autofind_names:
        det_data = _read_detector(_detector_path(name))
        if det_data is None:
            missing.append(name)
            continue
        if det_data.get("media_type", "") != media_type:
            continue
        reg_entry = find_by_name(name)
        if reg_entry is None:
            # Fallback: also accept registry entries whose name matches.
            for entry in list_detectors():
                if entry.get("name") == name:
                    reg_entry = entry
                    break
        detectors.append((name, det_data, reg_entry))
    return detectors, missing


def _score_detector_for_auto_detect(
    name: str,
    det_data: dict,
    reg_entry: dict | None,
    media_type: str,
    snap: dict,
    all_ids: list[int],
    X_all: Any,
) -> tuple[str, dict] | None:
    """Train (or reuse) one detector and score every media in *snap*."""
    import torch  # noqa: PLC0415

    # Poll BEFORE the catch-all try: CancelledError subclasses Exception, and
    # a cancel must propagate to the route (via future.result()), not be
    # swallowed as a failed detector.
    find_progress.check_cancelled()

    try:
        detector_id = reg_entry["id"] if reg_entry else name
        mlp, threshold, _diag = resolve_or_train_detector(
            detector_id,
            det_data,
            media_type,
            snap,
            progress_step=1,
            progress_total_steps=1,
        )
        if mlp is None:
            return None

        with torch.no_grad():
            X_in = X_all.to(next(mlp.parameters()).device)
            scores = sigmoid_to_finite_scores(mlp(X_in))

        positive_hits = []
        negative_hits = []
        for cid, score in zip(all_ids, scores, strict=True):
            clip_info = _media_info_for_response(snap[cid])
            clip_info["score"] = round(score, 4)
            if score >= threshold:
                positive_hits.append(clip_info)
            else:
                negative_hits.append(clip_info)

        positive_hits.sort(key=lambda x: x["score"], reverse=True)
        negative_hits.sort(key=lambda x: x["score"], reverse=True)

        return name, {
            "detector_name": name,
            "threshold": round(threshold, 4),
            "total_hits": len(positive_hits),
            "hits": positive_hits,
            "negative_hits": negative_hits,
        }
    except Exception:
        logger.exception("Auto-detect failed for detector %s", name)
        return None


def _collect_auto_detect_results(futures: list, results: dict[str, dict]) -> None:
    """Drain the auto-detect worker futures into *results*.

    A worker that saw the cancel flag raises :class:`CancelledError`; drop the
    queued workers (already-running ones finish their current detector, the
    queued ones bail at their entry poll), reset the tracker, and abort 409.
    """
    try:
        for future in futures:
            outcome = future.result()
            if outcome is not None:
                name, result = outcome
                results[name] = result
    except CancelledError:
        for f in futures:
            f.cancel()
        update_find_progress("idle", "", step=None, total_steps=None)
        abort(409, message="Find cancelled")


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
    call (``find_initial_labels``) for a 2x2 confusion, and sweeps the
    calibrated threshold across inclusion -10..10 (re-cutting the cached
    estimator behind the current cutoff) for false-positive / false-negative
    counts at every stop.
    Pure read; no new state.  See docs/plans/find-verification-workflow.md.
    """
    from vtscore.state.core import get_active_detector_context
    from vtscore.training.thresholds import INCLUSION_MAX, INCLUSION_MIN, threshold_from_fold_orderings
    from vtsearch.state import get_inclusion

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
    detector_positives = confirmed_good + culled_fp
    precision = confirmed_good / detector_positives if detector_positives else 0.0

    # Sweep FP/FN over ALL adopted items at every inclusion's threshold.
    # Adopted-bad above the line are false positives; adopted-good below it are
    # false negatives.  Thresholds come from the cached estimator behind the
    # detector's current cutoff - the fold-anchored population fit when safe
    # thresholds produced one, else the cached fold orderings - so the chart
    # plots the line the user would actually get at each stop, and the sweep
    # stays cheap (no refit, no re-scoring).
    anchored_cut = det_ctx.anchored_cut_cache
    cache = det_ctx.calibration_cache
    orderings = cache[1].orderings if (cache is not None and cache[1].fallback is None) else []
    good_scores = [scores[c] for c in good if c in scores]
    bad_scores = [scores[c] for c in bad if c in scores]
    sweep = []
    for incl in range(INCLUSION_MIN, INCLUSION_MAX + 1):
        if anchored_cut is not None:
            t_i = anchored_cut.threshold_at(incl)
        else:
            t_i = threshold_from_fold_orderings(orderings, incl) if orderings else det_ctx.threshold
        sweep.append(
            {
                "inclusion": incl,
                "threshold": round(t_i, 4),
                "false_pos": sum(1 for s in bad_scores if s >= t_i),
                "false_neg": sum(1 for s in good_scores if s < t_i),
            }
        )

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
        "precision": round(precision, 4),
        "inclusion": get_inclusion(),
        "threshold": round(det_ctx.threshold, 4),
        "stale": getattr(det_ctx, "find_eval_stale", False),
        "sweep": sweep,
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
    from vtscore.datasets.labelset import LabelSet, element_key
    from vtscore.detectors.dataset_sync import _detector_file_mtime, validated_vote_snapshot
    from vtscore.detectors.input_spec import extract_input_spec_from_medias
    from vtscore.detectors.label_sync import _label_sync_write_lock
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
    with _label_sync_write_lock:
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
            abort(400, message="No Find run to take corrections from. Score the dataset first.")

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
        )

        # Merge: a correction supersedes any prior entry for the same source media
        # (so a culled false-positive flips its old "good" entry to "bad").
        corr_keys = {element_key(el) for el in corrections_ls.elements}
        corr_keys.discard(None)
        merged_elements = [el for el in existing_ls.elements if element_key(el) not in corr_keys]
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
    description="No medias loaded, or no Auto-Find detectors match the active media type.",
)
@detector_scoring_bp.alt_response(404, description="Named detector is not flagged for Auto-Find.")
@detector_scoring_bp.alt_response(409, description="Find was cancelled via /api/find/cancel.")
def auto_detect(body: dict):
    """Score the active dataset with every detector flagged for Auto-Find.

    Iterates :func:`~vtsearch.settings.get_autofind_detectors` and trains each
    one's MLP on demand from its on-disk labelset.  Returns one result column
    per detector. Pass ``detector_name`` to run a single Auto-Find detector.
    """
    import torch  # noqa: PLC0415

    snap = snapshot_medias()
    if not snap:
        abort(400, message="No medias loaded")

    media_type = next(iter(snap.values())).get("media_type", "audio")

    # Clear a leftover cancel flag from a previously-cancelled run.
    find_progress.reset_cancel()

    autofind_names = _resolve_autofind_names(body, media_type)
    detectors_to_run, missing_detectors = _collect_detectors_for_media_type(autofind_names, media_type)
    if not detectors_to_run:
        if missing_detectors:
            abort(
                400,
                message=(
                    f"No Auto-Find detectors found for media type: {media_type}. "
                    f"Missing detector file(s) for: {', '.join(missing_detectors)}"
                ),
            )
        abort(400, message=f"No Auto-Find detectors found for media type: {media_type}")

    # Type gate: drop Auto-Find detectors whose locked embedder type the active
    # dataset can't supply (see _compatible_detectors).
    detectors_to_run = _compatible_detectors(detectors_to_run, snap)

    from vtscore.embedding.matrix import get_embedding_matrix_for_snap  # noqa: PLC0415
    from vtscore.state.core import get_active_context  # noqa: PLC0415

    # Each detector scores in the concrete embedder of its locked type, so build
    # one matrix per distinct embedder the Auto-Find detectors call for
    # (collapsing to a single shared matrix on the common single-embedder
    # dataset, where every type resolves to the same name).  A legacy/typeless
    # detector falls back to the dataset score precedence, matching
    # resolve_or_train_detector's cold-train space.
    from types import SimpleNamespace  # noqa: PLC0415

    from vtscore.embedding.binding import keying_embedder_for_snap  # noqa: PLC0415

    default_score = get_active_context().routed_embedder("score")
    det_embedders: dict[str, str | None] = {}
    for dname, ddata, _entry in detectors_to_run:
        keyed = keying_embedder_for_snap(SimpleNamespace(embedder_type=_detector_type(ddata)), snap)
        det_embedders[dname] = keyed or default_score
    matrices: dict[str | None, tuple[list[int], Any]] = {}
    for emb in set(det_embedders.values()):
        ids, embs = get_embedding_matrix_for_snap(snap, emb)
        matrices[emb] = (ids, torch.from_numpy(embs))

    # Size the worker cap off the biggest matrix actually built.  Indexing by
    # ``default_score`` would KeyError whenever no detector keys to it - the
    # normal case on a multi-embedder dataset where every Auto-Find detector is
    # type-locked to a semantic embedder while the score precedence picks the
    # patch/structural one.  ``matrices`` is never empty here because
    # ``_compatible_detectors`` aborts 400 when the type gate leaves nothing.
    cap_rows = max(len(ids) for ids, _X in matrices.values())
    cap_dim = max(int(X.shape[1]) if X.ndim > 1 else 0 for _ids, X in matrices.values())
    worker_cap = cap_workers_by_memory(
        cap_rows,
        cap_dim,
        max_workers=min(len(detectors_to_run), 8),
    )
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=worker_cap) as pool:
        futures = [
            pool.submit(
                _score_detector_for_auto_detect,
                name,
                data,
                entry,
                media_type,
                snap,
                *matrices[det_embedders[name]],
            )
            for name, data, entry in detectors_to_run
        ]
        _collect_auto_detect_results(futures, results)

    if results:
        from vtsearch.achievements import record_find  # noqa: PLC0415

        # Each detector scored the medias in its own embedder's matrix, so count
        # them per detector rather than multiplying one matrix's row count out.
        record_find(sum(len(matrices[det_embedders[name]][0]) for name in results))

    response = {
        "media_type": media_type,
        "detectors_run": len(results),
        "results": results,
        "missing_detectors": missing_detectors,
    }
    auto_export = _run_autofind_export(response)
    if auto_export is not None:
        response["auto_export"] = auto_export
    return response


def _run_autofind_export(response: dict) -> dict | None:
    """Run the configured Auto-Find results exporter on *response*.

    Returns ``None`` when no exporter is configured (the common case), or a
    status dict ``{exporter, success, message?, error?}`` otherwise. Export
    failures are reported in the status block rather than raised: the scored
    results are valuable on their own, so a misconfigured exporter must not
    sink the whole request. See ``docs/plans/auto-find-settings-tab.md``.
    """
    from vtsearch.settings import (  # noqa: PLC0415
        get_autofind_exporter,
        get_autofind_exporter_field_values,
    )

    exporter_name = get_autofind_exporter()
    if not exporter_name:
        return None

    from vtscore.exporters import get_exporter  # noqa: PLC0415

    exporter = get_exporter(exporter_name)
    if exporter is None:
        return {"exporter": exporter_name, "success": False, "error": f"Unknown exporter '{exporter_name}'"}

    field_values = dict(get_autofind_exporter_field_values().get(exporter_name, {}))
    try:
        from vtscore.plugins.normalize import normalize_field_values  # noqa: PLC0415

        normalize_field_values(exporter, field_values)
        outcome = exporter.export(response, field_values) or {}
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller, never raised
        logger.exception("Auto-Find export via %s failed", exporter_name)
        return {"exporter": exporter_name, "success": False, "error": str(exc)}

    status = {"exporter": exporter_name, "success": True, "message": outcome.get("message", "Export complete.")}
    for key, value in outcome.items():
        if key != "message":
            status[key] = value

    # An `open_url` reaches the browser, so it gets the same scheme allowlist
    # `POST /api/exporters/export` applies - a plugin must not be able to push
    # a `javascript:` URL to the frontend from here either. Unlike that route
    # this one drops the bad URL instead of failing: the export already
    # happened, and the scored results must survive a cosmetic field.
    if status.get("open_url") is not None:
        from vtscore.security.url_validation import validate_browser_url  # noqa: PLC0415

        try:
            status["open_url"] = validate_browser_url(str(status["open_url"]))
        except ValueError as exc:
            logger.error("Auto-Find exporter %r returned an unusable open_url: %s", exporter_name, exc)
            del status["open_url"]
            status["message"] = f"{status['message']} (the exporter returned an unusable URL, so nothing will open)"
    return status
