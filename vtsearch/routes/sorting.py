"""Blueprint for sorting and voting routes.

Migrated to ``flask_smorest`` so the routes are described in
``/api/openapi.json``. See ``docs/plans/openapi-schema.md``.

Schema-level validation failures (missing required ``text`` / ``job_id`` /
``examples`` / ``inclusion`` / ``safe_thresholds``; type-mismatched
``inclusion`` / ``safe_thresholds`` values) surface as 422 with the
standard ``errors`` envelope. Handler-level rejects (empty / whitespace
``text``, no votes, no medias, bad files in the multipart routes, etc.)
keep their HTTP codes (400 / 404 / 500) with the standard ``message``
envelope. The two multipart routes (``/api/example-sort``,
``/api/label-file-sort``) omit ``arguments`` and declare their error
responses via ``alt_response``; same pattern as ``add-to-pile`` and
``server-media-files/upload``.
"""

import json
import logging
import threading
from pathlib import Path

from flask import request
from flask_smorest import Blueprint, abort

from vtsearch.routes._shared import (
    format_exception_detail,
    get_embedder_for_medias,
    require_dataset_header,
    require_detector_header,
)

from vtscore.config import DATA_DIR
import vtscore.security.path_validation as _paths
from vtscore.detectors.labeling_progress import inject_live_model
from vtscore.detectors.training import train_and_score
from vtscore.embedding import embed_text_query
from vtsearch.schemas.sorting import (
    DiversityTreeNextResponseSchema,
    InclusionRequestSchema,
    InclusionResponseSchema,
    LabelFileSortResponseSchema,
    LearnedSortCancelResponseSchema,
    LearnedSortRequestSchema,
    LearnedSortResponseSchema,
    LearnedSortResultQuerySchema,
    OkResponseSchema,
    SafeThresholdsRequestSchema,
    SafeThresholdsResponseSchema,
    SeedFromExamplesRequestSchema,
    SeedFromExamplesResponseSchema,
    SortRequestSchema,
    SortResponseSchema,
    TextsortSuggestionRequestSchema,
    TextsortSuggestionsResponseSchema,
    VotesResponseSchema,
)
from vtscore.training.thresholds import calculate_gmm_threshold
from vtsearch.state import (
    add_textsort_suggestion,
    bad_votes,
    diversity_tree_next_sample,
    get_calibrate_count,
    get_calibration_fraction,
    get_diversity_tree,
    get_inclusion,
    get_learned_scores,
    get_safe_thresholds,
    get_textsort_suggestions,
    get_vote_click_times,
    good_votes,
    set_inclusion,
    set_safe_thresholds,
    snapshot_medias,
    update_learned_scores,
    vote_region_boxes,
)
from vtscore.concurrency.progress import update_sort_progress

sorting_bp = Blueprint(
    "sorting",
    __name__,
    description="Text / example / learned sort, votes, inclusion, safe-thresholds, diversity tree.",
)


def _cosine_sort(query_vec):
    """Sort all loaded medias by cosine similarity to *query_vec*.

    Returns ``(results, threshold)`` where *results* is a list of
    ``{"id": …, "similarity": …}`` dicts sorted descending, and
    *threshold* is the GMM-based boundary (rounded to 4 decimals).

    For datasets embedded with a patch-aware embedder (DINOv2, DINOv3,
    EUPE), each result also carries a ``best_region`` field containing the
    bounding box of the region that scored highest, in normalised
    image coordinates ``[x0, y0, x1, y1]``.  Single-vector embedders
    take a fast vectorised numpy path with no per-result box.

    Both paths live in :mod:`vtscore.training.region_similarity`.
    """
    from vtscore.training.region_similarity import cosine_sort_with_boxes  # noqa: PLC0415

    snap = snapshot_medias()
    results, sims_list = cosine_sort_with_boxes(snap, query_vec)
    threshold = calculate_gmm_threshold(sims_list)
    return results, round(threshold, 4)


_embedder_load_lock = threading.Lock()


def _get_embedder_for_loaded_data():
    """Return the appropriate embedder for the currently loaded dataset."""
    return get_embedder_for_medias(snapshot_medias())


def _load_embedder_with_progress(media_type, total_steps):
    """Load the embedding model, forwarding progress to the sort progress bar.

    If the model is already loaded this is a no-op.  A lock serialises
    concurrent callers so that only one request touches ``_on_progress``
    at a time, preventing the save/restore from trampling another
    request's callback.
    """
    emb = _get_embedder_for_loaded_data()
    if emb is None:
        return

    with _embedder_load_lock:
        if getattr(emb, "_model", None) is not None:
            return

        update_sort_progress("sorting", "Loading embedder…", 0, total_steps)
        original_cb = emb._on_progress
        emb._on_progress = lambda status, msg, cur, tot: update_sort_progress("sorting", msg, cur, tot)
        try:
            emb.load_models()
        finally:
            emb._on_progress = original_cb


@sorting_bp.route("/api/sort", methods=["POST"])
@sorting_bp.arguments(SortRequestSchema)
@sorting_bp.response(200, SortResponseSchema)
@sorting_bp.alt_response(400, description="Empty/whitespace text, no medias, or embedder doesn't support text.")
@sorting_bp.alt_response(500, description="Text sort failed (embedder error or unexpected exception).")
def sort_clips(body: dict):
    """Return medias sorted by cosine similarity to a text query."""
    text = body.get("text", "").strip()
    if not text:
        update_sort_progress("idle")
        abort(400, message="text is required")

    snap = snapshot_medias()
    if not snap:
        update_sort_progress("idle")
        abort(400, message="No medias loaded")

    first = next(iter(snap.values()))
    media_type = first.get("media_type", "audio")
    embedder_name = first.get("embedder", "")
    total_steps = 3  # load embedder, embed query, compute similarities

    # Reject the request up-front when the active embedder is vision-only
    # (e.g. DINOv3, Perception Encoder). Without this short-circuit we'd
    # waste time loading the model into RAM just to discover it can't embed
    # text; the frontend wouldn't get a clean ``supports_text=False``
    # signal to hide its text-search UI.
    if embedder_name:
        try:
            from vtscore.media import get_embedder  # noqa: PLC0415

            _active_emb = get_embedder(embedder_name)
        except KeyError:
            _active_emb = None
        if _active_emb is not None and not _active_emb.supports_text:
            update_sort_progress("idle")
            # flask-smorest's error handler only flows ``message`` and
            # ``errors`` from ``abort()`` kwargs into the response body,
            # so the original ``supports_text=False`` flag would be
            # silently dropped. The frontend already reads the same flag
            # from each embedder's ``EmbedderInfo`` (see
            # ``left-panel.component.ts: updateTextSortAvailable``), so
            # we don't need to ship it on the error response too.
            abort(
                400,
                message=(
                    f"Embedder '{_active_emb.name}' does not support text queries. "
                    "Use learned sort or load a saved sort instead."
                ),
            )

    try:
        _load_embedder_with_progress(media_type, total_steps)
        update_sort_progress("sorting", "Embedding text query…", 1, total_steps)

        from vtsearch import settings

        enrich = settings.get_enrich_descriptions()
        text_vec = embed_text_query(text, media_type, enrich=enrich, embedder_name=embedder_name)
        if text_vec is None:
            update_sort_progress("idle")
            abort(500, message=f"Could not embed text for media type {media_type}")

        update_sort_progress("sorting", "Computing similarities…", 2, total_steps)
        results, threshold = _cosine_sort(text_vec)
        update_sort_progress("idle")
        return {"results": results, "threshold": threshold}
    except Exception as exc:
        from werkzeug.exceptions import HTTPException

        if isinstance(exc, HTTPException):
            # ``abort()`` above raises HTTPException; let flask-smorest's
            # handler render its envelope unchanged instead of wrapping it
            # in a 500.
            raise
        logging.getLogger(__name__).exception("text sort failed")
        update_sort_progress("idle")
        abort(500, message=f"Text sort failed: {format_exception_detail(exc)}")


def _learned_sort_done_payload(job) -> dict:
    """Build the JSON body returned when a learned-sort job is finished."""
    result = job.result or {}
    return {
        "job_id": job.job_id,
        "status": "done",
        "results": result.get("results", []),
        "threshold": result.get("threshold", 0.0),
    }


def _resolve_active_labelset(det_ctx):
    """Resolve the labelset for the active detector → (labelset, media_type)."""
    from vtscore.datasets.labelset import LabelSet
    from vtscore.detectors.registry import get_detector
    from vtscore.detectors.store import _detector_path, _read_detector
    from vtscore.state.core import _empty_detector_context

    if det_ctx is _empty_detector_context or not det_ctx.detector_id:
        return None, ""
    if det_ctx.cached_labelset is not None:
        return det_ctx.cached_labelset, det_ctx.cached_labelset_media_type
    entry = get_detector(det_ctx.detector_id)
    if not entry or not entry.get("name"):
        return None, ""
    det_data = _read_detector(_detector_path(entry["name"]))
    if not det_data:
        return None, ""
    return LabelSet.from_dict(det_data.get("labelset") or {}), det_data.get("media_type", "") or ""


def _validate_learned_sort_inputs(labelset, good, bad) -> None:
    if labelset is not None:
        good_count = sum(1 for el in labelset.elements if el.label == "good")
        bad_count = sum(1 for el in labelset.elements if el.label == "bad")
        if good_count == 0 or bad_count == 0:
            abort(400, message="need at least one good and one bad vote")
        return
    if not good or not bad:
        abort(400, message="need at least one good and one bad vote")


def _resolve_labelset_local_state(labelset, snap):
    """Resolve labelset elements to local cids using the dataset snapshot.

    Returns (local_good, local_bad, training_medias, has_cross_dataset).
    All are None when ``labelset`` is None (the non-labelset path).
    """
    if labelset is None:
        return None, None, None, False

    from vtsearch.state import build_media_lookup, resolve_media_ids

    origin_lookup, md5_lookup, name_lookup = build_media_lookup(snap)
    local_good: set[int] = set()
    local_bad: set[int] = set()
    training_medias: dict[int, dict] = {}
    has_cross_dataset = False
    for el in labelset.elements:
        if el.label not in ("good", "bad"):
            continue
        cids = resolve_media_ids(el.to_dict(), origin_lookup, md5_lookup, name_lookup)
        if not cids:
            has_cross_dataset = True
            continue
        target = local_good if el.label == "good" else local_bad
        for cid in cids:
            target.add(cid)
            if cid in snap:
                training_medias[cid] = snap[cid]
    return local_good, local_bad, training_medias, has_cross_dataset


def _model_matches_local_votes(labelset, has_cross_dataset, local_good, local_bad, good, bad) -> bool:
    """Decide whether the trained model maps cleanly onto current-dataset votes.

    The progress cache is keyed on local cids, so we only inject when the
    training set is fully representable there; otherwise we'd return a
    cross-dataset model on a local-only replay.
    """
    if labelset is None:
        return True
    return not has_cross_dataset and local_good == set(good) and local_bad == set(bad)


def _update_det_ctx_with_trained_model(det_ctx, model, threshold, labelset, training_medias, snap, good, bad) -> None:
    det_ctx.model = model
    det_ctx.threshold = threshold
    if labelset is not None:
        det_ctx.training_medias = training_medias or {}
    else:
        training = {}
        for cid in list(good) + list(bad):
            if cid in snap:
                training[cid] = snap[cid]
        det_ctx.training_medias = training
    if snap:
        first = next(iter(snap.values()), {})
        det_ctx.embedder = first.get("embedder", "")
        det_ctx.media_type = first.get("media_type", "")


def _build_learned_sort_signature(
    *,
    det_ctx,
    ds_ctx,
    snap,
    labelset,
    good,
    bad,
    region_boxes_snapshot,
    inclusion_value,
    safe_thresholds_value,
    calibrate_count_value,
    calibration_fraction_value,
):
    if labelset is not None:
        labels_sig = tuple(sorted((el.label, _stable_element_id_for_sig(el)) for el in labelset.elements))
    else:
        labels_sig = (
            ("good", tuple(sorted(good))),
            ("bad", tuple(sorted(bad))),
            ("regions", tuple(sorted(region_boxes_snapshot.items()))),
        )
    return (
        det_ctx.detector_id,
        ds_ctx.dataset_id,
        tuple(sorted(snap.keys())),
        labels_sig,
        inclusion_value,
        safe_thresholds_value,
        calibrate_count_value,
        calibration_fraction_value,
    )


@sorting_bp.route("/api/learned-sort", methods=["POST"])
@sorting_bp.arguments(LearnedSortRequestSchema)
@sorting_bp.response(200, LearnedSortResponseSchema)
@sorting_bp.alt_response(400, description="No good/bad votes available for training.")
@sorting_bp.alt_response(500, description="Background learned-sort job failed (only when ``wait=true``).")
def learned_sort(body: dict):
    """Kick off (or short-circuit) a learned-sort training job.

    Training is GIL-bound and ran inline used to stall every other request
    served by the small ``gthread`` pool (votes polls, thumbnails, and even
    media bytes).  The endpoint now hands the work off to a background daemon
    thread and returns immediately with a ``job_id``; clients poll
    :func:`learned_sort_result` until ``status == "done"``.

    A small signature cache short-circuits the no-op case: when the votes,
    detector, inclusion and thresholding settings are unchanged from the
    most recent successful run, the previous result is returned directly.

    Tests can pass ``{"wait": true}`` in the body to block until the job
    completes and receive the result inline.  The frontend leaves it false.
    """
    from vtscore.detectors.labelset_training import labelset_train_and_score
    from vtscore.concurrency.async_jobs import learned_sort_jobs
    from vtscore.state.core import (
        _empty_detector_context,
        get_active_context,
        get_active_detector_context,
        thread_dataset_context,
        thread_detector_context,
    )

    wait = body["wait"]

    snap = snapshot_medias()

    det_ctx = get_active_detector_context()
    ds_ctx = get_active_context()
    labelset, det_media_type = _resolve_active_labelset(det_ctx)

    _validate_learned_sort_inputs(labelset, good_votes, bad_votes)

    inclusion_value = get_inclusion()
    safe_thresholds_value = get_safe_thresholds()
    calibrate_count_value = get_calibrate_count()
    calibration_fraction_value = get_calibration_fraction()
    region_boxes_snapshot = dict(vote_region_boxes)

    signature = _build_learned_sort_signature(
        det_ctx=det_ctx,
        ds_ctx=ds_ctx,
        snap=snap,
        labelset=labelset,
        good=good_votes,
        bad=bad_votes,
        region_boxes_snapshot=region_boxes_snapshot,
        inclusion_value=inclusion_value,
        safe_thresholds_value=safe_thresholds_value,
        calibrate_count_value=calibrate_count_value,
        calibration_fraction_value=calibration_fraction_value,
    )

    cached = learned_sort_jobs.cached_for(signature)
    if cached is not None:
        return _learned_sort_done_payload(cached)

    # _run is a closure over the resolved inputs above so we can pass it
    # straight to the job manager without threading a 10-arg dataclass
    # through the abstraction.  Its complexity is dominated by sequential
    # state updates rather than nested branching; splitting it further
    # would just smear cohesive logic across helpers.
    def _run(job):  # noqa: C901
        with thread_dataset_context(ds_ctx), thread_detector_context(det_ctx):
            if labelset is not None:
                results, threshold, model = labelset_train_and_score(
                    det_ctx,
                    labelset,
                    media_type=det_media_type,
                    clips_dict=snap,
                    inclusion_value=inclusion_value,
                    safe_thresholds=safe_thresholds_value,
                    calibrate_count=calibrate_count_value,
                    calibration_fraction=calibration_fraction_value,
                )
            else:
                results, threshold, model = train_and_score(
                    snap,
                    dict(good_votes),
                    dict(bad_votes),
                    inclusion_value,
                    safe_thresholds=safe_thresholds_value,
                    calibrate_count=calibrate_count_value,
                    calibration_fraction=calibration_fraction_value,
                    vote_region_boxes=region_boxes_snapshot,
                    det_ctx=det_ctx,
                )

            update_learned_scores({r["id"]: r["score"] for r in results})

            local_good, local_bad, training_medias, has_cross_dataset = _resolve_labelset_local_state(labelset, snap)

            if model is not None and _model_matches_local_votes(
                labelset, has_cross_dataset, local_good, local_bad, good_votes, bad_votes
            ):
                inject_live_model(good_votes, bad_votes, model, threshold)

            if det_ctx is not _empty_detector_context and model is not None:
                _update_det_ctx_with_trained_model(
                    det_ctx, model, threshold, labelset, training_medias, snap, good_votes, bad_votes
                )

            job.result = {"results": results, "threshold": round(threshold, 4)}

    job = learned_sort_jobs.start(
        signature,
        _run,
        dataset_id=ds_ctx.dataset_id,
        detector_id=det_ctx.detector_id,
    )

    if wait:
        job.done_event.wait(timeout=120)
        if job.status == "error":
            abort(500, message=job.error or "learned-sort failed")
        if job.status == "done":
            return _learned_sort_done_payload(job)

    return {"job_id": job.job_id, "status": "running", "current": 0, "total": 1}


def _stable_element_id_for_sig(el) -> str:
    """Return a stable identifier for a labelset element for signature use."""
    from vtscore.detectors.labelset_elements import stable_element_id

    return stable_element_id(el)


@sorting_bp.route("/api/learned-sort/result", methods=["GET"])
@sorting_bp.arguments(LearnedSortResultQuerySchema, location="query")
@sorting_bp.response(200, LearnedSortResponseSchema)
@sorting_bp.alt_response(404, description="Job not found.")
@sorting_bp.alt_response(500, description="Background learned-sort job failed.")
def learned_sort_result(query: dict):
    """Poll a learned-sort background job.

    Returns the same shape as the POST endpoint's ``done`` response when the
    job has finished, or a ``running`` snapshot otherwise.
    """
    from vtscore.concurrency.async_jobs import learned_sort_jobs

    job_id = query["job_id"]

    job = learned_sort_jobs.get(job_id)
    if job is None:
        # 404s are intercepted by the app-level ``NotFound`` errorhandler
        # in ``app.py`` and rendered with the legacy
        # ``{"error": "Not Found", "request_id": "..."}`` envelope; the
        # ``message`` kwarg and any extras (e.g. ``status="missing"``)
        # are dropped. Frontends rely on the HTTP status code for the
        # missing-job branch rather than a body field.
        abort(404, message="Job not found")

    if job.status in ("running", "pending"):
        return {
            "job_id": job.job_id,
            "status": "running",
            "current": job.current,
            "total": job.total,
        }
    if job.status == "error":
        abort(500, message=job.error or "learned-sort failed", job_id=job.job_id)
    if job.status == "cancelled":
        return {"job_id": job.job_id, "status": "cancelled"}
    return _learned_sort_done_payload(job)


@sorting_bp.route("/api/learned-sort/cancel/<job_id>", methods=["POST"])
@sorting_bp.response(200, LearnedSortCancelResponseSchema)
@sorting_bp.alt_response(404, description="Job not found.")
def cancel_learned_sort(job_id: str):
    """Cancel an in-flight learned-sort job.

    Sets the cancel flag on the :class:`AsyncJob`; the training loop
    polls it cooperatively. Returns 200 even when the job has already
    finished; the caller's contract is "make sure it's no longer
    running", which also holds for done / errored / already-cancelled
    jobs.
    """
    from vtscore.concurrency.async_jobs import learned_sort_jobs

    job = learned_sort_jobs.get(job_id)
    if job is None:
        abort(404, message="Job not found")
    job.cancel()
    return {"ok": True}


@sorting_bp.route("/api/votes", methods=["GET"])
@sorting_bp.response(200, VotesResponseSchema)
def get_votes():
    """Return current good/bad votes, click times, and learned scores."""
    from vtscore.state.core import _empty_detector_context, get_active_detector_context
    from vtscore.utils.scores import finite_or  # noqa: PLC0415

    click_times = get_vote_click_times()
    learned_scores = get_learned_scores()
    det_ctx = get_active_detector_context()
    if det_ctx is not _empty_detector_context and det_ctx.detector_id:
        labelset_good_count = det_ctx.labelset_good_count
        labelset_bad_count = det_ctx.labelset_bad_count
    else:
        labelset_good_count = len(good_votes)
        labelset_bad_count = len(bad_votes)
    # Defensive guard against non-finite scores poisoning the response: every
    # write site is already sanitised via ``sigmoid_to_finite_scores``, but
    # ``round(NaN, 4)`` returns ``NaN`` and Flask's default JSON provider
    # emits the literal token ``NaN``, which is invalid JSON that breaks every
    # browser ``JSON.parse``. Belt-and-braces audit M13.
    return {
        "good": sorted(good_votes),
        "bad": sorted(bad_votes),
        "click_times": {str(k): v for k, v in click_times.items()},
        "learned_scores": {str(k): round(finite_or(v), 4) for k, v in learned_scores.items()},
        "labelset_good_count": labelset_good_count,
        "labelset_bad_count": labelset_bad_count,
    }


@sorting_bp.route("/api/votes/clear", methods=["POST"])
@sorting_bp.response(200, OkResponseSchema)
@require_detector_header
def clear_votes_route():
    """Clear all votes without clearing medias.

    Used by the Label flow to reset votes before importing a model's labelset
    so that labels from a previous session don't contaminate the new model.
    """
    from vtsearch.state import clear_votes

    clear_votes()
    return {"ok": True}


@sorting_bp.route("/api/votes/seed-from-examples", methods=["POST"])
@sorting_bp.arguments(SeedFromExamplesRequestSchema)
@sorting_bp.response(200, SeedFromExamplesResponseSchema)
@require_dataset_header
@require_detector_header
def seed_votes_from_examples(body: dict):
    """Seed good votes from a model's media examples.

    For each ``type: "media"`` example, reads the file from
    ``data/example_media/``, computes its MD5, and either marks the
    matching loaded media as Good, or (if the example is new)
    embeds it, inserts it into the ``medias`` dict, and votes it Good.

    Returns::

        {"seeded": 2, "skipped": 1}
    """
    from vtscore.detectors.media_seeding import seed_good_votes_from_examples

    examples = body["examples"]

    seeded = seed_good_votes_from_examples(examples)
    skipped = len(examples) - seeded

    if seeded > 0:
        # Surface persistence failures explicitly instead of letting them
        # bubble as an uncaught 500; same C11/H30 pattern as
        # ``fill_labels_from_sort`` and ``vote_media``.  Without this, an
        # ``os.replace`` failure inside ``_write_detector`` would leave the
        # in-memory good votes committed while the on-disk labelset stayed
        # untouched, with no signal to the client beyond a generic 500.
        from vtscore.detectors.label_sync import sync_labels_to_loaded_detector

        try:
            sync_labels_to_loaded_detector()
        except Exception as exc:
            logging.getLogger(__name__).exception("seed_votes_from_examples: detector label sync failed")
            abort(500, message=f"Failed to persist seeded votes to detector store: {exc}")

        from vtscore.labels.sync import sync_to_labelset_source

        try:
            sync_to_labelset_source()
        except Exception:
            logging.getLogger(__name__).exception("seed_votes_from_examples: labelset source scheduling failed")

    return {"seeded": seeded, "skipped": skipped}


@sorting_bp.route("/api/textsort-suggestions", methods=["GET"])
@sorting_bp.response(200, TextsortSuggestionsResponseSchema)
def get_textsort_suggestions_route():
    """Return stored text-sort suggestions (most recent last)."""
    return {"suggestions": get_textsort_suggestions()}


@sorting_bp.route("/api/textsort-suggestions", methods=["POST"])
@sorting_bp.arguments(TextsortSuggestionRequestSchema)
@sorting_bp.response(200, OkResponseSchema)
@sorting_bp.alt_response(400, description="Empty or whitespace-only ``text``.")
def add_textsort_suggestion_route(body: dict):
    """Store a text-sort query as a suggested name for detectors/labelsets."""
    text = body.get("text", "").strip()
    if not text:
        abort(400, message="text is required")
    add_textsort_suggestion(text)
    return {"ok": True}


@sorting_bp.route("/api/inclusion", methods=["GET"])
@sorting_bp.response(200, InclusionResponseSchema)
def get_inclusion_route():
    """Get the current Inclusion setting."""
    return {"inclusion": get_inclusion()}


@sorting_bp.route("/api/inclusion", methods=["POST"])
@sorting_bp.arguments(InclusionRequestSchema)
@sorting_bp.response(200, InclusionResponseSchema)
def set_inclusion_route(body: dict):
    """Set the Inclusion setting (clamped to ``[-10, 10]``)."""
    new_inclusion = int(max(-10, min(10, body["inclusion"])))
    set_inclusion(new_inclusion)
    return {"inclusion": get_inclusion()}


@sorting_bp.route("/api/safe-thresholds", methods=["GET"])
@sorting_bp.response(200, SafeThresholdsResponseSchema)
def get_safe_thresholds_route():
    """Get the current Safe Thresholds setting."""
    return {"safe_thresholds": get_safe_thresholds()}


@sorting_bp.route("/api/safe-thresholds", methods=["POST"])
@sorting_bp.arguments(SafeThresholdsRequestSchema)
@sorting_bp.response(200, SafeThresholdsResponseSchema)
def set_safe_thresholds_route(body: dict):
    """Set the Safe Thresholds setting."""
    set_safe_thresholds(body["safe_thresholds"])
    return {"safe_thresholds": get_safe_thresholds()}


def _example_sort_from_path(file_path: Path) -> tuple:
    """Embed a media file and sort all loaded medias by cosine similarity.

    Returns ``(results_list, threshold)`` on success or raises on error.
    The file at *file_path* is embedded using the embedder of the currently
    loaded dataset.
    """
    snap = snapshot_medias()
    if not snap:
        raise ValueError("No medias loaded")

    emb = _get_embedder_for_loaded_data()
    if emb is None:
        raise ValueError("No embedder available for loaded dataset")
    from vtscore.media.embedder import media_from_path  # noqa: PLC0415

    example_embedding = emb.embed_media(media_from_path(file_path))

    if example_embedding is None:
        raise ValueError("Failed to embed media file")

    return _cosine_sort(example_embedding)


def _parse_crop_params(raw: str | None) -> dict | None:
    """Parse a JSON string of crop bounds, or return None if absent.

    Returns ``None`` when *raw* is empty/missing.  Returns a dict otherwise
    (caller validates the contents against the target media type).
    """
    if not raw:
        return None
    try:
        params = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(params, dict):
        return None
    return params


def _apply_crop_or_keep(temp_path: Path, crop_params: dict | None) -> Path:
    """Apply *crop_params* to *temp_path* in-place when set; otherwise keep file.

    Resolves the target media type from the loaded dataset's first media
    item (the embedder is the same one we're about to use).  Writes the
    cropped bytes back to *temp_path*.
    """
    if not crop_params:
        return temp_path

    snap = snapshot_medias()
    if not snap:
        return temp_path
    first_media = next(iter(snap.values()))
    media_type = first_media.get("media_type", "")

    from vtscore.media.cropping import crop_file_bytes

    cropped = crop_file_bytes(temp_path, media_type, crop_params)
    temp_path.write_bytes(cropped)
    return temp_path


@sorting_bp.route("/api/example-sort", methods=["POST"])
@sorting_bp.response(200, SortResponseSchema)
@sorting_bp.alt_response(
    400,
    description="No file provided, no filename, or no medias loaded.",
)
@sorting_bp.alt_response(500, description="Example sort failed (embedder error or unexpected exception).")
def example_sort():
    """Sort medias by similarity to an uploaded example media file.

    Optional ``crop_params`` form field carries a JSON object with the
    bounds for a user-cropped sub-region (e.g. ``{"start": 1.5, "end": 3}``
    for audio or ``{"box": [x1, y1, x2, y2]}`` for images).  When present
    the file is cropped server-side before embedding.
    """
    if "file" not in request.files:
        abort(400, message="No file provided")

    file = request.files["file"]
    if not file.filename:
        abort(400, message="No file selected")

    if not snapshot_medias():
        abort(400, message="No medias loaded")

    try:
        # Save uploaded file to a unique temp location to avoid race conditions
        import uuid

        suffix = Path(file.filename).suffix or ".bin"
        temp_path = DATA_DIR / f"temp_example_{uuid.uuid4().hex}{suffix}"
        DATA_DIR.mkdir(exist_ok=True)
        file.save(temp_path)

        try:
            crop_params = _parse_crop_params(request.form.get("crop_params"))
            _apply_crop_or_keep(temp_path, crop_params)
            results, thresh = _example_sort_from_path(temp_path)
        finally:
            # Clean up temp file even if sorting raises
            temp_path.unlink(missing_ok=True)

        return {"results": results, "threshold": thresh}

    except Exception as exc:
        from werkzeug.exceptions import HTTPException

        if isinstance(exc, HTTPException):
            raise
        logging.getLogger(__name__).exception("example-sort failed")
        abort(500, message=f"Example sort failed: {format_exception_detail(exc)}")


def _parse_label_file(file) -> list[dict]:
    """Parse the uploaded label file and return its ``labels`` list, or abort 400."""
    text = file.read().decode("utf-8")
    try:
        label_data = json.loads(text)
    except Exception:
        abort(400, message="Invalid label file format")
    labels = label_data.get("labels", [])
    if not labels:
        abort(400, message="No labels found in file")
    return labels


def _embed_external_labels(labels: list[dict], emb) -> tuple[list, list[float], int, int]:
    """Embed every well-formed entry in *labels* using *emb*.

    Returns ``(X_list, y_list, loaded_count, skipped_count)``. Entries are
    skipped (not aborted) when the label is malformed, the path is missing
    or escapes the allowed directory, the file doesn't exist, or the
    embedder returns None.
    """
    from vtscore.media.embedder import media_from_path  # noqa: PLC0415

    X_list: list = []
    y_list: list[float] = []
    loaded = 0
    skipped = 0
    file_base = _paths.get_file_access_base_dir()

    for entry in labels:
        label = entry.get("label")
        if label not in ("good", "bad"):
            skipped += 1
            continue

        raw_path = entry.get("path") or entry.get("file") or entry.get("filename")
        if not raw_path:
            skipped += 1
            continue

        media_path = Path(raw_path)
        try:
            _paths.validate_server_filepath(str(media_path), base_dir=file_base)
        except ValueError:
            skipped += 1
            continue
        if not media_path.exists():
            skipped += 1
            continue

        embedding = emb.embed_media(media_from_path(media_path))
        if embedding is None:
            skipped += 1
            continue

        X_list.append(embedding)
        y_list.append(1.0 if label == "good" else 0.0)
        loaded += 1

    return X_list, y_list, loaded, skipped


def _train_and_score_dataset(X_list: list, y_list: list[float]) -> tuple[list[dict], float]:
    """Train an MLP on (X, y), then score every media in the active dataset."""
    import torch  # noqa: PLC0415

    from vtscore.detectors.training import train_and_threshold
    from vtscore.embedding.matrix import get_embedding_matrix_for_snap  # noqa: PLC0415
    from vtscore.utils.scores import sigmoid_to_finite_scores  # noqa: PLC0415

    snap = snapshot_medias()
    model, threshold = train_and_threshold(X_list, y_list, snap=snap)

    all_ids, all_embs = get_embedding_matrix_for_snap(snap)
    X_all = torch.from_numpy(all_embs)
    with torch.no_grad():
        X_all = X_all.to(next(model.parameters()).device)
        scores = sigmoid_to_finite_scores(model(X_all))

    paired = sorted(zip(all_ids, scores, strict=True), key=lambda x: x[1], reverse=True)
    results = [{"id": cid, "score": round(s, 4)} for cid, s in paired]
    return results, threshold


@sorting_bp.route("/api/label-file-sort", methods=["POST"])
@sorting_bp.response(200, LabelFileSortResponseSchema)
@sorting_bp.alt_response(
    400,
    description=(
        "No file / no filename, no medias loaded, no embedder, invalid label file, "
        "no labels in file, too few valid labeled files, or missing good/bad split."
    ),
)
@sorting_bp.alt_response(500, description="Label file sort failed (unexpected exception).")
def label_file_sort():
    """Train MLP on external media files from a label file, then sort all medias."""
    if "file" not in request.files:
        abort(400, message="No file provided")

    file = request.files["file"]
    if not file.filename:
        abort(400, message="No file selected")

    if not snapshot_medias():
        abort(400, message="No medias loaded")

    emb = _get_embedder_for_loaded_data()
    if emb is None:
        abort(400, message="No embedder available for loaded dataset")

    try:
        labels = _parse_label_file(file)
        X_list, y_list, loaded, skipped = _embed_external_labels(labels, emb)

        if loaded < 2:
            abort(
                400,
                message=f"Need at least 2 valid labeled files (loaded {loaded}, skipped {skipped})",
            )

        from vtscore.detectors.training import validate_good_bad_split

        try:
            validate_good_bad_split(y_list)
        except ValueError:
            abort(400, message="Need at least one good and one bad labeled example")

        results, threshold = _train_and_score_dataset(X_list, y_list)
        return {
            "results": results,
            "threshold": round(threshold, 4),
            "loaded": loaded,
            "skipped": skipped,
        }

    except Exception as exc:
        from werkzeug.exceptions import HTTPException

        if isinstance(exc, HTTPException):
            raise
        logging.getLogger(__name__).exception("label-file-sort failed")
        abort(500, message="Label file sort failed")


@sorting_bp.route("/api/diversity-tree/next", methods=["GET", "POST"])
@sorting_bp.response(200, DiversityTreeNextResponseSchema)
@sorting_bp.alt_response(400, description="Invalid score keys/values or threshold value (POST only).")
def diversity_tree_next():
    """Return the next diverse sample from the Diversity Tree.

    Accepts an optional POST body with ``{"scores": {id: score, ...},
    "threshold": <float>}`` so the sort mode influences which element is
    picked from the next unseen node.  When a threshold is provided, the
    node's median score determines direction: above-threshold nodes yield
    the lowest-scored element (surprise in a "good" region), while
    below-threshold nodes yield the highest-scored element (surprise in a
    "bad" region).  Without scores the first element in the node is returned.

    Returns ``{"id": <media_id>}`` or ``{"id": null}`` when the tree is
    exhausted or not yet built.  Also includes ``diversity_level`` (the
    number of consecutive BFS-order seen nodes) so the frontend can display
    progress, and ``exhausted`` (bool) which is true when the tree exists
    but every node has already been seen.
    """
    scores: dict[int, float] | None = None
    threshold: float | None = None
    if request.method == "POST":
        # ``request.get_json(silent=True)`` keeps the legacy lenient body
        # handling; flask-smorest's ``arguments`` would 422 on a missing
        # body, but we want GET / POST to behave identically when nothing
        # is sent. The shape-level validation lives in the schema; per-
        # value int-coercion stays in the handler so we can return a 400
        # with a custom message.
        data = request.get_json(silent=True) or {}
        raw_scores = data.get("scores")
        if isinstance(raw_scores, dict):
            try:
                scores = {int(k): float(v) for k, v in raw_scores.items()}
            except (ValueError, TypeError):
                abort(400, message="Invalid score keys or values")
        raw_threshold = data.get("threshold")
        if raw_threshold is not None:
            try:
                threshold = float(raw_threshold)
            except (ValueError, TypeError):
                abort(400, message="Invalid threshold value")

    tree = get_diversity_tree()
    next_id = diversity_tree_next_sample(scores=scores, threshold=threshold)
    level = tree.diversity_level() if tree is not None else 0
    exhausted = tree is not None and next_id is None
    return {"id": next_id, "diversity_level": level, "exhausted": exhausted}
