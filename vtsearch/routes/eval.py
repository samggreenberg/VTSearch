"""Blueprint for evaluation and labeling progress routes.

Migrated to ``flask_smorest`` so the routes are described in
``/api/openapi.json``. See ``docs/plans/openapi-schema.md``. Schema-level
failures (missing ``metric`` / ``job_id``, invalid metric value) surface
as 422 with the standard ``errors`` envelope; handler-level rejects
(no votes / no label history, missing job) keep their HTTP codes
(400 / 404 / 500) with the standard ``message`` envelope.
"""

from flask_smorest import Blueprint, abort

from vtsearch.detectors.labeling_progress import (
    analyze_labeling_progress,
    calculate_diversity_level_over_time,
    calculate_error_cost_over_time,
    calculate_prediction_stability_over_time,
    compute_labeling_status,
)
from vtsearch.schemas.eval import (
    EvalTrainAndScoreRequestSchema,
    EvalTrainAndScoreResponseSchema,
    EvalTrainAndScoreResultQuerySchema,
    IndicatorScoreHistoryQuerySchema,
    IndicatorScoreHistoryResponseSchema,
    LabelingProgressResponseSchema,
    LabelingStatusResponseSchema,
)
from vtsearch.state import (
    bad_votes,
    get_diversity_tree,
    get_inclusion,
    good_votes,
    label_history,
    snapshot_medias,
)
from vtsearch.concurrency.progress import (
    get_eval_progress,
    update_eval_progress,
)

eval_bp = Blueprint(
    "eval",
    __name__,
    description="Labeling-progress analysis and learned-sort eval indicators.",
)


@eval_bp.route("/api/labeling-progress", methods=["POST"])
@eval_bp.response(200, LabelingProgressResponseSchema)
@eval_bp.alt_response(400, description="No good/bad votes, or no label history.")
@eval_bp.alt_response(500, description="Labeling-progress computation failed.")
def labeling_progress():
    """Analyze labeling progress and calculate stopping condition metrics."""
    if not good_votes or not bad_votes:
        abort(400, message="need at least one good and one bad vote")

    if not label_history:
        abort(400, message="no label history available")

    try:
        return analyze_labeling_progress(snapshot_medias(), label_history, good_votes, bad_votes, get_inclusion())
    except Exception:
        import logging

        logging.getLogger(__name__).exception("labeling-progress failed")
        abort(500, message="Labeling progress computation failed")


@eval_bp.route("/api/labeling-status", methods=["GET"])
@eval_bp.response(200, LabelingStatusResponseSchema)
@eval_bp.alt_response(500, description="Labeling-status computation failed.")
def labeling_status_indicator():
    """Return per-metric red/yellow/green labeling statuses.

    Returns ``smart``, ``stable``, and ``span`` sub-objects, each with a
    ``status`` field of ``"red"``, ``"yellow"``, or ``"green"``.
    """
    try:
        tree = get_diversity_tree()
        span = tree.span_info() if tree is not None else None
        return compute_labeling_status(
            snapshot_medias(), label_history, good_votes, bad_votes, get_inclusion(), span_info=span
        )
    except Exception:
        import logging

        logging.getLogger(__name__).exception("labeling-status failed")
        abort(500, message="Labeling status computation failed")


@eval_bp.route("/api/indicator-score-history", methods=["GET"])
@eval_bp.arguments(IndicatorScoreHistoryQuerySchema, location="query")
@eval_bp.response(200, IndicatorScoreHistoryResponseSchema)
@eval_bp.alt_response(500, description="Score-history computation failed.")
def indicator_score_history(query: dict):
    """Return cached indicator score history for a given metric.

    Reads only the per-step cache populated by the labeling-status
    polling — no models are retrained.
    """
    metric = query["metric"]

    clips = snapshot_medias()
    inclusion = get_inclusion()

    try:
        if metric == "smart":
            data = calculate_error_cost_over_time(clips, label_history, good_votes, bad_votes, inclusion)
        elif metric == "stable":
            data = calculate_prediction_stability_over_time(clips, label_history, inclusion)
        else:
            data = calculate_diversity_level_over_time(clips, label_history, inclusion)
        return {"metric": metric, "history": data}
    except Exception:
        import logging

        logging.getLogger(__name__).exception("indicator-score-history failed")
        abort(500, message="Score history computation failed")


_METRIC_KEY = {"smart": "error_cost", "stable": "stability", "diverse": "diversity"}


def _eval_done_payload(job) -> dict:
    """Build the JSON body for a finished eval job, including metric data."""
    result = job.result or {}
    metric = result.get("metric") or ""
    data_key = _METRIC_KEY.get(metric, "data")
    return {
        "job_id": job.job_id,
        "status": "done",
        "metric": metric,
        data_key: result.get("data", []),
    }


@eval_bp.route("/api/eval/train-and-score", methods=["POST"])
@eval_bp.arguments(EvalTrainAndScoreRequestSchema)
@eval_bp.response(200, EvalTrainAndScoreResponseSchema)
@eval_bp.alt_response(500, description="Evaluation computation failed (only when ``wait=true``).")
def eval_train_and_score(body: dict):
    """Start (or short-circuit) an eval train-and-score computation.

    The work walks the full ``label_history`` retraining a small MLP at
    every step, which used to block every other request on the gthread
    pool.  We now run it on a background daemon thread and return a
    ``job_id``; clients poll ``/api/eval/train-and-score/result`` for the
    metric data; the ``eval`` SSE channel on ``/api/events`` carries
    live progress.

    A signature cache keyed by ``(metric, history, votes, inclusion,
    dataset, detector)`` short-circuits identical re-runs.

    Tests can pass ``{"wait": true}`` to block until the job completes.
    """
    from vtsearch.concurrency.async_jobs import eval_jobs
    from vtsearch.state.core import (
        get_active_context,
        get_active_detector_context,
        set_thread_dataset_context,
        set_thread_detector_context,
    )

    metric = body["metric"]
    wait = body["wait"]

    clips = snapshot_medias()
    inclusion = get_inclusion()
    history = list(label_history)
    good_snap = dict(good_votes)
    bad_snap = dict(bad_votes)

    ds_ctx = get_active_context()
    det_ctx = get_active_detector_context()

    signature = (
        metric,
        ds_ctx.dataset_id,
        det_ctx.detector_id,
        tuple(sorted(clips.keys())),
        tuple(history),
        tuple(sorted(good_snap)),
        tuple(sorted(bad_snap)),
        inclusion,
    )

    cached = eval_jobs.cached_for(signature)
    if cached is not None:
        return _eval_done_payload(cached)

    n_total = max(len(history) - 1, 0)
    update_eval_progress("running", f"Computing {metric}...", 0, n_total)

    def _run(job):
        set_thread_dataset_context(ds_ctx)
        set_thread_detector_context(det_ctx)
        try:
            if metric == "smart":
                data = calculate_error_cost_over_time(clips, history, good_snap, bad_snap, inclusion)
            elif metric == "stable":
                data = calculate_prediction_stability_over_time(clips, history, inclusion)
            else:
                data = calculate_diversity_level_over_time(clips, history, inclusion)
            job.result = {"metric": metric, "data": data}
            update_eval_progress("idle", "Done", n_total, n_total)
        except Exception:
            update_eval_progress("idle", "Error", 0, 0)
            raise
        finally:
            set_thread_dataset_context(None)
            set_thread_detector_context(None)

    job = eval_jobs.start(signature, _run)

    if wait:
        job.done_event.wait(timeout=300)
        if job.status == "error":
            abort(500, message=job.error or "Evaluation computation failed")
        if job.status == "done":
            return _eval_done_payload(job)

    return {"job_id": job.job_id, "status": "running", "current": 0, "total": n_total}


@eval_bp.route("/api/eval/train-and-score/result", methods=["GET"])
@eval_bp.arguments(EvalTrainAndScoreResultQuerySchema, location="query")
@eval_bp.response(200, EvalTrainAndScoreResponseSchema)
@eval_bp.alt_response(404, description="Job not found.")
@eval_bp.alt_response(500, description="Background evaluation job failed.")
def eval_train_and_score_result(query: dict):
    """Poll a background eval train-and-score job."""
    from vtsearch.concurrency.async_jobs import eval_jobs

    job_id = query["job_id"]

    job = eval_jobs.get(job_id)
    if job is None:
        abort(404, message="Job not found", job_id=job_id, status="missing")

    if job.status in ("running", "pending"):
        prog = get_eval_progress()
        return {
            "job_id": job.job_id,
            "status": "running",
            "current": prog.get("current", 0),
            "total": prog.get("total", 0),
        }
    if job.status == "error":
        abort(500, message=job.error or "Evaluation computation failed", job_id=job.job_id)
    if job.status == "cancelled":
        return {"job_id": job.job_id, "status": "cancelled"}
    return _eval_done_payload(job)
