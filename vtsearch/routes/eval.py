"""Blueprint for evaluation and labeling progress routes."""

from flask import Blueprint, jsonify, request

from vtsearch.routes.helpers import get_json_or_400
from vtsearch.models import (
    analyze_labeling_progress,
    calculate_diversity_level_over_time,
    calculate_error_cost_over_time,
    calculate_prediction_stability_over_time,
    compute_labeling_status,
)
from vtsearch.utils import (
    bad_votes,
    get_diversity_tree,
    get_eval_progress,
    get_inclusion,
    good_votes,
    label_history,
    snapshot_medias,
    update_eval_progress,
)

eval_bp = Blueprint("eval", __name__)


@eval_bp.route("/api/labeling-progress", methods=["POST"])
def labeling_progress():
    """Analyze labeling progress and calculate stopping condition metrics."""
    if not good_votes or not bad_votes:
        return jsonify({"error": "need at least one good and one bad vote"}), 400

    if not label_history:
        return jsonify({"error": "no label history available"}), 400

    try:
        analysis = analyze_labeling_progress(snapshot_medias(), label_history, good_votes, bad_votes, get_inclusion())
        return jsonify(analysis)
    except Exception:
        import logging

        logging.getLogger(__name__).exception("labeling-progress failed")
        return jsonify({"error": "Labeling progress computation failed"}), 500


@eval_bp.route("/api/labeling-status", methods=["GET"])
def labeling_status_indicator():
    """Return per-metric red/yellow/green labeling statuses.

    Returns ``smart``, ``stable``, and ``span`` sub-objects, each with a
    ``status`` field of ``"red"``, ``"yellow"``, or ``"green"``.
    """
    try:
        tree = get_diversity_tree()
        span = tree.span_info() if tree is not None else None
        status = compute_labeling_status(
            snapshot_medias(), label_history, good_votes, bad_votes, get_inclusion(), span_info=span
        )
        return jsonify(status)
    except Exception:
        import logging

        logging.getLogger(__name__).exception("labeling-status failed")
        return jsonify({"error": "Labeling status computation failed"}), 500


@eval_bp.route("/api/indicator-score-history", methods=["GET"])
def indicator_score_history():
    """Return cached indicator score history for a given metric.

    Query parameter ``metric`` must be one of ``smart``, ``stable``, or
    ``diverse``.  Returns the cached per-step data without retraining
    models — only data already computed by the labeling-status polling
    is returned.
    """
    metric = request.args.get("metric", "").strip()
    if metric not in ("smart", "stable", "diverse"):
        return jsonify({"error": "metric must be one of: smart, stable, diverse"}), 400

    clips = snapshot_medias()
    inclusion = get_inclusion()

    try:
        if metric == "smart":
            data = calculate_error_cost_over_time(clips, label_history, good_votes, bad_votes, inclusion)
            return jsonify({"metric": "smart", "history": data})
        elif metric == "stable":
            data = calculate_prediction_stability_over_time(clips, label_history, inclusion)
            return jsonify({"metric": "stable", "history": data})
        else:
            data = calculate_diversity_level_over_time(clips, label_history, inclusion)
            return jsonify({"metric": "diverse", "history": data})
    except Exception:
        import logging

        logging.getLogger(__name__).exception("indicator-score-history failed")
        return jsonify({"error": "Score history computation failed"}), 500


_METRIC_KEY = {"smart": "error_cost", "stable": "stability", "diverse": "diversity"}


def _eval_done_payload(job) -> dict:
    """Build the JSON body for a finished eval job, including metric data."""
    result = job.result or {}
    metric = result.get("metric")
    data_key = _METRIC_KEY.get(metric, "data")
    return {
        "job_id": job.job_id,
        "status": "done",
        "metric": metric,
        data_key: result.get("data", []),
    }


@eval_bp.route("/api/eval/train-and-score", methods=["POST"])
def eval_train_and_score():
    """Start (or short-circuit) an eval train-and-score computation.

    The work walks the full ``label_history`` retraining a small MLP at
    every step, which used to block every other request on the gthread
    pool.  We now run it on a background daemon thread and return a
    ``job_id``; clients poll ``/api/eval/train-and-score/result`` for the
    metric data and ``/api/eval/voting-iterations`` for progress.

    A signature cache keyed by ``(metric, history, votes, inclusion,
    dataset, detector)`` short-circuits identical re-runs.

    Tests can pass ``{"wait": true}`` to block until the job completes.
    """
    from vtsearch.utils.async_jobs import eval_jobs
    from vtsearch.utils.state_core import (
        get_active_context,
        get_active_detector_context,
        set_thread_dataset_context,
        set_thread_detector_context,
    )

    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    metric = data.get("metric", "").strip()
    if metric not in ("smart", "stable", "diverse"):
        return jsonify({"error": "metric must be one of: smart, stable, diverse"}), 400

    wait = bool(data.get("wait"))

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
        return jsonify(_eval_done_payload(cached))

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
            return jsonify({"error": job.error or "Evaluation computation failed"}), 500
        if job.status == "done":
            return jsonify(_eval_done_payload(job))

    return jsonify({"job_id": job.job_id, "status": "running", "current": 0, "total": n_total})


@eval_bp.route("/api/eval/train-and-score/result", methods=["GET"])
def eval_train_and_score_result():
    """Poll a background eval train-and-score job."""
    from vtsearch.utils.async_jobs import eval_jobs

    job_id = request.args.get("job_id", "").strip()
    if not job_id:
        return jsonify({"error": "job_id is required"}), 400

    job = eval_jobs.get(job_id)
    if job is None:
        return jsonify({"status": "missing", "error": "Job not found"}), 404

    if job.status == "running":
        prog = get_eval_progress()
        return jsonify({
            "job_id": job.job_id,
            "status": "running",
            "current": prog.get("current", 0),
            "total": prog.get("total", 0),
        })
    if job.status == "error":
        return jsonify({
            "job_id": job.job_id,
            "status": "error",
            "error": job.error or "Evaluation computation failed",
        }), 500
    if job.status == "cancelled":
        return jsonify({"job_id": job.job_id, "status": "cancelled"})
    return jsonify(_eval_done_payload(job))


@eval_bp.route("/api/eval/voting-iterations", methods=["GET"])
def eval_voting_iterations():
    """Return progress of the current eval computation.

    Returns::

        {"progress": <int>, "total": <int>, "done": <bool>}
    """
    prog = get_eval_progress()
    done = prog["status"] == "idle" and prog["total"] > 0 and prog["current"] >= prog["total"]
    return jsonify(
        {
            "progress": prog["current"],
            "total": prog["total"],
            "done": done,
        }
    )
