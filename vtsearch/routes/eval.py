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


@eval_bp.route("/api/eval/train-and-score", methods=["POST"])
def eval_train_and_score():
    """Compute indicator score history for a given metric.

    Expects JSON::

        {"metric": "smart" | "stable" | "diverse"}

    Runs the computation in a background thread and returns the result
    synchronously. Progress can be polled via ``/api/eval/voting-iterations``.
    """
    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    metric = data.get("metric", "").strip()
    if metric not in ("smart", "stable", "diverse"):
        return jsonify({"error": "metric must be one of: smart, stable, diverse"}), 400

    clips = snapshot_medias()
    inclusion = get_inclusion()
    history = list(label_history)
    n_total = max(len(history) - 1, 0)

    update_eval_progress("running", f"Computing {metric}...", 0, n_total)

    try:
        if metric == "smart":
            result_data = calculate_error_cost_over_time(clips, history, good_votes, bad_votes, inclusion)
            update_eval_progress("idle", "Done", n_total, n_total)
            return jsonify({"error_cost": result_data})
        elif metric == "stable":
            result_data = calculate_prediction_stability_over_time(clips, history, inclusion)
            update_eval_progress("idle", "Done", n_total, n_total)
            return jsonify({"stability": result_data})
        else:
            result_data = calculate_diversity_level_over_time(clips, history, inclusion)
            update_eval_progress("idle", "Done", n_total, n_total)
            return jsonify({"diversity": result_data})
    except Exception:
        import logging

        logging.getLogger(__name__).exception("eval train-and-score failed")
        update_eval_progress("idle", "Error", 0, 0)
        return jsonify({"error": "Evaluation computation failed"}), 500


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
