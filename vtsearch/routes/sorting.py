"""Blueprint for sorting and voting routes."""

import json
import threading
from pathlib import Path

import numpy as np
from flask import Blueprint, jsonify, request

from vtsearch.routes.helpers import get_json_or_400

from vtsearch.config import DATA_DIR
import vtsearch.utils.paths as _paths
from vtsearch.models import (
    calculate_cross_calibration_threshold,
    calculate_gmm_threshold,
    calculate_safe_threshold,
    embed_text_query,
    inject_live_model,
    train_and_score,
    train_model,
)
from vtsearch.utils import (
    add_textsort_suggestion,
    bad_votes,
    diversity_tree_next_sample,
    get_calibrate_count,
    get_calibration_fraction,
    get_diversity_tree,
    get_inclusion,
    get_learned_scores,
    get_safe_thresholds,
    get_sort_progress,
    get_textsort_suggestions,
    get_vote_click_times,
    good_votes,
    set_inclusion,
    set_safe_thresholds,
    snapshot_medias,
    update_learned_scores,
    update_sort_progress,
)

sorting_bp = Blueprint("sorting", __name__)


@sorting_bp.route("/api/sort/progress")
def sort_progress():
    """Return the current progress of a text sort operation."""
    return jsonify(get_sort_progress())


def _cosine_sort(query_vec):
    """Sort all loaded medias by cosine similarity to *query_vec*.

    Returns ``(results, threshold)`` where *results* is a list of
    ``{"id": …, "similarity": …}`` dicts sorted descending, and
    *threshold* is the GMM-based boundary (rounded to 4 decimals).
    """
    snap = snapshot_medias()
    all_ids = list(snap.keys())
    all_embs = np.array([snap[cid]["embedding"] for cid in all_ids])
    query_norm = np.linalg.norm(query_vec)
    emb_norms = np.linalg.norm(all_embs, axis=1)
    norm_products = emb_norms * query_norm
    safe_norms = np.where(norm_products == 0, 1.0, norm_products)
    similarities = np.dot(all_embs, query_vec) / safe_norms
    similarities = np.where(norm_products == 0, 0.0, similarities)

    results = [{"id": cid, "similarity": round(float(sim), 4)} for cid, sim in zip(all_ids, similarities)]
    threshold = calculate_gmm_threshold(similarities.tolist())
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results, round(threshold, 4)


_embedder_load_lock = threading.Lock()


def _get_embedder_for_loaded_data():
    """Return the appropriate embedder for the currently loaded dataset."""
    snap = snapshot_medias()
    if not snap:
        return None
    first = next(iter(snap.values()))
    embedder_name = first.get("embedder", "")
    media_type = first.get("type", "audio")

    from vtsearch.media import embedders_for_type, get_embedder

    if embedder_name:
        try:
            return get_embedder(embedder_name)
        except KeyError:
            pass

    avail = embedders_for_type(media_type)
    return avail[0] if avail else None


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
def sort_clips():
    """Return medias sorted by cosine similarity to a text query."""
    data = get_json_or_400()
    if not isinstance(data, dict):
        update_sort_progress("idle")
        return data

    text = data.get("text", "").strip()
    if not text:
        update_sort_progress("idle")
        return jsonify({"error": "text is required"}), 400

    snap = snapshot_medias()
    if not snap:
        update_sort_progress("idle")
        return jsonify({"error": "No medias loaded"}), 400

    first = next(iter(snap.values()))
    media_type = first.get("type", "audio")
    embedder_name = first.get("embedder", "")
    total_steps = 3  # load embedder, embed query, compute similarities

    _load_embedder_with_progress(media_type, total_steps)
    update_sort_progress("sorting", "Embedding text query…", 1, total_steps)

    from vtsearch import settings

    enrich = settings.get_enrich_descriptions()
    text_vec = embed_text_query(text, media_type, enrich=enrich, embedder_name=embedder_name)
    if text_vec is None:
        update_sort_progress("idle")
        return jsonify({"error": f"Could not embed text for media type {media_type}"}), 500

    update_sort_progress("sorting", "Computing similarities…", 2, total_steps)
    results, threshold = _cosine_sort(text_vec)
    update_sort_progress("idle")
    return jsonify({"results": results, "threshold": threshold})


@sorting_bp.route("/api/learned-sort", methods=["POST"])
def learned_sort():
    """Train MLP on voted medias, return all medias sorted by predicted score."""
    if not good_votes or not bad_votes:
        return jsonify({"error": "need at least one good and one bad vote"}), 400
    snap = snapshot_medias()
    results, threshold, model = train_and_score(
        snap,
        good_votes,
        bad_votes,
        get_inclusion(),
        safe_thresholds=get_safe_thresholds(),
        calibrate_count=get_calibrate_count(),
        calibration_fraction=get_calibration_fraction(),
    )
    # Store scores so the /api/votes endpoint can provide confidence info.
    update_learned_scores({r["id"]: r["score"] for r in results})
    # Inject the live model into the progress cache so indicators and the
    # progress line-graph use the actual model that guided sorting, rather
    # than retraining an independent model from scratch.
    if model is not None:
        inject_live_model(good_votes, bad_votes, model, threshold)

    # Cache the trained MLP and threshold in the active DetectorContext so
    # that Find can use them directly without re-resolving and retraining.
    from vtsearch.utils.state_core import get_active_detector_context, _empty_detector_context

    det_ctx = get_active_detector_context()
    if det_ctx is not _empty_detector_context and model is not None:
        det_ctx.model = model
        det_ctx.threshold = threshold
        # Cache the voted media items with embeddings for cross-embedder scenarios.
        training = {}
        for cid in list(good_votes) + list(bad_votes):
            if cid in snap:
                training[cid] = snap[cid]
        det_ctx.training_medias = training
        # Record the embedder from the current dataset's media.
        if snap:
            first = next(iter(snap.values()), {})
            det_ctx.embedder = first.get("embedder", "")
            det_ctx.media_type = first.get("type", "")

    return jsonify({"results": results, "threshold": round(threshold, 4)})


@sorting_bp.route("/api/votes")
def get_votes():
    click_times = get_vote_click_times()
    learned_scores = get_learned_scores()
    return jsonify(
        {
            "good": sorted(good_votes),
            "bad": sorted(bad_votes),
            "click_times": {str(k): v for k, v in click_times.items()},
            "learned_scores": {str(k): round(v, 4) for k, v in learned_scores.items()},
        }
    )


@sorting_bp.route("/api/votes/clear", methods=["POST"])
def clear_votes_route():
    """Clear all votes without clearing medias.

    Used by the Label flow to reset votes before importing a model's labelset
    so that labels from a previous session don't contaminate the new model.
    """
    from vtsearch.utils import clear_votes

    clear_votes()
    return jsonify({"ok": True})


@sorting_bp.route("/api/votes/seed-from-examples", methods=["POST"])
def seed_votes_from_examples():
    """Seed good votes from a model's media examples.

    For each ``type: "media"`` example, reads the file from
    ``data/example_media/``, computes its MD5, and either marks the
    matching loaded media as Good, or — if the example is new —
    embeds it, inserts it into the ``medias`` dict, and votes it Good.

    Expects JSON::

        {"examples": [{"type": "media", "value": "abc123.wav"}, ...]}

    Returns::

        {"seeded": 2, "skipped": 1}
    """
    from vtsearch.models.media_seeding import seed_good_votes_from_examples

    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    examples = data.get("examples")
    if not isinstance(examples, list):
        return jsonify({"error": "examples must be a list"}), 400

    seeded = seed_good_votes_from_examples(examples)
    skipped = len(examples) - seeded

    if seeded > 0:
        from vtsearch.routes.trainable_models import sync_labels_to_loaded_model

        sync_labels_to_loaded_model()

    return jsonify({"seeded": seeded, "skipped": skipped})


@sorting_bp.route("/api/textsort-suggestions")
def get_textsort_suggestions_route():
    """Return stored text-sort suggestions (most recent last)."""
    return jsonify({"suggestions": get_textsort_suggestions()})


@sorting_bp.route("/api/textsort-suggestions", methods=["POST"])
def add_textsort_suggestion_route():
    """Store a text-sort query as a suggested name for detectors/labelsets."""
    data = get_json_or_400()
    if not isinstance(data, dict):
        return data
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400
    add_textsort_suggestion(text)
    return jsonify({"ok": True})


@sorting_bp.route("/api/inclusion")
def get_inclusion_route():
    """Get the current Inclusion setting."""
    return jsonify({"inclusion": get_inclusion()})


@sorting_bp.route("/api/inclusion", methods=["POST"])
def set_inclusion_route():
    """Set the Inclusion setting."""
    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    new_inclusion = data.get("inclusion")

    if isinstance(new_inclusion, bool) or not isinstance(new_inclusion, (int, float)):
        return jsonify({"error": "inclusion must be a number"}), 400

    # Clamp to -10 to +10 range
    new_inclusion = int(max(-10, min(10, new_inclusion)))
    set_inclusion(new_inclusion)

    return jsonify({"inclusion": get_inclusion()})


@sorting_bp.route("/api/safe-thresholds")
def get_safe_thresholds_route():
    """Get the current Safe Thresholds setting."""
    return jsonify({"safe_thresholds": get_safe_thresholds()})


@sorting_bp.route("/api/safe-thresholds", methods=["POST"])
def set_safe_thresholds_route():
    """Set the Safe Thresholds setting."""
    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    value = data.get("safe_thresholds")
    if not isinstance(value, bool):
        return jsonify({"error": "safe_thresholds must be a boolean"}), 400

    set_safe_thresholds(value)
    return jsonify({"safe_thresholds": get_safe_thresholds()})


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
    example_embedding = emb.embed_media(file_path)

    if example_embedding is None:
        raise ValueError("Failed to embed media file")

    return _cosine_sort(example_embedding)


@sorting_bp.route("/api/example-sort", methods=["POST"])
def example_sort():
    """Sort medias by similarity to an uploaded example media file."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    if not snapshot_medias():
        return jsonify({"error": "No medias loaded"}), 400

    try:
        # Save uploaded file to a unique temp location to avoid race conditions
        import uuid

        suffix = Path(file.filename).suffix or ".bin"
        temp_path = DATA_DIR / f"temp_example_{uuid.uuid4().hex}{suffix}"
        DATA_DIR.mkdir(exist_ok=True)
        file.save(temp_path)

        try:
            results, thresh = _example_sort_from_path(temp_path)
        finally:
            # Clean up temp file even if sorting raises
            temp_path.unlink(missing_ok=True)

        return jsonify({"results": results, "threshold": thresh})

    except Exception:
        import logging

        logging.getLogger(__name__).exception("example-sort failed")
        return jsonify({"error": "Example sort failed"}), 500


@sorting_bp.route("/api/label-file-sort", methods=["POST"])
def label_file_sort():
    """Train MLP on external media files from a label file, then sort all medias."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    if not snapshot_medias():
        return jsonify({"error": "No medias loaded"}), 400

    # Determine embedder from loaded dataset
    emb = _get_embedder_for_loaded_data()
    if emb is None:
        return jsonify({"error": "No embedder available for loaded dataset"}), 400

    try:
        # Parse the label file
        text = file.read().decode("utf-8")
        try:
            label_data = json.loads(text)
        except Exception:
            return jsonify({"error": "Invalid label file format"}), 400

        # Extract labels list
        labels = label_data.get("labels", [])
        if not labels:
            return jsonify({"error": "No labels found in file"}), 400

        # Load and embed each labeled media file
        X_list = []
        y_list = []
        loaded_count = 0
        skipped_count = 0
        _file_base = _paths.get_file_access_base_dir()

        for entry in labels:
            label = entry.get("label")
            if label not in ("good", "bad"):
                skipped_count += 1
                continue

            # Try to get media file path
            media_path = entry.get("path") or entry.get("file") or entry.get("filename")
            if not media_path:
                skipped_count += 1
                continue

            media_path = Path(media_path)
            # Ensure the path doesn't escape the allowed directory
            try:
                _paths.validate_server_filepath(str(media_path), base_dir=_file_base)
            except ValueError:
                skipped_count += 1
                continue
            if not media_path.exists():
                skipped_count += 1
                continue

            # Embed the media file using the dataset's embedder
            embedding = emb.embed_media(media_path)
            if embedding is None:
                skipped_count += 1
                continue

            X_list.append(embedding)
            y_list.append(1.0 if label == "good" else 0.0)
            loaded_count += 1

        if loaded_count < 2:
            return (
                jsonify(
                    {"error": f"Need at least 2 valid labeled files (loaded {loaded_count}, skipped {skipped_count})"}
                ),
                400,
            )

        # Check if we have both good and bad examples
        num_good = sum(1 for y in y_list if y == 1.0)
        num_bad = len(y_list) - num_good
        if num_good == 0 or num_bad == 0:
            return (
                jsonify({"error": "Need at least one good and one bad labeled example"}),
                400,
            )

        # Train MLP using the same approach as learned sort
        import torch  # noqa: PLC0415

        X = torch.tensor(np.array(X_list), dtype=torch.float32)
        y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)

        input_dim = X.shape[1]

        # Calculate threshold using k-fold calibration
        threshold = calculate_cross_calibration_threshold(
            X_list,
            y_list,
            input_dim,
            get_inclusion(),
            calibrate_count=get_calibrate_count(),
            calibration_fraction=get_calibration_fraction(),
        )

        # Train final model on all data
        model = train_model(X, y, input_dim, get_inclusion())

        # Score every media in the dataset
        snap = snapshot_medias()
        all_ids = sorted(snap.keys())
        all_embs = np.array([snap[cid]["embedding"] for cid in all_ids])
        X_all = torch.tensor(all_embs, dtype=torch.float32)
        with torch.no_grad():
            scores = torch.sigmoid(model(X_all)).squeeze(1).tolist()

        # Apply safe thresholds blending if enabled
        if get_safe_thresholds():
            threshold = calculate_safe_threshold(threshold, scores, len(y_list))

        # Sort by raw scores (full precision) before rounding for display.
        paired = sorted(zip(all_ids, scores), key=lambda x: x[1], reverse=True)
        results = [{"id": cid, "score": round(s, 4)} for cid, s in paired]

        return jsonify(
            {
                "results": results,
                "threshold": round(threshold, 4),
                "loaded": loaded_count,
                "skipped": skipped_count,
            }
        )

    except Exception:
        import logging

        logging.getLogger(__name__).exception("label-file-sort failed")
        return jsonify({"error": "Label file sort failed"}), 500


@sorting_bp.route("/api/diversity-tree/next", methods=["GET", "POST"])
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
    scores = None
    threshold = None
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
        raw_scores = data.get("scores")
        if isinstance(raw_scores, dict):
            try:
                scores = {int(k): float(v) for k, v in raw_scores.items()}
            except (ValueError, TypeError):
                return jsonify({"error": "Invalid score keys or values"}), 400
        raw_threshold = data.get("threshold")
        if raw_threshold is not None:
            try:
                threshold = float(raw_threshold)
            except (ValueError, TypeError):
                return jsonify({"error": "Invalid threshold value"}), 400

    tree = get_diversity_tree()
    next_id = diversity_tree_next_sample(scores=scores, threshold=threshold)
    level = tree.diversity_level() if tree is not None else 0
    exhausted = tree is not None and next_id is None
    return jsonify({"id": next_id, "diversity_level": level, "exhausted": exhausted})
