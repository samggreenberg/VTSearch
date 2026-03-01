"""Blueprint for sorting and voting routes."""

import json
from pathlib import Path

import numpy as np
from flask import Blueprint, jsonify, request

from vtsearch.config import DATA_DIR
from vtsearch.models import (
    analyze_labeling_progress,
    calculate_cross_calibration_threshold,
    calculate_gmm_threshold,
    calculate_safe_threshold,
    compute_labeling_status,
    embed_audio_file,
    embed_text_query,
    get_clap_model,
    train_and_score,
    train_model,
)
from vtsearch.utils import (
    add_textsort_suggestion,
    apply_label,
    apply_label_with_click_time,
    bad_votes,
    build_media_lookup,
    medias,
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
    label_history,
    resolve_media_ids,
    set_inclusion,
    set_safe_thresholds,
    update_learned_scores,
    update_sort_progress,
)

sorting_bp = Blueprint("sorting", __name__)


@sorting_bp.route("/api/sort/progress")
def sort_progress():
    """Return the current progress of a text sort operation."""
    return jsonify(get_sort_progress())


@sorting_bp.route("/api/sort", methods=["POST"])
def sort_clips():
    """Return medias sorted by cosine similarity to a text query."""
    try:
        data = request.get_json(force=True)
    except Exception:
        update_sort_progress("idle")
        return jsonify({"error": "Invalid request body"}), 400

    if data is None:
        update_sort_progress("idle")
        return jsonify({"error": "Invalid request body"}), 400

    text = data.get("text", "").strip()

    if not text:
        update_sort_progress("idle")
        return jsonify({"error": "text is required"}), 400

    # Determine media type from current medias
    if not medias:
        update_sort_progress("idle")
        return jsonify({"error": "No medias loaded"}), 400

    media_type = next(iter(medias.values())).get("type", "audio")

    # Total steps: 1 (embed) + len(medias) (similarities) + 1 (threshold)
    total_steps = 1 + len(medias) + 1

    # Check if the embedder needs loading (first use of this media type)
    from vtsearch.media import get as media_get

    try:
        mt = media_get(media_type)
        needs_loading = getattr(mt, "_model", None) is None
    except KeyError:
        needs_loading = False

    if needs_loading:
        # Temporarily redirect the media type's progress callback to the
        # sort progress system so the GUI's sort progress bar shows real
        # download / weight-loading percentages instead of an indeterminate
        # animation.
        update_sort_progress("sorting", "Loading embedder…", 0, total_steps)
        original_cb = mt._on_progress
        mt._on_progress = lambda status, msg, cur, tot: update_sort_progress("sorting", msg, cur, tot)
        try:
            mt.load_models()
        finally:
            mt._on_progress = original_cb
        update_sort_progress("sorting", "Embedding text query…", 0, total_steps)
    else:
        update_sort_progress("sorting", "Embedding text query…", 0, total_steps)

    # Embed text query using refactored module
    from vtsearch import settings

    enrich = settings.get_enrich_descriptions()
    text_vec = embed_text_query(text, media_type, enrich=enrich)
    if text_vec is None:
        update_sort_progress("idle")
        return (
            jsonify({"error": f"Could not embed text for media type {media_type}"}),
            500,
        )

    update_sort_progress("sorting", "Computing similarities…", 1, total_steps)

    # Vectorized cosine similarity: batch all embeddings into a matrix
    all_ids = list(medias.keys())
    all_embs = np.array([medias[cid]["embedding"] for cid in all_ids])
    text_norm = np.linalg.norm(text_vec)
    emb_norms = np.linalg.norm(all_embs, axis=1)
    norm_products = emb_norms * text_norm
    # Avoid division by zero
    safe_norms = np.where(norm_products == 0, 1.0, norm_products)
    similarities = np.dot(all_embs, text_vec) / safe_norms
    similarities = np.where(norm_products == 0, 0.0, similarities)

    results = [{"id": cid, "similarity": round(float(sim), 4)} for cid, sim in zip(all_ids, similarities)]
    scores = similarities.tolist()
    update_sort_progress("sorting", "Computing similarities…", 1 + len(medias), total_steps)

    # Calculate GMM-based threshold
    update_sort_progress("sorting", "Calculating threshold…", total_steps - 1, total_steps)
    threshold = calculate_gmm_threshold(scores)

    results.sort(key=lambda x: x["similarity"], reverse=True)
    update_sort_progress("idle")
    return jsonify({"results": results, "threshold": round(threshold, 4)})


@sorting_bp.route("/api/learned-sort", methods=["POST"])
def learned_sort():
    """Train MLP on voted medias, return all medias sorted by predicted score."""
    if not good_votes or not bad_votes:
        return jsonify({"error": "need at least one good and one bad vote"}), 400
    results, threshold = train_and_score(
        medias,
        good_votes,
        bad_votes,
        get_inclusion(),
        safe_thresholds=get_safe_thresholds(),
        calibrate_count=get_calibrate_count(),
        calibration_fraction=get_calibration_fraction(),
    )
    # Store scores so the /api/votes endpoint can provide confidence info.
    update_learned_scores({r["id"]: r["score"] for r in results})
    return jsonify({"results": results, "threshold": round(threshold, 4)})


@sorting_bp.route("/api/votes")
def get_votes():
    click_times = get_vote_click_times()
    learned_scores = get_learned_scores()
    return jsonify(
        {
            "good": list(good_votes),  # Maintains insertion order (dict keys)
            "bad": list(bad_votes),  # Maintains insertion order (dict keys)
            "click_times": {str(k): v for k, v in click_times.items()},
            "learned_scores": {str(k): round(v, 4) for k, v in learned_scores.items()},
        }
    )


@sorting_bp.route("/api/textsort-suggestions")
def get_textsort_suggestions_route():
    """Return stored text-sort suggestions (most recent last)."""
    return jsonify({"suggestions": get_textsort_suggestions()})


@sorting_bp.route("/api/textsort-suggestions", methods=["POST"])
def add_textsort_suggestion_route():
    """Store a text-sort query as a suggested name for detectors/labelsets."""
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid request body"}), 400
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400
    add_textsort_suggestion(text)
    return jsonify({"ok": True})


@sorting_bp.route("/api/labels/export")
def export_labels():
    """Export labels as a :class:`~vtsearch.datasets.labelset.LabelSet`.

    Each label entry includes the element's ``origin`` and ``origin_name``
    so that consumers know exactly where each labeled element came from.
    The format is a superset of the legacy export format — old consumers
    that only read ``md5`` and ``label`` keys continue to work unchanged.
    """
    from vtsearch.datasets.labelset import LabelSet

    labelset = LabelSet.from_clips_and_votes(medias, good_votes, bad_votes)
    result: dict = labelset.to_dict()
    return jsonify(result)


@sorting_bp.route("/api/labels/import", methods=["POST"])
def import_labels():
    """Import labels from JSON, matching medias by origin+origin_name (MD5 fallback)."""
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    labels = data.get("labels")
    if not isinstance(labels, list):
        return jsonify({"error": "labels must be a list"}), 400

    origin_lookup, md5_lookup = build_media_lookup(medias)

    applied = 0
    skipped = 0
    for entry in labels:
        label = entry.get("label")
        if label not in ("good", "bad"):
            skipped += 1
            continue
        cids = resolve_media_ids(entry, origin_lookup, md5_lookup)
        if not cids:
            skipped += 1
            continue

        for cid in cids:
            apply_label(cid, label)
        applied += 1

    return jsonify({"applied": applied, "skipped": skipped})


@sorting_bp.route("/api/labels/fill-from-sort", methods=["POST"])
def fill_labels_from_sort():
    """Fill labels from the current sort results.

    Assigns Good/Bad labels to currently-unlabeled medias based on their
    position relative to the sort threshold.

    Request body (JSON)::

        {
            "sort_results": [{"id": 1, "score": 0.8}, ...],
            "threshold": 0.5,
            "sides": "good" | "bad" | "both",
            "confirm": false
        }

    When ``confirm`` is false (dry run), returns the counts of unlabeled
    medias that would be labeled.  When ``confirm`` is true, applies the
    labels and returns the resulting data as a results dict suitable for
    any exporter.
    """
    data = request.get_json(force=True, silent=True) or {}

    sort_results = data.get("sort_results")
    if not isinstance(sort_results, list):
        return jsonify({"error": "sort_results must be a list"}), 400

    thresh = data.get("threshold")
    if thresh is None:
        return jsonify({"error": "threshold is required"}), 400
    try:
        thresh = float(thresh)
    except (TypeError, ValueError):
        return jsonify({"error": "threshold must be a number"}), 400

    sides = data.get("sides", "good")
    if sides not in ("good", "bad", "both"):
        return jsonify({"error": "sides must be 'good', 'bad', or 'both'"}), 400

    confirm = bool(data.get("confirm", False))

    # Find unlabeled medias above/below threshold
    good_candidates = []
    bad_candidates = []
    for entry in sort_results:
        cid = entry.get("id")
        score = entry.get("score", entry.get("similarity"))
        if cid is None or score is None:
            continue
        if cid in good_votes or cid in bad_votes:
            continue
        if cid not in medias:
            continue
        if score >= thresh:
            good_candidates.append({"id": cid, "score": float(score)})
        else:
            bad_candidates.append({"id": cid, "score": float(score)})

    if sides == "good":
        bad_candidates = []
    elif sides == "bad":
        good_candidates = []
    # "both" keeps both lists

    if not confirm:
        return jsonify({
            "good_count": len(good_candidates),
            "bad_count": len(bad_candidates),
        })

    # Apply labels
    for entry in good_candidates:
        apply_label_with_click_time(entry["id"], "good")

    for entry in bad_candidates:
        apply_label_with_click_time(entry["id"], "bad")

    # Build a results dict compatible with exporters
    good_hits = []
    for entry in good_candidates:
        cid = entry["id"]
        media = medias.get(cid, {})
        hit = {
            "id": cid,
            "filename": media.get("filename", f"media_{cid}"),
            "category": media.get("category", "unknown"),
            "score": round(entry["score"], 4),
            "label": "good",
        }
        if media.get("origin") is not None:
            hit["origin"] = media["origin"]
        if media.get("origin_name"):
            hit["origin_name"] = media["origin_name"]
        if media.get("md5"):
            hit["md5"] = media["md5"]
        good_hits.append(hit)

    bad_hits = []
    for entry in bad_candidates:
        cid = entry["id"]
        media = medias.get(cid, {})
        hit = {
            "id": cid,
            "filename": media.get("filename", f"media_{cid}"),
            "category": media.get("category", "unknown"),
            "score": round(entry["score"], 4),
            "label": "bad",
        }
        if media.get("origin") is not None:
            hit["origin"] = media["origin"]
        if media.get("origin_name"):
            hit["origin_name"] = media["origin_name"]
        if media.get("md5"):
            hit["md5"] = media["md5"]
        bad_hits.append(hit)

    media_type = "unknown"
    for media in medias.values():
        media_type = media.get("type", "unknown")
        break

    results_dict = {
        "media_type": media_type,
        "detectors_run": 1,
        "results": {
            "fill_from_sort": {
                "detector_name": "fill_from_sort",
                "threshold": round(thresh, 4),
                "total_hits": len(good_hits),
                "hits": good_hits,
                "negative_hits": bad_hits,
            },
        },
    }

    return jsonify({
        "good_applied": len(good_candidates),
        "bad_applied": len(bad_candidates),
        "results": results_dict,
    })


@sorting_bp.route("/api/inclusion")
def get_inclusion_route():
    """Get the current Inclusion setting."""
    return jsonify({"inclusion": get_inclusion()})


@sorting_bp.route("/api/inclusion", methods=["POST"])
def set_inclusion_route():
    """Set the Inclusion setting."""
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    new_inclusion = data.get("inclusion")

    if not isinstance(new_inclusion, (int, float)):
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
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    value = data.get("safe_thresholds")
    if not isinstance(value, bool):
        return jsonify({"error": "safe_thresholds must be a boolean"}), 400

    set_safe_thresholds(value)
    return jsonify({"safe_thresholds": get_safe_thresholds()})


def _example_sort_from_path(file_path: Path) -> tuple:
    """Embed a media file and sort all loaded medias by cosine similarity.

    Returns ``(results_list, threshold)`` on success or raises on error.
    The file at *file_path* is embedded using the media type of the currently
    loaded dataset.
    """
    if not medias:
        raise ValueError("No medias loaded")

    media_type = next(iter(medias.values())).get("type", "audio")
    from vtsearch.media import get as media_get

    mt = media_get(media_type)
    example_embedding = mt.embed_media(file_path)

    if example_embedding is None:
        raise ValueError("Failed to embed media file")

    # Vectorized cosine similarity with all medias
    all_ids = list(medias.keys())
    all_embs = np.array([medias[cid]["embedding"] for cid in all_ids])
    example_norm = np.linalg.norm(example_embedding)
    emb_norms = np.linalg.norm(all_embs, axis=1)
    norm_products = emb_norms * example_norm
    safe_norms = np.where(norm_products == 0, 1.0, norm_products)
    similarities = np.dot(all_embs, example_embedding) / safe_norms
    similarities = np.where(norm_products == 0, 0.0, similarities)

    results = [{"id": cid, "similarity": round(float(sim), 4)} for cid, sim in zip(all_ids, similarities)]
    scores = similarities.tolist()

    # Calculate GMM-based threshold
    threshold = calculate_gmm_threshold(scores)
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results, round(threshold, 4)


@sorting_bp.route("/api/example-sort", methods=["POST"])
def example_sort():
    """Sort medias by similarity to an uploaded example media file."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    if not medias:
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
            temp_path.unlink(missing_ok=True)

        return jsonify({"results": results, "threshold": thresh})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


#: Default directory for server-side example media files.
SERVER_MEDIA_DIR = DATA_DIR / "example_media"


@sorting_bp.route("/api/server-media-files", methods=["GET"])
def list_server_media_files():
    """List media files saved on the server in data/example_media/."""
    if not SERVER_MEDIA_DIR.is_dir():
        return jsonify({"files": []})

    files = []
    for p in sorted(SERVER_MEDIA_DIR.iterdir()):
        if p.is_file() and not p.name.startswith("."):
            files.append({
                "name": p.stem,
                "filename": p.name,
                "path": str(p.resolve()),
                "size_bytes": p.stat().st_size,
            })
    return jsonify({"files": files})


@sorting_bp.route("/api/example-sort-server", methods=["POST"])
def example_sort_server():
    """Sort medias by similarity to a server-side media file."""
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    filename = data.get("filename", "").strip()
    if not filename:
        return jsonify({"error": "filename is required"}), 400

    file_path = SERVER_MEDIA_DIR / filename
    if not file_path.is_file():
        return jsonify({"error": f"File not found: {filename}"}), 404

    # Ensure path doesn't escape the server media directory
    try:
        file_path.resolve().relative_to(SERVER_MEDIA_DIR.resolve())
    except ValueError:
        return jsonify({"error": "Invalid filename"}), 400

    try:
        results, thresh = _example_sort_from_path(file_path)
        return jsonify({"results": results, "threshold": thresh})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@sorting_bp.route("/api/label-file-sort", methods=["POST"])
def label_file_sort():
    """Train MLP on external audio files from a label file, then sort all medias."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    clap_model, clap_processor = get_clap_model()
    if clap_model is None or clap_processor is None:
        return jsonify({"error": "CLAP model not loaded"}), 500

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

        # Load and embed each labeled audio file
        X_list = []
        y_list = []
        loaded_count = 0
        skipped_count = 0

        for entry in labels:
            label = entry.get("label")
            if label not in ("good", "bad"):
                skipped_count += 1
                continue

            # Try to get audio file path
            audio_path = entry.get("path") or entry.get("file") or entry.get("filename")
            if not audio_path:
                skipped_count += 1
                continue

            audio_path = Path(audio_path)
            if not audio_path.exists():
                skipped_count += 1
                continue

            # Embed the audio file
            embedding = embed_audio_file(audio_path)
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
        all_ids = sorted(medias.keys())
        all_embs = np.array([medias[cid]["embedding"] for cid in all_ids])
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

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@sorting_bp.route("/api/labeling-progress", methods=["POST"])
def labeling_progress():
    """Analyze labeling progress and calculate stopping condition metrics."""
    if not good_votes or not bad_votes:
        return jsonify({"error": "need at least one good and one bad vote"}), 400

    if not label_history:
        return jsonify({"error": "no label history available"}), 400

    try:
        analysis = analyze_labeling_progress(medias, label_history, good_votes, bad_votes, get_inclusion())
        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@sorting_bp.route("/api/labeling-status", methods=["GET"])
def labeling_status_indicator():
    """Return per-metric red/yellow/green labeling statuses.

    Returns ``smart``, ``stable``, and ``span`` sub-objects, each with a
    ``status`` field of ``"red"``, ``"yellow"``, or ``"green"``.
    """
    try:
        tree = get_diversity_tree()
        span = tree.span_info() if tree is not None else None
        status = compute_labeling_status(medias, label_history, good_votes, bad_votes, get_inclusion(), span_info=span)
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@sorting_bp.route("/api/diversity-tree/next", methods=["GET", "POST"])
def diversity_tree_next():
    """Return the next diverse sample from the Diversity Tree.

    Accepts an optional POST body with ``{"scores": {id: score, ...}}``
    so the sort mode influences which element is picked from the next
    unseen node.  Without scores the first element in the node is returned.

    Returns ``{"id": <media_id>}`` or ``{"id": null}`` when the tree is
    exhausted or not yet built.  Also includes ``diversity_level`` so the
    frontend can display how many tree levels have been fully covered,
    and ``exhausted`` (bool) which is true when the tree exists but every
    node has already been seen.
    """
    scores = None
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
        raw_scores = data.get("scores")
        if isinstance(raw_scores, dict):
            scores = {int(k): float(v) for k, v in raw_scores.items()}

    tree = get_diversity_tree()
    next_id = diversity_tree_next_sample(scores=scores)
    level = tree.diversity_level() if tree is not None else -1
    exhausted = tree is not None and next_id is None
    return jsonify({"id": next_id, "diversity_level": level, "exhausted": exhausted})
