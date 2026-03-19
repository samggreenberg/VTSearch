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
    analyze_labeling_progress,
    calculate_cross_calibration_threshold,
    calculate_diversity_level_over_time,
    calculate_error_cost_over_time,
    calculate_gmm_threshold,
    calculate_prediction_stability_over_time,
    calculate_safe_threshold,
    compute_labeling_status,
    embed_text_query,
    inject_live_model,
    train_and_score,
    train_model,
)
from vtsearch.utils import (
    add_textsort_suggestion,
    apply_label,
    apply_label_with_click_time,
    bad_votes,
    build_media_hit,
    build_media_lookup,
    diversity_tree_next_sample,
    get_calibrate_count,
    get_calibration_fraction,
    get_diversity_tree,
    get_eval_progress,
    get_inclusion,
    get_learned_scores,
    get_media,
    get_safe_thresholds,
    get_sort_progress,
    get_textsort_suggestions,
    get_vote_click_times,
    good_votes,
    label_history,
    resolve_media_ids,
    set_inclusion,
    set_safe_thresholds,
    snapshot_medias,
    update_eval_progress,
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
    results, threshold, model = train_and_score(
        snapshot_medias(),
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
    from vtsearch.routes.trainable_models import _seed_good_votes_from_examples

    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    examples = data.get("examples")
    if not isinstance(examples, list):
        return jsonify({"error": "examples must be a list"}), 400

    seeded = _seed_good_votes_from_examples(examples)
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


@sorting_bp.route("/api/labels/export")
def export_labels():
    """Export labels as a :class:`~vtsearch.datasets.labelset.LabelSet`.

    Each label entry includes the element's ``origin`` and ``origin_name``
    so that consumers know exactly where each labeled element came from.
    The format is a superset of the legacy export format — old consumers
    that only read ``md5`` and ``label`` keys continue to work unchanged.

    Query params:
        goods_only: If ``"1"`` or ``"true"``, only export good labels.
        label_filter: ``"good"``, ``"bad"``, or ``"both"`` (default).
            Overrides ``goods_only`` when present.
        enrich: If ``"1"`` or ``"true"``, include per-item
            ``custom_metadata`` and a top-level ``available_columns`` list.
    """
    from vtsearch.datasets.labelset import LabelSet

    label_filter = request.args.get("label_filter", "").lower()
    if label_filter == "good":
        goods, bads = good_votes, {}
    elif label_filter == "bad":
        goods, bads = {}, bad_votes
    elif label_filter:
        goods, bads = good_votes, bad_votes
    else:
        # Backward compat: fall back to goods_only flag
        goods_only = request.args.get("goods_only", "").lower() in ("1", "true")
        goods = good_votes
        bads = {} if goods_only else bad_votes

    all_medias = snapshot_medias()
    labelset = LabelSet.from_clips_and_votes(all_medias, goods, bads)
    result: dict = labelset.to_dict()

    enrich = request.args.get("enrich", "").lower() in ("1", "true")
    if enrich:
        from vtsearch.media import get as get_media_type  # noqa: PLC0415

        md5_to_media: dict = {}
        for m in all_medias.values():
            md5_val = m.get("md5")
            if md5_val:
                md5_to_media[md5_val] = m

        all_meta_keys: set[str] = set()
        for entry in result["labels"]:
            media = md5_to_media.get(entry.get("md5"))
            if not media:
                continue
            media_type_id = media.get("type", "audio")
            try:
                mt = get_media_type(media_type_id)
                meta = mt.display_metadata(media)
            except KeyError:
                meta = {}
            importer_custom = media.get("custom_metadata")
            if importer_custom:
                meta.update(importer_custom)
            if meta:
                entry["custom_metadata"] = meta
                all_meta_keys.update(meta.keys())

        base_columns = ["label", "md5", "origin_name", "filename", "category"]
        base_lower = {c.lower() for c in base_columns}
        extra_keys = sorted(k for k in all_meta_keys if k.lower() not in base_lower)
        result["available_columns"] = base_columns + extra_keys

    return jsonify(result)


@sorting_bp.route("/api/labels/import", methods=["POST"])
def import_labels():
    """Import labels from JSON, matching medias by origin+origin_name (MD5 fallback)."""
    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    labels = data.get("labels")
    if not isinstance(labels, list):
        return jsonify({"error": "labels must be a list"}), 400

    origin_lookup, md5_lookup = build_media_lookup(snapshot_medias())

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

    from vtsearch.routes.trainable_models import sync_labels_to_loaded_model

    sync_labels_to_loaded_model()

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
        if not isinstance(score, (int, float)):
            continue
        if cid in good_votes or cid in bad_votes:
            continue
        if get_media(cid) is None:
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
        return jsonify(
            {
                "good_count": len(good_candidates),
                "bad_count": len(bad_candidates),
            }
        )

    # Apply labels
    for entry in good_candidates:
        apply_label_with_click_time(entry["id"], "good")

    for entry in bad_candidates:
        apply_label_with_click_time(entry["id"], "bad")

    # Build a results dict compatible with exporters
    snap = snapshot_medias()
    good_hits = [build_media_hit(e["id"], snap.get(e["id"], {}), e["score"], label="good") for e in good_candidates]
    bad_hits = [build_media_hit(e["id"], snap.get(e["id"], {}), e["score"], label="bad") for e in bad_candidates]

    media_type = "unknown"
    for media in snap.values():
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

    from vtsearch.routes.trainable_models import sync_labels_to_loaded_model

    sync_labels_to_loaded_model()

    return jsonify(
        {
            "good_applied": len(good_candidates),
            "bad_applied": len(bad_candidates),
            "results": results_dict,
        }
    )


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


#: Default directory for server-side example media files.
SERVER_MEDIA_DIR = DATA_DIR / "example_media"


@sorting_bp.route("/api/server-media-files/upload", methods=["POST"])
def upload_server_media_file():
    """Upload a media file to data/example_media/ and return its filename."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    import uuid

    suffix = Path(file.filename).suffix or ".bin"
    safe_name = f"{uuid.uuid4().hex}{suffix}"
    SERVER_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    dest = SERVER_MEDIA_DIR / safe_name
    file.save(dest)

    return jsonify({"filename": safe_name, "original_name": file.filename}), 201


@sorting_bp.route("/api/server-media-files", methods=["GET"])
def list_server_media_files():
    """List media files saved on the server in data/example_media/."""
    if not SERVER_MEDIA_DIR.is_dir():
        return jsonify({"files": []})

    files = []
    for p in sorted(SERVER_MEDIA_DIR.iterdir()):
        if p.is_file() and not p.name.startswith("."):
            files.append(
                {
                    "name": p.stem,
                    "filename": p.name,
                    "size_bytes": p.stat().st_size,
                }
            )
    return jsonify({"files": files})


@sorting_bp.route("/api/example-sort-server", methods=["POST"])
def example_sort_server():
    """Sort medias by similarity to a server-side media file."""
    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    filename = data.get("filename", "").strip()
    if not filename:
        return jsonify({"error": "filename is required"}), 400

    file_path = SERVER_MEDIA_DIR / filename

    # Ensure path doesn't escape the server media directory
    try:
        file_path.resolve().relative_to(SERVER_MEDIA_DIR.resolve())
    except ValueError:
        return jsonify({"error": "Invalid filename"}), 400

    if not file_path.is_file():
        return jsonify({"error": f"File not found: {filename}"}), 404

    try:
        results, thresh = _example_sort_from_path(file_path)
        return jsonify({"results": results, "threshold": thresh})
    except Exception:
        import logging

        logging.getLogger(__name__).exception("example-sort-server failed")
        return jsonify({"error": "Example sort failed"}), 500


@sorting_bp.route("/api/example-sort-origin", methods=["POST"])
def example_sort_origin():
    """Sort medias by similarity to a media file resolved from an origin.

    Accepts a JSON body with ``origin`` (an origin dict as stored on medias)
    and ``key`` (the item key / relative path within the source).  The file
    is fetched via the :class:`~vtsearch.datasets.sources.base.MediaSource`
    abstraction, embedded, and used for cosine-similarity sorting.

    Example request::

        {
            "origin": {"importer": "folder", "params": {"path": "/data/sounds"}},
            "key": "subdir/audio123.wav"
        }
    """
    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    origin = data.get("origin")
    if not isinstance(origin, dict):
        return jsonify({"error": "origin dict is required"}), 400

    key = data.get("key", "").strip()
    if not key:
        return jsonify({"error": "key is required"}), 400

    if not snapshot_medias():
        return jsonify({"error": "No medias loaded"}), 400

    from vtsearch.datasets.sources import get_source_for_origin

    source = get_source_for_origin(origin)
    if source is None:
        return jsonify({"error": f"No media source available for origin type: {origin.get('importer', '')}"}), 400

    try:
        file_path = source.fetch_item(key)
        if file_path is None:
            return jsonify({"error": f"File not found: {key}"}), 404

        results, thresh = _example_sort_from_path(file_path)
        return jsonify({"results": results, "threshold": thresh})
    except Exception:
        import logging

        logging.getLogger(__name__).exception("example-sort-origin failed")
        return jsonify({"error": "Example sort failed"}), 500
    finally:
        source.cleanup()


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


@sorting_bp.route("/api/labeling-progress", methods=["POST"])
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


@sorting_bp.route("/api/labeling-status", methods=["GET"])
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


@sorting_bp.route("/api/indicator-score-history", methods=["GET"])
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


# ---------------------------------------------------------------------------
# Eval routes
# ---------------------------------------------------------------------------


@sorting_bp.route("/api/eval/train-and-score", methods=["POST"])
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


@sorting_bp.route("/api/eval/voting-iterations", methods=["GET"])
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
