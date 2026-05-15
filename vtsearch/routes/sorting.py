"""Blueprint for sorting and voting routes."""

import json
import logging
import threading
from pathlib import Path

from flask import Blueprint, jsonify, request

from vtsearch.routes.helpers import (
    format_exception_detail,
    get_embedder_for_medias,
    get_json_or_400,
    get_json_safe,
)

from vtsearch.config import DATA_DIR
import vtsearch.utils.paths as _paths
from vtsearch.models import (
    calculate_gmm_threshold,
    embed_text_query,
    inject_live_model,
    train_and_score,
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
    vote_region_boxes,
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

    For datasets embedded with a patch-aware embedder (DINOv2, DINOv3,
    EUPE), each result also carries a ``best_region`` field — the
    bounding box of the region that scored highest, in normalised
    image coordinates ``[x0, y0, x1, y1]``.  Single-vector embedders
    take a fast vectorised numpy path with no per-result box.

    Both paths live in :mod:`vtsearch.models.region_similarity`.
    """
    from vtsearch.models.region_similarity import cosine_sort_with_boxes  # noqa: PLC0415

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

    # Reject the request up-front when the active embedder is vision-only
    # (e.g. DINOv3, Perception Encoder). Without this short-circuit we'd
    # waste time loading the model into RAM just to discover it can't embed
    # text — and the frontend wouldn't get a clean ``supports_text=False``
    # signal to hide its text-search UI.
    if embedder_name:
        try:
            from vtsearch.media import get_embedder  # noqa: PLC0415

            _active_emb = get_embedder(embedder_name)
        except KeyError:
            _active_emb = None
        if _active_emb is not None and not _active_emb.supports_text:
            update_sort_progress("idle")
            return (
                jsonify(
                    {
                        "error": (
                            f"Embedder '{_active_emb.name}' does not support text queries. "
                            "Use learned sort or load a saved sort instead."
                        ),
                        "supports_text": False,
                    }
                ),
                400,
            )

    try:
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
    except Exception as exc:
        logging.getLogger(__name__).exception("text sort failed")
        update_sort_progress("idle")
        return jsonify({"error": f"Text sort failed: {format_exception_detail(exc)}"}), 500


def _learned_sort_done_payload(job) -> dict:
    """Build the JSON body returned when a learned-sort job is finished."""
    result = job.result or {}
    return {
        "job_id": job.job_id,
        "status": "done",
        "results": result.get("results", []),
        "threshold": result.get("threshold", 0.0),
    }


@sorting_bp.route("/api/learned-sort", methods=["POST"])
def learned_sort():
    """Kick off (or short-circuit) a learned-sort training job.

    Training is GIL-bound and ran inline used to stall every other request
    served by the small ``gthread`` pool — votes polls, thumbnails, even
    media bytes.  The endpoint now hands the work off to a background daemon
    thread and returns immediately with a ``job_id``; clients poll
    :func:`learned_sort_result` until ``status == "done"``.

    A small signature cache short-circuits the no-op case: when the votes,
    detector, inclusion and thresholding settings are unchanged from the
    most recent successful run, the previous result is returned directly.

    Tests can pass ``{"wait": true}`` in the body to block until the job
    completes and receive the result inline.  The frontend leaves it false.
    """
    from vtsearch.datasets.labelset import LabelSet
    from vtsearch.models.detector_registry import get_detector
    from vtsearch.models.detector_store import _detector_path, _read_detector
    from vtsearch.models.labelset_training import labelset_train_and_score
    from vtsearch.utils.async_jobs import learned_sort_jobs
    from vtsearch.utils.state_core import (
        _empty_detector_context,
        get_active_context,
        get_active_detector_context,
        set_thread_dataset_context,
        set_thread_detector_context,
    )

    body = request.get_json(silent=True) or {}
    wait = bool(body.get("wait"))

    snap = snapshot_medias()

    # Resolve the active detector's labelset (if any).  Prefer the parsed
    # labelset cached on det_ctx by ``ensure_votes_match_active_dataset`` to
    # avoid re-reading the JSON file on every click.
    det_ctx = get_active_detector_context()
    ds_ctx = get_active_context()
    labelset: LabelSet | None = None
    det_media_type = ""
    if det_ctx is not _empty_detector_context and det_ctx.detector_id:
        if det_ctx.cached_labelset is not None:
            labelset = det_ctx.cached_labelset
            det_media_type = det_ctx.cached_labelset_media_type
        else:
            entry = get_detector(det_ctx.detector_id)
            if entry and entry.get("name"):
                det_data = _read_detector(_detector_path(entry["name"]))
                if det_data:
                    labelset = LabelSet.from_dict(det_data.get("labelset") or {})
                    det_media_type = det_data.get("media_type", "") or ""

    # Early validation so 400s come back before we spin a thread.
    if labelset is not None:
        good_count = sum(1 for el in labelset.elements if el.label == "good")
        bad_count = sum(1 for el in labelset.elements if el.label == "bad")
        if good_count == 0 or bad_count == 0:
            return jsonify({"error": "need at least one good and one bad vote"}), 400
    else:
        if not good_votes or not bad_votes:
            return jsonify({"error": "need at least one good and one bad vote"}), 400

    inclusion_value = get_inclusion()
    safe_thresholds_value = get_safe_thresholds()
    calibrate_count_value = get_calibrate_count()
    calibration_fraction_value = get_calibration_fraction()
    region_boxes_snapshot = dict(vote_region_boxes)

    # Signature: covers everything that changes the training output.  Re-sorts
    # with the same vote set + settings reuse the cached result.
    if labelset is not None:
        labels_sig = tuple(
            sorted((el.label, _stable_element_id_for_sig(el)) for el in labelset.elements)
        )
    else:
        labels_sig = (
            ("good", tuple(sorted(good_votes))),
            ("bad", tuple(sorted(bad_votes))),
            ("regions", tuple(sorted(region_boxes_snapshot.items()))),
        )
    signature = (
        det_ctx.detector_id,
        ds_ctx.dataset_id,
        tuple(sorted(snap.keys())),
        labels_sig,
        inclusion_value,
        safe_thresholds_value,
        calibrate_count_value,
        calibration_fraction_value,
    )

    cached = learned_sort_jobs.cached_for(signature)
    if cached is not None:
        return jsonify(_learned_sort_done_payload(cached))

    def _run(job):
        # Background thread needs the same dataset/detector context as the
        # request thread so proxies (good_votes, bad_votes, etc.) resolve
        # correctly when the training helpers reach for them.
        set_thread_dataset_context(ds_ctx)
        set_thread_detector_context(det_ctx)
        try:
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

            # Store scores so the /api/votes endpoint can provide confidence info.
            update_learned_scores({r["id"]: r["score"] for r in results})

            # In the labelset path, training used the on-disk labelset
            # (potentially cross-dataset) — derive the live-cache key and the
            # cached training snapshot from the labelset itself, not from
            # local cid-keyed votes.
            labelset_local_good: set[int] | None = None
            labelset_local_bad: set[int] | None = None
            labelset_training_medias: dict[int, dict] | None = None
            labelset_has_cross_dataset = False
            if labelset is not None:
                from vtsearch.utils import build_media_lookup, resolve_media_ids

                origin_lookup, md5_lookup, name_lookup = build_media_lookup(snap)
                labelset_local_good = set()
                labelset_local_bad = set()
                labelset_training_medias = {}
                for el in labelset.elements:
                    if el.label not in ("good", "bad"):
                        continue
                    cids = resolve_media_ids(
                        el.to_dict(), origin_lookup, md5_lookup, name_lookup
                    )
                    if not cids:
                        labelset_has_cross_dataset = True
                        continue
                    target = labelset_local_good if el.label == "good" else labelset_local_bad
                    for cid in cids:
                        target.add(cid)
                        if cid in snap:
                            labelset_training_medias[cid] = snap[cid]

            # Inject the live model into the progress cache so indicators and
            # the progress line-graph use the actual model that guided
            # sorting, rather than retraining an independent model from
            # scratch.  The cache is keyed by current-dataset cids, so only
            # inject when the model's training set is fully representable in
            # that key — otherwise the cache would return a cross-dataset
            # model when local-cids replay asks for a local-only one.
            if model is not None:
                if labelset is None:
                    inject_live_model(good_votes, bad_votes, model, threshold)
                elif (
                    not labelset_has_cross_dataset
                    and labelset_local_good == set(good_votes)
                    and labelset_local_bad == set(bad_votes)
                ):
                    inject_live_model(good_votes, bad_votes, model, threshold)

            if det_ctx is not _empty_detector_context and model is not None:
                det_ctx.model = model
                det_ctx.threshold = threshold
                # Cache the voted media items with embeddings for cross-embedder scenarios.
                if labelset is not None:
                    det_ctx.training_medias = labelset_training_medias or {}
                else:
                    training = {}
                    for cid in list(good_votes) + list(bad_votes):
                        if cid in snap:
                            training[cid] = snap[cid]
                    det_ctx.training_medias = training
                if snap:
                    first = next(iter(snap.values()), {})
                    det_ctx.embedder = first.get("embedder", "")
                    det_ctx.media_type = first.get("type", "")

            job.result = {"results": results, "threshold": round(threshold, 4)}
        finally:
            set_thread_dataset_context(None)
            set_thread_detector_context(None)

    job = learned_sort_jobs.start(signature, _run)

    if wait:
        job.done_event.wait(timeout=120)
        if job.status == "error":
            return jsonify({"error": job.error or "learned-sort failed"}), 500
        if job.status == "done":
            return jsonify(_learned_sort_done_payload(job))

    return jsonify({"job_id": job.job_id, "status": "running", "current": 0, "total": 1})


def _stable_element_id_for_sig(el) -> str:
    """Return a stable identifier for a labelset element for signature use."""
    from vtsearch.models.labelset_elements import stable_element_id

    return stable_element_id(el)


@sorting_bp.route("/api/learned-sort/result", methods=["GET"])
def learned_sort_result():
    """Poll a learned-sort background job.

    Returns the same shape as the POST endpoint's ``done`` response when the
    job has finished, or a ``running`` snapshot otherwise.
    """
    from vtsearch.utils.async_jobs import learned_sort_jobs

    job_id = request.args.get("job_id", "").strip()
    if not job_id:
        return jsonify({"error": "job_id is required"}), 400

    job = learned_sort_jobs.get(job_id)
    if job is None:
        return jsonify({"status": "missing", "error": "Job not found"}), 404

    if job.status == "running":
        return jsonify({
            "job_id": job.job_id,
            "status": "running",
            "current": job.current,
            "total": job.total,
        })
    if job.status == "error":
        return jsonify({
            "job_id": job.job_id,
            "status": "error",
            "error": job.error or "learned-sort failed",
        }), 500
    if job.status == "cancelled":
        return jsonify({"job_id": job.job_id, "status": "cancelled"})
    return jsonify(_learned_sort_done_payload(job))


@sorting_bp.route("/api/votes")
def get_votes():
    from vtsearch.utils.state_core import _empty_detector_context, get_active_detector_context

    click_times = get_vote_click_times()
    learned_scores = get_learned_scores()
    det_ctx = get_active_detector_context()
    if det_ctx is not _empty_detector_context and det_ctx.detector_id:
        labelset_good_count = det_ctx.labelset_good_count
        labelset_bad_count = det_ctx.labelset_bad_count
    else:
        labelset_good_count = len(good_votes)
        labelset_bad_count = len(bad_votes)
    return jsonify(
        {
            "good": sorted(good_votes),
            "bad": sorted(bad_votes),
            "click_times": {str(k): v for k, v in click_times.items()},
            "learned_scores": {str(k): round(v, 4) for k, v in learned_scores.items()},
            "labelset_good_count": labelset_good_count,
            "labelset_bad_count": labelset_bad_count,
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
        from vtsearch.models.label_sync import sync_labels_to_loaded_detector

        sync_labels_to_loaded_detector()

        from vtsearch.labels.sync import sync_to_labelset_source

        sync_to_labelset_source()

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
    from vtsearch.media.embedder import media_from_path  # noqa: PLC0415

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
    media_type = first_media.get("type", "")

    from vtsearch.media.cropping import crop_file_bytes

    cropped = crop_file_bytes(temp_path, media_type, crop_params)
    temp_path.write_bytes(cropped)
    return temp_path


@sorting_bp.route("/api/example-sort", methods=["POST"])
def example_sort():
    """Sort medias by similarity to an uploaded example media file.

    Optional ``crop_params`` form field carries a JSON object with the
    bounds for a user-cropped sub-region (e.g. ``{"start": 1.5, "end": 3}``
    for audio or ``{"box": [x1, y1, x2, y2]}`` for images).  When present
    the file is cropped server-side before embedding.
    """
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
            crop_params = _parse_crop_params(request.form.get("crop_params"))
            _apply_crop_or_keep(temp_path, crop_params)
            results, thresh = _example_sort_from_path(temp_path)
        finally:
            # Clean up temp file even if sorting raises
            temp_path.unlink(missing_ok=True)

        return jsonify({"results": results, "threshold": thresh})

    except Exception as exc:
        import logging

        logging.getLogger(__name__).exception("example-sort failed")
        return jsonify({"error": f"Example sort failed: {format_exception_detail(exc)}"}), 500


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
            from vtsearch.media.embedder import media_from_path  # noqa: PLC0415

            embedding = emb.embed_media(media_from_path(media_path))
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
        from vtsearch.models.detector_training import validate_good_bad_split

        try:
            validate_good_bad_split(y_list)
        except ValueError:
            return (
                jsonify({"error": "Need at least one good and one bad labeled example"}),
                400,
            )

        # Train MLP and compute threshold using the shared pipeline
        import torch  # noqa: PLC0415

        from vtsearch.models.detector_training import train_and_threshold

        snap = snapshot_medias()
        model, threshold = train_and_threshold(X_list, y_list, snap=snap)

        # Score every media in the dataset
        from vtsearch.models.embedding_matrix import get_embedding_matrix_for_snap  # noqa: PLC0415

        all_ids, all_embs = get_embedding_matrix_for_snap(snap)
        X_all = torch.from_numpy(all_embs)
        with torch.no_grad():
            X_all = X_all.to(next(model.parameters()).device)
            scores = torch.sigmoid(model(X_all)).squeeze(1).cpu().tolist()

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
        data = get_json_safe()
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
