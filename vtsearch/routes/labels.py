"""Blueprint for label management routes (export, import, fill-from-sort)."""

from flask import Blueprint, jsonify, request

from vtsearch.routes.helpers import get_json_or_400
from vtsearch.utils import (
    apply_label,
    apply_label_with_click_time,
    bad_votes,
    build_media_hit,
    build_media_lookup,
    get_media,
    good_votes,
    resolve_media_ids,
    snapshot_medias,
)

labels_bp = Blueprint("labels", __name__)


@labels_bp.route("/api/labels/export")
def export_labels():
    """Export labels as a :class:`~vtsearch.datasets.labelset.LabelSet`.

    Each label entry includes the element's ``origin`` and ``origin_name``
    so that consumers know exactly where each labeled element came from.
    The format is a superset of the legacy export format — old consumers
    that only read ``md5`` and ``label`` keys continue to work unchanged.

    Query params:
        goods_only: If ``"1"`` or ``"true"``, only export good labels.
        label_filter: ``"good"``, ``"bad"``, ``"both"`` (default), or
            ``"corrections"`` (only items where the user changed the
            detector's original label).  Overrides ``goods_only`` when
            present.
        enrich: If ``"1"`` or ``"true"``, include per-item
            ``custom_metadata`` and a top-level ``available_columns`` list.
    """
    from vtsearch.datasets.labelset import LabelSet
    from vtsearch.utils import get_find_initial_labels

    label_filter = request.args.get("label_filter", "").lower()
    corrections_only = label_filter == "corrections"

    if label_filter == "good":
        goods, bads = good_votes, {}
    elif label_filter == "bad":
        goods, bads = {}, bad_votes
    elif label_filter and not corrections_only:
        goods, bads = good_votes, bad_votes
    else:
        if not corrections_only:
            # Backward compat: fall back to goods_only flag
            goods_only = request.args.get("goods_only", "").lower() in ("1", "true")
            goods = good_votes
            bads = {} if goods_only else bad_votes
        else:
            goods, bads = good_votes, bad_votes

    all_medias = snapshot_medias()
    labelset = LabelSet.from_clips_and_votes(all_medias, goods, bads)
    result: dict = labelset.to_dict()

    # Annotate corrections: items where the user changed the detector's label.
    find_initial = get_find_initial_labels()
    if find_initial:
        for entry in result["labels"]:
            # Match by media ID — look up from medias by MD5
            md5 = entry.get("md5")
            media_id = None
            for mid, m in all_medias.items():
                if m.get("md5") == md5:
                    media_id = mid
                    break
            if media_id is not None and media_id in find_initial:
                original = find_initial[media_id]
                current = entry.get("label")
                entry["is_correction"] = current != original
            else:
                entry["is_correction"] = False

    if corrections_only:
        result["labels"] = [e for e in result["labels"] if e.get("is_correction")]

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

        base_columns = ["label", "md5", "origin_name", "filename", "category", "origin"]
        base_lower = {c.lower() for c in base_columns}
        extra_keys = sorted(k for k in all_meta_keys if k.lower() not in base_lower)
        result["available_columns"] = base_columns + extra_keys

    return jsonify(result)


@labels_bp.route("/api/labels/import", methods=["POST"])
def import_labels():
    """Import labels from JSON, matching medias by origin+origin_name (MD5 fallback)."""
    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    labels = data.get("labels")
    if not isinstance(labels, list):
        return jsonify({"error": "labels must be a list"}), 400

    origin_lookup, md5_lookup, _ = build_media_lookup(snapshot_medias())

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


@labels_bp.route("/api/labels/fill-from-sort", methods=["POST"])
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
