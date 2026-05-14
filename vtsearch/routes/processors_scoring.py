"""Extractor and localizer execution routes."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from flask import Blueprint, jsonify

from vtsearch.routes.helpers import get_json_or_400
from vtsearch.routes.processors_crud import _build_extractor, _build_localizer
from vtsearch.utils import (
    get_autorun_extractors_by_media,
    get_autorun_localizers_by_media,
    snapshot_medias,
)
from vtsearch.utils.memory_budget import cap_workers_by_memory

processors_scoring_bp = Blueprint("processors_scoring", __name__)

# Keys excluded from API responses (large binary/vector data).
_HEAVYWEIGHT_KEYS = ("embedding", "media_bytes", "media_string", "thumbnail_bytes")


def _media_info_for_response(media: dict) -> dict:
    """Return a copy of *media* without heavyweight fields."""
    return {k: v for k, v in media.items() if k not in _HEAVYWEIGHT_KEYS}


def _apply_processor_to_medias(processor, snap: dict, method: str, result_key: str) -> list[dict]:
    """Run a processor (extractor or localizer) on all medias, returning hits.

    Args:
        processor: An extractor or localizer instance with *method*.
        snap: A snapshot of the medias dict.
        method: The method name to call (``"extract"`` or ``"localize"``).
        result_key: The key to store results under (``"extractions"`` or ``"localizations"``).
    """
    func = getattr(processor, method)
    results = []
    for media_id in sorted(snap.keys()):
        media = snap[media_id]
        hits = func(media)
        if hits:
            info = _media_info_for_response(media)
            info[result_key] = hits
            results.append(info)
    return results


def _auto_run_processors(
    snap: dict, processors: dict, build_fn, type_key: str, method: str, result_key: str, name_key: str
) -> dict:
    """Run multiple processors in parallel via ThreadPoolExecutor.

    Args:
        snap: A snapshot of the medias dict.
        processors: ``{name: data_dict}`` of registered processors.
        build_fn: Factory ``(name, type_str, config) -> processor``.
        type_key: Key in *data_dict* holding the processor type string.
        method: Method name (``"extract"`` or ``"localize"``).
        result_key: Key for per-media results (``"extractions"`` or ``"localizations"``).
        name_key: Key for the processor name in the result dict.
    """

    def _run_single(proc_name, proc_data):
        try:
            proc = build_fn(proc_name, proc_data[type_key], proc_data["config"])
        except Exception:
            return None

        hits = _apply_processor_to_medias(proc, snap, method, result_key)
        return proc_name, {
            name_key: proc_name,
            "total_medias_with_hits": len(hits),
            "results": hits,
        }

    n_items = len(snap)
    first = next(iter(snap.values()), {}) if snap else {}
    embedding = first.get("embedding")
    try:
        embed_dim = int(len(embedding)) if embedding is not None else 0
    except TypeError:
        embed_dim = 0
    worker_cap = cap_workers_by_memory(
        n_items,
        embed_dim,
        max_workers=min(len(processors), 8),
    )
    results = {}
    with ThreadPoolExecutor(max_workers=worker_cap) as pool:
        futures = [pool.submit(_run_single, name, data) for name, data in processors.items()]
        for future in futures:
            outcome = future.result()
            if outcome is not None:
                name, result = outcome
                results[name] = result
    return results


# ---------------------------------------------------------------------------
# Extractor execution
# ---------------------------------------------------------------------------


@processors_scoring_bp.route("/api/extract", methods=["POST"])
def run_extract():
    """Run a single extractor on all medias and return per-media extraction results."""
    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    extractor_name = data.get("name", "").strip()
    extractor_type = data.get("extractor_type", "").strip()
    config = data.get("config")

    if not extractor_type:
        return jsonify({"error": "extractor_type is required"}), 400
    if not config or not isinstance(config, dict):
        return jsonify({"error": "config is required"}), 400

    snap = snapshot_medias()
    if not snap:
        return jsonify({"error": "No medias loaded"}), 400

    try:
        extractor = _build_extractor(extractor_name or "adhoc", extractor_type, config)
    except Exception as e:
        return jsonify({"error": f"Invalid extractor config: {e}"}), 400

    media_type = next(iter(snap.values())).get("type", "")
    if extractor.media_type != media_type:
        return (
            jsonify({"error": f"Extractor media type '{extractor.media_type}' does not match medias '{media_type}'"}),
            400,
        )

    results = _apply_processor_to_medias(extractor, snap, "extract", "extractions")

    return jsonify(
        {
            "extractor_name": extractor.name,
            "media_type": media_type,
            "total_medias_with_hits": len(results),
            "results": results,
        }
    )


@processors_scoring_bp.route("/api/auto-extract", methods=["POST"])
def auto_extract():
    """Run all autorun extractors for the current media type and return extraction results."""
    snap = snapshot_medias()
    if not snap:
        return jsonify({"error": "No medias loaded"}), 400

    media_type = next(iter(snap.values())).get("type", "")
    extractors = get_autorun_extractors_by_media(media_type)

    if not extractors:
        return jsonify({"error": f"No autorun extractors found for media type: {media_type}"}), 400

    results = _auto_run_processors(
        snap, extractors, _build_extractor, "extractor_type", "extract", "extractions", "extractor_name"
    )

    return jsonify(
        {
            "media_type": media_type,
            "extractors_run": len(results),
            "results": results,
        }
    )


# ---------------------------------------------------------------------------
# Localizer execution
# ---------------------------------------------------------------------------


@processors_scoring_bp.route("/api/localize", methods=["POST"])
def run_localize():
    """Run a single localizer on all clips and return per-clip localization results."""
    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    localizer_name = data.get("name", "").strip()
    localizer_type = data.get("localizer_type", "").strip()
    config = data.get("config")

    if not localizer_type:
        return jsonify({"error": "localizer_type is required"}), 400
    if not config or not isinstance(config, dict):
        return jsonify({"error": "config is required"}), 400

    snap = snapshot_medias()
    if not snap:
        return jsonify({"error": "No medias loaded"}), 400

    try:
        localizer = _build_localizer(localizer_name or "adhoc", localizer_type, config)
    except Exception as e:
        return jsonify({"error": f"Invalid localizer config: {e}"}), 400

    media_type = next(iter(snap.values())).get("type", "")
    if localizer.media_type != media_type:
        return (
            jsonify({"error": f"Localizer media type '{localizer.media_type}' does not match medias '{media_type}'"}),
            400,
        )

    results = _apply_processor_to_medias(localizer, snap, "localize", "localizations")

    return jsonify(
        {
            "localizer_name": localizer.name,
            "media_type": media_type,
            "total_medias_with_hits": len(results),
            "results": results,
        }
    )


@processors_scoring_bp.route("/api/auto-localize", methods=["POST"])
def auto_localize():
    """Run all autorun localizers for the current media type."""
    snap = snapshot_medias()
    if not snap:
        return jsonify({"error": "No medias loaded"}), 400

    media_type = next(iter(snap.values())).get("type", "")
    localizers = get_autorun_localizers_by_media(media_type)

    if not localizers:
        return jsonify({"error": f"No autorun localizers found for media type: {media_type}"}), 400

    results = _auto_run_processors(
        snap, localizers, _build_localizer, "localizer_type", "localize", "localizations", "localizer_name"
    )

    return jsonify(
        {
            "media_type": media_type,
            "localizers_run": len(results),
            "results": results,
        }
    )
