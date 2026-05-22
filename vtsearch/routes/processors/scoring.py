"""Extractor and localizer execution routes.

Migrated to ``flask_smorest`` so the routes are described in
``/api/openapi.json``. See ``docs/plans/openapi-schema.md``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from flask_smorest import Blueprint, abort

from vtscore.concurrency.memory_budget import cap_workers_by_memory
from vtsearch.routes.processors.crud import _build_extractor, _build_localizer
from vtsearch.schemas.processors import (
    AutoExtractResponseSchema,
    AutoLocalizeResponseSchema,
    ExtractRequestSchema,
    ExtractResponseSchema,
    LocalizeRequestSchema,
    LocalizeResponseSchema,
)
from vtsearch.autorun_processors import (
    get_autorun_extractors_by_media,
    get_autorun_localizers_by_media,
)
from vtsearch.state import snapshot_medias

processors_scoring_bp = Blueprint(
    "processors_scoring",
    __name__,
    description="Run extractors / localizers against the active dataset, either one-off or via autorun.",
)

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
@processors_scoring_bp.arguments(ExtractRequestSchema)
@processors_scoring_bp.response(200, ExtractResponseSchema)
@processors_scoring_bp.alt_response(400, description="No medias loaded, bad config, or media-type mismatch.")
def run_extract(body: dict):
    """Run a single extractor on all medias and return per-media extraction results."""
    extractor_name = body["name"].strip()
    extractor_type = body["extractor_type"].strip()
    config = body["config"]

    snap = snapshot_medias()
    if not snap:
        abort(400, message="No medias loaded")

    try:
        extractor = _build_extractor(extractor_name or "adhoc", extractor_type, config)
    except Exception as e:
        abort(400, message=f"Invalid extractor config: {e}")

    media_type = next(iter(snap.values())).get("media_type", "")
    if extractor.media_type != media_type:
        abort(
            400,
            message=f"Extractor media type '{extractor.media_type}' does not match medias '{media_type}'",
        )

    results = _apply_processor_to_medias(extractor, snap, "extract", "extractions")

    return {
        "extractor_name": extractor.name,
        "media_type": media_type,
        "total_medias_with_hits": len(results),
        "results": results,
    }


@processors_scoring_bp.route("/api/auto-extract", methods=["POST"])
@processors_scoring_bp.response(200, AutoExtractResponseSchema)
@processors_scoring_bp.alt_response(
    400, description="No medias loaded, or no autorun extractors for active media type."
)
def auto_extract():
    """Run all autorun extractors for the current media type and return extraction results."""
    snap = snapshot_medias()
    if not snap:
        abort(400, message="No medias loaded")

    media_type = next(iter(snap.values())).get("media_type", "")
    extractors = get_autorun_extractors_by_media(media_type)

    if not extractors:
        abort(400, message=f"No autorun extractors found for media type: {media_type}")

    results = _auto_run_processors(
        snap, extractors, _build_extractor, "extractor_type", "extract", "extractions", "extractor_name"
    )

    return {
        "media_type": media_type,
        "extractors_run": len(results),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Localizer execution
# ---------------------------------------------------------------------------


@processors_scoring_bp.route("/api/localize", methods=["POST"])
@processors_scoring_bp.arguments(LocalizeRequestSchema)
@processors_scoring_bp.response(200, LocalizeResponseSchema)
@processors_scoring_bp.alt_response(400, description="No medias loaded, bad config, or media-type mismatch.")
def run_localize(body: dict):
    """Run a single localizer on all clips and return per-clip localization results."""
    localizer_name = body["name"].strip()
    localizer_type = body["localizer_type"].strip()
    config = body["config"]

    snap = snapshot_medias()
    if not snap:
        abort(400, message="No medias loaded")

    try:
        localizer = _build_localizer(localizer_name or "adhoc", localizer_type, config)
    except Exception as e:
        abort(400, message=f"Invalid localizer config: {e}")

    media_type = next(iter(snap.values())).get("media_type", "")
    if localizer.media_type != media_type:
        abort(
            400,
            message=f"Localizer media type '{localizer.media_type}' does not match medias '{media_type}'",
        )

    results = _apply_processor_to_medias(localizer, snap, "localize", "localizations")

    return {
        "localizer_name": localizer.name,
        "media_type": media_type,
        "total_medias_with_hits": len(results),
        "results": results,
    }


@processors_scoring_bp.route("/api/auto-localize", methods=["POST"])
@processors_scoring_bp.response(200, AutoLocalizeResponseSchema)
@processors_scoring_bp.alt_response(
    400, description="No medias loaded, or no autorun localizers for active media type."
)
def auto_localize():
    """Run all autorun localizers for the current media type."""
    snap = snapshot_medias()
    if not snap:
        abort(400, message="No medias loaded")

    media_type = next(iter(snap.values())).get("media_type", "")
    localizers = get_autorun_localizers_by_media(media_type)

    if not localizers:
        abort(400, message=f"No autorun localizers found for media type: {media_type}")

    results = _auto_run_processors(
        snap, localizers, _build_localizer, "localizer_type", "localize", "localizations", "localizer_name"
    )

    return {
        "media_type": media_type,
        "localizers_run": len(results),
        "results": results,
    }
