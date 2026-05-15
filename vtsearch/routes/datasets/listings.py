"""Read-only listing endpoints for media types, embedders, clippers,
converters, and dataset importers."""

from collections import Counter

from flask import Blueprint, jsonify, request

from vtsearch.datasets import list_importers
from vtsearch.datasets.registry import list_datasets as _reg_list_all
from vtsearch.routes.datasets._helpers import _normalize_media_type_param

datasets_listings_bp = Blueprint("datasets_listings", __name__)


@datasets_listings_bp.route("/api/media-types")
def media_types_list():
    """Return all registered media types with their metadata."""
    from vtsearch.media import all_types_dict

    return jsonify({"media_types": all_types_dict()})


@datasets_listings_bp.route("/api/embedders")
def embedders_list():
    """Return all registered embedders, optionally filtered by media type.

    Query parameters:
        media_type: A ``type_id`` (e.g. ``"image"``) or ``folder_import_name``
            (e.g. ``"images"``).  When provided, only embedders whose
            ``media_type_id`` matches are returned.
    """
    from vtsearch.media import all_embedders_dict, embedders_for_type

    media_type = _normalize_media_type_param(request.args.get("media_type", ""))
    if media_type:
        embedders = [e.to_dict() for e in embedders_for_type(media_type)]
    else:
        embedders = all_embedders_dict()

    return jsonify({"embedders": embedders})


@datasets_listings_bp.route("/api/clippers")
def clippers_list():
    """Return all clippers, optionally filtered by media type.

    Query parameters:
        media_type: A ``type_id`` (e.g. ``"image"``) or ``folder_import_name``
            (e.g. ``"images"``).  When provided, only clippers whose
            ``media_type`` matches are returned.
    """
    from vtsearch.media import all_clippers_dict, clippers_for_type

    media_type = _normalize_media_type_param(request.args.get("media_type", ""))
    if media_type:
        clippers = [c.to_dict() for c in clippers_for_type(media_type)]
    else:
        clippers = all_clippers_dict()

    return jsonify({"clippers": clippers})


@datasets_listings_bp.route("/api/converters")
def converters_list():
    """Return all converters, optionally filtered by source or target media type.

    Query parameters:
        target: A ``type_id`` (e.g. ``"image"``) or ``folder_import_name``
            (e.g. ``"images"``).  When provided, only converters whose
            ``target_type`` matches are returned.
        source: A ``type_id`` (e.g. ``"video"``) or ``folder_import_name``
            (e.g. ``"videos"``).  When provided, only converters whose
            ``source_type`` matches are returned.
    """
    from vtsearch.converters import list_converters, list_converters_for_source, list_converters_for_target

    target = _normalize_media_type_param(request.args.get("target", ""))
    source = _normalize_media_type_param(request.args.get("source", ""))

    if target:
        converters = list_converters_for_target(target)
    elif source:
        converters = list_converters_for_source(source)
    else:
        converters = list_converters()

    return jsonify({"converters": [c.to_dict() for c in converters]})


@datasets_listings_bp.route("/api/dataset/importers")
def dataset_importers():
    """List all registered importers (excluding those with non-form UI)."""
    extended = [imp.to_dict() for imp in list_importers() if imp.ui_mode == "form"]
    return jsonify({"importers": extended})


@datasets_listings_bp.route("/api/dataset/all-importers")
def dataset_all_importers():
    """List all registered importers (including built-in ones)."""
    from vtsearch.datasets.importers.tabs import list_picker_tabs

    all_importers = [imp.to_dict() for imp in list_importers()]

    # Annotate combine_datasets with an enabled flag: requires 2+ saved
    # datasets sharing the same media type.
    type_counts = Counter(e.get("media_type") for e in _reg_list_all())
    can_combine = any(c >= 2 for c in type_counts.values())
    for imp_dict in all_importers:
        if imp_dict["name"] == "combine_datasets":
            imp_dict["enabled"] = can_combine
            break

    return jsonify({"importers": all_importers, "tabs": list_picker_tabs()})
