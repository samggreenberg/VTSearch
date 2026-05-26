"""Read-only listing endpoints for media types, embedders, clippers,
converters, and dataset importers.

Migrated to ``flask_smorest`` so these routes appear in
``/api/openapi.json``. See ``docs/plans/openapi-schema.md``. The plugin
``to_dict()`` payloads are declared as ``fields.Dict()`` rather than
nested schemas - see the module docstring in ``vtsearch/schemas/datasets.py``.
"""

from collections import Counter

from flask_smorest import Blueprint

from vtscore.datasets import list_importers
from vtscore.datasets.registry import list_datasets as _reg_list_all
from vtsearch.routes.datasets._helpers import _normalize_media_type_param
from vtsearch.settings import filter_visible_plugin_dicts, filter_visible_plugins
from vtsearch.schemas.datasets import (
    ClippersListQuerySchema,
    ClippersListResponseSchema,
    ConvertersListQuerySchema,
    ConvertersListResponseSchema,
    DatasetAllImportersListResponseSchema,
    DatasetImportersListResponseSchema,
    EmbeddersListQuerySchema,
    EmbeddersListResponseSchema,
    MediaTypesListResponseSchema,
)

datasets_listings_bp = Blueprint(
    "datasets_listings",
    __name__,
    description="Read-only listings: media types, embedders, clippers, converters, importers.",
)


@datasets_listings_bp.route("/api/media-types")
@datasets_listings_bp.response(200, MediaTypesListResponseSchema)
def media_types_list():
    """Return all registered media types with their metadata."""
    from vtscore.media import all_types_dict

    return {"media_types": filter_visible_plugin_dicts("media_types", all_types_dict(), id_key="type_id")}


@datasets_listings_bp.route("/api/embedders")
@datasets_listings_bp.arguments(EmbeddersListQuerySchema, location="query")
@datasets_listings_bp.response(200, EmbeddersListResponseSchema)
def embedders_list(query: dict):
    """Return all registered embedders, optionally filtered by media type.

    Query parameters:
        media_type: A ``type_id`` (e.g. ``"image"``) or ``folder_import_name``
            (e.g. ``"images"``).  When provided, only embedders whose
            ``media_type_id`` matches are returned.
    """
    from vtscore.media import all_embedders_dict, embedders_for_type

    media_type = _normalize_media_type_param(query.get("media_type", ""))
    if media_type:
        embedders = [e.to_dict() for e in filter_visible_plugins("embedders", embedders_for_type(media_type))]
    else:
        embedders = filter_visible_plugin_dicts("embedders", all_embedders_dict())

    return {"embedders": embedders}


@datasets_listings_bp.route("/api/clippers")
@datasets_listings_bp.arguments(ClippersListQuerySchema, location="query")
@datasets_listings_bp.response(200, ClippersListResponseSchema)
def clippers_list(query: dict):
    """Return all clippers, optionally filtered by media type.

    Query parameters:
        media_type: A ``type_id`` (e.g. ``"image"``) or ``folder_import_name``
            (e.g. ``"images"``).  When provided, only clippers whose
            ``media_type`` matches are returned.
    """
    from vtscore.media import all_clippers_dict, clippers_for_type

    media_type = _normalize_media_type_param(query.get("media_type", ""))
    if media_type:
        clippers = [c.to_dict() for c in filter_visible_plugins("clippers", clippers_for_type(media_type))]
    else:
        clippers = filter_visible_plugin_dicts("clippers", all_clippers_dict())

    return {"clippers": clippers}


@datasets_listings_bp.route("/api/converters")
@datasets_listings_bp.arguments(ConvertersListQuerySchema, location="query")
@datasets_listings_bp.response(200, ConvertersListResponseSchema)
def converters_list(query: dict):
    """Return all converters, optionally filtered by source or target media type.

    Query parameters:
        target: A ``type_id`` (e.g. ``"image"``) or ``folder_import_name``
            (e.g. ``"images"``).  When provided, only converters whose
            ``target_type`` matches are returned.
        source: A ``type_id`` (e.g. ``"video"``) or ``folder_import_name``
            (e.g. ``"videos"``).  When provided, only converters whose
            ``source_type`` matches are returned.
    """
    from vtscore.converters import list_converters, list_converters_for_source, list_converters_for_target

    target = _normalize_media_type_param(query.get("target", ""))
    source = _normalize_media_type_param(query.get("source", ""))

    if target:
        converters = list_converters_for_target(target)
    elif source:
        converters = list_converters_for_source(source)
    else:
        converters = list_converters()

    return {"converters": [c.to_dict() for c in filter_visible_plugins("converters", converters)]}


@datasets_listings_bp.route("/api/dataset/importers")
@datasets_listings_bp.response(200, DatasetImportersListResponseSchema)
def dataset_importers():
    """List all registered importers (excluding those with non-form UI)."""
    extended = [imp.to_dict() for imp in filter_visible_plugins("importers", list_importers()) if imp.ui_mode == "form"]
    return {"importers": extended}


@datasets_listings_bp.route("/api/dataset/all-importers")
@datasets_listings_bp.response(200, DatasetAllImportersListResponseSchema)
def dataset_all_importers():
    """List all registered importers (including built-in ones)."""
    from vtscore.datasets.importers.tabs import list_picker_tabs

    all_importers = [imp.to_dict() for imp in filter_visible_plugins("importers", list_importers())]

    # Annotate combine_datasets with an enabled flag: requires 2+ saved
    # datasets sharing the same media type.
    type_counts = Counter(e.get("media_type") for e in _reg_list_all())
    can_combine = any(c >= 2 for c in type_counts.values())
    for imp_dict in all_importers:
        if imp_dict["name"] == "combine_datasets":
            imp_dict["enabled"] = can_combine
            break

    return {"importers": all_importers, "tabs": list_picker_tabs()}
