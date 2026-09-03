"""Blueprint for datasource importers: fetch one media item into ``example_media/``.

Datasource importers (:mod:`vtscore.datasource_importers`) are the
single-item siblings of dataset importers: each fetches exactly one media
item from some source (a URL, a server path, a third-party service).  The
routes here power the dynamic example-media picker in the New Detector
modal, mirroring the dataset-importer trio of list / run / field-options
endpoints:

* ``GET  /api/datasource-importers`` ->
        :class:`~vtsearch.schemas.media.DatasourceImportersListResponseSchema`
* ``POST /api/datasource-import/<importer_name>`` ->
        :class:`~vtsearch.schemas.media.DatasourceImportResponseSchema`
        (plugin-dependent body; per-plugin typed routes are registered for
        the OpenAPI spec, file-field plugins use multipart)
* ``POST /api/datasource-import/<importer_name>/options`` -> dynamic
        select options, same contract as the dataset-importer variant.

The run endpoint saves the fetched bytes into the per-user
``example_media/`` directory and returns the same ``{filename,
original_name}`` contract as ``POST /api/server-media-files/upload``, so a
fetched item plugs into the existing ``{type: "media", value: <filename>}``
detector-example model unchanged.  It additionally returns the item's
durable ``origin`` (when the importer reports one) for the client to store
on the example, keeping the item re-fetchable after the cache file is gone.
"""

from __future__ import annotations

from flask_smorest import Blueprint, abort

from vtscore.datasource_importers import get_datasource_importer, list_datasource_importers
from vtsearch.routes._plugins import get_plugin_or_404, plugin_field_options, validate_plugin_args
from vtsearch.schemas.datasets import (
    ImporterFieldOptionsRequestSchema,
    ImporterFieldOptionsResponseSchema,
)
from vtsearch.schemas.media import (
    DatasourceImportersListResponseSchema,
    DatasourceImportResponseSchema,
)
from vtsearch.settings import filter_visible_plugins

datasource_importers_bp = Blueprint(
    "datasource_importers",
    __name__,
    description="Datasource importers: fetch a single media item into example_media/.",
)


@datasource_importers_bp.route("/api/datasource-importers")
@datasource_importers_bp.response(200, DatasourceImportersListResponseSchema)
def list_datasource_importers_route():
    """List available datasource importers and the shared picker tabs.

    ``tabs`` are the dataset-importer picker-tab declarations; datasource
    importers use the same category ids so both families share one tab
    bar in the example-media picker.
    """
    from vtscore.datasets.importers.tabs import list_picker_tabs

    importers = [imp.to_dict() for imp in filter_visible_plugins("datasource_importers", list_datasource_importers())]
    return {"importers": importers, "tabs": list_picker_tabs()}


@datasource_importers_bp.route("/api/datasource-import/<importer_name>", methods=["POST"])
def run_datasource_import(importer_name: str):
    """Fetch one media item via the named datasource importer.

    Plugin-dependent body shape (JSON, or multipart when the importer
    declares ``file`` fields).  On success the fetched bytes are saved
    into the per-user ``example_media/`` directory and the saved
    ``filename`` (persistence key) plus the human-readable
    ``original_name`` are returned, matching the upload endpoint's
    contract, along with the item's durable ``origin`` when the importer
    reports one (``null`` otherwise).
    """
    from vtsearch.routes.media.server import save_example_media_bytes

    importer, err = get_plugin_or_404(
        get_datasource_importer, list_datasource_importers, importer_name, "datasource importer"
    )
    if err:
        return err
    assert importer is not None  # narrowed by err check

    field_values = validate_plugin_args(importer)

    try:
        item = importer.fetch(field_values)
    except ValueError as exc:
        abort(400, message=str(exc))
    except NotImplementedError as exc:
        abort(501, message=str(exc) or f"{importer_name} does not implement fetch")
    except Exception as exc:  # noqa: BLE001 (surface source/network errors verbatim)
        abort(502, message=str(exc) or type(exc).__name__)

    if not item.data:
        abort(502, message=f"Datasource importer '{importer_name}' returned no data")

    safe_name = save_example_media_bytes(item.data, item.filename)

    return (
        DatasourceImportResponseSchema().dump(
            {"filename": safe_name, "original_name": item.filename, "origin": item.origin}
        ),
        201,
    )


@datasource_importers_bp.route("/api/datasource-import/<importer_name>/options", methods=["POST"])
@datasource_importers_bp.arguments(ImporterFieldOptionsRequestSchema)
@datasource_importers_bp.response(200, ImporterFieldOptionsResponseSchema)
@datasource_importers_bp.alt_response(400, description="Unknown or non-dynamic field key.")
@datasource_importers_bp.alt_response(404, description="Unknown datasource importer name.")
@datasource_importers_bp.alt_response(500, description="get_field_options did not return a list.")
@datasource_importers_bp.alt_response(501, description="Importer does not implement get_field_options.")
@datasource_importers_bp.alt_response(502, description="Remote service backing dynamic options raised an error.")
def datasource_importer_field_options(body: dict, importer_name: str):
    """Return dropdown options for a dynamic-options field.

    Same contract as ``POST /api/dataset/import/<name>/options``: the
    importer's ``get_field_options(field_key, current_values)`` is called
    with the supplied snapshot of current form values, and plugin errors
    (network failure, auth error, etc.) surface as a 502 with the
    original message so the frontend can display them inline.
    """
    importer, err = get_plugin_or_404(
        get_datasource_importer, list_datasource_importers, importer_name, "datasource importer"
    )
    if err:
        return err
    assert importer is not None  # narrowed by err check

    return plugin_field_options(importer, body)
