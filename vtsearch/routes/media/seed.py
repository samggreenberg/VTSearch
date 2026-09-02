"""Blueprint for seed importers: fetch a batch of unlabeled seeds into ``example_media/``.

Seed importers (:mod:`vtscore.seed_importers`) contribute *unlabeled* media
to a brand-new blank detector — items that are "close but not quite" what
the user is hunting for.  They are the batch, verdict-free sibling of the
single-item datasource importer and of the good/bad label importer, and the
routes here mirror the same list / run / field-options trio:

* ``GET  /api/seed-importers`` ->
        :class:`~vtsearch.schemas.media.SeedImportersListResponseSchema`
* ``POST /api/seed-import/<importer_name>`` ->
        :class:`~vtsearch.schemas.media.SeedImportResponseSchema`
        (plugin-dependent body; per-plugin typed routes are registered for
        the OpenAPI spec, file-field plugins use multipart)
* ``POST /api/seed-import/<importer_name>/options`` -> dynamic select
        options, same contract as the dataset-importer variant.

The run endpoint saves every returned item's bytes into the per-user
``example_media/`` directory and returns one ``{filename, original_name,
origin}`` entry apiece, so each seed plugs into the existing ``{type:
"media", value: <filename>}`` detector-example model unchanged.  The client
marks those examples ``"labeled": false``, which is what keeps a seed a
query rather than a good vote (see
:func:`~vtscore.detectors.media_seeding.is_labeled_example`).
"""

from __future__ import annotations

from flask_smorest import Blueprint, abort

from vtscore.seed_importers import get_seed_importer, list_seed_importers
from vtsearch.routes._shared import (
    get_plugin_or_404,
    plugin_field_options,
    validate_plugin_args,
)
from vtsearch.schemas.datasets import (
    ImporterFieldOptionsRequestSchema,
    ImporterFieldOptionsResponseSchema,
)
from vtsearch.schemas.media import (
    SeedImportersListResponseSchema,
    SeedImportResponseSchema,
)
from vtsearch.settings import filter_visible_plugins

seed_importers_bp = Blueprint(
    "seed_importers",
    __name__,
    description="Seed importers: fetch a batch of unlabeled seed media into example_media/.",
)


@seed_importers_bp.route("/api/seed-importers")
@seed_importers_bp.response(200, SeedImportersListResponseSchema)
def list_seed_importers_route():
    """List the available seed importers.

    Empty on a vanilla install: no seed importer ships in-tree, so the New
    Detector modal's Blank flow shows only its stock Text and media tabs
    until a third-party plugin registers one.
    """
    importers = [imp.to_dict() for imp in filter_visible_plugins("seed_importers", list_seed_importers())]
    return {"importers": importers}


@seed_importers_bp.route("/api/seed-import/<importer_name>", methods=["POST"])
def run_seed_import(importer_name: str):
    """Fetch a batch of unlabeled seed media via the named seed importer.

    Plugin-dependent body shape (JSON, or multipart when the importer
    declares ``file`` fields).  Every returned item's bytes are saved into
    the per-user ``example_media/`` directory; the response lists one
    ``{filename, original_name, origin}`` entry per saved seed.

    A run that returns more than the importer's ``max_items`` is truncated
    to that cap rather than filling the directory, and the response says
    so via ``truncated``.  An empty batch is a valid answer (``count: 0``),
    not an error: "nothing matched" is something the user should see rather
    than a failure that hides which field was too narrow.
    """
    from vtsearch.routes.media.server import save_example_media_bytes

    importer, err = get_plugin_or_404(get_seed_importer, list_seed_importers, importer_name, "seed importer")
    if err:
        return err
    assert importer is not None  # narrowed by err check

    field_values = validate_plugin_args(importer)

    try:
        items = importer.run(field_values)
    except ValueError as exc:
        abort(400, message=str(exc))
    except NotImplementedError as exc:
        abort(501, message=str(exc) or f"{importer_name} does not implement run")
    except Exception as exc:  # noqa: BLE001 (surface source/network errors verbatim)
        abort(502, message=str(exc) or type(exc).__name__)

    if not isinstance(items, list):
        abort(502, message=f"Seed importer '{importer_name}' did not return a list of seed items")

    max_items = max(0, int(getattr(importer, "max_items", 0) or 0))
    truncated = len(items) > max_items
    kept = items[:max_items]

    saved = []
    for item in kept:
        data = getattr(item, "data", b"")
        if not data:
            # One empty item shouldn't sink the batch: the rest are still
            # usable seeds, and ``count`` tells the caller how many landed.
            continue
        filename = getattr(item, "filename", "") or "seed.bin"
        saved.append(
            {
                "filename": save_example_media_bytes(data, filename),
                "original_name": filename,
                "origin": getattr(item, "origin", None),
            }
        )

    return (
        SeedImportResponseSchema().dump({"items": saved, "count": len(saved), "truncated": truncated}),
        201,
    )


@seed_importers_bp.route("/api/seed-import/<importer_name>/options", methods=["POST"])
@seed_importers_bp.arguments(ImporterFieldOptionsRequestSchema)
@seed_importers_bp.response(200, ImporterFieldOptionsResponseSchema)
@seed_importers_bp.alt_response(400, description="Unknown or non-dynamic field key.")
@seed_importers_bp.alt_response(404, description="Unknown seed importer name.")
@seed_importers_bp.alt_response(500, description="get_field_options did not return a list.")
@seed_importers_bp.alt_response(501, description="Importer does not implement get_field_options.")
@seed_importers_bp.alt_response(502, description="Remote service backing dynamic options raised an error.")
def seed_importer_field_options(body: dict, importer_name: str):
    """Return dropdown options for a dynamic-options field.

    Same contract as ``POST /api/dataset/import/<name>/options``.
    """
    importer, err = get_plugin_or_404(get_seed_importer, list_seed_importers, importer_name, "seed importer")
    if err:
        return err
    assert importer is not None  # narrowed by err check

    return plugin_field_options(importer, body)

