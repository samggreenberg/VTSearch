"""Schemas for ``/api/dataset/load-*`` and ``/api/dataset/clear`` (``vtsearch/routes/datasets/load.py``)."""

from __future__ import annotations

from marshmallow import Schema, fields, validate


class DatasetLoadDemoRequestSchema(Schema):
    """Body for ``POST /api/dataset/load-demo``.

    ``name`` must be a key of ``DEMO_DATASETS``; the handler returns 400
    if it isn't (the set isn't known at schema-build time).
    """

    name = fields.String(required=True, validate=validate.Length(min=1))
    embedder = fields.String(load_default="")
    embedders = fields.List(
        fields.String(),
        load_default=None,
        metadata={
            "description": (
                "v3 trio of create-time embedder picks (text / patch / structural). "
                "When set, every name is embedded so a multi-embedder dataset is produced; "
                "omitted falls back to the single `embedder`."
            )
        },
    )
    clipper = fields.String(load_default="")
    clipper_params = fields.Dict(
        load_default=None,
        metadata={
            "description": (
                'Optional parameter overrides for `clipper` (e.g. `{"duration": 5.0}`). '
                "Only applied when `clipper` names a real, non-default clipper."
            )
        },
    )
    cleaners = fields.List(
        fields.Dict(),
        load_default=None,
        metadata={
            "description": (
                "Cleanup gates to run on each finished unit before embedding, as "
                '`[{"name": "image_exif_orient", "params": {}}]`. Order is ignored; '
                "cleaners always run last, after the clipper / converter chain."
            )
        },
    )
    converter = fields.String(load_default="")
    dataset_name = fields.String(load_default="")
    outputs = fields.List(
        fields.Dict(),
        load_default=None,
        metadata={
            "description": (
                "Multi-dataset load: one object per dataset to make from the demo, in the shape "
                "`POST /api/dataset/import/{importer_name}` takes (`media_type`, `category`, `source_specs`, "
                "and per-dataset `embedder` / `embedders` / `clipper` / `clipper_params` / `cleaners`). "
                "Each output's `source_specs` is one row of the demo's own media type, direct or through one "
                "converter; converters run with their default parameters. The datasets are linked as one "
                "import, so a face can open the photo it was cut from. When set, the single-dataset "
                "`embedder` / `embedders` / `clipper` / `cleaners` / `converter` are ignored."
            )
        },
    )
    build_projection = fields.String(
        load_default="false",
        metadata={"description": "When 'true', compute + persist the 2-D Browse projection at ingest."},
    )
    merge_near_duplicates = fields.String(
        load_default="false",
        metadata={"description": "When 'true', collapse near-duplicate media into dupe sets at ingest."},
    )
    autofind = fields.String(
        load_default=None,
        allow_none=True,
        metadata={
            "description": (
                "'true' / 'false': whether to run the caller's AutoFind detectors on the dataset once it "
                "is saved, remembered as their `autofind_on_import` setting. Omitted, that setting decides."
            )
        },
    )


class DatasetLoadSourceRequestSchema(Schema):
    """Body for ``POST /api/dataset/load-source``.

    ``source`` is a raw origin dict (``{"importer": ..., "params": {...}}``)
    whose inner shape varies per importer and is validated by the
    handler (via ``can_reload_from_origin`` / ``reload_from_origin``).
    """

    source = fields.Dict(
        required=True,
        metadata={"description": "Origin dict as stored on medias."},
    )


class DatasetLoadStartedResponseSchema(Schema):
    """Response for ``POST /api/dataset/load-demo`` / ``load-file`` /
    ``load-source`` / ``import-local-folder`` and for
    ``POST /api/dataset/combine`` / ``promote`` in ``staging.py``.

    ``task_id`` is the background-task tracker id (string) used by the
    SSE progress stream; it may be empty when the load completes
    synchronously (rare).

    ``task_ids`` is set by the four routes that accept a multi-dataset
    ``outputs`` list (``import/{importer_name}``, ``import-local-folder``,
    ``import-local-files``, ``load-demo``): one task id per dataset the request started, in
    the outputs' order, so a client can follow (and cancel) each row.  A
    single-dataset request through those routes gets ``[task_id]``.
    """

    ok = fields.Boolean(required=True)
    message = fields.String(required=True)
    task_id = fields.String(required=True)
    task_ids = fields.List(
        fields.String(),
        metadata={
            "description": (
                "One task id per dataset a multi-dataset import started, in the request's ``outputs`` order "
                "(a single-dataset request lists its one ``task_id``).  Only the importer / local-folder / "
                "local-files / load-demo routes set it."
            )
        },
    )


class DatasetClearResponseSchema(Schema):
    """Response for ``POST /api/dataset/clear``."""

    ok = fields.Boolean(required=True)


__all__ = [
    "DatasetClearResponseSchema",
    "DatasetLoadDemoRequestSchema",
    "DatasetLoadSourceRequestSchema",
    "DatasetLoadStartedResponseSchema",
]
