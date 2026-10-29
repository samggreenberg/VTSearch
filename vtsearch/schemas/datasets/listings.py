"""Schemas for the read-only plugin listings (``vtsearch/routes/datasets/listings.py``).

``/api/media-types``, ``/api/embedders``, ``/api/clippers``,
``/api/cleaners``, ``/api/converters``, and the dataset-importer lists.

Every ``to_dict()`` payload here is enumerated as a nested schema so the
generated OpenAPI client carries a real typed model and the frontend can
delete its hand-written mirror.  Two flavours, depending on whether the
plugin family can add keys of its own:

* **Media types, embedders, converters, importers, picker tabs, registry
  entries** (the last in :mod:`vtsearch.schemas.datasets.registry`) have a
  *fixed* key set (no subclass overrides ``to_dict`` beyond the documented
  base), so their schemas are strict: an undeclared key is a bug, and
  dropping it on dump is the correct signal.
* **Clippers and cleaners** genuinely vary — concrete clippers append their
  own keys (``duration``, ``top_db``, ``box``, …) on top of the fixed base.
  Their schemas enumerate the base and inherit
  :class:`~vtsearch.schemas.common.PluginExtrasSchema`, so the fixed fields
  get real types while the plugin extras still reach the client verbatim.
  The generated model picks up an index signature from
  ``additionalProperties: true``, so the frontend gets that shape without
  hand-maintaining it.
"""

from __future__ import annotations

from marshmallow import Schema, fields, validate

from vtsearch.schemas.common import PluginExtrasSchema


class _MediaTypeFilterQuerySchema(Schema):
    """Query string shared by ``GET /api/embedders`` and ``/api/clippers``."""

    media_type = fields.String(
        load_default="",
        metadata={
            "description": (
                "Optional ``type_id`` (e.g. ``image``) or "
                "``folder_import_name`` (e.g. ``images``). When provided, "
                "only matching entries are returned."
            )
        },
    )


class EmbeddersListQuerySchema(_MediaTypeFilterQuerySchema):
    """Query for ``GET /api/embedders``."""


class ClippersListQuerySchema(_MediaTypeFilterQuerySchema):
    """Query for ``GET /api/clippers``."""


class CleanersListQuerySchema(_MediaTypeFilterQuerySchema):
    """Query for ``GET /api/cleaners``."""


class ConvertersListQuerySchema(Schema):
    """Query for ``GET /api/converters``.

    ``target`` and ``source`` are mutually exclusive filters; if both are
    supplied the handler prefers ``target``.
    """

    target = fields.String(load_default="")
    source = fields.String(load_default="")


class MediaTypeInfoSchema(Schema):
    """One ``MediaType.to_dict()`` payload (see ``vtscore/media/base.py``).

    Fixed shape: every media type emits the same keys, so this is a real
    nested schema rather than an opaque ``fields.Dict()``.  Only ``type_id``
    and ``name`` are guaranteed present enough to mark ``required``; the rest
    mirror the frontend's optional fields.
    """

    type_id = fields.String(required=True)
    name = fields.String(required=True)
    icon = fields.String()
    folder_import_name = fields.String()
    loops = fields.Boolean(
        metadata={"description": "Whether this media type's rendered form loops (e.g. short video/audio)."}
    )
    file_extensions = fields.List(
        fields.String(),
        metadata={"description": 'Glob patterns for files this media type claims, e.g. ``["*.jpg", "*.png"]``.'},
    )
    has_thumbnail = fields.Boolean(
        metadata={
            "description": (
                "Whether items of this type have a browsable thumbnail (image/video/document, and audio via "
                "its waveform PNG). Drives the VTSBrowse square-vs-hex bin shape and thumbnail painting."
            )
        }
    )
    importable = fields.Boolean(
        metadata={
            "description": (
                "Whether this type is a first-class ingestion category the user picks when importing (folder "
                "scan, file upload). ``false`` for a convert-in half type like ``face``."
            )
        }
    )
    embeddable = fields.Boolean(
        metadata={
            "description": (
                "Whether this type can be embedded (and therefore sorted / browsed / text-queried) on its own. "
                "``false`` for a convert-out half type like ``document`` that must be converted first."
            )
        }
    )
    converts_to = fields.List(
        fields.String(),
        metadata={
            "description": (
                "Embeddable target type_ids a non-embeddable type can convert into (first = default). "
                '``["image", "text"]`` for ``document``; empty for a directly-embeddable type.'
            )
        },
    )


class EmbedderInfoSchema(Schema):
    """One ``MediaEmbedder.to_dict()`` payload (see ``vtscore/media/embedder.py``).

    Fixed shape across all embedders (no subclass overrides ``to_dict``), so
    the payload is fully enumerated here instead of ``fields.Dict()``.
    """

    name = fields.String(required=True)
    display_name = fields.String(
        metadata={
            "description": (
                'Human-readable label shown in pickers, e.g. ``"SigLIP (general images)"``. Falls back to '
                "``name`` for legacy embedders that don't supply a friendlier label."
            )
        }
    )
    model_id = fields.String(
        allow_none=True,
        metadata={
            "description": (
                "Concrete pretrained-model identifier the embedder loads - usually a HuggingFace repo id "
                "(or a direct weights URL). ``null`` for embedders with no single downloadable model id "
                "(e.g. the classical SIFT/VLAD structural embedder)."
            )
        },
    )
    media_type_id = fields.String(required=True)
    is_default = fields.Boolean(
        metadata={
            "description": (
                "Whether this embedder is the recommended default for its media type (exactly one per media "
                'type). The dropdown surfaces this entry under a "Recommended" optgroup.'
            )
        }
    )
    supports_text = fields.Boolean(
        metadata={
            "description": (
                "Whether this embedder can embed text queries into the same vector space as its media. "
                "``false`` for vision-only encoders (DINOv3, Perception Encoder)."
            )
        }
    )
    supports_patch_regions = fields.Boolean(
        metadata={
            "description": (
                "Whether this embedder produces patch-level vectors and a hierarchical region tree per image. "
                "``true`` for patch-based encoders (DINOv2, DINOv3, EUPE)."
            )
        }
    )
    supports_geometric_verification = fields.Boolean(
        metadata={
            "description": (
                "Whether this embedder produces local features (keypoints + descriptors) for instance "
                "matching. ``true`` for structural embedders (SIFT/VLAD); ``false`` for every semantic embedder."
            )
        }
    )
    license_notice = fields.String(
        allow_none=True,
        metadata={
            "description": (
                "User-facing licence warning to show before the user picks this embedder. ``null`` for "
                "embedders with no special licensing constraints. Advisory only."
            )
        },
    )


class ImporterFieldSchema(Schema):
    """One ``PluginField.to_dict()`` payload (see ``vtscore/plugins/field.py``).

    Shared by every plugin family that renders a form: dataset importers,
    converters, label importers/exporters.  Fixed shape - ``PluginField`` is
    a dataclass and ``to_dict`` emits all of its wire fields unconditionally.

    Optionality mirrors what the frontend treats as optional rather than what
    the backend always emits (same convention as
    :class:`MediaTypeInfoSchema`): only ``key`` and ``field_type`` are the
    fields a consumer can never do without.
    """

    key = fields.String(required=True)
    field_type = fields.String(
        required=True,
        validate=validate.OneOf(
            ["file", "folder", "url", "text", "password", "email", "number", "select", "server_path", "checkbox"]
        ),
    )
    label = fields.String()
    description = fields.String()
    accept = fields.String(
        metadata={"description": 'For ``file`` fields: comma-separated extensions, e.g. ``".pkl"``.'}
    )
    options = fields.List(
        fields.String(),
        metadata={
            "description": (
                "For ``select`` fields: the statically-declared allowed values. Runtime-computed options come "
                "from ``POST /api/dataset/import/<importer>/options`` instead and carry their own "
                "``(value, label)`` shape; see ``ImporterFieldOptionsResponseSchema``."
            )
        },
    )
    default = fields.String()
    required = fields.Boolean()
    placeholder = fields.String(metadata={"description": "Hint shown as placeholder text inside the input widget."})
    hint = fields.String(
        metadata={
            "description": (
                "Inline format-hint text rendered as a visible chip below the input. Distinct from "
                "``description`` (which feeds the placeholder): the hint stays visible after the user starts "
                "typing, so it is the right place for accepted extensions or a short sample file layout."
            )
        }
    )
    dynamic_options = fields.Boolean(
        metadata={
            "description": (
                "When true, ``options`` is computed at runtime by calling "
                "``POST /api/dataset/import/<importer>/options`` with the current field values. The frontend "
                "re-fetches whenever any field listed in ``depends_on`` changes."
            )
        }
    )
    depends_on = fields.List(
        fields.String(), metadata={"description": "Field keys whose values this field's options depend on."}
    )
    allow_free_text = fields.Boolean(
        metadata={
            "description": (
                "For ``select`` fields: when true, render as a combobox the user can type an arbitrary value "
                "into. When the options refresh, a typed value absent from the new list is kept; a strict "
                "select clears it."
            )
        }
    )
    min = fields.String(metadata={"description": "For ``number`` fields: minimum allowed value (empty = no min)."})
    max = fields.String(metadata={"description": "For ``number`` fields: maximum allowed value (empty = no max)."})
    step = fields.String(
        metadata={"description": 'For ``number`` fields: step increment (empty / ``"any"`` = unconstrained).'}
    )
    clears = fields.List(
        fields.String(),
        metadata={
            "description": (
                "Field keys this field is mutually exclusive with. Entering a non-empty value here blanks each "
                "listed field (and they list this one back), so only one of the set is ever active at a time."
            )
        },
    )
    hidden = fields.Boolean(
        metadata={
            "description": (
                "When true, no GUI form renders a widget for this field: the plugin author fixes its value via "
                "``default`` and the user only sees the remaining fields (a plugin whose fields are all hidden "
                "presents as a bare action button). A GUI affordance only - the field still reaches the CLI, is "
                "still normalised and validated server-side, and is still listed here, so it is never a place to "
                "put a secret."
            )
        }
    )
    visible_when = fields.Dict(
        keys=fields.String(),
        values=fields.List(fields.String()),
        metadata={
            "description": (
                "Show this field only while each named field holds one of the listed values (every entry must "
                'match). Values are compared as strings after trimming; a checkbox controller reads ``"true"`` / '
                '``"false"``, and a blank or itself-hidden controller counts as its ``default``. While the '
                "condition fails the field is not rendered and not required, and the server resets it to its "
                "default. Empty means always shown."
            )
        },
    )
    template_vars = fields.List(
        fields.String(),
        metadata={
            "description": (
                "Framework-substituted template variable names (e.g. ``detector_name``, ``YYYYMMDD``) replaced "
                "in this field's value before the plugin runs. Advisory for clients; substitution is server-side."
            )
        },
    )


class ConverterInfoSchema(Schema):
    """One ``MediaConverter.to_dict()`` payload (see ``vtscore/converters/base.py``).

    Fixed shape across all converters.
    """

    name = fields.String(required=True)
    source_type = fields.String(required=True)
    target_type = fields.String(required=True)
    display_name = fields.String()
    description = fields.String()
    summary_template = fields.String(
        metadata={
            "description": (
                "One-line preview with ``{key}`` placeholders for each field. The native row of the importer "
                "source-specs picker substitutes the current field values. Falls back to ``description`` when empty."
            )
        }
    )
    #: Python attribute renamed to avoid shadowing ``marshmallow.fields``;
    #: mapped back to the ``fields`` wire key.
    field_list = fields.List(fields.Nested(ImporterFieldSchema), attribute="fields", data_key="fields")


class ClipperParameterSchema(Schema):
    """One entry of a clipper's ``parameters`` / ``creation_questions`` list.

    See ``MediaClipper.parameters`` in ``vtscore/media/clipper.py`` for the
    descriptor contract.  ``min`` / ``max`` / ``step`` are only meaningful for
    ``number`` parameters.
    """

    key = fields.String(required=True)
    label = fields.String(required=True)
    description = fields.String()
    type = fields.String(required=True, validate=validate.OneOf(["number", "string"]))
    default = fields.Raw(required=True, metadata={"oneOf": [{"type": "number"}, {"type": "string"}]})
    min = fields.Float()
    max = fields.Float()
    step = fields.Float()


class ClipperInfoSchema(PluginExtrasSchema):
    """One ``MediaClipper.to_dict()`` payload (see ``vtscore/media/clipper.py``).

    The base key set is fixed, but concrete clippers append their own
    resolved parameter values (``duration``, ``top_db``, ``box``, …), so this
    enumerates the base and lets the extras through untouched.  See the
    module docstring for why that beats either an opaque ``fields.Dict()`` or
    a strict schema that would silently drop those keys.
    """

    name = fields.String(required=True)
    media_type = fields.String(required=True)
    display_name = fields.String()
    description = fields.String()
    summary_template = fields.String(
        metadata={
            "description": (
                "One-line preview with ``{key}`` placeholders for each parameter. The native row of the "
                "importer source-specs picker substitutes the current parameter values, so the user sees a live "
                "summary of what the clipper will do. Falls back to ``description`` when empty."
            )
        }
    )
    parameters = fields.List(fields.Nested(ClipperParameterSchema))
    creation_questions = fields.List(
        fields.Nested(ClipperParameterSchema),
        metadata={
            "description": (
                "Parameters to ask about when the user first picks this clipper. Defaults to ``parameters``."
            )
        },
    )


class CleanerInfoSchema(ClipperInfoSchema):
    """One ``MediaCleaner.to_dict()`` payload (see ``vtscore/media/cleaner.py``).

    ``MediaCleaner`` subclasses ``MediaClipper``, so this is the clipper
    descriptor plus the cleaner-only ``default_enabled`` flag.
    """

    default_enabled = fields.Boolean(
        metadata={
            "description": (
                "Whether the import form checks this cleaner by default. True only for cleaners that fix an "
                "outright representation bug (EXIF orientation), where leaving it off ships known-wrong vectors."
            )
        }
    )


class ImporterInfoSchema(Schema):
    """One ``ImporterBase.to_dict()`` payload (see ``vtscore/datasets/importers/base/core.py``).

    Fixed shape: ``PluginBase.to_dict()`` emits the shared plugin metadata
    and ``ImporterBase.to_dict()`` appends ``picker_view`` / ``category`` /
    ``available_converters_by_media_type`` / ``supports_multi_output``.  No
    concrete importer overrides ``to_dict``, so this schema is strict.
    """

    name = fields.String(required=True)
    display_name = fields.String()
    description = fields.String()
    icon = fields.String()
    #: Python attribute renamed to avoid shadowing ``marshmallow.fields``;
    #: mapped back to the ``fields`` wire key.
    field_list = fields.List(fields.Nested(ImporterFieldSchema), attribute="fields", data_key="fields")
    ui_mode = fields.String()
    hidden_from_picker = fields.Boolean()
    picker_view = fields.String(
        metadata={
            "description": (
                "Which view the dataset-importer modal opens for this card: ``form`` (default), ``demo``, "
                "``server_folder``, ``local_folder``, or ``local_files``."
            )
        }
    )
    category = fields.String(
        metadata={
            "description": (
                "Picker tab this importer belongs to. One of ``services``, ``server``, ``local``, ``demo``, "
                'or ``""`` (uncategorised).'
            )
        }
    )
    supports_multi_output = fields.Boolean(
        metadata={
            "description": (
                "Whether one run of this importer may produce several datasets (the Add Dataset dialog's "
                "Multi-Dataset mode; ``POST /api/dataset/import/{importer_name}`` with ``outputs``). True when "
                "the importer declares a ``media_type`` field and has not opted out."
            )
        }
    )
    available_converters_by_media_type = fields.Dict(
        keys=fields.String(),
        values=fields.List(fields.Nested(ConverterInfoSchema)),
        metadata={
            "description": (
                "Map of output media-type id -> converters whose ``target_type`` matches that id. Drives the "
                '"Include rows" UI without an extra round-trip to ``/api/converters``.'
            )
        },
    )
    enabled = fields.Boolean(
        metadata={
            "description": (
                "Only set on ``combine_datasets`` by ``GET /api/dataset/all-importers``: false when fewer than "
                "two saved datasets share a media type, so there is nothing to combine."
            )
        }
    )


class ImporterPickerTabSchema(Schema):
    """One entry of the ``tabs`` array from ``GET /api/dataset/all-importers``.

    Registered by plugins via
    :func:`vtscore.datasets.importers.tabs.register_picker_tab`; ``id``
    matches the importers' ``category``.
    """

    id = fields.String(required=True)
    label = fields.String(required=True)
    icon = fields.String(metadata={"description": "``vt-icon`` type name."})
    order = fields.Integer(metadata={"description": "Lower values render first."})


class MediaTypesListResponseSchema(Schema):
    """Response for ``GET /api/media-types``."""

    media_types = fields.List(fields.Nested(MediaTypeInfoSchema), required=True)


class EmbeddersListResponseSchema(Schema):
    """Response for ``GET /api/embedders``."""

    embedders = fields.List(fields.Nested(EmbedderInfoSchema), required=True)


class ClippersListResponseSchema(Schema):
    """Response for ``GET /api/clippers``."""

    clippers = fields.List(fields.Nested(ClipperInfoSchema), required=True)


class CleanersListResponseSchema(Schema):
    """Response for ``GET /api/cleaners``."""

    cleaners = fields.List(fields.Nested(CleanerInfoSchema), required=True)


class ConvertersListResponseSchema(Schema):
    """Response for ``GET /api/converters``."""

    converters = fields.List(fields.Nested(ConverterInfoSchema), required=True)


class DatasetImportersListResponseSchema(Schema):
    """Response for ``GET /api/dataset/importers``."""

    importers = fields.List(fields.Nested(ImporterInfoSchema), required=True)


class DatasetAllImportersListResponseSchema(Schema):
    """Response for ``GET /api/dataset/all-importers``.

    ``tabs`` is the picker-tab layout; the ``combine_datasets`` importer is
    annotated with an ``enabled`` flag by the handler.
    """

    importers = fields.List(fields.Nested(ImporterInfoSchema), required=True)
    tabs = fields.List(fields.Nested(ImporterPickerTabSchema), required=True)


__all__ = [
    "CleanersListQuerySchema",
    "CleanersListResponseSchema",
    "ClippersListQuerySchema",
    "ClippersListResponseSchema",
    "ConvertersListQuerySchema",
    "ConvertersListResponseSchema",
    "DatasetAllImportersListResponseSchema",
    "DatasetImportersListResponseSchema",
    "EmbeddersListQuerySchema",
    "EmbeddersListResponseSchema",
    "MediaTypesListResponseSchema",
]
