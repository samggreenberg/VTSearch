"""Schemas for the Settings Import/Export and Sync Sources APIs.

Covers:

* ``GET  /api/settings-importers``        — :class:`SettingsImporterEntrySchema`
* ``POST /api/settings-exporters/export`` — :class:`RunSettingsExportRequestSchema` →
                                             :class:`RunSettingsExportResponseSchema`
* ``GET  /api/settings-exporters``        — :class:`SettingsExporterEntrySchema`
* ``GET  /api/settings-sources``          — :class:`SyncSourceEntrySchema`
* ``GET  /api/settings-sources/active``   — :class:`SyncSourceConfigSchema`
* ``PUT  /api/settings-sources/active``   — :class:`SetSyncSourceRequestSchema` →
                                             :class:`OkMessageSchema`
* ``POST /api/settings-sources/sync``     — :class:`SyncFromSourceResponseSchema`
* ``GET  /api/labelset-sources``          — :class:`SyncSourceEntrySchema`
* ``GET  /api/detectors/<n>/labelset-source``      — :class:`SyncSourceConfigSchema`
* ``PUT  /api/detectors/<n>/labelset-source``      — :class:`SetSyncSourceRequestSchema` →
                                                      :class:`OkMessageSchema`
* ``POST /api/detectors/<n>/labelset-source/sync`` — :class:`SyncFromLabelsetSourceResponseSchema`

The per-plugin shape of ``field_values`` on ``POST
/api/settings-exporters/export`` is intentionally declared as
``fields.Dict()``: the inner keys vary per exporter and ``field_values``
is validated inside the handler against the selected plugin's
:attr:`fields` declaration.  The multipart-or-JSON ``POST
/api/settings-importers/import/<importer_name>`` route stays on the
legacy plain-Flask path (no decorator) for the same reason — see the
*Resolved questions / Plugin field endpoints* section of
``docs/plans/openapi-schema.md``.
"""

from __future__ import annotations

from marshmallow import Schema, fields, validate


class _PluginEntrySchema(Schema):
    """Shared shape for plugin-listing endpoints.

    Mirrors :meth:`vtscore.plugins.PluginBase.to_dict`; the ``fields``
    array's inner shape mirrors :meth:`vtscore.plugins.PluginField.to_dict`
    but is declared as ``fields.Dict()`` to avoid duplicating the source
    of truth across schema and dataclass.
    """

    name = fields.String(required=True)
    display_name = fields.String(required=True)
    description = fields.String(required=True)
    icon = fields.String(required=True)
    ui_mode = fields.String(required=True)
    hidden_from_picker = fields.Boolean(required=True)
    # ``data_key`` / ``attribute`` keep the wire name as ``"fields"`` on both
    # load and dump without shadowing :attr:`marshmallow.Schema.fields` (a
    # ``dict[str, Field]`` registry on the base class).
    plugin_fields = fields.List(
        fields.Dict(),
        required=True,
        data_key="fields",
        attribute="fields",
    )


class SettingsImporterEntrySchema(_PluginEntrySchema):
    """One entry in ``GET /api/settings-importers``."""


class SettingsExporterEntrySchema(_PluginEntrySchema):
    """One entry in ``GET /api/settings-exporters``."""


class SyncSourceEntrySchema(_PluginEntrySchema):
    """One entry in ``GET /api/settings-sources`` and ``GET /api/labelset-sources``."""


# ---------------------------------------------------------------------------
# Settings exporter run
# ---------------------------------------------------------------------------


class RunSettingsExportRequestSchema(Schema):
    """Body for ``POST /api/settings-exporters/export``.

    ``field_values`` is permissive (``fields.Dict``) because its keys
    depend on the named exporter; the handler validates the inner shape
    against the selected plugin.
    """

    exporter_name = fields.String(required=True, validate=validate.Length(min=1))
    field_values = fields.Dict(load_default=dict)


class RunSettingsExportResponseSchema(Schema):
    """Response for ``POST /api/settings-exporters/export``.

    ``download`` / ``data`` / ``filename`` are populated by the
    ``local_json_file`` exporter; ``filepath`` by ``server_json_file``;
    other exporters can return any extra keys.
    """

    success = fields.Boolean(required=True)
    message = fields.String()
    download = fields.Boolean()
    data = fields.Dict()
    filename = fields.String()
    filepath = fields.String()

    class Meta:
        # Exporter-specific extension keys flow through on dump.
        unknown = "include"


# ---------------------------------------------------------------------------
# Sync source config (shared by settings-sources and labelset-sources)
# ---------------------------------------------------------------------------


class SyncSourceConfigSchema(Schema):
    """A persisted sync-source config.

    The wire shape is either ``null`` (no source configured) or
    ``{"source_name": "...", "field_values": {...}}``.  This schema
    describes the populated form; the ``null`` case is documented via
    ``response(... )`` and the handler returning ``None``.
    """

    source_name = fields.String(required=True)
    field_values = fields.Dict(load_default=dict)


class SetSyncSourceRequestSchema(Schema):
    """Body for ``PUT /api/settings-sources/active`` and
    ``PUT /api/detectors/<name>/labelset-source``.

    A body of ``{}`` or one with empty ``source_name`` clears the active
    source.  Otherwise both fields specify the new source.  ``source_name``
    is declared as optional so the clear-by-empty-body shape continues to
    work; the handler 404s on unknown source names.
    """

    source_name = fields.String(load_default="")
    field_values = fields.Dict(load_default=dict)


class OkMessageSchema(Schema):
    """``{ok: bool, message: str}`` — generic ack envelope."""

    ok = fields.Boolean(required=True)
    message = fields.String(required=True)


# ---------------------------------------------------------------------------
# Sync-from-source responses
# ---------------------------------------------------------------------------


class SettingsImportResponseSchema(Schema):
    """Response for ``POST /api/settings-importers/import/<importer_name>``.

    The route stays on the legacy plain-Flask path (request body is a
    plugin-field shape), but the success body shape is fixed.  Currently
    *not* attached to the route via ``@response`` — kept here for the
    eventual unified plugin-field migration.
    """

    success = fields.Boolean(required=True)
    message = fields.String(required=True)
    keys = fields.List(fields.String(), required=True)


class SyncFromSourceResponseSchema(Schema):
    """Response for ``POST /api/settings-sources/sync``.

    ``ok=true`` carries ``keys`` (imported setting names) and ``message``.
    ``ok=false`` (no source configured / source empty) carries only
    ``message``.
    """

    ok = fields.Boolean(required=True)
    message = fields.String(required=True)
    keys = fields.List(fields.String())


class SyncFromLabelsetSourceResponseSchema(Schema):
    """Response for ``POST /api/detectors/<name>/labelset-source/sync``."""

    ok = fields.Boolean(required=True)
    message = fields.String(required=True)


__all__ = [
    "OkMessageSchema",
    "RunSettingsExportRequestSchema",
    "RunSettingsExportResponseSchema",
    "SetSyncSourceRequestSchema",
    "SettingsExporterEntrySchema",
    "SettingsImportResponseSchema",
    "SettingsImporterEntrySchema",
    "SyncFromLabelsetSourceResponseSchema",
    "SyncFromSourceResponseSchema",
    "SyncSourceConfigSchema",
    "SyncSourceEntrySchema",
]
