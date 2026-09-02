"""Drift guards between plugin ``to_dict()`` payloads and their schemas.

Several schemas in :mod:`vtsearch.schemas.datasets` hand-transcribe a plugin
``to_dict()`` payload so the generated OpenAPI client carries a real typed
model instead of ``any``.  Nothing tied the two together, and the failure
mode is silent in the worst way: marshmallow **drops undeclared keys on
dump**, so adding a key to a ``to_dict()`` removes it from the API response
and from the generated TypeScript client without a single test going red.

These tests close that loop.  They are deliberately one-directional except
for :class:`~vtsearch.schemas.datasets.ImporterFieldSchema`: a key the wire
emits but the schema does not declare is a bug (it vanishes), while a key
the schema declares but no plugin currently emits is fine (routes annotate
some payloads themselves — ``ImporterInfoSchema.enabled`` is added by the
``/api/dataset/all-importers`` handler).

The last two tests pin the two halves of the dump-side contract in
:mod:`vtsearch.schemas.common`: opting in to the passthrough must let extras
through, and *not* opting in must keep dropping them.  The strict default is
a security boundary, not a formality — it is what keeps a media dict's
embedding vectors out of an auto-detect response.
"""

from __future__ import annotations

import pytest
from marshmallow import Schema, fields

from vtscore.converters import list_converters
from vtscore.datasets import list_importers
from vtscore.media import all_cleaners_dict, all_clippers_dict, all_embedders_dict, all_types_dict
from vtscore.plugins import PluginField
from vtsearch.schemas.common import PluginExtrasSchema
from vtsearch.schemas.datasets import (
    CleanerInfoSchema,
    ClipperInfoSchema,
    ConverterInfoSchema,
    EmbedderInfoSchema,
    ImporterFieldSchema,
    ImporterInfoSchema,
    MediaTypeInfoSchema,
)


def _wire_keys(schema: Schema) -> set[str]:
    """The key names *schema* emits on dump (``data_key`` where declared)."""
    return {f.data_key or name for name, f in schema.fields.items()}


def test_importer_field_schema_matches_plugin_field_exactly():
    """``ImporterFieldSchema`` is a hand-written mirror of ``PluginField.to_dict()``.

    Unlike the other schemas here, this one is checked for *equality*:
    ``PluginField`` is a dataclass whose ``to_dict`` emits all of its wire
    fields unconditionally, so a key on one side and not the other is drift
    in whichever direction it points.  ``PluginField`` itself is external
    plugin API; only this transcript of it is ours to fix.
    """
    wire = set(PluginField(key="k", label="Label", field_type="text").to_dict())
    declared = _wire_keys(ImporterFieldSchema())

    assert wire == declared, (
        "ImporterFieldSchema has drifted from PluginField.to_dict(). "
        f"Emitted but not declared (silently dropped from every response): {sorted(wire - declared)}. "
        f"Declared but never emitted: {sorted(declared - wire)}."
    )


@pytest.mark.parametrize(
    ("schema", "payloads"),
    [
        pytest.param(MediaTypeInfoSchema, lambda: all_types_dict(), id="media_types"),
        pytest.param(EmbedderInfoSchema, lambda: all_embedders_dict(), id="embedders"),
        pytest.param(ConverterInfoSchema, lambda: [c.to_dict() for c in list_converters()], id="converters"),
        pytest.param(ImporterInfoSchema, lambda: [i.to_dict() for i in list_importers()], id="importers"),
    ],
)
def test_fixed_shape_schemas_declare_every_emitted_key(schema, payloads):
    """Every key a registered plugin emits is declared by its schema.

    These families have a fixed key set (no subclass overrides ``to_dict``
    beyond the documented base), so an undeclared key is a plain bug: it is
    dropped on dump and never reaches the client.
    """
    declared = _wire_keys(schema())
    emitted: set[str] = set()
    for payload in payloads():
        emitted |= set(payload)

    assert emitted, "no plugins registered — the check would pass vacuously"
    assert emitted <= declared, (
        f"{schema.__name__} does not declare every key its plugins emit; "
        f"these are silently dropped on dump: {sorted(emitted - declared)}"
    )


@pytest.mark.parametrize(
    ("schema", "payloads"),
    [
        pytest.param(ClipperInfoSchema, lambda: all_clippers_dict(), id="clippers"),
        pytest.param(CleanerInfoSchema, lambda: all_cleaners_dict(), id="cleaners"),
    ],
)
def test_open_shape_schemas_declare_the_fixed_base_keys(schema, payloads):
    """Clippers and cleaners are the open families: extras pass through.

    So the assertion is narrower than the fixed-shape one.  Every key that
    *all* registered plugins in the family emit is a base key and must be
    declared, since that is what gets a real type in the generated client; a
    key only some of them emit is a concrete plugin's own resolved parameter
    (``duration``, ``top_db``, ``box``, …) and reaches the client via
    :class:`~vtsearch.schemas.common.PluginExtrasSchema` instead.
    """
    declared = _wire_keys(schema())
    plugins = payloads()

    assert len(plugins) > 1, "need at least two registered plugins to identify the shared base keys"
    base_keys = set(plugins[0])
    for payload in plugins[1:]:
        base_keys &= set(payload)

    assert base_keys <= declared, (
        f"{schema.__name__} leaves base keys undeclared, so they lose their type in the "
        f"generated client: {sorted(base_keys - declared)}"
    )


def test_plugin_extras_schema_passes_undeclared_keys_through_on_dump():
    """The opt-in half: ``unknown = INCLUDE`` alone would not do this."""

    class _Descriptor(PluginExtrasSchema):
        name = fields.String()

    dumped = _Descriptor().dump({"name": "blackout", "top_db": 30, "duration": 1.5})

    assert dumped == {"name": "blackout", "top_db": 30, "duration": 1.5}


def test_plain_schema_still_drops_undeclared_keys_on_dump():
    """The default half, and the reason the passthrough must stay opt-in.

    ``vtsearch.schemas.detectors._HitSchema`` relies on exactly this: a
    strict dump is the last line of defence keeping a media dict's embedding
    vectors out of an auto-detect response.
    """

    class _Strict(Schema):
        name = fields.String()

    dumped = _Strict().dump({"name": "blackout", "embedding": [0.1, 0.2, 0.3]})

    assert dumped == {"name": "blackout"}
