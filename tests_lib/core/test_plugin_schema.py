"""Unit tests for :mod:`vtscore.plugins.schema`.

Covers the per-field-type mapping (``PluginField`` →
:class:`marshmallow.fields.Field`) plus the high-level ``load()``
behaviour that the route helpers depend on:

* required text-like fields raise ``ValidationError`` on missing /
  empty / whitespace-only values;
* ``select`` fields enforce ``OneOf`` when ``options`` is static;
* ``number`` fields coerce strings to int/float per
  ``is_integer_number``;
* ``checkbox`` fields accept both ``bool`` and ``"true"`` / ``"false"``;
* unknown keys are dropped (``Meta.unknown = "exclude"``);
* the cache returns the same instance across calls.
"""

from __future__ import annotations

import pytest
from marshmallow import ValidationError

from vtscore.plugins import PluginBase, PluginField
from vtscore.plugins.schema import get_plugin_arg_schema, make_plugin_arg_schema


class _FakePlugin(PluginBase):
    """Bare-minimum stand-in for a real plugin instance."""

    def __init__(self, fields: list[PluginField]) -> None:
        self.fields = fields


class TestSchemaBuilder:
    def test_required_text_rejects_missing(self):
        plugin = _FakePlugin([PluginField(key="name", label="Name", field_type="text", required=True)])
        schema = make_plugin_arg_schema(plugin)()
        with pytest.raises(ValidationError) as exc:
            schema.load({})
        assert "name" in exc.value.messages

    def test_required_text_rejects_whitespace_only(self):
        plugin = _FakePlugin([PluginField(key="name", label="Name", field_type="text", required=True)])
        schema = make_plugin_arg_schema(plugin)()
        with pytest.raises(ValidationError) as exc:
            schema.load({"name": "   "})
        assert "name" in exc.value.messages

    def test_required_text_accepts_value(self):
        plugin = _FakePlugin([PluginField(key="name", label="Name", field_type="text", required=True)])
        schema = make_plugin_arg_schema(plugin)()
        assert schema.load({"name": "hello"}) == {"name": "hello"}

    def test_default_used_when_missing(self):
        plugin = _FakePlugin(
            [PluginField(key="name", label="Name", field_type="text", required=False, default="fallback")]
        )
        schema = make_plugin_arg_schema(plugin)()
        assert schema.load({}) == {"name": "fallback"}

    def test_select_rejects_unknown_option(self):
        plugin = _FakePlugin(
            [
                PluginField(
                    key="mode",
                    label="Mode",
                    field_type="select",
                    options=["a", "b", "c"],
                    required=True,
                )
            ]
        )
        schema = make_plugin_arg_schema(plugin)()
        with pytest.raises(ValidationError) as exc:
            schema.load({"mode": "z"})
        assert "mode" in exc.value.messages

    def test_select_dynamic_options_skips_oneof(self):
        plugin = _FakePlugin(
            [
                PluginField(
                    key="mode",
                    label="Mode",
                    field_type="select",
                    options=["a"],
                    dynamic_options=True,
                    required=False,
                )
            ]
        )
        schema = make_plugin_arg_schema(plugin)()
        assert schema.load({"mode": "anything-goes"}) == {"mode": "anything-goes"}

    def test_number_integer(self):
        plugin = _FakePlugin([PluginField(key="n", label="N", field_type="number", default="3", step="1")])
        schema = make_plugin_arg_schema(plugin)()
        assert schema.load({"n": 7}) == {"n": 7}
        # Marshmallow accepts numeric strings for Integer fields.
        assert schema.load({"n": "9"}) == {"n": 9}

    def test_number_float(self):
        plugin = _FakePlugin([PluginField(key="x", label="X", field_type="number", default="1.5", step="0.1")])
        schema = make_plugin_arg_schema(plugin)()
        assert schema.load({"x": 2.5}) == {"x": 2.5}

    def test_number_integer_range_enforced(self):
        plugin = _FakePlugin(
            [PluginField(key="n", label="N", field_type="number", default="128", step="1", min="8", max="512")]
        )
        schema = make_plugin_arg_schema(plugin)()
        # In-range values pass.
        assert schema.load({"n": 8}) == {"n": 8}
        assert schema.load({"n": 512}) == {"n": 512}
        # Below min and above max are both rejected.
        with pytest.raises(ValidationError) as exc:
            schema.load({"n": 7})
        assert "n" in exc.value.messages
        with pytest.raises(ValidationError) as exc:
            schema.load({"n": 10_000_000})
        assert "n" in exc.value.messages

    def test_number_float_range_enforced(self):
        plugin = _FakePlugin(
            [PluginField(key="t", label="T", field_type="number", default="0.5", step="0.05", min="0", max="1")]
        )
        schema = make_plugin_arg_schema(plugin)()
        assert schema.load({"t": 0.0}) == {"t": 0.0}
        assert schema.load({"t": 1.0}) == {"t": 1.0}
        with pytest.raises(ValidationError):
            schema.load({"t": -0.1})
        with pytest.raises(ValidationError):
            schema.load({"t": 1.5})

    def test_number_min_only_allows_arbitrary_upper(self):
        # synthetic.size, video2audio.ffmpeg_timeout etc. declare only a floor.
        plugin = _FakePlugin([PluginField(key="n", label="N", field_type="number", default="100", min="1", step="1")])
        schema = make_plugin_arg_schema(plugin)()
        assert schema.load({"n": 1_000_000}) == {"n": 1_000_000}
        with pytest.raises(ValidationError):
            schema.load({"n": 0})

    def test_number_no_range_when_unset(self):
        # No min / no max → no Range validator attached.
        plugin = _FakePlugin([PluginField(key="n", label="N", field_type="number", default="1", step="1")])
        schema = make_plugin_arg_schema(plugin)()
        assert schema.load({"n": -999_999}) == {"n": -999_999}
        assert schema.load({"n": 999_999_999}) == {"n": 999_999_999}

    def test_checkbox_accepts_string_true_false(self):
        plugin = _FakePlugin([PluginField(key="flag", label="Flag", field_type="checkbox", default="false")])
        schema = make_plugin_arg_schema(plugin)()
        assert schema.load({"flag": "true"}) == {"flag": True}
        assert schema.load({"flag": "false"}) == {"flag": False}
        assert schema.load({"flag": True}) == {"flag": True}

    def test_checkbox_uses_default_when_missing(self):
        plugin = _FakePlugin([PluginField(key="flag", label="Flag", field_type="checkbox", default="true")])
        schema = make_plugin_arg_schema(plugin)()
        assert schema.load({}) == {"flag": True}

    def test_file_field_skipped(self):
        plugin = _FakePlugin(
            [
                PluginField(key="upload", label="Upload", field_type="file", required=True),
                PluginField(key="name", label="Name", field_type="text", required=False),
            ]
        )
        schema = make_plugin_arg_schema(plugin)()
        # File fields are populated outside the schema; the validator
        # mustn't complain about them being missing.
        assert schema.load({"name": "ok"}) == {"name": "ok"}

    def test_unknown_keys_excluded(self):
        plugin = _FakePlugin([PluginField(key="name", label="Name", field_type="text", required=False)])
        schema = make_plugin_arg_schema(plugin)()
        loaded = schema.load({"name": "ok", "bogus": "junk", "chunk_size": 99})
        assert isinstance(loaded, dict)
        assert "bogus" not in loaded
        assert "chunk_size" not in loaded
        assert loaded["name"] == "ok"

    def test_cache_returns_same_instance(self):
        plugin = _FakePlugin([PluginField(key="x", label="X", field_type="text", required=False)])
        first = get_plugin_arg_schema(plugin)
        second = get_plugin_arg_schema(plugin)
        assert first is second
