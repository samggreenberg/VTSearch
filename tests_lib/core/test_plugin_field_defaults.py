"""A declared ``PluginField.default`` is a value, not a placeholder (issue #3874).

The default has to survive three different journeys into a plugin body, and
before this was pinned only the first of them worked:

* **CLI** - ``argparse`` applies the flag's default when the flag is omitted.
* **HTTP** - a GUI form posts every input it rendered, an untouched one as
  ``""``.  Marshmallow's ``load_default`` fires on a *missing* key only, so a
  blank arrived as a blank: a required field 422'd and an optional one
  reached ``run()`` as ``""``.
* **Saved settings** - the Auto-Find results exporter is handed a persisted
  ``field_values`` map with no schema in the loop at all, so
  :func:`~vtscore.plugins.normalize.normalize_field_values` is the only pass
  that can fill anything in.

Both ends are covered here because they back each other up: the schema hook
keeps a blank from being rejected before the default can apply, and the
normalize pass is what a schema-less caller relies on.
"""

from __future__ import annotations

import pytest
from marshmallow import ValidationError

from vtscore.plugins import PluginBase, PluginField
from vtscore.plugins.normalize import normalize_field_values
from vtscore.plugins.schema import make_plugin_arg_schema


class _FakePlugin(PluginBase):
    """Bare-minimum stand-in for a real plugin instance."""

    def __init__(self, fields: list[PluginField]) -> None:
        self.fields = fields


def _email_plugin(default: str = "ops@example.com") -> _FakePlugin:
    """The issue's own field: a required e-mail with a runtime-known default."""
    return _FakePlugin(
        [
            PluginField(
                key="email_address",
                label="Email Address",
                field_type="email",
                default=default,
                description="Email Address for Autorun Results.",
            )
        ]
    )


class TestSchemaAppliesDefaultToBlank:
    """``make_plugin_arg_schema`` — the HTTP ingress."""

    def test_required_field_with_default_accepts_an_empty_string(self):
        schema = make_plugin_arg_schema(_email_plugin())()
        assert schema.load({"email_address": ""}) == {"email_address": "ops@example.com"}

    def test_whitespace_only_counts_as_blank(self):
        schema = make_plugin_arg_schema(_email_plugin())()
        assert schema.load({"email_address": "   "}) == {"email_address": "ops@example.com"}

    def test_a_real_value_still_wins(self):
        schema = make_plugin_arg_schema(_email_plugin())()
        assert schema.load({"email_address": "me@example.com"}) == {"email_address": "me@example.com"}

    def test_required_field_without_a_default_still_rejects_a_blank(self):
        # The hook must not become a blanket "empty is fine": a field with
        # nothing to fall back to is exactly the one that needs the 422.
        schema = make_plugin_arg_schema(_email_plugin(default=""))()
        with pytest.raises(ValidationError) as exc:
            schema.load({"email_address": ""})
        assert "email_address" in exc.value.messages

    def test_blank_number_takes_its_default_instead_of_failing_to_parse(self):
        # Dropping the key ahead of ``load()`` is what makes this type-agnostic:
        # ``fields.Integer`` never sees the empty string it cannot parse.
        plugin = _FakePlugin([PluginField(key="count", label="Count", field_type="number", default="5", step="1")])
        schema = make_plugin_arg_schema(plugin)()
        assert schema.load({"count": ""}) == {"count": 5}

    def test_blank_select_takes_its_default_rather_than_failing_oneof(self):
        plugin = _FakePlugin(
            [PluginField(key="mode", label="Mode", field_type="select", options=["a", "b"], default="b")]
        )
        schema = make_plugin_arg_schema(plugin)()
        assert schema.load({"mode": ""}) == {"mode": "b"}

    def test_a_blank_loads_identically_to_an_omitted_key(self):
        # The whole mechanism is "a blank means the caller said nothing", so
        # the two inputs must not diverge anywhere — including marshmallow's
        # rule that a ``load_default`` skips the field's validators, which is
        # why an off-list default survives both loads rather than only one.
        plugin = _FakePlugin(
            [PluginField(key="mode", label="Mode", field_type="select", options=["a", "b"], default="zzz")]
        )
        schema = make_plugin_arg_schema(plugin)()
        assert schema.load({"mode": ""}) == schema.load({}) == {"mode": "zzz"}


class TestNormalizeAppliesDefaultToBlank:
    """``normalize_field_values`` — the CLI and the schema-less settings path."""

    def test_missing_key_takes_the_default(self):
        values: dict = {}
        normalize_field_values(_email_plugin(), values)
        assert values == {"email_address": "ops@example.com"}

    def test_blank_value_takes_the_default(self):
        values = {"email_address": "   "}
        normalize_field_values(_email_plugin(), values)
        assert values == {"email_address": "ops@example.com"}

    def test_a_real_value_still_wins(self):
        values = {"email_address": " me@example.com "}
        normalize_field_values(_email_plugin(), values)
        assert values == {"email_address": "me@example.com"}

    def test_a_declared_default_satisfies_required(self):
        # Before this, a saved settings map that never carried the key raised
        # "Email Address is required." from the Auto-Find export path.
        normalize_field_values(_email_plugin(), {})

    def test_required_field_without_a_default_still_raises(self):
        with pytest.raises(ValueError, match="Email Address is required"):
            normalize_field_values(_email_plugin(default=""), {"email_address": ""})

    def test_default_goes_through_template_substitution(self):
        plugin = _FakePlugin(
            [
                PluginField(
                    key="filepath",
                    label="File path",
                    field_type="text",
                    default="results_{YYYY}.json",
                    template_vars=("YYYY",),
                )
            ]
        )
        values: dict = {}
        normalize_field_values(plugin, values)
        assert values["filepath"].startswith("results_2")
        assert "{YYYY}" not in values["filepath"]

    def test_default_goes_through_the_url_validator(self):
        # The default is filled in *before* the security pass, so a plugin
        # cannot smuggle an unvalidated destination in through it.
        plugin = _FakePlugin([PluginField(key="url", label="URL", field_type="url", default="file:///etc/passwd")])
        with pytest.raises(ValueError, match="http or https"):
            normalize_field_values(plugin, {"url": ""})

    def test_non_string_input_is_left_alone(self):
        # ``0`` and ``False`` are input, not silence — only a missing or
        # blank-string value falls back.
        plugin = _FakePlugin([PluginField(key="count", label="Count", field_type="number", default="5", step="1")])
        values = {"count": 0}
        normalize_field_values(plugin, values)
        assert values == {"count": 0}

    def test_is_idempotent(self):
        values = {"email_address": ""}
        normalize_field_values(_email_plugin(), values)
        once = dict(values)
        normalize_field_values(_email_plugin(), values)
        assert values == once


class TestCliPresenceCheckDefersToDefaults:
    """``PluginBase.validate_cli_field_values`` — the CLI ingress."""

    def test_explicitly_blank_flag_takes_the_default(self):
        # ``--email-address ""`` used to fail where omitting the flag
        # succeeded, because the presence check ran ahead of the fallback.
        values = {"email_address": ""}
        _email_plugin().validate_cli_field_values(values)
        assert values == {"email_address": "ops@example.com"}

    def test_required_field_without_a_default_still_names_its_flag(self):
        with pytest.raises(ValueError, match="--email-address"):
            _email_plugin(default="").validate_cli_field_values({})
