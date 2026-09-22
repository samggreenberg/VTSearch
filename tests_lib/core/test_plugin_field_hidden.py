"""Behaviour of ``PluginField.hidden``.

``hidden`` marks a field the plugin author fills, not the user: no GUI form
renders a widget for it and its value comes from ``default``.  That is a
*presentation* flag, and every one of these tests exists to pin the half of
it that is deliberately **not** presentation - because the tempting reading
("hidden means gone") would quietly change what a plugin receives.

So a hidden field still:

* appears in ``to_dict()`` (the frontend needs to know it exists in order
  to skip it, and the value ships to the browser either way - which is why
  the docstring warns it is not a place for a secret);
* keeps its CLI flag, so a scripted or Auto-Find run can override the value
  a deployment fixed;
* is normalised, defaulted and template-substituted exactly like a visible
  field, so the plugin body cannot tell the difference.

The one thing that changes is what a form renders, and that half lives in
the frontend (``frontend/src/app/utils/plugin-fields.ts``).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pytest

from vtscore.plugins import PluginBase, PluginField
from vtscore.plugins.normalize import normalize_field_values


class _FakePlugin(PluginBase):
    """Bare-minimum stand-in for a real plugin instance."""

    name = "fake"
    display_name = "Fake"

    def __init__(self, fields: list[PluginField]) -> None:
        self.fields = fields


class TestHiddenDefault:
    def test_defaults_to_false(self):
        assert PluginField(key="k", label="K", field_type="text").hidden is False

    def test_on_the_wire_when_false(self):
        """Emitted unconditionally, like every other ``PluginField`` key.

        ``tests/api/test_schema_plugin_drift.py`` holds ``ImporterFieldSchema``
        to the same key set, so a payload that dropped this key would take it
        out of the generated TypeScript client too.
        """
        assert PluginField(key="k", label="K", field_type="text").to_dict()["hidden"] is False

    def test_on_the_wire_when_true(self):
        field = PluginField(key="k", label="K", field_type="text", hidden=True)
        assert field.to_dict()["hidden"] is True


class TestHiddenKeepsItsCliFlag:
    """Hiding is a GUI affordance; the CLI is the deliberate escape hatch."""

    def _parse(self, field: PluginField, argv: list[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        _FakePlugin([field]).add_cli_arguments(parser)
        return parser.parse_args(argv)

    def test_flag_is_registered(self):
        field = PluginField(
            key="url_template", label="URL Template", field_type="text", hidden=True, default="https://fixed/"
        )
        assert self._parse(field, []).url_template == "https://fixed/"

    def test_flag_overrides_the_fixed_default(self):
        field = PluginField(
            key="url_template", label="URL Template", field_type="text", hidden=True, default="https://fixed/"
        )
        assert self._parse(field, ["--url-template", "https://other/"]).url_template == "https://other/"


class TestHiddenIsNormalisedLikeAnyOtherField:
    def test_blank_value_falls_back_to_the_default(self):
        """The GUI never sends a hidden field, so this is the live path.

        A form that renders no widget submits nothing for the key; the
        framework's default fallback is what puts the author's value in
        front of the plugin body.
        """
        plugin = _FakePlugin(
            [PluginField(key="url_template", label="URL", field_type="text", hidden=True, default="https://fixed/")]
        )
        values: dict[str, object] = {}
        normalize_field_values(plugin, values)
        assert values["url_template"] == "https://fixed/"

    def test_template_vars_still_substitute(self):
        plugin = _FakePlugin(
            [
                PluginField(
                    key="url_template",
                    label="URL",
                    field_type="text",
                    hidden=True,
                    default="https://fixed/?y={YYYY}",
                    template_vars=("YYYY",),
                )
            ]
        )
        values: dict[str, object] = {}
        normalize_field_values(plugin, values)
        assert values["url_template"] == f"https://fixed/?y={datetime.now(timezone.utc):%Y}"

    def test_required_hidden_field_with_a_default_validates(self):
        """A default satisfies ``required``, hidden or not.

        This is the pairing the docstring recommends: hide the field and fix
        its value, and nothing downstream sees an empty required field.
        """
        plugin = _FakePlugin(
            [
                PluginField(
                    key="url_template",
                    label="URL",
                    field_type="text",
                    hidden=True,
                    required=True,
                    default="https://fixed/",
                )
            ]
        )
        plugin.validate_cli_field_values({})

    def test_required_hidden_field_without_a_default_still_fails(self):
        """Hiding does not waive ``required`` - it only removes the widget.

        Such a field is fillable from the CLI alone; a GUI submit is
        expected to fail here rather than to silently pass an empty value
        through to the plugin body.
        """
        plugin = _FakePlugin(
            [PluginField(key="url_template", label="URL", field_type="text", hidden=True, required=True)]
        )
        with pytest.raises(ValueError):
            plugin.validate_cli_field_values({})
