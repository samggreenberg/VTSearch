"""Behaviour of ``PluginField.visible_when`` (issue #5343).

``visible_when`` shows a field only while other fields hold particular
values - the canonical case is a checkbox that switches an e-mail box on.
The frontend hides the widget (``frontend/src/app/utils/plugin-fields.ts``),
but hiding is only half of it: a field the user can no longer see must not
block the submit as ``required``, and must not reach the plugin carrying a
value typed before it disappeared.  So every ingress point applies the same
evaluator, :func:`~vtscore.plugins.inactive_field_keys`:

* the per-plugin marshmallow schema (the HTTP routes and converters);
* :func:`~vtscore.plugins.normalize.normalize_field_values` (every path);
* :meth:`PluginBase.validate_cli_field_values` (the CLI);
* the label-importer YAML check and the labelset-source binding's
  missing-field report.

``TestEvaluator`` is the twin of the ``visible_when`` block in
``frontend/src/app/utils/plugin-fields.spec.ts``: the two evaluators must
agree case for case, or the form would hide a field the server still
requires (or the reverse).
"""

from __future__ import annotations

import argparse

import pytest
from marshmallow import ValidationError

from vtscore.plugins import PluginBase, PluginField, inactive_field_keys
from vtscore.plugins.normalize import normalize_field_values
from vtscore.plugins.schema import make_plugin_arg_schema


class _FakePlugin(PluginBase):
    """Bare-minimum stand-in for a real plugin instance."""

    name = "fake"
    display_name = "Fake"

    def __init__(self, fields: list[PluginField]) -> None:
        self.fields = fields


def _enable() -> PluginField:
    return PluginField(key="enable_email", label="Send e-mail", field_type="checkbox", default="false")


def _email(**overrides) -> PluginField:
    kwargs = {
        "key": "email",
        "label": "Address",
        "field_type": "email",
        "required": True,
        "visible_when": {"enable_email": True},
    }
    kwargs.update(overrides)
    return PluginField(**kwargs)


class TestEvaluator:
    def test_unticked_box_switches_the_field_off(self):
        assert inactive_field_keys([_enable(), _email()], {"enable_email": "false"}) == {"email"}

    def test_ticked_box_switches_it_on(self):
        assert inactive_field_keys([_enable(), _email()], {"enable_email": "true"}) == frozenset()

    @pytest.mark.parametrize("ticked", [True, "True", " true "])
    def test_checkbox_reads_through_parse_checkbox(self, ticked):
        assert inactive_field_keys([_enable(), _email()], {"enable_email": ticked}) == frozenset()

    def test_accepted_value_may_be_spelled_as_a_string(self):
        field = _email(visible_when={"enable_email": "True"})
        assert inactive_field_keys([_enable(), field], {"enable_email": True}) == frozenset()

    def test_blank_controller_reads_as_its_default(self):
        on = PluginField(key="enable_email", label="Send", field_type="checkbox", default="true")
        assert inactive_field_keys([on, _email()], {}) == frozenset()
        assert inactive_field_keys([on, _email()], {"enable_email": "  "}) == frozenset()
        assert inactive_field_keys([_enable(), _email()], {}) == {"email"}

    def test_values_listed_under_one_key_are_alternatives(self):
        source = PluginField(key="source", label="Source", field_type="select", options=["s3", "gcs", "local"])
        bucket = PluginField(key="bucket", label="Bucket", field_type="text", visible_when={"source": ["s3", "gcs"]})
        assert inactive_field_keys([source, bucket], {"source": "gcs"}) == frozenset()
        assert inactive_field_keys([source, bucket], {"source": " s3 "}) == frozenset()
        assert inactive_field_keys([source, bucket], {"source": "local"}) == {"bucket"}

    def test_every_key_must_match(self):
        mode = PluginField(key="mode", label="Mode", field_type="select", options=["smtp", "api"])
        port = PluginField(
            key="smtp_port",
            label="Port",
            field_type="number",
            visible_when={"enable_email": True, "mode": "smtp"},
        )
        fields = [_enable(), mode, port]
        assert inactive_field_keys(fields, {"enable_email": "true", "mode": "smtp"}) == frozenset()
        assert inactive_field_keys(fields, {"enable_email": "false", "mode": "smtp"}) == {"smtp_port"}
        assert inactive_field_keys(fields, {"enable_email": "true", "mode": "api"}) == {"smtp_port"}

    def test_switched_off_controller_counts_as_its_default(self):
        """``format`` is off while e-mail is off, so the plugin gets its
        default - and ``subject`` follows that, not the stale ``"html"``."""
        fmt = PluginField(
            key="format", label="Format", field_type="select", default="text", visible_when={"enable_email": True}
        )
        subject = PluginField(key="subject", label="Subject", field_type="text", visible_when={"format": "html"})
        fields = [_enable(), fmt, subject]
        assert inactive_field_keys(fields, {"enable_email": "false", "format": "html"}) == {"format", "subject"}
        assert inactive_field_keys(fields, {"enable_email": "true", "format": "html"}) == frozenset()

    def test_hidden_field_can_control_a_visible_one(self):
        tier = PluginField(key="tier", label="Tier", field_type="text", hidden=True, default="pro")
        extra = PluginField(key="extra", label="Extra", field_type="text", visible_when={"tier": "pro"})
        assert inactive_field_keys([tier, extra], {}) == frozenset()

    def test_key_naming_no_field_is_ignored(self):
        typo = _email(visible_when={"enabel_email": True})
        assert inactive_field_keys([_enable(), typo], {"enable_email": "false"}) == frozenset()

    def test_cycle_settles(self):
        a = PluginField(key="a", label="A", field_type="text", default="x", visible_when={"b": "y"})
        b = PluginField(key="b", label="B", field_type="text", default="y", visible_when={"a": "x"})
        assert inactive_field_keys([a, b], {}) == frozenset()

    def test_plugin_without_conditions_has_nothing_inactive(self):
        assert inactive_field_keys([_enable()], {"enable_email": "false"}) == frozenset()


class TestWire:
    def test_empty_by_default(self):
        assert PluginField(key="k", label="K", field_type="text").to_dict()["visible_when"] == {}

    def test_values_serialise_as_string_lists(self):
        """The frontend reads one shape: ``{key: [str, ...]}``."""
        field = PluginField(
            key="k", label="K", field_type="text", visible_when={"enable_email": True, "mode": ["smtp", "api"]}
        )
        assert field.to_dict()["visible_when"] == {"enable_email": ["true"], "mode": ["smtp", "api"]}


class TestSchema:
    """The HTTP path: the per-plugin marshmallow schema."""

    def _load(self, fields: list[PluginField], body: dict) -> dict:
        return make_plugin_arg_schema(_FakePlugin(fields))().load(body)

    def test_inactive_required_field_may_be_omitted(self):
        assert self._load([_enable(), _email()], {"enable_email": "false"})["email"] == ""

    def test_active_required_field_must_be_present(self):
        with pytest.raises(ValidationError) as exc:
            self._load([_enable(), _email()], {"enable_email": "true"})
        assert exc.value.messages == {"email": ["Missing data for required field."]}

    def test_active_required_field_may_not_be_blank(self):
        with pytest.raises(ValidationError) as exc:
            self._load([_enable(), _email()], {"enable_email": "true", "email": "  "})
        assert "email" in exc.value.messages

    def test_active_field_loads_its_value(self):
        loaded = self._load([_enable(), _email()], {"enable_email": "true", "email": "a@b.c"})
        assert loaded["email"] == "a@b.c"

    def test_inactive_field_value_is_dropped(self):
        loaded = self._load([_enable(), _email()], {"enable_email": "false", "email": "a@b.c"})
        assert loaded["email"] == ""

    def test_stale_select_choice_is_not_judged(self):
        """A choice the user left behind when the field vanished cannot fail ``OneOf``."""
        fmt = PluginField(
            key="format",
            label="Format",
            field_type="select",
            options=["text", "html"],
            default="text",
            visible_when={"enable_email": True},
        )
        loaded = self._load([_enable(), fmt], {"enable_email": "false", "format": "bogus"})
        assert loaded["format"] == "text"

    def test_stale_number_is_not_range_checked(self):
        port = PluginField(
            key="port", label="Port", field_type="number", min="1", max="65535", visible_when={"enable_email": True}
        )
        loaded = self._load([_enable(), port], {"enable_email": "false", "port": "0"})
        assert "port" not in loaded

    def test_unconditional_required_field_is_unchanged(self):
        plain = PluginField(key="name", label="Name", field_type="text", required=True)
        with pytest.raises(ValidationError) as exc:
            self._load([_enable(), _email(), plain], {"enable_email": "false"})
        assert exc.value.messages == {"name": ["Missing data for required field."]}


class TestNormalize:
    """Every path: the plugin never sees a value its user could no longer see."""

    def test_inactive_field_is_reset_to_blank(self):
        values: dict[str, object] = {"enable_email": "false", "email": "a@b.c"}
        normalize_field_values(_FakePlugin([_enable(), _email()]), values)
        assert values["email"] == ""

    def test_inactive_field_is_reset_to_its_default(self):
        field = _email(default="ops@example.com")
        values: dict[str, object] = {"enable_email": "false", "email": "a@b.c"}
        normalize_field_values(_FakePlugin([_enable(), field]), values)
        assert values["email"] == "ops@example.com"

    def test_inactive_required_field_is_not_required(self):
        normalize_field_values(_FakePlugin([_enable(), _email()]), {"enable_email": "false"})

    def test_active_required_field_is_still_required(self):
        with pytest.raises(ValueError, match="Address is required"):
            normalize_field_values(_FakePlugin([_enable(), _email()]), {"enable_email": "true"})

    def test_inactive_url_is_not_ssrf_checked(self):
        """The stale value is discarded before validation, not validated and kept."""
        hook = PluginField(key="hook", label="Hook", field_type="url", visible_when={"enable_email": True})
        values: dict[str, object] = {"enable_email": "false", "hook": "http://127.0.0.1/admin"}
        normalize_field_values(_FakePlugin([_enable(), hook]), values)
        assert values["hook"] == ""

    def test_inactive_file_arrives_as_none(self):
        upload = PluginField(key="attachment", label="File", field_type="file", visible_when={"enable_email": True})
        values: dict[str, object] = {"enable_email": "false", "attachment": object()}
        normalize_field_values(_FakePlugin([_enable(), upload]), values)
        assert values["attachment"] is None

    def test_inactive_checkbox_takes_its_default(self):
        cc = PluginField(
            key="cc_me", label="CC me", field_type="checkbox", default="true", visible_when={"enable_email": True}
        )
        values: dict[str, object] = {"enable_email": False, "cc_me": False}
        normalize_field_values(_FakePlugin([_enable(), cc]), values)
        assert values["cc_me"] == "true"

    def test_idempotent(self):
        plugin = _FakePlugin([_enable(), _email(default="ops@example.com")])
        values: dict[str, object] = {"enable_email": "false", "email": "a@b.c"}
        normalize_field_values(plugin, values)
        once = dict(values)
        normalize_field_values(plugin, values)
        assert values == once


class TestCli:
    def _parse(self, plugin: PluginBase, argv: list[str]) -> dict:
        parser = argparse.ArgumentParser()
        plugin.add_cli_arguments(parser)
        return vars(parser.parse_args(argv))

    def test_switched_off_field_is_not_a_missing_argument(self):
        plugin = _FakePlugin([_enable(), _email()])
        values = self._parse(plugin, [])
        plugin.validate_cli_field_values(values)
        assert values["email"] == ""

    def test_switched_on_field_is_required(self):
        plugin = _FakePlugin([_enable(), _email()])
        with pytest.raises(ValueError, match="--email"):
            plugin.validate_cli_field_values(self._parse(plugin, ["--enable-email"]))

    def test_switched_on_field_passes_through(self):
        plugin = _FakePlugin([_enable(), _email()])
        values = self._parse(plugin, ["--enable-email", "--email", "a@b.c"])
        plugin.validate_cli_field_values(values)
        assert values["email"] == "a@b.c"


class TestOtherRequiredChecks:
    def test_label_importer_yaml_check_skips_inactive_fields(self):
        from vtscore.cli_pipeline import _require_fields

        plugin = _FakePlugin([_enable(), _email()])
        _require_fields(plugin, "fake", {"enable_email": False})
        with pytest.raises(ValueError, match="email"):
            _require_fields(plugin, "fake", {"enable_email": True})

    def test_labelset_binding_does_not_report_inactive_fields_missing(self):
        from vtscore.labels.binding import _missing_required

        source = _FakePlugin([_enable(), _email()])
        assert _missing_required(source, {"enable_email": "false"}) == ()  # type: ignore[arg-type]
        assert _missing_required(source, {"enable_email": "true"}) == ("email",)  # type: ignore[arg-type]
