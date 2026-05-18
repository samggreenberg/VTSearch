"""Build marshmallow schemas from :class:`PluginField` declarations.

The five plugin-field HTTP routes
(``/api/dataset/import/<importer>``, ``/api/dataset/stage-import/<importer>``,
``/api/label-importers/import/<importer>``,
``/api/detectors/<name>/import-labels/<importer>``,
``/api/detectors/registry/from-labelset/<importer>``) accept a request
body whose shape depends on the named plugin's :attr:`PluginBase.fields`
declaration. Static marshmallow schemas can't express that shape because
each plugin has its own field list — so we build a schema *per plugin*
at request time, cache it on the plugin instance, and validate the
incoming body against it.

The mapping is:

============ =====================================================
field_type   marshmallow field
============ =====================================================
text         :class:`fields.String`
url          :class:`fields.String`
email        :class:`fields.Email`
password     :class:`fields.String`
folder       :class:`fields.String`
server_path  :class:`fields.String`
number       :class:`fields.Integer` (integer-looking)
             / :class:`fields.Float`
select       :class:`fields.String` + :class:`validate.OneOf`
             (static options); plain :class:`fields.String` for
             dynamic-options fields
checkbox     :class:`fields.Boolean` (with string-coercion loader)
file         skipped — file fields are populated from
             ``request.files`` outside the schema
============ =====================================================

Required text-like fields are marshmallow-required *and* validated as
non-empty after stripping whitespace (matching the pre-migration
hand-rolled validator that rejected ``"   "`` as well as ``""``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from marshmallow import Schema, ValidationError, fields, validate

if TYPE_CHECKING:
    from vtsearch.plugins import PluginBase, PluginField


_NON_EMPTY = validate.Length(min=1)


def _non_empty_after_strip(value: object) -> None:
    """Reject empty / whitespace-only strings.

    Marshmallow's :class:`validate.Length` checks the raw value length,
    but the pre-migration behaviour was to reject ``"   "`` as well as
    ``""``. We mirror that by stripping before measuring.
    """
    if isinstance(value, str) and not value.strip():
        raise ValidationError("Field may not be empty.")


def _coerce_checkbox(value: object) -> bool:
    """Accept ``true`` / ``false`` strings (legacy form-encoded source) and bools."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    if isinstance(value, (int, float)):
        return bool(value)
    raise ValidationError("Not a valid boolean.")


def _build_marshmallow_field(pf: PluginField) -> fields.Field | None:
    """Return the marshmallow field for *pf*, or ``None`` to skip it."""
    if pf.field_type == "file":
        # File fields are populated from ``request.files`` after schema load.
        return None

    kwargs: dict = {}
    if pf.required and not pf.default:
        kwargs["required"] = True
    elif pf.default:
        kwargs["load_default"] = pf.default
    else:
        kwargs["load_default"] = ""

    if pf.field_type == "checkbox":
        # ``load_default`` overrides any text-default we set above.
        kwargs["load_default"] = str(pf.default).lower() == "true"
        return fields.Function(deserialize=_coerce_checkbox, **kwargs)

    if pf.field_type == "number":
        kwargs.pop("load_default", None)
        if pf.default:
            try:
                kwargs["load_default"] = int(pf.default) if pf.is_integer_number() else float(pf.default)
            except (TypeError, ValueError):
                # Bad default — let marshmallow raise on missing input.
                kwargs.pop("load_default", None)
        if pf.is_integer_number():
            return fields.Integer(**kwargs)
        return fields.Float(**kwargs)

    if pf.field_type == "select":
        validators: list = []
        if pf.required:
            validators.append(_non_empty_after_strip)
        if pf.options and not pf.dynamic_options:
            validators.append(validate.OneOf([*pf.options, ""] if not pf.required else pf.options))
        if validators:
            kwargs["validate"] = validators
        return fields.String(**kwargs)

    # text / url / email / password / folder / server_path
    # Use fields.String (not fields.Email) for emails — empty strings are
    # acceptable for non-required fields and the frontend / plugin can
    # tighten the validation if it cares about RFC compliance.
    if pf.required:
        kwargs["validate"] = _non_empty_after_strip
    return fields.String(**kwargs)


def make_plugin_arg_schema(plugin: PluginBase) -> type[Schema]:
    """Build a marshmallow ``Schema`` class for *plugin*'s declared fields.

    Unknown keys are kept (``Meta.unknown = "include"``) so callers can
    read pass-through params (``converters``, ``source_specs``,
    ``clipper``, ``embedder``, ``dataset_name``, ``name``) alongside the
    plugin-declared fields without listing them as plugin fields.
    """
    attrs: dict = {}
    for pf in plugin.fields:
        mf = _build_marshmallow_field(pf)
        if mf is not None:
            attrs[pf.key] = mf

    attrs["Meta"] = type("Meta", (), {"unknown": "include"})
    cls_name = f"{type(plugin).__name__}ArgSchema"
    return type(cls_name, (Schema,), attrs)


def get_plugin_arg_schema(plugin: PluginBase) -> Schema:
    """Return a cached :class:`Schema` instance for *plugin*.

    Caches on the plugin instance so we pay the schema-build cost once
    per process. Plugin instances are long-lived (one per registered
    plugin per process), so this is safe.
    """
    cached = getattr(plugin, "_arg_schema_instance", None)
    if cached is not None:
        return cached
    schema_cls = make_plugin_arg_schema(plugin)
    instance = schema_cls()
    plugin._arg_schema_instance = instance  # type: ignore[attr-defined]
    return instance
