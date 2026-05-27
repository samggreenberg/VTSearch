"""Build marshmallow schemas from :class:`PluginField` declarations.

The five plugin-field HTTP routes
(``/api/dataset/import/<importer>``, ``/api/dataset/stage-import/<importer>``,
``/api/label-importers/import/<importer>``,
``/api/detectors/<name>/import-labels/<importer>``,
``/api/detectors/registry/from-labelset/<importer>``) accept a request
body whose shape depends on the named plugin's :attr:`PluginBase.fields`
declaration. Static marshmallow schemas can't express that shape because
each plugin has its own field list - so we build a schema *per plugin*
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
             / :class:`fields.Float`, with :class:`validate.Range`
             attached when :attr:`PluginField.min` / :attr:`max`
             are declared (out-of-range values are rejected at
             schema-load time, not silently coerced)
select       :class:`fields.String` + :class:`validate.OneOf`
             (static options); plain :class:`fields.String` for
             dynamic-options fields
checkbox     :class:`fields.Boolean` (with string-coercion loader)
file         skipped - file fields are populated from
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
    from vtscore.plugins import PluginBase, PluginField


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


def _presence_kwargs(pf: PluginField) -> dict:
    """Return marshmallow ``required`` / ``load_default`` kwargs for *pf*.

    The three states are mutually exclusive:
    - required AND no default → ``required=True`` (marshmallow raises
      on missing input).
    - has default → ``load_default=<default>``.
    - neither → ``load_default=""`` (preserves the legacy ``.get(key, "")``
      fallback).
    """
    if pf.required and not pf.default:
        return {"required": True}
    if pf.default:
        return {"load_default": pf.default}
    return {"load_default": ""}


def _build_checkbox(pf: PluginField, kwargs: dict) -> fields.Field:
    # Checkboxes always have a sensible default (``False`` if the plugin
    # doesn't set one) - drop ``required`` to avoid the
    # ``required + load_default`` conflict marshmallow would raise.
    kwargs.pop("required", None)
    kwargs["load_default"] = str(pf.default).lower() == "true"
    # ``fields.Function`` needs a paired serializer for apispec to
    # describe the field in the OpenAPI spec; identity is fine since
    # these schemas are only ever ``.load()``-ed (request bodies).
    return fields.Function(deserialize=_coerce_checkbox, serialize=lambda v: v, **kwargs)


def _build_number(pf: PluginField, kwargs: dict) -> fields.Field:
    kwargs.pop("load_default", None)
    cast = int if pf.is_integer_number() else float
    if pf.default:
        try:
            kwargs["load_default"] = cast(pf.default)
        except (TypeError, ValueError):
            # Bad default - let marshmallow raise on missing input.
            kwargs.pop("load_default", None)

    range_kwargs: dict = {}
    if pf.min:
        try:
            range_kwargs["min"] = cast(pf.min)
        except (TypeError, ValueError):
            pass
    if pf.max:
        try:
            range_kwargs["max"] = cast(pf.max)
        except (TypeError, ValueError):
            pass
    if range_kwargs:
        kwargs["validate"] = validate.Range(**range_kwargs)

    if pf.is_integer_number():
        return fields.Integer(**kwargs)
    return fields.Float(**kwargs)


def _build_select(pf: PluginField, kwargs: dict) -> fields.Field:
    validators: list = []
    if pf.required:
        validators.append(_non_empty_after_strip)
    if pf.options and not pf.dynamic_options:
        validators.append(validate.OneOf([*pf.options, ""] if not pf.required else pf.options))
    if validators:
        kwargs["validate"] = validators
    return fields.String(**kwargs)


def _build_text(pf: PluginField, kwargs: dict) -> fields.Field:
    # text / url / email / password / folder / server_path
    # Use ``fields.String`` for emails (not ``fields.Email``) - empty
    # strings are acceptable for non-required fields, and the frontend /
    # plugin can tighten validation if it needs RFC compliance.
    if pf.required:
        kwargs["validate"] = _non_empty_after_strip
    return fields.String(**kwargs)


def _build_marshmallow_field(pf: PluginField) -> fields.Field | None:
    """Return the marshmallow field for *pf*, or ``None`` to skip it."""
    if pf.field_type == "file":
        # File fields are populated from ``request.files`` after schema load.
        return None

    kwargs = _presence_kwargs(pf)

    if pf.field_type == "checkbox":
        return _build_checkbox(pf, kwargs)
    if pf.field_type == "number":
        return _build_number(pf, kwargs)
    if pf.field_type == "select":
        return _build_select(pf, kwargs)
    return _build_text(pf, kwargs)


def make_plugin_arg_schema(plugin: PluginBase) -> type[Schema]:
    """Build a marshmallow ``Schema`` class for *plugin*'s declared fields.

    Unknown keys are dropped (``Meta.unknown = "exclude"``) - the schema
    is a faithful description of the plugin's declared field set, not a
    free-form bag.  Route handlers that need to pass extra keys
    alongside the plugin-declared fields (e.g. ``converters``,
    ``source_specs``, ``clipper``, ``embedder``, ``dataset_name``,
    ``name``) declare them explicitly via the ``extra_keys`` argument to
    :func:`vtsearch.routes._shared.validate_plugin_args`.
    """
    attrs: dict = {}
    for pf in plugin.fields:
        mf = _build_marshmallow_field(pf)
        if mf is not None:
            attrs[pf.key] = mf

    attrs["Meta"] = type("Meta", (), {"unknown": "exclude"})
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


def make_plugin_route_schema(
    plugin: PluginBase,
    *,
    extra_keys: tuple[str, ...] = (),
    route_id: str,
) -> type[Schema]:
    """Build a marshmallow ``Schema`` class for a per-(plugin, route) pair.

    Like :func:`make_plugin_arg_schema`, but also declares any
    ``extra_keys`` (pass-through keys the route accepts alongside the
    plugin-declared fields) as :class:`fields.Raw`, and uses
    ``Meta.unknown = "include"`` so any other unknown keys (which the
    fallback parameterized route would have accepted) survive the
    ``schema.load()`` round trip.  The class name is namespaced by
    *route_id* so two routes that target the same plugin don't collide
    as OpenAPI components.
    """
    attrs: dict = {}
    for pf in plugin.fields:
        mf = _build_marshmallow_field(pf)
        if mf is not None:
            attrs[pf.key] = mf
    for key in extra_keys:
        if key not in attrs:
            attrs[key] = fields.Raw(load_default=None)

    attrs["Meta"] = type("Meta", (), {"unknown": "include"})
    cls_name = f"{route_id}_{type(plugin).__name__}_Args"
    return type(cls_name, (Schema,), attrs)
