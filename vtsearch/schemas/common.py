"""Validators and base schemas shared across :mod:`vtsearch.schemas`.

Two things live here, both of which were previously copy-pasted between
sibling schema modules:

* :func:`list_of_strings` — the "reject non-``str`` list items at the schema
  layer" validator, formerly duplicated byte-for-byte in ``datasets.py`` and
  ``detectors.py``.
* :class:`PluginExtrasSchema` — the ``unknown = INCLUDE`` + ``post_dump``
  re-merge pair that lets a plugin descriptor's undeclared keys reach the
  client, formerly written out three times.

Deliberately *not* here: a shared base for ``unknown`` on its own.  That
setting is a per-schema semantic decision rather than boilerplate — see the
class docstring below for why making it a one-word inheritance choice would
be a bad trade.
"""

from __future__ import annotations

from marshmallow import INCLUDE, Schema, ValidationError, post_dump


def list_of_strings(value: object) -> None:
    """Validator: *value* must be a ``list`` whose every entry is a ``str``.

    Used by the dataset- and detector-readers ACL endpoints so that numeric
    or otherwise non-string items are rejected at the schema layer (422)
    rather than silently coerced to strings by ``fields.String``'s
    deserializer.
    """
    if not isinstance(value, list) or not all(isinstance(r, str) for r in value):
        raise ValidationError("Must be a list of strings.")


class PluginExtrasSchema(Schema):
    """Base for a descriptor schema whose plugin may append keys of its own.

    Enumerating the fixed base keys gives the generated OpenAPI client real
    types; the ``unknown = INCLUDE`` / :meth:`_include_plugin_extras` pair
    then lets a concrete plugin's own keys through on both load *and* dump.
    Both halves are required: ``unknown`` is a **load-side** setting, and a
    declared schema drops undeclared keys on ``dump`` no matter what it says.
    The resulting ``additionalProperties: true`` is what gives the generated
    model its index signature.

    **Subclass this only for payloads whose extra keys are all safe to
    serve.**  The strict-dump behaviour it opts out of is a security
    boundary elsewhere in this package: it is the last line of defence
    keeping a media dict's embedding vectors out of a response (see
    ``vtsearch.schemas.detectors._HitSchema``, which documents in-place that
    a passthrough must never be added there).  Reach for this base when the
    payload is a plugin's own JSON-serialisable descriptor or outcome dict,
    not when it wraps a media entry.
    """

    class Meta:
        unknown = INCLUDE

    @post_dump(pass_original=True)
    def _include_plugin_extras(self, data: dict, original: object, **_: object) -> dict:
        if isinstance(original, dict):
            for key, value in original.items():
                if key not in data:
                    data[key] = value
        return data
