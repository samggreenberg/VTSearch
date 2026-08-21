"""Origin-serialisation policy helpers and the synthetic dataset-name field.

Pulled out of the importer base so the origin-policy knobs (which fields land
in a persisted origin, how their values stringify) and the user-typeable
dataset-name field live in one small module, independent of the importer
class hierarchy.
"""

from __future__ import annotations

import json
from typing import Any

from vtscore.plugins import PluginField

# Synthetic per-importer field that lets the user pick a name for the new
# dataset.  Appended to the end of every importer's serialised field list
# in :meth:`ImporterBase.to_dict` (just before the Advanced section in
# the UI), so users filling the form top-down have already entered the
# fields that feed the auto-derived default name by the time they reach
# it.  Routed through the per-plugin marshmallow schema as a regular field
# (see :func:`vtsearch.routes._shared.validate_plugin_args`) and read
# downstream by :meth:`ImporterBase.resolve_display_name`.
DATASET_NAME_FIELD_KEY = "dataset_name"


_ORIGIN_EXCLUDED_FIELD_TYPES = frozenset({"file", "password"})


def _field_in_origin(field: PluginField) -> bool:
    """Resolve whether *field*'s value should land in the persisted origin.

    Honors an explicit :attr:`PluginField.include_in_origin`; otherwise
    falls back to the field-type default (file and password fields are
    excluded).
    """
    if field.include_in_origin is not None:
        return field.include_in_origin
    return field.field_type not in _ORIGIN_EXCLUDED_FIELD_TYPES


def _serialise_origin_value(value: Any, serializer: Any) -> str:
    """Serialise *value* for inclusion in an origin ``params`` dict.

    Returns the empty string when *value* is falsy (mirroring the
    pre-refactor ``if val: params[key] = str(val)`` shape).  When
    *serializer* is set it runs first; otherwise list/dict values are
    JSON-encoded so an importer's structured ``field_values`` round-trip
    through the string-only origin contract.
    """
    if not value:
        return ""
    if serializer is not None:
        return str(serializer(value))
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return str(value)


def _dataset_name_field() -> PluginField:
    return PluginField(
        key=DATASET_NAME_FIELD_KEY,
        label="Dataset name",
        field_type="text",
        description="Leave blank to use a default name",
        required=False,
        placeholder="Leave blank to use a default name",
    )
