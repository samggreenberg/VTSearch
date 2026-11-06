"""A plugin's recorded params as the rows a person reads in a Stats window.

A dataset importer's origin and a detector's import record both keep the form
a plugin ran with as ``{<field key>: <str>}``
(:func:`~vtscore.datasets.importers.base.origin._origin_params`): every value
is the string the plugin wrote, so a checkbox reads ``"false"`` and a select
reads its option id.  :func:`param_rows` turns such a mapping into labelled
rows for display, using the plugin's own field labels and leaving each value's
wording to a callable the plugin family supplies (its importer's
``describe_origin_param`` hook).  Each family wraps it with its own rules:
:func:`vtscore.datasets.importers.origin_rows.origin_rows` for a dataset's
origin and :func:`vtscore.detectors.import_record.import_rows` for a
detector's import record (#5617, #5620).

Display only: nothing here writes a record, and a reload still reads the raw
params.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from vtscore.plugins.field import PluginField, parse_checkbox

__all__ = ["OriginRow", "describe_field_value", "format_param_key", "param_rows"]


@dataclass(frozen=True)
class OriginRow:
    """One recorded param as a Stats window lists it.

    *key* is the raw param key; *label* and *value* are what the row shows; and
    *description* is the plugin field's description (empty when the key names
    no field), for a tooltip.
    """

    key: str
    label: str
    value: str
    description: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def describe_field_value(field: PluginField | None, value: Any) -> str:
    """*value* of *field*: ``Yes`` / ``No`` for a checkbox, otherwise as given.

    The wording every family's ``describe_origin_param`` default starts from.
    *field* is ``None`` for a key that names no field.
    """
    if field is not None and field.field_type == "checkbox":
        return "Yes" if parse_checkbox(value) else "No"
    return str(value)


def format_param_key(key: str) -> str:
    """A label for a key that names no field: ``dig_archives`` → ``Dig archives``."""
    spaced = key.replace("_", " ")
    return spaced[:1].upper() + spaced[1:]


def param_rows(
    fields: Iterable[PluginField],
    params: Mapping[str, Any] | None,
    *,
    describe: Callable[[str, str], str],
    label_for: Callable[[str], str] = format_param_key,
) -> list[OriginRow]:
    """The rows a Stats window lists for *params*, in recorded order.

    A row's label is the *fields* entry's own label (``"Include subfolders"``,
    not ``recursive``), and its description that field's; a key that names no
    field is labelled by *label_for* (by default its key spelled out) with no
    description.  *describe* words each value from its key and recorded
    string; a value it words as ``""`` is left out, as is an empty one.
    *params* that is not a mapping has no rows.
    """
    if not isinstance(params, Mapping):
        return []
    by_key = {f.key: f for f in fields}
    rows: list[OriginRow] = []
    for key, raw in params.items():
        value = "" if raw is None else str(raw)
        if not value:
            continue
        shown = describe(key, value)
        if not shown:
            continue
        field = by_key.get(key)
        rows.append(
            OriginRow(
                key=key,
                label=field.label if field is not None else label_for(key),
                value=shown,
                description=field.description if field is not None else "",
            )
        )
    return rows
