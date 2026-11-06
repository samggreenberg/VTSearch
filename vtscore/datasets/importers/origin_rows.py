"""A dataset origin as the rows a person reads in a Stats window.

A dataset's origin (``{"importer": ..., "params": {...}}``, see
:meth:`~vtscore.datasets.importers.base.ImporterBase.build_origin`) is a
re-import recipe: every value is the string the importer wrote, so a checkbox
reads ``"false"``, a media type reads ``"face"`` and the ``source_specs`` rows
read as their JSON.  :func:`origin_rows` turns it into labelled rows for display,
using the importer's own field labels and asking the importer to word each value
(:meth:`~vtscore.datasets.importers.base.ImporterBase.describe_origin_param`).

Display only: nothing here writes an origin, and a reload still reads the raw
params.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from vtscore.plugins import PluginField, parse_checkbox

__all__ = [
    "INCLUDE_MEDIA_LABEL",
    "OriginRow",
    "describe_origin_value",
    "describe_source_specs",
    "importer_display_name",
    "origin_rows",
]

#: Label of the ``source_specs`` row: the name the Add Dataset dialog gives the
#: same choice.
INCLUDE_MEDIA_LABEL = "Include media"


@dataclass(frozen=True)
class OriginRow:
    """One origin param as a Stats window lists it.

    *key* is the raw param key; *label* and *value* are what the row shows; and
    *description* is the importer field's description (empty when the key
    names no field), for a tooltip.
    """

    key: str
    label: str
    value: str
    description: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _type_label(type_id: str) -> str:
    """The display name of a media type given as a type id or a folder name."""
    from vtscore.media import get, get_by_folder_name  # noqa: PLC0415

    for lookup in (get, get_by_folder_name):
        try:
            return lookup(type_id).name.strip()
        except KeyError:
            continue
    return type_id


def _type_id(raw: str) -> str:
    """Resolve a folder name (``"images"``) or type id to the type id."""
    from vtscore.media import get_by_folder_name  # noqa: PLC0415

    try:
        return get_by_folder_name(raw).type_id
    except KeyError:
        return raw


def _same_value(given: Any, default: Any) -> bool:
    """Whether *given* is the field default *default*, numbers compared as numbers.

    A converter's params arrive as JSON (``32``, ``0.5``) while field defaults
    are strings (``"32"``), so ``32`` and ``"32.0"`` count as the same value.
    """
    if str(given) == str(default):
        return True
    try:
        return float(given) == float(default)
    except (TypeError, ValueError):
        return False


def _field_value(field: PluginField, value: Any) -> str:
    """*value* of *field*: ``Yes`` / ``No`` for a checkbox, otherwise as given."""
    if field.field_type == "checkbox":
        return "Yes" if parse_checkbox(value) else "No"
    return str(value)


def _describe_converter(name: str, params: Any) -> str:
    """A converter's display name, with the parameters set away from their defaults.

    ``"Images → Faces"``, or ``"Images → Faces (Minimum face size (pixels):
    64)"`` for a spec that changed one.  A hidden field is the converter's own
    business and is left out.  An unregistered converter shows its name.
    """
    from vtscore.converters import get_converter  # noqa: PLC0415

    converter = get_converter(name)
    if converter is None:
        return name
    label = converter.display_name or name
    if not isinstance(params, Mapping):
        return label
    changed = [
        f"{f.label}: {_field_value(f, params[f.key])}"
        for f in converter.fields
        if not f.hidden and f.key in params and not _same_value(params[f.key], f.default)
    ]
    return f"{label} ({', '.join(changed)})" if changed else label


def describe_source_specs(raw: Any) -> str:
    """The ``source_specs`` an import recorded, as a person reads it.

    Each converted row reads as its converter's display name (``"Images →
    Faces"``).  When converted rows sit beside direct ones, each direct row reads
    as its media type with ``(direct)``: ``"Image (direct), Document →
    Images"``.  Rows that are all direct return ``""``: that is how every
    import of the dataset's own type reads, so there is nothing to say.  A value
    that is not a list of rows is returned as given.
    """
    if isinstance(raw, str):
        try:
            specs = json.loads(raw)
        except ValueError:
            return raw
    else:
        specs = raw
    if not isinstance(specs, list) or not all(isinstance(s, Mapping) for s in specs):
        return str(raw)
    if not any(s.get("converter") for s in specs):
        return ""
    parts = []
    for spec in specs:
        converter = str(spec.get("converter") or "")
        if converter:
            parts.append(_describe_converter(converter, spec.get("params")))
        else:
            parts.append(f"{_type_label(str(spec.get('source_type') or ''))} (direct)")
    return ", ".join(parts)


def describe_origin_value(field: PluginField | None, key: str, value: str) -> str:
    """The generic wording of origin param *key*'s *value*; ``""`` hides the row.

    *field* is the importer field *key* names, or ``None`` for a key that names
    none (``source_specs``, an :attr:`~ImporterBase.extra_origin_keys` entry).
    A checkbox reads ``Yes`` / ``No``, ``media_type`` the type's display name,
    and ``source_specs`` as :func:`describe_source_specs` words it.  Anything
    else is returned as recorded.
    """
    if key == "source_specs":
        return describe_source_specs(value)
    if field is not None and field.field_type == "checkbox":
        return _field_value(field, value)
    if key == "media_type":
        return _type_label(value)
    return value


def _format_key(key: str) -> str:
    """A label for a key that names no field: ``dig_archives`` → ``Dig archives``."""
    spaced = key.replace("_", " ")
    return spaced[:1].upper() + spaced[1:]


def importer_display_name(name: str) -> str:
    """The registered importer *name*'s display name, or *name* when it is not registered."""
    from vtscore.datasets.importers import get_importer  # noqa: PLC0415

    importer = get_importer(name) if name else None
    if importer is None:
        return name
    return importer.display_name or name


def origin_rows(origin: Mapping[str, Any] | None, *, media_type: str = "") -> list[OriginRow]:
    """The rows a Stats window lists for *origin*'s params, in recorded order.

    A row's label is the importer field's own label (``"Include subfolders"``,
    not ``recursive``); a key that names no field gets its key spelled out, and
    ``source_specs`` gets :data:`INCLUDE_MEDIA_LABEL`.  The importer words each
    value (:meth:`~ImporterBase.describe_origin_param`); a value worded as ``""``
    is left out, as is an empty one.  So is ``media_type`` when it is
    *media_type*, the dataset's own type, which the window shows already.  An
    origin from an importer that is no longer registered still lists its params,
    worded by :func:`describe_origin_value`.
    """
    if not isinstance(origin, Mapping):
        return []
    params = origin.get("params")
    if not isinstance(params, Mapping):
        return []
    from vtscore.datasets.importers import get_importer  # noqa: PLC0415

    name = str(origin.get("importer") or "")
    importer = get_importer(name) if name else None
    fields = {f.key: f for f in importer.fields} if importer is not None else {}

    rows: list[OriginRow] = []
    for key, raw in params.items():
        value = "" if raw is None else str(raw)
        if not value:
            continue
        if key == "media_type" and media_type and _type_id(value) == _type_id(media_type):
            continue
        field = fields.get(key)
        if importer is not None:
            shown = importer.describe_origin_param(key, value)
        else:
            shown = describe_origin_value(field, key, value)
        if not shown:
            continue
        if field is not None:
            label = field.label
        elif key == "source_specs":
            label = INCLUDE_MEDIA_LABEL
        else:
            label = _format_key(key)
        rows.append(OriginRow(key=key, label=label, value=shown, description=field.description if field else ""))
    return rows
