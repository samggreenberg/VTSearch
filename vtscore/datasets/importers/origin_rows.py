"""A dataset origin as the rows a person reads in a Stats window.

A dataset's origin (``{"importer": ..., "params": {...}}``, see
:meth:`~vtscore.datasets.importers.base.ImporterBase.build_origin`) is a
re-import recipe: every value is the string the importer wrote, so a checkbox
reads ``"false"``, a media type reads ``"face"`` and the ``source_specs`` rows
read as their JSON.  :func:`origin_rows` turns it into labelled rows for display
(the builder both Stats windows share, :func:`vtscore.plugins.origin_rows.param_rows`),
using the importer's own field labels and asking the importer to word each value
(:meth:`~vtscore.datasets.importers.base.ImporterBase.describe_origin_param`).

Display only: nothing here writes an origin, and a reload still reads the raw
params.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from vtscore.plugins import PluginField
from vtscore.plugins.origin_rows import OriginRow, describe_field_value, format_param_key, param_rows

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
        f"{f.label}: {describe_field_value(f, params[f.key])}"
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
        return describe_field_value(field, value)
    if key == "media_type":
        return _type_label(value)
    return value


def importer_display_name(name: str) -> str:
    """The registered importer *name*'s display name, or *name* when it is not registered."""
    from vtscore.datasets.importers import get_importer  # noqa: PLC0415

    importer = get_importer(name) if name else None
    if importer is None:
        return name
    return importer.display_name or name


def _label_for(key: str) -> str:
    """The label of a key that names no field: ``source_specs`` gets :data:`INCLUDE_MEDIA_LABEL`."""
    return INCLUDE_MEDIA_LABEL if key == "source_specs" else format_param_key(key)


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

    def _as_recorded(key: str, value: str) -> str:
        return describe_origin_value(None, key, value)

    shown = {
        key: raw
        for key, raw in params.items()
        if not (key == "media_type" and media_type and raw is not None and _type_id(str(raw)) == _type_id(media_type))
    }
    return param_rows(
        importer.fields if importer is not None else (),
        shown,
        describe=importer.describe_origin_param if importer is not None else _as_recorded,
        label_for=_label_for,
    )
