"""How a detector was imported: the label importer, the fields it ran with, and when (#5334).

A detector created from a label importer - the Dashboard's New Detector ›
Trained (``POST /api/detectors/registry/from-labelset/<importer>``) or the CLI's
``--import-labels-into <name> --create-detector`` - keeps that record as
``imported`` on its registry entry, the way a dataset keeps its importer's
origin as ``source``.  A detector made any other way has none.

The fields kept follow a dataset origin's policy (shared through
:func:`~vtscore.datasets.importers.base.origin._origin_params`): ``"file"`` and
``"password"`` fields are left out unless the importer opts them in with
:attr:`~vtscore.plugins.PluginField.include_in_origin`, so neither an upload
nor a secret is ever written to the registry.

:func:`import_rows` reads the record back as the rows the detector Stats
window's Import section lists (#5620): the importer's own field labels, each
value worded by its :meth:`~vtscore.labels.importers.base.LabelImporter.describe_origin_param`,
the way a dataset's origin reads in its Stats window
(:func:`vtscore.datasets.importers.origin_rows.origin_rows`).
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from vtscore.datasets.importers.base.origin import _origin_params
from vtscore.labels.importers import get_label_importer
from vtscore.plugins import PluginBase
from vtscore.plugins.origin_rows import OriginRow, param_rows

__all__ = ["build_import_record", "import_rows", "label_importer_display_name"]


def build_import_record(
    importer: PluginBase,
    field_values: dict[str, Any],
    *,
    imported_at: float | None = None,
) -> dict[str, Any]:
    """The record of *importer* creating a detector from *field_values*.

    Returns ``{"importer": <name>, "params": {<field key>: <str>}, "imported_at":
    <unix time>}``; *imported_at* defaults to now.
    """
    return {
        "importer": importer.name,
        "params": _origin_params(importer.fields, field_values),
        "imported_at": time.time() if imported_at is None else imported_at,
    }


def label_importer_display_name(name: str) -> str:
    """The registered label importer *name*'s display name, or *name* when it is not registered."""
    importer = get_label_importer(name) if name else None
    if importer is None:
        return name
    return importer.display_name or name


def _as_recorded(_key: str, value: str) -> str:
    return value


def import_rows(imported: Mapping[str, Any] | None) -> list[OriginRow]:
    """The rows the detector Stats window lists for *imported*'s params, in recorded order.

    A row's label is the label importer field's own label (``"Server path"``,
    not ``filepath``); a key that names no field gets its key spelled out.  The
    importer words each value
    (:meth:`~vtscore.labels.importers.base.LabelImporter.describe_origin_param`);
    a value worded as ``""`` is left out, as is an empty one.  A record from an
    importer that is no longer registered still lists its params, as recorded.
    ``None`` (a detector no importer created) has no rows.
    """
    if not isinstance(imported, Mapping):
        return []
    name = str(imported.get("importer") or "")
    importer = get_label_importer(name) if name else None
    if importer is None:
        return param_rows((), imported.get("params"), describe=_as_recorded)
    return param_rows(importer.fields, imported.get("params"), describe=importer.describe_origin_param)
