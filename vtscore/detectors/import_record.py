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
"""

from __future__ import annotations

import time
from typing import Any

from vtscore.datasets.importers.base.origin import _origin_params
from vtscore.plugins import PluginBase

__all__ = ["build_import_record"]


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
