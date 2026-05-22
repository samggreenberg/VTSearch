"""Picker tab definitions for the dataset-importer modal.

Each importer's ``category`` field references a tab id declared here.  The
frontend renders one tab per declaration, in :attr:`order`, and only shows
tabs that have at least one visible importer.

Extensions that introduce new categories can call :func:`register_picker_tab`
to give the new tab a pretty label and icon.  The frontend falls back to a
title-cased version of the id if no declaration exists, but registering one
yields a nicer result.
"""

from __future__ import annotations

from typing import Any

#: Built-in picker tabs.  ``order`` is ascending (smaller comes first).  The
#: ``icon`` field references a type understood by ``vt-icon`` (see
#: ``frontend/src/app/components/icon/icon.component.ts``).
_PICKER_TABS: list[dict[str, Any]] = [
    {"id": "services", "label": "Services", "icon": "lightning", "order": 10},
    {"id": "server", "label": "Server", "icon": "server", "order": 20},
    {"id": "local", "label": "Local", "icon": "house", "order": 30},
    {"id": "demo", "label": "Demo", "icon": "flask", "order": 40},
]


def list_picker_tabs() -> list[dict[str, Any]]:
    """Return the registered picker tabs sorted by ``order``."""
    return [dict(t) for t in sorted(_PICKER_TABS, key=lambda t: t.get("order", 100))]


def register_picker_tab(tab: dict[str, Any]) -> None:
    """Add or replace a picker tab definition.

    Args:
        tab: A dict with at least ``id`` (matching importer ``category``
            values) and ``label``.  Optional fields: ``icon`` (vt-icon type
            name), ``order`` (int — lower values render first).
    """
    tab_id = tab.get("id")
    if not tab_id:
        raise ValueError("picker tab requires an 'id' field")
    if not tab.get("label"):
        raise ValueError("picker tab requires a 'label' field")
    for i, existing in enumerate(_PICKER_TABS):
        if existing["id"] == tab_id:
            _PICKER_TABS[i] = dict(tab)
            return
    _PICKER_TABS.append(dict(tab))
