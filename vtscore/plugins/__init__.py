"""Generic plugin registry with auto-discovery.

Provides :class:`PluginRegistry`, a reusable registry that scans a package
directory for sub-packages exposing a sentinel attribute, and
:class:`PluginField` / :class:`PluginBase`, shared base types that eliminate
the duplicated field-dataclass and CLI / serialisation boilerplate across the
plugin families (dataset, datasource and seed importers, results exporters,
label importers, labelset sources, converters, and the app's settings
importers / exporters / sources).

Usage for creating a registry::

    from vtscore.plugins import PluginRegistry

    _registry: PluginRegistry[MyPlugin] = PluginRegistry(
        package="vtscore.exporters",
        sentinel="EXPORTER",
        label="exporter",
    )

    get_exporter    = _registry.get
    list_exporters  = _registry.list

Usage for defining a field / base class::

    from vtscore.plugins import PluginBase, PluginField

    class MyExporter(PluginBase):
        ...

The code lives in three modules, re-exported here so every name in
``__all__`` imports from ``vtscore.plugins`` itself:

- :mod:`vtscore.plugins.field` - :class:`PluginField`, its value types,
  :func:`parse_checkbox` and the ``visible_when`` evaluator
  :func:`inactive_field_keys`.
- :mod:`vtscore.plugins.base` - :class:`PluginBase` and the default-metadata
  derivation it runs on every subclass.
- :mod:`vtscore.plugins.registry` - :class:`PluginRegistry`,
  :func:`make_plugin_registry` and :class:`EntryPointTombstone`.

Those modules import from one another, never from this package root, so the
order of the imports below is not load-bearing.
"""

from __future__ import annotations

from vtscore.concurrency.notifications import Notification, notify
from vtscore.plugins.base import PluginBase
from vtscore.plugins.field import (
    ConditionValue,
    FieldOption,
    FieldType,
    PluginField,
    inactive_field_keys,
    parse_checkbox,
)
from vtscore.plugins.registry import EntryPointTombstone, PluginRegistry, make_plugin_registry

# Not in ``__all__``, but importable from here before the split, so kept.
from vtscore.plugins.registry import T as T

__all__ = [
    "ConditionValue",
    "EntryPointTombstone",
    "FieldOption",
    "FieldType",
    "Notification",
    "PluginBase",
    "PluginField",
    "PluginRegistry",
    "inactive_field_keys",
    "make_plugin_registry",
    "notify",
    "parse_checkbox",
]
