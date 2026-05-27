"""Base class for Settings Sources.

A settings source is a bidirectional sync target: it can both *load*
settings (like an importer) and *save* them back (like an exporter).
When an active source is configured, settings are auto-imported on
startup and auto-exported whenever they change.

Standalone importers and exporters remain fully functional regardless
of whether a source is active.
"""

from __future__ import annotations

from typing import Any

from vtscore.plugins import PluginField
from vtscore.sync import SyncSource

SettingsSourceField = PluginField

__all__ = ["SettingsSource", "SettingsSourceField"]


class SettingsSource(SyncSource[dict[str, Any], dict[str, Any]]):
    """Abstract base class for settings sources.

    Subclass this, set the class-level attributes, implement
    :meth:`load` and :meth:`save`, and expose a module-level
    ``SETTINGS_SOURCE = YourSource()`` - the registry picks it up
    automatically.

    ``load(field_values)`` returns a dict of settings key-value pairs
    to apply.  ``save(settings_data, field_values)`` persists the full
    settings dict.
    """
