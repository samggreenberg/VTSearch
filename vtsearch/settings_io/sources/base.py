"""Base class for Settings Sources.

A settings source is a bidirectional sync target: it can both *load*
settings (like an importer) and *save* them back (like an exporter).
When an active source is configured, settings are auto-imported on a
user's first settings access (and re-imported whenever the source's
freshness token changes) and auto-exported whenever they change.

Standalone importers and exporters remain fully functional regardless
of whether a source is active.

The three methods a source implements (``_do_load``, ``_do_save``,
``_do_peek_version``) are called by a sync engine with contracts this
class cannot express: ``peek_version`` sits on the hot path of every
settings read and must be cheap, local edits are protected from upstream
values by a dirty-key set, and no settings file lock is held while your
code runs. See ``docs/EXTENDING-plugins.md`` -> "Adding a Settings
Source" -> "What the sync engine expects of a source" before
implementing one (the engine itself is in ``docs/ARCHITECTURE.md`` ->
"The settings sync engine").
"""

from __future__ import annotations

from typing import Any

from vtscore.plugins import PluginField
from vtscore.sync import SyncSource

__all__ = ["PluginField", "SettingsSource"]


class SettingsSource(SyncSource[dict[str, Any], dict[str, Any]]):
    """Abstract base class for settings sources.

    Subclass this, set the class-level attributes, implement
    :meth:`_do_load` and :meth:`_do_save` (and optionally
    :meth:`_do_peek_version`), and expose a module-level
    ``SETTINGS_SOURCE = YourSource()`` - the registry picks it up
    automatically.  The public :meth:`load` / :meth:`save` /
    :meth:`peek_version` normalize ``field_values`` and then delegate to
    those hooks; do not override the public methods.

    ``_do_load(field_values)`` returns a dict of settings key-value pairs
    to apply.  ``_do_save(settings_data, field_values)`` persists the full
    settings dict.  ``_do_peek_version(field_values)`` returns a cheap
    freshness token, or ``None`` (the default) when the source can't
    cheaply check.
    """

    #: Abstract family base: no auto-derived metadata, and concrete
    #: subclasses strip ``SettingsSource`` from their class names.
    _is_plugin_family_base = True
