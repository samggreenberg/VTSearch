"""Shared base for bidirectional sync sources.

A *sync source* is a plugin that can both :meth:`load` data from an
external target (like an importer) and :meth:`save` data back to it
(like an exporter).  Concrete subclasses include
:class:`vtsearch.settings_io.sources.base.SettingsSource` (settings
round-trip) and :class:`vtscore.labels.sources.base.LabelsetSource`
(detector labels round-trip).

The two type parameters allow each kind of source to specify what it
returns from ``load()`` and accepts in ``save()`` independently.

Subclasses override :meth:`_do_load` / :meth:`_do_save` (not
``load`` / ``save``).  The public methods are framework-owned
wrappers that run
:func:`~vtscore.plugins.normalize.normalize_field_values` on a copy
of *field_values* before delegating - so subclass bodies trust the
dict they receive is whitespace-stripped, template-resolved, and
URL- / path-validated, the same guarantees the HTTP route layer
already provides.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from vtscore.plugins import PluginBase, PluginField

LoadT = TypeVar("LoadT")
SaveT = TypeVar("SaveT")

__all__ = ["SyncSource"]


class SyncSource(PluginBase, Generic[LoadT, SaveT]):
    """Abstract base class for bidirectional sync sources.

    Subclasses set the standard :class:`PluginBase` class attributes
    (``name``, ``display_name``, ``fields``, …) and override the
    underscored template methods :meth:`_do_load` / :meth:`_do_save`
    (and optionally :meth:`_do_peek_version`).  The public
    :meth:`load` / :meth:`save` / :meth:`peek_version` methods are
    framework wrappers that normalize *field_values* before dispatching
    - see this module's docstring.
    """

    icon: str = "\U0001f504"  # counterclockwise arrows (sync)
    fields: list[PluginField]

    # -- Public API (framework-owned; do not override) -----------------

    def load(self, field_values: dict[str, Any]) -> LoadT:
        """Import data from the source.

        Normalizes *field_values* then delegates to :meth:`_do_load`.
        Subclasses override :meth:`_do_load`, not this method.
        """
        return self._do_load(self._normalize(field_values))

    def save(self, data: SaveT, /, field_values: dict[str, Any]) -> None:
        """Export *data* to the source.

        The first parameter is positional-only so subclasses can name it
        according to what they save (``labelset``, ``settings``, …)
        without breaking the override contract.  Normalizes
        *field_values* then delegates to :meth:`_do_save`.  Subclasses
        override :meth:`_do_save`, not this method.
        """
        self._do_save(data, self._normalize(field_values))

    def peek_version(self, field_values: dict[str, Any]) -> Any | None:
        """Return an opaque token representing the source's current version.

        Used to cheaply detect whether the source has changed since the
        last successful :meth:`load`.  The caller compares the returned
        token against a previously-stashed value; if they differ, a full
        load is due.  Normalizes *field_values* then delegates to
        :meth:`_do_peek_version`.  Subclasses override
        :meth:`_do_peek_version`, not this method.
        """
        try:
            normalized = self._normalize(field_values)
        except Exception:
            # An unresolvable / invalid template should not crash the
            # caller's freshness probe - match the previous "return
            # None on error" contract.
            return None
        return self._do_peek_version(normalized)

    # -- Template methods (subclasses override) -----------------------

    def _do_load(self, field_values: dict[str, Any]) -> LoadT:
        """Subclass hook for :meth:`load`.

        Receives an already-normalized *field_values* dict.

        Raises:
            NotImplementedError: If the subclass has not implemented this.
        """
        raise NotImplementedError(f"{type(self).__name__}._do_load() is not implemented")

    def _do_save(self, data: SaveT, /, field_values: dict[str, Any]) -> None:
        """Subclass hook for :meth:`save`.

        Receives an already-normalized *field_values* dict.

        Raises:
            NotImplementedError: If the subclass has not implemented this.
        """
        raise NotImplementedError(f"{type(self).__name__}._do_save() is not implemented")

    def _do_peek_version(self, field_values: dict[str, Any]) -> Any | None:
        """Subclass hook for :meth:`peek_version`.

        Receives an already-normalized *field_values* dict.  Subclasses
        should return a cheap freshness probe (e.g. ``st_mtime_ns`` for
        a local file, ``ETag`` for an HTTP source).  The default returns
        ``None``, which means *"I can't cheaply check"* - the caller
        falls back to explicit manual sync.
        """
        return None

    # -- Helpers -------------------------------------------------------

    def _normalize(self, field_values: dict[str, Any]) -> dict[str, Any]:
        """Return a normalized copy of *field_values*.

        Caller's dict is not mutated - the copy carries the
        whitespace-stripped, template-resolved, URL- / path-validated
        values.
        """
        from vtscore.plugins.normalize import normalize_field_values  # noqa: PLC0415

        return normalize_field_values(self, dict(field_values))
