"""Base class for Labelset Sources.

A labelset source is a bidirectional sync target for detector labels:
it can both *load* labels (like a label importer) and *save* them back.
When a detector has a linked source, labels are auto-imported on load
and auto-exported whenever votes change.

Standalone label importers and exporters remain fully functional
regardless of whether a source is active.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vtsearch.utils.registry import PluginBase, PluginField

if TYPE_CHECKING:
    from vtsearch.datasets.labelset import LabelSet

LabelsetSourceField = PluginField

__all__ = ["LabelsetSource", "LabelsetSourceField"]


class LabelsetSource(PluginBase):
    """Abstract base class for labelset sources.

    Subclass this, set the class-level attributes, implement :meth:`load`
    and :meth:`save`, and expose a module-level
    ``LABELSET_SOURCE = YourSource()`` — the registry picks it up
    automatically.
    """

    icon: str = "\U0001f504"  # counterclockwise arrows (sync)
    fields: list[PluginField]

    def load(self, field_values: dict[str, Any]) -> list[dict[str, str]]:
        """Import labels from the source.

        Returns:
            A list of label dicts (``{"md5": "...", "label": "good"|"bad"}``).

        Raises:
            NotImplementedError: If the subclass has not implemented this.
        """
        raise NotImplementedError(f"{type(self).__name__}.load() is not implemented")

    def save(self, labelset: LabelSet, field_values: dict[str, Any]) -> None:
        """Export labels to the source.

        Args:
            labelset: The LabelSet to persist.
            field_values: Source configuration (e.g. filepath).

        Raises:
            NotImplementedError: If the subclass has not implemented this.
        """
        raise NotImplementedError(f"{type(self).__name__}.save() is not implemented")
