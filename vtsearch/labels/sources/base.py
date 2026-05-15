"""Base class for Labelset Sources.

A labelset source is a bidirectional sync target for detector labels:
it can both *load* labels (like a label importer) and *save* them back.
When a detector has a linked source, labels are auto-imported on load
and auto-exported whenever votes change.

Standalone label importers and exporters remain fully functional
regardless of whether a source is active.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vtsearch.plugins import PluginField
from vtsearch.sync import SyncSource

if TYPE_CHECKING:
    # Imported for the `"LabelSet"` forward reference in the generic
    # subscription below. Ruff can't see string-based references, so the
    # noqa suppresses an otherwise-correct F401.
    from vtsearch.datasets.labelset import LabelSet  # noqa: F401

LabelsetSourceField = PluginField

__all__ = ["LabelsetSource", "LabelsetSourceField"]


class LabelsetSource(SyncSource[list[dict[str, str]], "LabelSet"]):
    """Abstract base class for labelset sources.

    Subclass this, set the class-level attributes, implement
    :meth:`load` and :meth:`save`, and expose a module-level
    ``LABELSET_SOURCE = YourSource()`` — the registry picks it up
    automatically.

    ``load(field_values)`` returns a list of label dicts
    (``{"md5": ..., "label": "good"|"bad"}``).
    ``save(labelset, field_values)`` persists a :class:`LabelSet`.
    """
