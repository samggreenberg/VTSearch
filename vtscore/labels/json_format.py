"""Shared shape checks for the VTSearch label-JSON format.

Both the server-JSON label *importer* and the server-JSON labelset *source*
read the same on-disk shape::

    {"labels": [{"md5": "...", "label": "good"}, ...], "detector_meta": {...}}

The two plugins live in different packages (``vtscore.labels.importers`` and
``vtscore.labels.sources``) but validate that shape identically, and the error
text they raise is user-facing - it surfaces verbatim when someone points a
plugin at the wrong file.  Keeping the check here means the message can't
drift between the two paths, and a third reader of the format gets it for
free.

Parsing the file itself belongs to :func:`vtscore.io.read_server_json`; these
helpers only interpret an already-parsed object.
"""

from __future__ import annotations

from typing import Any

__all__ = ["extract_labels", "require_label_object"]


def require_label_object(data: Any) -> dict[str, Any]:
    """Return *data* as a label-file mapping, or raise :class:`ValueError`.

    Checks the two invariants every reader of the format depends on: the
    top level is an object, and it carries a ``"labels"`` list.  Callers that
    want the entries themselves should use :func:`extract_labels`; this one is
    for callers that go on to read sibling keys (``detector_meta``) out of the
    same object.
    """
    if not isinstance(data, dict):
        raise ValueError("JSON must contain an object at the top level.")
    if not isinstance(data.get("labels"), list):
        raise ValueError("JSON must contain a top-level 'labels' list.")
    return data


def extract_labels(data: Any) -> list[dict[str, str]]:
    """Pull the label entries out of an already-parsed label-file object.

    Non-dict entries are dropped rather than rejected: a hand-edited file with
    a stray ``null`` in the list should import the labels around it instead of
    failing outright.  A missing or non-list ``"labels"`` key *is* an error -
    that is the file being the wrong format, not one bad row.
    """
    return [entry for entry in require_label_object(data)["labels"] if isinstance(entry, dict)]
