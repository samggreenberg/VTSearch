"""Holder labelset exporter — export labels to a Holder package.

This exporter creates a new Holder package, makes ``Good`` and ``Bad``
folders, and writes the ``contentID`` of each labeled media into the
appropriate folder.  Only media that have a ``contentID`` (i.e. those
originally imported via ReCaller) are included — media from other origins
are silently skipped.

For each stored contentID, the exporter also writes per-entry metadata:
``{mediaID, md5, media_url, media_type}``.

Usage
-----
This exporter is invoked from the standard label-export flow:

1. ``GET /api/labels/export?enrich=true`` — build enriched labels.
2. Select columns and exporter in the UI.
3. ``POST /api/exporters/export`` with ``exporter_name="holder"``.

The response includes the newly-created ``holder_id``.

Data flow
---------
The exporter reads the label entries from ``results["labels"]``.  For
each entry it looks for ``contentID`` in:

1. ``entry.get("metadata", {}).get("contentID")``
2. ``entry.get("custom_metadata", {}).get("contentID")``
3. ``entry.get("origin", {}).get("params", {}).get("contentID")``

This means labels that came from a Holder import (which set ``metadata``)
or from an RC-imported dataset (which set ``custom_metadata`` and
``origin``) will both work.
"""

from __future__ import annotations

from typing import Any

from vtsearch.exporters.base import ExporterField, LabelsetExporter


# ---------------------------------------------------------------------------
# TODO(dev): Implement the Holder client functions below.
# ---------------------------------------------------------------------------

def _holder_create_package() -> str:
    """Create a new Holder package and return its *holderID*."""
    raise NotImplementedError("TODO: implement Holder API client")


def _holder_create_folder(holder_id: str, folder_name: str) -> None:
    """Create a named folder inside a Holder package."""
    raise NotImplementedError("TODO: implement Holder API client")


def _holder_write_entry(
    holder_id: str,
    folder_name: str,
    content_id: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write a contentID (with optional metadata) into a Holder folder."""
    raise NotImplementedError("TODO: implement Holder API client")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_content_id(entry: dict[str, Any]) -> str | None:
    """Try to find a contentID in various places on a label entry."""
    # 1. metadata (from HolderLabelImporter round-trip)
    md = entry.get("metadata") or {}
    cid = md.get("contentID")
    if cid:
        return cid

    # 2. custom_metadata (from enriched RC-imported media)
    cm = entry.get("custom_metadata") or {}
    cid = cm.get("contentID")
    if cid:
        return cid

    # 3. origin params (from RC importer origin)
    origin = entry.get("origin") or {}
    cid = origin.get("params", {}).get("contentID")
    return cid or None


def _extract_entry_metadata(entry: dict[str, Any]) -> dict[str, Any]:
    """Collect the per-entry metadata to store in Holder alongside the contentID."""
    sources = [
        entry.get("metadata") or {},
        entry.get("custom_metadata") or {},
        entry.get("origin", {}).get("params", {}),
    ]
    result: dict[str, Any] = {}
    for key in ("mediaID", "md5", "media_url", "media_type"):
        for src in sources:
            val = src.get(key)
            if val:
                result[key] = val
                break
        # md5 is also a top-level label field
        if key == "md5" and "md5" not in result:
            val = entry.get("md5")
            if val:
                result[key] = val
    return result


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------

class HolderLabelsetExporter(LabelsetExporter):
    """Export labels to a new Holder package (Good / Bad folders)."""

    name = "holder"
    display_name = "Holder Package"
    description = "Create a Holder package with Good/Bad folders of contentIDs."
    icon = "\U0001f4e6"  # package
    hidden_from_picker = True  # flip to False once API clients are implemented
    fields: list[ExporterField] = [
        # No user-supplied fields — a new package is created automatically.
        # Add Holder auth/URL fields here when needed:
        # ExporterField(key="holder_url", label="Holder API URL", field_type="text"),
    ]

    def export(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        labels = results.get("labels")
        if not isinstance(labels, list):
            raise ValueError("Expected labels-format results with a 'labels' key.")

        # Create package and folders
        holder_id = _holder_create_package()
        _holder_create_folder(holder_id, "Good")
        _holder_create_folder(holder_id, "Bad")

        exported = 0
        skipped = 0

        for entry in labels:
            content_id = _extract_content_id(entry)
            if not content_id:
                skipped += 1
                continue

            label = entry.get("label", "")
            if label == "good":
                folder = "Good"
            elif label == "bad":
                folder = "Bad"
            else:
                skipped += 1
                continue

            metadata = _extract_entry_metadata(entry)
            _holder_write_entry(holder_id, folder, content_id, metadata=metadata)
            exported += 1

        return {
            "message": (
                f"Created Holder package {holder_id}: "
                f"{exported} entries exported, {skipped} skipped (no contentID or invalid label)."
            ),
            "holder_id": holder_id,
            "exported": exported,
            "skipped": skipped,
        }


EXPORTER = HolderLabelsetExporter()
