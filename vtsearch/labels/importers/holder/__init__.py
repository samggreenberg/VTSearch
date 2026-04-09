"""Holder label importer — import labels from a Holder package.

Given a *holderID*, this importer reads the ``Good`` and ``Bad`` folders
from the Holder package.  Each folder contains ``contentID`` entries with
per-entry metadata: ``{mediaID, md5, media_url, media_type}``.

The returned label dicts include:

- ``md5`` and ``label`` — for standard VTSearch label matching.
- ``origin`` — reconstructed to match the ReCaller importer's per-media
  origin format, enabling origin-based matching::

      {"importer": "recaller", "params": {"contentID": "...", ...}}

- ``origin_name`` — set to the ``contentID``.
- ``metadata`` — carries ``{contentID, mediaID, media_url, media_type}``
  so that these fields survive re-export (e.g. back to Holder).
"""

from __future__ import annotations

from typing import Any

from vtsearch.labels.importers.base import LabelImporter, LabelImporterField


# ---------------------------------------------------------------------------
# TODO(dev): Implement the Holder client functions below.
# ---------------------------------------------------------------------------

def _holder_read_folder(holder_id: str, folder_name: str) -> list[dict[str, Any]]:
    """Read all entries from a folder in a Holder package.

    Returns a list of dicts, each with at least::

        {
            "contentID": str,
            "mediaID": str,      # from stored metadata
            "md5": str,          # from stored metadata
            "media_url": str,    # from stored metadata
            "media_type": str,   # from stored metadata
        }
    """
    raise NotImplementedError("TODO: implement Holder API client")


# ---------------------------------------------------------------------------
# Importer
# ---------------------------------------------------------------------------

class HolderLabelImporter(LabelImporter):
    """Import labels from a Holder package (Good / Bad folders)."""

    name = "holder"
    display_name = "Holder Package"
    description = "Import Good/Bad labels from a Holder package by its ID."
    icon = "\U0001f4e6"  # package
    fields = [
        LabelImporterField(
            key="holder_id",
            label="Holder ID",
            field_type="text",
            description="The ID of the Holder package containing Good/Bad folders.",
        ),
    ]

    def run(self, field_values: dict[str, Any]) -> list[dict[str, str]]:
        holder_id = (field_values.get("holder_id") or "").strip()
        if not holder_id:
            raise ValueError("A Holder ID is required.")

        labels: list[dict[str, Any]] = []

        for folder_name, label_value in [("Good", "good"), ("Bad", "bad")]:
            entries = _holder_read_folder(holder_id, folder_name)
            for entry in entries:
                labels.append(_entry_to_label(entry, label_value))

        return labels


def _entry_to_label(entry: dict[str, Any], label: str) -> dict[str, Any]:
    """Convert a Holder entry into a VTSearch label dict.

    Reconstructs the origin to match the ReCaller importer format so
    that origin-based matching works when importing into an RC-loaded
    dataset.
    """
    content_id = entry.get("contentID", "")
    media_id = entry.get("mediaID", "")
    media_url = entry.get("media_url", "")
    media_type = entry.get("media_type", "")
    md5 = entry.get("md5", "")

    # Reconstruct origin matching ReCaller importer's per-media format
    origin = {
        "importer": "recaller",
        "params": {
            "contentID": content_id,
            "mediaID": media_id,
            "media_url": media_url,
            "media_type": media_type,
        },
    }

    # Metadata for round-tripping through re-export
    metadata = {
        "contentID": content_id,
        "mediaID": media_id,
        "md5": md5,
        "media_url": media_url,
        "media_type": media_type,
    }

    return {
        "md5": md5,
        "label": label,
        "origin": origin,
        "origin_name": content_id,
        "metadata": metadata,
    }


LABEL_IMPORTER = HolderLabelImporter()
