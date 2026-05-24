"""Server JSON labelset source — bidirectional sync with a JSON file on the server.

Loads labels from, and saves labels to, a JSON file on the server
filesystem.  The ``filepath`` field supports ``{detector_id}`` and
``{detector_name}`` templates resolved at runtime.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vtscore.config import DATA_DIR
from vtscore.labels.sources.base import LabelsetSource, LabelsetSourceField

if TYPE_CHECKING:
    from vtscore.datasets.labelset import LabelSet


class ServerFileLabelsetSource(LabelsetSource):
    """Sync detector labels with a JSON file on the server filesystem."""

    name = "server_json_file"
    display_name = "Server JSON File"
    description = "Sync detector labels with a JSON file on the server filesystem."
    icon = "\U0001f5a5"  # desktop computer
    fields = [
        LabelsetSourceField(
            key="filepath",
            label="Save to (server path)",
            field_type="server_path",
            description="The JSON file on the server to sync this detector's labels with.",
            hint=("Absolute or relative server path.  Template variables: {detector_id}, {detector_name}."),
            placeholder=f"{DATA_DIR}/labels/{{detector_name}}.labels.json",
            template_vars=("detector_id", "detector_name"),
        ),
    ]

    def load(self, field_values: dict[str, Any]) -> list[dict[str, str]]:
        """Read labels from a JSON file on the server."""
        path = Path(_normalized(self, field_values)["filepath"])
        if not path.exists():
            return []
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        raw = path.read_bytes()
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc

        labels = data.get("labels")
        if not isinstance(labels, list):
            raise ValueError("JSON must contain a top-level 'labels' list.")
        return [entry for entry in labels if isinstance(entry, dict)]

    def load_full(self, field_values: dict[str, Any]) -> LabelSet:
        """Read labels *and* any ``detector_meta`` block into a :class:`LabelSet`."""
        from vtscore.datasets.labelset import LabelSet as _LabelSet

        path = Path(_normalized(self, field_values)["filepath"])
        if not path.exists():
            return _LabelSet()
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        raw = path.read_bytes()
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("JSON must contain an object at the top level.")
        if not isinstance(data.get("labels"), list):
            raise ValueError("JSON must contain a top-level 'labels' list.")
        return _LabelSet.from_dict(data)

    def save(self, labelset: LabelSet, field_values: dict[str, Any]) -> None:
        """Write labels to a JSON file on the server."""
        filepath = Path(_normalized(self, field_values)["filepath"])
        filepath.parent.mkdir(parents=True, exist_ok=True)
        tmp = filepath.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(labelset.to_dict(), indent=2) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, filepath)


def resolve_filepath_for(
    field_values: dict[str, Any],
    *,
    detector_id: str,
    detector_name: str,
) -> str:
    """Resolve the filepath using explicit detector identity values.

    Used by flows that need to resolve a path for a detector other than the
    currently-active one — notably the rename endpoint, which needs to
    resolve both the OLD and NEW paths to detect an orphaned labelset file.
    The framework's per-field normalize pass can't help here because the
    detector identity isn't the active context; this helper does the
    substitution + validation by hand using the same primitives.
    """
    from vtscore.security.path_validation import (
        get_file_access_base_dir,
        sanitize_template_value,
        validate_server_filepath,
    )

    filepath = (field_values.get("filepath") or "").strip()
    if not filepath:
        raise ValueError("A file path is required.")

    filepath = filepath.replace("{detector_id}", sanitize_template_value(detector_id))
    filepath = filepath.replace("{detector_name}", sanitize_template_value(detector_name))
    # The template itself may contain ``../`` even though the substituted
    # values are sanitised, so re-validate the resolved path before any
    # caller opens it.
    return str(validate_server_filepath(filepath, base_dir=get_file_access_base_dir()))


def _normalized(source: ServerFileLabelsetSource, field_values: dict[str, Any]) -> dict[str, Any]:
    """Return *field_values* with the source's declarative knobs applied.

    Sync-source callers (vtscore/labels/sync.py, the route layer when it
    loads the configured source) pass field_values straight from the
    detector config; that bypasses the route-level normalize hook, so we
    apply it here.  Idempotent: callers that already normalized get the
    same dict back.
    """
    from vtscore.plugins.normalize import normalize_field_values

    return normalize_field_values(source, dict(field_values))


LABELSET_SOURCE = ServerFileLabelsetSource()
