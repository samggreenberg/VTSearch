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
            description=(
                "Absolute or relative path to a labels JSON file on the "
                "server.  Supports {detector_id} and {detector_name} templates."
            ),
            placeholder=f"{DATA_DIR}/labels/{{detector_name}}.labels.json",
        ),
    ]

    def load(self, field_values: dict[str, Any]) -> list[dict[str, str]]:
        """Read labels from a JSON file on the server."""
        path = Path(_resolve_filepath(field_values))
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

        path = Path(_resolve_filepath(field_values))
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
        filepath = Path(_resolve_filepath(field_values))
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
    """
    from vtscore.security.path_validation import sanitize_template_value

    filepath = (field_values.get("filepath") or "").strip()
    if not filepath:
        raise ValueError("A file path is required.")

    filepath = filepath.replace("{detector_id}", sanitize_template_value(detector_id))
    filepath = filepath.replace("{detector_name}", sanitize_template_value(detector_name))
    return filepath


def _resolve_filepath(field_values: dict[str, Any]) -> str:
    """Resolve the filepath, expanding template variables.

    Substituted values are sanitized so that an attacker-controlled detector
    name like ``../../etc/passwd`` cannot escape the directory implied by the
    admin-configured template.
    """
    filepath = (field_values.get("filepath") or "").strip()
    if not filepath:
        raise ValueError("A file path is required.")

    if "{detector_id}" in filepath or "{detector_name}" in filepath:
        from vtscore.state.core import get_active_detector_context

        ctx = get_active_detector_context()
        if ctx is not None:
            return resolve_filepath_for(
                field_values,
                detector_id=ctx.detector_id,
                detector_name=ctx.name,
            )

    return filepath


LABELSET_SOURCE = ServerFileLabelsetSource()
