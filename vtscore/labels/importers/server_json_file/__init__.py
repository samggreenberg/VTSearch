"""Server JSON label importer – loads labels from a ``.json`` file on the server.

This importer reads a VTSearch label-format JSON file from the server
filesystem (the machine where the Python process is running), as opposed
to the local (browser-upload) importer which receives files uploaded from
the user's browser.  The file format is identical::

    {"labels": [{"md5": "...", "label": "good"}, ...]}

No additional pip packages are required; uses only Python's ``json`` and
``pathlib`` stdlib modules.
"""

from __future__ import annotations

from typing import Any

from vtscore.config import DATA_DIR
from vtscore.io import read_server_json
from vtscore.labels.importers.base import LabelImporter, LabelImporterField


class ServerJsonLabelImporter(LabelImporter):
    """Import labels from a JSON file on the server filesystem.

    The user provides a file path on the server (the machine running the
    Python process).  The file must be a JSON object with a top-level
    ``"labels"`` key whose value is a list of ``{"md5": "...", "label":
    "good"|"bad"}`` dicts.
    """

    name = "server_json_file"
    display_name = "Server JSON File"
    description = "Import labels from a VTSearch-format JSON file on the server filesystem."
    icon = "\U0001f5a5"  # desktop computer
    fields = [
        LabelImporterField(
            key="filepath",
            label="Path or URL",
            field_type="server_path",
            description="A VTSearch labels JSON file on the server to import labels from.",
            placeholder=f"{DATA_DIR}/labels/my_labels.json",
            hint=(
                "Example structure:\n"
                '  {"labels": [\n'
                '    {"md5": "d41d8cd98f00b204e9800998ecf8427e", "label": "good"},\n'
                '    {"md5": "e2fc714c4727ee9395f324cd2e7f331f", "label": "bad"}\n'
                "  ]}"
            ),
        ),
    ]

    def run(self, field_values: dict[str, Any]) -> list[dict[str, str]]:
        """Read and parse the JSON labels file from the server filesystem."""
        data = read_server_json(field_values["filepath"])
        return _extract_labels(data)

    def run_cli(self, field_values: dict[str, Any]) -> list[dict[str, str]]:
        """Load labels from a file-path string (CLI usage)."""
        return self.run(field_values)

    def add_cli_arguments(self, parser: Any) -> None:
        parser.add_argument(
            "--filepath",
            dest="filepath",
            help="Path to a VTSearch labels JSON file on the server.",
            required=False,
        )


def _extract_labels(data: Any) -> list[dict[str, str]]:
    """Pull the labels list out of an already-parsed JSON object."""
    labels = data.get("labels") if isinstance(data, dict) else None
    if not isinstance(labels, list):
        raise ValueError("JSON must contain a top-level 'labels' list.")
    return [entry for entry in labels if isinstance(entry, dict)]


LABEL_IMPORTER = ServerJsonLabelImporter()
