"""Base classes for dataset importers.

To add a new importer, subclass :class:`DatasetImporter` (or, for an importer
that drives its own ingestion and needs none of the source-spec / per-record
machinery, the thinner :class:`ImporterBase`), define its class attributes and
:meth:`~ImporterBase.run`, then expose a module-level ``IMPORTER`` instance
from a package under this directory.  The registry will discover it
automatically.

Each importer also supports CLI usage via :meth:`~ImporterBase.add_cli_arguments`
and :meth:`~ImporterBase.run_cli`.  The base class provides default
implementations that derive CLI arguments from the :attr:`fields` list, so most
importers work on the command line without any extra code.  ``field_type="file"``
values arrive at :meth:`run` as a
:class:`~vtscore.plugins.uploads.UploadedFile` regardless of whether the
import came in via the Flask request path (a Werkzeug ``FileStorage``,
which already satisfies the protocol) or the CLI path (a
:class:`~vtscore.plugins.uploads.CliUploadedFile` wrapping the
``--<field>`` path argument).  Plugin bodies should rely only on
``.filename`` / ``.read()`` / ``.save(dst)`` / ``.stream`` and never
mention Werkzeug.

The base is split across submodules:

- :mod:`~vtscore.datasets.importers.base.core` — :class:`ImporterBase`, the
  thin base every importer shares.
- :mod:`~vtscore.datasets.importers.base.dataset_importer` —
  :class:`DatasetImporter`, which adds the source-spec / converter / ingestion
  pipeline and the ``list_records`` / ``fetch_record`` hooks.
- :mod:`~vtscore.datasets.importers.base.specs` — :class:`SourceSpec` and the
  spec-parsing / converter-ingestion helpers.
- :mod:`~vtscore.datasets.importers.base.origin` — origin-serialisation policy
  helpers and the synthetic dataset-name field.

Example – a minimal SFTP importer skeleton::

    # vtsearch/datasets/importers/sftp/__init__.py
    from vtscore.datasets.importers.base import DatasetImporter, PluginField

    from vtscore.media import all_folder_names

    class SftpImporter(DatasetImporter):
        name         = "sftp"
        display_name = "SFTP Server"
        description  = "Download media files from an SFTP server."
        fields = [
            PluginField("host",       "Hostname",    "text"),
            PluginField("user",       "Username",    "text"),
            PluginField("password",   "Password",    "password"),
            PluginField("path",       "Remote Path", "text"),
            PluginField(
                "media_type", "Media Type", "select",
                options=all_folder_names(),
                default="audio",
            ),
        ]

        def run(self, field_values: dict, medias: dict) -> None:
            import paramiko
            ...  # download files, then call load_dataset_from_folder(...)

    IMPORTER = SftpImporter()

If the importer needs extra packages, add them to
``[project.dependencies]`` in the repo's ``pyproject.toml``. They are
picked up the next time you run ``bash scripts/install.sh`` (or any
editable install). pyproject.toml is the single source of truth; deptry
verifies that every imported package is declared there.
"""

from __future__ import annotations

from vtscore.plugins import PluginField

from .core import ImporterBase
from .dataset_importer import DatasetImporter
from .origin import DATASET_NAME_FIELD_KEY
from .specs import MissingMediaTypeError, PickerView, SourceSpec

__all__ = [
    "DATASET_NAME_FIELD_KEY",
    "DatasetImporter",
    "ImporterBase",
    "MissingMediaTypeError",
    "PickerView",
    "PluginField",
    "SourceSpec",
]
