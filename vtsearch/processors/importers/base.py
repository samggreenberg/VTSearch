"""Base classes for Processor Importers.

To add a new processor importer, subclass :class:`ProcessorImporter`, define its
class attributes and :meth:`~ProcessorImporter.run`, then expose a module-level
``PROCESSOR_IMPORTER`` instance from a package under this directory.  The registry
will discover it automatically.

Each importer also supports CLI usage via :meth:`~ProcessorImporter.add_cli_arguments`
and :meth:`~ProcessorImporter.run_cli`.  The base class provides default
implementations that derive CLI arguments from the :attr:`fields` list, so most
importers work on the command line without any extra code.  Importers whose
:meth:`run` expects non-string values (e.g. Werkzeug ``FileStorage`` objects)
should override :meth:`run_cli` to handle the CLI-appropriate types (file paths
as strings).

The processor data returned by :meth:`run` is a dict with at minimum::

    {
        "media_type": "audio",
        "weights": {"0.weight": [...], "0.bias": [...], ...},
        "threshold": 0.5,
    }

The route handler adds the user-supplied ``name`` and saves the result as a
autorun detector via :func:`~vtsearch.utils.add_autorun_detector`.

Example -- a minimal S3 processor importer skeleton::

    # vtsearch/processors/importers/s3/__init__.py
    from vtsearch.processors.importers.base import ProcessorImporter, ProcessorImporterField

    class S3ProcessorImporter(ProcessorImporter):
        name         = "s3"
        display_name = "S3 Detector File"
        description  = "Download a detector JSON file from an S3 bucket."
        icon         = "☁️"
        fields = [
            ProcessorImporterField("bucket",  "S3 Bucket", "text"),
            ProcessorImporterField("key",     "Object Key", "text"),
        ]

        def run(self, field_values: dict) -> dict:
            import boto3
            s3 = boto3.client("s3")
            ...  # download & parse JSON
            return {"media_type": "audio", "weights": weights, "threshold": 0.5}

    PROCESSOR_IMPORTER = S3ProcessorImporter()

Then create ``vtsearch/processors/importers/s3/requirements.txt`` containing
``boto3``.  It will be auto-discovered and installed by
``install-plugin-deps.sh``.
"""

from __future__ import annotations

from typing import Any

from vtsearch.utils.registry import PluginBase, PluginField

# Backward-compatible alias — existing plugins import ``ProcessorImporterField``.
ProcessorImporterField = PluginField



class ProcessorImporter(PluginBase):
    """Abstract base class for processor importers.

    Subclass this, set the class-level attributes, implement :meth:`run`,
    and expose a module-level ``PROCESSOR_IMPORTER = YourImporter()`` -- the
    registry picks it up automatically.

    The :meth:`run` method must return a dict with at minimum ``media_type``,
    ``weights``, and ``threshold`` keys.  The route handler will combine this
    with the user-supplied name and save it as a autorun detector.

    CLI support
    -----------
    Processor importers are used indirectly from the CLI via the autodetect
    workflow: add a processor recipe to the settings JSON file and run::

        python app.py --autodetect --dataset <file.pkl> --settings <settings.json>

    The default :meth:`add_cli_arguments` derives ``argparse`` arguments from
    :attr:`fields` and :meth:`run_cli` delegates to :meth:`run`.  Override
    either when the defaults are insufficient (e.g. when :meth:`run` expects a
    Werkzeug ``FileStorage`` rather than a plain file path).
    """

    #: Emoji or icon string shown next to the display name in the UI.
    icon: str = "\U0001f9e9"  # puzzle piece
    #: Ordered list of fields the user must fill before importing.
    fields: list[PluginField]

    def run(self, field_values: dict[str, Any]) -> dict[str, Any]:
        """Perform the import and return processor data.

        Args:
            field_values: Mapping of :attr:`ProcessorImporterField.key` to value.
                Fields with ``field_type="file"`` receive a Werkzeug
                :class:`~werkzeug.datastructures.FileStorage` object; all
                other fields receive plain strings.

        Returns:
            A dict with at minimum ``"media_type"`` (str), ``"weights"`` (dict),
            and ``"threshold"`` (float).  May also include ``"name"`` (suggested
            default name), ``"loaded"`` (int), and ``"skipped"`` (int) for
            status reporting.

        Raises:
            NotImplementedError: If the subclass has not implemented this.
            Exception: Any exception propagates to the route handler, which
                returns it as a 500 JSON error.
        """
        raise NotImplementedError(f"{type(self).__name__}.run() is not implemented")

    # ------------------------------------------------------------------
    # CLI support
    # ------------------------------------------------------------------

    def run_cli(self, field_values: dict[str, Any]) -> dict[str, Any]:
        """Import a processor from CLI-provided *field_values*.

        The default implementation simply delegates to :meth:`run`, which
        works for importers whose ``run()`` only expects plain string values.
        Importers that expect non-string objects (e.g. ``FileStorage``) must
        override this method to handle file-path strings appropriately.
        """
        return self.run(field_values)
