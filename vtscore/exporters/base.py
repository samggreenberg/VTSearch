"""Base classes for Labelset Exporters.

To add a new exporter, subclass :class:`LabelsetExporter`, define its class
attributes and :meth:`~LabelsetExporter.export`, then expose a module-level
``EXPORTER`` instance from a package under this directory.  The registry will
discover it automatically.

Each exporter also supports CLI usage via :meth:`~LabelsetExporter.add_cli_arguments`
and :meth:`~LabelsetExporter.export_cli`.  The base class provides default
implementations that derive CLI arguments from the :attr:`fields` list, so most
exporters work on the command line without any extra code.  Exporters whose
:meth:`export` expects non-string values should override :meth:`export_cli` to
handle the CLI-appropriate types.

Example – a minimal SFTP exporter skeleton::

    # vtsearch/exporters/sftp/__init__.py
    from vtscore.exporters.base import LabelsetExporter, ExporterField

    class SftpLabelsetExporter(LabelsetExporter):
        name         = "sftp"
        display_name = "SFTP Upload"
        description  = "Upload results JSON to a remote SFTP server."
        icon         = "📡"
        fields = [
            ExporterField("host",     "Hostname",    "text"),
            ExporterField("user",     "Username",    "text"),
            ExporterField("password", "Password",    "password"),
            ExporterField("path",     "Remote Path", "text",
                          default="/results/autodetect.json"),
        ]

        def export(self, results: dict, field_values: dict) -> dict:
            import paramiko
            ...  # connect, write JSON, disconnect
            return {"message": f"Uploaded to {field_values['host']}:{field_values['path']}"}

    EXPORTER = SftpLabelsetExporter()

If the exporter needs extra packages, add them to
``[project.dependencies]`` in the repo's ``pyproject.toml``. They are
picked up the next time you run ``bash scripts/install-cpu.sh`` (or any
editable install). pyproject.toml is the single source of truth - deptry
verifies that every imported package is declared there.
"""

from __future__ import annotations

from typing import Any

from vtscore.plugins import PluginBase, PluginField

# Backward-compatible alias - existing plugins import ``ExporterField``.
ExporterField = PluginField

__all__ = ["ExporterField", "LabelsetExporter"]


class LabelsetExporter(PluginBase):
    """Abstract base class for results exporters.

    Subclass this, set the class-level attributes, implement :meth:`export`,
    and expose a module-level ``EXPORTER = YourExporter()`` – the registry
    picks it up automatically.

    The :meth:`export` method receives the full results dict returned by
    ``/api/auto-detect`` and a flat mapping of field values supplied by the
    user via the UI.  It should return a dict with at minimum a ``"message"``
    key describing what happened (shown to the user as confirmation).
    """

    #: Emoji or icon string shown next to the display name in the UI.
    icon: str = "📤"
    #: Ordered list of fields the user must fill before exporting.
    #: Leave empty if the exporter needs no configuration.
    fields: list[PluginField]

    def export(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        """Perform the export and return a status dict.

        Args:
            results: The full auto-detect results dict from ``/api/auto-detect``.
                     Shape::

                         {
                           "media_type": "audio",
                           "detectors_run": 2,
                           "results": {
                             "detector_name": {
                               "detector_name": "...",
                               "threshold": 0.5,
                               "total_hits": 15,
                               "hits": [{...}, ...]
                             }
                           }
                         }

            field_values: Mapping of :attr:`ExporterField.key` → value string
                supplied by the user.

        Returns:
            A dict that **must** contain a ``"message"`` key with a short
            human-readable confirmation string.  It may also carry arbitrary
            extra keys (e.g. ``"filepath"`` for file-based exporters).

        Raises:
            NotImplementedError: If the subclass has not implemented this.
            Exception: Any exception propagates to the route handler, which
                returns it as a 500 JSON error.
        """
        raise NotImplementedError(f"{type(self).__name__}.export() is not implemented")

    # ------------------------------------------------------------------
    # CLI support
    # ------------------------------------------------------------------

    def export_cli(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        """Export results from CLI-provided *field_values*.

        The default implementation simply delegates to :meth:`export`, which
        works for exporters whose ``export()`` only expects plain string values.
        Exporters that need different behaviour on the command line (e.g. the
        GUI exporter, which has no browser) should override this method.
        """
        return self.export(results, field_values)
