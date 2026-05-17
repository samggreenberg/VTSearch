"""Base classes for Label Importers.

To add a new label importer, subclass :class:`LabelImporter`, define its class
attributes and :meth:`~LabelImporter.run`, then expose a module-level
``LABEL_IMPORTER`` instance from a package under this directory.  The registry
will discover it automatically.

Each importer also supports CLI usage via :meth:`~LabelImporter.add_cli_arguments`
and :meth:`~LabelImporter.run_cli`.  The base class provides default
implementations that derive CLI arguments from the :attr:`fields` list, so most
importers work on the command line without any extra code.  Importers whose
:meth:`run` expects non-string values (e.g. Werkzeug ``FileStorage`` objects)
should override :meth:`run_cli` to handle the CLI-appropriate types (file paths
as strings).

The label format used throughout is a list of dicts::

    [{"md5": "<media-md5>", "label": "good"}, ...]

where ``label`` is ``"good"`` or ``"bad"``.

Example – a minimal database label importer skeleton::

    # vtsearch/labels/importers/postgres/__init__.py
    from vtsearch.labels.importers.base import LabelImporter, LabelImporterField

    class PostgresLabelImporter(LabelImporter):
        name         = "postgres"
        display_name = "PostgreSQL Query"
        description  = "Import labels from a PostgreSQL database query."
        icon         = "🐘"
        fields = [
            LabelImporterField("host",     "Hostname", "text"),
            LabelImporterField("database", "Database", "text"),
            LabelImporterField("query",    "SQL Query", "text",
                               description="Must return md5 and label columns."),
        ]

        def run(self, field_values: dict) -> list[dict]:
            import psycopg2
            conn = psycopg2.connect(host=field_values["host"],
                                    database=field_values["database"])
            cur = conn.cursor()
            cur.execute(field_values["query"])
            return [{"md5": row[0], "label": row[1]} for row in cur.fetchall()]

    LABEL_IMPORTER = PostgresLabelImporter()

If the importer needs extra packages (e.g. ``psycopg2-binary``), add them
to ``[project.dependencies]`` in the repo's ``pyproject.toml``. They are
picked up the next time you run ``bash scripts/install-cpu.sh`` (or any
editable install). pyproject.toml is the single source of truth — deptry
verifies that every imported package is declared there.
"""

from __future__ import annotations

from typing import Any

from vtsearch.plugins import PluginBase, PluginField

# Backward-compatible alias — existing plugins import ``LabelImporterField``.
LabelImporterField = PluginField

__all__ = ["LabelImporter", "LabelImporterField"]


class LabelImporter(PluginBase):
    """Abstract base class for label importers.

    Subclass this, set the class-level attributes, implement :meth:`run`,
    and expose a module-level ``LABEL_IMPORTER = YourImporter()`` – the
    registry picks it up automatically.

    The :meth:`run` method must return a list of label dicts in the form::

        [{"md5": "<media-md5>", "label": "good"}, ...]

    where ``label`` is ``"good"`` or ``"bad"``.  The route handler applies
    these to the global vote state (``good_votes`` / ``bad_votes``) by
    matching media MD5 hashes.

    CLI support
    -----------
    Label importers are used via the web API (``POST /api/label-importers/import/<name>``).
    From the CLI, labels can be applied indirectly by using the ``label_file``
    processor importer in a settings file for the autodetect workflow.

    The default :meth:`add_cli_arguments` derives ``argparse`` arguments from
    :attr:`fields` and :meth:`run_cli` delegates to :meth:`run`.  Override
    either when the defaults are insufficient (e.g. when :meth:`run` expects a
    Werkzeug ``FileStorage`` rather than a plain file path).
    """

    #: Emoji or icon string shown next to the display name in the UI.
    icon: str = "🏷️"
    #: Ordered list of fields the user must fill before importing.
    fields: list[PluginField]

    def run(self, field_values: dict[str, Any]) -> list[dict[str, str]]:
        """Perform the import and return a list of label dicts.

        Args:
            field_values: Mapping of :attr:`LabelImporterField.key` → value.
                Fields with ``field_type="file"`` receive a Werkzeug
                :class:`~werkzeug.datastructures.FileStorage` object; all
                other fields receive plain strings.

        Returns:
            A list of dicts, each with ``"md5"`` and ``"label"`` keys.
            ``label`` must be ``"good"`` or ``"bad"``; any other value will
            be skipped by the route handler.

        Raises:
            NotImplementedError: If the subclass has not implemented this.
            Exception: Any exception propagates to the route handler, which
                returns it as a 500 JSON error.
        """
        raise NotImplementedError(f"{type(self).__name__}.run() is not implemented")

    # ------------------------------------------------------------------
    # CLI support
    # ------------------------------------------------------------------

    def run_cli(self, field_values: dict[str, Any]) -> list[dict[str, str]]:
        """Import labels from CLI-provided *field_values*.

        The default implementation simply delegates to :meth:`run`, which
        works for importers whose ``run()`` only expects plain string values.
        Importers that expect non-string objects (e.g. ``FileStorage``) must
        override this method to handle file-path strings appropriately.
        """
        return self.run(field_values)
