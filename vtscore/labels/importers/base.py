"""Base classes for Label Importers.

To add a new label importer, subclass :class:`LabelImporter`, define its class
attributes and :meth:`~LabelImporter.run`, then expose a module-level
``LABEL_IMPORTER`` instance from a package under this directory.  The registry
will discover it automatically.

Each importer also supports CLI usage via :meth:`~LabelImporter.add_cli_arguments`
and :meth:`~LabelImporter.run_cli`.  The base class provides default
implementations that derive CLI arguments from the :attr:`fields` list, so most
importers work on the command line without any extra code.  ``field_type="file"``
values arrive at :meth:`run` as a
:class:`~vtscore.plugins.uploads.UploadedFile` regardless of the
ingress path (Flask requests pass a Werkzeug ``FileStorage`` straight
through; CLI invocations wrap the path argument in
:class:`~vtscore.plugins.uploads.CliUploadedFile`), so plugin bodies
never mention Werkzeug.

The label format used throughout is a list of dicts::

    [{"md5": "<media-md5>", "label": "good"}, ...]

where ``label`` is ``"good"`` or ``"bad"``.

Example – a minimal database label importer skeleton::

    # vtscore/labels/importers/postgres/__init__.py
    from vtscore.labels.importers.base import LabelImporter, PluginField

    class PostgresLabelImporter(LabelImporter):
        name         = "postgres"
        display_name = "PostgreSQL Query"
        description  = "Import labels from a PostgreSQL database query."
        icon         = "🐘"
        fields = [
            PluginField("host",     "Hostname", "text"),
            PluginField("database", "Database", "text"),
            PluginField("query",    "SQL Query", "text",
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
picked up the next time you run ``bash scripts/install.sh`` (or any
editable install). pyproject.toml is the single source of truth - deptry
verifies that every imported package is declared there.
"""

from __future__ import annotations

from typing import Any

from vtscore.plugins import PluginBase, PluginField

__all__ = ["LabelImporter", "PluginField"]


def _filename_stem(value: Any) -> str:
    """Return the filename behind a ``file`` field's *value*, without its directory or extension.

    *value* is a plain filename string (the suggested-name route) or an
    upload carrying ``.filename``.  Internal whitespace is collapsed, so the
    result reads as a one-line detector name; ``""`` when there is no name.
    """
    filename = value if isinstance(value, str) else getattr(value, "filename", None)
    if not isinstance(filename, str):
        return ""
    base = filename.replace("\\", "/").rsplit("/", 1)[-1]
    dot = base.rfind(".")
    return " ".join((base[:dot] if dot > 0 else base).split())


class LabelImporter(PluginBase):
    """Abstract base class for label importers.

    Subclass this, set the class-level attributes, implement :meth:`run`,
    and expose a module-level ``LABEL_IMPORTER = YourImporter()`` – the
    registry picks it up automatically.

    The :meth:`run` method must return a list of label dicts in the form::

        [{"md5": "<media-md5>", "label": "good"}, ...]

    where ``label`` is ``"good"`` or ``"bad"``.  An entry may also carry
    ``origin`` / ``origin_name`` alongside (or instead of) ``md5``.  The route
    handler applies these to the vote state (``good_votes`` / ``bad_votes``),
    matching each entry to the medias that share its ``origin`` +
    ``origin_name`` *or* its ``md5`` (the union of both; see
    :func:`vtscore.state.media_lookup.resolve_media_ids`).  Entries that
    match no loaded media are auto-resolved: pulled in from their origins on
    a background task, then labelled.

    CLI support
    -----------
    Label importers are used via the web API (``POST /api/label-importers/import/<name>``).
    From the CLI, ``--import-labels-into <detector>`` (with
    ``--label-importer`` and ``--label-importer-file`` /
    ``--label-importer-field``) runs a label importer and merges its labels
    into that saved detector before an autodetect run scores with it (see
    :func:`vtscore.cli.import_labels_into_detector`).

    The default :meth:`add_cli_arguments` derives ``argparse`` arguments
    from :attr:`fields` and :meth:`run_cli` wraps any
    ``field_type="file"`` CLI argument in
    :class:`~vtscore.plugins.uploads.CliUploadedFile` before delegating
    to :meth:`run`, so plugin bodies written against the
    :class:`~vtscore.plugins.uploads.UploadedFile` surface work
    identically in both code paths.
    """

    #: Abstract family base: no auto-derived metadata, and concrete
    #: subclasses strip ``LabelImporter`` from their class names.
    _is_plugin_family_base = True

    #: Emoji or icon string shown next to the display name in the UI.
    icon: str = "🏷️"
    #: Ordered list of fields the user must fill before importing.
    fields: list[PluginField]

    def run(self, field_values: dict[str, Any]) -> list[dict[str, str]]:
        """Perform the import and return a list of label dicts.

        Args:
            field_values: Mapping of :attr:`PluginField.key` → value.
                Fields with ``field_type="file"`` receive an
                :class:`~vtscore.plugins.uploads.UploadedFile` (Flask
                requests pass a Werkzeug ``FileStorage`` straight
                through; CLI invocations wrap the path argument in
                :class:`~vtscore.plugins.uploads.CliUploadedFile`).
                All other fields receive plain strings.

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

    def default_detector_name(self, field_values: dict[str, Any]) -> str:
        """Return the default name for a detector created from *field_values*.

        The label-importer counterpart of
        :meth:`~vtscore.datasets.importers.base.core.ImporterBase.default_display_name`.
        Subclasses override this to derive a name the generic derivation
        cannot reach -- most usefully, to turn an opaque internal selection
        (an id, a saved-query key) into something human-readable.  The base
        implementation names the detector after the first ``file`` field
        holding a value (the filename without its extension), falling back
        to ``"<display_name> detector"``.

        This runs live while the user fills in the New Detector › Trained
        form: the frontend calls it through ``POST /api/label-importers/
        suggested-name/<name>`` on every field change and prefills the
        Detector name box with the result, which the user may overwrite.
        Keep it cheap and side-effect free; if it must reach a remote
        service to resolve a label, cache the lookup, because it is called
        far more often than :meth:`run`.  A ``file`` field's value arrives
        here as the chosen file's name: the upload itself is only sent with
        the import.
        """
        for field in self.fields:
            if field.field_type != "file":
                continue
            stem = _filename_stem(field_values.get(field.key))
            if stem:
                return stem
        return f"{self.display_name} detector"

    # ------------------------------------------------------------------
    # CLI support
    # ------------------------------------------------------------------

    def run_cli(self, field_values: dict[str, Any]) -> list[dict[str, str]]:
        """Import labels from CLI-provided *field_values*.

        The default implementation wraps any ``field_type="file"`` CLI
        path argument in :class:`~vtscore.plugins.uploads.CliUploadedFile`
        so :meth:`run` sees the same
        :class:`~vtscore.plugins.uploads.UploadedFile` shape it does for
        a Flask request, then delegates.
        """
        from vtscore.plugins.uploads import wrap_cli_file_fields  # noqa: PLC0415

        return self.run(wrap_cli_file_fields(self.fields, field_values))
