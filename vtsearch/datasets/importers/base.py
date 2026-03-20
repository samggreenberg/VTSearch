"""Base classes for dataset importers.

To add a new importer, subclass :class:`DatasetImporter`, define its class
attributes and :meth:`~DatasetImporter.run`, then expose a module-level
``IMPORTER`` instance from a package under this directory.  The registry will
discover it automatically.

Each importer also supports CLI usage via :meth:`~DatasetImporter.add_cli_arguments`
and :meth:`~DatasetImporter.run_cli`.  The base class provides default
implementations that derive CLI arguments from the :attr:`fields` list, so most
importers work on the command line without any extra code.  Importers whose
:meth:`run` expects non-string values (e.g. Werkzeug ``FileStorage`` objects)
should override :meth:`run_cli` to handle the CLI-appropriate types (file paths
as strings).

Example – a minimal SFTP importer skeleton::

    # vtsearch/datasets/importers/sftp/__init__.py
    from vtsearch.datasets.importers.base import DatasetImporter, ImporterField

    from vtsearch.media import all_folder_names

    class SftpImporter(DatasetImporter):
        name         = "sftp"
        display_name = "SFTP Server"
        description  = "Download media files from an SFTP server."
        fields = [
            ImporterField("host",       "Hostname",    "text"),
            ImporterField("user",       "Username",    "text"),
            ImporterField("password",   "Password",    "text"),
            ImporterField("path",       "Remote Path", "text"),
            ImporterField(
                "media_type", "Media Type", "select",
                options=all_folder_names(),
                default="sounds",
            ),
        ]

        def run(self, field_values: dict, medias: dict) -> None:
            import paramiko
            ...  # download files, then call load_dataset_from_folder(...)

    IMPORTER = SftpImporter()

Then add ``-r vtsearch/datasets/importers/sftp/requirements.txt`` to
``requirements-importers.txt`` (creating the file first with ``paramiko``).
"""

from __future__ import annotations

from typing import Any, Iterator

from vtsearch.utils.registry import PluginBase, PluginField

# Backward-compatible alias — existing plugins import ``ImporterField``.
ImporterField = PluginField

__all__ = ["DatasetImporter", "ImporterField"]


class DatasetImporter(PluginBase):
    """Abstract base class for dataset importers.

    Subclass this, set the class-level attributes, implement :meth:`run`,
    and expose a module-level ``IMPORTER = YourImporter()`` – the registry
    picks it up automatically.

    Custom metadata
    ---------------
    Importers can attach arbitrary per-media display metadata by setting
    ``media["custom_metadata"]`` to a ``dict[str, Any]`` mapping
    human-readable labels to values.  For example, an S3 importer might
    set ``{"Uploaded By": "alice", "Bucket": "my-data"}``.  These fields
    are merged with the media type's built-in display fields and rendered
    in the labeling UI.  The dict is also persisted through pickle
    export/import.

    Content vectors
    ---------------
    Some importers provide pre-computed content vectors (embeddings) alongside
    the media files.  To take advantage of this, populate
    :attr:`content_vectors` with a mapping of ``filename`` to
    ``numpy.ndarray`` during :meth:`run`.  When the dataset is later embedded
    (e.g. via :func:`~vtsearch.datasets.loader.load_dataset_from_folder`),
    files whose names appear in this mapping will reuse the supplied vector
    instead of running the embedding model.

    Content MD5s
    ------------
    Similarly, importers that already know the MD5 hash of each file can
    populate :attr:`content_md5s` with a mapping of ``filename`` to hex
    digest string.  :func:`~vtsearch.datasets.loader.load_dataset_from_folder`
    will skip its own MD5 calculation for any file whose name appears in
    this mapping.

    Custom metadata map
    -------------------
    Importers can populate :attr:`custom_metadata_map` with a mapping of
    ``filename`` to a metadata dict.  When a metadata dict contains a
    non-empty ``"md5"`` key, that value is used as the media's content hash
    (taking priority over both :attr:`content_md5s` and on-the-fly
    calculation).  The metadata dict is also attached to the media as
    ``custom_metadata``.

    CLI support
    -----------
    Every importer is automatically usable from the command line via
    ``python app.py --autodetect --importer <name> [importer args] --settings <file>``.

    The default :meth:`add_cli_arguments` derives ``argparse`` arguments from
    :attr:`fields` and :meth:`run_cli` delegates to :meth:`run`.  Override
    either method when the defaults are not sufficient (e.g. when :meth:`run`
    expects a Werkzeug ``FileStorage`` rather than a plain file path).
    """

    #: Emoji or icon string shown next to the display name in the UI.
    icon: str = "\U0001f50c"
    #: Ordered list of fields the user must fill before importing.
    fields: list[PluginField]

    def __init__(self) -> None:
        #: Mapping of filename to pre-computed embedding vector.  Importers
        #: that supply content vectors alongside media should populate this
        #: dict during :meth:`run` (keyed by the basename of each file).
        #: :func:`~vtsearch.datasets.loader.load_dataset_from_folder` will
        #: skip the embedding model for any file whose name appears here.
        self.content_vectors: dict[str, Any] = {}

        #: Mapping of filename to pre-computed MD5 hex digest string.
        #: Importers that already know the hash of each file should populate
        #: this dict during :meth:`run`.
        #: :func:`~vtsearch.datasets.loader.load_dataset_from_folder` will
        #: skip its own MD5 calculation for any file whose name appears here.
        self.content_md5s: dict[str, str] = {}

        #: Mapping of filename to a per-file custom metadata dict.  When a
        #: metadata dict contains a non-empty ``"md5"`` key, that value is
        #: used as the media's MD5 hash (skipping the normal calculation).
        #: The metadata dict is also attached to the media as
        #: ``custom_metadata``.  Keys follow the same lookup order as
        #: :attr:`content_vectors` (relative path first, then basename).
        self.custom_metadata_map: dict[str, dict[str, Any]] = {}

    def run(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        """Perform the import, populating *medias* in-place.

        Args:
            field_values: Mapping of :attr:`ImporterField.key` → value.
                Fields with ``field_type="file"`` receive a Werkzeug
                :class:`~werkzeug.datastructures.FileStorage` object; all
                other fields receive plain strings.
            medias: The global medias dict to populate.  Modify it in-place;
                do not replace the reference.
            thin: When ``True``, store a ``media_path`` file reference
                instead of loading media bytes into ``media_bytes``.  This
                saves memory for CLI workflows that only need embeddings.

        Raises:
            NotImplementedError: If the subclass has not implemented this.
            Exception: Any exception propagates to the route handler, which
                stores it in the progress tracker as an error message.
        """
        raise NotImplementedError(f"{type(self).__name__}.run() is not implemented")

    # ------------------------------------------------------------------
    # Chunked / piecewise loading
    # ------------------------------------------------------------------

    @property
    def supports_chunked(self) -> bool:
        """Whether this importer supports chunked (piecewise) loading.

        Importers that override :meth:`run_chunked` should return ``True``
        here so the CLI can advertise the capability.
        """
        return False

    def run_chunked(
        self,
        field_values: dict[str, Any],
        chunk_size: int,
        thin: bool = False,
    ) -> Iterator[dict[int, dict[str, Any]]]:
        """Yield chunks of medias for memory-efficient piecewise processing.

        Each yielded dict is a self-contained medias dict with sequential
        IDs starting at 1.  The caller processes each chunk independently
        and discards it before consuming the next, bounding memory usage
        to roughly *chunk_size* medias at a time.

        The default implementation calls :meth:`run` once and yields the
        result as a single chunk.  This means callers can always use the
        chunked code path — importers that have not overridden this method
        simply produce one chunk equal to the full dataset.

        Importers that handle large data sources should override this
        method (and set :attr:`supports_chunked` to ``True``) to yield
        genuine incremental chunks.

        Args:
            field_values: Same mapping as :meth:`run`.
            chunk_size: Maximum number of medias per yielded chunk.
            thin: When ``True``, store file path references instead of
                loading media bytes.

        Yields:
            A dict mapping int media IDs to media data dicts.  Each yielded
            dict contains at most *chunk_size* medias.
        """
        medias: dict[int, dict[str, Any]] = {}
        self.run(field_values, medias, thin=thin)
        yield medias

    def run_chunked_cli(
        self,
        field_values: dict[str, Any],
        chunk_size: int,
        thin: bool = False,
    ) -> Iterator[dict[int, dict[str, Any]]]:
        """CLI variant of :meth:`run_chunked`.

        The default implementation calls :meth:`run_cli` once and yields
        the result as a single chunk.  Importers that override
        :meth:`run_chunked` should also override this if their CLI path
        differs (e.g. file-path strings instead of ``FileStorage`` objects).
        """
        medias: dict[int, dict[str, Any]] = {}
        self.run_cli(field_values, medias, thin=thin)
        yield medias

    # ------------------------------------------------------------------
    # CLI support
    # ------------------------------------------------------------------

    def run_cli(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        """Load a dataset from CLI-provided *field_values* into *medias*.

        The default implementation simply delegates to :meth:`run`, which
        works for importers whose ``run()`` only expects plain string values.
        Importers that expect non-string objects (e.g. ``FileStorage``) must
        override this method to handle file-path strings appropriately.

        Args:
            field_values: Mapping of importer field keys to their CLI values.
            medias: The global medias dict to populate.
            thin: When ``True``, store file path references instead of
                loading media bytes.  Passed through to :meth:`run`.
        """
        self.run(field_values, medias, thin=thin)

    def build_cli_args(self, field_values: dict[str, Any]) -> str:
        """Build a CLI argument string that would recreate this import.

        The returned string contains only the importer-specific portion, e.g.
        ``"--importer folder --media-type sounds --path /data/audio"``.  The
        caller can prepend ``python app.py --autodetect`` and append
        ``--settings <file>`` as needed.

        Fields with ``field_type="file"`` are skipped because they correspond
        to browser uploads that don't translate directly to a CLI flag.

        Args:
            field_values: The same mapping passed to :meth:`run` /
                :meth:`run_cli`.

        Returns:
            A space-separated CLI argument string.
        """
        parts = [f"--importer {self.name}"]
        for f in self.fields:
            if f.field_type == "file":
                continue
            value = field_values.get(f.key, "")
            if value:
                arg_name = f"--{f.key.replace('_', '-')}"
                parts.append(f"{arg_name} {value}")
        return " ".join(parts)

    def build_origin(self, field_values: dict[str, Any]) -> dict[str, Any]:
        """Build an origin dict for elements imported by this importer.

        The returned dict is the serialised form of an
        :class:`~vtsearch.datasets.origin.Origin` object and is stored on
        each media as ``media["origin"]``.  It captures enough information to
        identify the data source (importer name + string-serialisable
        field values).

        Args:
            field_values: The field values used for the import.

        Returns:
            A dict with ``"importer"`` (str) and ``"params"`` (dict of str)
            keys.
        """
        params: dict[str, str] = {}
        for f in self.fields:
            if f.field_type == "file":
                continue
            val = field_values.get(f.key, "")
            if val:
                params[f.key] = str(val)
        return {"importer": self.name, "params": params}

    # ------------------------------------------------------------------
    # Origin display and reload
    # ------------------------------------------------------------------

    def origin_display(self, origin: dict[str, Any]) -> str:
        """Return a human-readable string for an origin dict from this importer.

        The default implementation returns ``"<name>:<first_param_value>"``
        or just ``"<name>"`` when there are no params.  Subclasses may
        override for a more descriptive representation.
        """
        params = origin.get("params", {})
        if params:
            first_val = next(iter(params.values()))
            return f"{self.name}:{first_val}"
        return self.name

    def can_reload_from_origin(self, origin: dict[str, Any]) -> bool:
        """Return whether this importer can re-load data from *origin*.

        The default implementation returns ``True`` — most importers can
        reload from their stored params.  Importers that require browser
        uploads (e.g. pickle) should return ``False`` unless the params
        contain enough info to reload (e.g. a server file path).

        Subclasses should override when their reload capability depends on
        the specific origin params (e.g. checking if a file still exists).
        """
        return True

    def reload_from_origin(self, origin: dict[str, Any]) -> dict[str, Any] | None:
        """Extract field_values from an origin dict suitable for :meth:`run`.

        Returns a field_values dict that can be passed to
        :meth:`run` / ``_run_importer_in_background()``, or ``None`` if
        the origin cannot be reloaded.

        The default implementation returns the origin's ``params`` dict
        directly, which works for importers whose fields accept plain
        strings.
        """
        return dict(origin.get("params", {}))

    def resolve_file(
        self,
        origin: dict[str, Any],
        origin_name: str = "",
        filename: str = "",
    ) -> "Path | None":
        """Resolve a media file from its origin information.

        Given the origin dict that this importer produced, plus the
        ``origin_name`` and ``filename`` stored on the media, return the
        :class:`~pathlib.Path` to the actual file on disk — or ``None``
        if the file cannot be found.

        .. important::

           Every importer whose media can be located on disk **must**
           override this method.  Cross-dataset features (e.g. applying a
           saved Detector to a different dataset via "Find") rely on
           resolving label entries back to files for re-embedding.  If this
           method is not overridden, those features silently produce empty
           results ("N/A" verdicts) with no error — making the root cause
           very hard to diagnose.

        The default implementation returns ``None``, which is only
        appropriate for importers whose media truly cannot be resolved
        from disk (e.g. ``pickle`` with browser-uploaded files and no
        server path).
        """
        return None
