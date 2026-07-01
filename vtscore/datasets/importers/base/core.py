"""The thin importer base (:class:`ImporterBase`).

``ImporterBase`` carries everything every importer needs regardless of how it
pulls media in: metadata/declarative attributes, dataset-name resolution,
origin building, CLI wrapping, chunked-loading scaffolding, the precomputed
embedding/MD5/metadata dicts, and the origin reload/display/resolve surface.

It deliberately does **not** know about source specs, converters, or the
per-record fetch hooks.  Importers that want the framework to drive the
source-spec → converter → ingestion pipeline (or the ``list_records`` /
``fetch_record`` convenience hooks) subclass
:class:`~vtscore.datasets.importers.base.dataset_importer.DatasetImporter`
instead, which layers that machinery on top of this base.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from vtscore.plugins import PluginBase, PluginField

from .origin import _dataset_name_field, _field_in_origin, _serialise_origin_value


class ImporterBase(PluginBase):
    """Thin abstract base class for dataset importers.

    Subclass this (or its richer subclass
    :class:`~vtscore.datasets.importers.base.dataset_importer.DatasetImporter`),
    set the class-level attributes, implement :meth:`run`, and expose a
    module-level ``IMPORTER = YourImporter()`` – the registry picks it up
    automatically.

    Embedding contract
    ------------------
    Importers do **not** call any embedder.  Emit media dicts with
    ``embedding=None`` (and ``embedder=""``); the framework
    :func:`~vtscore.datasets.stages.embedding.embed_missing` stage runs
    after the importer returns and bulk-embeds every item still at
    ``None`` using the user's selected embedder (or the default for the
    media type).  Items where the embedder returns ``None`` get dropped
    by the next stage.

    If your importer ships pre-computed vectors (e.g. an NPZ archive),
    use :attr:`content_vectors` or :attr:`custom_metadata_map` (below)
    so the framework treats them as already-embedded and skips them.

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
    ``numpy.ndarray`` during :meth:`run`.  Files whose names appear in
    this mapping land on the media dict with the supplied vector; the
    framework embed stage then leaves them alone.

    Content MD5s
    ------------
    Similarly, importers that already know the MD5 hash of each file can
    populate :attr:`content_md5s` with a mapping of ``filename`` to hex
    digest string.  :func:`~vtscore.datasets.loader.load_dataset_from_folder`
    will skip its own MD5 calculation for any file whose name appears in
    this mapping.

    Custom metadata map
    -------------------
    Importers can populate :attr:`custom_metadata_map` with a mapping of
    ``filename`` to a metadata dict.  When a metadata dict contains a
    non-empty ``"md5"`` key, that value is used as the media's content hash
    (taking priority over both :attr:`content_md5s` and on-the-fly
    calculation).  When it contains an ``"embedding"`` key, that value is
    used as the media's embedding vector (taking priority over
    :attr:`content_vectors` and the framework embed stage).  The metadata
    dict is also attached to the media as ``custom_metadata``.

    CLI support
    -----------
    Every importer is automatically usable from the command line via
    ``python app.py --autodetect --importer <name> [importer args] --settings <file>``.

    The default :meth:`add_cli_arguments` derives ``argparse`` arguments
    from :attr:`fields` and :meth:`run_cli` wraps any
    ``field_type="file"`` CLI argument in
    :class:`~vtscore.plugins.uploads.CliUploadedFile` before delegating
    to :meth:`run`, so plugin bodies written against the
    :class:`~vtscore.plugins.uploads.UploadedFile` surface work
    identically in both code paths.  Subclasses only need to override
    :meth:`run_cli` when their CLI-specific behaviour genuinely diverges
    from the request-time flow (e.g. the chunked-pickle fast path).
    """

    #: Emoji or icon string shown next to the display name in the UI.
    icon: str = "\U0001f50c"
    #: Ordered list of fields the user must fill before importing.
    fields: list[PluginField]

    #: Which view the dataset-importer modal opens when this card is clicked.
    #: ``"form"`` (default) builds a generic form from :attr:`fields`.  The
    #: other values trigger dedicated UI sections in the modal:
    #:
    #: - ``"local_folder"``: browser-side folder upload widget.
    #: - ``"local_files"``: browser-side multi-file upload widget.
    #: - ``"server_folder"``: server filesystem browser.
    #: - ``"demo"``: demo-dataset table.
    picker_view: str = "form"

    #: Picker tab this importer belongs to.  One of ``"services"``,
    #: ``"server"``, ``"local"``, ``"demo"``, or ``""`` (uncategorised;
    #: hidden from the tabbed picker entirely).  Database/API-style
    #: importers (extensions that fetch from a remote service) belong on
    #: the ``"services"`` tab, which is the default for any custom importer
    #: that doesn't override this attribute.
    category: str = "services"

    #: Extra keys to copy from ``field_values`` into the origin ``params``
    #: dict, in addition to the importer's declared :attr:`fields`.  Use
    #: this for transient request-time values that aren't first-class
    #: :class:`PluginField`\s.  List/dict values are JSON-encoded;
    #: everything else is stringified via ``str(...)``.  Empty values are
    #: skipped.  The framework automatically adds ``"source_specs"`` to
    #: this tuple, so importers usually leave this empty.
    extra_origin_keys: tuple[str, ...] = ()

    #: When ``True``, :meth:`build_origin` returns
    #: ``{"importer": self.name, "params": {}}`` regardless of
    #: ``field_values``.  Use this for importers whose dataset-level origin
    #: is intentionally empty (e.g. the recaller importer, which builds a
    #: useful per-media origin on each yielded record and has no useful
    #: dataset-wide identifier).
    origin_suppressed: bool = False

    #: When ``True``, the importer applies the clipper (and owns the
    #: resulting clipped + embedded artifacts) inside its own :meth:`run`,
    #: so the shared load pipeline must NOT run its ``_apply_clipper_stage``
    #: on top — doing so would clip the already-clipped media a second time.
    #: Set this on importers that self-clip (e.g. the demo importer, which
    #: clips + embeds + caches the final dataset in ``load_demo_dataset``).
    #: The dispatch (:func:`_run_importer_in_background`) keeps the full
    #: clipper config in ``field_values`` for these importers and suppresses
    #: the pipeline's clipper stage.
    handles_own_clipping: bool = False

    def __init__(self) -> None:
        #: Mapping of filename to pre-computed embedding vector.  Importers
        #: that supply content vectors alongside media should populate this
        #: dict during :meth:`run` (keyed by the basename of each file);
        #: or, preferably, call :meth:`yield_precomputed` once per file
        #: which writes to all three precomputed dicts atomically.
        #: :func:`~vtscore.datasets.loader.load_dataset_from_folder` will
        #: skip the embedding model for any file whose name appears here.
        self.content_vectors: dict[str, Any] = {}

        #: Mapping of filename to pre-computed MD5 hex digest string.
        #: Importers that already know the hash of each file should populate
        #: this dict during :meth:`run` (or call :meth:`yield_precomputed`).
        #: :func:`~vtscore.datasets.loader.load_dataset_from_folder` will
        #: skip its own MD5 calculation for any file whose name appears here.
        self.content_md5s: dict[str, str] = {}

        #: Mapping of filename to a per-file custom metadata dict.  When a
        #: metadata dict contains a non-empty ``"md5"`` key, that value is
        #: used as the media's MD5 hash (skipping the normal calculation).
        #: When it contains an ``"embedding"`` key, that value is used as
        #: the media's embedding vector (skipping both :attr:`content_vectors`
        #: and the embedding model).  The metadata dict is also attached to
        #: the media as ``custom_metadata``.  Keys follow the same lookup
        #: order as :attr:`content_vectors` (relative path first, then
        #: basename).
        self.custom_metadata_map: dict[str, dict[str, Any]] = {}

    def to_dict(self) -> dict[str, Any]:
        from vtscore.converters import list_converters_for_target  # noqa: PLC0415
        from vtscore.media import all_types_dict  # noqa: PLC0415

        d = super().to_dict()
        d["picker_view"] = self.picker_view
        d["category"] = self.category
        d["fields"] = d["fields"] + [_dataset_name_field().to_dict()]

        # For each media type the user can select, list N→M converters
        # that produce that type; the UI can show a datagrid of
        # available converters dynamically.
        converters_by_target: dict[str, list[dict]] = {}
        for mt_info in all_types_dict():
            type_id = mt_info["type_id"]
            convs = list_converters_for_target(type_id)
            if convs:
                converters_by_target[type_id] = [c.to_dict() for c in convs]
        d["available_converters_by_media_type"] = converters_by_target
        return d

    def default_display_name(self, field_values: dict[str, Any]) -> str:
        """Return the importer-computed default name for a dataset.

        Subclasses override this to derive a sensible default from
        *field_values* (e.g. the demo importer reads the demo entry's
        label).  The base implementation just returns :attr:`display_name`.
        The user-typed ``dataset_name`` (when present) takes priority over
        whatever this method returns (see :meth:`resolve_display_name`).
        """
        return self.display_name

    def resolve_display_name(self, field_values: dict[str, Any] | None) -> str:
        """Return the human-readable name to use for a dataset loaded with *field_values*.

        Importer subclasses should override :meth:`default_display_name`
        rather than this method.  ``resolve_display_name`` first honours
        the user-typed ``dataset_name`` field (when non-empty) and falls
        back to :meth:`default_display_name` otherwise.
        """
        user_name = (field_values.get("dataset_name") or "").strip() if field_values else ""
        if user_name:
            return user_name
        return self.default_display_name(field_values or {})

    def yield_precomputed(
        self,
        filename: str,
        *,
        embedding: Any = None,
        md5: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register pre-computed embedding / MD5 / metadata for *filename*.

        Single entry point for the three legacy precomputed dicts
        (:attr:`content_vectors`, :attr:`content_md5s`,
        :attr:`custom_metadata_map`).  Importers that already know any of
        these values up-front (e.g. when reading an NPZ sidecar or a
        manifest that ships hashes) should call this once per file
        instead of writing to the three dicts directly, so a single
        misspelled key (or one entry landing in only two of the three)
        cannot silently produce a mismatched-per-file precomputed state.

        Any combination of arguments may be omitted; only the supplied
        ones land in the matching dict.  The three legacy dicts remain
        public for back-compat; third-party importers that write to
        them directly continue to work, but new code should prefer this
        helper.

        Args:
            filename: The basename (or dataset-relative path) the
                framework uses to key precomputed lookups against the
                file the loader discovers on disk.  Must match the key
                that ``load_dataset_from_folder`` will see.
            embedding: Pre-computed embedding vector.  Goes into
                :attr:`content_vectors`.
            md5: Pre-computed MD5 hex digest string.  Goes into
                :attr:`content_md5s`.
            metadata: Per-file custom metadata dict.  Goes into
                :attr:`custom_metadata_map`.
        """
        if embedding is not None:
            self.content_vectors[filename] = embedding
        if md5 is not None:
            self.content_md5s[filename] = md5
        if metadata is not None:
            self.custom_metadata_map[filename] = metadata

    def run(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        """Perform the import, populating *medias* in-place.

        The thin base does not implement an import strategy.  Subclasses
        either:

        - override this method directly for full control (folder-shaped
          importers delegate to
          :func:`~vtscore.converters.runner.run_converters_on_folder`), or
        - subclass
          :class:`~vtscore.datasets.importers.base.dataset_importer.DatasetImporter`,
          whose default :meth:`run` drives the source-spec → converter →
          ingestion pipeline and the ``list_records`` / ``fetch_record``
          convenience hooks.

        Args:
            field_values: Mapping of :attr:`PluginField.key` → value.
                Fields with ``field_type="file"`` receive an
                :class:`~vtscore.plugins.uploads.UploadedFile` (Flask
                requests pass a Werkzeug ``FileStorage`` straight
                through; CLI invocations wrap the path argument in
                :class:`~vtscore.plugins.uploads.CliUploadedFile`).
                All other fields receive plain strings.
            medias: The global medias dict to populate.  Modify it in-place;
                do not replace the reference.
            thin: When ``True``, store a ``media_path`` file reference
                instead of loading media bytes into ``media_bytes``.  This
                saves memory for CLI workflows that only need embeddings.

        Raises:
            NotImplementedError: Always, unless overridden.  Subclass and
                override :meth:`run`, or subclass ``DatasetImporter`` for
                the source-spec / per-record hooks.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.run() is not implemented. Override run() "
            "directly, or subclass DatasetImporter to use the source-spec / "
            "per-record hooks (list_records/fetch_record, fetch_source_media)."
        )

    # ------------------------------------------------------------------
    # Dynamic field options
    # ------------------------------------------------------------------

    def get_field_options(
        self,
        field_key: str,
        current_values: dict[str, Any],
    ) -> list[str]:
        """Return the dropdown options for *field_key* given current form values.

        Override this on importers that declare any
        :class:`~vtscore.plugins.PluginField` with
        ``dynamic_options=True``.  The frontend calls this via
        ``POST /api/dataset/import/<name>/options`` whenever a field listed
        in another field's ``depends_on`` changes (e.g. a ``query_id``
        select might re-populate after the user picks a ``media_type``.

        Args:
            field_key: The :attr:`PluginField.key` of the field whose
                options are being requested.
            current_values: A snapshot of every form field's current value,
                keyed by :attr:`PluginField.key`.  Values are plain strings
                (or empty strings for unfilled fields).

        Returns:
            The list of allowed option strings for the dropdown.

        Raises:
            NotImplementedError: When the importer declares no dynamic
                fields, or has not implemented this hook for *field_key*.
                Subclasses should raise (or let the default raise) for any
                ``field_key`` they do not handle.
        """
        raise NotImplementedError(f"{type(self).__name__}.get_field_options({field_key!r}) is not implemented")

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
        chunked code path; importers that have not overridden this method
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

        The default implementation wraps any ``field_type="file"`` CLI
        path argument in :class:`~vtscore.plugins.uploads.CliUploadedFile`
        so :meth:`run` sees the same
        :class:`~vtscore.plugins.uploads.UploadedFile` shape it does for
        a Flask request, then delegates.  Subclasses only need to
        override this method when the CLI path needs genuinely different
        behaviour from the request-time flow (e.g. a chunked fast path).

        Args:
            field_values: Mapping of importer field keys to their CLI values.
            medias: The global medias dict to populate.
            thin: When ``True``, store file path references instead of
                loading media bytes.  Passed through to :meth:`run`.
        """
        from vtscore.plugins.uploads import wrap_cli_file_fields  # noqa: PLC0415

        self.run(wrap_cli_file_fields(self.fields, field_values), medias, thin=thin)

    def build_cli_args(self, field_values: dict[str, Any]) -> str:
        """Build a CLI argument string that would recreate this import.

        The returned string contains only the importer-specific portion, e.g.
        ``"--importer folder --media-type audio --path /data/audio"``.  The
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
            arg_name = f"--{f.key.replace('_', '-')}"
            if f.field_type == "checkbox":
                value = field_values.get(f.key, str(f.default).lower() == "true")
                truthy = value if isinstance(value, bool) else str(value).lower() == "true"
                no_arg = f"--no-{f.key.replace('_', '-')}"
                parts.append(arg_name if truthy else no_arg)
                continue
            value = field_values.get(f.key, "")
            if value:
                parts.append(f"{arg_name} {value}")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Origin building
    # ------------------------------------------------------------------

    def build_origin(self, field_values: dict[str, Any]) -> dict[str, Any]:
        """Build an origin dict for elements imported by this importer.

        The returned dict is the serialised form of an
        :class:`~vtscore.datasets.origin.Origin` object and is stored on
        each media as ``media["origin"]``.  It captures enough information to
        identify the data source (importer name + string-serialisable
        field values).

        Behaviour is driven declaratively by:

        - :attr:`origin_suppressed`: short-circuits to empty params.
        - :attr:`PluginField.include_in_origin`: per-field opt-out.  The
          default is ``False`` for ``"file"`` and ``"password"`` fields
          (don't persist uploads or secrets) and ``True`` for every
          other type.
        - :attr:`PluginField.origin_serializer`: per-field custom
          string conversion (e.g. comma-joining a list).
        - :attr:`extra_origin_keys`: non-``PluginField`` keys to copy
          from ``field_values``.  ``"source_specs"`` is always included.

        Subclasses may still override this method, in which case the
        override wins; every declarative knob above is ignored.  Prefer
        the declarative form unless you need something the framework
        can't express.

        Args:
            field_values: The field values used for the import.

        Returns:
            A dict with ``"importer"`` (str) and ``"params"`` (dict of str)
            keys.
        """
        if self.origin_suppressed:
            return {"importer": self.name, "params": {}}

        params: dict[str, str] = {}
        for f in self.fields:
            if not _field_in_origin(f):
                continue
            if f.field_type == "checkbox":
                val = field_values.get(f.key, str(f.default).lower() == "true")
                truthy = val if isinstance(val, bool) else str(val).lower() == "true"
                params[f.key] = "true" if truthy else "false"
                continue
            raw = field_values.get(f.key, "")
            serialised = _serialise_origin_value(raw, f.origin_serializer)
            if serialised:
                params[f.key] = serialised

        for key in self._effective_extra_origin_keys():
            raw = field_values.get(key, "")
            serialised = _serialise_origin_value(raw, None)
            if serialised:
                params[key] = serialised

        return {"importer": self.name, "params": params}

    def _effective_extra_origin_keys(self) -> tuple[str, ...]:
        """Return :attr:`extra_origin_keys`, auto-adding ``"source_specs"``."""
        if "source_specs" not in self.extra_origin_keys:
            return self.extra_origin_keys + ("source_specs",)
        return self.extra_origin_keys

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

        The default implementation returns ``True``; most importers can
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
    ) -> Path | None:
        """Resolve a media file from its origin information.

        Given the origin dict that this importer produced, plus the
        ``origin_name`` and ``filename`` stored on the media, return the
        :class:`~pathlib.Path` to the actual file on disk, or ``None``
        if the file cannot be found.

        .. important::

           Every importer whose media can be located on disk **must**
           override this method.  Cross-dataset features (e.g. applying a
           saved Detector to a different dataset via "Find") rely on
           resolving label entries back to files for re-embedding.  If this
           method is not overridden, those features silently produce empty
           results ("N/A" verdicts) with no error; making the root cause
           very hard to diagnose.

        The default implementation returns ``None``, which is only
        appropriate for importers whose media truly cannot be resolved
        from disk (e.g. ``pickle`` with browser-uploaded files and no
        server path).
        """
        return None
