"""Base classes for dataset importers.

To add a new importer, subclass :class:`DatasetImporter`, define its class
attributes and :meth:`~DatasetImporter.run`, then expose a module-level
``IMPORTER`` instance from a package under this directory.  The registry will
discover it automatically.

Each importer also supports CLI usage via :meth:`~DatasetImporter.add_cli_arguments`
and :meth:`~DatasetImporter.run_cli`.  The base class provides default
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

Example – a minimal SFTP importer skeleton::

    # vtsearch/datasets/importers/sftp/__init__.py
    from vtscore.datasets.importers.base import DatasetImporter, ImporterField

    from vtscore.media import all_folder_names

    class SftpImporter(DatasetImporter):
        name         = "sftp"
        display_name = "SFTP Server"
        description  = "Download media files from an SFTP server."
        fields = [
            ImporterField("host",       "Hostname",    "text"),
            ImporterField("user",       "Username",    "text"),
            ImporterField("password",   "Password",    "password"),
            ImporterField("path",       "Remote Path", "text"),
            ImporterField(
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
picked up the next time you run ``bash scripts/install-cpu.sh`` (or any
editable install). pyproject.toml is the single source of truth — deptry
verifies that every imported package is declared there.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Iterator

from vtscore.plugins import PluginBase, PluginField

# Backward-compatible alias — existing plugins import ``ImporterField``.
ImporterField = PluginField

__all__ = ["DatasetImporter", "ImporterField", "SourceSpec"]


@dataclass
class SourceSpec:
    """Declarative description of one media stream an importer should pull in.

    A multi-media import is a list of these.  Each spec asks the importer
    to fetch media of ``source_type`` and — when ``converter`` is set —
    pass them through that converter (with the supplied ``params``) to
    produce media of the dataset's output media type.

    When ``converter`` is ``None`` the source media is included directly
    and ``source_type`` must equal the importer's chosen output media
    type.

    See :meth:`DatasetImporter.effective_source_specs` for how importers
    obtain this list.
    """

    source_type: str
    converter: str | None = None
    params: dict[str, Any] = dc_field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "converter": self.converter,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceSpec:
        return cls(
            source_type=str(data.get("source_type") or ""),
            converter=(str(data["converter"]) if data.get("converter") else None),
            params=dict(data.get("params") or {}),
        )


def _parse_multi_media_specs(raw: Any, output_type: str) -> list[SourceSpec]:
    """Parse and validate the explicit ``source_specs`` form value.

    Falls back to a single pass-through spec when no value is submitted
    (or when the submitted value parses to an empty list) so an importer
    whose form omits ``source_specs`` still loads cleanly, and an empty
    spec-grid does not silently produce a zero-media dataset.
    """
    from vtscore.converters import get_converter  # noqa: PLC0415
    from vtscore.media import get_by_folder_name  # noqa: PLC0415

    specs_raw: list[dict[str, Any]]
    if raw is None or raw == "":
        specs_raw = []
    elif isinstance(raw, str):
        try:
            specs_raw = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid source_specs JSON: {exc}") from exc
    else:
        specs_raw = list(raw)

    if not output_type:
        raise ValueError("import requires a 'media_type' (output) field")

    if not specs_raw:
        specs_raw = [{"source_type": output_type, "converter": None, "params": {}}]

    specs: list[SourceSpec] = []
    for item in specs_raw:
        if not isinstance(item, dict):
            raise ValueError(f"source_specs entries must be objects, got {type(item).__name__}")
        spec = SourceSpec.from_dict(item)
        try:
            spec.source_type = get_by_folder_name(spec.source_type).type_id
        except (KeyError, AttributeError) as exc:
            raise ValueError(f"Unknown source_type: {spec.source_type!r}") from exc
        _validate_spec_converter(spec, output_type, get_converter)
        specs.append(spec)
    return specs


def _validate_spec_converter(spec: SourceSpec, output_type: str, get_converter) -> None:
    """Validate that *spec*'s converter (if any) bridges source→output.

    Also validates ``spec.params`` against the converter's own
    :class:`PluginField` schema, so declared ``min`` / ``max`` ranges
    are enforced before the params reach :meth:`MediaConverter.convert`.
    """
    from marshmallow import ValidationError  # noqa: PLC0415

    if spec.converter is None:
        if spec.source_type != output_type:
            raise ValueError(
                f"Direct (no-converter) source_type {spec.source_type!r} "
                f"does not match output media_type {output_type!r}",
            )
        return
    converter = get_converter(spec.converter)
    if converter is None:
        raise ValueError(f"Unknown converter: {spec.converter!r}")
    if converter.source_type != spec.source_type:
        raise ValueError(
            f"Converter {spec.converter!r} expects source_type {converter.source_type!r}, not {spec.source_type!r}",
        )
    if converter.target_type != output_type:
        raise ValueError(
            f"Converter {spec.converter!r} produces "
            f"{converter.target_type!r}, but output media_type is {output_type!r}",
        )
    try:
        converter.validate_params(spec.params)
    except ValidationError as exc:
        raise ValueError(f"Invalid params for converter {spec.converter!r}: {exc.messages}") from exc


PickerView = str  # one of: "form", "demo", "server_folder", "local"


# Synthetic per-importer field that lets the user pick a name for the new
# dataset.  Appended to the end of every importer's serialised field list
# in :meth:`DatasetImporter.to_dict` (just before the Advanced section in
# the UI), so users filling the form top-down have already entered the
# fields that feed the auto-derived default name by the time they reach
# it.  Routed through the per-plugin marshmallow schema as a regular field
# (see :func:`vtsearch.routes._shared.validate_plugin_args`) and read
# downstream by :meth:`DatasetImporter.resolve_display_name`.
DATASET_NAME_FIELD_KEY = "dataset_name"


_ORIGIN_EXCLUDED_FIELD_TYPES = frozenset({"file", "password"})


def _field_in_origin(field: PluginField) -> bool:
    """Resolve whether *field*'s value should land in the persisted origin.

    Honors an explicit :attr:`PluginField.include_in_origin`; otherwise
    falls back to the field-type default (file and password fields are
    excluded).
    """
    if field.include_in_origin is not None:
        return field.include_in_origin
    return field.field_type not in _ORIGIN_EXCLUDED_FIELD_TYPES


def _serialise_origin_value(value: Any, serializer: Any) -> str:
    """Serialise *value* for inclusion in an origin ``params`` dict.

    Returns the empty string when *value* is falsy (mirroring the
    pre-refactor ``if val: params[key] = str(val)`` shape).  When
    *serializer* is set it runs first; otherwise list/dict values are
    JSON-encoded so an importer's structured ``field_values`` round-trip
    through the string-only origin contract.
    """
    if not value:
        return ""
    if serializer is not None:
        return str(serializer(value))
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return str(value)


def _dataset_name_field() -> PluginField:
    return PluginField(
        key=DATASET_NAME_FIELD_KEY,
        label="Dataset Name",
        field_type="text",
        description="Leave blank to use a default name",
        required=False,
        placeholder="Leave blank to use a default name",
    )


class DatasetImporter(PluginBase):
    """Abstract base class for dataset importers.

    Subclass this, set the class-level attributes, implement :meth:`run`,
    and expose a module-level ``IMPORTER = YourImporter()`` – the registry
    picks it up automatically.

    Embedding contract
    ------------------
    Importers do **not** call any embedder.  Emit media dicts with
    ``embedding=None`` (and ``embedder=""``); the framework
    :func:`~vtscore.datasets.load_pipeline.embed_missing` stage runs
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
    #: - ``"local_folder"`` — browser-side folder upload widget.
    #: - ``"local_files"`` — browser-side multi-file upload widget.
    #: - ``"server_folder"`` — server filesystem browser.
    #: - ``"demo"`` — demo-dataset table.
    picker_view: str = "form"

    #: Picker tab this importer belongs to.  One of ``"services"``,
    #: ``"server"``, ``"local"``, ``"demo"``, or ``""`` (uncategorised —
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

    def to_dict(self) -> dict[str, Any]:
        from vtscore.converters import list_converters_for_target  # noqa: PLC0415
        from vtscore.media import all_types_dict  # noqa: PLC0415

        d = super().to_dict()
        d["picker_view"] = self.picker_view
        d["category"] = self.category
        d["fields"] = d["fields"] + [_dataset_name_field().to_dict()]

        # For each media type the user can select, list N→M converters
        # that produce that type — so the UI can show a datagrid of
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
        whatever this method returns — see :meth:`resolve_display_name`.
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

    def __init__(self) -> None:
        #: Mapping of filename to pre-computed embedding vector.  Importers
        #: that supply content vectors alongside media should populate this
        #: dict during :meth:`run` (keyed by the basename of each file) —
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
        these values up-front — e.g. when reading an NPZ sidecar or a
        manifest that ships hashes — should call this once per file
        instead of writing to the three dicts directly, so a single
        misspelled key (or one entry landing in only two of the three)
        cannot silently produce a mismatched-per-file precomputed state.

        Any combination of arguments may be omitted; only the supplied
        ones land in the matching dict.  The three legacy dicts remain
        public for back-compat — third-party importers that write to
        them directly continue to work — but new code should prefer this
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

        Subclasses pick one of four override points, in order of increasing
        control.  Hooks 1–3 leave conversion and ingestion to the framework;
        only hook 4 takes that responsibility back.  The numbering here
        matches ``docs/EXTENDING-plugins.md``.

        1. :meth:`list_records` + :meth:`fetch_record` (and optionally
           :meth:`_fetch_records_bulk_impl` for batched fetches) — a
           single-spec convenience for service importers that only pull one
           source type.  The default :meth:`fetch_source_media` delegates to
           these hooks, so single-spec importers can keep using the
           per-record split without thinking about specs.  This hook is also
           the fallback when :meth:`effective_source_specs` cannot resolve
           (no ``media_type`` declared).
        2. :meth:`fetch_source_media` — yield raw source-type media one record
           at a time, for one :class:`SourceSpec` at a time.  The framework
           loops over each spec produced by :meth:`effective_source_specs`,
           calls this method per spec, runs the spec's converter (when one is
           set) on every yielded media, and ingests the results.  Recommended
           for service-style multi-source-type importers whose backend
           serves one media type per query.
        3. :meth:`fetch_all_source_media` — yield ``(spec, raw_media)`` pairs
           for **all** specs in one pass.  Override this when one upstream
           call returns mixed source types and you want to make it just once
           (e.g. a service whose query returns "everything that matched" with
           a per-record type tag).  The framework still runs converters and
           ingests; subclasses never call
           :func:`~vtscore.converters.get_converter` themselves.  The default
           implementation delegates to :meth:`fetch_source_media` per spec,
           so importers using hook 1 or 2 don't need to know this hook exists.
        4. :meth:`run` directly — full control.  Folder-shaped importers
           override this and delegate to
           :func:`~vtscore.converters.runner.run_converters_on_folder`.

        Default flow:

        - When :meth:`effective_source_specs` resolves to one or more specs,
          call :meth:`fetch_all_source_media` (hook 3, which by default loops
          :meth:`fetch_source_media` — hook 2) once and pass each yielded
          ``(spec, raw)`` pair through ``spec.converter`` (when set) before
          assigning IDs and storing the result in *medias*.
        - When :meth:`effective_source_specs` cannot resolve (no
          ``media_type`` declared — i.e. a bare service importer), fall
          back to the :meth:`list_records` + :meth:`fetch_records_bulk`
          path (hook 1) with no conversion.

        .. note::

           Hooks 2 and 3 only run when :meth:`effective_source_specs`
           resolves to at least one spec.  An importer that overrides
           :meth:`fetch_source_media` or :meth:`fetch_all_source_media`
           but does **not** declare a ``media_type`` field (or accept a
           ``source_specs`` value) will fall through to the hook-1 path
           and raise :class:`NotImplementedError` from
           :meth:`list_records`.

        IDs are assigned as sequential integers starting at 1.  The default
        per-media origin is filled in from :meth:`build_origin` when the
        media dict did not already set one.

        Args:
            field_values: Mapping of :attr:`ImporterField.key` → value.
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
            NotImplementedError: If none of :meth:`run`,
                :meth:`fetch_all_source_media`, :meth:`fetch_source_media`,
                or the :meth:`list_records` + :meth:`fetch_record` hooks are
                implemented by the subclass.
            Exception: Any exception propagates to the route handler, which
                stores it in the progress tracker as an error message.
        """
        default_origin = self.build_origin(field_values)
        next_id = 1

        try:
            specs = self.effective_source_specs(field_values)
        except ValueError:
            specs = []

        if specs:
            next_id = self._ingest_spec_stream(
                self.fetch_all_source_media(specs, field_values, thin=thin),
                medias,
                default_origin,
                next_id,
            )
            return

        # Fallback: no spec set could be resolved.  Use the per-record hooks
        # directly with no conversion.  This path keeps the
        # list_records/fetch_record API working for importers that don't
        # declare a media_type / source_specs schema (e.g. tests).
        records = self.list_records(field_values)
        fetched = self.fetch_records_bulk(records, field_values, thin=thin)
        for media in fetched:
            if media is None:
                continue
            media["id"] = next_id
            media.setdefault("origin", default_origin)
            media.setdefault("origin_name", media.get("filename") or str(next_id))
            medias[next_id] = media
            next_id += 1

    def _ingest_spec_stream(
        self,
        stream: Iterator[tuple[SourceSpec, dict[str, Any]]],
        medias: dict,
        default_origin: dict[str, Any],
        next_id: int,
    ) -> int:
        """Convert+ingest each ``(spec, raw)`` pair from *stream* into *medias*.

        Resolves each spec's converter once and caches it across pairs so
        a bulk importer that interleaves specs doesn't re-resolve on
        every yield.  Returns the next available media id.
        """
        from vtscore.converters import get_converter  # noqa: PLC0415

        converter_cache: dict[str, Any] = {}
        for spec, raw in stream:
            if raw is None:
                continue
            if spec.converter is None:
                outs = [raw]
                target_type = spec.source_type
            else:
                converter = converter_cache.get(spec.converter)
                if converter is None:
                    resolved = get_converter(spec.converter)
                    if resolved is None:
                        raise ValueError(f"Unknown converter: {spec.converter!r}")
                    converter = resolved
                    converter_cache[spec.converter] = converter
                outs = converter.convert_normalized(raw, spec.params)
                target_type = converter.target_type
            for media in outs:
                if media is None:
                    continue
                media.setdefault("media_type", target_type)
                media["id"] = next_id
                media.setdefault("origin", default_origin)
                media.setdefault("origin_name", media.get("filename") or str(next_id))
                medias[next_id] = media
                next_id += 1
        return next_id

    # ------------------------------------------------------------------
    # Multi-media source hooks
    # ------------------------------------------------------------------
    #
    # Two override points for service-style importers.  In both cases the
    # framework drives the converter loop and ingestion, so subclasses
    # never call :func:`vtscore.converters.get_converter` themselves —
    # they just yield raw media of the appropriate ``spec.source_type``.
    #
    # Pick :meth:`fetch_source_media` when the backend serves one media
    # type per query (the framework loops it across specs for you).
    # Pick :meth:`fetch_all_source_media` when a single upstream call
    # returns mixed source types and you want to make it only once.

    def fetch_all_source_media(
        self,
        specs: list[SourceSpec],
        field_values: dict[str, Any],
        thin: bool = False,
    ) -> Iterator[tuple[SourceSpec, dict[str, Any]]]:
        """Yield ``(spec, raw_media)`` pairs for every spec the user picked.

        This is the bulk-fetch escape hatch for importers whose backend
        returns mixed source types in a single upstream call (e.g. one
        query that yields both images and videos, with a per-record type
        tag).  Override this to issue that one call, then yield each
        record paired with the :class:`SourceSpec` it satisfies — the
        framework handles converter dispatch and ingestion exactly as it
        does for :meth:`fetch_source_media`.

        For per-spec importers (the common case — one query per source
        type), override :meth:`fetch_source_media` instead; the default
        implementation of this method loops it for you.

        Each yielded ``raw_media`` dict must match the shape expected of
        media of ``spec.source_type`` (so a video spec yields
        ``type="video"`` dicts with ``media_bytes`` / ``media_path``; the
        framework hands them to the spec's converter, which produces
        e.g. ``type="image"`` dicts).  ``id`` and ``origin`` may be
        omitted — :meth:`run` assigns IDs and falls back to
        :meth:`build_origin` for unset origins.

        Args:
            specs: The full list of :class:`SourceSpec` rows resolved
                from *field_values*, in the user's submitted order.
            field_values: The same mapping passed to :meth:`run`.
            thin: When ``True``, skip downloading raw bytes — yield media
                dicts with ``media_url`` / ``media_path`` instead of
                ``media_bytes``.

        Yields:
            ``(spec, raw_media)`` tuples.  ``spec`` must be one of the
            entries in *specs* (the framework uses its ``converter`` and
            ``params`` to dispatch).
        """
        for spec in specs:
            for raw in self.fetch_source_media(spec, field_values, thin=thin):
                if raw is None:
                    continue
                yield spec, raw

    def fetch_source_media(
        self,
        spec: SourceSpec,
        field_values: dict[str, Any],
        thin: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Yield raw media dicts of ``spec.source_type``.

        For multi-source-type importers (service-style importers that mix
        e.g. images + videos in one import), override this method.  The
        framework calls it once per :class:`SourceSpec` returned by
        :meth:`effective_source_specs`; if the spec declares a converter,
        the framework runs ``converter.convert(raw, spec.params)`` on every
        yielded media before storing it.  Subclasses never invoke
        converters themselves.

        Each yielded dict should already match the shape expected of media
        of ``spec.source_type`` (so a video spec yields ``type="video"``
        dicts with ``media_bytes`` / ``media_path``; the framework hands
        them to the converter which produces e.g. ``type="image"`` dicts).
        Yield nothing if the spec resolves to zero records.

        Default implementation: delegates to
        :meth:`list_records` + :meth:`fetch_records_bulk`, ignoring *spec*.
        This keeps the legacy single-spec hooks (``list_records`` +
        ``fetch_record``) working for importers that only pull one source
        type per import.

        Args:
            spec: The :class:`SourceSpec` row being fetched.  ``source_type``
                is the canonical type id (e.g. ``"image"``, ``"video"``).
            field_values: The same mapping passed to :meth:`run`.
            thin: When ``True``, skip downloading raw bytes — yield media
                dicts with ``media_url`` / ``media_path`` instead of
                ``media_bytes``.

        Yields:
            Raw source-type media dicts.  ``id`` and ``origin`` may be
            omitted — :meth:`run` assigns IDs and falls back to
            :meth:`build_origin` for unset origins.
        """
        del spec  # default impl is single-spec
        records = self.list_records(field_values)
        fetched = self.fetch_records_bulk(records, field_values, thin=thin)
        for media in fetched:
            if media is not None:
                yield media

    # ------------------------------------------------------------------
    # Per-record / bulk-record hooks
    # ------------------------------------------------------------------
    #
    # Convenience hooks for service-style importers that pull a single
    # source type per import.  The default :meth:`fetch_source_media`
    # delegates here.  Multi-source-type importers should override
    # :meth:`fetch_source_media` directly instead.

    def list_records(self, field_values: dict[str, Any]) -> list[Any]:
        """Return the opaque list of records to import.

        Each "record" is whatever shape the importer wants — typically a
        dict with the identifiers and URLs needed by :meth:`fetch_record`.
        The framework only cares about the count (for progress) and order
        (preserved into *medias*).

        Override this together with :meth:`fetch_record` to use the default
        :meth:`run` implementation.  Importers that override :meth:`run`
        directly do not need to implement this.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.list_records() is not implemented "
            "(override this and fetch_record(), or override run() directly). "
            "If you intended to use fetch_source_media() or "
            "fetch_all_source_media() (hooks 2/3), make sure your importer "
            "declares a 'media_type' field (or accepts a 'source_specs' "
            "value) so effective_source_specs() resolves — otherwise "
            "run() falls back to this per-record path."
        )

    def fetch_record(
        self,
        record: Any,
        field_values: dict[str, Any],
        thin: bool = False,
    ) -> dict[str, Any] | None:
        """Convert a single *record* into a media dict.

        The returned dict should contain the standard media fields
        (``type``, ``filename``, ``embedding``, ``md5``, ``media_bytes`` /
        ``media_path``, etc.).  ``id`` is assigned by the framework and may
        be omitted.  ``origin`` and ``origin_name`` may also be omitted —
        :meth:`run` falls back to :meth:`build_origin` and the filename.

        Return ``None`` to skip the record (e.g. unsupported media type).

        Override together with :meth:`list_records`.  For batched fetching,
        override :meth:`_fetch_records_bulk_impl` instead — the default
        bulk impl loops over this method.
        """
        raise NotImplementedError(f"{type(self).__name__}.fetch_record() is not implemented")

    def fetch_records_bulk(
        self,
        records: list[Any],
        field_values: dict[str, Any],
        thin: bool = False,
    ) -> list[dict[str, Any] | None]:
        """Fetch every record in *records* and return a same-length list of
        media dicts (or ``None`` for skipped records).

        Default dispatches to :meth:`_fetch_records_bulk_impl`, which loops
        over :meth:`fetch_record` one record at a time.  Subclasses backed
        by a service that natively accepts many items per request — or that
        can pipeline downloads concurrently — should override
        :meth:`_fetch_records_bulk_impl`.

        The order of the returned list matches the order of *records*.
        """
        if not records:
            return []
        return self._fetch_records_bulk_impl(records, field_values, thin=thin)

    def _fetch_records_bulk_impl(
        self,
        records: list[Any],
        field_values: dict[str, Any],
        thin: bool = False,
    ) -> list[dict[str, Any] | None]:
        """Subclass hook: fetch a list of records.

        Default loops over :meth:`fetch_record`, emitting per-item progress
        via :func:`vtscore.concurrency.progress.update_progress` so long imports stay
        visible in the UI.  Override to replace the per-item loop with a
        single bulk request, batched HTTP, or a thread/async pool.  Bulk
        overrides are responsible for emitting their own progress updates.
        """
        from vtscore.concurrency.progress import update_progress

        total = len(records)
        results: list[dict[str, Any] | None] = []
        for i, record in enumerate(records):
            update_progress("loading", f"Importing {i + 1} of {total}…", i + 1, total)
            results.append(self.fetch_record(record, field_values, thin=thin))
        return results

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
        in another field's ``depends_on`` changes — e.g. a ``query_id``
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
    # Source specs (multi-media import)
    # ------------------------------------------------------------------

    def effective_source_specs(self, field_values: dict[str, Any]) -> list[SourceSpec]:
        """Resolve *field_values* into a flat list of :class:`SourceSpec`.

        The user submits the spec list explicitly via the ``source_specs``
        form key.  This helper parses that value (either a Python list or
        a JSON-encoded string), validates each entry against the
        registered media types and converter registry, and returns the
        typed list.

        Args:
            field_values: The same dict passed to :meth:`run`.

        Returns:
            A list of :class:`SourceSpec`.  Order matches the user's
            submitted order.

        Raises:
            ValueError: If a referenced media type or converter is
                unknown, if a converter's ``target_type`` does not match
                the importer's chosen output media type, or if the import
                is missing a ``media_type`` output declaration.
        """
        from vtscore.media import get_by_folder_name  # noqa: PLC0415

        output_type_raw = field_values.get("media_type", "") or ""
        try:
            output_type = get_by_folder_name(str(output_type_raw)).type_id if output_type_raw else ""
        except (KeyError, AttributeError):
            output_type = str(output_type_raw)

        return _parse_multi_media_specs(field_values.get("source_specs"), output_type)

    def build_origin(self, field_values: dict[str, Any]) -> dict[str, Any]:
        """Build an origin dict for elements imported by this importer.

        The returned dict is the serialised form of an
        :class:`~vtscore.datasets.origin.Origin` object and is stored on
        each media as ``media["origin"]``.  It captures enough information to
        identify the data source (importer name + string-serialisable
        field values).

        Behaviour is driven declaratively by:

        - :attr:`origin_suppressed` — short-circuits to empty params.
        - :attr:`PluginField.include_in_origin` — per-field opt-out.  The
          default is ``False`` for ``"file"`` and ``"password"`` fields
          (don't persist uploads or secrets) and ``True`` for every
          other type.
        - :attr:`PluginField.origin_serializer` — per-field custom
          string conversion (e.g. comma-joining a list).
        - :attr:`extra_origin_keys` — non-``PluginField`` keys to copy
          from ``field_values``.  ``"source_specs"`` is always included.

        Subclasses may still override this method, in which case the
        override wins — every declarative knob above is ignored.  Prefer
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
    ) -> Path | None:
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
