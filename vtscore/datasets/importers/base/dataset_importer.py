"""The full-featured importer base (:class:`DatasetImporter`).

``DatasetImporter`` layers the source-spec → converter → ingestion pipeline
(plus the ``list_records`` / ``fetch_record`` convenience hooks) on top of the
thin :class:`~vtscore.datasets.importers.base.core.ImporterBase`.  It is the
class most importers subclass; importers that take full control of their own
ingestion (and never touch source specs or the per-record hooks) can subclass
``ImporterBase`` directly for a leaner surface.
"""

from __future__ import annotations

from typing import Any, Iterator

from .core import ImporterBase
from .specs import SourceSpec, _fill_converter_output_fields, _parse_multi_media_specs


class DatasetImporter(ImporterBase):
    """Abstract base class for spec-driven and service-style dataset importers.

    Subclass this, set the class-level attributes, implement :meth:`run`
    (or one of the fetch hooks below), and expose a module-level
    ``IMPORTER = YourImporter()`` – the registry picks it up automatically.

    Everything :class:`~vtscore.datasets.importers.base.core.ImporterBase`
    documents (the embedding contract, custom metadata, content vectors /
    MD5s, the precomputed-dict helpers, CLI support) applies here too; this
    subclass adds the framework-driven ingestion machinery described under
    :meth:`run`.
    """

    def run(self, field_values: dict[str, Any], medias: dict, thin: bool = False) -> None:
        """Perform the import, populating *medias* in-place.

        Subclasses pick one of four override points, in order of increasing
        control.  Hooks 1–3 leave conversion and ingestion to the framework;
        only hook 4 takes that responsibility back.  The numbering here
        matches ``docs/EXTENDING-plugins.md``.

        1. :meth:`list_records` + :meth:`fetch_record` (and optionally
           :meth:`_fetch_records_bulk_impl` for batched fetches); a
           single-spec convenience for service importers that only pull one
           source type.  The default :meth:`fetch_source_media` delegates to
           these hooks, so single-spec importers can keep using the
           per-record split without thinking about specs.  This hook is also
           the fallback when :meth:`effective_source_specs` cannot resolve
           (no ``media_type`` declared).
        2. :meth:`fetch_source_media`: yield raw source-type media one record
           at a time, for one :class:`SourceSpec` at a time.  The framework
           loops over each spec produced by :meth:`effective_source_specs`,
           calls this method per spec, runs the spec's converter (when one is
           set) on every yielded media, and ingests the results.  Recommended
           for service-style multi-source-type importers whose backend
           serves one media type per query.
        3. :meth:`fetch_all_source_media`: yield ``(spec, raw_media)`` pairs
           for **all** specs in one pass.  Override this when one upstream
           call returns mixed source types and you want to make it just once
           (e.g. a service whose query returns "everything that matched" with
           a per-record type tag).  The framework still runs converters and
           ingests; subclasses never call
           :func:`~vtscore.converters.get_converter` themselves.  The default
           implementation delegates to :meth:`fetch_source_media` per spec,
           so importers using hook 1 or 2 don't need to know this hook exists.
        4. :meth:`run` directly: full control.  Folder-shaped importers
           override this and delegate to
           :func:`~vtscore.converters.runner.run_converters_on_folder`.

        Default flow:

        - When :meth:`effective_source_specs` resolves to one or more specs,
          call :meth:`fetch_all_source_media` (hook 3, which by default loops
          :meth:`fetch_source_media` (hook 2) once and pass each yielded
          ``(spec, raw)`` pair through ``spec.converter`` (when set) before
          assigning IDs and storing the result in *medias*.
        - When :meth:`effective_source_specs` cannot resolve (no
          ``media_type`` declared (i.e. a bare service importer), fall
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
                if spec.converter is not None:
                    _fill_converter_output_fields(media)
                medias[next_id] = media
                next_id += 1
        return next_id

    # ------------------------------------------------------------------
    # Multi-media source hooks
    # ------------------------------------------------------------------
    #
    # Two override points for service-style importers.  In both cases the
    # framework drives the converter loop and ingestion, so subclasses
    # never call :func:`vtscore.converters.get_converter` themselves;
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
        record paired with the :class:`SourceSpec` it satisfies; the
        framework handles converter dispatch and ingestion exactly as it
        does for :meth:`fetch_source_media`.

        For per-spec importers (the common case: one query per source
        type), override :meth:`fetch_source_media` instead; the default
        implementation of this method loops it for you.

        Each yielded ``raw_media`` dict must match the shape expected of
        media of ``spec.source_type`` (so a video spec yields
        ``type="video"`` dicts with ``media_bytes`` / ``media_path``; the
        framework hands them to the spec's converter, which produces
        e.g. ``type="image"`` dicts).  ``id`` and ``origin`` may be
        omitted; :meth:`run` assigns IDs and falls back to
        :meth:`build_origin` for unset origins.

        Args:
            specs: The full list of :class:`SourceSpec` rows resolved
                from *field_values*, in the user's submitted order.
            field_values: The same mapping passed to :meth:`run`.
            thin: When ``True``, skip downloading raw bytes; yield media
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
            thin: When ``True``, skip downloading raw bytes; yield media
                dicts with ``media_url`` / ``media_path`` instead of
                ``media_bytes``.

        Yields:
            Raw source-type media dicts.  ``id`` and ``origin`` may be
            omitted; :meth:`run` assigns IDs and falls back to
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

        Each "record" is whatever shape the importer wants (typically a
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
            "value) so effective_source_specs() resolves; otherwise "
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
        be omitted.  ``origin`` and ``origin_name`` may also be omitted;
        :meth:`run` falls back to :meth:`build_origin` and the filename.

        Return ``None`` to skip the record (e.g. unsupported media type).

        Override together with :meth:`list_records`.  For batched fetching,
        override :meth:`_fetch_records_bulk_impl` instead; the default
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
        by a service that natively accepts many items per request, or that
        can pipeline downloads concurrently, should override
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
