"""Base classes for Results Exporters.

An exporter is a *destination* - a file, an SMTP server, a webhook, a browser
tab.  What gets sent there is a separate axis, with three payload kinds
(:data:`PAYLOAD_KINDS`): a scored run, a detector's labelset, or the trained
classifiers.  To add a new exporter, subclass :class:`ResultsExporter`, define
its class attributes and a method per payload kind you support, then expose a
module-level ``EXPORTER`` instance from a package under this directory.  The
registry will discover it automatically.

:attr:`~ResultsExporter.supported_payloads` is derived from which methods you
overrode, so each picker offers you only for the kinds you can actually read.
An exporter written against the older single-:meth:`~ResultsExporter.export`
contract keeps working unchanged; see that class's docstring.

Each exporter also supports CLI usage via :meth:`~ResultsExporter.add_cli_arguments`
and :meth:`~ResultsExporter.export_cli`.  The base class provides default
implementations that derive CLI arguments from the :attr:`fields` list, so most
exporters work on the command line without any extra code.  Exporters that
expect non-string values should override :meth:`export_cli` to handle the
CLI-appropriate types.

For the CLI ``--stream-results`` path (scoring a media source larger than RAM),
an exporter can write hits incrementally instead of buffering the whole result
set: set :attr:`~ResultsExporter.supports_streaming` to ``True`` and implement
:meth:`~ResultsExporter.export_cli_streaming`.  See that method's docstring and
``docs/plans/cli-stream-massive-images.md`` for details.

Example – a minimal SFTP exporter skeleton::

    # vtsearch/exporters/sftp/__init__.py
    from vtscore.exporters.base import PluginField, ResultsExporter

    class SftpResultsExporter(ResultsExporter):
        name         = "sftp"
        display_name = "SFTP Upload"
        description  = "Upload results JSON to a remote SFTP server."
        icon         = "📡"
        fields = [
            PluginField("host",     "Hostname",    "text"),
            PluginField("user",     "Username",    "text"),
            PluginField("password", "Password",    "password"),
            PluginField("path",     "Remote Path", "text",
                          default="/results/autodetect.json"),
        ]

        def _upload(self, payload: dict, field_values: dict) -> dict:
            import paramiko
            ...  # connect, write JSON, disconnect
            return {"message": f"Uploaded to {field_values['host']}:{field_values['path']}"}

        def export_find_results(self, results: dict, field_values: dict) -> dict:
            return self._upload(results, field_values)

        def export_labelset(self, labelset: dict, field_values: dict) -> dict:
            return self._upload(labelset, field_values)

    EXPORTER = SftpResultsExporter()

If the exporter needs extra packages, add them to
``[project.dependencies]`` in the repo's ``pyproject.toml``. They are
picked up the next time you run ``bash scripts/install.sh`` (or any
editable install). pyproject.toml is the single source of truth - deptry
verifies that every imported package is declared there.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

from vtscore.plugins import PluginBase, PluginField

logger = logging.getLogger(__name__)

__all__ = [
    "PAYLOAD_KINDS",
    "LabelsetExporter",
    "PluginField",
    "ResultsExporter",
    "UnsupportedPayloadError",
    "resolve_stream_batch_size",
]

#: Every payload kind an exporter can be handed, in the order they are
#: documented.  ``find_results`` is a scored run (hit lists); ``labelset`` is a
#: detector's labels (origins and vote provenance); ``detector_bundles`` is the
#: trained classifiers themselves.  See :class:`ResultsExporter`.
PAYLOAD_KINDS: tuple[str, ...] = ("find_results", "labelset", "detector_bundles")


class UnsupportedPayloadError(ValueError):
    """Raised when an exporter is handed a payload kind it does not implement.

    A :class:`ValueError` subclass so ``POST /api/exporters/export`` turns it
    into a 400 through the handler's existing ``except ValueError`` arm rather
    than a 500: asking a labelset-only exporter for a find-results export is a
    bad request, not a server fault.
    """


def resolve_stream_batch_size(value: Any, default: int = 500) -> int:
    """Coerce a streaming ``batch_size`` field value to a positive ``int``.

    The delivery-style streaming exporters (``webhook``, ``email_smtp``) group
    incoming hits into fixed-size batches so peak memory stays bounded no
    matter how many hits a run produces.  Their ``batch_size`` field arrives as
    an ``int`` from the CLI (argparse coerces ``"number"`` fields) but may be a
    string or absent from other call paths, so this helper normalises every
    shape: ``None``/blank/non-numeric fall back to *default*, and any
    non-positive value is clamped to *default* (a batch size of zero would
    never flush).
    """
    if value is None or value == "":
        return default
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return n if n > 0 else default


class ResultsExporter(PluginBase):
    """Abstract base class for results exporters.

    Subclass this, set the class-level attributes, implement the payload
    method(s) you support, and expose a module-level
    ``EXPORTER = YourExporter()`` – the registry picks it up automatically.

    Payload kinds
    -------------
    An exporter is a *destination* (a file, an SMTP server, a webhook, a
    browser tab).  What gets sent there is a separate axis, and there are three
    kinds:

    ``find_results``
        A scored run: hit lists per detector, with scores and thresholds.
        Implement :meth:`export_find_results`.  Produced by
        ``POST /api/auto-detect``, the Auto-Find auto-export, and CLI
        ``--autodetect``.

    ``labelset``
        A detector's labels: origins, vote provenance, and the metadata that
        makes the export re-importable.  Implement :meth:`export_labelset`.
        Produced by the Export modal.

    ``detector_bundles``
        The trained classifiers themselves, for a portable scoring bundle.
        Implement :meth:`export_cli_detectors` (CLI/pipeline path only).

    Most destinations can carry more than one kind, and many carry two: a CSV
    file is a fine home for either a scored run or a labelset.  Implement each
    kind you actually understand and leave the rest alone – :attr:`supported_payloads`
    is derived from which methods you overrode, the pickers only offer you for
    the kinds you claim, and the route rejects anything else with a 400 rather
    than letting you deliver an empty export.

    Legacy single-method exporters
    ------------------------------
    Before the payload kinds were named, an exporter implemented one
    :meth:`export` method and told the two dict shapes apart itself (typically
    ``if "labels" in results``).  That still works: the default
    :meth:`export_find_results` and :meth:`export_labelset` delegate to
    :meth:`export`, so an existing out-of-tree plugin needs no changes.  Such an
    exporter is credited with **both** kinds (there is no way to know which it
    handles), which is exactly the pre-existing behaviour, and it gets a
    :class:`DeprecationWarning` pointing at the named methods.

    Streaming
    ---------
    Exporters that can write a scored run incrementally (for the CLI
    ``--stream-results`` path on a media source larger than RAM) override
    :attr:`supports_streaming` to return ``True`` and implement
    :meth:`export_cli_streaming`.  Streaming is a ``find_results`` mode only;
    there is no labelset equivalent.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # A subclass that overrode only the pre-payload-kinds ``export()`` gets
        # delegated to and keeps working, but it is credited with every
        # non-detector payload kind because nothing can tell which it actually
        # handles - so a picker may hand it one it silently no-ops on.  Say so
        # once, at import time, rather than leaving the author to discover it
        # from an empty export.
        #
        # Deliberately a log line and not ``warnings.warn(DeprecationWarning)``.
        # Class bodies are executed inside the registry's import ``try``, so a
        # caller running under ``-W error`` would turn this advisory into an
        # ImportError and tombstone the very plugin the delegation exists to
        # keep working - escalating a compatibility notice into the break it
        # was written to avoid.  A log line cannot do that, and reaches the
        # author where they are already looking (the server log at startup),
        # unlike a DeprecationWarning that Python hides by default anyway.
        overrides_legacy = cls.export is not ResultsExporter.export
        overrides_named = (
            cls.export_find_results is not ResultsExporter.export_find_results
            or cls.export_labelset is not ResultsExporter.export_labelset
        )
        # A detector-bundle exporter legitimately overrides ``export()`` only to
        # explain why hits are the wrong input; it is not a legacy exporter.
        exports_bundles = cls.export_cli_detectors is not ResultsExporter.export_cli_detectors
        if overrides_legacy and not overrides_named and not exports_bundles:
            logger.warning(
                "%s implements the legacy ResultsExporter.export(); it still works, but it is offered "
                "every payload kind because nothing can tell which shapes it handles. Override "
                "export_find_results() and/or export_labelset() instead - see "
                "vtscore/docs/extending/results-exporters.md.",
                cls.__name__,
            )

    #: Emoji or icon string shown next to the display name in the UI.
    icon: str = "📤"
    #: Ordered list of fields the user must fill before exporting.
    #: Leave empty if the exporter needs no configuration.
    fields: list[PluginField]

    #: When ``True``, this exporter's :meth:`export` returns an ``"open_url"``
    #: key and the frontend opens that URL in a new browser tab.  Declaring it
    #: is what lets the Export modal label the button "Open in <name>" *before*
    #: the export runs; the frontend still honours an ``open_url`` from an
    #: exporter that leaves this ``False`` (e.g. a delivery exporter whose
    #: remote happens to hand back a permalink), it just can't advertise it up
    #: front.  See :meth:`export` for the response contract.
    opens_url: bool = False

    @property
    def supported_payloads(self) -> frozenset[str]:
        """Which of :data:`PAYLOAD_KINDS` this exporter actually implements.

        Derived from which methods the subclass overrode, never declared.  A
        declared flag is a second place to forget something, and forgetting is
        precisely how an exporter ends up in a picker for a payload it cannot
        read; a derived set cannot drift from the implementation.

        The rules, in order:

        - :attr:`needs_trained_detectors` means "this exporter consumes the
          trained classifiers, not their output", so it reports
          ``{"detector_bundles"}`` alone.
        - Otherwise, overriding :meth:`export_find_results` or
          :meth:`export_labelset` claims that kind, and overriding the legacy
          :meth:`export` claims **both** (nothing can tell which dict shapes it
          handles - see the class docstring).
        - Overriding :meth:`export_cli_detectors` adds ``detector_bundles``.

        A subclass of a concrete exporter inherits its parent's overrides, and
        therefore its parent's payload kinds, which is the intended answer.
        """
        cls = type(self)
        if self.needs_trained_detectors:
            return frozenset({"detector_bundles"})

        kinds: set[str] = set()
        legacy = cls.export is not ResultsExporter.export
        if legacy or cls.export_find_results is not ResultsExporter.export_find_results:
            kinds.add("find_results")
        if legacy or cls.export_labelset is not ResultsExporter.export_labelset:
            kinds.add("labelset")
        if cls.export_cli_detectors is not ResultsExporter.export_cli_detectors:
            kinds.add("detector_bundles")
        return frozenset(kinds)

    def export_find_results(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        """Export a scored run - the hit lists a detector produced.

        Args:
            results: The auto-detect results dict.  Shape::

                         {
                           "media_type": "audio",
                           "detectors_run": 2,
                           "results": {
                             "<detector_name>": {
                               "detector_name": "...",
                               "threshold": 0.5,
                               "total_hits": 15,      # positives only
                               "hits": [{...}, ...],
                               "negative_hits": [{...}, ...],
                             }
                           },
                           "missing_detectors": [...],
                         }

                     Each hit comes from
                     :func:`vtscore.utils.hits.build_media_hit`: ``id``,
                     ``filename``, ``category``, ``score``, plus ``origin`` /
                     ``origin_name`` / ``md5`` and any clip bounds the media
                     carries.

                     **``negative_hits`` is conventionally ignored.** Every
                     built-in exporter writes positives only, because "the
                     items the detector found" is what an export is for.  The
                     key is nonetheless always present (the CLI fills it only
                     under ``--keep-negatives``), so an exporter that wants the
                     below-threshold items can read it.  ``missing_detectors``
                     is likewise informational.

            field_values: Mapping of :attr:`PluginField.key` -> value supplied
                by the user.

        Returns:
            A status dict; see :meth:`export` for the ``"message"`` /
            ``"open_url"`` / ``"display_results"`` contract, which is shared by
            all three payload methods.

        Raises:
            UnsupportedPayloadError: If this exporter does not handle a scored
                run (the default, unless the legacy :meth:`export` is
                overridden).
        """
        return self._delegate_to_legacy_export("find_results", results, field_values)

    def export_labelset(self, labelset: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        """Export a detector's labelset - what it was taught, not what it found.

        Args:
            labelset: A serialised
                :class:`~vtscore.datasets.labelset.LabelSet`.  Shape::

                         {
                           "labels": [{...}, ...],
                           "selected_columns": ["label", "md5", ...],
                         }

                     Each entry is a
                     :class:`~vtscore.datasets.labelset.LabeledElement` dict:
                     ``md5`` and ``label`` always, plus ``origin`` /
                     ``origin_name`` / ``filename`` / ``category`` /
                     ``metadata`` / ``region_box`` where present.  The
                     ``origin`` is what makes the export re-importable, so
                     preserve it rather than flattening it to a string.

                     ``selected_columns`` is the user's column choice from the
                     Export modal.  Honour it for tabular output; a delivery
                     exporter that sends the whole entry may ignore it.  It is
                     absent on non-UI call paths.

            field_values: Mapping of :attr:`PluginField.key` -> value supplied
                by the user.

        Returns:
            A status dict; see :meth:`export`.

        Raises:
            UnsupportedPayloadError: If this exporter does not handle a
                labelset (the default, unless the legacy :meth:`export` is
                overridden).
        """
        return self._delegate_to_legacy_export("labelset", labelset, field_values)

    def _delegate_to_legacy_export(
        self,
        kind: str,
        payload: dict[str, Any],
        field_values: dict[str, Any],
    ) -> dict[str, Any]:
        """Hand *payload* to a legacy :meth:`export`, or refuse it clearly.

        The compatibility hinge: an exporter written before the payload kinds
        were named implements :meth:`export` and sniffs the dict shape itself,
        so the named methods route to it unchanged.  One that implements
        neither gets an :class:`UnsupportedPayloadError` naming the kind, which
        the route turns into a 400.
        """
        if type(self).export is ResultsExporter.export:
            raise UnsupportedPayloadError(
                f"{type(self).__name__} does not export a {kind} payload "
                f"(supported: {', '.join(sorted(self.supported_payloads)) or 'none'})"
            )
        return self.export(payload, field_values)

    def export(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        """Legacy single-method export; prefer the payload-specific methods.

        Kept working, and not going anywhere, so an out-of-tree exporter written
        against it needs no changes: :meth:`export_find_results` and
        :meth:`export_labelset` both delegate here when the subclass hasn't
        overridden them.  New exporters should implement those instead - an
        exporter that only implements this one is credited with both payload
        kinds, so a picker can hand it a shape it doesn't read.

        This method's own contract is unchanged: *results* is whichever payload
        the caller had, and an implementation tells them apart with
        ``if "labels" in results``.

        The return contract below is shared by all three payload methods.

        Args:
            results: The full auto-detect results dict from ``/api/auto-detect``
                     (or a serialised LabelSet - see above).  Shape::

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

            field_values: Mapping of :attr:`PluginField.key` → value string
                supplied by the user.

        Returns:
            A dict that **must** contain a ``"message"`` key with a short
            human-readable confirmation string.  It may also carry arbitrary
            extra keys (e.g. ``"filepath"`` for file-based exporters).

            Two extra keys are understood by the frontend rather than merely
            passed through:

            ``"display_results"``
                The GUI exporter's pass-through of the results dict, rendered
                in the Auto-Detect Results modal.

            ``"open_url"``
                An ``http(s)`` URL the browser opens in a new tab.  This is how
                an exporter hands the user off to a third-party site that has
                no ingest API: format the labelset into the site's own URL
                (query string, fragment, path) and return it.  It also fits a
                delivery exporter whose remote returns a permalink to what was
                just uploaded.  Exporters that always return one should also
                set :attr:`opens_url` so the button can say so beforehand.

                The route re-validates the URL with
                :func:`vtscore.security.url_validation.validate_browser_url`
                (scheme allowlist — deliberately *not* the SSRF guard, since
                the browser makes the request and a ``localhost`` viewer is a
                legitimate target) and fails the export if it doesn't pass, so
                a plugin can never push a ``javascript:`` URL to the frontend.

                Anything encoded into the URL is visible to the target site and
                lands in the user's browser history; keep it to identifiers.

        Raises:
            NotImplementedError: If the subclass has not implemented this.
            Exception: Any exception propagates to the route handler, which
                returns it as a 500 JSON error.
        """
        raise NotImplementedError(f"{type(self).__name__}.export() is not implemented")

    # ------------------------------------------------------------------
    # Payload-specific export methods
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise exporter metadata, adding the exporter-only fields.

        :attr:`opens_url` rides along so ``GET /api/exporters`` tells the
        frontend which exporters end in a new browser tab.  It is declared
        here rather than on :class:`~vtscore.plugins.PluginBase` because no
        other plugin kind has anywhere to open a URL.

        :attr:`supported_payloads` rides along for the same reason: it is what
        lets each picker offer only the exporters that can read the payload it
        is about to send, instead of listing the whole registry and finding out
        afterwards.  Serialised sorted, so the API response is stable.
        """
        return {
            **super().to_dict(),
            "opens_url": self.opens_url,
            "supported_payloads": sorted(self.supported_payloads),
        }

    # ------------------------------------------------------------------
    # CLI support
    # ------------------------------------------------------------------

    def export_cli(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        """Export a scored run from CLI-provided *field_values*.

        The CLI autodetect/pipeline path only ever produces find results, so the
        default delegates to :meth:`export_find_results` (which in turn reaches
        a legacy :meth:`export`, if that is all the exporter implements).
        Exporters that need different behaviour on the command line (e.g. the
        GUI exporter, which has no browser) should override this method.
        """
        return self.export_find_results(results, field_values)

    # ------------------------------------------------------------------
    # Trained-detector CLI support (portable-detector export)
    # ------------------------------------------------------------------

    @property
    def needs_trained_detectors(self) -> bool:
        """Whether this exporter consumes the trained classifiers, not the hits.

        Almost every exporter serialises the scored *results* (the hit lists).
        The portable-detector exporter is the exception: it serialises the
        trained MLP itself (an ONNX scoring bundle), so it needs the in-memory
        detectors the CLI pipeline just trained rather than their output.  When
        this returns ``True`` the pipeline routes to :meth:`export_cli_detectors`
        instead of :meth:`export_cli`; results-consuming exporters leave it
        ``False``.
        """
        return False

    def export_cli_detectors(
        self,
        detectors: list[dict[str, Any]],
        field_values: dict[str, Any],
    ) -> dict[str, Any]:
        """Export the trained detectors themselves (not the scored hits).

        Called by the CLI/pipeline path for exporters whose
        :attr:`needs_trained_detectors` is ``True``.  *detectors* is the list
        of trained detectors the pipeline produced against the loaded dataset;
        each entry is a descriptor dict with these keys::

            {
              "detector_name": str,        # the detector's name
              "media_type":    str,        # audio / image / video / ...
              "weights":       dict,       # serialize_weights() nested lists
              "threshold":     float,      # decision threshold
              "embedder":      str,        # concrete embedder name it trained in
              "embedder_type": str,        # the detector's locked embedder type
              "good_count":    int,        # labelset good count
              "bad_count":     int,        # labelset bad count
            }

        Returns a status dict with at minimum a ``"message"`` key, like
        :meth:`export`.

        Raises:
            NotImplementedError: If the exporter does not export trained
                detectors (the default).
        """
        raise NotImplementedError(f"{type(self).__name__} does not export trained detectors")

    # ------------------------------------------------------------------
    # Streaming CLI support (massive sources)
    # ------------------------------------------------------------------

    @property
    def supports_streaming(self) -> bool:
        """Whether this exporter can write results incrementally.

        Exporters that override :meth:`export_cli_streaming` return ``True``
        so the ``--stream-results`` CLI path can route to them.  File-based
        exporters flush each hit as it streams; delivery-based ones
        (``webhook``, ``email_smtp``) batch hits into fixed-size groups and
        deliver each batch as it fills.  Exporters that leave this ``False``
        have no incremental mode, so requesting ``--stream-results`` with them
        is rejected with a clear error.

        See ``docs/plans/cli-stream-massive-images.md``.
        """
        return False

    def export_cli_streaming(
        self,
        header: dict[str, Any],
        records: Iterator[tuple[str, dict[str, Any]]],
        field_values: dict[str, Any],
    ) -> dict[str, Any]:
        """Write results incrementally as scored chunks stream in.

        Unlike :meth:`export_cli`, which receives the fully-materialised
        results dict, this method receives a lazy *records* iterator and is
        expected to write each record to the destination as it arrives,
        never buffering the whole set.  This is what lets ``--autodetect``
        run against a media source with more items (and more hits) than fit
        in RAM.

        Args:
            header: Metadata known before any hit streams, with keys
                ``"media_type"`` (str), ``"detectors"`` (a list of
                ``{"detector_name": str, "threshold": float}`` dicts), and
                ``"keep_negatives"`` (bool — whether below-threshold hits are
                included in *records*).
            records: Yields ``(detector_name, hit)`` tuples in chunk order
                (NOT globally sorted by score).  Each *hit* is the dict from
                :func:`vtscore.utils.hits.build_media_hit` plus a ``"label"``
                key (``"good"`` for above-threshold, ``"bad"`` otherwise).
            field_values: Mapping of :attr:`PluginField.key` → value.

        Returns:
            A status dict with a ``"message"`` key (and optionally
            ``"filepath"``), like :meth:`export`.

        Raises:
            NotImplementedError: If the exporter does not support streaming.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support streaming export")


#: Backwards-compatible alias for the pre-payload-kinds class name.
#:
#: ``LabelsetExporter`` described one of the three payloads the class has
#: always accepted, which is why every guide had to apologise for it.  The
#: alias is permanent: it costs a line and keeps every out-of-tree ``from
#: vtscore.exporters.base import LabelsetExporter`` and subclass working.
LabelsetExporter = ResultsExporter
