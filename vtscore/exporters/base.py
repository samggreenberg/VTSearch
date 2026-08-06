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

For the CLI ``--stream-results`` path (scoring a media source larger than RAM),
an exporter can write hits incrementally instead of buffering the whole result
set: set :attr:`~LabelsetExporter.supports_streaming` to ``True`` and implement
:meth:`~LabelsetExporter.export_cli_streaming`.  See that method's docstring and
``docs/plans/cli-stream-massive-images.md`` for details.

Example – a minimal SFTP exporter skeleton::

    # vtsearch/exporters/sftp/__init__.py
    from vtscore.exporters.base import LabelsetExporter, PluginField

    class SftpLabelsetExporter(LabelsetExporter):
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

        def export(self, results: dict, field_values: dict) -> dict:
            import paramiko
            ...  # connect, write JSON, disconnect
            return {"message": f"Uploaded to {field_values['host']}:{field_values['path']}"}

    EXPORTER = SftpLabelsetExporter()

If the exporter needs extra packages, add them to
``[project.dependencies]`` in the repo's ``pyproject.toml``. They are
picked up the next time you run ``bash scripts/install.sh`` (or any
editable install). pyproject.toml is the single source of truth - deptry
verifies that every imported package is declared there.
"""

from __future__ import annotations

from typing import Any, Iterator

from vtscore.plugins import PluginBase, PluginField

__all__ = ["LabelsetExporter", "PluginField", "resolve_stream_batch_size"]


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


class LabelsetExporter(PluginBase):
    """Abstract base class for results exporters.

    Subclass this, set the class-level attributes, implement :meth:`export`,
    and expose a module-level ``EXPORTER = YourExporter()`` – the registry
    picks it up automatically.

    The :meth:`export` method receives the full results dict returned by
    ``/api/auto-detect`` and a flat mapping of field values supplied by the
    user via the UI.  It should return a dict with at minimum a ``"message"``
    key describing what happened (shown to the user as confirmation).

    Exporters that can write results incrementally (for the CLI
    ``--stream-results`` path on a media source larger than RAM) override
    :attr:`supports_streaming` to return ``True`` and implement
    :meth:`export_cli_streaming`.
    """

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
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise exporter metadata, adding the exporter-only flags.

        :attr:`opens_url` rides along so ``GET /api/exporters`` tells the
        frontend which exporters end in a new browser tab.  It is declared
        here rather than on :class:`~vtscore.plugins.PluginBase` because no
        other plugin kind has anywhere to open a URL.
        """
        return {**super().to_dict(), "opens_url": self.opens_url}

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
