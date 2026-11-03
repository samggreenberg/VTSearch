"""CLI progress output formatting.

Selected via ``--progress-format {text,json}`` on ``python app.py``. In ``text``
mode (the default) the CLI prints human-readable prose, preserving the
pre-flag behaviour byte-for-byte. In ``json`` mode the same status updates
are emitted to stdout as **NDJSON** - one JSON object per line, each shaped
``{"event": <name>, "ts": <iso8601-z>, ...}`` - so scripts and CI runners
can consume progress without scraping prose or tqdm glyphs.

The events, from :mod:`vtscore.cli`, :mod:`vtscore.cli.pipeline` and the
app's ``vtsearch/cli/main.py``:

- ``chunk_start``    - a new chunk is about to be scored
  fields: ``chunk_num`` (int), ``chunk_size`` (int)
- ``chunks_done``    - chunked scoring finished
  fields: ``total_medias`` (int), ``chunks`` (int)
- ``resume_start``   - a resumable streamed run (``--resume``) begins, fresh
  or continuing from its checkpoint
  fields: ``checkpoint`` (str), ``chunks_done`` (int), ``resumed`` (bool)
- ``chunk_skipped``  - a resumed run passed a chunk it had already written
  and checked it against the checkpoint, without scoring it
  fields: ``chunk_num`` (int), ``chunk_size`` (int), ``read`` (bool -
  whether its files were re-read to check it, or only walked)
- ``resume_threshold_kept`` - a resumed run's retrained threshold moved by
  floating-point noise; the checkpoint's is kept
  fields: ``detector`` (str), ``retrained`` (float), ``recorded`` (float)
- ``resume_complete`` - a resumable run finished and removed its checkpoint
  fields: ``checkpoint`` (str), ``chunks`` (int)
- ``medias_unembedded`` - media still carry no embedding after the load's
  embed pass, and score only through a converter route
  fields: ``unembedded`` (int), ``unembedded_ids`` (list[int] - the first 100)
- ``medias_skipped`` - media with no usable embedding in a detector's space
  were left out of its scoring
  fields: ``skipped`` (int), ``skipped_ids`` (list[int] - the first 100),
  ``embedder`` (str)
- ``detector_skipped`` - a detector's ``media_type`` has no direct or
  converter route from the dataset's types, so it does not run
  fields: ``detector`` (str), ``detector_media_type`` (str),
  ``dataset_media_type`` (str)
- ``detector_reclip`` - the dataset is re-clipped to a detector's
  ``input_spec`` clipper at scoring time, or to the clipper override
  (``--override-clipper``) in its place
  fields: ``detector`` (str), ``detector_input_spec`` (dict),
  ``dataset_input_spec`` (dict), ``override_input_spec`` (dict? - only when
  the override did it)
- ``detector_centroid`` / ``detector_query`` - a detector's labels give no
  trained head, so it scores as the Goods' centroid or as its typed query's
  own set
  fields: ``detector`` (str), ``query`` (str - ``detector_query`` only), and
  the label quota's ``tier``, ``n_good``, ``n_bad``, ``goods_owed``,
  ``bads_owed``, ``good_quota``, ``bad_quota``, ``dry_bad_quota``,
  ``hold_labels``, ``hold_owed`` (all int but ``tier``, a str)
- ``detector_unchecked`` - a detector's line is exported unchecked, since a
  headless run cannot spot-check it
  fields: ``detector`` (str), ``beta`` (float), ``status`` (str),
  ``count`` (int - the size of the set the line keeps)
- ``labels_import_dry_run`` - ``--dry-run`` with ``--import-labels-into``:
  the label import that would run
  fields: ``detector`` (str), ``importer`` (str), ``filepath`` (str?),
  ``fields`` (dict), ``create_media_type`` (str? - set when
  ``--create-detector`` would make the detector)
- ``labels_imported`` - one-shot ``--import-labels-into`` (or a pipeline
  file's ``import_labels``) finished
  fields: ``detector`` (str), ``applied`` (int), ``skipped`` (int),
  ``created`` (bool - ``--create-detector`` made the detector)
- ``export_complete`` - exporter finished its run
  fields: ``message`` (str - the exporter's own confirmation text),
  ``open_url`` (str? - only when the exporter returned one; an ``http(s)``
  URL for the caller to open, since the CLI has no browser of its own)
- ``progress``       - a tick from the embedding / loading stack
  fields: ``status`` (str), ``message`` (str?), ``current`` (int?),
  ``total`` (int?), ``pct`` (float? - only when total > 0), ``dataset``
  (str? - the dataset's name, when a multi-dataset run imports several at
  once)
- ``import_progress`` - ``text`` mode only, one line per phase of a saving
  import (``json`` mode sends each tick as a ``progress`` event instead)
  fields: none
- ``dataset_saved``  - a saving run registered (or found) its dataset on the
  dashboard; once per dataset of a multi-dataset run
  fields: ``dataset_id`` (str), ``name`` (str), ``num_items`` (int),
  ``pkl_path`` (str), ``already_saved`` (bool - the pickle was already a
  registered dataset, so nothing new was imported)
- ``detection_skipped`` - a run had no detector to score its dataset with; a
  saved dataset stays saved, and a single-dataset run exits 0
  fields: ``reason`` (str), ``dataset`` (str? - the dataset's name, in a
  multi-dataset run, whose other datasets still run)
- ``dataset_deleted`` - a saving run deleted the dataset it imported once its
  detectors had run, because the user's ``autofind_cli_delete_dataset``
  setting is on
  fields: ``dataset_id`` (str), ``name`` (str)
- ``dataset_start``  - a multi-dataset run (``--outputs``, #4707) is about to
  score one of its datasets; what follows, up to the next one, is that
  dataset's
  fields: ``name`` (str), ``media_type`` (str), ``category`` (str),
  ``dataset_id`` (str? - when saved), ``index`` (int, 0-based), ``count`` (int)
- ``dataset_failed`` - one dataset of a multi-dataset run could not be
  imported; the others still run, and the run exits non-zero at the end
  fields: ``name`` (str), ``error`` (str)
- ``dry_run_plan`` - ``--dry-run`` in ``json`` mode: what the run would do
  (``text`` mode prints the plan as prose instead)
  fields: ``source`` (dict), ``settings_path`` (str?),
  ``autofind_detectors`` (list[dict] - one summary per detector the run
  would train), ``detectors_source`` (str - ``"autofind"`` or
  ``"override"``), ``override_clipper`` (dict? - the clipper override's
  ``input_spec``), ``exporter`` (str?), ``exporter_field_values`` (dict)
- ``authenticated_user`` - ``--user`` with ``--api-key`` authenticated; the
  run executes as that user
  fields: ``user`` (str)
- ``notification``   - a non-fatal message a plugin wanted the user to see
  (see :mod:`vtscore.concurrency.notifications`; in the GUI these become
  toasts). The run continues either way, including at ``level="error"``.
  fields: ``level`` (str), ``message`` (str), ``detail`` (str?),
  ``source`` (str?), ``items`` (list[str]?)
- ``error``          - fatal error; the process will exit non-zero
  fields: ``message`` (str)

In JSON mode every event is written to stdout (including errors), so a
single ``stdout`` pipe captures the full stream. Stderr is reserved for
unstructured noise (tqdm bars, library warnings); consumers can discard
it with ``2>/dev/null`` and still see error events on stdout.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, TextIO

if TYPE_CHECKING:
    from vtscore.concurrency.notifications import Notification

#: Allowed values for ``--progress-format``.
FORMATS = ("text", "json")

_format: str = "text"


def set_format(fmt: str) -> None:
    """Set the active progress output format (``"text"`` or ``"json"``)."""
    global _format
    if fmt not in FORMATS:
        raise ValueError(f"Unknown progress format: {fmt!r}. Choose one of {FORMATS}.")
    _format = fmt


def get_format() -> str:
    """Return the active progress output format."""
    return _format


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def emit(
    event: str,
    *,
    text: str | None = None,
    stream: TextIO | None = None,
    **fields: Any,
) -> None:
    """Emit a status update for humans or for scripts.

    - In ``text`` mode: writes *text* (if given) to *stream* with a trailing
      newline and a flush. When *text* is ``None`` nothing is written -
      callers use this for JSON-only events that have no prose analogue.
    - In ``json`` mode: writes one NDJSON line
      ``{"event": event, "ts": ..., **fields}`` to *stream* and flushes.

    *stream* defaults to ``sys.stdout``. Errors should also go to stdout in
    JSON mode (so a single pipe captures the whole stream); use
    :func:`emit_error` which routes correctly for each format.
    """
    out: TextIO = stream if stream is not None else sys.stdout
    if _format == "json":
        line = json.dumps({"event": event, "ts": _now(), **fields}, default=str)
        out.write(line + "\n")
        out.flush()
        return
    if text is not None:
        out.write(text + "\n")
        out.flush()


def emit_error(message: str) -> None:
    """Emit a fatal-error event.

    In text mode this prints ``Error: <message>`` to stderr (matching the
    pre-flag CLI behaviour). In JSON mode it writes an ``{"event":"error",
    ...}`` line to **stdout** so the same pipe captures the failure
    record. Callers still call :func:`sys.exit` themselves after emitting.
    """
    if _format == "json":
        emit("error", message=message)
    else:
        sys.stderr.write(f"Error: {message}\n")
        sys.stderr.flush()


def progress_callback(status: str, message: str = "", current: int = 0, total: int = 0) -> None:
    """:data:`~vtscore.media.base.ProgressCallback` that emits JSON events.

    In ``text`` mode this is a no-op (the embedding stack already paints
    its own tqdm bars on stderr, which the pre-flag CLI relied on). In
    ``json`` mode each tick becomes one ``progress`` event. Ticks with
    no useful content (no message and no total) are dropped to avoid
    spamming consumers with empty ``{"status":"idle"}`` records.
    """
    if _format != "json":
        return
    if not message and total <= 0:
        return
    fields: dict[str, Any] = {"status": status}
    if message:
        fields["message"] = message
    if total > 0:
        fields["current"] = current
        fields["total"] = total
        fields["pct"] = round(current * 100 / total, 1)
    emit("progress", **fields)


#: Prose label per notification level, for ``text`` mode. Mirrors the
#: ``Error:`` prefix :func:`emit_error` already uses, so a reader does not have
#: to learn a second vocabulary for the same severity ladder.
_NOTIFICATION_LABEL = {
    "info": "Note",
    "success": "Done",
    "warning": "Warning",
    "error": "Error",
}


def notification_subscriber(notification: "Notification") -> None:
    """Print a user notification, in whichever format the CLI is running.

    Subscribed to :data:`vtscore.concurrency.notifications.notifications` for
    the life of a CLI run so the messages plugins raise for the GUI's toasts
    are not silently dropped in headless mode.

    In ``text`` mode this writes one prose line to **stderr**, keeping stdout
    reserved for the run's real output (an exporter's confirmation, a results
    dump a caller may be piping). In ``json`` mode it writes a ``notification``
    NDJSON record to **stdout** with the rest of the event stream, so one pipe
    still captures everything.

    Note that ``level="error"`` here is *not* :func:`emit_error`: a
    notification never ends the run. It reports something the code decided to
    continue past, which is the entire point of the notification channel.
    """
    label = _NOTIFICATION_LABEL.get(notification.level, "Note")
    source = f" [{notification.source}]" if notification.source else ""
    detail = f" - {notification.detail}" if notification.detail else ""
    # The GUI keeps the item list behind a Details toggle; headless there is
    # nothing to toggle, so each item gets its own indented line.
    items = "".join(f"\n  - {item}" for item in notification.items or ())
    emit(
        "notification",
        text=f"{label}:{source} {notification.message}{detail}{items}",
        stream=None if _format == "json" else sys.stderr,
        level=notification.level,
        message=notification.message,
        detail=notification.detail,
        source=notification.source,
        items=list(notification.items) if notification.items else None,
    )
