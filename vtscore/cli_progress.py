"""CLI progress output formatting.

Selected via ``--progress-format {text,json}`` on ``python app.py``. In ``text``
mode (the default) the CLI prints human-readable prose, preserving the
pre-flag behaviour byte-for-byte. In ``json`` mode the same status updates
are emitted to stdout as **NDJSON** - one JSON object per line, each shaped
``{"event": <name>, "ts": <iso8601-z>, ...}`` - so scripts and CI runners
can consume progress without scraping prose or tqdm glyphs.

Event names emitted today:

- ``chunk_start``    - a new chunk is about to be scored
  fields: ``chunk_num`` (int), ``chunk_size`` (int)
- ``chunks_done``    - chunked scoring finished
  fields: ``total_medias`` (int), ``chunks`` (int)
- ``labels_imported`` - one-shot ``--import-labels-into`` finished
  fields: ``detector`` (str), ``applied`` (int), ``skipped`` (int)
- ``export_complete`` - exporter finished its run
  fields: ``message`` (str - the exporter's own confirmation text)
- ``progress``       - a tick from the embedding / loading stack
  fields: ``status`` (str), ``message`` (str?), ``current`` (int?),
  ``total`` (int?), ``pct`` (float? - only when total > 0)
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
from typing import Any, TextIO

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
