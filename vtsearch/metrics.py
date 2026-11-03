"""Usage metrics: the server-side sink for the SPA's metrics events (#5475).

The frontend's ``MetricsService`` hands every usage event (a view opened, a
dataset imported, results exported, a setting changed, a tagged control
clicked) to a pluggable provider. The default provider batches them to
``POST /api/metrics``, and this module appends each one to a JSON-lines file:

.. code-block:: json

    {"ts": "2026-10-30T16:31:24.420Z", "user": "default", "session": "k3f9q2",
     "group": "view", "attributes": {"view": "browse"}, "client_ts": 1793385084301}

``ts`` is the server's receive time (UTC); ``client_ts`` is when the browser
logged the event (epoch milliseconds), since a batch reaches the server a few
seconds after its first event. ``session`` identifies one page load, so a
user's events can be read as visits.

Files roll monthly (``events-YYYY-MM.jsonl``) under :data:`METRICS_DIR`, so no
file grows without bound and a month's usage is one file to read. Attributes
are kept flat: a nested value is dropped and a long string is cut, because a
metrics line answers "who used what", not "with which data".

The server-tier ``enable_metrics`` setting switches the sink off. Metrics are
best-effort throughout: a write that fails is logged and dropped, never
surfaced to the user as an error.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vtscore.config import DATA_DIR
from vtsearch import settings

logger = logging.getLogger(__name__)

#: Directory the monthly event files live in. Tests repoint it at ``tmp_path``.
METRICS_DIR: Path = DATA_DIR / "metrics"

#: Most attributes one event keeps; the rest are dropped.
MAX_ATTRIBUTES: int = 20

#: Longest attribute key / string value kept, in characters.
MAX_KEY_LENGTH: int = 64
MAX_STRING_LENGTH: int = 200

# Serialises appends within this process. Each batch goes to the file in one
# ``write`` on an ``O_APPEND`` handle, so batches from separate worker
# processes land whole as well.
_write_lock = threading.Lock()

AttributeValue = str | int | float | bool | None


def is_enabled() -> bool:
    """Whether the server records metrics (the ``enable_metrics`` setting)."""
    return bool(settings.get_enable_metrics())


def metrics_file(when: datetime) -> Path:
    """The file events received at *when* are appended to."""
    return METRICS_DIR / f"events-{when:%Y-%m}.jsonl"


def _clean_value(value: Any) -> tuple[bool, AttributeValue]:
    """Return ``(keep, cleaned)`` for one attribute value."""
    if value is None or isinstance(value, bool):
        return True, value
    if isinstance(value, int):
        return True, value
    if isinstance(value, float):
        return math.isfinite(value), value
    if isinstance(value, str):
        return True, value[:MAX_STRING_LENGTH]
    return False, None


def clean_attributes(raw: Mapping[str, Any]) -> dict[str, AttributeValue]:
    """Flatten *raw* to at most :data:`MAX_ATTRIBUTES` scalar entries.

    Strings, numbers, booleans and ``None`` survive (strings cut to
    :data:`MAX_STRING_LENGTH`); lists, dicts and non-finite floats are dropped.
    """
    out: dict[str, AttributeValue] = {}
    for key, value in raw.items():
        if len(out) >= MAX_ATTRIBUTES:
            break
        keep, cleaned = _clean_value(value)
        if keep:
            out[str(key)[:MAX_KEY_LENGTH]] = cleaned
    return out


def record_events(events: Iterable[Mapping[str, Any]], *, user: str, session: str | None = None) -> int:
    """Append *events* to this month's file and return how many were written.

    Each event carries a ``group`` and optional ``attributes`` and
    ``client_ts``, as validated by
    :class:`vtsearch.schemas.metrics.MetricsEventSchema`. Records nothing (and
    returns ``0``) when metrics are disabled or the write fails.
    """
    if not is_enabled():
        return 0
    now = datetime.now(timezone.utc)
    stamp = now.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    lines: list[str] = []
    for event in events:
        line: dict[str, Any] = {
            "ts": stamp,
            "user": user,
            "session": session,
            "group": event["group"],
            "attributes": clean_attributes(event.get("attributes") or {}),
        }
        if event.get("client_ts") is not None:
            line["client_ts"] = event["client_ts"]
        lines.append(json.dumps(line, ensure_ascii=False, separators=(",", ":")))
    if not lines:
        return 0
    data = ("\n".join(lines) + "\n").encode("utf-8")
    path = metrics_file(now)
    try:
        with _write_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("ab", buffering=0) as fh:
                fh.write(data)
    except OSError as exc:
        logger.warning("Could not record %d metrics event(s) to %s: %s", len(lines), path, exc)
        return 0
    return len(lines)
