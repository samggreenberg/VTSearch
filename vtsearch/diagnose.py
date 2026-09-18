"""One switch for a diagnostic session, and a log line saying what the bars were.

Two of the inconclusive sessions in issue #3853 failed for configuration
reasons rather than instrumentation ones, and both are fixed here.

**The bars were wrong, quietly.**  The 2026-09-15 ``car`` session ran at the
shipped ``VTSEARCH_SLOW_REQUEST_MS=1000``, so a vote taking 600-900 ms was
invisible to its log; the ``umbrella`` session got a lower bar only through
*uncommitted local edits* to two source files.  Since #3873 there is a third
way to get it wrong: setting ``VTSEARCH_GC_WARN_MS`` by hand bypasses the
coupling that keeps the GC bar under the phase bar.  :data:`PRESET` is the
whole set, applied together, by one switch.

**The log could not say which bars produced it.**  "Zero slow requests" reads
as *nothing was slow* and as *the bar was a second* equally well, and nothing
in a log distinguished them -- so every absence in this issue had to be
re-litigated from memory of how the process was launched.
:func:`log_effective_settings` writes one line at startup, at WARNING so it
survives a stock deployment's log level, and :func:`effective_settings` is
what it reports.

The preset is applied with ``setdefault`` semantics: an explicit environment
variable always wins, so ``VTSEARCH_DIAGNOSE=1 VTSEARCH_SLOW_PHASE_MS=50`` is
the preset with one bar overridden rather than a conflict.

This lives in the app tier because the set spans both: the log level and the
request bar are ``vtsearch`` names, the phase bar is a ``vtscore`` one.
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)

#: Env var: truthy turns on the whole diagnostic preset below.
DIAGNOSE_ENV = "VTSEARCH_DIAGNOSE"

_TRUTHY = {"1", "true", "yes", "on"}

#: What ``VTSEARCH_DIAGNOSE`` sets, unless the variable is already set.
#:
#: ``VTSEARCH_GC_WARN_MS`` is deliberately **absent**.  Left unset it tracks
#: the phase bar (``vtscore.concurrency.stalls.gc_warn_threshold_ms``), so
#: pinning it here would be the same mistake the coupling exists to prevent:
#: a collection under the GC bar is invisible while still inflating whatever
#: phase it lands in.  Lowering the phase bar to 150 ms lowers the GC bar to
#: 75 ms by itself.
PRESET: dict[str, str] = {
    "VTSEARCH_LOG_LEVEL": "INFO",  # per-request `request trace` lines
    "VTSEARCH_SLOW_REQUEST_MS": "400",
    "VTSEARCH_SLOW_PHASE_MS": "150",
}


def diagnose_enabled() -> bool:
    """Whether ``VTSEARCH_DIAGNOSE`` asks for the preset."""
    return os.environ.get(DIAGNOSE_ENV, "").strip().lower() in _TRUTHY


def apply_diagnostic_preset() -> dict[str, str]:
    """Apply :data:`PRESET` to the environment; returns what it actually set.

    Call this **before** :func:`vtsearch.logging_config.setup_logging`, which
    reads ``VTSEARCH_LOG_LEVEL`` once. Values already in the environment are
    left alone, so the return value is the subset this call is responsible
    for -- which is what the startup line reports, so a reader can tell a
    preset value from one somebody chose.
    """
    if not diagnose_enabled():
        return {}
    applied: dict[str, str] = {}
    for name, value in PRESET.items():
        if os.environ.get(name) is None:
            os.environ[name] = value
            applied[name] = value
    return applied


def effective_settings() -> dict[str, Any]:
    """Every threshold that decides what a stall leaves in the log.

    Resolved through the same accessors the instruments themselves call, not
    re-read from the environment here: a summary that parsed the variables
    separately could report a bar the code does not use, which is worse than
    no summary at all.
    """
    from vtscore.concurrency.stalls import (
        gc_freeze_enabled,
        gc_warn_threshold_ms,
        slow_phase_threshold_ms,
        watchdog_threshold_ms,
    )
    from vtscore.detectors.store import detector_write_mode

    from vtsearch.hooks import slow_request_threshold_ms

    return {
        "diagnose": diagnose_enabled(),
        "log_level": logging.getLevelName(logging.getLogger().level),
        "slow_request_ms": slow_request_threshold_ms(),
        "slow_phase_ms": slow_phase_threshold_ms(),
        "gc_warn_ms": gc_warn_threshold_ms(),
        "watchdog_ms": watchdog_threshold_ms(),
        "gc_freeze": gc_freeze_enabled(),
        "detector_write": detector_write_mode(),
        "log_file": os.environ.get("VTSEARCH_LOG_FILE") or "<none>",
    }


def log_effective_settings() -> dict[str, Any]:
    """Record the settings above in the log; returns them.

    **WARNING, not INFO, on purpose**, for the same reason the slow-request
    line is: ``VTSEARCH_LOG_LEVEL`` defaults to WARNING, and a log that omits
    its own thresholds makes every absence in it ambiguous. One line per
    process start.
    """
    settings = effective_settings()
    log.warning(
        "diagnostics config: diagnose=%s log_level=%s slow_request=%.0fms slow_phase=%.0fms "
        "gc_warn=%.0fms watchdog=%.0fms gc_freeze=%s detector_write=%s log_file=%s",
        settings["diagnose"],
        settings["log_level"],
        settings["slow_request_ms"],
        settings["slow_phase_ms"],
        settings["gc_warn_ms"],
        settings["watchdog_ms"],
        settings["gc_freeze"],
        settings["detector_write"],
        settings["log_file"],
    )
    return settings
