"""Stall diagnostics: a GIL-aware heartbeat watchdog, GC-pause logging, and
slow-phase / lock-wait timers for the request paths a labeling session hits.

Written for issue #3853, where the app froze for 5-20 s a few times per
labeling session and the per-request timer (``vtsearch.hooks``) could show
*that* every in-flight request finished at the same instant but not *why*: a
timer on each request cannot tell a thread that held the GIL from a lock
convoy from a process the kernel stopped scheduling.  Three instruments here
separate those, and all three run in-process so a stall on a live deployment
leaves a trace at the default log level:

* :class:`StallWatchdog` - a heartbeat thread that sleeps ``interval`` seconds
  at a time and measures how late it wakes.  A thread that cannot run for
  longer than the threshold has been kept off the interpreter, and the report
  says by what: it samples every thread's CPU time from ``/proc`` on each
  beat, so across the gap it can name the thread that burned the wall clock
  (a GIL hold - ``faulthandler`` has by then written that thread's frames,
  see below), or show that *no* thread ran (the process itself was stalled:
  memory pressure, a descheduled cgroup, a page fault storm - the report
  carries RSS, major faults and the cgroup memory counters for that case).

  Each beat also re-arms :func:`faulthandler.dump_traceback_later`.  That
  timer runs on a C thread that needs no GIL, so when the heartbeat misses it
  dumps every thread's Python frames *during* the stall, to a file the
  watchdog names.  This is the one decisive datum: the frame the GIL holder
  was in when everything else froze.

* :func:`install_gc_pause_logging` - ``gc.callbacks`` timing, logged at
  WARNING above ``VTSEARCH_GC_WARN_MS``.  A full collection holds the GIL for
  its whole duration and shows up in a ``faulthandler`` dump only as an
  arbitrary allocation site, so it is named here explicitly.  Unset, that
  threshold *tracks* ``VTSEARCH_SLOW_PHASE_MS`` (half of it, capped at the
  200 ms default): a collection shorter than the phase bar is invisible but
  still lands inside whatever phase was running, so a bar left behind while
  the phase bar was lowered silently inflates every phase it contaminates.
  That happened on 2026-09-15 (issue #3853), where three ``label_sync``
  outliers at a 150 ms phase bar each coincided with a ~206 ms collection
  that a 200 ms GC bar had only just failed to report.

* :class:`PhaseClock` and :func:`timed_lock` - breakdown timers for the
  paths a vote runs through (the labelset rewrite, the learned-sort retrain,
  the labeling-status replay) and wait timers on the locks they share.  Both
  log at WARNING above ``VTSEARCH_SLOW_PHASE_MS`` and are silent below it, so
  they cost a few counter reads on the fast path.

  A :class:`PhaseClock` reports **CPU and GC alongside wall time**, because
  wall time alone cannot say what a slow phase was doing.  ``cpu ~ wall``
  means the phase did the work; ``cpu << wall`` means it blocked on I/O or was
  descheduled (an 8-thread torch process in an 8-CPU cgroup on a shared node
  is descheduled routinely); ``gc`` names the part that was a collection
  freezing every thread.  ``gc`` is **not** additive with ``cpu``: a
  collection that ran on this thread burned this thread's CPU too, so a line
  reading ``total 1137ms cpu=1133ms gc=139ms`` means "spent on the CPU, and
  139ms of that was the collector", not 1272ms of anything.

* :func:`freeze_gc_after_preload` - ``gc.freeze()`` once the models are
  loaded, so full collections stop traversing the imported ML libraries'
  object graph (issue #3870).

Everything is opt-out by environment (``VTSEARCH_STALL_WATCHDOG_MS=0``
disables the watchdog) and nothing here imports Flask or ``vtsearch``; the
app wires it in from ``initialize_server``.
"""

from __future__ import annotations

import faulthandler
import gc
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import IO, Any, Callable, Iterator, Optional

log = logging.getLogger(__name__)

#: Env var: heartbeat-miss threshold in ms; ``0`` disables the watchdog.
WATCHDOG_MS_ENV = "VTSEARCH_STALL_WATCHDOG_MS"
_DEFAULT_WATCHDOG_MS = 1000.0

#: Env var: file the ``faulthandler`` thread dump is written to when the
#: heartbeat misses.  Defaults to ``VTSEARCH_LOG_FILE`` when set, else stderr.
DUMP_FILE_ENV = "VTSEARCH_STALL_DUMP_FILE"

#: Env var: GC pauses at least this long are logged at WARNING.
GC_WARN_MS_ENV = "VTSEARCH_GC_WARN_MS"
_DEFAULT_GC_WARN_MS = 200.0

#: Env var: a phase or a lock wait at least this long is logged at WARNING.
SLOW_PHASE_MS_ENV = "VTSEARCH_SLOW_PHASE_MS"
_DEFAULT_SLOW_PHASE_MS = 500.0

#: Env var: set falsey to skip the post-preload :func:`gc.freeze`.
GC_FREEZE_ENV = "VTSEARCH_GC_FREEZE"

_FALSEY = {"0", "false", "no", "off"}


def _env_ms(name: str, default: float) -> float:
    """Read a millisecond threshold from the environment.

    Read per call (not captured at import) so a restart with a new value takes
    effect and a test can lower the bar without reloading the module.  An
    unparseable value falls back to *default*: bad configuration must not be
    able to fault the paths these timers sit on.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Slow phases and lock waits
# ---------------------------------------------------------------------------


def slow_phase_threshold_ms() -> float:
    """Milliseconds above which :class:`PhaseClock` / :func:`timed_lock` log."""
    return _env_ms(SLOW_PHASE_MS_ENV, _DEFAULT_SLOW_PHASE_MS)


def watchdog_threshold_ms() -> float:
    """Heartbeat-miss threshold; ``0`` means the watchdog is off."""
    return _env_ms(WATCHDOG_MS_ENV, _DEFAULT_WATCHDOG_MS)


def thread_cpu_ms() -> float:
    """This thread's CPU time in ms, or ``0.0`` where the clock is missing.

    ``time.thread_time`` is documented as available only on some platforms.
    A missing clock must degrade to "no CPU figure", never to an exception on
    the vote path, so every caller reads a difference of two of these and a
    constant 0.0 simply reports ``cpu=0ms``.
    """
    clock = getattr(time, "thread_time", None)
    if clock is None:  # pragma: no cover - POSIX and Windows both have it
        return 0.0
    try:
        return clock() * 1000.0
    except (OSError, RuntimeError):  # pragma: no cover - defensive
        return 0.0


def _fmt_fields(fields: dict[str, Any]) -> str:
    return " ".join(f"{k}={v}" for k, v in fields.items())


class PhaseClock:
    """Time a multi-phase operation and log its breakdown when it was slow.

    ::

        with PhaseClock("learned_sort", labels=n) as clock:
            train()
            clock.mark("train")
            score()
            clock.mark("score")

    On exit (or an explicit :meth:`finish`) the total is compared against
    :func:`slow_phase_threshold_ms`; at or above it one WARNING line names
    the operation, its total, every phase's share, and the keyword fields
    given at construction or to :meth:`finish`.  Below it nothing is logged.
    Time between the last :meth:`mark` and :meth:`finish` is reported as
    ``rest``.
    """

    __slots__ = (
        "name",
        "fields",
        "_t0",
        "_last",
        "_phases",
        "_finished",
        "_logger",
        "_cpu0",
        "_gc0",
    )

    def __init__(self, name: str, *, logger: logging.Logger | None = None, **fields: Any) -> None:
        self.name = name
        self.fields: dict[str, Any] = dict(fields)
        self._t0 = time.perf_counter()
        self._last = self._t0
        self._phases: list[tuple[str, float]] = []
        self._finished = False
        self._logger = logger or log
        self._cpu0 = thread_cpu_ms()
        # Defined further down the module; resolved at call time.
        self._gc0 = gc_pause_ms_total()

    def mark(self, phase: str) -> float:
        """Close the phase that ran since the previous mark; returns its ms."""
        now = time.perf_counter()
        ms = (now - self._last) * 1000.0
        self._phases.append((phase, ms))
        self._last = now
        return ms

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def finish(self, **fields: Any) -> float:
        """Log the breakdown if the total crossed the threshold; returns total ms."""
        if self._finished:
            return self.elapsed_ms()
        self._finished = True
        now = time.perf_counter()
        total_ms = (now - self._t0) * 1000.0
        rest_ms = (now - self._last) * 1000.0
        self.fields.update(fields)
        if total_ms >= slow_phase_threshold_ms():
            cpu_ms = max(0.0, thread_cpu_ms() - self._cpu0)
            gc_ms = max(0.0, gc_pause_ms_total() - self._gc0)
            phases = list(self._phases)
            if self._phases and rest_ms >= 1.0:
                phases.append(("rest", rest_ms))
            breakdown = ", ".join(f"{p}={ms:.0f}ms" for p, ms in phases) or "no phases marked"
            self._logger.warning(
                "slow phase: %s total %.0fms cpu=%.0fms gc=%.0fms (%s) %s",
                self.name,
                total_ms,
                cpu_ms,
                gc_ms,
                breakdown,
                _fmt_fields(self.fields),
            )
        return total_ms

    def __enter__(self) -> "PhaseClock":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.finish()


@contextmanager
def timed_lock(lock: Any, name: str, *, logger: logging.Logger | None = None) -> Iterator[float]:
    """``with lock:`` that logs at WARNING when acquiring it waited too long.

    Yields the wait in milliseconds.  Works for :class:`threading.Lock` and
    :class:`threading.RLock` alike (it only calls ``acquire`` / ``release``).
    A long wait names the lock and nothing else: which thread held it is what
    the watchdog's thread dump and the phase clocks around the holder's work
    are for.
    """
    t0 = time.perf_counter()
    lock.acquire()
    waited_ms = (time.perf_counter() - t0) * 1000.0
    try:
        if waited_ms >= slow_phase_threshold_ms():
            (logger or log).warning("lock wait: %s waited %.0fms", name, waited_ms)
        yield waited_ms
    finally:
        lock.release()


# ---------------------------------------------------------------------------
# GC pauses
# ---------------------------------------------------------------------------


class _GcStats:
    __slots__ = ("start", "pauses", "gen2_pauses", "total_ms", "max_ms", "installed")

    def __init__(self) -> None:
        self.start: float | None = None
        self.pauses = 0
        self.gen2_pauses = 0
        self.total_ms = 0.0
        self.max_ms = 0.0
        self.installed = False


_gc_stats = _GcStats()


def gc_warn_threshold_ms() -> float:
    """Milliseconds above which a collection is logged.

    Unset, the bar **tracks the phase bar** at half of it, capped at the
    200 ms default.  A collection holds the GIL, so it lands inside whatever
    phase or request was running and is charged to that phase; if the GC bar
    sits above the phase bar, every such phase is reported as slow with no
    line anywhere saying a collection was the reason.  Lowering
    ``VTSEARCH_SLOW_PHASE_MS`` for a diagnostic session therefore raises GC
    resolution by itself, and cannot leave the two out of step (#3853).
    An explicit ``VTSEARCH_GC_WARN_MS`` still wins outright.
    """
    raw = os.environ.get(GC_WARN_MS_ENV)
    if raw is not None:
        return _env_ms(GC_WARN_MS_ENV, _DEFAULT_GC_WARN_MS)
    return min(_DEFAULT_GC_WARN_MS, slow_phase_threshold_ms() / 2.0)


def _gc_callback(phase: str, info: dict[str, Any]) -> None:
    # Runs on whichever thread triggered the collection, GIL held; a full
    # collection is the pause we want to name, so keep this cheap.
    if phase == "start":
        _gc_stats.start = time.perf_counter()
        return
    if _gc_stats.start is None:
        return
    ms = (time.perf_counter() - _gc_stats.start) * 1000.0
    _gc_stats.start = None
    _gc_stats.pauses += 1
    generation = int(info.get("generation", -1))
    if generation == 2:
        _gc_stats.gen2_pauses += 1
    _gc_stats.total_ms += ms
    if ms > _gc_stats.max_ms:
        _gc_stats.max_ms = ms
    if ms >= gc_warn_threshold_ms():
        log.warning(
            "gc pause: generation %d took %.0fms (collected=%s, uncollectable=%s)",
            generation,
            ms,
            info.get("collected"),
            info.get("uncollectable"),
        )


def install_gc_pause_logging() -> None:
    """Register the ``gc.callbacks`` timer once.  Idempotent."""
    if _gc_stats.installed:
        return
    gc.callbacks.append(_gc_callback)
    _gc_stats.installed = True


def uninstall_gc_pause_logging() -> None:
    """Remove the timer (tests)."""
    if not _gc_stats.installed:
        return
    try:
        gc.callbacks.remove(_gc_callback)
    except ValueError:
        pass
    _gc_stats.installed = False


def gc_pause_ms_total() -> float:
    """Total collection time since install, in ms; monotonic.

    Snapshotted at both ends of a window (a :class:`PhaseClock`, a request)
    to say how much of it was a collection.  Collections are process-wide and
    hold the GIL, so a pause that fires on another thread still froze this
    one, and charging it to this window is exactly right.
    """
    return _gc_stats.total_ms


def gc_pause_stats() -> dict[str, Any]:
    """Counters since install: ``pauses``, ``gen2_pauses``, ``total_ms``, ``max_ms``."""
    return {
        "pauses": _gc_stats.pauses,
        "gen2_pauses": _gc_stats.gen2_pauses,
        "total_ms": round(_gc_stats.total_ms, 1),
        "max_ms": round(_gc_stats.max_ms, 1),
    }


def gc_freeze_enabled() -> bool:
    """Whether :func:`freeze_gc_after_preload` will do anything."""
    return os.environ.get(GC_FREEZE_ENV, "1").strip().lower() not in _FALSEY


def freeze_gc_after_preload() -> tuple[int, float] | None:
    """Collect once, then :func:`gc.freeze`; returns ``(objects, ms)``.

    Issue #3870.  A labeling session logged a **gen-2 pause of ~300 ms every
    ~2 minutes** (35 of them over 2,400 votes, flat with label count), each
    freezing whatever was in flight - a vote POST to 333 ms, a sort to
    351 ms.  The pause is dominated by the long-lived graph the process
    starts with: ``transformers``, ``torch``, ``sklearn``, ``cuml``/``numba``
    contribute millions of tracked containers that every full collection
    traverses and never frees.  ``gc.freeze()`` moves everything alive at
    this point into the permanent generation, which full collections skip.
    Measured on the GRID with an otherwise identical run: ten pauses of
    290-360 ms became **zero**, and vote POST max fell 377 ms -> 121 ms.

    Call this **after the model preload and before serving**, which is what
    makes it safe: the frozen set is the imported libraries and the loaded
    embedders, all of which live for the process anyway.  Datasets and
    detectors load lazily *afterwards*, so they stay collectable and
    unloading one still frees its cycles.  Nothing is frozen that would
    otherwise have been freed.

    Returns ``None`` when ``VTSEARCH_GC_FREEZE`` is falsey, so a deployment
    can turn it off without a code change.
    """
    if not gc_freeze_enabled():
        log.info("gc freeze: skipped (%s=%s)", GC_FREEZE_ENV, os.environ.get(GC_FREEZE_ENV))
        return None
    t0 = time.perf_counter()
    gc.collect()
    gc.freeze()
    frozen = gc.get_freeze_count()
    ms = (time.perf_counter() - t0) * 1000.0
    log.info("gc freeze: froze %d objects into the permanent generation in %.0fms", frozen, ms)
    return frozen, ms


# ---------------------------------------------------------------------------
# /proc and cgroup readers (best effort; every failure reads as "unknown")
# ---------------------------------------------------------------------------

try:
    _CLK_TCK = float(os.sysconf("SC_CLK_TCK"))
except (AttributeError, ValueError, OSError):  # pragma: no cover - non-POSIX
    _CLK_TCK = 100.0
try:
    _PAGE_KB = float(os.sysconf("SC_PAGE_SIZE")) / 1024.0
except (AttributeError, ValueError, OSError):  # pragma: no cover - non-POSIX
    _PAGE_KB = 4.0


def _read_stat(path: str) -> tuple[float, int, int] | None:
    """``(cpu_seconds, majflt, rss_kb)`` from a ``/proc/.../stat`` line."""
    try:
        with open(path, encoding="ascii") as fh:
            line = fh.read()
    except OSError:
        return None
    # The comm field is parenthesised and may hold spaces; split after it.
    tail = line.rsplit(")", 1)[-1].split()
    # tail[0] is ``state``; utime/stime are stat fields 14/15 -> tail 11/12,
    # majflt is field 12 -> tail 9, rss (pages) is field 24 -> tail 21.
    try:
        cpu = (int(tail[11]) + int(tail[12])) / _CLK_TCK
        majflt = int(tail[9])
        rss_kb = int(int(tail[21]) * _PAGE_KB)
    except (IndexError, ValueError):
        return None
    return cpu, majflt, rss_kb


def _read_int_file(path: str) -> int | None:
    try:
        with open(path, encoding="ascii") as fh:
            raw = fh.read().strip()
    except OSError:
        return None
    if raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _read_cgroup_memory() -> dict[str, Any]:
    """Current / limit bytes and the limit-hit counter, cgroup v2 then v1."""
    out: dict[str, Any] = {}
    cur = _read_int_file("/sys/fs/cgroup/memory.current")
    if cur is not None:
        out["current"] = cur
        out["max"] = _read_int_file("/sys/fs/cgroup/memory.max")
        events: dict[str, int] = {}
        try:
            with open("/sys/fs/cgroup/memory.events", encoding="ascii") as fh:
                for line in fh:
                    k, _, v = line.partition(" ")
                    if v.strip().isdigit():
                        events[k] = int(v)
        except OSError:
            pass
        out["limit_hits"] = events.get("max", 0) + events.get("high", 0)
        return out
    cur = _read_int_file("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if cur is not None:
        out["current"] = cur
        limit = _read_int_file("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        # v1 reports "no limit" as a huge number rather than "max".
        out["max"] = limit if limit is not None and limit < (1 << 60) else None
        out["limit_hits"] = _read_int_file("/sys/fs/cgroup/memory/memory.failcnt") or 0
    return out


def default_sampler() -> dict[str, Any]:
    """One heartbeat sample: per-thread and process CPU, RSS, faults, GC, cgroup."""
    names: dict[int, tuple[str, int]] = {}
    for t in threading.enumerate():
        native = getattr(t, "native_id", None)
        if native is not None:
            names[native] = (t.name, t.ident or 0)
    threads: dict[int, tuple[str, int, float]] = {}
    try:
        tids = os.listdir("/proc/self/task")
    except OSError:
        tids = []
    for tid_s in tids:
        stat = _read_stat(f"/proc/self/task/{tid_s}/stat")
        if stat is None:
            continue
        tid = int(tid_s)
        name, ident = names.get(tid, ("<native>", 0))
        threads[tid] = (name, ident, stat[0])
    proc = _read_stat("/proc/self/stat")
    return {
        "wall": time.monotonic(),
        "threads": threads,
        "proc_cpu": proc[0] if proc else None,
        "majflt": proc[1] if proc else None,
        "rss_kb": proc[2] if proc else None,
        "gc": gc_pause_stats(),
        "cgroup": _read_cgroup_memory(),
    }


# ---------------------------------------------------------------------------
# The watchdog
# ---------------------------------------------------------------------------


def _faulthandler_armer(dump_file: IO[Any]) -> Callable[[float], None]:
    def arm(timeout_s: float) -> None:
        # Re-arming replaces the previous timer; ``repeat=False`` so a stall
        # produces one dump, not one per timeout for as long as it lasts.
        faulthandler.dump_traceback_later(timeout_s, repeat=False, file=dump_file, exit=False)

    return arm


class StallWatchdog:
    """Heartbeat thread that reports when it could not run for ``threshold``.

    Parameters
    ----------
    threshold_ms:
        A beat that wakes at least this late is a stall.
    arm:
        Called with the threshold in seconds on every beat; the production
        armer re-arms ``faulthandler.dump_traceback_later``.  ``None`` arms
        nothing (tests, or a caller that only wants the CPU accounting).
    sampler:
        Returns the per-beat sample (see :func:`default_sampler`).
    dump_path:
        Where the armed dump lands, for the report line only.
    """

    def __init__(
        self,
        threshold_ms: float,
        *,
        arm: Callable[[float], None] | None = None,
        sampler: Callable[[], dict[str, Any]] = default_sampler,
        dump_path: str = "<stderr>",
        logger: logging.Logger | None = None,
    ) -> None:
        self.threshold_s = max(threshold_ms, 1.0) / 1000.0
        # Beat often enough that a stall is measured to within a quarter
        # threshold, but never faster than 20 Hz (the sample reads /proc).
        self.interval_s = max(self.threshold_s / 4.0, 0.05)
        self._arm = arm
        self._sampler = sampler
        self.dump_path = dump_path
        self._logger = logger or log
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_beat: float | None = None
        self._last_sample: dict[str, Any] | None = None
        self.stalls = 0
        self.worst_lag_ms = 0.0
        self.last_report: str | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "StallWatchdog":
        if self._thread is not None:
            return self
        self._prime()
        self._thread = threading.Thread(target=self._run, name="stall-watchdog", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        if self._arm is not None:
            faulthandler.cancel_dump_traceback_later()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- the beat ------------------------------------------------------------

    def _prime(self) -> None:
        self._last_sample = self._sampler()
        self._last_beat = time.monotonic()
        if self._arm is not None:
            self._arm(self.threshold_s)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.beat()
            except Exception:  # noqa: BLE001 - the watchdog must outlive its own bugs
                self._logger.exception("stall watchdog beat failed")

    def beat(self, now: float | None = None) -> float | None:
        """One heartbeat.  Returns the lag in ms when it counted as a stall.

        Split out from the loop so a test can drive it with an explicit clock:
        the lag is ``now - last_beat - interval``, i.e. how much later than
        scheduled this beat ran.
        """
        now = time.monotonic() if now is None else now
        assert self._last_beat is not None
        lag_s = now - self._last_beat - self.interval_s
        sample = self._sampler()
        stalled: float | None = None
        if lag_s >= self.threshold_s:
            stalled = lag_s * 1000.0
            self.stalls += 1
            self.worst_lag_ms = max(self.worst_lag_ms, stalled)
            self._report(self._last_sample, sample, lag_s, now - self._last_beat)
        self._last_beat = now
        self._last_sample = sample
        if self._arm is not None:
            self._arm(self.threshold_s)
        return stalled

    # -- the report ----------------------------------------------------------

    def _report(
        self,
        before: dict[str, Any] | None,
        after: dict[str, Any],
        lag_s: float,
        gap_s: float,
    ) -> None:
        parts = [f"stall: heartbeat late by {lag_s * 1000.0:.0f}ms"]
        if before is not None:
            wall_ms = max(gap_s, 1e-6) * 1000.0
            proc_before, proc_after = before.get("proc_cpu"), after.get("proc_cpu")
            if proc_before is not None and proc_after is not None:
                cpu_ms = (proc_after - proc_before) * 1000.0
                parts.append(f"process cpu {cpu_ms:.0f}ms of {wall_ms:.0f}ms wall ({cpu_ms / wall_ms:.2f})")
            parts.append(self._top_threads(before, after))
            mf_before, mf_after = before.get("majflt"), after.get("majflt")
            if mf_before is not None and mf_after is not None:
                parts.append(f"majflt +{mf_after - mf_before}")
            gc_before, gc_after = before.get("gc") or {}, after.get("gc") or {}
            gen2 = int(gc_after.get("gen2_pauses", 0)) - int(gc_before.get("gen2_pauses", 0))
            parts.append(f"gc gen2 pauses +{gen2} (max {gc_after.get('max_ms', 0)}ms)")
        rss_kb = after.get("rss_kb")
        if rss_kb is not None:
            parts.append(f"rss {rss_kb / 1024.0:.0f}MB")
        parts.append(self._cgroup_part(before, after))
        parts.append(
            f"thread dump armed at {self.threshold_s * 1000.0:.0f}ms into the gap -> {self.dump_path}"
            if self._arm is not None
            else "no thread dump armed"
        )
        message = "; ".join(p for p in parts if p)
        self.last_report = message
        self._logger.warning("%s", message)

    @staticmethod
    def _top_threads(before: dict[str, Any], after: dict[str, Any], n: int = 3) -> str:
        prev: dict[int, tuple[str, int, float]] = before.get("threads") or {}
        cur: dict[int, tuple[str, int, float]] = after.get("threads") or {}
        deltas: list[tuple[float, int, str, int]] = []
        for tid, (name, ident, cpu) in cur.items():
            base = prev.get(tid)
            used = cpu - base[2] if base is not None else cpu
            if used > 0:
                deltas.append((used, tid, name, ident))
        if not deltas:
            return "no thread consumed cpu across the gap"
        deltas.sort(reverse=True)
        items = ", ".join(
            f"{name} (tid {tid}, ident 0x{ident:x}) {used * 1000.0:.0f}ms" for used, tid, name, ident in deltas[:n]
        )
        return f"top threads: {items}"

    @staticmethod
    def _cgroup_part(before: dict[str, Any] | None, after: dict[str, Any]) -> str:
        cg = after.get("cgroup") or {}
        if "current" not in cg:
            return ""
        cur_gb = cg["current"] / 1e9
        limit = cg.get("max")
        limit_s = f"{limit / 1e9:.1f}GB" if limit else "no limit"
        hits = int(cg.get("limit_hits") or 0)
        prev_hits = int(((before or {}).get("cgroup") or {}).get("limit_hits") or 0)
        return f"cgroup mem {cur_gb:.1f}GB of {limit_s}, limit hits +{hits - prev_hits}"


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

_active: StallWatchdog | None = None
_dump_file: IO[Any] | None = None


def resolve_dump_path() -> str | None:
    """The file the thread dump goes to: ``VTSEARCH_STALL_DUMP_FILE``, else
    ``VTSEARCH_LOG_FILE``, else ``None`` for stderr."""
    return os.environ.get(DUMP_FILE_ENV) or os.environ.get("VTSEARCH_LOG_FILE") or None


def start_stall_diagnostics_from_env() -> Optional[StallWatchdog]:
    """Install GC-pause logging and start the watchdog per the environment.

    Idempotent: a second call returns the running watchdog.  Returns ``None``
    when ``VTSEARCH_STALL_WATCHDOG_MS`` is ``0`` (GC logging is still
    installed - it is free until a pause crosses its own threshold).
    """
    global _active, _dump_file
    install_gc_pause_logging()
    if _active is not None and _active.running:
        return _active
    threshold_ms = watchdog_threshold_ms()
    if threshold_ms <= 0:
        return None
    path = resolve_dump_path()
    dump_file: IO[Any]
    if path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            dump_file = open(path, "a", encoding="utf-8")  # noqa: SIM115 - faulthandler needs it open for the process lifetime
        except OSError:
            log.exception("stall watchdog: cannot open %s for the thread dump; using stderr", path)
            dump_file, path = sys.stderr, None
    else:
        dump_file = sys.stderr
    _dump_file = dump_file
    _active = StallWatchdog(
        threshold_ms,
        arm=_faulthandler_armer(dump_file),
        dump_path=path or "<stderr>",
    ).start()
    log.info(
        "stall watchdog armed: threshold %.0fms, thread dump -> %s",
        threshold_ms,
        _active.dump_path,
    )
    return _active


def stop_stall_diagnostics() -> None:
    """Stop the watchdog started by :func:`start_stall_diagnostics_from_env` (tests)."""
    global _active
    if _active is not None:
        _active.stop()
        _active = None


def active_watchdog() -> Optional[StallWatchdog]:
    return _active
