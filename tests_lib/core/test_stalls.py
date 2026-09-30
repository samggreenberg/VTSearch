"""Tests for the stall diagnostics in :mod:`vtscore.concurrency.stalls` (#3853).

Three instruments, each pinned on the property that makes it useful on a live
deployment: the phase clock and lock timer are silent below their threshold
and name what they measured above it; the GC callback logs a pause at
WARNING; the watchdog turns a late heartbeat into a report that says which
thread burned the wall clock, with every thread's stack taken as it woke - and,
since #4345, never arms ``faulthandler``'s live dump unless asked to, because
that dump can segfault the process it watches.
"""

from __future__ import annotations

import gc
import io
import logging
import os
import re
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from vtscore.concurrency import stalls
from vtscore.concurrency.stalls import (
    PhaseClock,
    StallWatchdog,
    capture_thread_stacks,
    format_thread_stacks,
    freeze_gc_after_preload,
    gc_pause_ms_total,
    gc_pause_stats,
    gc_warn_threshold_ms,
    install_gc_pause_logging,
    slow_phase_threshold_ms,
    timed_lock,
    uninstall_gc_pause_logging,
)

LOGGER = "vtscore.concurrency.stalls"


def _messages(caplog: pytest.LogCaptureFixture, needle: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if needle in r.getMessage()]


def _field_ms(msg: str, field: str) -> float:
    """``cpu=41ms`` / ``total 270ms`` out of a log line, as a number."""
    m = re.search(rf"{re.escape(field)}=?\s?(\d+)ms", msg)
    assert m is not None, f"no {field!r} figure in {msg!r}"
    return float(m.group(1))


# ---------------------------------------------------------------------------
# Threshold parsing
# ---------------------------------------------------------------------------


class TestThresholds:
    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv(stalls.SLOW_PHASE_MS_ENV, raising=False)
        assert slow_phase_threshold_ms() == stalls._DEFAULT_SLOW_PHASE_MS

    def test_env_wins(self, monkeypatch):
        monkeypatch.setenv(stalls.SLOW_PHASE_MS_ENV, "12.5")
        assert slow_phase_threshold_ms() == 12.5

    def test_unparseable_falls_back(self, monkeypatch):
        """Bad configuration must not fault the vote path."""
        monkeypatch.setenv(stalls.SLOW_PHASE_MS_ENV, "soon")
        assert slow_phase_threshold_ms() == stalls._DEFAULT_SLOW_PHASE_MS


class TestGcThresholdTracksPhaseThreshold:
    """A GC bar left above the phase bar makes every phase it contaminates
    look slow with nothing anywhere saying a collection was the reason
    (#3853, 2026-09-15: three ``label_sync`` outliers at a 150ms phase bar,
    each coinciding with a ~206ms collection a 200ms GC bar just missed)."""

    def test_default_config_is_unchanged(self, monkeypatch):
        monkeypatch.delenv(stalls.GC_WARN_MS_ENV, raising=False)
        monkeypatch.delenv(stalls.SLOW_PHASE_MS_ENV, raising=False)
        assert gc_warn_threshold_ms() == stalls._DEFAULT_GC_WARN_MS

    def test_lowering_the_phase_bar_lowers_the_gc_bar(self, monkeypatch):
        monkeypatch.delenv(stalls.GC_WARN_MS_ENV, raising=False)
        monkeypatch.setenv(stalls.SLOW_PHASE_MS_ENV, "150")
        assert gc_warn_threshold_ms() == 75.0

    def test_gc_bar_never_sits_above_the_phase_bar(self, monkeypatch):
        """The property that matters, over the whole range."""
        monkeypatch.delenv(stalls.GC_WARN_MS_ENV, raising=False)
        for phase_ms in (10, 50, 150, 300, 500, 5000):
            monkeypatch.setenv(stalls.SLOW_PHASE_MS_ENV, str(phase_ms))
            assert gc_warn_threshold_ms() <= phase_ms

    def test_explicit_env_still_wins(self, monkeypatch):
        monkeypatch.setenv(stalls.GC_WARN_MS_ENV, "42")
        monkeypatch.setenv(stalls.SLOW_PHASE_MS_ENV, "150")
        assert gc_warn_threshold_ms() == 42.0


# ---------------------------------------------------------------------------
# PhaseClock
# ---------------------------------------------------------------------------


class TestPhaseClock:
    def test_logs_breakdown_and_fields_when_slow(self, caplog, monkeypatch):
        monkeypatch.setenv(stalls.SLOW_PHASE_MS_ENV, "0")
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            with PhaseClock("learned_sort", labels=42) as clock:
                clock.mark("train")
                clock.mark("score")
                clock.fields["corpus"] = 7
        records = _messages(caplog, "slow phase: learned_sort")
        assert len(records) == 1
        msg = records[0].getMessage()
        assert records[0].levelno == logging.WARNING
        assert "train=" in msg and "score=" in msg
        assert "labels=42" in msg and "corpus=7" in msg

    def test_silent_below_threshold(self, caplog, monkeypatch):
        monkeypatch.setenv(stalls.SLOW_PHASE_MS_ENV, "600000")
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            with PhaseClock("quiet") as clock:
                clock.mark("a")
        assert not _messages(caplog, "slow phase")

    def test_finish_fields_and_idempotence(self, caplog, monkeypatch):
        monkeypatch.setenv(stalls.SLOW_PHASE_MS_ENV, "0")
        clock = PhaseClock("x")
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            total = clock.finish(outcome="done")
            clock.finish(outcome="again")  # a second finish logs nothing
        assert total >= 0
        records = _messages(caplog, "slow phase: x")
        assert len(records) == 1
        assert "outcome=done" in records[0].getMessage()

    def test_unmarked_clock_says_so(self, caplog, monkeypatch):
        monkeypatch.setenv(stalls.SLOW_PHASE_MS_ENV, "0")
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            PhaseClock("bare").finish()
        assert "no phases marked" in _messages(caplog, "slow phase: bare")[0].getMessage()

    def test_reports_cpu_alongside_wall(self, caplog, monkeypatch):
        """A phase that burns CPU must be distinguishable from one that waits.

        This is what the wall-only clock could not do in #3853: six of the
        eight slow votes in the captured trace were slow *alone*, which says
        they released the GIL - but nothing recorded whether they had done
        work or blocked, so the candidates could not be separated.

        The two phases are compared against *each other* rather than each
        against its own wall clock. ``cpu >= total * 0.5`` reads as the same
        claim and is not: wall time absorbs whatever the scheduler took away,
        so under ``-n auto`` on a loaded box a spin loop's share of its own
        wall clock falls below half and the assertion fails on machine load
        rather than on anything the clock got wrong. A sleeping phase burns
        no CPU at any load, which makes the comparison load-independent.
        """
        monkeypatch.setenv(stalls.SLOW_PHASE_MS_ENV, "0")
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            with PhaseClock("busy"):
                n = 0
                for i in range(200_000):
                    n += i
            with PhaseClock("waiting"):
                time.sleep(0.05)
        busy = _messages(caplog, "slow phase: busy")[0].getMessage()
        waiting = _messages(caplog, "slow phase: waiting")[0].getMessage()
        assert _field_ms(busy, "cpu") > 0, busy
        assert _field_ms(busy, "cpu") > _field_ms(waiting, "cpu"), f"{busy!r} vs {waiting!r}"

    def test_reports_the_gc_pause_that_landed_inside_it(self, caplog, monkeypatch):
        """A collection holds the GIL, so it is charged to whatever phase was
        running - which is why an un-lowered GC bar silently inflates phases.
        The phase line now carries the figure itself."""
        monkeypatch.setenv(stalls.SLOW_PHASE_MS_ENV, "0")
        monkeypatch.setenv(stalls.GC_WARN_MS_ENV, "600000")  # bar is irrelevant here
        install_gc_pause_logging()
        try:
            with caplog.at_level(logging.WARNING, logger=LOGGER):
                with PhaseClock("collecting"):
                    gc.collect()
        finally:
            uninstall_gc_pause_logging()
        msg = _messages(caplog, "slow phase: collecting")[0].getMessage()
        assert _field_ms(msg, "gc") >= 0
        assert "cpu=" in msg

    def test_gc_column_is_zero_without_the_callback(self, caplog, monkeypatch):
        """The counter is only fed by the installed callback; with none
        installed the clock must report 0, not crash or guess."""
        monkeypatch.setenv(stalls.SLOW_PHASE_MS_ENV, "0")
        uninstall_gc_pause_logging()
        before = gc_pause_ms_total()
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            with PhaseClock("uninstrumented"):
                gc.collect()
        assert gc_pause_ms_total() == before
        assert "gc=0ms" in _messages(caplog, "slow phase: uninstrumented")[0].getMessage()


# ---------------------------------------------------------------------------
# timed_lock
# ---------------------------------------------------------------------------


class TestTimedLock:
    @pytest.mark.parametrize("lock_factory", [threading.Lock, threading.RLock])
    def test_acquires_and_releases(self, lock_factory, caplog, monkeypatch):
        monkeypatch.setenv(stalls.SLOW_PHASE_MS_ENV, "600000")
        lock = lock_factory()
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            with timed_lock(lock, "test") as waited:
                assert waited >= 0.0
                # Held: a non-blocking acquire from another thread must fail.
                other: list[bool] = []
                t = threading.Thread(target=lambda: other.append(lock.acquire(blocking=False)))
                t.start()
                t.join(5)
                assert other == [False]
        # Released on exit.
        assert lock.acquire(blocking=False)
        lock.release()
        assert not _messages(caplog, "lock wait")

    def test_logs_contended_wait(self, caplog, monkeypatch):
        """A wait over the threshold is logged with the lock's name."""
        monkeypatch.setenv(stalls.SLOW_PHASE_MS_ENV, "0")
        lock = threading.Lock()
        holding = threading.Event()
        release = threading.Event()

        def holder():
            with lock:
                holding.set()
                release.wait(5)

        t = threading.Thread(target=holder)
        t.start()
        assert holding.wait(5)
        # Let the holder go once we are (about to be) blocked on the lock.
        threading.Timer(0.05, release.set).start()
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            with timed_lock(lock, "_progress_lock/test"):
                pass
        t.join(5)
        records = _messages(caplog, "lock wait: _progress_lock/test")
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING


# ---------------------------------------------------------------------------
# GC pause logging
# ---------------------------------------------------------------------------


class TestGcPauseLogging:
    @pytest.fixture(autouse=True)
    def _installed(self):
        install_gc_pause_logging()
        install_gc_pause_logging()  # idempotent
        assert gc.callbacks.count(stalls._gc_callback) == 1
        yield
        uninstall_gc_pause_logging()
        assert stalls._gc_callback not in gc.callbacks

    def test_full_collection_logged_at_zero_threshold(self, caplog, monkeypatch):
        monkeypatch.setenv(stalls.GC_WARN_MS_ENV, "0")
        before = gc_pause_stats()
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            gc.collect()
        after = gc_pause_stats()
        assert after["gen2_pauses"] == before["gen2_pauses"] + 1
        records = _messages(caplog, "gc pause: generation 2")
        assert records and records[-1].levelno == logging.WARNING
        assert "collected=" in records[-1].getMessage()

    def test_silent_above_threshold(self, caplog, monkeypatch):
        monkeypatch.setenv(stalls.GC_WARN_MS_ENV, "600000")
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            gc.collect()
        assert not _messages(caplog, "gc pause")


# ---------------------------------------------------------------------------
# gc.freeze after preload (#3870)
# ---------------------------------------------------------------------------


class TestFreezeGcAfterPreload:
    """``gc.freeze()`` once the models are loaded, so full collections stop
    traversing the imported ML libraries' graph (#3870: ~300ms gen-2 pauses
    every ~2 minutes, each freezing whatever request was in flight)."""

    @pytest.fixture(autouse=True)
    def _unfreeze(self):
        """Put the freeze back the way the test found it.

        Both conftests freeze the session heap (``freeze_startup_heap``), so
        the state to restore is usually *frozen*: a bare ``gc.unfreeze()`` here
        handed the whole import-time heap back to the collector for the rest of
        the worker's session, and every later production ``gc.collect()`` paid
        to rescan it (issue #4152).  Whatever this test itself froze is undone
        first, so nothing it created outlives it in the permanent generation.
        """
        was_frozen = gc.get_freeze_count() > 0
        yield
        gc.unfreeze()
        if was_frozen:
            gc.collect()
            gc.freeze()

    def test_freezes_and_reports_what_it_froze(self, monkeypatch):
        monkeypatch.delenv(stalls.GC_FREEZE_ENV, raising=False)
        gc.unfreeze()
        assert gc.get_freeze_count() == 0
        result = freeze_gc_after_preload()
        assert result is not None
        frozen, ms = result
        # Not ``==``: a frozen object that dies leaves the permanent
        # generation, and any thread still alive in this worker (an embedder
        # warm-up, a registry preload left by an earlier test) can free one
        # between the function's read and this one. Nothing can *enter* it
        # without another freeze, so the count can only have fallen.
        assert 0 < gc.get_freeze_count() <= frozen
        assert ms >= 0.0

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF"])
    def test_env_can_turn_it_off(self, monkeypatch, value):
        """A deployment must be able to back this out without a code change."""
        monkeypatch.setenv(stalls.GC_FREEZE_ENV, value)
        gc.unfreeze()
        assert freeze_gc_after_preload() is None
        assert gc.get_freeze_count() == 0

    def test_objects_created_after_the_freeze_stay_collectable(self, monkeypatch):
        """The safety property the call site depends on: datasets and
        detectors load *after* this runs, so unloading one must still free
        its cycles. Only what was alive at freeze time is permanent."""
        monkeypatch.delenv(stalls.GC_FREEZE_ENV, raising=False)
        gc.unfreeze()
        freeze_gc_after_preload()

        class Node:
            peer: "Node | None" = None

        a, b = Node(), Node()
        a.peer, b.peer = b, a  # a cycle only the collector can break
        import weakref

        ref = weakref.ref(a)
        del a, b
        gc.collect()
        assert ref() is None, "post-freeze garbage was not collected"


# ---------------------------------------------------------------------------
# StallWatchdog
# ---------------------------------------------------------------------------


def _beat_base(wd: StallWatchdog) -> float:
    base = wd._last_beat
    assert base is not None
    return base


def _sample(cpu_by_tid: dict[int, float], proc_cpu: float, majflt: int = 0, gen2: int = 0) -> dict:
    return {
        "wall": 0.0,
        "threads": {tid: (f"thread-{tid}", 0x1000 + tid, cpu) for tid, cpu in cpu_by_tid.items()},
        "proc_cpu": proc_cpu,
        "majflt": majflt,
        "rss_kb": 2048,
        "gc": {"gen2_pauses": gen2, "max_ms": 5.0},
        "cgroup": {"current": 1_000_000_000, "max": 4_000_000_000, "limit_hits": 0},
    }


class TestThreadStacks:
    """The stacks a late beat writes, taken holding the GIL (#4345)."""

    def test_captures_a_parked_thread_where_it_is(self):
        entered = threading.Event()
        release = threading.Event()

        def parked_in_a_named_function():
            entered.set()
            release.wait(10)

        t = threading.Thread(target=parked_in_a_named_function, name="parker")
        t.start()
        try:
            assert entered.wait(10)
            stacks = capture_thread_stacks()
        finally:
            release.set()
            t.join(10)
        assert t.ident is not None
        functions = [func for _file, _line, func in stacks[t.ident]]
        # Innermost first: the Event's wait, then the frame that called it.
        assert "parked_in_a_named_function" in functions
        assert functions.index("wait") < functions.index("parked_in_a_named_function")
        assert all(isinstance(line, int) for _file, line, _func in stacks[t.ident])
        # The thread taking the snapshot is in it too; the watchdog drops itself.
        assert threading.get_ident() in stacks

    def test_values_are_fixed_when_taken(self):
        """A live frame's ``f_lineno`` moves as its thread runs on; the snapshot
        must report where the thread was, so it is reduced to plain values."""
        stacks = capture_thread_stacks()
        here = stacks[threading.get_ident()][0]
        assert isinstance(here, tuple)
        assert here[2] == "capture_thread_stacks"

    def test_format_is_faulthandlers_with_the_holder_first(self):
        stacks: dict[int, list[stalls.FrameLine]] = {
            0x1001: [("/app/idle.py", 3, "wait")],
            0x1002: [("/app/embed.py", 42, "encode"), ("/app/importer.py", 7, "run")],
            0x1003: [("/app/unknown.py", None, "mystery")],
        }
        text = format_thread_stacks(
            stacks,
            header="Stall snapshot (heartbeat late by 5000ms):",
            threads={0x1001: ("idle", 11, 0.0), 0x1002: ("importer", 12, 4900.0)},
        )
        lines = text.splitlines()
        assert lines[0] == "Stall snapshot (heartbeat late by 5000ms):"
        assert lines[1] == (
            'Thread 0x0000000000001002 ["importer", tid 12, 4900ms cpu across the gap] (most recent call first):'
        )
        assert lines[2] == '  File "/app/embed.py", line 42 in encode'
        assert lines[3] == '  File "/app/importer.py", line 7 in run'
        headers = [ln for ln in lines if ln.startswith("Thread 0x")]
        assert [h[:25] for h in headers] == [
            "Thread 0x0000000000001002",
            "Thread 0x0000000000001001",
            "Thread 0x0000000000001003",
        ]
        assert 'Thread 0x0000000000001003 ["?"] (most recent call first):' in lines
        assert '  File "/app/unknown.py", line ? in mystery' in lines

    def test_format_puts_an_exited_thread_after_the_measured_ones(self):
        """A thread gone before its CPU was read ran as the stall ended - the
        holder finishing its work is the usual way - so it is not filed with
        the idle threads."""
        stacks: dict[int, list[stalls.FrameLine]] = {
            1: [("/a.py", 1, "idle")],
            2: [("/b.py", 2, "busy")],
            3: [("/c.py", 3, "finished")],
        }
        text = format_thread_stacks(
            stacks,
            header="h",
            threads={1: ("idle", 11, None), 2: ("busy", 12, 30.0), 3: ("importer", 13, None)},
            exited=frozenset({3}),
        )
        headers = [ln for ln in text.splitlines() if ln.startswith("Thread 0x")]
        assert [h.split('"')[1] for h in headers] == ["busy", "importer", "idle"]
        assert '["importer", tid 13, exited before its cpu was read]' in headers[1]

    def test_format_marks_a_cut_stack(self):
        deep: list[stalls.FrameLine] = [("/app/r.py", i, "recurse") for i in range(stalls._MAX_FRAMES + 1)]
        lines = format_thread_stacks({1: deep}, header="h").splitlines()
        frames = [ln for ln in lines if ln.startswith("  File ")]
        assert len(frames) == stalls._MAX_FRAMES
        assert "  ..." in lines


class TestStallWatchdog:
    def test_report_names_the_thread_that_burned_the_gap(self, caplog):
        """A GIL hold: one thread's CPU accounts for the wall clock."""
        samples = iter(
            [
                _sample({1: 1.0, 2: 0.5}, proc_cpu=1.5),
                _sample({1: 1.0, 2: 5.4}, proc_cpu=6.4, gen2=1),
            ]
        )
        armed: list[float] = []
        wd = StallWatchdog(
            1000,
            arm=armed.append,
            sampler=lambda: next(samples),
            dump_path="/tmp/dump.log",
            logger=logging.getLogger(LOGGER),
        )
        wd._prime()
        assert armed == [1.0]  # armed at construction-time priming
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            # Beat scheduled at interval (0.25s) but ran 5.0s later.
            lag_ms = wd.beat(now=_beat_base(wd) + wd.interval_s + 5.0)
        assert lag_ms == pytest.approx(5000.0, abs=1.0)
        assert wd.stalls == 1 and wd.worst_lag_ms == pytest.approx(5000.0, abs=1.0)
        assert armed == [1.0, 1.0]  # re-armed after the beat
        msg = _messages(caplog, "stall: heartbeat late by")[0].getMessage()
        assert "thread-2 (tid 2, ident 0x1002) 4900ms" in msg
        assert "process cpu 4900ms of 5250ms wall (0.93)" in msg
        assert "gc gen2 pauses +1" in msg
        assert "live thread dump armed at 1000ms into the gap -> /tmp/dump.log" in msg
        assert "cgroup mem 1.0GB of 4.0GB" in msg

    def test_report_says_when_nothing_ran(self, caplog):
        """A stalled process: no thread consumed CPU across the gap."""
        samples = iter([_sample({1: 1.0}, proc_cpu=1.0, majflt=10), _sample({1: 1.0}, proc_cpu=1.0, majflt=900)])
        wd = StallWatchdog(1000, sampler=lambda: next(samples), logger=logging.getLogger(LOGGER))
        wd._prime()
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert wd.beat(now=_beat_base(wd) + wd.interval_s + 2.0) is not None
        msg = _messages(caplog, "stall: heartbeat late by")[0].getMessage()
        assert "no thread consumed cpu across the gap" in msg
        assert "majflt +890" in msg
        assert "no thread stacks" in msg

    def test_on_time_beat_is_silent(self, caplog):
        samples = iter([_sample({1: 1.0}, 1.0), _sample({1: 1.1}, 1.1)])
        wd = StallWatchdog(1000, sampler=lambda: next(samples), logger=logging.getLogger(LOGGER))
        wd._prime()
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert wd.beat(now=_beat_base(wd) + wd.interval_s + 0.01) is None
        assert wd.stalls == 0
        assert not _messages(caplog, "stall:")

    def test_thread_lifecycle_and_real_sampler(self):
        """The thread starts, beats with the real /proc sampler, and stops."""
        beats: list[float] = []
        wd = StallWatchdog(200, arm=beats.append)
        wd.start()
        try:
            assert wd.running
            assert wd.interval_s == pytest.approx(0.05)
            deadline = threading.Event()
            deadline.wait(0.3)
            assert len(beats) >= 2, "the heartbeat never beat"
        finally:
            wd.stop()
        assert not wd.running
        assert wd.stalls == 0

    def test_default_sampler_sees_this_thread(self):
        sample = stalls.default_sampler()
        assert sample["proc_cpu"] is None or sample["proc_cpu"] >= 0
        me = threading.get_native_id()
        if sample["threads"]:  # /proc present
            name, ident, cpu = sample["threads"][me]
            assert name == threading.current_thread().name
            assert ident == threading.get_ident()
            assert cpu >= 0

    def test_late_beat_takes_the_stacks_before_it_samples(self):
        """The sampler's /proc reads release the GIL; a snapshot taken after
        them would show the holder wherever it ran on to."""
        calls: list[str] = []
        base_sample = _sample({1: 1.0}, proc_cpu=1.0)

        def sampler() -> dict:
            calls.append("sample")
            return base_sample

        def snapshot() -> dict:
            calls.append("snapshot")
            return {}

        wd = StallWatchdog(
            1000, snapshot=snapshot, sampler=sampler, dump_file=io.StringIO(), logger=logging.getLogger(LOGGER)
        )
        wd._prime()
        calls.clear()
        wd.beat(now=_beat_base(wd) + wd.interval_s + 0.01)
        assert calls == ["sample"], "an on-time beat takes no stacks"
        calls.clear()
        assert wd.beat(now=_beat_base(wd) + wd.interval_s + 2.0) is not None
        assert calls == ["snapshot", "sample"]

    def test_stacks_are_written_before_the_report_line(self):
        """The dump sits immediately above the stall it belongs to, which is
        where ``analyze_app_log.py`` looks for it, holder first, without the
        watchdog's own thread."""
        samples = iter([_sample({1: 1.0, 2: 0.5}, proc_cpu=1.5), _sample({1: 1.0, 2: 5.4}, proc_cpu=6.4)])
        me = threading.get_ident()
        stacks: dict[int, list[stalls.FrameLine]] = {
            0x1001: [("/app/idle.py", 3, "wait")],
            0x1002: [("/app/embed.py", 42, "encode")],
            me: [("/app/watchdog.py", 1, "beat")],
        }
        dump = io.StringIO()
        seen_at_log_time: list[str] = []

        class _Recorder(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                if "stall: heartbeat late by" in record.getMessage():
                    seen_at_log_time.append(dump.getvalue())

        logger = logging.getLogger(f"{LOGGER}.ordering")
        handler = _Recorder()
        logger.addHandler(handler)
        try:
            wd = StallWatchdog(
                1000,
                snapshot=lambda: dict(stacks),
                dump_file=dump,
                sampler=lambda: next(samples),
                dump_path="/logs/app.log",
                logger=logger,
            )
            wd._prime()
            wd.beat(now=_beat_base(wd) + wd.interval_s + 5.0)
        finally:
            logger.removeHandler(handler)
        assert len(seen_at_log_time) == 1
        written = seen_at_log_time[0]
        assert written, "the stacks were not written before the report line"
        assert written.startswith("Stall snapshot (heartbeat late by 5000ms;")
        headers = [ln for ln in written.splitlines() if ln.startswith("Thread 0x")]
        assert headers[0].startswith('Thread 0x0000000000001002 ["thread-2", tid 2, 4900ms cpu across the gap]')
        assert len(headers) == 2, "the watchdog's own thread is not the story"
        assert f"0x{me:016x}" not in written
        assert wd.last_report is not None
        assert "thread stacks at wake -> /logs/app.log" in wd.last_report
        assert "live thread dump" not in wd.last_report

    def test_names_threads_as_the_snapshot_is_taken(self):
        """A thread alive for the snapshot but gone from the sample after it
        keeps its name and is marked exited, not left as ``"?"``."""
        entered = threading.Event()
        release = threading.Event()

        def importer_body():
            entered.set()
            release.wait(10)

        t = threading.Thread(target=importer_body, name="fixture-import")
        t.start()
        try:
            assert entered.wait(10)
            ident = t.ident
            assert ident is not None
            samples = iter([_sample({1: 1.0}, proc_cpu=1.0), _sample({1: 1.0}, proc_cpu=2.0)])
            dump = io.StringIO()
            wd = StallWatchdog(
                1000,
                snapshot=lambda: {ident: [("/app/importer.py", 9, "encode")]},
                dump_file=dump,
                sampler=lambda: next(samples),
                logger=logging.getLogger(LOGGER),
            )
            wd._prime()
            wd.beat(now=_beat_base(wd) + wd.interval_s + 2.0)
        finally:
            release.set()
            t.join(10)
        assert f'["fixture-import", tid {t.native_id}, exited before its cpu was read]' in dump.getvalue()

    def test_a_failed_snapshot_still_reports(self, caplog):
        samples = iter([_sample({1: 1.0}, proc_cpu=1.0), _sample({1: 3.0}, proc_cpu=3.0)])

        def broken() -> dict:
            raise RuntimeError("no frames today")

        wd = StallWatchdog(1000, snapshot=broken, sampler=lambda: next(samples), logger=logging.getLogger(LOGGER))
        wd._prime()
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert wd.beat(now=_beat_base(wd) + wd.interval_s + 2.0) is not None
        assert _messages(caplog, "taking the thread stacks failed")
        assert "no thread stacks" in _messages(caplog, "stall: heartbeat late by")[0].getMessage()


class TestWiring:
    @pytest.fixture(autouse=True)
    def _clean(self):
        stalls.stop_stall_diagnostics()
        uninstall_gc_pause_logging()
        yield
        stalls.stop_stall_diagnostics()
        uninstall_gc_pause_logging()

    def test_zero_disables_watchdog_but_installs_gc_logging(self, monkeypatch):
        monkeypatch.setenv(stalls.WATCHDOG_MS_ENV, "0")
        assert stalls.start_stall_diagnostics_from_env() is None
        assert stalls.active_watchdog() is None
        assert stalls._gc_callback in gc.callbacks

    def test_starts_once_and_dumps_to_the_log_file(self, monkeypatch, tmp_path):
        dump = tmp_path / "logs" / "app.log"
        monkeypatch.setenv(stalls.WATCHDOG_MS_ENV, "5000")
        monkeypatch.delenv(stalls.DUMP_FILE_ENV, raising=False)
        monkeypatch.setenv("VTSEARCH_LOG_FILE", str(dump))
        wd = stalls.start_stall_diagnostics_from_env()
        assert wd is not None and wd.running
        assert wd.dump_path == str(dump)
        assert dump.parent.is_dir()
        assert stalls.start_stall_diagnostics_from_env() is wd
        stalls.stop_stall_diagnostics()
        assert not wd.running

    def test_dump_file_env_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VTSEARCH_LOG_FILE", str(tmp_path / "app.log"))
        monkeypatch.setenv(stalls.DUMP_FILE_ENV, str(tmp_path / "stalls.log"))
        assert stalls.resolve_dump_path() == str(tmp_path / "stalls.log")

    def test_default_never_arms_the_live_dump(self, monkeypatch):
        """faulthandler's live dump segfaulted the app during a CPU import
        (#4345), so the default watchdog must never arm it: it takes the
        stacks itself, holding the GIL, when a beat finds it woke late."""
        armed: list[float] = []
        monkeypatch.setattr(stalls.faulthandler, "dump_traceback_later", lambda t, **kw: armed.append(t))
        monkeypatch.setattr(stalls.faulthandler, "cancel_dump_traceback_later", lambda: None)
        monkeypatch.setenv(stalls.WATCHDOG_MS_ENV, "5000")
        monkeypatch.delenv(stalls.LIVE_DUMP_ENV, raising=False)
        monkeypatch.delenv(stalls.DUMP_FILE_ENV, raising=False)
        monkeypatch.delenv("VTSEARCH_LOG_FILE", raising=False)
        wd = stalls.start_stall_diagnostics_from_env()
        assert wd is not None
        assert wd._arm is None
        assert wd._snapshot is capture_thread_stacks
        stalls.stop_stall_diagnostics()
        assert armed == []

    def test_live_dump_is_opt_in(self, monkeypatch):
        armed: list[float] = []
        monkeypatch.setattr(stalls.faulthandler, "dump_traceback_later", lambda t, **kw: armed.append(t))
        monkeypatch.setattr(stalls.faulthandler, "cancel_dump_traceback_later", lambda: None)
        monkeypatch.setenv(stalls.WATCHDOG_MS_ENV, "5000")
        monkeypatch.setenv(stalls.LIVE_DUMP_ENV, "1")
        monkeypatch.delenv(stalls.DUMP_FILE_ENV, raising=False)
        monkeypatch.delenv("VTSEARCH_LOG_FILE", raising=False)
        wd = stalls.start_stall_diagnostics_from_env()
        assert wd is not None
        assert wd._snapshot is capture_thread_stacks, "the stacks at wake are taken either way"
        stalls.stop_stall_diagnostics()
        assert armed and armed[0] == 5.0

    @pytest.mark.parametrize(
        ("value", "enabled"), [("1", True), ("on", True), ("TRUE", True), ("0", False), ("", False)]
    )
    def test_live_dump_env_parsing(self, monkeypatch, value, enabled):
        monkeypatch.setenv(stalls.LIVE_DUMP_ENV, value)
        assert stalls.live_dump_enabled() is enabled


# ---------------------------------------------------------------------------
# The stall path under frame churn (#4345)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]

_CHURN_SCRIPT = textwrap.dedent(
    """
    import logging, os, sys, threading
    from vtscore.concurrency.stalls import StallWatchdog, capture_thread_stacks

    stop = threading.Event()

    def recurse(n):
        return [0] * 64 if n == 0 else recurse(n - 1)

    def churn():
        # Deep enough to cross CPython's frame-stack chunks, whose release is
        # what faulthandler's GIL-free walk trips over.
        while not stop.is_set():
            for depth in (5, 300, 900, 50):
                recurse(depth)

    # Hand the GIL over every 10us, so the churn runs in the middle of every
    # snapshot's walk rather than only between beats.
    sys.setswitchinterval(1e-5)
    workers = [threading.Thread(target=churn, daemon=True) for _ in range(4)]
    for w in workers:
        w.start()
    quiet = logging.getLogger("churn")
    quiet.propagate = False
    quiet.addHandler(logging.NullHandler())
    sample = {"threads": {}, "proc_cpu": None, "majflt": None, "rss_kb": None, "gc": {}, "cgroup": {}}
    with open(os.devnull, "w") as dump:
        wd = StallWatchdog(1000, snapshot=capture_thread_stacks, dump_file=dump, sampler=lambda: sample, logger=quiet)
        wd._prime()
        for _ in range(200):
            wd.beat(now=wd._last_beat + wd.interval_s + 2.0)
    stop.set()
    print("stalls", wd.stalls)
    """
)


class TestSnapshotUnderFrameChurn:
    def test_stall_path_survives_threads_pushing_and_popping_frames(self):
        """The regression for #4345, run in a child so a crash fails the test
        instead of killing the worker.

        Under this churn, ``faulthandler.dump_traceback_later`` segfaults
        CPython 3.11 within about 0.1s, a few dozen dumps.  The watchdog's own
        stall path runs 200 times here.
        """
        env = dict(
            os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(_REPO_ROOT), os.environ.get("PYTHONPATH")]))
        )
        proc = subprocess.run(  # noqa: S603 - a fixed script run by this interpreter
            [sys.executable, "-c", _CHURN_SCRIPT],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            cwd=_REPO_ROOT,
        )
        assert proc.returncode == 0, f"exit {proc.returncode}\nstdout: {proc.stdout}\nstderr: {proc.stderr[-2000:]}"
        assert "stalls 200" in proc.stdout


# ---------------------------------------------------------------------------
# The dump as the log analyzer reads it
# ---------------------------------------------------------------------------

_ANALYZER = _REPO_ROOT / "scripts" / "experiments" / "stall_3853" / "analyze_app_log.py"


def _load_analyzer():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_stall_analyze_app_log", _ANALYZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stall_json(lag_ms: int, ts: str) -> str:
    import json

    return json.dumps({"ts": ts, "level": "WARNING", "msg": f"stall: heartbeat late by {lag_ms}ms; top threads: x"})


class TestAnalyzerReadsTheDump:
    """``analyze_app_log.py`` attaches a dump to the stall it precedes; the
    watchdog's layout must stay one it parses (#4345 changed who writes it)."""

    def _snapshot_text(self) -> str:
        return format_thread_stacks(
            {0x1002: [("/app/embed.py", 42, "encode")], 0x1001: [("/usr/lib/threading.py", 320, "wait")]},
            header="Stall snapshot (heartbeat late by 1500ms; stacks taken as the watchdog woke):",
            threads={0x1002: ("importer", 12, 1400.0), 0x1001: ("idle", 11, 0.0)},
        )

    def test_snapshot_is_parsed_and_attached(self, tmp_path, monkeypatch, capsys):
        analyzer = _load_analyzer()
        log_path = tmp_path / "app.log"
        live = 'Timeout (0:00:01)!\nThread 0x0000000000001002 (most recent call first):\n  File "/app/embed.py", line 41 in encode\n\n'
        log_path.write_text(
            live
            + self._snapshot_text()
            + _stall_json(1500, "2026-09-30T11:00:02Z")
            + "\n"
            + _stall_json(1200, "2026-09-30T11:05:00Z")
            + "\n"
        )
        events, dumps = analyzer.load(str(log_path))
        assert [d["live"] for d in dumps] == [True, False]
        snapshot = dumps[1]
        assert snapshot["threads"][0]["header"].startswith('Thread 0x0000000000001002 ["importer", tid 12, 1400ms')
        assert snapshot["threads"][0]["frames"] == ['File "/app/embed.py", line 42 in encode']
        assert [e["kind"] for e in events] == ["stall", "stall"]

        monkeypatch.setattr(sys, "argv", ["analyze_app_log.py", str(log_path)])
        analyzer.main()
        out = capsys.readouterr().out
        first, second = out.split("STALL 1200ms")
        # The first stall shows both: the live dump mid-stall, then the stacks at wake.
        assert "live faulthandler dump at log line 1" in first
        assert "thread stacks at wake at log line" in first
        assert "line 42 in encode" in first
        # The second stall has no dump of its own, and must not borrow the first's.
        assert "no thread dump found before this stall line" in second
