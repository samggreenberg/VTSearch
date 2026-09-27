"""Tests for the stall diagnostics in :mod:`vtscore.concurrency.stalls` (#3853).

Three instruments, each pinned on the property that makes it useful on a live
deployment: the phase clock and lock timer are silent below their threshold
and name what they measured above it; the GC callback logs a pause at
WARNING; the watchdog turns a late heartbeat into a report that says which
thread burned the wall clock, and re-arms its dump on every beat.
"""

from __future__ import annotations

import gc
import logging
import re
import threading
import time

import pytest

from vtscore.concurrency import stalls
from vtscore.concurrency.stalls import (
    PhaseClock,
    StallWatchdog,
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
        assert frozen == gc.get_freeze_count() > 0
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
        assert "/tmp/dump.log" in msg
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
        assert "no thread dump armed" in msg

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
