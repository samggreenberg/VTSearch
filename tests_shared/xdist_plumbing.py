"""Keep one dead xdist worker from wedging the run, and say why it died (issue #4925).

On the GRID's Xeon Silver 4114 nodes one test ran past the 300 s per-test
timeout, and ``./run-tests.sh`` then sat at 99% until its 30-minute cap, with
nothing in the log to name the test.  Two faults, both in the plugins:

* **pytest-xdist's loadgroup recovery hangs.**  When a worker dies,
  ``LoadScopeScheduling.remove_node`` (pytest-xdist 3.8.0, which
  ``--dist loadgroup`` inherits) puts back on the queue *every* unit the worker
  was ever given, finished ones too.  The worker started in its place is handed
  a finished unit, which is ``runtests([])``.  It never reports a test, so it is
  never given another unit, and the controller waits on it for good.  Fixed
  upstream (pytest-dev/pytest-xdist#1323, pytest-dev/pytest-xdist#1327) but in
  no release yet.  :func:`make_crash_safe_scheduler` requeues only what is
  unfinished.  It also counts the test that killed the worker as run (xdist
  reports it failed, by name) where upstream would run it again: a test past
  the timeout once does it again on the next worker, 300 s each time, until
  xdist's restart budget (four per worker) runs out.  ``--dist load`` already
  drops a crashed test.  And it takes back a unit whose send finds the worker
  already gone, which upstream lets escape as an ``INTERNALERROR`` that ends
  the run, and it offers work only to workers whose collection has arrived,
  where upstream also offers it to a replacement still collecting, another
  ``INTERNALERROR`` once two workers have died.
* **pytest-timeout's report goes to /dev/null.**  The ``thread`` method writes
  the stacks through the terminal writer, then calls ``os._exit(1)``.  In an
  xdist worker that writer is the worker's stdout, which execnet points at
  ``/dev/null``.  :func:`set_worker_timeout_timer` arms the same timer and
  writes the report to the worker's stderr, which reaches the run's log.

The conftests call both from two-line hooks (``pytest_xdist_make_scheduler``,
``pytest_timeout_set_timer``); each returns ``None`` where it does not apply,
which hands the hook back to the plugin's own implementation.

A third fault left ``--dist loadgroup`` grouping nothing at all (issues #5356,
#5352).  pytest-xdist 3.8.0 groups a test by appending ``@<group>`` to its
nodeid in the worker, by writing ``item._nodeid``.  pytest 9.2 computes
``nodeid`` from a structured ``item._id`` instead, so that write lands in an
attribute nothing reads, and every ``xdist_group`` test became its own unit.
Tests that share files under the session data dir then ran on several workers at
once, and one test's cleanup removed the files another was reading.  The
conftests' ``pytest_collection_modifyitems`` calls
:func:`restore_xdist_group_suffix`, which writes the suffix where pytest reads it.
"""

from __future__ import annotations

import os
import sys
import threading
import traceback

from xdist.scheduler import LoadGroupScheduling


class CrashSafeLoadGroupScheduling(LoadGroupScheduling):
    """``--dist loadgroup`` whose recovery from a dead worker cannot hang."""

    def remove_node(self, node):
        workload = self.assigned_work.pop(node)
        unfinished = [(scope, nodeid) for scope, unit in workload.items() for nodeid, done in unit.items() if not done]
        if not unfinished:
            return None
        # The worker died in the first test it had not finished.
        scope, crashitem = unfinished[0]
        workload[scope][crashitem] = True
        for scope, unit in workload.items():
            if not all(unit.values()):
                self.workqueue[scope] = unit
        for other in self.assigned_work:
            self._reschedule(other)
        return crashitem

    def _assign_work_unit(self, node):
        try:
            super()._assign_work_unit(node)
        except OSError:
            # The worker died just after reporting a test.  Its report is
            # handled before its death, and until execnet sees the pipe close
            # the worker does not read as shutting down, so the report hands it
            # another unit and the send finds the pipe closed.  It never got the
            # unit: put it back at the head of the queue for remove_node, which
            # runs next, to hand on.  (xdist's own shutdown() swallows the same
            # error.)
            scope, unit = self.assigned_work[node].popitem()
            self.workqueue[scope] = unit
            self.workqueue.move_to_end(scope, last=False)

    def schedule(self):
        # A worker runs a test only once it holds the next one (or a shutdown),
        # and the replacement for a dead worker joins holding nothing, so after
        # the first distribution every node is offered work twice, as upstream
        # does (pytest-dev/pytest-xdist#1327).  Only nodes whose collection has
        # arrived: two replacements can be in flight at once, and the one still
        # collecting has no collection to index the work by.  Its own
        # collection's arrival calls this again.  (Upstream offers to every
        # node, and raises ``KeyError`` there.)
        if self.collection is None:
            super().schedule()
            return
        for _ in range(2):
            for node in self.nodes:
                if node in self.registered_collections:
                    self._reschedule(node)


def restore_xdist_group_suffix(config, items) -> None:
    """Give each ``xdist_group`` test the ``@<group>`` nodeid suffix ``--dist loadgroup`` schedules by.

    In an xdist worker, pytest-xdist's own ``pytest_collection_modifyitems`` runs
    before the conftests' and appends the suffix itself on pytest < 9.2, so an
    item that already carries it is left alone.  That keeps this a no-op there,
    and once a pytest-xdist release writes the suffix where pytest 9.2 reads it.
    The name is built the way pytest-xdist 3.8.0 builds it: every group the test
    is marked with, sorted and joined with ``_``.  Call it before anything that
    matches items by nodeid (the pass cache records the suffixed form).
    """
    if not getattr(config.option, "loadgroup", False):
        return
    for item in items:
        names = {
            str(m.args[0] if m.args else m.kwargs.get("name", "default")) for m in item.iter_markers("xdist_group")
        }
        if not names:
            continue
        suffix = "@" + "_".join(sorted(names))
        if item.nodeid.endswith(suffix):
            continue
        # pytest 9.2's ``nodeid`` is ``str(self._id)``; ``parse`` keeps the exact string.
        item._id = type(item._id).parse(item.nodeid + suffix)


def make_crash_safe_scheduler(config, log):
    """The scheduler for ``--dist loadgroup``; ``None`` (xdist's own) for any other mode."""
    if config.getvalue("dist") != "loadgroup":
        return None
    return CrashSafeLoadGroupScheduling(config, log)


def set_worker_timeout_timer(item, settings):
    """pytest-timeout's ``thread`` timer, reporting on stderr, in an xdist worker; else ``None``."""
    if settings.method != "thread" or not hasattr(item.config, "workerinput"):
        return None
    timer = threading.Timer(settings.timeout, _report_timeout_and_exit, (item, settings))
    timer.name = f"{__name__} {item.nodeid}"

    def cancel():
        timer.cancel()
        timer.join()

    # pytest-timeout's own pytest_timeout_cancel_timer calls this.
    item.cancel_timeout = cancel
    timer.start()
    return True


def _report_timeout_and_exit(item, settings):
    """What pytest-timeout's ``timeout_timer`` does, written where a worker's output survives."""
    try:
        out = err = ""
        capman = item.config.pluginmanager.getplugin("capturemanager")
        if capman is not None:
            capman.suspend_global_capture(in_=True)
            out, err = capman.read_global_capture()
        worker = item.config.workerinput.get("workerid", "?")
        parts = [
            "",
            f"{'+' * 30} Timeout {'+' * 30}",
            f"{item.nodeid} ran past {settings.timeout:g}s on xdist worker {worker}, which now exits.",
        ]
        for title, text in (("Captured stdout", out), ("Captured stderr", err)):
            if text:
                parts += [f"{'~' * 20} {title} {'~' * 20}", text.rstrip()]
        names = {t.ident: t.name for t in threading.enumerate()}
        for ident, frame in sys._current_frames().items():
            if ident != threading.get_ident():
                parts += [f"{'~' * 20} Stack of {names.get(ident, '<unknown>')} ({ident}) {'~' * 20}"]
                parts += ["".join(traceback.format_stack(frame)).rstrip()]
        parts += ["+" * 69, ""]
        # Capture is suspended, so fd 2 is the worker's own stderr again.
        os.write(2, "\n".join(parts).encode(errors="replace"))
    except Exception:
        traceback.print_exc()
    finally:
        os._exit(1)
