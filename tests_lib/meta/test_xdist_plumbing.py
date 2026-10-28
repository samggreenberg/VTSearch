"""A dead xdist worker fails its test by name instead of wedging the run (issue #4925).

``tests_shared/xdist_plumbing.py`` replaces pytest-xdist's loadgroup scheduler
and, in a worker, pytest-timeout's ``thread`` timer.  What they fix only shows
across processes, so these run whole ``-n 2 --dist loadgroup`` sessions in a
subprocess, over a throwaway suite whose conftest wires the hooks as both
repo conftests do.  With pytest-xdist 3.8.0's own scheduler each of these
sessions hangs once its worker dies, and the test fails on ``subprocess.run``'s
timeout.

It also restores the ``@<group>`` nodeid suffix that pytest-xdist 3.8.0 can no
longer write on pytest 9.2, without which ``xdist_group`` groups nothing (#5356).
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests_shared.xdist_plumbing import make_crash_safe_scheduler, set_worker_timeout_timer

REPO_ROOT = Path(__file__).resolve().parents[2]

_CONFTEST = """
import pytest
from tests_shared.xdist_plumbing import make_crash_safe_scheduler, restore_xdist_group_suffix, set_worker_timeout_timer

def pytest_collection_modifyitems(items, config):
    restore_xdist_group_suffix(config, items)

@pytest.hookimpl(optionalhook=True)
def pytest_xdist_make_scheduler(config, log):
    return make_crash_safe_scheduler(config, log)

@pytest.hookimpl(optionalhook=True)
def pytest_timeout_set_timer(item, settings):
    return set_worker_timeout_timer(item, settings)
"""


def _quick(prefix: str, n: int) -> str:
    return "".join(f"def test_{prefix}{i}():\n    time.sleep(0.05)\n" for i in range(n))


def _session(tmp_path: Path, files: dict[str, str], *args: str, conftest_extra: str = "", returncode: int = 1) -> str:
    """Run pytest on ``files`` under ``-n 2 --dist loadgroup``; its stdout and stderr together."""
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "conftest.py").write_text(_CONFTEST + conftest_extra)
    for name, body in files.items():
        (tmp_path / name).write_text("import os, time\nimport pytest\n" + body)
    ours = ("PYTEST_ADDOPTS", "PYTEST_CURRENT_TEST")
    env = {k: v for k, v in os.environ.items() if k not in ours and not k.startswith("PYTEST_XDIST_")}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")]))
    r = subprocess.run(  # noqa: S603  # fixed argv, no shell
        [sys.executable, "-m", "pytest", "-c", "pytest.ini", "-p", "no:cacheprovider", "-n", "2"]
        + ["--dist", "loadgroup", "-q", *args],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
    )
    assert r.returncode == returncode, r.stdout
    return r.stdout


def test_a_test_past_the_timeout_fails_by_name_and_the_rest_still_run(tmp_path):
    """The worker had finished units when it died, and units were still queued: the 3.8.0 hang, both halves."""
    out = _session(
        tmp_path,
        {
            "test_a.py": _quick("a", 12),
            "test_b.py": "def test_hangs():\n    time.sleep(60)\n",
            "test_c.py": _quick("c", 12),
        },
        "--timeout=2",
        "--timeout-method=thread",
    )
    assert "crashed while running 'test_b.py::test_hangs'" in out
    assert "1 failed, 24 passed" in out
    # The timeout's own report, from the worker, and only once: the test is not run again.
    assert out.count("test_b.py::test_hangs ran past 2s on xdist worker gw") == 1, out
    assert "in test_hangs" in out


_RECORD_WORKER = (
    "def _record(name):\n    with open(f'ran-{name}', 'w') as f:\n        f.write(os.environ['PYTEST_XDIST_WORKER'])\n"
)


def test_a_group_runs_on_one_worker(tmp_path):
    """Every ``xdist_group`` test lands on one worker, even split across files (#5356).

    The group's two halves sit at either end of the collection, with ungrouped
    tests between them, so plain ``--dist load`` (what loadgroup degraded to
    when pytest-xdist's suffix stopped reaching the nodeid) hands them to both
    workers.
    """

    def grouped(prefix: str) -> str:
        return _RECORD_WORKER + "".join(
            f"@pytest.mark.xdist_group('g')\ndef test_{prefix}{i}():\n    _record('{prefix}{i}')\n    time.sleep(0.05)\n"
            for i in range(4)
        )

    out = _session(
        tmp_path,
        {"test_a.py": grouped("a"), "test_m.py": _quick("m", 16), "test_z.py": grouped("z")},
        "-v",
        returncode=0,
    )
    assert "test_a.py::test_a0@g" in out, out
    workers = {path.read_text() for path in tmp_path.glob("ran-*")}
    assert len(list(tmp_path.glob("ran-*"))) == 8, out
    assert len(workers) == 1, f"group 'g' ran on {sorted(workers)}"


@pytest.mark.xdist_group("xdist-group-suffix-probe")
def test_this_session_suffixes_its_own_grouped_tests(request):
    """The live suite's own grouped tests carry the suffix loadgroup schedules by (#5356)."""
    if not getattr(request.config.option, "loadgroup", False):
        pytest.skip("not running in a --dist loadgroup worker")
    assert request.node.nodeid.endswith("@xdist-group-suffix-probe")


def test_restore_xdist_group_suffix_leaves_a_suffixed_or_ungrouped_item_alone():
    from tests_shared.xdist_plumbing import restore_xdist_group_suffix

    def item(nodeid: str, *groups: str) -> Any:
        marks = [SimpleNamespace(args=(g,), kwargs={}) for g in groups]
        return SimpleNamespace(nodeid=nodeid, iter_markers=lambda name: iter(marks))

    already, plain = item("t.py::test_a@g", "g"), item("t.py::test_b")
    restore_xdist_group_suffix(SimpleNamespace(option=SimpleNamespace(loadgroup=True)), [already, plain])
    assert (already.nodeid, plain.nodeid) == ("t.py::test_a@g", "t.py::test_b")
    assert not hasattr(already, "_id") and not hasattr(plain, "_id")
    # Outside a loadgroup worker nothing is touched, not even a grouped item.
    grouped = item("t.py::test_c", "g")
    restore_xdist_group_suffix(SimpleNamespace(option=SimpleNamespace()), [grouped])
    assert not hasattr(grouped, "_id")


def test_a_crash_mid_group_hands_the_rest_of_the_group_on(tmp_path):
    out = _session(
        tmp_path,
        {
            "test_g.py": (
                "@pytest.mark.xdist_group('g')\ndef test_g1():\n    pass\n"
                "@pytest.mark.xdist_group('g')\ndef test_g2():\n    os._exit(1)\n"
                "@pytest.mark.xdist_group('g')\ndef test_g3():\n    pass\n"
            ),
            "test_x.py": _quick("x", 8),
        },
    )
    assert "crashed while running 'test_g.py::test_g2" in out
    assert "1 failed, 10 passed" in out


# Once the controller has the report of test_g1, a send to the worker that ran
# it fails as a send down a closed pipe does, and leaves a marker the waiting
# tests below hold out for.
_SEND_FAILS_AFTER_G1 = """
from xdist.workermanage import WorkerController

_send = WorkerController.send_runtest_some

def _send_or_fail(self, indices):
    if getattr(self, "reported_g1", False):
        open("send-failed", "w").close()
        raise OSError("cannot send (already closed?)")
    return _send(self, indices)

WorkerController.send_runtest_some = _send_or_fail

def pytest_runtest_logreport(report):
    if report.nodeid.startswith("test_g.py::test_g1") and report.when == "teardown" and hasattr(report, "node"):
        report.node.reported_g1 = True
"""

_WAIT_FOR_THE_FAILED_SEND = (
    "def _wait():\n"
    "    deadline = time.monotonic() + 60\n"
    "    while not os.path.exists('send-failed') and time.monotonic() < deadline:\n"
    "        time.sleep(0.01)\n"
)


def test_a_send_to_a_worker_already_gone_hands_the_unit_back(tmp_path):
    """The worker dies just after a report, and the report hands it more work (#4929).

    Until execnet sees the pipe close, a dead worker does not read as shutting
    down, so the report of its last test gets it another unit and the send
    raises, which ended the run in an ``INTERNALERROR``.  That window is too
    short to hit on purpose, so the conftest fails the send instead, and every
    other test waits until it has: the worker dies in test_g2 holding nothing it
    was not sent, and the other worker cannot drain the queue first.
    """
    out = _session(
        tmp_path,
        {
            "test_g.py": _WAIT_FOR_THE_FAILED_SEND
            + (
                "@pytest.mark.xdist_group('g')\ndef test_g1():\n    pass\n"
                "@pytest.mark.xdist_group('g')\ndef test_g2():\n    _wait()\n    os._exit(1)\n"
                "@pytest.mark.xdist_group('g')\ndef test_g3():\n    pass\n"
            ),
            "test_x.py": _WAIT_FOR_THE_FAILED_SEND + "".join(f"def test_x{i}():\n    _wait()\n" for i in range(8)),
        },
        conftest_extra=_SEND_FAILS_AFTER_G1,
    )
    assert (tmp_path / "send-failed").exists(), out
    assert "INTERNALERROR" not in out
    assert "crashed while running 'test_g.py::test_g2" in out
    assert "1 failed, 10 passed" in out


class _Node:
    """A worker as the scheduler sees one: hashable, and records what it is sent."""

    def __init__(self) -> None:
        self.shutting_down = False
        self.sent: list[list[int]] = []

    def send_runtest_some(self, indices: list[int]) -> None:
        self.sent.append(list(indices))

    def shutdown(self) -> None:
        self.shutting_down = True


def test_a_replacement_still_collecting_is_not_handed_work(tmp_path):
    """Two replacements join; the first to finish collecting must not schedule the other (#4966).

    A slow machine lost two workers to the timeout minutes apart, and the
    second's replacement was still collecting when the first's collection
    arrived.  Offering work to every node then reached the one whose collection
    the scheduler did not hold yet: ``KeyError`` on that worker, an
    ``INTERNALERROR`` that ended the run.  The queue is long enough to keep
    work pending for both offers.
    """
    from collections import OrderedDict

    from tests_shared.xdist_plumbing import CrashSafeLoadGroupScheduling

    collection = [f"test_{i}.py::test" for i in range(8)]
    # Built without __init__ (which reads the run's --tx specs) and fed stand-in
    # nodes, so typed loosely.
    sched: Any = CrashSafeLoadGroupScheduling.__new__(CrashSafeLoadGroupScheduling)
    sched.log = lambda *args: None
    sched.numnodes = 2
    sched.collection = collection
    sched.workqueue = OrderedDict((nodeid.rsplit("::", 1)[0], {nodeid: False}) for nodeid in collection)
    ready, collecting = _Node(), _Node()
    sched.assigned_work = {ready: {}, collecting: {}}
    sched.registered_collections = {ready: list(collection)}

    sched.schedule()

    assert ready.sent and not collecting.sent
    sched.registered_collections[collecting] = list(collection)
    sched.schedule()
    assert collecting.sent


def test_other_modes_keep_their_own_scheduler_and_timer():
    assert make_crash_safe_scheduler(SimpleNamespace(getvalue=lambda name: "load"), None) is None
    controller = SimpleNamespace(config=SimpleNamespace())
    assert set_worker_timeout_timer(controller, SimpleNamespace(method="thread", timeout=1.0)) is None
    worker = SimpleNamespace(config=SimpleNamespace(workerinput={}))
    assert set_worker_timeout_timer(worker, SimpleNamespace(method="signal", timeout=1.0)) is None


@pytest.mark.parametrize("tier", ["tests", "tests_lib"])
def test_both_conftests_take_both_hooks(tier):
    tree = ast.parse((REPO_ROOT / tier / "conftest.py").read_text())
    defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {"pytest_xdist_make_scheduler", "pytest_timeout_set_timer"} <= defined


@pytest.mark.parametrize("tier", ["tests", "tests_lib"])
def test_both_conftests_restore_the_group_suffix_before_the_pass_cache(tier):
    """The pass cache records suffixed nodeids, so the suffix must be on before it matches items."""
    tree = ast.parse((REPO_ROOT / tier / "conftest.py").read_text())
    hook = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "pytest_collection_modifyitems")
    calls = [
        n.value.func.id
        for n in hook.body
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
    ]
    assert "restore_xdist_group_suffix" in calls and "deselect_cached_passes" in calls, calls
    assert calls.index("restore_xdist_group_suffix") < calls.index("deselect_cached_passes"), calls
