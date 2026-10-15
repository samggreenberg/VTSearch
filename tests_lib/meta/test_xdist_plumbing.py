"""A dead xdist worker fails its test by name instead of wedging the run (issue #4925).

``tests_shared/xdist_plumbing.py`` replaces pytest-xdist's loadgroup scheduler
and, in a worker, pytest-timeout's ``thread`` timer.  What they fix only shows
across processes, so these run whole ``-n 2 --dist loadgroup`` sessions in a
subprocess, over a throwaway suite whose conftest wires the two hooks as both
repo conftests do.  With pytest-xdist 3.8.0's own scheduler each of these
sessions hangs once its worker dies, and the test fails on ``subprocess.run``'s
timeout.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests_shared.xdist_plumbing import make_crash_safe_scheduler, set_worker_timeout_timer

REPO_ROOT = Path(__file__).resolve().parents[2]

_CONFTEST = """
import pytest
from tests_shared.xdist_plumbing import make_crash_safe_scheduler, set_worker_timeout_timer

@pytest.hookimpl(optionalhook=True)
def pytest_xdist_make_scheduler(config, log):
    return make_crash_safe_scheduler(config, log)

@pytest.hookimpl(optionalhook=True)
def pytest_timeout_set_timer(item, settings):
    return set_worker_timeout_timer(item, settings)
"""


def _quick(prefix: str, n: int) -> str:
    return "".join(f"def test_{prefix}{i}():\n    time.sleep(0.05)\n" for i in range(n))


def _session(tmp_path: Path, files: dict[str, str], *args: str, conftest_extra: str = "") -> str:
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
    assert r.returncode == 1, r.stdout
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
