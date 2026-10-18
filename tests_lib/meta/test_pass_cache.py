"""The pass cache reuses a pass only under an identical tree (``tests_shared/pass_cache.py``).

Three layers.  The key: derived on a throwaway git repo, it must move for every
input a test can observe (a tracked edit, an untracked file, a ``VTSEARCH_*``
variable) and for nothing else (an ignored file, the per-session data dirs).
The pass rule: only a test whose setup, call and teardown all passed is
recorded.  And whole sessions: a tiny project in a tmp git repo, with a
conftest wired exactly as the two repo conftests are, run through
``python -m pytest`` in a subprocess -- single-process and ``-n 2`` -- so what
is checked is the real controller/worker split, the real exit status and the
real summary banner.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests_shared import pass_cache
from tests_shared.pass_cache import (
    CACHE_KEY,
    FRESH_ENV,
    KEY_ENV,
    PassCacheState,
    passed_nodeids,
    prune,
    record_pass_report,
    tree_key,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GIT = shutil.which("git") or "git"

pytestmark = pytest.mark.skipif(not shutil.which("git"), reason="needs git")

# A fixed stand-in for the installed-distributions digest: it is slow to derive
# and constant within a process, so the key tests pin it.
DISTS = "dists-digest"
ENV = {"VTSEARCH_FOO": "1", "HOME": "/nowhere"}


def _git(cwd: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    for k in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        env.pop(k, None)
    return subprocess.run([GIT, *args], cwd=cwd, env=env, check=True, capture_output=True, text=True).stdout.strip()  # noqa: S603  # fixed argv, no shell


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository with one commit, one tracked file and an ignore rule."""
    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / ".gitignore").write_text("ignored.txt\n.pytest_cache/\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def _key(repo: Path, env: dict[str, str] = ENV) -> str:
    key = tree_key(repo, env, dists_digest=DISTS)
    assert key is not None
    return key


class TestTreeKey:
    def test_stable_across_two_computations(self, repo: Path) -> None:
        assert _key(repo) == _key(repo)

    def test_a_tracked_edit_moves_it_staged_or_not_and_so_does_the_commit(self, repo: Path) -> None:
        clean = _key(repo)
        (repo / "a.py").write_text("x = 2\n")
        unstaged = _key(repo)
        assert unstaged != clean
        _git(repo, "add", "a.py")
        assert _key(repo) == unstaged  # the same content against HEAD, staged or not
        _git(repo, "commit", "-q", "-m", "edit")
        assert _key(repo) not in {clean, unstaged}

    def test_an_untracked_file_moves_it_by_path_and_by_content(self, repo: Path) -> None:
        clean = _key(repo)
        (repo / "new.py").write_text("y = 1\n")
        added = _key(repo)
        assert added != clean
        (repo / "new.py").write_text("y = 2\n")
        assert _key(repo) not in {clean, added}

    def test_an_ignored_file_does_not(self, repo: Path) -> None:
        clean = _key(repo)
        (repo / "ignored.txt").write_text("anything\n")
        (repo / ".pytest_cache").mkdir()
        (repo / ".pytest_cache" / "stuff").write_text("x")
        assert _key(repo) == clean

    def test_a_vtsearch_variable_moves_it_and_the_session_dirs_do_not(self, repo: Path) -> None:
        base = _key(repo)
        assert _key(repo, {**ENV, "VTSEARCH_FOO": "2"}) != base
        assert _key(repo, {**ENV, "VTSEARCH_BAR": "1"}) != base
        assert _key(repo, {**ENV, "VTSEARCH_DATA_DIR": "/tmp/x"}) == base
        assert _key(repo, {**ENV, "VTSEARCH_TEST_DATA_DIR": "/tmp/x"}) == base
        assert _key(repo, {**ENV, KEY_ENV: "stale"}) == base
        assert _key(repo, {**ENV, "OTHER": "1"}) == base

    def test_no_repository_or_no_commit_gives_no_key(self, tmp_path: Path) -> None:
        bare = tmp_path / "plain"
        bare.mkdir()
        assert tree_key(bare, ENV, dists_digest=DISTS) is None
        fresh = tmp_path / "fresh"
        fresh.mkdir()
        _git(fresh, "init", "-q", "-b", "main")
        assert tree_key(fresh, ENV, dists_digest=DISTS) is None


def _report(nodeid: str, when: str, *, passed: bool, xfail: bool = False):
    rep = SimpleNamespace(nodeid=nodeid, when=when, passed=passed)
    if xfail:
        rep.wasxfail = "reason"
    return rep


class TestPassRule:
    def _record(self, monkeypatch, reports, *, worker: bool = False) -> set[str]:
        state = PassCacheState()
        state.is_worker = worker
        monkeypatch.setattr(pass_cache, "_state", state)
        for rep in reports:
            record_pass_report(rep)
        return passed_nodeids(state.reports)

    def test_only_a_test_green_in_all_three_phases_is_recorded(self, monkeypatch) -> None:
        reports = [
            # ok: green throughout
            _report("t::ok", "setup", passed=True),
            _report("t::ok", "call", passed=True),
            _report("t::ok", "teardown", passed=True),
            # failed call
            _report("t::fail", "setup", passed=True),
            _report("t::fail", "call", passed=False),
            _report("t::fail", "teardown", passed=True),
            # skipped in setup: no call report at all
            _report("t::skip", "setup", passed=False),
            _report("t::skip", "teardown", passed=True),
            # xfail: the call is "skipped" with wasxfail
            _report("t::xfail", "setup", passed=True),
            _report("t::xfail", "call", passed=False, xfail=True),
            _report("t::xfail", "teardown", passed=True),
            # xpass: passed, but under wasxfail
            _report("t::xpass", "setup", passed=True),
            _report("t::xpass", "call", passed=True, xfail=True),
            _report("t::xpass", "teardown", passed=True),
            # teardown error
            _report("t::td", "setup", passed=True),
            _report("t::td", "call", passed=True),
            _report("t::td", "teardown", passed=False),
        ]
        assert self._record(monkeypatch, reports) == {"t::ok"}

    def test_a_worker_records_nothing_the_controller_does(self, monkeypatch) -> None:
        reports = [_report("t::ok", w, passed=True) for w in ("setup", "call", "teardown")]
        assert self._record(monkeypatch, reports, worker=True) == set()

    def test_without_a_session_the_hook_is_inert(self, monkeypatch) -> None:
        monkeypatch.setattr(pass_cache, "_state", None)
        record_pass_report(_report("t::ok", "call", passed=True))  # no exception


def test_prune_keeps_the_four_most_recent_keys() -> None:
    keys = {f"k{i}": {"when": f"2026-10-{10 + i:02d}T00:00:00Z", "nodeids": []} for i in range(7)}
    kept = prune(keys)
    assert set(kept) == {"k3", "k4", "k5", "k6"}
    assert prune({"a": {"when": None, "nodeids": []}, "b": {"when": "2026-01-01T00:00:00Z", "nodeids": []}}, 1) == {
        "b": {"when": "2026-01-01T00:00:00Z", "nodeids": []}
    }


# ---------------------------------------------------------------------------
# Whole sessions.
# ---------------------------------------------------------------------------

# The same wiring as tests/conftest.py and tests_lib/conftest.py, including the
# force-exit summary printer, so the banner text is what the real suite prints.
_CONFTEST = """
import pytest
from tests_shared.pass_cache import (
    configure_pass_cache, deselect_cached_passes, finish_pass_cache, note_worker_reused, record_pass_report,
)
from tests_shared.pytest_plumbing import print_summary_and_exit

def pytest_configure(config):
    configure_pass_cache(config)

def pytest_collection_modifyitems(items, config):
    deselect_cached_passes(config, items)

def pytest_runtest_logreport(report):
    record_pass_report(report)

def pytest_sessionfinish(session, exitstatus):
    finish_pass_cache(session)
    session.config._exitstatus = session.exitstatus

@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    print_summary_and_exit(config, getattr(config, "_exitstatus", 0))

@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
    note_worker_reused(node)
"""

_FILES = {
    "test_a.py": "def test_a1():\n    pass\n\ndef test_a2():\n    pass\n",
    "test_b.py": (
        "import os\nimport pytest\n\n"
        "def test_b1():\n    assert not os.environ.get('MAKE_B1_FAIL')\n\n"
        "@pytest.mark.skipif(os.environ.get('SKIP_B2') == '1', reason='asked')\n"
        "def test_b2():\n    pass\n"
    ),
}


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A committed four-test project whose conftest is wired like the repo's."""
    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / ".gitignore").write_text(".pytest_cache/\n__pycache__/\n")
    (tmp_path / "conftest.py").write_text(_CONFTEST)
    for name, body in _FILES.items():
        (tmp_path / name).write_text(body)
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "project")
    return tmp_path


def _run(project: Path, *args: str, **extra_env: str) -> subprocess.CompletedProcess:
    ours = ("PYTEST_ADDOPTS", "PYTEST_CURRENT_TEST", KEY_ENV, FRESH_ENV, "MAKE_B1_FAIL", "SKIP_B2")
    env = {k: v for k, v in os.environ.items() if k not in ours and not k.startswith("PYTEST_XDIST_")}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")]))
    env.update(extra_env)
    return subprocess.run(  # noqa: S603  # fixed argv, no shell
        [sys.executable, "-m", "pytest", "-c", "pytest.ini", "-q", "--no-header", *args],
        cwd=project,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
    )


def _recorded(project: Path) -> dict[str, list[str]]:
    data = json.loads((project / ".pytest_cache" / "v" / CACHE_KEY).read_text())
    return {k: v["nodeids"] for k, v in data["keys"].items()}


ALL_FOUR = ["test_a.py::test_a1", "test_a.py::test_a2", "test_b.py::test_b1", "test_b.py::test_b2"]
XDIST = ("-n", "2", "--dist", "loadgroup")


@pytest.mark.parametrize("mode", [(), XDIST], ids=["single-process", "xdist"])
def test_a_second_run_on_the_same_tree_reruns_nothing_and_exits_0(project: Path, mode) -> None:
    first = _run(project, *mode)
    assert first.returncode == 0, first.stdout
    assert "ALL 4 TESTS PASSED" in first.stdout
    assert "Pass cache: 4 passes recorded for this tree (none reused)." in first.stdout
    (recorded,) = _recorded(project).values()
    assert recorded == ALL_FOUR

    second = _run(project, *mode)
    assert second.returncode == 0, second.stdout
    assert "ALL 4 TESTS ALREADY PASSED ON THIS TREE (nothing re-run)" in second.stdout
    assert "Pass cache: 4 tests not re-run (passed on this identical tree at 20" in second.stdout
    assert "; 0 newly recorded." in second.stdout
    assert "ALL 0 TESTS PASSED" not in second.stdout


def test_a_group_run_s_passes_carry_into_the_full_run(project: Path) -> None:
    group = _run(project, "test_a.py")
    assert group.returncode == 0, group.stdout
    assert list(_recorded(project).values()) == [["test_a.py::test_a1", "test_a.py::test_a2"]]

    full = _run(project)
    assert full.returncode == 0, full.stdout
    assert "2 passed, 2 deselected" in full.stdout
    assert "Pass cache: 2 tests not re-run" in full.stdout and "; 2 newly recorded." in full.stdout
    assert list(_recorded(project).values()) == [ALL_FOUR]


def test_only_passes_are_recorded_so_a_failure_and_a_skip_run_again(project: Path) -> None:
    red = _run(project, MAKE_B1_FAIL="1", SKIP_B2="1")
    assert red.returncode == 1, red.stdout
    assert "TESTS FAILED: 1 failed" in red.stdout
    assert list(_recorded(project).values()) == [["test_a.py::test_a1", "test_a.py::test_a2"]]

    again = _run(project, MAKE_B1_FAIL="1", SKIP_B2="1")
    assert again.returncode == 1, again.stdout
    assert "1 failed, 1 skipped, 2 deselected" in again.stdout

    green = _run(project)
    assert green.returncode == 0, green.stdout
    assert "2 passed, 2 deselected" in green.stdout
    assert list(_recorded(project).values()) == [ALL_FOUR]


def test_any_edit_to_the_tree_reruns_everything(project: Path) -> None:
    assert _run(project).returncode == 0
    assert "nothing re-run" in _run(project).stdout
    with (project / "test_a.py").open("a") as f:
        f.write("# a comment is still a change\n")
    third = _run(project)
    assert third.returncode == 0, third.stdout
    assert "ALL 4 TESTS PASSED" in third.stdout
    assert "deselected" not in third.stdout
    assert len(_recorded(project)) == 2  # the old tree's entry is kept beside the new one


def test_fresh_reruns_everything_but_still_records(project: Path) -> None:
    assert _run(project).returncode == 0
    fresh = _run(project, **{FRESH_ENV: "1"})
    assert fresh.returncode == 0, fresh.stdout
    assert "ALL 4 TESTS PASSED" in fresh.stdout
    assert "deselected" not in fresh.stdout
    assert "nothing re-run" in _run(project).stdout


def test_without_the_cacheprovider_the_plugin_is_inert(project: Path) -> None:
    assert _run(project).returncode == 0
    plain = _run(project, "-p", "no:cacheprovider")
    assert plain.returncode == 0, plain.stdout
    assert "ALL 4 TESTS PASSED" in plain.stdout
    assert "Pass cache" not in plain.stdout


def test_outside_a_repository_the_plugin_says_so_once_and_records_nothing(tmp_path: Path) -> None:
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "conftest.py").write_text(_CONFTEST)
    (tmp_path / "test_a.py").write_text(_FILES["test_a.py"])
    env = {k: v for k, v in os.environ.items() if k not in (KEY_ENV, "GIT_DIR", "GIT_WORK_TREE")}
    env["PYTHONPATH"] = str(REPO_ROOT)
    out = subprocess.run(  # noqa: S603  # fixed argv, no shell
        [sys.executable, "-m", "pytest", "-c", "pytest.ini", "-q", "--no-header"],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stdout
    assert out.stdout.count("pass cache: no git tree key for this run") == 1
    assert not (tmp_path / ".pytest_cache" / "v" / CACHE_KEY).exists()


@pytest.mark.parametrize("tier", ["tests", "tests_lib"])
def test_both_conftests_wire_every_hook(tier: str) -> None:
    source = (REPO_ROOT / tier / "conftest.py").read_text()
    tree = ast.parse(source)
    defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    hooks = {
        "pytest_configure",
        "pytest_collection_modifyitems",
        "pytest_runtest_logreport",
        "pytest_sessionfinish",
        "pytest_testnodedown",
    }
    assert hooks <= defined
    for fn in ("configure_pass_cache", "deselect_cached_passes", "record_pass_report", "note_worker_reused"):
        assert f"{fn}(" in source
    # The force-exit must see the status the pass cache may have overridden.
    finish = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "pytest_sessionfinish")
    statements = [
        ast.unparse(s) for s in finish.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
    ]
    body = "\n".join(statements)
    assert body.index("finish_pass_cache(session)") < body.index("session.exitstatus")
    assert "= exitstatus" not in body
