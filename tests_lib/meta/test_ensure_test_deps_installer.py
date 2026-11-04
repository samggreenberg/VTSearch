"""Tests for .claude/hooks/ensure-test-deps.sh, the cloud container's installer.

Issue #4662: the cloud image shipped with `python` on 3.11 and `pip` on 3.13.
The hook installed with a bare `pip install`, so the whole stack landed in 3.13
while every documented `python ...` command ran on 3.11 and died on
`No module named 'flask'`. The hook's marker then made it worse: it was written
after the (wrong-interpreter) install, so every later run exited early and
re-running the hook changed nothing.

The installer is exercised for real, against stubs: a `python` on `PATH` that
records its `-m pip` calls and answers the import probe from a state file, and
a `pip` / `pip3` that records any call at all. No package is installed, and the
stack only "imports" once the stubbed `python` has installed the requirements,
which is the property the hook must now check rather than assume.

Issue #5549: the hook's `npm install` had no bound, so one stalled install ate
the whole 30-minute run-tests.sh cap. A stubbed `npm` stalls or fails on cue to
check that each attempt is cut short, retried from a clean tree, and that a
final failure names the step rather than leaving a partial `node_modules` the
next run would trust.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

HOOKS = Path(__file__).resolve().parents[2] / ".claude" / "hooks"
INSTALLER = HOOKS / "ensure-test-deps.sh"
PRUNE = HOOKS / "prune-stale-dist-info.py"

pytestmark = pytest.mark.skipif(shutil.which("sha256sum") is None, reason="the hook hashes itself with sha256sum")

# Answers `python -m pip ...` with success, and marks the stack importable once
# the requirements file goes in -- unless the test says the install is broken.
# The import probe (`python -c 'import flask, ...'`) succeeds only after that.
STUB_PYTHON = """#!/bin/bash
echo "python $*" >> "$STUB_LOG"
if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then
  if [[ " $* " == *"requirements/base.txt"* ]] && [ -z "${STUB_INSTALL_IS_BROKEN:-}" ]; then
    touch "$STUB_STATE/importable"
  fi
  exit 0
fi
if [ "$1" = "-c" ] && [[ "$2" == *"import flask"* ]]; then
  [ -f "$STUB_STATE/importable" ]
  exit $?
fi
exit 0
"""

STUB_BARE_PIP = """#!/bin/bash
echo "bare-pip $*" >> "$STUB_LOG"
exit 0
"""

# Each `install` writes its debug log into --logs-dir first, as the real one
# writes as it goes, and leaves a partial node_modules. The first
# $STUB_NPM_STALLS installs then hang, the next $STUB_NPM_FAILS exit 1, and the
# rest complete. A hang that is not cut short records that it outlived the cap.
STUB_NPM = """#!/bin/bash
if [ "$1" = "install" ]; then
  n=$(( $(cat "$STUB_STATE/npm-installs" 2>/dev/null || echo 0) + 1 ))
  echo "$n" > "$STUB_STATE/npm-installs"
  tree=clean
  [ -e node_modules ] && tree=dirty
  echo "npm $* [$tree tree]" >> "$STUB_LOG"
  logs=$STUB_STATE/no-logs-dir-given
  for arg in "$@"; do
    case "$arg" in --logs-dir=*) logs=${arg#--logs-dir=} ;; esac
  done
  mkdir -p node_modules/half-extracted "$logs"
  echo "http fetch GET 200 https://registry.npmjs.org/stalled-at-$n.tgz" > "$logs/attempt-$n-debug-0.log"
  stalls=${STUB_NPM_STALLS:-0}
  if [ "$n" -le "$stalls" ]; then
    sleep 20
    echo "npm stall $n outlived the cap" >> "$STUB_LOG"
    exit 1
  fi
  [ "$n" -le $(( stalls + ${STUB_NPM_FAILS:-0} )) ] && exit 1
  touch node_modules/.package-lock.json
  exit 0
fi
exit 0
"""


class Sandbox:
    def __init__(self, root: Path):
        self.root = root
        self.bin = root / "bin"
        self.state = root / "state"
        self.markers = root / "markers"
        self.log = root / "calls.log"
        self.frontend = root / "frontend"
        self.npm_cache = root / "npm-cache"
        for d in (self.bin, self.state, self.markers):
            d.mkdir()
        stubs = (("python", STUB_PYTHON), ("pip", STUB_BARE_PIP), ("pip3", STUB_BARE_PIP), ("npm", STUB_NPM))
        for name, body in stubs:
            exe = self.bin / name
            exe.write_text(body)
            exe.chmod(0o755)
        self.log.touch()
        script_hash = hashlib.sha256(INSTALLER.read_bytes() + PRUNE.read_bytes()).hexdigest()[:12]
        self.marker = self.markers / f".vtsearch-deps-installed-{script_hash}"

    def add_frontend(self) -> None:
        """Without a frontend/, the hook skips its npm step."""
        self.frontend.mkdir()
        for name in ("package.json", "package-lock.json"):
            (self.frontend / name).write_text("{}\n")

    def run(self, **extra_env: str) -> subprocess.CompletedProcess:
        env = {
            **{k: v for k, v in os.environ.items() if k.lower() != "npm_config_logs_dir"},
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "CLAUDE_CODE_REMOTE": "true",
            "VTSEARCH_DEPS_MARKER_DIR": str(self.markers),
            "STUB_LOG": str(self.log),
            "STUB_STATE": str(self.state),
            "npm_config_cache": str(self.npm_cache),
            **extra_env,
        }
        # cwd has no frontend/ until add_frontend(), so the npm step is skipped.
        return subprocess.run(  # noqa: S603  # bash + repo-local hook path, no shell
            ["bash", str(INSTALLER)],  # noqa: S607
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def calls(self) -> list[str]:
        return self.log.read_text().splitlines()

    def pip_installs(self) -> list[str]:
        return [c for c in self.calls() if c.startswith("python -m pip install")]

    def bare_pip_calls(self) -> list[str]:
        return [c for c in self.calls() if c.startswith("bare-pip")]

    def npm_installs(self) -> list[str]:
        return [c for c in self.calls() if c.startswith("npm install")]


@pytest.fixture
def sandbox(tmp_path) -> Sandbox:
    return Sandbox(tmp_path)


class TestInstallsIntoThePythonInUse:
    def test_every_install_goes_through_python_m_pip(self, sandbox):
        result = sandbox.run()
        assert result.returncode == 0, result.stdout + result.stderr
        assert sandbox.bare_pip_calls() == [], "a bare `pip` installs into whatever interpreter it belongs to"
        assert any("requirements/base.txt" in c for c in sandbox.pip_installs())
        assert sandbox.marker.exists()

    def test_the_user_site_upgrade_goes_through_python_m_pip_too(self, sandbox):
        """The stub enables the user site, so the one conditional install runs as well."""
        sandbox.run()
        assert any(" --user " in c for c in sandbox.pip_installs())


class TestPrunesAfterTheLastInstall:
    def test_the_prune_runs_after_every_pip_install(self, sandbox):
        """The stale dist-info it deletes is left by the installs, so it must follow all of them (#5153)."""
        sandbox.run()
        calls = sandbox.calls()
        prune = [n for n, c in enumerate(calls) if c == f"python {PRUNE}"]
        installs = [n for n, c in enumerate(calls) if c.startswith("python -m pip install")]
        assert len(prune) == 1, calls
        assert installs and prune[0] > max(installs), calls

    def test_an_edit_to_the_prune_invalidates_the_marker(self, sandbox):
        """The marker hashes the prune too: a fixed prune must re-run on a long-lived container."""
        installer_only = hashlib.sha256(INSTALLER.read_bytes()).hexdigest()[:12]
        assert not sandbox.marker.name.endswith(installer_only)
        sandbox.run()
        assert sandbox.marker.exists()


class TestMarker:
    def test_a_marker_whose_stack_does_not_import_is_not_trusted(self, sandbox):
        """The issue's trap: a marker from another interpreter short-circuited every re-run."""
        sandbox.marker.touch()
        result = sandbox.run()
        assert result.returncode == 0, result.stdout + result.stderr
        assert sandbox.pip_installs(), "a marker over an unimportable stack skipped the install"
        assert "reinstalling" in result.stdout

    def test_a_marker_whose_stack_imports_skips_the_install(self, sandbox):
        sandbox.marker.touch()
        (sandbox.state / "importable").touch()
        result = sandbox.run()
        assert result.returncode == 0, result.stdout + result.stderr
        assert sandbox.pip_installs() == []

    def test_an_install_that_leaves_the_stack_unimportable_writes_no_marker(self, sandbox):
        result = sandbox.run(STUB_INSTALL_IS_BROKEN="1")
        assert result.returncode != 0
        assert "do not import" in result.stderr
        assert not sandbox.marker.exists(), "the next run would trust a marker over a broken install"


class TestFrontendInstall:
    """Each npm attempt is bounded, and a failure is retried from a clean tree (#5549)."""

    def test_a_stalled_attempt_is_cut_short_and_retried_from_a_clean_tree(self, sandbox):
        sandbox.add_frontend()
        result = sandbox.run(STUB_NPM_STALLS="1", VTSEARCH_NPM_INSTALL_TIMEOUT="2")
        assert result.returncode == 0, result.stdout + result.stderr
        assert not [c for c in sandbox.calls() if "outlived the cap" in c]
        installs = sandbox.npm_installs()
        assert len(installs) == 2, installs
        assert installs[1].endswith("[clean tree]"), "the retry ran over the stalled attempt's partial tree"
        assert "npm install timed out after 2s" in result.stderr
        assert "stalled-at-1.tgz" in result.stderr, "a killed npm prints nothing; its debug log shows where it stopped"
        assert (sandbox.frontend / "node_modules").is_dir()
        assert sandbox.marker.exists()

    def test_when_every_attempt_fails_the_run_is_blocked_by_name(self, sandbox):
        """The issue's trap: the run's 1800s cap fired instead, and blamed a wedged run."""
        sandbox.add_frontend()
        result = sandbox.run(STUB_NPM_STALLS="1", STUB_NPM_FAILS="1", VTSEARCH_NPM_INSTALL_TIMEOUT="2")
        assert result.returncode != 0
        assert len(sandbox.npm_installs()) == 2
        assert "TESTS BLOCKED: npm install in frontend/ did not finish" in result.stderr
        assert "attempt 1: timed out after 2s" in result.stderr
        assert re.search(r"attempt 2: failed \(exit 1\) after \d+s", result.stderr), result.stderr
        assert f"npm's debug logs are in {sandbox.npm_cache / '_logs'}" in result.stderr
        assert not sandbox.marker.exists()

    def test_a_failed_install_leaves_no_partial_tree_for_the_next_run_to_trust(self, sandbox):
        """A partial node_modules is newer than the lockfile, so the next run's check would skip npm."""
        sandbox.add_frontend()
        assert sandbox.run(STUB_NPM_FAILS="2").returncode != 0
        assert not (sandbox.frontend / "node_modules").exists()
        result = sandbox.run()
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(sandbox.npm_installs()) == 3
        assert (sandbox.frontend / "node_modules").is_dir()
        assert sandbox.marker.exists()

    def test_npm_fails_a_hung_fetch_inside_the_attempt(self, sandbox):
        """npm's own default is 300s per request, tried three times: longer than a whole attempt."""
        sandbox.add_frontend()
        assert sandbox.run().returncode == 0
        assert [c for c in sandbox.npm_installs() if "--fetch-timeout=60000" in c] == sandbox.npm_installs() != []


# `pip install` / `pip3 install` / `pip3.13 install` not preceded by `-m `.
_BARE_PIP_INSTALL = re.compile(r"(?<!-m )\bpip(?:3(?:\.\d+)?)? install\b")


@pytest.mark.parametrize("hook", sorted(HOOKS.glob("*.sh")), ids=lambda p: p.name)
def test_no_hook_installs_with_a_bare_pip(hook: Path):
    """Covers the hooks the behavioural tests above do not run (session-start.sh)."""
    offenders = [
        f"{hook.name}:{n}: {line.strip()}"
        for n, line in enumerate(hook.read_text().splitlines(), 1)
        if not line.lstrip().startswith("#") and _BARE_PIP_INSTALL.search(line)
    ]
    assert offenders == [], "install with `python -m pip`, not a bare `pip` (issue #4662):\n" + "\n".join(offenders)
